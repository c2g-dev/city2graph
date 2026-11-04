"""Resolve calendars, service periods and agency metadata with their provenance."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from functools import lru_cache
from importlib.metadata import version
from itertools import pairwise
from typing import Any
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo
from zoneinfo import ZoneInfoNotFoundError

import pandas as pd

from .feed import _assumption
from .schema import _MODEL
from .schema import _REGISTRY
from .schema import TransportData
from .schema import _clock
from .schema import _default
from .schema import _GenerationConfig
from .schema import _present
from .schema import _records
from .schema import _seconds


def _geographical_timezone(data: TransportData) -> tuple[str | None, str]:
    """Infer a timezone only when every stop has the same unique land timezone.

    Geography is an estimate of local time rather than agency metadata.

    Parameters
    ----------
    data : TransportData
        Input with validated WGS84 stop coordinates.

    Returns
    -------
    tuple
        Unique timezone or None, and the lookup dataset version and limitation.
    """
    points = tuple(
        (float(row["stop_lon"]), float(row["stop_lat"])) for row in _records(data["stops"])
    )
    return _timezone_points(points)


@lru_cache(maxsize=128)
def _timezone_points(points: tuple[tuple[float, float], ...]) -> tuple[str | None, str]:
    """Cache offline lookups for repeated generation from unchanged stops.

    Ocean and ambiguous boundary results are rejected without choosing a zone.

    Parameters
    ----------
    points : tuple of tuple of float
        WGS84 longitude and latitude pairs.

    Returns
    -------
    tuple
        Unique timezone and the lookup dataset description.
    """
    from timezonefinder import TimezoneFinder  # noqa: PLC0415

    zones: set[str] = set()
    description = (
        f"Offline timezonefinder {version('timezonefinder')} with timezonefinder-data {version('timezonefinder-data')}; "
        "stop geography is an estimate, not observed agency metadata"
    )
    with TimezoneFinder() as finder:
        for longitude, latitude in points:
            matches = finder.timezones_at(lng=longitude, lat=latitude)
            if len(matches) != 1 or matches[0].startswith("Etc/"):
                return (
                    None,
                    description + "; stop timezone is ambiguous or outside a unique land timezone",
                )
            zones.add(matches[0])
    if len(zones) != 1:
        return None, description + "; stops span multiple land timezones"
    return next(iter(zones)), description


def _source_metadata(
    route: dict[str, Any],
    identifiers: list[str],
    sources: dict[str, dict[str, dict[str, Any]]],
    records: list[dict[str, object]],
) -> dict[str, tuple[str, str]]:
    """Collect only unambiguous agency fields from a route's source patterns.

    Conflicting source fields are recorded instead of selecting an arbitrary pattern.

    Parameters
    ----------
    route : dict
        Route whose source metadata is being collected.
    identifiers : list of str
        Source patterns belonging to this route.
    sources : dict
        Interpreted source information by pattern.
    records : list of dict
        Conflicting source fields to record as ignored.

    Returns
    -------
    dict
        Agency fields mapped to their source values and origins.
    """
    result = {}
    for field in ("agency_name", "agency_url", "agency_timezone"):
        values = {
            str(record["value"]): str(record["source"])
            for identifier in identifiers
            if (record := sources.get(str(identifier), {}).get(field)) is not None
        }
        if len(values) == 1:
            value, origin = next(iter(values.items()))
            result[field] = (value, origin)
        elif values:
            _assumption(
                records,
                str(route["route_id"]),
                field,
                ", ".join(sorted(values)),
                "ignored",
                "source metadata",
                "Conflicting agency metadata within the same route",
            )
    return result


def _valid_website(value: str) -> bool:
    """Check a source URL without fetching or correcting it.

    The original source value is not repaired or resolved through network requests.

    Parameters
    ----------
    value : str
        Candidate agency URL.

    Returns
    -------
    bool
        Whether an HTTP or HTTPS URL has a valid network location.
    """
    try:
        parts = urlsplit(value)
    except ValueError:
        return False
    return parts.scheme in {"http", "https"} and bool(parts.netloc)


def _feed_timezone(
    data: TransportData,
    metadata: list[dict[str, tuple[str, str]]],
    records: list[dict[str, object]],
) -> tuple[str, str, str, str]:
    """Resolve a single feed timezone before linking agencies.

    Source metadata precedes geographical inference and a warned UTC placeholder.

    Parameters
    ----------
    data : TransportData
        Normalised stops for geographical lookup.
    metadata : list of dict
        Source agency fields, with invalid zones removed in place.
    records : list of dict
        Rejected timezone information to retain.

    Returns
    -------
    tuple of str
        Timezone, origin kind, origin name and explanation.
    """
    source_zones: set[str] = set()
    for fields in metadata:
        if "agency_timezone" in fields:
            zone, origin = fields["agency_timezone"]
            try:
                ZoneInfo(zone)
                source_zones.add(zone)
            except (ZoneInfoNotFoundError, ValueError):
                _assumption(
                    records,
                    "agency",
                    "agency_timezone",
                    zone,
                    "ignored",
                    origin,
                    "Invalid source timezone",
                )
                fields.pop("agency_timezone")
    if len(source_zones) == 1:
        zone = next(iter(source_zones))
        zone_kind, zone_source, reason = (
            "source",
            "OpenStreetMap:timezone",
            "Unambiguous source timezone",
        )
    else:
        inferred, reason = _geographical_timezone(data)
        if source_zones:
            _assumption(
                records,
                "agency",
                "agency_timezone",
                ", ".join(sorted(source_zones)),
                "ignored",
                "source metadata",
                "Conflicting source timezones; geographical inference is attempted",
            )
            reason = "Conflicting source timezones ignored; " + reason
        zone = inferred or str(_default("agency_timezone"))
        zone_kind, zone_source = (
            ("estimated", "stop geography") if inferred else ("default", _MODEL)
        )
    return zone, zone_kind, zone_source, reason


def _agency_identity(
    route: dict[str, Any], fields: dict[str, tuple[str, str]], records: list[dict[str, object]]
) -> tuple[dict[str, object], dict[str, tuple[str, str, str]]]:
    """Resolve the operator name and website independently.

    Invalid website values are recorded before synthetic metadata is substituted.

    Parameters
    ----------
    route : dict
        Route receiving source metadata or placeholders.
    fields : dict
        Available source metadata and origins.
    records : list of dict
        Rejection records to extend.

    Returns
    -------
    tuple of dict
        Agency values and their effective origins.
    """
    result: dict[str, object] = {}
    origins: dict[str, tuple[str, str, str]] = {}
    for field in ("agency_name", "agency_url"):
        value, origin = fields.get(field, (str(_default(field)), _MODEL))
        valid = field != "agency_url" or _valid_website(value)
        if not valid:
            _assumption(
                records,
                str(route["route_id"]),
                field,
                value,
                "ignored",
                origin,
                "Invalid source website URL",
            )
            value, origin = str(_default(field)), _MODEL
        result[field] = value
        origins[field] = (
            "default" if origin == _MODEL else "source",
            origin,
            "Synthetic placeholder metadata"
            if origin == _MODEL
            else "Unambiguous route source metadata",
        )
    return result, origins


def _resolve_agency(
    data: TransportData,
    agency: pd.DataFrame | None,
    records: list[dict[str, object]],
    sources: dict[str, dict[str, dict[str, Any]]],
) -> pd.DataFrame:
    """
    Retain supplied agencies or fill source metadata before placeholders.

    Generated agency links retain their origin without being labelled as user input.

    Parameters
    ----------
    data : TransportData
        Independent normalised input; generated agency links may be added.
    agency : pandas.DataFrame or None
        Complete explicit agency table, if supplied.
    records : list of dict
        Effective values and rejection reasons to extend.
    sources : dict[str, dict[str, dict[str, Any]]]
        Interpreted source records indexed once for this generation.

    Returns
    -------
    pandas.DataFrame
        Valid agency metadata with one timezone across the feed.
    """
    if agency is not None:
        for index, row in enumerate(_records(agency)):
            for field, value in row.items():
                _assumption(
                    records, f"agency:{row.get('agency_id', index)}", field, value, "user", "input"
                )
        return agency.copy()
    routes = _records(data["routes"])
    route_patterns: dict[str, list[str]] = {}
    for pattern in data["patterns"].itertuples():
        route_patterns.setdefault(str(pattern.route_id), []).append(str(pattern.pattern_id))
    metadata = [
        _source_metadata(route, route_patterns.get(str(route["route_id"]), []), sources, records)
        for route in routes
    ]
    zone, zone_kind, zone_source, reason = _feed_timezone(data, metadata, records)
    agencies: dict[str, dict[str, object]] = {}
    agency_links = {
        str(record["record_id"]): record
        for record in reversed(records)
        if record["table_name"] == "routes" and record["field"] == "agency_id"
    }
    keys: dict[tuple[str, str, str], str] = {}
    for route, fields in zip(routes, metadata, strict=True):
        result, origins = _agency_identity(route, fields, records)
        result["agency_timezone"] = zone
        origins["agency_timezone"] = (zone_kind, zone_source, reason)
        if zone_kind == "source":
            if "agency_timezone" in fields:
                origins["agency_timezone"] = ("source", fields["agency_timezone"][1], reason)
            else:
                origins["agency_timezone"] = (
                    "estimated",
                    "feed timezone",
                    "Unique available source timezone reused for the feed; this route has no timezone metadata",
                )
        key = (str(result["agency_name"]), str(result["agency_url"]), zone)
        identifier = (
            str(route["agency_id"])
            if _present(route.get("agency_id"))
            else keys.setdefault(key, f"agency_{len(keys)}")
        )
        result["agency_id"] = identifier
        if previous := agencies.get(identifier):
            if previous != result:
                msg = f"Conflicting source agency metadata for agency_id {identifier!r}; supply agency explicitly"
                raise ValueError(msg)
        else:
            agencies[identifier] = result
            for field, value in result.items():
                kind, source, details = origins.get(
                    field, ("generated", _MODEL, "Deterministic agency link")
                )
                if field == "agency_id" and _present(route.get("agency_id")):
                    original = agency_links.get(str(route["route_id"]))
                    if original:
                        kind, source, details = (
                            str(original["kind"]),
                            str(original["source"]),
                            "Agency link retained from the input route",
                        )
                _assumption(records, f"agency:{identifier}", field, value, kind, source, details)
        route["agency_id"] = identifier
    data["routes"] = pd.DataFrame(routes)
    return pd.DataFrame(agencies.values())


@dataclass
class _ServiceContext:
    """Resolved pattern/calendar context and the generation-local ID allocator.

    Evaluated calendars and identifier reservations are shared within one call.
    """

    conditions: dict[str, dict[str, Any]]
    group: str
    size: int
    unresolved: bool
    offset: int
    trip_ids: set[str]
    dated_trip_ids: dict[tuple[str, str], str]
    user_fields: set[str]
    base_id: str
    days: dict[str, bool]


@dataclass
class _ServicePeriod:
    """Adopted values with their origin and resolved frequency scope.

    Conditional source periods carry or explicitly replace the shared budget.
    """

    row: dict[str, Any]
    origins: dict[str, tuple[str, str, str]]
    shared_headway: float | None = None
    shared_kind: str = "default"

    @property
    def frequency_scope(self) -> str:
        """Describe the already resolved departure policy.

        Reporting uses the adopted scope without repeating precedence rules.

        Returns
        -------
        str
            Scheduled, per-pattern or shared-group frequency scope.
        """
        if "departure_time" in self.origins:
            return "scheduled"
        return "shared route-direction budget" if self.shared_headway is not None else "per pattern"


def _calendar_days(config: _GenerationConfig, identifier: str) -> dict[str, bool]:
    """Evaluate declared calendars or construct the fixed representative week.

    Explicit calendar exceptions override weekly eligibility on the same date.

    Parameters
    ----------
    config : _GenerationConfig
        Explicit dates and calendars.
    identifier : str
        Service identifier before any source-driven splitting.

    Returns
    -------
    dict
        Declared dates and their activity, including explicit removals.
    """
    if config.calendar is None and config.calendar_dates is None:
        first = datetime.strptime(config.service_date or str(_default("service_date")), "%Y%m%d")
        count = 1 if config.service_date is not None else int(_default("service_days"))
        return {(first + timedelta(days=day)).strftime("%Y%m%d"): True for day in range(count)}
    days: dict[str, bool] = {}
    weekdays = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
    for row in _records(config.calendar if config.calendar is not None else pd.DataFrame()):
        if str(row["service_id"]) != identifier:
            continue
        for day in pd.date_range(str(row["start_date"]), str(row["end_date"])):
            days[day.strftime("%Y%m%d")] = int(row[weekdays[day.weekday()]]) == 1
    for row in _records(
        config.calendar_dates if config.calendar_dates is not None else pd.DataFrame()
    ):
        if str(row["service_id"]) == identifier:
            days[str(row["date"])] = int(row["exception_type"]) == 1
    return dict(sorted(days.items()))


def _service_groups(data: TransportData, selected: set[str]) -> dict[str, tuple[str, int, bool]]:
    """Share the baseline rate among selected route-direction patterns.

    Only default frequencies are shared; supplied intervals keep their own scope.

    Parameters
    ----------
    data : TransportData
        Patterns with optional explicit or source-derived service_group_id.
    selected : set of str
        Patterns targeted by the service input.

    Returns
    -------
    dict
        Pattern IDs mapped to a group, its size and unresolved-direction status.
    """
    membership = {}
    for row in _records(data["patterns"]):
        identifier = str(row["pattern_id"])
        if identifier not in selected:
            continue
        group = (
            str(row["service_group_id"])
            if _present(row.get("service_group_id"))
            else (f"route:{row['route_id']}|direction:{row['direction_id']}")
        )
        membership[identifier] = (group, row.get("service_group_direction") == "unresolved")
    sizes: dict[str, int] = {}
    for group, _ in membership.values():
        sizes[group] = sizes.get(group, 0) + 1
    return {
        key: (group, sizes[group], unresolved) for key, (group, unresolved) in membership.items()
    }


def _period_values(
    row: dict[str, Any],
    period: list[str] | None,
    conditions: dict[str, dict[str, Any]],
    config: _GenerationConfig,
    group_size: int,
) -> _ServicePeriod:
    """Choose explicit, source and fallback values for one initial period.

    Each field is selected independently before conditional periods are intersected.

    Parameters
    ----------
    row : dict
        Partial service input.
    period : list of str or None
        Adopted source opening window.
    conditions : dict
        Interpreted source fields.
    config : _GenerationConfig
        Global overrides.
    group_size : int
        Number of selected patterns sharing a default frequency budget.

    Returns
    -------
    tuple
        Effective row and field-specific origins and explanations.
    """
    values, origins = dict(row), {}
    shared_headway = None
    shared_kind = "default"
    if _present(row.get("departure_time")):
        origins["departure_time"] = (
            "user",
            "input",
            "Explicit fixed departure; no frequency is added",
        )
        return _ServicePeriod(values, origins)
    for field in ("start_time", "end_time", "headway_secs", "exact_times"):
        value = row.get(field)
        origin = ("user", "input", "Explicit service-row value takes precedence")
        if not _present(value):
            value = getattr(config, field, None)
            origin = ("user", "input", "Explicit global override")
        if field == "headway_secs" and value is None and config.group_headway_secs is not None:
            shared_headway, shared_kind = config.group_headway_secs, "user"
            value = shared_headway * group_size
            origin = (
                "user",
                "input",
                "Explicit group_headway_secs multiplied by selected pattern count",
            )
        if value is None:
            value = _default("group_headway_secs" if field == "headway_secs" else field)
            origin = ("default", str(_REGISTRY["model"]), "")
            if field in {"start_time", "end_time"} and period is not None:
                value = period[0 if field == "start_time" else 1]
                origin = (
                    "source",
                    str(conditions["opening_hours"]["source"]),
                    "Source window evaluated for this weekday",
                )
            elif field == "headway_secs":
                if field in conditions:
                    value = conditions[field]["value"]
                    origin = (
                        "source",
                        str(conditions[field]["source"]),
                        "Source interval applies to this pattern",
                    )
                else:
                    shared_headway = int(value)
                    value = shared_headway * group_size
                    origin = (
                        "default",
                        str(_REGISTRY["model"]),
                        str(_REGISTRY["group_allocation"]),
                    )
        values[field], origins[field] = value, origin
    if _seconds(values["end_time"]) <= _seconds(values["start_time"]):
        msg = "Resolved end_time must exceed start_time; supply compatible explicit and source windows"
        raise ValueError(msg)
    return _ServicePeriod(values, origins, shared_headway, shared_kind)


def _conditional_periods(
    period: _ServicePeriod,
    conditions: dict[str, dict[str, Any]],
    weekday: int,
) -> list[_ServicePeriod]:
    """Intersect supported conditional intervals with the resolved service window.

    Overnight source rules are evaluated against neighbouring service-day clocks.

    Parameters
    ----------
    period : _ServicePeriod
        Effective values, field origins and frequency scope before splitting.
    conditions : dict
        Interpreted source fields.
    weekday : int
        Weekday of the starting service date.

    Returns
    -------
    list
        Disjoint periods and their effective field origins.
    """
    values, origins = period.row, period.origins
    source = conditions.get("interval:conditional")
    if source is None or origins["headway_secs"][0] == "user":
        return [period]
    week = json.loads(str(source["value"]))
    lower, upper = _seconds(values["start_time"]), _seconds(values["end_time"])
    intervals = []
    for offset in (-1, 0, 1):
        for rule in week[(weekday + offset) % 7]:
            start = _seconds(rule["start_time"]) + offset * 86400
            end = _seconds(rule["end_time"]) + offset * 86400
            if start < upper and end > lower:
                intervals.append((max(lower, start), min(upper, end), rule["headway_secs"]))
    boundaries = sorted(
        {lower, upper, *(point for start, end, _ in intervals for point in (start, end))}
    )
    result = []
    for start, end in pairwise(boundaries):
        resolved, adopted = dict(values), dict(origins)
        shared_headway = period.shared_headway
        resolved["start_time"], resolved["end_time"] = _clock(start), _clock(end)
        for field, point, original in (("start_time", start, lower), ("end_time", end, upper)):
            if point != original:
                adopted[field] = (
                    "estimated",
                    str(source["source"]),
                    "Intersection of conditional interval and operating window",
                )
        for first, last, headway in intervals:
            if first <= start and end <= last:
                shared_headway = None
                resolved["headway_secs"] = headway
                adopted["headway_secs"] = (
                    "source",
                    str(source["source"]),
                    "Conditional interval for this date and time",
                )
                break
        result.append(_ServicePeriod(resolved, adopted, shared_headway, period.shared_kind))
    return result


def _source_windows(
    conditions: dict[str, dict[str, Any]],
    weekday: int,
    *,
    enabled: bool,
) -> tuple[list[list[str] | None], bool]:
    """Apply source opening hours only when no complete explicit window exists.

    Closed dates remain reportable while fully explicit windows take precedence.

    Parameters
    ----------
    conditions : dict
        Interpreted source operating fields.
    weekday : int
        Weekday to evaluate.
    enabled : bool
        Whether source opening hours apply to this service.

    Returns
    -------
    tuple
        Source windows and whether the source permits operation.
    """
    if not enabled:
        return [None], True
    periods = json.loads(str(conditions["opening_hours"]["value"]))[weekday]
    return periods or [None], bool(periods)


def _service_trip_id(
    supplied: object,
    index: int,
    date: str | None,
    occupied: set[str],
    dated: dict[tuple[str, str], str],
) -> str:
    """Allocate generated trip identifiers without replacing explicit ones.

    Repeated periods of an explicitly named trip share the same derived daily ID.

    Parameters
    ----------
    supplied : object
        Explicit trip ID, or a missing value.
    index : int
        Stable service-row index used for unnamed trips.
    date : str or None
        Source-driven split date, or None for an unsplit service.
    occupied : set of str
        Reserved explicit IDs and previously allocated generated IDs.
    dated : dict
        Previously allocated IDs keyed by explicit trip and split date.

    Returns
    -------
    str
        Explicit or collision-free generated trip ID.
    """
    if _present(supplied) and date is None:
        return str(supplied)
    key = (str(supplied), date) if _present(supplied) and date is not None else None
    if key is not None and key in dated:
        return dated[key]
    base = f"{supplied}:c2g:{date}" if key is not None else f"service_{index}"
    candidate, suffix = base, 0
    while candidate in occupied:
        suffix += 1
        candidate = f"{base}:{suffix}"
    occupied.add(candidate)
    if key is not None:
        dated[key] = candidate
    return candidate


def _record_service(
    period: _ServicePeriod,
    supplied: dict[str, Any],
    context: _ServiceContext,
    records: list[dict[str, object]],
    *,
    index: int,
    date: str,
    split: bool,
    service_id: str,
    effective_days: dict[str, bool],
) -> dict[str, Any]:
    """
    Assign stable IDs and record an effective service period.

    The adopted values and their origins are consumed without resolving them again.

    Parameters
    ----------
    period : _ServicePeriod
        Effective service values, field origins and frequency scope.
    supplied : dict[str, Any]
        Original service row used to identify explicit input fields.
    context : _ServiceContext
        Resolved pattern/calendar context and generation-local ID reservations.
    records : list[dict[str, object]]
        Provenance records to extend in generation order.
    index : int
        Stable service-row position used to allocate generated identifiers.
    date : str
        Service date used for source-driven weekday splitting.
    split : bool
        Whether this period uses a date-specific service identifier.
    service_id : str
        Effective service identifier for this period.
    effective_days : dict[str, bool]
        Service dates mapped to their resolved active status.

    Returns
    -------
    dict
        Service row with deterministic trip and service identifiers.
    """
    resolved, adopted = period.row, period.origins
    row = supplied
    scope = f"service:{index}"
    resolved["service_id"] = service_id
    resolved["trip_id"] = _service_trip_id(
        row.get("trip_id"),
        index,
        date if split else None,
        context.trip_ids,
        context.dated_trip_ids,
    )
    for field, (kind, source, reason) in adopted.items():
        _assumption(records, scope, field, resolved[field], kind, source, reason)
    for field in ("pattern_id", "service_id", "trip_id"):
        explicit = (
            field in context.user_fields
            and _present(supplied.get(field))
            and not (split and field != "pattern_id")
        )
        _assumption(
            records,
            scope,
            field,
            resolved[field],
            "user" if explicit else "generated",
            "input" if explicit else str(_REGISTRY["model"]),
            "Source weekday splitting creates derived service/trip IDs"
            if split
            else "Input selection or deterministic identifier",
        )
    for field, value in (
        ("service_dates", json.dumps(effective_days)),
        ("service_group_id", context.group),
        ("group_pattern_count", context.size),
    ):
        _assumption(
            records,
            scope,
            field,
            value,
            "generated",
            str(_REGISTRY["model"]),
            "Resolved date and grouping context",
        )
    shared_frequency = period.shared_headway is not None
    if shared_frequency:
        _assumption(
            records,
            scope,
            "group_headway_secs",
            period.shared_headway,
            period.shared_kind,
            "input" if period.shared_kind == "user" else str(_REGISTRY["model"]),
        )
    _assumption(records, scope, "frequency_scope", period.frequency_scope, "generated")
    if shared_frequency and context.unresolved:
        _assumption(
            records,
            scope,
            "direction_resolution",
            "unresolved parent; one shared departure budget",
            "estimated",
            "OpenStreetMap:route_master",
            "No directional service split is inferred",
        )
    return resolved


def _resolved_row(
    supplied: dict[str, Any],
    context: _ServiceContext,
    config: _GenerationConfig,
    records: list[dict[str, object]],
) -> tuple[list[dict[str, Any]], list[dict[str, object]]]:
    """Resolve one input row across its dates and source operating periods.

    Weekly source conditions receive deterministic date-specific service identifiers.

    Parameters
    ----------
    supplied : dict
        Original service row before missing identifiers are filled.
    context : _ServiceContext
        Pattern, group, evaluated calendar and source information.
    config : _GenerationConfig
        Explicit global options.
    records : list of dict
        Effective assumptions to extend.

    Returns
    -------
    tuple
        Materialised service periods and generated calendar exceptions.
    """
    row, conditions = dict(supplied), context.conditions
    pattern_id = str(row["pattern_id"])
    base = str(row.get("service_id", context.base_id))
    days = context.days
    row["service_id"] = base
    complete_window = all(
        _present(row.get(key)) or getattr(config, key) is not None
        for key in ("start_time", "end_time")
    )
    fixed = _present(row.get("departure_time"))
    use_hours = "opening_hours" in conditions and not fixed and not complete_window
    use_conditional = (
        "interval:conditional" in conditions
        and not fixed
        and not (
            _present(row.get("headway_secs"))
            or config.headway_secs is not None
            or config.group_headway_secs is not None
        )
    )
    split = use_hours or use_conditional
    for field, used in (("opening_hours", use_hours), ("interval:conditional", use_conditional)):
        if field in conditions and not used:
            source = conditions[field]
            _assumption(
                records,
                pattern_id,
                field,
                source["value"],
                "ignored",
                str(source["source"]),
                "Explicit service windows or departures override source conditions",
            )
    selected_days = (
        list(days.items()) if split else [(next(iter(days), str(_default("service_date"))), True)]
    )
    output: list[dict[str, Any]] = []
    exceptions: list[dict[str, object]] = []
    for date, calendar_active in selected_days:
        weekday = datetime.strptime(date, "%Y%m%d").weekday()
        windows, source_active = _source_windows(conditions, weekday, enabled=use_hours)
        service_id = f"{base}:c2g:{pattern_id}:{date}" if split else base
        effective_days = {date: calendar_active and source_active} if split else days
        if split or (config.calendar is None and config.calendar_dates is None):
            exceptions.extend(
                {"service_id": service_id, "date": day, "exception_type": 1 if active else 2}
                for day, active in effective_days.items()
            )
        for window in windows:
            initial = _period_values(row, window, conditions, config, context.size)
            periods = (
                [initial]
                if not use_conditional or not source_active
                else _conditional_periods(initial, conditions, weekday)
            )
            for period in periods:
                resolved = _record_service(
                    period,
                    supplied,
                    context,
                    records,
                    index=context.offset + len(output),
                    date=date,
                    split=split,
                    service_id=service_id,
                    effective_days=effective_days,
                )
                output.append(resolved)
    return output, exceptions


def _resolve_services(
    data: TransportData,
    services: pd.DataFrame | None,
    config: _GenerationConfig,
    records: list[dict[str, object]],
    sources: dict[str, dict[str, dict[str, Any]]],
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Resolve target patterns, preserve explicit calendars and add derived dates.

    The caller frames are preserved and calendar ambiguity is never guessed away.

    Parameters
    ----------
    data : TransportData
        Independent validated transport input.
    services : pandas.DataFrame or None
        Explicit partial services or None for all patterns.
    config : _GenerationConfig
        Global overrides and calendars.
    records : list of dict
        Effective assumptions to extend.
    sources : dict[str, dict[str, dict[str, Any]]]
        Interpreted source records indexed once for this generation.

    Returns
    -------
    tuple
        Resolved service rows and combined date exceptions.
    """
    frame = pd.DataFrame([{}]) if services is None else services.copy()
    fields = set() if services is None else set(frame.columns)
    if "pattern_id" not in frame:
        if "trip_id" in frame:
            msg = "Specify pattern_id when providing explicit trip_id values"
            raise ValueError(msg)
        frame = data["patterns"][["pattern_id"]].merge(frame, how="cross")
    specified = set(frame["pattern_id"].astype(str))
    selected = set(data["patterns"]["pattern_id"].astype(str))
    groups = _service_groups(data, selected)
    unknown = specified - selected
    if unknown:
        msg = f"pattern_id: unknown IDs {sorted(unknown)}"
        raise ValueError(msg)
    completed = selected - specified
    if completed:
        frame = pd.concat(
            [
                frame,
                data["patterns"].loc[
                    data["patterns"]["pattern_id"].astype(str).isin(completed), ["pattern_id"]
                ],
            ],
            ignore_index=True,
        )
    calendar_ids = {
        str(value)
        for table in (config.calendar, config.calendar_dates)
        if table is not None
        for value in table["service_id"]
    }
    explicit_calendar = config.calendar is not None or config.calendar_dates is not None
    rows: list[dict[str, Any]] = []
    exceptions: list[dict[str, object]] = []
    trip_ids = {
        str(value) for value in frame.get("trip_id", pd.Series(dtype=object)) if _present(value)
    }
    dated_trip_ids: dict[tuple[str, str], str] = {}
    calendar_cache: dict[str, dict[str, bool]] = {}
    for row in _records(frame):
        identifier = str(row["pattern_id"])
        if not _present(row.get("service_id")):
            if explicit_calendar and len(calendar_ids) != 1:
                msg = "services.service_id: specify an ID when the explicit calendar is not unique"
                raise ValueError(msg)
            row.pop("service_id", None)
        group, size, unresolved = groups[identifier]
        base_id = str(
            row.get(
                "service_id",
                next(iter(calendar_ids)) if explicit_calendar else f"synthetic:{identifier}",
            )
        )
        calendar_key = base_id if explicit_calendar else ""
        if calendar_key not in calendar_cache:
            calendar_cache[calendar_key] = _calendar_days(config, base_id)
        context = _ServiceContext(
            conditions=sources.get(identifier, {}),
            group=group,
            size=size,
            unresolved=unresolved,
            offset=len(rows),
            trip_ids=trip_ids,
            dated_trip_ids=dated_trip_ids,
            user_fields=set() if identifier in completed else fields,
            base_id=base_id,
            days=calendar_cache[calendar_key],
        )
        resolved, row_dates = _resolved_row(row, context, config, records)
        rows.extend(resolved)
        exceptions.extend(row_dates)
    frames = [pd.DataFrame(exceptions)] if exceptions else []
    if config.calendar_dates is not None:
        frames.insert(0, config.calendar_dates)
    dates = (
        pd.concat(frames, ignore_index=True).drop_duplicates(["service_id", "date"])
        if frames
        else None
    )
    return pd.DataFrame(rows), dates
