"""Interpret OSM and Overture input through explicit retrieval operations."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from ast import literal_eval
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from itertools import pairwise
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import Unpack
from typing import cast

import geopandas as gpd
import pandas as pd
from overturemaps.core import ALL_RELEASES
from shapely.geometry import LineString
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import Polygon

from .feed import _input_provenance
from .generation import create_gtfs
from .schema import _INPUT_KEYS
from .schema import _MODES
from .schema import _PROVENANCE_COLUMNS
from .schema import GenerationOptions
from .schema import TransportData
from .schema import _clock
from .schema import _default
from .schema import _present
from .schema import _seconds
from .validation import _matching_patterns
from .validation import _normalise_stops
from .validation import _select_patterns
from .validation import _selection
from .validation import _transport_modes
from .validation import _validate_generation_options
from .validation import _without_stop_wkb

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from collections.abc import Sequence

    import duckdb

_DAYS = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]


def _interval(value: str) -> int:
    """
    Parse integer minutes or hour-minute-second headways without guessing units.

    Invalid and non-positive source values are rejected as a whole.

    Parameters
    ----------
    value : str
        Source interval tag.

    Returns
    -------
    int
        Positive headway in seconds.
    """
    if re.fullmatch(r"\d+", value):
        result = int(value) * 60
    elif re.fullmatch(r"\d+:\d{2}(?::\d{2})?", value):
        parts = [int(part) for part in value.split(":")]
        if any(part >= 60 for part in parts[1:]):
            msg = "Invalid interval minutes or seconds"
            raise ValueError(msg)
        result = parts[0] * 3600 + parts[1] * 60 + (parts[2] if len(parts) == 3 else 0)
    else:
        msg = "Unsupported interval; expected integer minutes, HH:MM or HH:MM:SS"
        raise ValueError(msg)
    if result <= 0:
        msg = "interval must be positive"
        raise ValueError(msg)
    return result


def _weekdays(selector: str | None) -> set[int]:
    """
    Expand weekday lists and ranges, rejecting repeated days.

    Missing selectors cover the whole week, beginning with Monday.

    Parameters
    ----------
    selector : str or None
        Weekday list or range, such as Mo,We-Fr.

    Returns
    -------
    set of int
        Weekday indices from zero to six.
    """
    result: set[int] = set()
    for item in (selector or "Mo-Su").split(","):
        bounds = item.split("-")
        if len(bounds) not in {1, 2} or any(day not in _DAYS for day in bounds):
            msg = "Unsupported weekday selector"
            raise ValueError(msg)
        first, last = _DAYS.index(bounds[0]), _DAYS.index(bounds[-1])
        days = (
            {first}
            if len(bounds) == 1
            else {(first + offset) % 7 for offset in range((last - first) % 7 + 1)}
        )
        if result & days:
            msg = "Overlapping weekday selectors are unsupported"
            raise ValueError(msg)
        result |= days
    return result


def _opening_hours(value: str) -> list[list[tuple[str, str]]]:
    """
    Parse fixed weekly periods, rejecting the entire tag for unsupported grammar.

    Overnight periods belong to their starting service day and use extended hours.

    Parameters
    ----------
    value : str
        Source opening_hours tag.

    Returns
    -------
    list
        Seven lists of fixed periods, with an empty list meaning no operation.
    """
    if value == "24/7":
        return [[("00:00:00", "24:00:00")] for _ in range(7)]
    week: list[list[tuple[str, str]]] = [[] for _ in range(7)]
    assigned: set[int] = set()
    for clause in value.split(";"):
        match = re.fullmatch(
            r"(?:(?P<days>[A-Za-z,\-]+)\s+)?(?P<hours>off|closed|\d{2}:\d{2}-\d{2}:\d{2}(?:,\s*\d{2}:\d{2}-\d{2}:\d{2})*)",
            clause.strip(),
        )
        if match is None:
            msg = (
                "Unsupported opening_hours; only weekdays, fixed time ranges and off are supported"
            )
            raise ValueError(msg)
        days = _weekdays(match["days"])
        if assigned & days:
            msg = "Overlapping weekday rules are unsupported"
            raise ValueError(msg)
        assigned |= days
        periods: list[tuple[int, int]] = []
        if match["hours"] not in {"off", "closed"}:
            for interval in match["hours"].split(","):
                start, end = [
                    tuple(map(int, clock.split(":"))) for clock in interval.strip().split("-")
                ]
                if (
                    start[0] >= 24
                    or start[1] >= 60
                    or end[0] > 24
                    or end[1] >= 60
                    or (end[0] == 24 and end[1] != 0)
                ):
                    msg = "Invalid opening_hours clock value"
                    raise ValueError(msg)
                lower, upper = start[0] * 3600 + start[1] * 60, end[0] * 3600 + end[1] * 60
                periods.append((lower, upper if upper > lower else upper + 86400))
            periods.sort()
            if any(left[1] > right[0] for left, right in pairwise(periods)):
                msg = "Overlapping opening_hours intervals are unsupported"
                raise ValueError(msg)
        for day in days:
            week[day] = [(_clock(lower), _clock(upper)) for lower, upper in periods]
    _check_weekly_overlap(week)
    return week


def _check_weekly_overlap(week: list[list[tuple[str, str]]]) -> None:
    """Reject physical overlaps, including overnight and Sunday-to-Monday rules.

    A repeated weekly coordinate permits checks at the Sunday-to-Monday boundary.

    Parameters
    ----------
    week : list
        Seven lists of service-day time intervals.
    """
    intervals = [
        (day * 86400 + _seconds(start), day * 86400 + _seconds(end))
        for day, periods in enumerate(week)
        for start, end in periods
    ]
    intervals.extend(
        (start - 604800, end - 604800) for start, end in list(intervals) if end > 604800
    )
    intervals.sort()
    if any(left[1] > right[0] for left, right in pairwise(intervals)):
        msg = "Overlapping weekly or overnight source intervals are unsupported"
        raise ValueError(msg)


def _conditional_intervals(value: str) -> list[list[dict[str, object]]]:
    """Read disjoint weekday and fixed-clock conditional intervals.

    Unsupported conditions reject the complete family instead of its fragments.

    Parameters
    ----------
    value : str
        Semicolon-separated ``interval @ (condition)`` clauses.

    Returns
    -------
    list
        Seven lists of periods with a positive headway.
    """
    week: list[list[dict[str, object]]] = [[] for _ in range(7)]
    for clause in value.split(";"):
        match = re.fullmatch(r"\s*([^@;]+?)\s*@\s*\(([^()]+)\)\s*", clause)
        if match is None:
            msg = "Unsupported interval:conditional; expected interval @ (weekdays and fixed times)"
            raise ValueError(msg)
        headway = _interval(match[1].strip())
        condition = match[2].strip()
        if re.fullmatch(r"[A-Za-z,\-]+", condition):
            condition += " 00:00-24:00"
        for day, periods in enumerate(_opening_hours(condition)):
            if not periods and condition.endswith(("off", "closed")):
                msg = "Conditional intervals must specify positive headways and operating times"
                raise ValueError(msg)
            week[day].extend(
                {"start_time": start, "end_time": end, "headway_secs": headway}
                for start, end in periods
            )
    for conditional_periods in week:
        conditional_periods.sort(key=lambda row: str(row["start_time"]))
        if any(
            str(left["end_time"]) > str(right["start_time"])
            for left, right in pairwise(conditional_periods)
        ):
            msg = "Overlapping conditional intervals are unsupported"
            raise ValueError(msg)
    _check_weekly_overlap(
        [[(str(row["start_time"]), str(row["end_time"])) for row in periods] for periods in week]
    )
    return week


def _osm_parents(
    relations: list[dict[str, Any]], masters: list[dict[str, Any]]
) -> dict[int, list[dict[str, Any]]]:
    """Associate selected routes with their immediate route masters.

    Multiple candidate parents are retained for explicit ambiguity reporting.

    Parameters
    ----------
    relations : list of dict
        Selected route relations.
    masters : list of dict
        Retrieved parent relations.

    Returns
    -------
    dict
        Route IDs mapped to all candidate parents.
    """
    result: dict[int, list[dict[str, Any]]] = {row["id"]: [] for row in relations}
    for master in masters:
        for member in master.get("members", []):
            if member.get("type") == "relation" and member.get("ref") in result:
                result[member["ref"]].append(master)
    return result


def _route_operating_context(
    data: TransportData,
    relation: dict[str, Any],
    candidates: list[dict[str, Any]],
    records: list[dict[str, object]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Inherit only unambiguous parent fields and retain structural uncertainty.

    Parent membership is retained separately from an unknown directional split.

    Parameters
    ----------
    data : TransportData
        Prepared independent input.
    relation : dict
        Selected child route.
    candidates : list of dict
        Immediate route-master candidates.
    records : list of dict
        Source and rejection history to extend.

    Returns
    -------
    tuple
        Effective source tags and the unique parent, if any.
    """
    identifier = f"osm:relation:{relation['id']}"
    own = relation.get("tags", {})
    parent = candidates[0] if len(candidates) == 1 else None
    inherited = parent.get("tags", {}) if parent else {}
    tags = {**inherited, **own}
    if any(field in own for field in ("interval", "interval:conditional")):
        for field in ("interval", "interval:conditional"):
            if field not in own:
                tags.pop(field, None)
    if "duration" not in own:
        tags.pop("duration", None)
    selected = data["patterns"]["pattern_id"].astype(str).eq(identifier)
    current = data["patterns"].loc[selected].iloc[0]
    if parent and not _present(current.get("service_group_id")):
        group = f"osm:route_master:{parent['id']}"
        for field, value, kind in (
            ("service_group_id", group, "source"),
            ("service_group_direction", "unresolved", "estimated"),
        ):
            data["patterns"].loc[selected, field] = value
            records.append(
                {
                    "table_name": "patterns",
                    "record_id": identifier,
                    "field": field,
                    "kind": kind,
                    "source": f"OpenStreetMap:route_master:{parent['id']}",
                    "value": value,
                    "details": "Parent membership is known; directional service split is not inferred",
                }
            )
    if "duration" in inherited and "duration" not in own:
        records.append(
            {
                "table_name": "operation_source",
                "record_id": identifier,
                "field": "duration",
                "kind": "ignored",
                "source": "OpenStreetMap:route_master",
                "value": str(inherited["duration"]),
                "details": "Parent duration cannot be assigned to potentially different route variants",
            }
        )
    if len(candidates) > 1:
        records.append(
            {
                "table_name": "operation_source",
                "record_id": identifier,
                "field": "route_master",
                "kind": "ignored",
                "source": "OpenStreetMap",
                "value": json.dumps([row["id"] for row in candidates]),
                "details": "Multiple route_master candidates; common metadata and grouping not inherited",
            }
        )
    for field, keys in {
        "agency_name": ("operator",),
        "agency_url": ("operator:website", "website", "url"),
        "agency_timezone": ("timezone",),
    }.items():
        key = next((key for key in keys if _present(tags.get(key))), None)
        if key is not None:
            origin = f"OpenStreetMap:{key}"
            if parent and key not in own:
                origin += f":route_master:{parent['id']}"
            records.append(
                {
                    "table_name": "operation_source",
                    "record_id": identifier,
                    "field": field,
                    "kind": "source",
                    "value": str(tags[key]),
                    "source": origin,
                    "details": json.dumps({"raw": tags[key], "inherited": key not in own}),
                }
            )
    return tags, parent


def _operating_value(tag: str, tags: dict[str, Any]) -> tuple[str, str]:
    """Parse one source field, validating the complete interval family together.

    Raw tag text is preserved separately from each parsed effective value.

    Parameters
    ----------
    tag : str
        Operating tag to interpret.
    tags : dict
        Effective source tags after bounded inheritance.

    Returns
    -------
    tuple
        Effective field name and serialised source value.
    """
    value = str(tags[tag])
    if tag in {"interval", "interval:conditional"}:
        interval = _interval(str(tags["interval"])) if "interval" in tags else None
        conditional = (
            _conditional_intervals(str(tags["interval:conditional"]))
            if "interval:conditional" in tags
            else None
        )
        return (
            ("headway_secs", str(interval)) if tag == "interval" else (tag, json.dumps(conditional))
        )
    if tag == "duration":
        if not re.fullmatch(r"\d+:\d{2}(?::\d{2})?", value):
            msg = "Ambiguous duration; expected H:MM or H:MM:SS"
            raise ValueError(msg)
        return "duration_secs", str(_interval(value))
    return tag, json.dumps(_opening_hours(value))


def _add_osm_operating(
    data: TransportData,
    relations: list[dict[str, Any]],
    *,
    enabled: bool,
    masters: list[dict[str, Any]] | None = None,
) -> None:
    """Attach source operating values without replacing explicit user conditions.

    Source history remains distinct from explicit overrides and default assumptions.

    Parameters
    ----------
    data : TransportData
        Prepared input whose provenance is extended.
    relations : list of dict
        Selected OSM route relations.
    enabled : bool
        Whether operating tags can supply values below explicit overrides.
    masters : list of dict, optional
        Immediate parent relations for bounded inheritance.
    """
    records: list[dict[str, object]] = []
    identifiers = set(data["patterns"]["pattern_id"].astype(str))
    parents = _osm_parents(relations, masters or [])
    data["patterns"] = data["patterns"].copy()
    for relation in relations:
        identifier = f"osm:relation:{relation['id']}"
        if identifier not in identifiers:
            continue
        tags, parent = _route_operating_context(data, relation, parents[relation["id"]], records)
        for tag in ("interval", "interval:conditional", "opening_hours", "duration"):
            if tag not in tags:
                continue
            field: str = tag
            value, kind = str(tags[tag]), "ignored"
            reason = "Source operating tags disabled by use_source_tags=False"
            if enabled:
                try:
                    field, value = _operating_value(tag, tags)
                    kind, reason = "source", ""
                except ValueError as exc:
                    reason = str(exc)
                    if (
                        tag in {"interval", "interval:conditional"}
                        and "interval:conditional" in tags
                    ):
                        reason = f"Conditional interval family rejected as a whole: {reason}"
            origin = f"OpenStreetMap:{tag}"
            if parent and tag not in relation.get("tags", {}):
                origin += f":route_master:{parent['id']}"
            records.append(
                {
                    "table_name": "operation_source",
                    "record_id": identifier,
                    "field": field,
                    "kind": kind,
                    "source": origin,
                    "value": value,
                    "details": json.dumps({"raw": tags[tag], "reason": reason}),
                }
            )
    data["provenance"] = pd.concat(
        [data["provenance"], pd.DataFrame(records, columns=_PROVENANCE_COLUMNS)], ignore_index=True
    )


_OSM_ENDPOINT = str(_default("endpoint"))


_OSM_TIMEOUT = float(_default("timeout"))


_USE_SOURCE_TAGS = bool(_default("use_source_tags"))


_OVERWRITE_SOURCE = bool(_default("overwrite_source"))


def _osm_failures(issues: dict[str, str], usable: list[str], on_invalid: str) -> None:
    """Report source structure failures together without hiding empty results.

    Every failure includes its source identifier and repair instructions.

    Parameters
    ----------
    issues : dict of str to str
        Unusable pattern IDs and actionable source diagnostics.
    usable : list of str
        Remaining usable pattern IDs.
    on_invalid : str
        Raise by default, or explicitly skip invalid source patterns.
    """
    if issues and (on_invalid == "raise" or not usable):
        msg = "Invalid OSM relations:\n" + "\n".join(
            f"- {key}: {value}" for key, value in issues.items()
        )
        relation_ids = [
            int(key.rsplit(":", 1)[1])
            for key in usable
            if key.startswith("osm:relation:") and key.rsplit(":", 1)[1].isdigit()
        ]
        msg += f"\nUsable relation_ids: {relation_ids}"
        if usable:
            msg += "; correct the listed inputs or set on_invalid='skip' to exclude them"
        else:
            msg += "; no usable relations remain"
        raise ValueError(msg)


def _filter_source_services(
    services: pd.DataFrame | None, known: set[str], selected: set[str]
) -> pd.DataFrame | None:
    """Exclude service overrides outside a source selection.

    Unknown identifiers remain errors rather than disappearing through filtering.

    Parameters
    ----------
    services : pandas.DataFrame or None
        Explicit service overrides.
    known : set of str
        Known source or supplemental pattern IDs.
    selected : set of str
        Retained pattern IDs.

    Returns
    -------
    pandas.DataFrame or None
        Selected rows; an empty selection becomes default completion.
    """
    if services is None or "pattern_id" not in services:
        return services
    if unknown := set(services["pattern_id"].astype(str)) - known:
        msg = f"services.pattern_id: unknown IDs {sorted(unknown)}"
        raise ValueError(msg)
    result = services.loc[services["pattern_id"].astype(str).isin(selected)].copy()
    return None if result.empty else result


def _osm_supplement(
    data: Mapping[str, pd.DataFrame] | None, known: set[str], selected: set[str]
) -> dict[str, pd.DataFrame] | None:
    """Keep supplemental source records within the requested selection.

    Validate table keys before removing deliberately excluded source rows.

    Parameters
    ----------
    data : mapping or None
        Supplemental tables.
    known : set of str
        All retrieved source pattern identifiers.
    selected : set of str
        Source patterns retained for generation.

    Returns
    -------
    dict or None
        Supplemental tables without deliberately excluded source records.
    """
    if data is None:
        return None
    result = dict(data)
    for name, frame in data.items():
        if name not in _INPUT_KEYS:
            msg = f"Unsupported supplemental table {name!r}"
            raise ValueError(msg)
        keys = _INPUT_KEYS[name]
        if (
            not set(keys).issubset(frame)
            or frame.duplicated(keys).any()
            or not frame[keys].map(_present).all().all()
        ):
            msg = f"{name}: supply unique keys {keys}"
            raise ValueError(msg)
        field = (
            "pattern_id"
            if name in {"patterns", "stop_sequences"}
            else "route_id"
            if name == "routes"
            else None
        )
        if field:
            result[name] = frame.loc[~frame[field].astype(str).isin(known - selected)].copy()
    return result


@dataclass(frozen=True)
class _SourceIO:
    """Data facade operations supplied at the source boundary.

    Adapters use these operations without importing the public data module.
    """

    prepare_area: Callable[..., tuple[str, Polygon | MultiPolygon | None]]
    boundaries: Callable[[str], gpd.GeoDataFrame]
    overture: Callable[..., dict[str, gpd.GeoDataFrame]]


def _area(
    io: _SourceIO,
    area: list[float] | Polygon | MultiPolygon | gpd.GeoSeries | gpd.GeoDataFrame | None,
    place_name: str | None,
) -> tuple[str, Polygon | MultiPolygon | None]:
    """
    Resolve a geographical selection using the existing data utilities.

    Geographical tabular inputs must declare a CRS before conversion to WGS84.

    Parameters
    ----------
    io : _SourceIO
        Geographical lookup and download operations supplied by the data facade.
    area : list[float] | Polygon | MultiPolygon | gpd.GeoSeries | gpd.GeoDataFrame | None
        Bounding box or polygon selecting the area of interest.
    place_name : str | None
        Place name to resolve to a boundary instead of area.

    Returns
    -------
    tuple[str, Polygon | MultiPolygon | None]
        WGS84 bounding-box text and an optional precise polygon.
    """
    if (area is None) == (place_name is None):
        msg = "Provide exactly one of area and place_name"
        raise ValueError(msg)
    if place_name is not None:
        area = io.boundaries(place_name)
    if isinstance(area, (gpd.GeoSeries, gpd.GeoDataFrame)):
        if area.crs is None:
            msg = "area: a CRS is required"
            raise ValueError(msg)
        area = area.to_crs(4326).geometry.union_all()
    if area is None:
        msg = "area must not be empty"
        raise ValueError(msg)
    bbox, polygon = io.prepare_area(area)
    values = [float(value) for value in bbox.split(",")]
    if len(values) != 4 or not (
        -180 <= values[0] < values[2] <= 180 and -90 <= values[1] < values[3] <= 90
    ):
        msg = "area: expected [west, south, east, north] in WGS84"
        raise ValueError(msg)
    return bbox, polygon


def _source_record(source: str, details: dict[str, object]) -> dict[str, object]:
    """
    Record the request, source version and retrieval time.

    Store request context separately from assumptions about service operation.

    Parameters
    ----------
    source : str
        Name of the upstream data provider.
    details : dict[str, object]
        Source version and request parameters to retain.

    Returns
    -------
    dict[str, object]
        One source provenance record.
    """
    return {
        "table_name": "source",
        "record_id": source,
        "field": "request",
        "kind": "source",
        "source": source,
        "value": datetime.now(UTC).isoformat(),
        "details": json.dumps(details, default=str),
    }


def _complete_input(
    base: dict[str, pd.DataFrame],
    supplied: Mapping[str, pd.DataFrame] | None,
    *,
    overwrite_source: bool,
    source: str,
    metadata: dict[str, object],
) -> TransportData:
    """
    Fill missing source fields, requiring an explicit switch for replacement.

    Keep original values in provenance when an explicit override replaces them.

    Parameters
    ----------
    base : dict[str, pd.DataFrame]
        Retrieved tables prepared for the shared generator.
    supplied : Mapping[str, pd.DataFrame] | None
        Explicit supplemental fields and user-defined patterns.
    overwrite_source : bool
        Permit explicit replacement of source values.
    source : str
        Name of the upstream data provider.
    metadata : dict[str, object]
        Request and data-version information recorded with the feed.

    Returns
    -------
    TransportData
        Common generator input with supplementation history.
    """
    provenance = _input_provenance(base)
    for record in provenance:
        record.update(kind="source", source=source)
    provenance.append(_source_record(source, metadata))
    for name, supplied_frame in (supplied or {}).items():
        incoming = _without_stop_wkb(supplied_frame) if name == "stops" else supplied_frame
        if isinstance(incoming, gpd.GeoDataFrame):
            if incoming.crs is None:
                msg = f"{name}: supplemental geometry requires a CRS"
                raise ValueError(msg)
            incoming = incoming.to_crs(4326)
        if name not in _INPUT_KEYS:
            msg = f"Unsupported supplemental table {name!r}"
            raise ValueError(msg)
        keys = _INPUT_KEYS[name]
        if not set(keys).issubset(incoming.columns) or incoming.duplicated(keys).any():
            msg = f"{name}: supply unique keys {keys}"
            raise ValueError(msg)
        existing = base.get(name, pd.DataFrame(columns=keys)).copy()
        # A declared stopping pattern replaces the entire ordered sequence,
        # avoiding stale source stops when users define express services.
        if name == "stop_sequences" and overwrite_source and not existing.empty:
            existing = existing[~existing["pattern_id"].isin(incoming["pattern_id"])]
        rows = existing.to_dict("records")
        lookup = {tuple(str(row[key]) for key in keys): row for row in rows}
        for row in incoming.to_dict("records"):
            identity = tuple(str(row[key]) for key in keys)
            target = lookup.get(identity)
            if target is None:
                target = dict(zip(keys, identity, strict=True))
                rows.append(target)
                lookup[identity] = target
            for field, value in row.items():
                if not _present(value):
                    continue
                old = target.get(field)
                changed = _present(old) and str(old) != str(value)
                if changed and not overwrite_source:
                    msg = f"{name} {identity}, {field}: source conflict; use overwrite_source=True to replace explicitly"
                    raise ValueError(msg)
                target[field] = value
                provenance.append(
                    {
                        "table_name": name,
                        "record_id": "|".join(identity),
                        "field": field,
                        "kind": "override" if changed else "user",
                        "source": "input",
                        "value": str(value),
                        "details": json.dumps({"previous": str(old)}) if changed else "",
                    }
                )
        frame = pd.DataFrame(rows)
        if "geometry" in frame:
            frame = gpd.GeoDataFrame(frame, geometry="geometry", crs=4326)
        base[name] = frame
    base["provenance"] = pd.DataFrame(provenance, columns=_PROVENANCE_COLUMNS)
    return cast("TransportData", base)


def _request_osm(query: str, endpoint: str, timeout: float) -> dict[str, Any]:
    """
    Fetch one bounded Overpass response and distinguish transport failures.

    A server remark can indicate a truncated response and is treated as a failure.

    Parameters
    ----------
    query : str
        Overpass QL query selecting complete route relations.
    endpoint : str
        HTTP or HTTPS Overpass API endpoint.
    timeout : float
        HTTP request timeout in seconds.

    Returns
    -------
    dict[str, Any]
        Complete parsed Overpass response.
    """
    if urllib.parse.urlparse(endpoint).scheme not in {"http", "https"}:
        msg = "Overpass endpoint must use HTTP or HTTPS"
        raise ValueError(msg)
    request = urllib.request.Request(  # noqa: S310 - scheme validated above
        endpoint,
        data=urllib.parse.urlencode({"data": query}).encode(),
        headers={
            "User-Agent": "city2graph/transport",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        msg = f"OSM retrieval failed: {exc}"
        raise RuntimeError(msg) from exc
    if not isinstance(payload, dict) or "elements" not in payload or payload.get("remark"):
        msg = f"OSM returned an incomplete response: {payload.get('remark') if isinstance(payload, dict) else payload}"
        raise RuntimeError(msg)
    return cast("dict[str, Any]", payload)


def _chain_lines(lines: list[LineString]) -> LineString | None:
    """
    Join an explicitly ordered way sequence without routing across gaps.

    Only connected endpoints in the supplied order may be joined.

    Parameters
    ----------
    lines : list[LineString]
        Directed route alignments indexed by pattern_id.

    Returns
    -------
    LineString | None
        Continuous alignment, or None if the supplied order has gaps.
    """
    if not lines:
        return None
    for reverse in (False, True):
        coordinates = list(lines[0].coords)
        if reverse:
            coordinates.reverse()
        for line in lines[1:]:
            following = list(line.coords)
            if coordinates[-1] == following[-1]:
                following.reverse()
            if coordinates[-1] != following[0]:
                break
            coordinates.extend(following[1:])
        else:
            return LineString(coordinates)
    return None


def _osm_pattern(
    relation: dict[str, Any],
    elements: dict[tuple[str, int], dict[str, Any]],
    *,
    sequence_override: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]], LineString | None]:
    """
    Interpret a PTv2 relation, preserving member order and stop occurrences.

    Resolve boarding occurrences before assembling an optional contiguous path.

    Parameters
    ----------
    relation : dict[str, Any]
        OSM route relation with ordered members.
    elements : dict[tuple[str, int], dict[str, Any]]
        Retrieved OSM elements indexed by type and identifier.
    sequence_override : bool
        Allow unresolved source stops when a replacement sequence is supplied.

    Returns
    -------
    tuple[list[dict[str, object]], list[dict[str, object]], LineString | None]
        Boarding locations, ordered stop occurrences and optional route alignment.
    """
    pattern_id = f"osm:relation:{relation['id']}"
    if relation.get("tags", {}).get("public_transport:version") != "2" and not sequence_override:
        msg = (
            f"{pattern_id}: stop order requires a PTv2 relation; "
            "supply stop_sequences and overwrite_source=True for other tagging schemes"
        )
        raise ValueError(msg)
    stops: list[dict[str, object]] = []
    sequence: list[dict[str, object]] = []
    lines: list[LineString] = []
    previous_stop: dict[str, Any] | None = None
    for member in relation.get("members", []):
        element = elements.get((member["type"], member["ref"]), member)
        role = member.get("role", "")
        if role.startswith(("stop", "platform")):
            tags = element.get("tags", {})
            if role.startswith("platform") and previous_stop is not None:
                previous_tags = previous_stop.get("tags", {})
                same = any(
                    tags.get(key) and tags.get(key) == previous_tags.get(key)
                    for key in ("ref", "name")
                )
                if not same and not sequence_override:
                    msg = f"{pattern_id}: ambiguous stop/platform pair; supply stop_sequences and overwrite_source=True"
                    raise ValueError(msg)
                if same:
                    previous_stop = None
                    continue
            previous_stop = element if role.startswith("stop") else None
            if "lat" not in element or "lon" not in element:
                if sequence_override:
                    continue
                msg = (
                    f"{pattern_id}: boarding location for {member['type']} {member['ref']} is not a point; "
                    "supply stops and stop_sequences"
                )
                raise ValueError(msg)
            stop_id = f"osm:{member['type']}:{member['ref']}"
            stops.append(
                {
                    "stop_id": stop_id,
                    "stop_name": tags.get("name"),
                    "geometry": Point(element["lon"], element["lat"]),
                }
            )
            sequence.append(
                {
                    "pattern_id": pattern_id,
                    "stop_id": stop_id,
                    "stop_sequence": len(sequence),
                    "pickup_type": 1 if role.endswith("exit_only") else 0,
                    "drop_off_type": 1 if role.endswith("entry_only") else 0,
                }
            )
        elif member["type"] == "way" and role in {"", "forward", "backward"}:
            coords = member.get("geometry") or element.get("geometry")
            if coords:
                line = LineString([(coord["lon"], coord["lat"]) for coord in coords])
                lines.append(LineString(list(line.coords)[::-1]) if role == "backward" else line)
            else:
                lines.append(LineString())
    shape = None if any(line.is_empty for line in lines) else _chain_lines(lines)
    if (
        shape is not None
        and stops
        and shape.project(stops[0]["geometry"]) > shape.project(stops[-1]["geometry"])
    ):
        shape = LineString(list(shape.coords)[::-1])
    return stops, sequence, shape


def _source_stop_issues(
    prepared: TransportData,
    data: Mapping[str, pd.DataFrame] | None,
    sequences: pd.DataFrame,
    issues: dict[str, str],
) -> None:
    """
    Validate shared stops and localise source failures.

    Successful shared validation avoids repeating checks for each pattern.

    Parameters
    ----------
    prepared : TransportData
        Combined source and supplemental transport input.
    data : Mapping[str, pd.DataFrame] | None
        Explicit transport tables or supplemental input.
    sequences : pd.DataFrame
        Stop occurrences with string pattern identifiers.
    issues : dict[str, str]
        Source structural failures to extend with contextual reasons.
    """
    try:
        _normalise_stops(prepared["stops"])
    except ValueError:
        # Error messages must retain the original rows of each affected pattern.
        pass
    else:
        return
    pattern_stops = {
        identifier: frozenset(group["stop_id"].astype(str))
        for identifier, group in sequences.groupby("pattern_id", sort=False)
    }
    supplied = (
        set(data["stops"]["stop_id"].astype(str)) if data is not None and "stops" in data else set()
    )
    stop_ids = prepared["stops"]["stop_id"].astype(str)
    checked: dict[frozenset[str], ValueError | None] = {}
    for identifier in prepared["patterns"]["pattern_id"].astype(str):
        wanted = pattern_stops.get(identifier, frozenset())
        if wanted not in checked:
            try:
                _normalise_stops(prepared["stops"].loc[stop_ids.isin(wanted)])
            except ValueError as exc:
                checked[wanted] = exc
            else:
                checked[wanted] = None
        if (error := checked[wanted]) is not None:
            if supplied & wanted:
                raise error
            issues[identifier] = f"{error}; supply corrected stops"


def _finish_osm_selection(
    prepared: TransportData,
    services: pd.DataFrame | None,
    data: Mapping[str, pd.DataFrame] | None,
    issues: dict[str, str],
    on_invalid: str,
) -> tuple[TransportData, pd.DataFrame | None, list[str]]:
    """Validate supplemented stopping sequences and persist source exclusions.

    Explicit input errors remain fatal even when invalid source relations are skipped.

    Parameters
    ----------
    prepared : TransportData
        Combined source and supplemental input.
    services : pandas.DataFrame or None
        Service overrides within the source selection.
    data : mapping or None
        Original supplemental input.
    issues : dict
        Source structural failures collected so far.
    on_invalid : str
        Source error policy.

    Returns
    -------
    tuple
        Valid input, selected services and usable pattern IDs.
    """
    history = prepared["provenance"]
    source_patterns = set(
        history.loc[
            history["table_name"].eq("patterns")
            & history["field"].eq("pattern_id")
            & history["kind"].eq("source"),
            "record_id",
        ]
    )
    routes = {str(row["route_id"]): row for row in prepared["routes"].to_dict("records")}
    sequences = prepared["stop_sequences"].assign(
        pattern_id=lambda frame: frame["pattern_id"].astype(str)
    )
    sequence_counts = sequences.groupby("pattern_id", sort=False).size()
    explicit_sequences = (
        set(data["stop_sequences"]["pattern_id"].astype(str))
        if data is not None and "stop_sequences" in data
        else set()
    )
    for pattern in prepared["patterns"].itertuples():
        identifier, route_id = str(pattern.pattern_id), pattern.route_id
        route = routes.get(str(route_id))
        if route is not None and not any(
            _present(route.get(field)) for field in ("route_short_name", "route_long_name")
        ):
            reason = (
                f"routes {route_id}: supply route_short_name or route_long_name in data['routes']"
            )
            # User-defined routes are explicit input, not skippable source failures.
            if identifier not in source_patterns:
                raise ValueError(reason)
            issues.setdefault(identifier, reason)
        if sequence_counts.get(identifier, 0) < 2:
            reason = "supply at least two ordered stop occurrences using stops and stop_sequences"
            explicit_sequence = identifier in explicit_sequences
            if explicit_sequence or identifier not in source_patterns:
                msg = f"stop_sequences, pattern {identifier}: {reason}"
                raise ValueError(msg)
            issues.setdefault(identifier, reason)
    _source_stop_issues(prepared, data, sequences, issues)
    usable = [
        str(value) for value in prepared["patterns"]["pattern_id"] if str(value) not in issues
    ]
    _osm_failures(issues, usable, on_invalid)
    if issues:
        extra: list[dict[str, object]] = [
            {
                "table_name": "operation_source",
                "record_id": key,
                "field": "relation",
                "kind": "ignored",
                "source": "OpenStreetMap",
                "value": key,
                "details": f"Excluded by on_invalid='skip': {reason}",
            }
            for key, reason in issues.items()
        ]
        services = _filter_source_services(services, set(usable) | set(issues), set(usable))
        prepared, services = _select_patterns(
            cast("Mapping[str, pd.DataFrame]", prepared), services, usable, extra, explicit=False
        )
        prepared["provenance"] = pd.concat(
            [prepared["provenance"], pd.DataFrame(extra)], ignore_index=True
        )
    return prepared, services, usable


def _select_completed_osm(
    prepared: TransportData,
    services: pd.DataFrame | None,
    selected_modes: list[str] | None,
    requested: list[str] | None,
) -> tuple[TransportData, pd.DataFrame | None, list[str]]:
    """Apply selection to effective routes after supplemental input is merged.

    User-defined patterns obey the same mode and identifier filters as source patterns.

    Parameters
    ----------
    prepared : TransportData
        Combined source and supplemental tables.
    services : pandas.DataFrame or None
        Overrides for known source or supplemental patterns.
    selected_modes : list of str or None
        Mode filter, or None to retain all supplemental route types.
    requested : list of str or None
        Explicit pattern selection.

    Returns
    -------
    tuple
        Selected tables, service overrides and retained pattern IDs.
    """
    # Supplementation can introduce patterns or replace a route's mode. Apply
    # the effective selection again before validating any source structure.
    chosen = _matching_patterns(
        cast("Mapping[str, pd.DataFrame]", prepared), selected_modes, requested
    )
    if not chosen:
        msg = "No matching OSM patterns after mode and pattern selection"
        raise ValueError(msg)
    selection_records: list[dict[str, object]] = []
    prepared, services = _select_patterns(
        cast("Mapping[str, pd.DataFrame]", prepared),
        services,
        chosen,
        selection_records,
        explicit=requested is not None,
    )
    prepared["provenance"] = pd.concat(
        [prepared["provenance"], pd.DataFrame(selection_records)], ignore_index=True
    )
    return prepared, services, chosen


def _osm_query(
    bbox: str,
    relation_ids: list[int] | None,
    modes: list[str],
    filters: Mapping[str, str | None],
    timeout: float,
) -> str:
    """Build a bounded relation query with mode filtering at the source.

    Complete relation members and immediate route masters remain available.

    Parameters
    ----------
    bbox : str
        WGS84 west, south, east and north bounds, or empty for explicit IDs.
    relation_ids : list of int or None
        Explicit relation selection.
    modes : list of str
        Normalised OSM route tags.
    filters : mapping
        Optional exact route-name and reference filters.
    timeout : float
        Overpass timeout in seconds.

    Returns
    -------
    str
        Overpass query.
    """
    if relation_ids is not None:
        if not relation_ids or any(
            not isinstance(value, int) or value <= 0 for value in relation_ids
        ):
            msg = "relation_ids must contain positive integers"
            raise ValueError(msg)
        selector = f'rel(id:{",".join(map(str, relation_ids))})["type"="route"]'
    else:
        if not bbox:
            msg = "Supply area, place_name or relation_ids"
            raise ValueError(msg)
        west, south, east, north = bbox.split(",")
        selector = 'rel["type"="route"]'
        for field, value in filters.items():
            if value is not None:
                selector += f"[{json.dumps(field)}={json.dumps(value)}]"
        selector += f"({south},{west},{north},{east})"
    return (
        f"[out:json][timeout:{int(timeout)}];{selector}->.candidates;"
        ".candidates out tags;"
        f'rel.candidates["route"~"^({"|".join(modes)})$"]->.selected;'
        'rel(br.selected)["type"="route_master"]->.masters;'
        "(.selected;.selected >>;.masters;);out body geom;"
    )


def _osm_input(
    relations: list[dict[str, Any]],
    elements: dict[tuple[str, int], dict[str, Any]],
    data: Mapping[str, pd.DataFrame] | None,
    *,
    overwrite_source: bool,
    polygon: Polygon | MultiPolygon | None,
    relation_ids: list[int] | None,
    on_invalid: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """
    Interpret selected relations before applying supplementation.

    Structural failures are retained so explicit input can repair affected patterns.

    Parameters
    ----------
    relations : list[dict[str, Any]]
        Selected OSM route relations in the retrieval response.
    elements : dict[tuple[str, int], dict[str, Any]]
        Retrieved OSM elements indexed by type and identifier.
    data : Mapping[str, pd.DataFrame] | None
        Explicit transport tables or supplemental input.
    overwrite_source : bool
        Whether explicit supplemental values may replace source values.
    polygon : Polygon | MultiPolygon | None
        Precise geographical selection, when supplied.
    relation_ids : list[int] | None
        Explicit OSM relation IDs, bypassing polygon intersection filtering.
    on_invalid : str
        Raise for invalid source relations or explicitly skip them.

    Returns
    -------
    tuple
        Source transport tables and relation-specific structural failures.
    """
    routes, stops, patterns, sequences = [], [], [], []
    issues: dict[str, str] = {}
    overridden = (
        set(data["stop_sequences"]["pattern_id"])
        if overwrite_source and data is not None and "stop_sequences" in data
        else set()
    )
    for relation in sorted(relations, key=lambda item: item["id"]):
        pattern_id = f"osm:relation:{relation['id']}"
        tags = relation.get("tags", {})
        override = pattern_id in overridden
        try:
            locations, ordered, shape = _osm_pattern(relation, elements, sequence_override=override)
        except ValueError as exc:
            if override:
                raise
            issues[pattern_id] = str(exc)
            locations, ordered, shape = [], [], None
        if (
            polygon is not None
            and pattern_id not in issues
            and relation_ids is None
            and not any(polygon.intersects(row["geometry"]) for row in locations)
            and (shape is None or not shape.intersects(polygon))
        ):
            continue
        routes.append(
            {
                "route_id": pattern_id,
                "route_short_name": tags.get("ref"),
                "route_long_name": tags.get("name"),
                "route_type": _MODES[tags["route"]],
            }
        )
        stops.extend(locations)
        sequences.extend(ordered)
        patterns.append(
            {"pattern_id": pattern_id, "route_id": pattern_id, "direction_id": 0, "geometry": shape}
        )
    if not patterns:
        _osm_failures(issues, [], on_invalid)
        msg = "No OSM transport relations intersect the selected polygon"
        raise ValueError(msg)
    base = {
        "routes": pd.DataFrame(routes),
        "stops": gpd.GeoDataFrame(
            stops, columns=["stop_id", "stop_name", "geometry"], geometry="geometry", crs=4326
        ).drop_duplicates("stop_id"),
        "patterns": gpd.GeoDataFrame(patterns, geometry="geometry", crs=4326),
        "stop_sequences": pd.DataFrame(
            sequences,
            columns=["pattern_id", "stop_id", "stop_sequence", "pickup_type", "drop_off_type"],
        ),
    }
    return base, issues


def _load_osm_transport(  # noqa: PLR0913, PLR0915 - source selection and shared generation options
    io: _SourceIO,
    services: pd.DataFrame | None = None,
    *,
    agency: pd.DataFrame | None = None,
    area: list[float] | Polygon | MultiPolygon | gpd.GeoSeries | gpd.GeoDataFrame | None = None,
    place_name: str | None = None,
    route_name: str | None = None,
    route_ref: str | None = None,
    relation_ids: list[int] | None = None,
    modes: str | Sequence[str] | None = None,
    pattern_ids: str | Sequence[str] | None = None,
    on_invalid: Literal["raise", "skip"] = "raise",
    data: TransportData | Mapping[str, pd.DataFrame] | None = None,
    overwrite_source: bool = _OVERWRITE_SOURCE,
    endpoint: str = _OSM_ENDPOINT,
    timeout: float = _OSM_TIMEOUT,
    use_source_tags: bool = _USE_SOURCE_TAGS,
    **gtfs_options: Unpack[GenerationOptions],
) -> duckdb.DuckDBPyConnection:
    """
    Retrieve and prepare OSM transport input for generation.

    Selection precedes structural validation; supplemental input can repair source gaps.

    Parameters
    ----------
    io : _SourceIO
        Geographical lookup and download operations supplied by the data facade.
    services : pd.DataFrame | None
        Partial service overrides for the selected patterns.
    agency : pd.DataFrame | None
        Explicit agency metadata passed to generation.
    area : list[float] | Polygon | MultiPolygon | gpd.GeoSeries | gpd.GeoDataFrame | None
        Bounding box or geometry defining the retrieval area.
    place_name : str | None
        Place name to resolve instead of an explicit area.
    route_name : str | None
        Optional route or infrastructure name filter.
    route_ref : str | None
        Optional OSM route reference filter.
    relation_ids : list[int] | None
        Explicit OSM relation IDs, bypassing polygon intersection filtering.
    modes : str | Sequence[str] | None
        Requested transport mode groups, or all supported modes.
    pattern_ids : str | Sequence[str] | None
        Explicit pattern selection applied before generation.
    on_invalid : Literal['raise', 'skip']
        Raise for invalid source relations or explicitly skip them.
    data : TransportData | Mapping[str, pd.DataFrame] | None
        Explicit transport tables or supplemental input.
    overwrite_source : bool
        Whether explicit supplemental values may replace source values.
    endpoint : str
        Overpass HTTP endpoint.
    timeout : float
        Overpass timeout in seconds.
    use_source_tags : bool
        Whether interpretable OSM operating tags supply missing conditions.
    **gtfs_options : Unpack[GenerationOptions]
        Calendar and timing options forwarded to GTFS generation.

    Returns
    -------
    duckdb.DuckDBPyConnection
        Generated feed with source, selection and assumption history.
    """
    _validate_generation_options(
        services, {**gtfs_options, "agency": agency, "data": data, "pattern_ids": pattern_ids}
    )
    selected_modes = _transport_modes(modes)
    data = _osm_supplement(cast("Mapping[str, pd.DataFrame] | None", data), set(), set())
    if on_invalid not in {"raise", "skip"}:
        msg = "on_invalid must be 'raise' or 'skip'"
        raise ValueError(msg)
    bbox, polygon = "", None
    if area is not None or place_name is not None:
        bbox, polygon = _area(io, area, place_name)
    query = _osm_query(
        bbox, relation_ids, selected_modes, {"name": route_name, "ref": route_ref}, timeout
    )
    payload = _request_osm(query, endpoint, timeout)
    elements = {(item["type"], item["id"]): item for item in payload["elements"]}
    relations = [
        item
        for item in elements.values()
        if item["type"] == "relation"
        and item.get("tags", {}).get("type") == "route"
        and item.get("tags", {}).get("route") in selected_modes
        and (relation_ids is None or item["id"] in relation_ids)
    ]
    masters = [
        item
        for item in elements.values()
        if item["type"] == "relation" and item.get("tags", {}).get("type") == "route_master"
    ]
    if route_name:
        relations = [item for item in relations if item.get("tags", {}).get("name") == route_name]
    if route_ref:
        relations = [item for item in relations if item.get("tags", {}).get("ref") == route_ref]
    if not relations:
        msg = f"No matching OSM public transport relations for modes {selected_modes}"
        raise ValueError(msg)
    # Validate known service IDs before filtering explicit source selections.
    known_ids = {
        f"osm:relation:{item['id']}"
        for item in payload["elements"]
        if item["type"] == "relation" and item.get("tags", {}).get("type") == "route"
    }
    supplied_ids = (
        set(data["patterns"]["pattern_id"].astype(str))
        if data is not None and "patterns" in data
        else set()
    )
    source_ids = known_ids.copy()
    known_ids |= supplied_ids
    requested = _selection(pattern_ids, "pattern_ids")
    if requested is not None:
        if unknown := set(requested) - known_ids:
            msg = f"pattern_ids: unknown IDs {sorted(unknown)}"
            raise ValueError(msg)
        # Additional user patterns may reuse source stops; keep their backing
        # source data until the final selection after supplementation.
        needs_source_stops = bool(set(requested) & (supplied_ids - source_ids))
        relations = [
            item
            for item in relations
            if f"osm:relation:{item['id']}" in requested or needs_source_stops
        ]
        if not relations and not (set(requested) & (supplied_ids - source_ids)):
            msg = "No matching OSM patterns after mode and pattern selection"
            raise ValueError(msg)
    services = _filter_source_services(
        services,
        known_ids,
        {f"osm:relation:{item['id']}" for item in relations} | (supplied_ids - source_ids),
    )
    data = _osm_supplement(
        cast("Mapping[str, pd.DataFrame] | None", data),
        source_ids,
        {f"osm:relation:{item['id']}" for item in relations},
    )
    candidates = {
        (
            item.get("tags", {}).get("ref"),
            item.get("tags", {}).get("operator"),
            item.get("tags", {}).get("network"),
        )
        for item in relations
    }
    if route_name and relation_ids is None and len(candidates) > 1:
        msg = f"Ambiguous route_name; candidate relation_ids: {[item['id'] for item in relations]}"
        raise ValueError(msg)
    base, issues = _osm_input(
        relations,
        elements,
        data,
        overwrite_source=overwrite_source,
        polygon=polygon,
        relation_ids=relation_ids,
        on_invalid=on_invalid,
    )
    prepared = _complete_input(
        base,
        _osm_supplement(
            cast("Mapping[str, pd.DataFrame] | None", data),
            source_ids,
            set(base["patterns"]["pattern_id"].astype(str)),
        ),
        overwrite_source=overwrite_source,
        source="OpenStreetMap",
        metadata={
            "endpoint": endpoint,
            "timeout": timeout,
            "use_source_tags": use_source_tags,
            "modes": selected_modes,
            "pattern_ids": requested,
            "excluded_pattern_ids": sorted(
                source_ids - {f"osm:relation:{item['id']}" for item in relations}
            ),
            "on_invalid": on_invalid,
            "overwrite_source": overwrite_source,
            "query": query,
            "osm3s": payload.get("osm3s"),
            "relations": relations,
            "route_masters": masters,
            "direction_id": "0 labels each independently selected relation; no opposite service inferred",
        },
    )
    prepared, services, chosen = _select_completed_osm(
        prepared, services, selected_modes if modes is not None else None, requested
    )
    issues = {identifier: reason for identifier, reason in issues.items() if identifier in chosen}
    prepared, services, usable = _finish_osm_selection(
        prepared, services, cast("Mapping[str, pd.DataFrame] | None", data), issues, on_invalid
    )
    records = prepared["provenance"]
    labels = (
        records["table_name"].eq("patterns")
        & records["field"].eq("direction_id")
        & records["kind"].eq("source")
    )
    records.loc[labels, "kind"] = "estimated"
    records.loc[labels, "details"] = "Independent relation label; no reverse service inferred"
    _add_osm_operating(
        prepared,
        [item for item in relations if f"osm:relation:{item['id']}" in usable],
        enabled=use_source_tags,
        masters=masters,
    )
    return create_gtfs(prepared, services, agency=agency, **gtfs_options)


def _overture_selection(
    data: Mapping[str, pd.DataFrame],
    services: pd.DataFrame | None,
    modes: str | Sequence[str] | None,
    pattern_ids: str | Sequence[str] | None,
) -> tuple[Mapping[str, pd.DataFrame], pd.DataFrame | None, list[dict[str, object]]]:
    """Select explicitly supplied Overture route patterns before downloading.

    Mode identity comes from route_type rather than inferred infrastructure use.

    Parameters
    ----------
    data : mapping
        Caller-supplied transport tables.
    services : pandas.DataFrame or None
        Service overrides.
    modes : str, sequence of str or None
        Requested mode groups.
    pattern_ids : str, sequence of str or None
        Additional pattern selection.

    Returns
    -------
    tuple
        Selected tables, services and selection provenance.
    """
    selected_modes = _transport_modes(modes)
    selection_records: list[dict[str, object]] = []
    if modes is not None or pattern_ids is not None:
        required = {
            "routes": {"route_id", "route_type"},
            "patterns": {"pattern_id", "route_id"},
            "stop_sequences": {"pattern_id", "stop_id"},
        }
        for name, columns in required.items():
            if name not in data or not columns.issubset(data[name]):
                msg = f"{name}: mode/pattern selection requires columns {sorted(columns)}"
                raise ValueError(msg)
        requested = _selection(pattern_ids, "pattern_ids")
        if requested is not None and (
            unknown := set(requested) - set(data["patterns"]["pattern_id"].astype(str))
        ):
            msg = f"pattern_ids: unknown IDs {sorted(unknown)}"
            raise ValueError(msg)
        chosen = _matching_patterns(data, selected_modes if modes is not None else None, requested)
        if not chosen:
            msg = f"No matching Overture route patterns for modes {selected_modes}"
            raise ValueError(msg)
        selected, services = _select_patterns(
            data, services, chosen, selection_records, explicit=pattern_ids is not None
        )
        data = cast("Mapping[str, pd.DataFrame]", selected)
    return data, services, selection_records


def _load_overture_transport(
    io: _SourceIO,
    services: pd.DataFrame | None = None,
    *,
    agency: pd.DataFrame | None = None,
    data: TransportData | Mapping[str, pd.DataFrame],
    area: list[float] | Polygon | MultiPolygon | gpd.GeoSeries | gpd.GeoDataFrame | None = None,
    place_name: str | None = None,
    release: str | None = None,
    route_name: str | None = None,
    modes: str | Sequence[str] | None = None,
    pattern_ids: str | Sequence[str] | None = None,
    overwrite_source: bool = _OVERWRITE_SOURCE,
    **gtfs_options: Unpack[GenerationOptions],
) -> duckdb.DuckDBPyConnection:
    """
    Retrieve Overture infrastructure for explicit transport input.

    Routes and stop order remain caller-supplied while selected infrastructure supplies geometry.

    Parameters
    ----------
    io : _SourceIO
        Geographical lookup and download operations supplied by the data facade.
    services : pd.DataFrame | None
        Partial service overrides for the selected patterns.
    agency : pd.DataFrame | None
        Explicit agency metadata passed to generation.
    data : TransportData | Mapping[str, pd.DataFrame]
        Explicit transport tables or supplemental input.
    area : list[float] | Polygon | MultiPolygon | gpd.GeoSeries | gpd.GeoDataFrame | None
        Bounding box or geometry defining the retrieval area.
    place_name : str | None
        Place name to resolve instead of an explicit area.
    release : str | None
        Overture release identifier, or the latest available release.
    route_name : str | None
        Optional route or infrastructure name filter.
    modes : str | Sequence[str] | None
        Requested transport mode groups, or all supported modes.
    pattern_ids : str | Sequence[str] | None
        Explicit pattern selection applied before generation.
    overwrite_source : bool
        Whether explicit supplemental values may replace source values.
    **gtfs_options : Unpack[GenerationOptions]
        Calendar and timing options forwarded to GTFS generation.

    Returns
    -------
    duckdb.DuckDBPyConnection
        Generated feed with source, selection and assumption history.
    """
    _validate_generation_options(
        services, {**gtfs_options, "agency": agency, "data": data, "pattern_ids": pattern_ids}
    )
    selected_modes = _transport_modes(modes)
    data, services, selection_records = _overture_selection(
        cast("Mapping[str, pd.DataFrame]", data), services, modes, pattern_ids
    )
    bbox, polygon = _area(io, area, place_name)
    resolved_release = release if release is not None else max(ALL_RELEASES)
    requested_stops = set(data.get("stop_sequences", pd.DataFrame()).get("stop_id", []))
    supplied_stops = set(data.get("stops", pd.DataFrame()).get("stop_id", []))
    needs_candidates = any(
        str(value).startswith("overture:") for value in requested_stops - supplied_stops
    )
    raw = io.overture(
        area=polygon if polygon is not None else [float(value) for value in bbox.split(",")],
        types=["segment", "connector", "place"] if needs_candidates else ["segment", "connector"],
        release=resolved_release,
        save_to_file=False,
        keep_outer_neighbors=True,
    )
    segments = raw.get("segment", gpd.GeoDataFrame())
    if route_name is not None:
        if "names" not in segments:
            msg = f"No named Overture infrastructure matching {route_name!r}"
            raise ValueError(msg)
        segments = segments[
            segments["names"].map(lambda names: _nested(names).get("primary") == route_name)
        ]
    if segments.empty:
        msg = "No matching Overture transport infrastructure"
        raise ValueError(msg)
    supplied = dict(data)
    patterns = supplied.get("patterns")
    if patterns is not None and "segment_ids" in patterns:
        patterns = patterns.copy()
        lookup = segments.set_index("id").geometry
        geometries = []
        for row in patterns.to_dict("records"):
            ids = row["segment_ids"]
            if not isinstance(ids, list) or not ids or set(ids) - set(lookup.index):
                msg = f"pattern {row['pattern_id']}: segment_ids must identify an ordered list of retrieved segments"
                raise ValueError(msg)
            shape = _chain_lines([lookup.loc[key] for key in ids])
            if shape is None:
                msg = (
                    f"pattern {row['pattern_id']}: supplied segments do not form a continuous path"
                )
                raise ValueError(msg)
            if row.get("reverse_shape", False):
                shape = LineString(list(shape.coords)[::-1])
            geometries.append(shape)
        supplied["patterns"] = gpd.GeoDataFrame(patterns, geometry=geometries, crs=segments.crs)
    stops = []
    for row in raw.get("place", pd.DataFrame()).to_dict("records"):
        if f"overture:{row['id']}" not in requested_stops:
            continue
        category = str(_nested(row.get("categories")).get("primary", ""))
        if any(
            word in category
            for word in (
                "railway_station",
                "train_station",
                "subway_station",
                "bus_stop",
                "bus_station",
                "ferry_terminal",
                "tram_stop",
            )
        ) and isinstance(row.get("geometry"), Point):
            stops.append(
                {
                    "stop_id": f"overture:{row['id']}",
                    "stop_name": _nested(row.get("names")).get("primary"),
                    "geometry": row["geometry"],
                }
            )
    base = {
        "stops": gpd.GeoDataFrame(
            stops, columns=["stop_id", "stop_name", "geometry"], geometry="geometry", crs=4326
        )
    }
    prepared = _complete_input(
        base,
        supplied,
        overwrite_source=overwrite_source,
        source="Overture Maps",
        metadata={
            "release": resolved_release,
            "bbox": bbox,
            "route_name": route_name,
            "modes": selected_modes if modes is not None else None,
            "segments": segments.drop(columns="geometry").to_dict("records"),
            "connectors": raw.get("connector", pd.DataFrame())
            .drop(columns="geometry", errors="ignore")
            .to_dict("records"),
        },
    )
    if selection_records:
        prepared["provenance"] = pd.concat(
            [prepared["provenance"], pd.DataFrame(selection_records)], ignore_index=True
        )
    return create_gtfs(prepared, services, agency=agency, **gtfs_options)


def _nested(value: object) -> dict[str, Any]:
    """
    Read Overture nested properties supplied as dictionaries or JSON strings.

    GeoJSON properties may arrive as mappings, JSON or serialised Python literals.

    Parameters
    ----------
    value : object
        Input scalar or nested property to interpret.

    Returns
    -------
    dict[str, Any]
        Parsed property mapping, or an empty mapping for absent properties.
    """
    if isinstance(value, str):
        encoded = value
        try:
            value = json.loads(encoded)
        except json.JSONDecodeError:
            try:
                value = literal_eval(encoded)
            except (ValueError, SyntaxError):
                return {}
    return value if isinstance(value, dict) else {}
