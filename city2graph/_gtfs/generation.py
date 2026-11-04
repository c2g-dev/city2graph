"""Prepare shared timing inputs and materialise resolved service as GTFS tables."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import geopandas as gpd
import pandas as pd
from pyproj import Geod
from shapely.geometry import LineString
from shapely.ops import substring

from .feed import _assumption
from .feed import _connection
from .feed import _input_provenance
from .feed import _retrieval_assumptions
from .feed import _source_conditions
from .feed import _warn_generation
from .schema import _MODEL
from .schema import _PROVENANCE_COLUMNS
from .schema import _REGISTRY
from .schema import _ROUTE_TYPE_GROUPS
from .schema import _SERVICE_DATE
from .schema import TransportData
from .schema import _clock
from .schema import _default
from .schema import _GenerationConfig
from .schema import _number
from .schema import _present
from .schema import _records
from .schema import _seconds
from .schema import _text_frame
from .service import _resolve_agency
from .service import _resolve_services
from .validation import _fail
from .validation import _prepare_input
from .validation import _require
from .validation import _select_patterns
from .validation import _selection
from .validation import _validate_generation_options
from .validation import _validate_tables

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    import duckdb

_GEOD = Geod(ellps="WGS84")


@dataclass
class _TimingInput:
    """Ordered occurrences and geometry shared for one generation call.

    Spatial point construction is deferred until a leg requires it.
    """

    sequences: dict[str, list[dict[str, Any]]]
    sequence_numbers: dict[str, list[float]]
    stops: pd.DataFrame
    lines: dict[str, LineString]

    @cached_property
    def locations(self) -> gpd.GeoSeries:
        """Construct stop points only when a leg needs spatial estimation.

        The cached points belong exclusively to this generation call.

        Returns
        -------
        geopandas.GeoSeries
            WGS84 stop points indexed by stop identifier.
        """
        return gpd.GeoSeries(
            gpd.points_from_xy(
                self.stops["stop_lon"].astype(float), self.stops["stop_lat"].astype(float)
            ),
            index=self.stops["stop_id"],
            crs=4326,
        )


@dataclass
class _ResolvedGeneration:
    """Effective input and its provenance, ready for GTFS materialisation.

    Resolved services share the prepared timing input through final assembly.
    """

    data: TransportData
    services: pd.DataFrame
    agency: pd.DataFrame
    calendar_dates: pd.DataFrame | None
    provenance: list[dict[str, object]]
    shapes: list[dict[str, object]]
    timing: _TimingInput


def _shape_rows(patterns: pd.DataFrame) -> tuple[list[dict[str, object]], dict[str, LineString]]:
    """
    Build GTFS shapes and retain their directed alignment for distance calculation.

    Vertex distances use WGS84 geodesics and remain in metres.

    Parameters
    ----------
    patterns : pd.DataFrame
        Route patterns with identifiers, direction labels and optional geometry.

    Returns
    -------
    tuple[list[dict[str, object]], dict[str, LineString]]
        GTFS shape records and their corresponding directed alignments.
    """
    rows: list[dict[str, object]] = []
    lines: dict[str, LineString] = {}
    for pattern in _records(patterns):
        line = pattern.get("geometry")
        if line is None or not _present(line):
            continue
        if not isinstance(line, LineString) or line.is_empty or line.length == 0:
            msg = f"pattern {pattern['pattern_id']}: supply a non-empty LineString"
            raise ValueError(msg)
        pattern_id = str(pattern["pattern_id"])
        lines[pattern_id] = line
        distance = 0.0
        previous = None
        for index, coord in enumerate(line.coords):
            if previous is not None:
                distance += _GEOD.inv(previous[0], previous[1], coord[0], coord[1])[2]
            rows.append(
                {
                    "shape_id": pattern_id,
                    "shape_pt_sequence": index,
                    "shape_pt_lon": coord[0],
                    "shape_pt_lat": coord[1],
                    "shape_dist_traveled": distance,
                }
            )
            previous = coord
    return rows, lines


def _leg_distance(
    row: dict[str, object],
    previous: dict[str, object],
    line: LineString | None,
    locations: gpd.GeoSeries,
    method: str,
) -> tuple[float, str]:
    """
    Resolve incoming leg distance without inferring an unknown path.

    Supplied distances take precedence over projections and straight-line estimates.

    Parameters
    ----------
    row : dict[str, object]
        Current stop occurrence and its incoming-leg attributes.
    previous : dict[str, object]
        Previous stop occurrence in the same pattern.
    line : LineString | None
        Directed WGS84 route alignment, when available.
    locations : gpd.GeoSeries
        WGS84 stop points indexed by stop_id.
    method : str
        Distance policy: auto, shape or geodesic.

    Returns
    -------
    tuple[float, str]
        Incoming distance in metres and a description of its basis.
    """
    if _present(row.get("distance_m")):
        return _number(row["distance_m"], "distance_m", positive=True), "supplied distance_m"
    if _present(row.get("shape_dist_traveled")) and _present(previous.get("shape_dist_traveled")):
        distance = _number(row["shape_dist_traveled"], "shape_dist_traveled") - _number(
            previous["shape_dist_traveled"], "shape_dist_traveled"
        )
        return _number(
            distance, "shape distance difference", positive=True
        ), "supplied shape_dist_traveled"
    first, last = locations.loc[str(previous["stop_id"])], locations.loc[str(row["stop_id"])]
    usable_shape = line is not None and line.is_simple and not line.is_ring
    start, end = (
        (line.project(first), line.project(last))
        if usable_shape and line is not None
        else (0.0, 0.0)
    )
    if method == "geodesic" or (method == "auto" and (not usable_shape or end <= start)):
        distance = float(_GEOD.inv(first.x, first.y, last.x, last.y)[2])
        if distance <= 0 or first.equals(last):
            msg = "zero stop-to-stop distance; supply distance_m or travel_time_secs"
            raise ValueError(msg)
        return distance, "geodesic stop-to-stop"
    if not usable_shape or line is None:
        msg = "supply distance_m or ordered shape_dist_traveled for missing/looping geometry"
        raise ValueError(msg)
    if end <= start:
        msg = "stops do not follow the directed shape; supply explicit distances"
        raise ValueError(msg)
    return float(_GEOD.geometry_length(substring(line, start, end))), "shape geodesic length"


def _pattern_times(
    sequence: list[dict[str, Any]],
    line: LineString | None,
    timing: _TimingInput,
    speed_kph: float | None,
    dwell_time_secs: float | None,
    distance_method: str,
    provenance: list[dict[str, object]],
) -> list[dict[str, object]]:
    """
    Compute one pattern's relative schedule and record estimated values.

    Explicit offsets already include dwell; inferred offsets add it exactly once.

    Parameters
    ----------
    sequence : list of dict
        Ordered stop occurrences and supplied timing attributes.
    line : LineString | None
        Directed WGS84 route alignment, when available.
    timing : _TimingInput
        Shared stop geometry and numeric occurrence order.
    speed_kph : float | None
        Fallback running speed, explicitly supplied in kilometres per hour.
    dwell_time_secs : float | None
        Fallback stopping duration in seconds, without an implicit default.
    distance_method : str
        Distance policy used when no leg distance is supplied.
    provenance : list[dict[str, object]]
        Source and assumption records; estimation records are appended during generation.

    Returns
    -------
    list[dict[str, object]]
        Stop occurrences with relative arrival and departure offsets.
    """
    result: list[dict[str, object]] = []
    previous: dict[str, object] | None = None
    departure = 0.0
    numbers = timing.sequence_numbers[str(sequence[0]["pattern_id"])]
    for supplied, number in zip(sequence, numbers, strict=True):
        row = {**supplied, "stop_sequence": number}
        label = f"pattern {row['pattern_id']}, stop_sequence {row['stop_sequence']}"
        try:
            arrival_value, departure_value = (
                row.get("arrival_offset_secs"),
                row.get("departure_offset_secs"),
            )
            if _present(arrival_value) or _present(departure_value):
                arrival = _number(arrival_value, "arrival_offset_secs")
                next_departure = _number(departure_value, "departure_offset_secs")
            else:
                travel = 0.0
                if previous is not None:
                    if _present(row.get("travel_time_secs")):
                        travel = _number(row["travel_time_secs"], "travel_time_secs", positive=True)
                    else:
                        speed_value = row.get("speed_kph")
                        speed = _number(
                            speed_value if _present(speed_value) else speed_kph,
                            "speed_kph",
                            positive=True,
                        )
                        distance, basis = _leg_distance(
                            row, previous, line, timing.locations, distance_method
                        )
                        travel = distance / (speed / 3.6)
                        provenance.append(
                            {
                                "table_name": "stop_sequences",
                                "record_id": f"{row['pattern_id']}|{row['stop_sequence']}",
                                "field": "travel_time_secs",
                                "kind": "estimated",
                                "source": "create_gtfs",
                                "value": str(travel),
                                "details": json.dumps(
                                    {
                                        "distance_m": distance,
                                        "speed_kph": speed,
                                        "distance_basis": basis,
                                        "includes_dwell": False,
                                    }
                                ),
                            }
                        )
                dwell_value = row.get("dwell_time_secs")
                dwell = _number(
                    dwell_value if _present(dwell_value) else dwell_time_secs, "dwell_time_secs"
                )
                arrival = departure + travel
                next_departure = arrival + dwell
            if arrival < departure or next_departure < arrival:
                msg = "offsets must be monotonic; do not add dwell twice"
                raise ValueError(msg)  # noqa: TRY301 - attach stop context below
            result.append(
                {**row, "arrival_offset_secs": arrival, "departure_offset_secs": next_departure}
            )
            departure, previous = next_departure, row
        except ValueError as exc:
            msg = f"{label}: {exc}"
            raise ValueError(msg) from exc
    return result


def create_gtfs(  # noqa: PLR0913 - shared generation API with optional overrides
    data: TransportData,
    services: pd.DataFrame | None = None,
    *,
    agency: pd.DataFrame | None = None,
    calendar: pd.DataFrame | None = None,
    calendar_dates: pd.DataFrame | None = None,
    speed_kph: float | None = None,
    dwell_time_secs: float | None = None,
    distance_method: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    headway_secs: float | None = None,
    group_headway_secs: float | None = None,
    pattern_ids: str | Sequence[str] | None = None,
    service_date: str | None = None,
    optional_tables: Mapping[str, pd.DataFrame] | None = None,
) -> duckdb.DuckDBPyConnection:
    """
    Create a GTFS feed from stopping patterns and optional operating conditions.

    Each service row describes a scheduled departure or one headway period.
    Missing conditions use the deterministic simple-v1 reference-week model.
    Inputs are copied; source data and modelling assumptions remain traceable.

    Parameters
    ----------
    data : TransportData
        ``routes`` and ``stops`` use GTFS columns. Stops may instead be a Point
        GeoDataFrame with a CRS. ``patterns`` contains pattern_id, route_id,
        direction_id and optional LineString geometry (with CRS). Redundant
        binary geometry in coordinate-valued pandas stop tables is ignored.
        Integral numeric direction codes, including 0.0 and 1.0, are accepted.
        Whitespace-only cells are missing values. Documented extended route
        types inherit their mode group's default speed and retain their codes.
        Optional service_group_id changes grouping for default frequency allocation.
        ``stop_sequences`` contains pattern_id, stop_id and stop_sequence.
        Optional incoming-leg columns are travel_time_secs, distance_m,
        speed_kph and cumulative shape_dist_traveled (metres). Complete
        arrival_offset_secs and departure_offset_secs already include dwell.
        Otherwise, supplied travel times, speeds and dwell times override
        the model defaults; no operating columns are required.
    services : pandas.DataFrame, optional
        Rows may supply pattern_id and service_id, plus either departure_time or
        start_time, end_time and headway_secs. Optional exact_times defaults
        to 0; use 1 only for an explicitly anchored timetable. Optional trip_id
        defaults to a deterministic service-row identifier avoiding explicit IDs.
        Repeating a trip_id is allowed only for separate frequency periods
        of the same pattern.
        Omit the table to generate every pattern, or omit pattern_id to apply
        supplied conditions to all patterns. Missing fields are filled from
        global overrides, source information and defaults, in that order.
        Unlisted patterns are retained and completed. Use pattern_ids to select
        a subset. Unknown columns raise an error. Optional trip_headsign,
        trip_short_name, wheelchair_accessible and bikes_allowed are preserved.
    agency : pandas.DataFrame, optional
        GTFS agency metadata. Source fields precede synthetic placeholders.
        A unique geographical timezone is estimated before a warned UTC fallback.
    calendar : pandas.DataFrame, optional
        GTFS weekly service calendar. Without dates, use a fixed representative week.
    calendar_dates : pandas.DataFrame, optional
        GTFS date-specific service additions and removals.
    speed_kph : float, optional
        Speed for legs without timings or per-leg speed. Defaults depend on mode
        and are unverified average-speed assumptions including stopping effects.
    dwell_time_secs : float, optional
        Additional dwell for stops without per-stop values. Defaults to zero.
        Supplied offsets already include dwell and are not modified.
    distance_method : str, optional
        Defaults to "auto": use supplied distances, usable directed shape,
        then geodesic stop-to-stop distance. Explicit "shape" requires a usable
        directed alignment; "geodesic" skips shape projection.
    start_time : str, optional
        Global frequency-period start, default "06:00:00" without a source value.
    end_time : str, optional
        Global frequency-period end, default "22:00:00" without a source value.
    headway_secs : float, optional
        Global positive integer headway in seconds, per pattern. Exclusive with
        group_headway_secs. Individual service rows take precedence.
    group_headway_secs : float, optional
        Positive integer reference headway shared across each selected group.
        Each pattern receives this interval multiplied by the group's pattern
        count. Overrides source intervals, but not individual service rows.
        Without either override or source intervals, the shared default is 900 s.
        Explicit pattern overrides and closures do not redistribute unused shares.
    pattern_ids : str or sequence of str, optional
        Generate only these patterns. Defaults to all patterns, including those
        omitted from services. Known service rows outside the selection are
        excluded; unknown IDs and empty selections raise an error.
    service_date : str, optional
        Explicit YYYYMMDD service date for one day. If all date inputs are omitted,
        use seven days from 20260103 (Saturday). Cannot be combined with
        calendar or calendar_dates.
    optional_tables : mapping of str to pandas.DataFrame, optional
        Additional transfers, pathways, levels, feed_info or attributions tables.

    Returns
    -------
    duckdb.DuckDBPyConnection
        An in-memory feed compatible with load_gtfs and the analysis functions.

    Raises
    ------
    ValueError
        If inputs are incomplete, ambiguous or inconsistent with the contract.
    """
    config = _GenerationConfig(
        calendar=calendar,
        calendar_dates=calendar_dates,
        speed_kph=speed_kph,
        dwell_time_secs=dwell_time_secs,
        distance_method=distance_method,
        start_time=start_time,
        end_time=end_time,
        headway_secs=headway_secs,
        group_headway_secs=group_headway_secs,
        pattern_ids=pattern_ids,
        service_date=service_date,
    )
    resolved = _resolve_generation(data, services, agency, config)
    data, services, agency = resolved.data, resolved.services, resolved.agency
    calendar_dates, provenance = resolved.calendar_dates, resolved.provenance
    patterns, stops = data["patterns"], data["stops"]
    timing, shapes = resolved.timing, resolved.shapes
    schedules = {
        str(identifier): _pattern_times(
            timing.sequences[str(identifier)],
            timing.lines.get(str(identifier)),
            timing,
            speed_kph,
            dwell_time_secs,
            distance_method or str(_default("distance_method")),
            provenance,
        )
        for identifier in patterns["pattern_id"]
    }
    trips, times, frequencies = _build_trips(patterns, services, schedules, timing.lines)
    tables = {
        "agency": agency,
        "routes": data["routes"],
        "stops": stops,
        "trips": pd.DataFrame(trips),
        "stop_times": pd.DataFrame(times),
    }
    for name, rows in (("shapes", shapes), ("frequencies", frequencies)):
        if rows:
            tables[name] = pd.DataFrame(rows)
    if calendar is not None:
        tables["calendar"] = calendar
    if calendar_dates is not None:
        tables["calendar_dates"] = calendar_dates
    for name, frame in (optional_tables or {}).items():
        if name not in {"transfers", "pathways", "levels", "feed_info", "attributions"}:
            msg = f"optional_tables: unsupported table {name!r}"
            raise ValueError(msg)
        tables[name] = frame
    tables = {name: _text_frame(frame) for name, frame in tables.items()}
    _validate_tables(tables)
    _warn_generation(provenance)
    return _connection(tables, pd.DataFrame(provenance, columns=_PROVENANCE_COLUMNS))


def _build_trips(
    patterns: pd.DataFrame,
    services: pd.DataFrame,
    schedules: dict[str, list[dict[str, object]]],
    lines: dict[str, LineString],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """
    Materialise trip templates and scheduled departures from service rows.

    Repeated frequency identifiers share a template only when their pattern agrees.

    Parameters
    ----------
    patterns : pd.DataFrame
        Route patterns with identifiers, direction labels and optional geometry.
    services : pd.DataFrame
        Scheduled departures or frequency periods referring to pattern identifiers.
    schedules : dict[str, list[dict[str, object]]]
        Relative arrival and departure offsets indexed by pattern_id.
    lines : dict[str, LineString]
        Directed route alignments indexed by pattern_id.

    Returns
    -------
    tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]
        Trip, stop-time and frequency records ready for table validation.
    """
    trips: list[dict[str, object]] = []
    times: list[dict[str, object]] = []
    frequencies: list[dict[str, object]] = []
    seen: dict[str, tuple[str, str, bool]] = {}
    pattern_map = patterns.set_index("pattern_id").to_dict("index")
    for service in _records(services):
        pattern_id, trip_id = str(service["pattern_id"]), str(service["trip_id"])
        pattern, schedule = pattern_map[pattern_id], schedules[pattern_id]
        frequency = _present(service.get("headway_secs"))
        scheduled = _present(service.get("departure_time"))
        if frequency == scheduled:
            msg = f"service {trip_id}: supply departure_time OR a frequency period"
            raise ValueError(msg)
        identity = (pattern_id, str(service["service_id"]), frequency)
        if trip_id in seen and (seen[trip_id] != identity or not frequency):
            msg = f"service {trip_id}: conflicting or duplicate trip_id"
            raise ValueError(msg)
        if frequency:
            frequencies.append(
                {
                    "trip_id": trip_id,
                    "start_time": service.get("start_time"),
                    "end_time": service.get("end_time"),
                    "headway_secs": service["headway_secs"],
                    "exact_times": service.get("exact_times")
                    if _present(service.get("exact_times"))
                    else 0,
                }
            )
        if trip_id in seen:
            continue
        seen[trip_id] = identity
        first_departure = float(str(schedule[0]["departure_offset_secs"]))
        # Keep frequency template offsets non-negative; analysis subtracts its
        # first departure. Fixed departures are anchored at the first stop.
        offset = 0.0 if frequency else _seconds(service["departure_time"]) - first_departure
        if offset < 0:
            msg = f"service {trip_id}: first arrival precedes the service day"
            raise ValueError(msg)
        trip = {
            "trip_id": trip_id,
            "route_id": pattern["route_id"],
            "service_id": service["service_id"],
            "direction_id": pattern["direction_id"],
        }
        for field in ("trip_headsign", "trip_short_name", "wheelchair_accessible", "bikes_allowed"):
            if _present(service.get(field)):
                trip[field] = service[field]
        if pattern_id in lines:
            trip["shape_id"] = pattern_id
        trips.append(trip)
        for stop in schedule:
            result = {
                "trip_id": trip_id,
                "stop_id": stop["stop_id"],
                "stop_sequence": stop["stop_sequence"],
                "arrival_time": _clock(offset + float(str(stop["arrival_offset_secs"]))),
                "departure_time": _clock(offset + float(str(stop["departure_offset_secs"]))),
            }
            for field in (
                "pickup_type",
                "drop_off_type",
                "stop_headsign",
                "shape_dist_traveled",
                "timepoint",
            ):
                if _present(stop.get(field)):
                    result[field] = stop[field]
            times.append(result)
    return trips, times, frequencies


def _resolve_generation(
    data: TransportData,
    services: pd.DataFrame | None,
    agency: pd.DataFrame | None,
    config: _GenerationConfig,
) -> _ResolvedGeneration:
    """
    Copy and resolve optional operating inputs before constructing GTFS tables.

    Original history and effective assumptions are recorded separately.

    Parameters
    ----------
    data : TransportData
        Common transport input.
    services : pandas.DataFrame or None
        Optional partial service rows.
    agency : pandas.DataFrame or None
        Optional agency metadata.
    config : _GenerationConfig
        Calendar and global operating overrides.

    Returns
    -------
    _ResolvedGeneration
        Effective input, services, agency, dates, history and shared timing data.
    """
    _validate_generation_options(services, {**vars(config), "agency": agency, "data": data})
    issues: list[str] = []
    for name, columns in {
        "routes": ["route_id", "route_type"],
        "stops": ["stop_id"],
        "patterns": ["pattern_id", "route_id", "direction_id"],
        "stop_sequences": ["pattern_id", "stop_sequence", "stop_id"],
    }.items():
        if name not in data:
            issues.append(f"missing input table {name}")
        else:
            _require(data[name], columns, name, issues)  # type: ignore[literal-required]
    _fail(issues)
    raw = {
        **cast("Mapping[str, pd.DataFrame]", data),
        **{
            name: frame
            for name, frame in (
                ("services", services),
                ("agency", agency),
                ("calendar", config.calendar),
                ("calendar_dates", config.calendar_dates),
            )
            if frame is not None
        },
    }
    records = _input_provenance(raw)
    data, services = _select_patterns(
        cast("Mapping[str, pd.DataFrame]", data),
        services,
        _selection(config.pattern_ids, "pattern_ids"),
        records,
    )
    _retrieval_assumptions(records)
    _assumption(records, "generation", "model", _MODEL)
    _assumption(
        records,
        "generation",
        "distance_method",
        config.distance_method or str(_default("distance_method")),
        "default" if config.distance_method is None else "user",
        _MODEL if config.distance_method is None else "input",
    )
    patterns, sequence, stops = _prepare_input(data)
    result: TransportData = {
        **data,
        "patterns": patterns,
        "stop_sequences": sequence,
        "stops": stops,
    }

    sources = _source_conditions(result)
    shapes, lines = _shape_rows(patterns)
    sequences: dict[str, list[dict[str, Any]]] = {
        str(identifier): [] for identifier in sequence["pattern_id"].unique()
    }
    sequence_numbers: dict[str, list[float]] = {identifier: [] for identifier in sequences}
    ordered = sequence.assign(
        _sequence_number=pd.to_numeric(sequence["stop_sequence"], errors="raise")
    )
    for row in _records(ordered.sort_values("_sequence_number", kind="stable")):
        identifier = str(row["pattern_id"])
        sequence_numbers[identifier].append(row.pop("_sequence_number"))
        sequences[identifier].append(row)
    timing = _TimingInput(sequences, sequence_numbers, stops, lines)
    resolved_services, dates = _resolve_services(result, services, config, records, sources)
    _timing_defaults(result, config, records, sources, timing)
    resolved_agency = _resolve_agency(result, agency, records, sources)
    if config.calendar is None and config.calendar_dates is None:
        _assumption(
            records,
            "generation",
            "service_date",
            config.service_date or _SERVICE_DATE,
            "default" if config.service_date is None else "user",
            _MODEL if config.service_date is None else "input",
            "Fixed reference-week coordinate, not a historical reconstruction",
        )
        _assumption(
            records,
            "generation",
            "service_days",
            1 if config.service_date else _default("service_days"),
            "generated" if config.service_date else "default",
        )
        if not config.service_date:
            for pattern_id in result["patterns"]["pattern_id"].astype(str):
                if "opening_hours" not in sources.get(pattern_id, {}):
                    _assumption(records, pattern_id, "weekday_policy", _default("weekday_policy"))
    for name, frame, original in (
        ("calendar", config.calendar, config.calendar),
        ("calendar_dates", dates, config.calendar_dates),
    ):
        supplied = (
            set()
            if original is None
            else {
                tuple(sorted((key, str(value)) for key, value in row.items()))
                for row in _records(original)
            }
        )
        if frame is not None:
            for index, row in enumerate(_records(frame)):
                user = tuple(sorted((key, str(value)) for key, value in row.items())) in supplied
                for field, value in row.items():
                    _assumption(
                        records,
                        f"{name}:{index}",
                        field,
                        value,
                        "user" if user else "generated",
                        "input" if user else _MODEL,
                    )
    return _ResolvedGeneration(
        result, resolved_services, resolved_agency, dates, records, shapes, timing
    )


def _allocate_duration(
    rows: list[dict[str, Any]],
    source: dict[str, Any],
    line: LineString | None,
    timing: _TimingInput,
    config: _GenerationConfig,
    records: list[dict[str, object]],
) -> list[dict[str, Any]]:
    """Allocate source duration after reserving explicit times and intermediate dwell.

    Explicit timings retain precedence and derived segment times remain estimates.

    Parameters
    ----------
    rows : list of dict
        Ordered independent stop occurrences.
    source : dict
        Parsed whole-route duration record.
    line : LineString or None
        Directed pattern geometry.
    timing : _TimingInput
        Shared stop locations, constructed on first spatial use.
    config : _GenerationConfig
        Explicit speed, dwell and distance policies.
    records : list of dict
        Adopted values and rejection reasons.

    Returns
    -------
    list
        Rows with inferred missing travel times, or unchanged rows on source conflict.
    """
    pattern_id, origin = str(rows[0]["pattern_id"]), str(source["source"])
    explicit_offsets = any(
        _present(row.get(field))
        for row in rows
        for field in ("arrival_offset_secs", "departure_offset_secs")
    )
    reason = ""
    if explicit_offsets or config.speed_kph is not None:
        reason = "Explicit offsets or global speed override source route duration"
    elif all(
        _present(row.get("travel_time_secs")) or _present(row.get("speed_kph")) for row in rows[1:]
    ):
        reason = "All incoming-leg timings were explicitly supplied"
    if reason:
        _assumption(
            records, pattern_id, "duration_secs", source["value"], "ignored", origin, reason
        )
        return rows
    total = _number(source["value"], "source duration", positive=True)
    adopted = [dict(row) for row in rows]
    known = sum(
        _number(
            row.get("dwell_time_secs")
            if _present(row.get("dwell_time_secs"))
            else (
                config.dwell_time_secs
                if config.dwell_time_secs is not None
                else _default("dwell_time_secs")
            ),
            "dwell_time_secs",
        )
        for row in rows[1:-1]
    )
    missing: list[int] = []
    for index, row in enumerate(adopted[1:], 1):
        if _present(row.get("travel_time_secs")):
            known += _number(row["travel_time_secs"], "travel_time_secs", positive=True)
        elif _present(row.get("speed_kph")):
            distance, _ = _leg_distance(
                row,
                rows[index - 1],
                line,
                timing.locations,
                config.distance_method or str(_default("distance_method")),
            )
            row["travel_time_secs"] = distance / (
                _number(row["speed_kph"], "speed_kph", positive=True) / 3.6
            )
            known += float(row["travel_time_secs"])
        else:
            missing.append(index)
    remainder = total - known
    if remainder <= 0:
        _assumption(
            records,
            pattern_id,
            "duration_secs",
            total,
            "ignored",
            origin,
            "Source duration is not greater than explicit leg times and intermediate dwell",
        )
        return rows
    distances, bases = [], []
    for index in missing:
        if len(missing) == 1:
            distance, basis = (
                1.0,
                "Single missing leg receives the remaining duration; no spatial allocation",
            )
        else:
            distance, basis = _leg_distance(
                rows[index],
                rows[index - 1],
                line,
                timing.locations,
                config.distance_method or str(_default("distance_method")),
            )
        distances.append(distance)
        bases.append(basis)
    _assumption(
        records,
        pattern_id,
        "duration_secs",
        total,
        "source",
        origin,
        "First departure to last arrival; terminal layover is not allocated",
    )
    for index, row in enumerate(adopted[1:], 1):
        if _present(rows[index].get("speed_kph")) and not _present(
            rows[index].get("travel_time_secs")
        ):
            scope = f"leg:{pattern_id}|{row['stop_sequence']}"
            _assumption(records, scope, "speed_kph", row["speed_kph"], "user", "input")
            _assumption(
                records,
                scope,
                "travel_time_secs",
                row["travel_time_secs"],
                "estimated",
                "input",
                "Computed from an explicit speed and the resolved distance",
            )
    total_distance = sum(distances)
    for index, distance, basis in zip(missing, distances, bases, strict=True):
        travel = remainder * distance / total_distance
        adopted[index]["travel_time_secs"] = travel
        _assumption(
            records,
            f"leg:{pattern_id}|{rows[index]['stop_sequence']}",
            "travel_time_secs",
            travel,
            "estimated",
            origin,
            json.dumps(
                {
                    "method": _REGISTRY["duration_allocation"],
                    "duration_secs": total,
                    "reserved_secs": known,
                    "remaining_secs": remainder,
                    "allocation_share": distance / total_distance,
                    "distance_basis": basis,
                }
            ),
        )
    return adopted


def _timing_defaults(
    data: TransportData,
    config: _GenerationConfig,
    records: list[dict[str, object]],
    sources: dict[str, dict[str, dict[str, Any]]],
    timing: _TimingInput,
) -> None:
    """
    Resolve only the speed, dwell and duration values used by the generator.

    Only values actually needed by the timing calculation are recorded as effective.

    Parameters
    ----------
    data : TransportData
        Independent patterns, routes, normalised stops and stop sequences.
    config : _GenerationConfig
        Optional global timing overrides.
    records : list of dict
        Effective assumptions to extend.
    sources : dict[str, dict[str, dict[str, Any]]]
        Interpreted source records indexed once for this generation.
    timing : _TimingInput
        Ordered occurrences and shared geometry for this generation.

    Returns
    -------
    None
        Updates the generation-local occurrence rows and provenance in place.
    """
    modes = {}
    for route in _records(data["routes"]):
        route_type = int(_number(route["route_type"], "routes.route_type"))
        modes[str(route["route_id"])] = str(_ROUTE_TYPE_GROUPS.get(route_type, route_type))
    patterns = {
        str(row["pattern_id"]): modes[str(row["route_id"])] for row in _records(data["patterns"])
    }
    recorded = {
        str(row["record_id"])
        for row in records
        if row["table_name"] == "assumptions" and row["field"] == "travel_time_secs"
    }
    for pattern_id, supplied_rows in timing.sequences.items():
        rows = supplied_rows
        if source := sources.get(str(pattern_id), {}).get("duration_secs"):
            first_record = len(records)
            rows = _allocate_duration(
                rows, source, timing.lines.get(str(pattern_id)), timing, config, records
            )
            timing.sequences[pattern_id] = rows
            recorded.update(
                str(row["record_id"])
                for row in records[first_record:]
                if row["table_name"] == "assumptions" and row["field"] == "travel_time_secs"
            )
        for index, row in enumerate(rows):
            scope = f"leg:{row['pattern_id']}|{row['stop_sequence']}"
            if _present(row.get("arrival_offset_secs")) or _present(
                row.get("departure_offset_secs")
            ):
                _assumption(records, scope, "timing_method", "supplied offsets", "user", "input")
                continue
            fields = [("dwell_time_secs", _default("dwell_time_secs"))]
            if index > 0 and not _present(row.get("travel_time_secs")):
                fields.append(
                    (
                        "speed_kph",
                        _REGISTRY["speeds_kph"].get(
                            patterns[str(pattern_id)], _REGISTRY["speeds_kph"]["other"]
                        ),
                    )
                )
            for field, default in fields:
                value = row.get(field)
                if not _present(value):
                    value = getattr(config, field)
                kind = "default" if value is None else "user"
                row[field] = default if value is None else value
                _assumption(
                    records,
                    scope,
                    field,
                    row[field],
                    kind,
                    str(_REGISTRY["model"]) if kind == "default" else "input",
                )
            if index > 0 and _present(row.get("travel_time_secs")) and scope not in recorded:
                _assumption(
                    records, scope, "travel_time_secs", row["travel_time_secs"], "user", "input"
                )
