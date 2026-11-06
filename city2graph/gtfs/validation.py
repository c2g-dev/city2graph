"""Validate, normalise and select transport input and exported GTFS tables."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from collections.abc import Sequence
from datetime import datetime
from difflib import get_close_matches
from itertools import pairwise
from typing import Any
from typing import cast
from zoneinfo import ZoneInfo
from zoneinfo import ZoneInfoNotFoundError

import geopandas as gpd
import pandas as pd

from .schema import _KEYS
from .schema import _MODES
from .schema import _REFERENCES
from .schema import _REQUIRED
from .schema import _ROUTE_TYPE_GROUPS
from .schema import TransportData
from .schema import _GenerationConfig
from .schema import _number
from .schema import _present
from .schema import _records
from .schema import _seconds
from .schema import _text_frame


def _fail(issues: list[str]) -> None:
    """
    Raise one diagnostic containing all collected input issues.

    All diagnostics are retained so callers can correct several fields at once.

    Parameters
    ----------
    issues : list[str]
        Diagnostic messages collected during validation.
    """
    if issues:
        raise ValueError("Invalid transport data:\n- " + "\n- ".join(issues))


def _require(frame: pd.DataFrame, columns: list[str], label: str, issues: list[str]) -> bool:
    """
    Collect missing columns and null required cells.

    Collect field and row context instead of failing at the first missing cell.

    Parameters
    ----------
    frame : pd.DataFrame
        Table with string column names.
    columns : list[str]
        Required column names.
    label : str
        Field context included in validation errors.
    issues : list[str]
        Diagnostic messages collected during validation.

    Returns
    -------
    bool
        Whether all required columns exist.
    """
    missing = set(columns) - set(frame.columns)
    if missing:
        issues.append(f"{label}: missing columns {sorted(missing)}")
        return False
    for column in columns:
        bad = ~frame[column].map(_present)
        if bad.any():
            issues.append(f"{label}.{column}: missing values at rows {frame.index[bad].tolist()}")
    return True


def _validate_tables(tables: dict[str, pd.DataFrame]) -> None:
    """
    Validate supported GTFS tables before creating or saving a feed.

    Generation and export use the same table and reference checks.

    Parameters
    ----------
    tables : dict[str, pd.DataFrame]
        GTFS tables represented as string-valued DataFrames.
    """
    issues: list[str] = []
    required_tables = {"agency", "stops", "routes", "trips", "stop_times"}
    if missing := required_tables - tables.keys():
        issues.append(f"missing tables {sorted(missing)}")
    if not ({"calendar", "calendar_dates"} & tables.keys()):
        issues.append("supply calendar or calendar_dates")
    for name, frame in tables.items():
        if name not in _REQUIRED:
            issues.append(f"unsupported GTFS table {name!r}")
            continue
        if name == "stop_times":
            _require(frame, ["trip_id", "stop_id", "stop_sequence"], name, issues)
            if not {"arrival_time", "departure_time"}.issubset(frame):
                issues.append("stop_times: missing arrival_time/departure_time columns")
        else:
            _require(frame, _REQUIRED[name], name, issues)
        keys = _KEYS.get(name, [])
        if keys and set(keys).issubset(frame) and frame.duplicated(keys).any():
            issues.append(f"{name}: duplicate keys {keys}")
        if name in required_tables and frame.empty:
            issues.append(f"{name}: must not be empty")
    _fail(issues)
    reference_ids = {
        (target, key): set(tables.get(target, pd.DataFrame()).get(key, []))
        for target, key in set(_REFERENCES.values())
    }
    for name, frame in tables.items():
        for field, (target, key) in _REFERENCES.items():
            if field not in frame or (name == target and field == key):
                continue
            values = set(frame[field].dropna()) - {""}
            available = reference_ids[target, key]
            if unresolved := values - available:
                issues.append(f"{name}.{field}: unknown {target} IDs {sorted(unresolved)}")
    calendars = [
        tables[name]["service_id"] for name in ("calendar", "calendar_dates") if name in tables
    ]
    unknown = set(tables["trips"]["service_id"]) - set(pd.concat(calendars))
    if unknown:
        issues.append(f"trips.service_id: unknown services {sorted(unknown)}")
    _validate_metadata(tables, issues)
    _validate_schedule(tables, issues)
    _validate_optional(tables, issues)
    _fail(issues)


def _validate_metadata(tables: dict[str, pd.DataFrame], issues: list[str]) -> None:
    """
    Check agency, route, stop and calendar metadata.

    Validate operator and route fields before checking stops and operating dates.

    Parameters
    ----------
    tables : dict[str, pd.DataFrame]
        GTFS tables represented as string-valued DataFrames.
    issues : list[str]
        Diagnostic messages collected during validation.
    """
    agency = tables["agency"]
    if len(agency) > 1 and ("agency_id" not in agency or agency["agency_id"].isna().any()):
        issues.append("agency.agency_id: required for multiple agencies")
    zones = set(agency["agency_timezone"])
    if len(zones) != 1:
        issues.append("agency: all agencies must use the same timezone")
    for zone in zones:
        try:
            ZoneInfo(str(zone))
        except (ZoneInfoNotFoundError, ValueError):
            issues.append(f"agency_timezone: invalid timezone {zone!r}")
    routes = tables["routes"]
    for row in _records(routes):
        if not any(_present(row.get(key)) for key in ("route_short_name", "route_long_name")):
            issues.append(f"routes {row['route_id']}: supply route_short_name or route_long_name")
        mode = pd.to_numeric(row["route_type"], errors="coerce")
        if pd.isna(mode) or not math.isfinite(mode) or mode < 0 or int(mode) != mode:
            issues.append(f"routes {row['route_id']}: invalid route_type")
    _validate_stops(tables["stops"], issues)
    _validate_calendars(tables, issues)


def _validate_stops(stops: pd.DataFrame, issues: list[str]) -> None:
    """
    Check boarding locations and station hierarchy requirements.

    Different location types have different coordinate and parent requirements.

    Parameters
    ----------
    stops : pd.DataFrame
        Boarding locations and optional station metadata.
    issues : list[str]
        Diagnostic messages collected during validation.
    """
    for row in _records(stops):
        location = row.get("location_type") if _present(row.get("location_type")) else "0"
        if location not in {"0", "1", "2", "3", "4"}:
            issues.append(f"stops {row['stop_id']}: invalid location_type")
        if location in {"0", "1", "2"}:
            if not _present(row.get("stop_name")):
                issues.append(f"stops {row['stop_id']}: missing stop_name")
            for field, limit in (("stop_lat", 90), ("stop_lon", 180)):
                value = pd.to_numeric(str(row.get(field)), errors="coerce")
                if pd.isna(value) or not -limit <= value <= limit:
                    issues.append(f"stops {row['stop_id']}: invalid {field}")
        if location in {"2", "3", "4"} and not _present(row.get("parent_station")):
            issues.append(f"stops {row['stop_id']}: missing parent_station")


def _validate_calendars(tables: dict[str, pd.DataFrame], issues: list[str]) -> None:
    """
    Check service dates, weekday flags and calendar exceptions.

    Both weekly service and date-specific exceptions must have valid date fields.

    Parameters
    ----------
    tables : dict[str, pd.DataFrame]
        GTFS tables represented as string-valued DataFrames.
    issues : list[str]
        Diagnostic messages collected during validation.
    """
    for name in ("calendar", "calendar_dates"):
        if name not in tables:
            continue
        frame = tables[name]
        fields = ["start_date", "end_date"] if name == "calendar" else ["date"]
        for field in fields:
            for value in frame[field].unique():
                try:
                    if not re.fullmatch(r"\d{8}", str(value)):
                        raise ValueError  # noqa: TRY301 - collect malformed dates with parse errors
                    datetime.strptime(str(value), "%Y%m%d")
                except ValueError:
                    issues.append(f"{name}.{field}: invalid date {value!r}")
        if name == "calendar":
            if (frame["start_date"] > frame["end_date"]).any():
                issues.append("calendar: start_date must not exceed end_date")
            issues.extend(
                f"calendar.{field}: expected 0 or 1"
                for field in _REQUIRED[name][1:8]
                if not frame[field].isin(["0", "1"]).all()
            )
        elif not frame["exception_type"].isin(["1", "2"]).all():
            issues.append("calendar_dates.exception_type: expected 1 or 2")


def _validate_schedule(tables: dict[str, pd.DataFrame], issues: list[str]) -> None:
    """
    Validate stop timings and non-overlapping frequency periods.

    Known stop times must be ordered, including times after midnight.

    Parameters
    ----------
    tables : dict[str, pd.DataFrame]
        GTFS tables represented as string-valued DataFrames.
    issues : list[str]
        Diagnostic messages collected during validation.
    """
    times = tables["stop_times"].copy()
    sequence = pd.to_numeric(times["stop_sequence"], errors="coerce")
    if sequence.isna().any() or (sequence < 0).any() or (sequence % 1 != 0).any():
        issues.append("stop_times.stop_sequence: expected non-negative integers")
        return
    times["stop_sequence"] = sequence
    try:
        for field in ("arrival_time", "departure_time"):
            seconds = {
                value: _seconds(value) if _present(value) else float("nan")
                for value in times[field].unique()
            }
            times[field] = times[field].map(seconds)
    except ValueError as exc:
        issues.append(str(exc))
        return
    trip_order = times["trip_id"].drop_duplicates()
    ordered = times.sort_values(["trip_id", "stop_sequence"], kind="stable")
    groups = ordered.groupby("trip_id", sort=False)
    sizes = groups.size()
    endpoints = groups.cumcount().eq(0) | groups.cumcount(ascending=False).eq(0)
    missing_arrival = ordered["arrival_time"].isna()
    missing_departure = ordered["departure_time"].isna()
    exact = ordered.get("timepoint", pd.Series("0", index=ordered.index)).eq("1")
    bad_times = (
        (
            ((missing_arrival | missing_departure) & (endpoints | exact))
            | (missing_arrival != missing_departure)
        )
        .groupby(ordered["trip_id"], sort=False)
        .any()
    )
    known = ordered.dropna(subset=["arrival_time", "departure_time"])
    preceding = known.groupby("trip_id", sort=False)["departure_time"].shift()
    nonmonotonic = (
        ((known["departure_time"] < known["arrival_time"]) | (known["arrival_time"] < preceding))
        .groupby(known["trip_id"], sort=False)
        .any()
    )
    for trip_id in trip_order:
        if sizes[trip_id] < 2:
            issues.append(f"trip {trip_id}: at least two stop occurrences are required")
        if bad_times[trip_id]:
            issues.append(
                f"trip {trip_id}: supply arrival and departure at endpoints and exact timepoints"
            )
        if nonmonotonic.get(trip_id, False):
            issues.append(f"trip {trip_id}: arrival/departure times must be monotonic")
    missing = set(tables["trips"]["trip_id"]) - set(times["trip_id"])
    if missing:
        issues.append(f"trips without stop_times: {sorted(missing)}")
    if "frequencies" in tables:
        _validate_frequencies(tables["frequencies"], issues)


def _validate_frequencies(frequencies: pd.DataFrame, issues: list[str]) -> None:
    """
    Check headways and period overlap within each trip template.

    Adjacent periods are valid, but a trip cannot mix exact and non-exact periods.

    Parameters
    ----------
    frequencies : pandas.DataFrame
        GTFS frequency periods with string-valued fields.
    issues : list of str
        Diagnostic messages to extend.
    """
    for trip_id, frame in frequencies.groupby("trip_id", sort=False):
        periods: list[tuple[int, int]] = []
        exact_values: set[str] = set()
        for row in _records(frame):
            try:
                start, end = _seconds(row["start_time"]), _seconds(row["end_time"])
                headway = _number(row["headway_secs"], "headway_secs", positive=True)
                if start >= end or not headway.is_integer():
                    msg = "end_time must exceed start_time; headway_secs must be an integer"
                    raise ValueError(msg)  # noqa: TRY301 - attach trip context below
                exact = row.get("exact_times") if _present(row.get("exact_times")) else "0"
                if exact not in {"0", "1"}:
                    msg = "exact_times must be 0 or 1"
                    raise ValueError(msg)  # noqa: TRY301 - attach trip context below
                exact_values.add(exact)
                periods.append((start, end))
            except ValueError as exc:
                issues.append(f"frequencies {trip_id}: {exc}")
        periods.sort()
        if any(left[1] > right[0] for left, right in pairwise(periods)):
            issues.append(f"frequencies {trip_id}: overlapping periods")
        if len(exact_values) > 1:
            issues.append(f"frequencies {trip_id}: inconsistent exact_times")


def _without_stop_wkb(stops: pd.DataFrame) -> pd.DataFrame:
    """Use GTFS coordinates when a pandas stop table also contains stored WKB.

    DuckDB returns binary geometry alongside stop_lat and stop_lon. GeoDataFrame
    geometry remains authoritative and still requires a declared CRS.

    Parameters
    ----------
    stops : pandas.DataFrame
        Stop attributes, optionally read back from a loaded feed.

    Returns
    -------
    pandas.DataFrame
        Table with redundant binary geometry removed, without modifying input.
    """
    if (
        not isinstance(stops, gpd.GeoDataFrame)
        and {"geometry", "stop_lat", "stop_lon"}.issubset(stops)
        and stops["geometry"]
        .map(lambda value: isinstance(value, (bytes, bytearray, memoryview)) or not _present(value))
        .all()
    ):
        return stops.drop(columns="geometry")
    return stops


def _normalise_stops(stops: pd.DataFrame) -> pd.DataFrame:
    """
    Copy boarding locations and derive WGS84 coordinates from geometry.

    Geometry-derived coordinates replace coordinate columns only in the returned copy.

    Parameters
    ----------
    stops : pd.DataFrame
        Boarding locations and optional station metadata.

    Returns
    -------
    pd.DataFrame
        Independent GTFS stop table with WGS84 coordinates.
    """
    if isinstance(stops, gpd.GeoDataFrame):
        if stops.crs is None:
            msg = "stops: GeoDataFrame must have a CRS"
            raise ValueError(msg)
        stops = stops.to_crs(4326).copy()
        if stops.geometry.isna().any() or not stops.geometry.geom_type.eq("Point").all():
            msg = "stops: supply non-empty Point boarding locations"
            raise ValueError(msg)
        stops["stop_lon"], stops["stop_lat"] = stops.geometry.x, stops.geometry.y
    result = _text_frame(stops)
    issues: list[str] = []
    _require(result, ["stop_id"], "stops", issues)
    _validate_stops(result, issues)
    _fail(issues)
    return result


def _pattern_geometry(patterns: pd.DataFrame) -> pd.DataFrame:
    """
    Copy pattern geometry, preserving a declared coordinate reference system.

    Plain DataFrames cannot carry geometry without a declared coordinate system.

    Parameters
    ----------
    patterns : pd.DataFrame
        Route patterns with identifiers, direction labels and optional geometry.

    Returns
    -------
    pd.DataFrame
        Independent pattern table with WGS84 geometry where present.
    """
    if isinstance(patterns, gpd.GeoDataFrame):
        if patterns.crs is None:
            msg = "patterns: GeoDataFrame must have a CRS"
            raise ValueError(msg)
        return cast("pd.DataFrame", patterns.to_crs(4326).copy())
    if "geometry" in patterns and patterns["geometry"].notna().any():
        msg = "patterns: geometry requires a GeoDataFrame with a CRS"
        raise ValueError(msg)
    return patterns.copy()


def _prepare_input(data: TransportData) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Check the common input contract and copy input frames.

    Check identifier relationships before attempting geometry or timing operations.

    Parameters
    ----------
    data : TransportData
        Common transport input tables.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Validated copies of patterns, stop sequences and stops.
    """
    issues: list[str] = []
    patterns = _pattern_geometry(data["patterns"])
    sequence = data["stop_sequences"].copy()
    for frame, keys in (
        (patterns, ["pattern_id"]),
        (sequence, ["pattern_id", "stop_sequence"]),
        (data["stops"], ["stop_id"]),
        (data["routes"], ["route_id"]),
    ):
        if frame[keys].astype(str).duplicated(keys).any():
            issues.append(f"duplicate input keys {keys}")
    for frame in (patterns, sequence):
        frame["pattern_id"] = frame["pattern_id"].astype(str)
    for frame, field, allowed in (
        (patterns, "route_id", data["routes"]["route_id"]),
        (sequence, "stop_id", data["stops"]["stop_id"]),
        (sequence, "pattern_id", patterns["pattern_id"]),
    ):
        if unknown := set(frame[field].astype(str)) - set(allowed.astype(str)):
            issues.append(f"{field}: unknown IDs {sorted(unknown)}")
    occurrence_counts = sequence.groupby("pattern_id", sort=False).size()
    issues.extend(
        f"pattern {pattern_id}: supply at least two ordered stop occurrences"
        for pattern_id in patterns["pattern_id"]
        if occurrence_counts.get(pattern_id, 0) < 2
    )
    directions = pd.to_numeric(patterns["direction_id"], errors="coerce")
    if not directions.isin([0, 1]).all():
        issues.append("patterns.direction_id: expected 0 or 1")
    _fail(issues)
    patterns["direction_id"] = directions.astype("int64")
    return patterns, sequence, _normalise_stops(data["stops"])


def _validate_optional(tables: dict[str, pd.DataFrame], issues: list[str]) -> None:
    """
    Check supported optional fields and transfer/pathway requirements.

    These checks prevent saving dangling or structurally invalid connections.
    They do not imply that the summary graph performs transfer routing.

    Parameters
    ----------
    tables : dict of str to pandas.DataFrame
        Supported GTFS tables with string-valued fields.
    issues : list of str
        Diagnostics to extend without stopping at the first invalid field.
    """
    enums = {
        "trips": {
            "direction_id": {"0", "1"},
            "wheelchair_accessible": {"0", "1", "2"},
            "bikes_allowed": {"0", "1", "2"},
        },
        "stop_times": {
            "pickup_type": {"0", "1", "2", "3"},
            "drop_off_type": {"0", "1", "2", "3"},
            "timepoint": {"0", "1"},
        },
        "transfers": {"transfer_type": {"0", "1", "2", "3", "4", "5"}},
        "pathways": {
            "pathway_mode": {"1", "2", "3", "4", "5", "6", "7"},
            "is_bidirectional": {"0", "1"},
        },
    }
    for name, fields in enums.items():
        if name not in tables:
            continue
        for field, allowed in fields.items():
            if field in tables[name] and not tables[name][field].dropna().isin(allowed).all():
                issues.append(f"{name}.{field}: expected one of {sorted(allowed)}")
    for row in _records(tables.get("transfers", pd.DataFrame())):
        transfer_type = row.get("transfer_type")
        keys = (
            ["from_trip_id", "to_trip_id"]
            if transfer_type in {"4", "5"}
            else ["from_stop_id", "to_stop_id"]
        )
        if any(not _present(row.get(key)) for key in keys):
            issues.append(f"transfers type {transfer_type}: requires {keys}")
        if transfer_type == "2":
            try:
                duration = _number(row.get("min_transfer_time"), "min_transfer_time")
                if not duration.is_integer():
                    issues.append("transfers.min_transfer_time: expected integer seconds")
            except ValueError as exc:
                issues.append(str(exc))
    _validate_stop_hierarchy(tables, issues)
    if "shapes" in tables:
        _validate_shapes(tables["shapes"], issues)


def _validate_stop_hierarchy(tables: dict[str, pd.DataFrame], issues: list[str]) -> None:
    """
    Validate boarding locations and their parent stations.

    Station nodes cannot be used as vehicle boarding stop occurrences.

    Parameters
    ----------
    tables : dict of str to pandas.DataFrame
        GTFS tables including stops and stop_times.
    issues : list of str
        Diagnostics to extend when a parent or boarding location is invalid.
    """
    stops = tables["stops"].set_index("stop_id")
    if "location_type" in stops:
        types = stops.location_type.fillna("0")
        boarding = set(types[types.isin(["0", "4"])].index)
        if invalid := set(tables["stop_times"].stop_id) - boarding:
            issues.append(
                f"stop_times: locations must be boarding stops, not parent stations: {sorted(invalid)}"
            )
        if "parent_station" in stops:
            for stop_id, station in stops.iterrows():
                parent = str(station.get("parent_station"))
                if _present(parent) and parent in types.index:
                    expected = "0" if station.get("location_type") == "4" else "1"
                    if types.loc[parent] != expected or parent == stop_id:
                        issues.append(f"stops {stop_id}: invalid parent_station type")


def _validate_shapes(shapes: pd.DataFrame, issues: list[str]) -> None:
    """
    Check exported shape coordinates and cumulative distance ordering.

    Geometry must remain a directed sequence with finite WGS84 coordinates.

    Parameters
    ----------
    shapes : pandas.DataFrame
        GTFS shape vertices.
    issues : list of str
        Validation diagnostics to extend.
    """
    for field, lower, upper in (("shape_pt_lat", -90, 90), ("shape_pt_lon", -180, 180)):
        values = pd.to_numeric(shapes[field], errors="coerce")
        if values.isna().any() or not values.between(lower, upper).all():
            issues.append(f"shapes.{field}: invalid coordinate")
    sequence = pd.to_numeric(shapes["shape_pt_sequence"], errors="coerce")
    if sequence.isna().any() or (sequence < 0).any() or (sequence % 1 != 0).any():
        issues.append("shapes.shape_pt_sequence: expected non-negative integers")
        return
    # Numeric ordering catches equivalent string keys such as '01' and '1'.
    ordered = shapes.assign(shape_pt_sequence=sequence).sort_values(
        ["shape_id", "shape_pt_sequence"]
    )
    if ordered.duplicated(["shape_id", "shape_pt_sequence"]).any():
        issues.append("shapes: duplicate numeric sequence values")
    if "shape_dist_traveled" in ordered:
        distance = pd.to_numeric(ordered.shape_dist_traveled, errors="coerce")
        supplied = ordered.shape_dist_traveled.notna()
        if (
            distance[supplied].isna().any()
            or not distance[supplied].map(math.isfinite).all()
            or (distance < 0).any()
        ):
            issues.append("shapes.shape_dist_traveled: expected finite non-negative distances")
        if (distance.groupby(ordered.shape_id).diff() < 0).any():
            issues.append("shapes.shape_dist_traveled: distances must follow vertex order")


def _validate_service_options(services: pd.DataFrame, issues: list[str]) -> None:
    """
    Collect malformed service-row values without filling missing conditions.

    Required identifiers cannot contain null values when their columns exist.

    Parameters
    ----------
    services : pandas.DataFrame
        Explicit service rows before supplementation.
    issues : list of str
        Diagnostics to extend.
    """
    if services.empty:
        issues.append("services: must not be empty")
    allowed = {
        "pattern_id",
        "service_id",
        "trip_id",
        "departure_time",
        "start_time",
        "end_time",
        "headway_secs",
        "exact_times",
        "trip_headsign",
        "trip_short_name",
        "wheelchair_accessible",
        "bikes_allowed",
    }
    _unknown_fields(set(services.columns), allowed, "services", issues)
    for field in ("pattern_id", "service_id", "trip_id"):
        if field in services:
            _require(services, [field], "services", issues)
    for index, row in zip(services.index, _records(services), strict=True):
        label = f"services row {index} (pattern_id={row.get('pattern_id', 'all')!r})"
        for field in ("departure_time", "start_time", "end_time"):
            if _present(row.get(field)):
                try:
                    _seconds(row[field])
                except ValueError as exc:
                    issues.append(f"{label}, {field}: {exc}")
        if _present(row.get("headway_secs")):
            try:
                headway = _number(row["headway_secs"], "headway_secs", positive=True)
                if not headway.is_integer():
                    issues.append(f"{label}, headway_secs: must be an integer")
            except ValueError as exc:
                issues.append(f"{label}: {exc}")
        if _present(row.get("departure_time")) and any(
            _present(row.get(field)) for field in ("start_time", "end_time", "headway_secs")
        ):
            issues.append(f"{label}: supply departure_time OR a frequency period")
        if _present(row.get("exact_times")) and str(row["exact_times"]) not in {
            "0",
            "1",
            "0.0",
            "1.0",
        }:
            issues.append(f"{label}, exact_times: expected 0 or 1")


def _unknown_fields(fields: set[str], allowed: set[str], label: str, issues: list[str]) -> None:
    """Collect unsupported settings with spelling suggestions.

    Unsupported configuration is never silently replaced with defaults.

    Parameters
    ----------
    fields : set of str
        Supplied names.
    allowed : set of str
        Supported names.
    label : str
        Input context.
    issues : list of str
        Diagnostics to extend.
    """
    for field in sorted(fields - allowed, key=str):
        candidates = get_close_matches(str(field), sorted(allowed), n=1)
        hint = f"; did you mean {candidates[0]!r}?" if candidates else ""
        issues.append(f"{label}: unknown field {field!r}{hint}")


def _validate_generation_options(services: pd.DataFrame | None, options: Mapping[str, Any]) -> None:
    """
    Reject invalid explicit overrides before any source download.

    Missing optional conditions are distinct from malformed supplied values.

    Parameters
    ----------
    services : pandas.DataFrame or None
        Explicit service rows, if supplied.
    options : mapping
        Calendar and operating keyword overrides.
    """
    issues: list[str] = []
    _unknown_fields(
        set(options),
        set(_GenerationConfig.__dataclass_fields__) | {"agency", "data", "optional_tables"},
        "generation options",
        issues,
    )
    _selection(options.get("pattern_ids"), "pattern_ids")
    if options.get("headway_secs") is not None and options.get("group_headway_secs") is not None:
        issues.append("Specify headway_secs OR group_headway_secs, not both")
    for field in ("speed_kph", "dwell_time_secs", "headway_secs", "group_headway_secs"):
        value = options.get(field)
        if value is not None:
            try:
                number = _number(value, field, positive=field != "dwell_time_secs")
                if field in {"headway_secs", "group_headway_secs"} and not number.is_integer():
                    issues.append(f"{field}: must be an integer")
            except ValueError as exc:
                issues.append(str(exc))
    for field in ("start_time", "end_time"):
        if options.get(field) is not None:
            try:
                _seconds(options[field])
            except ValueError as exc:
                issues.append(f"{field}: {exc}")
    if (
        not issues
        and options.get("start_time") is not None
        and options.get("end_time") is not None
        and _seconds(options["end_time"]) <= _seconds(options["start_time"])
    ):
        issues.append("end_time must exceed start_time; use extended hours after midnight")
    method = options.get("distance_method")
    if method is not None and method not in {"auto", "shape", "geodesic"}:
        issues.append("distance_method must be 'auto', 'shape' or 'geodesic'")
    date = options.get("service_date")
    if date is not None:
        try:
            if not re.fullmatch(r"\d{8}", str(date)):
                raise ValueError  # noqa: TRY301 - collect the date diagnostic
            datetime.strptime(str(date), "%Y%m%d")
        except ValueError:
            issues.append("service_date: expected a valid YYYYMMDD date")
        if options.get("calendar") is not None or options.get("calendar_dates") is not None:
            issues.append("service_date cannot be combined with calendar or calendar_dates")
    if services is not None:
        _validate_service_options(services, issues)
    _validate_transport_options(options.get("data"), issues)
    _validate_timing_options(options.get("data"), issues)
    _validate_metadata_options(options, issues)
    _fail(issues)


def _validate_transport_options(data: Mapping[str, pd.DataFrame] | None, issues: list[str]) -> None:
    """Validate supplied values even on records excluded by source selection.

    Partial supplementation may omit fields, but present values must be valid.

    Parameters
    ----------
    data : mapping or None
        Complete or supplemental transport tables.
    issues : list of str
        Diagnostics to extend before source retrieval.
    """
    codes = {
        "patterns": {"direction_id": {0, 1}},
        "stops": {"location_type": {0, 1, 2, 3, 4}, "wheelchair_boarding": {0, 1, 2}},
        "stop_sequences": {"pickup_type": {0, 1, 2, 3}, "drop_off_type": {0, 1, 2, 3}},
    }
    for name in ("routes", "patterns", "stops", "stop_sequences"):
        if data is None or name not in data:
            continue
        frame = _without_stop_wkb(data[name]) if name == "stops" else data[name]
        if isinstance(frame, gpd.GeoDataFrame) and frame.crs is None:
            issues.append(f"{name}: supplemental geometry requires a CRS")
        for index, row in zip(frame.index, _records(frame), strict=True):
            identifier = row.get("pattern_id", row.get("route_id", row.get("stop_id")))
            label = f"{name} row {index} ({identifier!r})"
            if name in {"stops", "patterns"} and _present(row.get("geometry")):
                geometry = row["geometry"]
                expected = "Point" if name == "stops" else "LineString"
                if getattr(geometry, "geom_type", None) != expected or geometry.is_empty:
                    issues.append(f"{label}, geometry: supply a non-empty {expected}")
            for field, allowed in codes.get(name, {}).items():
                if (
                    _present(row.get(field))
                    and pd.to_numeric(row[field], errors="coerce") not in allowed
                ):
                    issues.append(f"{label}, {field}: expected one of {sorted(allowed)}")
            if name == "routes" and _present(row.get("route_type")):
                mode = pd.to_numeric(row["route_type"], errors="coerce")
                if pd.isna(mode) or not math.isfinite(mode) or mode < 0 or int(mode) != mode:
                    issues.append(f"{label}, route_type: expected a finite non-negative integer")
            if name == "stops":
                for field, limit in (("stop_lat", 90), ("stop_lon", 180)):
                    if _present(row.get(field)):
                        value = pd.to_numeric(row[field], errors="coerce")
                        if pd.isna(value) or not -limit <= value <= limit:
                            issues.append(
                                f"{label}, {field}: expected a number in [-{limit}, {limit}]"
                            )


def _validate_timing_options(data: Mapping[str, pd.DataFrame] | None, issues: list[str]) -> None:
    """Collect malformed explicit timing fields before source retrieval.

    This preflight does not require complete source tables or resolve unused legs.

    Parameters
    ----------
    data : mapping or None
        Complete or supplemental common transport tables.
    issues : list of str
        Diagnostics to extend without inventing missing fields.
    """
    if data is None or "stop_sequences" not in data:
        return
    _unknown_fields(
        set(data["stop_sequences"].columns),
        {
            "pattern_id",
            "stop_id",
            "stop_sequence",
            "speed_kph",
            "distance_m",
            "travel_time_secs",
            "dwell_time_secs",
            "arrival_offset_secs",
            "departure_offset_secs",
            "shape_dist_traveled",
            "pickup_type",
            "drop_off_type",
            "stop_headsign",
            "timepoint",
        },
        "stop_sequences",
        issues,
    )
    for index, row in zip(
        data["stop_sequences"].index, _records(data["stop_sequences"]), strict=True
    ):
        label = f"stop_sequences row {index} (pattern_id={row.get('pattern_id')!r})"
        if _present(row.get("stop_sequence")):
            try:
                sequence = _number(row["stop_sequence"], "stop_sequence")
                if not sequence.is_integer():
                    issues.append(f"{label}, stop_sequence: expected a non-negative integer")
            except ValueError as exc:
                issues.append(f"{label}: {exc}")
        for field in (
            "speed_kph",
            "distance_m",
            "travel_time_secs",
            "dwell_time_secs",
            "arrival_offset_secs",
            "departure_offset_secs",
            "shape_dist_traveled",
        ):
            if _present(row.get(field)):
                try:
                    _number(row[field], field, positive=field == "speed_kph")
                except ValueError as exc:
                    issues.append(f"{label}: {exc}")


def _validate_metadata_options(options: Mapping[str, Any], issues: list[str]) -> None:
    """
    Check supplied agency and calendar metadata before source retrieval.

    Calendar validation is shared with GTFS export after text normalisation.

    Parameters
    ----------
    options : mapping
        Explicit metadata and generation keyword arguments.
    issues : list of str
        Diagnostics to extend.
    """
    agency = options.get("agency")
    if agency is not None and _require(agency, _REQUIRED["agency"], "agency", issues):
        zones = set(agency["agency_timezone"])
        if len(zones) != 1:
            issues.append("agency: all agencies must use the same timezone")
        for zone in zones:
            try:
                ZoneInfo(str(zone))
            except (ZoneInfoNotFoundError, ValueError):
                issues.append(f"agency_timezone: invalid timezone {zone!r}")
    calendars = {}
    for name in ("calendar", "calendar_dates"):
        frame = options.get(name)
        if frame is not None:
            frame = _text_frame(frame)
            if _require(frame, _REQUIRED[name], name, issues):
                calendars[name] = frame
    _validate_calendars(calendars, issues)


def _selection(value: str | Sequence[str] | None, label: str) -> list[str] | None:
    """Normalise a non-empty selection while preserving input order.

    Reject malformed input before any source lookup.

    Parameters
    ----------
    value : str, sequence of str or None
        Requested identifiers, or all available identifiers.
    label : str
        Parameter name for diagnostics.

    Returns
    -------
    list of str or None
        Unique identifiers.
    """
    if value is None:
        return None
    values = [value] if isinstance(value, str) else value
    if (
        not isinstance(values, Sequence)
        or not values
        or any(not isinstance(item, str) or not item.strip() for item in values)
    ):
        msg = f"{label}: supply a non-empty string or sequence of non-empty strings"
        raise ValueError(msg)
    return list(dict.fromkeys(values))


def _select_patterns(
    data: Mapping[str, pd.DataFrame],
    services: pd.DataFrame | None,
    selected: list[str] | None,
    records: list[dict[str, object]],
    *,
    explicit: bool = True,
) -> tuple[TransportData, pd.DataFrame | None]:
    """Select linked transport records and retain parent stops.

    Validate service IDs against the original selection before dropping known rows.

    Parameters
    ----------
    data : mapping
        Transport tables, optionally without stops during source preparation.
    services : pandas.DataFrame or None
        Explicit service overrides.
    selected : list of str or None
        Pattern IDs, or all available patterns.
    records : list of dict
        Metadata records to extend with deliberate exclusions.
    explicit : bool, default True
        Whether pattern IDs were supplied directly rather than derived by a loader.

    Returns
    -------
    tuple
        Selected transport tables and service rows.
    """
    available = set(data["patterns"]["pattern_id"].astype(str))
    for label, requested in (
        ("pattern_ids", set(selected or [])),
        (
            "services.pattern_id",
            set(services["pattern_id"].astype(str))
            if services is not None and "pattern_id" in services
            else set(),
        ),
    ):
        if unknown := requested - available:
            msg = f"{label}: unknown IDs {sorted(unknown)}"
            raise ValueError(msg)
    result = cast("TransportData", dict(data))
    if selected is None:
        return result, services
    keep = set(selected)
    if not keep:
        msg = "pattern_ids: selection contains no patterns"
        raise ValueError(msg)
    records.append(
        {
            "table_name": "assumptions",
            "record_id": "generation",
            "field": "pattern_ids",
            "kind": "user" if explicit else "generated",
            "source": "input" if explicit else "selection",
            "value": json.dumps(selected),
            "details": "Pattern selection; excluded patterns: "
            + json.dumps(sorted(available - keep)),
        }
    )
    result["patterns"] = (
        data["patterns"].loc[data["patterns"]["pattern_id"].astype(str).isin(keep)].copy()
    )
    result["routes"] = (
        data["routes"]
        .loc[
            data["routes"]["route_id"].astype(str).isin(result["patterns"]["route_id"].astype(str))
        ]
        .copy()
    )
    result["stop_sequences"] = (
        data["stop_sequences"]
        .loc[data["stop_sequences"]["pattern_id"].astype(str).isin(keep)]
        .copy()
    )
    if "stops" in data:
        stops = data["stops"]
        stop_ids = set(result["stop_sequences"]["stop_id"].astype(str))
        if "parent_station" in stops:
            while True:
                parents = stops.loc[stops["stop_id"].astype(str).isin(stop_ids), "parent_station"]
                expanded = stop_ids | {str(item) for item in parents if _present(item)}
                if expanded == stop_ids:
                    break
                stop_ids = expanded
        result["stops"] = stops.loc[stops["stop_id"].astype(str).isin(stop_ids)].copy()
    if services is not None and "pattern_id" in services:
        services = services.loc[services["pattern_id"].astype(str).isin(keep)].copy()
        if services.empty:
            services = None
    return result, services


def _transport_modes(modes: str | Sequence[str] | None) -> list[str]:
    """Validate transport groups and expand the tram alias.

    Normalised names are used for source queries and response filtering.

    Parameters
    ----------
    modes : str, sequence of str or None
        Requested transport modes; None selects all supported modes.

    Returns
    -------
    list of str
        OSM route tags belonging to the selected groups.
    """
    values = _selection(modes, "modes")
    selected = set(_MODES) if values is None else {value.strip().lower() for value in values}
    if unknown := selected - set(_MODES):
        msg = f"modes: unknown names {sorted(unknown)}; choose from {sorted(_MODES)}"
        raise ValueError(msg)
    if selected & {"tram", "light_rail"}:
        selected.update({"tram", "light_rail"})
    return sorted(selected)


def _matching_patterns(
    data: Mapping[str, pd.DataFrame],
    modes: list[str] | None,
    requested: list[str] | None,
) -> list[str]:
    """
    Intersect effective route modes and pattern identifiers.

    The selection preserves input pattern order and retains original route-type codes.

    Parameters
    ----------
    data : Mapping[str, pd.DataFrame]
        Explicit transport tables or supplemental input.
    modes : list[str] | None
        Requested transport mode groups, or all supported modes.
    requested : list[str] | None
        Requested pattern IDs, or all matching patterns.

    Returns
    -------
    list of str
        Matching pattern identifiers in input order.
    """
    routes = data["routes"]
    if modes is not None:
        route_types = pd.to_numeric(routes["route_type"], errors="coerce").replace(
            _ROUTE_TYPE_GROUPS
        )
        routes = routes.loc[route_types.isin({_MODES[mode] for mode in modes})]
    allowed = set(routes["route_id"].astype(str))
    chosen: list[str] = (
        data["patterns"]
        .loc[data["patterns"]["route_id"].astype(str).isin(allowed), "pattern_id"]
        .astype(str)
        .tolist()
    )
    if requested is not None:
        identifiers = set(requested)
        chosen = [identifier for identifier in chosen if identifier in identifiers]
    return chosen
