"""
Overture Maps transportation segment rule evaluation and directed expansion.

This private module implements Overture Maps schema parsing for
``access_restrictions``, ``speed_limits``, ``road_flags``, ``subtype``, and
``class``, aligning linearly referenced ``between`` intervals with connector
split bounds.
"""

from __future__ import annotations

import itertools
import json
import math
from typing import TYPE_CHECKING
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

if TYPE_CHECKING:
    from collections.abc import Sequence

# Official Overture Maps TravelMode enum values.
OVERTURE_TRAVEL_MODES: frozenset[str] = frozenset(
    {
        "vehicle",
        "motor_vehicle",
        "car",
        "truck",
        "motorcycle",
        "foot",
        "bicycle",
        "bus",
        "hgv",
        "hov",
        "emergency",
    }
)

# Common non-Overture mode aliases mapped to helpful suggestions.
_MODE_HINTS: dict[str, str] = {
    "auto": "car",
    "bike": "bicycle",
    "cycle": "bicycle",
    "cycling": "bicycle",
    "drive": "car",
    "driving": "car",
    "pedestrian": "foot",
    "walk": "foot",
    "walking": "foot",
}

# TravelMode inheritance hierarchy (self + ancestor modes).
_MODE_ANCESTORS: dict[str, frozenset[str]] = {
    "vehicle": frozenset({"vehicle"}),
    "motor_vehicle": frozenset({"motor_vehicle", "vehicle"}),
    "car": frozenset({"car", "motor_vehicle", "vehicle"}),
    "motorcycle": frozenset({"motorcycle", "motor_vehicle", "vehicle"}),
    "bus": frozenset({"bus", "motor_vehicle", "vehicle"}),
    "truck": frozenset({"truck", "motor_vehicle", "vehicle"}),
    "hgv": frozenset({"hgv", "truck", "motor_vehicle", "vehicle"}),
    "hov": frozenset({"hov", "motor_vehicle", "vehicle"}),
    "emergency": frozenset({"emergency", "motor_vehicle", "vehicle"}),
    "bicycle": frozenset({"bicycle", "vehicle"}),
    "foot": frozenset({"foot"}),
}

_GENERAL_SPEED_MODES: frozenset[str] = frozenset({"vehicle", "motor_vehicle", "car"})
_MOTORIZED_MODES: frozenset[str] = frozenset(
    {
        "vehicle",
        "motor_vehicle",
        "car",
        "truck",
        "motorcycle",
        "bus",
        "hgv",
        "hov",
        "emergency",
    }
)
_VEHICULAR_ROAD_CLASSES: frozenset[str] = frozenset(
    {
        "motorway",
        "trunk",
        "primary",
        "secondary",
        "tertiary",
        "residential",
        "living_street",
        "unclassified",
        "service",
    }
)

# Default passable modes for each Overture RoadClass under subtype == "road".
_ROAD_CLASS_DEFAULT_ACCESS: dict[str, frozenset[str]] = {
    "motorway": _MOTORIZED_MODES,
    "trunk": OVERTURE_TRAVEL_MODES,
    "primary": OVERTURE_TRAVEL_MODES,
    "secondary": OVERTURE_TRAVEL_MODES,
    "tertiary": OVERTURE_TRAVEL_MODES,
    "residential": OVERTURE_TRAVEL_MODES,
    "living_street": OVERTURE_TRAVEL_MODES,
    "unclassified": OVERTURE_TRAVEL_MODES,
    "service": OVERTURE_TRAVEL_MODES,
    "pedestrian": frozenset({"foot", "emergency"}),
    "footway": frozenset({"foot"}),
    "steps": frozenset({"foot"}),
    "bridleway": frozenset({"foot"}),
    "cycleway": frozenset({"bicycle"}),
    "path": frozenset({"foot", "bicycle"}),
    "track": frozenset({"foot", "bicycle"}),
    "unknown": frozenset(),
}

_IMPASSABLE_ROAD_FLAGS: frozenset[str] = frozenset({"is_under_construction", "is_abandoned"})
_SPEED_UNIT_TO_KPH: dict[str, float] = {
    "km/h": 1.0,
    "mph": 1.609344,
}

# Tolerance for ignoring tiny boundary-snapping slivers at connector splits.
_JITTER_REL_TOL = 0.01
_JITTER_ABS_TOL = 1e-3


def validate_travel_mode(travel_mode: str | None) -> str | None:
    """
    Validate the ``travel_mode`` argument for Overture segment processing.

    Accepts ``None`` (no directed expansion), ``"all"`` (all traversable roads),
    or any official Overture ``TravelMode`` string.

    Parameters
    ----------
    travel_mode : str or None
        Requested travel mode.

    Returns
    -------
    str or None
        Validated travel mode string, or ``None``.

    Raises
    ------
    ValueError
        If ``travel_mode`` is not ``None``, ``"all"``, or a valid Overture
        ``TravelMode``.
    """
    if travel_mode is None or travel_mode == "all" or travel_mode in OVERTURE_TRAVEL_MODES:
        return travel_mode

    hint = _MODE_HINTS.get(travel_mode.lower()) if isinstance(travel_mode, str) else None
    valid_list = ", ".join(["'all'", *sorted(f"'{m}'" for m in OVERTURE_TRAVEL_MODES)])
    hint_text = f" Did you mean '{hint}'?" if hint else ""
    msg = f"Invalid travel_mode {travel_mode!r}.{hint_text} Expected None or one of: {valid_list}."
    raise ValueError(msg)


def _parse_sequence(value: object) -> list[object]:
    """
    Normalise a nested Overture attribute value into a Python list.

    Handles Python lists and tuples, PyArrow/GeoParquet ``numpy.ndarray``
    objects, JSON or Python-literal strings, single dicts, and null values.

    Parameters
    ----------
    value : object
        Raw cell or nested field value.

    Returns
    -------
    list[object]
        Normalised list of elements, or an empty list for null/invalid inputs.
    """
    result: list[object] = []
    if isinstance(value, (list, tuple)):
        result = list(value)
    elif isinstance(value, np.ndarray):
        result = _parse_sequence(value.item()) if value.ndim == 0 else list(value.tolist())
    elif isinstance(value, dict):
        result = [value]
    elif isinstance(value, str):
        stripped = value.strip()
        parsed = _decode_json_like(stripped) if stripped else None
        if isinstance(parsed, list):
            result = parsed
        elif parsed is not None:
            result = [parsed]
    return result


def _decode_json_like(text: str) -> object:
    """
    Decode a JSON string, falling back to Python-literal replacements.

    Matches the permissive string parsing used for Overture GeoJSON exports.

    Parameters
    ----------
    text : str
        Non-empty JSON or Python-literal text.

    Returns
    -------
    object
        Decoded Python object, or ``None`` if parsing fails.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        normalized = (
            text.replace("'", '"')
            .replace("None", "null")
            .replace("True", "true")
            .replace("False", "false")
        )
        try:
            return json.loads(normalized)
        except json.JSONDecodeError:
            return None


def _parse_rule_list(value: object) -> list[dict[str, Any]]:
    """
    Extract valid dictionary rules from an Overture rule column cell.

    Non-dictionary entries and null values are discarded.

    Parameters
    ----------
    value : object
        Raw rule list cell from a GeoDataFrame column.

    Returns
    -------
    list[dict[str, Any]]
        List of parsed rule dictionaries.
    """
    return [item for item in _parse_sequence(value) if isinstance(item, dict)]


def _extract_between(rule: dict[str, Any]) -> tuple[float, float] | None:
    """
    Extract a normalised linear-reference interval ``(start, end)`` from a rule.

    Missing or ``None`` ``between`` values denote the full segment ``(0.0, 1.0)``.
    Malformed intervals return ``None`` so callers can skip them safely.

    Parameters
    ----------
    rule : dict[str, Any]
        Overture rule dictionary.

    Returns
    -------
    tuple[float, float] or None
        Valid ``(start, end)`` interval within ``[0.0, 1.0]``, or ``None`` if
        malformed.
    """
    between = rule.get("between")
    if between is None:
        return (0.0, 1.0)
    values = _parse_sequence(between)
    if len(values) != 2:
        return None
    try:
        start = float(values[0])  # type: ignore[arg-type]
        end = float(values[1])  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(start) and math.isfinite(end)) or not (0.0 <= start < end <= 1.0):
        return None
    return (start, end)


def _atomic_subintervals(
    lo: float,
    hi: float,
    intervals: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    """
    Partition ``[lo, hi]`` into non-trivial sub-intervals induced by rule bounds.

    Tiny boundary-snapping slivers at connector split positions are dropped so
    floating-point projection jitter does not bleed rules across adjacent splits.

    Parameters
    ----------
    lo : float
        Lower bound of the sub-segment in ``[0.0, 1.0]``.
    hi : float
        Upper bound of the sub-segment in ``[0.0, 1.0]``.
    intervals : Sequence[tuple[float, float]]
        Rule intervals ``(start, end)`` in the original segment's linear reference.

    Returns
    -------
    list[tuple[float, float]]
        Ordered non-overlapping sub-intervals covering the non-jitter interior
        of ``[lo, hi]``.
    """
    if not intervals or all(start <= lo and end >= hi for start, end in intervals):
        return [(lo, hi)]

    points = {lo, hi}
    for start, end in intervals:
        if end <= lo or start >= hi:
            continue
        if lo < start < hi:
            points.add(start)
        if lo < end < hi:
            points.add(end)

    if len(points) == 2:
        return [(lo, hi)]

    sorted_points = sorted(points)
    tol = min(_JITTER_REL_TOL * (hi - lo), _JITTER_ABS_TOL)
    parts = [
        (left, right) for left, right in itertools.pairwise(sorted_points) if (right - left) >= tol
    ]
    return parts or [(lo, hi)]


def _clean_text(value: object) -> str | None:
    """
    Normalise an optional string attribute, treating nulls and blanks as ``None``.

    Non-string values and whitespace-only strings return ``None``.

    Parameters
    ----------
    value : object
        Cell value from ``subtype`` or ``class``.

    Returns
    -------
    str or None
        Stripped string value, or ``None`` when absent.
    """
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _impassable_flag_intervals(flag_rules: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """
    Collect linear-reference intervals flagged as under construction or abandoned.

    Malformed ``between`` bounds are ignored safely.

    Parameters
    ----------
    flag_rules : list[dict[str, Any]]
        Parsed ``road_flags`` rule dictionaries.

    Returns
    -------
    list[tuple[float, float]]
        Intervals where road flags mark the segment impassable.
    """
    intervals: list[tuple[float, float]] = []
    for rule in flag_rules:
        values = {str(v) for v in _parse_sequence(rule.get("values"))}
        if not (values & _IMPASSABLE_ROAD_FLAGS):
            continue
        interval = _extract_between(rule)
        if interval is not None:
            intervals.append(interval)
    return intervals


def _has_conditional_scope(when: dict[str, Any]) -> bool:
    """
    Check whether a ``when`` clause contains time or vehicle-dimension conditions.

    Static general routing skips rules scoped to specific hours or vehicle
    dimensions so temporary or heavy-vehicle limits do not block general traffic.

    Parameters
    ----------
    when : dict[str, Any]
        Rule ``when`` scoping dictionary.

    Returns
    -------
    bool
        ``True`` if ``during`` or ``vehicle`` constraints are present.
    """
    during = when.get("during")
    if isinstance(during, str) and during.strip():
        return True
    return bool(_parse_sequence(when.get("vehicle")))


def _filter_access_rules(
    access_rules: list[dict[str, Any]],
    *,
    travel_mode: str,
    road_class: str | None,
    heading: str,
) -> list[tuple[float, float, bool]]:
    """
    Filter and normalise ``access_restrictions`` for one mode and heading.

    Implements Overture's ``TravelMode`` inheritance, skips time/vehicle-scoped
    rules, treats standalone ``as_private`` allowances as non-public, and keeps
    pedestrian travel bidirectional on vehicular roads with mode-unscoped
    one-way restrictions.

    Parameters
    ----------
    access_rules : list[dict[str, Any]]
        Parsed ``access_restrictions`` rule dictionaries.
    travel_mode : str
        Target mode (``"all"`` or an Overture ``TravelMode``).
    road_class : str or None
        Segment ``class`` value, if available.
    heading : str
        Direction of travel (``"forward"`` or ``"backward"``).

    Returns
    -------
    list[tuple[float, float, bool]]
        Ordered ``(start, end, passable)`` tuples in rule evaluation order.
    """
    applicable: list[tuple[float, float, bool]] = []
    ancestors = _MODE_ANCESTORS.get(travel_mode)

    for rule in access_rules:
        interval = _extract_between(rule)
        if interval is None:
            continue

        when_raw = rule.get("when")
        when = when_raw if isinstance(when_raw, dict) else {}
        if _has_conditional_scope(when):
            continue

        rule_heading = _clean_text(when.get("heading"))
        if rule_heading is not None and rule_heading != heading:
            continue

        rule_modes = {str(m) for m in _parse_sequence(when.get("mode"))}
        if travel_mode == "all":
            if rule_modes:
                continue
        else:
            if rule_modes and ancestors is not None and not (rule_modes & ancestors):
                continue
            if (
                travel_mode == "foot"
                and not rule_modes
                and rule_heading is not None
                and road_class in _VEHICULAR_ROAD_CLASSES
            ):
                continue

        access_type = _clean_text(rule.get("access_type"))
        if access_type not in {"allowed", "designated", "denied"}:
            continue

        recognized = {str(r) for r in _parse_sequence(when.get("recognized"))}
        using = {str(u) for u in _parse_sequence(when.get("using"))}
        if recognized or using:
            # Real Overture exports frequently represent OSM access=private as a
            # standalone allowed rule with recognized=["as_private"].
            if (
                access_type in {"allowed", "designated"}
                and recognized == {"as_private"}
                and not using
            ):
                applicable.append((interval[0], interval[1], False))
            continue

        applicable.append((interval[0], interval[1], access_type != "denied"))

    return applicable


def _default_access(
    subtype: str | None,
    road_class: str | None,
    travel_mode: str,
) -> bool:
    """
    Determine baseline passability from Overture ``subtype`` and ``class``.

    Non-road subtypes default to impassable, while road classes consult the
    Overture mode-access matrix before explicit rule overrides.

    Parameters
    ----------
    subtype : str or None
        Overture segment ``subtype`` (``"road"``, ``"rail"``, ``"water"``).
    road_class : str or None
        Overture segment ``class``.
    travel_mode : str
        Target mode (``"all"`` or an Overture ``TravelMode``).

    Returns
    -------
    bool
        Baseline passability before ``road_flags`` and ``access_restrictions``.
    """
    if subtype is not None and subtype != "road":
        return False
    if travel_mode == "all" or road_class is None:
        return True
    return travel_mode in _ROAD_CLASS_DEFAULT_ACCESS.get(road_class, frozenset())


def _evaluate_heading_access(
    *,
    default_passable: bool,
    flag_intervals: Sequence[tuple[float, float]],
    rules: Sequence[tuple[float, float, bool]],
    lo: float,
    hi: float,
) -> bool:
    """
    Evaluate end-to-end passability along ``[lo, hi]`` for a single heading.

    Applies Overture's last-matching-rule-wins semantics on every non-trivial
    atomic sub-interval and requires all sub-intervals to be passable.

    Parameters
    ----------
    default_passable : bool
        Baseline passability from ``subtype`` and ``class``.
    flag_intervals : Sequence[tuple[float, float]]
        Intervals marked impassable by ``road_flags``.
    rules : Sequence[tuple[float, float, bool]]
        Filtered ``(start, end, passable)`` access rules in list order.
    lo : float
        Sub-segment lower bound.
    hi : float
        Sub-segment upper bound.

    Returns
    -------
    bool
        ``True`` if the segment is passable end-to-end in this heading.
    """
    all_intervals = [*flag_intervals, *((start, end) for start, end, _ in rules)]
    for left, right in _atomic_subintervals(lo, hi, all_intervals):
        midpoint = 0.5 * (left + right)
        blocked_by_flag = any(start <= midpoint <= end for start, end in flag_intervals)
        passable = False if blocked_by_flag else default_passable
        for start, end, rule_passable in rules:
            if start <= midpoint <= end:
                passable = rule_passable
        if not passable:
            return False
    return True


def _evaluate_row_access(
    *,
    subtype: str | None,
    road_class: str | None,
    access_rules: list[dict[str, Any]],
    flag_rules: list[dict[str, Any]],
    lo: float,
    hi: float,
    travel_mode: str,
) -> tuple[bool, bool]:
    """
    Evaluate ``(forward, backward)`` passability for a single segment row.

    Combines ``subtype`` and ``class`` defaults, impassable ``road_flags``, and
    filtered ``access_restrictions`` independently for both directions.

    Parameters
    ----------
    subtype : str or None
        Overture segment ``subtype``.
    road_class : str or None
        Overture segment ``class``.
    access_rules : list[dict[str, Any]]
        Parsed ``access_restrictions`` rules.
    flag_rules : list[dict[str, Any]]
        Parsed ``road_flags`` rules.
    lo : float
        Sub-segment lower bound in ``[0.0, 1.0]``.
    hi : float
        Sub-segment upper bound in ``[0.0, 1.0]``.
    travel_mode : str
        Target mode (``"all"`` or an Overture ``TravelMode``).

    Returns
    -------
    tuple[bool, bool]
        ``(forward_passable, backward_passable)``.
    """
    if travel_mode == "all" and subtype is not None and subtype != "road":
        return (False, False)

    default_passable = _default_access(subtype, road_class, travel_mode)
    flag_intervals = _impassable_flag_intervals(flag_rules)

    fwd_rules = _filter_access_rules(
        access_rules, travel_mode=travel_mode, road_class=road_class, heading="forward"
    )
    bwd_rules = _filter_access_rules(
        access_rules, travel_mode=travel_mode, road_class=road_class, heading="backward"
    )
    return (
        _evaluate_heading_access(
            default_passable=default_passable,
            flag_intervals=flag_intervals,
            rules=fwd_rules,
            lo=lo,
            hi=hi,
        ),
        _evaluate_heading_access(
            default_passable=default_passable,
            flag_intervals=flag_intervals,
            rules=bwd_rules,
            lo=lo,
            hi=hi,
        ),
    )


def _extract_speed_kph(rule: dict[str, Any]) -> float | None:
    """
    Extract a positive normalised speed in km/h from a ``speed_limits`` rule.

    Converts ``km/h`` and ``mph`` values to km/h and returns ``None`` for
    missing, non-positive, non-finite, or unsupported-unit entries.

    Parameters
    ----------
    rule : dict[str, Any]
        Overture speed limit rule dictionary.

    Returns
    -------
    float or None
        Normalised maximum speed in km/h, or ``None`` if absent or invalid.
    """
    max_speed = rule.get("max_speed")
    if not isinstance(max_speed, dict):
        return None
    unit = _clean_text(max_speed.get("unit"))
    factor = _SPEED_UNIT_TO_KPH.get(unit) if unit is not None else None
    if factor is None:
        return None
    try:
        numeric = float(max_speed["value"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric <= 0.0:
        return None
    return numeric * factor


def _evaluate_heading_speed(
    speed_rules: list[dict[str, Any]],
    *,
    lo: float,
    hi: float,
    travel_mode: str | None,
    heading: str,
) -> float:
    """
    Compute the effective maximum speed (km/h) along ``[lo, hi]`` for one heading.

    Uses last-matching-rule-wins per atomic sub-interval and aggregates distinct
    sub-interval speeds with the length-weighted harmonic mean so ``length /
    speed`` stays exact.

    Parameters
    ----------
    speed_rules : list[dict[str, Any]]
        Parsed ``speed_limits`` rules.
    lo : float
        Sub-segment lower bound.
    hi : float
        Sub-segment upper bound.
    travel_mode : str or None
        Requested travel mode, ``"all"``, or ``None``.
    heading : str
        Direction of travel (``"forward"`` or ``"backward"``).

    Returns
    -------
    float
        Effective maximum speed in km/h, or ``np.nan`` if no valid rule matches.
    """
    allowed_modes = (
        _GENERAL_SPEED_MODES
        if travel_mode is None or travel_mode == "all"
        else _MODE_ANCESTORS[travel_mode]
    )
    applicable: list[tuple[float, float, float]] = []

    for rule in speed_rules:
        speed_kph = _extract_speed_kph(rule)
        if speed_kph is None:
            continue
        interval = _extract_between(rule)
        if interval is None:
            continue

        when_raw = rule.get("when")
        when = when_raw if isinstance(when_raw, dict) else {}
        if _has_conditional_scope(when):
            continue
        if _parse_sequence(when.get("recognized")) or _parse_sequence(when.get("using")):
            continue

        rule_heading = _clean_text(when.get("heading"))
        if rule_heading is not None and rule_heading != heading:
            continue

        rule_modes = {str(m) for m in _parse_sequence(when.get("mode"))}
        if rule_modes and not (rule_modes & allowed_modes):
            continue

        applicable.append((interval[0], interval[1], speed_kph))

    if not applicable:
        return np.nan

    intervals = [(start, end) for start, end, _ in applicable]
    weight_sum = 0.0
    inv_sum = 0.0
    for left, right in _atomic_subintervals(lo, hi, intervals):
        midpoint = 0.5 * (left + right)
        speed: float | None = None
        for start, end, value in applicable:
            if start <= midpoint <= end:
                speed = value
        if speed is not None:
            span = right - left
            weight_sum += span
            inv_sum += span / speed

    if weight_sum <= 0.0 or inv_sum <= 0.0:
        return np.nan
    return weight_sum / inv_sum


def _evaluate_unexpanded_speed(
    speed_rules: list[dict[str, Any]],
    *,
    lo: float,
    hi: float,
) -> float:
    """
    Evaluate a single scalar maximum speed (km/h) for an unexpanded segment.

    Evaluates forward and backward directions and returns the conservative
    minimum when directional speed limits differ.

    Parameters
    ----------
    speed_rules : list[dict[str, Any]]
        Parsed ``speed_limits`` rules.
    lo : float
        Sub-segment lower bound.
    hi : float
        Sub-segment upper bound.

    Returns
    -------
    float
        Normalised maximum speed in km/h, or ``np.nan``.
    """
    fwd = _evaluate_heading_speed(speed_rules, lo=lo, hi=hi, travel_mode=None, heading="forward")
    bwd = _evaluate_heading_speed(speed_rules, lo=lo, hi=hi, travel_mode=None, heading="backward")
    fwd_ok = math.isfinite(fwd)
    bwd_ok = math.isfinite(bwd)
    if fwd_ok and bwd_ok:
        return min(fwd, bwd)
    if fwd_ok:
        return fwd
    if bwd_ok:
        return bwd
    return np.nan


def _split_bounds(segments_gdf: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract valid ``(split_from, split_to)`` arrays with ``(0.0, 1.0)`` fallback.

    Rows whose bounds are missing, non-finite, or out of order fall back to the
    full segment span ``(0.0, 1.0)``.

    Parameters
    ----------
    segments_gdf : geopandas.GeoDataFrame
        Segment frame, optionally carrying ``split_from`` and ``split_to``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Float arrays of lower and upper linear-reference bounds.
    """
    n = len(segments_gdf)
    if "split_from" in segments_gdf.columns:
        lo = (
            pd.to_numeric(segments_gdf["split_from"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
    else:
        lo = np.zeros(n, dtype=float)
    if "split_to" in segments_gdf.columns:
        hi = (
            pd.to_numeric(segments_gdf["split_to"], errors="coerce")
            .fillna(1.0)
            .to_numpy(dtype=float)
        )
    else:
        hi = np.ones(n, dtype=float)

    invalid = ~(np.isfinite(lo) & np.isfinite(hi) & (lo >= 0.0) & (lo < hi) & (hi <= 1.0))
    if invalid.any():
        lo = np.where(invalid, 0.0, lo)
        hi = np.where(invalid, 1.0, hi)
    return lo, hi


def _column_array(segments_gdf: gpd.GeoDataFrame, name: str) -> np.ndarray:
    """
    Extract a column as a numpy object array, or an array of ``None`` if absent.

    Avoids mutating the caller's GeoDataFrame when optional Overture attribute
    columns are missing.

    Parameters
    ----------
    segments_gdf : geopandas.GeoDataFrame
        Input segment GeoDataFrame.
    name : str
        Column name to extract.

    Returns
    -------
    numpy.ndarray
        Object array of length ``len(segments_gdf)``.
    """
    if name in segments_gdf.columns:
        return np.asarray(segments_gdf[name].to_numpy(dtype=object), dtype=object)
    return np.full(len(segments_gdf), None, dtype=object)


def apply_overture_routing_attributes(
    segments_gdf: gpd.GeoDataFrame,
    *,
    travel_mode: str | None,
    get_speed: bool,
) -> gpd.GeoDataFrame:
    """
    Expand Overture segments into directed rows and/or extract ``max_speed_kph``.

    When ``travel_mode`` is ``None``, rows are kept in place (1 physical segment
    per row) and ``max_speed_kph`` is added only when ``get_speed=True``.
    Otherwise, each segment is evaluated for ``travel_mode`` in forward and
    backward directions, impassable segments are omitted, and backward rows have
    their LineString geometries reversed.

    Parameters
    ----------
    segments_gdf : geopandas.GeoDataFrame
        Processed Overture segments (after optional connector splitting).
    travel_mode : str or None
        Validated travel mode (``None``, ``"all"``, or an Overture ``TravelMode``).
    get_speed : bool
        Whether to add the unit-normalised ``max_speed_kph`` column.

    Returns
    -------
    geopandas.GeoDataFrame
        Updated GeoDataFrame, expanded to directed rows when ``travel_mode`` is
        not ``None``.
    """
    if travel_mode is None and not get_speed:
        return segments_gdf

    lo_arr, hi_arr = _split_bounds(segments_gdf)
    speed_col = _column_array(segments_gdf, "speed_limits") if get_speed else None

    if travel_mode is None:
        assert speed_col is not None
        speeds = [
            _evaluate_unexpanded_speed(_parse_rule_list(raw), lo=float(lo), hi=float(hi))
            for raw, lo, hi in zip(speed_col, lo_arr, hi_arr, strict=True)
        ]
        result = segments_gdf.copy()
        result["max_speed_kph"] = pd.Series(speeds, index=result.index, dtype="float64")
        return result

    subtypes = _column_array(segments_gdf, "subtype")
    classes = _column_array(segments_gdf, "class")
    access_col = _column_array(segments_gdf, "access_restrictions")
    flags_col = _column_array(segments_gdf, "road_flags")

    row_indices: list[int] = []
    reverse_flags: list[bool] = []
    speeds_out: list[float] = []

    for idx in range(len(segments_gdf)):
        lo = float(lo_arr[idx])
        hi = float(hi_arr[idx])
        fwd, bwd = _evaluate_row_access(
            subtype=_clean_text(subtypes[idx]),
            road_class=_clean_text(classes[idx]),
            access_rules=_parse_rule_list(access_col[idx]),
            flag_rules=_parse_rule_list(flags_col[idx]),
            lo=lo,
            hi=hi,
            travel_mode=travel_mode,
        )
        if not fwd and not bwd:
            continue

        parsed_speeds = _parse_rule_list(speed_col[idx]) if speed_col is not None else []
        if fwd:
            row_indices.append(idx)
            reverse_flags.append(False)
            if get_speed:
                speeds_out.append(
                    _evaluate_heading_speed(
                        parsed_speeds,
                        lo=lo,
                        hi=hi,
                        travel_mode=travel_mode,
                        heading="forward",
                    )
                )
        if bwd:
            row_indices.append(idx)
            reverse_flags.append(True)
            if get_speed:
                speeds_out.append(
                    _evaluate_heading_speed(
                        parsed_speeds,
                        lo=lo,
                        hi=hi,
                        travel_mode=travel_mode,
                        heading="backward",
                    )
                )

    result = segments_gdf.iloc[row_indices].copy().reset_index(drop=True)
    if reverse_flags:
        rev_mask = np.asarray(reverse_flags, dtype=bool)
        if rev_mask.any():
            geoms = result.geometry.to_numpy().copy()
            to_reverse = rev_mask & pd.notna(geoms)
            if to_reverse.any():
                geoms[to_reverse] = shapely.reverse(geoms[to_reverse])
                result.geometry = gpd.GeoSeries(geoms, index=result.index, crs=segments_gdf.crs)

    if get_speed:
        result["max_speed_kph"] = pd.Series(speeds_out, index=result.index, dtype="float64")

    return result
