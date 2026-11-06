"""GTFS input contracts, scalar conversions and the shared defaults registry."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from dataclasses import field
from datetime import date
from importlib.resources import files
from typing import TYPE_CHECKING
from typing import Any
from typing import NotRequired
from typing import TypedDict
from typing import cast

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

_REGISTRY: dict[str, Any] = json.loads(
    files("city2graph.gtfs").joinpath("defaults.json").read_text(encoding="utf-8")
)


_ROUTE_TYPE_GROUPS: dict[int, int] = {
    extended: int(base)
    for base, codes in _REGISTRY["extended_route_types"].items()
    for extended in codes
}


def _default(field: str) -> str | int | float | bool:
    """Read a setting from the versioned generation registry.

    The same registry supplies the runtime model and generated documentation tables.

    Parameters
    ----------
    field : str
        Setting name.

    Returns
    -------
    object
        Registered default value.
    """
    return cast("str | int | float | bool", _REGISTRY["settings"][field]["value"])


class GTFSGenerationWarning(UserWarning):
    """Warn that generated service uses estimates, assumptions or placeholders.

    Inspect ``get_gtfs_summary`` for adopted values and rejection reasons.
    Standard Python warning filters can promote this warning to an error.
    """


class TransportData(TypedDict):
    """
    Tables accepted by :func:`create_gtfs` and prepared by source adapters.

    Identifiers link route patterns, stop occurrences and their source records.
    """

    routes: pd.DataFrame
    stops: pd.DataFrame
    patterns: pd.DataFrame
    stop_sequences: pd.DataFrame
    metadata: NotRequired[pd.DataFrame]


_METADATA = "_city2graph_metadata"
_METADATA_FILENAME = "city2graph_metadata.json"


_METADATA_COLUMNS = ["table_name", "record_id", "field", "kind", "source", "value", "details"]


_REQUIRED = {
    "agency": ["agency_name", "agency_url", "agency_timezone"],
    "stops": ["stop_id"],
    "routes": ["route_id", "route_type"],
    "trips": ["route_id", "service_id", "trip_id"],
    "stop_times": ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"],
    "calendar": [
        "service_id",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "start_date",
        "end_date",
    ],
    "calendar_dates": ["service_id", "date", "exception_type"],
    "shapes": ["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"],
    "frequencies": ["trip_id", "start_time", "end_time", "headway_secs"],
    "transfers": ["transfer_type"],
    "pathways": ["pathway_id", "from_stop_id", "to_stop_id", "pathway_mode", "is_bidirectional"],
    "levels": ["level_id", "level_index"],
    "feed_info": ["feed_publisher_name", "feed_publisher_url", "feed_lang"],
    "attributions": ["organization_name"],
}


_KEYS = {
    "agency": ["agency_id"],
    "stops": ["stop_id"],
    "routes": ["route_id"],
    "trips": ["trip_id"],
    "stop_times": ["trip_id", "stop_sequence"],
    "calendar": ["service_id"],
    "calendar_dates": ["service_id", "date"],
    "shapes": ["shape_id", "shape_pt_sequence"],
    "frequencies": ["trip_id", "start_time"],
    "pathways": ["pathway_id"],
    "levels": ["level_id"],
    "attributions": ["attribution_id"],
}


_REFERENCES = {
    "agency_id": ("agency", "agency_id"),
    "route_id": ("routes", "route_id"),
    "trip_id": ("trips", "trip_id"),
    "stop_id": ("stops", "stop_id"),
    "parent_station": ("stops", "stop_id"),
    "shape_id": ("shapes", "shape_id"),
    "level_id": ("levels", "level_id"),
    "from_stop_id": ("stops", "stop_id"),
    "to_stop_id": ("stops", "stop_id"),
    "from_route_id": ("routes", "route_id"),
    "to_route_id": ("routes", "route_id"),
    "from_trip_id": ("trips", "trip_id"),
    "to_trip_id": ("trips", "trip_id"),
}


def _present(value: object) -> bool:
    """
    Return whether a scalar is supplied rather than missing.

    Empty or whitespace-only strings and pandas missing scalars are absent.

    Parameters
    ----------
    value : object
        Input scalar or nested property to interpret.

    Returns
    -------
    bool
        Whether the scalar contains a supplied value.
    """
    return value is not None and str(value).strip() not in {"", "nan", "None", "<NA>"}


def _number(value: object, label: str, *, positive: bool = False) -> float:
    """
    Read a finite non-negative number, with an optional strict lower bound.

    Reject non-finite values before distance or time arithmetic.

    Parameters
    ----------
    value : object
        Input scalar or nested property to interpret.
    label : str
        Field context included in validation errors.
    positive : bool
        Require a strictly positive value when True.

    Returns
    -------
    float
        Validated numeric value.
    """
    try:
        result = float(str(value))
    except (TypeError, ValueError) as exc:
        msg = f"{label}: supply a number, got {value!r}"
        raise ValueError(msg) from exc
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        msg = f"{label}: must be finite and {'positive' if positive else 'non-negative'}"
        raise ValueError(msg)
    return result


def _seconds(value: object) -> int:
    """
    Parse a GTFS time without wrapping hours at midnight.

    Hours may exceed 24; minutes and seconds must be below 60.

    Parameters
    ----------
    value : object
        Input scalar or nested property to interpret.

    Returns
    -------
    int
        Integer seconds without midnight wrapping.
    """
    if not re.fullmatch(r"\d{2,}:\d{2}:\d{2}", str(value)):
        msg = f"Invalid GTFS time {value!r}; expected HH:MM:SS"
        raise ValueError(msg)
    hours, minutes, seconds = map(int, str(value).split(":"))
    if minutes >= 60 or seconds >= 60:
        msg = f"Invalid GTFS time {value!r}"
        raise ValueError(msg)
    return hours * 3600 + minutes * 60 + seconds


def _clock(seconds: float) -> str:
    """
    Format rounded elapsed seconds as a GTFS extended-hour time.

    Rounding is applied to cumulative times rather than each individual leg.

    Parameters
    ----------
    seconds : float
        Elapsed seconds since the service-day origin.

    Returns
    -------
    str
        Extended-hour HH:MM:SS representation.
    """
    value = round(seconds)
    return f"{value // 3600:02d}:{value % 3600 // 60:02d}:{value % 60:02d}"


def _text_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Copy a table into the string representation used by load_gtfs.

    Identifier strings retain their spelling, including leading zeroes.

    Parameters
    ----------
    frame : pd.DataFrame
        Table with string column names.

    Returns
    -------
    pd.DataFrame
        Independent string-valued table without geometry.
    """
    result = pd.DataFrame(frame.drop(columns=["geometry"], errors="ignore")).copy()
    for column in result:
        if str(column).endswith("_id") or column == "parent_station":
            result[column] = result[column].map(
                lambda value: str(value) if _present(value) else None
            )
            continue
        result[column] = result[column].map(
            lambda value: (
                None
                if not _present(value)
                else str(int(value))
                if isinstance(value, (bool, int, float)) and float(value).is_integer()
                else str(value)
            )
        )
    return result


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Read records from tables whose columns have string names.

    Pandas allows arbitrary column labels; these input contracts use string labels.

    Parameters
    ----------
    frame : pd.DataFrame
        Table with string column names.

    Returns
    -------
    list[dict[str, Any]]
        Rows indexed by their string field names.
    """
    return cast("list[dict[str, Any]]", frame.to_dict("records"))


_MODEL = str(_REGISTRY["model"])


@dataclass
class _GenerationConfig:
    """
    Optional global overrides, separate from the values resolved per pattern.

    None means that a source value or a model default may be adopted.
    """

    calendar: pd.DataFrame | None = None
    calendar_dates: pd.DataFrame | None = None
    speed_kph: float | None = None
    dwell_time_secs: float | None = None
    distance_method: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    headway_secs: float | None = None
    group_headway_secs: float | None = None
    pattern_ids: str | Sequence[str] | None = None
    service_date: str | None = None
    # Use the caller's local calendar date, captured once for the entire feed.
    generation_date: str = field(default_factory=lambda: date.today().strftime("%Y%m%d"))  # noqa: DTZ011


_MODES = {
    "tram": 0,
    "subway": 1,
    "train": 2,
    "bus": 3,
    "trolleybus": 11,
    "ferry": 4,
    "light_rail": 0,
    "monorail": 12,
    "funicular": 7,
}


_INPUT_KEYS = {
    "routes": ["route_id"],
    "stops": ["stop_id"],
    "patterns": ["pattern_id"],
    "stop_sequences": ["pattern_id", "stop_sequence"],
}


class GenerationOptions(TypedDict, total=False):
    """
    Keyword options forwarded unchanged to create_gtfs.

    Source adapters forward these calendar and timing options without modification.
    """

    calendar: pd.DataFrame | None
    calendar_dates: pd.DataFrame | None
    speed_kph: float | None
    dwell_time_secs: float | None
    distance_method: str | None
    start_time: str | None
    end_time: str | None
    headway_secs: float | None
    group_headway_secs: float | None
    service_date: str | None
    optional_tables: Mapping[str, pd.DataFrame]


def _time_to_seconds(value: str | float | None) -> float:
    """
    Convert a GTFS time value into seconds from midnight.

    GTFS may contain values beyond 24 hours (for trips after midnight), and this
    helper preserves those values as absolute seconds.

    String values must be in ``HH:MM:SS`` format. Invalid string
    representations like ``""``, ``"nan"``, ``"None"``, and bare numeric
    strings (e.g. ``"3600"``) are rejected with a ``ValueError``.

    Parameters
    ----------
    value : str | float | None
        GTFS time value, must be ``HH:MM:SS`` when a string is provided.

    Returns
    -------
    float
        Seconds from midnight.

    Raises
    ------
    ValueError
        If a string value is not in valid ``HH:MM:SS`` format.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0.0
    if not isinstance(value, str):
        return float(value)
    value = str(value).strip()
    if not value or value.lower() in {"nan", "none"}:
        msg = f"Invalid GTFS time value: {value!r}. Expected HH:MM:SS format."
        raise ValueError(msg)
    parts = value.split(":")
    if len(parts) != 3:
        msg = f"Invalid GTFS time value: {value!r}. Expected HH:MM:SS format."
        raise ValueError(msg)
    try:
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError as error:
        msg = f"Invalid GTFS time value: {value!r}. Expected HH:MM:SS format."
        raise ValueError(msg) from error
    return h * 3600 + m * 60 + s
