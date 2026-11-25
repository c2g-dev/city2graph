"""
GTFS transportation network utilities.

The functions in this module generate, inspect, save and load GTFS feeds,
retrieve transport service from OSM, and derive stop-to-stop
origin/destination records and transport summary graphs for analysis.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import warnings
import zipfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Literal
from typing import Unpack

import duckdb
import geopandas as gpd
import pandas as pd
from shapely import reverse
from shapely import wkt

from . import data as _data
from .gtfs.feed import _INCLUDE_METADATA
from .gtfs.feed import _ensure_graph_sql_support
from .gtfs.feed import save_gtfs
from .gtfs.generation import create_gtfs
from .gtfs.legs import _leg_diagnostics
from .gtfs.legs import _stop_time_legs
from .gtfs.metadata import _report_metadata
from .gtfs.metadata import _restore_metadata
from .gtfs.report import _generation_report
from .gtfs.schema import _METADATA_FILENAME
from .gtfs.schema import GenerationOptions
from .gtfs.schema import GTFSGenerationWarning
from .gtfs.schema import TransportData
from .gtfs.schema import _time_to_seconds as _time_to_seconds  # noqa: PLC0414 - retain analysis helper export
from .gtfs.sources import _OSM_ENDPOINT
from .gtfs.sources import _OSM_TIMEOUT
from .gtfs.sources import _OVERWRITE_SOURCE
from .gtfs.sources import _USE_SOURCE_TAGS
from .gtfs.sources import _load_osm_transport
from .gtfs.sources import _SourceIO
from .utils import gdf_to_nx

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    import networkx as nx
    from shapely.geometry import MultiPolygon
    from shapely.geometry import Polygon

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

__all__ = [
    "GTFSGenerationWarning",
    "TransportData",
    "create_gtfs",
    "get_gtfs_summary",
    "get_od_pairs",
    "load_gbfs",
    "load_gtfs",
    "load_osm_transport",
    "save_gtfs",
    "travel_summary_graph",
]

_ACTIVE_DATES_TABLE_SQL = """
    CREATE OR REPLACE TEMP TABLE _active_dates (
        service_id VARCHAR,
        active_date DATE
    )
"""

_CALENDAR_DATES_PARSED_SQL = """
    CREATE OR REPLACE TEMP TABLE _cal_dates_parsed AS
    SELECT service_id, exception_type, cast(strptime(date, '%Y%m%d') as date) as parsed_date
    FROM calendar_dates
"""

_BASE_ACTIVE_DATES_INSERT_SQL = """
    INSERT INTO _active_dates
    WITH RECURSIVE dates AS (
        SELECT cast(strptime('{start_date}', '%Y%m%d') as date) as d
        UNION ALL
        SELECT d + interval 1 day
        FROM dates
        WHERE d < cast(strptime('{end_date}', '%Y%m%d') as date)
    ),
    base_services AS (
        SELECT
            c.service_id,
            d.d as active_date
        FROM calendar c
        CROSS JOIN dates d
        WHERE d.d BETWEEN cast(strptime(c.start_date, '%Y%m%d') as date)
                    AND cast(strptime(c.end_date, '%Y%m%d') as date)
          AND 1 = CASE
              WHEN isodow(d.d) = 1 THEN try_cast(c.monday as integer)
              WHEN isodow(d.d) = 2 THEN try_cast(c.tuesday as integer)
              WHEN isodow(d.d) = 3 THEN try_cast(c.wednesday as integer)
              WHEN isodow(d.d) = 4 THEN try_cast(c.thursday as integer)
              WHEN isodow(d.d) = 5 THEN try_cast(c.friday as integer)
              WHEN isodow(d.d) = 6 THEN try_cast(c.saturday as integer)
              WHEN isodow(d.d) = 7 THEN try_cast(c.sunday as integer)
              ELSE 0
          END
    )
    SELECT service_id, active_date FROM base_services
"""

_STOP_GEOMETRY_SQL = """
    CASE
        WHEN try_cast(stop_lon AS DOUBLE) BETWEEN -180 AND 180
         AND try_cast(stop_lat AS DOUBLE) BETWEEN -90 AND 90
        THEN ST_Point(CAST(stop_lon AS DOUBLE), CAST(stop_lat AS DOUBLE))
    END
"""


def _list_tables(con: duckdb.DuckDBPyConnection) -> set[str]:
    """
    Return table names present in the current DuckDB connection.

    This helper centralizes table discovery so call sites stay concise.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Open DuckDB connection.

    Returns
    -------
    set[str]
        Set of relation names returned by ``SHOW TABLES``.
    """
    return {row[0] for row in con.execute("SHOW TABLES").fetchall()}


def _convert_wkt_column(
    df: pd.DataFrame,
    *,
    wkt_column: str = "geometry_wkt",
    geometry_column: str = "geometry",
) -> pd.DataFrame:
    """
    Convert a WKT column to shapely geometry and drop the original WKT column.

    This keeps repeated WKT parsing logic in one place.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing WKT text geometry.
    wkt_column : str, default="geometry_wkt"
        Column containing geometry encoded as WKT.
    geometry_column : str, default="geometry"
        Destination column for parsed shapely geometries.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with parsed geometry and without ``wkt_column``.
    """
    df[geometry_column] = df[wkt_column].apply(
        lambda value: wkt.loads(value) if pd.notna(value) else None
    )
    return df.drop(columns=[wkt_column])


def _build_active_dates(
    con: duckdb.DuckDBPyConnection,
    *,
    start_date: str,
    end_date: str,
    include_calendar: bool,
    include_calendar_dates: bool,
) -> None:
    """
    Create and populate temporary active service dates for a date window.

    Active dates combine weekly calendar service and date-level exceptions.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Open DuckDB connection.
    start_date : str
        Inclusive start date in ``YYYYMMDD`` format.
    end_date : str
        Inclusive end date in ``YYYYMMDD`` format.
    include_calendar : bool
        Whether to include ``calendar`` weekly service expansion.
    include_calendar_dates : bool
        Whether to apply ``calendar_dates`` exceptions.

    Returns
    -------
    None
        The function mutates temporary tables in ``con``.
    """
    con.execute(_ACTIVE_DATES_TABLE_SQL)

    if include_calendar:
        con.execute(_BASE_ACTIVE_DATES_INSERT_SQL.format(start_date=start_date, end_date=end_date))

    if include_calendar_dates:
        con.execute(_CALENDAR_DATES_PARSED_SQL)
        con.execute(
            """
            INSERT INTO _active_dates
            SELECT service_id, parsed_date
            FROM _cal_dates_parsed
            WHERE try_cast(exception_type as integer) = 1
              AND parsed_date BETWEEN cast(strptime(?, '%Y%m%d') as date)
                                  AND cast(strptime(?, '%Y%m%d') as date)
            """,
            [start_date, end_date],
        )
        con.execute(
            """
            DELETE FROM _active_dates
            WHERE (service_id, active_date) IN (
                SELECT service_id, parsed_date
                FROM _cal_dates_parsed
                WHERE try_cast(exception_type as integer) = 2
            )
            """
        )


def _table_columns(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
) -> set[str]:
    """
    Return column names for a DuckDB table.

    This helper centralizes schema inspection used by transport graph builders.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Open DuckDB connection.
    table_name : str
        Name of the relation to inspect.

    Returns
    -------
    set[str]
        Column names available on ``table_name``.
    """
    return {row[1] for row in con.execute(f"PRAGMA table_info('{table_name}')").fetchall()}


def _drop_temp_relation(con: duckdb.DuckDBPyConnection, name: str) -> None:
    """
    Drop an existing temporary DuckDB view or table matching its actual catalog type.

    DuckDB raises a ``CatalogException`` if ``DROP VIEW`` targets an existing
    table (or vice versa), so this helper inspects the temporary catalog first.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Open DuckDB connection.
    name : str
        Temporary view or table name to remove if present.

    Returns
    -------
    None
        Mutates temporary catalog entries in ``con``.
    """
    if con.execute(
        "SELECT 1 FROM duckdb_views() WHERE temporary AND view_name = ? LIMIT 1",
        [name],
    ).fetchone():
        con.execute(f"DROP VIEW IF EXISTS {name}")
    if con.execute(
        "SELECT 1 FROM duckdb_tables() WHERE temporary AND table_name = ? LIMIT 1",
        [name],
    ).fetchone():
        con.execute(f"DROP TABLE IF EXISTS {name}")


def _ensure_stops_geometry(con: duckdb.DuckDBPyConnection) -> None:
    """
    Create a temp relation named _stops_for_graph with a geometry column.

    If stops.geometry already exists, it is reused. Otherwise geometry is built
    from valid WGS84 coordinates in stop_lon/stop_lat.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Open DuckDB connection containing a ``stops`` table.

    Returns
    -------
    None
        The function creates a temporary relation named ``_stops_for_graph``.

    Raises
    ------
    ValueError
        If stop geometry is unavailable and ``stop_lon``/``stop_lat`` are
        missing.
    """
    stop_columns = _table_columns(con, "stops")

    _drop_temp_relation(con, "_stops_for_graph")

    if "geometry" in stop_columns:
        con.execute("CREATE TEMP VIEW _stops_for_graph AS SELECT * FROM stops")
        return

    if not {"stop_lon", "stop_lat"}.issubset(stop_columns):
        msg = "stops must contain either a geometry column or both stop_lon and stop_lat."
        raise ValueError(msg)

    con.execute(
        f"""
        CREATE TEMP TABLE _stops_for_graph AS
        SELECT *, {_STOP_GEOMETRY_SQL} AS geometry
        FROM stops
        """  # noqa: S608
    )


def _materialize_loaded_stop_geometry(con: duckdb.DuckDBPyConnection) -> None:
    """
    Add nullable WGS84 point geometry to the imported stops table.

    Keep the original coordinate fields and leave unusable coordinates without
    a point.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Open DuckDB connection containing the imported stops table.

    Raises
    ------
    ValueError
        If either coordinate column is missing.
    """
    if not {"stop_lon", "stop_lat"}.issubset(_table_columns(con, "stops")):
        msg = "GTFS stops must contain both stop_lon and stop_lat columns."
        raise ValueError(msg)
    con.execute("ALTER TABLE stops ADD COLUMN geometry GEOMETRY")
    con.execute(f"UPDATE stops SET geometry = {_STOP_GEOMETRY_SQL}")  # noqa: S608


def _feed_date_bounds(
    con: duckdb.DuckDBPyConnection, tables: set[str]
) -> tuple[datetime | None, datetime | None]:
    """Read date bounds across weekly calendars and date exceptions.

    Empty calendar tables contribute no bounds.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Feed connection.
    tables : set of str
        Available table names.

    Returns
    -------
    tuple of datetime or None
        Earliest and latest declared dates, or two missing values.
    """
    bounds: list[tuple[datetime, datetime]] = []
    for name, start, end in (
        ("calendar", "start_date", "end_date"),
        ("calendar_dates", "date", "date"),
    ):
        if name in tables:
            row = con.execute(
                f"SELECT MIN({start}), MAX({end}) FROM {name}"  # noqa: S608 - fixed table and column names
            ).fetchone()
            if row and row[0] and row[1]:
                bounds.append(
                    (
                        datetime.strptime(str(row[0]), "%Y%m%d"),
                        datetime.strptime(str(row[1]), "%Y%m%d"),
                    )
                )
    if not bounds:
        return None, None
    return min(start for start, _ in bounds), max(end for _, end in bounds)


def _resolve_service_window(
    con: duckdb.DuckDBPyConnection,
    *,
    calendar_start: str | None,
    calendar_end: str | None,
    tables: set[str],
) -> tuple[str | None, str | None, bool]:
    """
    Resolve the calendar window used for service-frequency counting.

    The resolved window combines feed-level date bounds with optional
    user-provided limits and supports feeds with ``calendar_dates`` only.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Open DuckDB connection.
    calendar_start : str | None
        Optional lower date bound in ``YYYYMMDD`` format.
    calendar_end : str | None
        Optional upper date bound in ``YYYYMMDD`` format.
    tables : set[str]
        Available relation names in the current feed.

    Returns
    -------
    tuple[str | None, str | None, bool]
        (resolved_start, resolved_end, use_date_aware_counts)

        If no usable calendar information exists, returns (None, None, False).

    Raises
    ------
    ValueError
        If requested bounds are malformed, out of range, or inconsistent.
    """
    has_calendar = "calendar" in tables
    has_calendar_dates = "calendar_dates" in tables

    if not (has_calendar or has_calendar_dates):
        if calendar_start or calendar_end:
            msg = (
                "calendar_start/calendar_end specified but GTFS feed has neither "
                "calendar.txt nor calendar_dates.txt"
            )
            raise ValueError(msg)
        return None, None, False

    gtfs_start, gtfs_end = _feed_date_bounds(con, tables)
    if gtfs_start is None or gtfs_end is None:
        if calendar_start or calendar_end:
            msg = (
                "calendar_start/calendar_end specified but calendar tables contain no usable dates"
            )
            raise ValueError(msg)
        return None, None, False

    try:
        user_start = datetime.strptime(calendar_start, "%Y%m%d") if calendar_start else None
        user_end = datetime.strptime(calendar_end, "%Y%m%d") if calendar_end else None
    except ValueError as error:
        msg = "Invalid calendar date format. Expected YYYYMMDD."
        raise ValueError(msg) from error

    if user_start and (user_start < gtfs_start or user_start > gtfs_end):
        msg = f"calendar_start ({calendar_start}) is outside the valid GTFS date range"
        raise ValueError(msg)

    if user_end and (user_end < gtfs_start or user_end > gtfs_end):
        msg = f"calendar_end ({calendar_end}) is outside the valid GTFS date range"
        raise ValueError(msg)

    resolved_start = user_start or gtfs_start
    resolved_end = user_end or gtfs_end

    if resolved_start > resolved_end:
        msg = (
            f"calendar_start ({resolved_start.strftime('%Y%m%d')}) "
            f"must be <= calendar_end ({resolved_end.strftime('%Y%m%d')})"
        )
        raise ValueError(msg)

    return (
        resolved_start.strftime("%Y%m%d"),
        resolved_end.strftime("%Y%m%d"),
        True,
    )


def load_gtfs(
    path: str | Path, *, include_metadata: bool = _INCLUDE_METADATA
) -> duckdb.DuckDBPyConnection:
    """
    Load a GTFS ZIP archive into an in-memory DuckDB database.

    The function reads ``*.txt`` GTFS files as tables, registers a time parsing
    UDF, and materializes ``stops.geometry`` as points from
    ``stop_lon``/``stop_lat`` when valid WGS84 stop coordinates are available.
    Stops with missing or invalid coordinate values are kept with null geometry.

    Parameters
    ----------
    path : str | Path
        Path to a GTFS ``.zip`` file.
    include_metadata : bool, default True
        Restore embedded ``city2graph_metadata.json`` when present.
        Set False to load only the GTFS tables.

    Returns
    -------
    duckdb.DuckDBPyConnection
        In-memory DuckDB connection containing imported GTFS tables.

    Raises
    ------
    ValueError
        If a stops table is present without both coordinate columns.
    """
    con = duckdb.connect(":memory:")
    try:
        _ensure_graph_sql_support(con)

        has_files = False
        embedded_metadata = None

        try:
            with tempfile.TemporaryDirectory() as tmp_dir, zipfile.ZipFile(path) as zf:
                if include_metadata and _METADATA_FILENAME in zf.namelist():
                    embedded_metadata = zf.read(_METADATA_FILENAME)
                for name in zf.namelist():
                    if name.endswith(".txt") and not name.endswith("/"):
                        has_files = True
                        table_name = name.rsplit("/", 1)[-1].replace(".txt", "")
                        file_path = str(Path(tmp_dir) / "table.csv")
                        with zf.open(name) as source, Path(file_path).open("wb") as target:
                            shutil.copyfileobj(source, target)
                        con.read_csv(file_path, all_varchar=True, auto_detect=True).create(
                            table_name
                        )
        except (OSError, zipfile.BadZipFile, duckdb.Error):
            logger.exception("Failed to read GTFS archive at %s", path)
            return con

        if not has_files:
            logger.warning("No GTFS files found in %s", path)
            return con

        tables = sorted(_list_tables(con))

        if "stops" in tables:
            _materialize_loaded_stop_geometry(con)

        if embedded_metadata is not None:
            _restore_metadata(con, embedded_metadata)

        logger.info("GTFS loaded: %s", ", ".join(tables))
        return con
    except Exception:
        con.close()
        raise


def load_gbfs(path: str | Path) -> duckdb.DuckDBPyConnection:
    """
    Load GBFS JSON feeds from a directory into an in-memory DuckDB database.

    The function reads GBFS JSON files from a local directory, flattens common
    feed structures into DuckDB tables, and materializes geometry columns from
    ``lon``/``lat`` fields when available. Unreadable files are logged and
    skipped so other feeds in the directory remain usable.

    Parameters
    ----------
    path : str | Path
        Path to a GBFS directory containing JSON files.

    Returns
    -------
    duckdb.DuckDBPyConnection
        In-memory DuckDB connection containing imported GBFS tables.
    """
    con = duckdb.connect(":memory:")
    con.execute("INSTALL spatial; LOAD spatial;")
    path = Path(path)
    has_files = False
    for file_path in path.rglob("*.json"):
        try:
            with file_path.open(encoding="utf-8") as file:
                raw = json.load(file)
        except (OSError, UnicodeError, json.JSONDecodeError):
            logger.exception("Failed to read GBFS file at %s", file_path)
            continue
        data = raw.get("data") if isinstance(raw, dict) else None
        if not isinstance(data, dict):
            logger.warning("GBFS file has no data object: %s", file_path)
            continue
        rows = next(
            (
                data[key]
                for key in ("stations", "bikes", "vehicles", "vehicle_types", "feeds")
                if key in data
            ),
            [data],
        )
        if not rows:
            continue
        table_name = file_path.stem.replace("-", "_").lower()
        try:
            con.register("_gbfs_tmp", pd.DataFrame(rows))
            try:
                con.execute(f"""
                    CREATE OR REPLACE TABLE {table_name} AS
                    SELECT *
                    FROM _gbfs_tmp
                """)  # noqa: S608
            finally:
                con.unregister("_gbfs_tmp")
        except (ValueError, duckdb.Error):
            logger.exception("Failed to load GBFS table from %s", file_path)
            continue
        has_files = True
    if not has_files:
        logger.warning("No GBFS JSON files found in %s", path)
        return con

    tables = sorted(_list_tables(con))
    for table in tables:
        columns = {row[1] for row in con.execute(f"PRAGMA table_info({table})").fetchall()}
        if {"lat", "lon"}.issubset(columns):
            con.execute(f"""
                ALTER TABLE {table} ADD COLUMN geometry GEOMETRY;
                UPDATE {table}
                SET geometry = ST_Point(
                    CAST(lon AS DOUBLE),
                    CAST(lat AS DOUBLE)
                )
                WHERE try_cast(lon AS DOUBLE) IS NOT NULL
                  AND try_cast(lat AS DOUBLE) IS NOT NULL
            """)  # noqa: S608
    logger.info("GBFS loaded: %s", ", ".join(tables))
    return con


def _build_frequency_periods(
    con: duckdb.DuckDBPyConnection,
    *,
    tables: set[str],
    use_frequencies: bool,
) -> None:
    """
    Prepare numeric frequency periods without expanding individual departures.

    Retain period boundaries so each graph leg can apply its own time offsets.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Open DuckDB connection containing GTFS tables.
    tables : set[str]
        GTFS tables represented as string-valued DataFrames.
    use_frequencies : bool
        Interpret frequency periods when True; otherwise retain template counts.
    """
    con.execute("""
        CREATE OR REPLACE TEMP TABLE _frequency_periods (
            trip_id VARCHAR, start_sec DOUBLE, end_sec DOUBLE,
            headway DOUBLE, exact_times DOUBLE
        )
    """)
    if not use_frequencies or "frequencies" not in tables:
        return
    columns = _table_columns(con, "frequencies")
    exact = (
        "CASE WHEN exact_times IS NULL OR trim(CAST(exact_times AS VARCHAR)) = '' "
        "THEN 0 ELSE try_cast(exact_times AS DOUBLE) END"
        if "exact_times" in columns
        else "0"
    )
    con.execute(f"""
        INSERT INTO _frequency_periods
        SELECT trip_id, time_to_seconds(CAST(start_time AS VARCHAR)),
            time_to_seconds(CAST(end_time AS VARCHAR)),
            try_cast(headway_secs AS DOUBLE), {exact}
        FROM frequencies
    """)  # noqa: S608
    invalid = con.execute("""
        SELECT trip_id FROM _frequency_periods
        WHERE start_sec IS NULL OR end_sec IS NULL
            OR headway IS NULL OR NOT isfinite(headway) OR headway <= 0
            OR headway != floor(headway) OR end_sec <= start_sec
            OR exact_times IS NULL OR exact_times NOT IN (0, 1)
        UNION ALL
        SELECT trip_id FROM (
            SELECT *, lag(end_sec) OVER (PARTITION BY trip_id ORDER BY start_sec) AS previous_end
            FROM _frequency_periods
        ) WHERE start_sec < previous_end
    """).fetchall()
    if invalid:
        msg = f"Invalid or overlapping frequency periods for trips: {invalid}"
        raise ValueError(msg)


def get_od_pairs(
    con: duckdb.DuckDBPyConnection,
    start_date: str | None = None,
    end_date: str | None = None,
    include_geometry: bool = True,
    directed: bool = False,  # noqa: ARG001 - retained for call compatibility
    *,
    interpolate_stop_times: bool = False,
) -> pd.DataFrame | gpd.GeoDataFrame:
    """
    Generate stop-to-stop OD pairs from GTFS trip stop sequences.

    For each ``trip_id``, consecutive stops are paired into directed legs with
    departure/arrival timestamps and per-leg travel time in seconds. Service
    activity is derived from ``calendar`` and optionally adjusted using
    ``calendar_dates``. Frequency-based trips remain template legs, not actual
    departures; their timestamps must not be used to infer connections.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        DuckDB connection with GTFS tables loaded.
    start_date : str | None, default=None
        Inclusive start date (``YYYYMMDD``). Defaults to the earliest date in
        calendar or calendar_dates.
    end_date : str | None, default=None
        Inclusive end date (``YYYYMMDD``). Defaults to the latest date in
        calendar or calendar_dates.
    include_geometry : bool, default=True
        If ``True``, include a line geometry for each OD pair when stops
        have prebuilt geometries or longitude and latitude columns.
    directed : bool, default=False
        Retained for call compatibility. Both values preserve the actual trip
        origin, destination, timestamps and line direction. Historically,
        ``False`` canonicalized the stop IDs, which could mislabel departures
        from the opposite stop. Use :func:`travel_summary_graph` with
        ``directed=False`` to aggregate trips into undirected edges instead.
    interpolate_stop_times : bool, default=False
        Estimate missing intermediate times by distance where available and
        by stop count otherwise. A single missing clock may copy its counterpart.
        Endpoints, exact timepoints and pickup/drop-off windows remain protected.

    Returns
    -------
    pd.DataFrame | gpd.GeoDataFrame
        One row per actual stop-to-stop leg. ``orig_stop_id`` identifies the
        departure stop and ``dest_stop_id`` identifies the arrival stop.
        With interpolation, ``departure_interpolated`` and ``arrival_interpolated``
        identify estimated clocks. The frame's ``attrs["stop_time_diagnostics"]``
        counts candidate and time-excluded legs once per active trip template.
    """
    empty = pd.DataFrame()
    empty.attrs["stop_time_diagnostics"] = {
        "candidate_legs": 0,
        "excluded_legs": 0,
        "excluded_by_reason": {},
    }
    if interpolate_stop_times:
        empty["departure_interpolated"] = pd.Series(dtype=bool)
        empty["arrival_interpolated"] = pd.Series(dtype=bool)
    tables = _list_tables(con)
    required = {"stop_times", "trips"}
    if not required.issubset(tables):
        logger.error("GTFS feed incomplete - need %s", ", ".join(required))
        return empty

    has_cal = "calendar" in tables

    s_dt, e_dt = start_date, end_date
    if not s_dt or not e_dt:
        first, last = _feed_date_bounds(con, tables)
        s_dt = s_dt or (first.strftime("%Y%m%d") if first else None)
        e_dt = e_dt or (last.strftime("%Y%m%d") if last else None)
    if not s_dt or not e_dt:
        return empty

    _build_active_dates(
        con,
        start_date=str(s_dt),
        end_date=str(e_dt),
        include_calendar=has_cal,
        include_calendar_dates="calendar_dates" in tables,
    )

    query = """
        SELECT v.trip_id, v.service_id, v.stop_id AS orig_stop_id, v.next_stop_id AS dest_stop_id,
            strftime(d.active_date, '%Y-%m-%d') AS date,
            d.active_date + to_seconds(v.departure_time_sec) AS departure_ts,
            d.active_date + to_seconds(v.next_arrival_time_sec) AS arrival_ts,
            v.next_arrival_time_sec - v.departure_time_sec AS travel_time_sec
    """
    if interpolate_stop_times:
        query += ", v.departure_interpolated, v.arrival_interpolated"

    stops_relation = "stops"
    include_geometry = include_geometry and "stops" in tables
    if include_geometry:
        stop_columns = _table_columns(con, "stops")
        if "geometry" not in stop_columns:
            include_geometry = {"stop_lon", "stop_lat"}.issubset(stop_columns)
            if include_geometry:
                _ensure_graph_sql_support(con)
                _ensure_stops_geometry(con)
                stops_relation = "_stops_for_graph"

    if include_geometry:
        query += """,
            ST_AsText(ST_MakeLine(s1.geometry, s2.geometry)) as geometry_wkt
        """

    query += """
        FROM _stop_time_legs v
        JOIN (SELECT DISTINCT service_id, active_date FROM _active_dates) d USING (service_id)
    """

    if include_geometry:
        query += f"""
        LEFT JOIN {stops_relation} s1 ON v.stop_id = s1.stop_id
        LEFT JOIN {stops_relation} s2 ON v.next_stop_id = s2.stop_id
        """

    query += " WHERE v.exclusion IS NULL ORDER BY v.trip_id, d.active_date, v.seq"
    with _stop_time_legs(con, interpolate=interpolate_stop_times, summary=False):
        diagnostics = _leg_diagnostics(con)
        od_pairs_df = con.execute(query).df()
    if not interpolate_stop_times and not diagnostics["candidate_legs"]:
        return empty
    if include_geometry:
        od_pairs_df = _convert_wkt_column(od_pairs_df)
        od_pairs_df = gpd.GeoDataFrame(od_pairs_df, geometry="geometry", crs="EPSG:4326")
    od_pairs_df.attrs["stop_time_diagnostics"] = diagnostics
    return od_pairs_df


def _resolve_time_bounds(
    start_time: str | None,
    end_time: str | None,
) -> tuple[float | None, float | None]:
    """
    Parse and validate optional summary-graph time filters.

    The returned bounds are expressed in seconds so the SQL aggregation query
    can reuse them directly.

    Parameters
    ----------
    start_time : str | None
        Optional lower bound (inclusive) in ``HH:MM:SS``.
    end_time : str | None
        Optional upper bound (inclusive) in ``HH:MM:SS``.

    Returns
    -------
    tuple[float | None, float | None]
        Parsed lower departure bound and upper arrival bound in seconds.
    """
    try:
        min_departure_sec = _time_to_seconds(start_time) if start_time is not None else None
        max_arrival_sec = _time_to_seconds(end_time) if end_time is not None else None
    except (TypeError, ValueError) as error:
        msg = "Invalid time format. Expected HH:MM:SS."
        raise ValueError(msg) from error

    if (
        min_departure_sec is not None
        and max_arrival_sec is not None
        and min_departure_sec > max_arrival_sec
    ):
        msg = (
            "start_time must be <= end_time. Use GTFS extended times "
            "(e.g. 25:30:00) for after-midnight windows."
        )
        raise ValueError(msg)

    return min_departure_sec, max_arrival_sec


def _prepare_service_counts(
    con: duckdb.DuckDBPyConnection,
    *,
    tables: set[str],
    calendar_start: str | None,
    calendar_end: str | None,
) -> None:
    """
    Create the temporary ``_service_counts`` table for edge aggregation.

    The table stores the effective number of active service days per
    ``service_id`` across the resolved calendar window.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Open DuckDB connection with GTFS tables loaded.
    tables : set[str]
        Available table names in the feed.
    calendar_start : str | None
        Optional calendar window start in ``YYYYMMDD``.
    calendar_end : str | None
        Optional calendar window end in ``YYYYMMDD``.

    Returns
    -------
    None
        The function mutates temporary tables in ``con``.
    """
    resolved_start, resolved_end, use_date_aware_counts = _resolve_service_window(
        con,
        calendar_start=calendar_start,
        calendar_end=calendar_end,
        tables=tables,
    )

    if use_date_aware_counts and resolved_start and resolved_end:
        _build_active_dates(
            con,
            start_date=resolved_start,
            end_date=resolved_end,
            include_calendar="calendar" in tables,
            include_calendar_dates="calendar_dates" in tables,
        )
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE _service_counts AS
            SELECT service_id, COUNT(DISTINCT active_date) AS sc
            FROM _active_dates
            GROUP BY service_id
            """
        )
        return

    logger.info(
        "No usable calendar/calendar_dates window found; falling back to sc=1 per service_id."
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE _service_counts AS
        SELECT DISTINCT service_id, 1 AS sc
        FROM trips
        WHERE service_id IS NOT NULL
        """
    )


def _build_travel_summary_edges(
    con: duckdb.DuckDBPyConnection,
    *,
    min_departure_sec: float | None,
    max_arrival_sec: float | None,
    directed: bool,
    interpolate: bool,
) -> gpd.GeoDataFrame:
    """
    Build aggregated stop-to-stop edges for the travel summary graph.

    The result includes weighted average travel times, service frequencies,
    and stop-to-stop line geometries.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Open DuckDB connection with prepared temp tables.
    min_departure_sec : float | None
        Optional lower departure bound in seconds.
    max_arrival_sec : float | None
        Optional upper arrival bound in seconds.
    directed : bool
        Whether to preserve edge direction.
    interpolate : bool
        Include the service count contributed by estimated times.

    Returns
    -------
    gpd.GeoDataFrame
        Aggregated edge records indexed by stop pairs.
    """
    estimated_count = (
        ", sum(service_count) FILTER (WHERE interpolated) AS interpolated_frequency"
        if interpolate
        else ""
    )
    estimated_column = (
        ", coalesce(a.interpolated_frequency, 0) AS interpolated_frequency" if interpolate else ""
    )
    edge_query = f"""
        WITH bounds AS (SELECT ?::DOUBLE AS lower_time, ?::DOUBLE AS upper_time),
        weighted AS (
            SELECT st.stop_id, st.next_stop_id, st.departure_interpolated OR st.arrival_interpolated AS interpolated,
                st.next_arrival_time_sec - st.departure_time_sec AS travel_time,
                sc.sc * CASE
                    WHEN f.trip_id IS NULL THEN
                        CASE WHEN (b.lower_time IS NULL OR st.departure_time_sec >= b.lower_time)
                            AND (b.upper_time IS NULL OR st.next_arrival_time_sec <= b.upper_time)
                        THEN 1.0 ELSE 0.0 END
                    WHEN f.exact_times = 0 THEN
                        greatest(0.0,
                            least(f.end_sec, coalesce(b.upper_time - (st.next_arrival_time_sec - st.origin), f.end_sec))
                            - greatest(f.start_sec, coalesce(b.lower_time - (st.departure_time_sec - st.origin), f.start_sec))
                        ) / f.headway
                    ELSE greatest(0.0,
                        least(ceil((f.end_sec - f.start_sec) / f.headway) - 1,
                            coalesce(floor((b.upper_time - (st.next_arrival_time_sec - st.origin) - f.start_sec) / f.headway),
                                ceil((f.end_sec - f.start_sec) / f.headway) - 1))
                        - greatest(0.0, coalesce(ceil((b.lower_time - (st.departure_time_sec - st.origin) - f.start_sec) / f.headway), 0.0))
                        + 1.0)
                END AS service_count
            FROM _stop_time_legs st
            JOIN _service_counts sc USING (service_id)
            LEFT JOIN _frequency_periods f ON st.trip_id = f.trip_id
            CROSS JOIN bounds b
            WHERE st.exclusion IS NULL
        ),
        valid_pairs AS (SELECT * FROM weighted WHERE service_count > 0),
        agg_pairs AS (
            SELECT
                stop_id,
                next_stop_id,
                SUM(travel_time * service_count) / SUM(service_count) AS travel_time_sec,
                SUM(service_count) AS frequency{estimated_count}
            FROM valid_pairs
            GROUP BY stop_id, next_stop_id
        )
        SELECT
            a.stop_id AS from_stop_id,
            a.next_stop_id AS to_stop_id,
            a.travel_time_sec,
            a.frequency{estimated_column},
            ST_AsText(
                CASE
                    WHEN s1.geometry IS NOT NULL AND s2.geometry IS NOT NULL
                    THEN ST_MakeLine(s1.geometry, s2.geometry)
                    ELSE NULL
                END
            ) AS geometry_wkt
        FROM agg_pairs a
        LEFT JOIN _stops_for_graph s1
          ON a.stop_id = s1.stop_id
        LEFT JOIN _stops_for_graph s2
          ON a.next_stop_id = s2.stop_id
        ORDER BY from_stop_id, to_stop_id
    """  # noqa: S608

    edges_df = con.execute(
        edge_query,
        [min_departure_sec, max_arrival_sec],
    ).df()
    edges_df = _convert_wkt_column(edges_df)

    if not directed and not edges_df.empty:
        swap_mask = edges_df["from_stop_id"] > edges_df["to_stop_id"]
        edges_df.loc[swap_mask, ["from_stop_id", "to_stop_id"]] = edges_df.loc[
            swap_mask, ["to_stop_id", "from_stop_id"]
        ].to_numpy()
        reversed_lines = reverse(edges_df.loc[swap_mask, "geometry"])
        edges_df.loc[swap_mask, "geometry"] = reversed_lines
        edges_df["_weighted_tt"] = edges_df["travel_time_sec"] * edges_df["frequency"]
        agg = edges_df.groupby(["from_stop_id", "to_stop_id"], sort=False).agg(
            _weighted_tt=("_weighted_tt", "sum"),
            frequency=("frequency", "sum"),
            **(
                {"interpolated_frequency": ("interpolated_frequency", "sum")} if interpolate else {}
            ),
            geometry=("geometry", "first"),
        )
        agg["travel_time_sec"] = agg["_weighted_tt"] / agg["frequency"]
        edges_df = agg.drop(columns=["_weighted_tt"]).reset_index()

    edges_gdf = gpd.GeoDataFrame(edges_df, geometry="geometry", crs="EPSG:4326")
    if {"from_stop_id", "to_stop_id"}.issubset(edges_gdf.columns):
        edges_gdf = edges_gdf.set_index(["from_stop_id", "to_stop_id"])
    return edges_gdf


def _build_travel_summary_nodes(con: duckdb.DuckDBPyConnection) -> gpd.GeoDataFrame:
    """
    Build summary-graph nodes from the prepared stop geometry relation.

    Stop attributes are preserved from ``_stops_for_graph`` and indexed by
    ``stop_id`` for downstream graph construction.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Open DuckDB connection with ``_stops_for_graph`` prepared.

    Returns
    -------
    gpd.GeoDataFrame
        Stop records indexed by ``stop_id``.
    """
    stops_df = con.execute(
        """
        SELECT
            * EXCLUDE (geometry),
            ST_AsText(geometry) AS geometry_wkt
        FROM _stops_for_graph
        ORDER BY stop_id
        """
    ).df()
    stops_df = _convert_wkt_column(stops_df)
    nodes_gdf = gpd.GeoDataFrame(stops_df, geometry="geometry", crs="EPSG:4326")
    if "stop_id" in nodes_gdf.columns:
        nodes_gdf = nodes_gdf.set_index("stop_id")
    return nodes_gdf


def _travel_summary_edges(
    con: duckdb.DuckDBPyConnection,
    *,
    start_time: str | None = None,
    end_time: str | None = None,
    calendar_start: str | None = None,
    calendar_end: str | None = None,
    directed: bool = False,
    use_frequencies: bool = True,
    interpolate_stop_times: bool = False,
    report_exclusions: bool = False,
) -> gpd.GeoDataFrame:
    """
    Prepare one calendar aggregation and build summary edges.

    Both public graph analysis and generation reports use this aggregation path.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Open DuckDB connection containing GTFS tables.
    start_time : str | None
        Optional inclusive lower departure bound in extended-hour GTFS time.
    end_time : str | None
        Optional inclusive upper arrival bound in extended-hour GTFS time.
    calendar_start : str | None
        Optional calendar window start in YYYYMMDD form.
    calendar_end : str | None
        Optional calendar window end in YYYYMMDD form.
    directed : bool
        Whether to preserve edge direction.
    use_frequencies : bool
        Whether frequency periods contribute expanded trip counts.
    interpolate_stop_times : bool, default=False
        Estimate missing intermediate times by distance where available and
        by stop count otherwise. A single missing clock may copy its counterpart.
        Endpoints, exact timepoints and pickup/drop-off windows remain protected.
    report_exclusions : bool
        Attach exclusion counts and warn about missing clocks.

    Returns
    -------
    geopandas.GeoDataFrame
        Aggregated stop-pair edges over the requested time and calendar window.
    """
    tables = _list_tables(con)
    required = {"stop_times", "stops", "trips"}
    if not required.issubset(tables):
        missing = sorted(required - tables)
        msg = f"GTFS must contain stop_times, stops, and trips. Missing: {', '.join(missing)}"
        raise ValueError(msg)

    _ensure_graph_sql_support(con)
    _ensure_stops_geometry(con)
    min_departure_sec, max_arrival_sec = _resolve_time_bounds(start_time, end_time)
    _prepare_service_counts(
        con,
        tables=tables,
        calendar_start=calendar_start,
        calendar_end=calendar_end,
    )
    _build_frequency_periods(con, tables=tables, use_frequencies=use_frequencies)
    with _stop_time_legs(con, interpolate=interpolate_stop_times, summary=True):
        edges = _build_travel_summary_edges(
            con,
            min_departure_sec=min_departure_sec,
            max_arrival_sec=max_arrival_sec,
            directed=directed,
            interpolate=interpolate_stop_times,
        )
        if report_exclusions:
            edges.attrs["stop_time_diagnostics"] = _leg_diagnostics(con)
    return edges


def travel_summary_graph(
    con: duckdb.DuckDBPyConnection,
    start_time: str | None = None,
    end_time: str | None = None,
    calendar_start: str | None = None,
    calendar_end: str | None = None,
    as_nx: bool | None = None,
    directed: bool = False,
    use_frequencies: bool = True,
    *,
    interpolate_stop_times: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.DiGraph | nx.Graph:
    """
    Aggregate GTFS service into a weighted stop-to-stop summary graph.

    The function builds a stop-level graph from the GTFS feed, where each edge
    represents the aggregated service between two consecutive stops.

    Edge attributes
    ~~~~~~~~~~~~~~~
    ``travel_time_sec``
        Service-weighted average travel time in seconds across all trips
        serving this stop pair.
    ``frequency``
        Total number of leg traversals in the resolved calendar
        window.  When ``use_frequencies=True`` and ``frequencies.txt`` is
        present, headway-based services are expanded into effective trip
        counts.  For example, a trip running every 10 minutes for 1 hour
        contributes 6 traversals per active service day. Non-exact frequency
        service uses expected counts (which may be fractional); exact service
        counts departures in the half-open frequency period. Time bounds apply
        to each leg, accounting for offsets from the first stop.

    ``interpolated_frequency``
        With interpolation enabled, traversals using at least one estimated clock.
        Edge frame attributes (or NetworkX graph attributes) include
        ``stop_time_diagnostics``, counting time exclusions once per active trip
        template before time-of-day filtering or frequency expansion.

    Edge geometry
    ~~~~~~~~~~~~~
    Edge geometry is represented as a straight stop-to-stop line. GeoDataFrame
    outputs retain stops with missing geometries and their incident edges,
    whose geometries are also missing. Conversion to NetworkX omits those
    stops and their incident edges.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        DuckDB connection with GTFS tables loaded.
    start_time : str | None, default=None
        Optional lower bound (inclusive) for departure time, ``HH:MM:SS``.
    end_time : str | None, default=None
        Optional upper bound (inclusive) for next-stop arrival time,
        ``HH:MM:SS``.
    calendar_start : str | None, default=None
        Optional calendar window start, ``YYYYMMDD``.
    calendar_end : str | None, default=None
        Optional calendar window end, ``YYYYMMDD``.
    as_nx : bool, optional
        If ``True``, return a NetworkX graph instead of GeoDataFrames.
        If not explicitly set, defaults to ``False`` (GeoDataFrame output).

        .. deprecated::
            ``as_nx`` is deprecated and will be removed in a future release.
            Use :func:`gdf_to_nx` on the returned GeoDataFrames instead to
            obtain a NetworkX graph.
    directed : bool, default=False
        If ``True``, return a directed graph (``nx.DiGraph``) preserving
        trip direction.  If ``False`` (default), edges in opposite
        directions are merged into undirected edges (``nx.Graph``).
    use_frequencies : bool, default=True
        If ``True`` and ``frequencies.txt`` exists, expand headway-based
        service into effective trip counts for the ``frequency`` attribute.
    interpolate_stop_times : bool, default=False
        Estimate missing intermediate times by distance where available and
        by stop count otherwise. A single missing clock may copy its counterpart.
        Endpoints, exact timepoints and pickup/drop-off windows remain protected.

    Returns
    -------
    tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.DiGraph | nx.Graph
        ``(nodes_gdf, edges_gdf)`` when ``as_nx`` is ``False``; otherwise
        a ``nx.DiGraph`` (``directed=True``) or ``nx.Graph``
        (``directed=False``) built from those GeoDataFrames via
        :func:`city2graph.utils.gdf_to_nx`, following its shared
        conventions: integer node identifiers with the original
        ``stop_id`` values stored in the ``_original_index`` node
        attribute, node ``pos`` coordinates, and graph-level CRS and
        index metadata.
    """
    if as_nx is not None:
        warnings.warn(
            "`as_nx` is deprecated and will be removed in a future release. "
            "Use `gdf_to_nx()` on the returned GeoDataFrames instead to "
            "obtain a NetworkX graph.",
            DeprecationWarning,
            stacklevel=2,
        )
    as_nx = bool(as_nx)

    edges_gdf = _travel_summary_edges(
        con,
        start_time=start_time,
        end_time=end_time,
        calendar_start=calendar_start,
        calendar_end=calendar_end,
        directed=directed,
        use_frequencies=use_frequencies,
        interpolate_stop_times=interpolate_stop_times,
        report_exclusions=True,
    )
    nodes_gdf = _build_travel_summary_nodes(con)

    if not as_nx:
        return nodes_gdf, edges_gdf

    graph = gdf_to_nx(nodes=nodes_gdf, edges=edges_gdf, directed=directed)
    graph.graph["stop_time_diagnostics"] = edges_gdf.attrs["stop_time_diagnostics"]
    return graph


def get_gtfs_summary(con: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    """
    Summarise generated GTFS service and inspect its recorded assumptions.

    The report uses persisted generation records and survives GTFS save/load.
    Edge counts are directed and cover all active service days in the feed.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Generated feed, optionally restored with embedded metadata.

    Returns
    -------
    dict of str to pandas.DataFrame
        ``summary`` contains one row with model, start_date, end_date,
        service_days, route_count, pattern_count, stop_count,
        expected_departures, mean_daily_departures, min_speed_kph,
        max_speed_kph, specified_pattern_count, completed_pattern_count and
        excluded_relation_count. Daily averages include closed dates between
        the bounds. Speed bounds cover adopted speeds for untimed legs only.
        ``assumptions`` lists adopted values and ignored source information;
        ``services`` lists each date and service period with its activity and
        number of departures; ``edges`` gives combined stop-to-stop traversals
        with start_date and end_date identifying the frequency aggregation period.
        Frequency-based departures are expected counts unless exact_times is 1.
        Service columns and their order are stable. Non-applicable frequency
        fields are null for fixed departures. Counts and seconds are numeric,
        integer fields are nullable Int64, and IDs, dates and clocks are strings.

    Raises
    ------
    ValueError
        If generation metadata is unavailable or predates effective records.
    """
    metadata = _report_metadata(con)
    edges = _travel_summary_edges(con, directed=True)
    return _generation_report(con, metadata, edges)


def load_osm_transport(  # noqa: PLR0913 - source selection and shared generation options
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
    stop_level: Literal["station", "platform"] = "station",
    on_invalid: Literal["raise", "skip"] = "skip",
    data: TransportData | Mapping[str, pd.DataFrame] | None = None,
    overwrite_source: bool = _OVERWRITE_SOURCE,
    endpoint: str = _OSM_ENDPOINT,
    timeout: float = _OSM_TIMEOUT,
    use_source_tags: bool = _USE_SOURCE_TAGS,
    **gtfs_options: Unpack[GenerationOptions],
) -> duckdb.DuckDBPyConnection:
    """
    Retrieve OSM public transport relations and create a GTFS feed.

    Relation IDs identify patterns as ``osm:relation:<id>``. Ordered PTv2 stop
    members are retained, including stops outside the selection boundary.
    Operating conditions are passed to the shared create_gtfs implementation.

    Parameters
    ----------
    services : pandas.DataFrame, optional
        Optional operating conditions referring to the OSM pattern identifiers.
        Omit the table for all patterns with source-first completion; omit
        pattern_id to apply supplied periods to every selected pattern.
    agency : pandas.DataFrame, optional
        Explicit agency metadata. Missing metadata uses available source fields,
        unique geographical timezone inference and warned placeholders.
    area : list, Polygon, MultiPolygon, GeoSeries or GeoDataFrame, optional
        WGS84 bounding box or polygon selection. Exclusive with place_name.
        May be omitted when relation_ids is supplied.
    place_name : str, optional
        Place boundary to resolve with get_boundaries.
    route_name : str, optional
        Exact route-name filter. Ambiguous matches require relation_ids.
    route_ref : str, optional
        Exact route reference filter within the selected area.
    relation_ids : list of int, optional
        Explicit route relation IDs. These select complete relations.
    modes : str or sequence of str, optional
        Select tram (including light_rail), subway, train, bus, trolleybus,
        ferry, monorail or funicular. Defaults to all supported modes.
        Combined with relation_ids and pattern_ids by intersection.
        Also filters supplemental routes by their effective route_type.
        Documented extended GTFS types use the corresponding mode group;
        original route_type values are preserved.
    pattern_ids : str or sequence of str, optional
        Additional explicit pattern selection. Intersects modes and relation_ids.
        Unlisted service patterns otherwise remain in the generated feed.
        Applies to supplemental patterns too. Mode-excluded source IDs are known
        from candidate tags, without downloading their member geometry.
    stop_level : {"station", "platform"}, default "station"
        Use named OSM stop areas to consolidate their boarding elements. Ambiguous
        or missing stop-area membership retains individual source IDs. Platform
        mode preserves every source boarding ID. Station grouping adds no transfer
        time or platform accessibility information.
    on_invalid : {"raise", "skip"}, default "skip"
        Report all invalid source relations together, or explicitly exclude them
        with a warning and persisted reasons. Network and explicit-input errors
        are never skipped; an empty result always raises an error.
    data : mapping of str to pandas.DataFrame, optional
        Supplemental routes, stops, patterns or stop_sequences using common IDs.
        Supplied values are validated before retrieval, including excluded rows.
        Missing source route names and stop information may be completed here.
    overwrite_source : bool, default False
        Allow explicit replacement of source fields and stopping sequences.
    endpoint : str, default "https://overpass-api.de/api/interpreter"
        Configurable Overpass endpoint.
    timeout : float, default 180
        HTTP and Overpass timeout in seconds.
    use_source_tags : bool, default True
        Adopt supported interval, opening_hours, duration and interval:conditional
        tags below explicit overrides. Unambiguous parent metadata can be inherited.
        Unsupported tags and rejection reasons remain in the generation report.
    **gtfs_options
        Calendar, timing, headway, representative-date and optional-table
        overrides forwarded to create_gtfs, including group_headway_secs and
        pattern selection. Missing values use simple-v1. Unlisted service patterns remain.

    Returns
    -------
    duckdb.DuckDBPyConnection
        Generated feed ready for analysis and save_gtfs.

    Raises
    ------
    ValueError
        If selection is empty, ambiguous or cannot supply the required inputs.
    RuntimeError
        If Overpass fails or returns an incomplete response.
    """
    return _load_osm_transport(
        _SourceIO(_data._prepare_area_and_bbox, _data.get_boundaries),
        services=services,
        agency=agency,
        area=area,
        place_name=place_name,
        route_name=route_name,
        route_ref=route_ref,
        relation_ids=relation_ids,
        modes=modes,
        pattern_ids=pattern_ids,
        on_invalid=on_invalid,
        stop_level=stop_level,
        data=data,
        overwrite_source=overwrite_source,
        endpoint=endpoint,
        timeout=timeout,
        use_source_tags=use_source_tags,
        **gtfs_options,
    )
