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
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Literal
from typing import Unpack

import duckdb
import geopandas as gpd
import pandas as pd
from shapely import wkt

from . import data as _data
from .gtfs.feed import _INCLUDE_METADATA
from .gtfs.feed import _ensure_graph_sql_support
from .gtfs.feed import save_gtfs
from .gtfs.generation import create_gtfs
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


def _ensure_stops_geometry(con: duckdb.DuckDBPyConnection) -> None:
    """
    Create a temp relation named _stops_for_graph with a geometry column.

    If stops.geometry already exists, it is reused. Otherwise geometry is built
    from stop_lon/stop_lat.

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

    con.execute("DROP VIEW IF EXISTS _stops_for_graph")
    con.execute("DROP TABLE IF EXISTS _stops_for_graph")

    if "geometry" in stop_columns:
        con.execute("CREATE TEMP VIEW _stops_for_graph AS SELECT * FROM stops")
        return

    if not {"stop_lon", "stop_lat"}.issubset(stop_columns):
        msg = "stops must contain either a geometry column or both stop_lon and stop_lat."
        raise ValueError(msg)

    con.execute(  # pragma: no cover - exercised by feeds without prebuilt stop geometry
        """
        CREATE TEMP TABLE _stops_for_graph AS
        SELECT
            *,
            CASE
                WHEN try_cast(NULLIF(stop_lon, '') AS DOUBLE) IS NOT NULL
                 AND try_cast(NULLIF(stop_lat, '') AS DOUBLE) IS NOT NULL
                THEN ST_Point(
                    CAST(NULLIF(stop_lon, '') AS DOUBLE),
                    CAST(NULLIF(stop_lat, '') AS DOUBLE)
                )
                ELSE NULL
            END AS geometry
        FROM stops
        """
    )


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


def _timestamp(gtfs_time: str | float | None, service_date: datetime | str) -> datetime | None:
    """
    Build an absolute timestamp from GTFS time and service date.

    Returns ``None`` when either value cannot be parsed safely.

    Parameters
    ----------
    gtfs_time : str | float | None
        Time value in GTFS format.
    service_date : datetime | str
        Service date as ``datetime`` or ``YYYYMMDD`` string.

    Returns
    -------
    datetime | None
        Parsed timestamp with day rollover support, or ``None`` on failure.
    """
    try:
        if isinstance(service_date, str):
            dt = datetime.strptime(str(service_date), "%Y%m%d")
        elif isinstance(service_date, datetime):
            dt = service_date
        else:  # pragma: no cover - public call sites pass strings or datetimes
            dt = datetime.strptime(str(int(service_date)), "%Y%m%d")

        secs = 0.0 if pd.isna(gtfs_time) or gtfs_time is None else _time_to_seconds(gtfs_time)

        h, remainder = divmod(int(secs), 3600)
        m, s = divmod(remainder, 60)
        day_offset, h = divmod(h, 24)
        return dt.replace(hour=h, minute=m, second=s) + timedelta(days=day_offset)
    except (TypeError, ValueError, OverflowError):
        return None


def load_gtfs(
    path: str | Path, *, include_metadata: bool = _INCLUDE_METADATA
) -> duckdb.DuckDBPyConnection:
    """
    Load a GTFS ZIP archive into an in-memory DuckDB database.

    The function reads ``*.txt`` GTFS files as tables, registers a time parsing
    UDF, and materializes ``stops.geometry`` as points from
    ``stop_lon``/``stop_lat`` when stop coordinates are available.

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
            # Build point geometries while guarding against empty coordinate strings.
            con.execute("""
                ALTER TABLE stops ADD COLUMN geometry GEOMETRY;
                UPDATE stops SET geometry = ST_Point(CAST(NULLIF(stop_lon, '') AS DOUBLE), CAST(NULLIF(stop_lat, '') AS DOUBLE))
                WHERE try_cast(NULLIF(stop_lon, '') as DOUBLE) IS NOT NULL AND try_cast(NULLIF(stop_lat, '') as DOUBLE) IS NOT NULL;
            """)

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
    ``lon``/``lat`` fields when available.

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
    try:
        for file_path in path.rglob("*.json"):
            with file_path.open(encoding="utf-8") as f:
                raw = json.load(f)
            data = raw.get("data", {})
            if "stations" in data:
                rows = data["stations"]
            elif "bikes" in data:
                rows = data["bikes"]
            elif "vehicles" in data:
                rows = data["vehicles"]
            elif "vehicle_types" in data:
                rows = data["vehicle_types"]
            elif "feeds" in data:
                rows = data["feeds"]
            else:
                rows = [data]
            if not rows:
                continue
            has_files = True
            table_name = file_path.stem.replace("-", "_").lower()
            frame = pd.DataFrame(rows)
            con.register("_gbfs_tmp", frame)
            con.execute(f"""
                CREATE OR REPLACE TABLE {table_name} AS
                SELECT *
                FROM _gbfs_tmp
            """)  # noqa: S608
            con.unregister("_gbfs_tmp")
    except (OSError, json.JSONDecodeError, duckdb.Error):
        logger.exception("Failed to read GBFS data at %s", path)
        return con
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
        WHERE headway IS NULL OR NOT isfinite(headway) OR headway <= 0
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
    directed: bool = False,
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
        If ``True`` and stop geometries are available, include a line geometry
        for each OD pair.
    directed : bool, default=False
        If ``True``, preserve the original trip direction for each OD pair.
        If ``False`` (default), canonicalize each pair so that
        ``orig_stop_id < dest_stop_id`` and aggregate both directions together.

    Returns
    -------
    pd.DataFrame | gpd.GeoDataFrame
        One row per directed (or undirected) stop-to-stop leg.
    """
    tables = _list_tables(con)
    required = {"stop_times", "trips"}
    if not required.issubset(tables):
        logger.error("GTFS feed incomplete - need %s", ", ".join(required))
        return pd.DataFrame()

    has_cal = "calendar" in tables

    s_dt, e_dt = start_date, end_date
    if not s_dt or not e_dt:
        first, last = _feed_date_bounds(con, tables)
        s_dt = s_dt or (first.strftime("%Y%m%d") if first else None)
        e_dt = e_dt or (last.strftime("%Y%m%d") if last else None)
    if not s_dt or not e_dt:
        return pd.DataFrame()

    _build_active_dates(
        con,
        start_date=str(s_dt),
        end_date=str(e_dt),
        include_calendar=has_cal,
        include_calendar_dates="calendar_dates" in tables,
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE _unique_dates AS
        SELECT DISTINCT
            service_id,
            active_date,
            strftime(active_date, '%Y%m%d') as active_date_str
        FROM _active_dates
        """
    )

    query = """
        WITH st AS (
            SELECT
                trip_id,
                stop_id as orig_stop_id,
                departure_time,
                LEAD(stop_id) OVER (PARTITION BY trip_id ORDER BY CAST(stop_sequence AS INTEGER)) as dest_stop_id,
                LEAD(arrival_time) OVER (PARTITION BY trip_id ORDER BY CAST(stop_sequence AS INTEGER)) as arrival_time_next
            FROM stop_times
            WHERE try_cast(stop_sequence as integer) IS NOT NULL
        ),
        valid_st AS (
            SELECT st.*, t.service_id
            FROM st
            JOIN trips t ON st.trip_id = t.trip_id
            WHERE st.dest_stop_id IS NOT NULL
        )
        SELECT
            v.trip_id,
            v.service_id,
            v.orig_stop_id,
            v.dest_stop_id,
            v.departure_time,
            v.arrival_time_next as arrival_time,
            d.active_date_str,
            strftime(d.active_date, '%Y-%m-%d') as date
    """

    if include_geometry and "stops" in tables:
        query += """,
            ST_AsText(ST_MakeLine(s1.geometry, s2.geometry)) as geometry_wkt
        """
    else:
        include_geometry = False

    query += """
        FROM valid_st v
        JOIN _unique_dates d ON v.service_id = d.service_id
    """

    if include_geometry:
        query += """
        LEFT JOIN stops s1 ON v.orig_stop_id = s1.stop_id
        LEFT JOIN stops s2 ON v.dest_stop_id = s2.stop_id
        """

    od_pairs_df = con.execute(query).df()

    if od_pairs_df.empty:
        return pd.DataFrame()

    for source, target in (("departure_time", "departure_ts"), ("arrival_time", "arrival_ts")):
        pairs = pd.MultiIndex.from_arrays([od_pairs_df[source], od_pairs_df["active_date_str"]])
        codes, unique_pairs = pd.factorize(pairs, sort=False)
        timestamps = pd.Series([_timestamp(time, date) for time, date in unique_pairs])
        od_pairs_df[target] = timestamps.take(codes).set_axis(od_pairs_df.index)

    od_pairs_df = od_pairs_df.dropna(subset=["departure_ts", "arrival_ts"])
    od_pairs_df["travel_time_sec"] = (
        od_pairs_df["arrival_ts"] - od_pairs_df["departure_ts"]
    ).dt.total_seconds()

    od_pairs_df = od_pairs_df.drop(columns=["departure_time", "arrival_time", "active_date_str"])

    if not directed:
        # Canonicalize: ensure orig_stop_id <= dest_stop_id
        swap = od_pairs_df["orig_stop_id"] > od_pairs_df["dest_stop_id"]
        od_pairs_df.loc[swap, ["orig_stop_id", "dest_stop_id"]] = od_pairs_df.loc[
            swap, ["dest_stop_id", "orig_stop_id"]
        ].to_numpy()
        od_pairs_df.loc[swap, ["departure_ts", "arrival_ts"]] = od_pairs_df.loc[
            swap, ["arrival_ts", "departure_ts"]
        ].to_numpy()

    od_pairs_df = od_pairs_df.sort_values(["trip_id", "date"]).reset_index(drop=True)

    if include_geometry and "geometry_wkt" in od_pairs_df.columns:
        od_pairs_df = _convert_wkt_column(od_pairs_df)
        return gpd.GeoDataFrame(od_pairs_df, geometry="geometry", crs="EPSG:4326")

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


def _prepare_summary_graph_support(
    con: duckdb.DuckDBPyConnection,
    *,
    tables: set[str],
    use_frequencies: bool,
) -> None:
    """
    Prepare optional temp tables used by the summary graph query.

    These tables encapsulate optional headway expansion so the main query stays
    focused on aggregation.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Open DuckDB connection with GTFS tables loaded.
    tables : set[str]
        Available GTFS table names.
    use_frequencies : bool
        Whether to expand ``frequencies.txt`` headways into trip counts.

    Returns
    -------
    None
        The function creates or replaces temporary support tables.
    """
    _build_frequency_periods(con, tables=tables, use_frequencies=use_frequencies)


def _build_travel_summary_edges(
    con: duckdb.DuckDBPyConnection,
    *,
    min_departure_sec: float | None,
    max_arrival_sec: float | None,
    directed: bool,
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

    Returns
    -------
    gpd.GeoDataFrame
        Aggregated edge records indexed by stop pairs.
    """
    edge_query = """
        WITH st AS (
            SELECT
                trip_id,
                stop_id,
                CASE
                    WHEN departure_time IS NULL
                      OR lower(trim(CAST(departure_time AS VARCHAR))) IN ('', 'nan', 'none')
                    THEN NULL
                    ELSE time_to_seconds(CAST(departure_time AS VARCHAR))
                END AS departure_time_sec,
                LEAD(stop_id) OVER (
                    PARTITION BY trip_id
                    ORDER BY CAST(stop_sequence AS INTEGER)
                ) AS next_stop_id,
                LEAD(
                    CASE
                        WHEN arrival_time IS NULL
                          OR lower(trim(CAST(arrival_time AS VARCHAR))) IN ('', 'nan', 'none')
                        THEN NULL
                        ELSE time_to_seconds(CAST(arrival_time AS VARCHAR))
                    END
                ) OVER (
                    PARTITION BY trip_id
                    ORDER BY CAST(stop_sequence AS INTEGER)
                ) AS next_arrival_time_sec
            FROM stop_times
            WHERE try_cast(stop_sequence AS INTEGER) IS NOT NULL
        ),
        trip_origins AS (
            SELECT trip_id, arg_min(departure_time_sec, CAST(stop_sequence AS INTEGER)) AS origin
            FROM (
                SELECT trip_id, stop_sequence,
                    time_to_seconds(CAST(departure_time AS VARCHAR)) AS departure_time_sec
                FROM stop_times
                WHERE departure_time IS NOT NULL
                  AND lower(trim(CAST(departure_time AS VARCHAR))) NOT IN ('', 'nan', 'none')
                  AND try_cast(stop_sequence AS INTEGER) IS NOT NULL
            ) GROUP BY trip_id
        ),
        bounds AS (SELECT ?::DOUBLE AS lower_time, ?::DOUBLE AS upper_time),
        weighted AS (
            SELECT st.stop_id, st.next_stop_id,
                st.next_arrival_time_sec - st.departure_time_sec AS travel_time,
                sc.sc * CASE
                    WHEN f.trip_id IS NULL THEN
                        CASE WHEN (b.lower_time IS NULL OR st.departure_time_sec >= b.lower_time)
                            AND (b.upper_time IS NULL OR st.next_arrival_time_sec <= b.upper_time)
                        THEN 1.0 ELSE 0.0 END
                    WHEN f.exact_times = 0 THEN
                        greatest(0.0,
                            least(f.end_sec, coalesce(b.upper_time - (st.next_arrival_time_sec - o.origin), f.end_sec))
                            - greatest(f.start_sec, coalesce(b.lower_time - (st.departure_time_sec - o.origin), f.start_sec))
                        ) / f.headway
                    ELSE greatest(0.0,
                        least(ceil((f.end_sec - f.start_sec) / f.headway) - 1,
                            coalesce(floor((b.upper_time - (st.next_arrival_time_sec - o.origin) - f.start_sec) / f.headway),
                                ceil((f.end_sec - f.start_sec) / f.headway) - 1))
                        - greatest(0.0, coalesce(ceil((b.lower_time - (st.departure_time_sec - o.origin) - f.start_sec) / f.headway), 0.0))
                        + 1.0)
                END AS service_count
            FROM st
            JOIN trips t ON st.trip_id = t.trip_id
            JOIN _service_counts sc ON t.service_id = sc.service_id
            JOIN trip_origins o ON st.trip_id = o.trip_id
            LEFT JOIN _frequency_periods f ON st.trip_id = f.trip_id
            CROSS JOIN bounds b
            WHERE st.next_stop_id IS NOT NULL
                AND st.departure_time_sec IS NOT NULL
                AND st.next_arrival_time_sec > st.departure_time_sec
                AND sc.sc > 0
        ),
        valid_pairs AS (SELECT * FROM weighted WHERE service_count > 0),
        agg_pairs AS (
            SELECT
                stop_id,
                next_stop_id,
                SUM(travel_time * service_count) / SUM(service_count) AS travel_time_sec,
                SUM(service_count) AS frequency
            FROM valid_pairs
            GROUP BY stop_id, next_stop_id
        )
        SELECT
            a.stop_id AS from_stop_id,
            a.next_stop_id AS to_stop_id,
            a.travel_time_sec,
            a.frequency,
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
    """

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
        edges_df["_weighted_tt"] = edges_df["travel_time_sec"] * edges_df["frequency"]
        agg = edges_df.groupby(["from_stop_id", "to_stop_id"], sort=False).agg(
            _weighted_tt=("_weighted_tt", "sum"),
            frequency=("frequency", "sum"),
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
    _prepare_summary_graph_support(
        con,
        tables=tables,
        use_frequencies=use_frequencies,
    )
    return _build_travel_summary_edges(
        con,
        min_departure_sec=min_departure_sec,
        max_arrival_sec=max_arrival_sec,
        directed=directed,
    )


def travel_summary_graph(
    con: duckdb.DuckDBPyConnection,
    start_time: str | None = None,
    end_time: str | None = None,
    calendar_start: str | None = None,
    calendar_end: str | None = None,
    as_nx: bool | None = None,
    directed: bool = False,
    use_frequencies: bool = True,
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

    Edge geometry
    ~~~~~~~~~~~~~
    Edge geometry is represented as a straight stop-to-stop line.

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
    )
    nodes_gdf = _build_travel_summary_nodes(con)

    if not as_nx:
        return nodes_gdf, edges_gdf

    return gdf_to_nx(nodes=nodes_gdf, edges=edges_gdf, directed=directed)


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
