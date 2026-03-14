"""
GTFS transportation network utilities.

The functions in this module load GTFS feeds into DuckDB, derive stop-to-stop
origin/destination records, and aggregate those records into transport summary
graphs for downstream network analysis.
"""

from __future__ import annotations

import logging
import tempfile
import zipfile
from contextlib import suppress
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import duckdb
import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely import wkt

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

__all__ = ["get_od_pairs", "load_gtfs", "travel_summary_graph"]

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


def _validate_calendar_window(
    con: duckdb.DuckDBPyConnection,
    *,
    calendar_start: str | None,
    calendar_end: str | None,
    tables: set[str],
) -> None:
    """
    Validate optional calendar window arguments against GTFS calendar bounds.

    The check enforces valid format, in-range dates, and ordered bounds.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Open DuckDB connection.
    calendar_start : str | None
        Requested lower bound date in ``YYYYMMDD`` format.
    calendar_end : str | None
        Requested upper bound date in ``YYYYMMDD`` format.
    tables : set[str]
        Available GTFS table names.

    Returns
    -------
    None
        Raises ``ValueError`` when validation fails.
    """
    if not (calendar_start or calendar_end):
        return

    if "calendar" not in tables:
        msg = "calendar_start/calendar_end specified but GTFS feed has no calendar.txt"
        raise ValueError(msg)

    cal_check = con.execute("SELECT COUNT(*) FROM calendar").fetchone()[0]
    if cal_check == 0:
        msg = "calendar_start/calendar_end specified but calendar.txt is empty"
        raise ValueError(msg)

    cal_bounds = con.execute("SELECT MIN(start_date), MAX(end_date) FROM calendar").fetchone()
    gtfs_start = datetime.strptime(str(cal_bounds[0]), "%Y%m%d")
    gtfs_end = datetime.strptime(str(cal_bounds[1]), "%Y%m%d")

    start_dt: datetime | None = None
    end_dt: datetime | None = None
    outside_msg: str | None = None

    try:
        if calendar_start:
            start_dt = datetime.strptime(calendar_start, "%Y%m%d")
            if start_dt < gtfs_start or start_dt > gtfs_end:
                outside_msg = (
                    f"calendar_start ({calendar_start}) is outside the valid GTFS date range"
                )
        if calendar_end:
            end_dt = datetime.strptime(calendar_end, "%Y%m%d")
            if end_dt < gtfs_start or end_dt > gtfs_end:
                outside_msg = f"calendar_end ({calendar_end}) is outside the valid GTFS date range"
    except ValueError as error:
        msg = "Invalid calendar date format. Expected YYYYMMDD."
        raise ValueError(msg) from error

    if outside_msg:
        raise ValueError(outside_msg)

    if start_dt and end_dt and start_dt > end_dt:
        msg = f"calendar_start ({calendar_start}) must be <= calendar_end ({calendar_end})"
        raise ValueError(msg)


def _build_service_counts(
    con: duckdb.DuckDBPyConnection,
    *,
    calendar_start: str | None,
    calendar_end: str | None,
    tables: set[str],
) -> None:
    """
    Create temporary per-service frequency counts used in edge aggregation.

    Counts represent the number of active days per service in the window.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Open DuckDB connection.
    calendar_start : str | None
        Optional lower bound date in ``YYYYMMDD``.
    calendar_end : str | None
        Optional upper bound date in ``YYYYMMDD``.
    tables : set[str]
        Available GTFS table names.

    Returns
    -------
    None
        The function creates or replaces ``_service_counts``.
    """
    if calendar_start and calendar_end:
        _build_active_dates(
            con,
            start_date=calendar_start,
            end_date=calendar_end,
            include_calendar=True,
            include_calendar_dates="calendar_dates" in tables,
        )
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE _service_counts AS
            SELECT service_id, COUNT(DISTINCT active_date) as sc
            FROM _active_dates
            GROUP BY service_id
            """
        )
        return

    con.execute(
        "CREATE OR REPLACE TEMP TABLE _service_counts AS SELECT DISTINCT service_id, 1 as sc FROM trips"
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


def _ensure_graph_sql_support(con: duckdb.DuckDBPyConnection) -> None:
    """
    Ensure SQL features required by transport graph queries are available.

    The function loads DuckDB spatial support and registers the
    ``time_to_seconds`` scalar UDF when missing.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Open DuckDB connection.

    Returns
    -------
    None
        The function mutates runtime SQL capabilities in ``con``.

    Raises
    ------
    RuntimeError
        If the DuckDB spatial extension cannot be loaded or installed.
    """
    try:
        con.execute("LOAD spatial;")
    except duckdb.Error:
        try:
            con.execute("INSTALL spatial; LOAD spatial;")
        except duckdb.Error as error:
            msg = "DuckDB spatial extension is required to build transport graph geometries."
            raise RuntimeError(msg) from error

    # Usually raises duckdb.Error if the function already exists.
    with suppress(duckdb.Error):
        con.create_function("time_to_seconds", _time_to_seconds, ["VARCHAR"], "DOUBLE")


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

    con.execute(
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

    bounds: list[tuple[datetime, datetime]] = []

    if has_calendar:
        row = con.execute(
            "SELECT MIN(start_date), MAX(end_date), COUNT(*) FROM calendar"
        ).fetchone()
        if row and row[2] and row[0] and row[1]:
            bounds.append(
                (
                    datetime.strptime(str(row[0]), "%Y%m%d"),
                    datetime.strptime(str(row[1]), "%Y%m%d"),
                )
            )

    if has_calendar_dates:
        row = con.execute("SELECT MIN(date), MAX(date), COUNT(*) FROM calendar_dates").fetchone()
        if row and row[2] and row[0] and row[1]:
            bounds.append(
                (
                    datetime.strptime(str(row[0]), "%Y%m%d"),
                    datetime.strptime(str(row[1]), "%Y%m%d"),
                )
            )

    if not bounds:
        if calendar_start or calendar_end:
            msg = (
                "calendar_start/calendar_end specified but calendar tables contain no usable dates"
            )
            raise ValueError(msg)
        return None, None, False

    gtfs_start = min(start for start, _ in bounds)
    gtfs_end = max(end for _, end in bounds)

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


def _time_to_seconds(value: str | float | None) -> float:
    """
    Convert a GTFS time value into seconds from midnight.

    GTFS may contain values beyond 24 hours (for trips after midnight), and this
    helper preserves those values as absolute seconds.

    Parameters
    ----------
    value : str | float | None
        GTFS time value, typically ``HH:MM:SS`` or a numeric value.

    Returns
    -------
    float
        Seconds from midnight.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0.0
    if not isinstance(value, str):
        return float(value)
    value = str(value).strip()
    if value in {"", "nan", "None"}:
        return 0.0
    try:
        h, m, s = map(int, value.split(":"))
        return h * 3600 + m * 60 + s
    except (ValueError, AttributeError):
        return float(value)


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
        else:
            dt = datetime.strptime(str(int(service_date)), "%Y%m%d")

        secs = 0.0 if pd.isna(gtfs_time) or gtfs_time is None else _time_to_seconds(gtfs_time)

        h, remainder = divmod(int(secs), 3600)
        m, s = divmod(remainder, 60)
        day_offset, h = divmod(h, 24)
        return dt.replace(hour=h, minute=m, second=s) + timedelta(days=day_offset)
    except (TypeError, ValueError, OverflowError):
        return None


def load_gtfs(path: str | Path) -> duckdb.DuckDBPyConnection:
    """
    Load a GTFS ZIP archive into an in-memory DuckDB database.

    The function reads ``*.txt`` GTFS files as tables, registers a time parsing
    UDF, and materializes optional spatial helpers:

    - ``stops.geometry`` as points from ``stop_lon``/``stop_lat``
    - ``shapes_geom`` as one polyline per ``shape_id``

    Parameters
    ----------
    path : str | Path
        Path to a GTFS ``.zip`` file.

    Returns
    -------
    duckdb.DuckDBPyConnection
        In-memory DuckDB connection containing imported GTFS tables.
    """
    con = duckdb.connect(":memory:")
    con.execute("INSTALL spatial; LOAD spatial;")

    # Register a scalar UDF used by SQL queries in this module.
    con.create_function("time_to_seconds", _time_to_seconds, ["VARCHAR"], "DOUBLE")

    tmp_dir = tempfile.mkdtemp()
    has_files = False

    try:
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if name.endswith(".txt") and not name.endswith("/"):
                    has_files = True
                    zf.extract(name, tmp_dir)
                    table_name = name.rsplit("/", 1)[-1].replace(".txt", "")
                    file_path = str(Path(tmp_dir) / name)
                    con.read_csv(file_path, all_varchar=True, auto_detect=True).create(table_name)
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

    if "shapes" in tables:
        con.execute("""
            CREATE TABLE shapes_geom AS
            SELECT
                shape_id,
                ST_MakeLine(
                    list(
                        ST_Point(
                            CAST(NULLIF(shape_pt_lon, '') AS DOUBLE),
                            CAST(NULLIF(shape_pt_lat, '') AS DOUBLE)
                        ) ORDER BY CAST(NULLIF(shape_pt_sequence, '') AS INTEGER)
                    )
                ) AS geometry
            FROM shapes
            WHERE try_cast(NULLIF(shape_pt_lon, '') as DOUBLE) IS NOT NULL AND try_cast(NULLIF(shape_pt_lat, '') as DOUBLE) IS NOT NULL
            GROUP BY shape_id;
        """)

    logger.info("GTFS loaded: %s", ", ".join(tables))
    return con


def get_od_pairs(
    con: duckdb.DuckDBPyConnection,
    start_date: str | None = None,
    end_date: str | None = None,
    include_geometry: bool = True,
) -> pd.DataFrame | gpd.GeoDataFrame:
    """
    Generate stop-to-stop OD pairs from GTFS trip stop sequences.

    For each ``trip_id``, consecutive stops are paired into directed legs with
    departure/arrival timestamps and per-leg travel time in seconds. Service
    activity is derived from ``calendar`` and optionally adjusted using
    ``calendar_dates``.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        DuckDB connection with GTFS tables loaded.
    start_date : str | None, default=None
        Inclusive start date (``YYYYMMDD``). Optional when ``calendar`` exists.
    end_date : str | None, default=None
        Inclusive end date (``YYYYMMDD``). Optional when ``calendar`` exists.
    include_geometry : bool, default=True
        If ``True`` and stop geometries are available, include a line geometry
        for each OD pair.

    Returns
    -------
    pd.DataFrame | gpd.GeoDataFrame
        One row per directed stop-to-stop leg.
    """
    tables = _list_tables(con)
    required = {"stop_times", "trips"}
    if not required.issubset(tables):
        logger.error("GTFS feed incomplete - need %s", ", ".join(required))
        return pd.DataFrame()

    has_cal = "calendar" in tables

    if has_cal:
        cal_df = con.execute(
            "SELECT MIN(start_date) as s, MAX(end_date) as e FROM calendar"
        ).fetchone()
        s_dt = start_date or cal_df[0]
        e_dt = end_date or cal_df[1]
    else:
        if not start_date or not end_date:
            return pd.DataFrame()
        s_dt = start_date
        e_dt = end_date

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

    od_pairs_df["departure_ts"] = od_pairs_df.apply(
        lambda row: _timestamp(row["departure_time"], row["active_date_str"]), axis=1
    )
    od_pairs_df["arrival_ts"] = od_pairs_df.apply(
        lambda row: _timestamp(row["arrival_time"], row["active_date_str"]), axis=1
    )

    od_pairs_df = od_pairs_df.dropna(subset=["departure_ts", "arrival_ts"])
    od_pairs_df["travel_time_sec"] = (
        od_pairs_df["arrival_ts"] - od_pairs_df["departure_ts"]
    ).dt.total_seconds()

    od_pairs_df = od_pairs_df.drop(columns=["departure_time", "arrival_time", "active_date_str"])
    od_pairs_df = od_pairs_df.sort_values(["trip_id", "date"]).reset_index(drop=True)

    if include_geometry and "geometry_wkt" in od_pairs_df.columns:
        od_pairs_df = _convert_wkt_column(od_pairs_df)
        return gpd.GeoDataFrame(od_pairs_df, geometry="geometry", crs="EPSG:4326")

    return od_pairs_df


def travel_summary_graph(
    con: duckdb.DuckDBPyConnection,
    start_time: str | None = None,
    end_time: str | None = None,
    calendar_start: str | None = None,
    calendar_end: str | None = None,
    as_nx: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph:
    """
    Aggregate GTFS service into a weighted stop-to-stop summary graph.

    Fixes compared with the original version:
    - requires trips, stop_times, and stops
    - honors one-sided calendar bounds by inferring the missing side
    - defaults to full-feed date-aware service counts when calendar data exists
    - supports calendar_dates-only feeds
    - constructs stop geometry if stops.geometry is absent
    - returns a directed NetworkX graph when as_nx=True

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
    as_nx : bool, default=False
        If ``True``, return a directed NetworkX graph instead of GeoDataFrames.

    Returns
    -------
    tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph
        ``(nodes_gdf, edges_gdf)`` when ``as_nx`` is ``False``; otherwise a
        directed NetworkX graph built from those GeoDataFrames.
    """
    tables = _list_tables(con)
    required = {"stop_times", "stops", "trips"}
    if not required.issubset(tables):
        missing = sorted(required - tables)
        msg = f"GTFS must contain stop_times, stops, and trips. Missing: {', '.join(missing)}"
        raise ValueError(msg)

    _ensure_graph_sql_support(con)
    _ensure_stops_geometry(con)

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
    else:
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
        st_filtered AS (
            SELECT *
            FROM st
            WHERE next_stop_id IS NOT NULL
              AND departure_time_sec IS NOT NULL
              AND next_arrival_time_sec IS NOT NULL
              AND (? IS NULL OR departure_time_sec >= ?)
              AND (? IS NULL OR next_arrival_time_sec <= ?)
        ),
        valid_pairs AS (
            SELECT
                st.stop_id,
                st.next_stop_id,
                st.next_arrival_time_sec - st.departure_time_sec AS travel_time,
                sc.sc AS service_count
            FROM st_filtered st
            JOIN trips t
              ON st.trip_id = t.trip_id
            JOIN _service_counts sc
              ON t.service_id = sc.service_id
            WHERE (st.next_arrival_time_sec - st.departure_time_sec) > 0
              AND sc.sc > 0
        ),
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
            CAST(a.frequency AS BIGINT) AS frequency,
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
        [min_departure_sec, min_departure_sec, max_arrival_sec, max_arrival_sec],
    ).df()
    edges_df = _convert_wkt_column(edges_df)
    edges_gdf = gpd.GeoDataFrame(edges_df, geometry="geometry", crs="EPSG:4326")
    if "from_stop_id" in edges_gdf.columns and "to_stop_id" in edges_gdf.columns:
        edges_gdf = edges_gdf.set_index(["from_stop_id", "to_stop_id"])

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

    if not as_nx:
        return nodes_gdf, edges_gdf

    graph = nx.DiGraph()
    graph.graph["crs"] = str(nodes_gdf.crs) if nodes_gdf.crs else None

    for stop_id, row in nodes_gdf.iterrows():
        graph.add_node(stop_id, **row.to_dict())

    for (from_stop_id, to_stop_id), row in edges_gdf.iterrows():
        graph.add_edge(from_stop_id, to_stop_id, **row.to_dict())

    return graph
