"""Resolve GTFS stop times and pair adjacent stops for transport analyses."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING

from .feed import _ensure_time_sql_support

if TYPE_CHECKING:
    from collections.abc import Iterator

    import duckdb

logger = logging.getLogger(__name__)


# Each anchor carries its position even when its relevant clock is missing.
# This prevents the window functions from looking through protected stops.
_INTERPOLATED_STOPS_SQL = """
    numbered AS (
        SELECT *, row_number() OVER trip AS pos, count(*) OVER (PARTITION BY trip_id) AS n
            {previous_distance}
        FROM parsed WINDOW trip AS (PARTITION BY trip_id ORDER BY seq)
    ),
    distances AS (
        SELECT *, {distance_run} AS distance_run FROM numbered
    ),
    -- Required clocks and pickup/drop-off windows split runs of missing times.
    classified AS (
        SELECT *, CASE
            WHEN pos IN (1, n) OR exact_time THEN 1
            WHEN has_window THEN 2
            WHEN invalid_clock THEN 3
        END AS protected_reason,
        coalesce(dep, arr) AS known_time
        FROM distances
    ),
    -- Check a one-sided stop against the nearest supplied clocks before copying.
    neighbours AS (
        SELECT *,
            lag(CASE WHEN known_time IS NOT NULL OR protected_reason IS NOT NULL
                THEN struct_pack(t := coalesce(dep, arr)) END IGNORE NULLS) OVER trip AS prev,
            lead(CASE WHEN known_time IS NOT NULL OR protected_reason IS NOT NULL
                THEN struct_pack(t := coalesce(arr, dep)) END IGNORE NULLS) OVER trip AS nxt
        FROM classified WINDOW trip AS (PARTITION BY trip_id ORDER BY seq)
    ),
    partials AS (
        SELECT *, coalesce(prev.t > known_time, false) OR coalesce(nxt.t < known_time, false) AS conflicts,
            CASE WHEN protected_reason IS NULL AND NOT conflicts THEN known_time END AS copied
        FROM neighbours
    ),
    -- Resolved partial stops can now bound runs of entirely untimed stops.
    anchors AS (
        SELECT *,
            lag(CASE WHEN known_time IS NOT NULL OR protected_reason IS NOT NULL
                THEN struct_pack(pos := pos, t := coalesce(dep, copied), d := distance, run := distance_run) END IGNORE NULLS)
                OVER trip AS earlier,
            lead(CASE WHEN known_time IS NOT NULL OR protected_reason IS NOT NULL
                THEN struct_pack(pos := pos, t := coalesce(arr, copied), d := distance, run := distance_run) END IGNORE NULLS)
                OVER trip AS later
        FROM partials WINDOW trip AS (PARTITION BY trip_id ORDER BY seq)
    ),
    estimates AS (
        SELECT *, CASE WHEN protected_reason IS NULL AND known_time IS NULL
            AND earlier.t <= later.t
            THEN earlier.t + round((later.t - earlier.t) * CASE
                WHEN earlier.run = later.run AND later.d > earlier.d
                THEN (distance - earlier.d) / (later.d - earlier.d)
                ELSE (pos - earlier.pos) / (later.pos - earlier.pos) END)
            ELSE copied END AS estimate,
            coalesce(protected_reason, CASE WHEN conflicts OR earlier.t > later.t
                                            THEN 3 ELSE 4 END) AS reason
        FROM anchors
    ),
    resolved AS (
        SELECT *, coalesce(dep, estimate) AS departure_time_sec,
            coalesce(arr, estimate) AS arrival_time_sec,
            dep IS NULL AND estimate IS NOT NULL AS departure_interpolated,
            arr IS NULL AND estimate IS NOT NULL AS arrival_interpolated
        FROM estimates
    )
"""

# Numeric codes let least() select a single reason when both endpoints are missing.
_STOP_TIME_REASONS = (
    "interpolation_disabled",
    "required_time_missing",
    "pickup_drop_off_window",
    "inconsistent_times",
    "missing_anchor",
    "nonpositive_duration",
)


@contextmanager
def _stop_time_legs(
    con: duckdb.DuckDBPyConnection, *, interpolate: bool, summary: bool
) -> Iterator[None]:
    """
    Prepare adjacent legs once per active trip.

    The consumer and diagnostics read the same temporary table. No legs are
    retained after the query finishes.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Connection with active dates or summary service counts prepared.
    interpolate : bool
        Fill eligible intermediate clocks before pairing stops.
    summary : bool
        Use summary service counts and exclude nonpositive durations.

    Yields
    ------
    None
        The temporary leg table is available for the duration of the context.
    """
    services = "_service_counts" if summary else "_active_dates"
    columns = con.table("stop_times").columns if interpolate else ()
    exact = "try_cast(timepoint AS DOUBLE) = 1" if "timepoint" in columns else "false"
    windows = [
        f"nullif(trim(CAST({name} AS VARCHAR)), '') IS NOT NULL"
        for name in ("start_pickup_drop_off_window", "end_pickup_drop_off_window")
        if name in columns
    ]
    has_window = " OR ".join(windows) or "false"
    # Summary historically rejects malformed clocks. OD omits the affected leg.
    parser = "time_to_seconds" if summary else "_od_time_to_seconds"
    if not summary:
        _ensure_time_sql_support(con, strict=False)
    distance = "NULL::DOUBLE"
    previous_distance = ""
    distance_run = "0"
    if "shape_dist_traveled" in columns:
        distance = "try_cast(shape_dist_traveled AS DOUBLE)"
        previous_distance = ", lag(distance) OVER trip AS previous_distance"
        # A change of run anywhere between the two anchors invalidates the whole gap.
        distance_run = """count(*) FILTER (WHERE NOT coalesce(
            isfinite(distance) AND isfinite(previous_distance) AND distance >= previous_distance,
            false)) OVER (PARTITION BY trip_id ORDER BY seq ROWS UNBOUNDED PRECEDING)"""
    resolved = (
        _INTERPOLATED_STOPS_SQL.format(
            previous_distance=previous_distance, distance_run=distance_run
        )
        if interpolate
        else """
        resolved AS (
            SELECT *, dep AS departure_time_sec, arr AS arrival_time_sec, 0 AS reason,
                false AS departure_interpolated, false AS arrival_interpolated FROM parsed
        )
    """
    )
    origin = (
        "first_value(departure_time_sec IGNORE NULLS) OVER "
        "(trip ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)"
        if summary
        else "NULL"
    )
    duration_reason = "WHEN next_arrival_time_sec <= departure_time_sec THEN 5" if summary else ""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _stop_time_legs AS
        WITH clocks AS (
            SELECT st.trip_id, t.service_id, st.stop_id, try_cast(st.stop_sequence AS INTEGER) AS seq,
                lower(trim(CAST(departure_time AS VARCHAR))) AS raw_dep,
                lower(trim(CAST(arrival_time AS VARCHAR))) AS raw_arr,
                {exact} AS exact_time, {has_window} AS has_window,
                departure_time, arrival_time, {distance} AS distance
            FROM stop_times st JOIN trips t ON st.trip_id = t.trip_id
            WHERE try_cast(st.stop_sequence AS INTEGER) IS NOT NULL
                AND t.service_id IN (SELECT service_id FROM {services})
        ), clock_values AS (
            SELECT DISTINCT unnest([raw_dep, raw_arr]) AS raw FROM clocks
        ), seconds AS MATERIALIZED (
            SELECT raw, {parser}(raw) AS sec FROM clock_values
            WHERE raw NOT IN ('', 'none', 'nan')
        ), parsed AS (
            SELECT clocks.*,
                CASE WHEN typeof(departure_time) <> 'VARCHAR' AND {not summary}
                     THEN trunc(try_cast(departure_time AS DOUBLE)) ELSE d.sec END AS dep,
                CASE WHEN typeof(arrival_time) <> 'VARCHAR' AND {not summary}
                     THEN trunc(try_cast(arrival_time AS DOUBLE)) ELSE a.sec END AS arr,
                (dep IS NULL AND raw_dep NOT IN ('', 'none', 'nan'))
                  OR (arr IS NULL AND raw_arr NOT IN ('', 'none', 'nan')) AS invalid_clock
            FROM clocks LEFT JOIN seconds d ON raw_dep = d.raw
                        LEFT JOIN seconds a ON raw_arr = a.raw
        ), {resolved},
        paired AS (
            SELECT trip_id, service_id, seq, stop_id, lead(stop_id) OVER trip AS next_stop_id,
                departure_time_sec, lead(arrival_time_sec) OVER trip AS next_arrival_time_sec,
                departure_interpolated, lead(arrival_interpolated) OVER trip AS arrival_interpolated,
                {origin} AS origin,
                least(CASE WHEN departure_time_sec IS NULL THEN reason END,
                      lead(CASE WHEN arrival_time_sec IS NULL THEN reason END) OVER trip) AS missing_reason
            FROM resolved WINDOW trip AS (PARTITION BY trip_id ORDER BY seq)
        )
        SELECT * EXCLUDE (missing_reason), CASE WHEN missing_reason IS NOT NULL THEN missing_reason
                       {duration_reason} END AS exclusion
        FROM paired WHERE next_stop_id IS NOT NULL
    """)  # noqa: S608
    try:
        yield
    finally:
        con.execute("DROP TABLE _stop_time_legs")


def _leg_diagnostics(con: duckdb.DuckDBPyConnection) -> dict[str, object]:
    """
    Count template legs before service expansion.

    Each excluded leg has one reason. Missing clocks produce one warning per call.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Connection containing the prepared leg table.

    Returns
    -------
    dict[str, object]
        Candidate and excluded leg counts, including counts by reason.
    """
    counts = con.execute(
        "SELECT exclusion, count(*) FROM _stop_time_legs GROUP BY exclusion ORDER BY exclusion"
    ).fetchall()
    reasons = {_STOP_TIME_REASONS[reason]: n for reason, n in counts if reason is not None}
    missing = {key: n for key, n in reasons.items() if key != "nonpositive_duration"}
    if missing:
        details = ", ".join(f"{key}={n}" for key, n in missing.items())
        logger.warning(
            "Excluded %d legs with missing stop times (%s).", sum(missing.values()), details
        )
    return {
        "candidate_legs": sum(n for _, n in counts),
        "excluded_legs": sum(reasons.values()),
        "excluded_by_reason": reasons,
    }
