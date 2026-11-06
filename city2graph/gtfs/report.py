"""Build GTFS summary tables from persisted effective generation conditions."""

from __future__ import annotations

import json
import math
from datetime import datetime
from typing import TYPE_CHECKING
from typing import Literal

import pandas as pd

from .schema import _MODEL
from .schema import _REGISTRY
from .schema import _seconds

if TYPE_CHECKING:
    import duckdb
    import geopandas as gpd


def _generation_report(
    con: duckdb.DuckDBPyConnection,
    metadata: pd.DataFrame,
    edges: gpd.GeoDataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Inspect effective assumptions, pattern services and shared-section counts.

    The report uses persisted generation records and survives GTFS save/load.
    Edge counts are directed and cover all active service days in the feed.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Generated feed, optionally restored with saved metadata.
    metadata : pd.DataFrame
        Validated persisted generation history.
    edges : gpd.GeoDataFrame
        Directed edge counts over the resolved service calendar.

    Returns
    -------
    dict of str to pandas.DataFrame
        ``assumptions`` lists adopted values and ignored source information;
        ``services`` lists each date and service period with its activity and
        number of departures; ``edges`` gives combined stop-to-stop traversals.
        Frequency-based departures are expected counts unless exact_times is 1.
        Service columns and their order are stable. Non-applicable frequency
        fields are null for fixed departures. Counts and seconds are numeric,
        integer fields are nullable Int64, and IDs, dates and clocks are strings.

    Raises
    ------
    ValueError
        If generation metadata is unavailable or predates effective records.
    """
    effective = metadata[metadata["table_name"].eq("assumptions")]
    assumptions = metadata[
        metadata["table_name"].isin(["assumptions", "operation_source"])
        | metadata["kind"].isin(["estimated", "ignored"])
    ].reset_index(drop=True)
    assumptions["unit"] = assumptions["field"].map(
        lambda field: str(
            _REGISTRY["settings"]
            .get(field, {})
            .get(
                "unit",
                {
                    "speed_kph": "km/h",
                    "headway_secs": "s",
                    "travel_time_secs": "s",
                    "duration_secs": "s",
                }.get(field, ""),
            )
        )
    )
    assumptions["reason"] = assumptions["details"].map(_readable_reason)
    assumptions["evidence"] = assumptions.apply(
        lambda row: (
            str(
                _REGISTRY["settings"]
                .get(row["field"], {})
                .get("evidence", "Unverified modelling assumption")
            )
            if row["kind"] == "default" and row["source"] == _MODEL
            else {
                "user": "Explicit input; not independently verified",
                "source": "Source information; not independently verified",
                "estimated": "Derived estimate; inspect the recorded method",
                "generated": "Generated identifier, selection or metadata",
                "ignored": "Not adopted",
            }.get(row["kind"], "Unverified assumption")
        ),
        axis=1,
    )
    active_dates = set(
        con.sql("SELECT service_id, strftime(active_date, '%Y%m%d') FROM _active_dates").fetchall()
    )
    rows = []
    for scope, frame in effective[effective["record_id"].str.startswith("service:")].groupby(
        "record_id", sort=False
    ):
        row = dict(zip(frame["field"], frame["value"], strict=True))
        if "headway_secs" in row:
            count = (_seconds(row["end_time"]) - _seconds(row["start_time"])) / float(
                row["headway_secs"]
            )
            if float(row["exact_times"]) == 1:
                count = math.ceil(count)
        else:
            count = 1.0
        row["record_id"] = scope
        for date in json.loads(row["service_dates"]):
            daily = {
                **row,
                "date": date,
                "weekday": (
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday",
                )[datetime.strptime(date, "%Y%m%d").weekday()],
            }
            daily["active_days"] = int((row["service_id"], date) in active_dates)
            daily["departures"] = count * daily["active_days"]
            rows.append(daily)
    service_frame = pd.DataFrame(rows)
    trip_columns = con.table("trips").df()[["trip_id", "route_id", "direction_id"]]
    service_frame = service_frame.merge(trip_columns, on="trip_id", how="left")
    service_frame = service_frame.merge(
        con.table("routes").df()[["route_id", "route_type"]], on="route_id", how="left"
    )
    columns = [
        "date",
        "weekday",
        "pattern_id",
        "route_id",
        "direction_id",
        "route_type",
        "service_id",
        "trip_id",
        "departure_time",
        "start_time",
        "end_time",
        "headway_secs",
        "group_headway_secs",
        "exact_times",
        "departures",
        "active_days",
        "service_group_id",
        "group_pattern_count",
        "frequency_scope",
        "service_dates",
        "record_id",
    ]
    service_frame = service_frame.reindex(columns=columns)
    integers = {
        "direction_id",
        "route_type",
        "headway_secs",
        "group_headway_secs",
        "exact_times",
        "active_days",
        "group_pattern_count",
    }
    for field in columns:
        if field in integers or field == "departures":
            service_frame[field] = pd.to_numeric(service_frame[field], errors="raise").astype(
                "Float64" if field == "departures" else "Int64"
            )
        else:
            service_frame[field] = service_frame[field].astype("string")
    summary = _feed_summary(con, effective, service_frame)
    edges = edges.copy()
    for field in ("start_date", "end_date"):
        edges[field] = pd.Series(summary.loc[0, field], index=edges.index, dtype="string")
    return {
        "summary": summary,
        "assumptions": assumptions,
        "services": service_frame,
        "edges": edges,
    }


def _feed_summary(
    con: duckdb.DuckDBPyConnection,
    effective: pd.DataFrame,
    services: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise recorded daily service and effective generation metadata.

    Daily averages include every date between the recorded bounds, including
    closed dates. Speed bounds describe only speeds adopted for untimed legs.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Generated feed connection.
    effective : pandas.DataFrame
        Persisted effective assumptions.
    services : pandas.DataFrame
        Daily service rows.

    Returns
    -------
    pandas.DataFrame
        One overview row with nullable numeric and string columns.
    """
    generation = effective[effective["record_id"].eq("generation")]
    values = dict(zip(generation["field"], generation["value"], strict=True))
    dates = services["date"]
    start, end = dates.min(), dates.max()
    days = (
        datetime.strptime(str(end), "%Y%m%d") - datetime.strptime(str(start), "%Y%m%d")
    ).days + 1
    service_input = effective.loc[
        effective["field"].eq("service_input"), ["record_id", "value"]
    ].drop_duplicates("record_id", keep="last")
    speeds = pd.to_numeric(effective.loc[effective["field"].eq("speed_kph"), "value"])
    total = services["departures"].sum(min_count=1)
    row = {
        "model": values["model"],
        "start_date": start,
        "end_date": end,
        "service_days": days,
        "route_count": services["route_id"].nunique(),
        "pattern_count": services["pattern_id"].nunique(),
        "stop_count": len(con.table("stops").df()),
        "expected_departures": total,
        "mean_daily_departures": total / days,
        "min_speed_kph": speeds.min() if not speeds.empty else pd.NA,
        "max_speed_kph": speeds.max() if not speeds.empty else pd.NA,
        "specified_pattern_count": service_input["value"].eq("specified").sum(),
        "completed_pattern_count": service_input["value"].eq("completed").sum(),
        "excluded_relation_count": values["excluded_relation_count"],
    }
    summary = pd.DataFrame([row])
    floats = {"expected_departures", "mean_daily_departures", "min_speed_kph", "max_speed_kph"}
    for field in summary:
        dtype: Literal["string", "Float64", "Int64"] = (
            "string"
            if field in {"model", "start_date", "end_date"}
            else ("Float64" if field in floats else "Int64")
        )
        summary[field] = summary[field].astype(dtype)
    return summary


def _readable_reason(value: str) -> str:
    """Extract a concise explanation while retaining the full original details.

    Structured estimation details remain available in the original details column.

    Parameters
    ----------
    value : str
        Plain-text or JSON metadata details.

    Returns
    -------
    str
        Recorded reason, estimation method, or an explicit history gap.
    """
    if not value:
        return "Explanation was not recorded by the original generator"
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value
    if isinstance(parsed, dict):
        return str(
            parsed.get("reason") or parsed.get("method") or parsed.get("distance_basis") or value
        )
    return value
