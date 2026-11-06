"""Compare reconstructed and approximated service against a fixed real GTFS feed.

Large stop_times tables are filtered in chunks. No network access is required;
optional source alignments can be supplied as a GeoJSON with route_id and
direction_id columns. Raw reference feeds are not copied into the repository.
"""

from __future__ import annotations

# The command reports benchmark results on stdout.
import argparse
import tempfile
import zipfile
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

import city2graph as c2g
from benchmarks.common import emit
from benchmarks.common import file_digest
from benchmarks.common import positive_int
from benchmarks.common import record

if TYPE_CHECKING:
    import duckdb


def read_table(archive: zipfile.ZipFile, name: str) -> pd.DataFrame:
    """Read a small reference table without coercing identifiers."""
    with archive.open(name + ".txt") as stream:
        return pd.read_csv(stream, dtype=str, keep_default_na=False)


def reference_tables(path: Path, date: str, routes: list[str]) -> dict[str, pd.DataFrame]:
    """Select complete active trips and their stops for the chosen routes."""
    with zipfile.ZipFile(path) as archive:
        weekday = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")[
            datetime.strptime(date, "%Y%m%d").weekday()
        ]
        active: set[str] = set()
        if "calendar.txt" in archive.namelist():
            calendar = read_table(archive, "calendar")
            active = set(
                calendar.loc[
                    (calendar.start_date <= date)
                    & (calendar.end_date >= date)
                    & calendar[weekday].eq("1"),
                    "service_id",
                ]
            )
        elif "calendar_dates.txt" not in archive.namelist():
            msg = "Reference feed needs calendar.txt or calendar_dates.txt"
            raise ValueError(msg)
        if "calendar_dates.txt" in archive.namelist():
            exceptions = read_table(archive, "calendar_dates")
            selected = exceptions[exceptions.date.eq(date)]
            active -= set(selected.loc[selected.exception_type.eq("2"), "service_id"])
            active |= set(selected.loc[selected.exception_type.eq("1"), "service_id"])
        trips = read_table(archive, "trips")
        trips = trips[trips.route_id.isin(routes) & trips.service_id.isin(active)].copy()
        if trips.empty:
            msg = "No active reference trips for the selected routes/date"
            raise ValueError(msg)
        identifiers = set(trips.trip_id)
        if "frequencies.txt" in archive.namelist():
            frequencies = read_table(archive, "frequencies")
            if frequencies.trip_id.isin(identifiers).any():
                msg = "Exact reconstruction requires scheduled trips; selected trips use frequencies.txt"
                raise ValueError(msg)
        chunks: list[pd.DataFrame] = []
        with archive.open("stop_times.txt") as stream:
            chunks.extend(
                chunk[chunk.trip_id.isin(identifiers)]
                for chunk in pd.read_csv(
                    stream, dtype=str, keep_default_na=False, chunksize=100_000
                )
            )
        times = pd.concat(chunks, ignore_index=True)
        if times.empty or set(times.trip_id) != identifiers:
            msg = "Every selected trip must have stop_times"
            raise ValueError(msg)
        for field in ("arrival_time", "departure_time"):
            if field not in times or times[field].str.strip().eq("").any():
                msg = f"Exact reconstruction requires complete {field} values"
                raise ValueError(msg)
            times[field] = times[field].map(lambda value: format_time(elapsed(value)))
        times["stop_sequence"] = pd.to_numeric(times.stop_sequence)
        times = times.sort_values(["trip_id", "stop_sequence"])
        stops = read_table(archive, "stops")
        wanted = set(times.stop_id)
        if "parent_station" in stops:
            wanted |= set(stops.loc[stops.stop_id.isin(wanted), "parent_station"]) - {""}
        stops = stops[stops.stop_id.isin(wanted)]
        route_table = read_table(archive, "routes")
        return {
            "agency": read_table(archive, "agency"),
            "routes": route_table[route_table.route_id.isin(routes)],
            "stops": stops,
            "trips": trips,
            "stop_times": times,
            "calendar_dates": pd.DataFrame(
                {"service_id": sorted(set(trips.service_id)), "date": date, "exception_type": "1"}
            ),
        }


def elapsed(value: str) -> int:
    """Convert extended-hour reference times into elapsed seconds."""
    try:
        hours, minutes, seconds = map(int, value.split(":"))
    except ValueError as exc:
        msg = f"Invalid GTFS time: {value!r}"
        raise ValueError(msg) from exc
    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        msg = f"Invalid GTFS time: {value!r}"
        raise ValueError(msg)
    return hours * 3600 + minutes * 60 + seconds


def format_time(seconds: int) -> str:
    """Normalise extended-hour times for exact stop-time comparisons."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def generator_input(tables: dict[str, pd.DataFrame]) -> tuple[c2g.TransportData, pd.DataFrame]:
    """Reconstruct each reference trip with its exact relative offsets."""
    patterns: list[dict[str, object]] = []
    sequences: list[dict[str, object]] = []
    services: list[dict[str, object]] = []
    trips = tables["trips"].set_index("trip_id")
    for trip_id, times in tables["stop_times"].groupby("trip_id", sort=False):
        trip = trips.loc[trip_id]
        pattern_id = str(trip_id)
        first_arrival = elapsed(times.iloc[0].arrival_time)
        patterns.append(
            {
                "pattern_id": pattern_id,
                "route_id": trip.route_id,
                "direction_id": trip.get("direction_id") or "0",
            }
        )
        services.append(
            {
                "pattern_id": pattern_id,
                "trip_id": trip_id,
                "service_id": trip.service_id,
                "departure_time": times.iloc[0].departure_time,
            }
        )
        sequences.extend(
            {
                "pattern_id": pattern_id,
                "stop_id": row["stop_id"],
                "stop_sequence": row["stop_sequence"],
                "arrival_offset_secs": elapsed(row["arrival_time"]) - first_arrival,
                "departure_offset_secs": elapsed(row["departure_time"]) - first_arrival,
            }
            for row in times.to_dict("records")
        )
    return {
        "routes": tables["routes"],
        "stops": tables["stops"],
        "patterns": pd.DataFrame(patterns),
        "stop_sequences": pd.DataFrame(sequences),
    }, pd.DataFrame(services)


def graph_metrics(reference: gpd.GeoDataFrame, candidate: gpd.GeoDataFrame) -> dict[str, object]:
    """Measure matched edges and weighted paths without adding transfers."""
    common = reference.index.intersection(candidate.index)
    reference_graph, candidate_graph = nx.DiGraph(), nx.DiGraph()
    for frame, graph in ((reference, reference_graph), (candidate, candidate_graph)):
        for (origin, destination), row in frame.iterrows():
            graph.add_edge(origin, destination, weight=float(row.travel_time_sec))
    errors = (
        candidate.loc[common, "travel_time_sec"] - reference.loc[common, "travel_time_sec"]
    ).abs()
    frequency_errors = (
        candidate.loc[common, "frequency"] - reference.loc[common, "frequency"]
    ).abs()
    positive = reference.loc[common, "travel_time_sec"] > 0
    path_errors: list[float] = []
    for origin in sorted(set(reference_graph) & set(candidate_graph)):
        before = nx.single_source_dijkstra_path_length(reference_graph, origin)
        after = nx.single_source_dijkstra_path_length(candidate_graph, origin)
        path_errors.extend(
            abs(after[target] - duration)
            for target, duration in before.items()
            if target != origin and target in after
        )
    return {
        "reference_edges": len(reference),
        "candidate_edges": len(candidate),
        "matched_edges": len(common),
        "edge_precision": len(common) / len(candidate) if len(candidate) else None,
        "edge_recall": len(common) / len(reference) if len(reference) else None,
        "travel_time_mae_sec": float(errors.mean()) if len(common) else None,
        "travel_time_mean_relative_error": float(
            (errors[positive] / reference.loc[common, "travel_time_sec"][positive]).mean()
        )
        if positive.any()
        else None,
        "relative_error_edges": int(positive.sum()),
        "relative_error_excluded_edges": int((~positive).sum()),
        "frequency_mae": float(frequency_errors.mean()) if len(common) else None,
        "reference_components": nx.number_weakly_connected_components(reference_graph),
        "candidate_components": nx.number_weakly_connected_components(candidate_graph),
        "matched_reachable_pairs": len(path_errors),
        "shortest_path_mae_sec": float(np.mean(path_errors)) if path_errors else None,
    }


def sensitivity_cases(
    reference: duckdb.DuckDBPyConnection,
    data: c2g.TransportData,
    tables: dict[str, pd.DataFrame],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    """Vary speed and route-direction frequency without calibrating either."""
    sequences = data["stop_sequences"].groupby("pattern_id", sort=False)
    signatures = {
        str(identifier): tuple(frame.sort_values("stop_sequence").stop_id)
        for identifier, frame in sequences
    }
    patterns = data["patterns"].copy()
    patterns["signature"] = patterns.pattern_id.map(signatures)
    patterns = patterns.drop_duplicates(["route_id", "direction_id", "signature"])
    patterns = patterns.drop(columns="signature")
    selected = data["stop_sequences"].pattern_id.isin(patterns.pattern_id)
    conditions = patterns[["pattern_id"]].copy()
    shares = patterns.groupby(["route_id", "direction_id"]).pattern_id.transform("size")
    conditions["pattern_count"] = shares.to_numpy()
    compact = cast(
        "c2g.TransportData",
        {
            **data,
            "patterns": patterns,
            "stop_sequences": data["stop_sequences"].loc[selected],
        },
    )
    reference_edges = c2g.travel_summary_graph(
        reference,
        calendar_start=args.date,
        calendar_end=args.date,
        directed=True,
    )[1]
    results = []
    for factor in args.speed_factors:
        for headway in args.headways:
            services = conditions[["pattern_id"]].copy()
            services["headway_secs"] = headway * conditions.pattern_count
            with c2g.create_gtfs(
                compact,
                services,
                agency=tables["agency"],
                service_date=args.date,
                speed_kph=args.speed_kph * factor,
                dwell_time_secs=args.dwell_secs,
                start_time=args.service_start,
                end_time=args.service_end,
                distance_method="shape" if args.alignments else "geodesic",
            ) as candidate:
                edges = c2g.travel_summary_graph(
                    candidate,
                    calendar_start=args.date,
                    calendar_end=args.date,
                    directed=True,
                )[1]
                results.append(
                    {
                        "speed_factor": factor,
                        "speed_kph": args.speed_kph * factor,
                        "route_direction_headway_secs": headway,
                        "unique_patterns": len(patterns),
                        "mean_leg_time_sec": float(edges.travel_time_sec.mean())
                        if len(edges)
                        else None,
                        "total_section_traversals": float(edges.frequency.sum()),
                        "comparison": graph_metrics(reference_edges, edges),
                    }
                )
    return results


def compare(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate exact reconstruction and a documented speed approximation."""
    tables = reference_tables(args.feed, args.date, args.routes)
    data, services = generator_input(tables)
    approximated = cast("c2g.TransportData", dict(data))
    approximated["stop_sequences"] = data["stop_sequences"].drop(
        columns=["arrival_offset_secs", "departure_offset_secs"]
    )
    if args.alignments:
        alignments = gpd.read_file(args.alignments).to_crs(4326)
        alignments["route_id"] = alignments["route_id"].astype(str)
        alignments["direction_id"] = alignments["direction_id"].astype(str)
        patterns = data["patterns"].merge(
            alignments[["route_id", "direction_id", "geometry"]],
            on=["route_id", "direction_id"],
            how="left",
            validate="many_to_one",
        )
        approximated["patterns"] = gpd.GeoDataFrame(patterns, geometry="geometry", crs=4326)
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        path = Path(directory) / "reference.zip"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, frame in tables.items():
                archive.writestr(name + ".txt", frame.to_csv(index=False))
        reference = stack.enter_context(c2g.load_gtfs(path))
        exact = stack.enter_context(
            c2g.create_gtfs(
                data, services, agency=tables["agency"], calendar_dates=tables["calendar_dates"]
            )
        )
        approximation = stack.enter_context(
            c2g.create_gtfs(
                approximated,
                services,
                agency=tables["agency"],
                calendar_dates=tables["calendar_dates"],
                speed_kph=args.speed_kph,
                dwell_time_secs=args.dwell_secs,
                distance_method="shape" if args.alignments else "geodesic",
            )
        )
        results: dict[str, Any] = {}
        for label, bounds in [
            ("whole_service_day", (None, None)),
            *[(f"{start}-{end}", (start, end)) for start, end in args.window],
        ]:
            graphs = [
                c2g.travel_summary_graph(
                    con,
                    calendar_start=args.date,
                    calendar_end=args.date,
                    start_time=bounds[0],
                    end_time=bounds[1],
                    directed=True,
                )[1]
                for con in (reference, exact, approximation)
            ]
            results[label] = {
                "exact": graph_metrics(graphs[0], graphs[1]),
                "approximation": graph_metrics(graphs[0], graphs[2]),
            }
        sensitivity = (
            sensitivity_cases(reference, approximated, tables, args) if args.sensitivity else []
        )
        before = reference.sql(
            "SELECT trip_id, stop_id, stop_sequence, arrival_time, departure_time "
            "FROM stop_times ORDER BY trip_id, CAST(stop_sequence AS INTEGER)"
        ).df()
        after = exact.sql(
            "SELECT trip_id, stop_id, stop_sequence, arrival_time, departure_time "
            "FROM stop_times ORDER BY trip_id, CAST(stop_sequence AS INTEGER)"
        ).df()
        pd.testing.assert_frame_equal(before, after)
        for window in results.values():
            exact_metrics = window["exact"]
            if exact_metrics["reference_edges"] == 0:
                assert exact_metrics["candidate_edges"] == 0
            else:
                assert exact_metrics["edge_precision"] == 1
                assert exact_metrics["edge_recall"] == 1
                assert exact_metrics["travel_time_mae_sec"] == 0
                assert exact_metrics["frequency_mae"] == 0
        saved = c2g.save_gtfs(exact, Path(directory) / "generated.zip")
        reloaded = c2g.load_gtfs(saved)
        try:
            pd.testing.assert_frame_equal(
                exact.table("stop_times").df(), reloaded.table("stop_times").df()
            )
        finally:
            reloaded.close()
    parameters = {
        "service_date": args.date,
        "routes": sorted(args.routes),
        "speed_kph": args.speed_kph,
        "dwell_secs": args.dwell_secs,
        "distance_method": "shape" if args.alignments else "geodesic",
        "windows": args.window,
        "sensitivity": args.sensitivity,
        "speed_factors": args.speed_factors,
        "headways": args.headways,
        "service_start": args.service_start,
        "service_end": args.service_end,
        "workload_version": 1,
    }
    inputs = {"feed": file_digest(args.feed)}
    if args.alignments:
        inputs["alignments"] = file_digest(args.alignments)
    return record(
        "gtfs_generation",
        "reference",
        parameters,
        inputs,
        {
            "trips": len(tables["trips"]),
            "stop_occurrences": len(before),
            "windows": results,
            "sensitivity": sensitivity,
        },
        {"exact_stop_times": True, "exact_graphs": True, "roundtrip": True},
    )


def main() -> None:
    """Run the explicitly selected, reproducible real-feed comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feed", type=Path, required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--routes", nargs="+", required=True)
    parser.add_argument("--speed-kph", type=float, required=True)
    parser.add_argument("--dwell-secs", type=float, required=True)
    parser.add_argument("--alignments", type=Path)
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument(
        "--window",
        nargs=2,
        action="append",
        metavar=("START", "END"),
        help="Additional comparison window; repeat to compare multiple windows",
    )
    parser.add_argument("--speed-factors", nargs="+", type=float, default=[0.75, 1.0, 1.25])
    parser.add_argument("--headways", nargs="+", type=positive_int, default=[600, 900, 1800])
    parser.add_argument("--service-start", default="06:00:00")
    parser.add_argument("--service-end", default="22:00:00")
    args = parser.parse_args()
    args.window = args.window or []
    if not np.isfinite(args.speed_kph) or args.speed_kph <= 0:
        parser.error("--speed-kph must be finite and positive")
    if not np.isfinite(args.dwell_secs) or args.dwell_secs < 0:
        parser.error("--dwell-secs must be finite and non-negative")
    if any(not np.isfinite(factor) or factor <= 0 for factor in args.speed_factors):
        parser.error("--speed-factors must be finite and positive")
    try:
        for start, end in [*args.window, (args.service_start, args.service_end)]:
            if elapsed(start) >= elapsed(end):
                msg = "Window start must precede end; use extended hours after midnight"
                parser.error(msg)
        emit(compare(args))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
