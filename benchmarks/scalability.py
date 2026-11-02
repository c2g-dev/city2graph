"""Measure public APIs in independent processes; emit JSON lines for comparison."""

from __future__ import annotations

# Workload imports follow --package-root selection; spies mirror external signatures.
# ruff: noqa: PLC0415, ANN401, T201
import argparse
import gc
import json
import platform
import resource
import subprocess
import sys
import time
import tracemalloc
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from collections.abc import Callable
from unittest.mock import patch

CASES = {
    "weighted": (1000, 2000, 4000),
    "paths": (100, 200, 400),
    "network": (500, 1000, 2000),
    "waxman": (1000, 2000, 4000),
    "mst": (100, 200, 400),
    "gabriel": (1000, 2000, 4000),
    "rng": (1000, 2000, 4000),
}


def measure(case: str, size: int) -> None:  # noqa: C901, PLR0915
    """Measure one workload after importing the selected package checkout."""
    import geopandas as gpd
    import networkx as nx
    import numpy as np
    import pandas as pd
    import shapely
    from scipy.sparse import csgraph

    import city2graph as c2g
    from city2graph import proximity

    def points(count: int) -> gpd.GeoDataFrame:
        coordinates = np.random.default_rng(42).random((count, 2))
        return gpd.GeoDataFrame(geometry=gpd.points_from_xy(*coordinates.T), crs=3857)

    details = {
        "max_distance_bytes": 0,
        "largest_join": 0,
        "searches": 0,
        "max_path_references": 0,
        "enriched_edges": 0,
    }
    nodes = points(size)
    if case == "weighted":
        index = pd.MultiIndex.from_arrays([np.arange(size - 1), np.arange(1, size)])
        edges = {
            ("a", "road", "a"): gpd.GeoDataFrame(
                {"cost": np.ones(size - 1)}, geometry=[None] * (size - 1), index=index, crs=3857
            )
        }

        def operation() -> Any:
            # This API adds its result to the supplied edge dictionary.
            # Each timing and allocation run must start from the original chain.
            return c2g.add_metapaths_by_weight(
                nodes={"a": nodes}, edges=dict(edges), weight="cost", threshold=1, endpoint_type="a"
            )
    elif case == "paths":
        sizes = [1, size, 1, size, 1]
        layers = {str(i): points(count) for i, count in enumerate(sizes)}
        edges = {}
        sequence = []
        for i in range(4):
            u = np.repeat(np.arange(sizes[i]), sizes[i + 1])
            v = np.tile(np.arange(sizes[i + 1]), sizes[i])
            key = (str(i), "r", str(i + 1))
            sequence.append(key)
            edges[key] = gpd.GeoDataFrame(
                geometry=[None] * len(u), index=pd.MultiIndex.from_arrays([u, v]), crs=3857
            )

        def operation() -> Any:
            return c2g.add_metapaths(nodes=layers, edges=edges, sequence=sequence, directed=True)
    elif case == "network":
        starts = np.column_stack([np.arange(size - 1), np.zeros(size - 1)])
        ends = np.column_stack([np.arange(1, size), np.zeros(size - 1)])
        roads = gpd.GeoDataFrame(
            geometry=shapely.linestrings(np.stack([starts, ends], axis=1)), crs=3857
        )
        nodes = gpd.GeoDataFrame(geometry=gpd.points_from_xy(np.arange(20), np.zeros(20)), crs=3857)

        def operation() -> Any:
            return c2g.knn_graph(nodes, k=1, distance_metric="network", network_gdf=roads)
    else:
        functions: dict[str, Callable[..., Any]] = {
            "waxman": c2g.waxman_graph,
            "mst": c2g.euclidean_minimum_spanning_tree,
            "gabriel": c2g.gabriel_graph,
            "rng": c2g.relative_neighborhood_graph,
        }
        kwargs: dict[str, Any] = (
            {"beta": 0.00001, "r0": 1, "seed": 42}
            if case == "waxman"
            else ({"distance_metric": "manhattan"} if case == "mst" else {})
        )

        def operation() -> Any:
            return functions[case](nodes, **kwargs)

    samples = []
    for _ in range(3):
        gc.collect()
        start = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - start)

    original_dijkstra = csgraph.dijkstra
    original_merge = pd.DataFrame.merge
    original_add = proximity.GraphBuilder.add_edges

    def matrix_spy(*args: Any, **kwargs: Any) -> Any:
        result = original_dijkstra(*args, **kwargs)
        details["max_distance_bytes"] = max(details["max_distance_bytes"], result.nbytes)
        return result

    def merge_spy(frame: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        result = original_merge(frame, *args, **kwargs)
        details["largest_join"] = max(details["largest_join"], len(result))
        return result

    def add_spy(builder: proximity.GraphBuilder, edges: Any) -> Any:
        details["enriched_edges"] += len(edges)
        return original_add(builder, edges)

    def search_spy(original: Callable[..., Any], *, full_paths: bool = False) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            details["searches"] += 1
            result = original(*args, **kwargs)
            if full_paths:
                details["max_path_references"] = max(
                    details["max_path_references"], sum(map(len, result[1].values()))
                )
            return result

        return wrapped

    gc.collect()
    with ExitStack() as stack:
        stack.enter_context(patch.object(csgraph, "dijkstra", matrix_spy))
        stack.enter_context(patch.object(pd.DataFrame, "merge", merge_spy))
        stack.enter_context(patch.object(proximity.GraphBuilder, "add_edges", add_spy))
        for name in (
            "single_source_dijkstra_path_length",
            "single_source_dijkstra",
            "dijkstra_predecessor_and_distance",
        ):
            stack.enter_context(
                patch.object(
                    nx,
                    name,
                    search_spy(getattr(nx, name), full_paths=name == "single_source_dijkstra"),
                )
            )
        tracemalloc.start()
        result = operation()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    output_edges = result[1]
    if isinstance(output_edges, dict):
        output_edges = list(output_edges.values())[-1]
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = rss if sys.platform == "darwin" else rss * 1024
    print(
        json.dumps(
            {
                "case": case,
                "size": size,
                "median_seconds": float(np.median(samples)),
                "traced_peak_bytes": peak,
                "process_peak_rss_bytes": rss_bytes,
                "output_edges": len(output_edges),
                **details,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "geopandas": gpd.__version__,
                "package": c2g.__file__,
            }
        ),
        flush=True,
    )


def main() -> None:
    """Run the suite or one case against the current or a baseline checkout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--size", type=int)
    args = parser.parse_args()
    if args.case and args.size:
        sys.path.insert(0, str(args.package_root.resolve()))
        measure(args.case, args.size)
    else:
        selected = {args.case: CASES[args.case]} if args.case else CASES
        for case, sizes in selected.items():
            for size in sizes:
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--package-root",
                        str(args.package_root.resolve()),
                        "--case",
                        case,
                        "--size",
                        str(size),
                    ],
                    check=True,
                )


if __name__ == "__main__":
    main()
