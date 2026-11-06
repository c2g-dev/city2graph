"""Measure spatial graph scalability through public APIs on deterministic inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from benchmarks.common import Workload
from benchmarks.common import digest
from benchmarks.common import run_suite

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import ExitStack

# Imports occur after the worker selects the requested package checkout.
# ruff: noqa: PLC0415, ANN401

CASES = {
    "weighted": (1000, 2000, 4000),
    "paths": (100, 200, 400),
    "network": (500, 1000, 2000),
    "waxman": (1000, 2000, 4000),
    "mst": (100, 200, 400),
    "gabriel": (1000, 2000, 4000),
    "rng": (1000, 2000, 4000),
}


def prepare(case: str, size: int, seed: int, _stack: ExitStack) -> Workload:
    """Prepare one graph workload outside the measured operation."""
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import shapely

    import city2graph as c2g

    def points(count: int) -> gpd.GeoDataFrame:
        coordinates = np.random.default_rng(seed).random((count, 2))
        return gpd.GeoDataFrame(geometry=gpd.points_from_xy(*coordinates.T), crs=3857)

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
            {"beta": 0.00001, "r0": 1, "seed": seed}
            if case == "waxman"
            else ({"distance_metric": "manhattan"} if case == "mst" else {})
        )

        def operation() -> Any:
            return functions[case](nodes, **kwargs)

    def describe(result: Any) -> dict[str, int]:
        output_edges = result[1]
        if isinstance(output_edges, dict):
            output_edges = list(output_edges.values())[-1]
        return {"edges": len(output_edges)}

    specification = {"generator": "spatial-v1", "case": case, "size": size, "seed": seed}
    return Workload(operation, describe, {"synthetic": digest(specification)})


if __name__ == "__main__":
    run_suite("scalability", CASES, prepare)
