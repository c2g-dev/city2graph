"""Measure OD network preparation and queries across facility categories."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING
from typing import Literal

# Delay package imports until the harness has selected the checkout.
# ruff: noqa: PLC0415
from benchmarks.common import Workload
from benchmarks.common import digest
from benchmarks.common import run_suite

if TYPE_CHECKING:
    from contextlib import ExitStack

    import pandas as pd

SIZES = {"node": 1_000_000, "edge": 50_000, "bounded": 3_000}
CASES = {
    f"{mode}-{phase}": (size,)
    for mode, size in SIZES.items()
    for phase in ("prepare", "connect", "first", "reuse")
}


def prepare(case: str, size: int, seed: int, _stack: ExitStack) -> Workload:
    """Generate reproducible points and a two-way grid with metre coordinates."""
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import shapely

    import city2graph as c2g

    mode, phase = case.split("-")
    connect_to: Literal["node", "edge"] = "edge" if mode == "edge" else "node"
    side = 64
    cells = np.arange(side**2).reshape(side, side)
    x, y = np.meshgrid(np.arange(side), np.arange(side))
    xy = np.column_stack((x.ravel(), y.ravel()))
    u = np.r_[cells[:, :-1].ravel(), cells[:-1].ravel()]
    v = np.r_[cells[:, 1:].ravel(), cells[1:].ravel()]
    nodes = gpd.GeoDataFrame(geometry=shapely.points(xy), crs=3857)
    edges = gpd.GeoDataFrame(
        {"cost": np.ones(len(u))},
        geometry=shapely.linestrings(np.stack([xy[u], xy[v]], axis=1)),
        index=pd.MultiIndex.from_arrays([u, v]),
        crs=3857,
    )
    rng = np.random.default_rng(seed)

    def sample(count: int) -> gpd.GeoSeries:
        road = rng.integers(len(u), size=count)
        fraction = rng.integers(1, 8, size=count)[:, None] / 8
        return gpd.GeoSeries(
            shapely.points(xy[u[road]] + (xy[v[road]] - xy[u[road]]) * fraction), crs=3857
        )

    origins = sample(size)
    first, second = sample(24), sample(24)
    inputs = {"grid": digest({"side": side, "seed": seed, "mode": mode, "origins": size})}

    def prepare_network() -> c2g.PreparedNetwork:
        return c2g.PreparedNetwork((nodes, edges), weight="cost", connect_to=connect_to)

    if phase == "prepare":
        return Workload(
            prepare_network, lambda _result: {"nodes": len(nodes), "edges": len(edges)}, inputs
        )
    network = prepare_network()
    if phase == "connect":
        return Workload(lambda: network.connect(origins), lambda _result: {"origins": size}, inputs)
    connected = network.connect(origins)

    def query(targets: gpd.GeoSeries) -> pd.DataFrame:
        return c2g.network_costs(
            connected,
            targets,
            network,
            k=2 if mode == "bounded" else 1,
            threshold=8 if mode == "bounded" else None,
        )

    if phase == "reuse":
        query(first)
    return Workload(lambda: query(second if phase == "reuse" else first), describe, inputs)


def describe(result: pd.DataFrame) -> dict[str, int]:
    """Include ordered labels and costs in the cross-checkout comparison."""
    import pandas as pd

    hashed = pd.util.hash_pandas_object(result, index=False).to_numpy().tobytes()
    return {"pairs": len(result), "sha256": int.from_bytes(hashlib.sha256(hashed).digest())}


if __name__ == "__main__":
    run_suite("routing", CASES, prepare)
