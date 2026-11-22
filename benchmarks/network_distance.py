"""Compare NetworkX and sparse-matrix network distances from many origins."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from benchmarks.common import Workload
from benchmarks.common import digest
from benchmarks.common import run_suite

if TYPE_CHECKING:
    from collections.abc import Sequence
    from contextlib import ExitStack

    import networkx as nx
    import numpy as np

# Imports occur after the worker selects the requested package checkout.
# ruff: noqa: PLC0415

ORIGINS = (1, 10, 100, 1000)
CASES = {
    f"{backend}-{search}": ORIGINS
    for search in ("full", "cutoff")
    for backend in ("networkx", "sparse")
}
GRID_SIDE = 60
TARGETS = 100
PATHS = 3
CUTOFF = 5.0


def synthetic_network(side: int, seed: int) -> nx.DiGraph:
    """
    Build a seeded directed street grid with independent arc weights.

    Each street has two arcs with weights in [1, 2); one in ten streets keeps a
    single direction.

    Parameters
    ----------
    side : int
        Number of nodes along each side of the grid.
    seed : int
        Random seed for weights and one-way streets.

    Returns
    -------
    networkx.DiGraph
        Grid whose nodes are integers ``x * side + y``.
    """
    import networkx as nx
    import numpy as np

    rng = np.random.default_rng(seed)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(side * side))
    for x in range(side):
        for y in range(side):
            node = x * side + y
            for neighbour in (
                node + side if x + 1 < side else None,
                node + 1 if y + 1 < side else None,
            ):
                if neighbour is None:
                    continue
                forward, backward = rng.uniform(1, 2, 2)
                graph.add_edge(node, neighbour, cost=float(forward))
                if rng.random() >= 0.1:
                    graph.add_edge(neighbour, node, cost=float(backward))
    return graph


def network_distances(
    backend: str,
    graph: nx.Graph,
    weight: str,
    origins: Sequence[Any],
    targets: Sequence[Any],
    cutoff: float | None,
    paths: int = PATHS,
) -> tuple[np.ndarray, list[list[list[Any] | None]]]:
    """
    Compute target distances and nearest-target paths from every origin.

    The sparse backend includes the conversion of ``graph`` to a matrix.

    Parameters
    ----------
    backend : str
        ``"networkx"`` or ``"sparse"``.
    graph : networkx.Graph
        Network with non-negative weights.
    weight : str
        Edge weight attribute.
    origins : sequence
        Search origins.
    targets : sequence
        Destination nodes.
    cutoff : float, optional
        Maximum explored distance.
    paths : int, default 3
        Number of nearest reachable targets whose paths are returned.

    Returns
    -------
    tuple[numpy.ndarray, list]
        Distances shaped ``(len(origins), len(targets))`` and the node paths to
        the nearest targets of each origin.
    """
    import numpy as np

    from city2graph.proximity import NetworkXSearch
    from city2graph.proximity import SparseNetwork

    if backend == "sparse":
        network = SparseNetwork.from_graph(graph, weight)
        if network is None:
            msg = "Weights cannot be searched as a sparse matrix"
            raise ValueError(msg)
        searches: Any = network.searches(origins, targets, cutoff)
    else:
        searches = (
            (search, search.distances(targets))
            for search in (NetworkXSearch(graph, o, weight, cutoff) for o in origins)
        )
    rows = []
    routes = []
    for search, row in searches:
        nearest = np.argsort(row, kind="stable")[:paths]
        nearest = nearest[row[nearest] < np.inf]
        rows.append(row)
        routes.append(search.paths([targets[j] for j in nearest]))
    return np.vstack(rows) if rows else np.empty((0, len(targets))), routes


def prepare(case: str, size: int, seed: int, _stack: ExitStack) -> Workload:
    """Prepare a fixed network, origins and targets outside the measurement."""
    import numpy as np

    # The harness verifies which checkout provides the imported package.
    import city2graph.proximity  # noqa: F401

    backend, search = case.split("-")
    graph = synthetic_network(GRID_SIDE, seed)
    rng = np.random.default_rng(seed + 1)
    origins = rng.integers(0, graph.number_of_nodes(), size).tolist()
    targets = rng.choice(graph.number_of_nodes(), TARGETS, replace=False).tolist()
    cutoff = CUTOFF if search == "cutoff" else None

    def operation() -> tuple[np.ndarray, list[list[list[Any] | None]]]:
        return network_distances(backend, graph, "cost", origins, targets, cutoff)

    def describe(result: tuple[np.ndarray, list[list[list[Any] | None]]]) -> dict[str, int]:
        distances, routes = result
        # Both backends must report identical counts for the same search.
        return {
            "origins": len(distances),
            "reachable": int(np.isfinite(distances).sum()),
            "path_nodes": sum(len(path or ()) for paths in routes for path in paths),
        }

    specification = {
        "generator": "directed-grid-v1",
        "side": GRID_SIDE,
        "targets": TARGETS,
        "paths": PATHS,
        "cutoff": cutoff,
        "origins": size,
        "seed": seed,
    }
    return Workload(operation, describe, {"synthetic": digest(specification)})


if __name__ == "__main__":
    run_suite("network_distance", CASES, prepare)
