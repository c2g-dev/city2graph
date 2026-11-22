"""Agreement between NetworkX and sparse-matrix network distance backends."""

from __future__ import annotations

from contextlib import ExitStack
from fractions import Fraction
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pytest

from benchmarks import network_distance
from benchmarks.common import measure

if TYPE_CHECKING:
    from collections.abc import Callable


def _directed() -> nx.DiGraph:
    """One-way arcs, so reverse distances differ or are unreachable."""
    graph = nx.DiGraph()
    graph.add_weighted_edges_from([(0, 1, 1.0), (1, 2, 2.0), (2, 0, 1.0), (2, 3, 0.5)], "cost")
    graph.add_node(4)
    return graph


def _zero_weight() -> nx.Graph:
    """Zero-weight edges, including a zero-weight cycle and equal-length routes."""
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [(0, 1, 0.0), (1, 2, 0.0), (2, 0, 0.0), (2, 3, 1.0), (0, 4, 1.0), (4, 3, 0.0), (3, 5, 0.0)],
        "cost",
    )
    return graph


def _parallel() -> nx.MultiDiGraph:
    """Parallel arcs, where NetworkX uses the smallest weight."""
    graph = nx.MultiDiGraph()
    graph.add_weighted_edges_from(
        [(0, 1, 3.0), (0, 1, 1.0), (1, 2, 2.0), (1, 2, 2.0), (0, 2, 4.0), (2, 0, 1.0)], "cost"
    )
    return graph


def _undirected_parallel() -> nx.MultiGraph:
    """Undirected parallel edges and a self-loop."""
    graph = nx.MultiGraph()
    graph.add_weighted_edges_from(
        [(0, 1, 2.0), (1, 0, 1.0), (1, 1, 0.0), (1, 2, 1.0), (2, 3, 1.5), (0, 3, 3.5)], "cost"
    )
    return graph


def _unreachable() -> nx.Graph:
    """Two components and an isolated node."""
    graph = nx.Graph()
    graph.add_weighted_edges_from([(0, 1, 1.0), (1, 2, 1.0), (3, 4, 1.0)], "cost")
    graph.add_node(5)
    return graph


def _impassable() -> nx.Graph:
    """Return a path whose infinite-weight edge NetworkX traverses with an infinite length."""
    graph = nx.Graph()
    graph.add_weighted_edges_from([(0, 1, 1.0), (1, 2, np.inf), (2, 3, 1.0)], "cost")
    return graph


def _ties() -> nx.Graph:
    """Return a unit grid with many shortest paths of equal length."""
    graph = nx.convert_node_labels_to_integers(nx.grid_2d_graph(5, 5))
    nx.set_edge_attributes(graph, 1, "cost")
    return graph


GRAPHS: dict[str, Callable[[], nx.Graph]] = {
    "directed": _directed,
    "zero-weight": _zero_weight,
    "parallel": _parallel,
    "undirected-parallel": _undirected_parallel,
    "unreachable": _unreachable,
    "impassable": _impassable,
    "ties": _ties,
    "synthetic": lambda: network_distance.synthetic_network(8, 3),
}


@pytest.mark.parametrize("cutoff", [None, 0.0, 1.0, 2.5, 3.0])
@pytest.mark.parametrize("name", GRAPHS)
def test_backends_agree(name: str, cutoff: float | None) -> None:
    """Distances, reachability and selected paths match NetworkX, cutoffs included."""
    graph = GRAPHS[name]()
    nodes = list(graph)
    expected = network_distance.network_distances(
        "networkx", graph, "cost", nodes, nodes, cutoff, paths=len(nodes)
    )
    actual = network_distance.network_distances(
        "sparse", graph, "cost", nodes, nodes, cutoff, paths=len(nodes)
    )
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1] == expected[1]


def test_distances_at_cutoff_are_reachable() -> None:
    """A node exactly at the cutoff is reached and the next node is not."""
    graph = nx.path_graph(4, create_using=nx.DiGraph)
    nx.set_edge_attributes(graph, 1.5, "cost")
    distances, paths = network_distance.network_distances(
        "sparse", graph, "cost", [0], [0, 1, 2, 3], 3.0, paths=4
    )
    np.testing.assert_array_equal(distances, [[0.0, 1.5, 3.0, np.inf]])
    assert paths == [[[0], [0, 1], [0, 1, 2]]]


@pytest.mark.parametrize(
    "weights",
    [
        pytest.param([Fraction(1, 3), Fraction(1, 3)], id="rational"),
        pytest.param([np.float32(0.1), np.float32(0.2)], id="float32"),
        pytest.param([1, 0.5], id="mixed-integer-and-float"),
    ],
)
def test_unsupported_weights_keep_networkx(weights: list[object]) -> None:
    """Weights that NetworkX adds differently from float64 are not converted."""
    graph = nx.path_graph(3)
    nx.set_edge_attributes(graph, dict(zip(graph.edges, weights, strict=True)), "cost")
    with pytest.raises(ValueError, match="sparse matrix"):
        network_distance.network_distances("sparse", graph, "cost", [0], [2], None)


@pytest.mark.parametrize("search", ["full", "cutoff"])
def test_workloads_report_identical_outputs(search: str) -> None:
    """Both backends perform the same measured work on the same inputs."""
    outputs = {}
    inputs = {}
    for backend in ("networkx", "sparse"):
        with ExitStack() as stack:
            workload = network_distance.prepare(f"{backend}-{search}", 5, 42, stack)
            outputs[backend] = [
                measure(workload, phase, repeat=1, warmup=0)["output"]
                for phase in ("time", "allocations", "rss")
            ]
            inputs[backend] = workload.inputs
    assert outputs["networkx"] == outputs["sparse"]
    assert inputs["networkx"] == inputs["sparse"]
    assert outputs["sparse"][0]["origins"] == 5
