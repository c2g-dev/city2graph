"""Exercise the public network routing APIs."""

from typing import Any
from typing import Literal
from unittest.mock import Mock

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csgraph
from shapely.geometry import LineString
from shapely.geometry import Point

from city2graph import PreparedNetwork
from city2graph import network_cost_matrix
from city2graph import network_costs


def points(*coords: tuple[float, float], labels: list[Any] | None = None) -> gpd.GeoSeries:
    """Build projected point inputs."""
    return gpd.GeoSeries([Point(xy) for xy in coords], index=labels, crs=3857)


@pytest.fixture
def streets() -> nx.MultiDiGraph:
    """Build an OSMnx-like graph with asymmetric parallel arcs and an isolated node."""
    graph = nx.MultiDiGraph(crs="EPSG:3857")
    for node, x in enumerate([0, 10, 20, 50]):
        graph.add_node(node, x=x, y=0)
    graph.add_edge(0, 1, time=10)
    graph.add_edge(0, 1, time=20)
    graph.add_edge(1, 0, time=30)
    graph.add_edge(1, 2, time=0)
    return graph


def test_zero_unreachable_and_labels(streets: nx.MultiDiGraph) -> None:
    """Explicit zeros survive CSR construction and duplicate labels stay distinct."""
    origins = points((10, 0), (50, 0), labels=["x", "x"])
    destinations = points((20, 0), (10, 0), labels=[9, 3])
    matrix = network_cost_matrix(origins, destinations, streets, weight="time", threshold=0)
    assert matrix.shape == (2, 2)
    assert matrix.indptr.tolist() == [0, 2, 2]
    assert matrix.data.tolist() == [0, 0]
    table = network_costs(origins, destinations, streets, weight="time", k=1, threshold=0)
    assert table.to_numpy().tolist() == [["x", 9, 0]]


def test_edge_cost_and_access(streets: nx.MultiDiGraph) -> None:
    """Reverse and forward partial traversals have different costs on the same line."""
    road = PreparedNetwork(streets, weight="time", connect_to="edge")
    origins = road.connect(points((2, 1), (8, 3), (40, 40)), max_connection_distance=3)
    table = network_costs(origins, points((5, 2)), road, k=1, include_connection_distance=True)
    np.testing.assert_allclose(table.cost, [3, 9])
    np.testing.assert_allclose(table.connection_distance, [3, 5])
    assert network_costs(origins, points((0, 0)), road, threshold=6.1).origin.tolist() == [0]


def test_within_and_one_way(streets: nx.MultiDiGraph) -> None:
    """Self exclusion is positional, and reaching the same vertex can cost zero."""
    origins = points((10, 0), (10, 0), (20, 0), (50, 0), labels=[8, 8, 4, 6])
    table = network_costs(origins, graph=streets, k=1, weight="time", threshold=0)
    assert table.to_numpy().tolist() == [[8, 8, 0], [8, 8, 0]]
    reverse = network_costs(points((18, 1)), points((12, 1)), streets, connect_to="edge")
    assert reverse.empty


def test_edge_frame_geometry_length() -> None:
    """An edge frame defaults to two-way travel and uses curved geometry length."""
    edges = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (0, 4), (3, 4)])], crs=3857)
    table = network_costs(points((3, 4)), points((0, 0)), edges)
    assert table.cost.tolist() == [7]


def test_heterogeneous_entry_layers() -> None:
    """Repeated node IDs in different layers stay distinct and transit remains traversable."""
    street = gpd.GeoDataFrame(geometry=points((0, 0), (10, 0)))
    stop = gpd.GeoDataFrame(geometry=points((1, 0), (9, 0)))

    def links(
        source: gpd.GeoDataFrame,
        target: gpd.GeoDataFrame,
        pairs: list[tuple[int, int]],
        times: list[int],
    ) -> gpd.GeoDataFrame:
        lines = [LineString([source.geometry.iloc[u], target.geometry.iloc[v]]) for u, v in pairs]
        return gpd.GeoDataFrame(
            {"time": times},
            geometry=lines,
            crs=3857,
            index=pd.MultiIndex.from_tuples(pairs, names=["u", "v"]),
        )

    edges = {
        ("street", "road", "street"): links(street, street, [(0, 1)], [15]),
        ("street", "board", "stop"): links(street, stop, [(0, 0)], [2]),
        ("stop", "ride", "stop"): links(stop, stop, [(0, 1)], [3]),
        ("stop", "alight", "street"): links(stop, street, [(1, 1)], [2]),
    }
    graph = ({"street": street, "stop": stop}, edges)
    table = network_costs(
        points((1, 0)),
        points((9, 0)),
        graph,
        weight="time",
        directed=True,
        connect_node_types="street",
        include_connection_distance=True,
    )
    assert table.cost.tolist() == [7]
    assert table.connection_distance.tolist() == [2]


@pytest.mark.parametrize("value", [-1, float("nan"), "2", None])
def test_invalid_cost(streets: nx.MultiDiGraph, value: object) -> None:
    """Validate every edge, even when another parallel edge is cheaper."""
    streets[0][1][1]["time"] = value
    with pytest.raises(ValueError, match="non-negative"):
        PreparedNetwork(streets, weight="time")


@pytest.mark.parametrize("reverse", [False, True])
def test_batched_costs_match_networkx(streets: nx.MultiDiGraph, reverse: bool) -> None:
    """Check both search directions and top-k pruning against NetworkX."""
    starts, ends = ([0, 1, 2, 3], [0, 2]) if reverse else ([0, 2], [0, 1, 2, 3])
    coords = [(0, 0), (10, 0), (20, 0), (50, 0)]
    origins = points(*(coords[i] for i in starts))
    destinations = points(*(coords[j] for j in ends))
    table = network_costs(origins, destinations, streets, weight="time", threshold=30, k=2)
    expected: list[tuple[int, int, float]] = []
    for row, source in enumerate(starts):
        lengths = nx.single_source_dijkstra_path_length(streets, source, cutoff=30, weight="time")
        ranked = sorted((lengths[node], col) for col, node in enumerate(ends) if node in lengths)
        expected.extend((row, col, cost) for cost, col in ranked[:2])
    assert list(table.itertuples(index=False, name=None)) == expected


@pytest.mark.parametrize("k", [None, 1, 2])
@pytest.mark.parametrize("within", [False, True])
@pytest.mark.parametrize("limit", [None, 6, 0])
def test_projected_routes_against_split_network(
    k: int | None, within: bool, limit: float | None
) -> None:
    """Compare projected costs to an independently subdivided directed multigraph."""
    graph = nx.MultiDiGraph(crs="EPSG:3857")
    xy = {0: (0, 0), 1: (8, 0), 2: (8, 8)}
    for node, position in xy.items():
        graph.add_node(node, pos=position)
    graph.add_edge(0, 1, cost=8)
    graph.add_edge(1, 2, cost=0)
    graph.add_edge(2, 0, cost=4)
    # The parallel return arc follows a separate alignment.
    graph.add_edge(1, 0, cost=8, geometry=LineString([(8, 0), (4, -4), (0, 0)]))
    stops = [(0, 0), (2, 0), (6, 0), (8, 0), (8, 2), (8, 6), (8, 8), (4, -4)]
    ref = nx.DiGraph()
    ref.add_weighted_edges_from(
        [
            (0, 1, 2),
            (1, 2, 4),
            (2, 3, 2),
            (3, 4, 0),
            (4, 5, 0),
            (5, 6, 0),
            (6, 0, 4),
            (3, 7, 4),
            (7, 0, 4),
        ]
    )
    src_ids = [2, 5, 7, 1, 6, 2, 0, 4, 3]
    dst_ids = src_ids if within else [5, 1, 7, 4, 2]
    src = points(*(stops[i] for i in src_ids))
    dst = None if within else points(*(stops[i] for i in dst_ids))
    network = PreparedNetwork(graph, weight="cost", connect_to="edge")
    origins = network.connect(src)
    expected: list[tuple[int, int, float]] = []
    for row, node in enumerate(src_ids):
        distances = nx.single_source_dijkstra_path_length(ref, node, cutoff=limit)
        ranked = sorted(
            (distances[v], col)
            for col, v in enumerate(dst_ids)
            if v in distances and not (within and k is not None and row == col)
        )
        expected.extend((row, col, cost) for cost, col in ranked[:k])
    table = network_costs(origins, dst, network, threshold=limit, k=k)
    assert list(table.itertuples(index=False, name=None)) == expected
    matrix = network_cost_matrix(origins, dst, network, threshold=limit, k=k).tocoo()
    assert matrix.shape == (len(src_ids), len(dst_ids))
    assert sorted(zip(matrix.row, matrix.col, matrix.data, strict=True)) == sorted(expected)


@pytest.mark.parametrize("connect_to", ["node", "edge"])
def test_empty_network_and_disconnected_points(connect_to: Literal["node", "edge"]) -> None:
    """Missing points, isolated nodes and empty inputs preserve matrix positions."""
    origins = gpd.GeoSeries([None, Point(0, 0)], crs=3857)
    graph = nx.Graph(crs="EPSG:3857")
    matrix = network_cost_matrix(origins, graph=graph, connect_to=connect_to)
    assert matrix.shape == (2, 2)
    assert matrix.nnz == 0
    graph.add_node(0, pos=(0, 0))
    matrix = network_cost_matrix(origins, graph=graph, connect_to=connect_to)
    assert matrix.nnz == int(connect_to == "node")
    empty = network_cost_matrix(origins.iloc[:0], origins, graph)
    assert empty.shape == (0, 2)


def test_cached_origins_and_reverse_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two destination categories share origin projections and use one reverse search each."""
    graph = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (16, 0)])], crs=3857)
    network = PreparedNetwork(graph, connect_to="edge")
    origins = network.connect(points((4, 1), (10, 3)))
    destinations = [network.connect(points((6, 0))), network.connect(points((8, 0)))]
    search = Mock(wraps=csgraph.dijkstra)
    monkeypatch.setattr(csgraph, "dijkstra", search)
    monkeypatch.setattr(network, "connect", Mock(side_effect=AssertionError("Repeated projection")))
    for dest, expected in zip(destinations, [[2, 4], [4, 2]], strict=True):
        result = network_costs(origins, dest, network, k=1)
        assert result.cost.tolist() == expected
    assert search.call_count == 2
    assert all(call.kwargs["min_only"] for call in search.call_args_list)
    assert search.call_args_list[0].args[0] is search.call_args_list[1].args[0]


def test_prepared_point_validation(streets: nx.MultiDiGraph) -> None:
    """Validate prepared network ownership, CRS and query limits at the public API."""
    network = PreparedNetwork(streets)
    raw = points((0, 0))
    connected = network.connect(raw)
    with pytest.raises(ValueError, match="different PreparedNetwork"):
        network_costs(connected, graph=PreparedNetwork(streets))
    with pytest.raises(ValueError, match="graph options"):
        network_costs(raw, graph=network, directed=True)
    with pytest.raises(ValueError, match="same CRS"):
        network.connect(raw.to_crs(4326))
    with pytest.raises(ValueError, match="must be points"):
        network.connect(gpd.GeoSeries([LineString([(0, 0), (1, 0)])], crs=3857))
    with pytest.raises(ValueError, match="threshold"):
        network_costs(connected, graph=network, threshold=-1)
    with pytest.raises(ValueError, match="positive integer"):
        network_costs(connected, graph=network, k=0)
    with pytest.raises(ValueError, match="max_connection_distance"):
        network_costs(raw, graph=network, max_connection_distance=-1)
    with pytest.raises(ValueError, match="max_connection_distance"):
        network.connect(raw, max_connection_distance=-1)
    off = network.connect(points((0, 2)))
    assert network_costs(off, raw, network, max_connection_distance=1).empty


@pytest.mark.parametrize("as_tuple", [True, False])
@pytest.mark.parametrize("as_columns", [True, False])
def test_endpoint_identifiers_at_crossing(as_tuple: bool, as_columns: bool) -> None:
    """Named columns and endpoint indices both distinguish coincident crossings."""
    xy = [(0, 0), (2, 0), (2, 0), (4, 0)]
    nodes = gpd.GeoDataFrame(geometry=points(*xy)).set_axis([10, 11, 20, 21])
    edges = gpd.GeoDataFrame(
        {"source_id": [10, 20], "target_id": [11, 21]},
        geometry=[LineString(xy[:2]), LineString(xy[2:])],
        crs=3857,
    )
    if not as_columns:
        edges = edges.set_index(["source_id", "target_id"])
    network = (nodes, edges) if as_tuple else edges
    assert network_costs(points(xy[0]), points(xy[-1]), network).empty


def test_geometric_endpoint_lookup_and_missing_linework() -> None:
    """A node frame supports coordinate lookup and indexed edges without stored lines."""
    source, dest = points((0, 0)), points((4, 0))
    nodes = gpd.GeoDataFrame(geometry=points((0, 0), (4, 0)))
    geometry_only = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (4, 0)])], crs=3857)
    assert network_costs(source, dest, (nodes, geometry_only)).cost.tolist() == [4]
    edges = gpd.GeoDataFrame({"u": [0], "v": [1], "seconds": [2]})
    assert network_costs(source, dest, (nodes, edges), weight="seconds").cost.tolist() == [2]
    with pytest.raises(ValueError, match="non-negative numeric cost"):
        network_costs(source, dest, (nodes, edges), weight="fare")
    with pytest.raises(ValueError, match="missing from its node layer"):
        network_costs(source, dest, (nodes.iloc[:1], edges))
    with pytest.raises(ValueError, match="continuous line"):
        network_costs(source, dest, edges)


def test_reciprocal_self_loop_and_invalid_geometry() -> None:
    """Oppositely oriented loop arcs retain independent partial traversal costs."""
    graph = nx.MultiDiGraph(crs="EPSG:3857")
    line = LineString([(0, 0), (4, 0), (4, 3), (0, 0)])
    graph.add_node(0, pos=(0, 0))
    graph.add_edge(0, 0, geometry=line, time=12)
    graph.add_edge(0, 0, geometry=line.reverse(), time=24)
    locations = points((4, 0), (4, 3))
    trips = network_costs(locations, graph=graph, weight="time", connect_to="edge", k=1)
    assert trips.cost.tolist() == [3, 6]
    bounded = network_costs(
        locations, graph=graph, weight="time", connect_to="edge", k=1, threshold=3
    )
    assert bounded.cost.tolist() == [3]
    graph.add_edge(0, 0, geometry=Point(1, 1), time=12)
    with pytest.raises(ValueError, match="continuous LineString"):
        PreparedNetwork(graph, weight="time", connect_to="edge")


def test_reversed_top_k_across_batches() -> None:
    """A later batch can displace every previously selected destination."""
    graph = nx.path_graph(340)
    nx.set_node_attributes(graph, {i: (i, 0) for i in graph}, "pos")
    origins = points(*[(i, 0) for i in graph])
    destinations = origins.iloc[::-1].iloc[:300]
    matrix = network_cost_matrix(origins, destinations, graph, k=2).tocoo()
    for row in [0, 41, 339]:
        hits = matrix.row == row
        selected = sorted(zip(matrix.data[hits], matrix.col[hits], strict=True))
        assert (
            selected
            == sorted((abs(row - node), col) for col, node in enumerate(destinations.index))[:2]
        )
