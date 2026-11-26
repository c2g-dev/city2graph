"""Tests for :mod:`city2graph.utils.distance`."""

from __future__ import annotations

from typing import Any
from typing import Literal
from unittest import mock

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csgraph
from shapely.geometry import LineString
from shapely.geometry import Point

from city2graph import network_cost_matrix
from city2graph import network_costs
from city2graph import utils
from tests.city2graph import helpers


def _points(*coords: tuple[float, float], labels: list[Any] | None = None) -> gpd.GeoSeries:
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


@pytest.fixture
def disconnected_streets(sample_crs: str) -> nx.Graph:
    """Two separate one-unit streets, with a centre on each."""
    graph = nx.Graph(crs=sample_crs)
    for node, x in enumerate([0, 1, 10, 11], start=1):
        graph.add_node(node, pos=(x, 0))
    graph.add_edge(1, 2, length=1.0)
    graph.add_edge(3, 4, length=1.0)
    return graph


class TestNetworkCosts:
    """Exercise ``network_costs`` and ``network_cost_matrix``."""

    def test_zero_unreachable_and_labels(self, streets: nx.MultiDiGraph) -> None:
        """Explicit zeros survive CSR construction and duplicate labels stay distinct."""
        origins = _points((10, 0), (50, 0), labels=["x", "x"])
        destinations = _points((20, 0), (10, 0), labels=[9, 3])
        matrix = network_cost_matrix(origins, destinations, streets, weight="time", threshold=0)
        assert matrix.shape == (2, 2)
        assert matrix.indptr.tolist() == [0, 2, 2]
        assert matrix.data.tolist() == [0, 0]
        table = network_costs(origins, destinations, streets, weight="time", k=1, threshold=0)
        assert table.to_numpy().tolist() == [["x", 9, 0]]

    def test_edge_cost_and_access(self, streets: nx.MultiDiGraph) -> None:
        """Reverse and forward partial traversals have different costs on the same line."""
        origins = _points((2, 1), (8, 3), (40, 40))
        table = network_costs(
            origins,
            _points((5, 2)),
            streets,
            weight="time",
            connect_to="edge",
            max_connection_distance=3,
            k=1,
            include_connection_distance=True,
        )
        np.testing.assert_allclose(table.cost, [3, 9])
        np.testing.assert_allclose(table.connection_distance, [3, 5])
        bounded = network_costs(
            origins,
            _points((0, 0)),
            streets,
            weight="time",
            connect_to="edge",
            max_connection_distance=3,
            threshold=6.1,
        )
        assert bounded.origin.tolist() == [0]

    def test_within_and_one_way(self, streets: nx.MultiDiGraph) -> None:
        """Self exclusion is positional, and reaching the same vertex can cost zero."""
        origins = _points((10, 0), (10, 0), (20, 0), (50, 0), labels=[8, 8, 4, 6])
        table = network_costs(origins, graph=streets, k=1, weight="time", threshold=0)
        assert table.to_numpy().tolist() == [[8, 8, 0], [8, 8, 0]]
        reverse = network_costs(_points((18, 1)), _points((12, 1)), streets, connect_to="edge")
        assert reverse.empty

    def test_edge_frame_geometry_length(self) -> None:
        """An edge frame defaults to two-way travel and uses curved geometry length."""
        edges = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (0, 4), (3, 4)])], crs=3857)
        table = network_costs(_points((3, 4)), _points((0, 0)), edges)
        assert table.cost.tolist() == [7]

    def test_heterogeneous_entry_layers(self) -> None:
        """Repeated node IDs in different layers stay distinct and transit remains traversable."""
        street = gpd.GeoDataFrame(geometry=_points((0, 0), (10, 0)))
        stop = gpd.GeoDataFrame(geometry=_points((1, 0), (9, 0)))

        def links(
            source: gpd.GeoDataFrame,
            target: gpd.GeoDataFrame,
            pairs: list[tuple[int, int]],
            times: list[int],
        ) -> gpd.GeoDataFrame:
            lines = [
                LineString([source.geometry.iloc[u], target.geometry.iloc[v]]) for u, v in pairs
            ]
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
            _points((1, 0)),
            _points((9, 0)),
            graph,
            weight="time",
            directed=True,
            connect_node_types="street",
            include_connection_distance=True,
        )
        assert table.cost.tolist() == [7]
        assert table.connection_distance.tolist() == [2]

    @pytest.mark.parametrize("value", [-1, float("nan"), "2", None])
    def test_invalid_cost(self, streets: nx.MultiDiGraph, value: object) -> None:
        """Validate every edge, even when another parallel edge is cheaper."""
        streets[0][1][1]["time"] = value
        with pytest.raises(ValueError, match="non-negative"):
            network_costs(_points((0, 0)), graph=streets, weight="time")

    @pytest.mark.parametrize("reverse", [False, True])
    def test_batched_costs_match_networkx(self, streets: nx.MultiDiGraph, reverse: bool) -> None:
        """Check both search directions and top-k pruning against NetworkX."""
        starts, ends = ([0, 1, 2, 3], [0, 2]) if reverse else ([0, 2], [0, 1, 2, 3])
        coords = [(0, 0), (10, 0), (20, 0), (50, 0)]
        origins = _points(*(coords[i] for i in starts))
        destinations = _points(*(coords[j] for j in ends))
        table = network_costs(origins, destinations, streets, weight="time", threshold=30, k=2)
        expected: list[tuple[int, int, float]] = []
        for row, source in enumerate(starts):
            lengths = nx.single_source_dijkstra_path_length(
                streets, source, cutoff=30, weight="time"
            )
            ranked = sorted(
                (lengths[node], col) for col, node in enumerate(ends) if node in lengths
            )
            expected.extend((row, col, cost) for cost, col in ranked[:2])
        assert list(table.itertuples(index=False, name=None)) == expected

    @pytest.mark.parametrize("k", [None, 1, 2])
    @pytest.mark.parametrize("within", [False, True])
    @pytest.mark.parametrize("limit", [None, 6, 0])
    def test_projected_routes_against_split_network(
        self, k: int | None, within: bool, limit: float | None
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
        src = _points(*(stops[i] for i in src_ids))
        dst = None if within else _points(*(stops[i] for i in dst_ids))
        expected: list[tuple[int, int, float]] = []
        for row, node in enumerate(src_ids):
            distances = nx.single_source_dijkstra_path_length(ref, node, cutoff=limit)
            ranked = sorted(
                (distances[v], col)
                for col, v in enumerate(dst_ids)
                if v in distances and not (within and k is not None and row == col)
            )
            expected.extend((row, col, cost) for cost, col in ranked[:k])
        table = network_costs(
            src, dst, graph, weight="cost", connect_to="edge", threshold=limit, k=k
        )
        assert list(table.itertuples(index=False, name=None)) == expected
        matrix = network_cost_matrix(
            src, dst, graph, weight="cost", connect_to="edge", threshold=limit, k=k
        ).tocoo()
        assert matrix.shape == (len(src_ids), len(dst_ids))
        assert sorted(zip(matrix.row, matrix.col, matrix.data, strict=True)) == sorted(expected)

    @pytest.mark.parametrize("connect_to", ["node", "edge"])
    def test_empty_network_and_disconnected_points(
        self, connect_to: Literal["node", "edge"]
    ) -> None:
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

    def test_nearest_edge_query_uses_single_reverse_search(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A k=1 query against a separate destination set runs one reverse search."""
        graph = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (16, 0)])], crs=3857)
        origins = _points((4, 1), (10, 3))
        destinations = _points((6, 0))
        search = mock.Mock(wraps=csgraph.dijkstra)
        monkeypatch.setattr(csgraph, "dijkstra", search)
        result = network_costs(origins, destinations, graph, connect_to="edge", k=1)
        assert result.cost.tolist() == [2, 4]
        assert search.call_count == 1
        assert search.call_args.kwargs["min_only"] is True

    def test_input_validation(self, streets: nx.MultiDiGraph) -> None:
        """Validate graph, CRS, geometry type, and query limits at the public API."""
        raw = _points((0, 0))
        with pytest.raises(ValueError, match="Supply a graph"):
            network_costs(raw)
        with pytest.raises(ValueError, match="connect_to"):
            network_costs(raw, graph=streets, connect_to="invalid")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="same CRS"):
            network_costs(raw.to_crs(4326), graph=streets)
        with pytest.raises(ValueError, match="must be points"):
            network_costs(gpd.GeoSeries([LineString([(0, 0), (1, 0)])], crs=3857), graph=streets)
        with pytest.raises(ValueError, match="threshold"):
            network_costs(raw, graph=streets, threshold=-1)
        with pytest.raises(ValueError, match="positive integer"):
            network_costs(raw, graph=streets, k=0)
        with pytest.raises(ValueError, match="max_connection_distance"):
            network_costs(raw, graph=streets, max_connection_distance=-1)
        assert network_costs(_points((0, 2)), raw, streets, max_connection_distance=1).empty
        missing_coords = nx.Graph(crs="EPSG:3857")
        missing_coords.add_node(0)
        with pytest.raises(ValueError, match="pos or x and y"):
            network_costs(raw, graph=missing_coords)

    @pytest.mark.parametrize("as_tuple", [True, False])
    @pytest.mark.parametrize("as_columns", [True, False])
    def test_endpoint_identifiers_at_crossing(self, as_tuple: bool, as_columns: bool) -> None:
        """Named columns and endpoint indices both distinguish coincident crossings."""
        xy = [(0, 0), (2, 0), (2, 0), (4, 0)]
        nodes = gpd.GeoDataFrame(geometry=_points(*xy)).set_axis([10, 11, 20, 21])
        edges = gpd.GeoDataFrame(
            {"source_id": [10, 20], "target_id": [11, 21]},
            geometry=[LineString(xy[:2]), LineString(xy[2:])],
            crs=3857,
        )
        if not as_columns:
            edges = edges.set_index(["source_id", "target_id"])
        network = (nodes, edges) if as_tuple else edges
        assert network_costs(_points(xy[0]), _points(xy[-1]), network).empty

    def test_geometric_endpoint_lookup_and_missing_linework(self) -> None:
        """A node frame supports coordinate lookup and indexed edges without stored lines."""
        source, dest = _points((0, 0)), _points((4, 0))
        nodes = gpd.GeoDataFrame(geometry=_points((0, 0), (4, 0)))
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

    def test_reciprocal_self_loop_and_invalid_geometry(self) -> None:
        """Oppositely oriented loop arcs retain independent partial traversal costs."""
        graph = nx.MultiDiGraph(crs="EPSG:3857")
        line = LineString([(0, 0), (4, 0), (4, 3), (0, 0)])
        graph.add_node(0, pos=(0, 0))
        graph.add_edge(0, 0, geometry=line, time=12)
        graph.add_edge(0, 0, geometry=line.reverse(), time=24)
        locations = _points((4, 0), (4, 3))
        trips = network_costs(locations, graph=graph, weight="time", connect_to="edge", k=1)
        assert trips.cost.tolist() == [3, 6]
        bounded = network_costs(
            locations, graph=graph, weight="time", connect_to="edge", k=1, threshold=3
        )
        assert bounded.cost.tolist() == [3]
        graph.add_edge(0, 0, geometry=Point(1, 1), time=12)
        with pytest.raises(ValueError, match="continuous LineString"):
            network_costs(locations, graph=graph, weight="time", connect_to="edge")

    def test_reversed_top_k_across_batches(self) -> None:
        """A later batch can displace every previously selected destination."""
        graph = nx.path_graph(340)
        nx.set_node_attributes(graph, {i: (i, 0) for i in graph}, "pos")
        origins = _points(*[(i, 0) for i in graph])
        destinations = origins.iloc[::-1].iloc[:300]
        matrix = network_cost_matrix(origins, destinations, graph, k=2).tocoo()
        for row in [0, 41, 339]:
            hits = matrix.row == row
            selected = sorted(zip(matrix.data[hits], matrix.col[hits], strict=True))
            assert (
                selected
                == sorted((abs(row - node), col) for col, node in enumerate(destinations.index))[:2]
            )


class TestFilterGraphByDistance:
    """Tests for ``filter_graph_by_distance``."""

    def test_preserves_integer_threshold_above_float_precision(self) -> None:
        """A road just above an exactly representable limit remains outside the subgraph."""
        graph = nx.Graph()
        graph.add_node(0, pos=(0, 0))
        graph.add_node(1, pos=(1, 0))
        graph.add_edge(0, 1, length=2**53 + 1)
        result = utils.filter_graph_by_distance(graph, Point(0, 0), threshold=2**53)
        assert set(result) == {0}

    @pytest.mark.parametrize(
        ("graph_fixture", "as_nx", "center_fixture", "distance", "expect_empty"),
        [
            ("sample_segments_gdf", False, "mg_center_point", 100.0, False),
            ("sample_segments_gdf", False, "mg_center_point", 0.01, True),
            ("sample_nx_graph", True, "sample_nodes_gdf", 1.0, False),
            ("empty_gdf", False, "mg_center_point", 100.0, True),
        ],
    )
    def test_graph_distance_filtering(
        self,
        graph_fixture: str,
        as_nx: bool,
        center_fixture: str,
        distance: float,
        expect_empty: bool,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test filtering graphs by distance from center points."""
        graph = request.getfixturevalue(graph_fixture)
        center_source = request.getfixturevalue(center_fixture)
        center_point = helpers.get_center_point(center_source)

        reachable = utils.filter_graph_by_distance(
            graph,
            center_point=center_point,
            threshold=distance,
            edge_attr="length",
        )

        if as_nx:
            assert isinstance(reachable, nx.Graph)
            assert (reachable.number_of_edges() == 0) == expect_empty
        else:
            helpers.assert_valid_gdf(reachable, expect_empty)

    @pytest.mark.parametrize("as_geoseries", [False, True])
    def test_multi_center_reaches_both_streets(
        self, disconnected_streets: nx.Graph, as_geoseries: bool
    ) -> None:
        """Centres on different streets contribute both connected components."""
        points = [Point(0, 0), Point(10, 0)]
        centers = (
            gpd.GeoSeries(points, crs=disconnected_streets.graph["crs"]) if as_geoseries else points
        )
        graph = utils.filter_graph_by_distance(
            disconnected_streets, center_point=centers, threshold=1
        )
        assert set(graph.nodes) == {1, 2, 3, 4}
        assert {tuple(sorted(pair)) for pair in graph.edges} == {(1, 2), (3, 4)}

    def test_filter_graph_edge_attr_none_falls_back_to_length(self, sample_crs: str) -> None:
        """edge_attr=None should preserve the existing length fallback."""
        G = nx.Graph()
        G.graph["crs"] = sample_crs
        G.add_node(1, pos=(0, 0))
        G.add_node(2, pos=(1, 0))
        G.add_node(3, pos=(0, 1))
        G.add_edge(1, 2, length=1.0)
        G.add_edge(1, 3, length=1.0)

        reachable_default = utils.filter_graph_by_distance(
            G,
            center_point=Point(0, 0),
            threshold=1.0,
            edge_attr=None,
        )
        reachable_length = utils.filter_graph_by_distance(
            G,
            center_point=Point(0, 0),
            threshold=1.0,
            edge_attr="length",
        )

        assert sorted(reachable_default.nodes()) == sorted(reachable_length.nodes())
        assert sorted(tuple(sorted(edge)) for edge in reachable_default.edges()) == sorted(
            tuple(sorted(edge)) for edge in reachable_length.edges()
        )

    def test_filter_graph_gdf_preserves_edge_membership_and_crs(self, sample_crs: str) -> None:
        """GeoDataFrame input should keep the same filtered edge set and CRS."""
        edges = gpd.GeoDataFrame(
            {
                "u": [1, 2],
                "v": [2, 3],
                "length": [1.0, 1.0],
                "geometry": [
                    LineString([(0, 0), (1, 0)]),
                    LineString([(1, 0), (2, 0)]),
                ],
            },
            crs=sample_crs,
        ).set_index(["u", "v"])

        reachable = utils.filter_graph_by_distance(edges, center_point=Point(0, 0), threshold=1.0)

        assert isinstance(reachable, gpd.GeoDataFrame)
        assert list(reachable.index) == [(1, 2)]
        assert reachable.crs == edges.crs

    def test_filter_graph_excludes_disconnected_components(self, sample_crs: str) -> None:
        """Filtering should not leak into disconnected components."""
        G = nx.Graph()
        G.graph["crs"] = sample_crs
        G.add_node(1, pos=(0, 0))
        G.add_node(2, pos=(1, 0))
        G.add_node(3, pos=(10, 10))
        G.add_node(4, pos=(11, 10))
        G.add_edge(1, 2, length=1.0)
        G.add_edge(3, 4, length=1.0)

        reachable = utils.filter_graph_by_distance(G, center_point=Point(0, 0), threshold=2.0)

        assert sorted(reachable.nodes()) == [1, 2]
        assert sorted(tuple(sorted(edge)) for edge in reachable.edges()) == [(1, 2)]

    def test_filter_graph_invalid_center_sequence_member(self, sample_nx_graph: nx.Graph) -> None:
        """Invalid sequence members should raise a clear TypeError."""
        with pytest.raises(TypeError, match="center_point must be a Point"):
            utils.filter_graph_by_distance(
                sample_nx_graph,
                center_point=[Point(0, 0), "bad"],
                threshold=1.0,
            )

    def test_filter_graph_empty_center_sequence_returns_empty(
        self, sample_nx_graph: nx.Graph
    ) -> None:
        """Empty center sequences should preserve empty-graph behavior."""
        reachable = utils.filter_graph_by_distance(
            sample_nx_graph,
            center_point=[],
            threshold=1.0,
        )

        assert isinstance(reachable, nx.Graph)
        assert len(reachable.nodes()) == 0

    def test_filter_graph_no_pos(self, sample_crs: str) -> None:
        """Test filter_graph_by_distance with graph missing 'pos' attribute."""
        G = nx.Graph()
        G.add_node(1)  # No pos
        G.graph = {"crs": sample_crs}

        # Should return empty graph or handle gracefully
        reachable = utils.filter_graph_by_distance(G, center_point=Point(0, 0), threshold=10)
        assert isinstance(reachable, nx.Graph)
        assert len(reachable) == 0
