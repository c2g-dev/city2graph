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
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import Polygon

from city2graph import network_cost_matrix
from city2graph import network_costs
from city2graph import utils
from city2graph.utils import distance as distance_utils
from city2graph.utils import gdf_to_nx
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


@pytest.fixture
def isochrone_ring_graph() -> nx.Graph:
    """Build a square road circuit whose buffer encloses a hole."""
    graph = nx.cycle_graph(4)
    graph.graph["crs"] = "EPSG:27700"
    positions = [(0, 0), (2000, 0), (2000, 2000), (0, 2000)]
    nx.set_node_attributes(graph, dict(enumerate(positions)), "pos")
    for source, target in graph.edges:
        graph.edges[source, target].update(
            length=1.0,
            geometry=LineString([positions[source], positions[target]]),
        )
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


class TestCreateIsochrone:
    """Tests for ``create_isochrone``."""

    @pytest.mark.parametrize("parts", [1, 2])
    @pytest.mark.parametrize("island", [False, True])
    def test_isochrone_fill_holes_preserves_separate_parts(
        self, isochrone_ring_graph: nx.Graph, parts: int, island: bool
    ) -> None:
        """Fill each ring without connecting distant parts or leaving overlaps."""
        graph = isochrone_ring_graph
        centres = [Point(0, 0)]
        if parts == 2:
            second = nx.relabel_nodes(graph, lambda node: node + 4)
            for data in second.nodes.values():
                x, y = data["pos"]
                data["pos"] = (x + 5000, y)
            for _source, _target, data in second.edges(data=True):
                data["geometry"] = LineString([(x + 5000, y) for x, y in data["geometry"].coords])
            graph = nx.compose(graph, second)
            centres.append(Point(5000, 0))
        if island:
            graph.add_node(99, pos=(1000, 1000))
            centres.append(Point(1000, 1000))

        original = utils.create_isochrone(
            graph, center_point=centres, threshold=2, edge_attr="length", method="buffer"
        )
        result = utils.create_isochrone(
            graph,
            center_point=centres,
            threshold=2,
            edge_attr="length",
            method="buffer",
            fill_holes=True,
        )
        before, after = original.geometry.iloc[0], result.geometry.iloc[0]
        before_parts = list(before.geoms) if isinstance(before, MultiPolygon) else [before]
        after_parts = list(after.geoms) if isinstance(after, MultiPolygon) else [after]
        assert sum(len(part.interiors) for part in before_parts) == parts
        assert len(after_parts) == parts
        assert all(not part.interiors for part in after_parts)
        assert after.is_valid
        assert after.covers(before)
        assert after.area > before.area
        assert after.bounds == before.bounds
        assert list(result.columns) == ["geometry"]
        assert result.crs == original.crs
        if parts == 2:
            assert not after.intersects(Point(3500, 1000))

    def test_isochrone_fill_holes_after_component_union(self) -> None:
        """A hole formed by separate road buffers must also be removed."""
        graph = nx.Graph(crs="EPSG:27700")
        segments = [
            [(50, 0), (1950, 0)],
            [(2000, 50), (2000, 1950)],
            [(1950, 2000), (50, 2000)],
            [(0, 1950), (0, 50)],
        ]
        for index, coords in enumerate(segments):
            graph.add_node(2 * index, pos=coords[0])
            graph.add_node(2 * index + 1, pos=coords[1])
            graph.add_edge(2 * index, 2 * index + 1, length=1, geometry=LineString(coords))
        centres = [Point(coords[0]) for coords in segments]
        original = utils.create_isochrone(
            graph, center_point=centres, threshold=1, edge_attr="length", method="buffer"
        )
        result = utils.create_isochrone(
            graph,
            center_point=centres,
            threshold=1,
            edge_attr="length",
            method="buffer",
            fill_holes=True,
        )
        before, after = original.geometry.iloc[0], result.geometry.iloc[0]
        assert len(before.interiors) == 1
        assert not after.interiors
        assert after.is_valid
        assert after.covers(before)
        assert after.contains(Point(1000, 1000))

    @pytest.mark.parametrize("threshold", [2, [2, 0, 1, 2]])
    def test_isochrone_fill_holes_defaults_preserve_legacy(
        self, isochrone_ring_graph: nx.Graph, threshold: float | list[float]
    ) -> None:
        """Omitted and disabled filling preserve the original ring footprint."""
        options: dict[str, Any] = {
            "center_point": Point(0, 0),
            "threshold": threshold,
            "edge_attr": "length",
            "method": "buffer",
            "join_style": 2,
            "cap_style": 3,
        }
        original = utils.create_isochrone(isochrone_ring_graph, **options)
        disabled = utils.create_isochrone(isochrone_ring_graph, fill_holes=False, **options)
        assert original.geometry.to_wkb().tolist() == disabled.geometry.to_wkb().tolist()
        expected = Polygon(
            [(-100, -100), (2100, -100), (2100, 2100), (-100, 2100)],
            holes=[[(100, 100), (1900, 100), (1900, 1900), (100, 1900)]],
        )
        assert original.geometry.iloc[0].equals(expected)
        if isinstance(threshold, list):
            assert original["threshold"].tolist() == threshold
            for value, geometry in zip(threshold, original.geometry, strict=True):
                scalar = utils.create_isochrone(
                    isochrone_ring_graph, **{**options, "threshold": value}
                )
                assert geometry.wkb == scalar.geometry.iloc[0].wkb

    @pytest.mark.parametrize("as_gdf", [False, True])
    def test_isochrone_fill_holes_layered_inputs(
        self, isochrone_ring_graph: nx.Graph, as_gdf: bool
    ) -> None:
        """Fill layers independently and retain threshold order and duplicates."""
        inputs: dict[str, Any] = {"graph": isochrone_ring_graph}
        if as_gdf:
            nodes, edges = utils.nx_to_gdf(isochrone_ring_graph)
            inputs = {"nodes": nodes, "edges": edges}
        thresholds = [2, 0, 1, 2]
        result = utils.create_isochrone(
            **inputs,
            center_point=Point(0, 0),
            threshold=thresholds,
            edge_attr="length",
            method="buffer",
            fill_holes=True,
        )
        assert result["threshold"].tolist() == thresholds
        assert list(result.columns) == ["threshold", "geometry"]
        assert result.crs == isochrone_ring_graph.graph["crs"]
        for value, geometry in zip(thresholds, result.geometry, strict=True):
            scalar = utils.create_isochrone(
                **inputs,
                center_point=Point(0, 0),
                threshold=value,
                edge_attr="length",
                method="buffer",
                fill_holes=True,
            )
            assert geometry.wkb == scalar.geometry.iloc[0].wkb
            assert geometry.is_valid
            assert not geometry.interiors

    @pytest.mark.parametrize("node_count", [0, 1, 2])
    @pytest.mark.parametrize("buffer_distance", [None, 0, 100])
    def test_isochrone_fill_holes_degenerate_inputs(
        self, node_count: int, buffer_distance: float | None
    ) -> None:
        """Empty and hole-free inputs retain their original geometry and schema."""
        graph = nx.path_graph(node_count)
        graph.graph["crs"] = "EPSG:27700"
        nx.set_node_attributes(graph, {i: (i * 50, 0) for i in graph}, "pos")
        for threshold in (2, [2, 0, 2]):
            options: dict[str, Any] = {
                "center_point": Point(0, 0),
                "threshold": threshold,
                "method": "buffer",
                "buffer_distance": buffer_distance,
            }
            original = utils.create_isochrone(graph, **options)
            result = utils.create_isochrone(graph, fill_holes=True, **options)
            assert list(result.columns) == list(original.columns)
            assert result.crs == original.crs
            assert len(result) == len(original)
            assert result.geometry.is_empty.tolist() == original.geometry.is_empty.tolist()
            assert result.geometry.is_valid.all()
            for before, after in zip(original.geometry, result.geometry, strict=True):
                assert before.equals(after)

    @pytest.mark.parametrize("fill_holes", [None, 0, 1, "True", [], np.bool_(1)])
    @pytest.mark.parametrize("empty", [False, True])
    def test_isochrone_fill_holes_validation(
        self, isochrone_ring_graph: nx.Graph, fill_holes: object, empty: bool
    ) -> None:
        """Reject non-bool values before any empty-graph early return."""
        graph = nx.Graph() if empty else isochrone_ring_graph
        with pytest.raises(ValueError, match="fill_holes must be a bool"):
            utils.create_isochrone(
                graph,
                center_point=Point(0, 0),
                threshold=2,
                method="buffer",
                fill_holes=fill_holes,
            )

    @pytest.mark.parametrize("method", ["convex_hull", "concave_hull_alpha", "concave_hull_knn"])
    def test_isochrone_fill_holes_is_buffer_only(
        self, isochrone_ring_graph: nx.Graph, method: str
    ) -> None:
        """Hull methods continue to ignore buffer-specific keyword arguments."""
        options: dict[str, Any] = {"center_point": Point(0, 0), "threshold": 2, "method": method}
        original = utils.create_isochrone(isochrone_ring_graph, **options)
        result = utils.create_isochrone(isochrone_ring_graph, fill_holes="ignored", **options)
        assert result.geometry.to_wkb().tolist() == original.geometry.to_wkb().tolist()

    @pytest.mark.parametrize("cache_bytes", [0, 64])
    def test_isochrone_neighbor_cache_eviction_preserves_hull(
        self, cache_bytes: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Recomputed neighbour arrays keep equal-distance ordering and geometry."""
        graph = nx.grid_2d_graph(8, 8)
        nx.set_node_attributes(graph, {node: node for node in graph}, "pos")
        graph.graph["crs"] = "EPSG:3857"
        nx.set_edge_attributes(graph, 1.0, "length")
        expected = utils.create_isochrone(graph, center_point=Point(0, 0), threshold=20, k=3)
        monkeypatch.setattr(distance_utils, "_HULL_NEIGHBOR_CACHE_BYTES", cache_bytes)
        actual = utils.create_isochrone(graph, center_point=Point(0, 0), threshold=20, k=3)
        assert actual.geometry.iloc[0].equals_exact(expected.geometry.iloc[0], 0)

    def test_create_isochrone_basic(self, sample_nx_graph: nx.Graph) -> None:
        """Test basic isochrone creation."""
        center = Point(0, 0)
        distance = 2.0

        isochrone = utils.create_isochrone(
            sample_nx_graph,
            center_point=center,
            threshold=distance,
            edge_attr="edge_feature1",  # Use custom attribute
            method="convex_hull",
        )

        assert isinstance(isochrone, gpd.GeoDataFrame)
        assert len(isochrone) == 1
        assert isochrone.geometry.iloc[0].geom_type == "Polygon"

    def test_create_isochrone_buffer(self, sample_nx_graph: nx.Graph) -> None:
        """Test isochrone creation with buffer method."""
        center = Point(0, 0)
        distance = 1.0

        isochrone = utils.create_isochrone(
            sample_nx_graph,
            center_point=center,
            threshold=distance,
            edge_attr="edge_feature1",
            method="buffer",
            buffer_distance=0.1,
        )

        assert isinstance(isochrone, gpd.GeoDataFrame)
        assert len(isochrone) == 1
        assert isochrone.geometry.iloc[0].geom_type in ["Polygon", "MultiPolygon"]

    def test_create_isochrone_multi_threshold_returns_layered_rows(
        self, sample_nx_graph: nx.Graph
    ) -> None:
        """Sequence thresholds should return one row per requested layer."""
        result = utils.create_isochrone(
            sample_nx_graph,
            center_point=Point(0, 0),
            threshold=[2.0, 1.0, 2.0],
            edge_attr="edge_feature1",
            method="convex_hull",
        )

        assert isinstance(result, gpd.GeoDataFrame)
        assert list(result["threshold"]) == [2.0, 1.0, 2.0]
        assert len(result) == 3
        assert result.crs == sample_nx_graph.graph["crs"]
        assert result.geometry.notna().all()

    def test_create_isochrone_scalar_threshold_keeps_legacy_shape(
        self, sample_nx_graph: nx.Graph
    ) -> None:
        """Scalar thresholds should keep the pre-existing single-row schema."""
        result = utils.create_isochrone(
            sample_nx_graph,
            center_point=Point(0, 0),
            threshold=2.0,
            edge_attr="edge_feature1",
            method="convex_hull",
        )

        assert len(result) == 1
        assert list(result.columns) == ["geometry"]

    def test_create_isochrone_multi_threshold_preserves_empty_layers(self) -> None:
        """Multi-threshold mode should keep rows even when no polygon is generated."""
        empty_graph = nx.Graph()
        empty_graph.graph["crs"] = "EPSG:4326"

        result = utils.create_isochrone(
            empty_graph,
            center_point=Point(0, 0),
            threshold=[10.0, 20.0],
            method="convex_hull",
        )

        assert list(result["threshold"]) == [10.0, 20.0]
        assert len(result) == 2
        assert result.geometry.is_empty.tolist() == [True, True]

    def test_create_isochrone_multi_threshold_cut_edge_types(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Cut edge filtering should still apply when layered output is requested."""
        graph = gdf_to_nx(nodes=sample_hetero_nodes_dict, edges=sample_hetero_edges_dict)
        cut_type = ("building", "connects_to", "road")

        iso_full = utils.create_isochrone(
            graph,
            center_point=Point(0, 0),
            threshold=[100.0],
            method="buffer",
            buffer_distance=1.0,
        )
        iso_cut = utils.create_isochrone(
            graph,
            center_point=Point(0, 0),
            threshold=[100.0],
            method="buffer",
            buffer_distance=1.0,
            cut_edge_types=[cut_type],
        )

        assert iso_cut.geometry.iloc[0].area < iso_full.geometry.iloc[0].area

    def test_create_isochrone_multi_threshold_from_gdf_inputs(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Layered isochrones should work with nodes/edges inputs."""
        result = utils.create_isochrone(
            nodes=sample_nodes_gdf,
            edges=sample_edges_gdf,
            center_point=Point(0, 0),
            threshold=[1.0, 2.0],
            method="convex_hull",
        )

        assert list(result["threshold"]) == [1.0, 2.0]
        assert len(result) == 2
        assert result.geometry.notna().all()

    def test_create_isochrone_multi_threshold_multi_center(self, sample_crs: str) -> None:
        """Layered output should still union reachability from multiple centers."""
        graph = nx.Graph()
        graph.graph["crs"] = sample_crs
        graph.add_node(1, pos=(0, 0))
        graph.add_node(2, pos=(1, 0))
        graph.add_node(3, pos=(10, 0))
        graph.add_node(4, pos=(11, 0))
        graph.add_edge(1, 2, length=1.0)
        graph.add_edge(3, 4, length=1.0)

        result = utils.create_isochrone(
            graph,
            center_point=gpd.GeoSeries([Point(0, 0), Point(10, 0)], crs=sample_crs),
            threshold=[1.0, 2.0],
            edge_attr="length",
            method="convex_hull",
        )

        assert list(result["threshold"]) == [1.0, 2.0]
        assert result.geometry.iloc[0].intersects(Point(0, 0))
        assert result.geometry.iloc[0].intersects(Point(10, 0))

    def test_create_isochrone_multi_threshold_computes_distances_once(
        self, sample_nx_graph: nx.Graph
    ) -> None:
        """Layered isochrones should reuse a single distance computation."""
        with mock.patch(
            "city2graph.utils.distance._compute_center_node_distances",
            wraps=distance_utils._compute_center_node_distances,
        ) as mock_compute:
            result = utils.create_isochrone(
                sample_nx_graph,
                center_point=Point(0, 0),
                threshold=[1.0, 2.0, 3.0],
                edge_attr="edge_feature1",
                method="convex_hull",
            )

        assert len(result) == 3
        assert mock_compute.call_count == 1

    @pytest.mark.parametrize("graph_class", [nx.Graph, nx.MultiDiGraph])
    @pytest.mark.parametrize("threshold", [10, [5, 10]])
    def test_isochrone_cuts_remove_isolated_stations_and_preserve_walks(
        self, graph_class: type[nx.Graph], threshold: int | list[int], sample_crs: str
    ) -> None:
        """Transport may enable walking reach without leaving station-only circles."""
        graph = graph_class(crs=sample_crs)
        positions = {0: (0, 0), 1: (100, 0), 2: (1000, 0), 3: (1100, 0), 4: (1200, 0), 5: (2000, 0)}
        for node, position in positions.items():
            graph.add_node(node, pos=position)
        rail = ("station", "ride", "station")
        walk = ("street", "walk", "street")
        if graph.is_multigraph():
            graph.add_edge(
                0, 1, cost=1, edge_type=rail, geometry=LineString([(0, 0), (0, 500), (100, 0)])
            )
        for u, v, edge_type in [
            (0, 1, walk),
            (1, 2, rail),
            (2, 3, rail),
            (3, 4, walk),
            (2, 5, rail),
        ]:
            graph.add_edge(
                u, v, cost=1, edge_type=edge_type, geometry=LineString([positions[u], positions[v]])
            )
        original_edges = graph.number_of_edges()
        result = utils.create_isochrone(
            graph,
            center_point=Point(0, 0),
            threshold=threshold,
            edge_attr="cost",
            cut_edge_types=[rail],
            method="buffer",
            buffer_distance=10,
        )
        for polygon in result.geometry:
            assert polygon.covers(Point(50, 0))
            assert polygon.covers(Point(1150, 0))
            assert not polygon.covers(Point(1000, 0))
            assert not polygon.covers(Point(2000, 0))
            assert not polygon.covers(Point(0, 250))
        assert graph.number_of_edges() == original_edges
        assert len(graph) == len(positions)

    def test_isochrone_with_cut_edge_types(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Test isochrone generation with edge type cutting."""
        graph = gdf_to_nx(nodes=sample_hetero_nodes_dict, edges=sample_hetero_edges_dict)
        center = Point(0, 0)
        distance = 100.0

        # Generate without cutting
        iso_full = utils.create_isochrone(
            graph,
            center_point=center,
            threshold=distance,
            method="buffer",
            buffer_distance=1.0,
        )

        # Generate with cutting
        # Assuming 'building', 'connects_to', 'road' is the edge type in sample
        cut_type = ("building", "connects_to", "road")
        iso_cut = utils.create_isochrone(
            graph,
            center_point=center,
            threshold=distance,
            method="buffer",
            buffer_distance=1.0,
            cut_edge_types=[cut_type],
        )

        assert not iso_full.empty
        assert not iso_cut.empty
        assert iso_cut.geometry.iloc[0].area < iso_full.geometry.iloc[0].area

    def test_isochrone_disconnected_components(self, sample_crs: str) -> None:
        """Test that cutting edges results in multiple polygons if graph becomes disconnected."""
        # Create two clusters connected by a single edge
        # Cluster 1
        c1_nodes = [Point(0, 0), Point(1, 0), Point(0, 1)]
        c1_ids = [1, 2, 3]

        # Cluster 2 (far away)
        c2_nodes = [Point(100, 100), Point(101, 100), Point(100, 101)]
        c2_ids = [11, 12, 13]

        nodes_gdf = gpd.GeoDataFrame(
            {"geometry": c1_nodes + c2_nodes},
            index=pd.Index(c1_ids + c2_ids, name="node_id"),
            crs=sample_crs,
        )

        # Edges
        edges_data = {
            "u": [1, 1, 11, 11, 3],
            "v": [2, 3, 12, 13, 11],  # 3->11 is the long connector
            "length": [1, 1, 1, 1, 10],
            "edge_type": [
                ("street", "street", "street"),
                ("street", "street", "street"),
                ("street", "street", "street"),
                ("street", "street", "street"),
                ("transit", "transit", "transit"),
            ],
            "geometry": [
                LineString([(0, 0), (1, 0)]),
                LineString([(0, 0), (0, 1)]),
                LineString([(100, 100), (101, 100)]),
                LineString([(100, 100), (100, 101)]),
                LineString([(0, 1), (100, 100)]),
            ],
        }
        edges_gdf = gpd.GeoDataFrame(edges_data, crs=sample_crs).set_index(["u", "v"])

        graph = gdf_to_nx(nodes=nodes_gdf, edges=edges_gdf)

        # Generate isochrone with cutting "transit"
        # Distance 20 is enough to reach cluster 2
        iso = utils.create_isochrone(
            graph,
            center_point=Point(0, 0),
            threshold=20,
            edge_attr="length",
            method="convex_hull",
            cut_edge_types=[("transit", "transit", "transit")],
        )

        # Should be MultiPolygon (or at least disjoint)
        geom = iso.geometry.iloc[0]
        assert geom.geom_type == "MultiPolygon"
        # Area should be small (approx 2 small triangles), not covering the gap
        assert geom.area < 100.0

    def test_isochrone_center_gdf(self, sample_nx_graph: nx.Graph) -> None:
        """Test isochrone creation with GeoDataFrame as center_point."""
        # Create a GeoDataFrame for center point
        center_gdf = gpd.GeoDataFrame(
            {"geometry": [Point(0, 0)]},
            index=[999],  # Mismatched index to ensure robustness
            crs=sample_nx_graph.graph["crs"],
        )

        isochrone = utils.create_isochrone(
            sample_nx_graph,
            center_point=center_gdf,
            threshold=2.0,
            edge_attr="edge_feature1",
            method="convex_hull",
        )

        assert isinstance(isochrone, gpd.GeoDataFrame)
        assert len(isochrone) == 1
        assert not isochrone.empty

    def test_create_isochrone_alpha_shape(self, sample_nx_graph: nx.Graph) -> None:
        """Test isochrone creation with alpha_shape method."""
        center = Point(0, 0)
        distance = 2.0

        isochrone = utils.create_isochrone(
            sample_nx_graph,
            center_point=center,
            threshold=distance,
            edge_attr="edge_feature1",
            method="concave_hull_alpha",
            hull_ratio=0.5,
        )

        assert isinstance(isochrone, gpd.GeoDataFrame)
        assert len(isochrone) == 1
        assert isochrone.geometry.iloc[0].geom_type in ["Polygon", "MultiPolygon"]

    def test_create_isochrone_default_method(self, sample_nx_graph: nx.Graph) -> None:
        """Test isochrone creation with default method (concave_hull_knn)."""
        center = Point(0, 0)
        distance = 2.0

        isochrone = utils.create_isochrone(
            sample_nx_graph,
            center_point=center,
            threshold=distance,
            edge_attr="edge_feature1",
        )

        assert isinstance(isochrone, gpd.GeoDataFrame)
        assert len(isochrone) == 1
        assert isochrone.geometry.iloc[0].geom_type in ["Polygon", "MultiPolygon"]

    def test_create_isochrone_concave_hull_knn_params(self, sample_nx_graph: nx.Graph) -> None:
        """Test concave_hull_knn with different k values."""
        center = Point(0, 0)
        distance = 2.0

        # k=1 (should work, might fallback or minimal hull)
        iso_k1 = utils.create_isochrone(
            sample_nx_graph,
            center_point=center,
            threshold=distance,
            method="concave_hull_knn",
            k=1,
        )
        assert not iso_k1.empty

        # k larger than n_points should clamp and still produce a polygonal hull.
        iso_k_large = utils.create_isochrone(
            sample_nx_graph,
            center_point=center,
            threshold=distance,
            method="concave_hull_knn",
            k=1000,
        )
        assert not iso_k_large.empty
        assert iso_k_large.geometry.iloc[0].geom_type == "Polygon"

    def test_create_isochrone_concave_hull_alpha_params(self, sample_nx_graph: nx.Graph) -> None:
        """Test concave_hull_alpha with different parameters."""
        center = Point(0, 0)
        distance = 2.0

        # High ratio (tighter fit)
        iso_tight = utils.create_isochrone(
            sample_nx_graph,
            center_point=center,
            threshold=distance,
            method="concave_hull_alpha",
            hull_ratio=0.9,
        )
        assert not iso_tight.empty

        # Allow holes
        iso_holes = utils.create_isochrone(
            sample_nx_graph,
            center_point=center,
            threshold=distance,
            method="concave_hull_alpha",
            allow_holes=True,
        )
        assert not iso_holes.empty

    def test_create_isochrone_degenerate_cases(self, sample_crs: str) -> None:
        """Test isochrone generation with degenerate graphs (1 or 2 nodes)."""
        # 1 node
        g1 = nx.Graph()
        g1.add_node(1, pos=(0, 0), geometry=Point(0, 0))
        g1.graph = {"crs": sample_crs}

        iso1 = utils.create_isochrone(
            g1, center_point=Point(0, 0), threshold=10, method="concave_hull_knn"
        )
        assert not iso1.empty
        # Output is always Polygon or MultiPolygon (buffered from Point)
        assert iso1.geometry.iloc[0].geom_type in ["Polygon", "MultiPolygon"]

        # 2 nodes
        g2 = nx.Graph()
        g2.add_node(1, pos=(0, 0), geometry=Point(0, 0))
        g2.add_node(2, pos=(1, 1), geometry=Point(1, 1))
        g2.add_edge(1, 2, length=1.414)
        g2.graph = {"crs": sample_crs}

        iso2 = utils.create_isochrone(
            g2, center_point=Point(0, 0), threshold=10, method="concave_hull_knn"
        )
        assert not iso2.empty
        # Output is always Polygon or MultiPolygon (convex hull of LineString)
        assert iso2.geometry.iloc[0].geom_type in ["Polygon", "MultiPolygon"]

    def test_create_isochrone_input_validation(self, sample_nx_graph: nx.Graph) -> None:
        """Test input validation for create_isochrone."""
        with pytest.raises(ValueError, match="Unknown method"):
            utils.create_isochrone(
                sample_nx_graph, center_point=Point(0, 0), threshold=10, method="invalid_method"
            )

        with pytest.raises(ValueError, match="center_point and threshold must be provided"):
            utils.create_isochrone(sample_nx_graph, center_point=None, threshold=10)

        with pytest.raises(
            ValueError, match="Either 'graph' or 'nodes' and 'edges' must be provided"
        ):
            utils.create_isochrone(
                graph=None, nodes=None, edges=None, center_point=Point(0, 0), threshold=10
            )

        with pytest.raises(ValueError, match="threshold sequence must not be empty"):
            utils.create_isochrone(sample_nx_graph, center_point=Point(0, 0), threshold=[])

        with pytest.raises(TypeError, match="center_point must be a Point"):
            utils.create_isochrone(
                sample_nx_graph,
                center_point=[Point(0, 0), "bad"],
                threshold=10,
            )

    @pytest.mark.parametrize(
        "centers",
        [
            [Point(0, 0), Point(10, 0)],
            (Point(0, 0), Point(10, 0)),
            gpd.GeoSeries([Point(0, 0), Point(10, 0)], crs=27700),
        ],
        ids=["list", "tuple", "geoseries"],
    )
    def test_isochrone_covers_each_center(
        self,
        disconnected_streets: nx.Graph,
        centers: list[Point] | tuple[Point, ...] | gpd.GeoSeries,
    ) -> None:
        """All supported centre containers include both streets in the isochrone."""
        iso = utils.create_isochrone(
            disconnected_streets,
            center_point=centers,
            threshold=1,
            edge_attr="length",
            method="convex_hull",
        )
        geom = iso.geometry.iloc[0]
        assert all(geom.intersects(Point(x, 0)) for x in (0, 1, 10, 11))

    def test_create_isochrone_empty_center_sequence_returns_empty(
        self, sample_nx_graph: nx.Graph
    ) -> None:
        """Empty center sequences should preserve empty-output behavior."""
        iso = utils.create_isochrone(
            sample_nx_graph,
            center_point=[],
            threshold=1.0,
            method="convex_hull",
        )

        assert iso.empty

    def test_create_isochrone_buffer_none(self, sample_nx_graph: nx.Graph) -> None:
        """Test buffer method with buffer_distance=None (should return Polygon/MultiPolygon)."""
        center = Point(0, 0)
        iso = utils.create_isochrone(
            sample_nx_graph,
            center_point=center,
            threshold=2.0,
            method="buffer",
            buffer_distance=None,
        )
        assert not iso.empty
        # Output is always Polygon or MultiPolygon
        assert iso.geometry.iloc[0].geom_type in ["Polygon", "MultiPolygon"]

    def test_create_isochrone_from_gdf_inputs(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test create_isochrone with nodes/edges input (returns GDF reachable)."""
        center = Point(0, 0)
        iso = utils.create_isochrone(
            nodes=sample_nodes_gdf,
            edges=sample_edges_gdf,
            center_point=center,
            threshold=2.0,
            method="convex_hull",
        )
        assert not iso.empty
        assert iso.geometry.iloc[0].geom_type == "Polygon"

    def test_create_isochrone_far_snap(self, sample_nx_graph: nx.Graph) -> None:
        """Test isochrone with far away center (should snap to nearest)."""
        center = Point(1000, 1000)  # Far away
        iso = utils.create_isochrone(
            sample_nx_graph,
            center_point=center,
            threshold=1.0,
            method="concave_hull_knn",
        )
        # Current implementation snaps to nearest node regardless of distance
        assert not iso.empty
        # Should be a geometry (Point, LineString, or Polygon depending on neighbors)
        assert iso.geometry.iloc[0].geom_type in ["Point", "LineString", "Polygon"]

    def test_isochrone_excludes_unreachable_street(self, disconnected_streets: nx.Graph) -> None:
        """A centre on the first street cannot reach the other component."""
        iso = utils.create_isochrone(
            disconnected_streets,
            center_point=Point(0, 0),
            threshold=2,
            edge_attr="length",
            method="convex_hull",
        )
        assert len(iso) == 1
        assert iso.geometry.iloc[0].intersects(Point(1, 0))
        assert not iso.geometry.iloc[0].intersects(Point(10, 0))

    def test_create_isochrone_all_methods(self, sample_nx_graph: nx.Graph) -> None:
        """Ensure all methods run without error on a standard graph."""
        center = Point(0, 0)
        methods = ["concave_hull_knn", "concave_hull_alpha", "convex_hull", "buffer"]

        for method in methods:
            iso = utils.create_isochrone(
                sample_nx_graph, center_point=center, threshold=2.0, method=method
            )
            assert not iso.empty
            assert isinstance(iso, gpd.GeoDataFrame)

    def test_prepare_isochrone_graph_edge_attr_injection(
        self, sample_nodes_gdf: gpd.GeoDataFrame, sample_edges_gdf: gpd.GeoDataFrame
    ) -> None:
        """Test that edge attribute is injected if missing when using GDF inputs."""
        # Remove length column if exists
        if "length" in sample_edges_gdf.columns:
            sample_edges_gdf = sample_edges_gdf.drop(columns=["length"])

        iso = utils.create_isochrone(
            nodes=sample_nodes_gdf,
            edges=sample_edges_gdf,
            center_point=Point(0, 0),
            threshold=10.0,
            edge_attr="length",  # Should be calculated from geometry
            method="convex_hull",
        )
        assert not iso.empty

    @pytest.mark.parametrize(
        ("coords", "method"),
        [
            pytest.param([(0, 0)], "concave_hull_knn", id="knn-single-point"),
            pytest.param([(0, 0), (1, 1)], "concave_hull_knn", id="knn-two-points"),
            pytest.param([(0, 0), (1, 1), (2, 2)], "concave_hull_knn", id="knn-collinear"),
            pytest.param([(0, 0), (1, 1)], "concave_hull_alpha", id="alpha-two-points"),
        ],
    )
    def test_isochrone_buffers_degenerate_hulls(
        self,
        coords: list[tuple[int, int]],
        method: str,
        sample_crs: str,
    ) -> None:
        """Degenerate reachable sets produce the specified buffer around their hull."""
        graph = nx.path_graph(len(coords))
        graph.graph["crs"] = sample_crs
        nx.set_node_attributes(graph, dict(enumerate(coords)), "pos")
        nx.set_edge_attributes(graph, 1.0, "length")
        result = utils.create_isochrone(
            graph,
            center_point=Point(coords[0]),
            threshold=10,
            method=method,
            degenerate_buffer_distance=0.25,
        )
        expected = Point(coords[0]) if len(coords) == 1 else LineString([coords[0], coords[-1]])
        assert len(result) == 1
        assert result.crs == sample_crs
        assert result.geometry.iloc[0].equals(expected.buffer(0.25))

    def test_concave_hull_knn_degenerate_triangle(self) -> None:
        """Retain list-input compatibility; public isochrones pass coordinate arrays."""
        points = [Point(0, 0), Point(1, 0), Point(0, 1)]
        poly = distance_utils._concave_hull_knn(points, k=3)
        assert isinstance(poly, Polygon)
        assert poly.area == 0.5

    def test_prepare_isochrone_graph_dict_edges_missing_attr(self, sample_crs: str) -> None:
        """Isochrones infer missing edge lengths from heterogeneous edge geometries."""
        nodes = {
            "node_type": gpd.GeoDataFrame(
                {"geometry": [Point(0, 0), Point(1, 1)]},
                index=[1, 2],
                crs=sample_crs,
            )
        }
        # Edges dict without 'length' attribute, but with geometry
        edges = {
            ("node_type", "rel", "node_type"): gpd.GeoDataFrame(
                {
                    "u": [1],
                    "v": [2],
                    "geometry": [LineString([(0, 0), (1, 1)])],
                },
                crs=sample_crs,
            ).set_index(["u", "v"])
        }

        # Should calculate length automatically
        result = utils.create_isochrone(
            nodes=nodes,
            edges=edges,
            center_point=Point(0, 0),
            threshold=2,
            edge_attr="length",
            method="convex_hull",
            degenerate_buffer_distance=0.25,
        )
        assert len(result) == 1
        assert result.geometry.iloc[0].equals(LineString([(0, 0), (1, 1)]).buffer(0.25))
        assert result.crs == sample_crs

    def test_find_best_candidate_no_valid(self) -> None:
        """Test _find_best_candidate when no valid candidate exists."""
        # Retain zero-length candidate handling: public hulls deduplicate these
        # coordinates before tracing, so this defensive branch is not reachable.
        coords = np.array([(0, 0), (1, 0), (0, 1)])
        current_idx = 0
        candidates = [1, 2]
        prev_vec = np.array([1, 0])
        hull_indices = [0]
        start_idx = 0

        # This should find a candidate
        best = distance_utils._find_best_candidate(
            coords, current_idx, candidates, prev_vec, hull_indices, start_idx
        )
        assert best is not None

        # Now force failure by making candidates invalid (e.g. zero length vector)
        # coords[1] same as coords[0]
        coords_degenerate = np.array([(0, 0), (0, 0)])
        best_degenerate = distance_utils._find_best_candidate(
            coords_degenerate, 0, [1], prev_vec, hull_indices, start_idx
        )
        assert best_degenerate is None

    def test_is_valid_edge_intersection(self) -> None:
        """Retain the Shapely fallback; public tracing supplies cached coordinate arrays."""
        # Construct a hull that is about to self-intersect
        coords = np.array([(0, 0), (10, 0), (10, 10), (0, 10), (5, 5)])
        # Hull: (0,0) -> (10,0) -> (10,10) -> (0,10)
        hull_indices = [0, 1, 2, 3]
        start_idx = 0
        hull_arr = np.asarray(hull_indices, dtype=int)
        seg_starts = coords[hull_arr[:-2]]
        seg_ends = coords[hull_arr[1:-1]]
        existing_lines = [
            LineString([start, end]) for start, end in zip(seg_starts, seg_ends, strict=True)
        ]
        seg_bounds_min = np.minimum(seg_starts, seg_ends)
        seg_bounds_max = np.maximum(seg_starts, seg_ends)

        # Edge from (0,10) to (5,5) is valid (inside)
        assert distance_utils._is_valid_edge(
            coords=coords,
            current_idx=3,
            candidate_idx=4,
            start_idx=start_idx,
            existing_lines=existing_lines,
            seg_bounds_min=seg_bounds_min,
            seg_bounds_max=seg_bounds_max,
        )

        # Edge from (0,10) to (5, -5) crosses (0,0)-(10,0)
        invalid_coords = np.vstack([coords, np.array([(5, -5)])])
        assert not distance_utils._is_valid_edge(
            coords=invalid_coords,
            current_idx=3,
            candidate_idx=5,
            start_idx=start_idx,
            existing_lines=existing_lines,
            seg_bounds_min=seg_bounds_min,
            seg_bounds_max=seg_bounds_max,
        )

    def test_create_isochrone_empty_result(self) -> None:
        """Test create_isochrone returning empty result."""
        # Threshold 0.0 should result in no reachable nodes/edges if center is not on a node?
        # Or if graph is empty.
        empty_graph = nx.Graph()
        empty_graph.graph["crs"] = "EPSG:4326"

        iso = utils.create_isochrone(empty_graph, center_point=Point(0, 0), threshold=10.0)
        assert iso.empty
        assert "geometry" in iso.columns

    def test_isochrone_all_edges_cut(self, sample_crs: str) -> None:
        """Removing every edge leaves no reachable street geometry to draw."""
        G = nx.Graph()
        G.graph["crs"] = sample_crs
        G.add_edge(1, 2, full_edge_type=("a", "b", "c"), length=1)
        G.add_node(1, pos=(0, 0))
        G.add_node(2, pos=(1, 1))

        iso = utils.create_isochrone(
            G,
            center_point=Point(0, 0),
            threshold=10,
            cut_edge_types=[("a", "b", "c")],
            method="convex_hull",
        )
        assert iso.empty
        assert len(G) == 2
        assert G.number_of_edges() == 1

    def test_prepare_isochrone_graph_errors(self) -> None:
        """Isochrones require a graph or both node and edge layers."""
        with pytest.raises(
            ValueError, match="Either 'graph' or 'nodes' and 'edges' must be provided"
        ):
            utils.create_isochrone(center_point=Point(0, 0), threshold=10)

    def test_process_component_empty(self, sample_crs: str) -> None:
        """Retain the empty-component guard bypassed by public empty-graph handling."""
        # Empty component
        G = nx.Graph()
        assert distance_utils._process_component(G, "convex_hull", sample_crs) is None

    def test_generate_buffer_options(self, sample_nx_graph: nx.Graph) -> None:
        """Test buffer method with non-default options."""
        iso = utils.create_isochrone(
            sample_nx_graph,
            center_point=Point(0, 0),
            threshold=1.0,
            method="buffer",
            buffer_distance=10.0,
            cap_style=2,
            join_style=2,
            resolution=4,
        )
        assert not iso.empty
        assert iso.geometry.iloc[0].geom_type in ["Polygon", "MultiPolygon"]

    def test_process_component_empty_geoms(self, sample_nx_graph: nx.Graph) -> None:
        """Test _process_component when geometries become empty/invalid."""
        with mock.patch("city2graph.utils.distance._extract_isochrone_geometries") as mock_extract:
            mock_extract.return_value = [Point()]  # Empty point

            iso = utils.create_isochrone(
                sample_nx_graph,
                center_point=Point(0, 0),
                threshold=100,
                method="convex_hull",
                edge_attr="edge_feature1",
            )
            assert iso.empty
            assert len(iso) == 0

    def test_concave_hull_knn_k_zero(self, sample_nx_graph: nx.Graph) -> None:
        """Test concave_hull_knn with k=0 to trigger scalar index handling."""
        # This hits line 3160: indices = [indices]
        iso = utils.create_isochrone(
            sample_nx_graph,
            center_point=Point(0, 0),
            threshold=100,
            method="concave_hull_knn",
            k=0,
            edge_attr="edge_feature1",
        )
        assert not iso.empty

    @pytest.mark.parametrize(
        "walk",
        [
            pytest.param([0, 1, 3, 2], id="valid-retry"),
            pytest.param([0, 1, 2, 3], id="self-intersection-fallback"),
        ],
    )
    def test_concave_hull_knn_retries_with_larger_k(self, walk: list[int]) -> None:
        """concave_hull_knn should retry the full walk with a larger neighborhood."""
        graph = nx.cycle_graph(4)
        graph.graph["crs"] = "EPSG:27700"
        nx.set_node_attributes(graph, dict(enumerate([(0, 0), (1, 0), (1, 1), (0, 1)])), "pos")
        nx.set_edge_attributes(graph, 1.0, "length")
        with mock.patch("city2graph.utils.distance._trace_concave_hull_once") as mock_trace:
            mock_trace.side_effect = [None, walk]
            result = utils.create_isochrone(
                graph,
                center_point=Point(0, 0),
                threshold=10,
                method="concave_hull_knn",
                k=2,
            )
        assert result.geometry.iloc[0].equals(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]))
        assert [call.args[2] for call in mock_trace.call_args_list] == [2, 3]

    def test_concave_hull_knn_fallback(self, sample_nx_graph: nx.Graph) -> None:
        """Test concave_hull_knn fallback to alpha hull when the walker cannot progress."""
        with (
            mock.patch("city2graph.utils.distance._find_next_hull_point") as mock_find,
            mock.patch("city2graph.utils.distance._concave_fallback_alpha") as mock_fallback,
        ):
            mock_find.return_value = None
            mock_fallback.return_value = Polygon([(0, 0), (1, 0), (0, 1)])

            iso = utils.create_isochrone(
                sample_nx_graph,
                center_point=Point(0, 0),
                threshold=100,
                method="concave_hull_knn",
                k=3,
                edge_attr="edge_feature1",
            )
            assert not iso.empty
            mock_fallback.assert_called_once()
            assert iso.geometry.iloc[0].equals(mock_fallback.return_value)

    def test_create_isochrone_falls_back_when_hull_walk_never_closes(self, sample_crs: str) -> None:
        """Isochrone generation should fall back when the concave walk exhausts without closure."""
        graph = nx.Graph()
        graph.graph["crs"] = sample_crs
        graph.add_node(1, pos=(0, 0))
        graph.add_node(2, pos=(2, 0))
        graph.add_node(3, pos=(1, 1))
        graph.add_edge(1, 2, length=1.0)
        graph.add_edge(2, 3, length=1.0)

        next_indices = iter([1, 2, 1, 2])

        with (
            mock.patch(
                "city2graph.utils.distance._find_next_hull_point",
                side_effect=lambda **_kwargs: next(next_indices),
            ),
            mock.patch(
                "city2graph.utils.distance._concave_fallback_alpha",
                return_value=Polygon([(0, 0), (2, 0), (1, 1)]),
            ) as mock_fallback,
        ):
            iso = utils.create_isochrone(
                graph,
                center_point=Point(0, 0),
                threshold=10.0,
                edge_attr="length",
                method="concave_hull_knn",
            )

        assert not iso.empty
        mock_fallback.assert_called_once()

    def test_buffer_negative_distance(self, sample_nx_graph: nx.Graph) -> None:
        """Test buffer method with negative distance (results in empty polygon)."""
        iso = utils.create_isochrone(
            sample_nx_graph,
            center_point=Point(0, 0),
            threshold=100,
            method="buffer",
            buffer_distance=-100.0,  # Should collapse points
            edge_attr="edge_feature1",
        )
        assert iso.empty
        assert len(iso) == 0

    def test_isochrone_no_edge_attr(self, sample_nx_graph: nx.Graph) -> None:
        """Test create_isochrone with edge_attr=None."""
        # Should rely on existing weights or default if not used by method
        iso = utils.create_isochrone(
            sample_nx_graph,
            center_point=Point(0, 0),
            threshold=10.0,
            edge_attr=None,
            method="convex_hull",
        )
        assert not iso.empty

    def test_isochrone_generate_polygon_returns_none(self, sample_nx_graph: nx.Graph) -> None:
        """Test _process_component when _generate_polygon returns None."""
        with mock.patch("city2graph.utils.distance._generate_polygon") as mock_gen:
            mock_gen.return_value = None
            iso = utils.create_isochrone(
                sample_nx_graph,
                center_point=Point(0, 0),
                threshold=10.0,
                method="convex_hull",
            )
            assert iso.empty

    def test_isochrone_degenerate_buffer_param(self, sample_crs: str) -> None:
        """Test degenerate_buffer_distance parameter."""
        # Create a graph that results in a degenerate geometry (Point)
        G = nx.Graph()
        G.graph["crs"] = sample_crs
        G.add_node(1, pos=(0, 0))

        # Default buffer is 1.0. Let's set it to 0.5.
        iso = utils.create_isochrone(
            G,
            center_point=Point(0, 0),
            threshold=10.0,
            method="convex_hull",  # Single point convex hull is Point
            degenerate_buffer_distance=0.5,
        )
        assert not iso.empty
        poly = iso.geometry.iloc[0]
        # Area of circle with radius 0.5 is pi * 0.25 ~= 0.785
        assert 0.7 < poly.area < 0.8

    def test_isochrone_directed_graph(self, sample_crs: str) -> None:
        """Test create_isochrone with a directed graph."""
        G = nx.DiGraph()
        G.graph["crs"] = sample_crs
        G.add_node(1, pos=(0, 0))
        G.add_node(2, pos=(1, 0))
        G.add_edge(1, 2, length=1.0)  # 1 -> 2

        # From 1, can reach 2.
        iso = utils.create_isochrone(
            G,
            center_point=Point(0, 0),
            threshold=2.0,
            edge_attr="length",
            method="convex_hull",
        )
        assert not iso.empty
        # Should cover both points

        # From 2, cannot reach 1.
        iso2 = utils.create_isochrone(
            G,
            center_point=Point(1, 0),
            threshold=2.0,
            edge_attr="length",
            method="convex_hull",
        )
        # Should only contain node 2 (Point) -> buffered to Polygon
        assert not iso2.empty
        assert iso2.geometry.iloc[0].area < 4.0  # Small buffer around point

    def test_isochrone_buffer_returns_non_polygon(self) -> None:
        """Test create_isochrone handles non-polygon geometry from buffer."""
        # Create a simple graph that might result in a geometry collection
        G = nx.Graph()
        G.add_node(1, pos=(0, 0), geometry=Point(0, 0))
        G.graph = {"crs": "EPSG:27700"}

        result = utils.create_isochrone(
            G,
            center_point=Point(0, 0),
            threshold=10,
            method="buffer",
            buffer_distance=0.01,
        )
        # Result should always be Polygon or MultiPolygon
        assert result.geometry.iloc[0].geom_type in ["Polygon", "MultiPolygon"]

    def test_create_isochrone_buffers_non_polygon_union(
        self,
        sample_nx_graph: nx.Graph,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """create_isochrone should normalize non-polygon unions back into polygon output."""
        monkeypatch.setattr(
            distance_utils,
            "_generate_component_polygons",
            lambda _reachable, _method, _crs, **_kwargs: [LineString([(0, 0), (1, 0)])],
        )

        result = utils.create_isochrone(
            sample_nx_graph,
            center_point=Point(0, 0),
            threshold=10.0,
            method="buffer",
        )

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 1

    def test_create_isochrone_with_duplicate_positions_uses_degenerate_knn_path(self) -> None:
        """Duplicate reachable positions should still produce an isochrone through the public API."""
        graph = nx.Graph()
        graph.add_node("a", pos=(0.0, 0.0))
        graph.add_node("b", pos=(0.0, 0.0))
        graph.add_node("c", pos=(1.0, 0.0))
        graph.add_edge("a", "b", length=0.0)
        graph.add_edge("b", "c", length=1.0)
        graph.graph["crs"] = "EPSG:27700"

        result = utils.create_isochrone(
            graph,
            center_point=Point(0, 0),
            threshold=5.0,
            method="concave_hull_knn",
        )

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 1
