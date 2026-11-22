"""Public contracts for proximity graph construction."""

from __future__ import annotations

from fractions import Fraction
from itertools import pairwise
from typing import TYPE_CHECKING
from typing import cast

if TYPE_CHECKING:
    from collections.abc import Callable

from typing import Literal
from unittest.mock import Mock

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csgraph
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from shapely import get_coordinates
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import Polygon

from city2graph.proximity import bridge_nodes
from city2graph.proximity import contiguity_graph
from city2graph.proximity import delaunay_graph
from city2graph.proximity import euclidean_minimum_spanning_tree
from city2graph.proximity import fixed_radius_graph
from city2graph.proximity import gabriel_graph
from city2graph.proximity import group_nodes
from city2graph.proximity import knn_graph
from city2graph.proximity import relative_neighborhood_graph
from city2graph.proximity import waxman_graph
from city2graph.utils import nx_to_gdf
from tests.city2graph.helpers import assert_as_nx_deprecation
from tests.city2graph.helpers import assert_valid_proximity_result
from tests.city2graph.helpers import make_network_edges
from tests.city2graph.helpers import make_points_simple
from tests.city2graph.helpers import make_poly_points_pair
from tests.city2graph.helpers import make_single_point
from tests.city2graph.helpers import make_square_polygons


@pytest.fixture
def small_points() -> gpd.GeoDataFrame:
    """Small square-ish point set for proximity generators."""
    return make_points_simple([(0, 0), (1, 0), (0, 1), (2, 2)])


@pytest.fixture
def network_edges() -> gpd.GeoDataFrame:
    """Tiny 2-edge network for network distance metric tests."""
    return make_network_edges(
        src_ids=[0, 1],
        dst_ids=[1, 2],
        geometries=[LineString([(0, 0), (1, 0)]), LineString([(1, 0), (0, 1)])],
        crs="EPSG:27700",
    )


def _raise_dense_matrix_error(_self: object, _coords: object) -> None:
    """Fail the test if any builder materialises a dense distance matrix."""
    msg = "dense distance matrix must not be materialised"
    raise AssertionError(msg)


def _make_chain_points_and_network() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Return four points sitting exactly on the nodes of a unit chain network."""
    pts = make_points_simple([(0, 0), (1, 0), (2, 0), (3, 0)])
    net = make_network_edges(
        src_ids=[0, 1, 2],
        dst_ids=[1, 2, 3],
        geometries=[
            LineString([(0, 0), (1, 0)]),
            LineString([(1, 0), (2, 0)]),
            LineString([(2, 0), (3, 0)]),
        ],
        crs=pts.crs,
    )
    return pts, net


def _count_searches(monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, Mock]:
    """Record sparse and NetworkX shortest-path searches."""
    sparse_search = Mock(wraps=csgraph.dijkstra)
    networkx_search = Mock(wraps=nx.dijkstra_predecessor_and_distance)
    monkeypatch.setattr(csgraph, "dijkstra", sparse_search)
    monkeypatch.setattr(nx, "dijkstra_predecessor_and_distance", networkx_search)
    return sparse_search, networkx_search


def _searched_origins(search: Mock) -> int:
    """Count origins passed to sparse shortest-path searches."""
    return sum(len(call.kwargs["indices"]) for call in search.call_args_list)


def _undirected_edge_weights(edges: gpd.GeoDataFrame) -> dict[tuple[object, object], float]:
    """Map sorted (u, v) index pairs to edge weights."""
    return {
        tuple(sorted((u, v))): float(w)
        for (u, v), w in zip(edges.index, edges["weight"], strict=True)
    }


class TestKnnGraph:
    """Tests for ``knn_graph``."""

    def test_knn_non_string_metric_defaults(self, small_points: gpd.GeoDataFrame) -> None:
        """Non-string metric argument falls back to euclidean (robust normalisation)."""
        nodes, edges = knn_graph(small_points, k=1, distance_metric=None)  # type: ignore[arg-type]
        assert_valid_proximity_result(nodes, edges, len(small_points), allow_empty_edges=False)

    @pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
    def test_knn_graph_coincident_points_have_no_self_loops(self, metric: str) -> None:
        """Coincident points must neighbour each other, never themselves.

        With several nodes at exactly the same location every distance to the query
        point is zero, so the query point is not necessarily returned first by the
        neighbour search. The graph must still link the coincident nodes together.
        """
        pts = make_points_simple([(0, 0), (0, 0), (0, 0), (5, 5)])

        nodes, edges = knn_graph(pts, k=2, distance_metric=metric)

        assert_valid_proximity_result(nodes, edges, len(pts))
        pairs = [frozenset(pair) for pair in edges.index]
        assert all(len(pair) == 2 for pair in pairs), "self-loop found"
        assert {frozenset({0, 1}), frozenset({0, 2}), frozenset({1, 2})} <= set(pairs)
        assert sum(3 in pair for pair in pairs) >= 2

    def test_knn_single_point_early_exit(self) -> None:
        """KNN with a single point returns no edges (early return path)."""
        gdf = make_single_point()
        nodes, edges = knn_graph(gdf, k=3)
        assert_valid_proximity_result(nodes, edges, 1)
        assert edges.empty

    def test_knn_zero_k_early_exit(self, small_points: gpd.GeoDataFrame) -> None:
        """K <= 0 triggers early return with no edges."""
        nodes, edges = knn_graph(small_points, k=0)
        assert_valid_proximity_result(nodes, edges, len(small_points))
        assert edges.empty

    def test_directed_knn_as_nx(self, small_points: gpd.GeoDataFrame) -> None:
        """Directed kNN with as_nx=True exercises _directed_graph return branch."""
        src = small_points.iloc[:2]
        dst = small_points.iloc[2:]
        G = cast("nx.Graph", knn_graph(src, k=1, target_gdf=dst, as_nx=True))
        assert hasattr(G, "number_of_nodes")
        assert G.number_of_edges() >= 1

    def test_directed_knn_crs_mismatch(self, small_points: gpd.GeoDataFrame) -> None:
        """Directed helpers validate CRS equality between src and dst."""
        src = small_points.iloc[:2]
        dst = small_points.iloc[2:].to_crs("EPSG:4326")
        with pytest.raises(ValueError, match="CRS mismatch between source and target"):
            knn_graph(src, k=1, target_gdf=dst)

    @pytest.mark.parametrize("k", [0, -1])
    def test_directed_nonpositive_k(self, k: int) -> None:
        """A non-positive neighbour count follows the single-layer empty-edge policy."""
        frame = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(1, 0)], crs=3857)
        nodes, edges = knn_graph(frame, k=k, target_gdf=frame)
        assert len(nodes) == 4
        assert edges.empty

    def test_knn_graph_network_metric_points_sharing_node_have_no_self_loops(self) -> None:
        """Points snapped to the same network node must neighbour each other, not themselves."""
        net = make_network_edges(
            src_ids=[0, 1, 2],
            dst_ids=[1, 2, 3],
            geometries=[
                LineString([(0, 0), (10, 0)]),
                LineString([(10, 0), (20, 0)]),
                LineString([(20, 0), (30, 0)]),
            ],
        )
        # Points 0-2 all snap to the network node at (0, 0); 3 and 4 snap to the next nodes
        pts = make_points_simple([(0.5, 0.3), (0.6, -0.2), (0.4, 0.1), (10, 1), (20, 1)])

        nodes, edges = knn_graph(pts, k=2, distance_metric="network", network_gdf=net)

        assert_valid_proximity_result(nodes, edges, len(pts))
        pairs = [frozenset(pair) for pair in edges.index]
        assert all(len(pair) == 2 for pair in pairs), "self-loop found"
        # Each co-located point has the other two co-located points as its 2 nearest neighbours
        assert {frozenset({0, 1}), frozenset({0, 2}), frozenset({1, 2})} <= set(pairs)

    def test_knn_network_distance_requires_roads(self, small_points: gpd.GeoDataFrame) -> None:
        """Network kNN needs a road layer."""
        with pytest.raises(ValueError, match="network_gdf is required"):
            knn_graph(small_points, k=1, distance_metric="network")

    def test_knn_network_metric_custom_weight(self, small_points: gpd.GeoDataFrame) -> None:
        """Custom network weight attribute overrides geometry-derived lengths."""
        pts = small_points.iloc[:2]
        net = make_network_edges(
            src_ids=[0],
            dst_ids=[1],
            geometries=[LineString([(0, 0), (1, 0)])],
            crs=pts.crs,
            extra_attrs={"travel_time": [7.5]},
        )
        _, edges = knn_graph(
            pts,
            k=1,
            distance_metric="network",
            network_gdf=net,
            network_weight="travel_time",
        )
        assert pytest.approx(edges.iloc[0]["weight"], rel=1e-6) == 7.5

    @pytest.mark.parametrize("cost", [-1.0, float("nan"), float("-inf"), Fraction(-1, 2)])
    def test_network_proximity_rejects_invalid_shortest_path_costs(
        self, cost: float | Fraction
    ) -> None:
        """An invalid user-supplied weight cannot silently remove connectivity."""
        points = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(1, 0)], crs=3857)
        network = gpd.GeoDataFrame(
            {"travel_cost": [cost]}, geometry=[LineString([(0, 0), (1, 0)])], crs=3857
        )

        with pytest.raises(ValueError, match=r"travel_cost.*non-negative.*NaN"):
            knn_graph(
                points,
                k=1,
                distance_metric="network",
                network_gdf=network,
                network_weight="travel_cost",
            )

    @pytest.mark.parametrize("cost", [0.0, 2.5, Fraction(5, 2)])
    def test_network_proximity_preserves_valid_costs(self, cost: float | Fraction) -> None:
        """Zero and finite positive custom costs retain their supplied values."""
        points = make_points_simple([(0, 0), (2, 0)])
        network = gpd.GeoDataFrame(
            {"travel_cost": [cost]}, geometry=[LineString([(0, 0), (2, 0)])], crs=points.crs
        )
        _, edges = knn_graph(
            points,
            k=1,
            distance_metric="network",
            network_gdf=network,
            network_weight="travel_cost",
        )
        assert len(edges) == 1
        assert edges["weight"].iloc[0] == pytest.approx(float(cost))

    def test_network_proximity_missing_weight_takes_precedence_and_caps_examples(self) -> None:
        """Single-pass validation prioritizes missing numeric weights and caps error samples."""
        points = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(5, 0)], crs=3857)
        geometries = [LineString([(i, 0), (i + 1, 0)]) for i in range(5)]

        # Both an invalid negative cost and four non-numeric costs are present.
        mixed_network = gpd.GeoDataFrame(
            {"travel_cost": [-1.0, "bad", "bad", "bad", "bad"]},
            geometry=geometries,
            crs=3857,
        )
        with pytest.raises(ValueError, match="missing numeric 'travel_cost'") as exc_missing:
            knn_graph(
                points,
                k=1,
                distance_metric="network",
                network_gdf=mixed_network,
                network_weight="travel_cost",
            )
        assert exc_missing.value.args[0].count("((") == 3

        # More than three negative/NaN edges still cap reported examples at three.
        invalid_network = gpd.GeoDataFrame(
            {"travel_cost": [-1.0, -2.0, float("nan"), -4.0, -5.0]},
            geometry=geometries,
            crs=3857,
        )
        with pytest.raises(ValueError, match=r"travel_cost.*non-negative.*NaN") as exc_invalid:
            knn_graph(
                points,
                k=1,
                distance_metric="network",
                network_gdf=invalid_network,
                network_weight="travel_cost",
            )
        assert exc_invalid.value.args[0].count("((") == 3

    @pytest.mark.parametrize("detour", [False, True])
    def test_network_proximity_preserves_infinite_cost_barriers(self, detour: bool) -> None:
        """A blocked direct edge permits a finite detour, otherwise no connection."""
        points = make_points_simple([(0, 0), (2, 0)])
        geometries = [LineString([(0, 0), (2, 0)])]
        costs = [float("inf")]
        if detour:
            geometries.extend([LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 0)])])
            costs.extend([2.0, 3.0])
        network = gpd.GeoDataFrame({"travel_cost": costs}, geometry=geometries, crs=points.crs)
        original = network.copy(deep=True)

        nodes, edges = knn_graph(
            points,
            k=1,
            distance_metric="network",
            network_gdf=network,
            network_weight="travel_cost",
        )

        assert len(nodes) == 2
        pd.testing.assert_frame_equal(network, original)
        if detour:
            assert len(edges) == 1
            assert edges["weight"].iloc[0] == pytest.approx(5.0)
            assert edges.geometry.iloc[0].equals(LineString([(0, 0), (1, 1), (2, 0)]))
        else:
            assert edges.empty

    def test_network_metric_invalid_weight_attribute(self, small_points: gpd.GeoDataFrame) -> None:
        """Network weight attribute that's missing or non-numeric raises ValueError.

        Covers lines 108-113 in _resolve_network_weight_attribute.
        """
        # Create network with edges that have a non-numeric weight attribute
        net = gpd.GeoDataFrame(
            {
                "geometry": [LineString([(0, 0), (1, 0)])],
                "bad_weight": ["not_a_number"],  # Non-numeric value
            },
            index=pd.MultiIndex.from_arrays([[0], [1]], names=("source_id", "target_id")),
            crs=small_points.crs,
        )

        # This should raise ValueError because bad_weight is not numeric
        with pytest.raises(ValueError, match="Edges missing numeric 'bad_weight' attribute"):
            knn_graph(
                small_points,
                k=1,
                distance_metric="network",
                network_gdf=net,
                network_weight="bad_weight",
            )

    def test_network_metric_precomputed_auto_length(self, small_points: gpd.GeoDataFrame) -> None:
        """Network with pre-existing __c2g_edge_length skips recomputation.

        Covers line 119 in _resolve_network_weight_attribute (continue path).
        """
        # Create network where edges already have the auto-computed length attribute
        net = make_network_edges(
            src_ids=[0],
            dst_ids=[1],
            geometries=[LineString([(0, 0), (1, 0)])],
            crs=small_points.crs,
            extra_attrs={"__c2g_edge_length": [1.5]},
        )

        # Should use the pre-existing __c2g_edge_length value
        pts = small_points.iloc[:2]
        nodes, edges = knn_graph(pts, k=1, distance_metric="network", network_gdf=net)
        assert_valid_proximity_result(nodes, edges, len(pts))

    def test_network_metric_edges_without_geometry(self, small_points: gpd.GeoDataFrame) -> None:
        """Network edges use position-based fallback when geometry lacks length.

        Covers lines 123-128 in _resolve_network_weight_attribute - the branch where
        edges use node positions to compute length when geometry is missing or invalid.
        """
        pts = small_points.iloc[:2]
        net = make_network_edges(
            src_ids=[0],
            dst_ids=[1],
            geometries=[LineString([(0, 0), (1, 0)])],
            crs=pts.crs,
        )

        # This will exercise the position-based fallback internally
        nodes, edges = knn_graph(pts, k=1, distance_metric="network", network_gdf=net)
        assert_valid_proximity_result(nodes, edges, len(pts))

    def test_network_metric_no_valid_positions(self) -> None:
        """Network without valid node positions raises ValueError.

        Covers lines 281-282 in DistanceMetric._get_network_support.
        """
        pts = make_points_simple([(0, 0), (1, 1)])

        # Create a network GeoDataFrame that won't produce pos attributes when converted to nx
        # This happens when edges lack proper geometry that gdf_to_nx can extract nodes from
        # Create an edge-only GeoDataFrame with no extractable node positions
        net = gpd.GeoDataFrame(
            {
                "geometry": [None],  # No valid geometry
            },
            index=pd.MultiIndex.from_arrays([[0], [1]], names=("source_id", "target_id")),
            crs=pts.crs,
        )

        with pytest.raises(
            ValueError, match="network_gdf must include geometries with valid node positions"
        ):
            knn_graph(pts, k=1, distance_metric="network", network_gdf=net)

    def test_selection_and_geometry_share_searches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Co-located sources share one traversal and no full path dictionary."""
        roads = gpd.GeoDataFrame(
            geometry=[LineString([(i, 0), (i + 1, 0)]) for i in range(30)], crs=3857
        )
        points = gpd.GeoDataFrame(geometry=[Point(i, 0) for i in [0, 0, 1, 3, 7]], crs=3857)
        sparse_search, networkx_search = _count_searches(monkeypatch)
        monkeypatch.setattr(
            nx, "single_source_dijkstra", Mock(side_effect=AssertionError("Full paths retained"))
        )
        _, edges = knn_graph(points, k=1, distance_metric="network", network_gdf=roads)
        # Unique shortest paths are traced from the sparse search without NetworkX.
        assert sparse_search.call_count == 1
        assert _searched_origins(sparse_search) == 4
        assert networkx_search.call_count == 0
        assert len(edges) == 4
        assert set(edges["weight"]) == {0, 1, 2, 4}

    @pytest.mark.parametrize(
        "geometry",
        [
            pytest.param(None, id="missing-linework"),
            pytest.param(LineString([(10, 10), (10, 11)]), id="mismatched-endpoints"),
        ],
    )
    def test_knn_graph_uses_position_fallback_for_unusable_linework(
        self,
        small_points: gpd.GeoDataFrame,
        monkeypatch: pytest.MonkeyPatch,
        geometry: LineString | None,
    ) -> None:
        """Paths fall back to node positions when edge linework cannot be used."""
        graph = nx.Graph()
        graph.add_node("a", pos=(0.0, 0.0))
        graph.add_node("b", pos=(1.0, 0.0))
        graph.add_edge("a", "b", geometry=geometry)

        dummy_network = make_network_edges(
            src_ids=[0],
            dst_ids=[1],
            geometries=[LineString([(0, 0), (1, 0)])],
            crs=small_points.crs,
        )

        def fake_gdf_to_nx(*, edges: gpd.GeoDataFrame) -> nx.Graph:
            _ = edges
            return graph

        monkeypatch.setattr("city2graph.proximity.gdf_to_nx", fake_gdf_to_nx)

        nodes, edges = knn_graph(
            small_points.iloc[:2],
            k=1,
            distance_metric="network",
            network_gdf=dummy_network,
        )

        assert_valid_proximity_result(nodes, edges, 2)
        assert edges.geometry.iloc[0].equals(LineString([(0, 0), (1, 0)]))

    def test_knn_graph_surfaces_missing_network_via_public_path(
        self,
        small_points: gpd.GeoDataFrame,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Public API should still raise the network_gdf error if early validation is bypassed."""
        monkeypatch.setattr(
            "city2graph.proximity.DistanceMetric.validate", lambda _self, _crs: None
        )

        with pytest.raises(ValueError, match="network_gdf is required for network distance metric"):
            knn_graph(small_points.iloc[:2], k=1, distance_metric="network", network_gdf=None)

    def test_knn_graph_handles_network_edges_without_positions_or_geometry(
        self,
        small_points: gpd.GeoDataFrame,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """KNN should tolerate network edges whose fallback length resolves to zero."""
        graph = nx.Graph()
        graph.add_node("a", pos=(0.0, 0.0))
        graph.add_node("b")
        graph.add_edge("a", "b", geometry=None)

        dummy_network = make_network_edges(
            src_ids=[0],
            dst_ids=[1],
            geometries=[LineString([(0, 0), (1, 0)])],
            crs=small_points.crs,
        )

        def fake_gdf_to_nx(*, edges: gpd.GeoDataFrame) -> nx.Graph:
            _ = edges
            return graph

        monkeypatch.setattr("city2graph.proximity.gdf_to_nx", fake_gdf_to_nx)

        nodes, edges = knn_graph(
            small_points.iloc[:2],
            k=1,
            distance_metric="network",
            network_gdf=dummy_network,
        )

        assert_valid_proximity_result(nodes, edges, 2)

    def test_knn_network_k_exceeds_point_count(self) -> None:
        """K larger than n - 1 connects every reachable pair on the network."""
        pts, net = _make_chain_points_and_network()
        _, edges = knn_graph(pts, k=10, distance_metric="network", network_gdf=net)

        weights = _undirected_edge_weights(edges)
        expected = {
            (0, 1): 1.0,
            (0, 2): 2.0,
            (0, 3): 3.0,
            (1, 2): 1.0,
            (1, 3): 2.0,
            (2, 3): 1.0,
        }
        assert set(weights) == set(expected)
        for pair, dist in expected.items():
            assert weights[pair] == pytest.approx(dist)


class TestFixedRadiusGraph:
    """Tests for ``fixed_radius_graph``."""

    @pytest.mark.parametrize("radius", [-2.0, float("nan"), float("inf")])
    @pytest.mark.parametrize("directed", [False, True], ids=["undirected", "directed"])
    def test_rejects_an_invalid_radius(self, radius: float, directed: bool) -> None:
        """Both graph directions reject invalid radii before selecting edges."""
        points = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(1, 0)], crs=3857)
        with pytest.raises(ValueError, match="radius must be finite and non-negative"):
            fixed_radius_graph(points, radius=radius, target_gdf=points if directed else None)

    def test_fixed_radius_zero_connects_coincident_but_not_distinct_points(self) -> None:
        """Zero is a well-defined inclusive cutoff for distinct node IDs."""
        points = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(0, 0), Point(1, 0)], crs=3857)
        _, edges = fixed_radius_graph(points, radius=0)
        assert list(edges.index) == [(0, 1)]

    def test_network_edge_weight_length_branch(self, small_points: gpd.GeoDataFrame) -> None:
        """Provide 'length' edge attribute on network to trigger weighted path branch."""
        net = make_network_edges(
            src_ids=[0, 1],
            dst_ids=[1, 2],
            geometries=[LineString([(0, 0), (1, 0)]), LineString([(1, 0), (0, 1)])],
            crs="EPSG:27700",
            extra_attrs={"length": [1.0, 2.0]},
        )
        # Use network metric to ensure _set_network_edge_geometries is invoked
        _, e = fixed_radius_graph(
            small_points, radius=2.5, distance_metric="network", network_gdf=net
        )
        assert set(e.columns) >= {"weight", "geometry"}

    def test_network_geometry_fallback_same_nearest_node(self) -> None:
        """When both endpoints map to the same nearest network node, use straight segment fallback.

        This specifically exercises the branch in network geometry construction where the
        shortest-path node sequence has length <= 1, so the resulting edge geometry should
        be a direct LineString between the original point coordinates (not along the network).
        """
        # Two sample points very close to the origin
        pts = gpd.GeoDataFrame(
            {"id": [0, 1]},
            geometry=[Point(0.0, 0.01), Point(0.02, 0.0)],
            crs="EPSG:27700",
        ).set_index("id")

        # A sparse network with one node at the origin and another far away, ensuring both
        # points map to the origin node as their nearest network node
        net = gpd.GeoDataFrame(
            {
                "geometry": [
                    LineString([(0.0, 0.0), (100.0, 100.0)]),
                ]
            },
            index=pd.MultiIndex.from_arrays([[0], [1]], names=("source_id", "target_id")),
            crs="EPSG:27700",
        )

        # Network distance between both points is 0 (same mapped network node), so any
        # positive radius will include the edge. Geometry should fall back to straight line
        # between the two points (not a path along the network, which would be degenerate).
        nodes, edges = fixed_radius_graph(
            pts, radius=0.1, distance_metric="network", network_gdf=net
        )
        assert len(nodes) == 2
        assert len(edges) == 1
        geom = edges.geometry.iloc[0]
        assert isinstance(geom, LineString)
        # The fallback straight segment should have non-zero length approximately equal to
        # the Euclidean distance between the two sample points (diagonal ~0.02236)
        assert geom.length > 0

    def test_fixed_radius_network_cutoff_matches_bruteforce(self) -> None:
        """Cutoff-bounded Dijkstra returns the exact edge set, including the boundary."""
        pts, net = _make_chain_points_and_network()
        _, edges = fixed_radius_graph(pts, radius=2.0, distance_metric="network", network_gdf=net)

        weights = _undirected_edge_weights(edges)
        # Path lengths on the unit chain: 1 for adjacent pairs, 2 for one-apart pairs.
        # The pair (0, 3) at distance 3 must be excluded; distance == radius is included.
        expected = {(0, 1): 1.0, (1, 2): 1.0, (2, 3): 1.0, (0, 2): 2.0, (1, 3): 2.0}
        assert set(weights) == set(expected)
        for pair, dist in expected.items():
            assert weights[pair] == pytest.approx(dist)

    def test_directed_network_radius_exact_edges(self) -> None:
        """Directed radius selection yields the exact edge set with network weights."""
        pts, net = _make_chain_points_and_network()
        src = pts.iloc[[0, 3]]
        dst = pts.iloc[[1, 2]]
        _, edges = fixed_radius_graph(
            src, radius=1.0, distance_metric="network", network_gdf=net, target_gdf=dst
        )

        pairs = {(u, v): float(w) for (u, v), w in zip(edges.index, edges["weight"], strict=True)}
        expected = {(("src", 0), ("dst", 1)): 1.0, (("src", 3), ("dst", 2)): 1.0}
        assert set(pairs) == set(expected)
        for pair, dist in expected.items():
            assert pairs[pair] == pytest.approx(dist)


class TestDelaunayGraph:
    """Tests for ``delaunay_graph``."""

    def test_delaunay_requires_three_points_to_connect_nodes(self) -> None:
        """Two nodes are retained without a triangulation."""
        points = make_points_simple([(0, 0), (1, 1)])
        nodes, edges = delaunay_graph(points)
        pd.testing.assert_index_equal(nodes.index, points.index)
        assert edges.empty


class TestEuclideanMinimumSpanningTree:
    """Tests for ``euclidean_minimum_spanning_tree``."""

    def test_mst_unknown_metric_raises(self, small_points: gpd.GeoDataFrame) -> None:
        """Unknown distance metric should raise from internal dispatcher via MST."""
        with pytest.raises(ValueError, match="Unknown distance metric"):
            euclidean_minimum_spanning_tree(small_points, distance_metric="invalid")

    @pytest.mark.parametrize("duplicate_ids", [False, True])
    def test_manhattan_mst_preserves_equal_weight_edge_order(self, duplicate_ids: bool) -> None:
        """Delayed geometry construction preserves legacy Kruskal tie selection."""
        points = gpd.GeoDataFrame(
            geometry=[Point(x, y) for x in range(3) for y in range(3)],
            index=[f"node-{i}" for i in range(9)],
            crs=3857,
        )
        if duplicate_ids:
            points.index = [*points.index[:-1], points.index[0]]
        point_lookup = dict(zip(points.index, points.geometry, strict=True))
        candidates = {(i, j) for i in range(len(points)) for j in range(i + 1, len(points))}
        named = {(points.index[i], points.index[j]) for i, j in candidates}
        reference = nx.Graph(crs=points.crs)
        reference.add_nodes_from(points.index)
        for u, v in named:
            a, b = point_lookup[u], point_lookup[v]
            reference.add_edge(
                u,
                v,
                weight=abs(a.x - b.x) + abs(a.y - b.y),
                geometry=LineString([(a.x, a.y), (b.x, a.y), (b.x, b.y)]),
            )
        expected = nx_to_gdf(nx.minimum_spanning_tree(reference), nodes=False, edges=True)
        _, actual = euclidean_minimum_spanning_tree(points, distance_metric="manhattan")
        assert isinstance(expected, gpd.GeoDataFrame)
        pd.testing.assert_frame_equal(actual, expected)

    def test_manhattan_mst_preserves_invalid_geometry_filtering(self) -> None:
        """MST selection retains the shared input validator's geometry filtering."""
        points = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(float("nan"), 1)], crs=3857)
        nodes, edges = euclidean_minimum_spanning_tree(points, distance_metric="manhattan")
        assert list(nodes.index) == [0]
        assert edges.empty

    def test_mst_networkx_output(self, small_points: gpd.GeoDataFrame) -> None:
        """The legacy NetworkX output spans the input points."""
        graph = euclidean_minimum_spanning_tree(small_points, as_nx=True)
        assert set(graph) == set(small_points.index)
        assert nx.is_tree(graph)


class TestWaxmanGraph:
    """Tests for ``waxman_graph``."""

    def test_waxman_manhattan_metric(self, small_points: gpd.GeoDataFrame) -> None:
        """Waxman with Manhattan metric exercises the dispatcher manhattan branch."""
        nodes, edges = waxman_graph(
            small_points, beta=0.4, r0=2.0, seed=42, distance_metric="manhattan"
        )
        assert_valid_proximity_result(nodes, edges, len(small_points))

    def test_waxman_invalid_metric_raises(self, small_points: gpd.GeoDataFrame) -> None:
        """Unknown metric propagates from internal dispatcher via public API."""
        with pytest.raises(ValueError, match="Unknown distance metric"):
            waxman_graph(small_points, beta=0.5, r0=2.0, distance_metric="invalid")  # type: ignore[arg-type]

    @pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
    def test_waxman_preserves_dense_seeded_selection_across_batches(
        self,
        metric: Literal["euclidean", "manhattan"],
    ) -> None:
        """Batch boundaries must preserve every random draw and selected distance."""
        coordinates = np.random.default_rng(42).random((270, 2))
        points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(*coordinates.T), crs=3857)
        distance_name = "cityblock" if metric == "manhattan" else metric
        distances = squareform(pdist(coordinates, metric=distance_name))
        draws = np.random.default_rng(17).random(distances.shape)
        selected = np.triu(draws <= 0.01 * np.exp(-distances / 0.7), 1)
        sources, targets = np.where(selected)

        _, edges = waxman_graph(points, beta=0.01, r0=0.7, seed=17, distance_metric=metric)

        assert list(edges.index) == list(zip(sources, targets, strict=True))
        np.testing.assert_array_equal(edges["weight"], distances[sources, targets])

    @pytest.mark.parametrize("metric", ["euclidean", "manhattan", "network"])
    def test_waxman_preserves_last_coordinate_weights_for_duplicate_ids(
        self,
        metric: Literal["euclidean", "manhattan", "network"],
    ) -> None:
        """Sampling uses row positions but edge attributes use the last coordinate per ID."""
        points = gpd.GeoDataFrame(
            geometry=[Point(0, 0), Point(1, 0), Point(10, 0)], index=["a", "b", "a"], crs=3857
        )
        network = make_network_edges(
            [0, 1],
            [1, 2],
            [LineString([(0, 0), (1, 0)]), LineString([(1, 0), (10, 0)])],
            crs="EPSG:3857",
        )
        nodes, edges = waxman_graph(
            points, beta=1, r0=1000, seed=17, distance_metric=metric, network_gdf=network
        )
        assert list(nodes.index) == ["a", "b"]
        assert list(edges.index) == [("a", "b"), ("a", "a")]
        np.testing.assert_array_equal(edges["weight"], [9.0, 0.0])
        assert edges.loc[("a", "a"), "geometry"].length == 0

    def test_waxman_network_requires_network_gdf(self, small_points: gpd.GeoDataFrame) -> None:
        """Waxman with network metric should require network_gdf (dispatcher path)."""
        # Waxman has a slightly different error message, so can't use helper
        with pytest.raises(ValueError, match="network_gdf is required for network distance metric"):
            waxman_graph(small_points, beta=0.3, r0=1.5, distance_metric="network")

    def test_waxman_network_crs_mismatch(self, small_points: gpd.GeoDataFrame) -> None:
        """Waxman with network metric and mismatched CRS raises from _network_dm."""
        # Build a tiny network in a different CRS than points
        net = gpd.GeoDataFrame(
            {"geometry": [LineString([(0, 0), (1, 1)])]},
            index=pd.MultiIndex.from_arrays([[0], [1]], names=("source_id", "target_id")),
            crs="EPSG:4326",
        )
        with pytest.raises(ValueError, match="CRS mismatch"):
            waxman_graph(small_points, beta=0.3, r0=1.5, distance_metric="network", network_gdf=net)

    def test_waxman_network_metric_success(
        self, small_points: gpd.GeoDataFrame, network_edges: gpd.GeoDataFrame
    ) -> None:
        """Waxman with network metric executes network distance dispatcher path."""
        nodes, edges = waxman_graph(
            small_points,
            beta=0.6,
            r0=3.0,
            distance_metric="network",
            network_gdf=network_edges,
            seed=7,
        )
        assert len(nodes) == len(small_points)
        # Probabilistic; ensure schema exists (execution path hit)
        assert set(edges.columns) >= {"weight", "geometry"}


class TestContiguityGraph:
    """Tests for ``contiguity_graph``."""

    @pytest.mark.parametrize(("contiguity", "extra"), [("rook", set()), ("queen", {(1, 2)})])
    def test_contiguity_edge_and_corner_neighbours(
        self, contiguity: str, extra: set[tuple[int, int]]
    ) -> None:
        """Only queen contiguity connects polygons 1 and 2 at their shared corner."""
        polygons = make_square_polygons()
        nodes, edges = contiguity_graph(polygons, contiguity=contiguity)
        expected = {(0, 1), (0, 2), (1, 3)} | extra
        pd.testing.assert_index_equal(nodes.index, polygons.index)
        assert len(edges) == len(expected)
        assert set(_undirected_edge_weights(edges)) == expected
        graph = contiguity_graph(polygons, contiguity=contiguity, as_nx=True)
        assert isinstance(graph, nx.Graph)
        assert not graph.is_directed()
        assert set(graph) == set(polygons.index)
        assert {tuple(sorted(pair)) for pair in graph.edges()} == expected

    def test_contiguity_network_distance_requires_roads(self) -> None:
        """Network-weighted adjacency needs a road layer."""
        polygons = make_square_polygons()
        with pytest.raises(ValueError, match="network_gdf is required"):
            contiguity_graph(polygons, distance_metric="network")

    def test_contiguity_manhattan_geometry(self) -> None:
        """The diagonal neighbours take two axis-aligned steps between centroids."""
        polygons = make_square_polygons()
        _, edges = contiguity_graph(polygons, distance_metric="manhattan")
        diagonal = edges.loc[(1, 2)]
        assert diagonal.weight == 2
        coords = list(diagonal.geometry.coords)
        assert {coords[0], coords[-1]} == {(1.5, 0.5), (0.5, 1.5)}
        assert diagonal.geometry.length == 2
        assert all(x1 == x2 or y1 == y2 for (x1, y1), (x2, y2) in pairwise(coords))

    def test_contiguity_invalid_type(self) -> None:
        """Invalid contiguity type raises ValueError."""
        gdf = make_square_polygons()
        with pytest.raises(ValueError, match="Invalid contiguity type"):
            contiguity_graph(gdf, contiguity="invalid")

    def test_contiguity_empty_input(self) -> None:
        """Empty input returns empty nodes/edges GeoDataFrames."""
        empty = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:3857")
        nodes, edges = contiguity_graph(empty)
        assert nodes.empty
        assert edges.empty

    def test_contiguity_duplicate_edges_completes_neighbor_queries(self) -> None:
        """duplicate_edges=True makes MultiIndex neighbour queries complete (issue #156)."""
        gdf = make_square_polygons()
        _, base_edges = contiguity_graph(gdf, contiguity="queen")
        _, dup_edges = contiguity_graph(gdf, contiguity="queen", duplicate_edges=True)

        assert len(dup_edges) == 2 * len(base_edges)
        srcs = base_edges.index.get_level_values(0)
        dsts = base_edges.index.get_level_values(1)
        for node in gdf.index:
            forward = set(base_edges.xs(node, level=0).index) if node in srcs else set()
            backward = set(base_edges.xs(node, level=1).index) if node in dsts else set()
            neighbors = set(dup_edges.xs(node, level=0).index)
            assert neighbors == forward | backward

    def test_contiguity_duplicate_edges_empty_input(self) -> None:
        """Empty input with duplicate_edges=True returns empty GeoDataFrames."""
        empty = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:3857")
        nodes, edges = contiguity_graph(empty, duplicate_edges=True)
        assert nodes.empty
        assert edges.empty

    @pytest.mark.parametrize(("use_cost", "expected"), [(False, 1.0), (True, 7.0)])
    def test_contiguity_network_weight(self, use_cost: bool, expected: float) -> None:
        """Adjacent cells use the connecting road's geometry and cost."""
        polygons = make_square_polygons().iloc[:2]
        road = LineString(polygons.geometry.centroid.tolist())
        network = make_network_edges([0], [1], [road], crs=polygons.crs)
        if use_cost:
            network["length"] = [7.0]
        _, edges = contiguity_graph(
            polygons,
            distance_metric="network",
            network_gdf=network,
            network_weight="length" if use_cost else None,
        )
        assert _undirected_edge_weights(edges) == {(0, 1): expected}
        assert edges.geometry.iloc[0].equals(road)

    @pytest.mark.parametrize("set_point_nodes", [False, True])
    def test_contiguity_custom_node_geometry(self, set_point_nodes: bool) -> None:
        """Custom points set the edge endpoints independently of the output node geometry."""
        polygons = make_square_polygons().iloc[:2].assign(node_pts=[Point(0, 0), Point(10, 0)])
        nodes, edges = contiguity_graph(
            polygons, node_geom_col="node_pts", set_point_nodes=set_point_nodes
        )
        if set_point_nodes:
            assert nodes.geometry.tolist() == polygons["node_pts"].tolist()
            assert nodes["original_geometry"].equals(polygons.geometry)
        else:
            assert nodes.geometry.equals(polygons.geometry)
        assert _undirected_edge_weights(edges) == {(0, 1): 10}
        assert edges.geometry.iloc[0].equals(LineString([(0, 0), (10, 0)]))

    def test_contiguity_set_point_nodes_centroid(self) -> None:
        """set_point_nodes swaps polygon geometry for centroid points in nodes layer."""
        gdf = make_square_polygons()

        nodes, _ = contiguity_graph(gdf, set_point_nodes=True)

        assert set(nodes.geometry.geom_type) == {"Point"}
        assert nodes.geometry.equals(gdf.geometry.centroid)
        assert nodes["original_geometry"].equals(gdf.geometry)

    def test_contiguity_node_geom_col_missing(self) -> None:
        """Missing node_geom_col raises a clear error."""
        gdf = make_square_polygons()
        with pytest.raises(ValueError, match="node_geom_col 'missing' not found"):
            contiguity_graph(gdf, node_geom_col="missing")

    def test_contiguity_invalid_metric_raises(self) -> None:
        """Unsupported contiguity distance_metric triggers ValueError."""
        gdf = make_square_polygons()
        with pytest.raises(ValueError, match="Unknown distance metric"):
            contiguity_graph(gdf, distance_metric="invalid")

    def test_contiguity_network_crs_mismatch(self) -> None:
        """Network CRS different from polygons CRS raises ValueError."""
        gdf = make_square_polygons().iloc[:2]
        c1 = gdf.geometry.iloc[0].centroid
        c2 = gdf.geometry.iloc[1].centroid
        # Deliberately different CRS for network
        net = gpd.GeoDataFrame(
            {"geometry": [LineString([c1, c2])]},
            index=pd.MultiIndex.from_arrays([[0], [1]], names=("source_id", "target_id")),
            crs="EPSG:4326",
        )
        with pytest.raises(ValueError, match="CRS mismatch"):
            contiguity_graph(gdf, distance_metric="network", network_gdf=net)

    def test_contiguity_empty_as_nx(self) -> None:
        """Empty contiguity result as NetworkX preserves metadata."""
        empty = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:3857")
        G = cast("nx.Graph", contiguity_graph(empty, as_nx=True))
        assert G.number_of_nodes() == 0
        assert G.number_of_edges() == 0
        assert G.graph.get("contiguity") == "queen"
        assert G.graph.get("distance_metric") == "euclidean"

    def test_contiguity_single_polygon_no_edges(self) -> None:
        """Single polygon yields no adjacency edges (weights.neighbors empty branch)."""
        gdf = gpd.GeoDataFrame(
            {"val": [0]},
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            crs="EPSG:3857",
        ).set_index(pd.Index([0], name="pid"))
        nodes, edges = contiguity_graph(gdf)
        assert len(nodes) == 1
        assert edges.empty

    def test_contiguity_input_validation_and_empty_neighbors(self) -> None:
        """Cover contiguity_graph input validation and disjoint polygons."""
        # Not a GeoDataFrame
        with pytest.raises(TypeError, match="Input must be a GeoDataFrame"):
            contiguity_graph("not a gdf")

        # Disjoint polygons -> empty weights.neighbors -> empty edges
        # Create two polygons far apart
        polys = gpd.GeoDataFrame(
            {"val": [1, 2]},
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(10, 10), (11, 10), (11, 11), (10, 11)]),
            ],
            crs="EPSG:3857",
        )
        nodes, edges = contiguity_graph(polys)
        assert len(nodes) == 2
        assert edges.empty

    def test_duplicate_edges_rejects_as_nx(self) -> None:
        """duplicate_edges=True with as_nx=True raises ValueError."""
        gdf = make_square_polygons()
        with pytest.raises(ValueError, match="not supported with as_nx=True"):
            contiguity_graph(gdf, duplicate_edges=True, as_nx=True)


class TestGroupNodes:
    """Tests for ``group_nodes``."""

    @pytest.mark.parametrize(
        ("predicate", "relation", "point_ids"),
        [
            (None, "covers", [1, 2]),
            ("within", "contains", [1]),
        ],
    )
    def test_polygon_membership_at_boundary(
        self, predicate: str | None, relation: str, point_ids: list[int]
    ) -> None:
        """The default includes the boundary point, while strict containment excludes it."""
        polygons, points = make_poly_points_pair()
        nodes, layers = (
            group_nodes(polygons, points)
            if predicate is None
            else group_nodes(polygons, points, predicate=predicate)
        )
        assert set(nodes) == {"polygon", "point"}
        key = ("polygon", relation, "point")
        assert set(layers) == {key}
        edges = layers[key]
        assert sorted(edges.index) == [("A", pid) for pid in point_ids]
        assert {"weight", "geometry"} <= set(edges.columns)

    @pytest.mark.parametrize("empty_layer", ["point", "polygon"])
    def test_empty_membership_preserves_both_layers(self, empty_layer: str) -> None:
        """Empty membership keeps each input's nodes and the typed edge table."""
        polygons, points = make_poly_points_pair()
        if empty_layer == "point":
            points = points.iloc[:0]
        else:
            polygons = polygons.iloc[:0]
        nodes, edges = group_nodes(polygons, points)
        pd.testing.assert_index_equal(nodes["point"].index, points.index)
        pd.testing.assert_index_equal(nodes["polygon"].index, polygons.index)
        assert edges[("polygon", "covers", "point")].empty

    def test_group_nodes_network_metric_requires_network(self) -> None:
        """Network metric without network_gdf raises explicit error for group_nodes."""
        polys, pts = make_poly_points_pair()
        # group_nodes takes two positional args, so we can't use the helper directly
        with pytest.raises(ValueError, match="network_gdf is required"):
            group_nodes(polys, pts, distance_metric="network")

    def test_group_nodes_node_geom_col_missing(self) -> None:
        """Missing node_geom_col should raise a clear error."""
        polys, pts = make_poly_points_pair()
        with pytest.raises(ValueError, match="node_geom_col 'missing' not found"):
            group_nodes(polys, pts, node_geom_col="missing")

    def test_group_nodes_node_geom_col_override(self) -> None:
        """Custom polygon point column drives edge geometries."""
        polys, pts = make_poly_points_pair()
        custom = Point(10, 0)
        polys = polys.assign(node_pt=[custom])
        _, edges = group_nodes(polys, pts, node_geom_col="node_pt")
        edge_key = next(iter(edges))
        geoms = edges[edge_key].geometry
        assert all(tuple(map(float, geom.coords[0])) == (custom.x, custom.y) for geom in geoms)

    def test_group_nodes_set_point_nodes_preserves_original_geometry(self) -> None:
        """set_point_nodes swaps polygon geometry to points and keeps originals."""
        polys, pts = make_poly_points_pair()
        nodes, edges = group_nodes(polys, pts, set_point_nodes=True)
        poly_nodes = nodes["polygon"]
        assert "original_geometry" in poly_nodes.columns
        assert poly_nodes.geometry.equals(polys.geometry.centroid)
        assert poly_nodes["original_geometry"].equals(polys.geometry)
        edge_key = next(iter(edges))
        assert len(edges[edge_key]) == 2

    def test_group_nodes_invalid_metric(self) -> None:
        """Unsupported distance metric raises ValueError via public API."""
        polys, pts = make_poly_points_pair()
        with pytest.raises(ValueError, match="Unknown distance metric"):
            group_nodes(polys, pts, distance_metric="invalid")  # type: ignore[arg-type]

    def test_group_nodes_crs_mismatch(self) -> None:
        """CRS mismatch path triggers ValueError."""
        polys, pts = make_poly_points_pair()
        pts2 = pts.to_crs("EPSG:4326")
        with pytest.raises(ValueError, match="same CRS"):
            group_nodes(polys, pts2)

    def test_group_nodes_missing_crs(self) -> None:
        """Both layers must declare their CRS."""
        polygons, points = make_poly_points_pair()
        with pytest.raises(ValueError, match="CRS"):
            group_nodes(
                polygons.set_crs(None, allow_override=True),
                points.set_crs(None, allow_override=True),
            )

    def test_group_nodes_network_crs_mismatch(self) -> None:
        """Network metric with mismatched network CRS raises."""
        polys, pts = make_poly_points_pair()
        # Simple network in different CRS
        poly_centroid = polys.geometry.iloc[0].centroid
        net = gpd.GeoDataFrame(
            {"geometry": [LineString([poly_centroid, poly_centroid])]},
            index=pd.MultiIndex.from_arrays([[0], [1]], names=("source_id", "target_id")),
            crs="EPSG:4326",
        )
        with pytest.raises(ValueError, match="CRS mismatch between inputs and network"):
            group_nodes(polys, pts, distance_metric="network", network_gdf=net)

    @pytest.mark.parametrize(("use_cost", "expected"), [(False, 1.0), (True, 7.0)])
    def test_group_nodes_network_weight(self, use_cost: bool, expected: float) -> None:
        """The interior point is at the centroid and the boundary point is one road away."""
        polygons, points = make_poly_points_pair()
        road = make_network_edges([0], [1], [LineString([(1, 1), (2, 1)])], crs=points.crs)
        if use_cost:
            road["length"] = [7.0]
        _, layers = group_nodes(
            polygons,
            points,
            distance_metric="network",
            network_gdf=road,
            network_weight="length" if use_cost else None,
        )
        edges = next(iter(layers.values()))
        assert edges["weight"].to_dict() == {("A", 1): 0.0, ("A", 2): expected}

    def test_group_nodes_no_matches_nonempty(self) -> None:
        """Non-empty inputs but no spatial matches hit empty-join early branch."""
        # Polygon far from points to ensure no containment matches
        polys = gpd.GeoDataFrame(
            {"name": ["A"]},
            geometry=[Polygon([(10, 10), (12, 10), (12, 12), (10, 12)])],
            crs="EPSG:3857",
        ).set_index(pd.Index(["A"], name="zone"))
        pts = gpd.GeoDataFrame(
            {"pid": [1, 2]},
            geometry=[Point(0, 0), Point(1, 1)],
            crs="EPSG:3857",
        ).set_index("pid")
        _nodes, edges = group_nodes(polys, pts)
        edge_key = next(iter(edges))
        assert edges[edge_key].empty

    def test_group_nodes_rejects_mismatched_crs_for_empty_layer(self) -> None:
        """An empty point layer still needs to match the polygon CRS."""
        polygons, points = make_poly_points_pair()
        with pytest.raises(ValueError, match="CRS mismatch between inputs"):
            group_nodes(polygons, points.iloc[:0].to_crs(4326))


class TestBridgeNodes:
    """Tests for ``bridge_nodes``."""

    @pytest.mark.parametrize("radius", [-2.0, float("nan"), float("inf")])
    def test_fixed_radius_rejects_an_invalid_distance(self, radius: float) -> None:
        """Layer bridging validates radii before selecting any edges."""
        points = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(1, 0)], crs=3857)
        with pytest.raises(ValueError, match="radius must be finite and non-negative"):
            bridge_nodes(
                {"source": points, "target": points},
                proximity_method="fixed_radius",
                radius=radius,
            )

    def test_bridge_nodes_fixed_radius_requires_a_radius(self) -> None:
        """The missing distance raises ValueError as documented."""
        points = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(1, 0)], crs=3857)
        with pytest.raises(ValueError, match="radius is required"):
            bridge_nodes({"source": points, "target": points}, proximity_method="fixed_radius")

    def test_bridge_nodes_type_filters(self, small_points: gpd.GeoDataFrame) -> None:
        """bridge_nodes can restrict sources/targets when filters are provided."""
        layers = {
            "a": small_points.iloc[[0]],
            "b": small_points.iloc[[1]],
            "c": small_points.iloc[[2]],
        }

        _, all_edges = bridge_nodes(layers, k=1)
        assert set(all_edges) == {
            ("a", "is_nearby", "b"),
            ("a", "is_nearby", "c"),
            ("b", "is_nearby", "a"),
            ("b", "is_nearby", "c"),
            ("c", "is_nearby", "a"),
            ("c", "is_nearby", "b"),
        }

        _, filtered = bridge_nodes(
            layers,
            k=1,
            source_node_types=["a"],
            target_node_types=["b", "c"],
        )
        assert set(filtered) == {("a", "is_nearby", "b"), ("a", "is_nearby", "c")}

        _, target_only = bridge_nodes(layers, k=1, target_node_types=["b"])
        assert set(target_only) == {("a", "is_nearby", "b"), ("c", "is_nearby", "b")}

        with pytest.raises(ValueError, match="Unknown source node types"):
            bridge_nodes(layers, k=1, source_node_types=["missing"])

    def test_bridge_nodes_errors_and_nx(self) -> None:
        """Cover bridge_nodes error branches and as_nx=True path."""
        p1 = make_single_point()

        # < 2 layers
        with pytest.raises(ValueError, match="needs at least two layers"):
            bridge_nodes({"a": p1})

        # Invalid method
        with pytest.raises(ValueError, match="proximity_method must be 'knn' or 'fixed_radius'"):
            bridge_nodes({"a": p1, "b": p1}, proximity_method="invalid")

        # Check networkx output

        other = make_points_simple([(1, 0)]).rename(index={0: 1})
        G = cast("nx.Graph", bridge_nodes({"a": p1, "b": other}, k=1, as_nx=True))
        assert G.number_of_nodes() == 2
        assert G.number_of_edges() == 2

    def test_bridge_nodes_network_fixed_radius_exact_edges(self) -> None:
        """bridge_nodes fixed_radius with network metric respects the radius cutoff."""
        pts, net = _make_chain_points_and_network()
        layers = {"a": pts.iloc[[0, 3]], "b": pts.iloc[[1, 2]]}
        _, edges = bridge_nodes(
            layers,
            proximity_method="fixed_radius",
            radius=1.0,
            distance_metric="network",
            network_gdf=net,
        )

        expected = {
            ("a", "is_nearby", "b"): {(0, 1), (3, 2)},
            ("b", "is_nearby", "a"): {(1, 0), (2, 3)},
        }
        assert set(edges) == set(expected)
        for relation, pairs in expected.items():
            gdf = edges[relation]
            assert set(gdf.index) == pairs
            assert all(w == pytest.approx(1.0) for w in gdf["weight"])

    @pytest.mark.parametrize("method", ["knn", "fixed_radius"])
    @pytest.mark.parametrize("empty_side", [None, "source", "target", "both"])
    def test_network_bridge_preserves_cleaned_and_empty_layers(
        self, method: str, empty_side: str | None
    ) -> None:
        """Shared searches use the same validated inputs as individual directed calls."""
        frame = gpd.GeoDataFrame(
            geometry=[Point(0, 0), None, Point(), Point(1, 0)], index=["a", "b", "c", "d"], crs=3857
        )
        source = frame.iloc[:0] if empty_side in {"source", "both"} else frame
        target = frame.iloc[:0] if empty_side in {"target", "both"} else frame
        network = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 0)])], crs=3857)
        layers = {"source": source, "target": target}
        nodes, edges = bridge_nodes(
            layers,
            source_node_types=["source"],
            proximity_method=method,
            distance_metric="network",
            network_gdf=network,
            k=1,
            radius=0,
        )
        assert nodes is layers
        result = edges[("source", "is_nearby", "target")]
        assert list(result.index) == ([] if empty_side else [("a", "a"), ("d", "d")])
        assert result.crs == frame.crs
        assert (result["weight"] == 0).all()

    @pytest.mark.parametrize("k", [0, -1])
    def test_network_bridge_nonpositive_k(self, k: int) -> None:
        """Sharing a source traversal does not change non-positive k semantics."""
        frame = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(1, 0)], crs=3857)
        network = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 0)])], crs=3857)
        _, edges = bridge_nodes(
            {"source": frame, "target": frame},
            k=k,
            distance_metric="network",
            network_gdf=network,
        )
        assert all(result.empty for result in edges.values())

    def test_bridge_searches_are_shared_across_target_layers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every source traversal serves both destination layers."""
        roads = gpd.GeoDataFrame(
            geometry=[LineString([(i, 0), (i + 1, 0)]) for i in range(10)], crs=3857
        )
        layers = {
            str(i): gpd.GeoDataFrame(geometry=[Point(2 * i, 0), Point(2 * i + 1, 0)], crs=3857)
            for i in range(3)
        }
        sparse_search, networkx_search = _count_searches(monkeypatch)
        _, edges = bridge_nodes(layers, k=1, distance_metric="network", network_gdf=roads)
        assert _searched_origins(sparse_search) == 6
        assert networkx_search.call_count == 0
        assert len(edges) == 6
        assert all(len(frame) == 2 for frame in edges.values())


@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
def test_knn_and_fixed_radius_basic(small_points: gpd.GeoDataFrame, metric: str) -> None:
    """Core sanity checks for kNN/fixed-radius under both Euclidean & Manhattan."""
    nodes, edges = knn_graph(small_points, k=2, distance_metric=metric)
    assert_valid_proximity_result(nodes, edges, len(small_points))

    nodes_r, edges_r = fixed_radius_graph(small_points, radius=1.5, distance_metric=metric)
    assert_valid_proximity_result(nodes_r, edges_r, len(small_points))
    assert {frozenset(pair) for pair in edges.index} == {
        frozenset(pair) for pair in [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
    }
    expected_radius = [(0, 1), (0, 2)]
    if metric == "euclidean":
        expected_radius.append((1, 2))
    assert {frozenset(pair) for pair in edges_r.index} == {
        frozenset(pair) for pair in expected_radius
    }
    expected_weights = [1, 1, 2**0.5] if metric == "euclidean" else [1, 1]
    assert sorted(edges_r["weight"]) == pytest.approx(expected_weights)

    if metric == "manhattan" and not edges.empty:
        # Manhattan edges have 3 coordinate points (L path) unless aligned
        geom = edges.geometry.iloc[0]
        assert isinstance(geom, LineString)
        assert 2 <= len(geom.coords) <= 3


def test_triangulation_hierarchy_matches_brute_force() -> None:
    """MST ⊆ RNG ⊆ Gabriel ⊆ Delaunay, and Gabriel matches its brute-force definition.

    Uses large projected-CRS-like coordinates to guard against floating-point
    regressions in the disc-emptiness predicate (previously valid Gabriel
    edges were dropped when coordinates were far from the origin).
    """
    rng = np.random.default_rng(42)
    coords = rng.uniform(0.0, 5000.0, size=(150, 2)) + np.array([350_000.0, 5_700_000.0])
    pts = make_points_simple([(float(x), float(y)) for x, y in coords], crs="EPSG:32633")

    def edge_set(edges: gpd.GeoDataFrame) -> set[tuple[int, int]]:
        return {(min(int(u), int(v)), max(int(u), int(v))) for u, v in edges.index}

    _, d_edges = delaunay_graph(pts)
    _, g_edges = gabriel_graph(pts)
    _, r_edges = relative_neighborhood_graph(pts)
    _, m_edges = euclidean_minimum_spanning_tree(pts)
    delaunay, gabriel, rng_graph, mst = map(edge_set, (d_edges, g_edges, r_edges, m_edges))

    # Brute-force Gabriel: keep a Delaunay edge iff no other point lies
    # strictly inside the open disc having the edge as its diameter.
    expected_gabriel = set()
    for i, j in delaunay:
        mid = 0.5 * (coords[i] + coords[j])
        rad2 = np.sum((coords[i] - coords[j]) ** 2) * 0.25
        d2 = np.sum((coords - mid) ** 2, axis=1)
        d2[[i, j]] = np.inf
        if not np.any(d2 < rad2):
            expected_gabriel.add((i, j))

    assert gabriel == expected_gabriel
    assert mst <= rng_graph <= gabriel <= delaunay


@pytest.mark.parametrize("builder", [gabriel_graph, relative_neighborhood_graph])
def test_two_point_graph_has_the_unique_edge(
    builder: Callable[..., tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]],
) -> None:
    """Two points determine one edge without triangulation."""
    _, edges = builder(make_points_simple([(0, 0), (1, 0)]))
    assert _undirected_edge_weights(edges) == {(0, 1): 1}


@pytest.mark.parametrize(
    ("generator_func", "kwargs"),
    [
        (delaunay_graph, {}),
        (gabriel_graph, {}),
        (relative_neighborhood_graph, {}),
        (euclidean_minimum_spanning_tree, {}),
        (fixed_radius_graph, {"radius": 1.0}),
        (waxman_graph, {"beta": 0.5, "r0": 1.0}),
    ],
    ids=["delaunay", "gabriel", "rng", "mst", "fixed_radius", "waxman"],
)
def test_generators_insufficient_points_early_exit(
    generator_func: Callable[..., tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]],
    kwargs: dict[str, object],
) -> None:
    """A single point remains an isolated node in every generator."""
    point = make_single_point()
    nodes, edges = generator_func(point, **kwargs)
    pd.testing.assert_index_equal(nodes.index, point.index)
    assert edges.empty


@pytest.mark.parametrize("method", ["knn", "radius"])
@pytest.mark.parametrize("empty_side", ["source", "target", "both"])
def test_directed_empty_inputs(method: str, empty_side: str) -> None:
    """Empty layers produce typed empty edges and retain the remaining nodes."""
    frame = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(1, 0)], index=["a", "b"], crs=3857)
    source = frame.iloc[:0] if empty_side in {"source", "both"} else frame
    target = frame.iloc[:0] if empty_side in {"target", "both"} else frame
    nodes, edges = (
        knn_graph(source, k=1, target_gdf=target)
        if method == "knn"
        else fixed_radius_graph(source, radius=2, target_gdf=target)
    )
    assert len(nodes) == len(source) + len(target)
    assert edges.empty
    assert isinstance(edges.index, pd.MultiIndex)
    assert edges.crs == frame.crs


def test_directed_variants_and_bridge(small_points: gpd.GeoDataFrame) -> None:
    """Directed kNN & radius plus bidirectional layer bridging produce edges."""
    src = small_points.iloc[:2]
    dst = small_points.iloc[2:]
    _, k_edges = knn_graph(src, k=1, target_gdf=dst)
    _, r_edges = fixed_radius_graph(src, radius=3.0, target_gdf=dst)
    assert len(k_edges) == len(src)
    assert len(r_edges) >= 1

    # bridge_nodes over two layers (both methods)
    layers = {"a": src, "b": dst}
    _, b_knn = bridge_nodes(layers, proximity_method="knn", k=1)
    _, b_rad = bridge_nodes(layers, proximity_method="fixed_radius", radius=5.0)
    assert len(b_knn) == 2  # a->b, b->a
    assert len(b_rad) == 2


def test_layer_builder_as_nx_warning(small_points: gpd.GeoDataFrame) -> None:
    """Layer builders attribute the warning to the public caller."""
    assert_as_nx_deprecation(contiguity_graph, make_square_polygons())
    assert_as_nx_deprecation(group_nodes, *make_poly_points_pair())
    assert_as_nx_deprecation(
        bridge_nodes, {"a": small_points.iloc[:2], "b": small_points.iloc[2:]}, k=1
    )


@pytest.mark.parametrize(
    ("builder", "expected_weights"),
    [
        (delaunay_graph, {(0, 1): 2, (0, 2): 5, (1, 2): 3}),
        (gabriel_graph, {(0, 1): 2, (0, 2): 5, (1, 2): 3}),
        (relative_neighborhood_graph, {(0, 1): 2, (0, 2): 5}),
        (euclidean_minimum_spanning_tree, {(0, 1): 2, (1, 2): 3}),
    ],
)
def test_network_weighted_graphs(
    builder: Callable[..., tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]],
    expected_weights: dict[tuple[int, int], int],
) -> None:
    """The 3-4-5 triangle has a two-road network with costs 2 and 3."""
    points = make_points_simple([(0, 0), (3, 0), (0, 4)])
    roads = make_network_edges(
        [0, 1],
        [1, 2],
        [LineString([(0, 0), (3, 0)]), LineString([(3, 0), (0, 4)])],
        extra_attrs={"cost": [2, 3]},
    )
    nodes, edges = builder(
        points, distance_metric="network", network_gdf=roads, network_weight="cost"
    )
    pd.testing.assert_index_equal(nodes.index, points.index)
    assert len(edges) == len(expected_weights)
    assert _undirected_edge_weights(edges) == expected_weights


def _without_sparse_network(*_args: object) -> None:
    """Keep every network search in NetworkX."""


def _parity_network(case: str) -> gpd.GeoDataFrame:
    """Return a seeded grid network with one distance edge case."""
    rng = np.random.default_rng(3)
    lines = [LineString([(x, y), (x + 1, y)]) for x in range(5) for y in range(6)]
    lines += [LineString([(x, y), (x, y + 1)]) for x in range(6) for y in range(5)]
    cost: np.ndarray = np.ones(len(lines)) if case == "ties" else rng.uniform(1, 2, len(lines))
    if case == "integer-weight":
        cost = rng.integers(1, 4, len(lines))
    if case == "zero-weight":
        cost[rng.choice(len(lines), 15, replace=False)] = 0
    elif case == "infinite-weight":
        cost[rng.choice(len(lines), 10, replace=False)] = np.inf
        # NetworkX also traverses an impassable edge, reporting an infinite distance.
        lines += [LineString([(5, 5), (20, 20)]), LineString([(20, 20), (21, 20)])]
        cost = np.concatenate([cost, [np.inf, 1]])
    elif case == "parallel":
        # Duplicate rows join the same nodes with a second cost.
        lines += lines[:20]
        cost = np.concatenate([cost, rng.uniform(0.5, 2.5, 20)])
    elif case == "unreachable":
        lines += [LineString([(20, 20), (21, 20)]), LineString([(21, 20), (21, 21)])]
        cost = np.concatenate([cost, [1, 1]])
    return gpd.GeoDataFrame({"cost": cost}, geometry=lines, crs=3857)


def _parity_points(case: str, count: int, seed: int) -> gpd.GeoDataFrame:
    """Return seeded points on the parity network, some in a separate component."""
    xy = np.random.default_rng(seed).uniform(0, 5, (count, 2))
    if case in {"unreachable", "infinite-weight"}:
        xy[::4] = xy[::4] / 5 + 20
    return gpd.GeoDataFrame(geometry=gpd.points_from_xy(*xy.T), crs=3857)


def _network_proximity_results(case: str) -> list[gpd.GeoDataFrame]:
    """Run every network proximity builder on one parity input."""
    network = {"distance_metric": "network", "network_gdf": _parity_network(case)}
    network["network_weight"] = "cost"
    points = _parity_points(case, 24, 0)
    others = _parity_points(case, 10, 1)
    results = [
        knn_graph(points, k=4, **network)[1],
        knn_graph(points, k=3, target_gdf=others, **network)[1],
        waxman_graph(points, beta=0.8, r0=2, seed=5, **network)[1],
        euclidean_minimum_spanning_tree(points, **network)[1],
        fixed_radius_graph(points, radius=2.0, **network)[1],
    ]
    _, bridges = bridge_nodes({"a": points, "b": others}, k=2, **network)
    return results + list(bridges.values())


@pytest.mark.parametrize(
    "case", ["ties", "integer-weight", "zero-weight", "infinite-weight", "parallel", "unreachable"]
)
def test_network_searches_match_networkx(monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    """Sparse network searches keep the NetworkX distances, edges and path geometries."""
    with monkeypatch.context() as patch:
        patch.setattr("city2graph.proximity.SparseNetwork.from_graph", _without_sparse_network)
        expected = _network_proximity_results(case)
    sparse_search, _ = _count_searches(monkeypatch)
    actual = _network_proximity_results(case)

    assert sparse_search.call_count > 0
    for result, reference in zip(actual, expected, strict=True):
        pd.testing.assert_index_equal(result.index, reference.index)
        np.testing.assert_array_equal(result["weight"], reference["weight"])
        assert result["weight"].dtype == reference["weight"].dtype
        assert list(result.geometry.to_wkb()) == list(reference.geometry.to_wkb())


@pytest.mark.parametrize(
    ("builder_name", "metric"),
    [
        ("knn", "euclidean"),
        ("knn", "manhattan"),
        ("knn", "network"),
        ("fixed_radius", "euclidean"),
        ("fixed_radius", "network"),
        ("mst", "euclidean"),
        ("mst", "network"),
        ("knn_directed", "network"),
        ("fixed_radius_directed", "network"),
    ],
)
def test_proximity_builders_avoid_dense_matrix(
    small_points: gpd.GeoDataFrame,
    network_edges: gpd.GeoDataFrame,
    monkeypatch: pytest.MonkeyPatch,
    builder_name: str,
    metric: str,
) -> None:
    """Graph builders never call DistanceMetric.matrix (regression for issue #190)."""
    monkeypatch.setattr("city2graph.proximity.DistanceMetric.matrix", _raise_dense_matrix_error)
    net = network_edges if metric == "network" else None
    src = small_points.iloc[:2]
    dst = small_points.iloc[2:]

    if builder_name == "knn":
        nodes, edges = knn_graph(small_points, k=1, distance_metric=metric, network_gdf=net)
        assert_valid_proximity_result(nodes, edges, len(small_points), allow_empty_edges=False)
    elif builder_name == "fixed_radius":
        nodes, edges = fixed_radius_graph(
            small_points, radius=2.5, distance_metric=metric, network_gdf=net
        )
        assert_valid_proximity_result(nodes, edges, len(small_points))
    elif builder_name == "mst":
        nodes, edges = euclidean_minimum_spanning_tree(
            small_points, distance_metric=metric, network_gdf=net
        )
        assert len(edges) <= max(len(small_points) - 1, 0)
    elif builder_name == "knn_directed":
        nodes, edges = knn_graph(src, k=1, distance_metric=metric, network_gdf=net, target_gdf=dst)
        assert_valid_proximity_result(nodes, edges, len(small_points))
    else:  # fixed_radius_directed
        nodes, edges = fixed_radius_graph(
            src, radius=2.5, distance_metric=metric, network_gdf=net, target_gdf=dst
        )
        assert_valid_proximity_result(nodes, edges, len(small_points))


@pytest.mark.parametrize(
    ("geometry_column", "secondary_geometry"),
    [
        pytest.param("geometry", False, id="default"),
        pytest.param("shape", False, id="renamed"),
        pytest.param("shape", True, id="secondary-geometry"),
    ],
)
def test_network_proximity_preserves_curved_path_geometry(
    monkeypatch: pytest.MonkeyPatch, geometry_column: str, secondary_geometry: bool
) -> None:
    """A selected network path follows the linework in either direction."""
    points = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(4, 0)], index=["west", "east"], crs=3857)
    network = gpd.GeoDataFrame(
        geometry=[
            LineString([(0, 0), (1, 3), (2, 0)]),
            LineString([(4, 0), (3, 2), (2, 0)]),
        ],
        crs=3857,
    )
    if geometry_column != "geometry":
        network = network.rename_geometry(geometry_column)
    if secondary_geometry:
        network["geometry"] = [
            LineString([(0, 0), (1, -1), (2, 0)]),
            LineString([(4, 0), (3, -1), (2, 0)]),
        ]
    extract_coordinates = Mock(wraps=get_coordinates)
    monkeypatch.setattr("city2graph.proximity.get_coordinates", extract_coordinates)

    _, result = bridge_nodes(
        {"start": points.iloc[:1], "end": points.iloc[1:]},
        k=1,
        distance_metric="network",
        network_gdf=network,
    )

    route = LineString([(0, 0), (1, 3), (2, 0), (3, 2), (4, 0)])
    forward = result[("start", "is_nearby", "end")]
    reverse = result[("end", "is_nearby", "start")]
    assert forward.geometry.iloc[0].equals_exact(route, 0)
    assert reverse.geometry.iloc[0].equals_exact(route.reverse(), 0)
    assert forward["weight"].iloc[0] == pytest.approx(route.length)
    assert reverse["weight"].iloc[0] == pytest.approx(route.length)
    assert extract_coordinates.call_count == len(network)

    _, nearby = fixed_radius_graph(points, radius=6, distance_metric="network", network_gdf=network)
    assert nearby.empty


@pytest.mark.parametrize(
    "build",
    [
        lambda gdf, **kw: knn_graph(gdf, k=2, **kw),
        delaunay_graph,
        gabriel_graph,
        relative_neighborhood_graph,
        euclidean_minimum_spanning_tree,
        lambda gdf, **kw: fixed_radius_graph(gdf, radius=1.5, **kw),
        lambda gdf, **kw: waxman_graph(gdf, beta=0.9, r0=2.0, seed=42, **kw),
    ],
    ids=["knn", "delaunay", "gabriel", "rng", "emst", "fixed_radius", "waxman"],
)
def test_duplicate_edges_adds_reverse_rows(
    small_points: gpd.GeoDataFrame,
    build: Callable[..., tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]],
) -> None:
    """duplicate_edges=True adds the reverse row of every undirected edge."""
    _, base_edges = build(small_points)
    _, dup_edges = build(small_points, duplicate_edges=True)

    pairs = set(base_edges.index)
    assert len(dup_edges) == 2 * len(base_edges)
    assert set(dup_edges.index) == pairs | {(v, u) for u, v in pairs}


def test_duplicate_edges_rejects_target_gdf(small_points: gpd.GeoDataFrame) -> None:
    """duplicate_edges=True with target_gdf (directed variant) raises ValueError."""
    with pytest.raises(ValueError, match="not supported with target_gdf"):
        knn_graph(small_points, k=2, target_gdf=small_points, duplicate_edges=True)
    with pytest.raises(ValueError, match="not supported with target_gdf"):
        fixed_radius_graph(small_points, radius=1.5, target_gdf=small_points, duplicate_edges=True)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("method", ["knn", "radius"])
def test_proximity_uses_cleaned_geometries(directed: bool, method: str) -> None:
    """Missing and empty geometries are removed before querying neighbours."""
    frame = gpd.GeoDataFrame(
        geometry=[Point(0, 0), None, Point(), Point(1, 0)],
        index=["a", "missing", "empty", "b"],
        crs=3857,
    )
    target = frame if directed else None
    nodes, edges = (
        knn_graph(frame, k=1, target_gdf=target)
        if method == "knn"
        else fixed_radius_graph(frame, radius=2, target_gdf=target)
    )
    assert len(nodes) == (4 if directed else 2)
    assert not edges.empty
    assert nodes.geometry.notna().all()
    assert not nodes.geometry.is_empty.any()
    assert len(frame) == 4


@pytest.mark.parametrize("method", ["knn", "radius"])
@pytest.mark.parametrize("directed", [False, True])
def test_proximity_all_invalid_geometries(method: str, directed: bool) -> None:
    """Cleaning an entirely invalid layer produces an empty graph."""
    frame = gpd.GeoDataFrame(geometry=[None, Point()], crs=3857)
    target = frame if directed else None
    nodes, edges = (
        knn_graph(frame, k=1, target_gdf=target)
        if method == "knn"
        else fixed_radius_graph(frame, radius=2, target_gdf=target)
    )
    assert nodes.empty
    assert edges.empty
    assert edges.crs == frame.crs


@pytest.mark.parametrize(
    ("builder", "kwargs"),
    [
        (knn_graph, {"k": 1}),
        (delaunay_graph, {}),
        (gabriel_graph, {}),
        (relative_neighborhood_graph, {}),
        (euclidean_minimum_spanning_tree, {}),
        (fixed_radius_graph, {"radius": 5}),
        (waxman_graph, {"beta": 0.5, "r0": 5, "seed": 0}),
    ],
)
def test_point_builder_as_nx_warning(
    builder: Callable[..., object], kwargs: dict[str, float], small_points: gpd.GeoDataFrame
) -> None:
    """Point builders warn only on explicit as_nx."""
    assert_as_nx_deprecation(builder, small_points, **kwargs)
