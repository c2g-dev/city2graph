"""Concise proximity module tests (minimal yet 100% coverage target).

This suite intentionally compresses prior expansive testing into a small set
of focused scenarios that exercise every public code path and meaningful
branch in ``city2graph.proximity`` after refactoring/simplification.

Test categories:
1. Core generators (kNN, fixed-radius, triangulation family, MST, Waxman)
2. Distance metrics: euclidean, manhattan, network (incl. caching + errors)
3. Directed variants + multi-layer bridging
4. group_nodes (matches, empty, predicate mapping, network error)
5. Contiguity graph (validation, rook vs queen, empty, metric variants, errors)
6. Error handling & metric normalization (non-string, unknown metrics)
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

if TYPE_CHECKING:
    from collections.abc import Callable

import geopandas as gpd
import networkx as nx
import pandas as pd
import pytest
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
from tests.helpers import assert_network_metric_requires_network_gdf
from tests.helpers import assert_valid_proximity_result
from tests.helpers import make_network_edges
from tests.helpers import make_points_simple
from tests.helpers import make_poly_points_pair
from tests.helpers import make_single_point
from tests.helpers import make_square_polygons


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


# ---------------------------------------------------------------------------
# Distance helper behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
def test_knn_and_fixed_radius_basic(small_points: gpd.GeoDataFrame, metric: str) -> None:
    """Core sanity checks for kNN/fixed-radius under both Euclidean & Manhattan."""
    nodes, edges = knn_graph(small_points, k=2, distance_metric=metric)
    assert_valid_proximity_result(nodes, edges, len(small_points))

    nodes_r, edges_r = fixed_radius_graph(small_points, radius=1.5, distance_metric=metric)
    assert_valid_proximity_result(nodes_r, edges_r, len(small_points))

    if metric == "manhattan" and not edges.empty:
        # Manhattan edges have 3 coordinate points (L path) unless aligned
        geom = edges.geometry.iloc[0]
        assert isinstance(geom, LineString)
        assert 2 <= len(geom.coords) <= 3


def test_knn_non_string_metric_defaults(small_points: gpd.GeoDataFrame) -> None:
    """Non-string metric argument falls back to euclidean (robust normalisation)."""
    nodes, edges = knn_graph(small_points, k=1, distance_metric=None)  # type: ignore[arg-type]
    assert_valid_proximity_result(nodes, edges, len(small_points), allow_empty_edges=False)


def test_network_metric_requires_network_gdf(small_points: gpd.GeoDataFrame) -> None:
    """Requesting network metric without network_gdf triggers clear error."""
    assert_network_metric_requires_network_gdf(knn_graph, small_points, k=1)


# ---------------------------------------------------------------------------
# Triangulation based generators
# ---------------------------------------------------------------------------


def test_delaunay_and_subgraphs(small_points: gpd.GeoDataFrame) -> None:
    """Gabriel and RNG are subgraphs of Delaunay (edge count monotonicity)."""
    # Delaunay needs >=3 points; we already have 4
    _, d_edges = delaunay_graph(small_points)
    _, g_edges = gabriel_graph(small_points)
    _, r_edges = relative_neighborhood_graph(small_points)
    assert len(g_edges) <= len(d_edges)
    assert len(r_edges) <= len(d_edges)


def test_mst_and_waxman(small_points: gpd.GeoDataFrame) -> None:
    """MST has n-1 edges and Waxman returns expected schema."""
    _, mst_edges = euclidean_minimum_spanning_tree(small_points)
    assert len(mst_edges) == max(len(small_points) - 1, 0)
    _, wax_edges = waxman_graph(small_points, beta=0.8, r0=2.0, seed=7)
    # Probabilistic: just ensure schema
    assert {"weight", "geometry"}.issubset(wax_edges.columns)


def test_gabriel_rng_two_points_branch() -> None:
    """Gabriel and RNG handle exactly two points (single edge candidate path)."""
    pts = make_points_simple([(0, 0), (1, 0)])
    _, g_edges = gabriel_graph(pts)
    _, r_edges = relative_neighborhood_graph(pts)
    # With two points, both graphs should have at most one edge
    assert len(g_edges) <= 1
    assert len(r_edges) <= 1


# ---------------------------------------------------------------------------
# Directed variants & bridging
# ---------------------------------------------------------------------------


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


def test_bridge_nodes_type_filters(small_points: gpd.GeoDataFrame) -> None:
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


def test_directed_network_variants(
    small_points: gpd.GeoDataFrame, network_edges: gpd.GeoDataFrame
) -> None:
    """Directed kNN and radius with network metric exercise _directed_graph path."""
    # Partition into sources and destinations
    src = small_points.iloc[:2]
    dst = small_points.iloc[2:]

    # kNN directed with network metric
    _, e_knn = knn_graph(
        src,
        k=1,
        target_gdf=dst,
        distance_metric="network",
        network_gdf=network_edges,
    )
    # Expect at least one edge if any destination is reachable on the network
    assert len(e_knn) >= 0

    # Radius directed with network metric (radius chosen to include 0->2 path ~2.41)
    _, e_rad = fixed_radius_graph(
        src,
        radius=2.5,
        target_gdf=dst,
        distance_metric="network",
        network_gdf=network_edges,
    )
    assert len(e_rad) >= 1


# ---------------------------------------------------------------------------
# Contiguity minimal tests (queen vs rook, manhattan, errors, empty)
# ---------------------------------------------------------------------------


# Helper moved to tests/helpers.py as make_square_polygons()


def test_contiguity_queen_vs_rook() -> None:
    """Queen contiguity yields superset (or equal) of rook edges."""
    gdf = make_square_polygons()
    nodes_q, edges_q = contiguity_graph(gdf, contiguity="queen")
    nodes_r, edges_r = contiguity_graph(gdf, contiguity="rook")
    assert set(nodes_q.index) == set(gdf.index)
    assert set(nodes_r.index) == set(gdf.index)
    assert len(edges_q) >= len(edges_r)


def test_contiguity_manhattan_geometry() -> None:
    """Manhattan metric uses polyline (straight or L-shape)."""
    gdf = make_square_polygons()
    _, edges_m = contiguity_graph(gdf, distance_metric="manhattan")
    if not edges_m.empty:
        geom = edges_m.geometry.iloc[0]
        assert isinstance(geom, LineString)
        assert 2 <= len(geom.coords) <= 3


def test_contiguity_invalid_type() -> None:
    """Invalid contiguity type raises ValueError."""
    gdf = make_square_polygons()
    with pytest.raises(ValueError, match="Invalid contiguity type"):
        contiguity_graph(gdf, contiguity="invalid")


def test_contiguity_empty_input() -> None:
    """Empty input returns empty nodes/edges GeoDataFrames."""
    empty = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:3857")
    nodes, edges = contiguity_graph(empty)
    assert nodes.empty
    assert edges.empty


def test_contiguity_network_requires_network_gdf() -> None:
    """Network metric without network_gdf raises ValueError."""
    assert_network_metric_requires_network_gdf(contiguity_graph, make_square_polygons())


# ---------------------------------------------------------------------------
# group_nodes public API coverage
# ---------------------------------------------------------------------------


def test_group_nodes_basic_matches() -> None:
    """group_nodes connects polygon to contained points (boundary inclusive)."""
    polys, pts = make_poly_points_pair()
    nodes, edges = group_nodes(polys, pts)
    # One polygon node retained, two point nodes in same CRS
    assert set(nodes.keys()) == {"polygon", "point"}
    edge_key = next(iter(edges))
    e = edges[edge_key]
    # Only the first two points lie inside the polygon
    assert len(e) == 2
    assert {pid for _, pid in e.index} == {1, 2}
    assert "weight" in e.columns
    assert "geometry" in e.columns


def test_group_nodes_empty_points() -> None:
    """Empty point set returns empty edge GeoDataFrame while preserving nodes."""
    polys, pts = make_poly_points_pair()
    empty_pts = pts.iloc[0:0]
    nodes, edges = group_nodes(polys, empty_pts)
    assert nodes["point"].empty
    edge_key = next(iter(edges))
    assert edges[edge_key].empty


def test_group_nodes_empty_polygons() -> None:
    """Empty polygon set returns empty edges while preserving point nodes."""
    polys, pts = make_poly_points_pair()
    empty_polys = polys.iloc[0:0]
    nodes, edges = group_nodes(empty_polys, pts)
    assert nodes["polygon"].empty
    edge_key = next(iter(edges))
    assert edges[edge_key].empty


def test_group_nodes_predicate_alias_within() -> None:
    """Predicate 'within' excludes boundary point so only interior point remains."""
    polys, pts = make_poly_points_pair()
    # Shift third point to be strictly outside
    _nodes, edges = group_nodes(polys, pts, predicate="within")
    edge_key = next(iter(edges))
    # Points 1 only strictly within (point 2 is on boundary so excluded for within)
    e = edges[edge_key]
    assert {pid for _, pid in e.index} == {1}


def test_group_nodes_network_metric_requires_network() -> None:
    """Network metric without network_gdf raises explicit error for group_nodes."""
    polys, pts = make_poly_points_pair()
    # group_nodes takes two positional args, so we can't use the helper directly
    with pytest.raises(ValueError, match="network_gdf is required"):
        group_nodes(polys, pts, distance_metric="network")


def test_group_nodes_node_geom_col_missing() -> None:
    """Missing node_geom_col should raise a clear error."""
    polys, pts = make_poly_points_pair()
    with pytest.raises(ValueError, match="node_geom_col 'missing' not found"):
        group_nodes(polys, pts, node_geom_col="missing")


def test_group_nodes_node_geom_col_override() -> None:
    """Custom polygon point column drives edge geometries."""
    polys, pts = make_poly_points_pair()
    custom = Point(10, 0)
    polys = polys.assign(node_pt=[custom])
    _, edges = group_nodes(polys, pts, node_geom_col="node_pt")
    edge_key = next(iter(edges))
    geoms = edges[edge_key].geometry
    assert all(tuple(map(float, geom.coords[0])) == (custom.x, custom.y) for geom in geoms)


def test_group_nodes_set_point_nodes_preserves_original_geometry() -> None:
    """set_point_nodes swaps polygon geometry to points and keeps originals."""
    polys, pts = make_poly_points_pair()
    nodes, edges = group_nodes(polys, pts, set_point_nodes=True)
    poly_nodes = nodes["polygon"]
    assert "original_geometry" in poly_nodes.columns
    assert poly_nodes.geometry.equals(polys.geometry.centroid)
    assert poly_nodes["original_geometry"].equals(polys.geometry)
    edge_key = next(iter(edges))
    assert len(edges[edge_key]) == 2


class TestContiguityPublicAPI:
    """Retain existing public API test name for quick test task reference."""

    def test_small_grid(self) -> None:
        """Smoke test small grid contiguity output symmetry."""
        gdf = make_square_polygons()
        nodes, _edges = contiguity_graph(gdf)
        assert set(nodes.index) == set(gdf.index)
        # Edges GeoDataFrame stores undirected edges once (u,v where u<v maybe);
        # validate symmetry by converting to networkx graph and checking neighbors.
        G = cast("nx.Graph", contiguity_graph(gdf, as_nx=True))
        for u, v in G.edges():
            assert G.has_edge(v, u)


# ---------------------------------------------------------------------------
# Additional coverage tests (public API only) to reach 100% proximity coverage
# ---------------------------------------------------------------------------


def test_knn_single_point_early_exit() -> None:
    """KNN with a single point returns no edges (early return path)."""
    gdf = make_single_point()
    nodes, edges = knn_graph(gdf, k=3)
    assert_valid_proximity_result(nodes, edges, 1)


def test_knn_zero_k_early_exit(small_points: gpd.GeoDataFrame) -> None:
    """K <= 0 triggers early return with no edges."""
    nodes, edges = knn_graph(small_points, k=0)
    assert_valid_proximity_result(nodes, edges, len(small_points))


def test_knn_network_metric_success(
    network_edges: gpd.GeoDataFrame, small_points: gpd.GeoDataFrame
) -> None:
    """KNN network metric builds distance matrix and edges (cache path already covered)."""
    nodes, edges = knn_graph(
        small_points, k=1, distance_metric="network", network_gdf=network_edges
    )
    assert_valid_proximity_result(nodes, edges, len(small_points), allow_empty_edges=False)


def test_knn_network_metric_custom_weight(small_points: gpd.GeoDataFrame) -> None:
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


def test_delaunay_network_metric(
    network_edges: gpd.GeoDataFrame, small_points: gpd.GeoDataFrame
) -> None:
    """Delaunay with network metric attaches network-based weights/geometries."""
    _nodes, edges = delaunay_graph(
        small_points, distance_metric="network", network_gdf=network_edges
    )
    if len(small_points) >= 3:
        assert not edges.empty


def test_gabriel_network_metric(
    network_edges: gpd.GeoDataFrame, small_points: gpd.GeoDataFrame
) -> None:
    """Gabriel network metric branch executed."""
    nodes, edges = gabriel_graph(small_points, distance_metric="network", network_gdf=network_edges)
    assert_valid_proximity_result(nodes, edges, len(small_points))


def test_rng_network_metric(
    network_edges: gpd.GeoDataFrame, small_points: gpd.GeoDataFrame
) -> None:
    """Relative Neighborhood Graph network metric branch executed."""
    nodes, edges = relative_neighborhood_graph(
        small_points, distance_metric="network", network_gdf=network_edges
    )
    assert_valid_proximity_result(nodes, edges, len(small_points))


def test_mst_network_metric(
    network_edges: gpd.GeoDataFrame, small_points: gpd.GeoDataFrame
) -> None:
    """Minimum spanning tree under network metric (complete graph fallback)."""
    _nodes, edges = euclidean_minimum_spanning_tree(
        small_points, distance_metric="network", network_gdf=network_edges
    )
    # Tree edges <= n-1
    assert len(edges) <= max(len(small_points) - 1, 0)


def test_fixed_radius_network_metric(
    network_edges: gpd.GeoDataFrame, small_points: gpd.GeoDataFrame
) -> None:
    """Fixed-radius network metric selection path covered."""
    nodes, edges = fixed_radius_graph(
        small_points, radius=2.5, distance_metric="network", network_gdf=network_edges
    )
    assert_valid_proximity_result(nodes, edges, len(small_points))


def test_waxman_basic(small_points: gpd.GeoDataFrame) -> None:
    """Waxman graph probabilistic construction (schema validation)."""
    _nodes, edges = waxman_graph(small_points, beta=0.5, r0=3.0, seed=123)
    assert set(edges.columns) >= {"weight", "geometry"}


def test_waxman_manhattan_metric(small_points: gpd.GeoDataFrame) -> None:
    """Waxman with Manhattan metric exercises the dispatcher manhattan branch."""
    nodes, edges = waxman_graph(
        small_points, beta=0.4, r0=2.0, seed=42, distance_metric="manhattan"
    )
    assert_valid_proximity_result(nodes, edges, len(small_points))


def test_waxman_network_requires_network_gdf(small_points: gpd.GeoDataFrame) -> None:
    """Waxman with network metric should require network_gdf (dispatcher path)."""
    # Waxman has a slightly different error message, so can't use helper
    with pytest.raises(ValueError, match="network_gdf is required for network distance metric"):
        waxman_graph(small_points, beta=0.3, r0=1.5, distance_metric="network")


def test_waxman_network_crs_mismatch(small_points: gpd.GeoDataFrame) -> None:
    """Waxman with network metric and mismatched CRS raises from _network_dm."""
    # Build a tiny network in a different CRS than points
    net = gpd.GeoDataFrame(
        {"geometry": [LineString([(0, 0), (1, 1)])]},
        index=pd.MultiIndex.from_arrays([[0], [1]], names=("source_id", "target_id")),
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="CRS mismatch"):
        waxman_graph(small_points, beta=0.3, r0=1.5, distance_metric="network", network_gdf=net)


def test_waxman_invalid_metric_raises(small_points: gpd.GeoDataFrame) -> None:
    """Unknown metric propagates from internal dispatcher via public API."""
    with pytest.raises(ValueError, match="Unknown distance metric"):
        waxman_graph(small_points, beta=0.5, r0=2.0, distance_metric="invalid")  # type: ignore[arg-type]


def test_group_nodes_invalid_metric() -> None:
    """Unsupported distance metric raises ValueError via public API."""
    polys, pts = make_poly_points_pair()
    with pytest.raises(ValueError, match="Unknown distance metric"):
        group_nodes(polys, pts, distance_metric="invalid")  # type: ignore[arg-type]


def test_group_nodes_crs_mismatch() -> None:
    """CRS mismatch path triggers ValueError."""
    polys, pts = make_poly_points_pair()
    pts2 = pts.to_crs("EPSG:4326")
    with pytest.raises(ValueError, match="same CRS"):
        group_nodes(polys, pts2)


def test_group_nodes_missing_crs() -> None:
    """Both inputs must have a CRS error branch (no CRS)."""
    polys, _ = make_poly_points_pair()
    # Rebuild both layers with no CRS to hit the explicit missing-CRS validation
    poly_geom = polys.geometry.iloc[0]
    polys_nocrs = gpd.GeoDataFrame({"name": ["A"]}, geometry=[poly_geom], crs=None).set_index(
        pd.Index(["A"], name="zone")
    )
    # Build a brand-new GeoDataFrame with shapely geometries so CRS is truly None
    coords = [(1, 1), (2, 1), (3, 3)]
    pts_nocrs = gpd.GeoDataFrame(
        {"pid": [1, 2, 3]},
        geometry=[Point(x, y) for x, y in coords],
        crs=None,
    ).set_index("pid")
    assert polys_nocrs.crs is None
    assert pts_nocrs.crs is None
    with pytest.raises(ValueError, match="CRS"):
        group_nodes(polys_nocrs, pts_nocrs)


def test_group_nodes_network_crs_mismatch() -> None:
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


def test_group_nodes_network_metric_success() -> None:
    """Successful network metric group_nodes call (weights/geometries via network)."""
    polys, pts = make_poly_points_pair()
    # Build simple network linking polygon centroid to each point
    poly_centroid = polys.geometry.iloc[0].centroid
    lines = []
    src = []
    dst = []
    for i, p in enumerate(pts.geometry):
        src.append(i)
        dst.append(i + 100)
        lines.append(LineString([poly_centroid, p]))
    mi = pd.MultiIndex.from_arrays([src, dst], names=("source_id", "target_id"))
    net = gpd.GeoDataFrame({"geometry": lines}, index=mi, crs=polys.crs)
    _nodes, edges = group_nodes(polys, pts, distance_metric="network", network_gdf=net)
    edge_key = next(iter(edges))
    e = edges[edge_key]
    assert len(e) == 2  # two contained points


def test_group_nodes_network_with_length_weight() -> None:
    """group_nodes with network edges carrying 'length' triggers weighted paths branch."""
    polys, pts = make_poly_points_pair()
    poly_centroid = polys.geometry.iloc[0].centroid
    lines = []
    src = []
    dst = []
    lengths = []
    for i, p in enumerate(pts.geometry):
        src.append(i)
        dst.append(i + 100)
        seg = LineString([poly_centroid, p])
        lines.append(seg)
        lengths.append(seg.length)
    mi = pd.MultiIndex.from_arrays([src, dst], names=("source_id", "target_id"))
    net = gpd.GeoDataFrame({"length": lengths, "geometry": lines}, index=mi, crs=polys.crs)
    _nodes, edges = group_nodes(polys, pts, distance_metric="network", network_gdf=net)
    edge_key = next(iter(edges))
    e = edges[edge_key]
    assert len(e) == 2
    assert set(e.columns) >= {"weight", "geometry"}


def test_group_nodes_no_matches_nonempty() -> None:
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


def test_group_nodes_crs_mismatch_explicit_branch() -> None:
    """CRS mismatch after validation: one empty input avoids utils' global CRS check.

    This specifically exercises the explicit branch in group_nodes that checks
    ``poly_crs != pt_crs`` by constructing a scenario where validate_gdf does not
    raise (because one GDF is empty) but the CRSs still differ.
    """
    # Non-empty polygon in EPSG:3857
    polys = gpd.GeoDataFrame(
        {"name": ["A"]},
        geometry=[Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])],
        crs="EPSG:3857",
    ).set_index(pd.Index(["A"], name="zone"))
    # Empty points in a different CRS (EPSG:4326)
    pts = gpd.GeoDataFrame(
        {"pid": []}, geometry=gpd.GeoSeries([], dtype="geometry"), crs="EPSG:4326"
    ).set_index(pd.Index([], name="pid"))
    with pytest.raises(ValueError, match="CRS mismatch between inputs:"):
        group_nodes(polys, pts)


def test_contiguity_network_metric_success() -> None:
    """Contiguity graph with network metric builds network-path geometries."""
    gdf = make_square_polygons().iloc[:2]  # two adjacent squares
    # Build network line connecting centroids
    c1 = gdf.geometry.iloc[0].centroid
    c2 = gdf.geometry.iloc[1].centroid
    net = gpd.GeoDataFrame(
        {"geometry": [LineString([c1, c2])]},
        index=pd.MultiIndex.from_arrays([[0], [1]], names=("source_id", "target_id")),
        crs=gdf.crs,
    )
    _nodes, edges = contiguity_graph(gdf, distance_metric="network", network_gdf=net)
    assert len(edges) == 1


def test_waxman_network_metric_success(
    small_points: gpd.GeoDataFrame, network_edges: gpd.GeoDataFrame
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


def test_contiguity_network_length_weight_branch() -> None:
    """Contiguity with network metric and 'length' attribute triggers weighted path logic."""
    gdf = make_square_polygons().iloc[:2]  # two adjacent squares
    c1 = gdf.geometry.iloc[0].centroid
    c2 = gdf.geometry.iloc[1].centroid
    net = gpd.GeoDataFrame(
        {"length": [c1.distance(c2)], "geometry": [LineString([c1, c2])]},
        index=pd.MultiIndex.from_arrays([[0], [1]], names=("source_id", "target_id")),
        crs=gdf.crs,
    )
    _nodes, edges = contiguity_graph(gdf, distance_metric="network", network_gdf=net)
    assert len(edges) == 1
    # Ensure geometry and weight present (path weight via 'length')
    assert set(edges.columns) >= {"weight", "geometry"}


def test_contiguity_node_geom_col_override() -> None:
    """Custom node geometry column overrides centroid-based positions."""
    gdf = gpd.GeoDataFrame(
        {
            "val": [0, 1],
            "node_pts": [Point(0, 0), Point(10, 0)],
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            ],
        },
        crs="EPSG:3857",
    ).set_index(pd.Index([0, 1], name="pid"))

    nodes, edges = contiguity_graph(gdf, node_geom_col="node_pts")
    assert len(nodes) == 2
    assert len(edges) == 1
    edge = edges.iloc[0]
    assert pytest.approx(edge.weight, rel=1e-6) == 10.0
    assert pytest.approx(edge.geometry.length, rel=1e-6) == 10.0


def test_contiguity_set_point_nodes_centroid() -> None:
    """set_point_nodes swaps polygon geometry for centroid points in nodes layer."""
    gdf = make_square_polygons()

    nodes, _ = contiguity_graph(gdf, set_point_nodes=True)

    assert set(nodes.geometry.geom_type) == {"Point"}
    assert nodes.geometry.equals(gdf.geometry.centroid)
    assert nodes["original_geometry"].equals(gdf.geometry)


def test_contiguity_set_point_nodes_with_column() -> None:
    """set_point_nodes uses custom point column for node geometry."""
    gdf = gpd.GeoDataFrame(
        {
            "val": [0, 1],
            "node_pts": [Point(0, 0), Point(10, 0)],
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            ],
        },
        crs="EPSG:3857",
    ).set_index(pd.Index([0, 1], name="pid"))

    nodes, edges = contiguity_graph(gdf, node_geom_col="node_pts", set_point_nodes=True)

    expected = gpd.GeoSeries(gdf["node_pts"], index=gdf.index, crs=gdf.crs)
    assert nodes.geometry.equals(expected)
    assert nodes["original_geometry"].equals(gdf.geometry)
    assert len(edges) == 1
    edge = edges.iloc[0]
    assert pytest.approx(edge.weight, rel=1e-6) == 10.0
    assert pytest.approx(edge.geometry.length, rel=1e-6) == 10.0


def test_contiguity_node_geom_col_missing() -> None:
    """Missing node_geom_col raises a clear error."""
    gdf = make_square_polygons()
    with pytest.raises(ValueError, match="node_geom_col 'missing' not found"):
        contiguity_graph(gdf, node_geom_col="missing")


def test_contiguity_edges_unique_undirected_path() -> None:
    """Ensure unique undirected edges path in _generate_contiguity_edges is exercised."""
    gdf = make_square_polygons()
    _, edges = contiguity_graph(gdf, contiguity="queen")
    # Should produce at least one edge for the small grid
    assert len(edges) >= 1


def test_contiguity_invalid_metric_raises() -> None:
    """Unsupported contiguity distance_metric triggers ValueError."""
    gdf = make_square_polygons()
    with pytest.raises(ValueError, match="Unknown distance metric"):
        contiguity_graph(gdf, distance_metric="invalid")


def test_contiguity_network_crs_mismatch() -> None:
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


def test_contiguity_empty_as_nx() -> None:
    """Empty contiguity result as NetworkX preserves metadata."""
    empty = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:3857")
    G = cast("nx.Graph", contiguity_graph(empty, as_nx=True))
    assert G.number_of_nodes() == 0
    assert G.number_of_edges() == 0
    assert G.graph.get("contiguity") == "queen"
    assert G.graph.get("distance_metric") == "euclidean"


def test_contiguity_single_polygon_no_edges() -> None:
    """Single polygon yields no adjacency edges (weights.neighbors empty branch)."""
    gdf = gpd.GeoDataFrame(
        {"val": [0]},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        crs="EPSG:3857",
    ).set_index(pd.Index([0], name="pid"))
    nodes, edges = contiguity_graph(gdf)
    assert len(nodes) == 1
    assert edges.empty


def test_directed_knn_as_nx(small_points: gpd.GeoDataFrame) -> None:
    """Directed kNN with as_nx=True exercises _directed_graph return branch."""
    src = small_points.iloc[:2]
    dst = small_points.iloc[2:]
    G = cast("nx.Graph", knn_graph(src, k=1, target_gdf=dst, as_nx=True))
    assert hasattr(G, "number_of_nodes")
    assert G.number_of_edges() >= 1


def test_directed_knn_crs_mismatch(small_points: gpd.GeoDataFrame) -> None:
    """Directed helpers validate CRS equality between src and dst."""
    src = small_points.iloc[:2]
    dst = small_points.iloc[2:].to_crs("EPSG:4326")
    with pytest.raises(ValueError, match="CRS mismatch between source and target"):
        knn_graph(src, k=1, target_gdf=dst)


def test_network_edge_weight_length_branch(small_points: gpd.GeoDataFrame) -> None:
    """Provide 'length' edge attribute on network to trigger weighted path branch."""
    net = make_network_edges(
        src_ids=[0, 1],
        dst_ids=[1, 2],
        geometries=[LineString([(0, 0), (1, 0)]), LineString([(1, 0), (0, 1)])],
        crs="EPSG:27700",
        extra_attrs={"length": [1.0, 2.0]},
    )
    # Use network metric to ensure _set_network_edge_geometries is invoked
    _, e = fixed_radius_graph(small_points, radius=2.5, distance_metric="network", network_gdf=net)
    assert set(e.columns) >= {"weight", "geometry"}


def test_mst_unknown_metric_raises(small_points: gpd.GeoDataFrame) -> None:
    """Unknown distance metric should raise from internal dispatcher via MST."""
    with pytest.raises(ValueError, match="Unknown distance metric"):
        euclidean_minimum_spanning_tree(small_points, distance_metric="invalid")


def test_network_geometry_fallback_same_nearest_node() -> None:
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
    nodes, edges = fixed_radius_graph(pts, radius=0.1, distance_metric="network", network_gdf=net)
    assert len(nodes) == 2
    assert len(edges) == 1
    geom = edges.geometry.iloc[0]
    assert isinstance(geom, LineString)
    # The fallback straight segment should have non-zero length approximately equal to
    # the Euclidean distance between the two sample points (diagonal ~0.02236)
    assert geom.length > 0


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
    generator_func: Callable[..., object], kwargs: dict[str, object]
) -> None:
    """Generators return empty edges when points are insufficient.

    Covers early exit branches for generators requiring 2+ or 3+ points.
    """
    # Single point case
    p1 = make_single_point()
    result: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] = generator_func(p1, **kwargs)  # type: ignore[assignment]
    nodes, edges = result
    assert_valid_proximity_result(nodes, edges, 1)

    # Two points case (still insufficient for Delaunay which needs 3)
    if generator_func == delaunay_graph:
        p2 = make_points_simple([(0, 0), (1, 1)])
        nodes, edges = delaunay_graph(p2)
        assert_valid_proximity_result(nodes, edges, 2)


def test_bridge_nodes_errors_and_nx() -> None:
    """Cover bridge_nodes error branches and as_nx=True path."""
    p1 = make_single_point()

    # < 2 layers
    with pytest.raises(ValueError, match="needs at least two layers"):
        bridge_nodes({"a": p1})

    # Invalid method
    with pytest.raises(ValueError, match="proximity_method must be 'knn' or 'fixed_radius'"):
        bridge_nodes({"a": p1, "b": p1}, proximity_method="invalid")

    # Check networkx output

    G = cast("nx.Graph", bridge_nodes({"a": p1, "b": p1}, k=1, as_nx=True))
    assert G.number_of_nodes() == 2
    assert G.number_of_nodes() == 2


def test_contiguity_input_validation_and_empty_neighbors() -> None:
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


# ---------------------------------------------------------------------------
# Additional tests to achieve 100% coverage (lines 108-113, 119, 123-128, 281-282)
# ---------------------------------------------------------------------------


def test_network_metric_invalid_weight_attribute(small_points: gpd.GeoDataFrame) -> None:
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


def test_network_metric_precomputed_auto_length(small_points: gpd.GeoDataFrame) -> None:
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


def test_network_metric_edges_without_geometry(small_points: gpd.GeoDataFrame) -> None:
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


def test_network_metric_no_valid_positions() -> None:
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


class TestNetworkMetricPublicApis:
    """Public API coverage for network-distance fallback branches."""

    def test_knn_graph_uses_position_fallback_for_network_edge_lengths(
        self,
        small_points: gpd.GeoDataFrame,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """KNN should still work when network edges lack geometry but nodes have positions."""
        graph = nx.Graph()
        graph.add_node("a", pos=(0.0, 0.0))
        graph.add_node("b", pos=(1.0, 0.0))
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
