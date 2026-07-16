"""Tests for :mod:`city2graph.utils.topology`."""

from __future__ import annotations

import warnings
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString
from shapely.geometry import MultiLineString
from shapely.geometry import Point
from shapely.geometry import Polygon

from city2graph import utils
from city2graph.utils import topology as topology_utils
from tests.utils.helpers import BaseGraphTest


class TestGraphStructures(BaseGraphTest):
    """Test graph structure operations like dual graph and segments conversion."""

    @pytest.mark.parametrize(
        (
            "nodes_fixture",
            "edges_fixture",
            "keep_geom",
            "edge_id_col",
            "should_error",
            "error_match",
        ),
        [
            # Success cases
            ("sample_nodes_gdf", "sample_edges_gdf", False, None, False, None),
            ("sample_nodes_gdf", "sample_edges_gdf", True, "edge_id", False, None),
            ("empty_gdf", "empty_gdf", False, None, False, None),
            # Error cases
            ("sample_segments_gdf", None, False, None, True, r"Input `graph` must be a tuple"),
            (
                "sample_nodes_gdf",
                "segments_gdf_no_crs",
                False,
                None,
                True,
                "Edges GeoDataFrame must have a CRS",
            ),
        ],
    )
    def test_dual_graph_conversion(
        self,
        nodes_fixture: str,
        edges_fixture: str | None,
        keep_geom: bool,
        edge_id_col: str | None,
        should_error: bool,
        error_match: str | None,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test dual graph conversion with comprehensive parameter combinations."""
        if edges_fixture is None:
            graph_input = request.getfixturevalue(nodes_fixture)
        else:
            nodes = request.getfixturevalue(nodes_fixture)
            edges = request.getfixturevalue(edges_fixture)
            graph_input = (nodes, edges)

        if should_error:
            with pytest.raises((TypeError, ValueError, AttributeError), match=error_match):
                utils.dual_graph(graph_input, edge_id_col=edge_id_col, keep_original_geom=keep_geom)
        else:
            dual_nodes, dual_edges = utils.dual_graph(
                graph_input,
                edge_id_col=edge_id_col,
                keep_original_geom=keep_geom,
            )

            # Handle empty case
            if isinstance(graph_input, tuple) and graph_input[1].empty:
                self.assert_valid_gdf(dual_nodes, expected_empty=True)
                self.assert_valid_gdf(dual_edges, expected_empty=True)
                return

            self.assert_valid_gdf(dual_nodes)
            self.assert_valid_gdf(dual_edges)
            self.assert_crs_consistency(dual_nodes, dual_edges)

            if keep_geom:
                assert "original_geometry" in dual_nodes.columns

    def test_dual_graph_invalid_edge_id_col(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test that providing a non-existent edge_id_col raises ValueError."""
        with pytest.raises(ValueError, match="Column 'non_existent_col' not found"):
            utils.dual_graph((sample_nodes_gdf, sample_edges_gdf), edge_id_col="non_existent_col")

    def test_dual_graph_accepts_networkx_input(self, simple_nx_graph: nx.Graph) -> None:
        """Dual graph should accept NetworkX graphs directly."""
        dual_nodes, dual_edges = utils.dual_graph(simple_nx_graph)
        self.assert_valid_gdf(dual_nodes)
        self.assert_valid_gdf(dual_edges, expected_empty=True)

    def test_dual_graph_warns_on_geographic_crs(self) -> None:
        """Warn users when dual graph is computed on geographic CRS data."""
        edges = gpd.GeoDataFrame(
            {
                "source": [1, 2],
                "target": [2, 3],
                "geometry": [
                    LineString([(0, 0), (1, 1)]),
                    LineString([(1, 1), (2, 2)]),
                ],
            },
            crs="EPSG:4326",
        ).set_index(["source", "target"])
        nodes = gpd.GeoDataFrame(
            {"geometry": [Point(0, 0), Point(1, 1), Point(2, 2)]},
            crs="EPSG:4326",
            index=[1, 2, 3],
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            utils.dual_graph((nodes, edges))
            assert any("geographic CRS" in str(item.message) for item in caught)

    def test_dual_graph_handles_mixed_edge_ids(self) -> None:
        """Mixed edge-id types should still yield deterministic adjacencies."""
        edges = gpd.GeoDataFrame(
            {
                "edge_id": ["a", 1],
                "source": [1, 2],
                "target": [2, 3],
                "geometry": [
                    LineString([(0, 0), (1, 1)]),
                    LineString([(1, 1), (2, 2)]),
                ],
            },
            crs="EPSG:27700",
        ).set_index(["source", "target"])
        nodes = gpd.GeoDataFrame(
            {"geometry": [Point(0, 0), Point(1, 1), Point(2, 2)]},
            crs="EPSG:27700",
            index=[1, 2, 3],
        )

        dual_nodes, dual_edges = utils.dual_graph((nodes, edges), edge_id_col="edge_id")
        self.assert_valid_gdf(dual_nodes)
        self.assert_valid_gdf(dual_edges)

    def test_dual_graph_empty_result_with_edge_id_col(
        self, sample_nodes_gdf: gpd.GeoDataFrame, sample_crs: str
    ) -> None:
        """Test dual_graph with empty result and edge_id_col specified (line 1527)."""
        # Create edges that won't form any dual edges
        single_edge = gpd.GeoDataFrame(
            {"edge_id": ["e1"]},
            geometry=[LineString([(0, 0), (1, 1)])],
            crs=sample_crs,
        )
        single_edge.index = pd.MultiIndex.from_arrays([[0], [1]], names=["u", "v"])

        dual_nodes, dual_edges = utils.dual_graph(
            (sample_nodes_gdf, single_edge), edge_id_col="edge_id"
        )

        # Should have nodes but no edges (single edge has no dual connections)
        assert not dual_nodes.empty
        assert dual_edges.empty
        assert dual_edges.index.names == ["from_edge_id", "to_edge_id"]

    def test_dual_graph_as_nx(
        self, sample_nodes_gdf: gpd.GeoDataFrame, sample_edges_gdf: gpd.GeoDataFrame
    ) -> None:
        """Test dual_graph with as_nx=True."""
        result = utils.dual_graph((sample_nodes_gdf, sample_edges_gdf), as_nx=True)
        assert isinstance(result, nx.Graph)
        assert len(result) > 0

    def test_canonical_edge_pair_self_loop(self) -> None:
        """Test _canonical_edge_pair with self-loop (line 1370)."""
        assert topology_utils._canonical_edge_pair(1, 1) == (1, 1)
        assert topology_utils._canonical_edge_pair("a", "a") == ("a", "a")


class TestCanonicalizeEdges(BaseGraphTest):
    """Test canonicalize_edges collapsing of reciprocal and parallel rows."""

    @staticmethod
    def _make_edges(
        tuples: list[tuple[Any, ...]],
        names: list[str] | None = None,
    ) -> gpd.GeoDataFrame:
        """Build an edge GeoDataFrame with one distinct row per index tuple."""
        if names is None:
            names = ["u", "v"] if len(tuples[0]) == 2 else ["u", "v", "k"]
        return gpd.GeoDataFrame(
            {"name": [f"e{i}" for i in range(len(tuples))]},
            geometry=[LineString([(i, 0), (i + 1, 1)]) for i in range(len(tuples))],
            index=pd.MultiIndex.from_tuples(tuples, names=names),
            crs="EPSG:27700",
        )

    def test_first_keeps_first_row_per_unordered_pair(self) -> None:
        """duplicates='first' keeps the first reciprocal row verbatim."""
        edges = self._make_edges([(0, 1), (1, 0), (1, 2)])
        result = utils.canonicalize_edges(edges)

        assert result.index.tolist() == [(0, 1), (1, 2)]
        assert result.index.names == ["u", "v"]
        assert list(result["name"]) == ["e0", "e2"]
        assert result.geometry.iloc[0] == edges.geometry.iloc[0]
        assert result.crs == edges.crs

    def test_key_keeps_all_rows_as_multigraph(self) -> None:
        """duplicates='key' keeps reciprocal rows under distinct keys."""
        edges = self._make_edges([(0, 1), (1, 0), (1, 2)])
        result = utils.canonicalize_edges(edges, duplicates="key")

        assert result.index.tolist() == [(0, 1, 0), (0, 1, 1), (1, 2, 0)]
        assert result.index.names == ["u", "v", "key"]
        assert list(result["name"]) == ["e0", "e1", "e2"]

    def test_error_reports_offending_pairs(self) -> None:
        """duplicates='error' raises with row and pair counts."""
        edges = self._make_edges([(0, 1), (1, 0)])
        with pytest.raises(ValueError, match=r"2 row\(s\) across 1 unordered pair\(s\)"):
            utils.canonicalize_edges(edges, duplicates="error")

    def test_error_passes_when_no_duplicates(self) -> None:
        """duplicates='error' returns canonicalized edges when keys are unique."""
        edges = self._make_edges([(1, 0), (2, 1)])
        result = utils.canonicalize_edges(edges, duplicates="error")
        assert result.index.tolist() == [(0, 1), (1, 2)]

    def test_three_level_input_preserves_distinct_keys(self) -> None:
        """Three-level input keeps distinct parallel keys under 'first'."""
        edges = self._make_edges([(1, 0, 0), (0, 1, 1), (1, 0, 1)])
        result = utils.canonicalize_edges(edges)

        # (1, 0, 1) duplicates (0, 1, 1) after canonicalization and is dropped.
        assert result.index.tolist() == [(0, 1, 0), (0, 1, 1)]
        assert result.index.names == ["u", "v", "k"]

    def test_three_level_input_regenerates_keys(self) -> None:
        """duplicates='key' regenerates keys per unordered pair."""
        edges = self._make_edges([(1, 0, 0), (0, 1, 0)])
        result = utils.canonicalize_edges(edges, duplicates="key")
        assert result.index.tolist() == [(0, 1, 0), (0, 1, 1)]
        assert result.index.names == ["u", "v", "k"]

    def test_self_loops_unchanged(self) -> None:
        """Self-loops keep their index values."""
        edges = self._make_edges([(2, 2), (1, 0)])
        result = utils.canonicalize_edges(edges)
        assert result.index.tolist() == [(2, 2), (0, 1)]

    def test_string_ids_sorted(self) -> None:
        """String identifiers are ordered lexicographically."""
        edges = self._make_edges([("b", "a")])
        result = utils.canonicalize_edges(edges)
        assert result.index.tolist() == [("a", "b")]

    def test_mixed_type_ids_use_factorize_fallback(self) -> None:
        """Non-comparable mixed-type identifiers fall back to appearance order."""
        edges = self._make_edges([("a", 1), (1, "a")])
        result = utils.canonicalize_edges(edges)
        assert result.index.tolist() == [("a", 1)]

    def test_empty_input_returns_copy(self) -> None:
        """An empty edge GeoDataFrame is returned unchanged."""
        edges = gpd.GeoDataFrame(
            {"name": []},
            geometry=[],
            index=pd.MultiIndex.from_arrays([[], []], names=["u", "v"]),
            crs="EPSG:27700",
        )
        result = utils.canonicalize_edges(edges)
        assert result.empty
        assert result is not edges

    def test_non_multiindex_raises(self) -> None:
        """A flat index is rejected."""
        edges = gpd.GeoDataFrame(
            {"name": ["e0"]},
            geometry=[LineString([(0, 0), (1, 1)])],
            crs="EPSG:27700",
        )
        with pytest.raises(ValueError, match="MultiIndex with at least"):
            utils.canonicalize_edges(edges)

    def test_invalid_duplicates_option_raises(self) -> None:
        """Unknown duplicates options are rejected."""
        edges = self._make_edges([(0, 1)])
        with pytest.raises(ValueError, match="duplicates must be one of"):
            utils.canonicalize_edges(edges, duplicates="drop")  # type: ignore[arg-type]


class TestSymmetrizeEdges(BaseGraphTest):
    """Test symmetrize_edges adding reverse rows of undirected edges."""

    @staticmethod
    def _make_edges(
        tuples: list[tuple[Any, ...]],
        names: list[str] | None = None,
    ) -> gpd.GeoDataFrame:
        """Build an edge GeoDataFrame with one distinct row per index tuple."""
        if names is None:
            names = ["u", "v"] if len(tuples[0]) == 2 else ["u", "v", "k"]
        return gpd.GeoDataFrame(
            {"name": [f"e{i}" for i in range(len(tuples))]},
            geometry=[LineString([(i, 0), (i + 1, 1)]) for i in range(len(tuples))],
            index=pd.MultiIndex.from_tuples(tuples, names=names),
            crs="EPSG:27700",
        )

    def test_adds_reverse_rows(self) -> None:
        """Each canonical edge gains a reverse row with copied attributes."""
        edges = self._make_edges([(0, 1), (1, 2)])
        result = utils.symmetrize_edges(edges)

        assert result.index.tolist() == [(0, 1), (1, 2), (1, 0), (2, 1)]
        assert result.index.names == ["u", "v"]
        assert list(result["name"]) == ["e0", "e1", "e0", "e1"]
        assert result.crs == edges.crs

    def test_reverse_rows_have_reversed_geometry(self) -> None:
        """Reverse rows reverse the LineString so it starts at the source node."""
        edges = self._make_edges([(0, 1)])
        result = utils.symmetrize_edges(edges)

        forward = list(result.geometry.loc[(0, 1)].coords)
        backward = list(result.geometry.loc[(1, 0)].coords)
        assert backward == forward[::-1]

    def test_self_loops_not_duplicated(self) -> None:
        """Self-loops appear only once in the output."""
        edges = self._make_edges([(2, 2), (0, 1)])
        result = utils.symmetrize_edges(edges)
        assert result.index.tolist() == [(2, 2), (0, 1), (1, 0)]

    def test_idempotent(self) -> None:
        """Applying symmetrize_edges twice equals applying it once."""
        edges = self._make_edges([(0, 1), (1, 2)])
        once = utils.symmetrize_edges(edges)
        twice = utils.symmetrize_edges(once)
        assert twice.equals(once)

    def test_already_bidirectional_input_unchanged(self) -> None:
        """Inputs already holding both directions gain no extra rows."""
        edges = self._make_edges([(0, 1), (1, 0)])
        result = utils.symmetrize_edges(edges)
        assert result.index.tolist() == [(0, 1), (1, 0)]
        assert list(result["name"]) == ["e0", "e1"]

    def test_three_level_index_keeps_keys(self) -> None:
        """Multigraph keys are preserved on reverse rows."""
        edges = self._make_edges([(0, 1, 0), (0, 1, 1)])
        result = utils.symmetrize_edges(edges)
        assert result.index.tolist() == [(0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1)]
        assert result.index.names == ["u", "v", "k"]

    def test_round_trip_with_canonicalize(self) -> None:
        """canonicalize_edges collapses symmetrized output back to the input."""
        edges = self._make_edges([(0, 1), (1, 2)])
        result = utils.canonicalize_edges(utils.symmetrize_edges(edges))
        assert result.equals(edges)

    def test_mixed_type_ids_supported(self) -> None:
        """Mixed, non-comparable identifier types are symmetrized verbatim."""
        edges = self._make_edges([("a", 1)])
        result = utils.symmetrize_edges(edges)
        assert result.index.tolist() == [("a", 1), (1, "a")]

    def test_empty_input_returns_copy(self) -> None:
        """An empty edge GeoDataFrame is returned unchanged."""
        edges = gpd.GeoDataFrame(
            {"name": []},
            geometry=[],
            index=pd.MultiIndex.from_arrays([[], []], names=["u", "v"]),
            crs="EPSG:27700",
        )
        result = utils.symmetrize_edges(edges)
        assert result.empty
        assert result is not edges

    def test_non_multiindex_raises(self) -> None:
        """A flat index is rejected."""
        edges = gpd.GeoDataFrame(
            {"name": ["e0"]},
            geometry=[LineString([(0, 0), (1, 1)])],
            crs="EPSG:27700",
        )
        with pytest.raises(ValueError, match="MultiIndex with at least"):
            utils.symmetrize_edges(edges)


class TestClipGraph(BaseGraphTest):
    """Test graph clipping functionality."""

    def test_clip_graph_basic(self, sample_crs: str) -> None:
        """Test basic clipping with a polygon (default: strict within)."""
        clip_poly = Polygon([(0, 0), (1, 0), (1, 2), (0, 2)])

        gdf = gpd.GeoDataFrame(
            {
                "geometry": [
                    LineString([(0, 0), (2, 0)]),  # partially inside
                    LineString([(0.5, 0.5), (0.5, 1.5)]),  # fully inside
                    LineString([(2, 0), (3, 0)]),  # fully outside
                ]
            },
            crs=sample_crs,
        )

        # Default behavior: geometric clipping
        clipped = utils.clip_graph(gdf, clip_poly)
        assert isinstance(clipped, gpd.GeoDataFrame)
        assert len(clipped) == 2  # Now keeps both fully inside and partially inside (clipped)

        # Check that the partially inside line was clipped
        _ = clipped[clipped.geometry.length < 1.1]  # The one that was clipped (length 1.0)
        # Note: Depending on order/index, we need to be careful.
        # Original: LineString([(0, 0), (2, 0)]) -> clipped to [(0,0), (1,0)]

        # Verify geometries
        # 1. [(0.5, 0.5), (0.5, 1.5)] - Unchanged, length 1.0
        # 2. [(0, 0), (1, 0)] - Clipped, length 1.0

        lengths = clipped.geometry.length.sort_values().to_numpy()
        assert len(lengths) == 2
        assert np.isclose(lengths[0], 1.0)
        assert np.isclose(lengths[1], 1.0)

    def test_clip_graph_keep_outer(self, sample_crs: str) -> None:
        """Test clipping with keep_outer_neighbors=True (intersects)."""
        clip_poly = Polygon([(0, 0), (1, 0), (1, 2), (0, 2)])
        gdf = gpd.GeoDataFrame(
            {
                "geometry": [
                    LineString([(0, 0), (2, 0)]),  # partially inside (intersects)
                    LineString([(0.5, 0.5), (0.5, 1.5)]),  # fully inside
                    LineString([(2, 0), (3, 0)]),  # fully outside
                ]
            },
            crs=sample_crs,
        )

        clipped = utils.clip_graph(gdf, clip_poly, keep_outer_neighbors=True)
        assert len(clipped) == 2

    def test_clip_graph_geometry_handling(self, sample_crs: str) -> None:
        """Test that MultiLineStrings are exploded."""
        clip_poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        gdf = gpd.GeoDataFrame(
            {
                "geometry": [
                    MultiLineString([LineString([(1, 1), (2, 2)]), LineString([(3, 3), (4, 4)])])
                ]
            },
            crs=sample_crs,
        )

        clipped = utils.clip_graph(gdf, clip_poly)
        assert isinstance(clipped, gpd.GeoDataFrame)
        assert len(clipped) == 2
        assert all(isinstance(g, LineString) for g in clipped.geometry)

    def test_clip_graph_empty_input(self, sample_crs: str) -> None:
        """Test with empty input GeoDataFrame."""
        empty_gdf = gpd.GeoDataFrame(geometry=[], crs=sample_crs)
        clip_poly = Polygon([(0, 0), (1, 1), (1, 0)])

        result = utils.clip_graph(empty_gdf, clip_poly)
        assert isinstance(result, gpd.GeoDataFrame)
        assert result.empty

    def test_clip_graph_area_as_gdf(self, sample_crs: str) -> None:
        """Test passing area as GeoDataFrame."""
        clip_poly = Polygon([(0, 0), (1, 0), (1, 2), (0, 2)])
        area_gdf = gpd.GeoDataFrame({"geometry": [clip_poly]}, crs=sample_crs)

        gdf = gpd.GeoDataFrame({"geometry": [LineString([(0.5, 0.5), (0.5, 1.5)])]}, crs=sample_crs)

        clipped = utils.clip_graph(gdf, area_gdf)
        assert len(clipped) == 1

    def test_clip_graph_with_tuple_input(self, sample_crs: str) -> None:
        """Test clipping with (nodes, edges) tuple input."""
        clip_poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])

        nodes = gpd.GeoDataFrame(
            {"geometry": [Point(0, 0), Point(1, 1), Point(3, 3)]},
            index=pd.Index([1, 2, 3], name="node_id"),
            crs=sample_crs,
        )
        edges = gpd.GeoDataFrame(
            {"geometry": [LineString([(0, 0), (1, 1)]), LineString([(1, 1), (3, 3)])]},
            index=pd.MultiIndex.from_tuples([(1, 2), (2, 3)], names=["u", "v"]),
            crs=sample_crs,
        )

        clipped_nodes, clipped_edges = utils.clip_graph((nodes, edges), clip_poly)
        assert isinstance(clipped_nodes, gpd.GeoDataFrame)
        assert isinstance(clipped_edges, gpd.GeoDataFrame)
        assert len(clipped_edges) == 1
        assert len(clipped_nodes) == 2
        assert isinstance(clipped_edges.index, pd.MultiIndex)
        assert clipped_edges.index.names == ["u", "v"]
        assert set(clipped_nodes.index.to_list()) == {1, 2}

    def test_clip_graph_preserves_multiindex_after_explode(self, sample_crs: str) -> None:
        """Test tuple clipping preserves edge MultiIndex after MultiLineString explode."""
        clip_poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

        nodes = gpd.GeoDataFrame(
            {"geometry": [Point(1, 1), Point(4, 4), Point(9, 9)]},
            index=pd.Index([1, 2, 3], name="node_id"),
            crs=sample_crs,
        )
        edges = gpd.GeoDataFrame(
            {
                "geometry": [
                    MultiLineString([LineString([(1, 1), (2, 2)]), LineString([(3, 3), (4, 4)])])
                ]
            },
            index=pd.MultiIndex.from_tuples([(1, 2)], names=["u", "v"]),
            crs=sample_crs,
        )

        clipped_nodes, clipped_edges = utils.clip_graph((nodes, edges), clip_poly)

        assert isinstance(clipped_nodes, gpd.GeoDataFrame)
        assert isinstance(clipped_edges.index, pd.MultiIndex)
        assert clipped_edges.index.names == ["u", "v"]
        assert len(clipped_edges) == 2
        assert set(clipped_nodes.index.to_list()) == {1, 2}

    def test_clip_graph_area_gdf_crs_alignment(self) -> None:
        """Test clipping aligns area CRS with edge CRS when area is GeoDataFrame."""
        nodes = gpd.GeoDataFrame(
            {
                "geometry": [
                    Point(-0.150, 51.505),
                    Point(-0.140, 51.510),
                    Point(-0.220, 51.600),
                ]
            },
            index=pd.Index([1, 2, 3], name="node_id"),
            crs="EPSG:4326",
        )
        edges = gpd.GeoDataFrame(
            {"geometry": [LineString([(-0.150, 51.505), (-0.140, 51.510)])]},
            index=pd.MultiIndex.from_tuples([(1, 2)], names=["u", "v"]),
            crs="EPSG:4326",
        )

        area_wgs84 = gpd.GeoDataFrame(
            {
                "geometry": [
                    Polygon(
                        [
                            (-0.170, 51.490),
                            (-0.120, 51.490),
                            (-0.120, 51.530),
                            (-0.170, 51.530),
                        ]
                    )
                ]
            },
            crs="EPSG:4326",
        )
        area_bng = area_wgs84.to_crs(epsg=27700)

        clipped_nodes, clipped_edges = utils.clip_graph((nodes, edges), area_bng)

        assert isinstance(clipped_nodes, gpd.GeoDataFrame)
        assert len(clipped_edges) == 1
        assert isinstance(clipped_edges.index, pd.MultiIndex)
        assert set(clipped_nodes.index.to_list()) == {1, 2}

    def test_clip_graph_removes_out_of_boundary_endpoints(self, sample_crs: str) -> None:
        """Test strict clipping removes outside endpoint nodes and crossing edges."""
        clip_poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])

        nodes = gpd.GeoDataFrame(
            {"geometry": [Point(0.5, 0.5), Point(1.5, 1.5), Point(3.0, 3.0)]},
            index=pd.Index([1, 2, 3], name="node_id"),
            crs=sample_crs,
        )
        edges = gpd.GeoDataFrame(
            {
                "geometry": [
                    LineString([(0.5, 0.5), (1.5, 1.5)]),
                    LineString([(1.5, 1.5), (3.0, 3.0)]),
                ]
            },
            index=pd.MultiIndex.from_tuples([(1, 2), (2, 3)], names=["u", "v"]),
            crs=sample_crs,
        )

        clipped_nodes, clipped_edges = utils.clip_graph(
            (nodes, edges),
            clip_poly,
            keep_outer_neighbors=False,
        )

        assert isinstance(clipped_nodes, gpd.GeoDataFrame)
        assert set(clipped_nodes.index.to_list()) == {1, 2}
        assert set(clipped_edges.index.to_list()) == {(1, 2)}

    def test_clip_graph_with_nx_input(self, sample_crs: str) -> None:
        """Test clipping with NetworkX graph input."""
        clip_poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])

        nodes = gpd.GeoDataFrame(
            {"geometry": [Point(0, 0), Point(1, 1), Point(3, 3)]},
            index=pd.Index([1, 2, 3], name="node_id"),
            crs=sample_crs,
        )
        edges = gpd.GeoDataFrame(
            {"geometry": [LineString([(0, 0), (1, 1)]), LineString([(1, 1), (3, 3)])]},
            index=pd.MultiIndex.from_tuples([(1, 2), (2, 3)], names=["u", "v"]),
            crs=sample_crs,
        )
        nx_graph = utils.gdf_to_nx(nodes=nodes, edges=edges)

        result = utils.clip_graph(nx_graph, clip_poly)
        assert isinstance(result, nx.Graph)

    def test_clip_graph_as_nx_output(self, sample_crs: str) -> None:
        """Test clipping with as_nx=True returns NetworkX graph."""
        clip_poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        gdf = gpd.GeoDataFrame(
            {"geometry": [LineString([(0.5, 0.5), (1, 1)])]},
            crs=sample_crs,
        )

        result = utils.clip_graph(gdf, clip_poly, as_nx=True)
        assert isinstance(result, nx.Graph)


class TestRemoveIsolatedComponents(BaseGraphTest):
    """Test remove_isolated_components functionality."""

    def test_remove_isolated_basic(self, sample_crs: str) -> None:
        """Test basic isolation removal with GeoDataFrame."""
        # Create a graph with two disconnected components
        gdf = gpd.GeoDataFrame(
            {
                "geometry": [
                    # Large component (3 edges)
                    LineString([(0, 0), (1, 0)]),
                    LineString([(1, 0), (2, 0)]),
                    LineString([(2, 0), (3, 0)]),
                    # Small isolated component (1 edge)
                    LineString([(10, 10), (11, 10)]),
                ]
            },
            crs=sample_crs,
        )

        result = utils.remove_isolated_components(gdf)
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 3  # Only large component kept

    def test_remove_isolated_with_tuple_input(self, sample_crs: str) -> None:
        """Test with (nodes, edges) tuple input."""
        nodes = gpd.GeoDataFrame(
            {"geometry": [Point(0, 0), Point(1, 0), Point(10, 10), Point(11, 10)]},
            index=pd.Index([1, 2, 3, 4], name="node_id"),
            crs=sample_crs,
        )
        edges = gpd.GeoDataFrame(
            {
                "geometry": [
                    LineString([(0, 0), (1, 0)]),  # Component 1
                    LineString([(10, 10), (11, 10)]),  # Component 2 (isolated)
                ]
            },
            index=pd.MultiIndex.from_tuples([(1, 2), (3, 4)], names=["u", "v"]),
            crs=sample_crs,
        )

        result_nodes, result_edges = utils.remove_isolated_components((nodes, edges))
        assert isinstance(result_nodes, gpd.GeoDataFrame)
        assert isinstance(result_edges, gpd.GeoDataFrame)
        # Both components have same size (1 edge), first one should be kept
        assert len(result_edges) == 1

    def test_remove_isolated_with_nx_input(self, sample_crs: str) -> None:
        """Test with NetworkX graph input."""
        nodes = gpd.GeoDataFrame(
            {"geometry": [Point(0, 0), Point(1, 0), Point(10, 10)]},
            index=pd.Index([1, 2, 3], name="node_id"),
            crs=sample_crs,
        )
        edges = gpd.GeoDataFrame(
            {"geometry": [LineString([(0, 0), (1, 0)])]},
            index=pd.MultiIndex.from_tuples([(1, 2)], names=["u", "v"]),
            crs=sample_crs,
        )
        nx_graph = utils.gdf_to_nx(nodes=nodes, edges=edges)
        # Add isolated node
        nx_graph.add_node(999, pos=(10, 10), geometry=Point(10, 10))

        result = utils.remove_isolated_components(nx_graph)
        assert isinstance(result, nx.Graph)

    def test_remove_isolated_as_nx_output(self, sample_crs: str) -> None:
        """Test with as_nx=True returns NetworkX graph."""
        gdf = gpd.GeoDataFrame(
            {"geometry": [LineString([(0, 0), (1, 0)])]},
            crs=sample_crs,
        )

        result = utils.remove_isolated_components(gdf, as_nx=True)
        assert isinstance(result, nx.Graph)

    def test_remove_isolated_empty_input(self, sample_crs: str) -> None:
        """Test with empty GeoDataFrame."""
        empty_gdf = gpd.GeoDataFrame(geometry=[], crs=sample_crs)

        result = utils.remove_isolated_components(empty_gdf)
        assert isinstance(result, gpd.GeoDataFrame)
        assert result.empty

    def test_remove_isolated_graph_conversion_error(self, sample_crs: str) -> None:
        """Test remove_isolated_components when graph conversion fails (covers lines 4936-4940)."""
        # Create a GeoDataFrame with geometry but invalid structure for graph conversion
        edges = gpd.GeoDataFrame(
            {"geometry": [LineString([(0, 0), (1, 0)])]},
            crs=sample_crs,
        )
        # Set a non-standard index that will cause issues
        edges.index = pd.Index(["edge1"], name="weird_id")

        # Should handle the error gracefully
        result = utils.remove_isolated_components(edges)
        assert isinstance(result, gpd.GeoDataFrame)

    def test_remove_isolated_invalid_graph_structure(self, sample_crs: str) -> None:
        """Test remove_isolated_components with edges that fail graph conversion."""
        # Create a GeoDataFrame with geometry but non-standard index
        edges = gpd.GeoDataFrame(
            {"geometry": [LineString([(0, 0), (1, 0)])]},
            crs=sample_crs,
        )
        # No MultiIndex, just a simple index that won't convert to graph properly
        edges.index = pd.Index(["edge1"], name="weird_id")

        # Should handle gracefully
        result = utils.remove_isolated_components(edges)
        assert isinstance(result, gpd.GeoDataFrame)


class TestPublicUtilityBranches(BaseGraphTest):
    """Coverage-oriented tests exercised through public utility APIs."""

    def test_clip_graph_rejects_invalid_input_type(self) -> None:
        """clip_graph should reject unsupported graph inputs."""
        with pytest.raises(TypeError, match="Input must be GeoDataFrame"):
            utils.clip_graph("not_a_graph", Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]))

    def test_clip_graph_returns_empty_nodes_when_edges_clip_out(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Tuple clipping should return empty nodes when no clipped edges remain."""
        clip_poly = Polygon([(100, 100), (101, 100), (101, 101), (100, 101)])

        clipped_nodes, clipped_edges = utils.clip_graph(
            (sample_nodes_gdf, sample_edges_gdf), clip_poly
        )

        assert clipped_edges.empty
        assert clipped_nodes is not None
        assert clipped_nodes.empty

    def test_clip_graph_preserves_nodes_for_non_multiindex_edges(self, sample_crs: str) -> None:
        """Tuple clipping should leave nodes untouched when edges have no connectivity index."""
        nodes = gpd.GeoDataFrame({"value": [1]}, geometry=[Point(0, 0)], crs=sample_crs)
        nodes.index = pd.Index(["n1"])
        edges = gpd.GeoDataFrame(
            {"value": [1]},
            geometry=[LineString([(0, 0), (1, 0)])],
            crs=sample_crs,
        )
        edges.index = pd.Index(["edge-1"])
        clip_poly = Polygon([(-1, -1), (2, -1), (2, 1), (-1, 1)])

        clipped_nodes, clipped_edges = utils.clip_graph((nodes, edges), clip_poly)

        assert not clipped_edges.empty
        assert clipped_nodes is not None
        assert clipped_nodes.equals(nodes)

    def test_remove_isolated_components_returns_original_on_conversion_error(
        self,
        sample_edges_gdf: gpd.GeoDataFrame,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """remove_isolated_components should fall back cleanly when conversion fails."""

        def raise_bad_graph(*_args: object, **_kwargs: object) -> object:
            msg = "bad graph"
            error = ValueError(msg)
            raise error

        monkeypatch.setattr(
            utils,
            "gdf_to_nx",
            raise_bad_graph,
        )

        result = utils.remove_isolated_components(sample_edges_gdf)

        assert isinstance(result, gpd.GeoDataFrame)
        assert result.equals(sample_edges_gdf)

    def test_remove_isolated_components_handles_empty_networkx_graph(
        self,
        sample_edges_gdf: gpd.GeoDataFrame,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """remove_isolated_components should fall back when conversion yields an empty graph."""
        monkeypatch.setattr(topology_utils, "gdf_to_nx", lambda *_args, **_kwargs: nx.Graph())

        result = utils.remove_isolated_components(sample_edges_gdf)

        assert isinstance(result, gpd.GeoDataFrame)
        assert result.equals(sample_edges_gdf)
