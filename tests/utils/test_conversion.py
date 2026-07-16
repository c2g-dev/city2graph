"""Tests for :mod:`city2graph.utils.conversion`."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import geopandas as gpd
import networkx as nx
import pandas as pd
import pytest
import rustworkx as rx
from shapely.geometry import LineString
from shapely.geometry import Point

from city2graph import morphology
from city2graph import utils
from city2graph.base import GeoDataProcessor
from city2graph.base import GraphMetadata
from city2graph.utils import NxConverter
from city2graph.utils import conversion as conversion_utils
from city2graph.utils import gdf_to_nx
from city2graph.utils import nx_to_gdf
from city2graph.utils import spatial as spatial_utils
from city2graph.utils import topology as topology_utils
from tests import helpers
from tests.utils.helpers import BaseConversionTest
from tests.utils.helpers import BaseGraphTest

if TYPE_CHECKING:
    from collections.abc import Callable


class TestGraphStructures(BaseGraphTest):
    """Test graph structure operations like dual graph and segments conversion."""

    def test_validate_nx_populates_pos_from_xy(self, sample_crs: str) -> None:
        """validate_nx should auto-create pos from x/y when missing."""
        G = nx.Graph()
        G.add_node(1, x=0.0, y=0.0)
        G.add_node(2, x=1.0, y=1.0)
        G.add_edge(1, 2)
        G.graph = {"crs": sample_crs, "is_hetero": False}

        GeoDataProcessor().validate_nx(G)
        assert "pos" in G.nodes[1]
        assert "pos" in G.nodes[2]

    @pytest.mark.parametrize(
        ("segments_fixture", "expect_empty", "multigraph"),
        [
            ("sample_segments_gdf", False, False),
            ("empty_gdf", True, False),
            ("segments_invalid_geom_gdf", True, False),
            ("sample_segments_gdf", False, True),  # Test multigraph functionality
        ],
    )
    def test_segments_to_graph_conversion(
        self,
        segments_fixture: str,
        expect_empty: bool,
        multigraph: bool,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test conversion of line segments to graph structure."""
        segments_gdf = request.getfixturevalue(segments_fixture)
        nodes_gdf, edges_gdf = morphology.segments_to_graph(segments_gdf, multigraph=multigraph)

        self.assert_valid_gdf(nodes_gdf, expect_empty)
        self.assert_valid_gdf(edges_gdf, expect_empty)
        self.assert_crs_consistency(nodes_gdf, edges_gdf)

        if not expect_empty:
            assert nodes_gdf.index.name == "node_id"
            helpers.assert_geometry_types(nodes_gdf, ["Point"])
            assert isinstance(edges_gdf.index, pd.MultiIndex)

            expected_index_names = (
                ["from_node_id", "to_node_id", "edge_key"]
                if multigraph
                else ["from_node_id", "to_node_id"]
            )
            helpers.assert_index_names(edges_gdf, expected_index_names)

    def test_segments_multigraph_duplicate_handling(
        self,
        duplicate_segments_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test multigraph handling of duplicate edge connections."""
        nodes_gdf, edges_gdf = morphology.segments_to_graph(duplicate_segments_gdf, multigraph=True)

        assert len(nodes_gdf) == 2  # Two unique points
        assert len(edges_gdf) == 2  # Both edges preserved
        assert edges_gdf.index.names == ["from_node_id", "to_node_id", "edge_key"]

        # Verify edge keys are different for duplicates
        edge_keys = edges_gdf.index.get_level_values("edge_key")
        assert list(edge_keys) == [0, 1]

    def test_segments_to_graph_default_is_multigraph(
        self,
        sample_segments_gdf: gpd.GeoDataFrame,
    ) -> None:
        """The default output carries a three-level multigraph index."""
        _, edges_gdf = morphology.segments_to_graph(sample_segments_gdf)
        assert edges_gdf.index.names == ["from_node_id", "to_node_id", "edge_key"]

    def test_segments_to_graph_multigraph_false_duplicates_raise(
        self,
        duplicate_segments_gdf: gpd.GeoDataFrame,
    ) -> None:
        """multigraph=False raises on duplicate node pairs instead of passing silently."""
        with pytest.raises(ValueError, match=r"1 duplicate node pair\(s\)"):
            morphology.segments_to_graph(duplicate_segments_gdf, multigraph=False)

    def test_segments_to_graph_directed_default_preserves_draw_order(
        self,
        sample_crs: str,
    ) -> None:
        """Default directed=True keeps reverse-drawn segments as reciprocal pairs."""
        segments = gpd.GeoDataFrame(
            {"name": ["fwd", "rev"]},
            geometry=[
                LineString([(0, 0), (1, 1)]),
                LineString([(1, 1), (0, 0)]),
            ],
            crs=sample_crs,
        )
        _, edges_gdf = morphology.segments_to_graph(segments)

        pairs = list(
            zip(
                edges_gdf.index.get_level_values("from_node_id"),
                edges_gdf.index.get_level_values("to_node_id"),
                strict=True,
            )
        )
        assert pairs == [(0, 1), (1, 0)]

    def test_segments_to_graph_undirected_canonicalizes_reverse_segments(
        self,
        sample_crs: str,
    ) -> None:
        """directed=False folds reverse-drawn segments into one unordered pair."""
        segments = gpd.GeoDataFrame(
            {"name": ["fwd", "rev"]},
            geometry=[
                LineString([(0, 0), (1, 1)]),
                LineString([(1, 1), (0, 0)]),
            ],
            crs=sample_crs,
        )
        _, edges_gdf = morphology.segments_to_graph(segments, directed=False)

        assert edges_gdf.index.tolist() == [(0, 1, 0), (0, 1, 1)]
        # Geometries are left unchanged; only the index order is normalized.
        assert list(edges_gdf.geometry) == list(segments.geometry)

    def test_segments_to_graph_empty_as_nx(
        self,
        empty_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Empty input honors as_nx=True instead of returning a tuple."""
        result = morphology.segments_to_graph(empty_gdf, as_nx=True)
        assert isinstance(result, nx.MultiGraph)
        assert result.number_of_nodes() == 0

        simple = morphology.segments_to_graph(empty_gdf, multigraph=False, as_nx=True)
        assert isinstance(simple, nx.Graph)
        assert not isinstance(simple, nx.MultiGraph)

    def test_segments_to_graph_as_nx_keeps_parallel_edges(
        self,
        duplicate_segments_gdf: gpd.GeoDataFrame,
    ) -> None:
        """as_nx=True with the multigraph default preserves parallel edges."""
        graph = morphology.segments_to_graph(duplicate_segments_gdf, as_nx=True)
        assert isinstance(graph, nx.MultiGraph)
        assert graph.number_of_edges() == 2


class TestNxConversions(BaseConversionTest):
    """Test conversions between GeoDataFrame and NetworkX formats."""

    def test_homogeneous_roundtrip_conversion(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test roundtrip conversion preserves data integrity for homogeneous graphs."""
        graph = gdf_to_nx(sample_nodes_gdf, sample_edges_gdf)
        nodes_converted, edges_converted = nx_to_gdf(graph)

        self.assert_roundtrip_consistency(
            sample_nodes_gdf,
            sample_edges_gdf,
            nodes_converted,
            edges_converted,
        )

    def test_heterogeneous_roundtrip_conversion(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Test roundtrip conversion for heterogeneous graphs."""
        graph = gdf_to_nx(
            nodes=sample_hetero_nodes_dict,
            edges=sample_hetero_edges_dict,
            multigraph=True,
        )
        nodes_dict_converted, edges_dict_converted = nx_to_gdf(graph)

        assert isinstance(nodes_dict_converted, dict)
        assert isinstance(edges_dict_converted, dict)
        assert sample_hetero_nodes_dict.keys() == nodes_dict_converted.keys()
        assert sample_hetero_edges_dict.keys() == edges_dict_converted.keys()

        for node_type, original_nodes in sample_hetero_nodes_dict.items():
            assert isinstance(nodes_dict_converted[node_type], gpd.GeoDataFrame)
            converted_nodes = nodes_dict_converted[node_type]
            # Use the first available edge type for consistency check
            first_edge_type = next(iter(sample_hetero_edges_dict.keys()))
            original_edges = sample_hetero_edges_dict[first_edge_type]
            assert isinstance(edges_dict_converted[first_edge_type], gpd.GeoDataFrame)
            converted_edges = edges_dict_converted[first_edge_type]
            self.assert_roundtrip_consistency(
                original_nodes,
                original_edges,
                converted_nodes,
                converted_edges,
            )

    @pytest.mark.parametrize(
        ("input_type", "gdf_fixture"),
        [("edges_only", "sample_edges_gdf"), ("hetero_edges_only", "sample_hetero_edges_dict")],
    )
    def test_edges_only_conversion(
        self,
        input_type: str,
        gdf_fixture: str,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test conversion with only edge data provided."""
        gdf = request.getfixturevalue(gdf_fixture)

        if input_type == "edges_only":
            graph = gdf_to_nx(edges=gdf)
            assert isinstance(graph, nx.Graph)
            assert graph.number_of_edges() == len(gdf)
            assert graph.number_of_nodes() > 0  # Nodes created from edge endpoints
        else:  # hetero_edges_only
            graph = gdf_to_nx(edges=gdf)
            assert isinstance(graph, nx.Graph)
            # Heterogeneous edges dict without nodes results in empty graph
            assert graph.number_of_edges() == 0

    def test_geometryless_edges_mapped_by_index(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Edges without geometry are mapped via the (source, target) MultiIndex."""
        edges = sample_edges_gdf.set_geometry(
            gpd.GeoSeries([None] * len(sample_edges_gdf), index=sample_edges_gdf.index),
            crs=sample_edges_gdf.crs,
        )
        # An edge referencing an unknown node id cannot be mapped and is dropped
        unknown = edges.iloc[[0]].copy()
        unknown.index = pd.MultiIndex.from_tuples([(1, 99)], names=edges.index.names)
        edges = pd.concat([edges, unknown])

        with caplog.at_level(logging.WARNING, logger="city2graph.utils"):
            graph = gdf_to_nx(nodes=sample_nodes_gdf, edges=edges, keep_geom=False)

        assert graph.number_of_edges() == len(sample_edges_gdf)
        original_edges = {
            (graph.nodes[u]["_original_index"], graph.nodes[v]["_original_index"])
            for u, v in graph.edges()
        }
        assert original_edges == set(sample_edges_gdf.index)
        assert "Could not find nodes for 1 edges without geometry" in caplog.text

    def test_geometryless_edges_without_multiindex_are_dropped(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Edges without geometry and without a (u, v) MultiIndex cannot be mapped."""
        flat_edges = sample_edges_gdf.reset_index(drop=True)
        edges = flat_edges.set_geometry(
            gpd.GeoSeries([None] * len(flat_edges), index=flat_edges.index),
            crs=flat_edges.crs,
        )

        with caplog.at_level(logging.WARNING, logger="city2graph.utils"):
            graph = gdf_to_nx(nodes=sample_nodes_gdf, edges=edges, keep_geom=False)

        assert graph.number_of_edges() == 0
        assert "MultiIndex is required" in caplog.text

    def test_nx_to_gdf_requires_nodes_or_edges(self, simple_nx_graph: nx.Graph) -> None:
        """Requesting neither nodes nor edges should raise."""
        with pytest.raises(ValueError, match="Must request at least one of nodes or edges"):
            nx_to_gdf(simple_nx_graph, nodes=False, edges=False)

    def test_directed_multigraph_conversion(
        self,
        directed_multigraph_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Directed + multigraph flags should produce a MultiDiGraph."""
        converter = NxConverter(directed=True, multigraph=True)
        graph = converter.gdf_to_nx(nodes=None, edges=directed_multigraph_edges_gdf)
        assert isinstance(graph, nx.MultiDiGraph)

    @pytest.mark.parametrize(
        ("nodes_arg", "edges_arg", "error_type", "error_match"),
        [
            (None, None, ValueError, "Either nodes or edges must be provided"),
            ("not_a_gdf", "sample_edges_gdf", TypeError, "Input must be a GeoDataFrame"),
            (
                "sample_hetero_nodes_dict",
                "sample_edges_gdf",
                TypeError,
                "If nodes is a dict, edges must also be a dict",
            ),
            (
                "sample_nodes_gdf_alt_crs",
                "sample_edges_gdf",
                ValueError,
                "All GeoDataFrames must have the same CRS",
            ),
        ],
    )
    def test_conversion_error_handling(
        self,
        nodes_arg: str | None,
        edges_arg: str | None,
        error_type: type[Exception],
        error_match: str,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test proper error handling for invalid conversion inputs."""
        nodes = request.getfixturevalue(nodes_arg) if nodes_arg else None
        edges = request.getfixturevalue(edges_arg) if edges_arg else None

        with pytest.raises(error_type, match=error_match):
            gdf_to_nx(nodes=nodes, edges=edges)

    def test_node_index_names_survive_roundtrip(
        self,
        single_name_index_nodes_gdf: gpd.GeoDataFrame,
        simple_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Roundtrips should respect single-level index names."""
        graph = gdf_to_nx(nodes=single_name_index_nodes_gdf, edges=simple_edges_gdf)
        nodes_back, _ = nx_to_gdf(graph)
        assert isinstance(nodes_back, gpd.GeoDataFrame)
        assert nodes_back.index.name == "single_name"

        graph.graph["node_index_names"] = None
        nodes_back, _ = nx_to_gdf(graph)
        assert isinstance(nodes_back, gpd.GeoDataFrame)
        assert nodes_back.index.name is None

    def test_multiindex_nodes_roundtrip(
        self,
        multiindex_nodes_gdf: gpd.GeoDataFrame,
        simple_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """MultiIndex node metadata should be preserved in graph metadata."""
        graph = gdf_to_nx(nodes=multiindex_nodes_gdf, edges=simple_edges_gdf)
        assert graph.graph["node_index_names"] == ["node_type", "node_id"]

    @pytest.mark.parametrize(
        ("graph_fixture", "expect_crs", "expect_geom"),
        [
            ("sample_nx_graph", True, True),
            ("sample_nx_graph_no_crs", False, True),
            ("sample_nx_graph_no_pos", True, True),
        ],
    )
    def test_nx_to_gdf_variants(
        self,
        graph_fixture: str,
        expect_crs: bool,
        expect_geom: bool,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test NetworkX to GDF conversion with different graph properties."""
        graph = request.getfixturevalue(graph_fixture)
        nodes, edges = nx_to_gdf(graph)

        # These are homogeneous graphs, so ensure they return GeoDataFrames not dicts
        assert isinstance(nodes, gpd.GeoDataFrame)
        assert isinstance(edges, gpd.GeoDataFrame)

        if expect_geom:
            assert "geometry" in nodes.columns
            assert "geometry" in edges.columns

        if expect_crs:
            assert nodes.crs is not None
            assert edges.crs is not None
        else:
            assert nodes.crs is None
            assert edges.crs is None

    def test_nx_to_gdf_handles_empty_edges(self, sample_crs: str) -> None:
        """Graphs with no edges should still return empty edge GeoDataFrames."""
        graph = nx.Graph()
        graph.add_node(1, pos=(0, 0), geometry=Point(0, 0))
        graph.graph = {"crs": sample_crs, "is_hetero": False}

        nodes_gdf, edges_gdf = nx_to_gdf(graph)
        self.assert_valid_gdf(nodes_gdf)
        self.assert_valid_gdf(edges_gdf, expected_empty=True)

    def test_nx_to_gdf_multigraph_edges(self, sample_crs: str) -> None:
        """Multigraph attributes should be preserved after conversion."""
        graph = nx.MultiGraph()
        graph.add_node(1, pos=(0, 0), geometry=Point(0, 0))
        graph.add_node(2, pos=(1, 1), geometry=Point(1, 1))
        graph.add_edge(1, 2, key=0, weight=1.0, geometry=LineString([(0, 0), (1, 1)]))
        graph.graph = {"crs": sample_crs, "is_hetero": False}

        _, edges_gdf = nx_to_gdf(graph)
        self.assert_valid_gdf(edges_gdf)
        assert isinstance(edges_gdf, gpd.GeoDataFrame)
        assert "weight" in edges_gdf.columns

    def test_nx_to_gdf_multiindex_edges(self, sample_crs: str) -> None:
        """List-based stored edge indices should be normalized during reconstruction."""
        graph = nx.MultiGraph()
        graph.add_node(1, pos=(0, 0), geometry=Point(0, 0))
        graph.add_node(2, pos=(1, 1), geometry=Point(1, 1))
        graph.add_edge(1, 2, key=0, geometry=LineString([(0, 0), (1, 1)]))
        graph.add_edge(1, 2, key=1, geometry=LineString([(0, 0), (1, 1)]))
        graph.graph = {"crs": sample_crs, "is_hetero": False}

        for u, v, k, attrs in graph.edges(data=True, keys=True):
            attrs["_original_edge_index"] = [u, v, k]

        _, edges_gdf = nx_to_gdf(graph)
        self.assert_valid_gdf(edges_gdf)
        assert len(edges_gdf) == 2

    def test_nx_to_gdf_single_coord_attr(self, sample_crs: str) -> None:
        """Missing pos attributes can be populated from alternative coordinate keys."""
        graph = nx.Graph()
        graph.add_node(1, coords=[0.0, 1.0])
        graph.add_node(2, coords=(2.0, 3.0))
        graph.graph = {"crs": sample_crs, "is_hetero": False}

        nodes_gdf = nx_to_gdf(graph, nodes=True, edges=False, set_missing_pos_from=("coords",))
        self.assert_valid_gdf(nodes_gdf)
        assert len(nodes_gdf) == 2

    def test_heterogeneous_edge_processing(
        self,
        regular_hetero_graph: nx.Graph,
        empty_hetero_graph: nx.Graph,
    ) -> None:
        """nx_to_gdf should return dictionaries for heterogeneous graphs."""
        nodes_dict, edges_dict = nx_to_gdf(regular_hetero_graph)
        assert isinstance(nodes_dict, dict)
        assert isinstance(edges_dict, dict)

        nodes_dict, edges_dict = nx_to_gdf(empty_hetero_graph)
        assert isinstance(edges_dict[("building", "connects", "road")], gpd.GeoDataFrame)
        assert edges_dict[("building", "connects", "road")].empty

    def test_empty_nodes_in_conversion(self, sample_crs: str) -> None:
        """Test conversion with completely empty nodes (line 995)."""
        # Create edges but with no nodes that will result in empty node records
        edges_gdf = gpd.GeoDataFrame(
            geometry=[LineString([(0, 0), (1, 1)])],
            crs=sample_crs,
        )

        # Create a graph and then remove all node attributes to trigger empty records
        converter = NxConverter()
        graph = converter.gdf_to_nx(edges=edges_gdf)

        # Manually clear node data to simulate empty records scenario
        for node in graph.nodes():
            # Keep only pos to trigger the empty records path
            graph.nodes[node].clear()
            graph.nodes[node]["pos"] = (0, 0)

        # Convert back - this should handle empty node records
        nodes_gdf, _edges_gdf_out = converter.nx_to_gdf(graph)
        assert isinstance(nodes_gdf, gpd.GeoDataFrame)

    def test_build_edge_index_empty(self) -> None:
        """Test _build_edge_index with empty original_indices (line 1225)."""
        converter = NxConverter()
        result = converter._build_edge_index([], None)
        assert isinstance(result, pd.Index)
        assert len(result) == 0

    def test_build_node_index_empty(self) -> None:
        """Test _build_node_index with empty original_indices (line 1243)."""
        converter = NxConverter()
        result = converter._build_node_index([], None)
        assert isinstance(result, pd.Index)
        assert len(result) == 0

    def test_empty_graph_node_reconstruction(self, sample_crs: str) -> None:
        """Test nx_to_gdf with graph that has nodes without attributes (line 996)."""
        # Create a minimal graph with nodes that have no custom attributes
        # This will result in empty records when reconstructing
        G = nx.Graph()
        G.add_node(0, pos=(0, 0))
        G.add_node(1, pos=(1, 1))
        G.graph = {"crs": sample_crs, "is_hetero": False, "node_index_names": None}

        # Convert to GDF - should handle empty records gracefully
        nodes_gdf, _edges_gdf = utils.nx_to_gdf(G)
        assert isinstance(nodes_gdf, gpd.GeoDataFrame)
        assert len(nodes_gdf) == 2
        assert "geometry" in nodes_gdf.columns

    def test_edge_iteration_fallback(self, sample_crs: str) -> None:
        """Test edge iteration handles unexpected edge format (line 1200)."""
        # Create a graph and manually add malformed edge to test the fallback
        converter = NxConverter()
        graph = nx.Graph()
        graph.add_node(0, pos=(0, 0))
        graph.add_node(1, pos=(1, 1))
        graph.add_edge(0, 1, geometry=LineString([(0, 0), (1, 1)]))
        graph.graph = {"crs": sample_crs, "is_hetero": False, "edge_index_names": None}

        # The actual iteration path that could hit line 1200 is very defensive
        # It's hard to trigger without modifying internal NetworkX behavior
        # Convert to GDF to exercise the edge reconstruction path
        _, edges_gdf = converter.nx_to_gdf(graph)
        assert isinstance(edges_gdf, gpd.GeoDataFrame)

    def test_gdf_to_nx_edges_none(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test gdf_to_nx raises ValueError when edges is None."""
        converter = NxConverter()
        with pytest.raises(ValueError, match="Edges GeoDataFrame cannot be None"):
            converter.gdf_to_nx(nodes=sample_nodes_gdf, edges=None)

    def test_nx_to_gdf_multiindex_nodes(self, sample_crs: str) -> None:
        """Test nx_to_gdf with MultiIndex nodes."""
        G = nx.Graph()
        # Add nodes with tuple indices
        G.add_node((0, "a"), pos=(0, 0), _original_index=(0, "a"))
        G.add_node((1, "b"), pos=(1, 1), _original_index=(1, "b"))
        G.graph = {"crs": sample_crs, "is_hetero": False, "node_index_names": ["id", "type"]}

        nodes_gdf, _ = utils.nx_to_gdf(G)
        assert isinstance(nodes_gdf, gpd.GeoDataFrame)
        assert isinstance(nodes_gdf.index, pd.MultiIndex)
        assert nodes_gdf.index.names == ["id", "type"]
        assert len(nodes_gdf) == 2

    def test_coerce_name_sequence_string(self) -> None:
        """Test _coerce_name_sequence with string input."""
        assert conversion_utils._coerce_name_sequence("index_name") == ["index_name"]
        assert conversion_utils._coerce_name_sequence(["a", "b"]) == ["a", "b"]
        assert conversion_utils._coerce_name_sequence(None) is None

    def test_nx_to_gdf_empty_graph_no_nodes(self, sample_crs: str) -> None:
        """Test nx_to_gdf with a completely empty graph (no nodes)."""
        G = nx.Graph()
        G.graph = {"crs": sample_crs, "is_hetero": False}
        nodes_gdf, edges_gdf = utils.nx_to_gdf(G)
        assert isinstance(nodes_gdf, gpd.GeoDataFrame)
        assert isinstance(edges_gdf, gpd.GeoDataFrame)
        assert nodes_gdf.empty
        assert edges_gdf.empty
        assert "geometry" in nodes_gdf.columns


class TestValidation:
    """Test validation functions for GeoDataFrames and NetworkX graphs."""

    def test_geo_processor_edge_cases(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        empty_gdf: gpd.GeoDataFrame,
        invalid_geom_gdf: gpd.GeoDataFrame,
        all_invalid_geom_gdf: gpd.GeoDataFrame,
    ) -> None:
        """GeoDataProcessor should handle geometry filtering and allow-empty toggles."""
        processor = GeoDataProcessor()

        result = processor.validate_gdf(
            sample_buildings_gdf,
            expected_geom_types=["Polygon", "MultiPolygon"],
        )
        assert isinstance(result, gpd.GeoDataFrame)

        assert processor.validate_gdf(empty_gdf, allow_empty=True) is not None
        with pytest.raises(ValueError, match="GeoDataFrame cannot be empty"):
            processor.validate_gdf(empty_gdf, allow_empty=False)

        filtered = processor.validate_gdf(invalid_geom_gdf)
        assert filtered is not None
        assert len(filtered) < len(invalid_geom_gdf)

        with pytest.raises(ValueError, match="GeoDataFrame cannot be empty"):
            processor.validate_gdf(all_invalid_geom_gdf, allow_empty=False)

    @pytest.mark.parametrize(
        ("nodes_fixture", "edges_fixture", "should_error", "error_match"),
        [
            # Success cases
            ("sample_nodes_gdf", "sample_edges_gdf", False, None),
            ("sample_nodes_gdf", None, False, None),
            (None, "sample_edges_gdf", False, None),
            ("empty_gdf", "sample_edges_gdf", False, None),
            # Error cases
            ("not_a_gdf", "sample_edges_gdf", True, "Input must be a GeoDataFrame"),
            (
                "sample_nodes_gdf_alt_crs",
                "sample_edges_gdf",
                True,
                "All GeoDataFrames must have the same CRS",
            ),
            (
                "sample_nodes_gdf",
                "edges_dict_for_hetero",
                True,
                "If edges is a dict, nodes must also be a dict or None",
            ),
            (
                "simple_nodes_dict_type1",
                "sample_edges_gdf",
                True,
                "If nodes is a dict, edges must also be a dict or None",
            ),
        ],
    )
    def test_gdf_validation(
        self,
        nodes_fixture: str | None,
        edges_fixture: str | None,
        should_error: bool,
        error_match: str | None,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test GeoDataFrame validation with various input combinations."""
        nodes = request.getfixturevalue(nodes_fixture) if nodes_fixture else None
        edges = request.getfixturevalue(edges_fixture) if edges_fixture else None

        if should_error:
            with pytest.raises((TypeError, ValueError), match=error_match):
                utils.validate_gdf(nodes, edges)
        else:
            utils.validate_gdf(nodes, edges)  # Should not raise

    @pytest.mark.parametrize(
        ("graph_fixture", "should_error", "error_match"),
        [
            ("sample_nx_graph", False, None),
            ("sample_nx_multigraph", False, None),
            ("not_a_gdf", True, "Input must be a NetworkX Graph or MultiGraph"),
        ],
    )
    def test_nx_validation(
        self,
        graph_fixture: str,
        should_error: bool,
        error_match: str | None,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test NetworkX graph validation."""
        graph = request.getfixturevalue(graph_fixture)

        if should_error:
            with pytest.raises(TypeError, match=error_match):
                utils.validate_nx(graph)
        else:
            utils.validate_nx(graph)  # Should not raise

    @pytest.mark.parametrize(
        ("graph_fixture", "error_match"),
        [
            ("graph_missing_crs", "Graph metadata is missing required key"),
            (
                "hetero_graph_no_node_types",
                "Heterogeneous graph metadata is missing 'node_types'",
            ),
            (
                "hetero_graph_no_edge_types",
                "Heterogeneous graph metadata is missing 'edge_types'",
            ),
            ("graph_no_pos_geom", "All nodes must have a 'pos' or 'geometry' attribute"),
            (
                "hetero_graph_no_node_type",
                "All nodes in a heterogeneous graph must have a 'node_type' attribute",
            ),
            (
                "hetero_graph_no_edge_type",
                "All edges in a heterogeneous graph must have an 'edge_type' attribute",
            ),
        ],
    )
    def test_validate_nx_rejects_invalid_graphs(
        self,
        graph_fixture: str,
        error_match: str,
        request: pytest.FixtureRequest,
    ) -> None:
        """GeoDataProcessor.validate_nx should surface detailed errors."""
        graph = request.getfixturevalue(graph_fixture)
        processor = GeoDataProcessor()
        with pytest.raises(ValueError, match=error_match):
            processor.validate_nx(graph)

    def test_validation_edge_cases(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        empty_gdf: gpd.GeoDataFrame,
        segments_invalid_geom_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test validation handles edge cases properly."""
        # Empty edges should be allowed
        utils.validate_gdf(sample_nodes_gdf, empty_gdf)

        # Invalid geometries should be handled with warning
        utils.validate_gdf(sample_nodes_gdf, segments_invalid_geom_gdf)

    def test_heterogeneous_validation_errors(
        self,
        nodes_dict_bad_keys: dict[int, gpd.GeoDataFrame],
        edges_dict_bad_tuple: dict[str, gpd.GeoDataFrame],
        simple_nodes_dict_type1: dict[str, gpd.GeoDataFrame],
        edges_dict_bad_elements: dict[tuple[int, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Ensure validation catches malformed heterogeneous inputs."""
        with pytest.raises(TypeError, match="Node type keys must be strings"):
            gdf_to_nx(nodes=nodes_dict_bad_keys, edges=None)

        with pytest.raises(TypeError, match="Edge type keys must be tuples"):
            gdf_to_nx(nodes=simple_nodes_dict_type1, edges=edges_dict_bad_tuple)

        with pytest.raises(TypeError, match="All elements in edge type tuples must be strings"):
            gdf_to_nx(nodes=simple_nodes_dict_type1, edges=edges_dict_bad_elements)

    def test_validate_nx_basic_structure_errors(self) -> None:
        """Graphs with missing metadata or topology should raise informative errors."""
        processor = GeoDataProcessor()

        empty_graph = nx.Graph()
        with pytest.raises(ValueError, match="Graph has no nodes"):
            processor.validate_nx(empty_graph)

        no_edges_graph = nx.Graph()
        no_edges_graph.add_node(1)
        with pytest.raises(ValueError, match="Graph has no edges"):
            processor.validate_nx(no_edges_graph)

        incomplete_graph = nx.Graph()
        incomplete_graph.add_node(1, pos=(0, 0))
        incomplete_graph.add_edge(1, 2)
        delattr(incomplete_graph, "graph")
        with pytest.raises(ValueError, match="Graph is missing 'graph' attribute dictionary"):
            utils.validate_nx(incomplete_graph)

    def test_validate_nx_pos_from_xy(self, sample_crs: str) -> None:
        """Test validate_nx creates pos from x and y attributes (line 2016)."""
        # Create a graph with x, y but no pos
        G = nx.Graph()
        G.add_node(1, x=10.0, y=20.0)
        G.add_node(2, x=30.0, y=40.0)
        G.add_edge(1, 2, geometry=LineString([(10, 20), (30, 40)]))
        G.graph = {"crs": sample_crs, "is_hetero": False}

        # This should set pos from x and y
        utils.validate_nx(G)

        assert "pos" in G.nodes[1]
        assert "pos" in G.nodes[2]
        assert G.nodes[1]["pos"] == (10.0, 20.0)
        assert G.nodes[2]["pos"] == (30.0, 40.0)


class TestGraphMetadata:
    """Test GraphMetadata class functionality."""

    def test_metadata_creation_and_conversion(self) -> None:
        """Test GraphMetadata creation, conversion, and validation."""
        metadata = GraphMetadata(crs="EPSG:4326", is_hetero=True)
        metadata.node_types = ["building", "road"]
        metadata.edge_types = [("building", "connects", "road")]

        # Test to_dict conversion
        result_dict = metadata.to_dict()
        assert result_dict["crs"] == "EPSG:4326"
        assert result_dict["is_hetero"] is True
        assert result_dict["node_types"] == ["building", "road"]

        # Test from_dict creation
        recreated = GraphMetadata.from_dict(result_dict)
        assert recreated.crs == metadata.crs
        assert recreated.is_hetero == metadata.is_hetero
        assert recreated.node_types == metadata.node_types

    @pytest.mark.parametrize(
        ("invalid_data", "error_type", "error_match"),
        [
            ({"crs": 123.45, "is_hetero": False}, TypeError, "CRS must be str, int, dict"),
            ({"crs": "EPSG:4326", "is_hetero": "not_bool"}, TypeError, "is_hetero must be bool"),
        ],
    )
    def test_metadata_validation_errors(
        self,
        invalid_data: dict[str, object],
        error_type: type[Exception],
        error_match: str,
    ) -> None:
        """Test GraphMetadata validation catches invalid inputs."""
        with pytest.raises(error_type, match=error_match):
            GraphMetadata.from_dict(invalid_data)

    def test_metadata_valid_crs_types(self) -> None:
        """Test GraphMetadata accepts various valid CRS formats."""
        valid_crs_values = ["EPSG:4326", 4326, {"init": "epsg:4326"}, None]

        for crs_value in valid_crs_values:
            metadata = GraphMetadata.from_dict({"crs": crs_value, "is_hetero": False})
            assert metadata.crs == crs_value


class TestRustworkxConversions:
    """Test conversions between NetworkX and rustworkx."""

    @pytest.mark.parametrize(
        "create_graph",
        [
            lambda: nx.Graph(name="undirected"),
            lambda: nx.DiGraph(name="directed"),
            lambda: nx.MultiGraph(name="multi_undirected"),
            lambda: nx.MultiDiGraph(name="multi_directed"),
        ],
    )
    def test_nx_rx_roundtrip(self, create_graph: Callable[[], nx.Graph]) -> None:
        """Test full roundtrip conversion preserves structure and attributes."""
        # Setup complex graph
        G = create_graph()
        G.graph["crs"] = "EPSG:4326"
        G.add_node("a", color="red", size=10)
        G.add_node(1, color="blue")

        # Add edges with attributes
        G.add_edge("a", 1, weight=0.5, type="road")

        # If multi-graph, add another edge between same nodes
        if G.is_multigraph():
            G.add_edge("a", 1, weight=0.8, type="path")

        # Convert to rustworkx
        rx_graph = utils.nx_to_rx(G)

        # Verify RX structure
        assert rx_graph.num_nodes() == G.number_of_nodes()
        assert rx_graph.num_edges() == G.number_of_edges()
        assert rx_graph.attrs["crs"] == "EPSG:4326"

        # Convert back to NetworkX
        G_restored = utils.rx_to_nx(rx_graph)

        # Verify restored graph
        assert nx.utils.graphs_equal(G, G_restored)

    def test_rx_to_nx_raw_input(self) -> None:
        """Test converting a raw rustworkx graph (no __nx_node_id__)."""
        rx_G = rx.PyGraph(multigraph=False)
        rx_G.attrs = {"test": "attr"}

        # Add nodes with raw payload (not dict) and dict payload
        idx1 = rx_G.add_node("raw_payload")
        idx2 = rx_G.add_node({"attr": "value"})

        rx_G.add_edge(idx1, idx2, {"weight": 0.5})

        if rx_G.multigraph:
            rx_G.add_edge(idx1, idx2, {"weight": 0.8})

        # Convert
        nx_G = utils.rx_to_nx(rx_G)

        assert isinstance(nx_G, nx.Graph)
        assert nx_G.graph["test"] == "attr"

        # Check nodes - should use integer indices since no __nx_node_id__
        assert idx1 in nx_G.nodes
        assert nx_G.nodes[idx1]["payload"] == "raw_payload"
        assert nx_G.nodes[idx2]["attr"] == "value"

        # Check edge
        assert nx_G.has_edge(idx1, idx2)
        assert nx_G.edges[idx1, idx2]["weight"] == 0.5

    def test_rx_to_nx_raw_edge_payload(self) -> None:
        """Test converting RX graph with non-dict edge payloads."""
        rx_G = rx.PyGraph(multigraph=False)
        i1 = rx_G.add_node(None)
        i2 = rx_G.add_node(None)
        rx_G.add_edge(i1, i2, "edge_label")

        nx_G = utils.rx_to_nx(rx_G)
        assert nx_G.edges[i1, i2]["payload"] == "edge_label"


class TestIdentifySourceTargetCols:
    """Test the _identify_source_target_cols function."""

    @pytest.fixture
    def basic_edges(self) -> gpd.GeoDataFrame:
        """Create basic edges fixture for testing."""
        return gpd.GeoDataFrame(
            {"u": [1, 2], "v": [2, 3], "weight": [1.0, 2.0]},
            geometry=[LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 2)])],
        )

    def test_explicit_columns(self, basic_edges: gpd.GeoDataFrame) -> None:
        """Test explicit column specification."""
        u, v = conversion_utils._identify_source_target_cols(
            basic_edges, source_col="u", target_col="v"
        )
        assert (u == [1, 2]).all()
        assert (v == [2, 3]).all()

    def test_explicit_index_levels(self, basic_edges: gpd.GeoDataFrame) -> None:
        """Test explicit index level specification."""
        edges = basic_edges.set_index(["u", "v"])
        u, v = conversion_utils._identify_source_target_cols(edges, source_col="u", target_col="v")
        assert (u == [1, 2]).all()
        assert (v == [2, 3]).all()

    def test_implicit_columns_u_v(self, basic_edges: gpd.GeoDataFrame) -> None:
        """Test implicit detection of 'u' and 'v' columns."""
        u, v = conversion_utils._identify_source_target_cols(basic_edges)
        assert (u == [1, 2]).all()
        assert (v == [2, 3]).all()

    def test_implicit_columns_source_target(self, basic_edges: gpd.GeoDataFrame) -> None:
        """Test implicit detection of 'source' and 'target' columns."""
        edges = basic_edges.rename(columns={"u": "source", "v": "target"})
        u, v = conversion_utils._identify_source_target_cols(edges)
        assert (u == [1, 2]).all()
        assert (v == [2, 3]).all()

    def test_implicit_index_from_to_node_id(self, basic_edges: gpd.GeoDataFrame) -> None:
        """Test implicit detection of 'from_node_id' and 'to_node_id' index levels."""
        edges = basic_edges.rename(columns={"u": "from_node_id", "v": "to_node_id"})
        edges = edges.set_index(["from_node_id", "to_node_id"])
        u, v = conversion_utils._identify_source_target_cols(edges)
        assert (u == [1, 2]).all()
        assert (v == [2, 3]).all()

    def test_implicit_index_generic(self, basic_edges: gpd.GeoDataFrame) -> None:
        """Test implicit detection from first two index levels."""
        edges = basic_edges.set_index(["u", "v"])
        # Rename levels to something generic
        edges.index.names = ["level_0", "level_1"]
        u, v = conversion_utils._identify_source_target_cols(edges)
        assert (u == [1, 2]).all()
        assert (v == [2, 3]).all()

    def test_fallback_first_two_columns(self, basic_edges: gpd.GeoDataFrame) -> None:
        """Test fallback to first two columns."""
        edges = basic_edges.rename(columns={"u": "col1", "v": "col2"})
        # Ensure col1 and col2 are first
        edges = edges[["col1", "col2", "weight", "geometry"]]
        u, v = conversion_utils._identify_source_target_cols(edges)
        assert (u == [1, 2]).all()
        assert (v == [2, 3]).all()

    def test_error_missing_explicit(self, basic_edges: gpd.GeoDataFrame) -> None:
        """Test error when explicit columns are missing."""
        with pytest.raises(ValueError, match=r"Source/Target column\(s\) not found: missing"):
            conversion_utils._identify_source_target_cols(
                basic_edges, source_col="missing", target_col="v"
            )

    def test_error_unable_to_identify(self) -> None:
        """Test error when unable to identify columns."""
        edges = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 1)])])
        with pytest.raises(ValueError, match="Could not identify source and target"):
            conversion_utils._identify_source_target_cols(edges)

    def test_only_source_missing_target(self, basic_edges: gpd.GeoDataFrame) -> None:
        """Test error when only source is found but target is missing."""
        # This tests line 1647 - missing target column error
        edges = basic_edges.copy()
        with pytest.raises(
            ValueError, match="Source/Target column\\(s\\) not found: missing_target"
        ):
            conversion_utils._identify_source_target_cols(
                edges, source_col="u", target_col="missing_target"
            )

    def test_standard_index_name_matching(self) -> None:
        """Test _get_col_or_level with standard index name matching (line 1600)."""
        # Create DataFrame with named index
        frame = pd.DataFrame(
            {"col1": [1, 2, 3]},
            index=pd.Index([10, 20, 30], name="my_index"),
        )
        result = conversion_utils._get_col_or_level(frame, "my_index")
        assert result is not None
        assert (result == [10, 20, 30]).all()


class TestPublicUtilityExports:
    """Verify compatibility exports and focused public submodules."""

    def test_root_all_is_unchanged(self) -> None:
        """The package-level wildcard API should match the legacy module."""
        assert utils.__all__ == [
            "canonicalize_edges",
            "clip_graph",
            "create_isochrone",
            "create_tessellation",
            "dual_graph",
            "filter_graph_by_distance",
            "gdf_to_nx",
            "nx_to_gdf",
            "nx_to_rx",
            "plot_graph",
            "remove_isolated_components",
            "rx_to_nx",
            "symmetrize_edges",
            "validate_gdf",
            "validate_nx",
        ]

    @pytest.mark.parametrize(
        ("name", "owner"),
        [
            ("gdf_to_nx", conversion_utils),
            ("nx_to_gdf", conversion_utils),
            ("nx_to_rx", conversion_utils),
            ("rx_to_nx", conversion_utils),
            ("validate_gdf", conversion_utils),
            ("validate_nx", conversion_utils),
            ("canonicalize_edges", topology_utils),
            ("clip_graph", topology_utils),
            ("dual_graph", topology_utils),
            ("remove_isolated_components", topology_utils),
            ("symmetrize_edges", topology_utils),
            ("create_isochrone", spatial_utils),
            ("create_tessellation", spatial_utils),
            ("filter_graph_by_distance", spatial_utils),
            ("plot_graph", spatial_utils),
        ],
    )
    def test_root_exports_are_direct_aliases(self, name: str, owner: object) -> None:
        """Root utility exports should directly alias their owning implementation."""
        assert getattr(utils, name) is getattr(owner, name)

    def test_legacy_non_wildcard_exports_remain_available(self) -> None:
        """Previously importable compatibility names should remain accessible."""
        assert utils.NxConverter is conversion_utils.NxConverter
        assert utils.MATPLOTLIB_AVAILABLE is spatial_utils.MATPLOTLIB_AVAILABLE


class TestRemoveIsolatedComponents(BaseGraphTest):
    """Test remove_isolated_components functionality."""

    def test_nx_to_gdf_pos_from_xy_attributes(self) -> None:
        """Test nx_to_gdf with pos populated from x/y attributes (covers line 2237)."""
        G = nx.Graph()
        G.add_node(1, x=0.0, y=0.0)
        G.add_node(2, x=1.0, y=1.0)
        G.add_edge(1, 2)
        G.graph = {"crs": "EPSG:27700", "is_hetero": False}

        # Should populate 'pos' from x/y when calling nx_to_gdf
        nodes_gdf = nx_to_gdf(G, nodes=True, edges=False)
        assert isinstance(nodes_gdf, gpd.GeoDataFrame)
        assert len(nodes_gdf) == 2
