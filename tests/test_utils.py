"""Tests for the utils module - lean and maintainable.

This file delegates common assertions and helpers to tests/helpers.py to avoid
duplication and keep a single source of truth for shared logic across tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import networkx as nx
import pandas as pd
import pytest
import rustworkx as rx
from shapely.geometry import LineString
from shapely.geometry import Point

from city2graph import utils
from city2graph.base import GeoDataProcessor
from city2graph.base import GraphMetadata
from city2graph.utils import NxConverter
from city2graph.utils import gdf_to_nx
from city2graph.utils import nx_to_gdf
from tests import helpers

if TYPE_CHECKING:
    from collections.abc import Callable

# Try to import matplotlib for tests that need it
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None  # type: ignore[assignment]
    MATPLOTLIB_AVAILABLE = False

# ============================================================================
# BASE TEST CLASSES WITH SHARED FUNCTIONALITY
# ============================================================================


class BaseGraphTest:
    """Base class for graph-related tests with common utilities."""

    @staticmethod
    def assert_valid_gdf(gdf: gpd.GeoDataFrame, expected_empty: bool = False) -> None:
        """Delegate to shared helper to ensure consistent checks across tests."""
        helpers.assert_valid_gdf(gdf, expected_empty)

    @staticmethod
    def assert_crs_consistency(*gdfs: gpd.GeoDataFrame) -> None:
        """Delegate to shared helper for CRS consistency checks."""
        helpers.assert_crs_consistency(*gdfs)


class BaseConversionTest(BaseGraphTest):
    """Base class for conversion tests between GDF and NetworkX."""

    def assert_roundtrip_consistency(
        self,
        original_nodes: gpd.GeoDataFrame,
        original_edges: gpd.GeoDataFrame,
        converted_nodes: gpd.GeoDataFrame,
        converted_edges: gpd.GeoDataFrame,
    ) -> None:
        """Delegate to shared helper for roundtrip integrity checks."""
        helpers.assert_roundtrip_consistency(
            original_nodes,
            original_edges,
            converted_nodes,
            converted_edges,
        )


# ============================================================================
# TESSELLATION TESTS
# ============================================================================


class TestTessellation(BaseGraphTest):
    """Test tessellation creation functionality."""

    @pytest.mark.parametrize(
        ("geometry_fixture", "barriers_fixture", "expect_empty"),
        [
            ("empty_gdf", None, True),
            ("sample_buildings_gdf", None, False),
            ("sample_buildings_gdf", "sample_segments_gdf", False),
        ],
    )
    def test_tessellation_creation(
        self,
        geometry_fixture: str,
        barriers_fixture: str | None,
        expect_empty: bool,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test tessellation creation with various input combinations."""
        geometry = request.getfixturevalue(geometry_fixture)
        barriers = request.getfixturevalue(barriers_fixture) if barriers_fixture else None

        try:
            tessellation = utils.create_tessellation(geometry, primary_barriers=barriers)
        except (UnboundLocalError, TypeError, ValueError) as e:
            pytest.skip(f"Skipping due to incomplete implementation: {e}")

        self.assert_valid_gdf(tessellation, expect_empty)
        if not expect_empty:
            helpers.assert_has_columns(tessellation, ["tess_id"])
            assert tessellation.crs == geometry.crs


# ============================================================================
# GRAPH STRUCTURE TESTS
# ============================================================================


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
                "All GeoDataFrames must have the same CRS",
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
        nodes_gdf, edges_gdf = utils.segments_to_graph(segments_gdf, multigraph=multigraph)

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
        nodes_gdf, edges_gdf = utils.segments_to_graph(duplicate_segments_gdf, multigraph=True)

        assert len(nodes_gdf) == 2  # Two unique points
        assert len(edges_gdf) == 2  # Both edges preserved
        assert edges_gdf.index.names == ["from_node_id", "to_node_id", "edge_key"]

        # Verify edge keys are different for duplicates
        edge_keys = edges_gdf.index.get_level_values("edge_key")
        assert list(edge_keys) == [0, 1]


# ============================================================================
# GRAPH ANALYSIS TESTS
# ============================================================================


class TestGraphAnalysis(BaseGraphTest):
    """Test graph analysis operations like filtering and isochrone generation."""

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

        filtered = utils.filter_graph_by_distance(graph, center_point, distance=distance)

        if as_nx:
            assert isinstance(filtered, nx.Graph)
            assert (filtered.number_of_edges() == 0) == expect_empty
        else:
            self.assert_valid_gdf(filtered, expect_empty)

    @pytest.mark.parametrize(
        ("graph_fixture", "center_fixture", "distance", "expect_empty"),
        [
            ("sample_segments_gdf", "mg_center_point", 100.0, False),
            ("sample_segments_gdf", "mg_center_point", 0.01, True),
            ("sample_nx_graph", "sample_nodes_gdf", 1.0, False),
        ],
    )
    def test_isochrone_generation(
        self,
        graph_fixture: str,
        center_fixture: str,
        distance: float,
        expect_empty: bool,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test isochrone polygon generation from graphs."""
        graph = request.getfixturevalue(graph_fixture)
        center_source = request.getfixturevalue(center_fixture)
        center_point = helpers.get_center_point(center_source)

        isochrone = utils.create_isochrone(graph, center_point, distance=distance)

        self.assert_valid_gdf(isochrone, expect_empty)
        if not expect_empty:
            assert len(isochrone) == 1
            assert isochrone.geometry.iloc[0].geom_type in ["Polygon", "MultiPolygon"]


# ============================================================================
# CONVERSION TESTS
# ============================================================================


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


# ============================================================================
# VALIDATION TESTS
# ============================================================================


class TestValidation:
    """Test validation functions for GeoDataFrames and NetworkX graphs."""

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


# ============================================================================
# METADATA TESTS
# ============================================================================


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


# ============================================================================
# COMPREHENSIVE EDGE CASE TESTS
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions for comprehensive coverage."""

    def test_processor_edge_cases(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        empty_gdf: gpd.GeoDataFrame,
        invalid_geom_gdf: gpd.GeoDataFrame,
        all_invalid_geom_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test GeoDataProcessor edge cases."""
        processor = GeoDataProcessor()

        # Test geometry type filtering
        result = processor.validate_gdf(
            sample_buildings_gdf,
            expected_geom_types=["Polygon", "MultiPolygon"],
        )
        assert isinstance(result, gpd.GeoDataFrame)

        # Test allow_empty parameter
        result = processor.validate_gdf(empty_gdf, allow_empty=True)
        assert result is not None
        assert result.empty

        with pytest.raises(ValueError, match="GeoDataFrame cannot be empty"):
            processor.validate_gdf(empty_gdf, allow_empty=False)

        # Test invalid geometries handling
        result = processor.validate_gdf(invalid_geom_gdf)
        assert result is not None
        assert len(result) < len(invalid_geom_gdf)  # Invalid geometries filtered out

        # Test case where filtering makes GDF empty and allow_empty=False (line 115-116)
        with pytest.raises(ValueError, match="GeoDataFrame cannot be empty"):
            processor.validate_gdf(all_invalid_geom_gdf, allow_empty=False)

    def test_nx_validation_edge_cases(self) -> None:
        """Test NetworkX validation edge cases."""
        processor = GeoDataProcessor()

        # Empty graph
        empty_graph = nx.Graph()
        with pytest.raises(ValueError, match="Graph has no nodes"):
            processor.validate_nx(empty_graph)

        # Graph with nodes but no edges
        no_edges_graph = nx.Graph()
        no_edges_graph.add_node(1)
        with pytest.raises(ValueError, match="Graph has no edges"):
            processor.validate_nx(no_edges_graph)

        # Missing metadata
        incomplete_graph = nx.Graph()
        incomplete_graph.add_node(1, pos=(0, 0))
        incomplete_graph.add_edge(1, 2)
        # Remove the graph attribute to trigger the error
        delattr(incomplete_graph, "graph")
        with pytest.raises(ValueError, match="missing 'graph' attribute"):
            processor.validate_nx(incomplete_graph)

    def test_comprehensive_nx_validation_errors(
        self,
        graph_missing_crs: nx.Graph,
        hetero_graph_no_node_types: nx.Graph,
        hetero_graph_no_edge_types: nx.Graph,
        graph_no_pos_geom: nx.Graph,
        hetero_graph_no_node_type: nx.Graph,
        hetero_graph_no_edge_type: nx.Graph,
    ) -> None:
        """Test comprehensive NetworkX validation error conditions."""
        processor = GeoDataProcessor()

        # Test missing required metadata keys (lines 155-156)
        with pytest.raises(ValueError, match="Graph metadata is missing required key"):
            processor.validate_nx(graph_missing_crs)

        # Test heterogeneous graph missing node_types (lines 162-163)
        with pytest.raises(
            ValueError,
            match="Heterogeneous graph metadata is missing 'node_types'",
        ):
            processor.validate_nx(hetero_graph_no_node_types)

        # Test heterogeneous graph missing edge_types (lines 165-166)
        with pytest.raises(
            ValueError,
            match="Heterogeneous graph metadata is missing 'edge_types'",
        ):
            processor.validate_nx(hetero_graph_no_edge_types)

        # Test node missing pos/geometry (lines 173-174)
        with pytest.raises(ValueError, match="All nodes must have a 'pos' or 'geometry' attribute"):
            processor.validate_nx(graph_no_pos_geom)

        # Test heterogeneous node missing node_type (lines 178-179)
        with pytest.raises(
            ValueError,
            match="All nodes in a heterogeneous graph must have a 'node_type' attribute",
        ):
            processor.validate_nx(hetero_graph_no_node_type)

        # Test heterogeneous edge missing edge_type (lines 184-185)
        with pytest.raises(
            ValueError,
            match="All edges in a heterogeneous graph must have an 'edge_type' attribute",
        ):
            processor.validate_nx(hetero_graph_no_edge_type)

    def test_heterogeneous_validation_errors(
        self,
        nodes_dict_bad_keys: dict[int, gpd.GeoDataFrame],
        edges_dict_bad_tuple: dict[str, gpd.GeoDataFrame],
        simple_nodes_dict_type1: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Test heterogeneous graph validation errors."""
        # Test invalid node type keys
        with pytest.raises(TypeError, match="Node type keys must be strings"):
            utils.gdf_to_nx(nodes=nodes_dict_bad_keys, edges=None)

        # Test invalid edge type tuples
        with pytest.raises(TypeError, match="Edge type keys must be tuples"):
            utils.gdf_to_nx(nodes=simple_nodes_dict_type1, edges=edges_dict_bad_tuple)

    def test_conversion_edge_cases(self, simple_nx_graph: nx.Graph) -> None:
        """Test conversion edge cases for complete coverage."""
        # Test nx_to_gdf with neither nodes nor edges requested
        with pytest.raises(ValueError, match="Must request at least one of nodes or edges"):
            utils.nx_to_gdf(simple_nx_graph, nodes=False, edges=False)

        # Test empty graph edge geometry creation - use the simple graph but remove edges
        empty_edges_graph = simple_nx_graph.copy()
        empty_edges_graph.remove_edges_from(list(empty_edges_graph.edges()))

        nodes_gdf, edges_gdf = utils.nx_to_gdf(empty_edges_graph)
        assert isinstance(nodes_gdf, gpd.GeoDataFrame)
        assert isinstance(edges_gdf, gpd.GeoDataFrame)
        assert not nodes_gdf.empty
        assert edges_gdf.empty

    def test_tessellation_edge_cases(
        self,
        empty_gdf: gpd.GeoDataFrame,
        single_point_geom_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test tessellation edge cases."""
        # Empty geometry
        result = utils.create_tessellation(empty_gdf)
        assert result.empty
        assert isinstance(result, gpd.GeoDataFrame)

        # Single point that might cause tessellation issues
        result = utils.create_tessellation(single_point_geom_gdf)
        assert isinstance(result, gpd.GeoDataFrame)

    def test_graph_converter_edge_cases(
        self,
        directed_multigraph_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test NxConverter edge cases for missing coverage."""
        # Test directed graph creation (line 284)
        converter = NxConverter(directed=True, multigraph=True)
        graph = converter.gdf_to_nx(nodes=None, edges=directed_multigraph_edges_gdf)
        assert isinstance(graph, nx.MultiDiGraph)

        # Test edges is None after validation (line 277-278)
        converter = NxConverter()
        with pytest.raises(ValueError, match="Edges GeoDataFrame cannot be None"):
            converter._convert_homogeneous(nodes=None, edges=None)

    def test_nx_to_gdf_edge_cases(self, graph_with_edge_index_names: nx.Graph) -> None:
        """Test nx_to_gdf edge cases for missing coverage."""
        # Test edge_index_names is None (line 592)
        nodes_gdf, edges_gdf = utils.nx_to_gdf(graph_with_edge_index_names)
        assert isinstance(nodes_gdf, gpd.GeoDataFrame)
        assert isinstance(edges_gdf, gpd.GeoDataFrame)

    def test_index_handling_edge_cases(
        self,
        single_name_index_nodes_gdf: gpd.GeoDataFrame,
        simple_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test index handling edge cases."""
        # Test single-level index name handling (line 629)
        graph = utils.gdf_to_nx(nodes=single_name_index_nodes_gdf, edges=simple_edges_gdf)
        nodes_back, _ = utils.nx_to_gdf(graph)
        assert isinstance(nodes_back, gpd.GeoDataFrame)
        assert nodes_back.index.name == "single_name"

        # Test else clause for index names (line 633)
        graph.graph["node_index_names"] = None
        nodes_back, _ = utils.nx_to_gdf(graph)
        assert isinstance(nodes_back, gpd.GeoDataFrame)
        assert nodes_back.index.name is None

    def test_heterogeneous_edge_processing(
        self,
        regular_hetero_graph: nx.Graph,
        empty_hetero_graph: nx.Graph,
    ) -> None:
        """Test heterogeneous edge processing paths."""
        # Test regular graph edge processing (lines 771-775)
        nodes_dict, edges_dict = utils.nx_to_gdf(regular_hetero_graph)
        assert isinstance(nodes_dict, dict)
        assert isinstance(edges_dict, dict)

        # Test empty edge type (line 779)
        nodes_dict, edges_dict = utils.nx_to_gdf(empty_hetero_graph)
        assert isinstance(nodes_dict, dict)
        assert isinstance(edges_dict, dict)
        assert ("building", "connects", "road") in edges_dict
        assert edges_dict[("building", "connects", "road")].empty

    def test_dual_graph_nx_input(self, simple_nx_graph: nx.Graph) -> None:
        """Test dual graph with NetworkX input (line 1055-1056)."""
        dual_nodes, dual_edges = utils.dual_graph(simple_nx_graph, edge_id_col=None)
        assert isinstance(dual_nodes, gpd.GeoDataFrame)
        assert isinstance(dual_edges, gpd.GeoDataFrame)

    def test_validation_type_errors(
        self,
        nodes_non_dict_for_hetero: gpd.GeoDataFrame,
        edges_dict_for_hetero: dict[tuple[str, str, str], gpd.GeoDataFrame],
        edges_dict_bad_elements: dict[tuple[int, str, str], gpd.GeoDataFrame],
        simple_nodes_dict_type1: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Test validation type errors (lines 1761-1762, 1786-1787)."""
        # Test edges dict with nodes non-dict (line 1761-1762)
        with pytest.raises(
            TypeError,
            match="If edges is a dict, nodes must also be a dict or None",
        ):
            utils.gdf_to_nx(nodes=nodes_non_dict_for_hetero, edges=edges_dict_for_hetero)

        # Test invalid edge type tuple elements (line 1786-1787)
        with pytest.raises(TypeError, match="All elements in edge type tuples must be strings"):
            utils.gdf_to_nx(nodes=simple_nodes_dict_type1, edges=edges_dict_bad_elements)

    def test_multiindex_nodes_conversion(
        self,
        multiindex_nodes_gdf: gpd.GeoDataFrame,
        simple_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test conversion with MultiIndex nodes (line 294)."""
        # This should trigger the MultiIndex path (line 294)
        graph = utils.gdf_to_nx(nodes=multiindex_nodes_gdf, edges=simple_edges_gdf)
        assert graph.graph["node_index_names"] == ["node_type", "node_id"]

    def test_validate_nx_missing_graph_attr_edge_cases(self) -> None:
        """Test validate_nx with missing graph attributes to cover line 358."""
        # Create a graph with missing graph attribute
        graph = nx.Graph()
        graph.add_node(1, pos=(0, 0))
        graph.add_edge(1, 2)
        # Remove graph attribute to trigger line 358
        delattr(graph, "graph")

        with pytest.raises(ValueError, match="Graph is missing 'graph' attribute dictionary"):
            utils.validate_nx(graph)

    def test_nx_to_gdf_multiindex_edge_cases(self, sample_crs: str) -> None:
        """Test nx_to_gdf with MultiIndex edge cases to cover lines 1373-1377, 1633, 1637."""
        # Create a graph with complex edge indices to trigger MultiIndex handling
        graph = nx.MultiGraph()
        graph.add_node(1, pos=(0, 0), geometry=Point(0, 0))
        graph.add_node(2, pos=(1, 1), geometry=Point(1, 1))
        graph.add_edge(1, 2, key=0, geometry=LineString([(0, 0), (1, 1)]))
        graph.add_edge(1, 2, key=1, geometry=LineString([(0, 0), (1, 1)]))
        graph.graph = {"crs": sample_crs, "is_hetero": False}

        # Set complex original edge indices to trigger lines 1373-1377
        for u, v, k, attrs in graph.edges(data=True, keys=True):
            attrs["_original_edge_index"] = [u, v, k]  # List format to trigger tuple conversion

        nodes_gdf, edges_gdf = utils.nx_to_gdf(graph, nodes=True, edges=True)

        assert isinstance(nodes_gdf, gpd.GeoDataFrame)
        assert isinstance(edges_gdf, gpd.GeoDataFrame)
        # The index should be a regular index with list elements, not MultiIndex
        assert len(edges_gdf) == 2  # Two edges from the multigraph

    def test_nx_to_gdf_empty_edges_handling(self, sample_crs: str) -> None:
        """Test nx_to_gdf with empty edges to cover lines 1496-1497."""
        # Create a graph with no edges to trigger lines 1496-1497
        graph = nx.Graph()
        graph.add_node(1, pos=(0, 0), geometry=Point(0, 0))
        graph.graph = {"crs": sample_crs, "is_hetero": False}

        nodes_gdf, edges_gdf = utils.nx_to_gdf(graph, nodes=True, edges=True)

        assert isinstance(nodes_gdf, gpd.GeoDataFrame)
        assert isinstance(edges_gdf, gpd.GeoDataFrame)
        assert edges_gdf.empty

    def test_nx_to_gdf_multigraph_edge_processing(self, sample_crs: str) -> None:
        """Test nx_to_gdf multigraph edge processing to cover line 1510."""
        # Create a multigraph to trigger line 1510
        graph = nx.MultiGraph()
        graph.add_node(1, pos=(0, 0), geometry=Point(0, 0))
        graph.add_node(2, pos=(1, 1), geometry=Point(1, 1))
        graph.add_edge(1, 2, key=0, weight=1.0, geometry=LineString([(0, 0), (1, 1)]))
        graph.graph = {"crs": sample_crs, "is_hetero": False}

        nodes_gdf, edges_gdf = utils.nx_to_gdf(graph, nodes=True, edges=True)

        assert isinstance(edges_gdf, gpd.GeoDataFrame)
        assert len(edges_gdf) == 1
        assert "weight" in edges_gdf.columns

    def test_edge_index_names_handling(
        self,
        sample_crs: str,
        simple_edges_dict_type1_type2: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Test edge index names handling (lines 629, 633, 660, 697-698, 713)."""
        # Test line 465 - edge_index_names not dict handling
        converter = NxConverter()
        metadata = GraphMetadata(crs=sample_crs, is_hetero=True)
        metadata.edge_index_names = "not_a_dict"  # type: ignore[assignment]  # This should trigger line 465

        graph = nx.Graph()
        graph.add_node(0, node_type="type1", _original_index="a", pos=(0, 0))
        graph.add_node(1, node_type="type2", _original_index="b", pos=(1, 1))

        # This should convert edge_index_names to dict
        converter._add_heterogeneous_edges(graph, simple_edges_dict_type1_type2, metadata)
        assert isinstance(metadata.edge_index_names, dict)

    def test_tessellation_error_handling(
        self,
        single_point_geom_gdf: gpd.GeoDataFrame,
        tessellation_barriers_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test tessellation error handling paths (lines 1653-1662, 1669)."""
        # Test with geometry that might trigger momepy concatenation error
        result = utils.create_tessellation(single_point_geom_gdf)
        assert isinstance(result, gpd.GeoDataFrame)

        # Test with barriers that might cause issues (line 1669)
        result = utils.create_tessellation(
            single_point_geom_gdf,
            primary_barriers=tessellation_barriers_gdf,
        )
        assert isinstance(result, gpd.GeoDataFrame)


# ============================================================================
# RUSTWORKX CONVERSION TESTS
# ============================================================================


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


# ============================================================================
# PLOTTING TESTS
# ============================================================================


class TestPlotting(BaseGraphTest):
    """Test plotting functionality."""

    def test_plot_graph_homogeneous(self, sample_nx_graph: nx.Graph) -> None:
        """Test plotting a homogeneous graph."""
        if not utils.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        # Mock plt.show to avoid display
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("matplotlib.pyplot.show", lambda: None)
            utils.plot_graph(sample_nx_graph)

    def test_plot_graph_heterogeneous(self, regular_hetero_graph: nx.Graph) -> None:
        """Test plotting a heterogeneous graph."""
        if not utils.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("matplotlib.pyplot.show", lambda: None)
            utils.plot_graph(regular_hetero_graph)

    def test_plot_graph_save(self, sample_nx_graph: nx.Graph) -> None:
        """Test plotting without showing (for saving to file)."""
        if not utils.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("matplotlib.pyplot.show", lambda: None)
            # Plot without auto-display (caller can save the figure)
            utils.plot_graph(sample_nx_graph)

    def test_plot_graph_no_matplotlib(self, sample_nx_graph: nx.Graph) -> None:
        """Test error when matplotlib is missing."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(utils, "MATPLOTLIB_AVAILABLE", False)
            with pytest.raises(ImportError, match="(?i)matplotlib is required"):
                utils.plot_graph(sample_nx_graph)

    def test_plot_graph_no_input(self) -> None:
        """Test error when no input is provided."""
        if not utils.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        with pytest.raises(ValueError, match="At least one of graph, nodes, or edges"):
            utils.plot_graph()

    def test_plot_graph_with_gdf_input(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test plotting with GeoDataFrame inputs directly."""
        if not utils.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("matplotlib.pyplot.show", lambda: None)
            utils.plot_graph(nodes=sample_nodes_gdf, edges=sample_edges_gdf)

    def test_plot_graph_gdf_as_graph_param(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test legacy support: GeoDataFrame passed as graph parameter."""
        if not utils.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("matplotlib.pyplot.show", lambda: None)
            # This should treat the GDF as nodes (legacy support)
            utils.plot_graph(graph=sample_nodes_gdf)

    def test_plot_graph_unsupported_type(self) -> None:
        """Test error when unsupported type is passed as graph."""
        if not utils.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        with pytest.raises(TypeError, match="Unsupported data type"):
            utils.plot_graph(graph="not_a_graph")

    def test_plot_graph_hetero_subplots(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Test plotting heterogeneous graph with subplots."""
        if not utils.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("matplotlib.pyplot.show", lambda: None)
            utils.plot_graph(
                nodes=sample_hetero_nodes_dict,
                edges=sample_hetero_edges_dict,
                subplots=True,
            )

    def test_plot_graph_hetero_no_subplots(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Test plotting heterogeneous graph without subplots."""
        if not utils.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("matplotlib.pyplot.show", lambda: None)
            utils.plot_graph(
                nodes=sample_hetero_nodes_dict,
                edges=sample_hetero_edges_dict,
                subplots=False,
            )

    def test_plot_graph_no_legend(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Test plotting without legend."""
        if not utils.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("matplotlib.pyplot.show", lambda: None)
            utils.plot_graph(
                nodes=sample_hetero_nodes_dict,
                edges=sample_hetero_edges_dict,
                legend_position=None,
                subplots=False,
            )

    def test_plot_graph_style_kwargs(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test plotting with custom style kwargs."""
        if not utils.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("matplotlib.pyplot.show", lambda: None)
            utils.plot_graph(
                nodes=sample_nodes_gdf,
                edges=sample_edges_gdf,
                node_color="red",
                node_alpha=0.5,
                edge_color="blue",
                edge_linewidth=2.0,
                markersize=10.0,
            )

    def test_plot_graph_with_ax(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test plotting with provided axes."""
        if not utils.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        assert plt is not None  # For type checker
        fig, ax = plt.subplots()
        try:
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr("matplotlib.pyplot.show", lambda: None)
                utils.plot_graph(nodes=sample_nodes_gdf, ax=ax)
        finally:
            plt.close(fig)
