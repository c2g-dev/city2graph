"""
Streamlined tests for the graph module.

This module provides comprehensive test coverage for city2graph.graph with improved
maintainability and clear organization. Tests are organized by core functionality
with minimal redundancy.

Key improvements:
- Simplified test structure focused on core functionality
- Reduced redundancy through better parameterization
- Clear separation of concerns
- Easier to maintain and extend
"""

from __future__ import annotations

from typing import Any
from typing import cast

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import Polygon
from torch_geometric.data import Data
from torch_geometric.data import HeteroData

from city2graph import graph as graph_module
from city2graph.base import GraphMetadata
from city2graph.utils import gdf_to_nx

# Import torch-related modules conditionally
try:
    import torch
    from torch_geometric.data import Data
    from torch_geometric.data import HeteroData

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Skip all tests if torch is not available
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")

# Import graph functions only if torch is available
if TORCH_AVAILABLE:
    from city2graph.graph import gdf_to_pyg
    from city2graph.graph import is_torch_available
    from city2graph.graph import nx_to_pyg
    from city2graph.graph import pyg_to_gdf
    from city2graph.graph import pyg_to_nx
    from city2graph.graph import validate_pyg


class TestTorchAvailability:
    """Test PyTorch availability detection and error handling."""

    def test_is_torch_available_returns_true(self) -> None:
        """Test that is_torch_available returns True when torch is available."""
        assert is_torch_available() is True

    def test_functions_raise_import_error_without_torch(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that functions raise ImportError when torch is not available."""
        monkeypatch.setattr(graph_module, "TORCH_AVAILABLE", False)

        functions_to_test: list[tuple[Any, tuple[Any, ...]]] = [
            (graph_module.gdf_to_pyg, ({}, {})),
            (graph_module.validate_pyg, ({},)),
            (graph_module.pyg_to_gdf, ({},)),
            (graph_module.pyg_to_nx, ({},)),
        ]

        for func, args in functions_to_test:
            if isinstance(args, tuple):
                with pytest.raises(ImportError, match="PyTorch and PyTorch Geometric required"):
                    func(*args)
            else:
                with pytest.raises(ImportError, match="PyTorch and PyTorch Geometric required"):
                    func(args)

        # Test nx_to_pyg with valid graph
        valid_graph = nx.Graph()
        valid_graph.add_node(1, feature1=10.0, pos=(0, 0))
        valid_graph.add_node(2, feature1=20.0, pos=(1, 1))
        valid_graph.add_edge(1, 2)
        valid_graph.graph["is_hetero"] = False
        valid_graph.graph["crs"] = "EPSG:4326"

        with pytest.raises(ImportError, match="PyTorch and PyTorch Geometric required"):
            graph_module.nx_to_pyg(valid_graph)


class TestGraphConversion:
    """Test graph conversion functionality between different formats."""

    @pytest.fixture
    def sample_geom_data(self) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Create sample data with curved geometries for testing."""
        # Create sample nodes
        nodes_data = {
            "node_id": [1, 2, 3],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
            "feature": [1.0, 2.0, 3.0],
        }
        nodes = gpd.GeoDataFrame(nodes_data).set_index("node_id")
        nodes.crs = "EPSG:4326"

        # Create sample edges
        edges_data = {
            "source_id": [1, 2],
            "target_id": [2, 3],
            "geometry": [
                LineString([(0, 0), (1, 1)]),
                LineString([(1, 1), (2, 2)]),
            ],
            "weight": [0.5, 1.5],
        }
        edges = gpd.GeoDataFrame(edges_data).set_index(["source_id", "target_id"])
        edges.crs = "EPSG:4326"

        return nodes, edges

    @pytest.mark.parametrize("graph_type", ["homogeneous", "heterogeneous"])
    def test_gdf_to_pyg_basic(
        self,
        graph_type: str,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Test basic conversion from GeoDataFrames to PyG objects."""
        if graph_type == "homogeneous":
            data = gdf_to_pyg(sample_nodes_gdf, sample_edges_gdf)
            assert isinstance(data, Data)
            assert data.num_nodes == len(sample_nodes_gdf)
            assert data.num_edges == len(sample_edges_gdf)
            assert not data.graph_metadata.is_hetero
        else:
            data = gdf_to_pyg(sample_hetero_nodes_dict, sample_hetero_edges_dict)
            assert isinstance(data, HeteroData)
            assert data.graph_metadata.is_hetero
            assert set(data.node_types) == set(sample_hetero_nodes_dict.keys())
            assert set(data.edge_types) == set(sample_hetero_edges_dict.keys())

    @pytest.mark.parametrize("graph_type", ["homogeneous", "heterogeneous"])
    def test_gdf_to_pyg_with_features(
        self,
        graph_type: str,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Test conversion with node and edge features."""
        if graph_type == "homogeneous":
            data = gdf_to_pyg(
                sample_nodes_gdf,
                sample_edges_gdf,
                node_feature_cols=["feature1"],
                node_label_cols=["label1"],
                edge_feature_cols=["edge_feature1"],
            )
            assert data.x.shape[1] == 1
            assert data.y.shape[1] == 1
            assert data.edge_attr.shape[1] == 1
        else:
            data = gdf_to_pyg(
                sample_hetero_nodes_dict,
                sample_hetero_edges_dict,
                node_feature_cols={"building": ["b_feat1"], "road": ["length"]},
                node_label_cols={"building": ["b_label"]},
                edge_feature_cols={"connects_to": ["conn_feat1"], "links_to": ["link_feat1"]},
            )
            assert data["building"].x.shape[1] == 1
            assert data["road"].x.shape[1] == 1
            assert data["building"].y.shape[1] == 1

    @pytest.mark.parametrize("graph_type", ["homogeneous", "heterogeneous"])
    def test_round_trip_conversion(
        self,
        graph_type: str,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Test that data survives round-trip conversion (GDF -> PyG -> GDF)."""
        if graph_type == "homogeneous":
            data = gdf_to_pyg(sample_nodes_gdf, sample_edges_gdf)
            nodes_restored, edges_restored = pyg_to_gdf(data)

            assert isinstance(nodes_restored, gpd.GeoDataFrame)
            assert isinstance(edges_restored, gpd.GeoDataFrame)
            assert len(nodes_restored) == len(sample_nodes_gdf)
            assert len(edges_restored) == len(sample_edges_gdf)
        else:
            data = gdf_to_pyg(sample_hetero_nodes_dict, sample_hetero_edges_dict)
            nodes_restored, edges_restored = pyg_to_gdf(data)

            assert isinstance(nodes_restored, dict)
            assert isinstance(edges_restored, dict)
            assert set(nodes_restored.keys()) == set(sample_hetero_nodes_dict.keys())
            assert set(edges_restored.keys()) == set(sample_hetero_edges_dict.keys())

    def test_nx_conversions(self, sample_nx_graph: nx.Graph) -> None:
        """Test NetworkX conversions."""
        # NX -> PyG
        data = nx_to_pyg(sample_nx_graph)
        assert isinstance(data, Data)
        assert data.num_nodes == sample_nx_graph.number_of_nodes()
        assert data.num_edges == sample_nx_graph.number_of_edges()

        # PyG -> NX (round trip)
        nx_restored = pyg_to_nx(data)
        assert isinstance(nx_restored, nx.Graph)
        assert nx_restored.graph.get("is_hetero") is False
        assert nx_restored.number_of_nodes() == sample_nx_graph.number_of_nodes()
        assert nx_restored.number_of_edges() == sample_nx_graph.number_of_edges()
        assert nx_restored.graph.get("crs") == sample_nx_graph.graph.get("crs")

    def test_pyg_to_nx_heterogeneous(self, sample_pyg_hetero_data: HeteroData) -> None:
        """Test PyG to NetworkX conversion for heterogeneous graphs."""
        nx_graph = pyg_to_nx(sample_pyg_hetero_data)
        assert isinstance(nx_graph, nx.Graph)
        assert nx_graph.graph.get("is_hetero") is True
        assert "node_types" in nx_graph.graph
        assert "edge_types" in nx_graph.graph

    def test_nx_default_feature_and_label_naming(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test NetworkX conversion with both default feature and label naming."""
        # Create data with both features and labels
        data = gdf_to_pyg(
            sample_nodes_gdf,
            node_feature_cols=["feature1"],
            node_label_cols=["label1"],
        )

        # Clear both feature and label column names to trigger default naming
        data.graph_metadata.node_feature_cols = None
        data.graph_metadata.node_label_cols = None

        nx_graph = pyg_to_nx(data)

        # Check that nodes have both default feature and label attributes
        for _node_id, node_data in nx_graph.nodes(data=True):
            assert isinstance(node_data, dict)
            # Should have default feature attributes
            feat_attrs = [key for key in node_data if key.startswith("feat_")]
            assert len(feat_attrs) > 0, "Should have default feature attributes"
            # Should have default label attributes
            label_attrs = [key for key in node_data if key.startswith("label_")]
            assert len(label_attrs) > 0, "Should have default label attributes"
            break

    def test_empty_inputs(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test handling of empty inputs."""
        # No edges
        data = gdf_to_pyg(sample_nodes_gdf, edges=None)
        assert data.edge_index.shape == (2, 0)
        assert data.edge_attr.shape == (0, 0)

        # Round trip with no edges
        nodes_restored, edges_restored = pyg_to_gdf(data)
        assert isinstance(nodes_restored, gpd.GeoDataFrame)
        assert isinstance(edges_restored, gpd.GeoDataFrame)

    def test_empty_features(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test handling of empty feature columns."""
        # Empty feature columns
        data = gdf_to_pyg(
            sample_nodes_gdf,
            sample_edges_gdf,
            node_feature_cols=[],
            node_label_cols=[],
            edge_feature_cols=[],
        )
        assert data.x.shape[1] == 0
        assert data.edge_attr.shape[1] == 0

        # Non-existent feature columns
        data = gdf_to_pyg(
            sample_nodes_gdf,
            sample_edges_gdf,
            node_feature_cols=["nonexistent"],
            edge_feature_cols=["nonexistent"],
        )
        assert data.x.shape[1] == 0
        assert data.edge_attr.shape[1] == 0

    def test_empty_nx_graph(self) -> None:
        """Test that empty NetworkX graph raises error."""
        empty_graph = nx.Graph()
        with pytest.raises(ValueError, match="Graph has no nodes"):
            nx_to_pyg(empty_graph)

    @pytest.mark.parametrize(
        ("feature_type", "graph_type"),
        [
            ("node_feature_cols", "homogeneous"),
            ("node_label_cols", "homogeneous"),
            ("edge_feature_cols", "homogeneous"),
            ("node_feature_cols", "heterogeneous"),
            ("node_label_cols", "heterogeneous"),
            ("edge_feature_cols", "heterogeneous"),
        ],
    )
    def test_invalid_feature_types(
        self,
        feature_type: str,
        graph_type: str,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Test invalid feature column types."""
        if graph_type == "homogeneous":
            kwargs: dict[str, Any] = {feature_type: {"invalid": "cols"}}
            expected_msg = f"{feature_type} must be a list"
            with pytest.raises(TypeError, match=expected_msg):
                gdf_to_pyg(sample_nodes_gdf, **kwargs)
        else:
            kwargs = {feature_type: ["invalid"]}
            expected_msg = f"{feature_type} must be a dict"
            with pytest.raises(TypeError, match=expected_msg):
                gdf_to_pyg(sample_hetero_nodes_dict, **kwargs)

    def test_geometry_handling(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test geometry handling edge cases."""
        # Null geometries
        nodes_with_null = sample_nodes_gdf.copy()
        nodes_with_null["geometry"] = None
        data = gdf_to_pyg(nodes_with_null)
        nodes_restored, _ = pyg_to_gdf(data)
        assert isinstance(nodes_restored, gpd.GeoDataFrame)

        # Missing position data
        data.pos = None
        nodes_restored, _ = pyg_to_gdf(data)
        assert isinstance(nodes_restored, gpd.GeoDataFrame)

    def test_node_attribute_preservation(self, sample_pyg_data: Data) -> None:
        """Test that node attributes are preserved in conversions."""
        nx_graph = pyg_to_nx(sample_pyg_data)
        for _node_id, node_data in nx_graph.nodes(data=True):
            assert "_original_index" in node_data
            if hasattr(sample_pyg_data, "pos") and sample_pyg_data.pos is not None:
                assert "pos" in node_data

    def test_heterogeneous_empty_edges(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Test heterogeneous conversion with empty edges."""
        empty_edges_dict = {
            ("building", "connects_to", "road"): gpd.GeoDataFrame(
                columns=["geometry"],
                index=pd.MultiIndex.from_arrays([[], []], names=["building_id", "road_id"]),
            ),
        }
        data = gdf_to_pyg(sample_hetero_nodes_dict, empty_edges_dict)
        edge_type = ("building", "connects_to", "road")
        assert data[edge_type].edge_index.shape == (2, 0)
        assert data[edge_type].edge_attr.shape[0] == 0

    def test_gdf_to_pyg_geographic_crs_handling(
        self,
        sample_nodes_gdf_alt_crs: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test geographic CRS handling in gdf_to_pyg (lines 808-809)."""
        # Use the fixture that provides nodes with geographic CRS (EPSG:4326)
        # Convert edges to same CRS to match
        edges_geographic = sample_edges_gdf.to_crs("EPSG:4326")

        # This should trigger the geographic CRS handling code path
        pyg_data = gdf_to_pyg(sample_nodes_gdf_alt_crs, edges_geographic)

        assert isinstance(pyg_data, Data)
        assert pyg_data.pos is not None
        assert pyg_data.pos.shape[0] == len(sample_nodes_gdf_alt_crs)  # All nodes

    def test_keep_geom_true(
        self,
        sample_geom_data: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
    ) -> None:
        """Test conversion and reconstruction with keep_geom=True."""
        nodes, edges = sample_geom_data

        # Convert with keep_geom=True (default)
        data = gdf_to_pyg(nodes, edges, keep_geom=True)

        # Check if geometries are stored in metadata
        assert data.graph_metadata.node_geometries is not None
        assert data.graph_metadata.edge_geometries is not None

        # Reconstruct with keep_geom=True
        nodes_rec, edges_rec = pyg_to_gdf(data, keep_geom=True)
        assert isinstance(nodes_rec, gpd.GeoDataFrame)
        assert isinstance(edges_rec, gpd.GeoDataFrame)

        # Geometries should be exactly the same (stored WKB)
        assert nodes_rec.geometry.equals(nodes.geometry)
        assert edges_rec.geometry.equals(edges.geometry)

    def test_keep_geom_false_conversion(
        self,
        sample_geom_data: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
    ) -> None:
        """Test conversion with keep_geom=False."""
        nodes, edges = sample_geom_data

        # Convert with keep_geom=False
        data = gdf_to_pyg(nodes, edges, keep_geom=False)

        # Check that geometries are NOT stored in metadata
        assert (
            not hasattr(data.graph_metadata, "node_geometries")
            or data.graph_metadata.node_geometries is None
        )
        assert (
            not hasattr(data.graph_metadata, "edge_geometries")
            or data.graph_metadata.edge_geometries is None
        )

        # Reconstruct (keep_geom=True/False shouldn't matter as they are not stored)
        nodes_rec, edges_rec = pyg_to_gdf(data)
        assert isinstance(nodes_rec, gpd.GeoDataFrame)
        assert isinstance(edges_rec, gpd.GeoDataFrame)

        # Geometries should be reconstructed from pos (centroids)
        # Use geom_equals_exact for floating point comparison with tolerance
        assert nodes_rec.geometry.geom_equals_exact(nodes.geometry, tolerance=1e-6).all()
        assert edges_rec.geometry.geom_equals_exact(edges.geometry, tolerance=1e-6).all()

    def test_keep_geom_false_reconstruction(
        self,
        sample_geom_data: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
    ) -> None:
        """Test reconstruction with keep_geom=False ignoring stored geometries."""
        nodes, edges = sample_geom_data

        # Convert with keep_geom=True (store them)
        gdf_to_pyg(nodes, edges, keep_geom=True)

        # Let's use a curved edge case
        edges_curved_data = {
            "source_id": [1],
            "target_id": [3],
            "geometry": [LineString([(0, 0), (0, 2), (2, 2)])],  # L-shape
            "weight": [1.0],
        }
        edges_curved = gpd.GeoDataFrame(edges_curved_data).set_index(["source_id", "target_id"])
        edges_curved.crs = "EPSG:4326"

        data_curved = gdf_to_pyg(nodes, edges_curved, keep_geom=True)

        # Reconstruct with keep_geom=True -> should preserve L-shape
        _, edges_rec_true = pyg_to_gdf(data_curved, keep_geom=True)
        assert isinstance(edges_rec_true, gpd.GeoDataFrame)
        assert edges_rec_true.geometry.iloc[0].equals(edges_curved.geometry.iloc[0])

        # Reconstruct with keep_geom=False -> should be straight line from (0,0) to (2,2)
        _, edges_rec_false = pyg_to_gdf(data_curved, keep_geom=False)
        assert isinstance(edges_rec_false, gpd.GeoDataFrame)

        expected_straight = LineString([(0, 0), (2, 2)])
        # Use equals_exact for floating point comparison
        assert edges_rec_false.geometry.iloc[0].equals_exact(expected_straight, tolerance=1e-6)
        assert not edges_rec_false.geometry.iloc[0].equals(edges_curved.geometry.iloc[0])

    def test_convert_invalid_inputs(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test invalid input types in convert method."""
        # Nodes GDF but edges not GDF/None
        with pytest.raises(TypeError, match="For homogeneous graphs, edges must be a GeoDataFrame"):
            gdf_to_pyg(sample_nodes_gdf, edges={"invalid": "type"})

        # Nodes dict but edges not dict/None
        with pytest.raises(TypeError, match="For heterogeneous graphs, edges must be a dictionary"):
            gdf_to_pyg({"n": sample_nodes_gdf}, edges=sample_nodes_gdf)

        # Nodes invalid type
        with pytest.raises(TypeError, match="Nodes must be a GeoDataFrame or a dictionary"):
            gdf_to_pyg("invalid_nodes")

    def test_convert_none_nodes(self) -> None:
        """Test None nodes in conversion methods."""
        converter = graph_module.PyGConverter()

        # Homogeneous
        with pytest.raises(ValueError, match="Nodes GeoDataFrame is required"):
            converter._convert_homogeneous(None, None)

        # Heterogeneous
        with pytest.raises(ValueError, match="Nodes dictionary is required"):
            converter._convert_heterogeneous(None, None)

    def test_reconstruct_missing_geometry_data(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test reconstruction when features exist but geometry/pos is missing."""
        # Create data with features but no pos/geometry
        data = gdf_to_pyg(sample_nodes_gdf)
        data.pos = None
        data.graph_metadata.node_geometries = None

        nodes_rec, _ = pyg_to_gdf(data)
        assert isinstance(nodes_rec, gpd.GeoDataFrame)
        # Note: implementation creates empty GeoSeries with None
        assert nodes_rec.geometry.isna().all()

    def test_reconstruct_edge_missing_geometry_data(
        self, sample_edges_gdf: gpd.GeoDataFrame, sample_nodes_gdf: gpd.GeoDataFrame
    ) -> None:
        """Test edge reconstruction when features exist but geometry is missing."""
        # Create data with features but no stored geometries
        data = gdf_to_pyg(sample_nodes_gdf, sample_edges_gdf, keep_geom=False)
        # Remove pos to prevent reconstruction from pos
        data.pos = None

        # This will trigger the fallback to empty geometry in _reconstruct_edge_gdf
        _, edges_rec = pyg_to_gdf(data)
        assert isinstance(edges_rec, gpd.GeoDataFrame)
        assert edges_rec.geometry.isna().all()

    def test_reconstruct_hetero_edge_errors(self, sample_pyg_hetero_data: HeteroData) -> None:
        """Test heterogeneous edge reconstruction errors."""
        converter = graph_module.PyGConverter()
        metadata = sample_pyg_hetero_data.graph_metadata

        with pytest.raises(TypeError, match="Edge type must be a tuple"):
            converter._reconstruct_edge_gdf(sample_pyg_hetero_data, metadata, edge_type="invalid")  # type: ignore[arg-type]

    def test_create_geometry_missing_pos(self, sample_pyg_data: Data) -> None:
        """Test geometry creation with missing pos."""
        converter = graph_module.PyGConverter()
        sample_pyg_data.pos = None
        assert converter._create_geometry_from_positions(sample_pyg_data) is None

    def test_create_edge_geometries_hetero_no_stored(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Test heterogeneous edge geometry creation from pos (keep_geom=False)."""
        # Convert with keep_geom=False
        data = gdf_to_pyg(sample_hetero_nodes_dict, sample_hetero_edges_dict, keep_geom=False)

        # Reconstruct
        _, edges_rec = pyg_to_gdf(data, keep_geom=False)
        assert isinstance(edges_rec, dict)

        # Edges should have geometries (straight lines)
        for gdf in edges_rec.values():
            assert not gdf.geometry.isna().all()
            assert all(isinstance(g, LineString) for g in gdf.geometry)

    def test_create_features_no_numeric(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test feature creation with no numeric columns."""
        # Create GDF with only string columns
        gdf = sample_nodes_gdf.copy()
        gdf["str_col"] = "a"
        gdf = gdf[["str_col", "geometry"]]

        converter = graph_module.PyGConverter()
        features = converter._create_features(gdf, feature_cols=None)
        assert features.shape[1] == 0

    def test_create_node_positions_no_geom(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test node position creation with no geometry."""
        gdf = sample_nodes_gdf.copy()
        # Remove geometry column to trigger the None return
        del gdf["geometry"]

        converter = graph_module.PyGConverter()
        assert converter._create_node_positions(gdf) is None

    def test_serialize_geometries_no_geom(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test geometry serialization with no geometry."""
        gdf = sample_nodes_gdf.copy()
        # Remove geometry column to trigger the None return
        del gdf["geometry"]

        converter = graph_module.PyGConverter()
        assert converter._serialize_geometries(gdf) is None


class TestGraphValidation:
    """Test validation functionality for PyG objects."""

    def test_valid_data(self, sample_pyg_data: Data, sample_pyg_hetero_data: HeteroData) -> None:
        """Test validation of valid PyG objects."""
        # Test homogeneous data
        metadata = validate_pyg(sample_pyg_data)
        assert isinstance(metadata, GraphMetadata)
        assert metadata.is_hetero is False

        # Test heterogeneous data
        metadata_hetero = validate_pyg(sample_pyg_hetero_data)
        assert isinstance(metadata_hetero, GraphMetadata)
        assert metadata_hetero.is_hetero is True

    def test_invalid_inputs(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test validation with invalid inputs."""
        # Invalid input type
        with pytest.raises(TypeError, match="Input must be a PyTorch Geometric"):
            validate_pyg("not_a_pyg_object")

        # Missing metadata
        data = gdf_to_pyg(sample_nodes_gdf)
        delattr(data, "graph_metadata")
        with pytest.raises(ValueError, match="PyG object is missing 'graph_metadata'"):
            validate_pyg(data)

        # Wrong metadata type
        data = gdf_to_pyg(sample_nodes_gdf)
        data.graph_metadata = "wrong_type"
        with pytest.raises(TypeError, match="PyG object has 'graph_metadata' of incorrect type"):
            validate_pyg(data)

    @pytest.mark.parametrize(
        ("graph_type", "inconsistency"),
        [
            ("homo_marked_as_hetero", "is Data but metadata.is_hetero is True"),
            ("hetero_marked_as_homo", "HeteroData but metadata.is_hetero is False"),
        ],
    )
    def test_inconsistent_metadata(
        self,
        graph_type: str,
        inconsistency: str,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Test validation with inconsistent metadata."""
        if graph_type == "homo_marked_as_hetero":
            data = gdf_to_pyg(sample_nodes_gdf)
            data.graph_metadata.is_hetero = True
        else:
            data = gdf_to_pyg(sample_hetero_nodes_dict)
            data.graph_metadata.is_hetero = False

        with pytest.raises(ValueError, match=f"Inconsistency detected.*{inconsistency}"):
            validate_pyg(data)

    def test_tensor_validation_errors(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test tensor validation errors."""
        # Position tensor mismatch
        data = gdf_to_pyg(sample_nodes_gdf)
        data.pos = torch.randn(1, 2)  # Wrong size
        with pytest.raises(ValueError, match="position tensor size.*doesn't match"):
            validate_pyg(data)

        # Label tensor mismatch
        data = gdf_to_pyg(sample_nodes_gdf, node_label_cols=["label1"])
        data.y = torch.randn(1, 1)  # Wrong size
        with pytest.raises(ValueError, match="label tensor size.*doesn't match"):
            validate_pyg(data)

    def test_hetero_label_tensor_mismatch(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Test heterogeneous label tensor size mismatch."""
        data = gdf_to_pyg(
            sample_hetero_nodes_dict,
            node_feature_cols={"building": ["b_feat1"], "road": ["length"]},
            node_label_cols={"building": ["b_label"]},
        )

        # Corrupt the label tensor size for building node type
        data["building"].y = torch.randn(1, 1)  # Wrong size

        with pytest.raises(
            ValueError,
            match="Node type 'building': label tensor size.*doesn't match",
        ):
            validate_pyg(data)

    def test_homo_validation_edge_cases(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test homogeneous graph validation edge cases."""
        # Missing default mapping
        data = gdf_to_pyg(sample_nodes_gdf)
        data.graph_metadata.node_mappings = {"wrong_key": {}}
        with pytest.raises(ValueError, match="should use 'default' key in node_mappings"):
            validate_pyg(data)

        # Dict feature cols instead of list
        data = gdf_to_pyg(sample_nodes_gdf)
        data.graph_metadata.node_feature_cols = {"wrong": ["feature1"]}
        with pytest.raises(ValueError, match="should have node_feature_cols as list, not dict"):
            validate_pyg(data)

        # Dict label cols instead of list
        data = gdf_to_pyg(sample_nodes_gdf, node_label_cols=["label1"])
        data.graph_metadata.node_label_cols = {"wrong": ["label1"]}
        with pytest.raises(ValueError, match="should have node_label_cols as list, not dict"):
            validate_pyg(data)

    def test_homo_edge_validation_cases(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test homogeneous edge validation cases."""
        # Dict edge feature cols instead of list
        data = gdf_to_pyg(sample_nodes_gdf, sample_edges_gdf, edge_feature_cols=["edge_feature1"])
        data.graph_metadata.edge_feature_cols = {"wrong": ["edge_feature1"]}
        with pytest.raises(ValueError, match="should have edge_feature_cols as list, not dict"):
            validate_pyg(data)

        # Edge attribute tensor mismatch
        data = gdf_to_pyg(sample_nodes_gdf, sample_edges_gdf, edge_feature_cols=["edge_feature1"])
        data.edge_attr = torch.randn(2, 1)  # Wrong size
        with pytest.raises(
            ValueError,
            match="Edge attribute tensor size.*doesn't match number of edges",
        ):
            validate_pyg(data)

    def test_heterogeneous_tensor_validation(self, sample_pyg_hetero_data: HeteroData) -> None:
        """Test tensor validation for heterogeneous graphs."""
        node_type = next(iter(sample_pyg_hetero_data.node_types))
        sample_pyg_hetero_data[node_type].pos = torch.randn(1, 2)  # Wrong size
        with pytest.raises(ValueError, match="position tensor size.*doesn't match"):
            validate_pyg(sample_pyg_hetero_data)

    def test_homogeneous_validation_errors(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test validation errors specific to homogeneous graphs."""
        data = gdf_to_pyg(sample_nodes_gdf)

        # Node types specified
        data.graph_metadata.node_types = ["some_type"]
        with pytest.raises(ValueError, match="should not have node_types"):
            validate_pyg(data)

        # Reset and test edge types
        data = gdf_to_pyg(sample_nodes_gdf)
        data.graph_metadata.edge_types = [("some", "edge", "type")]
        with pytest.raises(ValueError, match="should not have edge_types"):
            validate_pyg(data)

    def test_hetero_validation_errors(self, sample_pyg_hetero_data: HeteroData) -> None:
        """Test validation errors specific to heterogeneous graphs."""
        # Node types mismatch
        data = sample_pyg_hetero_data
        original_node_types = data.graph_metadata.node_types
        data.graph_metadata.node_types = ["wrong_type"]
        with pytest.raises(ValueError, match="Node types mismatch"):
            validate_pyg(data)

        # Reset and test edge types mismatch
        data.graph_metadata.node_types = original_node_types
        data.graph_metadata.edge_types = [("wrong", "type", "tuple")]
        with pytest.raises(ValueError, match="Edge types mismatch"):
            validate_pyg(data)


class TestDeviceHandling:
    """Test device and dtype handling."""

    def test_device_validation_errors(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test device validation with various invalid inputs."""
        # Test TypeError cases - invalid types
        invalid_type_inputs = [123, ["cuda"]]
        for device_input in invalid_type_inputs:
            with pytest.raises(TypeError, match="Device must be"):
                gdf_to_pyg(sample_nodes_gdf, device=cast("Any", device_input))

        # Test ValueError cases - invalid string values
        invalid_string_inputs = ["invalid_device", "gpu"]
        for device_input in invalid_string_inputs:
            with pytest.raises(ValueError, match="Device must be"):
                gdf_to_pyg(sample_nodes_gdf, device=device_input)

    def test_cuda_not_available(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test CUDA not available error when CUDA is requested but not available."""
        if not torch.cuda.is_available():
            with pytest.raises(ValueError, match="CUDA selected, but not available"):
                gdf_to_pyg(sample_nodes_gdf, device="cuda")


class TestGraphFeatures:
    """Test feature and label extraction logic."""

    def test_nx_default_feature_naming(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test NetworkX conversion with default feature naming."""
        data = gdf_to_pyg(sample_nodes_gdf, node_feature_cols=["feature1"])

        # Clear the feature column names to trigger default naming
        data.graph_metadata.node_feature_cols = None

        nx_graph = pyg_to_nx(data)

        # Check that nodes have attributes (default naming may or may not include feat_ prefix)
        for _node_id, node_data in nx_graph.nodes(data=True):
            assert isinstance(node_data, dict)
            assert len(node_data) > 0  # Should have some attributes
            break

    def test_homogeneous_feature_extraction_branches(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test homogeneous feature extraction branches - covers lines 1512-1513, 1523-1524."""
        # Create data with features and labels
        data = gdf_to_pyg(
            sample_nodes_gdf,
            node_feature_cols=["feature1"],
            node_label_cols=["label1"],
        )

        # Test the homogeneous branch in _extract_node_features_and_labels
        # by calling pyg_to_gdf which uses this function
        nodes_restored, _ = pyg_to_gdf(data)

        # Verify features and labels are extracted correctly
        assert isinstance(nodes_restored, gpd.GeoDataFrame)
        assert "feature1" in nodes_restored.columns
        assert "label1" in nodes_restored.columns

    def test_hetero_nx_default_naming(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Test heterogeneous NetworkX conversion with default naming - covers lines 1841, 1855."""
        # Create heterogeneous data with features and labels
        data = gdf_to_pyg(
            sample_hetero_nodes_dict,
            node_feature_cols={"building": ["b_feat1"], "road": ["length"]},
            node_label_cols={"building": ["b_label"], "road": ["r_label"]},
        )

        # Clear feature/label column names for one node type to trigger default naming
        data.graph_metadata.node_feature_cols = {"building": None, "road": ["length"]}
        data.graph_metadata.node_label_cols = {"building": None, "road": ["r_label"]}

        # Convert to NetworkX - this should trigger the default naming branches
        nx_graph = pyg_to_nx(data)

        # Check that nodes have attributes with default names
        building_nodes = [
            n for n, d in nx_graph.nodes(data=True) if d.get("node_type") == "building"
        ]
        if building_nodes:
            node_data = nx_graph.nodes[building_nodes[0]]
            # Should have default feature names like feat_0, label_0
            has_default_feat = any(key.startswith("feat_") for key in node_data)
            has_default_label = any(key.startswith("label_") for key in node_data)
            # At least one should be true if we have features/labels
            non_meta_keys = [
                k for k in node_data if not k.startswith("_") and k not in {"node_type", "pos"}
            ]
            assert has_default_feat or has_default_label or len(non_meta_keys) == 0

    def test_feature_extraction_minimal_data(self) -> None:
        """Test feature extraction with minimal node data."""
        minimal_nodes = gpd.GeoDataFrame(
            {
                "geometry": [Point(0, 0), Point(1, 1)],
            },
            index=[100, 200],
        )
        data = gdf_to_pyg(minimal_nodes)
        nodes_back, _ = pyg_to_gdf(data)
        assert isinstance(nodes_back, gpd.GeoDataFrame)
        assert list(nodes_back.index) == [100, 200]

    def test_missing_features_handling(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test handling of missing node features."""
        nodes_no_features = sample_nodes_gdf[["geometry"]]
        data = gdf_to_pyg(nodes_no_features, node_feature_cols=["nonexistent"])
        assert data.x.shape[1] == 0

    def test_nx_feature_naming(self, sample_pyg_data: Data) -> None:
        """Test NetworkX conversion with default feature naming."""
        data = sample_pyg_data
        data.graph_metadata.node_feature_cols = None
        data.graph_metadata.node_label_cols = None
        nx_graph = pyg_to_nx(data)
        for _node_id, node_data in nx_graph.nodes(data=True):
            if hasattr(data, "x") and data.x is not None and data.x.numel() > 0:
                assert any(key.startswith("feat_") for key in node_data)

    def test_hetero_feature_extraction(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Test heterogeneous feature extraction."""
        data = gdf_to_pyg(
            sample_hetero_nodes_dict,
            node_feature_cols={"building": ["b_feat1"], "road": ["r_feat1"]},
            node_label_cols={"building": ["b_label"], "road": ["r_label"]},
        )
        nodes_restored, _ = pyg_to_gdf(data)
        assert isinstance(nodes_restored, dict)
        assert isinstance(nodes_restored["building"], gpd.GeoDataFrame)
        assert isinstance(nodes_restored["road"], gpd.GeoDataFrame)
        assert "b_feat1" in nodes_restored["building"].columns
        # Note: r_feat1 doesn't exist in the fixture, so it won't be restored
        assert "r_label" in nodes_restored["road"].columns

    def test_edge_features_none_handling(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test edge feature extraction with None edge attributes."""
        data = gdf_to_pyg(sample_nodes_gdf, sample_edges_gdf)
        data.edge_attr = None
        _, edges_restored = pyg_to_gdf(data)
        assert isinstance(edges_restored, gpd.GeoDataFrame)
        assert len(edges_restored) == len(sample_edges_gdf)


METAPATH = [("building", "connects_to", "road"), ("road", "links_to", "road")]
RESULT_KEY = ("building", "metapath_0", "road")


@pytest.fixture
def sample_weight_graph_data() -> tuple[
    dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]
]:
    """Small heterogeneous graph tailored for add_metapaths_by_weight tests."""
    buildings = gpd.GeoDataFrame(
        {
            "geometry": [Point(0, 0), Point(10, 0), Point(20, 0)],
            "node_type": "building",
        },
        index=[1, 2, 3],
        crs="EPSG:4326",
    )

    streets = gpd.GeoDataFrame(
        {
            "geometry": [Point(0, 1), Point(10, 1), Point(20, 1)],
            "node_type": "street",
        },
        index=[101, 102, 103],
        crs="EPSG:4326",
    )

    nodes_dict = {"building": buildings, "street": streets}

    b_s_edges = gpd.GeoDataFrame(
        {
            "weight": [1.0, 1.0, 1.0],
            "edge_type": "access",
            "geometry": [
                LineString([(0, 0), (0, 1)]),
                LineString([(10, 0), (10, 1)]),
                LineString([(20, 0), (20, 1)]),
            ],
        },
        index=pd.MultiIndex.from_tuples([(1, 101), (2, 102), (3, 103)]),
        crs="EPSG:4326",
    )

    s_b_edges = gpd.GeoDataFrame(
        {
            "weight": [1.0, 1.0, 1.0],
            "edge_type": "access",
            "geometry": [
                LineString([(0, 1), (0, 0)]),
                LineString([(10, 1), (10, 0)]),
                LineString([(20, 1), (20, 0)]),
            ],
        },
        index=pd.MultiIndex.from_tuples([(101, 1), (102, 2), (103, 3)]),
        crs="EPSG:4326",
    )

    s_s_edges = gpd.GeoDataFrame(
        {
            "weight": [10.0, 10.0],
            "edge_type": "road",
            "geometry": [
                LineString([(0, 1), (10, 1)]),
                LineString([(10, 1), (20, 1)]),
            ],
        },
        index=pd.MultiIndex.from_tuples([(101, 102), (102, 103)]),
        crs="EPSG:4326",
    )

    s_s_edges_rev = gpd.GeoDataFrame(
        {
            "weight": [10.0, 10.0],
            "edge_type": "road",
            "geometry": [
                LineString([(10, 1), (0, 1)]),
                LineString([(20, 1), (10, 1)]),
            ],
        },
        index=pd.MultiIndex.from_tuples([(102, 101), (103, 102)]),
        crs="EPSG:4326",
    )

    edges_dict = {
        ("building", "access", "street"): b_s_edges,
        ("street", "access", "building"): s_b_edges,
        ("street", "road", "street"): s_s_edges,
        ("street", "road_rev", "street"): s_s_edges_rev,
    }

    return nodes_dict, edges_dict


WeightGraphData = tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]]


class TestMetapaths:
    """Test metapath addition functionality."""

    @pytest.mark.parametrize("agg_mode", ["sum", "mean", "callable", "callable_all_nan"])
    def test_add_metapaths_basic_aggregations(
        self,
        agg_mode: str,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Exercise aggregation choices: sum, mean, callable, and all-NaN callable path."""
        travel_time_values = {
            ("building", "connects_to", "road"): [10.0, 20.0, 30.0],
            ("road", "links_to", "road"): [5.0, 15.0],
        }
        edges_with_attr = {k: v.copy() for k, v in sample_hetero_edges_dict.items()}
        for ek, vals in travel_time_values.items():
            if ek in edges_with_attr:
                edges_with_attr[ek]["travel_time"] = vals

        if agg_mode == "sum":
            nodes_out, edges_out = graph_module.add_metapaths(
                (sample_hetero_nodes_dict, edges_with_attr),
                sequence=METAPATH,
                edge_attr="travel_time",
                edge_attr_agg="sum",
            )
            assert nodes_out is sample_hetero_nodes_dict
            result = edges_out[RESULT_KEY]
            assert "weight" in result.columns
            assert pd.api.types.is_integer_dtype(result["weight"].dtype)
            assert "travel_time" in result.columns
            assert "geometry" in result.columns
        elif agg_mode == "mean":
            _, edges_out = graph_module.add_metapaths(
                (sample_hetero_nodes_dict, edges_with_attr),
                sequence=METAPATH,
                edge_attr="travel_time",
                edge_attr_agg="mean",
            )
            assert (edges_out[RESULT_KEY]["travel_time"] > 0).all()
        elif agg_mode == "callable":
            _, edges_out = graph_module.add_metapaths(
                (sample_hetero_nodes_dict, edges_with_attr),
                sequence=METAPATH,
                edge_attr="travel_time",
                edge_attr_agg=np.nanmax,
            )
            assert all(
                isinstance(v, float) for v in edges_out[RESULT_KEY]["travel_time"].to_numpy()
            )
        else:
            # callable_all_nan
            edges_nan_attr = {k: v.copy() for k, v in sample_hetero_edges_dict.items()}
            for ek in list(edges_nan_attr):
                edges_nan_attr[ek]["travel_time"] = np.nan
            _, edges_out = graph_module.add_metapaths(
                (sample_hetero_nodes_dict, edges_nan_attr),
                sequence=METAPATH,
                edge_attr="travel_time",
                edge_attr_agg=np.nanmax,
            )
            result = edges_out[RESULT_KEY]
            if not result.empty:
                assert result["travel_time"].isna().all()

    def test_add_metapaths_edge_attr_list(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Passing ``edge_attr`` as a list should aggregate each requested column."""
        edges_with_attr = {k: v.copy() for k, v in sample_hetero_edges_dict.items()}
        for gdf in edges_with_attr.values():
            gdf["travel_time"] = 1.0

        _, edges_out = graph_module.add_metapaths(
            (sample_hetero_nodes_dict, edges_with_attr),
            sequence=METAPATH,
            edge_attr=["travel_time"],
            edge_attr_agg="sum",
        )

        result = edges_out[RESULT_KEY]
        assert "travel_time" in result.columns
        if not result.empty:
            assert (result["travel_time"] > 0).all()

    @pytest.mark.parametrize("mode", ["as_nx", "networkx_input", "metadata_merge"])
    def test_add_metapaths_return_formats(
        self,
        mode: str,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cover return as NetworkX, accepting NX input, and metadata merge behavior."""
        edges_with_attr = {k: v.copy() for k, v in sample_hetero_edges_dict.items()}
        for ek in list(edges_with_attr):
            edges_with_attr[ek]["travel_time"] = 1.0

        if mode == "as_nx":
            g = graph_module.add_metapaths(
                (sample_hetero_nodes_dict, edges_with_attr),
                sequence=METAPATH,
                edge_attr="travel_time",
                edge_attr_agg="sum",
                trace_path=True,
                as_nx=True,
                multigraph=True,
            )
            assert isinstance(g, nx.MultiGraph)
            assert RESULT_KEY in g.graph.get("metapath_dict", {})
        elif mode == "networkx_input":
            hetero_graph = gdf_to_nx(
                nodes=sample_hetero_nodes_dict, edges=sample_hetero_edges_dict, multigraph=True
            )
            _, edges_out = graph_module.add_metapaths(hetero_graph, sequence=METAPATH)
            assert RESULT_KEY in edges_out
        else:

            def fake_gdf_to_nx(*_a: object, **_k: object) -> nx.Graph:
                g = nx.MultiGraph()
                g.graph["metapath_dict"] = {"legacy": {"tag": 1}}
                return g

            monkeypatch.setattr(graph_module, "gdf_to_nx", fake_gdf_to_nx)
            g = graph_module.add_metapaths(
                (sample_hetero_nodes_dict, sample_hetero_edges_dict),
                sequence=METAPATH,
                as_nx=True,
                multigraph=True,
            )
            assert isinstance(g, nx.MultiGraph)
            assert "legacy" in g.graph["metapath_dict"]

    def test_add_metapaths_index_name_fallback(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Unnamed hop indices should fall back to sensible identifiers."""
        edges_no_names = {k: v.copy() for k, v in sample_hetero_edges_dict.items()}
        for gdf in edges_no_names.values():
            gdf.index = gdf.index.set_names([None, None])

        _, edges_out = graph_module.add_metapaths(
            (sample_hetero_nodes_dict, edges_no_names),
            sequence=METAPATH,
        )

        result = edges_out[RESULT_KEY]
        assert result.index.names == ["building_id", "road_id"]

    def test_add_metapaths_nx_edges_none(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When ``nx_to_gdf`` returns ``None`` for edges, it should normalise to {}."""
        hetero_graph = gdf_to_nx(
            nodes=sample_hetero_nodes_dict,
            edges=sample_hetero_edges_dict,
            multigraph=True,
        )

        monkeypatch.setattr(
            graph_module,
            "nx_to_gdf",
            lambda _g: (sample_hetero_nodes_dict, None),
        )

        nodes_out, edges_out = graph_module.add_metapaths(hetero_graph, sequence=METAPATH)
        assert nodes_out is sample_hetero_nodes_dict
        assert edges_out == {}

    def test_add_metapaths_nx_edges_wrong_type(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Non-dict edge data from ``nx_to_gdf`` should raise ``TypeError``."""
        hetero_graph = gdf_to_nx(
            nodes=sample_hetero_nodes_dict,
            edges=sample_hetero_edges_dict,
            multigraph=True,
        )

        monkeypatch.setattr(
            graph_module,
            "nx_to_gdf",
            lambda _g: (sample_hetero_nodes_dict, [1, 2, 3]),
        )

        with pytest.raises(TypeError, match="typed edges"):
            graph_module.add_metapaths(hetero_graph, sequence=METAPATH)

    @pytest.mark.parametrize("early", ["empty_metapaths", "empty_edges", "raw_edges_none"])
    def test_add_metapaths_input_normalization(
        self,
        early: str,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Validate early-return normalization paths for empty inputs and None edges."""
        if early == "empty_metapaths":
            nodes_out, edges_out = graph_module.add_metapaths(
                (sample_hetero_nodes_dict, sample_hetero_edges_dict), sequence=[]
            )
            assert nodes_out is sample_hetero_nodes_dict
            assert edges_out is sample_hetero_edges_dict
        elif early == "empty_edges":
            nodes_out, edges_out = graph_module.add_metapaths(
                (sample_hetero_nodes_dict, {}), sequence=METAPATH
            )
            assert nodes_out is sample_hetero_nodes_dict
            assert edges_out == {}
        else:
            nodes_out, edges_out = graph_module.add_metapaths(
                (sample_hetero_nodes_dict, None), sequence=METAPATH
            )
            assert nodes_out is sample_hetero_nodes_dict
            assert edges_out == {}

    @pytest.mark.parametrize(
        "direction_case", ["directed_true", "reverse_lookup", "edge_type_missing_directed"]
    )
    def test_add_metapaths_edge_direction_and_lookup(
        self,
        direction_case: str,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Check directed lookup, reverse fallback, and directed missing-edge error."""
        if direction_case == "directed_true":
            _, edges_out = graph_module.add_metapaths(
                (sample_hetero_nodes_dict, sample_hetero_edges_dict),
                sequence=METAPATH,
                directed=True,
            )
            assert RESULT_KEY in edges_out
        elif direction_case == "reverse_lookup":
            mp_rev = [("road", "connects_to", "building"), ("building", "connects_to", "road")]
            _, edges_rev = graph_module.add_metapaths(
                (sample_hetero_nodes_dict, sample_hetero_edges_dict),
                sequence=mp_rev,
                directed=False,
            )
            assert ("road", "metapath_0", "road") in edges_rev
        else:
            bad_mp = [("x", "y", "z"), ("z", "y", "x")]
            with pytest.raises(KeyError, match="Edge type .* not found"):
                graph_module.add_metapaths(
                    (sample_hetero_nodes_dict, sample_hetero_edges_dict),
                    sequence=bad_mp,
                    directed=True,
                )

    @pytest.mark.parametrize(
        "join_case", ["empty_hop", "disjoint", "nan_sources", "index_normalization"]
    )
    def test_add_metapaths_join_and_index_cases(
        self,
        join_case: str,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Cover empty hop, disjoint joins, NaN sources, and index name normalization."""
        if join_case == "empty_hop":
            edges_empty = {k: v.copy() for k, v in sample_hetero_edges_dict.items()}
            edges_empty[("road", "links_to", "road")] = (
                edges_empty[("road", "links_to", "road")].head(0).copy()
            )
            nodes = {"road": sample_hetero_nodes_dict["road"]}
            _, edges_out = graph_module.add_metapaths((nodes, edges_empty), sequence=METAPATH)
            res = edges_out[RESULT_KEY]
            assert res.empty
            assert res.crs == sample_hetero_nodes_dict["road"].crs
        elif join_case == "disjoint":
            edges_disjoint = {k: v.copy() for k, v in sample_hetero_edges_dict.items()}
            rl = edges_disjoint[("road", "links_to", "road")].copy()
            rl.index = pd.MultiIndex.from_tuples(
                [("r10", "r11"), ("r11", "r10")], names=rl.index.names
            )
            edges_disjoint[("road", "links_to", "road")] = rl
            _, edges_out = graph_module.add_metapaths(
                (sample_hetero_nodes_dict, edges_disjoint), sequence=METAPATH
            )
            assert edges_out[RESULT_KEY].empty
        elif join_case == "nan_sources":
            edges_nan = {k: v.copy() for k, v in sample_hetero_edges_dict.items()}
            con = edges_nan[("building", "connects_to", "road")].copy()
            con.index = pd.MultiIndex.from_arrays(
                [np.array([np.nan, np.nan, np.nan]), con.index.get_level_values(1)],
                names=con.index.names,
            )
            edges_nan[("building", "connects_to", "road")] = con
            _, edges_out = graph_module.add_metapaths(
                (sample_hetero_nodes_dict, edges_nan), sequence=METAPATH
            )
            assert edges_out[RESULT_KEY].empty
        else:
            edges_named = {k: v.copy() for k, v in sample_hetero_edges_dict.items()}
            e0 = edges_named[("building", "connects_to", "road")]
            e1 = edges_named[("road", "links_to", "road")]
            e0.index = pd.MultiIndex.from_tuples(list(e0.index), names=[1, 2])
            e1.index = pd.MultiIndex.from_tuples(list(e1.index), names=[3, 4])
            edges_named[("building", "connects_to", "road")] = e0
            edges_named[("road", "links_to", "road")] = e1
            _, edges_norm = graph_module.add_metapaths(
                (sample_hetero_nodes_dict, edges_named), sequence=METAPATH
            )
            res_norm = edges_norm[RESULT_KEY]
            assert all(isinstance(n, str) for n in res_norm.index.names)

    @pytest.mark.parametrize("geom_case", ["geometry_fallback", "safe_linestring_error"])
    def test_add_metapaths_geometry_edge_cases(
        self,
        geom_case: str,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Exercise geometry fallbacks and exception-safe linestring creation."""
        if geom_case == "geometry_fallback":
            nodes_geom = {k: v.copy() for k, v in sample_hetero_nodes_dict.items()}
            nodes_geom["building"].loc["b1", "geometry"] = None
            nodes_geom["building"].loc["b2", "geometry"] = Point()
            nodes_geom["road"].loc["r1", "geometry"] = Polygon(
                [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
            )
            _, edges_out = graph_module.add_metapaths(
                (nodes_geom, sample_hetero_edges_dict), sequence=METAPATH
            )
            result = edges_out[RESULT_KEY].sort_index()
            assert result["geometry"].isna().any() or result.empty
        else:
            bad_nodes = {k: v.copy() for k, v in sample_hetero_nodes_dict.items()}
            b_first = bad_nodes["building"].index[0]
            bad_nodes["building"].loc[b_first, "geometry"] = Polygon(
                [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
            )
            _, edges_badgeom = graph_module.add_metapaths(
                (bad_nodes, sample_hetero_edges_dict), sequence=METAPATH
            )
            res_badgeom = edges_badgeom[RESULT_KEY]
            assert res_badgeom["geometry"].isna().any() or res_badgeom.empty

    # --- Error cases ---
    def test_add_metapaths_tuple_length_error(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Tuple with invalid length should trigger ValueError in normalization."""
        with pytest.raises(ValueError, match="Graph tuple must contain"):
            graph_module.add_metapaths(
                (sample_hetero_nodes_dict, sample_hetero_edges_dict, {}), sequence=METAPATH
            )

    def test_add_metapaths_nodes_dict_type_error(
        self,
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Non-dict nodes argument should raise TypeError."""
        with pytest.raises(TypeError, match="nodes_dict must be a dictionary"):
            graph_module.add_metapaths(([1, 2, 3], sample_hetero_edges_dict), sequence=METAPATH)

    def test_add_metapaths_edges_dict_type_error(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Non-dict edges argument should raise TypeError."""
        with pytest.raises(TypeError, match="edges_dict must be a dictionary"):
            graph_module.add_metapaths((sample_hetero_nodes_dict, [1, 2, 3]), sequence=METAPATH)

    def test_add_metapaths_nx_nodes_not_dict(self) -> None:
        """Homogeneous NetworkX graph should fail due to missing typed nodes."""
        g = nx.Graph()
        g.add_node(1, pos=(0.0, 0.0))
        g.add_node(2, pos=(1.0, 1.0))
        g.add_edge(1, 2)
        g.graph["is_hetero"] = False
        g.graph["crs"] = "EPSG:4326"
        with pytest.raises(TypeError, match="requires a heterogeneous graph with typed nodes"):
            graph_module.add_metapaths(g, sequence=METAPATH)

    def test_add_metapaths_unsupported_input_type(self) -> None:
        """Unsupported input type should raise TypeError in _ensure_hetero_dict."""
        with pytest.raises(TypeError, match="Unsupported graph input type"):
            graph_module.add_metapaths(12345, sequence=METAPATH)

    def test_add_metapaths_neither_graph_nor_nodes(self) -> None:
        """Test add_metapaths raises error when neither graph nor nodes provided."""
        with pytest.raises(ValueError, match="Either 'graph' or 'nodes'"):
            graph_module.add_metapaths(sequence=METAPATH)

    def test_add_metapaths_sequence_none(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Test add_metapaths raises ValueError when sequence is None."""
        with pytest.raises(ValueError, match="sequence must be provided"):
            graph_module.add_metapaths(
                (sample_hetero_nodes_dict, sample_hetero_edges_dict), sequence=None
            )

    def test_add_metapaths_short_metapath_error(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Metapath shorter than two hops should raise ValueError."""
        with pytest.raises(ValueError, match="at least two edge types"):
            graph_module.add_metapaths(
                (sample_hetero_nodes_dict, sample_hetero_edges_dict), sequence=[("a", "b", "c")]
            )

    def test_add_metapaths_invalid_index_error(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Edge frame without MultiIndex must raise ValueError in materialization."""
        edges_bad_index = {k: v.copy() for k, v in sample_hetero_edges_dict.items()}

        e0 = edges_bad_index[("building", "connects_to", "road")].reset_index(drop=True)
        edges_bad_index[("building", "connects_to", "road")] = e0
        with pytest.raises(ValueError, match="must have a two-level MultiIndex"):
            graph_module.add_metapaths(
                (sample_hetero_nodes_dict, edges_bad_index), sequence=METAPATH
            )

    def test_add_metapaths_by_weight_basic(self, sample_weight_graph_data: WeightGraphData) -> None:
        """Connect buildings within a threshold and materialize geometry."""
        nodes_dict, edges_dict = sample_weight_graph_data
        nodes_out, edges_out = graph_module.add_metapaths_by_weight(
            (nodes_dict, edges_dict),
            endpoint_type="building",
            weight="weight",
            threshold=15.0,
            directed=True,
        )

        relation = ("building", "connected_within_0.0_15.0", "building")
        assert nodes_out is nodes_dict
        assert relation in edges_out

        new_edges = edges_out[relation]
        pairs = set(new_edges.index.tolist())
        assert pairs == {(1, 2), (2, 1), (2, 3), (3, 2)}
        assert new_edges.loc[(1, 2), "weight"] == pytest.approx(12.0)
        geom = new_edges.loc[(1, 2), "geometry"]
        assert isinstance(geom, LineString)
        assert list(geom.coords) == [(0.0, 0.0), (10.0, 0.0)]

    def test_add_metapaths_by_weight_threshold_controls(
        self,
        sample_weight_graph_data: WeightGraphData,
    ) -> None:
        """Threshold and min_threshold bounds should gate which pairs are emitted."""
        nodes_dict, edges_dict = sample_weight_graph_data

        _, edges_threshold = graph_module.add_metapaths_by_weight(
            (nodes_dict, edges_dict),
            endpoint_type="building",
            weight="weight",
            threshold=12.0,
            directed=True,
        )
        rel_max = ("building", "connected_within_0.0_12.0", "building")
        assert rel_max in edges_threshold
        assert (1, 2) in set(edges_threshold[rel_max].index.tolist())

        _, edges_filtered = graph_module.add_metapaths_by_weight(
            (nodes_dict, edges_dict),
            endpoint_type="building",
            weight="weight",
            threshold=20.0,
            min_threshold=13.0,
            directed=True,
        )
        rel_min = ("building", "connected_within_13.0_20.0", "building")
        if rel_min in edges_filtered:
            assert edges_filtered[rel_min].empty

    def test_add_metapaths_by_weight_edge_types_and_custom_name(
        self,
        sample_weight_graph_data: WeightGraphData,
    ) -> None:
        """Ensure edge filters and friendly relation labels behave as expected."""
        nodes_dict, edges_dict = sample_weight_graph_data

        _, edges_custom = graph_module.add_metapaths_by_weight(
            (nodes_dict, edges_dict),
            endpoint_type="building",
            weight="weight",
            threshold=15.0,
            new_relation_name="accessible",
        )
        custom_rel = ("building", "accessible", "building")
        assert custom_rel in edges_custom

        _, edges_filtered = graph_module.add_metapaths_by_weight(
            (nodes_dict, edges_dict),
            endpoint_type="building",
            weight="weight",
            threshold=100.0,
            edge_types=[
                ("building", "access", "street"),
                ("street", "access", "building"),
            ],
        )
        filtered_rel = ("building", "connected_within_0.0_100.0", "building")
        assert filtered_rel not in edges_filtered

    def test_add_metapaths_by_weight_networkx_io(
        self, sample_weight_graph_data: WeightGraphData
    ) -> None:
        """Cover NetworkX round-trips both as input and output."""
        nodes_dict, edges_dict = sample_weight_graph_data

        nx_result = graph_module.add_metapaths_by_weight(
            (nodes_dict, edges_dict),
            endpoint_type="building",
            weight="weight",
            threshold=15.0,
            as_nx=True,
        )
        assert isinstance(nx_result, nx.Graph)
        assert any(
            data.get("edge_type") == ("building", "connected_within_0.0_15.0", "building")
            for _, _, data in nx_result.edges(data=True)
        )

        hetero_graph = gdf_to_nx(nodes=nodes_dict, edges=edges_dict)
        nx_roundtrip = graph_module.add_metapaths_by_weight(
            hetero_graph,
            endpoint_type="building",
            weight="weight",
            threshold=15.0,
            as_nx=True,
        )
        assert isinstance(nx_roundtrip, nx.Graph)
        assert any(
            data.get("edge_type") == ("building", "connected_within_0.0_15.0", "building")
            for _, _, data in nx_roundtrip.edges(data=True)
        )

    def test_add_metapaths_by_weight_missing_endpoint(
        self,
        sample_weight_graph_data: WeightGraphData,
    ) -> None:
        """Unknown endpoint types should short-circuit and return originals."""
        nodes_dict, edges_dict = sample_weight_graph_data

        nodes_out, edges_out = graph_module.add_metapaths_by_weight(
            (nodes_dict, edges_dict),
            endpoint_type="nonexistent",
            weight="weight",
            threshold=15.0,
        )

        assert nodes_out is nodes_dict
        assert len(edges_out) == len(edges_dict)

    def test_add_metapaths_missing_edge_attr_error(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Missing edge attribute at hop level should raise KeyError."""
        with pytest.raises(KeyError, match=r"Edge attribute\(s\)"):
            graph_module.add_metapaths(
                (sample_hetero_nodes_dict, sample_hetero_edges_dict),
                sequence=METAPATH,
                edge_attr="missing_attr",
            )

    def test_add_metapaths_missing_join_attr_error(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Edge attribute missing in some steps should raise KeyError in join reduction."""
        edges_partial_attr = {k: v.copy() for k, v in sample_hetero_edges_dict.items()}
        edges_partial_attr[("building", "connects_to", "road")]["travel_time"] = [1.0] * len(
            edges_partial_attr[("building", "connects_to", "road")]
        )
        with pytest.raises(KeyError, match="missing in metapath steps"):
            graph_module.add_metapaths(
                (sample_hetero_nodes_dict, edges_partial_attr),
                sequence=METAPATH,
                edge_attr="travel_time",
            )

    def test_add_metapaths_invalid_edge_attr_agg_string(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Unsupported string for edge_attr_agg must raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported edge_attr_agg"):
            graph_module.add_metapaths(
                (sample_hetero_nodes_dict, sample_hetero_edges_dict),
                sequence=METAPATH,
                edge_attr_agg="median",
            )

    def test_add_metapaths_invalid_edge_attr_agg_type(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Non-string, non-callable edge_attr_agg must raise TypeError."""
        with pytest.raises(TypeError, match="edge_attr_agg must be"):
            graph_module.add_metapaths(
                (sample_hetero_nodes_dict, sample_hetero_edges_dict),
                sequence=METAPATH,
                edge_attr_agg=123,
            )

    def test_add_metapaths_attach_geometry_missing_nodes(
        self,
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Missing node GeoDataFrame for start or end types should raise KeyError."""
        nodes_missing = {k: v for k, v in sample_hetero_nodes_dict.items() if k != "road"}
        with pytest.raises(KeyError, match="Missing node GeoDataFrame"):
            graph_module.add_metapaths((nodes_missing, sample_hetero_edges_dict), sequence=METAPATH)

    def test_add_metapaths_by_weight_value_errors(
        self, sample_weight_graph_data: WeightGraphData
    ) -> None:
        """Test ValueError cases for add_metapaths_by_weight."""
        nodes_dict, edges_dict = sample_weight_graph_data

        # Missing weight parameter
        with pytest.raises(ValueError, match="weight must be provided"):
            graph_module.add_metapaths_by_weight(
                (nodes_dict, edges_dict),
                endpoint_type="building",
                threshold=15.0,
            )

        # Missing threshold parameter
        with pytest.raises(ValueError, match="threshold must be provided"):
            graph_module.add_metapaths_by_weight(
                (nodes_dict, edges_dict),
                endpoint_type="building",
                weight="weight",
            )

        # Both graph and nodes provided
        with pytest.raises(ValueError, match="Cannot provide both"):
            graph_module.add_metapaths_by_weight(
                (nodes_dict, edges_dict),
                nodes=nodes_dict,
                endpoint_type="building",
                weight="weight",
                threshold=15.0,
            )

        # Neither graph nor nodes provided
        with pytest.raises(ValueError, match="Either 'graph' or 'nodes'"):
            graph_module.add_metapaths_by_weight(
                endpoint_type="building",
                weight="weight",
                threshold=15.0,
            )

    def test_add_metapaths_by_weight_endpoint_type_none(
        self, sample_weight_graph_data: WeightGraphData
    ) -> None:
        """Test that endpoint_type=None returns original graph with warning."""
        nodes_dict, edges_dict = sample_weight_graph_data

        # endpoint_type is None - should return original graph
        nodes_out, edges_out = graph_module.add_metapaths_by_weight(
            (nodes_dict, edges_dict),
            endpoint_type=None,
            weight="weight",
            threshold=15.0,
        )
        assert nodes_out is nodes_dict
        assert edges_out is edges_dict

    def test_add_metapaths_by_weight_empty_data(
        self, sample_weight_graph_data: WeightGraphData
    ) -> None:
        """Test add_metapaths_by_weight with no valid edges (empty data list)."""
        nodes_dict, edges_dict = sample_weight_graph_data

        # Filter out all edges by specifying non-matching edge types
        nodes_out, edges_out = graph_module.add_metapaths_by_weight(
            (nodes_dict, edges_dict),
            endpoint_type="building",
            weight="weight",
            threshold=15.0,
            edge_types=[("nonexistent", "nonexistent", "nonexistent")],
        )
        # Should return original graph since no data for sparse matrix
        assert nodes_out is nodes_dict
        assert edges_out is edges_dict

    def test_add_metapaths_by_weight_multigraph_input(
        self, sample_weight_graph_data: WeightGraphData
    ) -> None:
        """Test add_metapaths_by_weight with MultiGraph input."""
        nodes_dict, edges_dict = sample_weight_graph_data

        # Convert to MultiGraph
        nx_multigraph = gdf_to_nx(nodes=nodes_dict, edges=edges_dict, multigraph=True)

        result = graph_module.add_metapaths_by_weight(
            nx_multigraph,
            endpoint_type="building",
            weight="weight",
            threshold=15.0,
            as_nx=True,
            multigraph=True,
        )
        assert isinstance(result, (nx.Graph, nx.MultiGraph))

    def test_add_metapaths_by_weight_as_nx_from_tuple(
        self, sample_weight_graph_data: WeightGraphData
    ) -> None:
        """Test add_metapaths_by_weight with as_nx=True from tuple input."""
        nodes_dict, edges_dict = sample_weight_graph_data

        result = graph_module.add_metapaths_by_weight(
            (nodes_dict, edges_dict),
            endpoint_type="building",
            weight="weight",
            threshold=15.0,
            as_nx=True,
            multigraph=False,
        )
        assert isinstance(result, nx.Graph)

    def test_add_metapaths_with_graph_and_nodes_error(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Test add_metapaths raises error when both graph and nodes provided."""
        with pytest.raises(ValueError, match="Cannot provide both"):
            graph_module.add_metapaths(
                graph=(sample_hetero_nodes_dict, sample_hetero_edges_dict),
                nodes=sample_hetero_nodes_dict,
                sequence=METAPATH,
            )

    def test_add_metapaths_with_nodes_only(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Test add_metapaths with nodes parameter only (no graph)."""
        _, edges_out = graph_module.add_metapaths(
            nodes=sample_hetero_nodes_dict,
            edges=sample_hetero_edges_dict,
            sequence=METAPATH,
        )
        assert RESULT_KEY in edges_out

    def test_add_metapaths_with_new_relation_name(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Test add_metapaths with custom new_relation_name."""
        _, edges_out = graph_module.add_metapaths(
            (sample_hetero_nodes_dict, sample_hetero_edges_dict),
            sequence=METAPATH,
            new_relation_name="custom_path",
        )
        custom_key = ("building", "custom_path", "road")
        assert custom_key in edges_out

    def test_add_metapaths_trace_path_logging(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Test add_metapaths with trace_path=True triggers debug logging."""
        # trace_path=True should just log and continue without error
        _, edges_out = graph_module.add_metapaths(
            (sample_hetero_nodes_dict, sample_hetero_edges_dict),
            sequence=METAPATH,
            trace_path=True,
        )
        assert RESULT_KEY in edges_out

    def test_add_metapaths_by_weight_nodes_param(
        self, sample_weight_graph_data: WeightGraphData
    ) -> None:
        """Test add_metapaths_by_weight using nodes/edges params instead of graph."""
        nodes_dict, edges_dict = sample_weight_graph_data

        # Use nodes parameter instead of graph
        nodes_out, edges_out = graph_module.add_metapaths_by_weight(
            nodes=nodes_dict,
            edges=edges_dict,
            endpoint_type="building",
            weight="weight",
            threshold=15.0,
        )

        relation = ("building", "connected_within_0.0_15.0", "building")
        assert nodes_out is nodes_dict
        assert relation in edges_out

    def test_add_metapaths_by_weight_edge_type_filter_nx(
        self, sample_weight_graph_data: WeightGraphData
    ) -> None:
        """Test add_metapaths_by_weight edge_types filter with NetworkX input."""
        nodes_dict, edges_dict = sample_weight_graph_data

        # Convert to NetworkX (non-multigraph to hit the else branch in line 2716)
        nx_graph = gdf_to_nx(nodes=nodes_dict, edges=edges_dict, multigraph=False)

        # Filter to only use specific edge types
        result = graph_module.add_metapaths_by_weight(
            nx_graph,
            endpoint_type="building",
            weight="weight",
            threshold=15.0,
            edge_types=[("building", "access", "street"), ("street", "access", "building")],
            as_nx=True,
        )
        assert isinstance(result, nx.Graph)


class TestGraphGeometry:
    """Test geometry handling in graph conversions."""

    def test_hetero_geometry_dict_missing_type(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Test _get_stored_geometries when type_key not in geometry dict."""
        # Create hetero data with keep_geom=True
        data = gdf_to_pyg(sample_hetero_nodes_dict, sample_hetero_edges_dict, keep_geom=True)

        # Manually modify node_geometries to have only one type
        # This tests line 1538 where type_key is not found in the dict
        if data.graph_metadata.node_geometries is not None:
            # Keep only 'building' geometry, remove 'road'
            data.graph_metadata.node_geometries = {
                "building": data.graph_metadata.node_geometries.get("building", [])
            }

        # Reconstruct should handle missing 'road' geometry
        nodes_rec, _ = pyg_to_gdf(data)
        assert isinstance(nodes_rec, dict)
        assert "road" in nodes_rec  # Should still have road nodes but geometry from pos
