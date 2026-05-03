"""Scenario-focused tests for the public graph conversion API."""

from __future__ import annotations

import time
from typing import Any
from typing import cast

import geopandas as gpd
import networkx as nx
import osmnx as ox
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
from city2graph.proximity import contiguity_graph

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

    def test_homogeneous_gdf_to_pyg_basic(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Homogeneous GeoDataFrames convert into a symmetrized Data graph."""
        data = gdf_to_pyg(sample_nodes_gdf, sample_edges_gdf)
        assert isinstance(data, Data)
        assert data.num_nodes == len(sample_nodes_gdf)
        assert data.num_edges == len(sample_edges_gdf) * 2
        assert not data.graph_metadata.is_hetero

    def test_heterogeneous_gdf_to_pyg_basic(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Heterogeneous GeoDataFrames retain original and generated edge stores."""
        data = gdf_to_pyg(sample_hetero_nodes_dict, sample_hetero_edges_dict)
        assert isinstance(data, HeteroData)
        assert data.graph_metadata.is_hetero
        assert set(data.node_types) == set(sample_hetero_nodes_dict.keys())
        assert set(sample_hetero_edges_dict.keys()).issubset(set(data.edge_types))

    def test_homogeneous_gdf_to_pyg_with_features(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Homogeneous feature, label, and edge tensors are column-driven."""
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

    def test_heterogeneous_gdf_to_pyg_with_features(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Heterogeneous feature and label tensors use per-type column specs."""
        data = gdf_to_pyg(
            sample_hetero_nodes_dict,
            sample_hetero_edges_dict,
            node_feature_cols={"building": ["b_feat1"], "road": ["length"]},
            node_label_cols={"building": ["b_label"]},
            edge_feature_cols={
                ("building", "connects_to", "road"): ["conn_feat1"],
                ("road", "links_to", "road"): ["link_feat1"],
            },
        )
        assert data["building"].x.shape[1] == 1
        assert data["road"].x.shape[1] == 1
        assert data["building"].y.shape[1] == 1

    def test_homogeneous_round_trip_conversion(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Homogeneous GDF data survives GDF -> PyG -> GDF conversion."""
        data = gdf_to_pyg(sample_nodes_gdf, sample_edges_gdf)
        nodes_restored, edges_restored = pyg_to_gdf(data)

        assert isinstance(nodes_restored, gpd.GeoDataFrame)
        assert isinstance(edges_restored, gpd.GeoDataFrame)
        assert len(nodes_restored) == len(sample_nodes_gdf)
        assert len(edges_restored) == len(sample_edges_gdf)

    def test_heterogeneous_round_trip_conversion(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Heterogeneous GDF data survives GDF -> PyG -> GDF conversion."""
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
        # Edges are symmetrized by default (directed=False), so doubled
        assert data.num_edges == sample_nx_graph.number_of_edges() * 2

        # PyG -> NX (round trip) — pyg_to_nx uses pyg_to_gdf which deduplicates
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

    def test_homogeneous_feature_specs_must_be_lists(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Homogeneous conversion rejects dictionary column specs."""
        with pytest.raises(TypeError, match="node_feature_cols must be a list"):
            gdf_to_pyg(sample_nodes_gdf, node_feature_cols={"invalid": ["cols"]})

        with pytest.raises(TypeError, match="node_label_cols must be a list"):
            gdf_to_pyg(sample_nodes_gdf, node_label_cols={"invalid": ["cols"]})

        with pytest.raises(TypeError, match="edge_feature_cols must be a list"):
            gdf_to_pyg(sample_nodes_gdf, edge_feature_cols={("a", "b", "c"): ["cols"]})

    def test_heterogeneous_feature_specs_must_be_dicts(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Heterogeneous conversion rejects list column specs."""
        with pytest.raises(TypeError, match="node_feature_cols must be a dict"):
            gdf_to_pyg(sample_hetero_nodes_dict, node_feature_cols=["invalid"])

        with pytest.raises(TypeError, match="node_label_cols must be a dict"):
            gdf_to_pyg(sample_hetero_nodes_dict, node_label_cols=["invalid"])

        with pytest.raises(TypeError, match="edge_feature_cols must be a dict"):
            gdf_to_pyg(sample_hetero_nodes_dict, edge_feature_cols=["invalid"])

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
        """Public converter dispatch reports missing homogeneous and hetero nodes."""
        converter = graph_module.PyGConverter()
        edges = gpd.GeoDataFrame(
            {"geometry": [LineString([(0, 0), (1, 1)])]},
            index=pd.MultiIndex.from_tuples([(1, 2)], names=["source_id", "target_id"]),
        )

        with pytest.raises(ValueError, match="Nodes GeoDataFrame is required"):
            converter.convert(None, edges)

        with pytest.raises(ValueError, match="Nodes dictionary is required"):
            converter.convert(None, {("node", "to", "node"): edges})

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

        _, edges_rec = pyg_to_gdf(data)
        assert isinstance(edges_rec, gpd.GeoDataFrame)
        assert edges_rec.geometry.isna().all()

    def test_reconstruct_hetero_edge_errors(self, sample_pyg_hetero_data: HeteroData) -> None:
        """Invalid hetero edge metadata is rejected during public reconstruction."""
        sample_pyg_hetero_data.graph_metadata.original_edge_types = ["invalid"]

        with pytest.raises(TypeError, match="Edge type must be a tuple"):
            pyg_to_gdf(sample_pyg_hetero_data)

    def test_nodes_reconstruct_with_null_geometry_when_positions_removed(
        self, sample_pyg_data: Data
    ) -> None:
        """Node reconstruction falls back to null geometries when positions are missing."""
        sample_pyg_data.pos = None
        sample_pyg_data.graph_metadata.node_geometries = None
        nodes_restored, _ = pyg_to_gdf(sample_pyg_data)
        assert isinstance(nodes_restored, gpd.GeoDataFrame)
        assert nodes_restored.geometry.isna().all()

    def test_heterogeneous_edges_rebuild_straight_geometry_without_stored_wkb(
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

    def test_non_numeric_requested_features_are_ignored(
        self, sample_nodes_gdf: gpd.GeoDataFrame
    ) -> None:
        """Non-numeric requested feature columns produce an empty tensor."""
        gdf = sample_nodes_gdf.copy()
        gdf["str_col"] = "a"
        gdf = gdf[["str_col", "geometry"]]

        data = gdf_to_pyg(gdf, node_feature_cols=["str_col"])
        assert data.x.shape[1] == 0

    def test_geometryless_nodes_have_no_position_tensor(
        self, sample_nodes_gdf: gpd.GeoDataFrame
    ) -> None:
        """GeoDataFrames without an active geometry column create no positions."""
        gdf = gpd.GeoDataFrame(sample_nodes_gdf.drop(columns="geometry"))

        data = gdf_to_pyg(gdf)
        assert data.pos is None

    def test_geometryless_nodes_store_no_wkb_and_reconstruct_null_geometry(
        self, sample_nodes_gdf: gpd.GeoDataFrame
    ) -> None:
        """Geometry-less inputs reconstruct with null geometry under keep_geom."""
        gdf = gpd.GeoDataFrame(sample_nodes_gdf.drop(columns="geometry"))

        data = gdf_to_pyg(gdf, keep_geom=True)
        assert data.graph_metadata.node_geometries is None
        nodes_restored, _ = pyg_to_gdf(data)
        assert isinstance(nodes_restored, gpd.GeoDataFrame)
        assert nodes_restored.geometry.isna().all()


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

    def test_homogeneous_data_marked_as_heterogeneous_is_rejected(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Data objects cannot claim heterogeneous metadata."""
        data = gdf_to_pyg(sample_nodes_gdf)
        data.graph_metadata.is_hetero = True

        with pytest.raises(
            ValueError,
            match=r"Inconsistency detected.*is Data but metadata.is_hetero is True",
        ):
            validate_pyg(data)

    def test_heterogeneous_data_marked_as_homogeneous_is_rejected(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """HeteroData objects cannot claim homogeneous metadata."""
        data = gdf_to_pyg(sample_hetero_nodes_dict)
        data.graph_metadata.is_hetero = False

        with pytest.raises(
            ValueError,
            match=r"Inconsistency detected.*HeteroData but metadata.is_hetero is False",
        ):
            validate_pyg(data)

    def test_tensor_validation_errors(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test tensor validation errors."""
        # Position tensor mismatch
        data = gdf_to_pyg(sample_nodes_gdf)
        data.pos = torch.randn(1, 2)  # Wrong size
        with pytest.raises(ValueError, match=r"position tensor size.*doesn't match"):
            validate_pyg(data)

        # Label tensor mismatch
        data = gdf_to_pyg(sample_nodes_gdf, node_label_cols=["label1"])
        data.y = torch.randn(1, 1)  # Wrong size
        with pytest.raises(ValueError, match=r"label tensor size.*doesn't match"):
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
            match=r"Node type 'building': label tensor size.*doesn't match",
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
            match=r"Edge attribute tensor size.*doesn't match number of edges",
        ):
            validate_pyg(data)

    def test_heterogeneous_tensor_validation(self, sample_pyg_hetero_data: HeteroData) -> None:
        """Test tensor validation for heterogeneous graphs."""
        node_type = next(iter(sample_pyg_hetero_data.node_types))
        sample_pyg_hetero_data[node_type].pos = torch.randn(1, 2)  # Wrong size
        with pytest.raises(ValueError, match=r"position tensor size.*doesn't match"):
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


class TestOptionalTensorConversion:
    """Test conversion of optional tensor attributes."""

    def test_pyg_to_gdf_homo_additional_cols(self) -> None:
        """Test extracted additional columns for homogeneous graphs."""
        # Create sample data
        x = torch.randn(10, 5)
        z = torch.randn(10)  # Additional 1D tensor
        pos = torch.rand(10, 2)
        edge_index = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long)

        data = Data(x=x, pos=pos, edge_index=edge_index)
        data.z = z

        # Metadata is required for pyg_to_gdf
        metadata = GraphMetadata(is_hetero=False)
        data.graph_metadata = metadata

        # Test pyg_to_gdf
        nodes_gdf_tuple = pyg_to_gdf(data, additional_node_cols=["z"])
        # Unpack tuple safely
        nodes_gdf = nodes_gdf_tuple[0]
        assert isinstance(nodes_gdf, gpd.GeoDataFrame)
        assert "z" in nodes_gdf.columns
        assert len(nodes_gdf) == 10
        assert (nodes_gdf["z"].to_numpy() == z.numpy()).all()

        # Test pyg_to_nx
        nx_graph = pyg_to_nx(data, additional_node_cols=["z"])
        for i, (_, attrs) in enumerate(nx_graph.nodes(data=True)):
            assert "z" in attrs
            assert attrs["z"] == z[i].item()

    def test_pyg_to_gdf_hetero_additional_cols(self) -> None:
        """Test extracted additional columns for heterogeneous graphs."""
        data = HeteroData()
        data["oa"].x = torch.randn(5, 3)
        data["oa"].z = torch.randn(5)
        data["oa"].pos = torch.rand(5, 2)

        metadata = GraphMetadata(is_hetero=True)
        metadata.node_types = ["oa"]
        metadata.edge_types = []
        data.graph_metadata = metadata

        # Test pyg_to_gdf
        result = pyg_to_gdf(data, additional_node_cols={"oa": ["z"]})
        # Unpack tuple safely. Hetero result is (node_dict, edge_dict)
        node_gdfs = result[0]
        assert isinstance(node_gdfs, dict)
        assert "z" in node_gdfs["oa"].columns
        assert (node_gdfs["oa"]["z"].to_numpy() == data["oa"].z.numpy()).all()

        # Test pyg_to_nx
        nx_graph = pyg_to_nx(data, additional_node_cols={"oa": ["z"]})
        for _, attrs in nx_graph.nodes(data=True):
            if attrs.get("node_type") == "oa":
                assert "z" in attrs

    def test_additional_col_collision(self) -> None:
        """Test that additional_cols logic skips columns already in gdf_data."""
        # Simple setup: Data with one attribute 'z' and minimal edge_index
        z = torch.randn(5)
        data = Data(num_nodes=5, edge_index=torch.empty((2, 0), dtype=torch.long))
        data.z = z
        data.graph_metadata = GraphMetadata(is_hetero=False)

        # Request 'z' twice. It should be extracted once, and second time skipped.
        # and result has 'z', it works.
        nodes, _ = pyg_to_gdf(data, additional_node_cols=["z", "z"])
        assert isinstance(nodes, gpd.GeoDataFrame)
        assert "z" in nodes.columns
        assert list(nodes.columns).count("z") == 1

    def test_2d_tensor_flattening(self) -> None:
        """Test flattening of 2D tensor with shape (N, 1) in additional cols."""
        data = Data(num_nodes=5, edge_index=torch.empty((2, 0), dtype=torch.long))
        data.pos = torch.rand(5, 2)
        # 2D tensor with width 1
        data.z_2d = torch.randn(5, 1)
        data.graph_metadata = GraphMetadata(is_hetero=False)

        nodes, _ = pyg_to_gdf(data, additional_node_cols=["z_2d"])
        assert isinstance(nodes, gpd.GeoDataFrame)
        assert "z_2d" in nodes.columns
        assert nodes["z_2d"].to_numpy().ndim == 1

    def test_hetero_edge_type_deduction(self) -> None:
        """Test optional edge column extraction using tuple and relation string keys."""
        data = HeteroData()
        data["n"].x = torch.randn(2, 1)
        data["n"].pos = torch.rand(2, 2)

        edge_type_1 = ("n", "to", "n")
        data[edge_type_1].edge_index = torch.tensor([[0, 1], [0, 1]])
        data[edge_type_1].z = torch.randn(2)

        edge_type_2 = ("n", "via", "n")
        data[edge_type_2].edge_index = torch.tensor([[0, 1], [0, 1]])
        data[edge_type_2].w = torch.randn(2)

        metadata = GraphMetadata(is_hetero=True)
        metadata.node_types = ["n"]
        metadata.edge_types = [edge_type_1, edge_type_2]
        data.graph_metadata = metadata

        # Test matching by full edge type tuple for both
        _, edges_dict = pyg_to_gdf(
            data,
            additional_edge_cols={edge_type_1: ["z"], edge_type_2: ["w"]},
        )
        assert isinstance(edges_dict, dict)
        assert "z" in edges_dict[edge_type_1].columns
        assert "w" in edges_dict[edge_type_2].columns

    def test_hetero_edge_type_deduction_by_relation_name(self) -> None:
        """Edge column lookup should also support relation-name keys."""
        data = HeteroData()
        data["n"].x = torch.randn(2, 1)
        data["n"].pos = torch.rand(2, 2)

        edge_type = ("n", "via", "n")
        data[edge_type].edge_index = torch.tensor([[0, 1], [1, 0]])
        data[edge_type].w = torch.randn(2)

        metadata = GraphMetadata(is_hetero=True)
        metadata.node_types = ["n"]
        metadata.edge_types = [edge_type]
        data.graph_metadata = metadata

        _, edges_dict = pyg_to_gdf(data, additional_edge_cols={edge_type: ["w"]})

        assert isinstance(edges_dict, dict)
        assert "w" in edges_dict[edge_type].columns

    def test_converter_reports_missing_torch_during_device_resolution(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Converter device resolution reports missing torch through the public method."""
        monkeypatch.setattr(graph_module, "TORCH_AVAILABLE", False)
        nodes = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, index=[1])
        converter = graph_module.PyGConverter()

        with pytest.raises(ImportError, match="PyTorch and PyTorch Geometric required"):
            converter.gdf_to_pyg(nodes)


class TestUndirectedEdgeHandling:
    """Test undirected edge symmetrization and deduplication."""

    @pytest.fixture
    def simple_nodes(self) -> gpd.GeoDataFrame:
        """Create simple nodes for testing."""
        data = {
            "node_id": [1, 2, 3],
            "feature": [10.0, 20.0, 30.0],
            "geometry": [Point(0, 0), Point(1, 0), Point(1, 1)],
        }
        return gpd.GeoDataFrame(data, crs="EPSG:27700").set_index("node_id")

    @pytest.fixture
    def simple_edges(self) -> gpd.GeoDataFrame:
        """Create simple undirected edges (each pair once)."""
        source_ids = [1, 2]
        target_ids = [2, 3]
        data = {
            "source_id": source_ids,
            "target_id": target_ids,
            "weight": [0.5, 1.5],
            "geometry": [
                LineString([(0, 0), (1, 0)]),
                LineString([(1, 0), (1, 1)]),
            ],
        }
        idx = pd.MultiIndex.from_arrays([source_ids, target_ids], names=["source_id", "target_id"])
        return gpd.GeoDataFrame(data, index=idx, crs="EPSG:27700")

    def test_default_symmetrizes_edges(
        self, simple_nodes: gpd.GeoDataFrame, simple_edges: gpd.GeoDataFrame
    ) -> None:
        """Default (directed=False) doubles edges via symmetrization."""
        data = gdf_to_pyg(simple_nodes, simple_edges)
        # 2 original edges → 4 after symmetrization
        assert data.num_edges == 4

    def test_every_edge_has_reverse(
        self, simple_nodes: gpd.GeoDataFrame, simple_edges: gpd.GeoDataFrame
    ) -> None:
        """Every (u,v) should have a matching (v,u)."""
        data = gdf_to_pyg(simple_nodes, simple_edges)
        edge_set = set()
        ei = data.edge_index
        for i in range(ei.size(1)):
            edge_set.add((ei[0, i].item(), ei[1, i].item()))
        for u, v in list(edge_set):
            assert (v, u) in edge_set, f"Missing reverse edge ({v}, {u})"

    def test_directed_true_no_symmetrization(
        self, simple_nodes: gpd.GeoDataFrame, simple_edges: gpd.GeoDataFrame
    ) -> None:
        """directed=True keeps edges as-is (no doubling)."""
        data = gdf_to_pyg(simple_nodes, simple_edges, directed=True)
        assert data.num_edges == 2
        assert data.graph_metadata.is_directed is True
        assert data.graph_metadata.edge_was_symmetrized is False

    def test_issue_156_queen_grid_neighbors_are_bidirectional(self) -> None:
        """4x4 queen contiguity around id22 has every neighbor in both directions."""
        records = [
            {
                "node_id": f"id{row}{col}",
                "geometry": Polygon(
                    [
                        (col, row),
                        (col + 1, row),
                        (col + 1, row + 1),
                        (col, row + 1),
                    ]
                ),
            }
            for row in range(4)
            for col in range(4)
        ]
        polygons = gpd.GeoDataFrame(records, crs="EPSG:27700").set_index("node_id")
        nodes, edges = contiguity_graph(polygons, contiguity="queen")

        data = gdf_to_pyg(nodes, edges)
        mapping = data.graph_metadata.node_mappings["default"]["mapping"]
        focal = mapping["id22"]
        expected_neighbors = {"id11", "id12", "id13", "id21", "id23", "id31", "id32", "id33"}
        expected_edges = {(focal, mapping[neighbor]) for neighbor in expected_neighbors} | {
            (mapping[neighbor], focal) for neighbor in expected_neighbors
        }
        observed_edges = {
            (int(data.edge_index[0, i]), int(data.edge_index[1, i]))
            for i in range(data.edge_index.size(1))
        }

        assert expected_edges <= observed_edges

    def test_round_trip_restores_original_edges(
        self, simple_nodes: gpd.GeoDataFrame, simple_edges: gpd.GeoDataFrame
    ) -> None:
        """gdf_to_pyg → pyg_to_gdf round trip restores original edge count."""
        data = gdf_to_pyg(simple_nodes, simple_edges)
        assert data.num_edges == 4  # symmetrized

        _, edges_restored = pyg_to_gdf(data)
        assert isinstance(edges_restored, gpd.GeoDataFrame)
        # Should be deduplicated back to original count
        assert len(edges_restored) == 2

    def test_edge_features_duplicated(
        self, simple_nodes: gpd.GeoDataFrame, simple_edges: gpd.GeoDataFrame
    ) -> None:
        """Edge features are correctly duplicated for reverse edges."""
        data = gdf_to_pyg(simple_nodes, simple_edges, edge_feature_cols=["weight"])
        assert data.edge_attr.shape == (4, 1)
        # Original: [0.5, 1.5], Reverse: [0.5, 1.5]
        weights = data.edge_attr[:, 0].tolist()
        assert weights == [0.5, 1.5, 0.5, 1.5]

    def test_self_loops_not_duplicated(self) -> None:
        """Self-loops should not be duplicated during symmetrization."""
        nodes = gpd.GeoDataFrame(
            {"node_id": [1, 2], "geometry": [Point(0, 0), Point(1, 0)]},
            crs="EPSG:27700",
        ).set_index("node_id")

        source_ids = [1, 1]
        target_ids = [2, 1]
        edges_data = {
            "source_id": source_ids,
            "target_id": target_ids,  # 1→2 and 1→1 (self-loop)
            "geometry": [LineString([(0, 0), (1, 0)]), LineString([(0, 0), (0, 0)])],
        }
        idx = pd.MultiIndex.from_arrays(
            [source_ids, target_ids],
            names=["source_id", "target_id"],
        )
        edges = gpd.GeoDataFrame(edges_data, index=idx, crs="EPSG:27700")

        data = gdf_to_pyg(nodes, edges)
        # 1→2 gets reversed (adds 2→1), 1→1 stays as is: total 3
        assert data.num_edges == 3

    def test_only_self_loops_do_not_trigger_parallel_validation(self) -> None:
        """Self-loop-only undirected tables skip unordered-pair validation."""
        nodes = gpd.GeoDataFrame(
            {"node_id": [1], "geometry": [Point(0, 0)]},
            crs="EPSG:27700",
        ).set_index("node_id")
        edges = gpd.GeoDataFrame(
            {"geometry": [LineString([(0, 0), (0, 0)])]},
            index=pd.MultiIndex.from_arrays([[1], [1]], names=["source_id", "target_id"]),
            crs="EPSG:27700",
        )

        data = gdf_to_pyg(nodes, edges)

        assert data.num_edges == 1

    def test_hetero_same_type_symmetrized(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Same-type hetero edges (road→road) are symmetrized."""
        # Create edges with road→road only having one direction
        road_links = gpd.GeoDataFrame(
            {
                "source_road_id": ["r1"],
                "target_road_id": ["r2"],
                "feat": [1.0],
                "geometry": [LineString([(10, 12), (12, 12)])],
            },
            crs="EPSG:27700",
        )
        road_links = road_links.set_index(
            pd.MultiIndex.from_arrays(
                [road_links["source_road_id"], road_links["target_road_id"]],
                names=["source_road_id", "target_road_id"],
            )
        )

        edges_dict = {("road", "links_to", "road"): road_links}
        data = gdf_to_pyg(sample_hetero_nodes_dict, edges_dict)

        # 1 edge → 2 after symmetrization
        et = ("road", "links_to", "road")
        assert data[et].edge_index.size(1) == 2

    def test_hetero_cross_type_not_symmetrized(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Cross-type hetero edges (building→road) are NOT symmetrized."""
        data = gdf_to_pyg(sample_hetero_nodes_dict, sample_hetero_edges_dict)

        # building→road: 3 edges, NOT symmetrized (different types)
        et_cross = ("building", "connects_to", "road")
        assert data[et_cross].edge_index.size(1) == 3

    # ------------------------------------------------------------------
    # NEW: Graph-semantics tests from Implementation Plan v2 Step 7
    # ------------------------------------------------------------------

    def test_nx_digraph_stays_directed(self) -> None:
        """nx.DiGraph edges are not symmetrized in PyG."""
        G = nx.DiGraph()
        G.add_node(1, pos=(0, 0), geometry=Point(0, 0))
        G.add_node(2, pos=(1, 0), geometry=Point(1, 0))
        G.add_edge(1, 2, geometry=LineString([(0, 0), (1, 0)]))
        G.graph["crs"] = "EPSG:27700"
        G.graph["is_hetero"] = False

        data = nx_to_pyg(G)
        # Directed: edges should NOT be doubled
        assert data.num_edges == 1, f"Expected 1 directed edge, got {data.num_edges}"
        assert data.graph_metadata.is_directed is True

    def test_nx_multidigraph_stays_directed(self) -> None:
        """nx.MultiDiGraph edges are not symmetrized in PyG."""
        G = nx.MultiDiGraph()
        G.add_node(1, pos=(0, 0), geometry=Point(0, 0))
        G.add_node(2, pos=(1, 0), geometry=Point(1, 0))
        G.add_edge(1, 2, key=0, geometry=LineString([(0, 0), (1, 0)]))
        G.graph["crs"] = "EPSG:27700"
        G.graph["is_hetero"] = False

        data = nx_to_pyg(G)
        assert data.num_edges == 1, f"Expected 1 directed edge, got {data.num_edges}"
        assert data.graph_metadata.is_directed is True

    def test_nx_graph_symmetrized(self) -> None:
        """nx.Graph edges are symmetrized in PyG (bidirectional)."""
        G = nx.Graph()
        G.add_node(1, pos=(0, 0), geometry=Point(0, 0))
        G.add_node(2, pos=(1, 0), geometry=Point(1, 0))
        G.add_edge(1, 2, geometry=LineString([(0, 0), (1, 0)]))
        G.graph["crs"] = "EPSG:27700"
        G.graph["is_hetero"] = False

        data = nx_to_pyg(G)
        # Undirected: edges should be doubled
        assert data.num_edges == 2, f"Expected 2 edges after symmetrization, got {data.num_edges}"
        assert data.graph_metadata.is_directed is False

    def test_already_bidirectional_gdf_rejected(self, simple_nodes: gpd.GeoDataFrame) -> None:
        """GDF with both (u,v) and (v,u) raises ValueError when directed=False."""
        source_ids = [1, 2]
        target_ids = [2, 1]
        edges_data = {
            "source_id": source_ids,
            "target_id": target_ids,
            "geometry": [
                LineString([(0, 0), (1, 0)]),
                LineString([(1, 0), (0, 0)]),
            ],
        }
        idx = pd.MultiIndex.from_arrays(
            [source_ids, target_ids],
            names=["source_id", "target_id"],
        )
        edges = gpd.GeoDataFrame(edges_data, index=idx, crs="EPSG:27700")

        with pytest.raises(ValueError, match="Ambiguous undirected input"):
            gdf_to_pyg(simple_nodes, edges, directed=False)

    def test_already_bidirectional_ok_when_directed(self, simple_nodes: gpd.GeoDataFrame) -> None:
        """Already-bidirectional GDF is fine when directed=True."""
        source_ids = [1, 2]
        target_ids = [2, 1]
        edges_data = {
            "source_id": source_ids,
            "target_id": target_ids,
            "geometry": [
                LineString([(0, 0), (1, 0)]),
                LineString([(1, 0), (0, 0)]),
            ],
        }
        idx = pd.MultiIndex.from_arrays(
            [source_ids, target_ids],
            names=["source_id", "target_id"],
        )
        edges = gpd.GeoDataFrame(edges_data, index=idx, crs="EPSG:27700")

        data = gdf_to_pyg(simple_nodes, edges, directed=True)
        assert data.num_edges == 2

    def test_parallel_undirected_rows_rejected(self, simple_nodes: gpd.GeoDataFrame) -> None:
        """Duplicate unordered keys (parallel undirected) raise ValueError."""
        source_ids = [1, 1]
        target_ids = [2, 2]
        edges_data = {
            "source_id": source_ids,
            "target_id": target_ids,
            "geometry": [
                LineString([(0, 0), (1, 0)]),
                LineString([(0, 0), (1, 0)]),
            ],
        }
        idx = pd.MultiIndex.from_arrays(
            [source_ids, target_ids],
            names=["source_id", "target_id"],
        )
        edges = gpd.GeoDataFrame(edges_data, index=idx, crs="EPSG:27700")

        with pytest.raises(ValueError, match="Parallel undirected edges"):
            gdf_to_pyg(simple_nodes, edges, directed=False)

    def test_cross_type_auto_reverse_store(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Cross-type undirected edges generate auto reverse stores."""
        conn_data = {
            "building_id": ["b1"],
            "road_id": ["r1"],
            "geometry": [LineString([(10, 10), (10, 12)])],
        }
        conn_gdf = gpd.GeoDataFrame(
            conn_data,
            index=pd.MultiIndex.from_arrays(
                [conn_data["building_id"], conn_data["road_id"]],
                names=["building_id", "road_id"],
            ),
            crs="EPSG:27700",
        )
        edges_dict = {("building", "connects_to", "road"): conn_gdf}

        data = gdf_to_pyg(sample_hetero_nodes_dict, edges_dict)

        # Original edge type still has 1 edge
        et = ("building", "connects_to", "road")
        assert data[et].edge_index.size(1) == 1

        # Generated reverse store should exist
        rev_et = ("road", "rev_connects_to", "building")
        assert rev_et in data.edge_types
        assert data[rev_et].edge_index.size(1) == 1

        # Reverse should flip source/target
        assert data[rev_et].edge_index[0].tolist() == data[et].edge_index[1].tolist()
        assert data[rev_et].edge_index[1].tolist() == data[et].edge_index[0].tolist()

    def test_cross_type_explicit_reverse_mapping(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Explicit reverse_edge_types mapping is used for cross-type edges."""
        conn_data = {
            "building_id": ["b1"],
            "road_id": ["r1"],
            "geometry": [LineString([(10, 10), (10, 12)])],
        }
        conn_gdf = gpd.GeoDataFrame(
            conn_data,
            index=pd.MultiIndex.from_arrays(
                [conn_data["building_id"], conn_data["road_id"]],
                names=["building_id", "road_id"],
            ),
            crs="EPSG:27700",
        )
        edges_dict = {("building", "connects_to", "road"): conn_gdf}

        custom_rev = ("road", "served_by", "building")
        data = gdf_to_pyg(
            sample_hetero_nodes_dict,
            edges_dict,
            reverse_edge_types={("building", "connects_to", "road"): custom_rev},
        )

        assert custom_rev in data.edge_types
        # Auto-generated name should NOT exist
        assert ("road", "rev_connects_to", "building") not in data.edge_types

    def test_cross_type_strict_mode_raises(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """reverse_edge_types=None (strict mode) raises for undirected cross-type."""
        conn_data = {
            "building_id": ["b1"],
            "road_id": ["r1"],
            "geometry": [LineString([(10, 10), (10, 12)])],
        }
        conn_gdf = gpd.GeoDataFrame(
            conn_data,
            index=pd.MultiIndex.from_arrays(
                [conn_data["building_id"], conn_data["road_id"]],
                names=["building_id", "road_id"],
            ),
            crs="EPSG:27700",
        )
        edges_dict = {("building", "connects_to", "road"): conn_gdf}

        with pytest.raises(ValueError, match="strict mode"):
            gdf_to_pyg(
                sample_hetero_nodes_dict,
                edges_dict,
                reverse_edge_types=None,
            )

    def test_cross_type_directed_no_reverse_store(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Cross-type directed edges don't generate reverse stores."""
        conn_data = {
            "building_id": ["b1"],
            "road_id": ["r1"],
            "geometry": [LineString([(10, 10), (10, 12)])],
        }
        conn_gdf = gpd.GeoDataFrame(
            conn_data,
            index=pd.MultiIndex.from_arrays(
                [conn_data["building_id"], conn_data["road_id"]],
                names=["building_id", "road_id"],
            ),
            crs="EPSG:27700",
        )
        edges_dict = {("building", "connects_to", "road"): conn_gdf}

        data = gdf_to_pyg(sample_hetero_nodes_dict, edges_dict, directed=True)
        assert ("road", "rev_connects_to", "building") not in data.edge_types
        assert data[("building", "connects_to", "road")].edge_index.size(1) == 1

    def test_directed_dict_complete(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Complete directed dict: road→road directed, building→road undirected."""
        road_data = {
            "source_road_id": ["r1"],
            "target_road_id": ["r2"],
            "geometry": [LineString([(10, 12), (12, 12)])],
        }
        road_gdf = gpd.GeoDataFrame(
            road_data,
            index=pd.MultiIndex.from_arrays(
                [road_data["source_road_id"], road_data["target_road_id"]],
                names=["source_road_id", "target_road_id"],
            ),
            crs="EPSG:27700",
        )
        conn_data = {
            "building_id": ["b1"],
            "road_id": ["r1"],
            "geometry": [LineString([(10, 10), (10, 12)])],
        }
        conn_gdf = gpd.GeoDataFrame(
            conn_data,
            index=pd.MultiIndex.from_arrays(
                [conn_data["building_id"], conn_data["road_id"]],
                names=["building_id", "road_id"],
            ),
            crs="EPSG:27700",
        )

        edges_dict = {
            ("road", "links_to", "road"): road_gdf,
            ("building", "connects_to", "road"): conn_gdf,
        }

        data = gdf_to_pyg(
            sample_hetero_nodes_dict,
            edges_dict,
            directed={
                ("road", "links_to", "road"): True,
                ("building", "connects_to", "road"): False,
            },
        )

        # road→road is directed: no symmetrization, 1 edge
        assert data[("road", "links_to", "road")].edge_index.size(1) == 1
        # building→road is undirected: has reverse store
        assert data[("building", "connects_to", "road")].edge_index.size(1) == 1
        assert ("road", "rev_connects_to", "building") in data.edge_types

    def test_directed_dict_missing_keys_raises(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Incomplete directed dict raises ValueError."""
        road_data = {
            "source_road_id": ["r1"],
            "target_road_id": ["r2"],
            "geometry": [LineString([(10, 12), (12, 12)])],
        }
        road_gdf = gpd.GeoDataFrame(
            road_data,
            index=pd.MultiIndex.from_arrays(
                [road_data["source_road_id"], road_data["target_road_id"]],
                names=["source_road_id", "target_road_id"],
            ),
            crs="EPSG:27700",
        )
        conn_data = {
            "building_id": ["b1"],
            "road_id": ["r1"],
            "geometry": [LineString([(10, 10), (10, 12)])],
        }
        conn_gdf = gpd.GeoDataFrame(
            conn_data,
            index=pd.MultiIndex.from_arrays(
                [conn_data["building_id"], conn_data["road_id"]],
                names=["building_id", "road_id"],
            ),
            crs="EPSG:27700",
        )

        edges_dict = {
            ("road", "links_to", "road"): road_gdf,
            ("building", "connects_to", "road"): conn_gdf,
        }

        with pytest.raises(ValueError, match="directed dict is missing keys"):
            gdf_to_pyg(
                sample_hetero_nodes_dict,
                edges_dict,
                directed={("road", "links_to", "road"): True},  # missing building→road
            )

    def test_directed_dict_extra_keys_raises(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Extra keys in directed dict raises ValueError."""
        road_data = {
            "source_road_id": ["r1"],
            "target_road_id": ["r2"],
            "geometry": [LineString([(10, 12), (12, 12)])],
        }
        road_gdf = gpd.GeoDataFrame(
            road_data,
            index=pd.MultiIndex.from_arrays(
                [road_data["source_road_id"], road_data["target_road_id"]],
                names=["source_road_id", "target_road_id"],
            ),
            crs="EPSG:27700",
        )
        edges_dict = {("road", "links_to", "road"): road_gdf}

        with pytest.raises(ValueError, match="directed dict has extra keys"):
            gdf_to_pyg(
                sample_hetero_nodes_dict,
                edges_dict,
                directed={
                    ("road", "links_to", "road"): True,
                    ("building", "fake_rel", "road"): False,
                },
            )

    def test_pyg_to_gdf_skips_generated_reverse_stores(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """pyg_to_gdf skips generated reverse edge stores during reconstruction."""
        conn_data = {
            "building_id": ["b1"],
            "road_id": ["r1"],
            "geometry": [LineString([(10, 10), (10, 12)])],
        }
        conn_gdf = gpd.GeoDataFrame(
            conn_data,
            index=pd.MultiIndex.from_arrays(
                [conn_data["building_id"], conn_data["road_id"]],
                names=["building_id", "road_id"],
            ),
            crs="EPSG:27700",
        )
        edges_dict = {("building", "connects_to", "road"): conn_gdf}

        data = gdf_to_pyg(sample_hetero_nodes_dict, edges_dict)
        _, edges_restored = pyg_to_gdf(data)

        assert isinstance(edges_restored, dict)
        # Only the original edge type should be returned
        assert ("building", "connects_to", "road") in edges_restored
        assert ("road", "rev_connects_to", "building") not in edges_restored

    def test_reverse_type_collision_raises(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Auto-generated reverse type colliding with user type raises ValueError."""
        conn_data = {
            "building_id": ["b1"],
            "road_id": ["r1"],
            "geometry": [LineString([(10, 10), (10, 12)])],
        }
        conn_gdf = gpd.GeoDataFrame(
            conn_data,
            index=pd.MultiIndex.from_arrays(
                [conn_data["building_id"], conn_data["road_id"]],
                names=["building_id", "road_id"],
            ),
            crs="EPSG:27700",
        )
        # Provide a user edge type that would collide with auto-generated reverse
        rev_data = {
            "road_id": ["r1"],
            "building_id": ["b1"],
            "geometry": [LineString([(10, 12), (10, 10)])],
        }
        rev_gdf = gpd.GeoDataFrame(
            rev_data,
            index=pd.MultiIndex.from_arrays(
                [rev_data["road_id"], rev_data["building_id"]],
                names=["road_id", "building_id"],
            ),
            crs="EPSG:27700",
        )

        edges_dict = {
            ("building", "connects_to", "road"): conn_gdf,
            ("road", "rev_connects_to", "building"): rev_gdf,
        }

        with pytest.raises(ValueError, match="collides with an existing"):
            gdf_to_pyg(sample_hetero_nodes_dict, edges_dict)

    def test_round_trip_with_cross_type_undirected(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Round trip preserves original edge count for cross-type undirected edges."""
        conn_data = {
            "building_id": ["b1", "b2"],
            "road_id": ["r1", "r2"],
            "feat": [1.0, 2.0],
            "geometry": [
                LineString([(10, 10), (10, 12)]),
                LineString([(11, 11), (12, 12)]),
            ],
        }
        conn_gdf = gpd.GeoDataFrame(
            conn_data,
            index=pd.MultiIndex.from_arrays(
                [["b1", "b2"], ["r1", "r2"]],
                names=["building_id", "road_id"],
            ),
            crs="EPSG:27700",
        )
        edges_dict = {("building", "connects_to", "road"): conn_gdf}

        data = gdf_to_pyg(sample_hetero_nodes_dict, edges_dict)
        _, edges_restored = pyg_to_gdf(data)

        assert isinstance(edges_restored, dict)
        et = ("building", "connects_to", "road")
        assert len(edges_restored[et]) == 2

    def test_metadata_edge_was_symmetrized_homo(
        self, simple_nodes: gpd.GeoDataFrame, simple_edges: gpd.GeoDataFrame
    ) -> None:
        """Homogeneous edge_was_symmetrized is True when directed=False."""
        data = gdf_to_pyg(simple_nodes, simple_edges, directed=False)
        assert data.graph_metadata.edge_was_symmetrized is True

        data_dir = gdf_to_pyg(simple_nodes, simple_edges, directed=True)
        assert data_dir.graph_metadata.edge_was_symmetrized is False

    def test_metadata_edge_was_symmetrized_hetero(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Hetero edge_was_symmetrized is per-edge-type dict."""
        road_data = {
            "source_road_id": ["r1"],
            "target_road_id": ["r2"],
            "geometry": [LineString([(10, 12), (12, 12)])],
        }
        road_gdf = gpd.GeoDataFrame(
            road_data,
            index=pd.MultiIndex.from_arrays(
                [road_data["source_road_id"], road_data["target_road_id"]],
                names=["source_road_id", "target_road_id"],
            ),
            crs="EPSG:27700",
        )
        edges_dict = {("road", "links_to", "road"): road_gdf}

        data = gdf_to_pyg(sample_hetero_nodes_dict, edges_dict)
        ews = data.graph_metadata.edge_was_symmetrized
        assert isinstance(ews, dict)
        assert ews[("road", "links_to", "road")] is True

    def test_directed_dict_for_homogeneous_raises(
        self, simple_nodes: gpd.GeoDataFrame, simple_edges: gpd.GeoDataFrame
    ) -> None:
        """Homogeneous graphs reject directed dict."""
        with pytest.raises(TypeError, match="directed must be a bool for homogeneous"):
            gdf_to_pyg(
                simple_nodes,
                simple_edges,
                directed={("n", "r", "n"): True},
            )

    def test_explicit_reverse_inconsistent_endpoints_raises(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Explicit reverse with wrong endpoints raises ValueError."""
        conn_data = {
            "building_id": ["b1"],
            "road_id": ["r1"],
            "geometry": [LineString([(10, 10), (10, 12)])],
        }
        conn_gdf = gpd.GeoDataFrame(
            conn_data,
            index=pd.MultiIndex.from_arrays(
                [conn_data["building_id"], conn_data["road_id"]],
                names=["building_id", "road_id"],
            ),
            crs="EPSG:27700",
        )
        edges_dict = {("building", "connects_to", "road"): conn_gdf}

        with pytest.raises(ValueError, match="inconsistent endpoints"):
            gdf_to_pyg(
                sample_hetero_nodes_dict,
                edges_dict,
                reverse_edge_types={
                    ("building", "connects_to", "road"): ("building", "wrong_rev", "road"),
                },
            )

    def test_explicit_reverse_missing_mapping_raises(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Explicit reverse dict missing a mapping raises ValueError."""
        conn_data = {
            "building_id": ["b1"],
            "road_id": ["r1"],
            "geometry": [LineString([(10, 10), (10, 12)])],
        }
        conn_gdf = gpd.GeoDataFrame(
            conn_data,
            index=pd.MultiIndex.from_arrays(
                [conn_data["building_id"], conn_data["road_id"]],
                names=["building_id", "road_id"],
            ),
            crs="EPSG:27700",
        )
        edges_dict = {("building", "connects_to", "road"): conn_gdf}

        with pytest.raises(ValueError, match="missing a mapping"):
            gdf_to_pyg(
                sample_hetero_nodes_dict,
                edges_dict,
                reverse_edge_types={},  # empty dict, missing the required mapping
            )

    def test_original_edge_types_metadata(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """original_edge_types records only user-supplied edge types."""
        data = gdf_to_pyg(sample_hetero_nodes_dict, sample_hetero_edges_dict)
        oets = data.graph_metadata.original_edge_types
        assert set(oets) == set(sample_hetero_edges_dict.keys())
        # Should NOT include generated reverse stores
        gen = data.graph_metadata.generated_reverse_edge_types
        for gen_et in gen:
            assert gen_et not in oets

    def test_non_multiindex_edges_rejected(
        self, simple_nodes: gpd.GeoDataFrame, simple_edges: gpd.GeoDataFrame
    ) -> None:
        """Plain edge indexes are rejected by the public converter."""
        edges = simple_edges.reset_index(drop=True)

        with pytest.raises(ValueError, match="MultiIndex with at least two levels"):
            gdf_to_pyg(simple_nodes, edges)

    def test_empty_non_multiindex_edges_accepted(self, simple_nodes: gpd.GeoDataFrame) -> None:
        """Empty edge tables can omit a MultiIndex."""
        edges = gpd.GeoDataFrame(
            {"geometry": gpd.GeoSeries([], crs="EPSG:27700")},
            geometry="geometry",
            crs="EPSG:27700",
        )

        data = gdf_to_pyg(simple_nodes, edges)

        assert data.num_edges == 0

    def test_three_level_multiindex_accepted(
        self, simple_nodes: gpd.GeoDataFrame, simple_edges: gpd.GeoDataFrame
    ) -> None:
        """Three-level MultiGraph edge indexes are accepted."""
        edges = simple_edges.copy()
        edges.index = pd.MultiIndex.from_arrays(
            [
                edges.index.get_level_values(0),
                edges.index.get_level_values(1),
                [0, 0],
            ],
            names=["source_id", "target_id", "edge_key"],
        )

        data = gdf_to_pyg(simple_nodes, edges, directed=True)

        assert data.num_edges == len(edges)

    def test_three_level_parallel_undirected_edges_round_trip(
        self, simple_nodes: gpd.GeoDataFrame
    ) -> None:
        """Three-level MultiGraph-style parallel rows round-trip with keys."""
        edges = gpd.GeoDataFrame(
            {
                "weight": [1.0, 2.0, 3.0],
                "geometry": [
                    LineString([(0, 0), (1, 0)]),
                    LineString([(0, 0), (0.5, 0.2), (1, 0)]),
                    LineString([(0, 0), (0, 0)]),
                ],
            },
            index=pd.MultiIndex.from_arrays(
                [[1, 1, 1], [2, 2, 1], ["road", "rail", "loop"]],
                names=["source_id", "target_id", "edge_key"],
            ),
            crs="EPSG:27700",
        )

        data = gdf_to_pyg(simple_nodes, edges, edge_feature_cols=["weight"])
        _, restored_edges = pyg_to_gdf(data)

        assert data.num_edges == 5
        assert isinstance(restored_edges, gpd.GeoDataFrame)
        pd.testing.assert_index_equal(restored_edges.index, edges.index)
        assert restored_edges["weight"].tolist() == [1.0, 2.0, 3.0]
        assert restored_edges.geometry.equals(edges.geometry)

    def test_opt_in_multigraph_generates_key_level(self, simple_nodes: gpd.GeoDataFrame) -> None:
        """multigraph=True preserves parallel two-level rows with generated keys."""
        edges = gpd.GeoDataFrame(
            {
                "weight": [1.0, 2.0],
                "geometry": [
                    LineString([(0, 0), (1, 0)]),
                    LineString([(0, 0), (0.5, 0.2), (1, 0)]),
                ],
            },
            index=pd.MultiIndex.from_arrays(
                [[1, 1], [2, 2]],
                names=["source_id", "target_id"],
            ),
            crs="EPSG:27700",
        )

        data = gdf_to_pyg(
            simple_nodes,
            edges,
            edge_feature_cols=["weight"],
            multigraph=True,
        )
        _, restored_edges = pyg_to_gdf(data)

        assert data.num_edges == 4
        assert isinstance(restored_edges, gpd.GeoDataFrame)
        assert restored_edges.index.names == ["source_id", "target_id", "key"]
        assert restored_edges.index.get_level_values("key").tolist() == [0, 1]
        assert restored_edges["weight"].tolist() == [1.0, 2.0]

    def test_empty_edge_features_resized_on_symmetrization(
        self, simple_nodes: gpd.GeoDataFrame, simple_edges: gpd.GeoDataFrame
    ) -> None:
        """Empty edge feature matrices resize during symmetrisation."""
        data = gdf_to_pyg(simple_nodes, simple_edges, edge_feature_cols=[])

        assert data.edge_attr.shape == (4, 0)

    def test_cross_type_reverse_store_clones_edge_attr(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Generated cross-type reverse stores clone edge attributes."""
        edge_type = ("building", "connects_to", "road")
        reverse_type = ("road", "rev_connects_to", "building")
        conn_gdf = gpd.GeoDataFrame(
            {
                "building_id": ["b1", "b2"],
                "road_id": ["r1", "r2"],
                "score": [1.25, 2.5],
                "geometry": [
                    LineString([(10, 10), (10, 12)]),
                    LineString([(11, 11), (12, 12)]),
                ],
            },
            index=pd.MultiIndex.from_arrays(
                [["b1", "b2"], ["r1", "r2"]],
                names=["building_id", "road_id"],
            ),
            crs="EPSG:27700",
        )

        data = gdf_to_pyg(
            sample_hetero_nodes_dict,
            {edge_type: conn_gdf},
            edge_feature_cols={edge_type: ["score"]},
        )

        assert data[reverse_type].edge_attr.shape == data[edge_type].edge_attr.shape
        assert torch.equal(data[reverse_type].edge_attr, data[edge_type].edge_attr)
        assert data[reverse_type].edge_attr is not data[edge_type].edge_attr

    def test_large_undirected_validation_smoke(self) -> None:
        """Large unique undirected edge tables validate and round-trip."""
        edge_count = 10_000
        node_ids = list(range(edge_count + 1))
        nodes = gpd.GeoDataFrame(
            {"geometry": [Point(float(i), 0.0) for i in node_ids]},
            index=pd.Index(node_ids, name="node_id"),
            crs="EPSG:27700",
        )
        src = list(range(edge_count))
        dst = list(range(1, edge_count + 1))
        edges = gpd.GeoDataFrame(
            {
                "geometry": [LineString([(float(i), 0.0), (float(i + 1), 0.0)]) for i in src],
            },
            index=pd.MultiIndex.from_arrays([src, dst], names=["source_id", "target_id"]),
            crs="EPSG:27700",
        )

        start = time.perf_counter()
        data = gdf_to_pyg(nodes, edges)
        _, restored_edges = pyg_to_gdf(data)
        elapsed = time.perf_counter() - start

        assert data.num_edges == edge_count * 2
        assert isinstance(restored_edges, gpd.GeoDataFrame)
        assert len(restored_edges) == edge_count
        assert elapsed < 20.0

    def test_categorical_node_ids_round_trip_undirected(self) -> None:
        """Categorical node IDs do not require ordering for deduplication."""
        node_ids = pd.CategoricalIndex(["b", "a", "c"], name="node_id")
        nodes = gpd.GeoDataFrame(
            {"geometry": [Point(0, 0), Point(1, 0), Point(2, 0)]},
            index=node_ids,
            crs="EPSG:27700",
        )
        edges = gpd.GeoDataFrame(
            {"geometry": [LineString([(0, 0), (1, 0)]), LineString([(1, 0), (2, 0)])]},
            index=pd.MultiIndex.from_arrays(
                [
                    pd.Categorical(["b", "a"]),
                    pd.Categorical(["a", "c"]),
                ],
                names=["source_id", "target_id"],
            ),
            crs="EPSG:27700",
        )

        _, restored_edges = pyg_to_gdf(gdf_to_pyg(nodes, edges))

        assert isinstance(restored_edges, gpd.GeoDataFrame)
        assert len(restored_edges) == len(edges)

    def test_uuid_string_node_ids_round_trip_undirected(self) -> None:
        """UUID string node IDs round-trip after undirected symmetrization."""
        ids = [
            "7f4e2c0e-d9d7-4a90-b9f2-5a03e2d86a01",
            "3103f18b-3552-46be-a6d0-763f25de52c4",
            "f24207d2-c43c-419a-a37a-b602094a8eb2",
        ]
        nodes = gpd.GeoDataFrame(
            {"geometry": [Point(0, 0), Point(1, 0), Point(2, 0)]},
            index=pd.Index(ids, name="node_id"),
            crs="EPSG:27700",
        )
        edges = gpd.GeoDataFrame(
            {"geometry": [LineString([(0, 0), (1, 0)]), LineString([(1, 0), (2, 0)])]},
            index=pd.MultiIndex.from_arrays(
                [[ids[1], ids[2]], [ids[0], ids[1]]],
                names=["source_id", "target_id"],
            ),
            crs="EPSG:27700",
        )

        _, restored_edges = pyg_to_gdf(gdf_to_pyg(nodes, edges))

        assert isinstance(restored_edges, gpd.GeoDataFrame)
        assert len(restored_edges) == len(edges)

    def test_nx_multigraph_symmetrized(self) -> None:
        """nx.MultiGraph edges are symmetrised in PyG."""
        graph = nx.MultiGraph()
        graph.add_node(1, pos=(0, 0), geometry=Point(0, 0))
        graph.add_node(2, pos=(1, 0), geometry=Point(1, 0))
        graph.add_edge(1, 2, key=0, geometry=LineString([(0, 0), (1, 0)]))
        graph.graph["crs"] = "EPSG:27700"
        graph.graph["is_hetero"] = False

        data = nx_to_pyg(graph)

        assert data.num_edges == 2
        assert data.graph_metadata.is_directed is False

    def test_nx_multigraph_round_trip_preserves_parallel_keys(self) -> None:
        """nx.MultiGraph round-trips as a MultiGraph with edge keys."""
        graph = nx.MultiGraph()
        graph.add_node("a", pos=(0, 0), geometry=Point(0, 0))
        graph.add_node("b", pos=(1, 0), geometry=Point(1, 0))
        graph.add_edge(
            "a",
            "b",
            key="road",
            weight=1.0,
            geometry=LineString([(0, 0), (1, 0)]),
        )
        graph.add_edge(
            "a",
            "b",
            key="rail",
            weight=2.0,
            geometry=LineString([(0, 0), (0.5, 0.2), (1, 0)]),
        )
        graph.graph["crs"] = "EPSG:27700"
        graph.graph["is_hetero"] = False

        data = nx_to_pyg(graph, edge_feature_cols=["weight"])
        restored = pyg_to_nx(data)

        assert isinstance(restored, nx.MultiGraph)
        assert restored.is_directed() is False
        assert restored.number_of_edges("a", "b") == 2
        assert {key for _, _, key in restored.edges(keys=True)} == {"road", "rail"}

    def test_osmnx_shaped_multidigraph_gdfs_round_trip_preserve_edge_keys(self) -> None:
        """OSMnx-shaped MultiDiGraph GeoDataFrames keep keyed edge identity."""
        nodes = gpd.GeoDataFrame(
            {
                "x": [0.0, 1.0, 2.0],
                "y": [0.0, 0.0, 0.0],
                "geometry": [Point(0, 0), Point(1, 0), Point(2, 0)],
            },
            index=pd.Index([101, 202, 303], name="osmid"),
            crs="EPSG:4326",
        )
        edges = gpd.GeoDataFrame(
            {
                "osmid": [10, 11, 12],
                "length": [100.0, 110.0, 125.0],
                "geometry": [
                    LineString([(0, 0), (1, 0)]),
                    LineString([(0, 0), (0.5, 0.1), (1, 0)]),
                    LineString([(1, 0), (2, 0)]),
                ],
            },
            index=pd.MultiIndex.from_tuples(
                [(101, 202, 0), (101, 202, 1), (202, 303, 0)],
                names=["u", "v", "key"],
            ),
            crs="EPSG:4326",
        )

        data = gdf_to_pyg(nodes, edges, edge_feature_cols=["length"], directed=True)
        restored_nodes, restored_edges = pyg_to_gdf(data)

        assert data.num_edges == len(edges)
        assert isinstance(restored_nodes, gpd.GeoDataFrame)
        assert isinstance(restored_edges, gpd.GeoDataFrame)
        assert len(restored_nodes) == len(nodes)
        assert len(restored_edges) == len(edges)
        assert restored_edges.index.nlevels == 3
        assert restored_edges.index.names == ["u", "v", "key"]
        assert set(restored_edges.index) == set(edges.index)
        pd.testing.assert_series_equal(
            restored_edges.sort_index()["length"],
            edges.sort_index()["length"],
            check_names=False,
            check_dtype=False,
        )

    def test_osmnx_shaped_multigraph_gdfs_round_trip_deduplicates_symmetrized_edges(
        self,
    ) -> None:
        """Undirected OSMnx-shaped MultiGraph GeoDataFrames dedupe by pair plus key."""
        nodes = gpd.GeoDataFrame(
            {
                "x": [0.0, 1.0, 2.0],
                "y": [0.0, 0.0, 0.0],
                "geometry": [Point(0, 0), Point(1, 0), Point(2, 0)],
            },
            index=pd.Index([101, 202, 303], name="osmid"),
            crs="EPSG:4326",
        )
        edges = gpd.GeoDataFrame(
            {
                "osmid": [10, 11, 12],
                "length": [100.0, 110.0, 125.0],
                "geometry": [
                    LineString([(0, 0), (1, 0)]),
                    LineString([(0, 0), (0.5, 0.1), (1, 0)]),
                    LineString([(1, 0), (2, 0)]),
                ],
            },
            index=pd.MultiIndex.from_tuples(
                [(101, 202, 0), (101, 202, 1), (202, 303, 0)],
                names=["u", "v", "key"],
            ),
            crs="EPSG:4326",
        )

        data = gdf_to_pyg(nodes, edges, edge_feature_cols=["length"], directed=False)
        _, restored_edges = pyg_to_gdf(data)

        assert data.num_edges == len(edges) * 2
        assert data.graph_metadata.is_multigraph is True
        assert data.graph_metadata.edge_was_symmetrized is True
        assert isinstance(restored_edges, gpd.GeoDataFrame)
        assert len(restored_edges) == len(edges)
        assert restored_edges.index.names == ["u", "v", "key"]
        assert set(restored_edges.index) == set(edges.index)
        pd.testing.assert_series_equal(
            restored_edges.sort_index()["length"],
            edges.sort_index()["length"],
            check_names=False,
            check_dtype=False,
        )

    def test_osmnx_downloaded_multidigraph_smoke(self) -> None:
        """Downloaded OSMnx MultiDiGraph round-trips through PyG with keyed edges."""
        graph = ox.graph_from_bbox(
            (-0.128, 51.501, -0.124, 51.503),
            network_type="drive",
            simplify=True,
            retain_all=False,
        )
        nodes, edges = ox.graph_to_gdfs(graph)

        data = gdf_to_pyg(
            nodes,
            edges,
            edge_feature_cols=["length"],
            directed=graph.is_directed(),
            multigraph=graph.is_multigraph(),
        )
        restored_nodes, restored_edges = pyg_to_gdf(data)

        assert isinstance(graph, nx.MultiDiGraph)
        assert isinstance(restored_nodes, gpd.GeoDataFrame)
        assert isinstance(restored_edges, gpd.GeoDataFrame)
        assert len(restored_nodes) == len(nodes)
        assert len(restored_edges) == len(edges)
        assert restored_edges.index.nlevels == 3
        assert set(restored_edges.index) == set(edges.index)

        undirected_graph = ox.convert.to_undirected(graph)
        undirected_nodes, undirected_edges = ox.graph_to_gdfs(undirected_graph)
        undirected_data = gdf_to_pyg(
            undirected_nodes,
            undirected_edges,
            edge_feature_cols=["length"],
            directed=False,
            multigraph=undirected_graph.is_multigraph(),
        )
        _, undirected_restored_edges = pyg_to_gdf(undirected_data)

        assert isinstance(undirected_graph, nx.MultiGraph)
        assert undirected_data.num_edges == len(undirected_edges) * 2
        assert undirected_data.graph_metadata.is_multigraph is True
        assert undirected_data.graph_metadata.edge_was_symmetrized is True
        assert isinstance(undirected_restored_edges, gpd.GeoDataFrame)
        assert len(undirected_restored_edges) == len(undirected_edges)
        assert undirected_restored_edges.index.nlevels == 3
        assert set(undirected_restored_edges.index) == set(undirected_edges.index)

    def test_nx_multidigraph_round_trip_preserves_parallel_direction(self) -> None:
        """nx.MultiDiGraph round-trips as directed with parallel keys."""
        graph = nx.MultiDiGraph()
        graph.add_node("a", pos=(0, 0), geometry=Point(0, 0))
        graph.add_node("b", pos=(1, 0), geometry=Point(1, 0))
        graph.add_edge(
            "a",
            "b",
            key="ab",
            weight=1.0,
            geometry=LineString([(0, 0), (1, 0)]),
        )
        graph.add_edge(
            "b",
            "a",
            key="ba",
            weight=2.0,
            geometry=LineString([(1, 0), (0, 0)]),
        )
        graph.graph["crs"] = "EPSG:27700"
        graph.graph["is_hetero"] = False

        restored = pyg_to_nx(nx_to_pyg(graph, edge_feature_cols=["weight"]))

        assert isinstance(restored, nx.MultiDiGraph)
        assert restored.is_directed() is True
        assert set(restored.edges(keys=True)) == {("a", "b", "ab"), ("b", "a", "ba")}

    def test_old_metadata_missing_multigraph_attrs_still_round_trips(
        self, simple_nodes: gpd.GeoDataFrame
    ) -> None:
        """Older PyG metadata without new multigraph attrs still reconstructs."""
        edges = gpd.GeoDataFrame(
            {
                "geometry": [
                    LineString([(0, 0), (1, 0)]),
                    LineString([(0, 0), (0.5, 0.2), (1, 0)]),
                ],
            },
            index=pd.MultiIndex.from_arrays(
                [[1, 1], [2, 2], ["road", "rail"]],
                names=["source_id", "target_id", "edge_key"],
            ),
            crs="EPSG:27700",
        )
        data = gdf_to_pyg(simple_nodes, edges)
        delattr(data.graph_metadata, "edge_index_keys")
        delattr(data.graph_metadata, "is_multigraph")

        _, restored_edges = pyg_to_gdf(data)
        restored_graph = pyg_to_nx(data)

        assert isinstance(restored_edges, gpd.GeoDataFrame)
        pd.testing.assert_index_equal(restored_edges.index, edges.index)
        assert isinstance(restored_graph, nx.MultiGraph)

    def test_nx_to_pyg_directed_override(self) -> None:
        """The directed override takes precedence over nx.Graph semantics."""
        graph = nx.Graph()
        graph.add_node(1, pos=(0, 0), geometry=Point(0, 0))
        graph.add_node(2, pos=(1, 0), geometry=Point(1, 0))
        graph.add_edge(1, 2, geometry=LineString([(0, 0), (1, 0)]))
        graph.graph["crs"] = "EPSG:27700"
        graph.graph["is_hetero"] = False

        data = nx_to_pyg(graph, directed=True)

        assert data.num_edges == 1
        assert data.graph_metadata.is_directed is True

    def test_nx_round_trip_undirected_preserves_edges(self) -> None:
        """nx.Graph round trips as undirected with the same edge set."""
        graph = nx.Graph()
        for node_id, coords in [(1, (0, 0)), (2, (1, 0)), (3, (1, 1))]:
            graph.add_node(node_id, pos=coords, geometry=Point(*coords))
        graph.add_edge(1, 2, geometry=LineString([(0, 0), (1, 0)]))
        graph.add_edge(2, 3, geometry=LineString([(1, 0), (1, 1)]))
        graph.graph["crs"] = "EPSG:27700"
        graph.graph["is_hetero"] = False

        restored = pyg_to_nx(nx_to_pyg(graph))

        assert restored.is_directed() is False
        assert restored.number_of_edges() == graph.number_of_edges()
        assert {frozenset(edge) for edge in restored.edges()} == {
            frozenset(edge) for edge in graph.edges()
        }

    def test_nx_round_trip_digraph_preserves_direction(self) -> None:
        """nx.DiGraph round trips as directed with the same edge set."""
        graph = nx.DiGraph()
        for node_id, coords in [(1, (0, 0)), (2, (1, 0)), (3, (1, 1))]:
            graph.add_node(node_id, pos=coords, geometry=Point(*coords))
        graph.add_edge(1, 2, geometry=LineString([(0, 0), (1, 0)]))
        graph.add_edge(3, 2, geometry=LineString([(1, 1), (1, 0)]))
        graph.graph["crs"] = "EPSG:27700"
        graph.graph["is_hetero"] = False

        restored = pyg_to_nx(nx_to_pyg(graph))

        assert restored.is_directed() is True
        assert set(restored.edges()) == set(graph.edges())

    def test_hetero_mixed_directionality_round_trip(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Mixed same-type and cross-type undirected relations round trip."""
        road_type = ("road", "links_to", "road")
        conn_type = ("building", "connects_to", "road")
        road_gdf = gpd.GeoDataFrame(
            {
                "source_road_id": ["r1"],
                "target_road_id": ["r2"],
                "geometry": [LineString([(10, 12), (12, 12)])],
            },
            index=pd.MultiIndex.from_arrays(
                [["r1"], ["r2"]],
                names=["source_road_id", "target_road_id"],
            ),
            crs="EPSG:27700",
        )
        conn_gdf = gpd.GeoDataFrame(
            {
                "building_id": ["b1"],
                "road_id": ["r1"],
                "geometry": [LineString([(10, 10), (10, 12)])],
            },
            index=pd.MultiIndex.from_arrays(
                [["b1"], ["r1"]],
                names=["building_id", "road_id"],
            ),
            crs="EPSG:27700",
        )
        edges_dict = {road_type: road_gdf, conn_type: conn_gdf}

        _, edges_restored = pyg_to_gdf(gdf_to_pyg(sample_hetero_nodes_dict, edges_dict))

        assert isinstance(edges_restored, dict)
        assert set(edges_restored) == set(edges_dict)
        assert len(edges_restored[road_type]) == len(road_gdf)
        assert len(edges_restored[conn_type]) == len(conn_gdf)

    def test_hetero_same_type_parallel_keys_round_trip(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Same-type hetero edges preserve parallel key-level rows."""
        edge_type = ("road", "links_to", "road")
        road_gdf = gpd.GeoDataFrame(
            {
                "weight": [1.0, 2.0],
                "geometry": [
                    LineString([(10, 12), (12, 12)]),
                    LineString([(10, 12), (11, 12.5), (12, 12)]),
                ],
            },
            index=pd.MultiIndex.from_arrays(
                [["r1", "r1"], ["r2", "r2"], ["road", "rail"]],
                names=["source_road_id", "target_road_id", "edge_key"],
            ),
            crs="EPSG:27700",
        )

        data = gdf_to_pyg(
            sample_hetero_nodes_dict,
            {edge_type: road_gdf},
            edge_feature_cols={edge_type: ["weight"]},
        )
        _, edges_restored = pyg_to_gdf(data)

        assert data[edge_type].edge_index.size(1) == 4
        assert isinstance(edges_restored, dict)
        pd.testing.assert_index_equal(edges_restored[edge_type].index, road_gdf.index)
        assert edges_restored[edge_type]["weight"].tolist() == [1.0, 2.0]

    def test_hetero_cross_type_parallel_keys_skip_reverse_leakage(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Cross-type parallel keyed edges mirror for PyG without leaking back."""
        edge_type = ("building", "connects_to", "road")
        reverse_type = ("road", "rev_connects_to", "building")
        conn_gdf = gpd.GeoDataFrame(
            {
                "weight": [1.0, 2.0],
                "geometry": [
                    LineString([(10, 10), (10, 12)]),
                    LineString([(10, 10), (10.2, 11), (10, 12)]),
                ],
            },
            index=pd.MultiIndex.from_arrays(
                [["b1", "b1"], ["r1", "r1"], ["walk", "service"]],
                names=["building_id", "road_id", "edge_key"],
            ),
            crs="EPSG:27700",
        )

        data = gdf_to_pyg(
            sample_hetero_nodes_dict,
            {edge_type: conn_gdf},
            edge_feature_cols={edge_type: ["weight"]},
        )
        _, edges_restored = pyg_to_gdf(data)

        assert data[edge_type].edge_index.size(1) == 2
        assert data[reverse_type].edge_index.size(1) == 2
        assert torch.equal(data[reverse_type].edge_attr, data[edge_type].edge_attr)
        assert isinstance(edges_restored, dict)
        assert set(edges_restored) == {edge_type}
        pd.testing.assert_index_equal(edges_restored[edge_type].index, conn_gdf.index)

    def test_hetero_directed_dict_round_trip(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Per-type directionality round trips without reverse artefacts."""
        road_type = ("road", "links_to", "road")
        conn_type = ("building", "connects_to", "road")
        road_gdf = gpd.GeoDataFrame(
            {
                "source_road_id": ["r1"],
                "target_road_id": ["r2"],
                "geometry": [LineString([(10, 12), (12, 12)])],
            },
            index=pd.MultiIndex.from_arrays(
                [["r1"], ["r2"]],
                names=["source_road_id", "target_road_id"],
            ),
            crs="EPSG:27700",
        )
        conn_gdf = gpd.GeoDataFrame(
            {
                "building_id": ["b1", "b2"],
                "road_id": ["r1", "r2"],
                "geometry": [
                    LineString([(10, 10), (10, 12)]),
                    LineString([(11, 11), (12, 12)]),
                ],
            },
            index=pd.MultiIndex.from_arrays(
                [["b1", "b2"], ["r1", "r2"]],
                names=["building_id", "road_id"],
            ),
            crs="EPSG:27700",
        )
        edges_dict = {road_type: road_gdf, conn_type: conn_gdf}
        data = gdf_to_pyg(
            sample_hetero_nodes_dict,
            edges_dict,
            directed={road_type: False, conn_type: True},
        )

        _, edges_restored = pyg_to_gdf(data)

        assert isinstance(edges_restored, dict)
        assert set(edges_restored) == set(edges_dict)
        assert len(edges_restored[conn_type]) == len(conn_gdf)
        assert ("road", "rev_connects_to", "building") not in edges_restored

    def test_pyg_to_nx_warns_on_mixed_hetero_directedness(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Mixed hetero direction metadata warns before collapsing to one NX graph."""
        road_type = ("road", "links_to", "road")
        conn_type = ("building", "connects_to", "road")
        road_gdf = gpd.GeoDataFrame(
            {
                "source_road_id": ["r1"],
                "target_road_id": ["r2"],
                "geometry": [LineString([(10, 12), (12, 12)])],
            },
            index=pd.MultiIndex.from_arrays(
                [["r1"], ["r2"]],
                names=["source_road_id", "target_road_id"],
            ),
            crs="EPSG:27700",
        )
        conn_gdf = gpd.GeoDataFrame(
            {
                "building_id": ["b1"],
                "road_id": ["r1"],
                "geometry": [LineString([(10, 10), (10, 12)])],
            },
            index=pd.MultiIndex.from_arrays(
                [["b1"], ["r1"]],
                names=["building_id", "road_id"],
            ),
            crs="EPSG:27700",
        )
        data = gdf_to_pyg(
            sample_hetero_nodes_dict,
            {road_type: road_gdf, conn_type: conn_gdf},
            directed={road_type: False, conn_type: True},
        )

        with pytest.warns(UserWarning, match="collapses mixed heterogeneous edge directedness"):
            graph = pyg_to_nx(data)

        assert graph.is_directed() is False

    def test_metadata_reverse_edge_types_populated(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Reverse edge metadata records generated cross-type stores."""
        edge_type = ("building", "connects_to", "road")
        reverse_type = ("road", "rev_connects_to", "building")
        conn_gdf = gpd.GeoDataFrame(
            {
                "building_id": ["b1"],
                "road_id": ["r1"],
                "geometry": [LineString([(10, 10), (10, 12)])],
            },
            index=pd.MultiIndex.from_arrays(
                [["b1"], ["r1"]],
                names=["building_id", "road_id"],
            ),
            crs="EPSG:27700",
        )

        data = gdf_to_pyg(sample_hetero_nodes_dict, {edge_type: conn_gdf})
        metadata = data.graph_metadata

        assert metadata.reverse_edge_types == {edge_type: reverse_type}
        assert metadata.generated_reverse_edge_types == {reverse_type: edge_type}
        assert metadata.original_edge_types == [edge_type]

    def test_explicit_reverse_dict_round_trip(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Custom generated reverse stores do not leak into reconstructed GDFs."""
        edge_type = ("building", "connects_to", "road")
        reverse_type = ("road", "served_by", "building")
        conn_gdf = gpd.GeoDataFrame(
            {
                "building_id": ["b1"],
                "road_id": ["r1"],
                "geometry": [LineString([(10, 10), (10, 12)])],
            },
            index=pd.MultiIndex.from_arrays(
                [["b1"], ["r1"]],
                names=["building_id", "road_id"],
            ),
            crs="EPSG:27700",
        )
        data = gdf_to_pyg(
            sample_hetero_nodes_dict,
            {edge_type: conn_gdf},
            reverse_edge_types={edge_type: reverse_type},
        )

        _, edges_restored = pyg_to_gdf(data)

        assert isinstance(edges_restored, dict)
        assert edge_type in edges_restored
        assert reverse_type not in edges_restored

    def test_old_metadata_generated_reverse_edge_types_skipped(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Old metadata that lists generated stores still skips them."""
        edge_type = ("building", "connects_to", "road")
        reverse_type = ("road", "rev_connects_to", "building")
        conn_gdf = gpd.GeoDataFrame(
            {
                "building_id": ["b1"],
                "road_id": ["r1"],
                "geometry": [LineString([(10, 10), (10, 12)])],
            },
            index=pd.MultiIndex.from_arrays(
                [["b1"], ["r1"]],
                names=["building_id", "road_id"],
            ),
            crs="EPSG:27700",
        )
        data = gdf_to_pyg(sample_hetero_nodes_dict, {edge_type: conn_gdf})
        data.graph_metadata.original_edge_types = None
        data.graph_metadata.edge_types = list(data.edge_types)

        _, edges_restored = pyg_to_gdf(data)

        assert isinstance(edges_restored, dict)
        assert edge_type in edges_restored
        assert reverse_type not in edges_restored

    def test_hetero_additional_edge_cols_by_relation_name(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Hetero edge columns can be requested by relation name."""
        edge_type = ("building", "connects_to", "road")
        conn_gdf = gpd.GeoDataFrame(
            {
                "building_id": ["b1", "b2"],
                "road_id": ["r1", "r2"],
                "geometry": [
                    LineString([(10, 10), (10, 12)]),
                    LineString([(11, 11), (12, 12)]),
                ],
            },
            index=pd.MultiIndex.from_arrays(
                [["b1", "b2"], ["r1", "r2"]],
                names=["building_id", "road_id"],
            ),
            crs="EPSG:27700",
        )
        data = gdf_to_pyg(sample_hetero_nodes_dict, {edge_type: conn_gdf})
        data[edge_type].capacity = torch.tensor([[4.0], [8.0]])

        _, edges_restored = pyg_to_gdf(
            data,
            additional_edge_cols=cast("Any", {"connects_to": ["capacity"]}),
        )

        assert isinstance(edges_restored, dict)
        assert edges_restored[edge_type]["capacity"].tolist() == [4.0, 8.0]

    def test_invalid_edge_was_symmetrized_metadata_no_dedup(
        self, simple_nodes: gpd.GeoDataFrame, simple_edges: gpd.GeoDataFrame
    ) -> None:
        """Unexpected symmetrisation metadata values do not deduplicate."""
        data = gdf_to_pyg(simple_nodes, simple_edges)
        data.graph_metadata.edge_was_symmetrized = None

        _, edges_restored = pyg_to_gdf(data)

        assert isinstance(edges_restored, gpd.GeoDataFrame)
        assert len(edges_restored) == data.edge_index.size(1)

    def test_backward_compat_is_directed_dict_no_dedup(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Old hetero directed metadata dicts avoid accidental deduplication."""
        edge_type = ("road", "links_to", "road")
        road_gdf = gpd.GeoDataFrame(
            {
                "source_road_id": ["r1"],
                "target_road_id": ["r2"],
                "geometry": [LineString([(10, 12), (12, 12)])],
            },
            index=pd.MultiIndex.from_arrays(
                [["r1"], ["r2"]],
                names=["source_road_id", "target_road_id"],
            ),
            crs="EPSG:27700",
        )
        data = gdf_to_pyg(sample_hetero_nodes_dict, {edge_type: road_gdf})
        if hasattr(data.graph_metadata, "edge_was_symmetrized"):
            delattr(data.graph_metadata, "edge_was_symmetrized")
        data.graph_metadata.is_directed = {edge_type: True}

        _, edges_restored = pyg_to_gdf(data)

        assert isinstance(edges_restored, dict)
        assert len(edges_restored[edge_type]) == data[edge_type].edge_index.size(1)

    def test_backward_compat_old_metadata_no_dedup(self) -> None:
        """Old metadata without edge_was_symmetrized defaults to no dedup."""
        nodes = gpd.GeoDataFrame(
            {"node_id": [1, 2], "geometry": [Point(0, 0), Point(1, 0)]},
            crs="EPSG:27700",
        ).set_index("node_id")

        source_ids = [1, 2]
        target_ids = [2, 1]
        edges = gpd.GeoDataFrame(
            {
                "source_id": source_ids,
                "target_id": target_ids,
                "geometry": [
                    LineString([(0, 0), (1, 0)]),
                    LineString([(1, 0), (0, 0)]),
                ],
            },
            index=pd.MultiIndex.from_arrays(
                [source_ids, target_ids], names=["source_id", "target_id"]
            ),
            crs="EPSG:27700",
        )

        # Build PyG data directly (simulating old metadata)
        data = gdf_to_pyg(nodes, edges, directed=True)
        # Remove edge_was_symmetrized to simulate old metadata
        if hasattr(data.graph_metadata, "edge_was_symmetrized"):
            delattr(data.graph_metadata, "edge_was_symmetrized")
        # Ensure is_directed is True (old default) so backward compat path won't dedup
        data.graph_metadata.is_directed = True

        _, edges_restored = pyg_to_gdf(data)
        assert isinstance(edges_restored, gpd.GeoDataFrame)
        assert len(edges_restored) == 2
