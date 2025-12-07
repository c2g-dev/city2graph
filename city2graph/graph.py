"""
Module for creating heterogeneous graph representations of urban environments.

This module provides comprehensive functionality for converting spatial data
(GeoDataFrames and NetworkX objects) into PyTorch Geometric Data and HeteroData objects,
supporting both homogeneous and heterogeneous graphs. It handles the complex mapping between
geographical coordinates, node/edge features, and the tensor representations
required by graph neural networks.

The module serves as a bridge between geospatial data analysis tools and deep
learning frameworks, enabling seamless integration of spatial urban data with
Graph Neural Networks (GNNs) for tasks of GeoAI such as urban modeling, traffic prediction,
and spatial analysis.
"""

# Future annotations for type hints
from __future__ import annotations

# Standard library imports
import functools
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

if TYPE_CHECKING:
    from collections.abc import Callable

# Third-party imports
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csgraph
from shapely import wkb
from shapely.geometry import LineString

# Internal imports from city2graph package
from city2graph.base import BaseGraphConverter
from city2graph.base import GraphMetadata
from city2graph.utils import gdf_to_nx
from city2graph.utils import nx_to_gdf
from city2graph.utils import validate_nx

# PyTorch Geometric imports with availability checking
try:
    import torch
    from torch_geometric.data import Data
    from torch_geometric.data import HeteroData

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - makes life easier for docs build.
    TORCH_AVAILABLE = False

    # Create stubs for documentation and fallback functionality
    if TYPE_CHECKING:
        from torch_geometric.data import Data
        from torch_geometric.data import HeteroData
    else:
        torch = None

        class HeteroData:
            """Fallback stub when torch is unavailable."""

        class Data:
            """Fallback stub when torch is unavailable."""


logger = logging.getLogger(__name__)

__all__ = [
    "add_metapaths",
    "add_metapaths_by_weight",
    "gdf_to_pyg",
    "is_torch_available",
    "nx_to_pyg",
    "pyg_to_gdf",
    "pyg_to_nx",
    "validate_pyg",
]


# Constants for error messages
TORCH_ERROR_MSG = "PyTorch and PyTorch Geometric required for graph conversion functionality."
DEVICE_ERROR_MSG = "Device must be 'cuda', 'cpu', a torch.device object, or None"
GRAPH_NO_NODES_MSG = "Graph has no nodes"


# ============================================================================
# GRAPH CONVERSION FUNCTIONS
# ============================================================================


class PyGConverter(BaseGraphConverter):
    """
    Graph converter for PyTorch Geometric.

    Handles conversion between GeoDataFrames and PyG Data/HeteroData objects.
    This converter handles feature extraction, tensor creation, and manages
    the bidirectional conversion while preserving spatial and attribute information.

    Parameters
    ----------
    node_feature_cols : dict[str, list[str]] or list[str], optional
        Column names to use as node features.
    node_label_cols : dict[str, list[str]] or list[str], optional
        Column names to use as node labels.
    edge_feature_cols : dict[str, list[str]] or list[str], optional
        Column names to use as edge features.
    device : str or torch.device, optional
        Target device for tensor placement.
    dtype : torch.dtype, optional
        Data type for float tensors.
    keep_geom : bool, default True
        Whether to preserve geometry information during conversion.
    """

    def __init__(
        self,
        node_feature_cols: dict[str, list[str]] | list[str] | None = None,
        node_label_cols: dict[str, list[str]] | list[str] | None = None,
        edge_feature_cols: dict[str, list[str]] | list[str] | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        keep_geom: bool = True,
    ) -> None:
        """
        Initialize PyGConverter.

        Configure feature columns, label columns, and tensor properties.

        Parameters
        ----------
        node_feature_cols : dict[str, list[str]] or list[str], optional
            Column names to use as node features.
        node_label_cols : dict[str, list[str]] or list[str], optional
            Column names to use as node labels.
        edge_feature_cols : dict[str, list[str]] or list[str], optional
            Column names to use as edge features.
        device : str or torch.device, optional
            Target device for tensor placement.
        dtype : torch.dtype, optional
            Data type for float tensors.
        keep_geom : bool, default True
            Whether to preserve geometry information during conversion.
            If True, original geometries are serialized and stored in metadata.
            If False, geometries are reconstructed from node positions during
            conversion back to GeoDataFrames.
        """
        super().__init__(keep_geom=keep_geom)
        self.node_feature_cols = node_feature_cols
        self.node_label_cols = node_label_cols
        self.edge_feature_cols = edge_feature_cols
        self.device = device
        self.dtype = dtype

    def gdf_to_pyg(
        self,
        nodes: dict[str, gpd.GeoDataFrame] | gpd.GeoDataFrame,
        edges: dict[tuple[str, str, str], gpd.GeoDataFrame] | gpd.GeoDataFrame | None = None,
    ) -> Data | HeteroData:
        """
        Convert GeoDataFrames to PyG object.

        This method converts spatial data represented as GeoDataFrames into PyTorch
        Geometric format with automatic validation.

        Parameters
        ----------
        nodes : dict[str, gpd.GeoDataFrame] or gpd.GeoDataFrame
            Node data as GeoDataFrame or dictionary of GeoDataFrames.
        edges : dict[tuple[str, str, str], gpd.GeoDataFrame] or gpd.GeoDataFrame, optional
            Edge data as GeoDataFrame or dictionary of GeoDataFrames.

        Returns
        -------
        Data or HeteroData
            PyTorch Geometric data object.
        """
        data = self.convert(nodes, edges)
        validate_pyg(data)
        return data

    def pyg_to_gdf(
        self,
        data: Data | HeteroData,
        _node_types: str | list[str] | None = None,
        _edge_types: str | list[tuple[str, str, str]] | None = None,
    ) -> (
        tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]]
        | tuple[gpd.GeoDataFrame | None, gpd.GeoDataFrame | None]
    ):
        """
        Convert PyG object to GeoDataFrames.

        This method reconstructs GeoDataFrames from PyTorch Geometric data using
        stored metadata for proper structure and attributes.

        Parameters
        ----------
        data : Data or HeteroData
            PyTorch Geometric data object.
        _node_types : str or list[str], optional
            Node types to extract (unused, kept for API compatibility).
        _edge_types : str or list[tuple[str, str, str]], optional
            Edge types to extract (unused, kept for API compatibility).

        Returns
        -------
        tuple
            Either homogeneous (nodes_gdf, edges_gdf) or heterogeneous
            (dict of node gdfs, dict of edge gdfs) tuple.
        """
        return self.reconstruct(data)

    def _validate_homogeneous_columns(
        self,
    ) -> tuple[list[str] | None, list[str] | None, list[str] | None]:
        """
        Validate and extract column specifications for homogeneous graphs.

        This method ensures that column specifications provided to the converter
        are appropriate for homogeneous graph conversion.

        Returns
        -------
        tuple
            Validated (node_feature_cols, node_label_cols, edge_feature_cols).

        Raises
        ------
        TypeError
            If column specifications are not lists for homogeneous graphs.
        """
        if isinstance(self.node_feature_cols, list) or self.node_feature_cols is None:
            node_feature_cols_homo = self.node_feature_cols
        else:
            msg = "node_feature_cols must be a list for homogeneous graphs"
            raise TypeError(msg)

        if isinstance(self.node_label_cols, list) or self.node_label_cols is None:
            node_label_cols_homo = self.node_label_cols
        else:
            msg = "node_label_cols must be a list for homogeneous graphs"
            raise TypeError(msg)

        if isinstance(self.edge_feature_cols, list) or self.edge_feature_cols is None:
            edge_feature_cols_homo = self.edge_feature_cols
        else:
            msg = "edge_feature_cols must be a list for homogeneous graphs"
            raise TypeError(msg)

        return node_feature_cols_homo, node_label_cols_homo, edge_feature_cols_homo

    def _convert_homogeneous(
        self,
        nodes: gpd.GeoDataFrame | None,
        edges: gpd.GeoDataFrame | None,
    ) -> Data:
        """
        Convert homogeneous GeoDataFrames to PyG Data.

        Extended summary for homogeneous graph conversion.

        Parameters
        ----------
        nodes : gpd.GeoDataFrame, optional
            Node data as GeoDataFrame.
        edges : gpd.GeoDataFrame, optional
            Edge data as GeoDataFrame.

        Returns
        -------
        Data
            PyTorch Geometric Data object.
        """
        if nodes is None:
            msg = "Nodes GeoDataFrame is required for PyG conversion"
            raise ValueError(msg)

        # Validate column types
        node_feature_cols_homo, node_label_cols_homo, edge_feature_cols_homo = (
            self._validate_homogeneous_columns()
        )

        device = _get_device(self.device)

        # Node processing
        id_mapping, id_col_name, original_ids = self._create_node_id_mapping(nodes)

        x = self._create_features(nodes, node_feature_cols_homo)
        pos = self._create_node_positions(nodes)

        # Handle labels
        y = None
        if node_label_cols_homo:
            y = self._create_features(nodes, node_label_cols_homo)

        # Edge processing
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        edge_attr = torch.empty((0, 0), dtype=self.dtype or torch.float32, device=device)

        if edges is not None and not edges.empty:
            edge_pairs = self._create_edge_indices(
                edges,
                id_mapping,
                id_mapping,
            )
            if edge_pairs:
                edge_index = torch.tensor(
                    np.array(edge_pairs).T,
                    dtype=torch.long,
                    device=device,
                )
            edge_attr = self._create_features(edges, edge_feature_cols_homo)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos)

        # Store metadata
        metadata = self._create_homogeneous_metadata(
            nodes,
            edges,
            id_mapping,
            id_col_name,
            original_ids,
            node_feature_cols_homo,
            node_label_cols_homo,
            edge_feature_cols_homo,
        )

        data.graph_metadata = metadata
        return data

    def _create_homogeneous_metadata(
        self,
        nodes: gpd.GeoDataFrame,
        edges: gpd.GeoDataFrame | None,
        id_mapping: dict[str | int, int],
        id_col_name: str,
        original_ids: list[str | int],
        node_feature_cols: list[str] | None,
        node_label_cols: list[str] | None,
        edge_feature_cols: list[str] | None,
    ) -> GraphMetadata:
        """
        Create metadata for homogeneous graph.

        This method populates a GraphMetadata object with all necessary information
        for reconstructing homogeneous GeoDataFrames from PyG Data objects.

        Parameters
        ----------
        nodes : gpd.GeoDataFrame
            Node GeoDataFrame.
        edges : gpd.GeoDataFrame or None
            Edge GeoDataFrame.
        id_mapping : dict
            Node ID mapping.
        id_col_name : str
            ID column name.
        original_ids : list
            Original node IDs.
        node_feature_cols : list or None
            Node feature columns.
        node_label_cols : list or None
            Node label columns.
        edge_feature_cols : list or None
            Edge feature columns.

        Returns
        -------
        GraphMetadata
            Populated metadata object.
        """
        metadata = GraphMetadata(is_hetero=False)
        metadata.node_mappings = {
            "default": {
                "mapping": id_mapping,
                "id_col": id_col_name,
                "original_ids": original_ids,
            },
        }
        metadata.node_feature_cols = node_feature_cols or []
        metadata.node_label_cols = node_label_cols or []
        metadata.edge_feature_cols = edge_feature_cols or []

        # Store index names and values for preservation
        metadata.node_index_names = nodes.index.names if hasattr(nodes.index, "names") else None
        if edges is not None and hasattr(edges.index, "names"):
            metadata.edge_index_names = edges.index.names

            # Store original edge index values for reconstruction
            metadata.edge_index_values = [
                edges.index.get_level_values(i).tolist() for i in range(edges.index.nlevels)
            ]
        else:
            metadata.edge_index_names = None
            metadata.edge_index_values = None

        # Set CRS
        if hasattr(nodes, "crs") and nodes.crs:
            metadata.crs = nodes.crs

        # Serialize and store geometries for exact reconstruction if keep_geom is True
        if self.keep_geom:
            metadata.node_geometries = self._serialize_geometries(nodes)
            if edges is not None and not edges.empty:
                metadata.edge_geometries = self._serialize_geometries(edges)

        return metadata

    def _convert_heterogeneous(
        self,
        nodes: dict[str, gpd.GeoDataFrame] | None,
        edges: dict[tuple[str, str, str], gpd.GeoDataFrame] | None,
    ) -> HeteroData:
        """
        Convert heterogeneous GeoDataFrames to PyG HeteroData.

        Extended summary for heterogeneous graph conversion.

        Parameters
        ----------
        nodes : dict[str, gpd.GeoDataFrame], optional
            Node data as dictionary of GeoDataFrames.
        edges : dict[tuple[str, str, str], gpd.GeoDataFrame], optional
            Edge data as dictionary of GeoDataFrames.

        Returns
        -------
        HeteroData
            PyTorch Geometric HeteroData object.
        """
        if nodes is None:
            msg = "Nodes dictionary is required for PyG conversion"
            raise ValueError(msg)

        # Type narrowing for heterogeneous graphs
        if isinstance(self.node_feature_cols, dict) or self.node_feature_cols is None:
            node_feature_cols_hetero = self.node_feature_cols
        else:
            msg = "node_feature_cols must be a dict for heterogeneous graphs"
            raise TypeError(msg)

        if isinstance(self.node_label_cols, dict) or self.node_label_cols is None:
            node_label_cols_hetero = self.node_label_cols
        else:
            msg = "node_label_cols must be a dict for heterogeneous graphs"
            raise TypeError(msg)

        if isinstance(self.edge_feature_cols, dict) or self.edge_feature_cols is None:
            edge_feature_cols_hetero = self.edge_feature_cols
        else:
            msg = "edge_feature_cols must be a dict for heterogeneous graphs"
            raise TypeError(msg)

        data = HeteroData()

        # Default empty dicts
        edges_dict = edges or {}

        # Process nodes and get mappings
        node_mappings = self._process_hetero_nodes(
            data,
            nodes,
            node_feature_cols_hetero,
            node_label_cols_hetero,
        )

        # Process edges
        self._process_hetero_edges(
            data,
            edges_dict,
            node_mappings,
            edge_feature_cols_hetero,
        )

        # Store metadata
        self._store_hetero_metadata(
            data,
            node_mappings,
            nodes,
            edges_dict,
            node_feature_cols_hetero,
            node_label_cols_hetero,
            edge_feature_cols_hetero,
        )

        return data

    def _process_hetero_nodes(
        self,
        data: HeteroData,
        nodes_dict: dict[str, gpd.GeoDataFrame],
        node_feature_cols: dict[str, list[str]] | None,
        node_label_cols: dict[str, list[str]] | None,
    ) -> dict[str, dict[str, dict[str | int, int] | str | list[str | int]]]:
        """
        Process all node types for heterogeneous graph.

        Extended summary of node processing for heterogeneous graphs.

        Parameters
        ----------
        data : HeteroData
            HeteroData object to populate with node information.
        nodes_dict : dict[str, gpd.GeoDataFrame]
            Dictionary mapping node types to GeoDataFrames.
        node_feature_cols : dict[str, list[str]], optional
            Dictionary mapping node types to feature column names.
        node_label_cols : dict[str, list[str]], optional
            Dictionary mapping node types to label column names.

        Returns
        -------
        dict
            Dictionary mapping node types to mapping information.
        """
        node_mappings: dict[str, dict[str, dict[str | int, int] | str | list[str | int]]] = {}

        for node_type, node_gdf in nodes_dict.items():
            id_mapping, id_col_name, original_ids = self._create_node_id_mapping(node_gdf)

            # Store mapping with metadata in unified structure
            node_mappings[node_type] = {
                "mapping": id_mapping,
                "id_col": id_col_name,
                "original_ids": original_ids,
            }

            # Features
            feature_cols = node_feature_cols.get(node_type) if node_feature_cols else None
            data[node_type].x = self._create_features(node_gdf, feature_cols)

            # Positions
            data[node_type].pos = self._create_node_positions(node_gdf)

            # Labels
            label_cols = node_label_cols.get(node_type) if node_label_cols else None
            if label_cols:
                data[node_type].y = self._create_features(node_gdf, label_cols)

        return node_mappings

    def _process_hetero_edges(
        self,
        data: HeteroData,
        edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
        node_mappings: dict[str, dict[str, dict[str | int, int] | str | list[str | int]]],
        edge_feature_cols: dict[str, list[str]] | None,
    ) -> None:
        """
        Process all edge types for heterogeneous graph.

        Extended summary of edge processing for heterogeneous graphs.

        Parameters
        ----------
        data : HeteroData
            HeteroData object to populate with edge information.
        edges_dict : dict[tuple[str, str, str], gpd.GeoDataFrame]
            Dictionary mapping edge types to GeoDataFrames.
        node_mappings : dict
            Dictionary containing node mapping information.
        edge_feature_cols : dict[str, list[str]], optional
            Dictionary mapping relation types to feature column names.
        """
        device = _get_device(self.device)

        for edge_type, edge_gdf in edges_dict.items():
            # Extract source, relation, and destination types from edge_type tuple
            src_type, rel_type, dst_type = edge_type

            # Get the mapping dictionaries (not the full metadata)
            # The type system guarantees these are dictionaries based on _process_hetero_nodes
            src_mapping_raw = node_mappings[src_type]["mapping"]
            dst_mapping_raw = node_mappings[dst_type]["mapping"]

            # Type assertion for mypy - these are guaranteed to be dicts by construction
            assert isinstance(src_mapping_raw, dict), f"Expected dict mapping for {src_type}"
            assert isinstance(dst_mapping_raw, dict), f"Expected dict mapping for {dst_type}"

            src_mapping: dict[str | int, int] = src_mapping_raw
            dst_mapping: dict[str | int, int] = dst_mapping_raw

            if edge_gdf is not None and not edge_gdf.empty:
                edge_pairs = self._create_edge_indices(
                    edge_gdf,
                    src_mapping,
                    dst_mapping,
                )
                edge_index = (
                    torch.tensor(np.array(edge_pairs).T, dtype=torch.long, device=device)
                    if edge_pairs
                    else torch.zeros((2, 0), dtype=torch.long, device=device)
                )
                data[edge_type].edge_index = edge_index

                feature_cols = edge_feature_cols.get(rel_type) if edge_feature_cols else None
                data[edge_type].edge_attr = self._create_features(edge_gdf, feature_cols)
            else:
                data[edge_type].edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                data[edge_type].edge_attr = torch.empty(
                    (0, 0),
                    dtype=self.dtype or torch.float32,
                    device=device,
                )

    def _store_hetero_metadata(
        self,
        data: HeteroData,
        node_mappings: dict[str, dict[str, dict[str | int, int] | str | list[str | int]]],
        nodes_dict: dict[str, gpd.GeoDataFrame],
        edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
        node_feature_cols: dict[str, list[str]] | None,
        node_label_cols: dict[str, list[str]] | None,
        edge_feature_cols: dict[str, list[str]] | None,
    ) -> None:
        """
        Store metadata for heterogeneous graph.

        Extended summary of metadata storage for heterogeneous graphs.

        Parameters
        ----------
        data : HeteroData
            HeteroData object to attach metadata to.
        node_mappings : dict
            Dictionary containing node mapping information.
        nodes_dict : dict[str, gpd.GeoDataFrame]
            Dictionary mapping node types to GeoDataFrames.
        edges_dict : dict[tuple[str, str, str], gpd.GeoDataFrame]
            Dictionary mapping edge types to GeoDataFrames.
        node_feature_cols : dict[str, list[str]], optional
            Dictionary mapping node types to feature column names.
        node_label_cols : dict[str, list[str]], optional
            Dictionary mapping node types to label column names.
        edge_feature_cols : dict[str, list[str]], optional
            Dictionary mapping relation types to feature column names.
        """
        # Store mappings and column metadata
        metadata = GraphMetadata(is_hetero=True)
        metadata.node_types = list(nodes_dict.keys())
        metadata.edge_types = list(edges_dict.keys())
        metadata.node_mappings = node_mappings
        metadata.node_feature_cols = node_feature_cols or {}
        metadata.node_label_cols = node_label_cols or {}
        metadata.edge_feature_cols = edge_feature_cols or {}

        # Store index names for reconstruction
        metadata.node_index_names = {}
        for node_type, node_gdf in nodes_dict.items():
            if hasattr(node_gdf.index, "names"):
                metadata.node_index_names[node_type] = node_gdf.index.names

        # Store edge index names and values for reconstruction
        metadata.edge_index_names = {}
        metadata.edge_index_values = {}
        for edge_type, edge_gdf in edges_dict.items():
            if edge_gdf is not None and hasattr(edge_gdf.index, "names"):
                # Store edge index names
                metadata.edge_index_names[edge_type] = edge_gdf.index.names

                # Store original edge index values for reconstruction
                metadata.edge_index_values[edge_type] = [
                    edge_gdf.index.get_level_values(i).tolist()
                    for i in range(edge_gdf.index.nlevels)
                ]

        # Set CRS
        crs_values = [gdf.crs for gdf in nodes_dict.values() if hasattr(gdf, "crs") and gdf.crs]
        if crs_values and all(crs == crs_values[0] for crs in crs_values):
            metadata.crs = crs_values[0]
            data.crs = metadata.crs

        # Serialize and store geometries for exact reconstruction if keep_geom is True
        if self.keep_geom:
            metadata.node_geometries = {}
            for node_type, node_gdf in nodes_dict.items():
                geoms = self._serialize_geometries(node_gdf)
                if geoms is not None:
                    metadata.node_geometries[node_type] = geoms

            metadata.edge_geometries = {}
            for edge_type, edge_gdf in edges_dict.items():
                if edge_gdf is not None and not edge_gdf.empty:
                    geoms = self._serialize_geometries(edge_gdf)
                    if geoms is not None:
                        metadata.edge_geometries[edge_type] = geoms

        data.graph_metadata = metadata

    def _create_node_id_mapping(
        self,
        node_gdf: gpd.GeoDataFrame,
    ) -> tuple[dict[str | int, int], str, list[str | int]]:
        """
        Create mapping from node IDs (from index) to sequential integer indices.

        PyTorch Geometric requires nodes to be identified by sequential integers starting from 0.
        This function creates the necessary mapping from original node identifiers (taken from
        the GeoDataFrame index) to these indices.

        Parameters
        ----------
        node_gdf : geopandas.GeoDataFrame
            GeoDataFrame containing node data. The index is used for node IDs.

        Returns
        -------
        dict[str | int, int]
            Dictionary mapping original IDs to integer indices.
        str
            Always "index", indicating the DataFrame index was used.
        list[str | int]
            List of original IDs in order.
        """
        # Use DataFrame index as the node identifier
        original_ids = node_gdf.index.tolist()
        id_mapping = {node_id: i for i, node_id in enumerate(original_ids)}
        return id_mapping, "index", original_ids

    def _create_edge_indices(
        self,
        edge_gdf: gpd.GeoDataFrame,
        source_mapping: dict[str | int, int],
        target_mapping: dict[str | int, int] | None = None,
    ) -> list[list[int]]:
        """
        Create edge connectivity matrix from edge data using MultiIndex.

        Extracts source and target node IDs from the MultiIndex of the edge GeoDataFrame
        and maps them to sequential integer indices required by PyTorch Geometric.

        Parameters
        ----------
        edge_gdf : geopandas.GeoDataFrame
            GeoDataFrame with MultiIndex containing (source_id, target_id) pairs.
        source_mapping : dict[str | int, int]
            Mapping from original source node IDs to integer indices.
        target_mapping : dict[str | int, int], optional
            Mapping from original target node IDs to integer indices.
            If None, uses source_mapping.

        Returns
        -------
        list[list[int]]
            Edge connectivity matrix as [source_indices, target_indices].
        """
        target_mapping = target_mapping or source_mapping

        # Extract source and target IDs from MultiIndex
        source_ids = edge_gdf.index.get_level_values(0)
        target_ids = edge_gdf.index.get_level_values(1)

        # Convert types if needed and validate
        source_ids = pd.Series(source_ids) if isinstance(source_ids, pd.Index) else source_ids
        target_ids = pd.Series(target_ids) if isinstance(target_ids, pd.Index) else target_ids

        # Find edges with valid source and target nodes
        valid_src_mask = source_ids.isin(source_mapping.keys())
        valid_dst_mask = target_ids.isin(target_mapping.keys())
        valid_edges_mask = valid_src_mask & valid_dst_mask

        # Process valid edges using vectorized operations
        valid_sources = source_ids[valid_edges_mask]
        valid_targets = target_ids[valid_edges_mask]

        # Map original node IDs to integer indices
        from_indices: np.ndarray[tuple[int, ...], np.dtype[np.int64]] = cast(
            "np.ndarray[tuple[int, ...], np.dtype[np.int64]]",
            valid_sources.map(
                source_mapping,
            ).to_numpy(dtype=int),
        )
        to_indices: np.ndarray[tuple[int, ...], np.dtype[np.int64]] = cast(
            "np.ndarray[tuple[int, ...], np.dtype[np.int64]]",
            valid_targets.map(
                target_mapping,
            ).to_numpy(dtype=int),
        )

        combined_array = np.column_stack([from_indices, to_indices]).astype(int)
        result: list[list[int]] = combined_array.tolist()
        return result

    def _reconstruct_node_gdf(
        self,
        data: Data | HeteroData,
        metadata: GraphMetadata,
        node_type: str | None = None,
    ) -> gpd.GeoDataFrame:
        """
        Reconstruct node GeoDataFrame from PyTorch Geometric data.

        This method reconstructs a GeoDataFrame containing node features and
        geometries from the PyTorch Geometric data object.

        Parameters
        ----------
        data : Data or HeteroData
            PyTorch Geometric data object.
        metadata : GraphMetadata
            Graph metadata.
        node_type : str, optional
            Node type name.

        Returns
        -------
        gpd.GeoDataFrame
            Reconstructed node GeoDataFrame.
        """
        is_hetero = metadata.is_hetero
        obj_data = data[node_type] if is_hetero and node_type else data

        # Extract features
        gdf_data = self._extract_features(obj_data, metadata, node_type, is_node=True)

        # Create geometry: use stored geometries if available, otherwise create from positions
        geometry = self._get_stored_geometries(metadata, is_hetero, node_type, is_node=True)

        # Fall back to creating geometry from positions if not stored
        if geometry is None:
            geometry = self._create_geometry_from_positions(obj_data)

        # Reconstruct index
        mapping_key = "default"
        if is_hetero and node_type:
            mapping_key = str(node_type)

        index_values: pd.Index | pd.MultiIndex | list[str | int] | None = None
        mapping_info = metadata.node_mappings.get(mapping_key)
        if mapping_info:
            original_ids = mapping_info.get("original_ids")
            if original_ids:
                num_nodes = obj_data.num_nodes
                # Ensure list and slice
                ids_list = (
                    original_ids if isinstance(original_ids, list) else list(range(num_nodes))
                )
                index_values = ids_list[:num_nodes]

        # Create GeoDataFrame
        if geometry is not None:
            gdf = gpd.GeoDataFrame(gdf_data, geometry=geometry, index=index_values)
        else:
            # Handle missing geometry
            length = len(next(iter(gdf_data.values()))) if gdf_data else 0
            empty_geom = gpd.GeoSeries([None] * length, crs=metadata.crs if metadata.crs else None)
            gdf = gpd.GeoDataFrame(gdf_data, geometry=empty_geom, index=index_values)

        # Set index names and CRS
        self._set_gdf_index_and_crs(gdf, node_type, metadata)

        return gdf

    def _reconstruct_edge_gdf(
        self,
        data: Data | HeteroData,
        metadata: GraphMetadata,
        edge_type: tuple[str, str, str] | None = None,
    ) -> gpd.GeoDataFrame:
        """
        Reconstruct edge GeoDataFrame from PyTorch Geometric data.

        This method reconstructs a GeoDataFrame containing edge features and
        geometries from the PyTorch Geometric data object.

        Parameters
        ----------
        data : Data or HeteroData
            PyTorch Geometric data object.
        metadata : GraphMetadata
            Graph metadata.
        edge_type : tuple, optional
            Edge type tuple.

        Returns
        -------
        gpd.GeoDataFrame
            Reconstructed edge GeoDataFrame.
        """
        is_hetero = metadata.is_hetero

        # Validate edge_type for heterogeneous graphs
        if is_hetero and edge_type is not None and not isinstance(edge_type, tuple):
            msg = "Edge type must be a tuple of (source_type, relation_type, target_type) for heterogeneous graphs"
            raise TypeError(msg)

        obj_data = data[edge_type] if is_hetero and edge_type else data

        # Extract features
        gdf_data = self._extract_features(obj_data, metadata, edge_type, is_node=False)

        # Create geometry: use stored geometries if available, otherwise create from positions
        geometry = self._get_stored_geometries(metadata, is_hetero, edge_type, is_node=False)

        # Fall back to creating geometry from positions if not stored
        if geometry is None:
            geometry = self._create_edge_geometries(obj_data, edge_type, is_hetero, data)

        # Reconstruct edge index
        stored_values: list[list[str | int]] | None = None
        if is_hetero and edge_type and isinstance(metadata.edge_index_values, dict):
            stored_values = metadata.edge_index_values.get(edge_type)
        elif not is_hetero and isinstance(metadata.edge_index_values, list):
            stored_values = metadata.edge_index_values

        index_values: pd.Index | pd.MultiIndex | list[str | int] | None = None
        if stored_values:
            # Determine number of rows based on edge data or stored values
            num_rows = len(next(iter(gdf_data.values()))) if gdf_data else len(stored_values[0])
            # Handle MultiIndex case
            arrays = [stored_values[i][:num_rows] for i in range(len(stored_values))]
            index_values = pd.MultiIndex.from_arrays(arrays)

        # Create GeoDataFrame
        if geometry is not None:
            gdf = gpd.GeoDataFrame(gdf_data, geometry=geometry, index=index_values)
        else:
            # Handle missing geometry
            length = len(next(iter(gdf_data.values()))) if gdf_data else 0
            empty_geom = gpd.GeoSeries([None] * length, crs=metadata.crs if metadata.crs else None)
            gdf = gpd.GeoDataFrame(gdf_data, geometry=empty_geom, index=index_values)

        # Set index names and CRS
        self._set_edge_index_names(gdf, edge_type, is_hetero, metadata)
        if metadata.crs:
            if gdf.empty or (gdf.geometry is not None and gdf.geometry.isna().all()):
                gdf.crs = metadata.crs
            else:
                gdf.set_crs(metadata.crs, allow_override=True, inplace=True)

        return gdf

    def _extract_tensor_columns(
        self,
        tensor: torch.Tensor | None,
        cols: list[str] | None,
    ) -> dict[str, np.ndarray[tuple[int, ...], np.dtype[np.float32]]]:
        """
        Extract tensor data into a dictionary.

        Helper function to convert tensor to numpy array and map to column names.

        Parameters
        ----------
        tensor : torch.Tensor or None
            Input tensor.
        cols : list[str] or None
            List of column names.

        Returns
        -------
        dict
            Dictionary mapping column names to numpy arrays.
        """
        if tensor is None or tensor.numel() == 0 or cols is None:
            return {}
        features_array = tensor.detach().cpu().numpy()
        num_cols = min(len(cols), features_array.shape[1])
        return {cols[i]: features_array[:, i] for i in range(num_cols)}

    def _extract_features(
        self,
        obj_data: Data | HeteroData,
        metadata: GraphMetadata,
        type_name: str | tuple[str, str, str] | None,
        is_node: bool,
    ) -> dict[str, np.ndarray[tuple[int, ...], np.dtype[np.float32]]]:
        """
        Extract features from data object.

        Generic method for extracting features for both nodes and edges.

        Parameters
        ----------
        obj_data : Data or HeteroData
            PyTorch Geometric data object.
        metadata : GraphMetadata
            Graph metadata.
        type_name : str or tuple, optional
            Node or edge type name.
        is_node : bool
            Whether to extract features for nodes.

        Returns
        -------
        dict
            Dictionary mapping feature names to numpy arrays.
        """
        gdf_data = {}
        is_hetero = metadata.is_hetero

        if is_node:
            # Extract node features (x)
            if hasattr(obj_data, "x") and obj_data.x is not None:
                feature_cols = metadata.node_feature_cols
                cols_list = None
                if is_hetero and type_name and isinstance(feature_cols, dict):
                    cols_list = feature_cols.get(str(type_name))
                elif not is_hetero and isinstance(feature_cols, list):
                    cols_list = feature_cols

                if cols_list is None:
                    num_features = obj_data.x.shape[1]
                    cols_list = [f"feat_{i}" for i in range(num_features)]

                gdf_data.update(self._extract_tensor_columns(obj_data.x, cols_list))

            # Extract node labels (y)
            if hasattr(obj_data, "y") and obj_data.y is not None:
                label_cols = metadata.node_label_cols
                cols_list = None
                if is_hetero and type_name and isinstance(label_cols, dict):
                    cols_list = label_cols.get(str(type_name))
                elif not is_hetero and isinstance(label_cols, list):
                    cols_list = label_cols

                if cols_list is None:
                    num_labels = obj_data.y.shape[1]
                    cols_list = [f"label_{i}" for i in range(num_labels)]

                gdf_data.update(self._extract_tensor_columns(obj_data.y, cols_list))

        # Extract edge features (edge_attr)
        elif hasattr(obj_data, "edge_attr") and obj_data.edge_attr is not None:
            feature_cols = metadata.edge_feature_cols
            cols_list = None
            if is_hetero and isinstance(type_name, tuple) and isinstance(feature_cols, dict):
                cols_list = feature_cols.get(type_name[1])  # relation type
            elif not is_hetero and isinstance(feature_cols, list):
                cols_list = feature_cols

            if cols_list is None:
                num_features = obj_data.edge_attr.shape[1]
                cols_list = [f"edge_feat_{i}" for i in range(num_features)]

            gdf_data.update(self._extract_tensor_columns(obj_data.edge_attr, cols_list))

        return gdf_data

    def _create_geometry_from_positions(
        self, node_data: Data | HeteroData
    ) -> gpd.array.GeometryArray | None:
        """
        Create geometry from node positions.

        This function converts node position tensors into GeoPandas geometry objects
        for spatial analysis and visualization.

        Parameters
        ----------
        node_data : Data or HeteroData
            PyTorch Geometric data object containing node positions.

        Returns
        -------
        gpd.array.GeometryArray or None
            Array of Point geometries created from node positions, or None if
            no position data is available.
        """
        if not hasattr(node_data, "pos") or node_data.pos is None:
            return None
        pos_array: np.ndarray[tuple[int, ...], np.dtype[np.float32]] = (
            node_data.pos.detach().cpu().numpy()
        )
        return gpd.points_from_xy(pos_array[:, 0], pos_array[:, 1])

    def _set_gdf_index_and_crs(
        self,
        gdf: gpd.GeoDataFrame,
        node_type: str | None,
        metadata: GraphMetadata,
    ) -> None:
        """
        Set index names and CRS on GeoDataFrame.

        This function configures the index names and coordinate reference system
        for a GeoDataFrame based on metadata information.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            The GeoDataFrame to configure.
        node_type : str or None
            The type of nodes in the GeoDataFrame.
        metadata : GraphMetadata
            Graph metadata containing index names and CRS information.
        """
        # Set index names
        if metadata.node_index_names:
            index_names: list[str] | None = None
            # Get index names based on heterogeneity and node type
            if metadata.is_hetero and node_type and isinstance(metadata.node_index_names, dict):
                index_names = metadata.node_index_names.get(node_type)
            elif not metadata.is_hetero and isinstance(metadata.node_index_names, list):
                index_names = metadata.node_index_names

            # Set index name if available
            if (
                index_names
                and hasattr(gdf.index, "names")
                and isinstance(index_names, list)
                and len(index_names) > 0
            ):
                gdf.index.name = index_names[0]

        # Set CRS
        if metadata.crs and hasattr(gdf, "geometry") and gdf.geometry is not None:
            # Check if the geometry column is truly empty or all null
            if gdf.empty or gdf.geometry.isna().all():
                gdf.crs = metadata.crs
            else:
                # Use set_crs for non-empty geometries
                gdf.set_crs(metadata.crs, allow_override=True, inplace=True)
        # If no geometry column, we can't set CRS - skip silently

    def _create_edge_geometries(
        self,
        edge_data: Data,
        edge_type: str | tuple[str, str, str] | None,
        is_hetero: bool,
        data: Data | HeteroData,
    ) -> gpd.array.GeometryArray | None:
        """
        Create edge geometries from edge indices and node positions.

        This function creates LineString geometries for edges by connecting
        the positions of source and destination nodes.

        Parameters
        ----------
        edge_data : Data
            PyTorch Geometric data object containing edge information.
        edge_type : str, tuple, or None
            The type of edges to create geometries for.
        is_hetero : bool
            Whether the graph is heterogeneous.
        data : Data or HeteroData
            Complete PyTorch Geometric data object containing node positions.

        Returns
        -------
        gpd.array.GeometryArray or None
            Array of LineString geometries for the edges, or None if
            node positions are not available.
        """
        # Get edge index array
        edge_index_array = edge_data.edge_index.detach().cpu().numpy()

        # Set default positions as None
        src_pos_array: np.ndarray[tuple[int, ...], np.dtype[np.float64]] | None = None
        dst_pos_array: np.ndarray[tuple[int, ...], np.dtype[np.float64]] | None = None

        # If hetero and specific edge type, get source and destination positions
        if is_hetero and isinstance(edge_type, tuple) and len(edge_type) == 3:
            src_type, _, dst_type = edge_type
            if hasattr(data[src_type], "pos") and data[src_type].pos is not None:
                src_pos_array = data[src_type].pos.detach().cpu().numpy()
            if hasattr(data[dst_type], "pos") and data[dst_type].pos is not None:
                dst_pos_array = data[dst_type].pos.detach().cpu().numpy()

        # If not hetero or no specific edge type, use default positions
        elif hasattr(data, "pos") and data.pos is not None:
            pos_array: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = (
                data.pos.detach().cpu().numpy()
            )
            src_pos_array = pos_array
            dst_pos_array = pos_array

        if src_pos_array is None or dst_pos_array is None:
            return None

        if edge_index_array.size == 0:
            geometries = []
        else:
            src_indices = edge_index_array[0]
            dst_indices = edge_index_array[1]

            # Vectorized bounds checking
            valid_src_mask = src_indices < len(src_pos_array)
            valid_dst_mask = dst_indices < len(dst_pos_array)
            valid_mask = valid_src_mask & valid_dst_mask

            # Get valid indices and coordinates
            valid_src_indices = src_indices[valid_mask]
            valid_dst_indices = dst_indices[valid_mask]
            src_coords = src_pos_array[valid_src_indices][:, :2]
            dst_coords = dst_pos_array[valid_dst_indices][:, :2]

            # Create LineStrings using vectorized coordinate pairing
            coord_pairs = np.stack([src_coords, dst_coords], axis=1)

            # Vectorized LineString creation - use map for better performance
            valid_geometries = list(map(LineString, coord_pairs))

            # Vectorized assignment using fancy indexing
            geometries_array = np.full(len(src_indices), None, dtype=object)
            geometries_array[valid_mask] = valid_geometries
            geometries = geometries_array.tolist()

        return gpd.array.from_shapely(geometries)

    def _set_edge_index_names(
        self,
        gdf: gpd.GeoDataFrame,
        edge_type: str | tuple[str, str, str] | None,
        is_hetero: bool,
        metadata: GraphMetadata,
    ) -> None:
        """
        Set index names on edge GeoDataFrame.

        This function configures the index names for an edge GeoDataFrame
        based on metadata information.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            The edge GeoDataFrame to configure.
        edge_type : str, tuple, or None
            The type of edges in the GeoDataFrame.
        is_hetero : bool
            Whether the graph is heterogeneous.
        metadata : GraphMetadata
            Graph metadata containing edge index name information.
        """
        index_names: list[str] | None = None
        if is_hetero and edge_type and isinstance(metadata.edge_index_names, dict):
            if isinstance(edge_type, tuple):
                index_names = metadata.edge_index_names.get(edge_type)
        elif not is_hetero and isinstance(metadata.edge_index_names, list):
            index_names = metadata.edge_index_names

        if (
            hasattr(gdf.index, "names")
            and isinstance(index_names, list)
            and len(index_names) > 1
            and isinstance(gdf.index, pd.MultiIndex)
        ):
            gdf.index.names = index_names

    def _extract_metadata(self, graph_data: Data | HeteroData) -> GraphMetadata:
        """
        Extract metadata from PyG object.

        Extended summary of metadata extraction from PyG objects.

        Parameters
        ----------
        graph_data : Data or HeteroData
            PyTorch Geometric data object.

        Returns
        -------
        GraphMetadata
            Extracted metadata from the graph.
        """
        validate_pyg(graph_data)
        # Type assertion for mypy
        metadata: GraphMetadata = graph_data.graph_metadata
        return metadata

    def _reconstruct_homogeneous(
        self,
        graph_data: Data,
        metadata: GraphMetadata,
        nodes: bool,
        edges: bool,
    ) -> tuple[gpd.GeoDataFrame | None, gpd.GeoDataFrame | None]:
        """
        Reconstruct homogeneous GeoDataFrames.

        Extended summary of homogeneous GeoDataFrame reconstruction.

        Parameters
        ----------
        graph_data : Data
            PyTorch Geometric Data object.
        metadata : GraphMetadata
            Graph metadata for reconstruction.
        nodes : bool
            Whether to reconstruct nodes.
        edges : bool
            Whether to reconstruct edges.

        Returns
        -------
        tuple
            Tuple of (nodes_gdf, edges_gdf) GeoDataFrames.
        """
        nodes_gdf = None
        edges_gdf = None

        if nodes:
            nodes_gdf = self._reconstruct_node_gdf(graph_data, metadata, None)

        if edges:
            edges_gdf = self._reconstruct_edge_gdf(graph_data, metadata, None)

        return nodes_gdf, edges_gdf

    def _reconstruct_heterogeneous(
        self,
        graph_data: HeteroData,
        metadata: GraphMetadata,
        nodes: bool,
        edges: bool,
    ) -> tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]]:
        """
        Reconstruct heterogeneous GeoDataFrames.

        Extended summary of heterogeneous GeoDataFrame reconstruction.

        Parameters
        ----------
        graph_data : HeteroData
            PyTorch Geometric HeteroData object.
        metadata : GraphMetadata
            Graph metadata for reconstruction.
        nodes : bool
            Whether to reconstruct nodes.
        edges : bool
            Whether to reconstruct edges.

        Returns
        -------
        tuple
            Tuple of (nodes_dict, edges_dict) dictionaries of GeoDataFrames.
        """
        nodes_dict = {}
        edges_dict = {}

        if nodes and metadata.node_types:
            for node_type in metadata.node_types:
                nodes_dict[node_type] = self._reconstruct_node_gdf(graph_data, metadata, node_type)

        if edges and metadata.edge_types:
            for edge_type in metadata.edge_types:
                edges_dict[edge_type] = self._reconstruct_edge_gdf(graph_data, metadata, edge_type)

        return nodes_dict, edges_dict

    def _create_features(
        self,
        gdf: gpd.GeoDataFrame,
        feature_cols: list[str] | None = None,
    ) -> torch.Tensor:
        """
        Convert attributes to PyTorch feature tensors.

        Generic method for creating both node and edge feature tensors.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            GeoDataFrame containing data.
        feature_cols : list[str], optional
            Column names to use as features.

        Returns
        -------
        torch.Tensor
            Feature tensor.
        """
        device = _get_device(self.device)
        dtype = self.dtype or torch.float32

        if feature_cols is None:
            # Return empty tensor when no feature columns specified
            return torch.zeros((len(gdf), 0), dtype=dtype, device=device)

        # Find valid columns that exist in the GeoDataFrame
        valid_cols = list(set(feature_cols) & set(gdf.columns))

        if not valid_cols:
            return torch.zeros((len(gdf), 0), dtype=dtype, device=device)

        # Select only numeric columns from valid_cols to prevent conversion errors
        # This logic was previously only in _create_edge_features but is good for nodes too
        numeric_cols = gdf[valid_cols].select_dtypes(include=np.number).columns.tolist()

        # Map torch dtype to numpy dtype for consistency
        numpy_dtype = torch.tensor(0, dtype=dtype).numpy().dtype

        # Ensure consistent column order based on feature_cols
        ordered_cols = [col for col in feature_cols if col in numeric_cols]

        features_array = gdf[ordered_cols].to_numpy().astype(numpy_dtype)
        return torch.from_numpy(features_array).to(device=device, dtype=dtype)

    def _create_node_positions(
        self,
        node_gdf: gpd.GeoDataFrame,
    ) -> torch.Tensor | None:
        """
        Extract spatial coordinates from node geometries.

        Extended summary of node position extraction from geometries.

        Parameters
        ----------
        node_gdf : gpd.GeoDataFrame
            GeoDataFrame containing node geometries.

        Returns
        -------
        torch.Tensor or None
            Node position tensor or None if no geometries.
        """
        device = _get_device(self.device)
        dtype = self.dtype or torch.float32

        # Get the geometry column
        if not hasattr(node_gdf, "geometry") or node_gdf.geometry is None:
            return None

        geom_series = node_gdf.geometry

        # Get centroids of geometries
        if geom_series.crs and geom_series.crs.is_geographic:
            # Reproject to a suitable projected CRS (UTM) to get accurate centroids
            utm_crs = geom_series.estimate_utm_crs()
            centroids = geom_series.to_crs(utm_crs).centroid.to_crs(geom_series.crs)
        else:
            centroids = geom_series.centroid

        # Map torch dtype to numpy dtype for consistency
        numpy_dtype = torch.tensor(0, dtype=dtype).numpy().dtype
        pos_data = np.column_stack(
            [
                centroids.x.to_numpy(),
                centroids.y.to_numpy(),
            ],
        ).astype(numpy_dtype)

        return torch.tensor(pos_data, dtype=dtype, device=device)

    def _serialize_geometries(self, gdf: gpd.GeoDataFrame) -> list[str] | None:
        """
        Serialize geometries to WKB hexadecimal format.

        This method converts geometry data from a GeoDataFrame into a list of
        WKB (Well-Known Binary) hexadecimal strings for storage in metadata.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            GeoDataFrame containing geometries to serialize.

        Returns
        -------
        list[str] or None
            List of WKB hexadecimal strings, or None if no geometry column.
        """
        if not hasattr(gdf, "geometry") or gdf.geometry is None:
            return None

        # Convert each geometry to WKB hex format
        return [geom.wkb_hex if geom is not None else "" for geom in gdf.geometry]

    def _deserialize_geometries(
        self, wkb_list: list[str], crs: object = None
    ) -> gpd.array.GeometryArray:
        """
        Deserialize WKB hexadecimal strings to geometries.

        This method reconstructs geometry data from WKB hexadecimal strings
        stored in metadata.

        Parameters
        ----------
        wkb_list : list[str]
            List of WKB hexadecimal strings.
        crs : object, optional
            Coordinate reference system for the geometries.

        Returns
        -------
        gpd.array.GeometryArray
            Array of reconstructed geometries.
        """
        # Reconstruct geometries from WKB hex
        geometries = []
        for wkb_hex in wkb_list:
            if wkb_hex:
                geom = wkb.loads(bytes.fromhex(wkb_hex))
                geometries.append(geom)
            else:
                geometries.append(None)

        return gpd.array.from_shapely(geometries, crs=crs)

    def _get_stored_geometries(
        self,
        metadata: GraphMetadata,
        is_hetero: bool,
        type_key: str | tuple[str, str, str] | None,
        is_node: bool = True,
    ) -> gpd.array.GeometryArray | None:
        """
        Retrieve stored geometries from metadata.

        This method extracts serialized geometries from metadata and deserializes
        them back to GeoPandas geometry arrays for reconstruction.

        Parameters
        ----------
        metadata : GraphMetadata
            Graph metadata containing stored geometries.
        is_hetero : bool
            Whether the graph is heterogeneous.
        type_key : str or tuple or None
            Node type or edge type key.
        is_node : bool, default True
            Whether retrieving node or edge geometries.

        Returns
        -------
        gpd.array.GeometryArray or None
            Reconstructed geometries or None if not stored or if keep_geom is False.
        """
        # If keep_geom is False, don't use stored geometries even if they exist
        if not self.keep_geom:
            return None

        geometries_dict = metadata.node_geometries if is_node else metadata.edge_geometries

        if geometries_dict is None:
            return None

        # Check for stored geometries
        if is_hetero and type_key and isinstance(geometries_dict, dict):
            # Type narrowing: dict could be dict[str, ...] or dict[tuple[str, str, str], ...]
            wkb_list = cast("dict[object, list[str]]", geometries_dict).get(type_key)
            if wkb_list is not None:
                return self._deserialize_geometries(wkb_list, metadata.crs)
        elif not is_hetero and isinstance(geometries_dict, list):
            return self._deserialize_geometries(geometries_dict, metadata.crs)

        return None


def gdf_to_pyg(
    nodes: dict[str, gpd.GeoDataFrame] | gpd.GeoDataFrame,
    edges: dict[tuple[str, str, str], gpd.GeoDataFrame] | gpd.GeoDataFrame | None = None,
    node_feature_cols: dict[str, list[str]] | list[str] | None = None,
    node_label_cols: dict[str, list[str]] | list[str] | None = None,
    edge_feature_cols: dict[str, list[str]] | list[str] | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    keep_geom: bool = True,
) -> Data | HeteroData:
    """
    Convert GeoDataFrames (nodes/edges) to a PyTorch Geometric object.

    This function serves as the main entry point for converting spatial data into
    PyTorch Geometric graph objects. It automatically detects whether to create
    homogeneous or heterogeneous graphs based on input structure. Node identifiers
    are taken from the GeoDataFrame index. Edge relationships are defined by a
    MultiIndex on the edge GeoDataFrame (source ID, target ID).

    The operation multiplies typed adjacency tables to connect terminal node
    pairs and can aggregate additional numeric edge attributes along the way.

    Parameters
    ----------
    nodes : dict[str, geopandas.GeoDataFrame] or geopandas.GeoDataFrame
        Node data. For homogeneous graphs, provide a single GeoDataFrame.
        For heterogeneous graphs, provide a dictionary mapping node type names
        to their respective GeoDataFrames. The index of these GeoDataFrames
        will be used as node identifiers.
    edges : dict[tuple[str, str, str], geopandas.GeoDataFrame] or geopandas.GeoDataFrame, optional
        Edge data. For homogeneous graphs, provide a single GeoDataFrame.
        For heterogeneous graphs, provide a dictionary mapping edge type tuples
        (source_type, relation_type, target_type) to their GeoDataFrames.
        The GeoDataFrame must have a MultiIndex where the first level represents
        source node IDs and the second level represents target node IDs.
    node_feature_cols : dict[str, list[str]] or list[str], optional
        Column names to use as node features. For heterogeneous graphs,
        provide a dictionary mapping node types to their feature columns.
    node_label_cols : dict[str, list[str]] or list[str], optional
        Column names to use as node labels for supervised learning tasks.
        For heterogeneous graphs, provide a dictionary mapping node types
        to their label columns.
    edge_feature_cols : dict[str, list[str]] or list[str], optional
        Column names to use as edge features. For heterogeneous graphs,
        provide a dictionary mapping relation types to their feature columns.
    device : str or torch.device, optional
        Target device for tensor placement ('cpu', 'cuda', or torch.device).
        If None, automatically selects CUDA if available, otherwise CPU.
    dtype : torch.dtype, optional
        Data type for float tensors (e.g., torch.float32, torch.float16).
        If None, uses torch.float32 (default PyTorch float type).
    keep_geom : bool, default True
        Whether to preserve geometry information during conversion.
        If True, original geometries are serialized and stored in metadata for
        exact reconstruction. If False, geometries are reconstructed from node
        positions during conversion back to GeoDataFrames (creating straight-line
        edges between nodes).

    Returns
    -------
    torch_geometric.data.Data or torch_geometric.data.HeteroData
        PyTorch Geometric Data object for homogeneous graphs or HeteroData
        object for heterogeneous graphs. The returned object contains:

        - Node features (x), positions (pos), and labels (y) if available
        - Edge connectivity (edge_index) and features (edge_attr) if available
        - Metadata for reconstruction including ID mappings and column names

    Raises
    ------
    ImportError
        If PyTorch Geometric is not installed.
    ValueError
        If input GeoDataFrames are invalid or incompatible.

    See Also
    --------
    pyg_to_gdf : Convert PyTorch Geometric data back to GeoDataFrames.
    nx_to_pyg : Convert NetworkX graph to PyTorch Geometric object.
    city2graph.utils.validate_gdf : Validate GeoDataFrame structure.

    Notes
    -----
    This function automatically detects the graph type based on input structure.
    For heterogeneous graphs, provide dictionaries mapping types to GeoDataFrames.
    Node positions are automatically extracted from geometry centroids when available.
    - Preserves original coordinate reference systems (CRS)
    - Maintains index structure for bidirectional conversion
    - Handles both Point and non-Point geometries (using centroids)
    - Creates empty tensors for missing features/edges
    - For heterogeneous graphs, ensures consistent node/edge type mapping

    Examples
    --------
    Create a homogeneous graph from single GeoDataFrames:

    >>> import geopandas as gpd
    >>> from city2graph.graph import gdf_to_pyg
    >>>
    >>> # Load and prepare node data
    >>> nodes_gdf = gpd.read_file("nodes.geojson").set_index("node_id")
    >>> edges_gdf = gpd.read_file("edges.geojson").set_index(["source_id", "target_id"])
    >>>
    >>> # Convert to PyTorch Geometric
    >>> data = gdf_to_pyg(nodes_gdf, edges_gdf,
    ...                   node_feature_cols=['population', 'area'])

    Create a heterogeneous graph from dictionaries:

    >>> # Prepare heterogeneous data
    >>> buildings_gdf = buildings_gdf.set_index("building_id")
    >>> roads_gdf = roads_gdf.set_index("road_id")
    >>> connections_gdf = connections_gdf.set_index(["building_id", "road_id"])
    >>>
    >>> # Define node and edge types
    >>> nodes_dict = {'building': buildings_gdf, 'road': roads_gdf}
    >>> edges_dict = {('building', 'connects', 'road'): connections_gdf}
    >>>
    >>> # Convert to heterogeneous graph with labels
    >>> data = gdf_to_pyg(nodes_dict, edges_dict,
    ...                   node_label_cols={'building': ['type'], 'road': ['category']})
    """
    if not TORCH_AVAILABLE:
        raise ImportError(TORCH_ERROR_MSG)

    converter = PyGConverter(
        node_feature_cols=node_feature_cols,
        node_label_cols=node_label_cols,
        edge_feature_cols=edge_feature_cols,
        device=device,
        dtype=dtype,
        keep_geom=keep_geom,
    )
    return converter.gdf_to_pyg(nodes, edges)


def pyg_to_gdf(
    data: Data | HeteroData,
    node_types: str | list[str] | None = None,
    edge_types: str | list[tuple[str, str, str]] | None = None,
    keep_geom: bool = True,
) -> (
    tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]]
    | tuple[gpd.GeoDataFrame | None, gpd.GeoDataFrame | None]
):
    """
    Convert PyTorch Geometric data to GeoDataFrames.

    Reconstructs the original GeoDataFrame structure from PyTorch Geometric
    Data or HeteroData objects. This function provides bidirectional conversion
    capability, preserving spatial information, feature data, and metadata.

    Parameters
    ----------
    data : torch_geometric.data.Data or torch_geometric.data.HeteroData
        PyTorch Geometric data object to convert back to GeoDataFrames.
    node_types : str or list[str], optional
        For heterogeneous graphs, specify which node types to reconstruct.
        If None, reconstructs all available node types.
    edge_types : str or list[tuple[str, str, str]], optional
        For heterogeneous graphs, specify which edge types to reconstruct.
        Edge types are specified as (source_type, relation_type, target_type) tuples.
        If None, reconstructs all available edge types.
    keep_geom : bool, default True
        Whether to use stored geometries for reconstruction. If True and geometries
        are stored in metadata, uses the original geometries. If False or no stored
        geometries exist, reconstructs geometries from node positions (creating
        straight-line edges between nodes).

    Returns
    -------
    tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]]
        **For Data input:** Returns a tuple containing:
            - First element: GeoDataFrame containing nodes
            - Second element: GeoDataFrame containing edges (or None if no edges)
        **For HeteroData input:** Returns a tuple containing:
            - First element: dict mapping node type names to GeoDataFrames
            - Second element: dict mapping edge types to GeoDataFrames

    See Also
    --------
    gdf_to_pyg : Convert GeoDataFrames to PyTorch Geometric object.
    pyg_to_nx : Convert PyTorch Geometric data to NetworkX graph.

    Notes
    -----
    - Preserves original index structure and names when available
    - Reconstructs geometry from stored position tensors
    - Maintains coordinate reference system (CRS) information
    - Converts feature tensors back to named DataFrame columns
    - Handles both homogeneous and heterogeneous graph structures

    Examples
    --------
    Convert homogeneous PyTorch Geometric data back to GeoDataFrames:

    >>> from city2graph.graph import pyg_to_gdf
    >>>
    >>> # Convert back to GeoDataFrames
    >>> nodes_gdf, edges_gdf = pyg_to_gdf(data)

    Convert heterogeneous data with specific node types:

    >>> # Convert only specific node types
    >>> node_gdfs, edge_gdfs = pyg_to_gdf(hetero_data,
    ...                                   node_types=['building', 'road'])
    """
    converter = PyGConverter(keep_geom=keep_geom)
    return converter.pyg_to_gdf(data, node_types, edge_types)


# ============================================================================
# NETWORKX CONVERSION FUNCTIONS
# ============================================================================


def pyg_to_nx(data: Data | HeteroData, keep_geom: bool = True) -> nx.Graph:
    """
    Convert a PyTorch Geometric object to a NetworkX graph.

    Converts PyTorch Geometric Data or HeteroData objects to NetworkX graphs,
    preserving node and edge features as graph attributes. This enables
    compatibility with the extensive NetworkX ecosystem for graph analysis.

    Parameters
    ----------
    data : torch_geometric.data.Data or torch_geometric.data.HeteroData
        PyTorch Geometric data object to convert.
    keep_geom : bool, default True
        Whether to use stored geometries for reconstruction. If True and geometries
        are stored in metadata, uses the original geometries. If False or no stored
        geometries exist, reconstructs geometries from node positions.

    Returns
    -------
    networkx.Graph
        The converted NetworkX graph with node and edge attributes.
        For heterogeneous graphs, node and edge types are stored as attributes.

    Raises
    ------
    ImportError
        If PyTorch Geometric is not installed.

    See Also
    --------
    nx_to_pyg : Convert NetworkX graph to PyTorch Geometric object.
    pyg_to_gdf : Convert PyTorch Geometric data to GeoDataFrames.

    Notes
    -----
    - Node features, positions, and labels are stored as node attributes
    - Edge features are stored as edge attributes
    - For heterogeneous graphs, type information is preserved
    - Geometry information is converted from tensor positions
    - Maintains compatibility with NetworkX analysis algorithms

    Examples
    --------
    Convert PyTorch Geometric data to NetworkX:

    >>> from city2graph.graph import pyg_to_nx
    >>> import networkx as nx
    >>>
    >>> # Convert to NetworkX graph
    >>> nx_graph = pyg_to_nx(data)
    >>>
    >>> # Use NetworkX algorithms
    >>> centrality = nx.betweenness_centrality(nx_graph)
    >>> communities = nx.community.greedy_modularity_communities(nx_graph)
    """
    nodes, edges = pyg_to_gdf(data, keep_geom=keep_geom)
    return gdf_to_nx(nodes, edges)


def nx_to_pyg(
    graph: nx.Graph,
    node_feature_cols: list[str] | None = None,
    node_label_cols: list[str] | None = None,
    edge_feature_cols: list[str] | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    keep_geom: bool = True,
) -> Data | HeteroData:
    """
    Convert NetworkX graph to PyTorch Geometric Data object.

    Converts a NetworkX Graph to a PyTorch Geometric Data object by first
    converting to GeoDataFrames then using the main conversion pipeline. This
    provides a bridge between NetworkX's rich graph analysis tools and PyTorch
    Geometric's deep learning capabilities.

    Parameters
    ----------
    graph : networkx.Graph
        NetworkX graph to convert.
    node_feature_cols : list[str], optional
        List of node attribute names to use as features.
    node_label_cols : list[str], optional
        List of node attribute names to use as labels.
    edge_feature_cols : list[str], optional
        List of edge attribute names to use as features.
    device : torch.device or str, optional
        Target device for tensor placement ('cpu', 'cuda', or torch.device).
        If None, automatically selects CUDA if available, otherwise CPU.
    dtype : torch.dtype, optional
        Data type for float tensors (e.g., torch.float32, torch.float16).
        If None, uses torch.float32 (default PyTorch float type).
    keep_geom : bool, default True
        Whether to preserve geometry information during conversion.
        If True, original geometries are serialized and stored in metadata for
        exact reconstruction. If False, geometries are reconstructed from node
        positions during conversion back to GeoDataFrames.

    Returns
    -------
    torch_geometric.data.Data or torch_geometric.data.HeteroData
        PyTorch Geometric Data object for homogeneous graphs or HeteroData
        object for heterogeneous graphs. The returned object contains:

        - Node features (x), positions (pos), and labels (y) if available
        - Edge connectivity (edge_index) and features (edge_attr) if available
        - Metadata for reconstruction including ID mappings and column names

    Raises
    ------
    ImportError
        If PyTorch Geometric is not installed.
    ValueError
        If the NetworkX graph is invalid or empty.

    See Also
    --------
    pyg_to_nx : Convert PyTorch Geometric data to NetworkX graph.
    gdf_to_pyg : Convert GeoDataFrames to PyTorch Geometric object.
    city2graph.utils.nx_to_gdf : Convert NetworkX graph to GeoDataFrames.

    Notes
    -----
    - Uses intermediate GeoDataFrame conversion for consistency
    - Preserves all graph attributes and metadata
    - Handles spatial coordinates if present in node attributes
    - Maintains compatibility with existing city2graph workflows
    - Automatically creates geometry from 'x', 'y' coordinates if available

    Examples
    --------
    Convert a NetworkX graph with spatial data:

    >>> import networkx as nx
    >>> from city2graph.graph import nx_to_pyg
    >>>
    >>> # Create NetworkX graph with spatial attributes
    >>> G = nx.Graph()
    >>> G.add_node(0, x=0.0, y=0.0, population=1000)
    >>> G.add_node(1, x=1.0, y=1.0, population=1500)
    >>> G.add_edge(0, 1, weight=0.5, road_type='primary')
    >>>
    >>> # Convert to PyTorch Geometric
    >>> data = nx_to_pyg(G,
    ...                  node_feature_cols=['population'],
    ...                  edge_feature_cols=['weight'])

    Convert from graph analysis results:

    >>> # Use NetworkX for analysis, then convert for ML
    >>> communities = nx.community.greedy_modularity_communities(G)
    >>> # Add community labels to nodes
    >>> for i, community in enumerate(communities):
    ...     for node in community:
    ...         G.nodes[node]['community'] = i
    >>>
    >>> # Convert with community labels
    >>> data = nx_to_pyg(G, node_label_cols=['community'])
    """
    # Validate NetworkX graph (includes type checking)
    validate_nx(graph)

    # Get nodes and edges GeoDataFrames
    nodes_gdf, edges_gdf = nx_to_gdf(graph, nodes=True, edges=True)

    # Convert to PyG using existing function
    return gdf_to_pyg(
        nodes=nodes_gdf,
        edges=edges_gdf,
        node_feature_cols=node_feature_cols,
        node_label_cols=node_label_cols,
        edge_feature_cols=edge_feature_cols,
        device=device,
        dtype=dtype,
        keep_geom=keep_geom,
    )


# ============================================================================
# TORCH UTILITIES FUNCTIONS
# ============================================================================


def is_torch_available() -> bool:
    """
    Check if PyTorch Geometric is available.

    This utility function checks whether the required PyTorch and PyTorch Geometric
    packages are installed and can be imported. It's useful for conditional
    functionality and providing helpful error messages.

    Returns
    -------
    bool
        True if PyTorch Geometric can be imported, False otherwise.

    See Also
    --------
    gdf_to_pyg : Convert GeoDataFrames to PyTorch Geometric (requires torch).
    pyg_to_gdf : Convert PyTorch Geometric to GeoDataFrames (requires torch).

    Notes
    -----
    - Returns False if either PyTorch or PyTorch Geometric is missing
    - Used internally by torch-dependent functions to provide helpful error messages

    Examples
    --------
    Check availability before using torch-dependent functions:

    >>> from city2graph.graph import is_torch_available
    >>>
    >>> if is_torch_available():
    ...     from city2graph.graph import gdf_to_pyg
    ...     data = gdf_to_pyg(nodes_gdf, edges_gdf)
    ... else:
    ...     print("PyTorch Geometric not available.")
    """
    return TORCH_AVAILABLE


def _get_device(device: str | torch.device | None) -> torch.device:
    """
    Normalize the device argument and return a torch.device instance.

    This function provides a consistent interface for device specification across
    the library, handling automatic device selection and validation.

    Parameters
    ----------
    device : str, torch.device, or None
        Device specification. Can be 'cpu', 'cuda', a torch.device object, or None.
        If None, automatically selects CUDA if available, otherwise CPU.

    Returns
    -------
    torch.device
        Normalized torch.device object.

    Raises
    ------
    ImportError
        If PyTorch is not available.
    ValueError
        If device string is not 'cpu' or 'cuda', or if 'cuda' is selected but not available.
    TypeError
        If device is not a valid type.

    See Also
    --------
    torch.device : PyTorch device specification.

    Examples
    --------
    >>> device = _normalize_device("cuda")
    >>> device = _normalize_device(None)  # Auto-selects best available
    """
    if not TORCH_AVAILABLE or torch is None:
        raise ImportError(TORCH_ERROR_MSG)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Check for invalid types first
    if not isinstance(device, (str, torch.device)):
        raise TypeError(DEVICE_ERROR_MSG)

    try:
        result_device = torch.device(device)
    except RuntimeError as e:
        # Convert RuntimeError from torch.device() to ValueError for consistency
        raise ValueError(DEVICE_ERROR_MSG) from e

    if result_device.type == "cuda" and not torch.cuda.is_available():
        msg = f"CUDA selected, but not available. {DEVICE_ERROR_MSG}"
        raise ValueError(msg)

    return result_device


def validate_pyg(data: Data | HeteroData) -> GraphMetadata:
    """
    Validate PyTorch Geometric Data or HeteroData objects and return metadata.

    This centralized validation function performs comprehensive validation of PyG objects,
    including type checking, metadata validation, and structural consistency checks.
    It serves as the single point of validation for all PyG objects in city2graph.

    Parameters
    ----------
    data : torch_geometric.data.Data or torch_geometric.data.HeteroData
        PyTorch Geometric data object to validate.

    Returns
    -------
    GraphMetadata
        Metadata object containing graph information for reconstruction.

    Raises
    ------
    ImportError
        If PyTorch Geometric is not installed.
    TypeError
        If data is not a valid PyTorch Geometric object.
    ValueError
        If the data object is missing required metadata or is inconsistent.

    See Also
    --------
    pyg_to_gdf : Convert PyG objects to GeoDataFrames.
    pyg_to_nx : Convert PyG objects to NetworkX graphs.

    Examples
    --------
    >>> data = gdf_to_pyg(nodes_gdf, edges_gdf)
    >>> metadata = validate_pyg(data)
    """
    # Check PyTorch availability first
    if not TORCH_AVAILABLE:
        raise ImportError(TORCH_ERROR_MSG)

    # Comprehensive type checking for PyG objects
    if not isinstance(data, (Data, HeteroData)):
        # Provide detailed error message based on the actual type
        actual_type = type(data).__name__
        msg = (
            f"Input must be a PyTorch Geometric Data or HeteroData object, "
            f"got {actual_type}. Ensure you have PyTorch Geometric installed "
            f"and are passing a valid PyG object."
        )
        raise TypeError(msg)

    # Validate metadata presence and type
    if not hasattr(data, "graph_metadata"):
        msg = (
            "PyG object is missing 'graph_metadata' attribute. "
            "This object may not have been created by city2graph. "
            "Use city2graph.graph.gdf_to_pyg() or city2graph.graph.nx_to_pyg() "
            "to create compatible PyG objects."
        )
        raise ValueError(msg)

    if not isinstance(data.graph_metadata, GraphMetadata):
        actual_metadata_type = type(data.graph_metadata).__name__
        msg = (
            f"PyG object has 'graph_metadata' of incorrect type: {actual_metadata_type}. "
            f"Expected GraphMetadata. This object may not have been created by city2graph."
        )
        raise TypeError(msg)

    metadata = data.graph_metadata
    is_hetero = isinstance(data, HeteroData)

    # Validate consistency between PyG object type and metadata
    if is_hetero and not metadata.is_hetero:
        msg = (
            "Inconsistency detected: PyG object is HeteroData but metadata.is_hetero is False. "
            "This indicates corrupted metadata or an incorrectly constructed object."
        )
        raise ValueError(msg)
    if not is_hetero and metadata.is_hetero:
        msg = (
            "Inconsistency detected: PyG object is Data but metadata.is_hetero is True. "
            "This indicates corrupted metadata or an incorrectly constructed object."
        )
        raise ValueError(msg)

    # Additional structural validation for heterogeneous graphs
    if is_hetero:
        _validate_hetero_structure(data, metadata)
    else:
        _validate_homo_structure(data, metadata)

    return metadata


def _validate_hetero_structure(data: HeteroData, metadata: GraphMetadata) -> None:
    """
    Validate structural consistency of heterogeneous PyG data.

    Performs comprehensive validation of heterogeneous graph structure, ensuring
    that node types, edge types, and tensor dimensions are consistent.

    Parameters
    ----------
    data : HeteroData
        The heterogeneous PyTorch Geometric data object to validate.
    metadata : GraphMetadata
        Metadata containing expected graph structure information.

    See Also
    --------
    _validate_homo_structure : Validate homogeneous graph structure.

    Examples
    --------
    >>> data = gdf_to_pyg(nodes_dict, edges_dict)
    >>> metadata = data.graph_metadata
    >>> _validate_hetero_structure(data, metadata)
    """
    # Check that node types in metadata match actual node types in data
    if metadata.node_types:
        actual_node_types = set(data.node_types)
        expected_node_types = set(metadata.node_types)
        if actual_node_types != expected_node_types:
            msg = (
                f"Node types mismatch: metadata expects {expected_node_types}, "
                f"but PyG object has {actual_node_types}"
            )
            raise ValueError(msg)

    # Check that edge types in metadata match actual edge types in data
    if metadata.edge_types:
        actual_edge_types = set(data.edge_types)
        expected_edge_types = set(metadata.edge_types)
        if actual_edge_types != expected_edge_types:
            msg = (
                f"Edge types mismatch: metadata expects {expected_edge_types}, "
                f"but PyG object has {actual_edge_types}"
            )
            raise ValueError(msg)

    # Validate tensor shape consistency for each node type
    for node_type in data.node_types:
        node_data = data[node_type]
        if hasattr(node_data, "x") and node_data.x is not None:
            num_nodes = node_data.x.size(0)

            # Check position tensor consistency
            if (
                hasattr(node_data, "pos")
                and node_data.pos is not None
                and node_data.pos.size(0) != num_nodes
            ):
                msg = (
                    f"Node type '{node_type}': position tensor size ({node_data.pos.size(0)}) "
                    f"doesn't match node feature tensor size ({num_nodes})"
                )
                raise ValueError(msg)

            # Check label tensor consistency
            if (
                hasattr(node_data, "y")
                and node_data.y is not None
                and node_data.y.size(0) != num_nodes
            ):
                msg = (
                    f"Node type '{node_type}': label tensor size ({node_data.y.size(0)}) "
                    f"doesn't match node feature tensor size ({num_nodes})"
                )
                raise ValueError(msg)


def _validate_homo_structure(data: Data, metadata: GraphMetadata) -> None:
    """
    Validate structural consistency of homogeneous PyG data.

    Performs comprehensive validation of homogeneous graph structure, ensuring
    that tensor dimensions are consistent and metadata is properly structured.

    Parameters
    ----------
    data : Data
        The homogeneous PyTorch Geometric data object to validate.
    metadata : GraphMetadata
        Metadata containing expected graph structure information.

    See Also
    --------
    _validate_hetero_structure : Validate heterogeneous graph structure.

    Examples
    --------
    >>> data = gdf_to_pyg(nodes_gdf, edges_gdf)
    >>> metadata = data.graph_metadata
    >>> _validate_homo_structure(data, metadata)
    """
    # Validate that metadata has the expected structure for homogeneous graphs
    if metadata.node_types and len(metadata.node_types) > 0:
        msg = "Homogeneous graph metadata should not have node_types specified"
        raise ValueError(msg)

    if metadata.edge_types and len(metadata.edge_types) > 0:
        msg = "Homogeneous graph metadata should not have edge_types specified"
        raise ValueError(msg)

    # Validate that node mappings use the "default" key for homogeneous graphs
    if metadata.node_mappings and "default" not in metadata.node_mappings:
        msg = "Homogeneous graph metadata should use 'default' key in node_mappings"
        raise ValueError(msg)

    # Validate that feature/label columns are lists, not dicts
    if metadata.node_feature_cols and not isinstance(metadata.node_feature_cols, list):
        msg = "Homogeneous graph metadata should have node_feature_cols as list, not dict"
        raise ValueError(msg)

    if metadata.node_label_cols and not isinstance(metadata.node_label_cols, list):
        msg = "Homogeneous graph metadata should have node_label_cols as list, not dict"
        raise ValueError(msg)

    if metadata.edge_feature_cols and not isinstance(metadata.edge_feature_cols, list):
        msg = "Homogeneous graph metadata should have edge_feature_cols as list, not dict"
        raise ValueError(msg)

    # Validate tensor shape consistency
    if hasattr(data, "x") and data.x is not None:
        num_nodes = data.x.size(0)

        # Check position tensor consistency
        if hasattr(data, "pos") and data.pos is not None and data.pos.size(0) != num_nodes:
            msg = (
                f"Node position tensor size ({data.pos.size(0)}) "
                f"doesn't match node feature tensor size ({num_nodes})"
            )
            raise ValueError(msg)

        # Check label tensor consistency
        if hasattr(data, "y") and data.y is not None and data.y.size(0) != num_nodes:
            msg = (
                f"Node label tensor size ({data.y.size(0)}) "
                f"doesn't match node feature tensor size ({num_nodes})"
            )
            raise ValueError(msg)

    # Validate edge tensor consistency
    if hasattr(data, "edge_index") and data.edge_index is not None:
        num_edges = data.edge_index.size(1)

        # Check edge attribute tensor consistency
        if (
            hasattr(data, "edge_attr")
            and data.edge_attr is not None
            and data.edge_attr.size(0) != num_edges
        ):
            msg = (
                f"Edge attribute tensor size ({data.edge_attr.size(0)}) "
                f"doesn't match number of edges ({num_edges})"
            )
            raise ValueError(msg)


def _validate_and_resolve_parameters(
    new_relation_name: str | None,
    endpoint_type: str | None,
    min_threshold: float,
    threshold: float,
) -> tuple[str, tuple[str, str, str]] | None:
    """
    Validate and resolve parameters for metapath connection.

    Validates parameter consistency and constructs the target edge type tuple
    from the provided parameters.

    Parameters
    ----------
    new_relation_name : str | None
        Target edge relation name.
    endpoint_type : str | None
        The node type to connect.
    min_threshold : float
        Minimum cost threshold.
    threshold : float
        Maximum cost threshold.

    Returns
    -------
    tuple[str, tuple[str, str, str]] | None
        Tuple of (endpoint_type, target_edge_type_tuple) if valid, None otherwise.
    """
    resolved_endpoint_type = endpoint_type
    relation_name = None
    target_edge_type_tuple = None

    if new_relation_name is not None:
        relation_name = new_relation_name

    if resolved_endpoint_type is None:
        logger.warning("endpoint_type not provided.")
        return None

    if target_edge_type_tuple is None:
        relation_name = relation_name or f"connected_within_{min_threshold}_{threshold}"
        target_edge_type_tuple = (resolved_endpoint_type, relation_name, resolved_endpoint_type)

    return resolved_endpoint_type, target_edge_type_tuple


def add_metapaths_by_weight(  # noqa: PLR0913
    graph: (
        tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]]
        | nx.Graph
        | nx.MultiGraph
        | None
    ) = None,
    nodes: dict[str, gpd.GeoDataFrame] | None = None,
    edges: dict[tuple[str, str, str], gpd.GeoDataFrame] | None = None,
    weight: str | None = None,
    threshold: float | None = None,
    new_relation_name: str | None = None,
    min_threshold: float = 0.0,
    edge_types: list[tuple[str, str, str]] | None = None,
    endpoint_type: str | None = None,
    directed: bool = False,
    multigraph: bool = False,
    as_nx: bool = False,
) -> (
    nx.Graph
    | nx.MultiGraph
    | tuple[
        dict[str, gpd.GeoDataFrame],
        dict[tuple[str, str, str], gpd.GeoDataFrame],
    ]
):
    """
    Connect nodes of a specific type if they are reachable within a cost threshold band.

    This function dynamically adds metapaths (edges) between nodes of a specified
    `endpoint_type` if they are reachable within a given cost band [`min_threshold`,
    `threshold`] based on edge weights (e.g., travel time). It uses Dijkstra's
    algorithm for path finding via `scipy.sparse.csgraph` for efficiency.

    Parameters
    ----------
    graph : tuple or networkx.Graph or networkx.MultiGraph, optional
        Input graph. Can be a tuple of (nodes_dict, edges_dict) or a NetworkX graph.
    nodes : dict[str, geopandas.GeoDataFrame], optional
        Dictionary of node GeoDataFrames.
    edges : dict[tuple[str, str, str], geopandas.GeoDataFrame], optional
        Dictionary of edge GeoDataFrames.
    weight : str
        The edge attribute to use as weight (e.g., 'travel_time').
    threshold : float
        The maximum cost threshold for connection.
    new_relation_name : str, optional
        Name of the new edge relation.
    min_threshold : float, default 0.0
        The minimum cost threshold for connection.
    edge_types : list[tuple[str, str, str]], optional
        List of edge types to consider for traversal. If None, all edges are used.
    endpoint_type : str, optional
        The node type to connect (e.g., 'building').
    directed : bool, default False
        If True, creates a directed graph for traversal.
    multigraph : bool, default False
        If True, returns a MultiGraph (only relevant if as_nx=True).
    as_nx : bool, default False
        If True, returns a NetworkX graph.

    Returns
    -------
    nx.Graph or nx.MultiGraph or tuple
        The graph with added metapaths. Format depends on `as_nx` parameter.
    """
    if weight is None:
        msg = "weight must be provided"
        raise ValueError(msg)
    if threshold is None:
        msg = "threshold must be provided"
        raise ValueError(msg)

    # Normalize input to graph tuple or nx graph
    if nodes is not None:
        if graph is not None:
            msg = "Cannot provide both 'graph' and 'nodes'/'edges'."
            raise ValueError(msg)
        graph = (nodes, edges or {})
    elif graph is None:
        msg = "Either 'graph' or 'nodes' (and optionally 'edges') must be provided."
        raise ValueError(msg)
    # Validate and resolve parameters
    params = _validate_and_resolve_parameters(
        new_relation_name, endpoint_type, min_threshold, threshold
    )
    if params is None:
        return _return_original_graph(graph)

    endpoint_type, target_edge_type_tuple = params

    # Prepare data for sparse matrix construction
    (
        node_to_idx,
        idx_to_node,
        endpoint_indices,
        row_indices,
        col_indices,
        data,
        num_nodes,
    ) = _prepare_sparse_graph_data(graph, endpoint_type, edge_types, weight)

    if not data:
        return _return_original_graph(graph)

    if not endpoint_indices:
        logger.warning("No nodes of type '%s' found.", endpoint_type)
        return _return_original_graph(graph)

    # Build sparse matrix
    adj_matrix = sparse.csr_matrix((data, (row_indices, col_indices)), shape=(num_nodes, num_nodes))

    # Run Dijkstra's algorithm
    dist_matrix = csgraph.dijkstra(
        csgraph=adj_matrix,
        directed=directed or (isinstance(graph, nx.Graph) and graph.is_directed()),
        indices=endpoint_indices,
        limit=threshold,
    )

    # Extract metapath edges
    new_edges_data = _extract_metapath_edges(
        dist_matrix,
        endpoint_indices,
        idx_to_node,
        min_threshold,
        threshold,
        graph,
        directed,
        target_edge_type_tuple,
        weight,
    )

    # Construct and return result
    return _construct_metapath_result(
        graph,
        new_edges_data,
        as_nx,
        directed,
        multigraph,
        target_edge_type_tuple,
        weight,
        endpoint_type,
    )


def _return_original_graph(
    graph: (
        tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]]
        | nx.Graph
        | nx.MultiGraph
    ),
) -> (
    nx.Graph
    | nx.MultiGraph
    | tuple[
        dict[str, gpd.GeoDataFrame],
        dict[tuple[str, str, str], gpd.GeoDataFrame],
    ]
):
    """
    Return the original graph as-is.

    This function is only called when as_nx=False, so it just returns
    the graph without any conversion.

    Parameters
    ----------
    graph : tuple or networkx.Graph or networkx.MultiGraph
        Input graph in either GeoDataFrame tuple or NetworkX format.

    Returns
    -------
    nx.Graph or nx.MultiGraph or tuple
        The original graph unchanged.
    """
    # This function is only called with as_nx=False, so just return the graph as-is
    return graph


def _prepare_sparse_graph_data(
    graph: (
        tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]]
        | nx.Graph
        | nx.MultiGraph
    ),
    endpoint_type: str,
    edge_types: list[tuple[str, str, str]] | None,
    weight: str,
) -> tuple[
    dict[object, int],
    dict[int, object],
    list[int],
    list[int],
    list[int],
    list[float],
    int,
]:
    """
    Prepare sparse graph data by delegating to appropriate handler.

    Dispatches to GeoDataFrame or NetworkX specific handler based on input type.

    Parameters
    ----------
    graph : tuple or networkx.Graph or networkx.MultiGraph
        Input graph.
    endpoint_type : str
        Node type to identify as endpoints.
    edge_types : list[tuple[str, str, str]] or None
        Edge types to include in traversal.
    weight : str
        Edge attribute to use as weight.

    Returns
    -------
    tuple
        Seven-element tuple containing node mappings, indices, and edge data.
    """
    if isinstance(graph, tuple):
        return _prepare_graph_data_from_gdf(graph, endpoint_type, edge_types, weight)
    return _prepare_graph_data_from_nx(graph, endpoint_type, edge_types, weight)


def _prepare_graph_data_from_gdf(
    graph: tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]],
    endpoint_type: str,
    edge_types: list[tuple[str, str, str]] | None,
    weight: str,
) -> tuple[
    dict[object, int],
    dict[int, object],
    list[int],
    list[int],
    list[int],
    list[float],
    int,
]:
    """
    Build sparse graph data from GeoDataFrame tuple.

    Extracts node indices and edge weights from GeoDataFrame dictionaries.

    Parameters
    ----------
    graph : tuple[dict[str, GeoDataFrame], dict[tuple[str, str, str], GeoDataFrame]]
        Input graph as GeoDataFrame tuple.
    endpoint_type : str
        Node type to identify as endpoints.
    edge_types : list[tuple[str, str, str]] or None
        Edge types to include in traversal.
    weight : str
        Edge attribute to use as weight.

    Returns
    -------
    tuple
        Seven-element tuple containing node mappings, indices, and edge data.
    """
    nodes_dict, edges_dict = graph
    node_to_idx: dict[object, int] = {}
    idx_to_node: dict[int, object] = {}
    endpoint_indices: list[int] = []
    row_indices: list[int] = []
    col_indices: list[int] = []
    data: list[float] = []
    current_idx = 0

    # Build node index
    for n_type, gdf in nodes_dict.items():
        for node_id in gdf.index:
            key = (n_type, node_id)
            if key not in node_to_idx:
                node_to_idx[key] = current_idx
                idx_to_node[current_idx] = key
                current_idx += 1

            if n_type == endpoint_type:
                endpoint_indices.append(node_to_idx[key])

    # Collect edges
    for e_type, gdf in edges_dict.items():
        if edge_types and e_type not in edge_types:
            continue

        src_type, _, dst_type = e_type
        us = gdf.index.get_level_values(0)
        vs = gdf.index.get_level_values(1)
        ws = gdf[weight].to_numpy()

        for u, v, w in zip(us, vs, ws, strict=False):
            u_key = (src_type, u)
            v_key = (dst_type, v)
            u_idx = node_to_idx[u_key]
            v_idx = node_to_idx[v_key]
            w_val = float(w)

            row_indices.append(u_idx)
            col_indices.append(v_idx)
            data.append(w_val)

    return (
        node_to_idx,
        idx_to_node,
        endpoint_indices,
        row_indices,
        col_indices,
        data,
        current_idx,
    )


def _prepare_graph_data_from_nx(
    graph: nx.Graph | nx.MultiGraph,
    endpoint_type: str,
    edge_types: list[tuple[str, str, str]] | None,
    weight: str,
) -> tuple[
    dict[object, int],
    dict[int, object],
    list[int],
    list[int],
    list[int],
    list[float],
    int,
]:
    """
    Build sparse graph data from NetworkX graph.

    Extracts node indices and edge weights from NetworkX graph structure.

    Parameters
    ----------
    graph : networkx.Graph or networkx.MultiGraph
        Input NetworkX graph.
    endpoint_type : str
        Node type to identify as endpoints.
    edge_types : list[tuple[str, str, str]] or None
        Edge types to include in traversal.
    weight : str
        Edge attribute to use as weight.

    Returns
    -------
    tuple
        Seven-element tuple containing node mappings, indices, and edge data.
    """
    node_to_idx: dict[object, int] = {}
    idx_to_node: dict[int, object] = {}
    endpoint_indices: list[int] = []
    row_indices: list[int] = []
    col_indices: list[int] = []
    data: list[float] = []
    current_idx = 0

    # Build node index
    for n, d in graph.nodes(data=True):
        if n not in node_to_idx:
            node_to_idx[n] = current_idx
            idx_to_node[current_idx] = n
            current_idx += 1

        if d.get("node_type") == endpoint_type:
            endpoint_indices.append(node_to_idx[n])

    edges_iter = (
        graph.edges(data=True, keys=True) if graph.is_multigraph() else graph.edges(data=True)
    )

    min_weights: dict[tuple[int, int], float] = {}

    for edge in edges_iter:
        if graph.is_multigraph():
            u, v, _, d = edge
        else:
            u, v, d = edge

        if edge_types and d.get("edge_type") not in edge_types:
            continue

        w_val = d.get(weight)
        u_idx = node_to_idx[u]
        v_idx = node_to_idx[v]
        w_float = float(w_val)

        pair = (u_idx, v_idx)
        if pair not in min_weights or w_float < min_weights[pair]:
            min_weights[pair] = w_float

    for (u_idx, v_idx), w in min_weights.items():
        row_indices.append(u_idx)
        col_indices.append(v_idx)
        data.append(w)

    return (
        node_to_idx,
        idx_to_node,
        endpoint_indices,
        row_indices,
        col_indices,
        data,
        current_idx,
    )


def _extract_metapath_edges(
    dist_matrix: np.ndarray,
    endpoint_indices: list[int],
    idx_to_node: dict[int, object],
    min_threshold: float,
    threshold: float,
    graph: (
        tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]]
        | nx.Graph
        | nx.MultiGraph
    ),
    directed: bool,
    target_edge_type: tuple[str, str, str],
    weight: str,
) -> list[dict[str, Any]]:
    """
    Extract metapath edges from Dijkstra distance matrix.

    Filters distances and creates edge records for valid connections.

    Parameters
    ----------
    dist_matrix : numpy.ndarray
        Distance matrix from Dijkstra algorithm.
    endpoint_indices : list[int]
        Indices of endpoint nodes.
    idx_to_node : dict[int, object]
        Mapping from indices to node keys.
    min_threshold : float
        Minimum distance threshold.
    threshold : float
        Maximum distance threshold.
    graph : tuple or networkx.Graph or networkx.MultiGraph
        Original input graph for node attribute lookup.
    directed : bool
        Whether graph is directed.
    target_edge_type : tuple[str, str, str]
        Edge type for new metapath edges.
    weight : str
        Edge weight attribute name.

    Returns
    -------
    list[dict[str, Any]]
        List of edge data dictionaries.
    """
    new_edges_data = []
    endpoint_indices_arr = np.array(endpoint_indices)

    for i, start_idx in enumerate(endpoint_indices):
        dists = dist_matrix[i]
        dists_to_endpoints = dists[endpoint_indices_arr]

        valid_mask = (
            (dists_to_endpoints >= min_threshold)
            & (dists_to_endpoints <= threshold)
            & (dists_to_endpoints != np.inf)
        )
        valid_mask[i] = False

        valid_target_indices_in_endpoints = np.where(valid_mask)[0]

        if valid_target_indices_in_endpoints.size > 0:
            start_node_key = idx_to_node[start_idx]

            for target_idx_in_endpoints in valid_target_indices_in_endpoints:
                target_idx = endpoint_indices[target_idx_in_endpoints]
                dist = dists_to_endpoints[target_idx_in_endpoints]
                end_node_key = idx_to_node[target_idx]

                if isinstance(graph, tuple):
                    start_node = cast("tuple[str, object]", start_node_key)[1]
                    end_node = cast("tuple[str, object]", end_node_key)[1]
                else:
                    start_node = start_node_key
                    end_node = end_node_key

                start_orig = start_node
                end_orig = end_node

                if isinstance(graph, (nx.Graph, nx.MultiGraph)):
                    start_orig = graph.nodes[start_node].get("_original_index", start_node)
                    end_orig = graph.nodes[end_node].get("_original_index", end_node)

                orig_edge_index = (start_orig, end_orig)

                if not directed and start_orig > end_orig:  # type: ignore[operator]
                    orig_edge_index = (end_orig, start_orig)

                new_edges_data.append(
                    {
                        "u": start_node,
                        "v": end_node,
                        weight: float(dist),
                        "edge_type": target_edge_type,
                        "_original_edge_index": orig_edge_index,
                    }
                )
    return new_edges_data


def _add_edges_to_nx_graph(
    nx_graph: nx.Graph | nx.MultiGraph,
    new_edges_data: list[dict[str, Any]],
    target_edge_type: tuple[str, str, str],
    weight: str,
) -> nx.Graph | nx.MultiGraph:
    """
    Add metapath edges to a NetworkX graph.

    This function adds edges from the provided edge data list to an existing
    NetworkX graph and updates the graph's edge type registry.

    Parameters
    ----------
    nx_graph : networkx.Graph or networkx.MultiGraph
        NetworkX graph to add edges to.
    new_edges_data : list[dict[str, Any]]
        List of edge data dictionaries.
    target_edge_type : tuple[str, str, str]
        Edge type for new metapath edges.
    weight : str
        Edge weight attribute name.

    Returns
    -------
    networkx.Graph or networkx.MultiGraph
        Graph with edges added.
    """
    edges_to_add = [
        (
            e["u"],
            e["v"],
            {
                weight: e[weight],
                "edge_type": e["edge_type"],
                "_original_edge_index": e["_original_edge_index"],
            },
        )
        for e in new_edges_data
    ]
    nx_graph.add_edges_from(edges_to_add)

    if "edge_types" in nx_graph.graph and target_edge_type not in nx_graph.graph["edge_types"]:
        nx_graph.graph["edge_types"].append(target_edge_type)

    return nx_graph


def _create_edge_gdf_from_data(
    new_edges_data: list[dict[str, Any]],
    endpoint_gdf: gpd.GeoDataFrame,
    weight: str,
) -> gpd.GeoDataFrame:
    """
    Create GeoDataFrame for metapath edges.

    This function constructs a GeoDataFrame from edge data dictionaries,
    creating LineString geometries between endpoint nodes.

    Parameters
    ----------
    new_edges_data : list[dict[str, Any]]
        List of edge data dictionaries.
    endpoint_gdf : geopandas.GeoDataFrame
        GeoDataFrame of endpoint nodes for geometry creation.
    weight : str
        Edge weight attribute name.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing the new edges.
    """
    edges_df = pd.DataFrame(new_edges_data)

    u_geoms = endpoint_gdf.loc[edges_df["u"]].geometry.to_numpy()
    v_geoms = endpoint_gdf.loc[edges_df["v"]].geometry.to_numpy()

    geometries = [
        LineString([u.centroid, v.centroid]) for u, v in zip(u_geoms, v_geoms, strict=False)
    ]

    new_gdf = gpd.GeoDataFrame(
        edges_df[[weight, "edge_type"]],
        geometry=geometries,
        crs=endpoint_gdf.crs,
    )

    new_gdf.index = pd.MultiIndex.from_frame(edges_df[["u", "v"]])
    new_gdf.index.names = ["u", "v"]

    return new_gdf


def _construct_metapath_result(
    graph: (
        tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]]
        | nx.Graph
        | nx.MultiGraph
    ),
    new_edges_data: list[dict[str, Any]],
    as_nx: bool,
    directed: bool,
    multigraph: bool,
    target_edge_type: tuple[str, str, str],
    weight: str,
    endpoint_type: str,
) -> (
    nx.Graph
    | nx.MultiGraph
    | tuple[
        dict[str, gpd.GeoDataFrame],
        dict[tuple[str, str, str], gpd.GeoDataFrame],
    ]
):
    """
    Construct the result graph with metapath edges added.

    Adds new edges to the graph in NetworkX or GeoDataFrame format.

    Parameters
    ----------
    graph : tuple or networkx.Graph or networkx.MultiGraph
        Original input graph.
    new_edges_data : list[dict[str, Any]]
        List of new edge data to add.
    as_nx : bool
        Whether to return NetworkX format.
    directed : bool
        Whether graph is directed.
    multigraph : bool
        Whether to use MultiGraph format.
    target_edge_type : tuple[str, str, str]
        Edge type for new metapath edges.
    weight : str
        Edge weight attribute name.
    endpoint_type : str
        Node type of endpoints.

    Returns
    -------
    nx.Graph or nx.MultiGraph or tuple
        Graph with metapath edges added.
    """
    if as_nx:
        # Convert to NetworkX if needed
        if isinstance(graph, tuple):
            nodes_dict_in, edges_dict_in = graph
            nx_graph = gdf_to_nx(
                nodes_dict_in, edges_dict_in, directed=directed, multigraph=multigraph
            )
        else:
            nx_graph = graph

        # Add edges to NetworkX graph
        return _add_edges_to_nx_graph(nx_graph, new_edges_data, target_edge_type, weight)

    # Return as GeoDataFrame tuple
    if isinstance(graph, tuple):
        nodes_dict, edges_dict = graph
    else:
        nodes_dict, edges_dict = nx_to_gdf(graph)  # pragma: no cover

    if not new_edges_data:
        return nodes_dict, edges_dict

    # Create GeoDataFrame for new edges
    endpoint_gdf = nodes_dict[endpoint_type]
    new_gdf = _create_edge_gdf_from_data(new_edges_data, endpoint_gdf, weight)
    edges_dict[target_edge_type] = new_gdf

    return nodes_dict, edges_dict


def add_metapaths(
    graph: (
        tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]]
        | nx.Graph
        | nx.MultiGraph
        | None
    ) = None,
    nodes: dict[str, gpd.GeoDataFrame] | None = None,
    edges: dict[tuple[str, str, str], gpd.GeoDataFrame] | None = None,
    sequence: list[tuple[str, str, str]] | None = None,
    new_relation_name: str | None = None,
    edge_attr: str | list[str] | None = None,
    edge_attr_agg: str | object | None = "sum",
    directed: bool = False,
    trace_path: bool = False,
    multigraph: bool = False,
    as_nx: bool = False,
    **_: object,
) -> (
    nx.Graph
    | nx.MultiGraph
    | tuple[
        dict[str, gpd.GeoDataFrame],
        dict[tuple[str, str, str], gpd.GeoDataFrame],
    ]
):
    """
    Add metapath-derived edges to a heterogeneous graph.

    The operation multiplies typed adjacency tables to connect terminal node
    pairs and can aggregate additional numeric edge attributes along the way.

    Parameters
    ----------
    graph : tuple or networkx.Graph or networkx.MultiGraph, optional
        Heterogeneous graph input expressed as typed GeoDataFrame dictionaries or
        a city2graph-compatible NetworkX graph.
    nodes : dict[str, geopandas.GeoDataFrame], optional
        Dictionary of node GeoDataFrames.
    edges : dict[tuple[str, str, str], geopandas.GeoDataFrame], optional
        Dictionary of edge GeoDataFrames.
    sequence : list[tuple[str, str, str]]
        Sequence of metapath specifications; every edge type is a
        ``(src_type, relation, dst_type)`` tuple and the path must contain at
        least two steps.
    new_relation_name : str, optional
        Target edge relation name for the new metapath edges.
        If None (default), edges are named ``metapath_0``.
    edge_attr : str | list[str] | None, optional
        Numeric edge attributes to aggregate along metapaths. When ``None``, only
        path weights are produced.
    edge_attr_agg : str | object | None, optional
        Aggregation strategy for ``edge_attr`` columns. Supported values are
        ``"sum"`` and ``"mean"`` (default ``"sum"``).
    directed : bool, optional
        Treat metapaths as directed when ``True``; otherwise both edge
        directions are accepted when available in the input graph.
    trace_path : bool, optional
        When ``True``, attempt to create traced geometries. Currently ignored but
        retained for API compatibility.
    multigraph : bool, optional
        When returning NetworkX data, build a ``networkx.MultiGraph`` if ``True``.
    as_nx : bool, optional
        Return the result as a NetworkX graph when ``True``.
    **_ : object
        Ignored placeholder for future keyword extensions.

    Returns
    -------
    tuple[dict[str, geopandas.GeoDataFrame], dict[tuple[str, str, str], geopandas.GeoDataFrame]] | networkx.Graph | networkx.MultiGraph
        The graph with metapath-derived edges.
        If as_nx is False (default), returns a tuple of node and edge GeoDataFrames.
        If as_nx is True, returns a NetworkX graph (Graph or MultiGraph).

    Notes
    -----
    Legacy scaffolding for path-tracing geometries has been removed because it
    was never executed. The trace_path argument is preserved for API
    compatibility but remains a no-op while straight-line geometries are
    generated for all metapath edges.
    """
    if trace_path:  # pragma: no cover
        logger.debug("trace_path option is not implemented; ignoring request.")

    if sequence is None:
        msg = "sequence must be provided"
        raise ValueError(msg)

    nodes_dict, edges_dict = _ensure_hetero_dict(graph, nodes, edges)

    if not sequence or not edges_dict:
        return _finalize_metapath_result(
            nodes_dict,
            edges_dict,
            as_nx,
            multigraph,
            directed,
            None,
        )

    edge_attrs = _normalize_edge_attrs(edge_attr)
    aggregation = _resolve_edge_attr_agg(edge_attr_agg)

    updated_edges = dict(edges_dict)
    metapath_metadata: dict[tuple[str, str, str], dict[str, object]] = {}

    edge_key, result_gdf, metadata_entry = _materialize_metapath(
        mp_index=0,
        metapath=sequence,
        nodes=nodes_dict,
        edges=edges_dict,
        directed=directed,
        edge_attrs=edge_attrs,
        aggregation=aggregation,
        new_relation_name=new_relation_name,
    )
    updated_edges[edge_key] = result_gdf
    metapath_metadata[edge_key] = metadata_entry

    return _finalize_metapath_result(
        nodes_dict,
        updated_edges,
        as_nx,
        multigraph,
        directed,
        metapath_metadata,
    )


def _ensure_hetero_dict(
    graph: (
        tuple[
            dict[str, gpd.GeoDataFrame],
            dict[tuple[str, str, str], gpd.GeoDataFrame],
        ]
        | nx.Graph
        | nx.MultiGraph
        | None
    ) = None,
    nodes: dict[str, gpd.GeoDataFrame] | None = None,
    edges: dict[tuple[str, str, str], gpd.GeoDataFrame] | None = None,
) -> tuple[
    dict[str, gpd.GeoDataFrame],
    dict[tuple[str, str, str], gpd.GeoDataFrame],
]:
    """
    Normalize supported inputs to hetero GeoDataFrame dictionaries.

    Ensures callers can work with a predictable pair of hetero mappings.

    Parameters
    ----------
    graph : tuple or networkx.Graph or networkx.MultiGraph, optional
        Heterogeneous graph representation.
    nodes : dict[str, geopandas.GeoDataFrame], optional
        Dictionary of node GeoDataFrames.
    edges : dict[tuple[str, str, str], geopandas.GeoDataFrame], optional
        Dictionary of edge GeoDataFrames.

    Returns
    -------
    tuple[dict[str, geopandas.GeoDataFrame], dict[tuple[str, str, str], geopandas.GeoDataFrame]]
        Normalised node and edge dictionaries.
    """
    if graph is None and nodes is None:
        msg = "Either 'graph' or 'nodes' (and optionally 'edges') must be provided."
        raise ValueError(msg)

    if nodes is not None:
        if graph is not None:
            msg = "Cannot provide both 'graph' and 'nodes'/'edges'."
            raise ValueError(msg)
        return nodes, edges or {}

    if isinstance(graph, tuple):
        if len(graph) != 2:
            msg = "Graph tuple must contain (nodes_dict, edges_dict)"
            raise ValueError(msg)

        nodes_dict, raw_edges = graph

        if not isinstance(nodes_dict, dict):
            msg = "nodes_dict must be a dictionary"
            raise TypeError(msg)

        if raw_edges is None:
            empty_edges: dict[tuple[str, str, str], gpd.GeoDataFrame] = {}
            return nodes_dict, empty_edges

        if not isinstance(raw_edges, dict):
            msg = "edges_dict must be a dictionary"
            raise TypeError(msg)

        normalized_edges = cast(
            "dict[tuple[str, str, str], gpd.GeoDataFrame]",
            raw_edges,
        )

        return nodes_dict, normalized_edges

    if isinstance(graph, (nx.Graph, nx.MultiGraph, nx.DiGraph, nx.MultiDiGraph)):
        validate_nx(graph)
        nodes_data, edges_data = nx_to_gdf(graph)

        if not isinstance(nodes_data, dict):
            msg = "add_metapaths requires a heterogeneous graph with typed nodes"
            raise TypeError(msg)

        if edges_data is None:
            return nodes_data, {}

        if not isinstance(edges_data, dict):
            msg = "add_metapaths requires a heterogeneous graph with typed edges"
            raise TypeError(msg)

        normalized_edges_graph = cast(
            "dict[tuple[str, str, str], gpd.GeoDataFrame]",
            edges_data,
        )
        return nodes_data, normalized_edges_graph

    msg = "Unsupported graph input type for add_metapaths"
    raise TypeError(msg)


@dataclass(slots=True)
class _EdgeAttrAggregation:
    """
    Container describing how to reduce edge attributes along a metapath.

    Stores the callable reducers used during metapath aggregation.
    """

    tag: object
    row_reducer: Callable[[pd.DataFrame], pd.Series]
    group_reducer: str | Callable[[pd.Series], float]


def _resolve_edge_attr_agg(
    edge_attr_agg: str | object | None,
) -> _EdgeAttrAggregation:
    """
    Normalise the aggregation option for edge attributes.

    Translate the user supplied definition into reusable reducers.

    Parameters
    ----------
    edge_attr_agg : str | object | None
        Aggregation strategy requested by the caller. Supported values are
        ``"sum"``, ``"mean"``, a callable reducer, or ``None`` which defaults to
        ``"sum"``.

    Returns
    -------
    _EdgeAttrAggregation
        Resolved aggregation metadata containing callable reducers for row and
        group operations.

    Raises
    ------
    ValueError
        If a string option is supplied that is not ``"sum"`` or ``"mean"``.
    TypeError
        If the option is neither a recognised string, a callable, nor ``None``.
    """
    option = edge_attr_agg if edge_attr_agg is not None else "sum"

    if isinstance(option, str):
        normalized = option.lower()
        if normalized == "sum":
            return _EdgeAttrAggregation("sum", _row_reduce_sum, "sum")
        if normalized == "mean":
            return _EdgeAttrAggregation("mean", _row_reduce_mean, "mean")
        msg = f"Unsupported edge_attr_agg '{option}'"
        raise ValueError(msg)

    if callable(option):
        return _EdgeAttrAggregation(
            option,
            functools.partial(_row_reduce_callable, func=option),
            functools.partial(_group_reduce_callable, func=option),
        )

    msg = "edge_attr_agg must be 'sum', 'mean', a callable, or None"
    raise TypeError(msg)


def _row_reduce_sum(block: pd.DataFrame) -> pd.Series:
    """
    Return row-wise sums for a hop attribute block.

    Missing values are treated as zeros before summation.

    Parameters
    ----------
    block : pandas.DataFrame
        Normalised attribute columns for a single hop across multiple paths.

    Returns
    -------
    pandas.Series
        Row-wise sums with missing values treated as zeros.
    """
    numeric = block.apply(pd.to_numeric, errors="coerce")
    result = numeric.fillna(0.0).sum(axis=1)
    return cast("pd.Series", result)


def _row_reduce_mean(block: pd.DataFrame) -> pd.Series:
    """
    Return row-wise means while skipping missing values.

    Each row is reduced to the arithmetic mean of its valid entries.

    Parameters
    ----------
    block : pandas.DataFrame
        Normalised attribute columns for a single hop across multiple paths.

    Returns
    -------
    pandas.Series
        Row-wise means calculated with ``NaN`` values ignored.
    """
    numeric = block.apply(pd.to_numeric, errors="coerce")
    result = numeric.mean(axis=1, skipna=True)
    return cast("pd.Series", result)


def _row_reduce_callable(
    block: pd.DataFrame,
    *,
    func: Callable[[np.ndarray], float],
) -> pd.Series:
    """
    Apply a custom reducer to each row of a hop attribute block.

    Allows user supplied reducers to participate in per-path aggregation.

    Parameters
    ----------
    block : pandas.DataFrame
        Normalised attribute columns for a single hop across multiple paths.
    func : Callable[[numpy.ndarray], float]
        Callable that reduces the non-null numeric values in a row to a scalar.

    Returns
    -------
    pandas.Series
        Row-wise reductions produced by ``func``.
    """
    numeric = block.apply(pd.to_numeric, errors="coerce")
    # Avoid passing `func` as a keyword to DataFrame.apply since it clashes with
    # the DataFrame.apply(func=...) parameter name itself, causing
    # "multiple values for argument 'func'" TypeError. Use a lambda to forward
    # the user-supplied reducer to our row helper.
    result = numeric.apply(lambda row: _apply_callable_row(row, func=func), axis=1)
    return cast("pd.Series", result)


def _apply_callable_row(
    row: pd.Series,
    *,
    func: Callable[[np.ndarray], float],
) -> float:
    """
    Reduce a single hop row using ``func`` while handling empty inputs.

    Helper used by :func:`_row_reduce_callable` to evaluate user reducers.

    Parameters
    ----------
    row : pandas.Series
        Hop attribute values for a single metapath traversal.
    func : Callable[[numpy.ndarray], float]
        Callable that reduces numeric values to a scalar result.

    Returns
    -------
    float
        Reduced value for the row; ``NaN`` when all values are missing.
    """
    valid = row.dropna().to_numpy()
    if valid.size == 0:
        return float("nan")
    return float(func(valid))


def _group_reduce_callable(
    series: pd.Series,
    *,
    func: Callable[[np.ndarray], float],
) -> float:
    """
    Reduce grouped path values with ``func`` while ignoring missing data.

    Ensures custom reducers can be applied during terminal node aggregation.

    Parameters
    ----------
    series : pandas.Series
        Row reductions belonging to the same terminal node pair.
    func : Callable[[numpy.ndarray], float]
        Callable that combines numeric values into a scalar summary.

    Returns
    -------
    float
        Aggregated value for the group; ``NaN`` when the group is empty.
    """
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float(func(numeric.to_numpy()))


def _safe_linestring(start_geom: object, end_geom: object) -> LineString | None:
    """
    Safely build a ``LineString`` between two geometries when possible.

    Invalid or missing geometries result in ``None``.

    Parameters
    ----------
    start_geom : object
        Geometry for the source node of the metapath edge.
    end_geom : object
        Geometry for the destination node of the metapath edge.

    Returns
    -------
    shapely.geometry.LineString | None
        Straight segment between the input geometries, or ``None`` when either
        geometry is missing or invalid.
    """
    if start_geom is None or end_geom is None:
        return None
    if getattr(start_geom, "is_empty", False) or getattr(end_geom, "is_empty", False):
        return None
    try:
        return LineString([start_geom, end_geom])
    except (TypeError, ValueError):
        return None


def _normalize_edge_attrs(edge_attr: str | list[str] | None) -> list[str] | None:
    """
    Normalise edge attribute selection to a list.

    Produces predictable iterable input for downstream helpers.

    Parameters
    ----------
    edge_attr : str | list[str] | None
        User-supplied edge attribute selector, a single column name, a list of
        names, or ``None``.

    Returns
    -------
    list[str] | None
        Normalised list of attribute names, or ``None`` when no attributes were
        requested.
    """
    if edge_attr is None:
        return None
    if isinstance(edge_attr, str):
        return [edge_attr]
    return list(edge_attr)


def _materialize_metapath(
    *,
    mp_index: int,
    metapath: list[tuple[str, str, str]],
    nodes: dict[str, gpd.GeoDataFrame],
    edges: dict[tuple[str, str, str], gpd.GeoDataFrame],
    directed: bool,
    edge_attrs: list[str] | None,
    aggregation: _EdgeAttrAggregation,
    new_relation_name: str | None = None,
) -> tuple[tuple[str, str, str], gpd.GeoDataFrame, dict[str, object]]:
    """
    Materialise a single metapath into an aggregated GeoDataFrame.

    Collects hop data, aggregates attributes, and attaches straight geometries.

    Parameters
    ----------
    mp_index : int
        Index of the metapath.
    metapath : list[tuple[str, str, str]]
        List of edge types defining the metapath.
    nodes : dict[str, geopandas.GeoDataFrame]
        Dictionary of node GeoDataFrames.
    edges : dict[tuple[str, str, str], geopandas.GeoDataFrame]
        Dictionary of edge GeoDataFrames.
    directed : bool
        Whether the graph is directed.
    edge_attrs : list[str] or None
        List of edge attributes to aggregate.
    aggregation : _EdgeAttrAggregation
        Aggregation strategy.
    new_relation_name : str, optional
        Target edge relation name.

    Returns
    -------
    tuple
        Tuple containing the edge key, the result GeoDataFrame, and metadata.
    """
    if len(metapath) < 2:
        msg = "Each metapath must contain at least two edge types"
        raise ValueError(msg)

    start_type = metapath[0][0]
    end_type = metapath[-1][-1]

    if new_relation_name is not None:
        edge_key = (start_type, new_relation_name, end_type)
    else:
        edge_key = (start_type, f"metapath_{mp_index}", end_type)
    metadata = {
        "metapath_spec": tuple(metapath),
        "edge_attr": None if edge_attrs is None else tuple(edge_attrs),
        "edge_attr_agg": aggregation.tag,
    }

    # 1. Build canonical frames for each hop
    frames: list[pd.DataFrame] = []
    start_index_name = f"{start_type}_id"
    end_index_name = f"{end_type}_id"

    for step_idx, edge_type in enumerate(metapath):
        edge_gdf, reversed_lookup = _get_edge_frame(edges, edge_type, directed)

        # Update index names from the first/last hop
        if step_idx == 0:
            idx_name = edge_gdf.index.names[1 if reversed_lookup else 0]
            start_index_name = _normalise_index_name(idx_name, f"{start_type}_id")
        if step_idx == len(metapath) - 1:
            idx_name = edge_gdf.index.names[0 if reversed_lookup else 1]
            end_index_name = _normalise_index_name(idx_name, f"{end_type}_id")

        frame = _build_hop_frame(
            edge_gdf=edge_gdf,
            step_idx=step_idx,
            reversed_lookup=reversed_lookup,
            edge_attrs=edge_attrs,
        )
        if frame.empty:
            return (
                edge_key,
                _empty_metapath_gdf(
                    nodes, start_type, end_type, edge_attrs, start_index_name, end_index_name
                ),
                metadata,
            )
        frames.append(frame)

    # 2. Join hops
    joined = frames[0]
    for idx in range(1, len(frames)):
        joined = joined.merge(
            frames[idx],
            left_on=f"dst_{idx - 1}",
            right_on=f"src_{idx}",
            how="inner",
            copy=False,
        )
        # Drop intermediate join columns to save memory
        joined = joined.drop(columns=[f"dst_{idx - 1}", f"src_{idx}"], errors="ignore")

        if joined.empty:
            return (
                edge_key,
                _empty_metapath_gdf(
                    nodes, start_type, end_type, edge_attrs, start_index_name, end_index_name
                ),
                metadata,
            )

    # 3. Aggregate paths
    aggregated = _aggregate_paths(
        joined,
        step_count=len(frames),
        edge_attrs=edge_attrs,
        aggregation=aggregation,
        start_index_name=start_index_name,
        end_index_name=end_index_name,
    )

    if aggregated.empty:
        return (
            edge_key,
            _empty_metapath_gdf(
                nodes, start_type, end_type, edge_attrs, start_index_name, end_index_name
            ),
            metadata,
        )

    # 4. Attach geometry
    result = _attach_metapath_geometry(aggregated, nodes, start_type, end_type)
    return edge_key, result, metadata


def _get_edge_frame(
    edges: dict[tuple[str, str, str], gpd.GeoDataFrame],
    edge_type: tuple[str, str, str],
    directed: bool,
) -> tuple[gpd.GeoDataFrame, bool]:
    """
    Fetch the GeoDataFrame for an edge type, optionally using the reverse.

    Checks for the edge type in the dictionary. If not found and the graph is
    undirected, checks for the reverse edge type.

    Parameters
    ----------
    edges : dict[tuple[str, str, str], geopandas.GeoDataFrame]
        Dictionary of edge GeoDataFrames.
    edge_type : tuple[str, str, str]
        The edge type to fetch.
    directed : bool
        Whether the graph is directed.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, bool]
        The edge GeoDataFrame and a boolean indicating if it is reversed.
    """
    if edge_type in edges:
        return edges[edge_type], False

    if not directed:
        reverse_key = (edge_type[2], edge_type[1], edge_type[0])
        if reverse_key in edges:
            return edges[reverse_key], True

    msg = f"Edge type {edge_type} not found in edges dictionary"
    raise KeyError(msg)


def _build_hop_frame(
    *,
    edge_gdf: gpd.GeoDataFrame,
    step_idx: int,
    reversed_lookup: bool,
    edge_attrs: list[str] | None,
) -> pd.DataFrame:
    """
    Convert one hop into a canonical DataFrame used for joins.

    Extracts source and destination indices and optional attributes.

    Parameters
    ----------
    edge_gdf : geopandas.GeoDataFrame
        The edge GeoDataFrame for this hop.
    step_idx : int
        The index of the current step in the metapath.
    reversed_lookup : bool
        Whether the edge is being traversed in reverse.
    edge_attrs : list[str] or None
        List of edge attributes to include.

    Returns
    -------
    pandas.DataFrame
        Canonical DataFrame with source and destination columns.
    """
    if not isinstance(edge_gdf.index, pd.MultiIndex) or edge_gdf.index.nlevels < 2:
        msg = "Edge GeoDataFrame must have a two-level MultiIndex"
        raise ValueError(msg)

    src_level = 1 if reversed_lookup else 0
    dst_level = 0 if reversed_lookup else 1

    src_col = f"src_{step_idx}"
    dst_col = f"dst_{step_idx}"

    data = {
        src_col: edge_gdf.index.get_level_values(src_level).to_numpy(),
        dst_col: edge_gdf.index.get_level_values(dst_level).to_numpy(),
    }

    if edge_attrs:
        missing_attrs = [attr for attr in edge_attrs if attr not in edge_gdf.columns]
        if missing_attrs:
            msg = f"Edge attribute(s) {missing_attrs} missing in metapath steps"
            raise KeyError(msg)
        for attr in edge_attrs:
            data[f"{attr}_step{step_idx}"] = edge_gdf[attr].to_numpy()

    return pd.DataFrame(data)


def _aggregate_paths(
    combined: pd.DataFrame,
    *,
    step_count: int,
    edge_attrs: list[str] | None,
    aggregation: _EdgeAttrAggregation,
    start_index_name: str,
    end_index_name: str,
) -> pd.DataFrame:
    """
    Group joined paths into terminal node pairs with aggregated weights.

    Aggregates weights and optional attributes for each path.

    Parameters
    ----------
    combined : pandas.DataFrame
        DataFrame containing all joined paths.
    step_count : int
        Number of steps in the metapath.
    edge_attrs : list[str] or None
        List of edge attributes to aggregate.
    aggregation : _EdgeAttrAggregation
        Aggregation strategy for edge attributes.
    start_index_name : str
        Name of the start node index.
    end_index_name : str
        Name of the end node index.

    Returns
    -------
    pandas.DataFrame
        Aggregated DataFrame with weights.
    """
    src_col = "src_0"
    dst_col = f"dst_{step_count - 1}"

    # Prepare aggregation dictionary
    agg_map: dict[str, str | Callable[[pd.Series], float]] = {"weight": "sum"}

    # Base workload with path count (weight=1 for each path)
    workload_data = {
        "src": combined[src_col].to_numpy(),
        "dst": combined[dst_col].to_numpy(),
        "weight": np.ones(len(combined), dtype=float),
    }

    if edge_attrs:
        for attr in edge_attrs:
            # Collect columns for this attribute across all steps
            step_columns = [
                f"{attr}_step{i}"
                for i in range(step_count)
                if f"{attr}_step{i}" in combined.columns
            ]

            # Row reduction (e.g. sum of times along the path)
            block = combined[step_columns]
            workload_data[attr] = aggregation.row_reducer(block).to_numpy()
            agg_map[attr] = aggregation.group_reducer

    workload = pd.DataFrame(workload_data)

    # Group by terminal nodes and aggregate (e.g. sum of weights = number of paths)
    aggregated = workload.groupby(["src", "dst"], sort=False).agg(agg_map)

    if aggregated.empty:
        return _empty_metapath_frame(edge_attrs, start_index_name, end_index_name)

    aggregated.index = aggregated.index.set_names([start_index_name, end_index_name])
    return aggregated


def _empty_metapath_frame(
    edge_attrs: list[str] | None,
    start_index_name: str,
    end_index_name: str,
) -> pd.DataFrame:
    """
    Create an empty aggregation frame with a consistent schema.

    Ensures downstream consumers receive predictable column ordering and index
    names even when no metapath traversals are available.

    Parameters
    ----------
    edge_attrs : list[str] | None
        Edge attributes expected in the aggregated output.
    start_index_name : str
        MultiIndex level name representing the metapath start nodes.
    end_index_name : str
        MultiIndex level name representing the metapath end nodes.

    Returns
    -------
    pandas.DataFrame
        Empty DataFrame with the requested columns and index structure.
    """
    columns = ["weight"] + (edge_attrs if edge_attrs else [])
    frame = pd.DataFrame(columns=columns)
    frame.index = pd.MultiIndex.from_tuples([], names=[start_index_name, end_index_name])
    return frame


def _empty_metapath_gdf(
    nodes: dict[str, gpd.GeoDataFrame],
    start_type: str,
    end_type: str,
    edge_attrs: list[str] | None,
    start_index_name: str,
    end_index_name: str,
) -> gpd.GeoDataFrame:
    """
    Create an empty GeoDataFrame placeholder for unmet metapaths.

    Ensures downstream consumers receive consistent structure even with no data.

    Parameters
    ----------
    nodes : dict[str, geopandas.GeoDataFrame]
        Mapping of node types to their GeoDataFrames.
    start_type : str
        Node type at the beginning of the metapath.
    end_type : str
        Node type at the end of the metapath.
    edge_attrs : list[str] | None
        Optional edge attributes expected in the result.
    start_index_name : str
        Name for the source level in the result index.
    end_index_name : str
        Name for the destination level in the result index.

    Returns
    -------
    geopandas.GeoDataFrame
        Empty GeoDataFrame matching the expected schema for the metapath.
    """
    start_nodes = nodes.get(start_type)
    end_nodes = nodes.get(end_type)

    crs = None
    if start_nodes is not None and start_nodes.crs:
        crs = start_nodes.crs
    elif end_nodes is not None:
        crs = end_nodes.crs

    base_frame = _empty_metapath_frame(edge_attrs, start_index_name, end_index_name)
    geometry = gpd.GeoSeries([], crs=crs)
    return gpd.GeoDataFrame(
        base_frame.assign(geometry=geometry),
        geometry="geometry",
        crs=crs,
    )


def _attach_metapath_geometry(
    aggregated: pd.DataFrame,
    nodes: dict[str, gpd.GeoDataFrame],
    start_type: str,
    end_type: str,
) -> gpd.GeoDataFrame:
    """
    Attach straight geometries between terminal node pairs.

    Generates straight-line links that connect the start and end node positions.

    Parameters
    ----------
    aggregated : pandas.DataFrame
        Aggregated metapath table produced by :func:`_reduce_metapath_paths`.
    nodes : dict[str, geopandas.GeoDataFrame]
        Mapping of node types to their GeoDataFrames.
    start_type : str
        Node type for the source nodes.
    end_type : str
        Node type for the destination nodes.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing aggregated metrics and straight-line geometries.

    Raises
    ------
    KeyError
        If node GeoDataFrames for ``start_type`` or ``end_type`` are missing.
    """
    start_nodes = nodes.get(start_type)
    end_nodes = nodes.get(end_type)

    if start_nodes is None or end_nodes is None:
        msg = f"Missing node GeoDataFrame for start '{start_type}' or end '{end_type}'"
        raise KeyError(msg)

    start_series = start_nodes.geometry.reindex(aggregated.index.get_level_values(0))
    end_series = end_nodes.geometry.reindex(aggregated.index.get_level_values(1))

    geometries = [
        _safe_linestring(start_geom, end_geom)
        for start_geom, end_geom in zip(start_series, end_series, strict=False)
    ]

    crs = start_nodes.crs or end_nodes.crs
    result = gpd.GeoDataFrame(aggregated, geometry=geometries, crs=crs)

    if "weight" in result.columns and not result.empty:
        weight_series = result["weight"]
        if weight_series.notna().all():
            rounded = weight_series.round()
            if np.allclose(weight_series.to_numpy(), rounded.to_numpy()):
                result["weight"] = rounded.astype(int)

    return result


def _normalise_index_name(raw_name: object, fallback: str) -> str:
    """
    Return a sensible index level name for merged metapaths.

    Provides deterministic naming when original indices are unnamed.

    Parameters
    ----------
    raw_name : object
        Original index level name extracted from a ``MultiIndex``.
    fallback : str
        Name to use when the source level is unnamed.

    Returns
    -------
    str
        Normalised index level name.
    """
    if isinstance(raw_name, str) and raw_name:
        return raw_name
    if raw_name is None:
        return fallback
    return str(raw_name)


def _finalize_metapath_result(
    nodes_dict: dict[str, gpd.GeoDataFrame],
    edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    as_nx: bool,
    multigraph: bool,
    directed: bool,
    metadata: dict[tuple[str, str, str], dict[str, object]] | None,
) -> (
    nx.Graph
    | nx.MultiGraph
    | tuple[
        dict[str, gpd.GeoDataFrame],
        dict[tuple[str, str, str], gpd.GeoDataFrame],
    ]
):
    """
    Return hetero dictionaries or convert to NetworkX with metadata.

    Handles the optional conversion to NetworkX while keeping metadata in sync.

    Parameters
    ----------
    nodes_dict : dict[str, geopandas.GeoDataFrame]
        Mapping of node types to their GeoDataFrames.
    edges_dict : dict[tuple[str, str, str], geopandas.GeoDataFrame]
        Mapping of edge types to their GeoDataFrames.
    as_nx : bool
        Whether to convert the result to a NetworkX graph.
    multigraph : bool
        Build a :class:`networkx.MultiGraph` when returning NetworkX data.
    directed : bool
        Produce a directed NetworkX graph if requested.
    metadata : dict[tuple[str, str, str], dict[str, object]] | None
        Metadata describing metapath-derived edges to attach to the graph.

    Returns
    -------
    networkx.Graph | networkx.MultiGraph | tuple[dict[str, geopandas.GeoDataFrame], dict[tuple[str, str, str], geopandas.GeoDataFrame]]
        Either the hetero GeoDataFrame dictionaries or, when ``as_nx`` is set,
        a NetworkX graph updated with the provided metadata.
    """
    if not as_nx:
        return nodes_dict, edges_dict

    nx_graph = gdf_to_nx(
        nodes=nodes_dict,
        edges=edges_dict,
        multigraph=multigraph,
        directed=directed,
    )

    if metadata:
        existing = nx_graph.graph.get("metapath_dict")
        if isinstance(existing, dict):
            existing.update(metadata)
        else:
            existing = dict(metadata)
        nx_graph.graph["metapath_dict"] = existing

    return nx_graph
