"""
Core Utilities Module.

This module provides essential utilities for graph conversion, data validation,
and spatial analysis operations. It serves as the foundation for the city2graph
package, offering a standardized data format for handling geospatial relations
across GeoPandas, NetworkX objects, and eventually PyTorch Geometric objects.
The module enables seamless integration between different graph representations
and geospatial data formats through robust data structures and conversion functions.
"""

# Standard library imports
import logging
import math
import typing
import warnings
from collections import defaultdict
from collections.abc import Iterable
from collections.abc import Sequence
from itertools import combinations
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

# Third-party imports
import geopandas as gpd
import momepy
import networkx as nx
import numpy as np
import pandas as pd
import rustworkx as rx
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import LineString
from shapely.geometry import MultiPoint
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import Polygon

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    MATPLOTLIB_AVAILABLE = False

if TYPE_CHECKING:
    import matplotlib.axes


# Import foundational classes from base module
from .base import BaseGraphConverter
from .base import GeoDataProcessor
from .base import GraphMetadata

# Public API definition
__all__ = [
    "create_isochrone",
    "create_tessellation",
    "dual_graph",
    "filter_graph_by_distance",
    "gdf_to_nx",
    "nx_to_gdf",
    "nx_to_rx",
    "plot_graph",
    "rx_to_nx",
    "validate_gdf",
    "validate_nx",
]

# Module logger configuration
logger = logging.getLogger(__name__)

# =============================================================================
# GRAPH CONVERSION ENGINE
# =============================================================================


class NxConverter(BaseGraphConverter):
    """
    Unified graph conversion engine for both homogeneous and heterogeneous graphs.

    This class provides methods to convert between GeoDataFrames and NetworkX graphs,
    supporting both homogeneous and heterogeneous graph structures. Inherits from
    BaseGraphConverter to leverage common conversion logic and validation.

    See Also
    --------
    gdf_to_nx : Convert GeoDataFrames to NetworkX graph.
    BaseGraphConverter : Base class for graph conversion.

    Examples
    --------
    >>> # Basic usage example
    >>> pass
    """

    def gdf_to_nx(
        self,
        nodes: gpd.GeoDataFrame | dict[str, gpd.GeoDataFrame] | None = None,
        edges: gpd.GeoDataFrame | dict[tuple[str, str, str], gpd.GeoDataFrame] | None = None,
    ) -> nx.Graph | nx.MultiGraph:
        """
        Convert GeoDataFrames to a NetworkX graph.

        This method serves as the main entry point for converting geospatial data,
        represented as GeoDataFrames, into a NetworkX graph. It automatically
        detects whether to create a homogeneous or heterogeneous graph based on the
        input types and dispatches to the appropriate conversion method.

        Parameters
        ----------
        nodes : geopandas.GeoDataFrame or dict[str, geopandas.GeoDataFrame], optional
            Node data. For homogeneous graphs, a single GeoDataFrame. For
            heterogeneous graphs, a dictionary mapping node type names to
            GeoDataFrames.
        edges : geopandas.GeoDataFrame or dict, optional
            Edge data. For homogeneous graphs, a single GeoDataFrame. For
            heterogeneous graphs, a dictionary mapping edge type tuples to
            GeoDataFrames.

        Returns
        -------
        networkx.Graph or networkx.MultiGraph
            A NetworkX graph object representing the spatial network.

        Raises
        ------
        ValueError
            If both nodes and edges are None.

        See Also
        --------
        nx_to_gdf : Convert a NetworkX graph back to GeoDataFrames.

        Examples
        --------
        >>> converter = NxConverter()
        >>> G_homo = converter.gdf_to_nx(nodes=nodes_gdf, edges=edges_gdf)
        >>> H_hetero = converter.gdf_to_nx(nodes=nodes_dict, edges=edges_dict)
        """
        return self.convert(nodes, edges)

    def nx_to_gdf(
        self,
        graph: nx.Graph | nx.MultiGraph,
        nodes: bool = True,
        edges: bool = True,
    ) -> (
        tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
        | tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]]
    ):
        """
        Convert a NetworkX graph to GeoDataFrames.

        This method reconstructs GeoDataFrames from a NetworkX graph, effectively
        reversing the `gdf_to_nx` conversion. It uses metadata stored within the
        graph to determine whether to reconstruct a homogeneous or heterogeneous
        set of GeoDataFrames and preserves spatial information by converting node
        positions back to geometries.

        Parameters
        ----------
        graph : networkx.Graph or networkx.MultiGraph
            The NetworkX graph to convert. Expected to have metadata in `graph.graph`.
        nodes : bool, default True
            If True, reconstructs the nodes GeoDataFrame.
        edges : bool, default True
            If True, reconstructs the edges GeoDataFrame.

        Returns
        -------
        geopandas.GeoDataFrame or tuple
            The reconstructed GeoDataFrames. The return type depends on the graph structure
            (homogeneous vs heterogeneous) and the requested components (`nodes`, `edges`).

            **Homogeneous Graphs:**

            *   If ``nodes=True`` and ``edges=True``: Returns ``(nodes_gdf, edges_gdf)``
                where both are :class:`geopandas.GeoDataFrame`.
            *   If only ``nodes=True``: Returns ``nodes_gdf`` (:class:`geopandas.GeoDataFrame`).
            *   If only ``edges=True``: Returns ``edges_gdf`` (:class:`geopandas.GeoDataFrame`).

            **Heterogeneous Graphs:**

            *   Returns ``(nodes_dict, edges_dict)`` where:

                *   ``nodes_dict`` is ``dict[str, geopandas.GeoDataFrame]`` mapping node types to GeoDataFrames.
                *   ``edges_dict`` is ``dict[tuple[str, str, str], geopandas.GeoDataFrame]`` mapping edge types to GeoDataFrames.

            Note: For heterogeneous graphs, the return value is always a tuple of dictionaries,
            even if only one component is requested (the other will be an empty dict).

        See Also
        --------
        gdf_to_nx : Convert GeoDataFrames to a NetworkX graph.

        Examples
        --------
        >>> converter = NxConverter()
        >>> nodes_gdf, edges_gdf = converter.nx_to_gdf(G_homo)
        >>> nodes_dict, edges_dict = converter.nx_to_gdf(H_hetero)
        """
        return self.reconstruct(graph, nodes, edges)

    def _extract_metadata(self, graph: nx.Graph | nx.MultiGraph) -> GraphMetadata:
        """
        Extract metadata from NetworkX graph.

        This method retrieves the `GraphMetadata` object stored within the
        `graph.graph` attribute of a NetworkX graph. This metadata is crucial
        for understanding the graph's structure (e.g., homogeneous/heterogeneous,
        CRS, original node/edge types) and for reconstructing GeoDataFrames.

        Parameters
        ----------
        graph : nx.Graph or nx.MultiGraph
            The NetworkX graph from which to extract metadata. The graph is
            expected to have a `graph` attribute containing a dictionary
            from which `GraphMetadata` can be constructed.

        Returns
        -------
        GraphMetadata
            An instance of `GraphMetadata` containing the structural and
            geospatial metadata of the graph.
        """
        return GraphMetadata.from_dict(graph.graph)

    def _convert_homogeneous(
        self,
        nodes: gpd.GeoDataFrame | None,
        edges: gpd.GeoDataFrame | None,
    ) -> nx.Graph | nx.MultiGraph:
        """
        Convert homogeneous GeoDataFrames to a NetworkX graph.

        This internal method handles the specific logic for converting a single set of
        node and edge GeoDataFrames into a homogeneous NetworkX graph. It validates
        the inputs, creates the graph structure, adds nodes and edges with their
        attributes, and stores the necessary metadata for potential reconstruction.

        Parameters
        ----------
        nodes : geopandas.GeoDataFrame or None
            The GeoDataFrame containing node data.
        edges : geopandas.GeoDataFrame or None
            The GeoDataFrame containing edge data.

        Returns
        -------
        networkx.Graph or networkx.MultiGraph
            The resulting homogeneous NetworkX graph.

        Raises
        ------
        ValueError
            If the edges GeoDataFrame is None.

        See Also
        --------
        _convert_heterogeneous : Convert heterogeneous GeoDataFrames to NetworkX graph.
        gdf_to_nx : Public interface for converting GeoDataFrames to NetworkX.

        Examples
        --------
        >>> import geopandas as gpd
        >>> from shapely.geometry import Point, LineString
        >>> nodes = gpd.GeoDataFrame({'geometry': [Point(0, 0), Point(1, 1)]})
        >>> edges = gpd.GeoDataFrame({'geometry': [LineString([(0, 0), (1, 1)])]})
        >>> converter = NxConverter()
        >>> graph = converter._convert_homogeneous(nodes, edges)
        >>> graph.number_of_nodes()
        2
        """
        # Validate inputs
        nodes, edges = self._validate_homo_inputs(nodes, edges)

        # Create graph and metadata
        graph, metadata = self._init_homo_graph(edges.crs)

        # Add nodes
        if nodes is not None:
            self._add_homogeneous_nodes(graph, nodes)
            metadata.node_geom_cols = list(nodes.select_dtypes(include=["geometry"]).columns)
            metadata.node_index_names = typing.cast(
                "list[str] | None", self._get_node_index_names(nodes)
            )

        # Add edges
        self._add_homogeneous_edges(graph, edges, nodes)
        metadata.edge_geom_cols = list(edges.select_dtypes(include=["geometry"]).columns)
        metadata.edge_index_names = typing.cast(
            "list[str] | None", _coerce_name_sequence(edges.index.names)
        )

        # Store metadata
        graph.graph.update(metadata.to_dict())
        return graph

    def _validate_homo_inputs(
        self,
        nodes: gpd.GeoDataFrame | None,
        edges: gpd.GeoDataFrame | None,
    ) -> tuple[gpd.GeoDataFrame | None, gpd.GeoDataFrame]:
        """
        Validate inputs for homogeneous graph conversion.

        Checks that input GeoDataFrames are valid and consistent with each other,
        ensuring they share the same CRS and that edges are not None.

        Parameters
        ----------
        nodes : geopandas.GeoDataFrame or None
            The nodes GeoDataFrame to validate.
        edges : geopandas.GeoDataFrame or None
            The edges GeoDataFrame to validate.

        Returns
        -------
        tuple[geopandas.GeoDataFrame | None, geopandas.GeoDataFrame]
            The validated nodes and edges GeoDataFrames.
        """
        nodes = self.processor.validate_gdf(nodes, allow_empty=True)
        edges = self.processor.validate_gdf(
            edges,
            ["LineString", "MultiLineString"],
            allow_empty=True,
        )
        # mypy: ensure edges is GeoDataFrame
        if edges is None:
            msg = "Edges GeoDataFrame cannot be None"
            raise ValueError(msg)

        self.processor.ensure_crs_consistency(nodes, edges)
        return nodes, edges

    def _init_homo_graph(self, crs: Any) -> tuple[nx.Graph | nx.MultiGraph, GraphMetadata]:  # noqa: ANN401
        """
        Initialize homogeneous graph and metadata.

        Creates the appropriate NetworkX graph instance (Graph/DiGraph/MultiGraph)
        and initializes the metadata object.

        Parameters
        ----------
        crs : Any
            The Coordinate Reference System for the graph.

        Returns
        -------
        tuple[networkx.Graph | networkx.MultiGraph, GraphMetadata]
            The initialized graph and metadata.
        """
        if self.multigraph:
            graph = nx.MultiDiGraph() if self.directed else nx.MultiGraph()
        else:
            graph = nx.DiGraph() if self.directed else nx.Graph()
        metadata = GraphMetadata(crs=crs, is_hetero=False)
        return graph, metadata

    def _get_node_index_names(self, nodes: gpd.GeoDataFrame) -> list[str | None]:
        """
        Get node index names as a list of strings.

        Extracts index names from the GeoDataFrame, handling both MultiIndex
        and standard Index cases.

        Parameters
        ----------
        nodes : geopandas.GeoDataFrame
            The nodes GeoDataFrame.

        Returns
        -------
        list[str | None]
            A list of index names.
        """
        if isinstance(nodes.index, pd.MultiIndex):
            return [str(name) for name in nodes.index.names]
        return [str(nodes.index.name) if nodes.index.name is not None else "index"]

    def _convert_heterogeneous(
        self,
        nodes_dict: dict[str, gpd.GeoDataFrame] | None,
        edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame] | None,
    ) -> nx.Graph | nx.MultiGraph:
        """
        Convert heterogeneous GeoDataFrames to a NetworkX graph.

        This internal method handles the logic for converting dictionaries of typed
        node and edge GeoDataFrames into a single, unified NetworkX graph. It adds
        nodes and edges with type information as attributes and stores comprehensive
        metadata to support reconstruction of the typed structure.

        Parameters
        ----------
        nodes_dict : dict[str, geopandas.GeoDataFrame] or None
            A dictionary mapping node type names to their GeoDataFrames.
        edges_dict : dict[tuple[str, str, str], geopandas.GeoDataFrame] or None
            A dictionary mapping edge type tuples to their GeoDataFrames.

        Returns
        -------
        networkx.Graph or networkx.MultiGraph
            The resulting heterogeneous NetworkX graph with typed nodes and edges.

        See Also
        --------
        _convert_homogeneous : Convert homogeneous GeoDataFrames to NetworkX graph.
        gdf_to_nx : Public interface for converting GeoDataFrames to NetworkX.

        Examples
        --------
        >>> import geopandas as gpd
        >>> from shapely.geometry import Point, LineString
        >>> nodes_dict = {'building': gpd.GeoDataFrame({'geometry': [Point(0, 0)]})}
        >>> edges_dict = {('building', 'connects', 'street'): gpd.GeoDataFrame({'geometry': [LineString([(0, 0), (1, 1)])]})}
        >>> converter = NxConverter()
        >>> graph = converter._convert_heterogeneous(nodes_dict, edges_dict)
        >>> graph.number_of_nodes()
        1
        """
        # Validate inputs
        self._validate_hetero_inputs(nodes_dict, edges_dict)

        # Create graph and metadata
        graph, metadata = self._init_hetero_graph(nodes_dict, edges_dict)

        # Add nodes and edges
        if nodes_dict:
            self._add_heterogeneous_nodes(graph, nodes_dict, metadata)
        if edges_dict:
            self._add_heterogeneous_edges(graph, edges_dict, metadata)

        # Store metadata
        graph.graph.update(metadata.to_dict())
        return graph

    def _validate_hetero_inputs(
        self,
        nodes_dict: dict[str, gpd.GeoDataFrame] | None,
        edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame] | None,
    ) -> None:
        """
        Validate inputs for heterogeneous graph conversion.

        Ensures that all provided node and edge GeoDataFrames are valid and
        contain the expected geometry types.

        Parameters
        ----------
        nodes_dict : dict[str, geopandas.GeoDataFrame] or None
            Dictionary of node GeoDataFrames.
        edges_dict : dict[tuple[str, str, str], geopandas.GeoDataFrame] or None
            Dictionary of edge GeoDataFrames.
        """
        if nodes_dict is not None:
            for node_type, node_gdf in nodes_dict.items():
                nodes_dict[node_type] = self.processor.validate_gdf(node_gdf, allow_empty=True)

        if edges_dict is not None:
            for edge_type, edge_gdf in edges_dict.items():
                edges_dict[edge_type] = self.processor.validate_gdf(
                    edge_gdf,
                    ["LineString", "MultiLineString"],
                    allow_empty=True,
                )

    def _init_hetero_graph(
        self,
        nodes_dict: dict[str, gpd.GeoDataFrame] | None,
        edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame] | None,
    ) -> tuple[nx.Graph | nx.MultiGraph, GraphMetadata]:
        """
        Initialize heterogeneous graph and metadata.

        Creates the appropriate NetworkX graph instance and initializes metadata
        populated with types and CRS from the input dictionaries.

        Parameters
        ----------
        nodes_dict : dict[str, geopandas.GeoDataFrame] or None
            Dictionary of node GeoDataFrames.
        edges_dict : dict[tuple[str, str, str], geopandas.GeoDataFrame] or None
            Dictionary of edge GeoDataFrames.

        Returns
        -------
        tuple[networkx.Graph | networkx.MultiGraph, GraphMetadata]
            The initialized graph and metadata.
        """
        if self.multigraph:
            graph = nx.MultiDiGraph() if self.directed else nx.MultiGraph()
        else:
            graph = nx.DiGraph() if self.directed else nx.Graph()
        metadata = GraphMetadata(is_hetero=True)

        if nodes_dict is not None and nodes_dict:
            metadata.crs = next(iter(nodes_dict.values())).crs
            metadata.node_types = list(nodes_dict.keys())

        if edges_dict is not None and edges_dict:
            if not metadata.crs:
                metadata.crs = next(iter(edges_dict.values())).crs
            metadata.edge_types = list(edges_dict.keys())

        return graph, metadata

    def _add_homogeneous_nodes(
        self,
        graph: nx.Graph | nx.MultiGraph,
        nodes_gdf: gpd.GeoDataFrame,
    ) -> None:
        """
        Add homogeneous nodes to the graph.

        This method processes a GeoDataFrame of nodes and adds them to the NetworkX
        graph. It extracts attributes, computes centroids for spatial positioning,
        and preserves the original index of the GeoDataFrame for metadata tracking.

        Parameters
        ----------
        graph : networkx.Graph or networkx.MultiGraph
            The NetworkX graph to which the nodes will be added.
        nodes_gdf : geopandas.GeoDataFrame
            The GeoDataFrame containing the node data to add.
        """
        centroids = self.processor.compute_centroids(nodes_gdf)
        node_data = nodes_gdf if self.keep_geom else nodes_gdf.drop(columns="geometry")

        # Convert to list of dictionaries for attributes
        node_attrs_list = node_data.to_dict("records")

        # Create nodes with attributes
        nodes_to_add = [
            (
                idx,
                {
                    **attrs,
                    "_original_index": orig_idx,
                    "pos": (centroid.x, centroid.y),
                },
            )
            for idx, (orig_idx, attrs, centroid) in enumerate(
                zip(nodes_gdf.index, node_attrs_list, centroids, strict=False),
            )
        ]

        graph.add_nodes_from(nodes_to_add)

    def _add_homogeneous_edges(
        self,
        graph: nx.Graph | nx.MultiGraph,
        edges_gdf: gpd.GeoDataFrame,
        nodes_gdf: gpd.GeoDataFrame,
    ) -> None:
        """
        Add homogeneous edges to the graph.

        This method adds edges from a GeoDataFrame to the NetworkX graph. It maps
        the edge geometries to the corresponding nodes in the graph, either by using
        a pre-existing node-to-coordinate mapping or by creating nodes from the edge
        endpoints if no nodes GeoDataFrame is provided. All attributes from the
        edges GeoDataFrame are carried over to the graph edges.

        Parameters
        ----------
        graph : networkx.Graph or networkx.MultiGraph
            The NetworkX graph to which the edges will be added.
        edges_gdf : geopandas.GeoDataFrame
            The GeoDataFrame containing the edge data.
        nodes_gdf : geopandas.GeoDataFrame
            The GeoDataFrame containing the node data, used for mapping edges.
        """
        if edges_gdf.empty:
            return

        # Create node mapping (either from existing nodes or from edge coordinates)
        if nodes_gdf is not None and not nodes_gdf.empty:
            coord_to_node = {
                node_data["pos"]: node_id for node_id, node_data in graph.nodes(data=True)
            }
        else:
            # Extract unique coordinates from edges and add them as nodes
            start_coords = self.processor.extract_coordinates(edges_gdf, start=True)
            end_coords = self.processor.extract_coordinates(edges_gdf, start=False)
            all_coords = pd.concat([start_coords, end_coords]).unique()

            nodes_to_add = [(coord, {"pos": coord}) for coord in all_coords]
            graph.add_nodes_from(nodes_to_add)

            # For implicit nodes, the ID is the coordinate itself
            coord_to_node = {coord: coord for coord in all_coords}

        # Map edge endpoints to node IDs
        start_coords = self.processor.extract_coordinates(edges_gdf, start=True)
        end_coords = self.processor.extract_coordinates(edges_gdf, start=False)

        u_nodes = start_coords.map(coord_to_node)
        v_nodes = end_coords.map(coord_to_node)

        # Filter valid edges
        valid_mask = u_nodes.notna() & v_nodes.notna()
        valid_edges = edges_gdf[valid_mask]
        valid_u = u_nodes[valid_mask]
        valid_v = v_nodes[valid_mask]

        self._create_edge_list(graph, valid_u, valid_v, valid_edges, self.keep_geom)

    def _add_heterogeneous_nodes(
        self,
        graph: nx.Graph | nx.MultiGraph,
        nodes_dict: dict[str, gpd.GeoDataFrame],
        metadata: "GraphMetadata",
    ) -> dict[str, int]:
        """
        Add heterogeneous nodes to the graph.

        This method iterates through a dictionary of typed node GeoDataFrames and adds
        them to the NetworkX graph. Each node is given a `node_type` attribute, and
        an offset is used to ensure unique node identifiers across all types. It also
        updates the graph's metadata with index and type information.

        Parameters
        ----------
        graph : networkx.Graph or networkx.MultiGraph
            The NetworkX graph to which the nodes will be added.
        nodes_dict : dict[str, geopandas.GeoDataFrame]
            A dictionary mapping node type names to their GeoDataFrames.
        metadata : GraphMetadata
            The metadata object to be updated with node information.

        Returns
        -------
        dict[str, int]
            A dictionary mapping each node type to its starting offset in the graph.
        """
        node_index_names = _ensure_metadata_index_dict(metadata, "node_index_names")
        node_offset = {}
        current_offset = 0

        for node_type, node_gdf in nodes_dict.items():
            node_offset[node_type] = current_offset
            node_index_names[node_type] = self._get_node_index_names(node_gdf)

            centroids = self.processor.compute_centroids(node_gdf)
            node_data = node_gdf if self.keep_geom else node_gdf.drop(columns="geometry")

            nodes_to_add = [
                (
                    current_offset + idx,
                    {
                        **attrs,
                        "node_type": node_type,
                        "_original_index": orig_idx,
                        "pos": (centroid.x, centroid.y),
                    },
                )
                for idx, (orig_idx, attrs, centroid) in enumerate(
                    zip(node_gdf.index, node_data.to_dict("records"), centroids, strict=False),
                )
            ]

            graph.add_nodes_from(nodes_to_add)
            current_offset += len(node_gdf)

        return node_offset

    def _add_heterogeneous_edges(
        self,
        graph: nx.Graph | nx.MultiGraph,
        edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
        metadata: "GraphMetadata",
    ) -> None:
        """
        Add heterogeneous edges to the graph.

        This method adds edges from a dictionary of typed GeoDataFrames to the
        NetworkX graph. It uses a node lookup to connect edges to the correct typed
        nodes and assigns an `edge_type` attribute to each edge. It also updates
        the graph's metadata with edge index information.

        Parameters
        ----------
        graph : networkx.Graph or networkx.MultiGraph
            The NetworkX graph to which the edges will be added.
        edges_dict : dict[tuple[str, str, str], geopandas.GeoDataFrame]
            A dictionary mapping edge type tuples to their GeoDataFrames.
        metadata : GraphMetadata
            The metadata object to be updated with edge information.

        See Also
        --------
        _add_homogeneous_edges : Add edges for homogeneous graphs.
        _create_edge_list : Create and add edge lists to graphs.

        Examples
        --------
        >>> converter = NxConverter()
        >>> # edges_dict and metadata would be prepared beforehand
        >>> converter._add_heterogeneous_edges(graph, edges_dict, metadata)
        """
        edge_index_names = _ensure_metadata_index_dict(metadata, "edge_index_names")

        # Identify all relevant node types
        relevant_node_types = set()
        for src_type, _, dst_type in edges_dict:
            relevant_node_types.add(src_type)
            relevant_node_types.add(dst_type)

        # Create node lookup once for all types
        node_lookup = self._create_node_lookup(graph, list(relevant_node_types))

        for edge_type, edge_gdf in edges_dict.items():
            self._add_single_hetero_edge_type(graph, edge_type, edge_gdf, node_lookup)
            edge_index_names[edge_type] = _coerce_name_sequence(edge_gdf.index.names)

    def _add_single_hetero_edge_type(
        self,
        graph: nx.Graph | nx.MultiGraph,
        edge_type: tuple[str, str, str],
        edge_gdf: gpd.GeoDataFrame,
        node_lookup: dict[str, dict[str, int]],
    ) -> None:
        """
        Process and add a single type of heterogeneous edges to the graph.

        This helper handles the mapping of edge indices to node IDs, updates
        metadata, and adds the edges to the graph.

        Parameters
        ----------
        graph : networkx.Graph or networkx.MultiGraph
            The graph to add edges to.
        edge_type : tuple[str, str, str]
            The type of the edges (source_type, relation_type, target_type).
        edge_gdf : geopandas.GeoDataFrame
            The GeoDataFrame containing the edge data.
        node_lookup : dict[str, dict[str, int]]
            Pre-computed lookup table for node IDs.
        """
        # Get edge type components
        src_type, _, dst_type = edge_type

        # Map edge indices to node IDs
        src_indices = edge_gdf.index.get_level_values(0)
        dst_indices = edge_gdf.index.get_level_values(1)

        u_nodes = pd.Series(src_indices.values, index=edge_gdf.index).map(
            node_lookup.get(src_type, {}),
        )
        v_nodes = pd.Series(dst_indices.values, index=edge_gdf.index).map(
            node_lookup.get(dst_type, {}),
        )

        valid_mask = u_nodes.notna() & v_nodes.notna()
        if not valid_mask.all():
            logger.warning(
                "Could not find nodes for %d edges of type %s",
                (~valid_mask).sum(),
                edge_type,
            )

        valid_edges = edge_gdf[valid_mask]
        valid_u = u_nodes[valid_mask]
        valid_v = v_nodes[valid_mask]

        self._create_edge_list(graph, valid_u, valid_v, valid_edges, self.keep_geom, edge_type)

    def _create_edge_list(
        self,
        graph: nx.Graph | nx.MultiGraph,
        u_nodes: pd.Series,
        v_nodes: pd.Series,
        edges_gdf: gpd.GeoDataFrame,
        keep_geom: bool,
        edge_type: str | tuple[str, str, str] | None = None,
    ) -> None:
        """
        Create and add a list of edges to the graph.

        This helper function constructs a list of edges with their attributes and adds
        them to the graph in a single, efficient operation. It handles both regular
        graphs and multigraphs by including edge keys where appropriate. It also
        attaches the original edge index and type information as attributes.

        Parameters
        ----------
        graph : networkx.Graph or networkx.MultiGraph
            The graph to which the edges will be added.
        u_nodes : pandas.Series
            A Series of source node identifiers.
        v_nodes : pandas.Series
            A Series of target node identifiers.
        edges_gdf : geopandas.GeoDataFrame
            The GeoDataFrame containing the edge data and attributes.
        keep_geom : bool
            If True, preserves the geometry attribute.
        edge_type : str or tuple, optional
            The type of the edges being added, for heterogeneous graphs.
        """
        attrs_df = edges_gdf if keep_geom else edges_gdf.drop(columns="geometry")
        edge_attrs = attrs_df.to_dict("records")

        if (
            isinstance(graph, nx.MultiGraph)
            and isinstance(edges_gdf.index, pd.MultiIndex)
            and edges_gdf.index.nlevels >= 2  # Check for at least u, v
        ):
            self._add_multigraph_edges(
                graph,
                u_nodes,
                v_nodes,
                edges_gdf,
                edge_attrs,
                edge_type,
            )
        else:
            self._add_simple_graph_edges(
                graph,
                u_nodes,
                v_nodes,
                edges_gdf,
                edge_attrs,
                edge_type,
            )

    def _add_multigraph_edges(
        self,
        graph: nx.MultiGraph,
        u_nodes: pd.Series,
        v_nodes: pd.Series,
        edges_gdf: gpd.GeoDataFrame,
        edge_attrs: list[dict[str, Any]],
        edge_type: str | tuple[str, str, str] | None,
    ) -> None:
        """
        Add edges to a MultiGraph with keys.

        This helper iterates through the edge data and adds edges to the MultiGraph,
        ensuring that edge keys are preserved or generated correctly.

        Parameters
        ----------
        graph : networkx.MultiGraph
            The MultiGraph to add edges to.
        u_nodes : pandas.Series
            Series of source node identifiers.
        v_nodes : pandas.Series
            Series of target node identifiers.
        edges_gdf : geopandas.GeoDataFrame
            The GeoDataFrame containing edge data.
        edge_attrs : list[dict[str, Any]]
            List of dictionaries containing edge attributes.
        edge_type : str or tuple, optional
            The type of the edges being added.
        """
        keys = (
            edges_gdf.index.get_level_values(2)
            if edges_gdf.index.nlevels == 3
            else range(len(edges_gdf))
        )
        edges_to_add_multi = [
            (
                u,
                v,
                k,
                {
                    **attrs,
                    "_original_edge_index": orig_idx,
                    **({"edge_type": edge_type} if edge_type else {}),
                },
            )
            for u, v, k, orig_idx, attrs in zip(
                u_nodes,
                v_nodes,
                keys,
                edges_gdf.index,
                edge_attrs,
                strict=True,
            )
        ]
        graph.add_edges_from(edges_to_add_multi)

    def _add_simple_graph_edges(
        self,
        graph: nx.Graph | nx.MultiGraph,
        u_nodes: pd.Series,
        v_nodes: pd.Series,
        edges_gdf: gpd.GeoDataFrame,
        edge_attrs: list[dict[str, Any]],
        edge_type: str | tuple[str, str, str] | None,
    ) -> None:
        """
        Add edges to a Graph or MultiGraph without explicit keys.

        This helper iterates through the edge data and adds edges to the graph.
        It handles the assignment of attributes and the original edge index.

        Parameters
        ----------
        graph : networkx.Graph or networkx.MultiGraph
            The graph to add edges to.
        u_nodes : pandas.Series
            Series of source node identifiers.
        v_nodes : pandas.Series
            Series of target node identifiers.
        edges_gdf : geopandas.GeoDataFrame
            The GeoDataFrame containing edge data.
        edge_attrs : list[dict[str, Any]]
            List of dictionaries containing edge attributes.
        edge_type : str or tuple, optional
            The type of the edges being added.
        """
        edges_to_add_simple = [
            (
                u,
                v,
                {
                    **attrs,
                    "_original_edge_index": orig_idx,
                    **({"edge_type": edge_type} if edge_type else {}),
                },
            )
            for u, v, orig_idx, attrs in zip(
                u_nodes,
                v_nodes,
                edges_gdf.index,
                edge_attrs,
                strict=True,
            )
        ]
        graph.add_edges_from(edges_to_add_simple)

    def _create_node_lookup(
        self,
        graph: nx.Graph | nx.MultiGraph,
        node_types: list[str],
    ) -> dict[str, dict[str, int]]:
        """
        Create a lookup dictionary for heterogeneous nodes.

        This method builds a nested dictionary that maps original node indices to the
        actual node identifiers in the NetworkX graph, organized by node type. This
        is essential for correctly connecting edges in a heterogeneous graph where
        node identifiers are offset.

        Parameters
        ----------
        graph : networkx.Graph or networkx.MultiGraph
            The graph containing the heterogeneous nodes.
        node_types : list[str]
            The list of node types to include in the lookup.

        Returns
        -------
        dict[str, dict[str, int]]
            A nested dictionary: `{node_type: {original_index: graph_node_id}}`.
        """
        node_lookup: dict[str, dict[str, int]] = {}
        for node_id, node_data in graph.nodes(data=True):
            node_type = node_data.get("node_type")
            orig_idx = node_data.get("_original_index")

            if node_type in node_types and orig_idx is not None:
                if node_type not in node_lookup:
                    node_lookup[node_type] = {}
                node_lookup[node_type][orig_idx] = node_id

        return node_lookup

    def _reconstruct_homogeneous(
        self,
        graph: nx.Graph | nx.MultiGraph,
        metadata: "GraphMetadata",
        nodes: bool = True,
        edges: bool = True,
    ) -> tuple[gpd.GeoDataFrame | None, gpd.GeoDataFrame | None] | gpd.GeoDataFrame:
        """
        Reconstruct homogeneous GeoDataFrames from a NetworkX graph.

        This internal method handles the logic for converting a homogeneous NetworkX
        graph back into node and edge GeoDataFrames. It uses the graph's metadata
        to correctly reconstruct geometries, attributes, and indices.

        Parameters
        ----------
        graph : networkx.Graph or networkx.MultiGraph
            The homogeneous NetworkX graph to reconstruct.
        metadata : GraphMetadata
            The metadata object containing information for reconstruction.
        nodes : bool, default True
            Whether to reconstruct the nodes GeoDataFrame.
        edges : bool, default True
            Whether to reconstruct the edges GeoDataFrame.

        Returns
        -------
        tuple[geopandas.GeoDataFrame | None, geopandas.GeoDataFrame | None] or geopandas.GeoDataFrame
            A tuple of (nodes_gdf, edges_gdf), or a single GeoDataFrame if only one
            is requested.
        """
        result: list[gpd.GeoDataFrame] = []

        if nodes:
            nodes_gdf = self._create_homogeneous_nodes_gdf(graph, metadata)
            result.append(nodes_gdf)

        if edges:
            edges_gdf = self._create_homogeneous_edges_gdf(graph, metadata)
            result.append(edges_gdf)

        if len(result) == 1:
            return result[0]
        return (result[0], result[1])

    def _reconstruct_heterogeneous(
        self,
        graph: nx.Graph | nx.MultiGraph,
        metadata: "GraphMetadata",
        nodes: bool = True,
        edges: bool = True,
    ) -> tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]]:
        """
        Reconstruct heterogeneous GeoDataFrames from a NetworkX graph.

        This internal method handles the complex logic of converting a NetworkX graph
        with typed nodes and edges back into separate, typed GeoDataFrames. It uses
        the graph's extensive metadata to split the nodes and edges by their type
        and reconstruct each GeoDataFrame with its correct attributes, geometry,
        and index.

        Parameters
        ----------
        graph : networkx.Graph or networkx.MultiGraph
            The heterogeneous NetworkX graph to reconstruct.
        metadata : GraphMetadata
            The metadata object containing type and index information.
        nodes : bool, default True
            Whether to reconstruct the node GeoDataFrames.
        edges : bool, default True
            Whether to reconstruct the edge GeoDataFrames.

        Returns
        -------
        tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]]
            A tuple containing a dictionary of node GeoDataFrames and a dictionary
            of edge GeoDataFrames.
        """
        nodes_dict = {}
        edges_dict = {}

        _ensure_metadata_index_dict(metadata, "node_index_names")
        _ensure_metadata_index_dict(metadata, "edge_index_names")

        if nodes:
            nodes_dict = self._create_heterogeneous_nodes_dict(graph, metadata)

        if edges:
            edges_dict = self._create_heterogeneous_edges_dict(graph, metadata)

        return nodes_dict, edges_dict

    def _create_homogeneous_nodes_gdf(
        self,
        graph: nx.Graph | nx.MultiGraph,
        metadata: "GraphMetadata",
    ) -> gpd.GeoDataFrame:
        """
        Create a homogeneous nodes GeoDataFrame from a NetworkX graph.

        This method extracts all nodes from a homogeneous graph, reconstructs their
        geometries from position attributes, and restores their original attributes
        and index using the provided metadata. The result is a single GeoDataFrame
        representing all nodes in the graph.

        Parameters
        ----------
        graph : networkx.Graph or networkx.MultiGraph
            The homogeneous NetworkX graph.
        metadata : GraphMetadata
            The metadata for reconstruction.

        Returns
        -------
        geopandas.GeoDataFrame
            The reconstructed nodes GeoDataFrame.
        """
        node_data = dict(graph.nodes(data=True))
        records, original_indices = self._create_node_records(node_data)

        index_names = _resolve_index_names(typing.cast("Any", metadata.node_index_names))
        index = self._build_node_index(original_indices, index_names)

        if not records:
            gdf = gpd.GeoDataFrame({"geometry": []}, index=index, crs=metadata.crs)
        else:
            gdf = gpd.GeoDataFrame(records, index=index, crs=metadata.crs)

        # Convert geometry columns
        for col in metadata.node_geom_cols:
            if col in gdf.columns:
                gdf[col] = gpd.GeoSeries(gdf[col], crs=metadata.crs)

        return gdf

    def _create_node_records(
        self,
        node_data: dict[object, dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[Any]]:
        """
        Create node records and extract original indices.

        Iterates through node data to separate attributes for the GeoDataFrame
        from the original indices needed for reconstruction.

        Parameters
        ----------
        node_data : dict[object, dict[str, Any]]
            Dictionary mapping node IDs to their attributes.

        Returns
        -------
        tuple[list[dict[str, Any]], list[Any]]
            A tuple containing a list of attribute dictionaries and a list of
            original indices.
        """
        original_indices = [attrs.get("_original_index", nid) for nid, attrs in node_data.items()]

        records = [
            {
                **{k: v for k, v in attrs.items() if k not in ["pos", "_original_index"]},
                "geometry": attrs["geometry"]
                if "geometry" in attrs and attrs["geometry"] is not None
                else (Point(attrs["pos"]) if "pos" in attrs else None),
            }
            for nid, attrs in node_data.items()
        ]
        return records, original_indices

    def _build_node_index(
        self,
        original_indices: list[object],
        index_names: Sequence[str | None] | None,
    ) -> pd.Index | pd.MultiIndex:
        """
        Build pandas Index or MultiIndex for nodes.

        Constructs the appropriate pandas Index based on the original indices
        and provided index names, handling MultiIndex creation if necessary.

        Parameters
        ----------
        original_indices : list[object]
            List of original node indices.
        index_names : Sequence[str | None] or None
            Names for the index levels.

        Returns
        -------
        pandas.Index or pandas.MultiIndex
            The reconstructed index.
        """
        if index_names and len(index_names) > 1:
            tuple_indices = [
                tuple(idx) if isinstance(idx, (list, tuple)) else (idx,) for idx in original_indices
            ]
            return pd.MultiIndex.from_tuples(tuple_indices, names=list(index_names))

        name = index_names[0] if index_names else None
        return pd.Index(original_indices, name=name)

    def _create_heterogeneous_nodes_dict(
        self,
        graph: nx.Graph | nx.MultiGraph,
        metadata: "GraphMetadata",
    ) -> dict[str, gpd.GeoDataFrame]:
        """
        Create a dictionary of heterogeneous node GeoDataFrames.

        This method filters the nodes of a heterogeneous graph by their `node_type`
        attribute and creates a separate GeoDataFrame for each type. It reconstructs
        the geometry, attributes, and original index for each typed GeoDataFrame,
        returning them in a dictionary.

        Parameters
        ----------
        graph : networkx.Graph or networkx.MultiGraph
            The heterogeneous NetworkX graph.
        metadata : GraphMetadata
            The metadata for reconstruction.

        Returns
        -------
        dict[str, geopandas.GeoDataFrame]
            A dictionary mapping node type names to their reconstructed GeoDataFrames.
        """
        nodes_dict = {}

        for node_type in metadata.node_types:
            type_nodes = [
                (n, d) for n, d in graph.nodes(data=True) if d.get("node_type") == node_type
            ]

            node_ids, attrs_list = zip(*type_nodes, strict=False)
            node_data = dict(zip(node_ids, attrs_list, strict=False))

            records, original_indices = self._create_node_records(node_data)

            # Filter out node_type from records as it's implicit in the dict key
            for record in records:
                record.pop("node_type", None)

            index_names = _resolve_index_names(
                typing.cast("Any", metadata.node_index_names), node_type
            )
            index = self._build_node_index(original_indices, index_names)
            gdf = gpd.GeoDataFrame(records, geometry="geometry", index=index, crs=metadata.crs)

            nodes_dict[node_type] = gdf

        return nodes_dict

    def _create_homogeneous_edges_gdf(
        self,
        graph: nx.Graph | nx.MultiGraph,
        metadata: "GraphMetadata",
    ) -> gpd.GeoDataFrame:
        """
        Create a homogeneous edges GeoDataFrame from a NetworkX graph.

        This method extracts all edges from a homogeneous graph, reconstructs their
        LineString geometries from the positions of their source and target nodes,
        and restores their original attributes and index from the provided metadata.

        Parameters
        ----------
        graph : networkx.Graph or networkx.MultiGraph
            The homogeneous NetworkX graph.
        metadata : GraphMetadata
            The metadata for reconstruction.

        Returns
        -------
        geopandas.GeoDataFrame
            The reconstructed edges GeoDataFrame.
        """
        if graph.number_of_edges() == 0:
            # Create empty GeoDataFrame with expected columns
            return gpd.GeoDataFrame({"geometry": []}, crs=metadata.crs)

        records, original_indices = self._create_edge_records(graph)
        index_names = _resolve_index_names(typing.cast("Any", metadata.edge_index_names))
        index = self._build_edge_index(original_indices, index_names)

        gdf = gpd.GeoDataFrame(records, index=index, crs=metadata.crs)

        # Convert geometry columns
        for col in metadata.edge_geom_cols:
            if col in gdf.columns:
                gdf[col] = gpd.GeoSeries(gdf[col], crs=metadata.crs)

        return gdf

    def _create_edge_records(
        self,
        graph: nx.Graph | nx.MultiGraph,
        edge_type: tuple[str, str, str] | None = None,
    ) -> tuple[list[dict[str, Any]], list[Any]]:
        """
        Create edge records and extract original indices.

        Iterates through graph edges to extract attributes and geometries,
        handling both simple and multi-graphs, and filtering by edge type if specified.

        Parameters
        ----------
        graph : networkx.Graph or networkx.MultiGraph
            The graph to extract edges from.
        edge_type : tuple[str, str, str], optional
            The specific edge type to filter for in heterogeneous graphs.

        Returns
        -------
        tuple[list[dict[str, Any]], list[Any]]
            A tuple containing:
            - A list of dictionaries representing edge attributes.
            - A list of original edge indices.
        """
        is_multigraph = isinstance(graph, nx.MultiGraph)
        edge_data: list[Any] = []

        if edge_type:
            # Filter for specific edge type
            if is_multigraph:
                edge_data = [
                    (u, v, k, d)
                    for u, v, k, d in graph.edges(data=True, keys=True)
                    if d.get("edge_type") == edge_type
                ]
            else:
                edge_data = [
                    (u, v, d)
                    for u, v, d in graph.edges(data=True)
                    if d.get("edge_type") == edge_type
                ]
        # Homogeneous case
        elif is_multigraph:
            # cast to MultiGraph to satisfy mypy
            multi_graph = typing.cast("nx.MultiGraph", graph)
            edge_data = list(multi_graph.edges(data=True, keys=True))
        else:
            edge_data = list(graph.edges(data=True))

        records = []
        original_indices = []
        for edge in edge_data:
            if is_multigraph and len(edge) == 4:
                u, v, k, attrs = edge
                default_idx: tuple[Any, ...] = (u, v, k)
            elif len(edge) == 3:
                u, v, attrs = edge
                default_idx = (u, v)
            else:  # pragma: no cover
                continue  # Should not happen

            original_indices.append(attrs.get("_original_edge_index", default_idx))

            geom = attrs.get("geometry")
            if geom is None and "pos" in graph.nodes[u] and "pos" in graph.nodes[v]:
                geom = LineString([graph.nodes[u]["pos"], graph.nodes[v]["pos"]])

            records.append(
                {
                    **{
                        k: v
                        for k, v in attrs.items()
                        if k not in ["_original_edge_index", "full_edge_type"]
                    },
                    "geometry": geom,
                },
            )
        return records, original_indices

    def _build_edge_index(
        self,
        original_indices: list[object],
        index_names: Sequence[str | None] | None,
    ) -> pd.Index | pd.MultiIndex:
        """
        Build pandas Index or MultiIndex for edges.

        Constructs the appropriate pandas Index for edges, handling MultiIndex
        tuples and assigning names from metadata.

        Parameters
        ----------
        original_indices : list[Any]
            List of original edge indices.
        index_names : list[str] or dict or None
            Names for the index levels.

        Returns
        -------
        pandas.Index or pandas.MultiIndex
            The reconstructed index.
        """
        if not original_indices:
            return pd.Index([])

        index: pd.Index | pd.MultiIndex

        # Handle MultiIndex
        if isinstance(original_indices[0], (tuple, list)):
            # Ensure all elements are properly converted to tuples
            tuple_indices = [
                tuple(idx) if isinstance(idx, list) else idx for idx in original_indices
            ]
            index = pd.MultiIndex.from_tuples(typing.cast("list[tuple[Any, ...]]", tuple_indices))
        else:
            index = pd.Index(original_indices)

        # Restore index names
        if index_names and hasattr(index, "names"):
            index.names = list(index_names)

        return index

    def _create_heterogeneous_edges_dict(
        self,
        graph: nx.Graph | nx.MultiGraph,
        metadata: "GraphMetadata",
    ) -> dict[tuple[str, str, str], gpd.GeoDataFrame]:
        """
        Create a dictionary of heterogeneous edge GeoDataFrames.

        This method filters the edges of a heterogeneous graph by their `edge_type`
        attribute and creates a separate GeoDataFrame for each type. It reconstructs
        the geometry, attributes, and original MultiIndex for each typed edge
        GeoDataFrame, returning them in a dictionary keyed by the edge type tuple.

        Parameters
        ----------
        graph : networkx.Graph or networkx.MultiGraph
            The heterogeneous NetworkX graph.
        metadata : GraphMetadata
            The metadata for reconstruction.

        Returns
        -------
        dict[tuple[str, str, str], geopandas.GeoDataFrame]
            A dictionary mapping edge type tuples to their reconstructed GeoDataFrames.
        """
        edges_dict = {}

        for edge_type in metadata.edge_types:
            edge_gdf = self._reconstruct_single_hetero_edge_type(graph, edge_type, metadata)
            edges_dict[edge_type] = edge_gdf

        return edges_dict

    def _reconstruct_single_hetero_edge_type(
        self,
        graph: nx.Graph | nx.MultiGraph,
        edge_type: tuple[str, str, str],
        metadata: "GraphMetadata",
    ) -> gpd.GeoDataFrame:
        """
        Reconstruct a single heterogeneous edge type GeoDataFrame.

        This helper extracts edges of a specific type from the graph and
        reconstructs them into a GeoDataFrame, restoring their geometry and
        attributes.

        Parameters
        ----------
        graph : networkx.Graph or networkx.MultiGraph
            The heterogeneous NetworkX graph.
        edge_type : tuple[str, str, str]
            The edge type to reconstruct.
        metadata : GraphMetadata
            The metadata for reconstruction.

        Returns
        -------
        geopandas.GeoDataFrame
            The reconstructed GeoDataFrame for the given edge type.
        """
        records, original_indices = self._create_edge_records(graph, edge_type)

        if not records:
            return gpd.GeoDataFrame(geometry=[], crs=metadata.crs)

        index_names = _resolve_index_names(typing.cast("Any", metadata.edge_index_names), edge_type)
        index = self._build_edge_index(original_indices, index_names)
        return gpd.GeoDataFrame(
            records,
            geometry="geometry",
            index=index,
            crs=metadata.crs,
        )


def _normalize_index_name(name: object) -> str | None:
    """
    Normalize an index name to a string or None.

    Ensures that index names are consistently represented as strings, converting
    None to None and other types to their string representation.

    Parameters
    ----------
    name : object
        The index name to normalize.

    Returns
    -------
    str or None
        The normalized index name.
    """
    return None if name is None else str(name)


def _coerce_name_sequence(
    names: Sequence[str | None] | str | None,
) -> list[str | None] | None:
    """
    Convert various index name formats into a normalized list.

    Handles single strings, lists of strings, or None, ensuring the output is
    always a list of normalized strings or None.

    Parameters
    ----------
    names : Sequence[str | None] | str | None
        The index names to coerce.

    Returns
    -------
    list[str | None] or None
        A list of normalized index names.
    """
    if names is None:
        return None
    if isinstance(names, str):
        return [_normalize_index_name(names)]
    return [_normalize_index_name(name) for name in names]


def _resolve_index_names(
    names: Sequence[str | None] | dict[object, Sequence[str | None] | None] | str | None,
    key: object | None = None,
) -> list[str | None] | None:
    """
    Resolve index names from metadata, handling dict/list/str inputs uniformly.

    Retrieves the appropriate index names based on the provided key if the input
    is a dictionary, or processes the input directly if it's a sequence or string.

    Parameters
    ----------
    names : Sequence | dict | str | None
        The index names or metadata structure.
    key : object, optional
        The key to look up if names is a dictionary.

    Returns
    -------
    list[str | None] or None
        The resolved list of index names.
    """
    if names is None:
        return None
    if isinstance(names, dict):
        target = names.get(key) if key is not None else next(iter(names.values()), None)
        return _coerce_name_sequence(target)
    return _coerce_name_sequence(names)


def _ensure_metadata_index_dict(
    metadata: GraphMetadata,
    attr: Literal["node_index_names", "edge_index_names"],
) -> dict[object, list[str | None] | None]:
    """
    Ensure GraphMetadata stores index names for hetero graphs as dictionaries.

    Checks if the specified attribute is already a dictionary; if not, initializes
    it as an empty dictionary to support heterogeneous graph metadata.

    Parameters
    ----------
    metadata : GraphMetadata
        The metadata object to update.
    attr : {'node_index_names', 'edge_index_names'}
        The attribute name to ensure is a dictionary.

    Returns
    -------
    dict
        The dictionary stored in the metadata attribute.
    """
    value = getattr(metadata, attr)
    result: dict[object, list[str | None] | None]
    if isinstance(value, dict):
        result = value
    else:
        result = {}
        setattr(metadata, attr, result)
    return result


def _safe_sort_key(value: object) -> tuple[str, str]:
    """
    Provide a deterministic sort key even for incomparable edge identifiers.

    Generates a tuple containing the type name and string representation of the
    value, allowing for consistent sorting of mixed types.

    Parameters
    ----------
    value : object
        The value to generate a sort key for.

    Returns
    -------
    tuple[str, str]
        A tuple of (type_name, repr_string) for sorting.
    """
    return (type(value).__name__, repr(value))


def _canonical_edge_pair(edge_a: object, edge_b: object) -> tuple[object, object]:
    """
    Return a deterministic ordering for an undirected edge pair.

    Sorts the two edge identifiers to ensure that the pair (u, v) is treated
    identically to (v, u), handling potential type comparison errors.

    Parameters
    ----------
    edge_a : object
        The first edge identifier.
    edge_b : object
        The second edge identifier.

    Returns
    -------
    tuple[object, object]
        The sorted pair of edge identifiers.
    """
    if edge_a == edge_b:
        return edge_a, edge_b
    try:
        return (
            (edge_a, edge_b)
            if typing.cast("Any", edge_a) <= typing.cast("Any", edge_b)
            else (edge_b, edge_a)
        )
    except TypeError:
        key_a = _safe_sort_key(edge_a)
        key_b = _safe_sort_key(edge_b)
        return (edge_a, edge_b) if key_a <= key_b else (edge_b, edge_a)


def _build_dual_edge_pairs(
    edge_ids: Iterable[object],
    u_values: Iterable[object],
    v_values: Iterable[object],
) -> list[tuple[object, object]]:
    """
    Compute unique dual-edge pairs given source/target node identifiers.

    Identifies pairs of edges that share a common node, effectively constructing
    the adjacency list for the dual graph.

    Parameters
    ----------
    edge_ids : Iterable[object]
        The identifiers of the edges.
    u_values : Iterable[object]
        The source node identifiers for each edge.
    v_values : Iterable[object]
        The target node identifiers for each edge.

    Returns
    -------
    list[tuple[object, object]]
        A list of unique pairs of adjacent edges.
    """
    adjacency: dict[object, set[object]] = defaultdict(set)
    for edge_id, u, v in zip(edge_ids, u_values, v_values, strict=False):
        adjacency[u].add(edge_id)
        adjacency[v].add(edge_id)

    pairs: set[tuple[Any, Any]] = set()
    for edges in adjacency.values():
        if len(edges) < 2:
            continue
        for edge_a, edge_b in combinations(edges, 2):
            pairs.add(_canonical_edge_pair(edge_a, edge_b))

    return sorted(
        pairs,
        key=lambda pair: (_safe_sort_key(pair[0]), _safe_sort_key(pair[1])),
    )


def _empty_dual_edge_gdf(crs: object, edge_id_col: str | None) -> gpd.GeoDataFrame:
    """
    Create an empty dual-edge GeoDataFrame with consistent index names.

    This helper ensures that even when no dual edges are produced, the resulting
    GeoDataFrame has the correct MultiIndex structure and column definitions.

    Parameters
    ----------
    crs : object
        The Coordinate Reference System.
    edge_id_col : str or None
        The name of the edge ID column.

    Returns
    -------
    geopandas.GeoDataFrame
        An empty GeoDataFrame for dual edges.
    """
    names = (
        [f"from_{edge_id_col}", f"to_{edge_id_col}"]
        if edge_id_col
        else ["from_edge_id", "to_edge_id"]
    )
    empty_index = pd.MultiIndex.from_arrays([[], []], names=names)
    return gpd.GeoDataFrame(geometry=[], crs=crs, index=empty_index)


# =============================================================================
# PUBLIC API FUNCTIONS
# =============================================================================


def dual_graph(
    graph: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph | nx.MultiGraph,
    edge_id_col: str | None = None,
    keep_original_geom: bool = False,
    source_col: str | None = None,
    target_col: str | None = None,
    as_nx: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph | nx.MultiGraph:
    """
    Convert a primal graph represented by nodes and edges GeoDataFrames to its dual graph.

    In the dual graph, original edges become nodes and original nodes become edges connecting
    adjacent original edges.

    Parameters
    ----------
    graph : tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame] or networkx.Graph or networkx.MultiGraph
        A graph containing nodes and edges GeoDataFrames or a NetworkX graph of the primal graph.
    edge_id_col : str, optional
        The name of the column in the edges GeoDataFrame to be used as unique identifiers
        for dual graph nodes. If None, the index of the edges GeoDataFrame is used.
        Default is None.
    keep_original_geom : bool, default False
        If True, preserve the original geometry of the edges in a new column named
        'original_geometry' in the dual nodes GeoDataFrame.
    source_col : str, optional
        Name of the column or index level representing the source node ID in the edges GeoDataFrame.
        If provided, it overrides automatic detection.
    target_col : str, optional
        Name of the column or index level representing the target node ID in the edges GeoDataFrame.
        If provided, it overrides automatic detection.
    as_nx : bool, default False
        If True, return the dual graph as a NetworkX graph instead of GeoDataFrames.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame]
        A tuple containing the nodes and edges of the dual graph as GeoDataFrames.

        - Dual nodes GeoDataFrame: Nodes represent original edges. The geometry is the
          centroid of the original edge's geometry. The index is derived from `edge_id_col`
          or the original edge index.
        - Dual edges GeoDataFrame: Edges represent adjacency between original edges (i.e.,
          they shared a node in the primal graph). The geometry is a LineString connecting
          the centroids of the two dual nodes. The index is a MultiIndex of the connected
          dual node IDs.

    See Also
    --------
    segments_to_graph : Convert LineString segments to a graph structure.

    Examples
    --------
    >>> import geopandas as gpd
    >>> import pandas as pd
    >>> from shapely.geometry import Point, LineString
    >>> # Primal graph nodes
    >>> nodes = gpd.GeoDataFrame(
    ...     {"node_id": [0, 1, 2]},
    ...     geometry=[Point(0, 0), Point(1, 1), Point(1, 0)],
    ...     crs="EPSG:32633"
    ... ).set_index("node_id")
    >>> # Primal graph edges
    >>> edges = gpd.GeoDataFrame(
    ...     {"u": [0, 1], "v": [1, 2]},
    ...     geometry=[LineString([(0, 0), (1, 1)]), LineString([(1, 1), (1, 0)])],
    ...     crs="EPSG:32633"
    ... )
    >>> # Convert to dual graph
    >>> dual_nodes, dual_edges = dual_graph((nodes, edges))
    """
    # Validate input type and extract GeoDataFrames
    nodes_gdf, edges_gdf = _validate_dual_graph_input(graph)

    # Ensure edges have a CRS
    if edges_gdf.crs is None:
        msg = "Edges GeoDataFrame must have a CRS."
        raise ValueError(msg)

    # Work on a copy to avoid modifying the input
    edges_clean = edges_gdf.copy()
    crs = edges_clean.crs

    # Handle empty edges case
    if edges_clean.empty:
        dual_nodes = gpd.GeoDataFrame(geometry=[], crs=crs)
        dual_edges = gpd.GeoDataFrame(geometry=[], crs=crs)
        return dual_nodes, dual_edges

    # edges_clean is guaranteed to be non-None and non-empty here
    assert edges_clean is not None
    assert not edges_clean.empty

    # 1. Create Dual Nodes
    # --------------------
    # Dual nodes are simply the centroids of the primal edges.
    # We preserve the original edge attributes in the dual nodes.
    dual_nodes = edges_clean.copy()
    if dual_nodes.crs.is_geographic:
        # Warn if using geographic CRS for centroid calculation
        warnings.warn(
            "Geometry is in a geographic CRS. Results from 'centroid' are likely incorrect. "
            "Use 'GeoSeries.to_crs()' to re-project geometries to a projected CRS before this operation.",
            UserWarning,
            stacklevel=2,
        )
    dual_nodes["geometry"] = dual_nodes.geometry.centroid

    if keep_original_geom:
        dual_nodes["original_geometry"] = edges_clean.geometry

    # Handle edge_id_col if provided
    if edge_id_col:
        if edge_id_col in dual_nodes.columns:
            dual_nodes = dual_nodes.set_index(edge_id_col)
        elif dual_nodes.index.name != edge_id_col:
            # If it's not a column and not the current index name, raise an error.
            msg = f"Column '{edge_id_col}' not found in edges GeoDataFrame."
            raise ValueError(msg)

    # 2. Create Dual Edges
    # --------------------
    # Dual edges connect dual nodes (primal edges) that share a primal node.
    # We can find these by looking at the start and end nodes of the primal edges.

    # We avoid reset_index() because it can fail if index names conflict with column names.
    u_values = None
    v_values = None

    # Identify source and target columns/indices
    u_values, v_values = _identify_source_target_cols(
        edges_clean, source_col=source_col, target_col=target_col
    )

    edge_pairs = _build_dual_edge_pairs(dual_nodes.index, u_values, v_values)

    if not edge_pairs:
        dual_edges = _empty_dual_edge_gdf(edges_clean.crs, edge_id_col)
        return (dual_nodes, dual_edges) if not as_nx else gdf_to_nx(dual_nodes, dual_edges)

    names = (
        [f"from_{edge_id_col}", f"to_{edge_id_col}"]
        if edge_id_col
        else ["from_edge_id", "to_edge_id"]
    )
    dual_index = pd.MultiIndex.from_tuples(edge_pairs, names=names)

    edge_ids_from = [pair[0] for pair in edge_pairs]
    edge_ids_to = [pair[1] for pair in edge_pairs]
    geom_series = dual_nodes.geometry
    p1 = geom_series.reindex(edge_ids_from).tolist()
    p2 = geom_series.reindex(edge_ids_to).tolist()
    geoms = [LineString([g1, g2]) for g1, g2 in zip(p1, p2, strict=False)]

    dual_edges = gpd.GeoDataFrame(
        geometry=geoms,
        crs=edges_clean.crs,
        index=dual_index,
    )

    if as_nx:
        return gdf_to_nx(dual_nodes, dual_edges)
    return dual_nodes, dual_edges


def _validate_dual_graph_input(
    graph: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph | nx.MultiGraph,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Validate and extract nodes and edges for dual graph conversion.

    This helper checks the input graph format and converts it to node and edge
    GeoDataFrames if necessary, ensuring a consistent starting point for dual graph creation.

    Parameters
    ----------
    graph : tuple or networkx.Graph
        The input graph representation.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame]
        The nodes and edges GeoDataFrames.
    """
    if not (
        isinstance(graph, (nx.Graph, nx.MultiGraph))
        or (isinstance(graph, tuple) and len(graph) == 2)
    ):
        msg = "Input `graph` must be a tuple of (nodes_gdf, edges_gdf) or a NetworkX graph."
        raise TypeError(msg)

    if isinstance(graph, (nx.Graph, nx.MultiGraph)):
        # If input is a NetworkX graph, convert it to GeoDataFrames
        nodes_gdf, edges_gdf = nx_to_gdf(graph, nodes=True, edges=True)
    else:
        # Input is guaranteed to be tuple[GeoDataFrame, GeoDataFrame] by type annotation
        nodes_gdf, edges_gdf = graph

    return nodes_gdf, edges_gdf


def _get_col_or_level(df: pd.DataFrame | gpd.GeoDataFrame, name: str) -> Any | None:  # noqa: ANN401
    """
    Get values from a column or index level by name.

    This helper abstracts the logic for retrieving data whether it resides in a
    DataFrame column or an index level, prioritizing columns.

    Parameters
    ----------
    df : pandas.DataFrame or geopandas.GeoDataFrame
        The DataFrame to search.
    name : str
        The name of the column or index level.

    Returns
    -------
    Any or None
        The values if found, otherwise None.
    """
    if name in df.columns:
        return df[name].to_numpy()
    if isinstance(df.index, pd.MultiIndex) and name in df.index.names:
        return df.index.get_level_values(name)
    # Handle standard Index with a name
    if df.index.name == name:
        return df.index.to_numpy()
    return None


def _identify_source_target_cols(
    edges_df: gpd.GeoDataFrame,
    source_col: str | None = None,
    target_col: str | None = None,
) -> tuple[Any, Any]:
    """
    Identify source and target node identifiers from a GeoDataFrame.

    Checks index levels and columns for standard naming conventions.
    If source_col and target_col are provided, they are used explicitly.

    Parameters
    ----------
    edges_df : geopandas.GeoDataFrame
        The edges GeoDataFrame.
    source_col : str, optional
        Name of the column or index level representing the source node ID.
    target_col : str, optional
        Name of the column or index level representing the target node ID.

    Returns
    -------
    tuple
        (u_values, v_values) arrays or Series of source and target node IDs.

    Raises
    ------
    ValueError
        If source/target columns cannot be identified.
    """
    # 1. Explicit Specification
    if source_col and target_col:
        u = _get_col_or_level(edges_df, source_col)
        v = _get_col_or_level(edges_df, target_col)

        if u is not None and v is not None:
            return u, v

        # If one is missing, raise error
        missing = []
        if u is None:
            missing.append(source_col)
        if v is None:
            missing.append(target_col)
        msg = f"Source/Target column(s) not found: {', '.join(missing)}"
        raise ValueError(msg)

    # 2. Implicit Candidates
    # List of (u_name, v_name) pairs to check
    candidates = [
        ("from_node_id", "to_node_id"),
        ("source_id", "target_id"),
        ("u", "v"),
        ("source", "target"),
    ]

    for u_name, v_name in candidates:
        u = _get_col_or_level(edges_df, u_name)
        v = _get_col_or_level(edges_df, v_name)
        if u is not None and v is not None:
            return u, v

    # 3. Fallback: MultiIndex position (first two levels)
    if isinstance(edges_df.index, pd.MultiIndex) and edges_df.index.nlevels >= 2:
        return edges_df.index.get_level_values(0), edges_df.index.get_level_values(1)

    # 4. Fallback: Column position (first two columns)
    if len(edges_df.columns) >= 2:
        return edges_df.iloc[:, 0].to_numpy(), edges_df.iloc[:, 1].to_numpy()

    msg = "Could not identify source and target node columns/indices in edges GeoDataFrame."
    raise ValueError(msg)


def gdf_to_nx(
    nodes: gpd.GeoDataFrame | dict[str, gpd.GeoDataFrame] | None = None,
    edges: gpd.GeoDataFrame | dict[tuple[str, str, str], gpd.GeoDataFrame] | None = None,
    keep_geom: bool = True,
    multigraph: bool = False,
    directed: bool = False,
) -> nx.Graph | nx.MultiGraph | nx.DiGraph | nx.MultiDiGraph:
    """
    Convert GeoDataFrames of nodes and edges to a NetworkX graph.

    This function provides a high-level interface to convert geospatial data,
    represented as GeoDataFrames, into a NetworkX graph. It supports both
    homogeneous and heterogeneous graphs.

    For homogeneous graphs, provide a single GeoDataFrame for nodes and edges.
    For heterogeneous graphs, provide dictionaries mapping type names to
    GeoDataFrames.

    Parameters
    ----------
    nodes : geopandas.GeoDataFrame or dict[str, geopandas.GeoDataFrame], optional
        Node data. For homogeneous graphs, a single GeoDataFrame. For
        heterogeneous graphs, a dictionary mapping node type names to
        GeoDataFrames. Node IDs are taken from the GeoDataFrame index.
    edges : geopandas.GeoDataFrame or dict[tuple[str, str, str], geopandas.GeoDataFrame], optional
        Edge data. For homogeneous graphs, a single GeoDataFrame. For
        heterogeneous graphs, a dictionary mapping edge type tuples
        (source_type, relation_type, target_type) to GeoDataFrames.
        Edge relationships are defined by a MultiIndex on the edge
        GeoDataFrame (source ID, target ID). For MultiGraphs, a third level
        in the index can be used for edge keys.
    keep_geom : bool, default True
        If True, the geometry of the nodes and edges GeoDataFrames will be
        preserved as attributes in the NetworkX graph.
    multigraph : bool, default False
        If True, a `networkx.MultiGraph` is created, which can store multiple
        edges between the same two nodes.
    directed : bool, default False
        If True, a directed graph (`networkx.DiGraph` or `networkx.MultiDiGraph`)
        is created. Otherwise, an undirected graph is created.

    Returns
    -------
    networkx.Graph or networkx.MultiGraph or networkx.DiGraph or networkx.MultiDiGraph
        A NetworkX graph object representing the spatial network. Graph-level
        metadata, such as CRS and heterogeneity information, is stored in
        `graph.graph`.

    See Also
    --------
    nx_to_gdf : Convert a NetworkX graph back to GeoDataFrames.

    Examples
    --------
    >>> # Homogeneous graph
    >>> import geopandas as gpd
    >>> import pandas as pd
    >>> from shapely.geometry import Point, LineString
    >>> nodes_gdf = gpd.GeoDataFrame(
    ...     geometry=[Point(0, 0), Point(1, 1)],
    ...     index=pd.Index([10, 20], name="node_id")
    ... )
    >>> edges_gdf = gpd.GeoDataFrame(
    ...     {"length": [1.414]},
    ...     geometry=[LineString([(0, 0), (1, 1)])],
    ...     index=pd.MultiIndex.from_tuples([(10, 20)], names=["u", "v"])
    ... )
    >>> G = gdf_to_nx(nodes=nodes_gdf, edges=edges_gdf)
    >>> print(G.nodes(data=True))
    >>> [(0, {'geometry': <POINT (0 0)>,
    ...       '_original_index': 10,
    ...       'pos': (0.0, 0.0)}),
    ...  (1, {'geometry': <POINT (1 1)>,
    ...       '_original_index': 20,
    ...       'pos': (1.0, 1.0)})]
    >>> print(G.edges(data=True))
    >>> [(0, 1, {'length': 1.414,
    ...          'geometry': <LINESTRING (0 0, 1 1)>,
    ...          '_original_edge_index': (10, 20)})]

    >>> # Heterogeneous graph
    >>> buildings_gdf = gpd.GeoDataFrame(geometry=[Point(0, 0)], index=pd.Index(['b1'], name="b_id"))
    >>> streets_gdf = gpd.GeoDataFrame(geometry=[Point(1, 1)], index=pd.Index(['s1'], name="s_id"))
    >>> connections_gdf = gpd.GeoDataFrame(
    ...     geometry=[LineString([(0,0), (1,1)])],
    ...     index=pd.MultiIndex.from_tuples([('b1', 's1')])
    ... )
    >>> nodes_dict = {"building": buildings_gdf, "street": streets_gdf}
    >>> edges_dict = {("building", "connects_to", "street"): connections_gdf}
    >>> H = gdf_to_nx(nodes=nodes_dict, edges=edges_dict)
    >>> print(H.nodes(data=True))
    >>> [(0, {'geometry': <POINT (0 0)>,
    ...       'node_type': 'building',
    ...       '_original_index': 'b1',
    ...       'pos': (0.0, 0.0)}),
    ...  (1, {'geometry': <POINT (1 1)>,
    ...       'node_type': 'street',
    ...       '_original_index': 's1',
    ...       'pos': (1.0, 1.0)})]
    >>> print(H.edges(data=True))
    >>> [(0, 1, {'geometry': <LINESTRING (0 0, 1 1)>,
    ...          'full_edge_type': ('building', 'connects_to', 'street'),
    ...          '_original_edge_index': ('b1', 's1')})]
    """
    # Validate inputs using enhanced validation with type detection
    validated_nodes, validated_edges, _ = validate_gdf(
        nodes_gdf=nodes,
        edges_gdf=edges,
        allow_empty=True,
    )

    converter = NxConverter(keep_geom=keep_geom, multigraph=multigraph, directed=directed)
    return converter.gdf_to_nx(validated_nodes, validated_edges)


def nx_to_gdf(
    G: nx.Graph | nx.MultiGraph,
    nodes: bool = True,
    edges: bool = True,
    set_missing_pos_from: tuple[str, ...] | None = ("x", "y"),
) -> (
    gpd.GeoDataFrame
    | tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
    | tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]]
):
    """
    Convert a NetworkX graph to GeoDataFrames for nodes and/or edges.

    This function reconstructs GeoDataFrames from a NetworkX graph that was
    created by `gdf_to_nx` or follows a similar structure. It can handle both
    homogeneous and heterogeneous graphs, extracting node and edge attributes
    and reconstructing geometries from position data.

    Parameters
    ----------
    G : networkx.Graph or networkx.MultiGraph
        The NetworkX graph to convert. It is expected to have metadata stored
        in `G.graph` to guide the conversion, including CRS and heterogeneity
        information. Node positions are expected in a 'pos' attribute.
    nodes : bool, default True
        If True, a GeoDataFrame for nodes will be created and returned.
    edges : bool, default True
        If True, a GeoDataFrame for edges will be created and returned.
    set_missing_pos_from : tuple[str, ...] | None, default ("x", "y")
        If provided (or None implies ("x", "y")) set missing node 'pos' from
        the given attribute name(s). With two names use them as x/y; with one
        name expect a 2-length coordinate.

    Returns
    -------
    geopandas.GeoDataFrame or tuple
        The reconstructed GeoDataFrames. The return type depends on the graph structure
        (homogeneous vs heterogeneous) and the requested components (`nodes`, `edges`).

        **Homogeneous Graphs:**

        *   If nodes=True and edges=True: Returns (nodes_gdf, edges_gdf)
            where both are geopandas.GeoDataFrame.
        *   If only nodes=True: Returns nodes_gdf (geopandas.GeoDataFrame).
        *   If only edges=True: Returns edges_gdf (geopandas.GeoDataFrame).

        **Heterogeneous Graphs:**

        *   Returns (nodes_dict, edges_dict) where:

            *   nodes_dict is dict[str, geopandas.GeoDataFrame] mapping node types to GeoDataFrames.
            *   edges_dict is dict[tuple[str, str, str], geopandas.GeoDataFrame] mapping edge types to GeoDataFrames.

        Note: For heterogeneous graphs, the return value is always a tuple of dictionaries,
        even if only one component is requested (the other will be an empty dict).

    Raises
    ------
    ValueError
        If both `nodes` and `edges` are False.

    See Also
    --------
    gdf_to_nx : Convert GeoDataFrames to a NetworkX graph.

    Examples
    --------
    >>> import networkx as nx
    >>> # Create a simple graph with spatial attributes
    >>> G = nx.Graph(is_hetero=False, crs="EPSG:4326")
    >>> G.add_node(0, pos=(0, 0), population=100, geometry=Point(0,0))
    >>> G.add_node(1, pos=(1, 1), population=200, geometry=Point(1,1))
    >>> G.add_edge(0, 1, weight=1.5, geometry=LineString([(0, 0), (1, 1)]))
    >>> # Convert back to GeoDataFrames
    >>> nodes_gdf, edges_gdf = nx_to_gdf(G)
    >>> print(nodes_gdf)
    >>> print(edges_gdf)
    >>>           population     geometry
    ... 0         100           POINT (0 0)
    ... 1         200           POINT (1 1)
    ...           weight        geometry
    ... 0 1       1.5           LINESTRING (0 0, 1 1)
    """
    if not (nodes or edges):
        msg = "Must request at least one of nodes or edges"
        raise ValueError(msg)

    # Minimal pre-processing to set missing 'pos' from provided attribute names
    existing_pos = nx.get_node_attributes(G, "pos")

    if not existing_pos and set_missing_pos_from is not None:
        if len(set_missing_pos_from) == 2:
            x_name, y_name = set_missing_pos_from[0], set_missing_pos_from[1]
            x_attr = nx.get_node_attributes(G, x_name)
            y_attr = nx.get_node_attributes(G, y_name)
            eligible = x_attr.keys() & y_attr.keys()
            to_set = {
                n: (x_attr[n], y_attr[n])
                for n in eligible
                if x_attr[n] is not None and y_attr[n] is not None
            }
            if to_set:
                nx.set_node_attributes(G, to_set, "pos")

        elif len(set_missing_pos_from) == 1:
            coord_name = set_missing_pos_from[0]
            coord_attr = nx.get_node_attributes(G, coord_name)
            to_set = {
                n: (v[0], v[1]) if not isinstance(v, tuple) else v
                for n, v in coord_attr.items()
                if isinstance(v, (list, tuple)) and len(v) == 2
            }
            if to_set:
                nx.set_node_attributes(G, to_set, "pos")

    converter = NxConverter()
    return converter.nx_to_gdf(G, nodes, edges)


def filter_graph_by_distance(
    graph: gpd.GeoDataFrame | nx.Graph | nx.MultiGraph,
    center_point: Point | gpd.GeoSeries | gpd.GeoDataFrame,
    threshold: float,
    edge_attr: str | None = "length",
    node_id_col: str | None = None,  # noqa: ARG001
) -> gpd.GeoDataFrame | nx.Graph | nx.MultiGraph:
    """
    Filter a graph to include only elements within a specified threshold from a center point.

    This function calculates the shortest path from a center point to all nodes
    in the graph and returns a subgraph containing only the nodes (and their
    induced edges) that are within the given threshold. The input can be a
    NetworkX graph or an edges GeoDataFrame.

    Parameters
    ----------
    graph : geopandas.GeoDataFrame or networkx.Graph or networkx.MultiGraph
        The graph to filter. If a GeoDataFrame, it represents the edges of the
        graph and will be converted to a NetworkX graph internally.
    center_point : Point or geopandas.GeoSeries or geopandas.GeoDataFrame
        The origin point(s) for the distance calculation. If multiple points
        are provided, the filter will include nodes reachable from any of them.
    threshold : float
        The maximum shortest-path distance (or cost) for a node to be included in the
        filtered graph.
    edge_attr : str, default "length"
        The name of the edge attribute to use as weight for shortest path
        calculations (e.g., 'length', 'travel_time').
    node_id_col : str, optional
        The name of the node identifier column if the input graph is a
        GeoDataFrame. Defaults to the index.

    Returns
    -------
    geopandas.GeoDataFrame or networkx.Graph or networkx.MultiGraph
        The filtered subgraph. The return type matches the input `graph` type.
        If the input was a GeoDataFrame, the output is a GeoDataFrame of the
        filtered edges.

    See Also
    --------
    create_isochrone : Generate an isochrone polygon from a graph.

    Examples
    --------
    >>> import networkx as nx
    >>> from shapely.geometry import Point
    >>> # Create a graph
    >>> G = nx.Graph()
    >>> G.add_node(0, pos=(0, 0))
    >>> G.add_node(1, pos=(10, 0))
    >>> G.add_node(2, pos=(20, 0))
    >>> G.add_edge(0, 1, length=10)
    >>> G.add_edge(1, 2, length=10)
    >>> # Filter the graph
    >>> center = Point(1, 0)
    >>> filtered_graph = filter_graph_by_distance(G, center, threshold=12)
    >>> print(list(filtered_graph.nodes))
    >>> [0, 1]
    """
    is_graph_input = isinstance(graph, (nx.Graph, nx.MultiGraph))

    # Convert to NetworkX if needed
    if is_graph_input:
        nx_graph = graph
        original_crs = nx_graph.graph.get("crs")
    else:
        converter = NxConverter()
        nx_graph = converter.gdf_to_nx(edges=graph)
        original_crs = graph.crs if hasattr(graph, "crs") else None

    # Extract node positions
    pos_dict = nx.get_node_attributes(nx_graph, "pos")
    if not pos_dict:
        if is_graph_input:
            graph_type = type(graph)
            return graph_type()
        return gpd.GeoDataFrame(geometry=[], crs=original_crs)

    # Prepare KDTree for fast nearest neighbor search
    node_ids = list(pos_dict.keys())
    coordinates = list(pos_dict.values())
    tree = cKDTree(coordinates)

    # Normalize center points
    if isinstance(center_point, gpd.GeoDataFrame):
        center_points = center_point.geometry
    elif isinstance(center_point, gpd.GeoSeries):
        center_points = center_point
    else:
        center_points = [center_point]

    center_points_list = (
        center_points.tolist() if hasattr(center_points, "tolist") else list(center_points)
    )

    # Compute reachable nodes
    all_reachable = set()
    for point in center_points_list:
        # Find nearest node using KDTree
        _, idx = tree.query((point.x, point.y))
        source_node = node_ids[idx]

        # Dijkstra
        lengths = nx.single_source_dijkstra_path_length(
            nx_graph,
            source_node,
            cutoff=threshold,
            weight=edge_attr or "length",
        )
        all_reachable.update(lengths.keys())

    # Create subgraph
    subgraph = nx_graph.subgraph(all_reachable)

    if is_graph_input:
        return subgraph

    # Convert back to GeoDataFrame
    converter = NxConverter()
    return converter.nx_to_gdf(subgraph, nodes=False, edges=True)


def create_isochrone(
    graph: nx.Graph | nx.MultiGraph | None = None,
    nodes: gpd.GeoDataFrame | dict[str, gpd.GeoDataFrame] | None = None,
    edges: gpd.GeoDataFrame | dict[tuple[str, str, str], gpd.GeoDataFrame] | None = None,
    center_point: Point | gpd.GeoSeries | gpd.GeoDataFrame | None = None,
    threshold: float | None = None,
    edge_attr: str | None = None,
    cut_edge_types: list[tuple[str, str, str]] | None = None,
    method: str = "concave_hull_knn",
    **kwargs: Any,  # noqa: ANN401
) -> gpd.GeoDataFrame:
    """
    Generate an isochrone polygon from a graph.

    An isochrone represents the area reachable from a center point within a
    given travel threshold (distance or time). This function computes the set of reachable
    edges and nodes in a network and generates a polygon that encloses this
    reachable area.

    Parameters
    ----------
    graph : networkx.Graph or networkx.MultiGraph, optional
        The network graph.
    nodes : geopandas.GeoDataFrame or dict, optional
        Nodes of the graph.
    edges : geopandas.GeoDataFrame or dict, optional
        Edges of the graph.
    center_point : Point or geopandas.GeoSeries or geopandas.GeoDataFrame
        The origin point(s) for the isochrone calculation.
    threshold : float
        The maximum travel distance (or time) that defines the boundary of the
        isochrone.
    edge_attr : str, default "travel_time"
        The edge attribute to use for distance calculation (e.g., 'length',
        'travel_time'). If None, the function will use the default edge attribute.
    cut_edge_types : list[tuple[str, str, str]] | None, default None
        List of edge types to remove from the graph before processing (e.g.,
        [("bus_stop", "is_next_to", "bus_stop")]).
    method : str, default "concave_hull_knn"
        The method to generate the isochrone polygon. Options are:

        - "concave_hull_knn": Creates a concave hull (k-NN) around reachable nodes.
        - "concave_hull_alpha": Creates a concave hull (alpha shape) around reachable nodes.
        - "convex_hull": Creates a convex hull around reachable nodes.
        - "buffer": Creates a buffer around reachable edges/nodes.
    **kwargs : Any
        Additional parameters for specific isochrone generation methods:

        For method="concave_hull_knn":
            k : int, default 100
                The number of nearest neighbors to consider.

        For method="concave_hull_alpha":
            hull_ratio : float, default 0.3
                The ratio for concave hull generation (0.0 to 1.0). Higher values mean tighter fit.
            allow_holes : bool, default False
                Whether to allow holes in the concave hull.

        For method="buffer":
            buffer_distance : float, default 100
                The distance to buffer reachable geometries.
            cap_style : int, default 1
                The cap style for buffering. 1=Round, 2=Flat, 3=Square.
            join_style : int, default 1
                The join style for buffering. 1=Round, 2=Mitre, 3=Bevel.
            resolution : int, default 16
                The resolution of the buffer (number of segments per quarter circle).

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing a single Polygon or MultiPolygon geometry that
        represents the isochrone.

    Raises
    ------
    ValueError
        If required inputs are missing or invalid.
    """
    valid_methods = {"concave_hull_knn", "concave_hull_alpha", "convex_hull", "buffer"}
    if method not in valid_methods:
        msg = f"Unknown method: {method}. Must be one of {valid_methods}."
        raise ValueError(msg)

    if center_point is None or threshold is None:
        msg = "center_point and threshold must be provided."
        raise ValueError(msg)

    # Prepare the graph
    nx_graph = _prepare_isochrone_graph(graph, nodes, edges, edge_attr)

    # Compute Reachable Subgraph
    reachable = filter_graph_by_distance(nx_graph, center_point, threshold, edge_attr)

    # Filter Edge Types if requested
    if cut_edge_types and isinstance(reachable, (nx.Graph, nx.MultiGraph)):
        reachable = _filter_edges_by_type(reachable, cut_edge_types)

    # Get CRS
    crs = nx_graph.graph.get("crs")

    # Generate polygons
    polygons = _generate_component_polygons(reachable, method, crs, **kwargs)

    if not polygons:
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    # Union all component polygons
    final_geom = gpd.GeoSeries(polygons, crs=crs).union_all()

    # Ensure the final geometry is a Polygon or MultiPolygon
    if not isinstance(final_geom, (Polygon, MultiPolygon)):
        final_geom = final_geom.buffer(0)

    return gpd.GeoDataFrame(geometry=[final_geom], crs=crs)


def _prepare_isochrone_graph(
    graph: nx.Graph | nx.MultiGraph | None,
    nodes: gpd.GeoDataFrame | dict[str, gpd.GeoDataFrame] | None,
    edges: gpd.GeoDataFrame | dict[tuple[str, str, str], gpd.GeoDataFrame] | None,
    edge_attr: str | None,
) -> nx.Graph | nx.MultiGraph:
    """
    Prepare the graph for isochrone generation.

    Validates inputs and converts GeoDataFrames to a NetworkX graph if necessary.
    Also handles edge attribute injection for dict inputs.

    Parameters
    ----------
    graph : networkx.Graph or networkx.MultiGraph or None
        Existing graph object.
    nodes : geopandas.GeoDataFrame or dict or None
        Node data.
    edges : geopandas.GeoDataFrame or dict or None
        Edge data.
    edge_attr : str or None
        Edge attribute to use for distance.

    Returns
    -------
    networkx.Graph or networkx.MultiGraph
        The prepared graph.
    """
    if graph is not None:
        return graph

    if nodes is None and edges is None:
        msg = "Either 'graph' or 'nodes' and 'edges' must be provided."
        raise ValueError(msg)

    # If edges is a dict, ensure length attribute exists
    if isinstance(edges, dict) and edge_attr:
        edges = {
            k: (
                gdf.assign(**{edge_attr: gdf.geometry.length})
                if edge_attr not in gdf.columns and "geometry" in gdf.columns
                else gdf
            )
            for k, gdf in edges.items()
        }
    elif (
        isinstance(edges, gpd.GeoDataFrame)
        and edge_attr
        and edge_attr not in edges.columns
        and "geometry" in edges.columns
    ):
        edges = edges.assign(**{edge_attr: edges.geometry.length})

    converter = NxConverter()
    return converter.gdf_to_nx(nodes=nodes, edges=edges)


def _filter_edges_by_type(
    graph: nx.Graph | nx.MultiGraph,
    cut_edge_types: list[tuple[str, str, str]],
) -> nx.Graph | nx.MultiGraph:
    """
    Remove edges of specified types from the graph.

    Iterates through the graph edges and removes those that match any of the
    specified types in `cut_edge_types`.

    Parameters
    ----------
    graph : networkx.Graph or networkx.MultiGraph
        The input graph.
    cut_edge_types : list[tuple[str, str, str]]
        List of edge types to remove.

    Returns
    -------
    networkx.Graph or networkx.MultiGraph
        The graph with specified edges removed.
    """
    graph = graph.copy()
    edges_to_remove = [
        (u, v)
        for u, v, d in graph.edges(data=True)
        if (d.get("full_edge_type") or d.get("edge_type")) in cut_edge_types
    ]
    if edges_to_remove:
        graph.remove_edges_from(edges_to_remove)
    return graph


def _generate_component_polygons(
    reachable: nx.Graph | nx.MultiGraph | gpd.GeoDataFrame,
    method: str,
    crs: Any,  # noqa: ANN401
    **kwargs: Any,  # noqa: ANN401
) -> list[Polygon | MultiPolygon]:
    """
    Generate polygons for each connected component of the reachable graph.

    Splits the graph into connected components and generates a polygon for each
    component using the specified method.

    Parameters
    ----------
    reachable : networkx.Graph or networkx.MultiGraph or geopandas.GeoDataFrame
        The reachable subgraph or edge GeoDataFrame.
    method : str
        The generation method.
    crs : Any
        The Coordinate Reference System.
    **kwargs : Any
        Additional arguments for the method.

    Returns
    -------
    list[Polygon | MultiPolygon]
        A list of generated Polygon or MultiPolygon geometries.
    """
    components = _get_graph_components(reachable)

    polygons = []
    for comp in components:
        poly = _process_component(comp, method, crs, **kwargs)
        if poly is not None:
            polygons.append(poly)

    return polygons


def _get_graph_components(
    reachable: nx.Graph | nx.MultiGraph,
) -> list[nx.Graph | nx.MultiGraph]:
    """
    Determine connected components of the reachable graph.

    If the input is a GeoDataFrame, it is treated as a single component.
    Otherwise, it computes weakly or strongly connected components depending on
    graph directionality.

    Parameters
    ----------
    reachable : networkx.Graph or networkx.MultiGraph or geopandas.GeoDataFrame
        The reachable subgraph or edge GeoDataFrame.

    Returns
    -------
    list
        A list of connected components (subgraphs).
    """
    if len(reachable) == 0:
        return []

    # Check if graph is already connected
    if _is_graph_connected(reachable):
        return [reachable]

    # Split into connected components
    component_fn = (
        nx.weakly_connected_components if reachable.is_directed() else nx.connected_components
    )
    return [reachable.subgraph(c).copy() for c in component_fn(reachable)]


def _is_graph_connected(graph: nx.Graph | nx.MultiGraph) -> bool:
    """
    Check if graph is connected (weakly for directed, strongly for undirected).

    This utility function abstracts the difference between directed and undirected
    graphs when checking for connectivity.

    Parameters
    ----------
    graph : networkx.Graph or networkx.MultiGraph
        The input graph.

    Returns
    -------
    bool
        True if connected, False otherwise.
    """
    return bool(nx.is_weakly_connected(graph) if graph.is_directed() else nx.is_connected(graph))


def _process_component(
    component: nx.Graph | nx.MultiGraph,
    method: str,
    crs: Any,  # noqa: ANN401
    **kwargs: Any,  # noqa: ANN401
) -> Polygon | MultiPolygon | None:
    """
    Process a single component to generate its polygon.

    Extracts geometries from the component and generates a polygon using the
    specified method. Returns None if the result is invalid or empty.

    Parameters
    ----------
    component : networkx.Graph or networkx.MultiGraph
        The connected component to process.
    method : str
        The generation method.
    crs : Any
        The Coordinate Reference System.
    **kwargs : Any
        Additional arguments for the method.

    Returns
    -------
    Polygon or MultiPolygon or None
        The generated Polygon or MultiPolygon geometry, or None if failed or empty.
    """
    geoms = _extract_isochrone_geometries(component, method)
    if not geoms:
        return None

    gs = gpd.GeoSeries(geoms, crs=crs)
    # Filter invalid geometries
    gs = gs[~gs.is_empty & gs.is_valid]

    if gs.empty:
        return None

    poly = _generate_polygon(gs, method, **kwargs)

    if poly is None or poly.is_empty:
        return None

    # Ensure Polygon/MultiPolygon output
    if isinstance(poly, (Polygon, MultiPolygon)):
        return poly

    # Handle degenerate cases (Point/LineString) by buffering
    # Use a small default buffer distance if not provided
    buffer_dist = kwargs.get("degenerate_buffer_distance", 1.0)
    # Ensure positive buffer distance
    buffer_dist = max(buffer_dist, 1e-6) if buffer_dist is not None else 1.0

    buffered = poly.buffer(buffer_dist)
    return buffered if not buffered.is_empty and buffered.is_valid else None


def _generate_polygon(
    gs: gpd.GeoSeries,
    method: str,
    **kwargs: Any,  # noqa: ANN401
) -> Polygon | MultiPolygon | LineString | Point | None:
    """
    Generate polygon using the specified method.

    This function acts as a dispatcher, calling the appropriate geometry generation
    function based on the provided method name. Note that this function may return
    non-polygon geometries (LineString, Point) for degenerate cases, which are
    then converted to polygons by the caller.

    Parameters
    ----------
    gs : geopandas.GeoSeries
        Input geometries.
    method : str
        The generation method name.
    **kwargs : Any
        Additional arguments for the method.

    Returns
    -------
    Polygon or MultiPolygon or LineString or Point or None
        The generated geometry (may be non-polygon for degenerate cases).
    """
    # Dispatch to method-specific handler
    handlers = {
        "concave_hull_knn": _generate_concave_hull_knn,
        "concave_hull_alpha": _generate_concave_hull_alpha,
        "convex_hull": _generate_convex_hull,
        "buffer": _generate_buffer,
    }

    handler = handlers[method]
    return handler(gs, **kwargs)


def _generate_concave_hull_knn(
    gs: gpd.GeoSeries,
    **kwargs: Any,  # noqa: ANN401
) -> Polygon | LineString | Point | None:
    """
    Generate concave hull using k-NN method.

    Extracts points from the input geometries and computes the concave hull using
    the k-nearest neighbors algorithm. May return LineString or Point for degenerate
    cases (< 3 points), which are converted to polygons by the caller.

    Parameters
    ----------
    gs : geopandas.GeoSeries
        Input geometries.
    **kwargs : Any
        Additional arguments including 'k'.

    Returns
    -------
    Polygon or LineString or Point or None
        The concave hull geometry, or None if empty.
    """
    k = kwargs.get("k", 100)
    points = _extract_points_from_geometries(gs)
    return _concave_hull_knn(points, k=k) if points else None


def _generate_concave_hull_alpha(
    gs: gpd.GeoSeries,
    **kwargs: Any,  # noqa: ANN401
) -> Polygon | MultiPolygon | None:
    """
    Generate concave hull using alpha shape method.

    Extracts points and computes the alpha shape (concave hull) using Shapely's
    implementation.

    Parameters
    ----------
    gs : geopandas.GeoSeries
        Input geometries.
    **kwargs : Any
        Additional arguments including 'hull_ratio', 'allow_holes'.

    Returns
    -------
    Polygon or MultiPolygon or None
        The concave hull geometry, or None if empty.
    """
    hull_ratio = kwargs.get("hull_ratio", 0.3)
    allow_holes = kwargs.get("allow_holes", False)
    points = _extract_points_from_geometries(gs)
    return (
        _concave_hull_alpha(points, ratio=hull_ratio, allow_holes=allow_holes) if points else None
    )


def _generate_convex_hull(
    gs: gpd.GeoSeries,
    **kwargs: Any,  # noqa: ANN401, ARG001
) -> Polygon | LineString | Point:
    """
    Generate convex hull.

    Computes the convex hull of the union of all input geometries. May return
    LineString or Point for degenerate cases, which are converted to polygons
    by the caller.

    Parameters
    ----------
    gs : geopandas.GeoSeries
        Input geometries.
    **kwargs : Any
        Additional arguments (unused).

    Returns
    -------
    Polygon or LineString or Point
        The convex hull geometry.
    """
    return gs.union_all().convex_hull


def _generate_buffer(
    gs: gpd.GeoSeries,
    **kwargs: Any,  # noqa: ANN401
) -> Polygon | MultiPolygon | None:
    """
    Generate buffer around geometries.

    Creates a buffer around the input geometries with the specified distance and
    style parameters.

    Parameters
    ----------
    gs : geopandas.GeoSeries
        Input geometries.
    **kwargs : Any
        Additional arguments including 'buffer_distance', 'cap_style', 'join_style', 'resolution'.

    Returns
    -------
    Polygon or MultiPolygon or None
        The buffered geometry, or None if empty.
    """
    buffer_distance = kwargs.get("buffer_distance", 100)

    # Early return if no buffering requested
    if buffer_distance is None:
        return gs.union_all()

    cap_style = kwargs.get("cap_style", 1)
    join_style = kwargs.get("join_style", 1)
    resolution = kwargs.get("resolution", 16)

    buffered = gs.buffer(
        buffer_distance, cap_style=cap_style, join_style=join_style, resolution=resolution
    )
    buffered = buffered[~buffered.is_empty & buffered.is_valid]
    return buffered.union_all() if not buffered.empty else None


def _extract_isochrone_geometries(
    reachable: nx.Graph | nx.MultiGraph,
    method: str,
) -> list[Any]:
    """
    Extract geometries from reachable subgraph for isochrone construction.

    Retrieves node positions and optionally edge geometries from the graph or
    GeoDataFrame, depending on the generation method.

    Parameters
    ----------
    reachable : networkx.Graph or networkx.MultiGraph or geopandas.GeoDataFrame
        The reachable subgraph or edge GeoDataFrame.
    method : str
        The isochrone generation method.

    Returns
    -------
    list
        A list of geometries (Points or LineStrings).
    """
    # Always extract node positions
    geoms = _extract_node_geometries(reachable)

    # Extract edge geometries only if needed
    if method in {"buffer", "concave_hull_alpha"}:
        geoms.extend(_extract_edge_geometries(reachable))

    return geoms


def _extract_node_geometries(graph: nx.Graph | nx.MultiGraph) -> list[Point]:
    """
    Extract node positions as Point geometries.

    Iterates through graph nodes and creates Point objects from their 'pos' attribute.

    Parameters
    ----------
    graph : networkx.Graph or networkx.MultiGraph
        The input graph.

    Returns
    -------
    list[Point]
        List of Point geometries for nodes.
    """
    pos_dict = nx.get_node_attributes(graph, "pos")
    return [Point(p) for p in pos_dict.values()] if pos_dict else []


def _extract_edge_geometries(graph: nx.Graph | nx.MultiGraph) -> list[Any]:
    """
    Extract edge geometries from graph.

    Iterates through graph edges and retrieves their 'geometry' attribute.

    Parameters
    ----------
    graph : networkx.Graph or networkx.MultiGraph
        The input graph.

    Returns
    -------
    list
        A list of edge geometries.
    """
    return [data["geometry"] for _, _, data in graph.edges(data=True) if "geometry" in data]


def _concave_hull_knn(points: list[Point] | np.ndarray, k: int) -> Polygon | LineString | Point:
    """
    Compute the concave hull of a set of points using the k-nearest neighbors approach.

    This function implements the k-nearest neighbors algorithm to generate a concave hull
    from a set of points. It constructs the hull by iteratively finding the next point
    that forms the largest right-turn angle, ensuring a tight fit around the point cloud.
    Based on the algorithm by Moreira and Santos (2007).

    Parameters
    ----------
    points : list[Point] or np.ndarray
        The input points.
    k : int
        The number of nearest neighbors to consider.

    Returns
    -------
    Polygon or LineString or Point
        The concave hull geometry.
    """
    # Handle degenerate cases
    if len(points) < 3:
        return points[0] if len(points) == 1 else LineString(points)

    # Prepare coordinates
    coords = np.array([(p.x, p.y) for p in points]) if isinstance(points, list) else points
    coords = np.unique(coords, axis=0)
    n_points = len(coords)

    if n_points < 3:
        return LineString(coords) if n_points == 2 else Point(coords[0])

    # Build KDTree and find starting point (lowest Y, then lowest X)
    tree = cKDTree(coords)
    start_idx = np.lexsort((coords[:, 0], coords[:, 1]))[0]

    # Initialize hull
    hull_indices = [start_idx]
    current_idx = start_idx
    prev_vec = np.array([1.0, 0.0])  # Initial direction: East
    visited = {start_idx}

    # Loop until we close the polygon or fail
    while True:
        next_idx = _find_next_hull_point(
            tree, coords, current_idx, k, prev_vec, hull_indices, start_idx, visited, n_points
        )

        if next_idx is None:
            # Fallback to convex hull if we get stuck
            return MultiPoint(points).convex_hull

        if next_idx == start_idx:
            return Polygon(coords[hull_indices])

        hull_indices.append(next_idx)
        visited.add(next_idx)
        current_idx = next_idx
        prev_vec = coords[next_idx] - coords[hull_indices[-2]]


def _find_next_hull_point(
    tree: cKDTree,
    coords: np.ndarray,
    current_idx: int,
    k: int,
    prev_vec: np.ndarray,
    hull_indices: list[int],
    start_idx: int,
    visited: set[int],
    n_points: int,
) -> int | None:
    """
    Find the next point in the concave hull.

    This helper function queries the KDTree for nearest neighbors and iterates through
    candidates to find the best next point that satisfies the concave hull criteria.
    It handles increasing k if no valid candidate is found initially.

    Parameters
    ----------
    tree : scipy.spatial.cKDTree
        The KDTree for nearest neighbor search.
    coords : np.ndarray
        Array of all point coordinates.
    current_idx : int
        Index of the current point in the hull.
    k : int
        The number of nearest neighbors to consider.
    prev_vec : np.ndarray
        Vector of the previous edge in the hull.
    hull_indices : list[int]
        List of indices currently in the hull.
    start_idx : int
        Index of the starting point of the hull.
    visited : set[int]
        Set of visited point indices.
    n_points : int
        Total number of points.

    Returns
    -------
    int or None
        The index of the next point, or None if no valid point found.
    """
    current_k = k
    while current_k <= n_points:
        # Query k nearest neighbors
        query_k = min(current_k + 1, n_points)
        _, indices = tree.query(coords[current_idx], k=query_k)

        if not isinstance(indices, (list, np.ndarray)):
            indices = [indices]

        # Filter candidates
        candidates = [
            idx
            for idx in indices
            if idx != current_idx
            and (idx not in visited or (idx == start_idx and len(hull_indices) >= 3))
        ]

        if candidates:
            best_idx = _find_best_candidate(
                coords, current_idx, candidates, prev_vec, hull_indices, start_idx
            )
            if best_idx is not None:
                return best_idx

        # Increase k if no valid candidate found
        current_k = min(current_k + 5, n_points)
        if current_k == n_points:  # Avoid infinite loop
            break

    return None


def _find_best_candidate(
    coords: np.ndarray,
    current_idx: int,
    candidates: list[int],
    prev_vec: np.ndarray,
    hull_indices: list[int],
    start_idx: int,
) -> int | None:
    """
    Find the best candidate point with the largest right-turn angle.

    This function evaluates a list of candidate points by calculating the angle
    deviation from the previous vector. It prioritizes points that form the sharpest
    right turn (largest inner angle) to ensure the hull wraps tightly around the shape.

    Parameters
    ----------
    coords : np.ndarray
        Array of all point coordinates.
    current_idx : int
        Index of the current point in the hull.
    candidates : list[int]
        List of indices of candidate points.
    prev_vec : np.ndarray
        Vector of the previous edge in the hull.
    hull_indices : list[int]
        List of indices currently in the hull.
    start_idx : int
        Index of the starting point of the hull.

    Returns
    -------
    int or None
        The index of the best candidate, or None if no valid candidate found.
    """
    current_pos = coords[current_idx]
    candidate_pos = coords[candidates]

    # Calculate vectors and normalize
    vecs = (candidate_pos - current_pos).astype(float)
    norms = np.linalg.norm(vecs, axis=1)

    # Filter zero-length vectors
    valid_mask = norms > 0
    if not np.any(valid_mask):
        return None

    vecs = vecs[valid_mask]
    candidates_array = np.array(candidates)[valid_mask]
    vecs /= norms[valid_mask][:, np.newaxis]

    # Calculate angles relative to previous vector
    prev_angle = np.arctan2(prev_vec[1], prev_vec[0])
    angles = np.arctan2(vecs[:, 1], vecs[:, 0])

    # Calculate angle difference (preference for right turns / largest inner angle)
    # We want the point that is "most right" relative to our current direction
    diffs = (angles - prev_angle + np.pi) % (2 * np.pi) - np.pi
    sorted_indices = np.argsort(diffs)

    # Check validity of candidates in order
    for idx in sorted_indices:
        candidate_idx = int(candidates_array[idx])
        new_edge = LineString([coords[current_idx], coords[candidate_idx]])

        if _is_valid_edge(new_edge, coords, hull_indices, start_idx, candidate_idx):
            return candidate_idx

    return None


def _is_valid_edge(
    new_edge: LineString,
    coords: np.ndarray,
    hull_indices: list[int],
    start_idx: int,
    candidate_idx: int,
) -> bool:
    """
    Check if the new edge intersects with any existing hull edges.

    This validation ensures that adding the proposed edge does not create a self-intersecting
    polygon. It checks for intersections with all non-adjacent edges in the current hull.

    Parameters
    ----------
    new_edge : shapely.geometry.LineString
        The potential new edge to add to the hull.
    coords : np.ndarray
        Array of all point coordinates.
    hull_indices : list[int]
        List of indices currently in the hull.
    start_idx : int
        Index of the starting point of the hull.
    candidate_idx : int
        Index of the candidate point for the new edge.

    Returns
    -------
    bool
        True if the edge is valid (no self-intersection), False otherwise.
    """
    if len(hull_indices) < 3:
        return True

    # Check against existing edges (excluding the immediate predecessor)
    # We only need to check segments that could possibly intersect
    # Optimization: check bounding box first? (Shapely does this internally)

    # Create a MultiLineString of existing edges to check against in one go?
    # Or just iterate. Iterating is fine for now.

    # We skip the last added edge (predecessor) because we share a vertex with it.
    for i in range(len(hull_indices) - 2):
        p1 = coords[hull_indices[i]]
        p2 = coords[hull_indices[i + 1]]
        existing_edge = LineString([p1, p2])

        if new_edge.intersects(existing_edge):
            intersection = new_edge.intersection(existing_edge)

            # Allow touching at start point when closing the loop
            if (
                candidate_idx == start_idx
                and i == 0
                and (isinstance(intersection, Point) or intersection.is_empty)
            ):
                continue

            # If intersection is just a point and it's not the shared vertex (which we skipped)
            # it might be a problem if it's not an endpoint.
            # But intersects() returns true for shared endpoints.
            # We skipped the immediate predecessor, so we shouldn't share endpoints with checked edges
            # UNLESS we are closing the loop (checked above).

            if not intersection.is_empty:
                # If it's a proper intersection (not just touching), reject
                # Or if it touches at a place that isn't allowed
                return False

    return True


def _extract_points_from_geometries(geoms: gpd.GeoSeries | list[Any]) -> list[Point]:
    """
    Extract all points from a list of geometries.

    This function uses shapely.get_coordinates for efficient extraction.

    Parameters
    ----------
    geoms : geopandas.GeoSeries or list
        Input geometries (Point, LineString, MultiPoint, Polygon, etc.).

    Returns
    -------
    list[Point]
        A list of shapely Points extracted from the inputs.
    """
    # Use shapely.get_coordinates for efficient extraction (requires shapely >= 2.0)
    # Since we use shapely.concave_hull elsewhere, we assume shapely >= 2.0
    coords = shapely.get_coordinates(geoms)
    return [Point(c) for c in coords]


def _concave_hull_alpha(
    points: list[Point], ratio: float, allow_holes: bool
) -> Polygon | MultiPolygon:
    """
    Compute the alpha shape (concave hull) of a set of points.

    This function uses shapely.concave_hull to generate the geometry.

    Parameters
    ----------
    points : list[Point]
        The input points.
    ratio : float
        The ratio for the concave hull (0.0 to 1.0).
    allow_holes : bool
        Whether to allow holes in the hull.

    Returns
    -------
    Polygon or MultiPolygon
        The computed alpha shape.
    """
    unique_coords = {(p.x, p.y) for p in points}
    if len(unique_coords) >= 3:
        unique_points = MultiPoint([Point(c) for c in unique_coords])
        return shapely.concave_hull(unique_points, ratio=ratio, allow_holes=allow_holes)

    # Fallback to convex hull for degenerate cases
    return MultiPoint(points).convex_hull


def create_tessellation(
    geometry: gpd.GeoDataFrame | gpd.GeoSeries,
    primary_barriers: gpd.GeoDataFrame | gpd.GeoSeries | None = None,
    shrink: float = 0.4,
    segment: float = 0.5,
    threshold: float = 0.05,
    n_jobs: int = -1,
    **kwargs: object,
) -> gpd.GeoDataFrame:
    """
    Create tessellations from given geometries, with optional barriers.

    This function generates either morphological or enclosed tessellations based on
    the input geometries. If `primary_barriers` are provided, it creates an
    enclosed tessellation; otherwise, it generates a morphological tessellation.

    Parameters
    ----------
    geometry : geopandas.GeoDataFrame or geopandas.GeoSeries
        The geometries (typically building footprints) to tessellate around.
    primary_barriers : geopandas.GeoDataFrame or geopandas.GeoSeries, optional
        Geometries (typically road network) to use as barriers for enclosed
        tessellation. If provided, `momepy.enclosed_tessellation` is used.
        Default is None.
    shrink : float, default 0.4
        The distance to shrink the geometry for the skeleton endpoint generation.
        Passed to `momepy.morphological_tessellation` or `momepy.enclosed_tessellation`.
    segment : float, default 0.5
        The segment length for discretizing the geometry. Passed to
        `momepy.morphological_tessellation` or `momepy.enclosed_tessellation`.
    threshold : float, default 0.05
        The threshold for snapping skeleton endpoints to the boundary. Only used
        for enclosed tessellation.
    n_jobs : int, default -1
        The number of jobs to use for parallel processing. -1 means using all
        available processors. Only used for enclosed tessellation.
    **kwargs : object, optional
        Additional keyword arguments passed to the underlying `momepy`
        tessellation function.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing the tessellation cells as polygons. Each cell
        has a unique `tess_id`.

    Raises
    ------
    ValueError
        If `primary_barriers` are not provided and the geometry is in a
        geographic CRS (e.g., EPSG:4326), as morphological tessellation
        requires a projected CRS.

    See Also
    --------
    momepy.morphological_tessellation : Generate morphological tessellation.
    momepy.enclosed_tessellation : Generate enclosed tessellation.

    Examples
    --------
    >>> import geopandas as gpd
    >>> from shapely.geometry import Polygon
    >>> # Create some building footprints
    >>> buildings = gpd.GeoDataFrame(
    ...     geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
    ...               Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])],
    ...     crs="EPSG:32633"
    ... )
    >>> # Generate morphological tessellation
    >>> tessellation = create_tessellation(buildings)
    >>> print(tessellation.head())

    >>> # Generate enclosed tessellation with roads as barriers
    >>> from shapely.geometry import LineString
    >>> roads = gpd.GeoDataFrame(
    ...     geometry=[LineString([(0, -1), (3, -1)]), LineString([(1.5, -1), (1.5, 4)])],
    ...     crs="EPSG:32633"
    ... )
    >>> enclosed_tess = create_tessellation(buildings, primary_barriers=roads)
    >>> print(enclosed_tess.head())
    """
    if geometry.empty:
        if primary_barriers is not None:
            # Enclosed tessellation needs 'enclosure_index' column
            return gpd.GeoDataFrame(
                columns=["geometry", "enclosure_index"],
                geometry="geometry",
                crs=geometry.crs,
            )
        return gpd.GeoDataFrame(
            columns=["geometry", "tess_id"],
            geometry="geometry",
            crs=geometry.crs,
        )

    if primary_barriers is not None:
        return _create_enclosed_tessellation(
            geometry,
            primary_barriers,
            shrink,
            segment,
            threshold,
            n_jobs,
            **kwargs,
        )

    return _create_morphological_tessellation(
        geometry,
        shrink,
        segment,
    )


def _create_enclosed_tessellation(
    geometry: gpd.GeoDataFrame | gpd.GeoSeries,
    primary_barriers: gpd.GeoDataFrame | gpd.GeoSeries,
    shrink: float,
    segment: float,
    threshold: float,
    n_jobs: int,
    **kwargs: object,
) -> gpd.GeoDataFrame:
    """
    Create enclosed tessellation.

    This helper generates enclosed tessellations using momepy, handling the
    creation of enclosures from barriers and managing potential errors during
    generation.

    Parameters
    ----------
    geometry : geopandas.GeoDataFrame or geopandas.GeoSeries
        The geometries to tessellate around.
    primary_barriers : geopandas.GeoDataFrame or geopandas.GeoSeries
        Geometries to use as barriers.
    shrink : float
        The distance to shrink the geometry.
    segment : float
        The segment length for discretizing the geometry.
    threshold : float
        The threshold for snapping skeleton endpoints.
    n_jobs : int
        The number of jobs to use for parallel processing.
    **kwargs : object
        Additional keyword arguments passed to momepy.

    Returns
    -------
    geopandas.GeoDataFrame
        The enclosed tessellation.
    """
    enclosures = momepy.enclosures(
        primary_barriers=primary_barriers,
        limit=None,
        additional_barriers=None,
        enclosure_id="eID",
        clip=False,
    )

    if not enclosures.empty:
        try:
            tessellation = momepy.enclosed_tessellation(
                geometry=geometry,
                enclosures=enclosures,
                shrink=shrink,
                segment=segment,
                threshold=threshold,
                n_jobs=n_jobs,
                **kwargs,
            )
        except ValueError as e:
            if "No objects to concatenate" in str(e):
                logger.warning(
                    "Momepy could not generate tessellation, returning empty GeoDataFrame.",
                )
                return _create_empty_tessellation(geometry.crs)
            raise
    else:
        tessellation = _create_empty_tessellation(geometry.crs, include_tess_id=False)

    if tessellation.empty:
        return _create_empty_tessellation(geometry.crs, include_tess_id=False)

    tessellation["tess_id"] = [
        f"{i}_{j}"
        for i, j in zip(tessellation["enclosure_index"], tessellation.index, strict=False)
    ]
    return tessellation.reset_index(drop=True)


def _create_empty_tessellation(
    crs: Any,  # noqa: ANN401
    include_tess_id: bool = True,
) -> gpd.GeoDataFrame:
    """
    Create an empty tessellation GeoDataFrame.

    This helper generates a properly structured but empty GeoDataFrame for tessellations,
    ensuring consistent column names and types when no tessellation can be generated.

    Parameters
    ----------
    crs : Any
        The Coordinate Reference System.
    include_tess_id : bool, default True
        Whether to include the 'tess_id' column.

    Returns
    -------
    geopandas.GeoDataFrame
        An empty tessellation GeoDataFrame.
    """
    columns = ["geometry", "enclosure_index"]
    if include_tess_id:
        columns.append("tess_id")
    return gpd.GeoDataFrame(columns=columns, geometry="geometry", crs=crs)


def _create_morphological_tessellation(
    geometry: gpd.GeoDataFrame | gpd.GeoSeries,
    shrink: float,
    segment: float,
) -> gpd.GeoDataFrame:
    """
    Create morphological tessellation.

    This helper generates morphological tessellations using momepy, which relies
    on the geometry itself without external barriers.

    Parameters
    ----------
    geometry : geopandas.GeoDataFrame or geopandas.GeoSeries
        The geometries to tessellate around.
    shrink : float
        The distance to shrink the geometry.
    segment : float
        The segment length for discretizing the geometry.

    Returns
    -------
    geopandas.GeoDataFrame
        The morphological tessellation.
    """
    tessellation = momepy.morphological_tessellation(
        geometry=geometry,
        clip="bounding_box",
        shrink=shrink,
        segment=segment,
    )

    tessellation["tess_id"] = tessellation.index
    return tessellation


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def validate_gdf(
    nodes_gdf: gpd.GeoDataFrame | dict[str, gpd.GeoDataFrame] | None = None,
    edges_gdf: gpd.GeoDataFrame | dict[tuple[str, str, str], gpd.GeoDataFrame] | None = None,
    allow_empty: bool = True,
) -> tuple[
    gpd.GeoDataFrame | dict[str, gpd.GeoDataFrame] | None,
    gpd.GeoDataFrame | dict[tuple[str, str, str], gpd.GeoDataFrame] | None,
    bool,
]:
    """
    Validate node and edge GeoDataFrames with type detection.

    This function validates both homogeneous and heterogeneous GeoDataFrame inputs,
    performs type checking, and determines whether the input represents a
    heterogeneous graph structure.

    Parameters
    ----------
    nodes_gdf : geopandas.GeoDataFrame or dict[str, geopandas.GeoDataFrame], optional
        The GeoDataFrame containing node data to validate, or a dictionary mapping
        node type names to GeoDataFrames for heterogeneous graphs.
    edges_gdf : geopandas.GeoDataFrame or dict[tuple[str, str, str], geopandas.GeoDataFrame], optional
        The GeoDataFrame containing edge data to validate, or a dictionary mapping
        edge type tuples to GeoDataFrames for heterogeneous graphs.
    allow_empty : bool, default True
        If True, allows the GeoDataFrames to be empty. If False, raises an error.

    Returns
    -------
    tuple
        A tuple containing:

        - validated nodes_gdf (same type as input)
        - validated edges_gdf (same type as input)
        - is_hetero: boolean indicating if this is a heterogeneous graph

    Raises
    ------
    TypeError
        If an input is not a GeoDataFrame or appropriate dictionary type.
    ValueError
        If the input types are inconsistent or invalid.

    See Also
    --------
    validate_nx : Validate a NetworkX graph.

    Examples
    --------
    >>> import geopandas as gpd
    >>> from shapely.geometry import Point, LineString
    >>> nodes = gpd.GeoDataFrame(geometry=[Point(0, 0)])
    >>> edges = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 1)])])
    >>> try:
    ...     validated_nodes, validated_edges, is_hetero = validate_gdf(nodes, edges)
    ...     print(f"Validation successful. Heterogeneous: {is_hetero}")
    ... except (TypeError, ValueError) as e:
    ...     print(f"Validation failed: {e}")
    Validation successful. Heterogeneous: False
    """
    processor = GeoDataProcessor()

    # Type detection and validation
    is_nodes_dict = isinstance(nodes_gdf, dict)
    is_edges_dict = isinstance(edges_gdf, dict)

    # Check for type consistency
    if is_nodes_dict and edges_gdf is not None and not is_edges_dict:
        msg = "If nodes is a dict, edges must also be a dict or None."
        raise TypeError(msg)
    if not is_nodes_dict and is_edges_dict and nodes_gdf is not None:
        msg = "If edges is a dict, nodes must also be a dict or None."
        raise TypeError(msg)

    is_hetero = is_nodes_dict or is_edges_dict

    validated_nodes: dict[str, gpd.GeoDataFrame] | gpd.GeoDataFrame | None = None
    validated_edges: dict[tuple[str, str, str], gpd.GeoDataFrame] | gpd.GeoDataFrame | None = None

    if is_hetero:
        # Validate heterogeneous inputs
        if nodes_gdf is not None:
            validated_nodes = {}
            for node_type, node_gdf in nodes_gdf.items():
                if not isinstance(node_type, str):
                    msg = "Node type keys must be strings"
                    raise TypeError(msg)
                validated_nodes[node_type] = processor.validate_gdf(node_gdf, allow_empty=True)

        if edges_gdf is not None:
            validated_edges = {}
            for edge_type, edge_gdf in edges_gdf.items():
                if not isinstance(edge_type, tuple) or len(edge_type) != 3:
                    msg = (
                        "Edge type keys must be tuples of (source_type, relation_type, target_type)"
                    )
                    raise TypeError(msg)
                if not all(isinstance(t, str) for t in edge_type):
                    msg = "All elements in edge type tuples must be strings"
                    raise TypeError(msg)
                validated_edges[edge_type] = processor.validate_gdf(
                    edge_gdf,
                    ["LineString", "MultiLineString"],
                    allow_empty=allow_empty,
                )
    else:
        # Validate homogeneous inputs
        if nodes_gdf is not None:
            validated_nodes = processor.validate_gdf(
                nodes_gdf,
                allow_empty=allow_empty,
            )

        if edges_gdf is not None:
            validated_edges = processor.validate_gdf(
                edges_gdf,
                ["LineString", "MultiLineString"],
                allow_empty=allow_empty,
            )

    # Ensure CRS consistency
    all_gdfs_to_check: list[gpd.GeoDataFrame] = []
    if validated_nodes is not None:
        all_gdfs_to_check.extend(
            validated_nodes.values() if is_hetero else [validated_nodes],
        )
    if validated_edges is not None:
        all_gdfs_to_check.extend(
            validated_edges.values() if is_hetero else [validated_edges],
        )

    processor.ensure_crs_consistency(*all_gdfs_to_check)

    return validated_nodes, validated_edges, is_hetero


def validate_nx(graph: nx.Graph | nx.MultiGraph) -> None:
    """
    Validate a NetworkX graph with comprehensive type checking.

    Checks if the input is a NetworkX graph, ensures it is not empty
    (i.e., it has both nodes and edges), and verifies that it contains the
    necessary metadata for conversion back to GeoDataFrames or PyG objects.

    Parameters
    ----------
    graph : networkx.Graph or networkx.MultiGraph
        The NetworkX graph to validate.

    Returns
    -------
    None
        This function does not return a value.

    Raises
    ------
    TypeError
        If the input is not a NetworkX graph.
    ValueError
        If the graph has no nodes, no edges, or is missing essential metadata.

    See Also
    --------
    validate_gdf : Validate GeoDataFrames for graph conversion.

    Examples
    --------
    >>> import networkx as nx
    >>> from shapely.geometry import Point
    >>> G = nx.Graph(is_hetero=False, crs="EPSG:4326")
    >>> G.add_node(0, pos=(0, 0))
    >>> G.add_node(1, pos=(1, 1))
    >>> G.add_edge(0, 1)
    >>> try:
    ...     validate_nx(G)
    ...     print("Validation successful.")
    ... except (TypeError, ValueError) as e:
    ...     print(f"Validation failed: {e}")
    Validation successful.
    """
    # Type validation
    if not isinstance(graph, (nx.Graph, nx.MultiGraph)):
        msg = "Input must be a NetworkX Graph or MultiGraph"
        raise TypeError(msg)

    processor = GeoDataProcessor()
    processor.validate_nx(graph)


def nx_to_rx(graph: nx.Graph | nx.MultiGraph) -> rx.PyGraph | rx.PyDiGraph:
    """
    Convert a NetworkX graph to a rustworkx graph.

    This function converts a NetworkX graph object into a rustworkx graph object,
    preserving node, edge, and graph attributes. It handles both directed and
    undirected graphs, as well as multigraphs.

    Parameters
    ----------
    graph : networkx.Graph or networkx.MultiGraph
        The NetworkX graph to convert.

    Returns
    -------
    rustworkx.PyGraph or rustworkx.PyDiGraph
        The converted rustworkx graph.

    See Also
    --------
    rx_to_nx : Convert a rustworkx graph back to NetworkX.

    Examples
    --------
    >>> rx_G = nx_to_rx(G)
    """
    is_directed = graph.is_directed()
    is_multigraph = graph.is_multigraph()

    out_graph = (
        rx.PyDiGraph(multigraph=is_multigraph)
        if is_directed
        else rx.PyGraph(multigraph=is_multigraph)
    )

    # Copy graph attributes
    out_graph.attrs = graph.graph.copy()

    # Add nodes and attributes
    node_mapping = {}
    for node, data in graph.nodes(data=True):
        # Prepare node payload
        payload = data.copy()

        # Store original NetworkX node ID for reversibility
        payload["__nx_node_id__"] = node

        # Add node to rustworkx graph
        rx_idx = out_graph.add_node(payload)

        # Map original NetworkX node ID to rustworkx node index
        node_mapping[node] = rx_idx

    # Add edges and attributes
    edges_to_add = []
    if is_multigraph:
        for u, v, k, data in graph.edges(data=True, keys=True):
            payload = data.copy()
            payload["__nx_edge_key__"] = k
            edges_to_add.append((node_mapping[u], node_mapping[v], payload))
    else:
        for u, v, data in graph.edges(data=True):
            edges_to_add.append((node_mapping[u], node_mapping[v], data.copy()))

    out_graph.add_edges_from(edges_to_add)

    return out_graph


def rx_to_nx(graph: rx.PyGraph | rx.PyDiGraph) -> nx.Graph | nx.MultiGraph:
    """
    Convert a rustworkx graph to a NetworkX graph.

    This function converts a rustworkx graph object into a NetworkX graph object,
    restoring node, edge, and graph attributes.

    Parameters
    ----------
    graph : rustworkx.PyGraph or rustworkx.PyDiGraph
        The rustworkx graph to convert.

    Returns
    -------
    networkx.Graph or networkx.MultiGraph
        The converted NetworkX graph.

    See Also
    --------
    nx_to_rx : Convert a NetworkX graph to rustworkx.

    Examples
    --------
    >>> # Assuming rx_G is a rustworkx graph
    >>> nx_G = rx_to_nx(rx_G)
    """
    is_directed = isinstance(graph, rx.PyDiGraph)
    is_multigraph = graph.multigraph

    out_graph = (
        (nx.MultiDiGraph() if is_multigraph else nx.DiGraph())
        if is_directed
        else nx.MultiGraph()
        if is_multigraph
        else nx.Graph()
    )

    # Restore graph attributes
    if isinstance(graph.attrs, dict):
        out_graph.graph.update(graph.attrs)

    # Restore nodes and attributes
    rx_to_nx_mapping = {}
    for rx_idx in graph.node_indices():
        payload = graph[rx_idx]
        node_data = {}

        # Attempt to retrieve original node ID and data
        if isinstance(payload, dict):
            node_data = payload.copy()
            nx_id = node_data.pop("__nx_node_id__", rx_idx)
        else:
            nx_id = rx_idx
            node_data = {"payload": payload} if payload is not None else {}

        out_graph.add_node(nx_id, **node_data)
        rx_to_nx_mapping[rx_idx] = nx_id

    # Restore edges and attributes
    # edge_index_map returns {edge_idx: (u_idx, v_idx, data)}
    for u_rx, v_rx, data in graph.edge_index_map().values():
        u_nx = rx_to_nx_mapping[u_rx]
        v_nx = rx_to_nx_mapping[v_rx]
        edge_data = (
            data if isinstance(data, dict) else ({"payload": data} if data is not None else {})
        )

        if is_multigraph:
            key = edge_data.pop("__nx_edge_key__", None)
            out_graph.add_edge(u_nx, v_nx, key, **edge_data)

        else:
            out_graph.add_edge(u_nx, v_nx, **edge_data)

    return out_graph


# =============================================================================
# PLOTTING UTILITIES
# =============================================================================

PLOT_DEFAULTS = {
    "node_color": "#22d3ee",  # Cyan
    "node_edgecolor": "none",
    "node_alpha": 0.8,
    "node_zorder": 2,
    "markersize": 4.0,
    "edge_color": "#ffffff",  # White
    "edge_linewidth": 0.5,
    "edge_alpha": 0.3,
    "edge_zorder": 1,
    "title_color": "white",
}


def plot_graph(  # noqa: PLR0913
    graph: nx.Graph | nx.MultiGraph | None = None,
    nodes: gpd.GeoDataFrame | dict[str, gpd.GeoDataFrame] | None = None,
    edges: gpd.GeoDataFrame | dict[tuple[str, str, str], gpd.GeoDataFrame] | None = None,
    ax: "matplotlib.axes.Axes | np.ndarray | None" = None,
    bgcolor: str = "#000000",
    figsize: tuple[float, float] = (12, 12),
    subplots: bool = True,
    ncols: int | None = None,
    legend_position: str | None = "upper left",
    labelcolor: str = "white",
    title_color: str | None = None,
    node_color: str | float | pd.Series | dict[str, Any] | None = None,
    node_alpha: float | pd.Series | dict[str, Any] | None = None,
    node_zorder: int | pd.Series | dict[str, Any] | None = None,
    node_edgecolor: str | pd.Series | dict[str, Any] | None = None,
    markersize: float | pd.Series | dict[str, Any] | None = None,
    edge_color: str | float | pd.Series | dict[tuple[str, str, str], Any] | None = None,
    edge_linewidth: float | pd.Series | dict[tuple[str, str, str], Any] | None = None,
    edge_alpha: float | pd.Series | dict[tuple[str, str, str], Any] | None = None,
    edge_zorder: int | pd.Series | dict[tuple[str, str, str], Any] | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> "matplotlib.axes.Axes | np.ndarray | None":
    """
    Plot a graph with a unified interface.

    This function provides a unified interface for plotting spatial network data,
    supporting both GeoDataFrame-based and NetworkX-based inputs. NetworkX graphs
    are automatically converted to GeoDataFrames before plotting. It can handle
    homogeneous and heterogeneous graphs with customizable styling.

    Parameters
    ----------
    graph : networkx.Graph or networkx.MultiGraph, optional
        The NetworkX graph to plot. If provided without nodes/edges, the function
        will convert it to GeoDataFrames before plotting.
    nodes : geopandas.GeoDataFrame or dict[str, geopandas.GeoDataFrame], optional
        Nodes to plot. Can be a single GeoDataFrame (homogeneous) or a dictionary
        mapping node type names to GeoDataFrames (heterogeneous).
    edges : geopandas.GeoDataFrame or dict[tuple[str, str, str], geopandas.GeoDataFrame], optional
        Edges to plot. Can be a single GeoDataFrame (homogeneous) or a dictionary
        mapping edge type tuples (src_type, rel_type, dst_type) to GeoDataFrames (heterogeneous).
    ax : matplotlib.axes.Axes or numpy.ndarray, optional
        The axes on which to plot. If None, a new figure and axes are created.
    bgcolor : str, default "#000000"
        Background color for the plot (Black theme).
    figsize : tuple[float, float], default (12, 12)
        Figure size as (width, height) in inches.
    subplots : bool, default True
        If True and the graph is heterogeneous, plot each node/edge type in a
        separate subplot. Uses 'ax' as array of subplots if provided.
    ncols : int, optional
        Number of columns (subplots per row) when plotting heterogeneous graphs
        with subplots=True. If None, defaults to min(3, number_of_edge_types).
    legend_position : str or None, default "upper left"
        Position of the legend for heterogeneous graphs. Common values include
        "upper left", "upper right", "lower left", "lower right", "center", etc.
        If None, no legend is displayed.
    labelcolor : str, default "white"
        Color of the legend text labels.
    title_color : str, optional
        Color for subplot titles when ``subplots=True``. Falls back to a white
        title on black backgrounds if not provided.
    node_color : str, float, pd.Series, or dict, optional
        Color for nodes. Can be a scalar, column name, Series, or a dictionary
        mapping node types to colors for heterogeneous graphs.
    node_alpha : float, pd.Series, or dict, optional
        Transparency for nodes (0.0-1.0). Can be a scalar, column name, Series,
        or a dictionary mapping node types to transparency values.
    node_zorder : int, pd.Series, or dict, optional
        Drawing order for nodes. Can be a scalar, column name, Series, or a
        dictionary mapping node types to zorder values.
    node_edgecolor : str, pd.Series, or dict, optional
        Color for node borders. Can be a scalar, column name, Series, or a
        dictionary mapping node types to edge colors.
    markersize : float, pd.Series, or dict, optional
        Size of the node markers. Can be a scalar, column name, Series, or a
        dictionary mapping node types to marker sizes.
    edge_color : str, float, pd.Series, or dict, optional
        Color for edges. Can be a scalar, column name, Series, or a dictionary
        mapping edge types to colors for heterogeneous graphs.
    edge_linewidth : float, pd.Series, or dict, optional
        Line width for edges. Can be a scalar, column name, Series, or a
        dictionary mapping edge types to line widths.
    edge_alpha : float, pd.Series, or dict, optional
        Transparency for edges (0.0-1.0). Can be a scalar, column name, Series,
        or a dictionary mapping edge types to transparency values.
    edge_zorder : int, pd.Series, or dict, optional
        Drawing order for edges. Can be a scalar, column name, Series, or a
        dictionary mapping edge types to zorder values.
    **kwargs : Any
        Additional keyword arguments passed to the GeoPandas plotting functions.

        Supports attribute-based styling where parameters can be specified as:

        - **Scalar values** (str/float): Applied uniformly to all geometries
        - **Column names** (str): If the string matches a column in the GeoDataFrame,
          that column's values are used for styling
        - **pd.Series**: Direct values for each geometry

        Other common options: etc.

    Returns
    -------
    matplotlib.axes.Axes or numpy.ndarray or None
        The axes object(s) used for plotting.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If no valid input is provided (all parameters are None).
    TypeError
        If the input data types are not supported.

    Examples
    --------
    >>> import geopandas as gpd
    >>> import pandas as pd
    >>> import networkx as nx
    >>> # Plot from NetworkX graph (automatically converted to GeoDataFrames)
    >>> G = nx.Graph()
    >>> G.add_node(0, pos=(0, 0))
    >>> G.add_edge(0, 1)
    >>> plot_graph(graph=G)
    >>> # Plot from GeoDataFrames with scalar styling
    >>> plot_graph(nodes=nodes_gdf, edges=edges_gdf, node_color='red')
    >>> # Plot with attribute-based node colors (by column name)
    >>> plot_graph(nodes=nodes_gdf, edges=edges_gdf, node_color='building_type')
    >>> # Plot with pd.Series for edge linewidth
    >>> edge_widths = pd.Series([1.0, 2.0, 1.5], index=edges_gdf.index)
    >>> plot_graph(nodes=nodes_gdf, edges=edges_gdf, edge_linewidth=edge_widths)
    >>> # Plot heterogeneous graph
    >>> plot_graph(nodes=nodes_dict, edges=edges_dict)
    """
    if not MATPLOTLIB_AVAILABLE:
        msg = "Matplotlib is required for plotting functionality."
        raise ImportError(msg)

    # Input validation
    if graph is None and nodes is None and edges is None:
        msg = "At least one of graph, nodes, or edges must be provided"
        raise ValueError(msg)

    # Convert NetworkX graph to GeoDataFrames if provided
    nodes, edges = _normalize_graph_input(graph, nodes, edges)

    # Collect style arguments
    style_kwargs = {
        "node_color": node_color,
        "node_alpha": node_alpha,
        "node_zorder": node_zorder,
        "node_edgecolor": node_edgecolor,
        "markersize": markersize,
        "edge_color": edge_color,
        "edge_linewidth": edge_linewidth,
        "edge_alpha": edge_alpha,
        "edge_zorder": edge_zorder,
        "legend_position": legend_position,
        "labelcolor": labelcolor,
        "title_color": title_color,
        **kwargs,
    }

    # Handle heterogeneous subplots
    is_hetero = isinstance(nodes, dict) or isinstance(edges, dict)

    if subplots and is_hetero:
        return _plot_hetero_subplots(
            nodes, edges, figsize, bgcolor, ax=ax, ncols=ncols, **style_kwargs
        )

    # Setup figure and axes
    if ax is None:
        ax = _setup_plot_axes(figsize, bgcolor)
    elif not isinstance(ax, np.ndarray):
        # Apply bgcolor to provided axes
        ax.set_facecolor(bgcolor)
        ax.set_axis_off()
        if hasattr(ax, "figure") and ax.figure is not None:
            ax.figure.patch.set_facecolor(bgcolor)

    # GeoDataFrame-based plotting
    is_hetero = isinstance(nodes, dict) or isinstance(edges, dict)
    if is_hetero:
        _plot_hetero_graph(nodes, edges, ax, **style_kwargs)
    else:
        _plot_homo_graph(nodes, edges, ax, **style_kwargs)

    return ax


def _setup_plot_axes(
    figsize: tuple[float, float],
    bgcolor: str,
) -> "matplotlib.axes.Axes":
    """
    Create and return a matplotlib axes configured for C2G plots.

    Centralizes figure styling for all C2G visualizations.

    Parameters
    ----------
    figsize : tuple[float, float]
        Width and height of the figure in inches.
    bgcolor : str
        Background color for figure and axes.

    Returns
    -------
    matplotlib.axes.Axes
        Configured axes instance.
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(bgcolor)
    ax.set_facecolor(bgcolor)
    ax.set_axis_off()
    return ax


def _normalize_graph_input(
    graph: nx.Graph | nx.MultiGraph | None,
    nodes: gpd.GeoDataFrame | dict[str, gpd.GeoDataFrame] | None,
    edges: gpd.GeoDataFrame | dict[tuple[str, str, str], gpd.GeoDataFrame] | None,
) -> tuple[
    gpd.GeoDataFrame | dict[str, gpd.GeoDataFrame] | None,
    gpd.GeoDataFrame | dict[tuple[str, str, str], gpd.GeoDataFrame] | None,
]:
    """
    Normalize graph input to GeoDataFrames.

    Converts various input formats into standardized GeoDataFrames.

    Parameters
    ----------
    graph : networkx.Graph or networkx.MultiGraph, optional
        NetworkX graph to convert.
    nodes : geopandas.GeoDataFrame or dict, optional
        Existing nodes data.
    edges : geopandas.GeoDataFrame or dict, optional
        Existing edges data.

    Returns
    -------
    tuple
        A tuple of (nodes, edges) as GeoDataFrames or dictionaries.
    """
    if graph is not None and nodes is None and edges is None:
        if isinstance(graph, (nx.Graph, nx.MultiGraph)):
            converter = NxConverter()
            nodes, edges = converter.nx_to_gdf(graph, nodes=True, edges=True)
        elif isinstance(graph, gpd.GeoDataFrame):
            nodes = graph
        else:
            msg = f"Unsupported data type for graph parameter: {type(graph)}"
            raise TypeError(msg)
    return nodes, edges


def _resolve_plot_parameter(
    gdf: gpd.GeoDataFrame,
    param_value: str | float | pd.Series | None,
    _param_name: str,
    default_value: Any,  # noqa: ANN401
) -> str | float | pd.Series:
    """
    Resolve a plot parameter to a value usable by GeoPandas plot().

    Handles None, Series, column names, and scalar values.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing potential attribute columns.
    param_value : str, float, pd.Series, or None
        The parameter value to resolve.
    _param_name : str
        Name of the parameter (unused).
    default_value : Any
        Default value if param_value is None.

    Returns
    -------
    str, float, or pd.Series
        Resolved parameter value.
    """
    if param_value is None:
        return default_value  # type: ignore[no-any-return]
    if isinstance(param_value, pd.Series):
        return param_value
    if isinstance(param_value, str) and param_value in gdf.columns:
        return gdf[param_value]  # type: ignore[no-any-return]
    return param_value


def _get_color_for_type(i: int) -> str | tuple[float, float, float, float]:
    """
    Get color for a specific type index using matplotlib's tab10 colormap.

    Uses cyclic indexing to support any number of types.

    Parameters
    ----------
    i : int
        The index of the current type being colored.

    Returns
    -------
    str or tuple
        A color from the tab10 colormap.
    """
    cmap = plt.get_cmap("tab10")
    return cmap(i % 10)


def _resolve_type_parameter(
    param: Any,  # noqa: ANN401
    type_key: str | tuple[str, str, str],
) -> Any:  # noqa: ANN401
    """
    Resolve a parameter that might be a dictionary keyed by type.

    Looks up value if param is a dict, otherwise returns param as-is.

    Parameters
    ----------
    param : Any
        The parameter value (scalar or dict).
    type_key : str or tuple
        The key identifying the node or edge type.

    Returns
    -------
    Any
        The resolved parameter value for the specific type.
    """
    return param.get(type_key) if isinstance(param, dict) else param


def _resolve_style_kwargs(
    global_kwargs: dict[str, Any],
    type_key: str | tuple[str, str, str] | None,
    is_edge: bool,
    default_color: Any = None,  # noqa: ANN401
) -> dict[str, Any]:
    """
    Resolve style arguments for a specific graph element type.

    Extracts and resolves style parameters from kwargs, handling type-specific
    overrides if a type_key is provided.

    Parameters
    ----------
    global_kwargs : dict
        The kwargs passed to the main plot_graph function.
    type_key : str or tuple or None
        The key identifying the node or edge type. None for homogeneous graphs.
    is_edge : bool
        True if resolving for edges, False for nodes.
    default_color : Any, optional
        Fallback color if not specified in kwargs.

    Returns
    -------
    dict
        Resolved style arguments ready for _plot_gdf.
    """

    def _get_param(name: str) -> Any:  # noqa: ANN401
        """
        Get parameter value from kwargs with optional type-specific lookup.

        This helper looks up a parameter by name from the enclosing function's
        global_kwargs dict, applying type-specific resolution if a type_key is set.

        Parameters
        ----------
        name : str
            Parameter name to look up.

        Returns
        -------
        Any
            The parameter value, or type-specific value if type_key is set.
        """
        val = global_kwargs.get(name)
        return _resolve_type_parameter(val, type_key) if type_key else val

    def _or_default(val: Any, default: Any) -> Any:  # noqa: ANN401
        """
        Return val if not None, otherwise return default.

        This helper provides a None-safe default value fallback, ensuring that
        explicit None values are replaced with the specified default.

        Parameters
        ----------
        val : Any
            Value to check.
        default : Any
            Default value to use if val is None.

        Returns
        -------
        Any
            Val if not None, otherwise default.
        """
        return val if val is not None else default

    # Keys reserved for C2G-specific styling logic
    c2g_keys = {
        "node_color",
        "node_alpha",
        "node_zorder",
        "node_edgecolor",
        "markersize",
        "edge_color",
        "edge_linewidth",
        "edge_alpha",
        "edge_zorder",
        "legend_position",
        "labelcolor",
        "title_color",
    }

    # Start with all global kwargs, resolving potential type-specific dictionaries
    resolved = {}
    for k, v in global_kwargs.items():
        if k not in c2g_keys:
            resolved[k] = _resolve_type_parameter(v, type_key) if type_key else v

    if is_edge:
        resolved.update(
            {
                "linewidth": _or_default(
                    _get_param("edge_linewidth"), PLOT_DEFAULTS["edge_linewidth"]
                ),
                "alpha": _or_default(_get_param("edge_alpha"), PLOT_DEFAULTS["edge_alpha"]),
                "zorder": _or_default(_get_param("edge_zorder"), PLOT_DEFAULTS["edge_zorder"]),
                "color": _or_default(_get_param("edge_color"), default_color),
            }
        )
    else:
        resolved.update(
            {
                "color": _or_default(
                    _get_param("node_color"), default_color or PLOT_DEFAULTS["node_color"]
                ),
                "alpha": _or_default(_get_param("node_alpha"), PLOT_DEFAULTS["node_alpha"]),
                "zorder": _or_default(_get_param("node_zorder"), PLOT_DEFAULTS["node_zorder"]),
                "edgecolor": _or_default(
                    _get_param("node_edgecolor"), PLOT_DEFAULTS["node_edgecolor"]
                ),
                "markersize": _or_default(_get_param("markersize"), PLOT_DEFAULTS["markersize"]),
            }
        )
    return resolved


def _plot_gdf(
    gdf: gpd.GeoDataFrame,
    ax: Any,  # noqa: ANN401
    **kwargs: Any,  # noqa: ANN401
) -> None:
    """
    Plot a GeoDataFrame with resolved parameters.

    Delegates to GeoPandas plot method after resolving style parameters.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The GeoDataFrame to plot.
    ax : matplotlib.axes.Axes
        The axes to plot on.
    **kwargs : Any
        Style parameters (color, alpha, linewidth, markersize, zorder, etc.).
    """
    if gdf.empty:
        return

    # Start with all kwargs as potential plot arguments
    plot_kwargs: dict[str, Any] = kwargs.copy()
    plot_kwargs["ax"] = ax

    param_defaults = {
        "color": PLOT_DEFAULTS["node_color"],
        "alpha": PLOT_DEFAULTS["node_alpha"],
        "linewidth": PLOT_DEFAULTS["edge_linewidth"],
        "markersize": PLOT_DEFAULTS["markersize"],
        "zorder": PLOT_DEFAULTS["node_zorder"],
        "edgecolor": PLOT_DEFAULTS["edge_color"],
    }

    for param_name, default_val in param_defaults.items():
        val = _resolve_plot_parameter(gdf, kwargs.get(param_name), param_name, default_val)
        if val is not None:
            if param_name == "color" and isinstance(val, pd.Series):
                plot_kwargs["column"] = val
                plot_kwargs.pop("color", None)
            else:
                plot_kwargs[param_name] = val

    # Handle label separately to avoid it being resolved as a column name
    if "label" in kwargs:
        plot_kwargs["label"] = kwargs["label"]

    gdf.plot(**plot_kwargs)


def _plot_hetero_graph(
    nodes: dict[str, gpd.GeoDataFrame] | None,
    edges: dict[tuple[str, str, str], gpd.GeoDataFrame] | None,
    ax: Any,  # noqa: ANN401
    **kwargs: Any,  # noqa: ANN401
) -> None:
    """
    Plot heterogeneous graph with per-type styling and legend.

    Draws edges and nodes grouped by their semantic types.

    Parameters
    ----------
    nodes : dict[str, geopandas.GeoDataFrame], optional
        Mapping of node type names to GeoDataFrames.
    edges : dict[tuple[str, str, str], geopandas.GeoDataFrame], optional
        Mapping of edge type tuples to GeoDataFrames.
    ax : matplotlib.axes.Axes
        Axes instance for rendering.
    **kwargs : Any
        Additional styling arguments.
    """
    # Plot edges first
    if edges is not None and isinstance(edges, dict):
        for i, (edge_type, edge_gdf) in enumerate(edges.items()):
            default_color = _get_color_for_type(i)
            style_kwargs = _resolve_style_kwargs(
                kwargs,
                edge_type,
                is_edge=True,
                default_color=default_color,
            )
            _plot_gdf(edge_gdf, ax, label=str(edge_type), **style_kwargs)

    # Plot nodes
    if nodes is not None and isinstance(nodes, dict):
        for i, (node_type, node_gdf) in enumerate(nodes.items()):
            default_color = _get_color_for_type(i)
            style_kwargs = _resolve_style_kwargs(
                kwargs,
                node_type,
                is_edge=False,
                default_color=default_color,
            )
            _plot_gdf(node_gdf, ax, label=node_type, **style_kwargs)

    # Add legend for heterogeneous plots with sophisticated styling
    _style_legend(ax, **kwargs)


def _style_legend(ax: Any, **kwargs: Any) -> None:  # noqa: ANN401
    """
    Apply sophisticated styling to the legend.

    This helper configures the legend appearance, including position, frame,
    label styling, and handle adjustments for better readability.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to style the legend for.
    **kwargs : Any
        Additional styling arguments.
    """
    legend_position = kwargs.get("legend_position", "upper left")
    if legend_position is not None and ax.get_legend_handles_labels()[0]:
        legend = ax.legend(
            loc=legend_position,
            frameon=False,  # Remove frame/border
            labelcolor=kwargs.get("labelcolor", "white"),
            fontsize=8,
            handlelength=1.5,  # Shorter legend handles
            handleheight=1.2,
            handletextpad=0.5,  # Reduce space between handle and text
            borderpad=0.3,
            labelspacing=0.3,  # Reduce spacing between labels
            markerscale=0.5,  # Fixed marker scale in legend
        )
        # Make legend markers non-transparent and fixed size for better recognition
        for handle in legend.legend_handles:
            handle.set_alpha(1.0)
            # Set fixed marker size for point markers (nodes)
            if hasattr(handle, "set_markersize"):
                handle.set_markersize(5)  # Fixed size for all node markers
            if hasattr(handle, "set_sizes"):
                handle.set_sizes([25])  # Fixed size for scatter plot markers
            # Increase line width for line markers (edges)
            if hasattr(handle, "set_linewidth"):
                handle.set_linewidth(2.5)


def _plot_hetero_subplots(
    nodes: dict[str, gpd.GeoDataFrame] | None,
    edges: dict[tuple[str, str, str], gpd.GeoDataFrame] | None,
    figsize: tuple[float, float],
    bgcolor: str,
    ax: "matplotlib.axes.Axes | np.ndarray | None" = None,
    ncols: int | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> "matplotlib.axes.Axes | np.ndarray | None":
    """
    Plot heterogeneous graph components in separate subplots.

    Creates a grid layout with one subplot per edge type.

    Parameters
    ----------
    nodes : dict[str, geopandas.GeoDataFrame], optional
        Mapping of node type names to GeoDataFrames.
    edges : dict[tuple[str, str, str], geopandas.GeoDataFrame], optional
        Mapping of edge type tuples to GeoDataFrames.
    figsize : tuple[float, float]
        Figure size (width, height) in inches.
    bgcolor : str
        Background color for figure and axes.
    ax : matplotlib.axes.Axes or numpy.ndarray, optional
        Existing axes to plot on. Can be a single axis or an array of axes.
    ncols : int, optional
        Number of columns in the subplot grid. If None, defaults to
        min(3, number_of_edge_types).
    **kwargs : Any
        Additional styling arguments.

    Returns
    -------
    matplotlib.axes.Axes or numpy.ndarray or None
        The axes object(s) used for plotting.
    """
    # Collect non-empty edge types to plot
    edge_items = [(k, v) for k, v in (edges or {}).items() if not v.empty]
    n_items = len(edge_items)
    if n_items == 0:
        return None

    returned_axes: matplotlib.axes.Axes | np.ndarray | None = None

    if ax is None:
        # Calculate grid layout
        cols = ncols if ncols is not None else min(3, n_items)
        cols = max(1, min(cols, n_items))  # Ensure cols is between 1 and n_items
        rows = math.ceil(n_items / cols)

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        fig.patch.set_facecolor(bgcolor)

        returned_axes = axes

        # Ensure axes is iterable
        axes_flat = [axes] if n_items == 1 else axes.flatten()
    else:
        # Use provided axes
        returned_axes = ax
        if hasattr(ax, "flatten"):
            axes_flat = ax.flatten()
        elif isinstance(ax, (list, tuple)):
            axes_flat = list(ax)
        else:
            axes_flat = [ax]

        # Apply bgcolor to provided axes' figure if available
        if (
            len(axes_flat) > 0
            and hasattr(axes_flat[0], "figure")
            and axes_flat[0].figure is not None
        ):
            axes_flat[0].figure.patch.set_facecolor(bgcolor)

    # Calculate total bounds for fixed extent
    xlim, ylim = _calculate_total_bounds(nodes, edges)

    # Plot each edge type
    for _, (subplot_ax, (edge_key, edge_gdf)) in enumerate(
        zip(axes_flat, edge_items, strict=False)
    ):
        subplot_ax.set_facecolor(bgcolor)
        subplot_ax.set_axis_off()
        subplot_ax.set_xlim(xlim)
        subplot_ax.set_ylim(ylim)

        # Get colors for this subplot
        colors = {
            "edge": None,
            "src": None,
            "dst": None,
        }

        _plot_hetero_subplot_item(subplot_ax, edge_key, edge_gdf, nodes, colors, **kwargs)

    # Hide unused axes
    for j in range(len(edge_items), len(axes_flat)):
        axes_flat[j].set_visible(False)

    return returned_axes


def _calculate_total_bounds(
    nodes: dict[str, gpd.GeoDataFrame] | None,
    edges: dict[tuple[str, str, str], gpd.GeoDataFrame] | None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Calculate total bounds for all nodes and edges with 5% padding.

    Computes the combined bounding box of all GeoDataFrames.

    Parameters
    ----------
    nodes : dict, optional
        Dictionary of node GeoDataFrames.
    edges : dict, optional
        Dictionary of edge GeoDataFrames.

    Returns
    -------
    tuple
        A tuple of ((minx, maxx), (miny, maxy)).
    """
    bounds = [float("inf"), float("inf"), float("-inf"), float("-inf")]

    for gdf_dict in [nodes, edges]:
        if gdf_dict:
            for gdf in gdf_dict.values():
                if not gdf.empty:
                    b = gdf.total_bounds
                    bounds[0] = min(bounds[0], b[0])
                    bounds[1] = min(bounds[1], b[1])
                    bounds[2] = max(bounds[2], b[2])
                    bounds[3] = max(bounds[3], b[3])

    dx, dy = bounds[2] - bounds[0], bounds[3] - bounds[1]
    pad_x, pad_y = dx * 0.05, dy * 0.05
    return (bounds[0] - pad_x, bounds[2] + pad_x), (bounds[1] - pad_y, bounds[3] + pad_y)


def _plot_hetero_subplot_item(
    ax: Any,  # noqa: ANN401
    edge_key: tuple[str, str, str],
    edge_gdf: gpd.GeoDataFrame,
    nodes: dict[str, gpd.GeoDataFrame] | None,
    colors: dict[str, Any],
    **kwargs: Any,  # noqa: ANN401
) -> None:
    """
    Plot a single heterogeneous subplot item.

    Renders an edge type and its connected source/target nodes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    edge_key : tuple
        The edge type tuple.
    edge_gdf : geopandas.GeoDataFrame
        The edge GeoDataFrame.
    nodes : dict, optional
        Dictionary of node GeoDataFrames.
    colors : dict
        Dictionary of fallback colors for this subplot.
    **kwargs : Any
        Additional arguments.
    """
    # Plot edges with resolved styling
    style_kwargs_edge = _resolve_style_kwargs(
        kwargs,
        edge_key,
        is_edge=True,
        default_color=colors.get("edge", PLOT_DEFAULTS["edge_color"]),
    )
    # Ensure alpha is at least 0.5 for visibility in subplots if not specified
    if kwargs.get("edge_alpha") is None:
        style_kwargs_edge["alpha"] = 0.5

    _plot_gdf(edge_gdf, ax, label=str(edge_key), **style_kwargs_edge)

    # Plot connected nodes
    src_type, _, dst_type = edge_key

    if nodes and src_type in nodes and not nodes[src_type].empty:
        style_kwargs_src = _resolve_style_kwargs(
            kwargs,
            src_type,
            is_edge=False,
            default_color=colors.get("src", PLOT_DEFAULTS["node_color"]),
        )
        _plot_gdf(nodes[src_type], ax, label=src_type, **style_kwargs_src)

    if nodes and dst_type in nodes and not nodes[dst_type].empty and dst_type != src_type:
        style_kwargs_dst = _resolve_style_kwargs(
            kwargs,
            dst_type,
            is_edge=False,
            default_color=colors.get("dst", PLOT_DEFAULTS["node_color"]),
        )
        _plot_gdf(nodes[dst_type], ax, label=dst_type, **style_kwargs_dst)

    # Set title
    title_color = kwargs.get("title_color")
    ax.set_title(
        f"{edge_key}",
        color=title_color if title_color is not None else PLOT_DEFAULTS["title_color"],
        fontsize=10,
    )


def _plot_homo_graph(
    nodes: gpd.GeoDataFrame | None,
    edges: gpd.GeoDataFrame | None,
    ax: Any,  # noqa: ANN401
    **kwargs: Any,  # noqa: ANN401
) -> None:
    """
    Plot homogeneous graph (edges first, then nodes on top).

    Renders a single node/edge GeoDataFrame pair on the provided axes.

    Parameters
    ----------
    nodes : geopandas.GeoDataFrame, optional
        Node geometries to render.
    edges : geopandas.GeoDataFrame, optional
        Edge geometries to render.
    ax : matplotlib.axes.Axes
        Target axes for rendering.
    **kwargs : Any
        Additional styling arguments.
    """
    # Plot edges first (in background)
    if edges is not None and isinstance(edges, gpd.GeoDataFrame):
        style_kwargs = _resolve_style_kwargs(kwargs, None, is_edge=True)
        _plot_gdf(edges, ax, **style_kwargs)

    # Plot nodes on top
    if nodes is not None and isinstance(nodes, gpd.GeoDataFrame):
        style_kwargs = _resolve_style_kwargs(kwargs, None, is_edge=False)
        _plot_gdf(nodes, ax, **style_kwargs)
