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
from typing import Any

# Third-party imports
import geopandas as gpd
import momepy
import networkx as nx
import pandas as pd
import rustworkx as rx
from shapely.geometry import LineString
from shapely.geometry import Point

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

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
    "segments_to_graph",
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
        tuple
            A tuple containing the reconstructed GeoDataFrames. For homogeneous graphs,
            it's `(nodes_gdf, edges_gdf)`. For heterogeneous graphs, it's
            `(nodes_dict, edges_dict)`.

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

        # Create graph and metadata
        if self.multigraph:
            graph = nx.MultiDiGraph() if self.directed else nx.MultiGraph()
        else:
            graph = nx.DiGraph() if self.directed else nx.Graph()
        metadata = GraphMetadata(crs=edges.crs, is_hetero=False)

        # Add nodes
        if nodes is not None:
            self._add_homogeneous_nodes(graph, nodes)
            metadata.node_geom_cols = list(nodes.select_dtypes(include=["geometry"]).columns)
            if isinstance(nodes.index, pd.MultiIndex):
                # Ensure we have proper list of strings for node index names
                metadata.node_index_names = [str(name) for name in nodes.index.names]
            else:
                metadata.node_index_names = [
                    str(nodes.index.name) if nodes.index.name is not None else "index",
                ]

        # Add edges
        self._add_homogeneous_edges(graph, edges, nodes)
        metadata.edge_geom_cols = list(edges.select_dtypes(include=["geometry"]).columns)
        metadata.edge_index_names = edges.index.names

        # Store metadata
        graph.graph.update(metadata.to_dict())
        return graph

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

        # Create graph and metadata
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

        # Add nodes and edges
        if nodes_dict:
            self._add_heterogeneous_nodes(graph, nodes_dict, metadata)
        if edges_dict:
            self._add_heterogeneous_edges(graph, edges_dict, metadata)

        # Store metadata
        graph.graph.update(metadata.to_dict())
        return graph

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

        if nodes_gdf is not None and not nodes_gdf.empty:
            # Use node mapping
            coord_to_node = {
                node_data["pos"]: node_id for node_id, node_data in graph.nodes(data=True)
            }

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
        else:
            # Use coordinate tuples as node IDs
            start_coords = self.processor.extract_coordinates(edges_gdf, start=True)
            end_coords = self.processor.extract_coordinates(edges_gdf, start=False)

            # Add unique nodes
            all_coords = pd.concat([start_coords, end_coords]).unique()
            nodes_to_add = [(coord, {"pos": coord}) for coord in all_coords]
            graph.add_nodes_from(nodes_to_add)

            self._create_edge_list(graph, start_coords, end_coords, edges_gdf, self.keep_geom)

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
        if metadata.node_index_names is None or not isinstance(metadata.node_index_names, dict):
            metadata.node_index_names = {}

        node_offset = {}
        current_offset = 0

        for node_type, node_gdf in nodes_dict.items():
            node_offset[node_type] = current_offset
            metadata.node_index_names[node_type] = node_gdf.index.name

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
        if metadata.edge_index_names is None:
            metadata.edge_index_names = {}

        # Ensure edge_index_names is a dict for type safety
        if not isinstance(metadata.edge_index_names, dict):
            metadata.edge_index_names = {}

        for edge_type, edge_gdf in edges_dict.items():
            # Get edge type components
            src_type, rel_type, dst_type = edge_type
            metadata.edge_index_names[edge_type] = edge_gdf.index.names

            # Create node lookup
            node_lookup = self._create_node_lookup(graph, [src_type, dst_type])

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
        else:
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

        if metadata.node_index_names is None:
            metadata.node_index_names = {}
        if metadata.edge_index_names is None:
            metadata.edge_index_names = {}

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

        # Extract original indices and create records
        original_indices = [attrs.get("_original_index", nid) for nid, attrs in node_data.items()]

        # Use list comprehension for records, prioritize geometry over pos
        records = [
            {
                **{k: v for k, v in attrs.items() if k not in ["pos", "_original_index"]},
                "geometry": attrs["geometry"]
                if "geometry" in attrs and attrs["geometry"] is not None
                else (Point(attrs["pos"]) if "pos" in attrs else None),
            }
            for nid, attrs in node_data.items()
        ]

        index_names = metadata.node_index_names

        # Handle different types of index_names
        if isinstance(index_names, list):
            if len(index_names) > 1:
                # Ensure we have valid tuples for MultiIndex
                tuple_indices = [
                    tuple(idx) if isinstance(idx, (list, tuple)) else (idx,)
                    for idx in original_indices
                ]
                index = pd.MultiIndex.from_tuples(tuple_indices, names=index_names)
            else:
                # Single level index
                name = index_names[0] if index_names else None
                index = pd.Index(original_indices, name=name)
        else:
            # Handle str, None, or other types
            index = pd.Index(
                original_indices,
                name=index_names if isinstance(index_names, str) else None,
            )

        gdf = gpd.GeoDataFrame(records, index=index, crs=metadata.crs)

        # Convert geometry columns
        for col in metadata.node_geom_cols:
            if col in gdf.columns:
                gdf[col] = gpd.GeoSeries(gdf[col], crs=metadata.crs)

        return gdf

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
            indices = [attrs.get("_original_index") for attrs in attrs_list]

            # Use list comprehension for records, prioritize geometry over pos
            records = [
                {
                    **{
                        k: v
                        for k, v in attrs.items()
                        if k not in ["pos", "node_type", "_original_index"]
                    },
                    "geometry": attrs["geometry"]
                    if "geometry" in attrs and attrs["geometry"] is not None
                    else (Point(attrs["pos"]) if "pos" in attrs else None),
                }
                for attrs in attrs_list
            ]

            # Handle index names safely
            index_names = (
                metadata.node_index_names.get(node_type)
                if isinstance(metadata.node_index_names, dict)
                else None
            )

            index = (
                pd.Index(indices, name=index_names)
                if isinstance(index_names, str)
                else pd.Index(indices, name=None)
            )
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
            return gpd.GeoDataFrame({"weight": [], "geometry": []}, crs=metadata.crs)

        is_multigraph = isinstance(graph, nx.MultiGraph)
        if is_multigraph:
            edge_data = list(graph.edges(data=True, keys=True))
            original_indices = [
                attrs.get("_original_edge_index", (u, v, k)) for u, v, k, attrs in edge_data
            ]
        else:
            edge_data = list(graph.edges(data=True))
            original_indices = [
                attrs.get("_original_edge_index", (u, v)) for u, v, attrs in edge_data
            ]

        records = []
        for edge in edge_data:
            if is_multigraph:
                # Multigraph edges have format (u, v, k, attrs)
                u, v, _, attrs = edge
            else:
                # Regular edges have format (u, v, attrs)
                u, v, attrs = edge

            geom = attrs.get("geometry")
            if geom is None and "pos" in graph.nodes[u] and "pos" in graph.nodes[v]:
                geom = LineString([graph.nodes[u]["pos"], graph.nodes[v]["pos"]])

            records.append(
                {
                    **{
                        k: v
                        for k, v in attrs.items()
                        if k not in ["_original_edge_index", "weight"]
                    },
                    "weight": attrs.get("weight"),
                    "geometry": geom,
                },
            )

        # Handle MultiIndex
        if original_indices and isinstance(original_indices[0], tuple):
            index = pd.MultiIndex.from_tuples(original_indices)
        else:
            index = pd.Index(original_indices)

        gdf = gpd.GeoDataFrame(records, index=index, crs=metadata.crs)

        # Restore index names
        index_names = metadata.edge_index_names

        if index_names and hasattr(gdf.index, "names"):
            gdf.index.names = index_names

        # Convert geometry columns
        for col in metadata.edge_geom_cols:
            if col in gdf.columns:
                gdf[col] = gpd.GeoSeries(gdf[col], crs=metadata.crs)

        return gdf

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
        is_multigraph = isinstance(graph, nx.MultiGraph)

        for edge_type in metadata.edge_types:
            src_type, rel_type, dst_type = edge_type

            if is_multigraph:
                multigraph_edges: list[tuple[object, object, object, dict[str, object]]] = [
                    (u, v, k, d)
                    for u, v, k, d in graph.edges(data=True, keys=True)
                    if d.get("edge_type") == edge_type
                ]
                # Convert to unified format for processing
                type_edges: list[tuple[object, object, object, dict[str, object]]] = (
                    multigraph_edges
                )
            else:
                regular_edges: list[tuple[object, object, dict[str, object]]] = [
                    (u, v, d)
                    for u, v, d in graph.edges(data=True)
                    if d.get("edge_type") == edge_type
                ]
                # Convert to unified format for processing (adding None for key)
                type_edges = [(u, v, None, d) for u, v, d in regular_edges]

            if not type_edges:
                edges_dict[edge_type] = gpd.GeoDataFrame(geometry=[], crs=metadata.crs)
                continue

            original_indices = [edge[-1].get("_original_edge_index") for edge in type_edges]
            records = []
            for edge in type_edges:
                # Unified format: (u, v, k_or_None, attrs)
                u, v, k, attrs = edge

                geom = attrs.get("geometry")
                if geom is None and "pos" in graph.nodes[u] and "pos" in graph.nodes[v]:
                    geom = LineString([graph.nodes[u]["pos"], graph.nodes[v]["pos"]])

                records.append(
                    {
                        **{
                            k: v
                            for k, v in attrs.items()
                            if k not in ["full_edge_type", "_original_edge_index"]
                        },
                        "geometry": geom,
                    },
                )

            # Handle MultiIndex - ensure we have proper tuples
            if original_indices and all(isinstance(idx, (tuple, list)) for idx in original_indices):
                # Ensure all elements are properly converted to tuples
                tuple_indices = []
                for idx in original_indices:
                    if isinstance(idx, list):
                        tuple_indices.append(tuple(idx))
                    elif isinstance(idx, tuple):
                        tuple_indices.append(idx)
                    else:
                        tuple_indices.append((idx,))
                multi_index = pd.MultiIndex.from_tuples(tuple_indices)
                gdf = gpd.GeoDataFrame(
                    records,
                    geometry="geometry",
                    index=multi_index,
                    crs=metadata.crs,
                )
            else:
                # Fall back to regular index if not proper tuples
                regular_index = pd.Index(original_indices)
                gdf = gpd.GeoDataFrame(
                    records,
                    geometry="geometry",
                    index=regular_index,
                    crs=metadata.crs,
                )

            # Restore index names safely
            if isinstance(metadata.edge_index_names, dict):
                index_names = metadata.edge_index_names.get(edge_type)
                if isinstance(index_names, list) and hasattr(gdf.index, "names"):
                    gdf.index.names = index_names

            edges_dict[edge_type] = gdf

        return edges_dict


class GraphAnalyzer:
    """
    Unified graph analysis operations.

    This class provides methods for analyzing and filtering graphs based on
    spatial and topological criteria. It encapsulates functionalities for
    spatial filtering, isochrone generation, and other graph-based analyses
    relevant to urban and spatial data.
    """

    def __init__(self) -> None:
        """
        Initialize GraphAnalyzer with processor and converter instances.

        This constructor creates a new GraphAnalyzer instance with default
        GeoDataProcessor and NxConverter components for spatial analysis.
        """
        self.processor = GeoDataProcessor()
        self.converter = NxConverter()

    def filter_graph_by_distance(
        self,
        graph: gpd.GeoDataFrame | nx.Graph | nx.MultiGraph,
        center_point: Point | gpd.GeoSeries,
        distance: float,
        edge_attr: str = "length",
        node_id_col: str | None = None,
    ) -> gpd.GeoDataFrame | nx.Graph | nx.MultiGraph:
        """
        Filter a graph to include only elements within a specified distance.

        This function extracts a subgraph containing all nodes and edges that are
        within a given shortest-path distance from a specified center point. It can
        operate on both NetworkX graphs and GeoDataFrames of edges, making it a
        versatile tool for spatial network analysis.

        The filtering process involves identifying the nearest graph nodes to the
        `center_point` and then performing a shortest-path search (e.g., Dijkstra's
        algorithm) to find all reachable nodes and edges within the specified `distance`.
        The resulting subgraph maintains the original graph's structure and attributes,
        allowing for further analysis on the spatially constrained network.
        algorithm) to find all reachable nodes and edges within the specified `distance`.
        The resulting subgraph maintains the original graph's structure and attributes,
        allowing for further analysis on the spatially constrained network.

        Parameters
        ----------
        graph : geopandas.GeoDataFrame or networkx.Graph or networkx.MultiGraph
            The graph to filter.
        center_point : Point or geopandas.GeoSeries
            The origin point(s) for the distance calculation.
        distance : float
            The maximum shortest-path distance.
        edge_attr : str, default "length"
            The edge attribute to use as weight for path calculations.
        node_id_col : str, optional
            The node identifier column if the input is a GeoDataFrame.

        Returns
        -------
        gpd.GeoDataFrame or nx.Graph or nx.MultiGraph
            The filtered subgraph, with the same type as the input.
        """
        is_graph_input = isinstance(graph, (nx.Graph, nx.MultiGraph))

        # Convert to NetworkX if needed
        if is_graph_input:
            nx_graph = graph
            original_crs = nx_graph.graph.get("crs")
        else:
            nx_graph = self.converter.gdf_to_nx(edges=graph)
            original_crs = graph.crs if hasattr(graph, "crs") else None

        # Extract node positions
        pos_dict = self._extract_node_positions(nx_graph)
        if not pos_dict:
            graph_type = type(graph) if is_graph_input else nx.Graph
            return self._create_empty_result(is_graph_input, original_crs, graph_type)

        # Create nodes GeoDataFrame for distance calculations
        node_id_name = node_id_col or "node_id"
        nodes_gdf = self._create_nodes_gdf(pos_dict, node_id_name, original_crs)

        # Normalize center points
        center_points = self._normalize_center_points(center_point)

        # Compute nodes within distance
        nodes_within_distance = self._compute_nodes_within_distance(
            nx_graph,
            center_points,
            nodes_gdf,
            distance,
            edge_attr,
            node_id_name,
        )

        # Create subgraph
        subgraph = nx_graph.subgraph(nodes_within_distance)

        if is_graph_input:
            return subgraph

        # Convert back to GeoDataFrame
        return self.converter.nx_to_gdf(subgraph, nodes=False, edges=True)

    def create_isochrone(
        self,
        graph: gpd.GeoDataFrame | nx.Graph | nx.MultiGraph,
        center_point: Point | gpd.GeoSeries | gpd.GeoDataFrame,
        distance: float,
        edge_attr: str = "length",
    ) -> gpd.GeoDataFrame:
        """
        Generate an isochrone polygon for a given graph and center point.

        This function computes the area reachable from a center point within a specified
        distance along the network. It first filters the graph to find all reachable
        nodes and edges, then generates a convex hull around them to create the
        isochrone polygon, which is useful for visualizing accessibility.

        Parameters
        ----------
        graph : geopandas.GeoDataFrame or networkx.Graph or networkx.MultiGraph
            The network graph.
        center_point : Point or geopandas.GeoSeries or geopandas.GeoDataFrame
            The origin point(s) for the isochrone.
        distance : float
            The maximum travel distance defining the isochrone boundary.
        edge_attr : str, default "length"
            The edge attribute to use as travel cost.

        Returns
        -------
        geopandas.GeoDataFrame
            A GeoDataFrame containing the isochrone polygon.
        """
        reachable = self.filter_graph_by_distance(graph, center_point, distance, edge_attr)

        # Convert to GeoDataFrame if NetworkX
        if isinstance(reachable, (nx.Graph, nx.MultiGraph)):
            reachable = self.converter.nx_to_gdf(reachable, nodes=False, edges=True)

        if reachable.empty:
            return gpd.GeoDataFrame(geometry=[], crs=getattr(reachable, "crs", None))

        # Create convex hull
        union_geom = reachable.union_all()
        hull = union_geom.convex_hull
        return gpd.GeoDataFrame(geometry=[hull], crs=reachable.crs)

    def _extract_node_positions(
        self,
        graph: nx.Graph | nx.MultiGraph,
    ) -> dict[object, object] | None:
        """
        Extract node positions from a NetworkX graph.

        This helper function retrieves the spatial positions of nodes from their
        `pos` attribute in the graph. It provides a consistent way to access
        coordinate data, which is fundamental for any spatial analysis or
        conversion task.

        Parameters
        ----------
        graph : networkx.Graph or networkx.MultiGraph
            The graph from which to extract node positions.

        Returns
        -------
        dict[object, object] or None
            A dictionary mapping node identifiers to their position tuples, or None
            if no `pos` attribute is found.
        """
        pos_dict: dict[object, object] = nx.get_node_attributes(graph, "pos")

        if pos_dict:
            return pos_dict

        return None

    def _create_nodes_gdf(
        self,
        pos_dict: dict[object, object],
        node_id_col: str,
        crs: str | int | None,
    ) -> gpd.GeoDataFrame:
        """
        Create a GeoDataFrame from a dictionary of node positions.

        This function converts a dictionary of node positions into a GeoDataFrame,
        which is a necessary intermediate step for performing spatial operations
        like distance calculations. It creates Point geometries from the coordinates
        and assigns the specified CRS.

        Parameters
        ----------
        pos_dict : dict[object, object]
            A dictionary mapping node identifiers to their position tuples.
        node_id_col : str
            The name to assign to the node identifier column.
        crs : str or int or None
            The Coordinate Reference System to assign to the new GeoDataFrame.

        Returns
        -------
        geopandas.GeoDataFrame
            A GeoDataFrame of nodes with Point geometries.
        """
        node_ids, coordinates = zip(*pos_dict.items(), strict=False)
        geometries = [Point(coord) for coord in coordinates]

        return gpd.GeoDataFrame(
            {node_id_col: node_ids, "geometry": geometries},
            crs=crs,
        )

    def _normalize_center_points(
        self,
        center_point: Point | gpd.GeoSeries,
    ) -> list[Point] | gpd.GeoSeries:
        """
        Normalize the center point input to a consistent format.

        This helper function ensures that the center point input, whether it's a
        single Point or a GeoSeries of points, is handled consistently. It returns
        a list or GeoSeries of points that can be iterated over for distance
        calculations.

        Parameters
        ----------
        center_point : Point or geopandas.GeoSeries
            The center point(s) to normalize.

        Returns
        -------
        list[Point] or geopandas.GeoSeries
            A list or GeoSeries of center points.
        """
        if isinstance(center_point, gpd.GeoSeries):
            return center_point
        return [center_point]

    def _compute_nodes_within_distance(
        self,
        graph: nx.Graph | nx.MultiGraph,
        center_points: list[Point] | gpd.GeoSeries,
        nodes_gdf: gpd.GeoDataFrame,
        distance: float,
        edge_attr: str,
        node_id_name: str,
    ) -> set[object]:
        """
        Compute the set of all nodes within a given distance from any center point.

        This function first finds the nearest graph node for each center point and then
        performs a single-source Dijkstra search from each of these source nodes to find
        all reachable nodes within the specified distance. The results from all
        searches are combined into a single set of unique node identifiers.

        Parameters
        ----------
        graph : networkx.Graph or networkx.MultiGraph
            The graph to search within.
        center_points : list[Point] or geopandas.GeoSeries
            The center points for the distance calculation.
        nodes_gdf : geopandas.GeoDataFrame
            A GeoDataFrame of the graph nodes, used for finding the nearest nodes.
        distance : float
            The maximum shortest-path distance.
        edge_attr : str
            The edge attribute to use as weight.
        node_id_name : str
            The name of the node identifier column.

        Returns
        -------
        set[object]
            A set of node identifiers that are within the distance.
        """
        center_points_list = (
            center_points.tolist() if hasattr(center_points, "tolist") else list(center_points)
        )

        # Get nearest nodes for all center points
        source_nodes = []
        for point in center_points_list:
            nearest_node = self._get_nearest_node(point, nodes_gdf, node_id_name)
            source_nodes.append(nearest_node)

        # Compute single-source shortest paths from all sources
        all_reachable = set()
        for source in source_nodes:
            lengths = nx.single_source_dijkstra_path_length(
                graph,
                source,
                cutoff=distance,
                weight=edge_attr,
            )
            all_reachable.update(lengths.keys())
        return all_reachable

    def _get_nearest_node(
        self,
        point: Point | gpd.GeoSeries,
        nodes_gdf: gpd.GeoDataFrame,
        node_id: str,
    ) -> object:
        """
        Find the nearest node in a GeoDataFrame to a given point.

        This function efficiently finds the closest node in a GeoDataFrame of nodes
        to a specified point by calculating the Euclidean distance. It is a key step
        in connecting off-network locations to the graph for analysis.

        Parameters
        ----------
        point : Point or geopandas.GeoSeries
            The point to find the nearest node to.
        nodes_gdf : geopandas.GeoDataFrame
            The GeoDataFrame of nodes to search within.
        node_id : str
            The name of the node identifier column.

        Returns
        -------
        object
            The identifier of the nearest node.
        """
        nearest_idx = nodes_gdf.distance(point).idxmin()
        return nodes_gdf.loc[nearest_idx, node_id]

    def _create_empty_result(
        self,
        is_graph_input: bool,
        original_crs: str | int | None,
        graph_type: type = nx.Graph,
    ) -> gpd.GeoDataFrame | nx.Graph | nx.MultiGraph:
        """
        Create an empty result in the appropriate format.

        This helper function generates an empty result container—either a GeoDataFrame
        or a NetworkX graph—that matches the expected output type. This is used to
        provide a consistent return value when an operation results in no data, such
        as when no nodes are found within a given distance.

        Parameters
        ----------
        is_graph_input : bool
            True if the original input was a NetworkX graph.
        original_crs : str or int or None
            The original CRS to assign to an empty GeoDataFrame.
        graph_type : type, default networkx.Graph
            The type of empty graph to create if needed.

        Returns
        -------
        gpd.GeoDataFrame or nx.Graph or nx.MultiGraph
            An empty result of the appropriate type.
        """
        return (
            gpd.GeoDataFrame(geometry=[], crs=original_crs) if not is_graph_input else graph_type()
        )


# =============================================================================
# PUBLIC API FUNCTIONS
# =============================================================================


def dual_graph(
    graph: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph | nx.MultiGraph,
    edge_id_col: str | None = None,
    keep_original_geom: bool = False,
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
    ...     {"edge_id": ["a", "b"]},
    ...     geometry=[LineString([(0, 0), (1, 1)]), LineString([(1, 1), (1, 0)])],
    ...     crs="EPSG:32633"
    ... ).set_index(pd.MultiIndex.from_tuples([(0, 1), (1, 2)]))
    >>> # Convert to dual graph
    >>> dual_nodes, dual_edges = dual_graph(
    ...     graph=(nodes, edges), edge_id_col="edge_id", keep_original_geom=True
    ... )
    >>> print(dual_nodes)
    >>> print(dual_edges)
    >>>                     geometry      original_geometry    mm_len
    ... edge_id
    ... a        LINESTRING (0 0, 1 1)  LINESTRING (0 0, 1 1)  1.414214
    ... b        LINESTRING (1 1, 1 0)  LINESTRING (1 1, 1 0)  1.000000
    ...                          angle  geometry
    ... from_edge_id to_edge_id
    ... a            b           135.0  LINESTRING (0.5 0.5, 1 0.5)
    """
    processor = GeoDataProcessor()

    # Validate input type
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

    processor.ensure_crs_consistency(nodes_gdf, edges_gdf)

    # Validate edges_gdf is a GeoDataFrame and clean it.
    # This will raise TypeError for non-GDF input, fixing one test failure.
    edges_clean = processor.validate_gdf(
        edges_gdf,
        ["LineString", "MultiLineString"],
        allow_empty=True,
    )

    # Handle empty or cleaned-to-empty edges GeoDataFrame.
    # This will fix the StopIteration test failure.
    if edges_clean is None or edges_clean.empty:
        crs = getattr(edges_gdf, "crs", None)
        dual_nodes = gpd.GeoDataFrame(geometry=[], crs=crs)
        dual_edges = gpd.GeoDataFrame(geometry=[], crs=crs)
        return dual_nodes, dual_edges

    # edges_clean is guaranteed to be non-None and non-empty here
    assert edges_clean is not None
    assert not edges_clean.empty

    if keep_original_geom:
        edges_clean["original_geometry"] = gpd.GeoSeries(
            edges_clean.geometry.copy(),
            crs=edges_clean.crs,
        )

    # If no edge_id_col, we'll use the index. Let's add it as a column
    # so it's carried over as a node attribute in the dual graph.
    preserve_index = edge_id_col is None
    # momepy uses the index of the input GDF as node IDs in the dual graph
    graph_nx = momepy.gdf_to_nx(
        edges_clean,
        approach="dual",
        multigraph=False,
        preserve_index=preserve_index,
    )

    # Ensure all edges from the primal graph are present as nodes in the dual graph, with their attributes.
    if preserve_index:
        if edges_clean is not None:
            node_attrs = edges_clean.to_dict("index")
            for node_id, attrs in node_attrs.items():
                if node_id not in graph_nx.nodes:
                    graph_nx.add_node(node_id, **attrs)
    elif edges_clean is not None:
        records = edges_clean.to_dict("records")
        nodes_to_add = [(i, attrs) for i, attrs in enumerate(records) if i not in graph_nx.nodes]
        graph_nx.add_nodes_from(nodes_to_add)

    # Add edge attributes of geometry as "geometry" of linestrings between centroids of nodes
    for u, v in graph_nx.edges():
        u_geom = graph_nx.nodes[u]["geometry"]
        v_geom = graph_nx.nodes[v]["geometry"]
        line = LineString([u_geom.centroid, v_geom.centroid])
        graph_nx.edges[u, v]["geometry"] = line

    # Convert the NetworkX graph to GeoDataFrames
    dual_nodes, dual_edges = nx_to_gdf(graph_nx, nodes=True, edges=True)

    # Ensure dual_nodes is a GeoDataFrame for type checking
    assert isinstance(dual_nodes, gpd.GeoDataFrame)
    assert isinstance(dual_edges, gpd.GeoDataFrame)

    # Handle index mapping based on whether edge_id_col is provided
    if edge_id_col is not None:
        # Create a mapping from the old index (used by momepy) to the new index values
        id_map = dual_nodes[edge_id_col]

        # Set the new index for the dual nodes
        dual_nodes = dual_nodes.set_index(edge_id_col)

        # Remap the dual edges' MultiIndex to use the new node IDs
        if isinstance(dual_edges, gpd.GeoDataFrame) and not dual_edges.empty:
            level_0 = dual_edges.index.get_level_values(0).map(id_map).to_list()
            level_1 = dual_edges.index.get_level_values(1).map(id_map).to_list()
            dual_edges.index = pd.MultiIndex.from_arrays([level_0, level_1])
            dual_edges.index.names = [f"from_{edge_id_col}", f"to_{edge_id_col}"]
    # When edge_id_col is None, use the existing index structure
    elif isinstance(dual_edges, gpd.GeoDataFrame) and not dual_edges.empty:
        dual_edges.index.names = ["from_edge_id", "to_edge_id"]

    return dual_nodes, dual_edges if not as_nx else gdf_to_nx(dual_nodes, dual_edges)


def segments_to_graph(
    segments_gdf: gpd.GeoDataFrame,
    multigraph: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    r"""
    Convert a GeoDataFrame of LineString segments into a graph structure.

    This function takes a GeoDataFrame of LineStrings and processes it into a
    topologically explicit graph representation, consisting of a GeoDataFrame of
    unique nodes (the endpoints of the lines) and a GeoDataFrame of edges.

    The resulting nodes GeoDataFrame contains unique points representing the start
    and end points of the input line segments. The edges GeoDataFrame is a copy
    of the input, but with a new MultiIndex (`from_node_id`, `to_node_id`) that
    references the IDs in the new nodes GeoDataFrame. If `multigraph` is True
    and there are multiple edges between the same pair of nodes, an additional
    index level (`edge_key`) is added to distinguish them.

    Parameters
    ----------
    segments_gdf : geopandas.GeoDataFrame
        A GeoDataFrame where each row represents a line segment, and the
        'geometry' column contains LineString objects.
    multigraph : bool, default False
        If True, supports multiple edges between the same pair of nodes by
        adding an `edge_key` level to the MultiIndex. This is useful when
        the input contains duplicate node-to-node connections that should
        be preserved as separate edges.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame]
        A tuple containing two GeoDataFrames:

        - nodes_gdf: A GeoDataFrame of unique nodes (Points), indexed by `node_id`.
        - edges_gdf: A GeoDataFrame of edges (LineStrings), with a MultiIndex
          mapping to the `node_id` in `nodes_gdf`. If `multigraph` is True,
          the index includes a third level (`edge_key`) for duplicate connections.

    Examples
    --------
    >>> import geopandas as gpd
    >>> from shapely.geometry import LineString
    >>> # Create a GeoDataFrame of line segments
    >>> segments = gpd.GeoDataFrame(
    ...     {"road_name": ["A", "B"]},
    ...     geometry=[LineString([(0, 0), (1, 1)]), LineString([(1, 1), (1, 0)])],
    ...     crs="EPSG:32633"
    ... )
    >>> # Convert to graph representation
    >>> nodes_gdf, edges_gdf = segments_to_graph(segments)
    >>> print(nodes_gdf)
    >>> print(edges_gdf)
    node_id  geometry
    0        POINT (0 0)
    1        POINT (1 1)
    2        POINT (1 0)
                                    road_name   geometry
    from_node_id to_node_id
    0            1                  A           LINESTRING (0 0, 1 1)
    1            2                  B           LINESTRING (1 1, 1 0)

    >>> # Example with duplicate connections (multigraph)
    >>> segments_with_duplicates = gpd.GeoDataFrame(
    ...     {"road_name": ["A", "B", "C"]},
    ...     geometry=[LineString([(0, 0), (1, 1)]),
    ...               LineString([(0, 0), (1, 1)]),
    ...               LineString([(1, 1), (1, 0)])],
    ...     crs="EPSG:32633"
    ... )
    >>> nodes_gdf, edges_gdf = segments_to_graph(segments_with_duplicates, multigraph=True)
    >>> print(edges_gdf.index.names)
    ['from_node_id', 'to_node_id', 'edge_key']
    """
    processor = GeoDataProcessor()

    # Validate input
    segments_clean = processor.validate_gdf(segments_gdf, ["LineString"])

    if segments_clean is None or segments_clean.empty:
        empty_nodes = gpd.GeoDataFrame(columns=["geometry"], crs=segments_gdf.crs)
        empty_edges = gpd.GeoDataFrame(columns=["geometry"], crs=segments_gdf.crs)
        return empty_nodes, empty_edges

    # Extract coordinates
    start_coords = processor.extract_coordinates(segments_clean, start=True)
    end_coords = processor.extract_coordinates(segments_clean, start=False)

    # Create unique nodes
    all_coords = pd.concat([start_coords, end_coords]).drop_duplicates()
    coord_to_id = {coord: i for i, coord in enumerate(all_coords)}

    # Create nodes GeoDataFrame efficiently using gpd.points_from_xy
    coords_array = all_coords.to_numpy()
    x_coords = [coord[0] for coord in coords_array]
    y_coords = [coord[1] for coord in coords_array]

    # Create nodes GeoDataFrame with unique node IDs
    nodes_gdf = gpd.GeoDataFrame(
        {
            "node_id": range(len(all_coords)),
            "geometry": gpd.points_from_xy(x_coords, y_coords),
        },
        crs=segments_clean.crs,
    ).set_index("node_id", drop=True)

    # Create edges with MultiIndex
    from_ids = start_coords.map(coord_to_id)
    to_ids = end_coords.map(coord_to_id)

    edges_gdf = segments_clean.copy()

    if multigraph:
        # For multigraph, handle potential duplicate node pairs by adding edge keys
        edge_pairs_df = pd.DataFrame({"from_id": from_ids, "to_id": to_ids})
        edge_keys = edge_pairs_df.groupby(["from_id", "to_id"]).cumcount()

        edges_gdf.index = pd.MultiIndex.from_arrays(
            [from_ids, to_ids, edge_keys],
            names=["from_node_id", "to_node_id", "edge_key"],
        )
    else:
        edges_gdf.index = pd.MultiIndex.from_arrays(
            [from_ids, to_ids],
            names=["from_node_id", "to_node_id"],
        )

    return nodes_gdf, edges_gdf


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
        The returned type depends on the graph type and input parameters:
        - Homogeneous graph:
            - `(nodes_gdf, edges_gdf)` if `nodes` and `edges` are True.
            - `nodes_gdf` if only `nodes` is True.
            - `edges_gdf` if only `edges` is True.
        - Heterogeneous graph:
            - `(nodes_dict, edges_dict)` where dicts map types to GeoDataFrames.

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
    center_point: Point | gpd.GeoSeries,
    distance: float,
    edge_attr: str = "length",
    node_id_col: str | None = None,
) -> gpd.GeoDataFrame | nx.Graph | nx.MultiGraph:
    """
    Filter a graph to include only elements within a specified distance from a center point.

    This function calculates the shortest path from a center point to all nodes
    in the graph and returns a subgraph containing only the nodes (and their
    induced edges) that are within the given distance. The input can be a
    NetworkX graph or an edges GeoDataFrame.

    Parameters
    ----------
    graph : geopandas.GeoDataFrame or networkx.Graph or networkx.MultiGraph
        The graph to filter. If a GeoDataFrame, it represents the edges of the
        graph and will be converted to a NetworkX graph internally.
    center_point : Point or geopandas.GeoSeries
        The origin point(s) for the distance calculation. If multiple points
        are provided, the filter will include nodes reachable from any of them.
    distance : float
        The maximum shortest-path distance for a node to be included in the
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
    >>> filtered_graph = filter_graph_by_distance(G, center, distance=12)
    >>> print(list(filtered_graph.nodes))
    >>> [0, 1]
    """
    analyzer = GraphAnalyzer()
    return analyzer.filter_graph_by_distance(graph, center_point, distance, edge_attr, node_id_col)


def create_isochrone(
    graph: gpd.GeoDataFrame | nx.Graph | nx.MultiGraph,
    center_point: Point | gpd.GeoSeries | gpd.GeoDataFrame,
    distance: float,
    edge_attr: str = "length",
) -> gpd.GeoDataFrame:
    """
    Generate an isochrone polygon from a graph.

    An isochrone represents the area reachable from a center point within a
    given travel distance or time. This function computes the set of reachable
    edges and nodes in a network and generates a polygon (the convex hull)
    that encloses this reachable area.

    Parameters
    ----------
    graph : geopandas.GeoDataFrame or networkx.Graph or networkx.MultiGraph
        The network graph. If a GeoDataFrame, it represents the edges of the
        graph.
    center_point : Point or geopandas.GeoSeries or geopandas.GeoDataFrame
        The origin point(s) for the isochrone calculation.
    distance : float
        The maximum travel distance (or time) that defines the boundary of the
        isochrone.
    edge_attr : str, default "length"
        The edge attribute to use as the cost of travel (e.g., 'length',
        'travel_time').

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing a single Polygon geometry that represents the
        isochrone.

    See Also
    --------
    filter_graph_by_distance : Filter a graph by distance from a center point.

    Examples
    --------
    >>> import networkx as nx
    >>> from shapely.geometry import Point
    >>> # Create a graph
    >>> G = nx.Graph(crs="EPSG:32633")
    >>> G.add_node(0, pos=(0, 0))
    >>> G.add_node(1, pos=(10, 0))
    >>> G.add_node(2, pos=(0, 10))
    >>> G.add_edge(0, 1, length=10)
    >>> G.add_edge(0, 2, length=10)
    >>> # Create an isochrone
    >>> center = Point(0, 0)
    >>> isochrone = create_isochrone(G, center, distance=12)
    >>> print(isochrone.geometry.iloc[0].wkt)
    POLYGON ((0 0, 10 0, 0 10, 0 0))
    """
    analyzer = GraphAnalyzer()
    return analyzer.create_isochrone(graph, center_point, distance, edge_attr)


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
        return gpd.GeoDataFrame(geometry=[], crs=geometry.crs)

    if primary_barriers is not None:
        # Enclosed tessellation
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
                    return gpd.GeoDataFrame(
                        columns=["geometry", "enclosure_index", "tess_id"],
                        geometry="geometry",
                        crs=geometry.crs,
                    )
        else:
            tessellation = gpd.GeoDataFrame(
                columns=["geometry", "enclosure_index"],
                geometry="geometry",
                crs=geometry.crs,
            )

        if tessellation.empty:
            return gpd.GeoDataFrame(
                columns=["geometry", "enclosure_index"],
                geometry="geometry",
                crs=geometry.crs,
            )

        tessellation["tess_id"] = [
            f"{i}_{j}"
            for i, j in zip(tessellation["enclosure_index"], tessellation.index, strict=False)
        ]
        tessellation = tessellation.reset_index(drop=True)
    else:
        # Morphological tessellation
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
    tuple[geopandas.GeoDataFrame | dict[str, geopandas.GeoDataFrame] | None,
          geopandas.GeoDataFrame | dict[tuple[str, str, str], geopandas.GeoDataFrame] | None,
          bool]
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
    undirected graphs, as well as multigraphs. The original NetworkX node IDs
    are stored in the node payload under the key ``__nx_node_id__`` to ensure
    reversibility.

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
    >>> import networkx as nx
    >>> G = nx.Graph()
    >>> G.add_node("a", color="red")
    >>> G.add_edge("a", "b", weight=2)
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
    restoring node, edge, and graph attributes. It attempts to restore original
    NetworkX node IDs if they were preserved during a previous conversion using
    ``nx_to_rx`` (via the ``__nx_node_id__`` key).

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
}


def plot_graph(  # noqa: PLR0913
    graph: nx.Graph | nx.MultiGraph | None = None,
    nodes: gpd.GeoDataFrame | dict[str, gpd.GeoDataFrame] | None = None,
    edges: gpd.GeoDataFrame | dict[tuple[str, str, str], gpd.GeoDataFrame] | None = None,
    ax: Any | None = None,  # noqa: ANN401
    bgcolor: str = "#000000",
    figsize: tuple[float, float] = (12, 12),
    subplots: bool = True,
    ncols: int | None = None,
    legend_position: str | None = "upper left",
    labelcolor: str = "white",
    node_color: str | float | pd.Series | dict[str, Any] | None = None,
    node_alpha: float | pd.Series | dict[str, Any] | None = None,
    node_zorder: int | pd.Series | dict[str, Any] | None = None,
    node_edgecolor: str | pd.Series | dict[str, Any] | None = None,
    markersize: float | pd.Series | dict[str, Any] | None = None,
    edge_color: str | float | pd.Series | dict[tuple[str, str, str], Any] | None = None,
    edge_linewidth: float | pd.Series | dict[tuple[str, str, str], Any] | None = None,
    edge_alpha: float | pd.Series | dict[tuple[str, str, str], Any] | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> None:
    """
    Plot a graph with beautiful defaults.

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
    ax : matplotlib.axes.Axes, optional
        The axes on which to plot. If None, a new figure and axes are created.
    bgcolor : str, default "#000000"
        Background color for the plot (Black theme).
    figsize : tuple[float, float], default (12, 12)
        Figure size as (width, height) in inches.
    subplots : bool, default True
        If True and the graph is heterogeneous, plot each node/edge type in a
        separate subplot. Ignores 'ax' if True.
    ncols : int, optional
        Number of columns (subplots per row) when plotting heterogeneous graphs
        with subplots=True. If None, defaults to min(3, number_of_edge_types).
    legend_position : str or None, default "upper left"
        Position of the legend for heterogeneous graphs. Common values include
        "upper left", "upper right", "lower left", "lower right", "center", etc.
        If None, no legend is displayed.
    labelcolor : str, default "white"
        Color of the legend text labels.
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
    **kwargs : Any
        Additional keyword arguments passed to the GeoPandas plotting functions.

        Supports attribute-based styling where parameters can be specified as:

        - **Scalar values** (str/float): Applied uniformly to all geometries
        - **Column names** (str): If the string matches a column in the GeoDataFrame,
          that column's values are used for styling
        - **pd.Series**: Direct values for each geometry

        Other common options: etc.

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
        "legend_position": legend_position,
        "labelcolor": labelcolor,
        **kwargs,
    }

    # Handle heterogeneous subplots
    is_hetero = isinstance(nodes, dict) or isinstance(edges, dict)

    if subplots and is_hetero:
        _plot_hetero_subplots(nodes, edges, figsize, bgcolor, ncols=ncols, **style_kwargs)
        return

    # Setup figure and axes
    if ax is None:
        ax = _setup_plot_axes(figsize, bgcolor)

    # GeoDataFrame-based plotting
    _plot_geodataframes(nodes, edges, ax, **style_kwargs)


def _setup_plot_axes(
    figsize: tuple[float, float],
    bgcolor: str,
) -> Any:  # noqa: ANN401
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

    if is_edge:
        return {
            "color": _or_default(
                _get_param("edge_color"), default_color or PLOT_DEFAULTS["edge_color"]
            ),
            "linewidth": _or_default(_get_param("edge_linewidth"), PLOT_DEFAULTS["edge_linewidth"]),
            "alpha": _or_default(_get_param("edge_alpha"), PLOT_DEFAULTS["edge_alpha"]),
            "zorder": PLOT_DEFAULTS["edge_zorder"],
        }

    return {
        "color": _or_default(
            _get_param("node_color"), default_color or PLOT_DEFAULTS["node_color"]
        ),
        "alpha": _or_default(_get_param("node_alpha"), PLOT_DEFAULTS["node_alpha"]),
        "zorder": _or_default(_get_param("node_zorder"), PLOT_DEFAULTS["node_zorder"]),
        "edgecolor": _or_default(_get_param("node_edgecolor"), PLOT_DEFAULTS["node_edgecolor"]),
        "markersize": _or_default(_get_param("markersize"), PLOT_DEFAULTS["markersize"]),
    }


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

    plot_kwargs: dict[str, Any] = {"ax": ax}
    param_defaults = {
        "color": PLOT_DEFAULTS["node_color"],
        "alpha": PLOT_DEFAULTS["node_alpha"],
        "linewidth": PLOT_DEFAULTS["edge_linewidth"],
        "markersize": PLOT_DEFAULTS["markersize"],
        "zorder": PLOT_DEFAULTS["node_zorder"],
        "edgecolor": PLOT_DEFAULTS["edge_color"],
        "label": None,
    }

    for param_name, default_val in param_defaults.items():
        val = _resolve_plot_parameter(gdf, kwargs.get(param_name), param_name, default_val)
        if val is not None:
            if param_name == "color" and isinstance(val, pd.Series):
                plot_kwargs["column"] = val
            else:
                plot_kwargs[param_name] = val

    gdf.plot(**plot_kwargs)


def _plot_geodataframes(
    nodes: gpd.GeoDataFrame | dict[str, gpd.GeoDataFrame] | None,
    edges: gpd.GeoDataFrame | dict[tuple[str, str, str], gpd.GeoDataFrame] | None,
    ax: Any,  # noqa: ANN401
    **kwargs: Any,  # noqa: ANN401
) -> None:
    """
    Render nodes and edges on axes.

    Dispatches to homogeneous or heterogeneous renderer based on input types.

    Parameters
    ----------
    nodes : geopandas.GeoDataFrame or dict, optional
        Node geometries to draw.
    edges : geopandas.GeoDataFrame or dict, optional
        Edge geometries to draw.
    ax : matplotlib.axes.Axes
        Axes instance for rendering.
    **kwargs : Any
        Additional keyword arguments for plotting.
    """
    is_hetero = isinstance(nodes, dict) or isinstance(edges, dict)
    if is_hetero:
        _plot_hetero_graph(nodes, edges, ax, **kwargs)
    else:
        _plot_homo_graph(nodes, edges, ax, **kwargs)


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
    ncols: int | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> None:
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
    ncols : int, optional
        Number of columns in the subplot grid. If None, defaults to
        min(3, number_of_edge_types).
    **kwargs : Any
        Additional styling arguments.
    """
    # Collect non-empty edge types to plot
    edge_items = [(k, v) for k, v in (edges or {}).items() if not v.empty]
    n_items = len(edge_items)
    if n_items == 0:
        return

    # Calculate grid layout
    cols = ncols if ncols is not None else min(3, n_items)
    cols = max(1, min(cols, n_items))  # Ensure cols is between 1 and n_items
    rows = math.ceil(n_items / cols)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.patch.set_facecolor(bgcolor)

    # Ensure axes is iterable
    axes_flat = [axes] if n_items == 1 else axes.flatten()

    # Calculate total bounds for fixed extent
    xlim, ylim = _calculate_total_bounds(nodes, edges)

    for i, (edge_key, edge_gdf) in enumerate(edge_items):
        ax = axes_flat[i]
        ax.set_facecolor(bgcolor)
        ax.set_axis_off()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # Get colors for this subplot
        # We do NOT want to force tab10 colors here. We want to use PLOT_DEFAULTS
        # unless the user has specified something else in kwargs (which is handled by _resolve_style_kwargs).
        # So we pass None for the default colors.
        colors = {
            "edge": None,
            "src": None,
            "dst": None,
        }

        _plot_hetero_subplot_item(ax, edge_key, edge_gdf, nodes, colors, **kwargs)

    # Hide unused axes
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)


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
    ax.set_title(f"{edge_key}", color="white", fontsize=10)


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
