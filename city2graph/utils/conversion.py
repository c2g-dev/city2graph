"""Graph conversion, metadata, and validation utilities."""

# Standard library imports
import logging
import typing
from collections.abc import Sequence
from typing import Any
from typing import Literal

# Third-party imports
import geopandas as gpd
import networkx as nx
import pandas as pd
import rustworkx as rx
from shapely.geometry import LineString
from shapely.geometry import Point

from city2graph.base import BaseGraphConverter
from city2graph.base import GeoDataProcessor
from city2graph.base import GraphMetadata

__all__ = [
    "gdf_to_nx",
    "nx_to_gdf",
    "nx_to_rx",
    "rx_to_nx",
    "validate_gdf",
    "validate_nx",
]

logger = logging.getLogger("city2graph.utils")


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
                "list[str | None] | None", self._get_node_index_names(nodes)
            )

        # Add edges
        self._add_homogeneous_edges(graph, edges, nodes)
        metadata.edge_geom_cols = list(edges.select_dtypes(include=["geometry"]).columns)
        metadata.edge_index_names = _coerce_name_sequence(edges.index.names)

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
        additional_node_cols: list[str] | None = None,  # noqa: ARG002
        additional_edge_cols: list[str] | None = None,  # noqa: ARG002
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
        additional_node_cols : list[str] or None, optional
            Additional columns to extract. Not used in this implementation.
        additional_edge_cols : list[str] or None, optional
            Additional columns to extract. Not used in this implementation.

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
        additional_node_cols: dict[str, list[str]] | None = None,  # noqa: ARG002
        additional_edge_cols: dict[str, list[str]] | None = None,  # noqa: ARG002
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
        additional_node_cols : dict[str, list[str]] or None, optional
            Additional columns to extract. Not used in this implementation.
        additional_edge_cols : dict[str, list[str]] or None, optional
            Additional columns to extract. Not used in this implementation.

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
