"""
Base Module for Graph Conversion.

This module provides foundational classes for graph conversion operations,
including metadata management, data validation, and base converter interfaces.
These classes serve as the core building blocks for the city2graph package's
graph conversion functionality across different formats (NetworkX, PyTorch Geometric).
"""

# Standard library imports
import logging
import typing
import warnings
from abc import ABC
from abc import abstractmethod
from collections.abc import Iterator
from collections.abc import Sequence
from typing import Any

# Third-party imports
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree
from shapely.geometry import Point

# Module logger configuration
logger = logging.getLogger(__name__)


class GraphMetadata:
    """
    Centralized graph metadata management.

    This class provides a centralized way to manage metadata for graph objects,
    including coordinate reference systems and heterogeneous graph information.

    Parameters
    ----------
    crs : str, int, dict, or None, optional
        Coordinate reference system specification.
    is_hetero : bool, default False
        Whether the graph is heterogeneous.

    See Also
    --------
    gdf_to_nx : Convert GeoDataFrame to NetworkX graph.
    nx_to_gdf : Convert NetworkX graph to GeoDataFrame.

    Examples
    --------
    >>> metadata = GraphMetadata(crs='EPSG:4326', is_hetero=False)
    >>> metadata.crs
    'EPSG:4326'
    """

    def __init__(
        self,
        crs: str | int | dict[str, object] | object | None = None,
        is_hetero: bool = False,
    ) -> None:
        """
        Initialize GraphMetadata with coordinate reference system and graph type.

        This constructor creates a new GraphMetadata instance to store essential
        information about graph structure and spatial properties for conversion
        between different graph representations.

        Parameters
        ----------
        crs : str, int, dict, object, or None, optional
            Coordinate reference system specification.
        is_hetero : bool, default False
            Whether the graph is heterogeneous.

        See Also
        --------
        to_dict : Convert metadata to dictionary.
        from_dict : Create metadata from dictionary.

        Examples
        --------
        >>> metadata = GraphMetadata(crs='EPSG:4326', is_hetero=False)
        >>> metadata.crs
        'EPSG:4326'
        """
        # Core metadata
        self.crs = crs
        self.is_hetero = is_hetero

        # Graph structure metadata
        self.node_types: list[str] = []
        self.edge_types: list[tuple[str, str, str]] = []

        # Index management
        self.node_index_names: dict[str, list[str | None] | None] | list[str | None] | None = None
        self.edge_index_names: (
            dict[tuple[str, str, str], list[str | None] | None] | list[str | None] | None
        ) = None

        # Geometry column tracking
        self.node_geom_cols: list[str] = []
        self.edge_geom_cols: list[str] = []

        # PyTorch Geometric specific metadata
        self.node_mappings: dict[str, dict[str, dict[str | int, int] | str | list[str | int]]] = {}
        self.node_feature_cols: dict[str, list[str]] | list[str] | None = None
        self.node_label_cols: dict[str, list[str]] | list[str] | None = None
        self.edge_feature_cols: dict[tuple[str, str, str], list[str]] | list[str] | None = None
        self.edge_index_values: (
            dict[tuple[str, str, str], list[np.ndarray[tuple[int, ...], np.dtype[Any]]]]
            | list[np.ndarray[tuple[int, ...], np.dtype[Any]]]
            | None
        ) = None
        self.edge_index_keys: (
            dict[tuple[str, str, str], np.ndarray[tuple[int, ...], np.dtype[Any]]]
            | np.ndarray[tuple[int, ...], np.dtype[Any]]
            | None
        ) = None

        # Geometry storage for exact reconstruction (WKB hexadecimal format)
        self.node_geometries: dict[str, list[str]] | list[str] | None = None
        self.edge_geometries: dict[tuple[str, str, str], list[str]] | list[str] | None = None

        # Directed flag: tracks whether edges are directed in the PyG representation.
        # When False (default), gdf_to_pyg symmetrizes edges and pyg_to_gdf deduplicates.
        self.is_directed: bool | dict[tuple[str, str, str], bool] = False

        # Per-edge-type symmetrization tracking.
        # Records what city2graph actually symmetrized during conversion.
        # Reconstruction uses this (not just is_directed) to decide deduplication,
        # because a hetero graph can mix directed and undirected edge types.
        self.edge_was_symmetrized: bool | dict[tuple[str, str, str], bool] = False

        # Multigraph flag: true when edge identity includes an explicit key level
        # or when conversion generated keys for opt-in multigraph handling.
        self.is_multigraph: bool | dict[tuple[str, str, str], bool] = False

        # Cross-type reverse edge type mappings.
        # Maps original cross-type edge types to generated reverse edge types.
        self.reverse_edge_types: dict[tuple[str, str, str], tuple[str, str, str]] = {}
        # Maps generated reverse edge types back to their original edge types.
        self.generated_reverse_edge_types: dict[tuple[str, str, str], tuple[str, str, str]] = {}

        # Edge types supplied by the user before city2graph added any generated
        # reverse stores. Generated reverse edge stores are implementation
        # artefacts for PyG message passing and should be skipped by default
        # during pyg_to_gdf reconstruction.
        self.original_edge_types: list[tuple[str, str, str]] = []

    def to_dict(self) -> dict[str, object]:
        """
        Convert to dictionary for NetworkX graph metadata.

        This method serializes the GraphMetadata instance into a dictionary
        format suitable for storage as NetworkX graph attributes.

        Returns
        -------
        dict[str, object]
            Dictionary containing metadata for NetworkX graph storage.

        See Also
        --------
        from_dict : Create GraphMetadata from dictionary.

        Examples
        --------
        >>> metadata = GraphMetadata(crs='EPSG:4326')
        >>> metadata.to_dict()
        {'crs': 'EPSG:4326', 'is_hetero': False, ...}
        """
        return self.__dict__

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "GraphMetadata":
        """
        Create from dictionary.

        This class method creates a GraphMetadata instance from a dictionary,
        typically used when reconstructing metadata from NetworkX graph attributes.

        Parameters
        ----------
        data : dict[str, object]
            Dictionary containing metadata information.

        Returns
        -------
        GraphMetadata
            New GraphMetadata instance created from the dictionary.

        See Also
        --------
        to_dict : Convert GraphMetadata to dictionary.

        Examples
        --------
        >>> data = {'crs': 'EPSG:4326', 'is_hetero': False}
        >>> metadata = GraphMetadata.from_dict(data)
        >>> metadata.crs
        'EPSG:4326'
        """
        crs = data.get("crs")
        is_hetero_obj = data.get("is_hetero", False)

        # Type check the parameters
        if crs is not None and not isinstance(crs, (str, int, dict)) and not hasattr(crs, "to_wkt"):
            msg = "CRS must be str, int, dict, a CRS-like object, or None"
            raise TypeError(msg)
        if not isinstance(is_hetero_obj, bool):
            msg = "is_hetero must be bool"
            raise TypeError(msg)
        is_hetero: bool = is_hetero_obj

        metadata = cls(crs, is_hetero)
        for key, value in data.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)
        return metadata


class BaseGraphConverter(ABC):
    """
    Abstract base class for graph conversion.

    This class defines the interface and common logic for converting between
    GeoDataFrames and various graph formats (NetworkX, PyTorch Geometric).

    Parameters
    ----------
    keep_geom : bool, default True
        Whether to preserve geometry information during conversion.
    multigraph : bool, default False
        Whether to create a multigraph (allows multiple edges between nodes).
    directed : bool, default False
        Whether to create a directed graph.
    """

    def __init__(
        self,
        keep_geom: bool = True,
        multigraph: bool = False,
        directed: bool = False,
    ) -> None:
        """
        Initialize BaseGraphConverter with conversion options.

        This constructor creates a new BaseGraphConverter instance with options
        for controlling how geometries are handled and what graph type is created.

        Parameters
        ----------
        keep_geom : bool, default True
            Whether to preserve geometry information during conversion.
        multigraph : bool, default False
            Whether to create a multigraph (allows multiple edges between nodes).
        directed : bool, default False
            Whether to create a directed graph.
        """
        self.keep_geom = keep_geom
        self.multigraph = multigraph
        self.directed = directed
        self.processor = GeoDataProcessor()

    def convert(
        self,
        nodes: gpd.GeoDataFrame | dict[str, gpd.GeoDataFrame] | None = None,
        edges: gpd.GeoDataFrame | dict[tuple[str, str, str], gpd.GeoDataFrame] | None = None,
    ) -> Any:  # noqa: ANN401
        """
        Convert GeoDataFrames to graph object.

        Dispatches to homogeneous or heterogeneous conversion based on input.

        Parameters
        ----------
        nodes : gpd.GeoDataFrame or dict[str, gpd.GeoDataFrame] or None, optional
            Node data as a single GeoDataFrame (homogeneous) or dictionary of
            GeoDataFrames keyed by node type (heterogeneous).
        edges : gpd.GeoDataFrame or dict[tuple[str, str, str], gpd.GeoDataFrame] or None, optional
            Edge data as a single GeoDataFrame (homogeneous) or dictionary of
            GeoDataFrames keyed by edge type tuple (heterogeneous).

        Returns
        -------
        Any
            Graph object in the target format (NetworkX, PyTorch Geometric, etc.).
        """
        if nodes is None and edges is None:
            msg = "Either nodes or edges must be provided."
            raise ValueError(msg)

        # Type validation
        is_nodes_dict = isinstance(nodes, dict)
        is_nodes_gdf = isinstance(nodes, gpd.GeoDataFrame)
        is_edges_dict = isinstance(edges, dict)
        is_edges_gdf = isinstance(edges, gpd.GeoDataFrame)

        # Validate nodes type
        if nodes is not None and not is_nodes_dict and not is_nodes_gdf:
            msg = "Nodes must be a GeoDataFrame or a dictionary of GeoDataFrames"
            raise TypeError(msg)

        # Validate edges type based on nodes type
        if edges is not None:
            if is_nodes_gdf and not is_edges_gdf:
                msg = "For homogeneous graphs, edges must be a GeoDataFrame or None"
                raise TypeError(msg)
            if is_nodes_dict and not is_edges_dict:
                msg = "For heterogeneous graphs, edges must be a dictionary or None"
                raise TypeError(msg)

        is_hetero = is_nodes_dict or is_edges_dict

        if is_hetero:
            return self._convert_heterogeneous(nodes, edges)
        return self._convert_homogeneous(nodes, edges)

    def reconstruct(
        self,
        graph_data: Any,  # noqa: ANN401
        nodes: bool = True,
        edges: bool = True,
        additional_node_cols: Any = None,  # noqa: ANN401
        additional_edge_cols: Any = None,  # noqa: ANN401
    ) -> (
        tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
        | tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]]
    ):
        """
        Reconstruct GeoDataFrames from graph object.

        Dispatches to homogeneous or heterogeneous reconstruction based on metadata.

        Parameters
        ----------
        graph_data : Any
            Graph object to reconstruct from (NetworkX, PyTorch Geometric, etc.).
        nodes : bool, default True
            Whether to reconstruct node GeoDataFrames.
        edges : bool, default True
            Whether to reconstruct edge GeoDataFrames.
        additional_node_cols : Any, optional
            Additional node attributes to extract.
        additional_edge_cols : Any, optional
            Additional edge attributes to extract.

        Returns
        -------
        tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] or tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]]
            Reconstructed node and edge GeoDataFrames. For homogeneous graphs,
            returns a tuple of two GeoDataFrames. For heterogeneous graphs,
            returns a tuple of dictionaries mapping types to GeoDataFrames.
        """
        metadata = self._extract_metadata(graph_data)

        if metadata.is_hetero:
            return self._reconstruct_heterogeneous(
                graph_data,
                metadata,
                nodes,
                edges,
                additional_node_cols=additional_node_cols,
                additional_edge_cols=additional_edge_cols,
            )
        return self._reconstruct_homogeneous(
            graph_data,
            metadata,
            nodes,
            edges,
            additional_node_cols=additional_node_cols,
            additional_edge_cols=additional_edge_cols,
        )

    @abstractmethod
    def _convert_homogeneous(
        self,
        nodes: gpd.GeoDataFrame | None,
        edges: gpd.GeoDataFrame | None,
    ) -> Any:  # noqa: ANN401
        """
        Convert homogeneous GeoDataFrames to graph.

        This abstract method must be implemented by subclasses to convert
        homogeneous node and edge GeoDataFrames into the target graph format.

        Parameters
        ----------
        nodes : gpd.GeoDataFrame or None
            Node GeoDataFrame with geometry and attributes.
        edges : gpd.GeoDataFrame or None
            Edge GeoDataFrame with geometry and attributes.
        """

    @abstractmethod
    def _convert_heterogeneous(
        self,
        nodes: dict[str, gpd.GeoDataFrame] | None,
        edges: dict[tuple[str, str, str], gpd.GeoDataFrame] | None,
    ) -> Any:  # noqa: ANN401
        """
        Convert heterogeneous GeoDataFrames to graph.

        This abstract method must be implemented by subclasses to convert
        heterogeneous node and edge GeoDataFrames into the target graph format.

        Parameters
        ----------
        nodes : dict[str, gpd.GeoDataFrame] or None
            Dictionary mapping node types to GeoDataFrames.
        edges : dict[tuple[str, str, str], gpd.GeoDataFrame] or None
            Dictionary mapping edge types (src_type, edge_type, dst_type) to GeoDataFrames.
        """

    @abstractmethod
    def _extract_metadata(self, graph_data: Any) -> GraphMetadata:  # noqa: ANN401
        """
        Extract metadata from graph object.

        This abstract method must be implemented by subclasses to extract
        metadata from a graph object for reconstruction purposes.

        Parameters
        ----------
        graph_data : Any
            Graph object to extract metadata from.
        """

    @abstractmethod
    def _reconstruct_homogeneous(
        self,
        graph_data: Any,  # noqa: ANN401
        metadata: GraphMetadata,
        nodes: bool,
        edges: bool,
        additional_node_cols: Any | None = None,  # noqa: ANN401
        additional_edge_cols: Any | None = None,  # noqa: ANN401
    ) -> tuple[gpd.GeoDataFrame | None, gpd.GeoDataFrame | None] | gpd.GeoDataFrame:
        """
        Reconstruct homogeneous GeoDataFrames.

        This abstract method must be implemented by subclasses to reconstruct
        homogeneous GeoDataFrames from a graph object.

        Parameters
        ----------
        graph_data : Any
            Graph object to reconstruct from.
        metadata : GraphMetadata
            Metadata extracted from the graph object.
        nodes : bool
            Whether to reconstruct node GeoDataFrame.
        edges : bool
            Whether to reconstruct edge GeoDataFrame.
        additional_node_cols : Any or None, optional
            Additional node attributes to extract.
        additional_edge_cols : Any or None, optional
            Additional edge attributes to extract.
        """

    @abstractmethod
    def _reconstruct_heterogeneous(
        self,
        graph_data: Any,  # noqa: ANN401
        metadata: GraphMetadata,
        nodes: bool,
        edges: bool,
        additional_node_cols: Any | None = None,  # noqa: ANN401
        additional_edge_cols: Any | None = None,  # noqa: ANN401
    ) -> tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]]:
        """
        Reconstruct heterogeneous GeoDataFrames.

        This abstract method must be implemented by subclasses to reconstruct
        heterogeneous GeoDataFrames from a graph object.

        Parameters
        ----------
        graph_data : Any
            Graph object to reconstruct from.
        metadata : GraphMetadata
            Metadata extracted from the graph object.
        nodes : bool
            Whether to reconstruct node GeoDataFrames.
        edges : bool
            Whether to reconstruct edge GeoDataFrames.
        additional_node_cols : Any or None, optional
            Additional node attributes to extract.
        additional_edge_cols : Any or None, optional
            Additional edge attributes to extract.
        """


class GeoDataProcessor:
    """
    Common processor for GeoDataFrame operations.

    This class provides static methods for validating and processing
    GeoDataFrames in preparation for graph conversion operations.

    See Also
    --------
    GraphConverter : Main graph conversion class.

    Examples
    --------
    >>> processor = GeoDataProcessor()
    >>> # Use static methods for validation
    >>> GeoDataProcessor.validate_gdf(gdf)
    """

    @staticmethod
    def validate_gdf(
        gdf: gpd.GeoDataFrame | None,
        expected_geom_types: list[str] | None = None,
        allow_empty: bool = True,
    ) -> gpd.GeoDataFrame | None:
        """
        Unified GeoDataFrame validation.

        This function validates a GeoDataFrame for common issues including
        geometry types, empty geometries, and coordinate reference systems.
        It provides comprehensive validation to ensure data quality before
        processing in spatial analysis workflows.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame or None
            GeoDataFrame to validate.
        expected_geom_types : list[str] or None, optional
            Expected geometry types (e.g., ['Point', 'LineString']).
        allow_empty : bool, default True
            Whether to allow empty GeoDataFrames.

        Returns
        -------
        gpd.GeoDataFrame or None
            Validated GeoDataFrame, or None if input was None.

        See Also
        --------
        validate_nx : Validate NetworkX graphs.
        ensure_crs_consistency : Ensure consistent CRS across GeoDataFrames.

        Examples
        --------
        >>> import geopandas as gpd
        >>> from shapely.geometry import Point
        >>> gdf = gpd.GeoDataFrame({'geometry': [Point(0, 0)]})
        >>> validated = GeoDataProcessor.validate_gdf(gdf, ['Point'])
        >>> validated is not None
        True
        """
        if gdf is None:
            return None

        if not isinstance(gdf, gpd.GeoDataFrame):
            msg = "Input must be a GeoDataFrame"
            raise TypeError(msg)

        if gdf.empty and not allow_empty:
            msg = "GeoDataFrame cannot be empty"
            raise ValueError(msg)

        if gdf.empty:
            return gdf

        # Validate geometry types
        if expected_geom_types:
            valid_mask = gdf.geometry.geom_type.isin(expected_geom_types)
            if not valid_mask.all():
                invalid_count = (~valid_mask).sum()
                logger.warning("Removed %d geometries with invalid types", invalid_count)
                gdf = gdf[valid_mask]

        # Remove invalid geometries
        invalid_mask = gdf.geometry.isna() | ~gdf.geometry.is_valid | gdf.geometry.is_empty
        if invalid_mask.any():
            invalid_count = invalid_mask.sum()
            logger.warning("Removed %d invalid geometries", invalid_count)
            gdf = gdf[~invalid_mask]

        if gdf.empty and not allow_empty:
            msg = "GeoDataFrame cannot be empty"
            raise ValueError(msg)

        return gdf

    @staticmethod
    def validate_nx(graph: nx.Graph | nx.MultiGraph) -> None:
        """
        Validate a NetworkX graph.

        Checks if the input is a NetworkX graph, ensures it is not empty,
        and verifies that it contains the necessary metadata for conversion
        back to GeoDataFrames or PyG objects.

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
        validate_gdf : Validate a GeoDataFrame.

        Examples
        --------
        >>> import networkx as nx
        >>> G = nx.Graph(is_hetero=False, crs="EPSG:4326")
        >>> G.add_node(0, pos=(0, 0))
        >>> G.add_node(1, pos=(0, 1))
        >>> try:
        ...     GeoDataProcessor.validate_nx(G)
        ... except ValueError as e:
        ...     print(e)
        Graph has no edges
        """
        if graph.number_of_nodes() == 0:
            msg = "Graph has no nodes"
            raise ValueError(msg)
        if graph.number_of_edges() == 0:
            msg = "Graph has no edges"
            raise ValueError(msg)

        # Check for essential graph-level metadata
        if not hasattr(graph, "graph") or not isinstance(graph.graph, dict):
            msg = "Graph is missing 'graph' attribute dictionary for metadata."
            raise ValueError(msg)

        # If 'is_hetero' is not set, default to False
        is_hetero = graph.graph.setdefault("is_hetero", False)

        metadata_keys = ["is_hetero", "crs"]
        for key in metadata_keys:
            if key not in graph.graph:
                msg = f"Graph metadata is missing required key: '{key}'"
                raise ValueError(msg)

        # Create 'pos' from 'x' and 'y' if it's missing
        pos_dict = {
            node: (attrs["x"], attrs["y"])
            for node, attrs in graph.nodes(data=True)
            if "pos" not in attrs and "x" in attrs and "y" in attrs
        }
        if pos_dict:
            nx.set_node_attributes(graph, pos_dict, "pos")

        # Check for node-level attributes in a single pass
        if is_hetero:
            if "node_types" not in graph.graph or not graph.graph["node_types"]:
                msg = "Heterogeneous graph metadata is missing 'node_types'."
                raise ValueError(msg)
            if "edge_types" not in graph.graph:
                msg = "Heterogeneous graph metadata is missing 'edge_types'."
                raise ValueError(msg)

        # Validate all node attributes in a single loop
        for _, node_data in graph.nodes(data=True):
            # Check for position/geometry attributes
            if "pos" not in node_data and "geometry" not in node_data:
                msg = "All nodes must have a 'pos' or 'geometry' attribute."
                raise ValueError(msg)

            # Check for node_type in heterogeneous graphs
            if is_hetero and "node_type" not in node_data:
                msg = "All nodes in a heterogeneous graph must have a 'node_type' attribute."
                raise ValueError(msg)

        # Validate edge attributes for heterogeneous graphs
        if is_hetero:
            for _, _, edge_data in graph.edges(data=True):
                if "edge_type" not in edge_data:
                    msg = "All edges in a heterogeneous graph must have an 'edge_type' attribute."
                    raise ValueError(msg)

    @staticmethod
    def ensure_crs_consistency(
        *gdfs: gpd.GeoDataFrame | None,
    ) -> tuple[gpd.GeoDataFrame | None, ...]:
        """
        Ensure all GeoDataFrames have a consistent Coordinate Reference System (CRS).

        This function iterates through a list of GeoDataFrames and verifies that they all
        share the same CRS. It is a crucial validation step before performing any spatial
        operations that require alignment between different geospatial datasets.

        Parameters
        ----------
        *gdfs : geopandas.GeoDataFrame or None
            A variable number of GeoDataFrames to check for CRS consistency.

        Returns
        -------
        tuple[geopandas.GeoDataFrame | None, ...]
            The original tuple of GeoDataFrames if all are consistent.

        Raises
        ------
        ValueError
            If any of the GeoDataFrames have a different CRS.

        See Also
        --------
        validate_gdf : Validate a GeoDataFrame, including its CRS.

        Examples
        --------
        >>> import geopandas as gpd
        >>> from shapely.geometry import Point
        >>> gdf1 = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:4326")
        >>> gdf2 = gpd.GeoDataFrame(geometry=[Point(1, 1)], crs="EPSG:4326")
        >>> try:
        ...     _ = GeoDataProcessor.ensure_crs_consistency(gdf1, gdf2)
        ...     print("CRS is consistent.")
        ... except ValueError as e:
        ...     print(e)
        CRS is consistent.
        """
        non_empty_gdfs = [gdf for gdf in gdfs if gdf is not None and not gdf.empty]
        if not non_empty_gdfs:
            return gdfs

        reference_crs = getattr(non_empty_gdfs[0], "crs", None)
        for gdf in non_empty_gdfs[1:]:
            if getattr(gdf, "crs", None) != reference_crs:
                msg = "All GeoDataFrames must have the same CRS"
                raise ValueError(msg)

        return gdfs

    @staticmethod
    def extract_coordinates(gdf: gpd.GeoDataFrame, start: bool = True) -> pd.Series:
        """
        Extract start or end coordinates from LineString geometries.

        This utility function efficiently extracts the first (start) or last (end) coordinate
        pair from a GeoSeries of LineString objects. It is useful for creating graph
        topologies from road networks or other linear features where the endpoints
        represent nodes.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            The GeoDataFrame containing the LineString geometries.
        start : bool, default True
            If True, extracts the start coordinate of each LineString. If False, extracts
            the end coordinate.

        Returns
        -------
        pandas.Series
            A Series containing the (x, y) coordinate tuples for each geometry.

        See Also
        --------
        segments_to_graph : Convert LineString segments to a graph structure.

        Examples
        --------
        >>> import geopandas as gpd
        >>> from shapely.geometry import LineString
        >>> gdf = gpd.GeoDataFrame(
        ...     geometry=[LineString([(0, 0), (1, 1)]), LineString([(2, 2), (3, 3)])]
        ... )
        >>> start_points = GeoDataProcessor.extract_coordinates(gdf, start=True)
        >>> print(start_points)
        0    (0.0, 0.0)
        1    (2.0, 2.0)
        Name: geometry, dtype: object
        """
        if start:
            coords: pd.Series = gdf.geometry.apply(lambda g: g.coords[0] if g else None)
        else:
            coords = gdf.geometry.apply(lambda g: g.coords[-1] if g else None)
        return coords

    @staticmethod
    def compute_centroids(gdf: gpd.GeoDataFrame) -> gpd.GeoSeries:
        """
        Compute centroids efficiently.

        This function calculates the geometric centroid for each geometry in a
        GeoDataFrame. It provides a simple and direct way to get the central point
        of polygons or lines, which is often used to represent the location of a
        larger geometry in graph-based analyses. The CRS of the input is
        explicitly propagated to the resulting GeoSeries.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            The GeoDataFrame for which to compute centroids.

        Returns
        -------
        geopandas.GeoSeries
            A GeoSeries containing the centroid points for each input geometry.

        See Also
        --------
        create_tessellation : Create tessellations from geometries.

        Examples
        --------
        >>> import geopandas as gpd
        >>> from shapely.geometry import Polygon
        >>> gdf = gpd.GeoDataFrame(
        ...     geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
        ... )
        >>> centroids = GeoDataProcessor.compute_centroids(gdf)
        >>> print(centroids)
        0    POINT (0.5 0.5)
        dtype: geometry
        """
        centroids = gdf.geometry.centroid
        # Explicitly carry CRS (some versions keep it automatically, we enforce)
        return centroids.set_crs(gdf.crs, allow_override=True)

    @staticmethod
    def harmonize_crs(
        source: gpd.GeoDataFrame | gpd.GeoSeries,
        target_crs: Any,  # noqa: ANN401
        *,
        warn: bool = True,
        stacklevel: int = 2,
    ) -> gpd.GeoDataFrame | gpd.GeoSeries:
        """
        Reproject a GeoDataFrame or GeoSeries to a target CRS when needed.

        This function compares the Coordinate Reference System (CRS) of the
        source against the target CRS and reprojects the source when they
        differ. It centralizes the CRS-harmonization behaviour that spatial
        operations rely on before combining datasets, optionally warning the
        caller that a reprojection took place.

        Parameters
        ----------
        source : geopandas.GeoDataFrame or geopandas.GeoSeries
            The geospatial data to check and potentially reproject.
        target_crs : Any
            The target coordinate reference system to align with.
        warn : bool, default True
            Whether to emit a ``RuntimeWarning`` when a reprojection occurs.
        stacklevel : int, default 2
            Stack level passed to :func:`warnings.warn` so the warning points
            at the caller's code.

        Returns
        -------
        geopandas.GeoDataFrame or geopandas.GeoSeries
            The source data, reprojected to the target CRS if necessary.

        Warns
        -----
        RuntimeWarning
            If a CRS mismatch is detected, reprojection is performed, and
            ``warn`` is True.

        See Also
        --------
        ensure_crs_consistency : Validate that GeoDataFrames share one CRS.
        warn_crs_issues : Warn about missing or geographic CRS values.

        Examples
        --------
        >>> import geopandas as gpd
        >>> from shapely.geometry import Point
        >>> gdf = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:4326")
        >>> aligned = GeoDataProcessor.harmonize_crs(gdf, "EPSG:4326")
        >>> aligned.crs.to_epsg()
        4326
        """
        if source.crs != target_crs:
            if warn:
                warnings.warn(
                    "CRS mismatch detected, reprojecting",
                    RuntimeWarning,
                    stacklevel=stacklevel,
                )
            source = source.to_crs(target_crs)
        return source

    @staticmethod
    def warn_crs_issues(
        gdf: gpd.GeoDataFrame,
        *,
        missing_message: str | None = None,
        geographic_message: str | None = None,
        stacklevel: int = 2,
    ) -> None:
        """
        Warn about missing or geographic CRS values on a GeoDataFrame.

        This function inspects the Coordinate Reference System (CRS) of a
        GeoDataFrame and emits a ``UserWarning`` when the CRS is missing or
        geographic. Distance- and length-based computations are unreliable in
        both situations, so callers pass context-specific messages describing
        the consequences for their own outputs.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            GeoDataFrame whose CRS will be inspected.
        missing_message : str or None, optional
            Warning message emitted when the CRS is missing. When None, a
            missing CRS is silently accepted.
        geographic_message : str or None, optional
            Warning message emitted when the CRS is geographic. When None, a
            geographic CRS is silently accepted.
        stacklevel : int, default 2
            Stack level passed to :func:`warnings.warn` so the warning points
            at the caller's code.

        Returns
        -------
        None
            This function only emits warnings.

        Warns
        -----
        UserWarning
            If the CRS is missing and ``missing_message`` is provided, or if
            the CRS is geographic and ``geographic_message`` is provided.

        See Also
        --------
        harmonize_crs : Reproject data to a target CRS when needed.
        ensure_crs_consistency : Validate that GeoDataFrames share one CRS.

        Examples
        --------
        >>> import geopandas as gpd
        >>> from shapely.geometry import Point
        >>> gdf = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:27700")
        >>> GeoDataProcessor.warn_crs_issues(gdf, geographic_message="Geographic CRS")
        """
        crs = gdf.crs
        if crs is None:
            if missing_message:
                warnings.warn(missing_message, UserWarning, stacklevel=stacklevel)
            return

        # pyproj.CRS exposes is_geographic when available
        if geographic_message and getattr(crs, "is_geographic", False):
            warnings.warn(geographic_message, UserWarning, stacklevel=stacklevel)

    @staticmethod
    def create_empty_edges_gdf(
        crs: Any,  # noqa: ANN401
        from_col: str,
        to_col: str,
        extra_cols: list[str] | None = None,
    ) -> gpd.GeoDataFrame:
        """
        Create an empty edges GeoDataFrame with specified column structure.

        This helper function generates an empty GeoDataFrame suitable for representing
        graph edges, ensuring it has the correct column names for 'from' and 'to'
        node IDs, and optionally additional columns, along with a geometry column
        and a specified Coordinate Reference System (CRS). This is useful for
        initializing empty edge GeoDataFrames when no connections are found or
        when setting up a new graph structure.

        Parameters
        ----------
        crs : Any
            Coordinate reference system.
        from_col : str
            Name for the 'from' node ID column.
        to_col : str
            Name for the 'to' node ID column.
        extra_cols : list[str], optional
            Optional list of additional column names.

        Returns
        -------
        geopandas.GeoDataFrame
            Empty GeoDataFrame with specified columns.

        See Also
        --------
        validate_gdf : Validate a GeoDataFrame.

        Examples
        --------
        >>> empty = GeoDataProcessor.create_empty_edges_gdf(
        ...     "EPSG:4326", "from_id", "to_id"
        ... )
        >>> list(empty.columns)
        ['from_id', 'to_id', 'geometry']
        """
        # Initialize list of column names with 'from' and 'to' ID columns
        columns = [from_col, to_col]
        # Add any extra columns if provided
        if extra_cols:
            columns.extend(extra_cols)
        # Add the 'geometry' column name (standard for GeoDataFrames)
        columns.append("geometry")

        # Create an empty GeoDataFrame with the defined columns, geometry column, and CRS
        return gpd.GeoDataFrame(columns=columns, geometry="geometry", crs=crs)

    @staticmethod
    def wrap_layer_ids(ids: list[Any], layer: str) -> list[tuple[str, Any]]:
        """
        Wrap node identifiers as ``(layer, id)`` tuples.

        This function namespaces node identifiers with a layer label so that
        nodes originating from different input layers remain distinguishable
        after they are combined into a single graph. The tuples produced here
        are the values later unwrapped by :meth:`unwrap_layer_edge_index`.

        Parameters
        ----------
        ids : list[Any]
            Node identifiers to wrap.
        layer : str
            Layer label used as the first tuple element (e.g., ``"src"``).

        Returns
        -------
        list[tuple[str, Any]]
            List of ``(layer, id)`` tuples in input order.

        See Also
        --------
        wrap_layer_index : Wrap a pandas Index into a layered MultiIndex.
        unwrap_layer_edge_index : Recover original identifiers from wrapped edges.

        Examples
        --------
        >>> GeoDataProcessor.wrap_layer_ids([1, 2], "src")
        [('src', 1), ('src', 2)]
        """
        return [(layer, i) for i in ids]

    @staticmethod
    def wrap_layer_index(index: pd.Index, layer: str) -> pd.MultiIndex:
        """
        Wrap a pandas Index into a layered MultiIndex.

        This function converts a flat node index into a two-level MultiIndex
        whose first level is a constant layer label. It mirrors
        :meth:`wrap_layer_ids` for node GeoDataFrames so that node and edge
        identifiers stay aligned when layers are concatenated.

        Parameters
        ----------
        index : pandas.Index
            Node index to wrap.
        layer : str
            Layer label used as the first index level (e.g., ``"src"``).

        Returns
        -------
        pandas.MultiIndex
            MultiIndex with levels ``["layer", index.name or "node_id"]``.

        See Also
        --------
        wrap_layer_ids : Wrap identifiers as ``(layer, id)`` tuples.
        unwrap_layer_edge_index : Recover original identifiers from wrapped edges.

        Examples
        --------
        >>> import pandas as pd
        >>> GeoDataProcessor.wrap_layer_index(pd.Index([1, 2]), "src").tolist()
        [('src', 1), ('src', 2)]
        """
        return pd.MultiIndex.from_tuples(
            [(layer, i) for i in index],
            names=["layer", index.name or "node_id"],
        )

    @staticmethod
    def unwrap_layer_edge_index(
        index: pd.MultiIndex,
        names: tuple[str, str] = ("source", "target"),
    ) -> pd.MultiIndex:
        """
        Recover original node identifiers from a layered edge MultiIndex.

        This function strips the ``(layer, id)`` wrappers produced by
        :meth:`wrap_layer_ids` from both levels of an edge MultiIndex,
        returning a MultiIndex of the original node identifiers. It is the
        inverse operation applied when layered edges are handed back to
        callers that expect the original identifier space.

        Parameters
        ----------
        index : pandas.MultiIndex
            Edge MultiIndex whose level values are ``(layer, id)`` tuples.
        names : tuple[str, str], default ("source", "target")
            Names assigned to the resulting MultiIndex levels.

        Returns
        -------
        pandas.MultiIndex
            MultiIndex of original source and target identifiers.

        See Also
        --------
        wrap_layer_ids : Wrap identifiers as ``(layer, id)`` tuples.
        wrap_layer_index : Wrap a pandas Index into a layered MultiIndex.

        Examples
        --------
        >>> import pandas as pd
        >>> idx = pd.MultiIndex.from_tuples([(("src", 1), ("dst", 2))])
        >>> GeoDataProcessor.unwrap_layer_edge_index(idx).tolist()
        [(1, 2)]
        """
        src_ids = [t[1] for t in index.get_level_values(0)]
        dst_ids = [t[1] for t in index.get_level_values(1)]
        return pd.MultiIndex.from_arrays([src_ids, dst_ids], names=list(names))


def row_batches(rows: int, columns: int) -> Iterator[slice]:
    """
    Generate bounded slices in original row order.

    A single row may exceed the byte budget; rows are never truncated.

    Parameters
    ----------
    rows : int
        Total number of rows.
    columns : int
        Number of float64 values per row.

    Yields
    ------
    slice
        Contiguous row slice containing at most 256 rows.
    """
    size = max(1, min(256, (8 * 1024 * 1024) // max(8 * columns, 1)))
    for start in range(0, rows, size):
        yield slice(start, min(start + size, rows))


def frame_records(frame: pd.DataFrame) -> Iterator[dict[Any, Any]]:
    """
    Yield independent attribute dictionaries.

    Bounded record batches preserve pandas scalar boxing, including nullable values,
    without retaining a full list of attributes.

    Parameters
    ----------
    frame : pandas.DataFrame
        Source table; its column order is retained.

    Yields
    ------
    dict[str, Any]
        Attributes for one row.
    """
    for batch in row_batches(len(frame), len(frame.columns)):
        yield from frame.iloc[batch].to_dict("records")


def column_or_level(df: pd.DataFrame | gpd.GeoDataFrame, name: str) -> Any | None:  # noqa: ANN401
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
    if df.index.name == name:
        return df.index.to_numpy()
    return None


def edge_endpoints(edges_df: pd.DataFrame) -> tuple[Any, Any] | None:
    """
    Read named endpoints or the first two levels of an edge index.

    A frame without endpoint identifiers can use geometry-based topology.

    Parameters
    ----------
    edges_df : pandas.DataFrame
        Spatial edge records in their original order.

    Returns
    -------
    tuple or None
        Source and target identifiers, or None for implicit nodes.
    """
    candidates = [
        ("from_node_id", "to_node_id"),
        ("source_id", "target_id"),
        ("u", "v"),
        ("source", "target"),
    ]
    for u_name, v_name in candidates:
        u = column_or_level(edges_df, u_name)
        v = column_or_level(edges_df, v_name)
        if u is not None and v is not None:
            return u, v
    if isinstance(edges_df.index, pd.MultiIndex) and edges_df.index.nlevels >= 2:
        return edges_df.index.get_level_values(0), edges_df.index.get_level_values(1)
    return None


def identify_source_target_cols(
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
    if source_col and target_col:
        u = column_or_level(edges_df, source_col)
        v = column_or_level(edges_df, target_col)
        if u is not None and v is not None:
            return u, v
        missing = []
        if u is None:
            missing.append(source_col)
        if v is None:
            missing.append(target_col)
        msg = f"Source/Target column(s) not found: {', '.join(missing)}"
        raise ValueError(msg)

    endpoints = edge_endpoints(edges_df)
    if endpoints is not None:
        return endpoints

    if len(edges_df.columns) >= 2:
        return edges_df.iloc[:, 0].to_numpy(), edges_df.iloc[:, 1].to_numpy()

    msg = "Could not identify source and target node columns/indices in edges GeoDataFrame."
    raise ValueError(msg)


def adjacency(row: np.ndarray, col: np.ndarray, data: np.ndarray, n: int) -> sparse.csr_matrix:
    """
    Build directed CSR adjacency from weighted arcs.

    Parallel arcs retain their minimum cost and explicit zero-cost arcs remain traversable.

    Parameters
    ----------
    row : ndarray
        Source vertex indices for each arc.
    col : ndarray
        Destination vertex indices for each arc.
    data : ndarray
        Non-negative traversal weights in arc order.
    n : int
        Total number of vertices, including isolated vertices.

    Returns
    -------
    scipy.sparse.csr_matrix
        Weighted adjacency in vertex order.
    """
    order = np.lexsort((data, col, row))
    row, col, data = row[order], col[order], data[order]
    first = np.ones(len(row), dtype=bool)
    first[1:] = (row[1:] != row[:-1]) | (col[1:] != col[:-1])
    return sparse.csr_matrix((data[first], (row[first], col[first])), shape=(n, n))


def distance_batches(
    matrix: sparse.csr_matrix, sources: np.ndarray, limit: float, *, paths: bool = False
) -> Iterator[tuple[slice, np.ndarray, np.ndarray | None]]:
    """
    Search from multiple vertices in bounded batches.

    Path-producing proximity graphs can retain predecessors from the same
    traversal. Cost-only callers allocate just the distance rows.

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix
        Weighted directed adjacency.
    sources : ndarray
        Vertex indices in search order.
    limit : float
        Inclusive distance cutoff.
    paths : bool, optional
        Request predecessor rows alongside distances.

    Yields
    ------
    tuple
        Source slice, distance rows and optional predecessor rows.
    """
    for batch in row_batches(len(sources), matrix.shape[0]):
        result = csgraph.dijkstra(
            matrix, indices=sources[batch], limit=limit, return_predecessors=paths
        )
        if paths:
            yield batch, result[0], result[1]
        else:
            yield batch, result, None


def closest_positions(costs: np.ndarray, k: int) -> np.ndarray:
    """
    Select the k lowest costs with stable ties.

    Partition once to find the boundary cost and sort only the retained rows.

    Parameters
    ----------
    costs : ndarray
        Candidate costs in destination order.
    k : int
        Maximum number of positions to return.

    Returns
    -------
    ndarray
        Destination positions ranked by cost and then input order.
    """
    if k >= len(costs):
        return np.argsort(costs, kind="stable")
    boundary = np.partition(costs, k - 1)[k - 1]
    nearer = np.flatnonzero(costs < boundary)
    tied = np.flatnonzero(costs == boundary)[: k - len(nearer)]
    return np.concatenate((nearer[np.argsort(costs[nearer], kind="stable")], tied))


def normalize_center_points(
    center_point: Point | Sequence[Point] | gpd.GeoSeries | gpd.GeoDataFrame,
) -> list[Point]:
    """
    Normalize center point inputs to a list of Point geometries.

    This ensures downstream snapping logic can always iterate over a
    homogeneous list of Point objects.

    Parameters
    ----------
    center_point : Point or Sequence[Point] or geopandas.GeoSeries or geopandas.GeoDataFrame
        The center point input.

    Returns
    -------
    list[Point]
        The normalized center points.
    """
    if isinstance(center_point, gpd.GeoDataFrame):
        center_points = center_point.geometry.tolist()
    elif isinstance(center_point, gpd.GeoSeries):
        center_points = center_point.tolist()
    elif isinstance(center_point, Sequence) and not isinstance(center_point, str | bytes):
        center_points = list(center_point)
    else:
        center_points = [center_point]

    invalid_points = [point for point in center_points if not isinstance(point, Point)]
    if invalid_points:
        msg = (
            "center_point must be a Point, a sequence of Point objects, GeoSeries, or GeoDataFrame."
        )
        raise TypeError(msg)

    return typing.cast("list[Point]", center_points)


def compute_center_node_distances(
    graph: nx.Graph | nx.MultiGraph,
    center_point: Point | Sequence[Point] | gpd.GeoSeries | gpd.GeoDataFrame,
    edge_attr: str | None,
    cutoff: float,
) -> dict[Any, float]:
    """
    Compute shortest-path distances from snapped center points once.

    Each center point is snapped to its nearest node, and a single multi-source
    Dijkstra run is used to obtain distances to all reachable nodes.

    Parameters
    ----------
    graph : networkx.Graph or networkx.MultiGraph
        The graph to traverse.
    center_point : Point or Sequence[Point] or geopandas.GeoSeries or geopandas.GeoDataFrame
        Center points that will be snapped to the nearest graph nodes.
    edge_attr : str or None
        Edge attribute used as traversal weight.
    cutoff : float
        Maximum distance used to limit traversal.

    Returns
    -------
    dict[Any, float]
        Reachable node distances keyed by node id.
    """
    pos_dict = nx.get_node_attributes(graph, "pos")
    if not pos_dict:
        return {}

    node_ids = list(pos_dict.keys())
    coordinates = list(pos_dict.values())
    tree = cKDTree(coordinates)

    points = normalize_center_points(center_point)
    if not len(points):
        return {}
    _, nearest = tree.query([(p.x, p.y) for p in points])
    deduped_sources = [node_ids[i] for i in np.unique(nearest)]

    distances = nx.multi_source_dijkstra_path_length(
        graph,
        deduped_sources,
        cutoff=cutoff,
        weight=edge_attr or "length",
    )
    return typing.cast("dict[Any, float]", distances)


def build_reachable_subgraph(
    graph: nx.Graph | nx.MultiGraph,
    distances: dict[Any, float],
    threshold: float,
) -> nx.Graph | nx.MultiGraph:
    """
    Induce a reachable subgraph from precomputed node distances.

    Nodes with shortest-path distance less than or equal to the threshold are
    retained, preserving graph type and edge attributes.

    Parameters
    ----------
    graph : networkx.Graph or networkx.MultiGraph
        Source graph.
    distances : dict[Any, float]
        Precomputed shortest-path distances.
    threshold : float
        Threshold used to keep nodes.

    Returns
    -------
    networkx.Graph or networkx.MultiGraph
        Reachable subgraph view.
    """
    reachable_nodes = [node for node, distance in distances.items() if distance <= threshold]
    return graph.subgraph(reachable_nodes)
