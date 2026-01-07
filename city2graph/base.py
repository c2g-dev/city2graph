"""
Base Module for Graph Conversion.

This module provides foundational classes for graph conversion operations,
including metadata management, data validation, and base converter interfaces.
These classes serve as the core building blocks for the city2graph package's
graph conversion functionality across different formats (NetworkX, PyTorch Geometric).
"""

# Standard library imports
import logging
from abc import ABC
from abc import abstractmethod
from typing import Any

# Third-party imports
import geopandas as gpd
import networkx as nx
import pandas as pd

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
        self.node_index_names: dict[str, list[str] | None] | list[str] | None = None
        self.edge_index_names: dict[tuple[str, str, str], list[str] | None] | list[str] | None = (
            None
        )

        # Geometry column tracking
        self.node_geom_cols: list[str] = []
        self.edge_geom_cols: list[str] = []

        # PyTorch Geometric specific metadata
        self.node_mappings: dict[str, dict[str, dict[str | int, int] | str | list[str | int]]] = {}
        self.node_feature_cols: dict[str, list[str]] | list[str] | None = None
        self.node_label_cols: dict[str, list[str]] | list[str] | None = None
        self.edge_feature_cols: dict[tuple[str, str, str], list[str]] | list[str] | None = None
        self.edge_index_values: (
            dict[tuple[str, str, str], list[list[str | int]]] | list[list[str | int]] | None
        ) = None

        # Geometry storage for exact reconstruction (WKB hexadecimal format)
        self.node_geometries: dict[str, list[str]] | list[str] | None = None
        self.edge_geometries: dict[tuple[str, str, str], list[str]] | list[str] | None = None

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
        >>> G.add_edge(0, 1)
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
        ...     GeoDataProcessor.ensure_crs_consistency(gdf1, gdf2)
        ...     print("CRS is consistent.")
        ... except ValueError as e:
        ...     print(e)
        CRS is consistent.
        """
        non_empty_gdfs = [gdf for gdf in gdfs if gdf is not None and not gdf.empty]
        if not non_empty_gdfs:
            return gdfs

        reference_crs = non_empty_gdfs[0].crs
        for gdf in non_empty_gdfs[1:]:
            if gdf.crs != reference_crs:
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
        dtype: object
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
        larger geometry in graph-based analyses.

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
        0    POINT (0.50000 0.50000)
        dtype: geometry
        """
        return gdf.geometry.centroid
