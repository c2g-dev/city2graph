"""Spatial analysis, tessellation, and plotting utilities."""

# Standard library imports
import logging
import math
import typing
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING
from typing import Any

# Third-party imports
import geopandas as gpd
import momepy
import networkx as nx
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import GeometryCollection
from shapely.geometry import LineString
from shapely.geometry import MultiPoint
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    MATPLOTLIB_AVAILABLE = False

if TYPE_CHECKING:
    import matplotlib.axes

from .conversion import NxConverter

__all__ = [
    "create_isochrone",
    "create_tessellation",
    "filter_graph_by_distance",
    "plot_graph",
]

logger = logging.getLogger("city2graph.utils")

_KNN_HULL_MAX_ATTEMPTS = 50
_COARSE_GRID_SIZE: float = 1e-3
_JITTER_MAGNITUDE: float = 0.01

PLOT_DEFAULTS = {
    "node_color": "#22d3ee",
    "node_edgecolor": "none",
    "node_alpha": 0.8,
    "node_zorder": 2,
    "markersize": 4.0,
    "edge_color": "#ffffff",
    "edge_linewidth": 0.5,
    "edge_alpha": 0.3,
    "edge_zorder": 1,
    "title_color": "white",
}


def filter_graph_by_distance(
    graph: gpd.GeoDataFrame | nx.Graph | nx.MultiGraph,
    center_point: Point | Sequence[Point] | gpd.GeoSeries | gpd.GeoDataFrame,
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
    center_point : Point or Sequence[Point] or geopandas.GeoSeries or geopandas.GeoDataFrame
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

    distances = _compute_center_node_distances(
        nx_graph,
        center_point=center_point,
        edge_attr=edge_attr,
        cutoff=threshold,
    )
    if not distances and not nx.get_node_attributes(nx_graph, "pos"):
        if is_graph_input:
            graph_type = type(graph)
            return graph_type()
        return gpd.GeoDataFrame(geometry=[], crs=original_crs)

    subgraph = _build_reachable_subgraph(nx_graph, distances, threshold)

    if is_graph_input:
        return subgraph

    # Convert back to GeoDataFrame
    converter = NxConverter()
    return converter.nx_to_gdf(subgraph, nodes=False, edges=True)


def create_isochrone(
    graph: nx.Graph | nx.MultiGraph | None = None,
    nodes: gpd.GeoDataFrame | dict[str, gpd.GeoDataFrame] | None = None,
    edges: gpd.GeoDataFrame | dict[tuple[str, str, str], gpd.GeoDataFrame] | None = None,
    center_point: Point | Sequence[Point] | gpd.GeoSeries | gpd.GeoDataFrame | None = None,
    threshold: float | Sequence[float] | None = None,
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
    center_point : Point or Sequence[Point] or geopandas.GeoSeries or geopandas.GeoDataFrame
        The origin point(s) for the isochrone calculation. When multiple
        points are provided, reachability is unioned across all centers.
    threshold : float or Sequence[float]
        The maximum travel distance (or time) that defines the boundary of the
        isochrone. When a sequence is provided, the function returns one
        isochrone layer per threshold using a single shared distance
        calculation.
    edge_attr : str, default "travel_time"
        The edge attribute to use for distance calculation (e.g., 'length',
        'travel_time'). If None, the function will use the default edge attribute.
    cut_edge_types : list[tuple[str, str, str]] | None, default None
        List of edge types to remove from the graph before processing (e.g.,
        [("bus_stop", "is_next_to", "bus_stop")]).
    method : str, default "concave_hull_knn"
        The method to generate the isochrone polygon. Options are:

        - "concave_hull_knn": Creates a concave hull (k-NN) around reachable nodes.
          This iterative algorithm scales poorly with point count; for large
          networks (tens of thousands of reachable nodes), prefer
          "concave_hull_alpha", which uses shapely's C implementation and is
          orders of magnitude faster.
        - "concave_hull_alpha": Creates a concave hull (alpha shape) around reachable nodes.
        - "convex_hull": Creates a convex hull around reachable nodes.
        - "buffer": Creates a buffer around reachable edges/nodes.
    **kwargs : Any
        Additional parameters for specific isochrone generation methods:

        For method="concave_hull_knn":
            k : int, default 50
                The number of nearest neighbors to consider. Increasing `k`
                generally produces a smoother, less concave boundary that moves
                closer to the convex hull.

        For method="concave_hull_alpha":
            hull_ratio : float, default 0.0
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
        A GeoDataFrame containing Polygon or MultiPolygon geometry. Scalar
        threshold input returns a single-row GeoDataFrame matching the existing
        API. Sequence input returns one row per threshold with columns
        ``threshold`` and ``geometry``.

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

    thresholds, is_multi_threshold = _normalize_isochrone_thresholds(threshold)

    # Prepare the graph
    nx_graph = _prepare_isochrone_graph(graph, nodes, edges, edge_attr)
    crs = nx_graph.graph.get("crs")

    distances = _compute_center_node_distances(
        nx_graph,
        center_point=center_point,
        edge_attr=edge_attr,
        cutoff=max(thresholds),
    )

    if is_multi_threshold:
        rows: list[dict[str, float | Polygon | MultiPolygon | GeometryCollection]] = []
        for layer_threshold in thresholds:
            reachable = _build_reachable_subgraph(nx_graph, distances, layer_threshold)
            if cut_edge_types:
                reachable = _filter_edges_by_type(reachable, cut_edge_types)

            geometry = _build_isochrone_geometry(reachable, method, crs, **kwargs)
            rows.append(
                {
                    "threshold": layer_threshold,
                    "geometry": geometry if geometry is not None else GeometryCollection(),
                }
            )

        return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)

    reachable = _build_reachable_subgraph(nx_graph, distances, thresholds[0])

    # Filter Edge Types if requested
    if cut_edge_types:
        reachable = _filter_edges_by_type(reachable, cut_edge_types)

    final_geom = _build_isochrone_geometry(reachable, method, crs, **kwargs)

    if final_geom is None:
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    return gpd.GeoDataFrame(geometry=[final_geom], crs=crs)


def _normalize_isochrone_thresholds(
    threshold: float | Sequence[float],
) -> tuple[list[float], bool]:
    """
    Normalize scalar or layered isochrone thresholds.

    This helper converts all threshold inputs to a float list and records
    whether the caller requested layered output.

    Parameters
    ----------
    threshold : float or Sequence[float]
        The requested threshold input.

    Returns
    -------
    tuple[list[float], bool]
        Normalized thresholds and a flag indicating sequence input.

    Raises
    ------
    ValueError
        If a threshold sequence is empty.
    """
    if isinstance(threshold, Sequence) and not isinstance(threshold, str | bytes):
        if not threshold:
            msg = "threshold sequence must not be empty."
            raise ValueError(msg)
        return [float(value) for value in threshold], True

    return [float(threshold)], False


def _normalize_center_points(
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


def _compute_center_node_distances(
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

    source_nodes = []
    for point in _normalize_center_points(center_point):
        _, idx = tree.query((point.x, point.y))
        source_nodes.append(node_ids[idx])

    if not source_nodes:
        return {}

    deduped_sources = list(dict.fromkeys(source_nodes))
    distances = nx.multi_source_dijkstra_path_length(
        graph,
        deduped_sources,
        cutoff=cutoff,
        weight=edge_attr or "length",
    )
    return typing.cast("dict[Any, float]", distances)


def _build_reachable_subgraph(
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


def _build_isochrone_geometry(
    reachable: nx.Graph | nx.MultiGraph,
    method: str,
    crs: Any,  # noqa: ANN401
    **kwargs: Any,  # noqa: ANN401
) -> Polygon | MultiPolygon | None:
    """
    Build the final isochrone geometry for a reachable subgraph.

    Component polygons are generated per connected component and merged into a
    single polygonal geometry for output.

    Parameters
    ----------
    reachable : networkx.Graph or networkx.MultiGraph
        Reachable subgraph.
    method : str
        Isochrone generation method.
    crs : Any
        Coordinate reference system.
    **kwargs : Any
        Geometry generation options.

    Returns
    -------
    Polygon or MultiPolygon or None
        Final geometry, or None when no component polygons can be generated.
    """
    polygons = _generate_component_polygons(reachable, method, crs, **kwargs)
    if not polygons:
        return None

    final_geom = gpd.GeoSeries(polygons, crs=crs).union_all()

    if not isinstance(final_geom, (Polygon, MultiPolygon)):
        final_geom = final_geom.buffer(0)

    return final_geom


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
    if method == "concave_hull_knn":
        coords = _extract_node_coordinate_array(component)
        coords = coords[np.isfinite(coords).all(axis=1)]
        if len(coords) == 0:
            return None
        gs = gpd.GeoSeries(shapely.points(coords[:, 0], coords[:, 1]), crs=crs)
        poly = _generate_concave_hull_knn(gs, **kwargs)
    else:
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
    k = kwargs.get("k", 50)
    coords = _extract_coordinate_array_from_geometries(gs)
    return _concave_hull_knn(coords, k=k) if len(coords) else None


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
    hull_ratio = kwargs.get("hull_ratio", 0.0)
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


def _extract_node_coordinate_array(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """
    Extract node positions as a dense (n, 2) NumPy array.

    This avoids building temporary Point geometries when only raw coordinates are
    needed, such as for the k-NN concave hull path.

    Parameters
    ----------
    graph : networkx.Graph or networkx.MultiGraph
        Input graph containing node ``pos`` attributes.

    Returns
    -------
    np.ndarray
        Node coordinates with shape ``(n, 2)`` and dtype float. An empty array
        is returned when coordinates are unavailable or malformed.
    """
    pos_dict = nx.get_node_attributes(graph, "pos")
    coords = np.asarray(list(pos_dict.values()), dtype=float) if pos_dict else np.empty((0, 2))
    if coords.ndim != 2 or coords.shape[1] < 2:
        return np.empty((0, 2), dtype=float)
    return coords[:, :2]


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


def _concave_hull_knn(
    points: list[Point] | np.ndarray, k: int
) -> Polygon | MultiPolygon | LineString | Point:
    """
    Compute the concave hull of a set of points using the k-nearest neighbors approach.

    This function implements the k-nearest neighbors algorithm to generate a concave hull
    from a set of points. It constructs the hull by iteratively finding the next point
    that forms the largest right-turn angle, ensuring a tight fit around the point cloud.
    Larger values of `k` usually produce a smoother, less detailed boundary and can
    make the result approach the convex hull, while smaller values allow tighter,
    more locally adaptive concavities.

    Parameters
    ----------
    points : list[Point] or np.ndarray
        The input points.
    k : int
        The number of nearest neighbors to consider. Increasing `k` generally
        reduces concavity and smooths the hull.

    Returns
    -------
    Polygon or LineString or Point
        The concave hull geometry.

    References
    ----------
    .. [1] Moreira, Adriano and Santos, Maribel Yasmina. "Concave hull:
       A k-nearest neighbours approach for the computation of the region occupied
       by a set of points." International Conference on Computer Graphics Theory
       and Applications, vol. 2, pp. 61-68, SciTePress, 2007.
    """
    raw_coords = (
        np.asarray([(p.x, p.y) for p in points], dtype=float)
        if isinstance(points, list)
        else np.asarray(points, dtype=float)
    )

    raw_coords = np.atleast_2d(raw_coords) if raw_coords.ndim == 1 else raw_coords
    if raw_coords.size == 0 or raw_coords.shape[1] < 2:
        return LineString()

    raw_coords = raw_coords[:, :2]
    raw_coords = raw_coords[np.isfinite(raw_coords).all(axis=1)]

    # Preserve original degenerate-case behavior before de-duplication.
    if len(raw_coords) < 3:
        return Point(raw_coords[0]) if len(raw_coords) == 1 else LineString(raw_coords)

    coords = np.unique(raw_coords, axis=0)
    n_points = len(coords)

    if n_points < 3:
        return LineString(coords) if n_points == 2 else Point(coords[0])

    start_idx = int(np.lexsort((coords[:, 0], coords[:, 1]))[0])
    min_k = max(2, min(int(k), n_points - 1))
    tree = cKDTree(coords)
    neighbor_cache: dict[int, np.ndarray] = {}

    # Cap the number of k-escalation retries: each retry is a full O(n) hull
    # walk, so an unbounded `range(min_k, n_points)` can degrade to O(n^2) on
    # large point clouds. Past the cap, fall back to the alpha-shape hull.
    max_k = min(n_points - 1, min_k + _KNN_HULL_MAX_ATTEMPTS - 1)

    for current_k in range(min_k, max_k + 1):
        hull_indices = _trace_concave_hull_once(
            coords,
            start_idx,
            current_k,
            tree=tree,
            neighbor_cache=neighbor_cache,
        )
        if hull_indices is None or len(hull_indices) < 3:
            continue

        poly = Polygon(coords[hull_indices])
        if not poly.is_valid:
            poly = poly.buffer(0)

        if poly.is_empty or not isinstance(poly, (Polygon, MultiPolygon)):
            continue

        if _polygon_covers_all_points(poly, coords):
            return poly

    return _concave_fallback_alpha(coords)


@dataclass(slots=True)
class _HullWalkState:
    """
    Mutable state for a single greedy concave-hull walk.

    The state tracks accepted edge endpoints and their axis-aligned bounds so
    edge intersection checks can run fully vectorized in NumPy.
    """

    seg_starts: np.ndarray
    seg_ends: np.ndarray
    seg_bounds_min: np.ndarray
    seg_bounds_max: np.ndarray
    segment_count: int
    neighbor_cache: dict[int, np.ndarray]

    @classmethod
    def create(
        cls, max_segments: int, *, neighbor_cache: dict[int, np.ndarray] | None = None
    ) -> "_HullWalkState":
        """
        Build an empty walk state with preallocated segment-bound arrays.

        This initializer centralizes array allocation and optional cache wiring
        so each hull trace starts from a predictable memory layout.

        Parameters
        ----------
        max_segments : int
            Upper bound for the number of segments expected during one hull
            trace. The preallocated arrays are sized from this value.
        neighbor_cache : dict[int, np.ndarray] or None, optional
            Optional shared cache for KDTree neighbor index lookups.

        Returns
        -------
        _HullWalkState
            Initialized state ready for a hull walk.
        """
        return cls(
            seg_starts=np.empty((max_segments, 2), dtype=float),
            seg_ends=np.empty((max_segments, 2), dtype=float),
            seg_bounds_min=np.empty((max_segments, 2), dtype=float),
            seg_bounds_max=np.empty((max_segments, 2), dtype=float),
            segment_count=0,
            neighbor_cache={} if neighbor_cache is None else neighbor_cache,
        )

    def append_segment(self, start: np.ndarray, end: np.ndarray) -> None:
        """
        Append an accepted hull segment and cache its endpoints and bbox.

        Raw endpoint coordinates and min/max bounds are written into
        preallocated arrays so intersection checks stay fully vectorized.

        Parameters
        ----------
        start : np.ndarray
            Segment start coordinate as a length-2 array.
        end : np.ndarray
            Segment end coordinate as a length-2 array.
        """
        if self.segment_count >= len(self.seg_bounds_min):
            grow_by = max(1, len(self.seg_bounds_min))
            grow = np.empty((grow_by, 2), dtype=float)
            self.seg_starts = np.vstack([self.seg_starts, grow])
            self.seg_ends = np.vstack([self.seg_ends, grow.copy()])
            self.seg_bounds_min = np.vstack([self.seg_bounds_min, grow.copy()])
            self.seg_bounds_max = np.vstack([self.seg_bounds_max, grow.copy()])
        self.seg_starts[self.segment_count] = start
        self.seg_ends[self.segment_count] = end
        self.seg_bounds_min[self.segment_count] = np.minimum(start, end)
        self.seg_bounds_max[self.segment_count] = np.maximum(start, end)
        self.segment_count += 1

    def bounds_min_view(self) -> np.ndarray:
        """
        Return the populated minimum bounds view.

        This slices the preallocated bounds array to only the rows associated
        with currently accepted segments.

        Returns
        -------
        np.ndarray
            Array of shape ``(segment_count, 2)`` with per-segment minimum
            coordinates.
        """
        return self.seg_bounds_min[: self.segment_count]

    def bounds_max_view(self) -> np.ndarray:
        """
        Return the populated maximum bounds view.

        This slices the preallocated bounds array to only the rows associated
        with currently accepted segments.

        Returns
        -------
        np.ndarray
            Array of shape ``(segment_count, 2)`` with per-segment maximum
            coordinates.
        """
        return self.seg_bounds_max[: self.segment_count]

    def starts_view(self) -> np.ndarray:
        """
        Return the populated segment start-point view.

        This slices the preallocated endpoint array to only the rows
        associated with currently accepted segments.

        Returns
        -------
        np.ndarray
            Array of shape ``(segment_count, 2)`` with per-segment start
            coordinates.
        """
        return self.seg_starts[: self.segment_count]

    def ends_view(self) -> np.ndarray:
        """
        Return the populated segment end-point view.

        This slices the preallocated endpoint array to only the rows
        associated with currently accepted segments.

        Returns
        -------
        np.ndarray
            Array of shape ``(segment_count, 2)`` with per-segment end
            coordinates.
        """
        return self.seg_ends[: self.segment_count]


def _trace_concave_hull_once(
    coords: np.ndarray,
    start_idx: int,
    k: int,
    *,
    tree: cKDTree | None = None,
    neighbor_cache: dict[int, np.ndarray] | None = None,
) -> list[int] | None:
    """
    Run one greedy k-NN hull walk.

    The walk iteratively chooses valid next hull points until it closes on the
    start index or fails to find a legal extension.

    Parameters
    ----------
    coords : np.ndarray
        Unique 2D point coordinates used to build the hull.
    start_idx : int
        Index of the starting vertex in ``coords``.
    k : int
        Number of nearest neighbors considered at each step.
    tree : scipy.spatial.cKDTree, optional
        Optional KDTree built from ``coords`` to avoid rebuilding it.
    neighbor_cache : dict[int, np.ndarray] or None, optional
        Optional cache of neighbor indices keyed by point index.

    Returns
    -------
    list[int] or None
        Ordered hull vertex indices when a closed walk is found, otherwise
        ``None``.
    """
    n_points = len(coords)
    tree = cKDTree(coords) if tree is None else tree

    hull_indices = [start_idx]
    current_idx = start_idx
    prev_vec = np.array([1.0, 0.0], dtype=float)
    visited = {start_idx}
    hull_state = _HullWalkState.create(max(0, n_points - 1), neighbor_cache=neighbor_cache)

    for _ in range(n_points + 1):
        next_idx = _find_next_hull_point(
            tree=tree,
            coords=coords,
            current_idx=current_idx,
            k=k,
            prev_vec=prev_vec,
            hull_indices=hull_indices,
            start_idx=start_idx,
            visited=visited,
            n_points=n_points,
            hull_state=hull_state,
        )

        if next_idx is None:
            return None

        if next_idx == start_idx:
            return hull_indices

        if len(hull_indices) >= 2:
            hull_state.append_segment(coords[hull_indices[-2]], coords[hull_indices[-1]])

        hull_indices.append(next_idx)
        visited.add(next_idx)
        prev_vec = coords[next_idx] - coords[current_idx]
        current_idx = next_idx

    return None


def _polygon_covers_all_points(
    poly: Polygon | MultiPolygon, coords: MultiPoint | np.ndarray, tol: float = 1e-9
) -> bool:
    """
    Check whether all points are covered by the polygonal hull.

    A small positive buffer is applied before testing to reduce numerical
    precision issues near polygon boundaries.

    Parameters
    ----------
    poly : Polygon or MultiPolygon
        Candidate hull geometry.
    coords : MultiPoint or np.ndarray
        Input points to test against the hull.
    tol : float, default=1e-9
        Buffer distance used to stabilize boundary inclusion checks.

    Returns
    -------
    bool
        ``True`` when all points are covered or touched by the buffered hull.
    """
    test_poly = poly.buffer(tol)
    point_coords = (
        shapely.get_coordinates(coords)
        if isinstance(coords, MultiPoint)
        else np.asarray(coords, dtype=float)
    )
    point_geoms = shapely.points(point_coords)
    shapely.prepare(test_poly)
    return bool(np.all(shapely.covers(test_poly, point_geoms)))


def _concave_fallback_alpha(
    coords: np.ndarray,
) -> Polygon | MultiPolygon | LineString | Point:
    """
    Build a fallback concave hull using alpha-shape candidates.

    Multiple alpha ratios are attempted from tighter to smoother boundaries
    before returning a final fallback geometry.

    Parameters
    ----------
    coords : np.ndarray
        Input 2D coordinates.

    Returns
    -------
    Polygon or MultiPolygon or LineString or Point
        First non-empty polygonal result, or the final alpha-shape fallback for
        degenerate inputs.
    """
    points = [Point(x, y) for x, y in coords]

    for ratio in (0.05, 0.1, 0.2, 0.35, 0.5):
        geom = _concave_hull_alpha(points, ratio=ratio, allow_holes=False)
        if isinstance(geom, (Polygon, MultiPolygon)) and not geom.is_empty:
            return geom

    return _concave_hull_alpha(points, ratio=0.5, allow_holes=False)


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
    hull_state: _HullWalkState | None = None,
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
    hull_state : _HullWalkState or None, optional
        Cached hull edges, segment bounds, and nearest-neighbor query results.

    Returns
    -------
    int or None
        The index of the next point, or None if no valid point found.
    """
    current_k = max(2, min(int(k), n_points - 1))
    neighbor_cache = {} if hull_state is None else hull_state.neighbor_cache
    neighbor_indices = neighbor_cache.get(current_idx)

    # Candidates rejected at a smaller k stay rejected: nothing in the hull
    # state changes between escalations, so only the newly added neighbors
    # need to be evaluated on each retry.
    tested = 0

    while current_k <= n_points - 1:
        required = min(current_k + 1, n_points)

        if neighbor_indices is None or len(neighbor_indices) < required:
            query_k = min(
                n_points,
                max(
                    required,
                    len(neighbor_indices) * 2 if neighbor_indices is not None else required,
                ),
            )
            _, neighbor_indices = tree.query(coords[current_idx], k=query_k)
            neighbor_indices = np.atleast_1d(neighbor_indices)
            neighbor_cache[current_idx] = neighbor_indices

        # Filter candidates
        candidates = [
            int(idx)
            for idx in neighbor_indices[tested:required]
            if idx != current_idx
            and (idx not in visited or (idx == start_idx and len(hull_indices) >= 3))
        ]
        tested = required

        if candidates:
            best_idx = _find_best_candidate(
                coords,
                current_idx,
                candidates,
                prev_vec,
                hull_indices,
                start_idx,
                hull_state=hull_state,
            )
            if best_idx is not None:
                return best_idx

        current_k = n_points if current_k == n_points - 1 else min(current_k + 5, n_points - 1)

    return None


def _find_best_candidate(
    coords: np.ndarray,
    current_idx: int,
    candidates: list[int],
    prev_vec: np.ndarray,
    hull_indices: list[int],
    start_idx: int,
    hull_state: _HullWalkState | None = None,
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
    hull_state : _HullWalkState or None, optional
        Cached hull edges and segment bounds used to prevent self-intersections.

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
    candidates_array = np.asarray(candidates, dtype=int)[valid_mask]
    vecs /= norms[valid_mask][:, np.newaxis]

    # Calculate angles relative to previous vector
    prev_angle = np.arctan2(prev_vec[1], prev_vec[0])
    angles = np.arctan2(vecs[:, 1], vecs[:, 0])

    # Calculate angle difference (preference for right turns / largest inner angle)
    # We want the point that is "most right" relative to our current direction
    diffs = (angles - prev_angle + np.pi) % (2 * np.pi) - np.pi
    sorted_indices = np.argsort(diffs)

    if hull_state is None:
        hull_state = _HullWalkState.create(max(0, len(hull_indices) - 2))

    if hull_state.segment_count == 0:
        return int(candidates_array[sorted_indices[0]])

    # Validate candidates in blocks so the segment-intersection test runs as
    # a single vectorized operation per block instead of one Python call per
    # candidate. The first valid candidate in preference order wins.
    ordered = candidates_array[sorted_indices]
    block_size = 32
    for block_start in range(0, len(ordered), block_size):
        block = ordered[block_start : block_start + block_size]
        valid = _edges_valid_batch(
            start=current_pos.astype(float),
            ends=coords[block].astype(float),
            closing_mask=block == start_idx,
            hull_state=hull_state,
        )
        hits = np.flatnonzero(valid)
        if len(hits):
            return int(block[hits[0]])

    return None


def _edges_valid_batch(
    start: np.ndarray,
    ends: np.ndarray,
    closing_mask: np.ndarray,
    hull_state: _HullWalkState,
) -> np.ndarray:
    """
    Validate a batch of candidate edges against the accepted hull segments.

    All candidate edges share the same start vertex. An edge is invalid when
    it intersects any existing hull segment, except that the loop-closing
    edge may touch the first hull segment in a single point.

    Parameters
    ----------
    start : np.ndarray
        Shared start coordinate of the candidate edges as a length-2 array.
    ends : np.ndarray
        Candidate end coordinates with shape ``(m, 2)``.
    closing_mask : np.ndarray
        Boolean mask of shape ``(m,)`` flagging candidates that close the
        hull loop back to the start vertex.
    hull_state : _HullWalkState
        Accepted hull segments with cached endpoints and bounds.

    Returns
    -------
    np.ndarray
        Boolean mask of shape ``(m,)`` that is ``True`` for valid edges.
    """
    seg_bounds_min = hull_state.bounds_min_view()
    seg_bounds_max = hull_state.bounds_max_view()

    # Prefilter segments against the union bbox of all candidate edges.
    union_min = np.minimum(ends.min(axis=0), start)
    union_max = np.maximum(ends.max(axis=0), start)
    overlap = np.flatnonzero(
        np.all((seg_bounds_max >= union_min) & (seg_bounds_min <= union_max), axis=1)
    )
    if len(overlap) == 0:
        return np.ones(len(ends), dtype=bool)

    seg_starts = hull_state.starts_view()[overlap]
    seg_ends = hull_state.ends_view()[overlap]
    seg_dirs = seg_ends - seg_starts
    new_dirs = ends - start

    # Orientation terms; d1 depends only on the shared start vertex.
    d1 = _cross_2d(seg_dirs, start - seg_starts)
    d2 = _cross_2d(seg_dirs[np.newaxis], ends[:, np.newaxis, :] - seg_starts[np.newaxis])
    d3 = _cross_2d(new_dirs[:, np.newaxis, :], (seg_starts - start)[np.newaxis])
    d4 = _cross_2d(new_dirs[:, np.newaxis, :], (seg_ends - start)[np.newaxis])

    proper = (d1[np.newaxis] * d2 < 0) & (d3 * d4 < 0)

    seg_lo = np.minimum(seg_starts, seg_ends)
    seg_hi = np.maximum(seg_starts, seg_ends)
    start_on = (d1 == 0) & np.all((start >= seg_lo) & (start <= seg_hi), axis=1)
    ends_on = (d2 == 0) & np.all(
        (ends[:, np.newaxis, :] >= seg_lo[np.newaxis])
        & (ends[:, np.newaxis, :] <= seg_hi[np.newaxis]),
        axis=2,
    )
    edge_lo = np.minimum(ends, start)[:, np.newaxis, :]
    edge_hi = np.maximum(ends, start)[:, np.newaxis, :]
    seg_starts_on = (d3 == 0) & np.all(
        (seg_starts[np.newaxis] >= edge_lo) & (seg_starts[np.newaxis] <= edge_hi), axis=2
    )
    seg_ends_on = (d4 == 0) & np.all(
        (seg_ends[np.newaxis] >= edge_lo) & (seg_ends[np.newaxis] <= edge_hi), axis=2
    )

    intersects = proper | start_on[np.newaxis] | ends_on | seg_starts_on | seg_ends_on

    # Allow the loop-closing edge to touch the first hull segment in a single
    # point (shared start vertex or transversal crossing); only a collinear
    # overlap of positive length still invalidates it.
    closing_rows = np.flatnonzero(closing_mask)
    first_seg_pos = np.flatnonzero(overlap == 0)
    if len(closing_rows) and len(first_seg_pos):
        for row in closing_rows:
            allowed = _closing_touch_allowed_mask(
                start,
                ends[row],
                seg_starts[first_seg_pos],
                seg_ends[first_seg_pos],
                overlap[first_seg_pos],
            )
            intersects[row, first_seg_pos] &= ~allowed

    return typing.cast("np.ndarray", ~np.any(intersects, axis=1))


def _bbox_overlap_indices(
    start: np.ndarray,
    end: np.ndarray,
    seg_bounds_min: np.ndarray,
    seg_bounds_max: np.ndarray,
) -> np.ndarray:
    """
    Return indices of existing segments whose bounding boxes overlap the candidate segment.

    This is only a prefilter. Exact validity remains determined by the Shapely
    intersection checks in `_is_valid_edge`.

    Parameters
    ----------
    start : np.ndarray
        Start coordinate of the candidate segment.
    end : np.ndarray
        End coordinate of the candidate segment.
    seg_bounds_min : np.ndarray
        Per-segment minimum ``(x, y)`` bounds.
    seg_bounds_max : np.ndarray
        Per-segment maximum ``(x, y)`` bounds.

    Returns
    -------
    np.ndarray
        Integer indices of potentially overlapping segments.
    """
    edge_min = np.minimum(start, end)
    edge_max = np.maximum(start, end)

    overlap_mask = (
        (seg_bounds_max[:, 0] >= edge_min[0])
        & (seg_bounds_min[:, 0] <= edge_max[0])
        & (seg_bounds_max[:, 1] >= edge_min[1])
        & (seg_bounds_min[:, 1] <= edge_max[1])
    )
    return np.flatnonzero(overlap_mask)


def _is_valid_edge(
    coords: np.ndarray,
    current_idx: int,
    candidate_idx: int,
    start_idx: int,
    existing_lines: Sequence[LineString],
    seg_bounds: Sequence[tuple[float, float, float, float]] | None = None,
    seg_bounds_min: np.ndarray | None = None,
    seg_bounds_max: np.ndarray | None = None,
    seg_starts: np.ndarray | None = None,
    seg_ends: np.ndarray | None = None,
) -> bool:
    """
    Check if the new edge intersects with any existing hull edges.

    This validation ensures that adding the proposed edge does not create a self-intersecting
    polygon. It checks for intersections with all non-adjacent edges in the current hull
    using a vectorized NumPy orientation test, avoiding per-edge GEOS calls.

    Parameters
    ----------
    coords : np.ndarray
        Array of all point coordinates.
    current_idx : int
        Index of the current hull point.
    candidate_idx : int
        Index of the candidate point for the new edge.
    start_idx : int
        Index of the starting point of the hull.
    existing_lines : Sequence[LineString]
        Cached existing hull edges excluding the immediate predecessor edge.
        Used to derive segment endpoints when ``seg_starts``/``seg_ends`` are
        not provided.
    seg_bounds : Sequence[tuple[float, float, float, float]] or None, optional
        Cached per-edge bounding boxes used when available.
    seg_bounds_min : np.ndarray
        Minimum x/y values for the cached hull edge bounding boxes.
    seg_bounds_max : np.ndarray
        Maximum x/y values for the cached hull edge bounding boxes.
    seg_starts : np.ndarray or None, optional
        Per-segment start coordinates with shape ``(n, 2)``. Takes precedence
        over ``existing_lines`` when provided together with ``seg_ends``.
    seg_ends : np.ndarray or None, optional
        Per-segment end coordinates with shape ``(n, 2)``.

    Returns
    -------
    bool
        True if the edge is valid (no self-intersection), False otherwise.
    """
    if seg_starts is None or seg_ends is None:
        if not existing_lines:
            return True
        endpoint_pairs = shapely.get_coordinates(list(existing_lines)).reshape(-1, 2, 2)
        seg_starts = endpoint_pairs[:, 0]
        seg_ends = endpoint_pairs[:, 1]
    elif len(seg_starts) == 0:
        return True

    start = coords[current_idx]
    end = coords[candidate_idx]
    seg_bounds_min_arr, seg_bounds_max_arr = _coerce_seg_bounds_arrays(
        seg_bounds=seg_bounds,
        seg_bounds_min=seg_bounds_min,
        seg_bounds_max=seg_bounds_max,
    )
    possible_hits = _bbox_overlap_indices(start, end, seg_bounds_min_arr, seg_bounds_max_arr)
    if len(possible_hits) == 0:
        return True

    hit_starts = seg_starts[possible_hits]
    hit_ends = seg_ends[possible_hits]
    intersects = _segments_intersect_mask(start, end, hit_starts, hit_ends)

    if not bool(np.any(intersects)):
        return True

    # Allow touching at start point when closing the loop: the closing edge
    # may meet the first hull segment in a single point (their shared start
    # vertex or a transversal crossing), but not in a collinear overlap.
    if candidate_idx == start_idx:
        closing_ok = _closing_touch_allowed_mask(start, end, hit_starts, hit_ends, possible_hits)
        intersects &= ~closing_ok

    return not bool(np.any(intersects))


def _cross_2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute the scalar 2D cross product with broadcasting.

    This replaces ``np.cross`` for 2D inputs, whose 2D-vector support is
    deprecated since NumPy 2.0.

    Parameters
    ----------
    a : np.ndarray
        First vector(s), last axis of size 2.
    b : np.ndarray
        Second vector(s), last axis of size 2.

    Returns
    -------
    np.ndarray
        Scalar cross products ``a.x * b.y - a.y * b.x``.
    """
    return np.asarray(a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0])


def _segments_intersect_mask(
    start: np.ndarray,
    end: np.ndarray,
    seg_starts: np.ndarray,
    seg_ends: np.ndarray,
) -> np.ndarray:
    """
    Vectorized segment intersection test against a batch of segments.

    A segment pair is flagged when it properly crosses or touches at any
    point (including endpoint contact and collinear overlap), matching the
    non-empty-intersection semantics of the previous GEOS-based check.

    Parameters
    ----------
    start : np.ndarray
        Start coordinate of the candidate segment as a length-2 array.
    end : np.ndarray
        End coordinate of the candidate segment as a length-2 array.
    seg_starts : np.ndarray
        Batch of segment start coordinates with shape ``(n, 2)``.
    seg_ends : np.ndarray
        Batch of segment end coordinates with shape ``(n, 2)``.

    Returns
    -------
    np.ndarray
        Boolean mask of shape ``(n,)`` that is ``True`` where the candidate
        segment intersects the batch segment.
    """
    seg_dirs = seg_ends - seg_starts
    new_dir = end - start

    d1 = _cross_2d(seg_dirs, start - seg_starts)
    d2 = _cross_2d(seg_dirs, end - seg_starts)
    d3 = _cross_2d(new_dir, seg_starts - start)
    d4 = _cross_2d(new_dir, seg_ends - start)

    proper = (d1 * d2 < 0) & (d3 * d4 < 0)
    touch = (
        ((d1 == 0) & _points_within_bbox(start, seg_starts, seg_ends))
        | ((d2 == 0) & _points_within_bbox(end, seg_starts, seg_ends))
        | ((d3 == 0) & _points_within_bbox_single(seg_starts, start, end))
        | ((d4 == 0) & _points_within_bbox_single(seg_ends, start, end))
    )
    return typing.cast("np.ndarray", proper | touch)


def _points_within_bbox(point: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """
    Check whether a single point lies within each segment's bounding box.

    Combined with a zero cross product, this confirms the point lies on the
    segment itself.

    Parameters
    ----------
    point : np.ndarray
        Query coordinate as a length-2 array.
    starts : np.ndarray
        Segment start coordinates with shape ``(n, 2)``.
    ends : np.ndarray
        Segment end coordinates with shape ``(n, 2)``.

    Returns
    -------
    np.ndarray
        Boolean mask of shape ``(n,)``.
    """
    lo = np.minimum(starts, ends)
    hi = np.maximum(starts, ends)
    return typing.cast("np.ndarray", np.all((point >= lo) & (point <= hi), axis=1))


def _points_within_bbox_single(
    points: np.ndarray, start: np.ndarray, end: np.ndarray
) -> np.ndarray:
    """
    Check whether each point lies within a single segment's bounding box.

    Combined with a zero cross product, this confirms the points lie on the
    segment itself.

    Parameters
    ----------
    points : np.ndarray
        Query coordinates with shape ``(n, 2)``.
    start : np.ndarray
        Segment start coordinate as a length-2 array.
    end : np.ndarray
        Segment end coordinate as a length-2 array.

    Returns
    -------
    np.ndarray
        Boolean mask of shape ``(n,)``.
    """
    lo = np.minimum(start, end)
    hi = np.maximum(start, end)
    return typing.cast("np.ndarray", np.all((points >= lo) & (points <= hi), axis=1))


def _closing_touch_allowed_mask(
    start: np.ndarray,
    end: np.ndarray,
    hit_starts: np.ndarray,
    hit_ends: np.ndarray,
    hit_indices: np.ndarray,
) -> np.ndarray:
    """
    Identify allowed single-point contacts for the loop-closing edge.

    The closing edge is permitted to meet the first hull segment in exactly
    one point. Only a collinear overlap of positive length (a shared line
    section rather than a point) still invalidates the edge, mirroring the
    previous ``isinstance(intersection, Point)`` check.

    Parameters
    ----------
    start : np.ndarray
        Start coordinate of the closing edge as a length-2 array.
    end : np.ndarray
        End coordinate of the closing edge (the hull start vertex).
    hit_starts : np.ndarray
        Start coordinates of bbox-filtered hull segments, shape ``(n, 2)``.
    hit_ends : np.ndarray
        End coordinates of bbox-filtered hull segments, shape ``(n, 2)``.
    hit_indices : np.ndarray
        Original hull segment indices of the bbox-filtered segments.

    Returns
    -------
    np.ndarray
        Boolean mask of shape ``(n,)`` that is ``True`` where an intersection
        should be tolerated.
    """
    is_first_segment = hit_indices == 0
    if not bool(np.any(is_first_segment)):
        return np.zeros(len(hit_indices), dtype=bool)

    seg_dirs = hit_ends - hit_starts
    new_dir = end - start
    collinear = (
        (_cross_2d(seg_dirs, start - hit_starts) == 0)
        & (_cross_2d(seg_dirs, end - hit_starts) == 0)
        & (_cross_2d(new_dir, seg_dirs) == 0)
    )

    # Positive-length overlap of the two collinear segments along the
    # dominant axis means the intersection is a LineString, not a Point.
    axis = np.where(
        np.abs(hit_ends[:, 0] - hit_starts[:, 0]) >= np.abs(hit_ends[:, 1] - hit_starts[:, 1]),
        0,
        1,
    )
    idx = np.arange(len(hit_starts))
    seg_lo = np.minimum(hit_starts[idx, axis], hit_ends[idx, axis])
    seg_hi = np.maximum(hit_starts[idx, axis], hit_ends[idx, axis])
    edge_lo = np.minimum(start[axis], end[axis])
    edge_hi = np.maximum(start[axis], end[axis])
    overlap = np.minimum(seg_hi, edge_hi) - np.maximum(seg_lo, edge_lo)

    point_contact = ~(collinear & (overlap > 0))
    return typing.cast("np.ndarray", is_first_segment & point_contact)


def _coerce_seg_bounds_arrays(
    seg_bounds: Sequence[tuple[float, float, float, float]] | None = None,
    seg_bounds_min: np.ndarray | None = None,
    seg_bounds_max: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize cached segment bounds to ``(n, 2)`` NumPy arrays.

    This keeps `_is_valid_edge()` on the vectorized bbox-overlap path even when
    callers still provide legacy tuple-based bounds.

    Parameters
    ----------
    seg_bounds : Sequence[tuple[float, float, float, float]] or None, optional
        Legacy sequence of segment bounds as ``(minx, miny, maxx, maxy)``.
    seg_bounds_min : np.ndarray or None, optional
        Optional array of minimum coordinates with shape ``(n, 2)``.
    seg_bounds_max : np.ndarray or None, optional
        Optional array of maximum coordinates with shape ``(n, 2)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Pair ``(min_bounds, max_bounds)`` where each array has shape ``(n, 2)``.
    """
    bounds_arr = np.asarray([] if seg_bounds is None else seg_bounds, dtype=float)
    if seg_bounds_min is not None or seg_bounds_max is not None:
        min_arr = (
            np.empty((0, 2), dtype=float)
            if seg_bounds_min is None
            else np.asarray(seg_bounds_min, dtype=float).reshape(-1, 2)
        )
        max_arr = (
            np.empty((0, 2), dtype=float)
            if seg_bounds_max is None
            else np.asarray(seg_bounds_max, dtype=float).reshape(-1, 2)
        )
        bounds_arr = (
            np.hstack([min_arr, max_arr]) if len(min_arr) or len(max_arr) else np.empty((0, 4))
        )

    bounds_arr = (
        np.empty((0, 4), dtype=float) if bounds_arr.size == 0 else np.atleast_2d(bounds_arr)
    )
    return bounds_arr[:, :2], bounds_arr[:, 2:]


def _extract_coordinate_array_from_geometries(geoms: gpd.GeoSeries | list[Any]) -> np.ndarray:
    """
    Extract coordinates from geometries as a NumPy array.

    This is faster than converting coordinates into temporary Point objects when
    only raw coordinates are needed.

    Parameters
    ----------
    geoms : geopandas.GeoSeries or list[Any]
        Input geometries from which coordinates are extracted.

    Returns
    -------
    np.ndarray
        Coordinate array with shape ``(n, 2)`` and dtype float.
    """
    coords = np.asarray(shapely.get_coordinates(geoms), dtype=float)
    return np.empty((0, 2), dtype=float) if coords.size == 0 else coords[:, :2]


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
    limit: gpd.GeoDataFrame | gpd.GeoSeries | BaseGeometry | None = None,
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
    limit : geopandas.GeoDataFrame, geopandas.GeoSeries, shapely geometry, or None, optional
        Boundary passed to `momepy.enclosures` for enclosed tessellation. When
        None, a buffered (100 m) union of input geometry and barriers is
        computed and the enclosures are clipped to it.
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
            return _create_empty_tessellation(geometry.crs)
        # Morphological tessellation has no enclosures.
        return gpd.GeoDataFrame(
            columns=["geometry", "tess_id"],
            geometry="geometry",
            crs=geometry.crs,
        )

    if primary_barriers is not None:
        return _create_enclosed_tessellation(
            geometry,
            primary_barriers,
            limit,
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


def _polygonal_tessellation(tessellation: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Return only polygonal tessellation cells, salvaging GeometryCollections.

    Degenerate or overlapping footprints can make momepy emit GeometryCollection
    cells (especially on the ``simplify=False`` retry path). Downstream
    consumers such as libpysal contiguity weights only accept (Multi)Polygon
    geometries, so collections are replaced by the union of their polygonal
    parts and any remaining non-polygonal or empty rows are dropped.

    Parameters
    ----------
    tessellation : geopandas.GeoDataFrame
        Raw tessellation returned by momepy.

    Returns
    -------
    geopandas.GeoDataFrame
        The polygonal subset of ``tessellation``.
    """
    collections = tessellation.geom_type == "GeometryCollection"
    if collections.any():
        logger.warning(
            "Tessellation produced %d GeometryCollection cell(s); keeping only "
            "their polygonal parts.",
            int(collections.sum()),
        )
        tessellation = tessellation.copy()
        tessellation.loc[collections, tessellation.geometry.name] = tessellation.loc[
            collections
        ].geometry.apply(
            lambda geom: shapely.unary_union(
                [part for part in geom.geoms if part.geom_type in ("Polygon", "MultiPolygon")],
            ),
        )

    keep = tessellation.geometry.notna() & ~tessellation.geometry.is_empty
    keep &= tessellation.geom_type.isin(["Polygon", "MultiPolygon"])
    if not keep.all():
        tessellation = tessellation.loc[keep].copy()
    return tessellation


def _overfilled_enclosures(
    tessellation: gpd.GeoDataFrame,
    enclosures: gpd.GeoDataFrame,
    tolerance: float = 1.05,
) -> list[Any]:
    """
    Return enclosure indices whose cells' total area exceeds the enclosure area.

    A valid enclosed tessellation partitions each enclosure, so the summed cell
    area can never meaningfully exceed the enclosure's own area. A large excess
    means ``shapely.voronoi_polygons`` degenerated and every cell spans (nearly)
    the whole enclosure; such cells all contain the same building points and
    would explode downstream spatial joins.

    Parameters
    ----------
    tessellation : geopandas.GeoDataFrame
        Raw ``momepy.enclosed_tessellation`` output with an ``enclosure_index``
        column referring to ``enclosures`` index values.
    enclosures : geopandas.GeoDataFrame
        The enclosures the tessellation was generated from.
    tolerance : float, default 1.05
        Allowed ratio of summed cell area to enclosure area before an
        enclosure is reported as degenerate.

    Returns
    -------
    list[Any]
        The ``enclosure_index`` values of degenerate enclosures.
    """
    if tessellation.empty or "enclosure_index" not in tessellation.columns:
        return []
    cell_areas = tessellation.geometry.area.groupby(tessellation["enclosure_index"]).sum()
    enclosure_areas = enclosures.geometry.area.reindex(cell_areas.index)
    # NaN enclosure areas (unmatched indices) compare as False and are kept.
    broken = cell_areas > enclosure_areas * tolerance
    return list(cell_areas.index[broken])


def _jitter_hash_unit(coords: np.ndarray, salt: float) -> np.ndarray:
    """
    Map coordinate pairs to deterministic pseudo-random values in [0, 1).

    The value is a pure function of the coordinates and the salt, so repeated
    calls with the same input produce identical output. This keeps the
    geometry jitter reproducible across runs and consistent for vertices
    shared between geometries.

    Parameters
    ----------
    coords : numpy.ndarray
        Array of shape (n, 2) with x/y coordinates.
    salt : float
        Salt distinguishing independent channels (e.g. x- and y-offsets).

    Returns
    -------
    numpy.ndarray
        One value in [0, 1) per coordinate pair, identical for identical
        input pairs.
    """
    return np.abs(np.sin(coords[:, 0] * 12.9898 + coords[:, 1] * 78.233 + salt) * 43758.5453) % 1.0


def _shift_jittered_coordinates(coords: np.ndarray, magnitude: float) -> np.ndarray:
    """
    Offset an (n, 2) coordinate array by the deterministic jitter.

    Both offset channels derive from :func:`_jitter_hash_unit`, so the shift
    is reproducible and identical for identical coordinates.

    Parameters
    ----------
    coords : numpy.ndarray
        Array of shape (n, 2) with x/y coordinates.
    magnitude : float
        Maximum absolute offset per axis in map units.

    Returns
    -------
    numpy.ndarray
        The offset coordinates.
    """
    dx = (_jitter_hash_unit(coords, 0.0) - 0.5) * 2.0 * magnitude
    dy = (_jitter_hash_unit(coords, 1.0) - 0.5) * 2.0 * magnitude
    return coords + np.column_stack([dx, dy])


def _jitter_geometry(
    geometry: gpd.GeoDataFrame | gpd.GeoSeries,
    magnitude: float = _JITTER_MAGNITUDE,
) -> gpd.GeoDataFrame | gpd.GeoSeries:
    """
    Displace every vertex by a deterministic sub-centimetre offset.

    The offset is a pure function of the vertex coordinates, so identical
    vertices move identically: rings stay closed, shared party walls stay
    coincident, and repeated runs produce identical output. Used by the
    tessellation retry ladder to break the exact collinearity that makes
    ``shapely.voronoi_polygons`` degenerate on rectilinear footprints.

    Parameters
    ----------
    geometry : geopandas.GeoDataFrame or geopandas.GeoSeries
        Geometries to jitter.
    magnitude : float, default _JITTER_MAGNITUDE
        Maximum absolute offset per axis in map units.

    Returns
    -------
    geopandas.GeoDataFrame or geopandas.GeoSeries
        A copy with jittered vertex coordinates.
    """
    shift = partial(_shift_jittered_coordinates, magnitude=magnitude)
    jittered = geometry.copy()
    if isinstance(jittered, gpd.GeoDataFrame):
        jittered.geometry = shapely.transform(geometry.geometry, shift)
    else:
        jittered = gpd.GeoSeries(
            shapely.transform(geometry, shift),
            index=geometry.index,
            crs=geometry.crs,
        )
    return jittered


def _classify_tessellation_failure(error: Exception) -> str | None:
    """
    Classify a tessellation failure for the retry ladder.

    Known failures are the errors the retry ladder in
    :func:`_run_tessellation_with_retries` recovers from: the ``ValueError``
    momepy raises when an enclosure produces nothing to concatenate, the
    ``TypeError`` raised by ``shapely.coverage_simplify`` on non-polygonal
    cells, and any ``GEOSException``. Anything else must propagate.

    Parameters
    ----------
    error : Exception
        The exception raised by ``momepy.enclosed_tessellation``.

    Returns
    -------
    str or None
        ``"empty"``, ``"simplify"`` or ``"geos"`` for known failures, or
        ``None`` for unknown errors that must propagate.
    """
    if isinstance(error, ValueError):
        return "empty" if "No objects to concatenate" in str(error) else None
    if isinstance(error, TypeError):
        return "simplify" if "incorrect geometry type" in str(error) else None
    return "geos" if isinstance(error, shapely.errors.GEOSException) else None


def _next_retry_overrides(
    kind: str,
    overrides: dict[str, object],
    kwargs: dict[str, object],
) -> dict[str, object] | None:
    """
    Return the next retry overrides for a classified tessellation failure.

    A coarser ``grid_size`` repairs the underlying voronoi partition, so it
    is always tried before ``simplify=False``, which would keep degenerate
    cells; the ``simplify=False`` rung only applies to ``coverage_simplify``
    failures. Options pinned by the caller in ``kwargs`` (or already applied
    via ``overrides``) are never overridden, and the momepy concat failure
    has no remedy at all.

    Parameters
    ----------
    kind : str
        The failure kind from :func:`_classify_tessellation_failure`.
    overrides : dict[str, object]
        Retry overrides already in effect for the failed attempt.
    kwargs : dict[str, object]
        The caller-supplied momepy options (used to respect pinned values).

    Returns
    -------
    dict[str, object] or None
        The overrides for the next attempt, or ``None`` when the ladder is
        exhausted and the unit must degrade to an empty tessellation.
    """
    if kind != "empty" and "_jitter" not in overrides:
        if "grid_size" not in kwargs and "grid_size" not in overrides:
            return {**overrides, "grid_size": _COARSE_GRID_SIZE}
        if kind == "simplify" and "simplify" not in kwargs and "simplify" not in overrides:
            return {**overrides, "simplify": False}
        # The final jitter rung replaces the earlier workarounds instead of
        # stacking on them: snapping to the coarse grid would re-align the
        # jittered coordinates and reintroduce the degeneracy.
        return {"_jitter": True}
    return None


def _log_tessellation_retry(
    kind: str,
    error: Exception,
    added_option: str,
) -> None:
    """
    Log the retry decision for a known tessellation failure.

    The wording mirrors the failure kind and the option added by the next
    rung so operators can follow the ladder in the logs.

    Parameters
    ----------
    kind : str
        The failure kind from :func:`_classify_tessellation_failure`.
    error : Exception
        The failure that triggered the retry.
    added_option : str
        The option the next rung adds: ``"grid_size"``, ``"simplify"`` or
        ``"_jitter"``.
    """
    if added_option == "grid_size" and kind == "geos":
        logger.warning(
            "Tessellation hit a GEOS topology error (%s); retrying with coarser grid_size=1e-3.",
            error,
        )
    elif added_option == "grid_size":
        logger.warning(
            "Tessellation boundary simplification failed (%s); retrying with grid_size=1e-3.",
            error,
        )
    elif added_option == "_jitter":
        logger.warning(
            "Tessellation stayed degenerate (%s); retrying with jittered geometry.",
            error,
        )
    else:
        logger.warning(
            "Tessellation boundary simplification failed (%s); retrying with simplify=False.",
            error,
        )


def _log_tessellation_degradation(error: Exception) -> None:
    """
    Log a known terminal failure before degrading the unit to an empty result.

    The warning message depends on the failure kind: the momepy concat
    ``ValueError`` keeps its historical wording, while simplification and GEOS
    failures report the exhausted retry ladder.

    Parameters
    ----------
    error : Exception
        The known failure (see :func:`_classify_tessellation_failure`) that
        exhausted the retry ladder.
    """
    if isinstance(error, ValueError):
        logger.warning("Momepy could not generate tessellation, returning empty GeoDataFrame.")
        return
    logger.warning(
        "Tessellation still failed at coarser precision; returning empty "
        "tessellation for this unit.",
    )


def _log_repair_retry_failure(error: Exception) -> None:
    """
    Log an exhausted overlap-repair retry.

    Unlike the main ladder, a failed repair retry does not degrade the unit
    to empty output: the original cells are kept and only the degenerate
    enclosures are dropped by :func:`_repair_or_drop_degenerate_enclosures`.

    Parameters
    ----------
    error : Exception
        The known failure that exhausted the repair retry.
    """
    logger.warning(
        "Tessellation retry at coarser precision failed (%s); keeping the original cells.",
        error,
    )


def _run_tessellation_with_retries(
    run_tessellation: typing.Callable[..., gpd.GeoDataFrame],
    kwargs: dict[str, object],
    overrides: dict[str, object] | None = None,
    log_exhausted: typing.Callable[[Exception], None] = _log_tessellation_degradation,
) -> tuple[gpd.GeoDataFrame | None, dict[str, object]]:
    """
    Run momepy enclosed tessellation through the known-failure retry ladder.

    ``shapely.voronoi_polygons`` can fail numerically inside an enclosure at
    momepy's default snapping precision; depending on the options this
    surfaces as a ``TypeError`` from ``shapely.coverage_simplify``, a
    ``GEOSException``, or as a silently degenerate partition (handled
    separately by :func:`_repair_or_drop_degenerate_enclosures`). Each known
    failure escalates to the next rung chosen by
    :func:`_next_retry_overrides` — the coarser grid first, then
    ``simplify=False`` — until the options are exhausted and the unit
    degrades to an empty tessellation. Caller-pinned ``simplify``/
    ``grid_size`` values are never overridden; only unknown errors propagate.

    Parameters
    ----------
    run_tessellation : Callable[..., geopandas.GeoDataFrame]
        Closure invoking ``momepy.enclosed_tessellation``; keyword arguments
        are applied as retry overrides on top of the caller-supplied options.
    kwargs : dict[str, object]
        The caller-supplied momepy options (used to respect pinned values).
    overrides : dict[str, object], optional
        Retry overrides to start the ladder from (e.g. the coarser
        ``grid_size`` for the overlap repair); the first attempt runs with
        them applied.
    log_exhausted : Callable[[Exception], None], optional
        Warning emitted when the ladder is exhausted, receiving the terminal
        failure. Defaults to :func:`_log_tessellation_degradation`.

    Returns
    -------
    tuple[geopandas.GeoDataFrame or None, dict[str, object]]
        ``(tessellation, overrides)`` where ``tessellation`` is ``None`` when
        momepy definitively failed and the caller should return an empty
        tessellation, and ``overrides`` records the retry options that
        produced the result.
    """
    overrides = dict(overrides or {})
    while True:
        try:
            return run_tessellation(**overrides), overrides
        except (ValueError, TypeError, shapely.errors.GEOSException) as error:
            kind = _classify_tessellation_failure(error)
            if kind is None:
                raise
            next_overrides = _next_retry_overrides(kind, overrides, kwargs)
            if next_overrides is None:
                log_exhausted(error)
                return None, overrides
            added_option = next(key for key in next_overrides if key not in overrides)
            _log_tessellation_retry(kind, error, added_option)
            overrides = next_overrides


def _repair_or_drop_degenerate_enclosures(
    tessellation: gpd.GeoDataFrame,
    enclosures: gpd.GeoDataFrame,
    run_tessellation: typing.Callable[..., gpd.GeoDataFrame],
    kwargs: dict[str, object],
    overrides: dict[str, object],
) -> gpd.GeoDataFrame:
    """
    Validate a tessellation and retry or drop degenerate enclosures.

    A degenerate voronoi partition can come back without any error (notably
    under ``simplify=False``), so the cells are validated against the
    enclosure areas before they can poison downstream spatial joins. A broken
    partition is retried once with a coarser ``grid_size`` (unless one is
    already in effect) and then once with deterministically jittered geometry
    (which breaks the exact collinearity that degenerates the voronoi
    partition of rectilinear footprints); when a retry itself fails with a
    known error the original cells are kept, and enclosures that stay
    degenerate have their cells dropped with a warning.

    Parameters
    ----------
    tessellation : geopandas.GeoDataFrame
        Raw ``momepy.enclosed_tessellation`` output.
    enclosures : geopandas.GeoDataFrame
        The enclosures the tessellation was generated from.
    run_tessellation : Callable[..., geopandas.GeoDataFrame]
        Closure invoking ``momepy.enclosed_tessellation`` with overrides.
    kwargs : dict[str, object]
        The caller-supplied momepy options (used to respect pinned values).
    overrides : dict[str, object]
        Retry overrides already applied to produce ``tessellation``.

    Returns
    -------
    geopandas.GeoDataFrame
        The validated tessellation, with degenerate enclosures repaired or
        removed.
    """
    broken = _overfilled_enclosures(tessellation, enclosures)
    if broken and "grid_size" not in kwargs and "grid_size" not in overrides:
        logger.warning(
            "Tessellation produced overlapping cells in %d enclosure(s); retrying with "
            "grid_size=1e-3.",
            len(broken),
        )
        retried, overrides = _run_tessellation_with_retries(
            run_tessellation,
            kwargs,
            overrides={**overrides, "grid_size": _COARSE_GRID_SIZE},
            log_exhausted=_log_repair_retry_failure,
        )
        if retried is not None:
            tessellation = retried
            broken = _overfilled_enclosures(tessellation, enclosures)
    if broken and "_jitter" not in overrides:
        logger.warning(
            "Tessellation kept overlapping cells in %d enclosure(s); retrying with "
            "jittered geometry.",
            len(broken),
        )
        # Jitter replaces the earlier workarounds (see _next_retry_overrides):
        # the coarse grid would re-align the jittered coordinates.
        retried, _ = _run_tessellation_with_retries(
            run_tessellation,
            kwargs,
            overrides={"_jitter": True},
            log_exhausted=_log_repair_retry_failure,
        )
        if retried is not None:
            tessellation = retried
            broken = _overfilled_enclosures(tessellation, enclosures)
    if broken:
        dropped = tessellation["enclosure_index"].isin(broken)
        logger.warning(
            "Dropping %d enclosure(s) whose tessellation cells overlap instead of "
            "partitioning the enclosure; %d cell(s) removed and their buildings "
            "degrade to footprint fallback cells.",
            len(broken),
            int(dropped.sum()),
        )
        tessellation = tessellation.loc[~dropped]
    return tessellation


def _run_enclosed_tessellation(
    *,
    geometry: gpd.GeoDataFrame | gpd.GeoSeries,
    enclosures: gpd.GeoDataFrame,
    shrink: float,
    segment: float,
    threshold: float,
    n_jobs: int,
    kwargs: dict[str, object],
    **extra_kwargs: object,
) -> gpd.GeoDataFrame:
    """
    Run momepy enclosed tessellation with retry-specific option overrides.

    The retry ladder passes only override options into this helper. This
    function merges those options with the caller-supplied momepy options and
    handles the internal ``_jitter`` marker before dispatching to momepy.

    Parameters
    ----------
    geometry : geopandas.GeoDataFrame or geopandas.GeoSeries
        The geometries to tessellate around.
    enclosures : geopandas.GeoDataFrame
        The enclosures to tessellate within.
    shrink : float
        The distance to shrink the geometry.
    segment : float
        The segment length for discretizing the geometry.
    threshold : float
        The threshold for snapping skeleton endpoints.
    n_jobs : int
        The number of jobs to use for parallel processing.
    kwargs : dict[str, object]
        Caller-supplied momepy options.
    **extra_kwargs : object
        Retry overrides, such as ``simplify``, ``grid_size`` or the internal
        ``_jitter`` marker.

    Returns
    -------
    geopandas.GeoDataFrame
        The enclosed tessellation.
    """
    options = {**kwargs, **extra_kwargs}
    tess_geometry = _jitter_geometry(geometry) if options.pop("_jitter", False) else geometry
    return momepy.enclosed_tessellation(
        geometry=tess_geometry,
        enclosures=enclosures,
        shrink=shrink,
        segment=segment,
        threshold=threshold,
        n_jobs=n_jobs,
        **options,
    )


def _create_enclosed_tessellation(
    geometry: gpd.GeoDataFrame | gpd.GeoSeries,
    primary_barriers: gpd.GeoDataFrame | gpd.GeoSeries,
    limit: gpd.GeoDataFrame | gpd.GeoSeries | BaseGeometry | None,
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
    limit : geopandas.GeoDataFrame, geopandas.GeoSeries, shapely geometry, or None
        Boundary passed to `momepy.enclosures`. If None, a buffered union of
        geometry and barriers is derived and the enclosures are clipped to it.
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
    derived_limit = limit is None
    if derived_limit:
        limit = _compute_enclosure_limit(geometry, primary_barriers)

    # The derived limit is a non-convex buffered union that may contain holes;
    # polygonization turns those holes into faces outside the limit, so they
    # must be clipped away. User-supplied limits keep clip=False because
    # momepy's clip requires a polygonal limit and clipping would change the
    # documented semantics of an explicit ``limit``.
    enclosures = momepy.enclosures(
        primary_barriers=primary_barriers,
        limit=limit,
        additional_barriers=None,
        enclosure_id="eID",
        clip=derived_limit and limit is not None,
    )

    if not enclosures.empty:
        run_enclosed_tessellation = partial(
            _run_enclosed_tessellation,
            geometry=geometry,
            enclosures=enclosures,
            shrink=shrink,
            segment=segment,
            threshold=threshold,
            n_jobs=n_jobs,
            kwargs=kwargs,
        )
        tessellation, overrides = _run_tessellation_with_retries(
            run_enclosed_tessellation,
            kwargs,
        )
        if tessellation is None:
            # ``momepy.enclosed_tessellation`` crashes with "No objects to
            # concatenate" when no enclosure holds two or more buildings,
            # although every single-building enclosure is a perfectly valid
            # cell; recover those units instead of degrading them to empty.
            tessellation = _recover_tessellation_without_splits(geometry, enclosures)
            if tessellation is None:
                return _create_empty_tessellation(geometry.crs)
        else:
            tessellation = _repair_or_drop_degenerate_enclosures(
                tessellation,
                enclosures,
                run_enclosed_tessellation,
                kwargs,
                overrides,
            )
    else:
        tessellation = _create_empty_tessellation(geometry.crs)

    tessellation = _polygonal_tessellation(tessellation)
    if tessellation.empty:
        return _create_empty_tessellation(geometry.crs)

    tessellation["tess_id"] = [
        f"{i}_{j}"
        for i, j in zip(tessellation["enclosure_index"], tessellation.index, strict=False)
    ]
    return tessellation.reset_index(drop=True)


def _recover_tessellation_without_splits(
    geometry: gpd.GeoDataFrame | gpd.GeoSeries,
    enclosures: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame | None:
    """
    Recover the tessellation after momepy's empty-concatenation crash.

    ``momepy.enclosed_tessellation`` voronoi-partitions only enclosures that
    intersect two or more buildings and crashes with "No objects to
    concatenate" when none exist, although single-building and empty
    enclosures are perfectly valid cells. When that precondition holds (and
    at least one polygonal building sits in an enclosure) the equivalent
    output is built directly; otherwise ``None`` signals that the failure was
    genuine and the unit must degrade to an empty tessellation.

    Parameters
    ----------
    geometry : geopandas.GeoDataFrame or geopandas.GeoSeries
        Building footprints passed to the tessellation.
    enclosures : geopandas.GeoDataFrame
        Enclosure polygons produced by ``momepy.enclosures``.

    Returns
    -------
    geopandas.GeoDataFrame or None
        The recovered tessellation, or ``None`` when recovery does not apply.
    """
    polygonal = geometry.geometry.notna() & geometry.geometry.geom_type.isin(
        ["Polygon", "MultiPolygon"],
    )
    buildings = geometry[polygonal]
    if buildings.empty:
        return None
    enclosure_positions, _ = buildings.sindex.query(
        enclosures.geometry,
        predicate="intersects",
    )
    if len(enclosure_positions) == 0:
        return None
    _, counts = np.unique(enclosure_positions, return_counts=True)
    if (counts > 1).any():
        # Some enclosure did need splitting, so the crash came from elsewhere.
        return None
    logger.warning(
        "Momepy could not tessellate because no enclosure holds two or more "
        "buildings; assigning single-building enclosures as cells directly.",
    )
    return _enclosure_cells_without_splits(buildings, enclosures)


def _enclosure_cells_without_splits(
    geometry: gpd.GeoDataFrame | gpd.GeoSeries,
    enclosures: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Build the enclosed tessellation when no enclosure needs splitting.

    Equivalent of the ``momepy.enclosed_tessellation`` output for the case its
    own implementation crashes on: every enclosure intersecting exactly one
    building becomes that building's cell (indexed by the building), and
    enclosures without buildings keep momepy's negative-index convention.

    Parameters
    ----------
    geometry : geopandas.GeoDataFrame or geopandas.GeoSeries
        Building footprints.
    enclosures : geopandas.GeoDataFrame
        Enclosure polygons produced by ``momepy.enclosures``.

    Returns
    -------
    geopandas.GeoDataFrame
        One cell per enclosure with the ``enclosure_index`` column set,
        matching the raw momepy output schema.
    """
    enclosure_positions, building_positions = geometry.sindex.query(
        enclosures.geometry,
        predicate="intersects",
    )
    cell_index = np.full(len(enclosures), None, dtype=object)
    cell_index[enclosure_positions] = geometry.geometry.index.to_numpy()[building_positions]
    empty_mask = np.array([index is None for index in cell_index])
    cell_index[empty_mask] = range(-int(empty_mask.sum()), 0)
    cells = gpd.GeoDataFrame(
        {"enclosure_index": enclosures.index},
        geometry=enclosures.geometry.to_numpy(),
        index=pd.Index(cell_index),
        crs=enclosures.crs,
    )
    # Match the raw momepy output column order (geometry first).
    return cells[["geometry", "enclosure_index"]]


def _compute_enclosure_limit(
    geometry: gpd.GeoDataFrame | gpd.GeoSeries,
    primary_barriers: gpd.GeoDataFrame | gpd.GeoSeries,
    buffer: float = 100,
) -> BaseGeometry | None:
    """
    Compute a buffered-union limit for momepy enclosures.

    The limit needs to include both buildings and street barriers so buildings
    near the outer street loops are not excluded from enclosed tessellation.
    It follows the built fabric (buffered union) rather than a convex hull:
    a hull leaves a vast outermost enclosure whose Voronoi cells stretch from
    street-front buildings deep into empty land as needle-shaped artifacts.

    Parameters
    ----------
    geometry : gpd.GeoDataFrame | gpd.GeoSeries
        Building geometries used to compute boundaries.
    primary_barriers : gpd.GeoDataFrame | gpd.GeoSeries
        Street geometries forming enclosures.
    buffer : float, default 100
        Distance in map units to buffer each geometry before the union.

    Returns
    -------
    BaseGeometry | None
        A single geometry representing the enclosure limit, or None if no valid
        geometries exist. The result may contain holes (block interiors farther
        than ``buffer`` from any street or building); callers must clip the
        enclosures to the limit so those holes do not become enclosures.
    """
    geometries = [
        geom
        for source in (geometry.geometry, primary_barriers.geometry)
        for geom in source
        if geom is not None and not geom.is_empty
    ]
    if not geometries:
        return None

    return gpd.GeoSeries(geometries, crs=geometry.crs).buffer(buffer).union_all()


def _create_empty_tessellation(crs: Any) -> gpd.GeoDataFrame:  # noqa: ANN401
    """
    Create an empty enclosed-tessellation GeoDataFrame.

    This helper generates a properly structured but empty GeoDataFrame when no
    tessellation can be generated, matching the column schema of non-empty
    enclosed tessellations so downstream consumers see a uniform shape.

    Parameters
    ----------
    crs : Any
        The Coordinate Reference System.

    Returns
    -------
    geopandas.GeoDataFrame
        An empty tessellation GeoDataFrame with the columns ``geometry``,
        ``enclosure_index`` and ``tess_id``.
    """
    return gpd.GeoDataFrame(
        columns=["geometry", "enclosure_index", "tess_id"],
        geometry="geometry",
        crs=crs,
    )


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
) -> str | float | np.ndarray | pd.Series:
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
    str, float, pd.Series, or np.ndarray
        Resolved parameter value.
    """
    if param_value is None:
        return typing.cast("str | float | np.ndarray | pd.Series", default_value)
    if isinstance(param_value, pd.Series):
        return param_value.to_numpy()
    if isinstance(param_value, str) and param_value in gdf.columns:
        return typing.cast("str | float | np.ndarray | pd.Series", gdf[param_value].to_numpy())
    return typing.cast("str | float | np.ndarray | pd.Series", param_value)


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


def _get_style_param(
    global_kwargs: dict[str, Any],
    name: str,
    type_key: str | tuple[str, str, str] | None,
) -> Any:  # noqa: ANN401
    """
    Get a style parameter value with optional type-specific lookup.

    Type-specific plotting options are stored as dictionaries keyed by node or
    edge type. This helper resolves those dictionaries when a type key is
    available and otherwise returns the raw parameter value.

    Parameters
    ----------
    global_kwargs : dict[str, Any]
        The kwargs passed to the main plot_graph function.
    name : str
        Parameter name to look up.
    type_key : str or tuple or None
        The key identifying the node or edge type.

    Returns
    -------
    Any
        The parameter value, or type-specific value if ``type_key`` is set.
    """
    val = global_kwargs.get(name)
    return _resolve_type_parameter(val, type_key) if type_key else val


def _or_default(val: Any, default: Any) -> Any:  # noqa: ANN401
    """
    Return a fallback value when the resolved style value is None.

    Plotting defaults should replace missing values while preserving explicit
    falsy values such as ``0`` or ``False``.

    Parameters
    ----------
    val : Any
        Value to check.
    default : Any
        Default value to use if ``val`` is None.

    Returns
    -------
    Any
        ``val`` if not None, otherwise ``default``.
    """
    return val if val is not None else default


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
        "legend",
        "legend_kwargs",
        "title",
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
                    _get_style_param(global_kwargs, "edge_linewidth", type_key),
                    PLOT_DEFAULTS["edge_linewidth"],
                ),
                "alpha": _or_default(
                    _get_style_param(global_kwargs, "edge_alpha", type_key),
                    PLOT_DEFAULTS["edge_alpha"],
                ),
                "zorder": _or_default(
                    _get_style_param(global_kwargs, "edge_zorder", type_key),
                    PLOT_DEFAULTS["edge_zorder"],
                ),
                "color": _or_default(
                    _get_style_param(global_kwargs, "edge_color", type_key),
                    default_color,
                ),
            }
        )
    else:
        resolved.update(
            {
                "color": _or_default(
                    _get_style_param(global_kwargs, "node_color", type_key),
                    default_color or PLOT_DEFAULTS["node_color"],
                ),
                "alpha": _or_default(
                    _get_style_param(global_kwargs, "node_alpha", type_key),
                    PLOT_DEFAULTS["node_alpha"],
                ),
                "zorder": _or_default(
                    _get_style_param(global_kwargs, "node_zorder", type_key),
                    PLOT_DEFAULTS["node_zorder"],
                ),
                "edgecolor": _or_default(
                    _get_style_param(global_kwargs, "node_edgecolor", type_key),
                    PLOT_DEFAULTS["node_edgecolor"],
                ),
                "markersize": _or_default(
                    _get_style_param(global_kwargs, "markersize", type_key),
                    PLOT_DEFAULTS["markersize"],
                ),
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
            if param_name == "color" and isinstance(val, (pd.Series, np.ndarray)):
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
    title = kwargs.get("title")
    legend = kwargs.get("legend")
    legend_kwargs = kwargs.get("legend_kwargs")
    plot_kwargs = {k: v for k, v in kwargs.items() if k not in {"title", "legend", "legend_kwargs"}}

    # Plot edges first (in background)
    if edges is not None and isinstance(edges, gpd.GeoDataFrame):
        style_kwargs = _resolve_style_kwargs(plot_kwargs, None, is_edge=True)
        _plot_gdf(edges, ax, **style_kwargs)

    # Plot nodes on top
    if nodes is not None and isinstance(nodes, gpd.GeoDataFrame):
        style_kwargs = _resolve_style_kwargs(plot_kwargs, None, is_edge=False)
        if legend is not None:
            style_kwargs["legend"] = legend
        if legend_kwargs is not None:
            style_kwargs["legend_kwds"] = legend_kwargs
        _plot_gdf(nodes, ax, **style_kwargs)

    if title is not None:
        title_color = kwargs.get("title_color")
        ax.set_title(
            title,
            color=title_color if title_color is not None else PLOT_DEFAULTS["title_color"],
        )
