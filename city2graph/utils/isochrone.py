"""Isochrone polygon generation utilities."""

from __future__ import annotations

import logging
import typing
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import GeometryCollection
from shapely.geometry import LineString
from shapely.geometry import MultiPoint
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import Polygon

from city2graph.base import build_reachable_subgraph
from city2graph.base import compute_center_node_distances
from city2graph.base import row_batches

from .conversion import gdf_to_nx

__all__ = ["create_isochrone"]

logger = logging.getLogger("city2graph.utils")

_KNN_HULL_MAX_ATTEMPTS = 50
_HULL_NEIGHBOR_CACHE_BYTES = 8 * 1024 * 1024


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
        Edge types to omit from polygon construction after calculating travel
        distances (e.g., [("bus_stop", "is_next_to", "bus_stop")]). Nodes left
        isolated by these removals are omitted too; the excluded edges still
        contribute to reachability.
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
                Whether to allow holes in the concave hull. This controls hull
                construction; ``fill_holes`` instead removes holes after the
                buffer method has combined its polygons.

        For method="buffer":
            buffer_distance : float or None, default 100
                The distance to buffer reachable geometries, in CRS units.
                Use a suitable projected CRS for distance-based buffering.
                None skips this buffering step; degenerate points and lines
                still receive the existing fallback buffer.
            fill_holes : bool, default False
                Remove all interior rings after combining the buffered
                geometries. Separate polygon parts remain separate unless
                filling a hole absorbs a part contained inside that hole.
                This does not join gaps between separate buffers. Unlike
                ``allow_holes`` for "concave_hull_alpha", this is a
                post-processing step for "buffer" only.
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
        If required inputs are missing or invalid, or ``fill_holes`` is not
        a bool when ``method="buffer"``.

    Notes
    -----
    Hole filling can include areas without reachable roads, including water
    bodies. To respect a study boundary or excluded areas, clip the result
    after filling, using a boundary in the same CRS. Each threshold layer is
    processed independently; hole filling does not add cumulative unions
    between layers or change network reachability.
    """
    valid_methods = {"concave_hull_knn", "concave_hull_alpha", "convex_hull", "buffer"}
    if method not in valid_methods:
        msg = f"Unknown method: {method}. Must be one of {valid_methods}."
        raise ValueError(msg)

    if method == "buffer" and not isinstance(kwargs.get("fill_holes", False), bool):
        msg = "fill_holes must be a bool for method='buffer'."
        raise ValueError(msg)

    if center_point is None or threshold is None:
        msg = "center_point and threshold must be provided."
        raise ValueError(msg)

    thresholds, is_multi_threshold = _normalize_isochrone_thresholds(threshold)

    # Prepare the graph
    nx_graph = _prepare_isochrone_graph(graph, nodes, edges, edge_attr)
    crs = nx_graph.graph.get("crs")

    distances = compute_center_node_distances(
        nx_graph,
        center_point=center_point,
        edge_attr=edge_attr,
        cutoff=max(thresholds),
    )

    if is_multi_threshold:
        rows: list[dict[str, float | Polygon | MultiPolygon | GeometryCollection]] = []
        for layer_threshold in thresholds:
            reachable = build_reachable_subgraph(nx_graph, distances, layer_threshold)
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

    reachable = build_reachable_subgraph(nx_graph, distances, thresholds[0])

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

    if method == "buffer" and kwargs.get("fill_holes", False):
        if isinstance(final_geom, Polygon):
            final_geom = Polygon(final_geom.exterior)
        else:
            # A filled exterior may contain another part that occupied its hole.
            # Dissolve those overlaps without bridging gaps between exteriors.
            final_geom = shapely.union_all([Polygon(part.exterior) for part in final_geom.geoms])

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

    return gdf_to_nx(nodes=nodes, edges=edges)


def _filter_edges_by_type(
    graph: nx.Graph | nx.MultiGraph,
    cut_edge_types: list[tuple[str, str, str]],
) -> nx.Graph | nx.MultiGraph:
    """
    Remove edges of specified types from the graph.

    Remove matching edges by key and their newly isolated endpoints. Original
    isolated nodes remain eligible for point-based isochrones.

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
    affected_nodes: set[Any] = set()
    if isinstance(graph, nx.MultiGraph):
        keyed_edges = [
            (u, v, key)
            for u, v, key, data in graph.edges(keys=True, data=True)
            if (data.get("full_edge_type") or data.get("edge_type")) in cut_edge_types
        ]
        affected_nodes.update(node for u, v, _ in keyed_edges for node in (u, v))
        graph.remove_edges_from(keyed_edges)
    else:
        edges = [
            (u, v)
            for u, v, data in graph.edges(data=True)
            if (data.get("full_edge_type") or data.get("edge_type")) in cut_edge_types
        ]
        affected_nodes.update(node for edge in edges for node in edge)
        graph.remove_edges_from(edges)
    graph.remove_nodes_from(node for node in affected_nodes if graph.degree(node) == 0)
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
    neighbor_cache: dict[int, np.ndarray] = _HullNeighborCache()

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


class _HullNeighborCache(dict[int, np.ndarray]):
    """
    Bound retained neighbours while preserving query sizes and tie ordering.

    Evicted arrays can be recomputed with the same query size, including ties.
    """

    def __init__(self) -> None:
        """
        Initialise cache accounting.

        Query sizes survive eviction so subsequent queries retain their ordering.
        """
        super().__init__()
        self.sizes: dict[int, int] = {}
        self.nbytes = 0

    def __setitem__(self, key: int, value: np.ndarray) -> None:
        """
        Store neighbours within the cache budget.

        Query sizes are recorded even when the corresponding array is evicted.

        Parameters
        ----------
        key : int
            Point position used as a query origin.
        value : numpy.ndarray
            Ordered neighbour positions returned by the spatial index.
        """
        self.sizes[key] = len(value)
        previous = self.pop(key, None)
        if previous is not None:
            self.nbytes -= previous.nbytes
        budget = _HULL_NEIGHBOR_CACHE_BYTES
        while self and self.nbytes + value.nbytes > budget:
            self.nbytes -= self.pop(next(iter(self))).nbytes
        if value.nbytes <= budget:
            super().__setitem__(key, value)
            self.nbytes += value.nbytes


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
    ) -> _HullWalkState:
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
            previous_size = (
                neighbor_cache.sizes.get(current_idx, 0)
                if isinstance(neighbor_cache, _HullNeighborCache)
                else (len(neighbor_indices) if neighbor_indices is not None else 0)
            )
            query_k = min(
                n_points,
                max(required, previous_size if previous_size >= required else previous_size * 2),
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
    Validate hull candidates in bounded batches.

    Candidate blocks retain original order and use the same exact intersection predicates.

    Parameters
    ----------
    start : numpy.ndarray
        Shared candidate start coordinate.
    ends : numpy.ndarray
        Candidate endpoint coordinates.
    closing_mask : numpy.ndarray
        Whether each edge closes the hull.
    hull_state : _HullWalkState
        Previously accepted hull segments.

    Returns
    -------
    numpy.ndarray
        Boolean validity mask in candidate order.
    """
    valid = np.empty(len(ends), dtype=bool)
    for batch in row_batches(len(ends), max(1, hull_state.segment_count * 8)):
        valid[batch] = _edges_valid_block(start, ends[batch], closing_mask[batch], hull_state)
    return valid


def _edges_valid_block(
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
