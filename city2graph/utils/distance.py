"""Network shortest-path and distance utilities."""

from __future__ import annotations

import heapq
import math
import numbers
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely
from scipy import sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree

from city2graph.base import GeoDataProcessor
from city2graph.base import adjacency
from city2graph.base import build_reachable_subgraph
from city2graph.base import closest_positions
from city2graph.base import compute_center_node_distances
from city2graph.base import distance_batches
from city2graph.base import edge_endpoints
from city2graph.base import node_positions
from city2graph.base import row_batches

from .conversion import gdf_to_nx
from .conversion import nx_to_gdf

if TYPE_CHECKING:
    from collections.abc import Sequence

    from shapely.geometry import Point

__all__ = [
    "filter_graph_by_distance",
    "network_cost_matrix",
    "network_costs",
]


type _GraphArrays = tuple[
    np.ndarray, np.ndarray, object, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]
type _Points = gpd.GeoDataFrame | gpd.GeoSeries
type _GraphInput = (
    nx.Graph
    | gpd.GeoDataFrame
    | tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
    | tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]]
)


@dataclass(slots=True)
class _EdgeData:
    """
    Projected line metadata for edge-based point attachment.

    Stores endpoint vertex pairs, directional costs, and positive line lengths.
    """

    ends: np.ndarray
    rates: np.ndarray
    lengths: np.ndarray


@dataclass(slots=True)
class _Attachments:
    """
    Point attachments in original input order.

    Feature indices refer to the eligible node or edge index for the query.
    """

    labels: pd.Index
    feature: np.ndarray
    along: np.ndarray
    distance: np.ndarray


def _edge_groups(feature: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Group connected rows by physical edge for direct interior trips.

    Only connected points take part in same-edge comparisons.

    Parameters
    ----------
    feature : ndarray
        Connected feature index per input row, or -1 if disconnected.

    Returns
    -------
    tuple
        Sorted positions, distinct edge IDs and group boundaries.
    """
    rows = np.flatnonzero(feature >= 0)
    rows = rows[np.argsort(feature[rows], kind="stable")]
    keys, first = np.unique(feature[rows], return_index=True)
    return rows, keys, np.r_[first, len(rows)]


def _attachment_groups(feature: np.ndarray, along: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Index distinct attachments and their original row positions.

    Coincident projections share one endpoint search.

    Parameters
    ----------
    feature : ndarray
        Connected feature index per input row, or -1 if disconnected.
    along : ndarray
        Linear offset along the connected line in CRS units.

    Returns
    -------
    tuple
        Sorted positions and group boundaries.
    """
    pos = np.flatnonzero(feature >= 0)
    order = pos[np.lexsort((along[pos], feature[pos]))]
    breaks = np.flatnonzero((np.diff(feature[order]) != 0) | (np.diff(along[order]) != 0)) + 1
    bounds = np.r_[0, breaks, len(order)] if len(order) else np.array([0])
    return order, bounds


def _vertices(feature: np.ndarray, edge_data: _EdgeData | None) -> np.ndarray:
    """
    Return the base nodes available to each connected point.

    Node connections have one column and edge connections have two.

    Parameters
    ----------
    feature : ndarray
        Connected feature index per input row.
    edge_data : _EdgeData or None
        Projected edge endpoints, or None for node attachment.

    Returns
    -------
    ndarray
        Vertex positions for each connection.
    """
    if edge_data is None:
        return feature[:, None]
    return np.asarray(edge_data.ends[feature])


def _endpoint_costs(
    points: _Attachments, edge_data: _EdgeData | None, *, arrival: bool = False
) -> np.ndarray:
    """
    Return directional costs between projected points and their edge endpoints.

    One-way restrictions apply to interior projections.

    Parameters
    ----------
    points : _Attachments
        Connected points in input order.
    edge_data : _EdgeData or None
        Projected edge geometry metadata, or None for node attachment.
    arrival : bool, default False
        Compute incoming costs from endpoints when True, outgoing otherwise.

    Returns
    -------
    ndarray
        Network costs in endpoint order.
    """
    if edge_data is None:
        return np.zeros((len(points.feature), 1))
    length = edge_data.lengths[points.feature]
    fractions = np.column_stack((points.along, length - points.along)) / length[:, None]
    rates = edge_data.rates[points.feature, ::-1] if arrival else edge_data.rates[points.feature]
    return _fraction_cost(fractions, rates)


def _merge_lines(lines: np.ndarray) -> np.ndarray:
    """
    Merge multipart edge geometries while preferring directed continuity.

    Directed merging preserves part orientation when segments align; remaining
    multipart geometries fall back to undirected merging.

    Parameters
    ----------
    lines : ndarray
        Edge geometries in arc order.

    Returns
    -------
    ndarray
        Merged line geometries in input order.
    """
    multipart = shapely.get_type_id(lines) == 5
    if not np.any(multipart):
        return lines
    merged = lines.copy()
    parts = shapely.line_merge(merged[multipart], directed=True)
    still_multi = shapely.get_type_id(parts) == 5
    if np.any(still_multi):
        parts[still_multi] = shapely.line_merge(parts[still_multi])
    merged[multipart] = parts
    return merged


def _segments(
    coords: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    lines: np.ndarray,
    costs: np.ndarray,
    directed: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Group coincident road geometries and retain the cheapest arc in each direction.

    Distinct parallel geometries retain their own projection costs.

    Parameters
    ----------
    coords : ndarray
        Node coordinates in adjacency order.
    u : ndarray
        Source vertex of each directed arc.
    v : ndarray
        Target vertex of each directed arc.
    lines : ndarray
        Edge geometries in arc order.
    costs : ndarray
        Network costs in input order.
    directed : bool
        Whether arc orientation restricts traversal.

    Returns
    -------
    tuple
        Line geometries, endpoint IDs and directional costs.
    """
    if not len(lines):
        return lines, np.empty((0, 2), int), np.empty((0, 2))
    lines = _merge_lines(lines)
    if np.any(shapely.get_type_id(lines) != 1) or np.any(shapely.is_empty(lines)):
        msg = "Edge connection requires a continuous LineString"
        raise ValueError(msg)
    start, end = np.split(_endpoints(lines), 2)
    flipped = np.linalg.norm(start - coords[u], axis=1) > np.linalg.norm(end - coords[u], axis=1)
    backward = u > v
    reverse = flipped ^ backward
    lines[reverse] = shapely.reverse(lines[reverse])
    # Self-loops have no node ordering to distinguish reciprocal arcs.
    loops = u == v
    if np.any(loops):
        reversed_loops = shapely.reverse(lines[loops])
        swap = shapely.to_wkb(lines[loops]) > shapely.to_wkb(reversed_loops)
        lines[np.flatnonzero(loops)[swap]] = reversed_loops[swap]
        backward[loops] = swap
    a, b = np.minimum(u, v), np.maximum(u, v)
    keys = pd.MultiIndex.from_arrays([a, b, shapely.to_wkb(lines)])
    codes, unique = pd.factorize(keys)
    first = np.unique(codes, return_index=True)[1]
    rates = np.full((len(unique), 2), np.inf)
    np.minimum.at(rates, (codes, (~backward).astype(int)), costs)
    if not directed:
        rates[:] = rates.min(axis=1, keepdims=True)
    return lines[first], np.column_stack((a[first], b[first])), rates


def _nx_arrays(graph: nx.Graph, weight: str | None) -> _GraphArrays:
    """
    Read coordinates, layer labels and directed arcs in a common integer node order.

    Node iteration order establishes a stable index for the CSR matrix.

    Parameters
    ----------
    graph : nx.Graph
        Network with pos or x and y node attributes.
    weight : str | None
        Edge attribute for traversal cost. Omit to use planar length.

    Returns
    -------
    tuple
        Coordinates, layer labels, CRS, endpoints, geometries and weights.
    """
    coords_list: list[tuple[float, float]] = []
    layers_list: list[Any] = []
    for _, data in graph.nodes(data=True):
        if "pos" in data:
            coords_list.append((data["pos"][0], data["pos"][1]))
        elif "x" in data and "y" in data:
            coords_list.append((data["x"], data["y"]))
        else:
            msg = "Routing nodes require pos or x and y coordinates"
            raise ValueError(msg)
        layers_list.append(data.get("node_type"))
    coord = np.asarray(coords_list).reshape(-1, 2)
    layers = np.asarray(layers_list)
    ids = {node: i for i, node in enumerate(graph)}
    u, v, lines, values = [], [], [], []
    for source, target, data in graph.edges(data=True):
        u.append(ids[source])
        v.append(ids[target])
        lines.append(data.get("geometry"))
        if weight is not None:
            values.append(data.get(weight))
    return (
        coord,
        layers,
        graph.graph.get("crs"),
        np.array(u, int),
        np.array(v, int),
        np.array(lines, object),
        np.asarray(values),
    )


def _unpack_geograph(
    graph: _GraphInput,
) -> tuple[dict[Any, gpd.GeoDataFrame], dict[tuple[Any, Any, Any], gpd.GeoDataFrame]]:
    """
    Normalise GeoDataFrame graph inputs into typed node and edge dictionaries.

    Validates container structure before coordinate and topology extraction.

    Parameters
    ----------
    graph : _GraphInput
        Edges GeoDataFrame, ``(nodes, edges)`` pair, or heterogeneous layer dicts.

    Returns
    -------
    tuple
        Node-layer dictionary and edge-relation dictionary.
    """
    if isinstance(graph, gpd.GeoDataFrame):
        nodes_raw: Any = None
        edges_raw: Any = graph
    elif isinstance(graph, tuple) and len(graph) == 2:
        nodes_raw, edges_raw = graph
    else:
        msg = "graph must be a NetworkX graph, GeoDataFrame, or (nodes, edges) tuple"
        raise TypeError(msg)
    if isinstance(edges_raw, gpd.GeoDataFrame):
        relations: dict[tuple[Any, Any, Any], gpd.GeoDataFrame] = {(None, None, None): edges_raw}
    elif isinstance(edges_raw, dict):
        relations = edges_raw
    else:
        msg = "graph edges must be a GeoDataFrame or a dictionary of GeoDataFrames"
        raise TypeError(msg)
    if nodes_raw is None:
        frames: dict[Any, gpd.GeoDataFrame] = {}
    elif isinstance(nodes_raw, gpd.GeoDataFrame):
        frames = {None: nodes_raw}
    elif isinstance(nodes_raw, dict):
        frames = nodes_raw
    else:
        msg = "graph nodes must be None, a GeoDataFrame, or a dictionary of GeoDataFrames"
        raise TypeError(msg)
    return frames, relations


def _node_xy(frame: gpd.GeoDataFrame) -> np.ndarray:
    """
    Extract planar 2D coordinates for a node layer, using centroids for non-points.

    Point layers read coordinates directly to avoid centroid overhead.

    Parameters
    ----------
    frame : geopandas.GeoDataFrame
        Node layer with an active geometry column.

    Returns
    -------
    ndarray
        Two-column float array of node coordinates.
    """
    if frame.empty:
        return np.empty((0, 2), dtype=float)
    geom = frame.geometry.to_numpy()
    if np.any(shapely.is_missing(geom) | shapely.is_empty(geom)):
        msg = "Node geometries cannot be missing or empty"
        raise ValueError(msg)
    non_point = shapely.get_type_id(geom) != 0
    if np.any(non_point):
        geom = geom.copy()
        geom[non_point] = shapely.centroid(geom[non_point])
    return np.asarray(shapely.get_coordinates(geom), dtype=float)


def _register_implicit(
    layer: Any,  # noqa: ANN401
    keys: Any,  # noqa: ANN401
    xy: np.ndarray,
    implicit_ids: dict[tuple[Any, Any], int],
    implicit_coords: list[np.ndarray],
    types: list[Any],
) -> np.ndarray:
    """
    Assign stable vertex positions for implicit edge-only endpoints.

    Nodes in the same layer share one identifier namespace across edge relations.

    Parameters
    ----------
    layer : Any
        Node layer label for the endpoint column.
    keys : Any
        Sequence or array of node identifiers or coordinate tuples.
    xy : ndarray
        Endpoint coordinates in row order.
    implicit_ids : dict
        Running map of ``(layer, key)`` to vertex position.
    implicit_coords : list[ndarray]
        Running list of vertex coordinates.
    types : list[Any]
        Running list of vertex layer labels.

    Returns
    -------
    ndarray
        Integer vertex positions for each endpoint row.
    """
    out = np.empty(len(xy), dtype=int)
    for idx, (key, pt) in enumerate(zip(keys, xy, strict=True)):
        node_key = (layer, key)
        pos = implicit_ids.get(node_key)
        if pos is None:
            pos = len(implicit_ids)
            implicit_ids[node_key] = pos
            implicit_coords.append(pt)
            types.append(layer)
        out[idx] = pos
    return out


def _resolve_explicit_endpoints(
    source: Any,  # noqa: ANN401
    target: Any,  # noqa: ANN401
    endpoints: tuple[Any, Any] | None,
    lines: np.ndarray,
    layer_ids: dict[Any, pd.Index],
    layer_coords: dict[Any, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map edge endpoints onto explicit node layer positions by ID or coordinates.

    Validates that both referenced node layers exist before indexing.

    Parameters
    ----------
    source : Any
        Source node layer name.
    target : Any
        Target node layer name.
    endpoints : tuple or None
        Extracted endpoint identifier arrays, or None for coordinate lookup.
    lines : ndarray
        Edge geometries used when falling back to coordinate matching.
    layer_ids : dict
        Node indices keyed by layer name.
    layer_coords : dict
        Node coordinate arrays keyed by layer name.

    Returns
    -------
    tuple
        Local vertex positions for source and target endpoints.
    """
    if source not in layer_ids or target not in layer_ids:
        missing_layers = [layer for layer in (source, target) if layer not in layer_ids]
        msg = f"Edge layer references missing node layer(s): {missing_layers}"
        raise ValueError(msg)
    if endpoints is not None:
        return (
            np.asarray(layer_ids[source].get_indexer(endpoints[0])),
            np.asarray(layer_ids[target].get_indexer(endpoints[1])),
        )
    a, b = np.split(_endpoints(lines), 2)
    u = pd.MultiIndex.from_arrays(list(layer_coords[source].T)).get_indexer(
        pd.MultiIndex.from_arrays(list(a.T))
    )
    v = pd.MultiIndex.from_arrays(list(layer_coords[target].T)).get_indexer(
        pd.MultiIndex.from_arrays(list(b.T))
    )
    return np.asarray(u), np.asarray(v)


def _resolve_implicit_endpoints(
    source: Any,  # noqa: ANN401
    target: Any,  # noqa: ANN401
    endpoints: tuple[Any, Any] | None,
    lines: np.ndarray,
    *,
    single_layer: bool,
    coords: list[np.ndarray],
    implicit_ids: dict[tuple[Any, Any], int],
    implicit_coords: list[np.ndarray],
    types: list[Any],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resolve endpoints for edges-only networks without an explicit node table.

    Uses a vectorised factorisation path for single-layer inputs and per-layer
    registration when multiple layers or relations are present.

    Parameters
    ----------
    source : Any
        Source node layer label.
    target : Any
        Target node layer label.
    endpoints : tuple or None
        Extracted endpoint identifiers, or None for coordinate matching.
    lines : ndarray
        Edge geometries in arc order.
    single_layer : bool
        Whether the entire graph consists of one homogeneous edge layer.
    coords : list[ndarray]
        Accumulator for node coordinate arrays.
    implicit_ids : dict
        Running map of ``(layer, key)`` to vertex position.
    implicit_coords : list[ndarray]
        Running list of vertex coordinates.
    types : list[Any]
        Running list of vertex layer labels.

    Returns
    -------
    tuple
        Source and target vertex position arrays.
    """
    ends = _endpoints(lines)
    if single_layer:
        labels = (
            pd.Index(endpoints[0]).append(pd.Index(endpoints[1]))
            if endpoints is not None
            else pd.MultiIndex.from_arrays(list(ends.T))
        )
        codes, unique = pd.factorize(labels, sort=False)
        first = np.unique(codes, return_index=True)[1]
        coords.append(ends[first])
        types.extend([source] * len(unique))
        u, v = np.split(codes, 2)
        return u, v
    a_xy, b_xy = np.split(ends, 2)
    src_keys = endpoints[0] if endpoints is not None else [tuple(p) for p in a_xy]
    dst_keys = endpoints[1] if endpoints is not None else [tuple(p) for p in b_xy]
    u = _register_implicit(source, src_keys, a_xy, implicit_ids, implicit_coords, types)
    v = _register_implicit(target, dst_keys, b_xy, implicit_ids, implicit_coords, types)
    return u, v


def _edge_arrays(graph: _GraphInput, weight: str | None) -> _GraphArrays:
    """
    Read geospatial columns in node-layer order and map edge endpoints to integer positions.

    Each node layer has its own index and a disjoint range of CSR positions.

    Parameters
    ----------
    graph : _GraphInput
        Input spatial graph.
    weight : str | None
        Edge attribute for traversal cost. Omit to use planar length.

    Returns
    -------
    tuple
        Coordinates, layer labels, CRS, endpoints, geometries and weights.
    """
    if isinstance(graph, nx.Graph):
        return _nx_arrays(graph, weight)
    frames, relations = _unpack_geograph(graph)
    geom_frames = [
        f
        for f in (*frames.values(), *relations.values())
        if not f.empty and f.active_geometry_name is not None
    ]
    GeoDataProcessor.ensure_crs_consistency(*geom_frames)
    crs = next((f.crs for f in geom_frames if f.crs is not None), None)
    coords: list[np.ndarray] = []
    layer_ids: dict[Any, pd.Index] = {}
    layer_offsets: dict[Any, int] = {}
    types: list[Any] = []
    size = 0
    for layer, frame in frames.items():
        layer_ids[layer] = frame.index
        layer_offsets[layer] = size
        coords.append(_node_xy(frame))
        types.extend([layer] * len(frame))
        size += len(frame)
    layer_coords = dict(zip(frames, coords, strict=True))
    arc_sources, arc_targets, geometries, weights = [], [], [], []
    implicit_ids: dict[tuple[Any, Any], int] = {}
    implicit_coords: list[np.ndarray] = []
    single_implicit = not frames and len(relations) == 1
    for (source, _, target), frame in relations.items():
        if frame.empty:
            continue
        lines = (
            frame.geometry.to_numpy(copy=True)
            if frame.active_geometry_name
            else np.full(len(frame), None)
        )
        endpoints = edge_endpoints(frame)
        if not frames:
            u, v = _resolve_implicit_endpoints(
                source,
                target,
                endpoints,
                lines,
                single_layer=single_implicit and source == target,
                coords=coords,
                implicit_ids=implicit_ids,
                implicit_coords=implicit_coords,
                types=types,
            )
        else:
            u, v = _resolve_explicit_endpoints(
                source, target, endpoints, lines, layer_ids, layer_coords
            )
        if np.any(u < 0) or np.any(v < 0):
            msg = "An edge endpoint is missing from its node layer"
            raise ValueError(msg)
        arc_sources.append(u + layer_offsets.get(source, 0))
        arc_targets.append(v + layer_offsets.get(target, 0))
        geometries.append(lines)
        if weight is not None:
            if weight not in frame:
                msg = f"Edges require a non-negative numeric cost for {weight!r}"
                raise ValueError(msg)
            weights.append(frame[weight].to_numpy())
    if implicit_coords:
        coords.append(np.vstack(implicit_coords))
    return (
        np.concatenate(coords) if coords else np.empty((0, 2)),
        np.asarray(types),
        crs,
        np.concatenate(arc_sources) if arc_sources else np.empty(0, int),
        np.concatenate(arc_targets) if arc_targets else np.empty(0, int),
        np.concatenate(geometries) if geometries else np.empty(0, object),
        np.concatenate(weights) if weights else np.empty(0),
    )


def _endpoints(lines: np.ndarray) -> np.ndarray:
    """
    Extract oriented coordinates for networks with implicit nodes.

    Continuous multipart lines are merged before reading their endpoints.

    Parameters
    ----------
    lines : ndarray
        Edge geometries in arc order.

    Returns
    -------
    ndarray
        Start coordinates followed by end coordinates in input order.
    """
    continuous = _merge_lines(lines)
    ends = np.concatenate((shapely.get_point(continuous, 0), shapely.get_point(continuous, -1)))
    xy = np.asarray(shapely.get_coordinates(ends))
    if len(xy) != 2 * len(lines):
        msg = "Edges without explicit node coordinates require continuous line geometries"
        raise ValueError(msg)
    return xy


def _weights(values: np.ndarray, weight: str | None) -> np.ndarray:
    """
    Validate impedances before parallel arcs are reduced.

    Check the original values so numeric strings cannot enter the adjacency.

    Parameters
    ----------
    values : ndarray
        Edge attribute values before conversion.
    weight : str | None
        Edge attribute for traversal cost. Omit to use planar length.

    Returns
    -------
    ndarray
        Non-negative float costs, including infinite barriers.
    """
    raw = np.asarray(values)
    if raw.dtype.kind == "b" or (
        raw.dtype.kind not in "iuf"
        and not all(isinstance(v, numbers.Real) and not isinstance(v, bool) for v in raw)
    ):
        msg = f"Edges require a non-negative numeric cost for {weight!r}"
        raise ValueError(msg)
    costs = raw.astype(float, copy=False)
    if not np.all(costs >= 0):
        msg = f"Edges require a non-negative numeric cost for {weight!r}"
        raise ValueError(msg)
    return costs


def _snap_points(
    points: _Points,
    crs: object,
    connect_to: Literal["node", "edge"],
    max_connection_distance: float,
    nodes: np.ndarray,
    lines: np.ndarray,
    tree: cKDTree | shapely.STRtree | None,
) -> _Attachments:
    """
    Attach query points to eligible network nodes or line segments.

    Missing points and points beyond the distance limit retain their input positions.

    Parameters
    ----------
    points : GeoSeries or GeoDataFrame
        Spatial points to attach to the network.
    crs : object
        Expected coordinate reference system of the network.
    connect_to : {"node", "edge"}
        Attachment mode.
    max_connection_distance : float
        Inclusive snapping distance limit in planar CRS units.
    nodes : ndarray
        Eligible vertex indices for node attachment.
    lines : ndarray
        Eligible LineString geometries for edge attachment.
    tree : cKDTree or STRtree or None
        Spatial index for eligible attachment features.

    Returns
    -------
    _Attachments
        Input labels, connected feature indices, offsets and distances.
    """
    if crs is not None and points.crs != crs:
        msg = "Points and graph must have the same CRS"
        raise ValueError(msg)
    geometry = points.geometry.to_numpy()
    valid = ~(shapely.is_missing(geometry) | shapely.is_empty(geometry))
    if np.any(shapely.get_type_id(geometry[valid]) != 0):
        msg = "Origins and destinations must be points"
        raise ValueError(msg)
    indices = np.flatnonzero(valid)
    if len(indices):
        xy = shapely.get_coordinates(geometry[indices])
        finite = np.all(np.isfinite(xy), axis=1)
        indices, xy = indices[finite], xy[finite]
    else:
        xy = np.empty((0, 2), dtype=float)
    feature = np.full(len(points), -1)
    distance = np.full(len(points), np.inf)
    along = np.zeros(len(points))
    if len(indices) and connect_to == "node" and isinstance(tree, cKDTree):
        dist, nearest = tree.query(xy)
        feature[indices], distance[indices] = nodes[nearest], dist
    elif len(indices) and connect_to == "edge" and isinstance(tree, shapely.STRtree) and lines.size:
        pairs, dist = tree.query_nearest(geometry[indices], return_distance=True)
        # Resolve equidistant lines in their stable graph order.
        order = np.lexsort((pairs[1], pairs[0]))
        _, first = np.unique(pairs[0, order], return_index=True)
        hits = order[first]
        points_at, lines_at = indices[pairs[0, hits]], pairs[1, hits]
        feature[points_at], distance[points_at] = lines_at, dist[hits]
        along[points_at] = shapely.line_locate_point(lines[lines_at], geometry[points_at])
    feature[distance > max_connection_distance] = -1
    return _Attachments(points.index.copy(), feature, along, distance)


def _fraction_cost(fraction: np.ndarray, rate: np.ndarray | float) -> np.ndarray:
    """
    Allow endpoint access at zero distance even on one-way edges.

    A zero-length connection remains free at a one-way endpoint.

    Parameters
    ----------
    fraction : ndarray
        Proportion of the edge to traverse.
    rate : ndarray | float
        Full traversal cost in this direction.

    Returns
    -------
    ndarray
        Directional cost of travelling the specified fraction.
    """
    result = np.zeros_like(fraction, dtype=float)
    np.multiply(fraction, rate, out=result, where=fraction != 0)
    return result


def _nearest_labels(
    matrix: sparse.csr_matrix, seeds: np.ndarray, labels: np.ndarray, limit: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assign equal-cost destinations by input order on the shortest-path arcs.

    Traverse optimal arcs in destination order to resolve ties through zero-cost cycles.

    Parameters
    ----------
    matrix : sparse.csr_matrix
        Directed search adjacency.
    seeds : ndarray
        Vertices at which the traversal begins.
    labels : ndarray
        Destination row positions associated with search vertices.
    limit : float
        Inclusive upper bound in network cost units.

    Returns
    -------
    tuple
        Distances and owning destination positions for every vertex.
    """
    distances = np.asarray(csgraph.dijkstra(matrix, indices=seeds, limit=limit, min_only=True))
    owner = np.full(matrix.shape[0], -1)
    order = np.argsort(labels)
    pointer, neighbors, weights = matrix.indptr, matrix.indices, matrix.data
    for i in order:
        seed, label = seeds[i], labels[i]
        if owner[seed] >= 0:
            continue
        owner[seed] = label
        stack = [seed]
        while stack:
            vertex = stack.pop()
            for arc in range(pointer[vertex], pointer[vertex + 1]):
                other = neighbors[arc]
                if (
                    owner[other] < 0
                    and distances[other] < np.inf
                    and distances[vertex] + weights[arc] == distances[other]
                ):
                    owner[other] = label
                    stack.append(other)
    return distances[:, None], owner[:, None]


def _two_labels(
    matrix: sparse.csr_matrix,
    points: _Attachments,
    edge_data: _EdgeData | None,
    limit: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep the closest two distinct input rows for leave-one-out queries.

    The second label is needed when the nearest row is the origin itself.

    Parameters
    ----------
    matrix : sparse.csr_matrix
        Directed search adjacency.
    points : _Attachments
        Connected destinations in input order.
    edge_data : _EdgeData or None
        Projected edge geometry metadata, or None for node attachment.
    limit : float
        Inclusive upper bound in network cost units.

    Returns
    -------
    tuple
        Costs and destination row positions in the closest two slots.
    """
    arrival = _endpoint_costs(points, edge_data, arrival=True)
    point_vertices = _vertices(points.feature, edge_data)
    rows, slots = np.nonzero(
        (points.feature[:, None] >= 0) & np.isfinite(arrival) & (arrival <= limit)
    )
    vertices, offsets = point_vertices[rows, slots], arrival[rows, slots]
    order = np.lexsort((rows, offsets, vertices))
    _, first, groups = np.unique(vertices[order], return_index=True, return_inverse=True)
    winner = order[first]
    other = order[rows[order] != rows[winner[groups]]]
    _, second = np.unique(vertices[other], return_index=True)
    keep = np.concatenate((winner, other[second]))
    queue = list(
        zip(offsets[keep].tolist(), rows[keep].tolist(), vertices[keep].tolist(), strict=True)
    )
    heapq.heapify(queue)
    owners = np.full((matrix.shape[0], 2), -1)
    distances = np.full(owners.shape, np.inf)
    while queue:
        cost, label, vertex = heapq.heappop(queue)
        first, second = owners[vertex]
        if second >= 0 or first == label:
            continue
        slot = int(first >= 0)
        owners[vertex, slot] = label
        distances[vertex, slot] = cost
        for arc in range(matrix.indptr[vertex], matrix.indptr[vertex + 1]):
            neighbor = matrix.indices[arc]
            length = cost + matrix.data[arc]
            if length <= limit and owners[neighbor, 1] < 0 and owners[neighbor, 0] != label:
                heapq.heappush(queue, (length, label, neighbor))
    return distances, owners


def _nearest_direct(
    start: _Attachments,
    end: _Attachments,
    edge_data: _EdgeData,
    within: bool,
    costs: np.ndarray,
    labels: np.ndarray,
) -> None:
    """
    Compare endpoint detours with the closest downstream and upstream projections.

    Update the selected labels and costs in place for trips along one physical edge.

    Parameters
    ----------
    start : _Attachments
        Connected origins in input order.
    end : _Attachments
        Connected destinations in input order.
    edge_data : _EdgeData
        Projected edge geometry metadata.
    within : bool
        Whether origins and destinations refer to the same input rows.
    costs : ndarray
        Current best cost per origin, updated in place.
    labels : ndarray
        Current best destination position per origin, updated in place.
    """
    source_order, source_edges, source_bounds = _edge_groups(start.feature)
    dest_order, dest_edges, dest_bounds = _edge_groups(end.feature)
    shared, origins, destinations = np.intersect1d(
        source_edges, dest_edges, assume_unique=True, return_indices=True
    )
    for edge, src_group, dst_group in zip(shared, origins, destinations, strict=True):
        rows = source_order[source_bounds[src_group] : source_bounds[src_group + 1]]
        cols = dest_order[dest_bounds[dst_group] : dest_bounds[dst_group + 1]]
        for sign, rate in zip((-1, 1), edge_data.rates[edge], strict=True):
            ordered = cols[np.lexsort((cols, sign * end.along[cols]))]
            offsets = sign * end.along[ordered]
            positions = np.searchsorted(offsets, sign * start.along[rows])
            first = np.r_[ordered, -1, -1]
            second = first[1:]
            if rate == 0:
                # Every downstream destination ties, including those farther along.
                first = np.full(len(ordered) + 1, -1)
                second = first.copy()
                for i in range(len(ordered) - 1, -1, -1):
                    a, b = first[i + 1], second[i + 1]
                    col = ordered[i]
                    if a < 0 or col < a:
                        first[i], second[i] = col, a
                    else:
                        first[i], second[i] = a, col if b < 0 else min(b, col)
            choice = first[positions]
            if within:
                choice = np.where(choice == rows, second[positions], choice)
            delta = sign * (end.along[choice] - start.along[rows])
            candidate = _fraction_cost(delta / edge_data.lengths[edge], rate)
            improve = (choice >= 0) & (
                (candidate < costs[rows]) | ((candidate == costs[rows]) & (choice < labels[rows]))
            )
            costs[rows[improve]] = candidate[improve]
            labels[rows[improve]] = choice[improve]


def _closest(
    matrix: sparse.csr_matrix,
    start: _Attachments,
    end: _Attachments,
    edge_data: _EdgeData | None,
    limit: float,
    within: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find the closest destination with one reverse multi-source traversal.

    Endpoint distances serve every origin attached to the same road vertices.

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix
        Directed adjacency matrix.
    start : _Attachments
        Connected origins in input order.
    end : _Attachments
        Connected destinations in input order.
    edge_data : _EdgeData or None
        Projected edge geometry metadata, or None for node attachment.
    limit : float
        Inclusive upper bound in network cost units.
    within : bool
        Whether origins and destinations refer to the same input rows.

    Returns
    -------
    tuple
        Origin positions, destination positions and minimum costs.
    """
    reverse = matrix.T.tocsr()
    valid = np.flatnonzero(end.feature >= 0)
    if within:
        distances, owners = _two_labels(reverse, end, edge_data, limit)
    elif edge_data is None:
        seeds, first = np.unique(end.feature[valid], return_index=True)
        distances, owners = _nearest_labels(reverse, seeds, valid[first], limit)
    else:
        # Only the cheapest destination offset at each road node can win.
        arrival = _endpoint_costs(end, edge_data, arrival=True)
        end_vertices = _vertices(end.feature, edge_data)
        rows, slots = np.nonzero(np.isfinite(arrival) & (end.feature[:, None] >= 0))
        vertices, offsets = end_vertices[rows, slots], arrival[rows, slots]
        order = np.lexsort((rows, offsets, vertices))
        seeds, first = np.unique(vertices[order], return_index=True)
        winners = order[first]
        n = reverse.shape[0]
        m = len(seeds)
        seeded = sparse.csr_matrix(
            (
                np.r_[reverse.data, offsets[winners]],
                np.r_[reverse.indices, seeds],
                np.r_[reverse.indptr, reverse.nnz + np.arange(1, m + 1)],
            ),
            shape=(n + m, n + m),
        )
        distances, owners = _nearest_labels(seeded, np.arange(n, n + m), rows[winners], limit)
    if not within and edge_data is None:
        labels = owners[start.feature, 0]
        rows = np.flatnonzero((start.feature >= 0) & (labels >= 0))
        return rows, labels[rows], distances[start.feature[rows], 0]
    start_vertices = _vertices(start.feature, edge_data)
    departure = _endpoint_costs(start, edge_data, arrival=False)
    rows = np.arange(len(start.feature))
    costs = np.full(len(rows), np.inf)
    labels = np.full(len(rows), -1)
    for slot in range(start_vertices.shape[1]):
        vertices = start_vertices[:, slot]
        for rank in range(owners.shape[1]):
            choice = owners[vertices, rank]
            candidate = distances[vertices, rank] + departure[:, slot]
            valid_choice = (choice >= 0) & ((choice != rows) if within else True)
            improve = valid_choice & (
                (candidate < costs) | ((candidate == costs) & (choice < labels))
            )
            costs[improve], labels[improve] = candidate[improve], choice[improve]
    if edge_data is not None:
        _nearest_direct(start, end, edge_data, within, costs, labels)
    keep = (start.feature >= 0) & np.isfinite(costs) & (costs <= limit)
    return rows[keep], labels[keep], costs[keep]


def _rank(
    blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray]], k: int | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Select at most k finite pairs per origin in stable cost order.

    Destination input order breaks ties at the selection boundary.

    Parameters
    ----------
    blocks : list[tuple[ndarray, ndarray, ndarray]]
        Blocks of origin positions, destination positions and costs.
    k : int | None
        Maximum destinations retained per origin.

    Returns
    -------
    tuple
        Selected row positions, column positions and costs.
    """
    rows, cols, costs = (np.concatenate(arrays) for arrays in zip(*blocks, strict=True))
    if k is None:
        return rows, cols, costs
    order = np.lexsort((cols, costs, rows))
    rows, cols, costs = rows[order], cols[order], costs[order]
    if len(rows):
        first = np.r_[True, rows[1:] != rows[:-1]]
        pos = np.arange(len(rows))
        keep = pos - np.maximum.accumulate(np.where(first, pos, 0)) < k
        rows, cols, costs = rows[keep], cols[keep], costs[keep]
    return rows, cols, costs


def _target_costs(
    source: _Attachments,
    rep: int,
    target: _Attachments,
    target_vertices: np.ndarray,
    edge_data: _EdgeData | None,
    lengths: np.ndarray,
    offsets: np.ndarray,
    reverse: bool,
) -> np.ndarray:
    """
    Combine distances through endpoints with direct trips on the same edge.

    Two points on one segment can reach each other before either endpoint.

    Parameters
    ----------
    source : _Attachments
        Point set on the search side.
    rep : int
        Representative row of the current search group.
    target : _Attachments
        Point set on the opposite side.
    target_vertices : ndarray
        Endpoint vertex positions for every target row.
    edge_data : _EdgeData or None
        Projected edge geometry metadata, or None for node attachment.
    lengths : ndarray
        Shortest path costs to road vertices.
    offsets : ndarray
        Directional endpoint costs for the target rows.
    reverse : bool
        Whether destinations are the search origins.

    Returns
    -------
    ndarray
        Minimum cost to each target in its input order.
    """
    costs = np.min(lengths[target_vertices] + offsets, axis=1)
    if edge_data is not None:
        edge = source.feature[rep]
        same = np.flatnonzero(target.feature == edge)
        delta = target.along[same] - source.along[rep]
        if reverse:
            delta = -delta
        direct = _fraction_cost(
            abs(delta) / edge_data.lengths[edge],
            edge_data.rates[edge, (delta >= 0).astype(int)],
        )
        costs[same] = np.minimum(costs[same], direct)
    return costs


def _pairs(
    matrix: sparse.csr_matrix,
    start: _Attachments,
    end: _Attachments,
    edge_data: _EdgeData | None,
    limit: float,
    k: int | None,
    within: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Batch searches from the side with fewer unique attachments.

    Keep search distances bounded by the existing row-batch memory budget.

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix
        Directed adjacency matrix.
    start : _Attachments
        Connected origins in input order.
    end : _Attachments
        Connected destinations in input order.
    edge_data : _EdgeData or None
        Projected edge geometry metadata, or None for node attachment.
    limit : float
        Inclusive upper bound in network cost units.
    k : int | None
        Maximum destinations retained per origin.
    within : bool
        Whether origins and destinations refer to the same input rows.

    Returns
    -------
    tuple
        Origin positions, destination positions and network costs.
    """
    empty = (np.empty(0, int), np.empty(0, int), np.empty(0))
    if not np.any(start.feature >= 0) or not np.any(end.feature >= 0):
        return empty
    if k == 1:
        return _closest(matrix, start, end, edge_data, limit, within)
    start_order, start_bounds = _attachment_groups(start.feature, start.along)
    end_order, end_bounds = (
        (start_order, start_bounds) if within else _attachment_groups(end.feature, end.along)
    )
    reverse = len(end_bounds) < len(start_bounds)
    source, target = (end, start) if reverse else (start, end)
    order, bounds = (end_order, end_bounds) if reverse else (start_order, start_bounds)
    representatives = order[bounds[:-1]]
    base = matrix.T.tocsr() if reverse else matrix
    source_vertices = _vertices(source.feature, edge_data)
    target_vertices = _vertices(target.feature, edge_data)
    source_offsets = _endpoint_costs(source, edge_data, arrival=reverse)
    offsets = _endpoint_costs(target, edge_data, arrival=not reverse)
    blocks = []
    pending = 0
    for batch in row_batches(len(representatives), base.shape[0] * source_vertices.shape[1]):
        batch_rows = representatives[batch]
        batch_vertices = source_vertices[batch_rows]
        batch_offsets = source_offsets[batch_rows]
        finite_slots = np.isfinite(batch_offsets)
        seeds, inverse = np.unique(batch_vertices[finite_slots], return_inverse=True)
        by_endpoint = np.vstack(
            [distances for _, distances, _ in distance_batches(base, seeds, limit)]
        )
        endpoint_rows = np.full(batch_vertices.shape, -1, dtype=int)
        endpoint_rows[finite_slots] = inverse
        for local, rep in enumerate(batch_rows):
            index = batch.start + local
            lengths = np.full(base.shape[0], np.inf)
            for slot in range(source_vertices.shape[1]):
                row_idx = endpoint_rows[local, slot]
                if row_idx >= 0:
                    lengths = np.minimum(lengths, by_endpoint[row_idx] + batch_offsets[local, slot])
            costs = _target_costs(
                source, rep, target, target_vertices, edge_data, lengths, offsets, reverse
            )
            hits = np.flatnonzero((target.feature >= 0) & np.isfinite(costs) & (costs <= limit))
            group = order[bounds[index] : bounds[index + 1]]
            if k is not None:
                if reverse:
                    group = np.sort(group)[: k + int(within)]
                else:
                    hits = hits[closest_positions(costs[hits], k + int(within))]
            if not len(hits):
                continue
            a, b = np.repeat(group, len(hits)), np.tile(hits, len(group))
            rows, cols = (b, a) if reverse else (a, b)
            keep = rows != cols if within and k is not None else slice(None)
            blocks.append((rows[keep], cols[keep], costs[b[keep]]))
            pending += len(blocks[-1][0])
            if reverse and k is not None and pending >= max(4096, len(start.labels) * k):
                blocks = [_rank(blocks, k)]
                pending = 0
    if not blocks:
        return empty
    return _rank(blocks, k)


def _validate_route_options(
    graph: _GraphInput | None,
    directed: bool | None,
    threshold: float | None,
    k: int | None,
    connect_to: Literal["node", "edge"],
    max_connection_distance: float,
) -> tuple[_GraphInput, bool, float]:
    """
    Validate public routing arguments and resolve traversal direction and cutoff.

    Ensures all query bounds are non-negative and prevents treating undirected
    NetworkX graphs as directed.

    Parameters
    ----------
    graph : _GraphInput or None
        Candidate spatial graph input.
    directed : bool or None
        Requested traversal direction override.
    threshold : float or None
        Optional inclusive cost cutoff.
    k : int or None
        Optional maximum destination count per origin.
    connect_to : {"node", "edge"}
        Attachment mode.
    max_connection_distance : float
        Maximum straight-line snapping distance.

    Returns
    -------
    tuple
        Validated graph, resolved direction flag, and numeric cutoff.
    """
    if graph is None:
        msg = "Supply a graph"
        raise ValueError(msg)
    if connect_to not in ("node", "edge"):
        msg = "connect_to must be 'node' or 'edge'"
        raise ValueError(msg)
    limit = math.inf if threshold is None else threshold
    if not limit >= 0:
        msg = "threshold must be non-negative"
        raise ValueError(msg)
    if k is not None and (isinstance(k, bool) or not isinstance(k, numbers.Integral) or k <= 0):
        msg = "k must be a positive integer"
        raise ValueError(msg)
    if not max_connection_distance >= 0:
        msg = "max_connection_distance must be non-negative"
        raise ValueError(msg)
    if directed is None:
        resolved_directed = isinstance(graph, nx.Graph) and graph.is_directed()
    elif directed and isinstance(graph, nx.Graph) and not graph.is_directed():
        msg = "directed=True requires a directed NetworkX graph"
        raise ValueError(msg)
    else:
        resolved_directed = directed
    return graph, resolved_directed, limit


def _route(
    origins: _Points,
    destinations: _Points | None,
    graph: _GraphInput | None,
    weight: str | None,
    directed: bool | None,
    threshold: float | None,
    k: int | None,
    connect_to: Literal["node", "edge"],
    max_connection_distance: float,
    connect_node_types: str | Sequence[str] | None,
) -> tuple[_Attachments, _Attachments, np.ndarray, np.ndarray, np.ndarray]:
    """
    Connect query points and compute their qualifying shortest-path costs.

    Builds the sparse adjacency and spatial index once per query, attaches query
    points within the snapping limit, and runs batched or reverse searches.

    Parameters
    ----------
    origins : GeoDataFrame or GeoSeries
        Origin points in graph CRS.
    destinations : GeoDataFrame or GeoSeries, optional
        Destination points in graph CRS. Omitting routes among origins.
    graph : _GraphInput or None
        Network supplying arcs and node coordinates.
    weight : str or None
        Edge cost attribute, or None for geometry length.
    directed : bool or None
        Traversal direction override, inferred from NetworkX when omitted.
    threshold : float or None
        Inclusive network cost cutoff.
    k : int or None
        Maximum number of destinations retained per origin.
    connect_to : {"node", "edge"}
        Feature used for point attachment, either node or edge.
    max_connection_distance : float
        Inclusive snapping distance limit in planar CRS units.
    connect_node_types : str or sequence of str, optional
        Eligible node layers for access and egress.

    Returns
    -------
    tuple
        Connected point sets, row positions, column positions and network costs.
    """
    valid_graph, is_directed, limit = _validate_route_options(
        graph, directed, threshold, k, connect_to, max_connection_distance
    )
    coords, layers, crs, u, v, raw_lines, values = _edge_arrays(valid_graph, weight)
    types = (
        [connect_node_types]
        if isinstance(connect_node_types, str)
        else list(connect_node_types)
        if connect_node_types is not None
        else None
    )
    allowed = np.ones(len(coords), bool) if types is None else np.isin(layers, types)
    nodes = np.flatnonzero(allowed)
    if weight is None or connect_to == "edge":
        missing = shapely.is_missing(raw_lines)
        if missing.any():
            raw_lines[missing] = shapely.linestrings(
                np.stack((coords[u[missing]], coords[v[missing]]), axis=1)
            )
    costs = _weights(shapely.length(raw_lines) if weight is None else values, weight)
    finite = np.isfinite(costs)
    a, b, data = u[finite], v[finite], costs[finite]
    if not is_directed:
        a, b, data = np.r_[a, b], np.r_[b, a], np.tile(data, 2)
    matrix = adjacency(a, b, data, len(coords))

    tree: cKDTree | shapely.STRtree | None = None
    lines = np.empty(0, object)
    edge_data: _EdgeData | None = None
    if connect_to == "node":
        if nodes.size:
            tree = cKDTree(coords[nodes])
    else:
        eligible = finite & allowed[u] & allowed[v]
        lines, ends, rates = _segments(
            coords, u[eligible], v[eligible], raw_lines[eligible], costs[eligible], is_directed
        )
        lengths = shapely.length(lines)
        edge_data = _EdgeData(ends=ends, rates=rates, lengths=np.where(lengths > 0, lengths, 1.0))
        tree = shapely.STRtree(lines)

    start = _snap_points(origins, crs, connect_to, max_connection_distance, nodes, lines, tree)
    end = (
        start
        if destinations is None
        else _snap_points(
            destinations, crs, connect_to, max_connection_distance, nodes, lines, tree
        )
    )
    rows, cols, trip_costs = _pairs(matrix, start, end, edge_data, limit, k, destinations is None)
    return start, end, rows, cols, trip_costs


def network_costs(
    origins: _Points,
    destinations: _Points | None = None,
    graph: _GraphInput | None = None,
    *,
    weight: str | None = None,
    directed: bool | None = None,
    threshold: float | None = None,
    k: int | None = None,
    connect_to: Literal["node", "edge"] = "node",
    max_connection_distance: float = math.inf,
    connect_node_types: str | Sequence[str] | None = None,
    include_connection_distance: bool = False,
) -> pd.DataFrame:
    """
    Compute shortest-path OD costs for reachable point pairs.

    Each result keeps the labels of its input points, including duplicate labels.

    Parameters
    ----------
    origins, destinations : GeoDataFrame or GeoSeries
        Points in graph CRS. Omitting destinations routes among origins. With k
        set, the same row is excluded.
    graph : GeoDataFrame, tuple, or NetworkX graph
        Edges, (nodes, edges), a pair of typed layer dictionaries, or a NetworkX
        graph.
    weight : str, optional
        Non-negative numeric edge attribute. Defaults to planar geometry length.
    directed : bool, optional
        Follow edge direction. Inferred for NetworkX and false for GeoDataFrames.
    threshold : float, optional
        Inclusive upper bound in edge weight units.
    k : int, optional
        Maximum destinations per origin. Ties follow destination input order.
    connect_to : {"node", "edge"}, default "node"
        Nearest node or projection onto a line with proportional traversal cost.
    max_connection_distance : float, default inf
        Maximum straight-line distance to each endpoint's network connection.
    connect_node_types : str or sequence of str, optional
        Eligible node layers for snapping. Both ends of an eligible edge must
        belong to these layers. All graph layers remain available for traversal.
    include_connection_distance : bool, default False
        Include the sum of access and egress distances in graph CRS units.

    Returns
    -------
    pandas.DataFrame
        Original origin and destination labels and network cost, ordered by
        origin position, cost and destination position.
    """
    start, end, rows, cols, costs = _route(
        origins,
        destinations,
        graph,
        weight,
        directed,
        threshold,
        k,
        connect_to,
        max_connection_distance,
        connect_node_types,
    )
    if k is None:
        order = np.lexsort((cols, costs, rows))
        rows, cols, costs = rows[order], cols[order], costs[order]
    result = pd.DataFrame(
        {
            "origin": start.labels.take(rows).to_numpy(),
            "destination": end.labels.take(cols).to_numpy(),
            "cost": costs,
        }
    )
    if include_connection_distance:
        result["connection_distance"] = start.distance[rows] + end.distance[cols]
    return result


def network_cost_matrix(
    origins: _Points,
    destinations: _Points | None = None,
    graph: _GraphInput | None = None,
    *,
    weight: str | None = None,
    directed: bool | None = None,
    threshold: float | None = None,
    k: int | None = None,
    connect_to: Literal["node", "edge"] = "node",
    max_connection_distance: float = math.inf,
    connect_node_types: str | Sequence[str] | None = None,
) -> sparse.csr_matrix:
    """
    Compute sparse OD costs with explicit entries for reachable zero-cost pairs.

    Row and column positions match the input order even when a point is disconnected.

    Parameters
    ----------
    origins, destinations : GeoDataFrame or GeoSeries
        Points in graph CRS. Omitting destinations routes among origins. With k
        set, the same row is excluded.
    graph : GeoDataFrame, tuple, or NetworkX graph
        Edges, (nodes, edges), a pair of typed layer dictionaries, or a NetworkX
        graph.
    weight : str, optional
        Non-negative numeric edge attribute. Defaults to planar geometry length.
    directed : bool, optional
        Follow edge direction. Inferred for NetworkX and false for GeoDataFrames.
    threshold : float, optional
        Inclusive upper bound in edge weight units.
    k : int, optional
        Maximum destinations per origin. Ties follow destination input order.
    connect_to : {"node", "edge"}, default "node"
        Nearest node or projection onto a line with proportional traversal cost.
    max_connection_distance : float, default inf
        Maximum straight-line distance to each endpoint's network connection.
    connect_node_types : str or sequence of str, optional
        Eligible node layers for snapping. Both ends of an eligible edge must
        belong to these layers. All graph layers remain available for traversal.

    Returns
    -------
    scipy.sparse.csr_matrix
        Origins by destinations in input order. Unreachable and excluded pairs
        have no stored entry.
    """
    start, end, rows, cols, costs = _route(
        origins,
        destinations,
        graph,
        weight,
        directed,
        threshold,
        k,
        connect_to,
        max_connection_distance,
        connect_node_types,
    )
    return sparse.csr_matrix((costs, (rows, cols)), shape=(len(start.labels), len(end.labels)))


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
    [0, 1]
    """
    is_graph_input = isinstance(graph, (nx.Graph, nx.MultiGraph))

    # Convert to NetworkX if needed
    nx_graph = graph if is_graph_input else gdf_to_nx(edges=graph)

    distances = compute_center_node_distances(
        nx_graph,
        center_point=center_point,
        edge_attr=edge_attr,
        cutoff=threshold,
    )
    if not distances and not node_positions(nx_graph):
        if is_graph_input:
            graph_type = type(graph)
            return graph_type()
        original_crs = graph.crs if hasattr(graph, "crs") else None
        return gpd.GeoDataFrame(geometry=[], crs=original_crs)

    subgraph = build_reachable_subgraph(nx_graph, distances, threshold)

    if is_graph_input:
        return subgraph

    # Convert back to GeoDataFrame
    return nx_to_gdf(subgraph, nodes=False, edges=True)
