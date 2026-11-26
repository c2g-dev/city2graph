"""Network shortest-path and distance utilities."""

from __future__ import annotations

import heapq
import logging
import math
import numbers
import typing
from collections.abc import Sequence
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
from shapely.geometry import Point

from city2graph._processing import row_batches
from city2graph.base import GeoDataProcessor

from .conversion import NxConverter
from .conversion import _edge_endpoints

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "filter_graph_by_distance",
    "network_cost_matrix",
    "network_costs",
]

logger = logging.getLogger("city2graph.utils")


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


def _distance_batches(
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


def _adjacency(row: np.ndarray, col: np.ndarray, data: np.ndarray, n: int) -> sparse.csr_matrix:
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


def _closest_positions(costs: np.ndarray, k: int) -> np.ndarray:
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
    multipart = shapely.get_type_id(lines) == 5
    lines[multipart] = shapely.line_merge(lines[multipart])
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
    nodes, edges = (None, graph) if isinstance(graph, gpd.GeoDataFrame) else graph
    relations = edges if isinstance(edges, dict) else {(None, None, None): edges}
    frames = nodes if isinstance(nodes, dict) else {None: nodes} if nodes is not None else {}
    GeoDataProcessor.ensure_crs_consistency(
        *(f for f in (*frames.values(), *relations.values()) if f.active_geometry_name is not None)
    )
    crs = next(
        (
            f.crs
            for f in (*frames.values(), *relations.values())
            if f.active_geometry_name is not None and f.crs is not None
        ),
        None,
    )
    coords = []
    layer_ids = {}
    layer_offsets = {}
    types = []
    size = 0
    for layer, frame in frames.items():
        layer_ids[layer] = frame.index
        layer_offsets[layer] = size
        coords.append(np.column_stack((frame.geometry.x, frame.geometry.y)))
        types.extend([layer] * len(frame))
        size += len(frame)
    arc_sources, arc_targets, geometries, weights = [], [], [], []
    for (source, _, target), frame in relations.items():
        if frame.empty:
            continue
        lines = (
            frame.geometry.to_numpy(copy=True)
            if frame.active_geometry_name
            else np.full(len(frame), None)
        )
        endpoints = _edge_endpoints(frame)
        if not frames:
            # An edges-only network may supply explicit IDs at coincident crossings.
            ends = _endpoints(lines)
            labels = (
                pd.Index(endpoints[0]).append(pd.Index(endpoints[1]))
                if endpoints is not None
                else pd.MultiIndex.from_arrays(list(ends.T))
            )
            codes, unique = pd.factorize(labels, sort=False)
            first = np.unique(codes, return_index=True)[1]
            coords.append(ends[first])
            types.extend([None] * len(unique))
            u, v = np.split(codes, 2)
        elif endpoints is not None:
            u = layer_ids[source].get_indexer(endpoints[0])
            v = layer_ids[target].get_indexer(endpoints[1])
        else:
            ends = _endpoints(lines)
            a, b = np.split(ends, 2)
            lookup = dict(zip(frames, coords, strict=True))
            u = pd.MultiIndex.from_arrays(list(lookup[source].T)).get_indexer(
                pd.MultiIndex.from_arrays(list(a.T))
            )
            v = pd.MultiIndex.from_arrays(list(lookup[target].T)).get_indexer(
                pd.MultiIndex.from_arrays(list(b.T))
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
    continuous = (
        shapely.line_merge(lines, directed=True)
        if np.any(shapely.get_type_id(lines) == 5)
        else lines
    )
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
    if raw.dtype.kind not in "iuf" and not all(isinstance(v, numbers.Real) for v in raw):
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
    feature = np.full(len(points), -1)
    distance = np.full(len(points), np.inf)
    along = np.zeros(len(points))
    if connect_to == "node" and isinstance(tree, cKDTree):
        dist, nearest = tree.query(shapely.get_coordinates(geometry[indices]))
        feature[indices], distance[indices] = nodes[nearest], dist
    elif connect_to == "edge" and isinstance(tree, shapely.STRtree) and lines.size:
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
            if length <= limit and owners[neighbor, 1] < 0:
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
        seeds, inverse = np.unique(source_vertices[batch_rows], return_inverse=True)
        by_endpoint = np.vstack(
            [distances for _, distances, _ in _distance_batches(base, seeds, limit)]
        )
        endpoint_rows = inverse.reshape(len(batch_rows), -1)
        for local, rep in enumerate(batch_rows):
            index = batch.start + local
            lengths = by_endpoint[endpoint_rows[local, 0]] + source_offsets[rep, 0]
            for slot in range(1, source_vertices.shape[1]):
                lengths = np.minimum(
                    lengths, by_endpoint[endpoint_rows[local, slot]] + source_offsets[rep, slot]
                )
            costs = _target_costs(
                source, rep, target, target_vertices, edge_data, lengths, offsets, reverse
            )
            hits = np.flatnonzero((target.feature >= 0) & np.isfinite(costs) & (costs <= limit))
            group = order[bounds[index] : bounds[index + 1]]
            if k is not None:
                if reverse:
                    group = np.sort(group)[: k + int(within)]
                else:
                    hits = hits[_closest_positions(costs[hits], k + int(within))]
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

    coords, layers, crs, u, v, raw_lines, values = _edge_arrays(graph, weight)
    if directed is None:
        directed = isinstance(graph, nx.Graph) and graph.is_directed()
    types = [connect_node_types] if isinstance(connect_node_types, str) else connect_node_types
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
    if not directed:
        a, b, data = np.r_[a, b], np.r_[b, a], np.tile(data, 2)
    matrix = _adjacency(a, b, data, len(coords))

    tree: cKDTree | shapely.STRtree | None = None
    lines = np.empty(0, object)
    edge_data: _EdgeData | None = None
    if connect_to == "node":
        if nodes.size:
            tree = cKDTree(coords[nodes])
    else:
        eligible = finite & allowed[u] & allowed[v]
        lines, ends, rates = _segments(
            coords, u[eligible], v[eligible], raw_lines[eligible], costs[eligible], directed
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

    points = _normalize_center_points(center_point)
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
