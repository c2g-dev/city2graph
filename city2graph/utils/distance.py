"""Network shortest-path, distance, and reachability utilities."""

from __future__ import annotations

import heapq
import logging
import math
import numbers
import typing
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
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
from shapely.geometry import GeometryCollection
from shapely.geometry import LineString
from shapely.geometry import MultiPoint
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import Polygon

from city2graph._processing import row_batches
from city2graph.base import GeoDataProcessor

from .conversion import NxConverter
from .conversion import _edge_endpoints

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "create_isochrone",
    "filter_graph_by_distance",
    "network_cost_matrix",
    "network_costs",
]

logger = logging.getLogger("city2graph.utils")

_KNN_HULL_MAX_ATTEMPTS = 50
_HULL_NEIGHBOR_CACHE_BYTES = 8 * 1024 * 1024


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


def minimum_distances(
    matrix: sparse.csr_matrix, indices: Sequence[int] | np.ndarray, limit: float
) -> np.ndarray:
    """
    Run a bounded multi-source search.

    The direction of each CSR arc determines traversal direction.

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix
        Weighted directed adjacency in search order.
    indices : array-like
        Source vertex indices.
    limit : float
        Inclusive cost cutoff for exploration.

    Returns
    -------
    ndarray
        Minimum cost from any source for each vertex, infinity if unreachable.
    """
    return np.asarray(csgraph.dijkstra(matrix, indices=indices, limit=limit, min_only=True))


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
    # Keep the smallest weight of each parallel or reversed duplicate.
    order = np.lexsort((data, col, row))
    row, col, data = row[order], col[order], data[order]
    first = np.ones(len(row), dtype=bool)
    first[1:] = (row[1:] != row[:-1]) | (col[1:] != col[:-1])
    return sparse.csr_matrix((data[first], (row[first], col[first])), shape=(n, n))


def sparse_adjacency(
    graph: nx.Graph, weight: str, index: dict[Any, int]
) -> tuple[sparse.csr_matrix, bool, bool] | None:
    """
    Convert NetworkX weights for sparse shortest-path searches.

    Unsupported numeric types retain the NetworkX distance metric's fallback behaviour.

    Parameters
    ----------
    graph : networkx.Graph or spatial graph
        Network supplying arcs and node coordinates.
    weight : str or None
        Edge cost attribute, or None for geometry length.
    index : dict
        Mapping from node label to its matrix position.

    Returns
    -------
    tuple or None
        Adjacency, presence of infinite weights and integral weight status, or None.
    """
    rows: list[int] = []
    cols: list[int] = []
    weights: list[float] = []
    integers = 0
    exact_integer_limit = (2**53) // max(1, len(index) - 1)
    for u, v, value in graph.edges(data=weight, default=1):
        if isinstance(value, (int, np.integer)):
            if int(value) > exact_integer_limit:
                return None
            integers += 1
        # numpy.float64 subclasses float; other numpy floats do not.
        elif not isinstance(value, float):
            return None
        rows.append(index[u])
        cols.append(index[v])
        weights.append(float(value))
    if 0 < integers < len(weights):
        return None
    row = np.asarray(rows, dtype=np.int64)
    col = np.asarray(cols, dtype=np.int64)
    data = np.asarray(weights, dtype=np.float64)
    if not graph.is_directed():
        row, col = np.concatenate([row, col]), np.concatenate([col, row])
        data = np.concatenate([data, data])
    if np.any(np.isnan(data)):
        return None
    finite = data < np.inf
    has_infinite = not bool(finite.all())
    row, col, data = row[finite], col[finite], data[finite]
    matrix = adjacency(row, col, data, len(index))
    return matrix, has_infinite, integers > 0


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


type _GraphArrays = tuple[
    np.ndarray, np.ndarray, object, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]

type Points = gpd.GeoDataFrame | gpd.GeoSeries
type GraphInput = (
    nx.Graph
    | gpd.GeoDataFrame
    | tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
    | tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]]
)


@dataclass
class _Connections:
    """
    Store point attachments in input order.

    Feature indices refer to the spatial index owned by the prepared network.
    """

    network: _SpatialNetwork
    labels: pd.Index
    feature: np.ndarray
    along: np.ndarray
    distance: np.ndarray

    @cached_property
    def edge_groups(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Group connected rows by physical edge for direct interior trips.

        Only connected points take part in same-edge comparisons.

        Returns
        -------
        tuple
            Sorted positions, distinct edge IDs and group boundaries.
        """
        rows = np.flatnonzero(self.feature >= 0)
        rows = rows[np.argsort(self.feature[rows], kind="stable")]
        keys, first = np.unique(self.feature[rows], return_index=True)
        bounds = np.r_[first, len(rows)]
        return rows, keys, bounds

    @cached_property
    def groups(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Index distinct attachments and their original row positions.

        Coincident projections share one endpoint search.

        Returns
        -------
        tuple
            Sorted positions and group boundaries.
        """
        pos = np.flatnonzero(self.feature >= 0)
        order = pos[np.lexsort((self.along[pos], self.feature[pos]))]
        breaks = (
            np.flatnonzero((np.diff(self.feature[order]) != 0) | (np.diff(self.along[order]) != 0))
            + 1
        )
        bounds = np.r_[0, breaks, len(order)] if len(order) else np.array([0])
        return order, bounds

    @cached_property
    def vertices(self) -> np.ndarray:
        """
        Return the base nodes available to each connected point.

        Node connections have one column and edge connections have two.

        Returns
        -------
        ndarray
            Vertex positions for each connection.
        """
        if self.network.connect_to == "node":
            return self.feature[:, None]
        return np.asarray(self.network.ends[self.feature])

    @cached_property
    def departure(self) -> np.ndarray:
        """
        Return costs from the projected points to each endpoint.

        One-way restrictions apply to interior projections.

        Returns
        -------
        ndarray
            Outgoing network costs in endpoint order.
        """
        if self.network.connect_to == "node":
            return np.zeros((len(self.feature), 1))
        length = self.network.edge_lengths[self.feature]
        fractions = np.column_stack((self.along, length - self.along)) / length[:, None]
        return _fraction_cost(fractions, self.network.rates[self.feature])

    @cached_property
    def arrival(self) -> np.ndarray:
        """
        Return costs from each endpoint to the projected points.

        A directed edge can have different access and egress costs.

        Returns
        -------
        ndarray
            Incoming network costs in endpoint order.
        """
        if self.network.connect_to == "node":
            return self.departure
        length = self.network.edge_lengths[self.feature]
        fractions = np.column_stack((self.along, length - self.along)) / length[:, None]
        return _fraction_cost(fractions, self.network.rates[self.feature, ::-1])


def _position(data: dict[str, Any]) -> tuple[float, float]:
    """
    Read a node's planar coordinates.

    NetworkX pos and OSMnx x and y attributes are supported.

    Parameters
    ----------
    data : dict
        NetworkX node attributes.

    Returns
    -------
    tuple of float
        Planar x and y coordinates.
    """
    if "pos" in data:
        return data["pos"][0], data["pos"][1]
    try:
        return data["x"], data["y"]
    except KeyError as exc:
        msg = "Routing nodes require pos or x and y coordinates"
        raise ValueError(msg) from exc


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
    coord = np.asarray([_position(data) for _, data in graph.nodes(data=True)]).reshape(-1, 2)
    layers = np.asarray([data.get("node_type") for _, data in graph.nodes(data=True)])
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


def _edge_arrays(graph: GraphInput, weight: str | None) -> _GraphArrays:
    """
    Read geospatial columns in node-layer order and map edge endpoints to integer positions.

    Each node layer has its own index and a disjoint range of CSR positions.

    Parameters
    ----------
    graph : GraphInput
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


class _SpatialNetwork:
    """
    Internal weighted graph representation for OD shortest-path queries.

    Stores the CSR adjacency and spatial index used to attach query points.

    Parameters
    ----------
    graph : networkx.Graph or spatial graph
        Network supplying arcs and node coordinates.
    weight : str or None
        Edge cost attribute, or None for geometry length.
    directed : bool or None
        Traversal direction override, inferred from NetworkX when omitted.
    connect_to : str
        Feature used for point attachment, either node or edge.
    connect_node_types : str or sequence of str, optional
        Eligible node layers for access and egress.
    """

    def __init__(
        self,
        graph: GraphInput,
        *,
        weight: str | None = None,
        directed: bool | None = None,
        connect_to: Literal["node", "edge"] = "node",
        connect_node_types: str | Sequence[str] | None = None,
    ) -> None:
        """
        Build the adjacency and connection index.

        Only eligible features participate in snapping. All traversable arcs participate in routing.

        Parameters
        ----------
        graph : networkx.Graph or spatial graph
            Network supplying arcs and node coordinates.
        weight : str or None
            Edge cost attribute, or None for geometry length.
        directed : bool or None
            Traversal direction override, inferred from NetworkX when omitted.
        connect_to : str
            Feature used for point attachment, either node or edge.
        connect_node_types : str or sequence of str, optional
            Eligible node layers for access and egress.
        """
        if connect_to not in ("node", "edge"):
            msg = "connect_to must be 'node' or 'edge'"
            raise ValueError(msg)
        coords, layers, self.crs, u, v, lines, values = _edge_arrays(graph, weight)
        self.connect_to = connect_to
        if directed is None:
            directed = isinstance(graph, nx.Graph) and graph.is_directed()
        types = [connect_node_types] if isinstance(connect_node_types, str) else connect_node_types
        allowed = np.ones(len(coords), bool) if types is None else np.isin(layers, types)
        self.nodes = np.flatnonzero(allowed)
        if weight is None or connect_to == "edge":
            missing = shapely.is_missing(lines)
            if missing.any():
                lines[missing] = shapely.linestrings(
                    np.stack((coords[u[missing]], coords[v[missing]]), axis=1)
                )
        costs = _weights(shapely.length(lines) if weight is None else values, weight)
        finite = np.isfinite(costs)
        a, b, data = u[finite], v[finite], costs[finite]
        if not directed:
            a, b, data = np.r_[a, b], np.r_[b, a], np.tile(data, 2)
        self.matrix = adjacency(a, b, data, len(coords))
        self.tree = None
        self.lines = np.empty(0, object)
        if connect_to == "node":
            if self.nodes.size:
                self.tree = cKDTree(coords[self.nodes])
            return
        eligible = finite & allowed[u] & allowed[v]
        self.lines, self.ends, self.rates = _segments(
            coords, u[eligible], v[eligible], lines[eligible], costs[eligible], directed
        )
        lengths = shapely.length(self.lines)
        # All projections on a zero-length feature coincide at its start.
        self.edge_lengths = np.where(lengths > 0, lengths, 1.0)
        self.tree = shapely.STRtree(self.lines)

    @cached_property
    def _reverse(self) -> sparse.csr_matrix:
        """
        Transpose the base adjacency once.

        Multiple destination sets can reuse the reverse graph.

        Returns
        -------
        scipy.sparse.csr_matrix
            Adjacency for searches from destination vertices.
        """
        return self.matrix.T.tocsr()

    @cached_property
    def _reverse_with_seeds(self) -> sparse.csr_matrix:
        """
        Reserve one incoming seed per node for repeated weighted multi-source searches.

        Only the seed weights change between facility categories.

        Returns
        -------
        sparse.csr_matrix
            Reverse adjacency with an extra seed row per road vertex.
        """
        base = self._reverse
        n = base.shape[0]
        # The final n arcs have private seed tails and can never be reached from a road.
        return sparse.csr_matrix(
            (
                np.r_[base.data, np.zeros(n)],
                np.r_[base.indices, np.arange(n)],
                np.r_[base.indptr, base.nnz + np.arange(1, n + 1)],
            ),
            shape=(2 * n, 2 * n),
        )

    def connect(self, points: Points, max_connection_distance: float = math.inf) -> _Connections:
        """
        Connect points to the prepared graph.

        Missing points and points beyond the distance limit retain their input positions.

        Parameters
        ----------
        points : GeoSeries or GeoDataFrame
            Spatial points to connect to the network.
        max_connection_distance : float
            Inclusive snapping distance limit in planar CRS units.

        Returns
        -------
        _Connections
            Input labels, connected feature indices, offsets and distances.
        """
        if self.crs is not None and points.crs != self.crs:
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
        if self.connect_to == "node" and self.tree is not None:
            dist, nearest = self.tree.query(shapely.get_coordinates(geometry[indices]))
            feature[indices], distance[indices] = self.nodes[nearest], dist
        elif self.connect_to == "edge" and self.tree is not None and self.lines.size:
            pairs, dist = self.tree.query_nearest(geometry[indices], return_distance=True)
            # Resolve equidistant lines in their stable graph order.
            order = np.lexsort((pairs[1], pairs[0]))
            _, first = np.unique(pairs[0, order], return_index=True)
            hits = order[first]
            points_at, lines_at = indices[pairs[0, hits]], pairs[1, hits]
            feature[points_at], distance[points_at] = lines_at, dist[hits]
            along[points_at] = shapely.line_locate_point(self.lines[lines_at], geometry[points_at])
        feature[distance > max_connection_distance] = -1
        return _Connections(self, points.index.copy(), feature, along, distance)


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
    distances = minimum_distances(matrix, seeds, limit)
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
    matrix: sparse.csr_matrix, points: _Connections, limit: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep the closest two distinct input rows for leave-one-out queries.

    The second label is needed when the nearest row is the origin itself.

    Parameters
    ----------
    matrix : sparse.csr_matrix
        Directed search adjacency.
    points : _Connections
        Projected destinations and directional costs from road endpoints.
    limit : float
        Inclusive upper bound in network cost units.

    Returns
    -------
    tuple
        Costs and destination row positions in the closest two slots.
    """
    rows, slots = np.nonzero(
        (points.feature[:, None] >= 0) & np.isfinite(points.arrival) & (points.arrival <= limit)
    )
    vertices, offsets = points.vertices[rows, slots], points.arrival[rows, slots]
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
    start: _Connections, end: _Connections, within: bool, costs: np.ndarray, labels: np.ndarray
) -> None:
    """
    Compare endpoint detours with the closest downstream and upstream projections.

    Update the selected labels and costs in place for trips along one physical edge.

    Parameters
    ----------
    start : _Connections
        Connected origins in input order.
    end : _Connections
        Connected destinations in input order.
    within : bool
        Whether origins and destinations refer to the same input rows.
    costs : ndarray
        Current best cost per origin, updated in place.
    labels : ndarray
        Current best destination position per origin, updated in place.
    """
    source_order, source_edges, source_bounds = start.edge_groups
    dest_order, dest_edges, dest_bounds = end.edge_groups
    shared, origins, destinations = np.intersect1d(
        source_edges, dest_edges, assume_unique=True, return_indices=True
    )
    for edge, src_group, dst_group in zip(shared, origins, destinations, strict=True):
        rows = source_order[source_bounds[src_group] : source_bounds[src_group + 1]]
        cols = dest_order[dest_bounds[dst_group] : dest_bounds[dst_group + 1]]
        for sign, rate in zip((-1, 1), start.network.rates[edge], strict=True):
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
            candidate = _fraction_cost(delta / start.network.edge_lengths[edge], rate)
            improve = (choice >= 0) & (
                (candidate < costs[rows]) | ((candidate == costs[rows]) & (choice < labels[rows]))
            )
            costs[rows[improve]] = candidate[improve]
            labels[rows[improve]] = choice[improve]


def _closest(
    start: _Connections, end: _Connections, limit: float, within: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find the closest destination with one reverse multi-source traversal.

    Endpoint distances serve every origin attached to the same road vertices.

    Parameters
    ----------
    start : _Connections
        Connected origins in input order.
    end : _Connections
        Connected destinations in input order.
    limit : float
        Inclusive upper bound in network cost units.
    within : bool
        Whether origins and destinations refer to the same input rows.

    Returns
    -------
    tuple
        Origin positions, destination positions and minimum costs.
    """
    valid = np.flatnonzero(end.feature >= 0)
    if within:
        distances, owners = _two_labels(end.network._reverse, end, limit)
    elif end.network.connect_to == "node":
        seeds, first = np.unique(end.feature[valid], return_index=True)
        distances, owners = _nearest_labels(end.network._reverse, seeds, valid[first], limit)
    else:
        # Only the cheapest destination offset at each road node can win.
        rows, slots = np.nonzero(np.isfinite(end.arrival) & (end.feature[:, None] >= 0))
        vertices, offsets = end.vertices[rows, slots], end.arrival[rows, slots]
        order = np.lexsort((rows, offsets, vertices))
        seeds, first = np.unique(vertices[order], return_index=True)
        winners = order[first]
        matrix = end.network._reverse_with_seeds
        n = end.network.matrix.shape[0]
        matrix.data[matrix.nnz - n + seeds] = offsets[winners]
        distances, owners = _nearest_labels(matrix, n + seeds, rows[winners], limit)
    if not within and end.network.connect_to == "node":
        labels = owners[start.feature, 0]
        rows = np.flatnonzero((start.feature >= 0) & (labels >= 0))
        return rows, labels[rows], distances[start.feature[rows], 0]
    rows = np.arange(len(start.feature))
    costs = np.full(len(rows), np.inf)
    labels = np.full(len(rows), -1)
    for slot in range(start.vertices.shape[1]):
        vertices = start.vertices[:, slot]
        for rank in range(owners.shape[1]):
            choice = owners[vertices, rank]
            candidate = distances[vertices, rank] + start.departure[:, slot]
            valid_choice = (choice >= 0) & ((choice != rows) if within else True)
            improve = valid_choice & (
                (candidate < costs) | ((candidate == costs) & (choice < labels))
            )
            costs[improve], labels[improve] = candidate[improve], choice[improve]
    if end.network.connect_to == "edge":
        _nearest_direct(start, end, within, costs, labels)
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
    source: _Connections,
    rep: int,
    target: _Connections,
    lengths: np.ndarray,
    offsets: np.ndarray,
    reverse: bool,
) -> np.ndarray:
    """
    Combine distances through endpoints with direct trips on the same edge.

    Two points on one segment can reach each other before either endpoint.

    Parameters
    ----------
    source : _Connections
        Point set on the search side.
    rep : int
        Representative row of the current search group.
    target : _Connections
        Point set on the opposite side.
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
    costs = np.min(lengths[target.vertices] + offsets, axis=1)
    if source.network.connect_to == "edge":
        edge = source.feature[rep]
        same = np.flatnonzero(target.feature == edge)
        delta = target.along[same] - source.along[rep]
        if reverse:
            delta = -delta
        direct = _fraction_cost(
            abs(delta) / source.network.edge_lengths[edge],
            source.network.rates[edge, (delta >= 0).astype(int)],
        )
        costs[same] = np.minimum(costs[same], direct)
    return costs


def _pairs(
    start: _Connections, end: _Connections, limit: float, k: int | None, within: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Batch searches from the side with fewer unique attachments.

    Keep search distances bounded by the existing row-batch memory budget.

    Parameters
    ----------
    start : _Connections
        Connected origins in input order.
    end : _Connections
        Connected destinations in input order.
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
        return _closest(start, end, limit, within)
    reverse = len(end.groups[1]) < len(start.groups[1])
    source, target = (end, start) if reverse else (start, end)
    order, bounds = source.groups
    net = start.network
    representatives = order[bounds[:-1]]
    base = net._reverse if reverse else net.matrix
    source_offsets = source.arrival if reverse else source.departure
    offsets = target.departure if reverse else target.arrival
    blocks = []
    pending = 0
    for batch in row_batches(len(representatives), base.shape[0] * source.vertices.shape[1]):
        batch_rows = representatives[batch]
        seeds, inverse = np.unique(source.vertices[batch_rows], return_inverse=True)
        by_endpoint = np.vstack(
            [distances for _, distances, _ in distance_batches(base, seeds, limit)]
        )
        endpoint_rows = inverse.reshape(len(batch_rows), -1)
        for local, rep in enumerate(batch_rows):
            index = batch.start + local
            lengths = by_endpoint[endpoint_rows[local, 0]] + source_offsets[rep, 0]
            for slot in range(1, source.vertices.shape[1]):
                lengths = np.minimum(
                    lengths, by_endpoint[endpoint_rows[local, slot]] + source_offsets[rep, slot]
                )
            costs = _target_costs(source, rep, target, lengths, offsets, reverse)
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


def _route(
    origins: Points,
    destinations: Points | None,
    graph: GraphInput | None,
    weight: str | None,
    directed: bool | None,
    threshold: float | None,
    k: int | None,
    connect_to: Literal["node", "edge"],
    max_connection_distance: float,
    connect_node_types: str | Sequence[str] | None,
) -> tuple[_Connections, _Connections, np.ndarray, np.ndarray, np.ndarray]:
    """
    Connect query points and compute their qualifying shortest-path costs.

    Builds the internal sparse adjacency and spatial index once per query, attaches
    query points within the snapping limit, and runs batched or reverse searches.

    Parameters
    ----------
    origins : GeoDataFrame or GeoSeries
        Origin points in graph CRS.
    destinations : GeoDataFrame or GeoSeries, optional
        Destination points in graph CRS. Omitting routes among origins.
    graph : GraphInput or None
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
    network = _SpatialNetwork(
        graph,
        weight=weight,
        directed=directed,
        connect_to=connect_to,
        connect_node_types=connect_node_types,
    )
    start = network.connect(origins, max_connection_distance)
    end = start if destinations is None else network.connect(destinations, max_connection_distance)
    rows, cols, costs = _pairs(start, end, limit, k, destinations is None)
    return start, end, rows, cols, costs


def network_costs(
    origins: Points,
    destinations: Points | None = None,
    graph: GraphInput | None = None,
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
    origins: Points,
    destinations: Points | None = None,
    graph: GraphInput | None = None,
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

    points = _normalize_center_points(center_point)
    if not len(points):
        return {}
    _, nearest = tree.query([(p.x, p.y) for p in points])
    deduped_sources = [node_ids[i] for i in np.unique(nearest)]

    index = {node: i for i, node in enumerate(graph)}
    converted = sparse_adjacency(graph, edge_attr or "length", index)
    if (
        converted is not None
        and cutoff >= 0
        and np.all(converted[0].data >= 0)
        and not (math.isinf(cutoff) and converted[1])
    ):
        lengths = minimum_distances(converted[0], [index[node] for node in deduped_sources], cutoff)
        return {node: lengths[i] for node, i in index.items() if np.isfinite(lengths[i])}
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

    converter = NxConverter()
    return converter.gdf_to_nx(nodes=nodes, edges=edges)


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
