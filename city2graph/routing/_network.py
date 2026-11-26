"""Prepare spatial networks and reusable point attachments."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

import math
import numbers
from dataclasses import dataclass
from functools import cached_property
from threading import Lock
from typing import Any
from typing import Literal

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely
from scipy import sparse
from scipy.spatial import cKDTree

from city2graph._processing import edge_endpoints
from city2graph.base import GeoDataProcessor

from ._sparse import adjacency

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

    network: PreparedNetwork
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
        endpoints = edge_endpoints(frame)
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


class PreparedNetwork:
    """
    Prepare a weighted graph for repeated OD queries.

    Cache the CSR adjacency and spatial index. Call connect once to reuse the same origins.

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
        self._seed_lock = Lock()
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
        if not max_connection_distance >= 0:
            msg = "max_connection_distance must be non-negative"
            raise ValueError(msg)
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
