"""
Proximity-Based Graph Generation Module.

This module provides comprehensive functionality for generating graph networks based
on spatial proximity relationships between geographic features. It implements several
classical proximity models commonly used in spatial network analysis and geographic
information systems. The module is particularly useful for constructing heterogeneous
graphs from multiple domains of geospatial relations, enabling complex spatial analysis
across different feature types and scales.
"""

# Future annotations for type hints
from __future__ import annotations

# Standard library imports
import logging
from dataclasses import dataclass
from itertools import combinations
from numbers import Real
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import cast

if TYPE_CHECKING:  # Only needed for typing annotations
    from collections.abc import Iterable

# Third-party imports
import geopandas as gpd
import libpysal
import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.spatial import Delaunay
from scipy.spatial import distance as sdist
from shapely.geometry import LineString
from sklearn.neighbors import NearestNeighbors

# Local imports
from .utils import gdf_to_nx
from .utils import nx_to_gdf
from .utils import validate_gdf

# Module logger configuration
logger = logging.getLogger(__name__)

__all__ = [
    "bridge_nodes",
    "contiguity_graph",
    "delaunay_graph",
    "euclidean_minimum_spanning_tree",
    "fixed_radius_graph",
    "gabriel_graph",
    "group_nodes",
    "knn_graph",
    "relative_neighborhood_graph",
    "waxman_graph",
]

# Simple type alias for readability
EdgePair = tuple[Any, Any]

AUTO_NETWORK_LENGTH_ATTR = "__c2g_edge_length"


def _resolve_network_weight_attribute(
    graph: nx.Graph,
    pos: dict[Any, tuple[float, float]],
    preferred_attr: str | None,
) -> str:
    """
    Return edge weight attribute, computing lengths when unspecified.

    This helper function determines which edge attribute to use for network
    distance calculations. If a preferred attribute is specified, it validates
    that all edges have numeric values for that attribute. Otherwise, it
    computes edge lengths automatically from geometries or node positions.

    Parameters
    ----------
    graph : networkx.Graph
        The network graph containing edges.
    pos : dict[Any, tuple[float, float]]
        Mapping from node IDs to (x, y) coordinate tuples.
    preferred_attr : str or None
        Preferred edge attribute name to use as weight. If None, lengths
        are computed automatically.

    Returns
    -------
    str
        Name of the edge attribute to use for network weights.

    Raises
    ------
    ValueError
        If preferred_attr is specified but some edges are missing this
        attribute or have non-numeric values.
    """
    if preferred_attr:
        missing = [
            (u, v)
            for u, v, data in graph.edges(data=True)
            if not isinstance(data.get(preferred_attr), Real)
        ]
        if missing:
            sample = ", ".join(map(str, missing[:3]))
            msg = (
                f"Edges missing numeric '{preferred_attr}' attribute required for network weights;"
                f" examples: {sample}"
            )
            raise ValueError(msg)
        return preferred_attr

    for u, v, data in graph.edges(data=True):
        existing = data.get(AUTO_NETWORK_LENGTH_ATTR)
        if isinstance(existing, Real):
            continue
        geom = data.get("geometry")
        if geom is not None and hasattr(geom, "length"):
            candidate = float(getattr(geom, "length", 0.0))
        elif u in pos and v in pos:
            ux, uy = pos[u]
            vx, vy = pos[v]
            candidate = float(np.hypot(vx - ux, vy - uy))
        else:
            candidate = 0.0
        data[AUTO_NETWORK_LENGTH_ATTR] = candidate

    return AUTO_NETWORK_LENGTH_ATTR


class DistanceMetric:
    """
    Encapsulate distance-metric normalisation and distance-matrix creation.

    Instances provide a unified interface that hides the implementation detail
    behind Euclidean, Manhattan, or network-based measurements. The class
    normalizes metric names, validates that required resources (such as a
    network GeoDataFrame) are available when needed, and provides methods
    for computing distance matrices.

    Parameters
    ----------
    metric : str
        Raw metric name (``euclidean``, ``manhattan``, or ``network``).
    network_gdf : geopandas.GeoDataFrame, optional
        Auxiliary network edges required when ``metric`` equals ``network``.
    network_weight : str, optional
        Edge attribute present in ``network_gdf`` to use as the shortest-path weight.
        Defaults to automatically computed geometry lengths when omitted.
    """

    def __init__(
        self,
        metric: str,
        network_gdf: gpd.GeoDataFrame | None = None,
        network_weight: str | None = None,
    ) -> None:
        """
        Initialize the distance metric configuration.

        Normalizes the metric name and validates that any required resources
        (such as a network GeoDataFrame) are provided.

        Parameters
        ----------
        metric : str
            Raw metric name (``euclidean``, ``manhattan``, or ``network``).
        network_gdf : geopandas.GeoDataFrame, optional
            Auxiliary network edges required when ``metric`` equals ``network``.
        network_weight : str, optional
            Edge attribute inside ``network_gdf`` that supplies weights for network
            distances. When omitted, lengths are derived from edge geometries.
        """
        self.name = self._normalize_metric(metric)
        if self.name not in {"euclidean", "manhattan", "network"}:
            msg = f"Unknown distance metric: {metric}"
            raise ValueError(msg)
        self.network_gdf = network_gdf
        self.network_weight = network_weight
        self._network_cache: (
            tuple[
                nx.Graph,
                dict[Any, tuple[float, float]],
                npt.NDArray[np.floating],
                list[Any],
                str,
            ]
            | None
        ) = None

    def validate(self, crs: object) -> None:
        """
        Validate metric requirements against a CRS.

        The check guarantees that network-based distances only operate when a
        compatible network GeoDataFrame and matching CRS are present.

        Parameters
        ----------
        crs : object
            The Coordinate Reference System to validate against.

        Raises
        ------
        ValueError
            If the metric is 'network' and the CRS does not match the network's CRS.
        """
        if self.name == "network":
            if self.network_gdf is None:
                msg = "network_gdf is required for network distance metric"
                raise ValueError(msg)
            if crs and self.network_gdf.crs != crs:
                msg = f"CRS mismatch between inputs and network: {crs} != {self.network_gdf.crs}"
                raise ValueError(msg)

    def matrix(self, coords: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """
        Compute a distance matrix that matches the configured metric.

        This method dispatches to the appropriate helper and therefore hides
        metric-specific implementation details from callers.

        Parameters
        ----------
        coords : npt.NDArray[np.floating]
            Array of coordinates.

        Returns
        -------
        npt.NDArray[np.floating]
            Distance matrix.

        Raises
        ------
        ValueError
            If the metric is unknown.
        """
        if self.name == "manhattan":
            return self._manhattan_dm(coords)
        if self.name == "network":
            return self._network_dm(coords)
        return self._euclidean_dm(coords)

    def _get_network_support(
        self,
    ) -> tuple[nx.Graph, dict[Any, tuple[float, float]], npt.NDArray[np.floating], list[Any], str]:
        """
        Return cached NetworkX graph plus positional helpers.

        This method builds and caches the network support infrastructure needed
        for network-based distance calculations. It converts the network GeoDataFrame
        to a NetworkX graph, extracts node positions, and resolves the weight attribute
        to use for shortest path calculations.

        Returns
        -------
        tuple[networkx.Graph, dict[Any, tuple[float, float]], numpy.ndarray, list[Any], str]
            A 5-tuple containing:
            - NetworkX graph representation of the network
            - Dictionary mapping node IDs to (x, y) positions
            - NumPy array of network node coordinates
            - List of network node IDs
            - str Weight attribute name for shortest path calculations

        Raises
        ------
        ValueError
            If network_gdf is None or if the network lacks valid node positions.
        """
        if self.network_gdf is None:
            msg = "network_gdf is required for network distance metric"
            raise ValueError(msg)

        if self._network_cache is None:
            net_nx = gdf_to_nx(edges=self.network_gdf)
            pos = nx.get_node_attributes(net_nx, "pos")
            if not pos:
                msg = "network_gdf must include geometries with valid node positions"
                raise ValueError(msg)
            weight_attr = _resolve_network_weight_attribute(net_nx, pos, self.network_weight)
            net_coords = np.asarray(list(pos.values()), dtype=float)
            net_ids = list(pos.keys())
            self._network_cache = (net_nx, pos, net_coords, net_ids, weight_attr)

        return self._network_cache

    @staticmethod
    def _normalize_metric(metric: object) -> str:
        """
        Normalize a user-provided distance metric label.

        The helper accepts loosely typed inputs and falls back to the default
        Euclidean metric whenever the argument is not a non-empty string.

        Parameters
        ----------
        metric : object
            Raw metric identifier supplied by the caller.

        Returns
        -------
        str
            Lower-case metric name compatible with :class:`DistanceMetric`.
        """
        if not isinstance(metric, str) or not metric:
            return "euclidean"
        return metric.lower()

    @staticmethod
    def _euclidean_dm(coords: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """
        Compute a dense pairwise Euclidean distance matrix.

        This routine is a thin wrapper around :func:`scipy.spatial.distance.pdist`
        that guarantees a NumPy matrix suitable for subsequent graph construction.

        Parameters
        ----------
        coords : numpy.typing.NDArray[np.floating]
            Array of point coordinates shaped as (n_samples, 2).

        Returns
        -------
        numpy.typing.NDArray[np.floating]
            Symmetric matrix of Euclidean distances.
        """
        return cast("npt.NDArray[np.floating]", sdist.squareform(sdist.pdist(coords)))

    @staticmethod
    def _manhattan_dm(coords: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """
        Compute a dense pairwise Manhattan (city-block) distance matrix.

        The resulting matrix matches the shape of the Euclidean counterpart but
        uses the L1 norm, which is often preferred in rectilinear grids.

        Parameters
        ----------
        coords : numpy.typing.NDArray[np.floating]
            Array of point coordinates shaped as (n_samples, 2).

        Returns
        -------
        numpy.typing.NDArray[np.floating]
            Symmetric matrix of Manhattan distances.
        """
        return cast(
            "npt.NDArray[np.floating]", sdist.squareform(sdist.pdist(coords, metric="cityblock"))
        )

    def _network_dm(self, coords: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """
        Compute a network-based distance matrix using shortest paths.

        The routine projects sample points onto the auxiliary network, runs a
        Dijkstra traversal between nearest nodes, and backfills the distances.

        Parameters
        ----------
        coords : npt.NDArray[np.floating]
            Array of coordinates.

        Returns
        -------
        npt.NDArray[np.floating]
            Network distance matrix.
        """
        net_nx, _pos, net_coords, net_ids, weight_attr = self._get_network_support()

        # Map sample points to nearest network nodes
        nn = NearestNeighbors(n_neighbors=1).fit(net_coords)
        _, idx = nn.kneighbors(coords)
        nearest = [net_ids[i[0]] for i in idx]

        # Pre-allocate distance matrix
        n = len(coords)
        dm: npt.NDArray[np.floating] = np.full((n, n), np.inf)
        np.fill_diagonal(dm, 0)

        # Calculate all-pairs shortest paths
        use_weight = weight_attr

        for i in range(n):
            lengths = nx.single_source_dijkstra_path_length(net_nx, nearest[i], weight=use_weight)
            for j in range(i + 1, n):
                dist = lengths.get(nearest[j], np.inf)
                dm[i, j] = dm[j, i] = dist
        return dm


class GraphBuilder:
    """
    Helper that consolidates GeoDataFrame data into NetworkX graphs.

    The builder centralises node preparation, distance computation, and edge
    enrichment so that the public APIs can focus on graph semantics.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing nodes.
    metric : DistanceMetric
        Distance metric used for weights and validation.
    directed : bool, default False
        Whether the resulting graph should be directed.
    """

    def __init__(
        self, gdf: gpd.GeoDataFrame, metric: DistanceMetric, directed: bool = False
    ) -> None:
        """
        Initialize the GraphBuilder with the core graph context.

        The object stores references to the source GeoDataFrame, preferred
        metric, and graph directionality so repeated operations remain cheap.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            GeoDataFrame containing nodes.
        metric : DistanceMetric
            Distance metric to use.
        directed : bool, default False
            Whether the graph is directed.
        """
        self.gdf = gdf
        self.metric = metric
        self.directed = directed
        self.G = nx.DiGraph() if directed else nx.Graph()
        self.coords: npt.NDArray[np.floating] | None = None
        self.node_ids: list[Any] | None = None
        self.dm: npt.NDArray[np.floating] | None = None

    def prepare_nodes(self) -> None:
        """
        Add nodes from GeoDataFrame to the graph.

        This method extracts coordinates and attributes from the GeoDataFrame
        and adds them to the NetworkX graph.
        """
        validate_gdf(nodes_gdf=self.gdf)
        centroids = self.gdf.geometry.centroid
        self.coords = np.column_stack([centroids.x, centroids.y])
        self.node_ids = list(self.gdf.index)

        # Bulk-add nodes
        attrs_list = self.gdf.to_dict("records")
        self.G.add_nodes_from(
            (
                node_id,
                {"pos": (float(x), float(y)), **attrs},
            )
            for node_id, attrs, (x, y) in zip(self.node_ids, attrs_list, self.coords, strict=False)
        )
        self.G.graph["crs"] = self.gdf.crs

        # Validate metric against CRS
        self.metric.validate(self.gdf.crs)

    def compute_distance_matrix(self) -> None:
        """
        Compute and cache distance matrix.

        This method computes the pairwise distance matrix between all nodes
        using the configured distance metric.
        """
        assert self.coords is not None

        self.dm = self.metric.matrix(self.coords)

    def add_edges(self, edges: list[EdgePair] | set[EdgePair]) -> None:
        """
        Add edges with weights and geometries.

        Any missing node preparation work is completed automatically so that
        callers can provide lightweight edge tuples without additional context.

        Parameters
        ----------
        edges : list[EdgePair] | set[EdgePair]
            Collection of edges to add, where each edge is a tuple of (u, v).
        """
        if not edges:
            return

        edge_list = list(edges)
        self.G.add_edges_from(edge_list)

        assert self.coords is not None
        assert self.node_ids is not None

        weights, geoms = self._compute_edge_data(edge_list)
        nx.set_edge_attributes(self.G, dict(zip(edge_list, weights, strict=False)), "weight")
        nx.set_edge_attributes(self.G, dict(zip(edge_list, geoms, strict=False)), "geometry")

    def _compute_edge_data(self, edges: list[EdgePair]) -> tuple[list[float], list[LineString]]:
        """
        Compute weights and geometries for edges.

        The method inspects the active distance metric to derive both scalar
        weights and representative geometries for each candidate edge.

        Parameters
        ----------
        edges : list[EdgePair]
            List of edges to compute data for.

        Returns
        -------
        tuple[list[float], list[LineString]]
            Tuple containing lists of weights and geometries.
        """
        assert self.node_ids is not None
        assert self.coords is not None

        idx_map = {n: i for i, n in enumerate(self.node_ids)}
        u_indices = [idx_map[u] for u, v in edges]
        v_indices = [idx_map[v] for u, v in edges]
        p1 = self.coords[u_indices]
        p2 = self.coords[v_indices]

        weights: list[float] = []
        geoms: list[LineString] = []

        if self.metric.name == "network":
            return self._compute_network_edge_data(edges, idx_map)

        if self.metric.name == "manhattan":
            for i in range(len(edges)):
                w = abs(p1[i][0] - p2[i][0]) + abs(p1[i][1] - p2[i][1])
                weights.append(w)
                geoms.append(
                    LineString([(p1[i][0], p1[i][1]), (p2[i][0], p1[i][1]), (p2[i][0], p2[i][1])])
                )
        else:  # Euclidean
            if self.dm is not None:
                for u, v in edges:
                    weights.append(self.dm[idx_map[u], idx_map[v]])
            else:
                d = np.hypot(p1[:, 0] - p2[:, 0], p1[:, 1] - p2[:, 1])
                weights = d.tolist()
            geoms = [LineString([pt1, pt2]) for pt1, pt2 in zip(p1, p2, strict=False)]

        return weights, geoms

    def _compute_network_edge_data(
        self, edges: list[EdgePair], idx_map: dict[Any, int]
    ) -> tuple[list[float], list[LineString]]:
        """
        Compute weights and geometries for network edges.

        When network distances are enabled, each graph edge must follow the
        auxiliary network, so we map sample points to their closest network
        nodes and trace the shortest paths.

        Parameters
        ----------
        edges : list[EdgePair]
            List of edges to compute data for.
        idx_map : dict[Any, int]
            Mapping from node ID to index in coords array.

        Returns
        -------
        tuple[list[float], list[LineString]]
            Tuple containing lists of weights and geometries.
        """
        assert self.metric.network_gdf is not None
        assert self.coords is not None
        assert self.node_ids is not None

        net_nx, pos, net_coords, net_ids_list, weight_attr = self.metric._get_network_support()

        nn = NearestNeighbors(n_neighbors=1).fit(net_coords)
        _, idxs = nn.kneighbors(self.coords)
        nearest = {self.node_ids[i]: net_ids_list[j[0]] for i, j in enumerate(idxs)}

        edges_by_src: dict[Any, list[int]] = {}
        for i, (u, _v) in enumerate(edges):
            edges_by_src.setdefault(u, []).append(i)

        use_weight = weight_attr
        weights = [0.0] * len(edges)
        geoms = [LineString()] * len(edges)

        for u, indices in edges_by_src.items():
            src_nn = nearest[u]
            dists, paths = nx.single_source_dijkstra(net_nx, src_nn, weight=use_weight)

            for i in indices:
                _, v = edges[i]
                dst_nn = nearest[v]

                if self.dm is not None:
                    weights[i] = self.dm[idx_map[u], idx_map[v]]
                else:
                    weights[i] = dists.get(dst_nn, float("inf"))

                path_nodes = paths.get(dst_nn)
                if path_nodes and len(path_nodes) >= 2:
                    geoms[i] = LineString([pos[p] for p in path_nodes])
                else:
                    geoms[i] = LineString([self.coords[idx_map[u]], self.coords[idx_map[v]]])

        return weights, geoms

    def to_output(self, as_nx: bool) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph:
        """
        Return graph in requested format.

        Final consumers can request either GeoDataFrame outputs or the raw
        NetworkX object depending on their downstream workflow needs.

        Parameters
        ----------
        as_nx : bool
            If True, return a NetworkX graph. Otherwise, return GeoDataFrames.

        Returns
        -------
        tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph
            The generated graph.
        """
        return self.G if as_nx else nx_to_gdf(self.G, nodes=True, edges=True)


# ============================================================================
# GRAPH GENERATORS
# ============================================================================


def knn_graph(
    gdf: gpd.GeoDataFrame,
    k: int = 5,
    distance_metric: str = "euclidean",
    network_gdf: gpd.GeoDataFrame | None = None,
    network_weight: str | None = None,
    *,
    target_gdf: gpd.GeoDataFrame | None = None,
    as_nx: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph:
    r"""
    Generate a k-nearest neighbour graph from a GeoDataFrame of points.

    This function constructs a graph where each node is connected to its k nearest neighbors
    based on the specified distance metric. The resulting graph captures local spatial
    relationships and is commonly used in spatial analysis, clustering, and network topology
    studies.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing the points (nodes) for the graph. The index of this
        GeoDataFrame will be used as node IDs.
    k : int, default 5
        The number of nearest neighbors to connect to each node. Must be positive and
        less than the total number of nodes.
    distance_metric : str, default "euclidean"
        The distance metric to use for calculating nearest neighbors. Options are:
        - "euclidean": Straight-line distance
        - "manhattan": City-block distance (L1 norm)
        - "network": Shortest path distance along a network
    network_gdf : geopandas.GeoDataFrame, optional
        A GeoDataFrame representing a network (e.g., roads, paths) to use for "network"
        distance calculations. Required if `distance_metric` is "network".
    network_weight : str, optional
        Edge attribute name in `network_gdf` to use as the network distance weight.
        When omitted, weights default to the geometry length of each network edge.
    target_gdf : geopandas.GeoDataFrame, optional
        If provided, creates a directed graph where edges connect nodes from `gdf` to
        their k nearest neighbors in `target_gdf`. If None, creates an undirected graph
        within `gdf` itself.
    as_nx : bool, default False
        If True, returns a NetworkX graph object. Otherwise, returns a tuple of
        GeoDataFrames (nodes, edges).

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame] or networkx.Graph
        If `as_nx` is False, returns a tuple of GeoDataFrames:
        - nodes_gdf: GeoDataFrame of nodes with spatial and attribute information
        - edges_gdf: GeoDataFrame of edges with 'weight' and 'geometry' attributes
        If `as_nx` is True, returns a NetworkX graph object with spatial attributes.

    Raises
    ------
    ValueError
        If `distance_metric` is "network" but `network_gdf` is not provided.
        If `k` is greater than or equal to the number of available nodes.
    """
    # Handle directed variant
    if target_gdf is not None:
        return _directed_graph(
            src_gdf=gdf,
            dst_gdf=target_gdf,
            distance_metric=distance_metric,
            method="knn",
            param=k,
            as_nx=as_nx,
            network_gdf=network_gdf,
            network_weight=network_weight,
        )

    metric = DistanceMetric(distance_metric, network_gdf, network_weight)
    builder = GraphBuilder(gdf, metric)
    builder.prepare_nodes()
    assert builder.coords is not None
    assert builder.node_ids is not None

    if len(builder.coords) <= 1 or k <= 0:
        return builder.to_output(as_nx)

    if metric.name == "network":
        builder.compute_distance_matrix()
        assert builder.dm is not None
        # Use argsort to find nearest neighbors in distance matrix
        # Skip the first one (self)
        order = np.argsort(builder.dm, axis=1)[:, 1 : k + 1]
        edges = [
            (builder.node_ids[i], builder.node_ids[j])
            for i in range(len(builder.node_ids))
            for j in order[i]
            if builder.dm[i, j] < np.inf
        ]
    else:
        nn_metric = "cityblock" if metric.name == "manhattan" else "euclidean"
        n_neigh = min(k + 1, len(builder.coords))
        nn = NearestNeighbors(n_neighbors=n_neigh, metric=nn_metric).fit(builder.coords)
        _, idxs = nn.kneighbors(builder.coords)
        edges = [
            (builder.node_ids[i], builder.node_ids[j])
            for i, neigh in enumerate(idxs)
            for j in neigh[1:]
        ]

    builder.add_edges(edges)
    return builder.to_output(as_nx)


def delaunay_graph(
    gdf: gpd.GeoDataFrame,
    distance_metric: str = "euclidean",
    network_gdf: gpd.GeoDataFrame | None = None,
    network_weight: str | None = None,
    *,
    as_nx: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph:
    r"""
    Generate a Delaunay triangulation graph from a GeoDataFrame of points.

    This function constructs a graph based on the Delaunay triangulation of the
    input points. Each edge in the graph corresponds to an edge in the Delaunay
    triangulation.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing the points (nodes) for the graph. The index of this
        GeoDataFrame will be used as node IDs.
    distance_metric : str, default "euclidean"
        The distance metric to use for calculating edge weights. Options are
        "euclidean", "manhattan", or "network".
    network_gdf : geopandas.GeoDataFrame, optional
        A GeoDataFrame representing a network (e.g., roads) to use for "network"
        distance calculations. Required if `distance_metric` is "network".
    network_weight : str, optional
        Edge attribute name in `network_gdf` used as the path weight when
        `distance_metric` is ``"network"``. Defaults to geometry length.
    as_nx : bool, default False
        If True, returns a NetworkX graph object. Otherwise, returns a tuple of
        GeoDataFrames (nodes, edges).

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame] or networkx.Graph
        If `as_nx` is False, returns a tuple of GeoDataFrames:
        - nodes_gdf: GeoDataFrame of nodes with spatial and attribute information
        - edges_gdf: GeoDataFrame of edges with 'weight' and 'geometry' attributes
        If `as_nx` is True, returns a NetworkX graph object with spatial attributes.

    Raises
    ------
    ValueError
        If `distance_metric` is "network" but `network_gdf` is not provided.

    See Also
    --------
    knn_graph : Generate a k-nearest neighbour graph.
    fixed_radius_graph : Generate a fixed-radius graph.
    waxman_graph : Generate a probabilistic Waxman graph.

    Notes
    -----
    - Node IDs are preserved from the input GeoDataFrame's index.
    - Edge weights represent the distance between connected nodes.
    - Edge geometries are LineStrings connecting the centroids of the nodes.
    - If the input gdf has fewer than 3 points, an empty graph will be returned as Delaunay
      triangulation requires at least 3 non-collinear points.

    References
    ----------
    .. [1] Lee, D. T., & Schachter, B. J. (1980). Two algorithms for constructing a
       Delaunay triangulation. International Journal of Computer & Information Sciences, 9(3),
       219-242. https://doi.org/10.1007/BF00977785
    """
    metric = DistanceMetric(distance_metric, network_gdf, network_weight)
    builder = GraphBuilder(gdf, metric)
    builder.prepare_nodes()
    assert builder.coords is not None
    assert builder.node_ids is not None

    if len(builder.coords) < 3:
        return builder.to_output(as_nx)

    tri = Delaunay(builder.coords)
    edges = {
        (builder.node_ids[i], builder.node_ids[j])
        for simplex in tri.simplices
        for i, j in combinations(simplex, 2)
    }

    builder.add_edges(edges)
    return builder.to_output(as_nx)


def gabriel_graph(
    gdf: gpd.GeoDataFrame,
    distance_metric: str = "euclidean",
    network_gdf: gpd.GeoDataFrame | None = None,
    network_weight: str | None = None,
    *,
    as_nx: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph:
    r"""
    Generate a Gabriel graph from a GeoDataFrame of points.

    In a Gabriel graph two nodes u and v are connected iff the closed disc that has
    $uv$ as its diameter contains no other node of the set.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input point layer. The GeoDataFrame index is preserved as the node id.
    distance_metric : {"euclidean", "manhattan", "network"}, default "euclidean"
        Metric used for edge weights / geometries (see the other generators).
    network_gdf : geopandas.GeoDataFrame, optional
        Required when `distance_metric='network'`.
    network_weight : str, optional
        Edge attribute in `network_gdf` that supplies weights for network distances.
        Defaults to geometry length when not provided.
    as_nx : bool, default False
        If True return a NetworkX graph, otherwise return two GeoDataFrames (nodes, edges)
        via `nx_to_gdf`.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame] or networkx.Graph
        If `as_nx` is False, returns a tuple of GeoDataFrames:
        - nodes_gdf: GeoDataFrame of nodes with spatial and attribute information
        - edges_gdf: GeoDataFrame of edges with 'weight' and 'geometry' attributes
        If `as_nx` is True, returns a NetworkX graph object with spatial attributes.

    Notes
    -----
    - The Gabriel graph is a sub-graph of the Delaunay triangulation; therefore the
      implementation first builds the Delaunay edges then filters them according to the
      disc-emptiness predicate, achieving an overall $O(n \\log n + mk)$
      complexity (m = Delaunay edges, k = average neighbours tested per edge).
    - When the input layer has exactly two points, the unique edge is returned.
    - If the layer has fewer than two points, an empty graph is produced.

    References
    ----------
    .. [1] Gabriel, K. R., & Sokal, R. R. (1969). A new statistical approach to geographic
       variation analysis. Systematic zoology, 18(3), 259-278.
       https://doi.org/10.2307/2412323
    """
    metric = DistanceMetric(distance_metric, network_gdf, network_weight)
    builder = GraphBuilder(gdf, metric)
    builder.prepare_nodes()
    assert builder.coords is not None
    assert builder.node_ids is not None

    n_points = len(builder.coords)
    if n_points < 2:
        return builder.to_output(as_nx)

    # Candidate edges
    if n_points == 2:
        delaunay_edges = {(0, 1)}
    else:
        delaunay_edges = {
            tuple(sorted((i, j)))
            for simplex in Delaunay(builder.coords).simplices
            for i, j in combinations(simplex, 2)
        }

    # Gabriel filtering
    kept_edges = set()
    tol = 1e-12
    coords = builder.coords

    for i, j in delaunay_edges:
        mid = 0.5 * (coords[i] + coords[j])
        rad2 = np.sum((coords[i] - coords[j]) ** 2) * 0.25
        d2 = np.sum((coords - mid) ** 2, axis=1)
        mask = d2 <= rad2 + tol
        if np.count_nonzero(mask) == 2:
            kept_edges.add((builder.node_ids[i], builder.node_ids[j]))

    builder.add_edges(kept_edges)
    return builder.to_output(as_nx)


def relative_neighborhood_graph(
    gdf: gpd.GeoDataFrame,
    distance_metric: str = "euclidean",
    network_gdf: gpd.GeoDataFrame | None = None,
    network_weight: str | None = None,
    *,
    as_nx: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph:
    r"""
    Generate a Relative-Neighbourhood Graph (RNG) from a GeoDataFrame.

    In an RNG two nodes u and v are connected iff there is no third node *w* such that both
    $d(u,w) < d(u,v)$ and $d(v,w) < d(u,v)$. Equivalently, the intersection of the
    two open discs having radius $d(u,v)$ and centres u and v (the lune) is empty.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input point layer whose index provides the node ids.
    distance_metric : {"euclidean", "manhattan", "network"}, default "euclidean"
        Metric used to attach edge weights / geometries (see the other generators).
    network_gdf : geopandas.GeoDataFrame, optional
        Required when `distance_metric='network'`.
    network_weight : str, optional
        Edge attribute in `network_gdf` used for network distances. Defaults to
        geometry length when omitted.
    as_nx : bool, default False
        If True return a NetworkX graph, otherwise return two GeoDataFrames (nodes, edges)
        via `nx_to_gdf`.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame] or networkx.Graph
        If `as_nx` is False, returns a tuple of GeoDataFrames:
        - nodes_gdf: GeoDataFrame of nodes with spatial and attribute information
        - edges_gdf: GeoDataFrame of edges with 'weight' and 'geometry' attributes
        If `as_nx` is True, returns a NetworkX graph object with spatial attributes.

    Notes
    -----
    - The RNG is a sub-graph of the Delaunay triangulation; therefore the
      implementation first collects Delaunay edges ($O(n \\log n)$) and then filters
      them according to the lune-emptiness predicate.
    - When the input layer has exactly two points the unique edge is returned.
    - If the layer has fewer than two points, an empty graph is produced.

    References
    ----------
    .. [1] Toussaint, G. T. (1980). The relative neighbourhood graph of a finite planar
       set. Pattern recognition, 12(4), 261-268.
       https://doi.org/10.1016/0031-3203(80)90066-7
    """
    metric = DistanceMetric(distance_metric, network_gdf, network_weight)
    builder = GraphBuilder(gdf, metric)
    builder.prepare_nodes()
    assert builder.coords is not None
    assert builder.node_ids is not None

    n_points = len(builder.coords)
    if n_points < 2:
        return builder.to_output(as_nx)

    if n_points == 2:
        cand_edges = {(0, 1)}
    else:
        cand_edges = {
            tuple(sorted((i, j)))
            for simplex in Delaunay(builder.coords).simplices
            for i, j in combinations(simplex, 2)
        }

    kept_edges = set()
    coords = builder.coords

    for i, j in cand_edges:
        dij2 = np.dot(coords[i] - coords[j], coords[i] - coords[j])
        di2 = np.sum((coords - coords[i]) ** 2, axis=1) < dij2
        dj2 = np.sum((coords - coords[j]) ** 2, axis=1) < dij2
        closer_both = np.where(di2 & dj2)[0]
        if len(closer_both) == 0:
            kept_edges.add((builder.node_ids[i], builder.node_ids[j]))

    builder.add_edges(kept_edges)
    return builder.to_output(as_nx)


def euclidean_minimum_spanning_tree(
    gdf: gpd.GeoDataFrame,
    distance_metric: str = "euclidean",
    network_gdf: gpd.GeoDataFrame | None = None,
    network_weight: str | None = None,
    *,
    as_nx: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph:
    r"""
    Generate a (generalised) Euclidean Minimum Spanning Tree from a GeoDataFrame of points.

    The classical Euclidean Minimum Spanning Tree (EMST) is the minimum-total-length tree
    that connects a set of points when edge weights are the straight-line ($L_2$)
    distances. For consistency with the other generators this implementation also supports
    manhattan and network metrics - it simply computes the minimum-weight spanning tree
    under the chosen metric. When the metric is euclidean the edge search is restricted to
    the Delaunay triangulation (EMST ⊆ Delaunay), guaranteeing an $O(n \log n)$
    overall complexity. With other metrics, or degenerate cases where the triangulation
    cannot be built, the algorithm gracefully falls back to the complete graph.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input point layer. The index is preserved as the node identifier.
    distance_metric : {"euclidean", "manhattan", "network"}, default "euclidean"
        Metric used for the edge weights / geometries.
    network_gdf : geopandas.GeoDataFrame, optional
        Required when `distance_metric='network'`. Must contain the network arcs with
        valid pos attributes on its nodes.
    network_weight : str, optional
        Edge attribute name in `network_gdf` used for weighting shortest paths when
        `distance_metric='network'`. Defaults to geometry length.
    as_nx : bool, default False
        If True return a NetworkX graph, otherwise return two GeoDataFrames (nodes, edges)
        via `nx_to_gdf`.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame] or networkx.Graph
        If `as_nx` is False, returns a tuple of GeoDataFrames:
        - nodes_gdf: GeoDataFrame of nodes with spatial and attribute information
        - edges_gdf: GeoDataFrame of edges with 'weight' and 'geometry' attributes
        If `as_nx` is True, returns a NetworkX graph object with spatial attributes.

    See Also
    --------
    delaunay_graph : Generate a Delaunay triangulation graph.
    gabriel_graph : Generate a Gabriel graph.
    relative_neighborhood_graph : Generate a Relative Neighborhood Graph.

    Notes
    -----
    - The resulting graph always contains n - 1 edges (or 0 / 1 when the input has < 2 points).
    - For planar Euclidean inputs the computation is $O(n \\log n)$ thanks to the
      Delaunay pruning.
    - All the usual spatial attributes (weight, geometry, CRS checks, etc.) are attached
      through the shared private helpers.

    References
    ----------
    .. [1] March, W. B., Ram, P., & Gray, A. G. (2010, July). Fast euclidean minimum
       spanning tree: algorithm, analysis, and applications. In Proceedings of the 16th ACM
       SIGKDD international conference on Knowledge discovery and data mining (pp.
       603-612). https://doi.org/10.1145/1835804.1835882
    """
    metric = DistanceMetric(distance_metric, network_gdf, network_weight)
    builder = GraphBuilder(gdf, metric)
    builder.prepare_nodes()
    assert builder.coords is not None
    assert builder.node_ids is not None

    n_points = len(builder.coords)
    if n_points < 2:
        return builder.to_output(as_nx)

    # Candidate edges
    if metric.name == "euclidean" and n_points >= 3:
        tri = Delaunay(builder.coords)
        cand_edges = {
            tuple(sorted((i, j))) for simplex in tri.simplices for i, j in combinations(simplex, 2)
        }
    else:
        cand_edges = {(i, j) for i in range(n_points) for j in range(i + 1, n_points)}

    # Convert to node IDs
    named_edges = {(builder.node_ids[i], builder.node_ids[j]) for i, j in cand_edges}

    # Add all candidate edges with weights
    builder.compute_distance_matrix()  # Ensure DM is ready for weight assignment if needed
    builder.add_edges(named_edges)

    # Compute MST
    mst_G = nx.minimum_spanning_tree(builder.G, weight="weight", algorithm="kruskal")
    return mst_G if as_nx else nx_to_gdf(mst_G, nodes=True, edges=True)


def fixed_radius_graph(
    gdf: gpd.GeoDataFrame,
    radius: float,
    distance_metric: str = "euclidean",
    network_gdf: gpd.GeoDataFrame | None = None,
    network_weight: str | None = None,
    *,
    target_gdf: gpd.GeoDataFrame | None = None,
    as_nx: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph:
    r"""
    Generate a fixed-radius graph from a GeoDataFrame of points.

    This function constructs a graph where nodes are connected if the distance
    between them is within a specified radius. This model is particularly useful for
    modeling communication networks, ecological connectivity, and spatial influence
    zones where interaction strength has a clear distance threshold.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing the source points (nodes) for the graph. The index
        of this GeoDataFrame will be used as node IDs.
    radius : float
        The maximum distance for connecting nodes. Nodes within this radius will
        have an edge between them. Must be positive.
    distance_metric : str, default "euclidean"
        The distance metric to use for determining connections. Options are:
        - "euclidean": Straight-line distance
        - "manhattan": City-block distance (L1 norm)
        - "network": Shortest path distance along a network
    network_gdf : geopandas.GeoDataFrame, optional
        A GeoDataFrame representing a network (e.g., roads) to use for "network"
        distance calculations. Required if `distance_metric` is "network".
    network_weight : str, optional
        Edge attribute in `network_gdf` used as path weights when
        `distance_metric="network"`. Defaults to geometry length when not provided.
    target_gdf : geopandas.GeoDataFrame, optional
        If provided, creates a directed graph where edges connect nodes from `gdf` to
        nodes in `target_gdf` within the specified radius. If None, creates an undirected
        graph from `gdf` itself.
    as_nx : bool, default False
        If True, returns a NetworkX graph object. Otherwise, returns a tuple of
        GeoDataFrames (nodes, edges).

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame] or networkx.Graph
        If `as_nx` is False, returns a tuple of GeoDataFrames:
        - nodes_gdf: GeoDataFrame of nodes with spatial and attribute information
        - edges_gdf: GeoDataFrame of edges with 'weight' and 'geometry' attributes
        If `as_nx` is True, returns a NetworkX graph object with spatial attributes.

    Raises
    ------
    ValueError
        If `distance_metric` is "network" but `network_gdf` is not provided.
        If `radius` is not positive.

    See Also
    --------
    knn_graph : Generate a k-nearest neighbour graph.
    delaunay_graph : Generate a Delaunay triangulation graph.
    waxman_graph : Generate a probabilistic Waxman graph.

    Notes
    -----
    - Node IDs are preserved from the input GeoDataFrame's index
    - Edge weights represent the actual distance between connected nodes
    - Edge geometries are LineStrings connecting node centroids
    - The graph stores the radius parameter in G.graph["radius"]
    - For Manhattan distance, edges follow L-shaped geometric paths

    References
    ----------
    .. [1] Bentley, J. L., Stanat, D. F., & Williams Jr, E. H. (1977). The complexity of
       finding fixed-radius near neighbors. Information processing letters, 6(6),
       209-212. https://doi.org/10.1016/0020-0190(77)90070-9
    """
    if target_gdf is not None:
        return _directed_graph(
            src_gdf=gdf,
            dst_gdf=target_gdf,
            distance_metric=distance_metric,
            method="radius",
            param=radius,
            as_nx=as_nx,
            network_gdf=network_gdf,
            network_weight=network_weight,
        )

    metric = DistanceMetric(distance_metric, network_gdf, network_weight)
    builder = GraphBuilder(gdf, metric)
    builder.prepare_nodes()
    assert builder.coords is not None
    assert builder.node_ids is not None

    if len(builder.coords) < 2:
        return builder.to_output(as_nx)

    if metric.name == "network":
        builder.compute_distance_matrix()
        assert builder.dm is not None
        mask = (builder.dm <= radius) & np.triu(np.ones_like(builder.dm, dtype=bool), 1)
        edge_idx = np.column_stack(np.where(mask))
        edges = [
            (builder.node_ids[i], builder.node_ids[j])
            for i, j in edge_idx
            if builder.dm[i, j] < np.inf
        ]
    else:
        nn_metric = "cityblock" if metric.name == "manhattan" else "euclidean"
        nn = NearestNeighbors(radius=radius, metric=nn_metric).fit(builder.coords)
        idxs = nn.radius_neighbors(builder.coords, return_distance=False)
        edges = [
            (builder.node_ids[i], builder.node_ids[j])
            for i, neigh in enumerate(idxs)
            for j in neigh
            if i < j
        ]

    builder.add_edges(edges)
    builder.G.graph["radius"] = radius
    return builder.to_output(as_nx)


def waxman_graph(
    gdf: gpd.GeoDataFrame,
    beta: float,
    r0: float,
    seed: int | None = None,
    distance_metric: Literal["euclidean", "manhattan", "network"] = "euclidean",
    network_gdf: gpd.GeoDataFrame | None = None,
    network_weight: str | None = None,
    *,
    as_nx: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph:
    r"""
    Generate a probabilistic Waxman graph from a GeoDataFrame of points.

    This function constructs a random graph where the probability of an edge
    existing between two nodes decreases exponentially with their distance. The model is
    based on the Waxman random graph model, commonly used to simulate realistic
    network topologies in telecommunications, transportation, and social networks where
    connection probability diminishes with distance.

    The connection probability follows the formula:

    $$
    P(u,v) = \beta \times \exp \left(-\frac{\text{dist}(u,v)}{r_0}\right)
    $$

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing the points (nodes) for the graph. The index of this
        GeoDataFrame will be used as node IDs.
    beta : float
        Parameter controlling the overall probability of edge creation. Higher
        values (closer to 1.0) increase the likelihood of connections. Must be between 0 and 1.
    r0 : float
        Parameter controlling the decay rate of probability with distance. Higher
        values result in longer-range connections being more likely. Must be positive.
    seed : int, optional
        Seed for the random number generator to ensure reproducibility of results.
        If None, results will vary between runs.
    distance_metric : str, default "euclidean"
        The distance metric to use for calculating distances between nodes. Options are:
        - "euclidean": Straight-line distance
        - "manhattan": City-block distance (L1 norm)
        - "network": Shortest path distance along a network
    network_gdf : geopandas.GeoDataFrame, optional
        A GeoDataFrame representing a network (e.g., roads) to use for "network"
        distance calculations. Required if `distance_metric` is "network".
    network_weight : str, optional
        Edge attribute name in `network_gdf` used for network distances. Defaults to
        geometry length when omitted.
    as_nx : bool, default False
        If True, returns a NetworkX graph object. Otherwise, returns a tuple of
        GeoDataFrames (nodes, edges).

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame] or networkx.Graph
        If `as_nx` is False, returns a tuple of GeoDataFrames:
        - nodes_gdf: GeoDataFrame of nodes with spatial and attribute information
        - edges_gdf: GeoDataFrame of edges with 'weight' and 'geometry' attributes
        If `as_nx` is True, returns a NetworkX graph object with spatial attributes.

    Raises
    ------
    ValueError
        If `distance_metric` is "network" but `network_gdf` is not provided.
        If `beta` is not between 0 and 1, or if `r0` is not positive.

    See Also
    --------
    knn_graph : Generate a k-nearest neighbour graph.
    delaunay_graph : Generate a Delaunay triangulation graph.
    fixed_radius_graph : Generate a fixed-radius graph.

    Notes
    -----
    - Node IDs are preserved from the input GeoDataFrame's index
    - Edge weights represent the actual distance between connected nodes
    - Edge geometries are LineStrings connecting node centroids
    - The graph stores parameters in G.graph["beta"] and G.graph["r0"]
    - Results are stochastic; use seed parameter for reproducible outputs
    - The graph is undirected with symmetric edge probabilities

    References
    ----------
    .. [1] Waxman, B. M. (2002). Routing of multipoint connections. IEEE journal on
       selected areas in communications, 6(9), 1617-1622.
       https://doi.org/10.1109/49.12889
    """
    rng = np.random.default_rng(seed)
    metric = DistanceMetric(distance_metric, network_gdf, network_weight)
    builder = GraphBuilder(gdf, metric)
    builder.prepare_nodes()
    assert builder.coords is not None
    assert builder.node_ids is not None

    if len(builder.coords) < 2:
        return builder.to_output(as_nx)

    builder.compute_distance_matrix()
    assert builder.dm is not None

    with np.errstate(divide="ignore"):
        probs = beta * np.exp(-builder.dm / r0)
    probs[builder.dm == np.inf] = 0

    rand = rng.random(builder.dm.shape)
    mask = (rand <= probs) & np.triu(np.ones_like(builder.dm, dtype=bool), 1)
    edge_idx = np.column_stack(np.where(mask))
    edges = [(builder.node_ids[i], builder.node_ids[j]) for i, j in edge_idx]

    builder.add_edges(edges)
    builder.G.graph.update({"beta": beta, "r0": r0})
    return builder.to_output(as_nx)


def _normalize_layer_types(
    values: Iterable[str] | None,
    label: str,
    node_order: tuple[str, ...],
    node_set: set[str],
) -> list[str]:
    """
    Return requested node types if they are defined.

    The helper validates requested node types against the available set and
    preserves the original ordering to maintain deterministic layer pairing.

    Parameters
    ----------
    values : Iterable[str] | None
        Requested node types (``None`` means all types).
    label : str
        Role label used for error messaging.
    node_order : tuple[str, ...]
        Ordered tuple of available node types.
    node_set : set[str]
        Lookup set for validating requested node types.

    Returns
    -------
    list[str]
        Ordered list of validated node types.
    """
    if values is None:
        return list(node_order)
    unique_values = list(dict.fromkeys(values))
    missing = [v for v in unique_values if v not in node_set]
    if missing:
        sorted_missing = ", ".join(sorted(missing))
        msg = f"Unknown {label} node types: {sorted_missing}"
        raise ValueError(msg)
    return unique_values


def bridge_nodes(
    nodes_dict: dict[str, gpd.GeoDataFrame],
    proximity_method: str = "knn",
    *,
    source_node_types: Iterable[str] | None = None,
    target_node_types: Iterable[str] | None = None,
    multigraph: bool = False,
    as_nx: bool = False,
    **kwargs: float | str | bool,
) -> tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]] | nx.Graph:
    r"""
    Build directed proximity edges between every ordered pair of node layers.

    This function creates a multi-layer spatial network by generating directed
    proximity edges from nodes in one GeoDataFrame layer to nodes in another. It
    systematically processes all ordered pairs of layers and applies either k-nearest
    neighbors (KNN) or fixed-radius method to establish inter-layer connections. This
    function is specifically designed for constructing heterogeneous graphs by
    generating new edge types ("is_nearby") between different types of nodes, enabling the
    modeling of complex relationships across multiple domains. It is useful for
    modeling complex urban systems, ecological networks, or multi-modal transportation
    systems where different types of entities interact through spatial proximity.

    Parameters
    ----------
    nodes_dict : dict[str, geopandas.GeoDataFrame]
        A dictionary where keys are layer names (strings) and values are
        GeoDataFrames representing the nodes of each layer. Each GeoDataFrame should
        contain point geometries with consistent CRS across all layers.
    proximity_method : str, default "knn"
        The method to use for generating proximity edges between layers. Options are:
        - "knn": k-nearest neighbors method
        - "fixed_radius": fixed-radius method
    source_node_types : Iterable[str], optional
        Node types from ``nodes_dict`` that should act as sources. When None, all
        node types are considered sources.
    target_node_types : Iterable[str], optional
        Node types from ``nodes_dict`` that should act as targets. When None, all
        node types are considered targets.
    multigraph : bool, default False
        If True, the resulting NetworkX graph will be a MultiGraph, allowing
        multiple edges between the same pair of nodes from different proximity relationships.
    as_nx : bool, default False
        If True, returns a NetworkX graph object containing all nodes and
        inter-layer edges. Otherwise, returns dictionaries of GeoDataFrames.
    **kwargs : Any
        Additional keyword arguments passed to the underlying proximity method:

        For `proximity_method="knn"`:
            - k : int, default 1
              Number of nearest neighbors to connect to in target layer
            - distance_metric : str, default "euclidean"
              Distance metric ("euclidean", "manhattan", "network")
            - network_gdf : geopandas.GeoDataFrame, optional
              Network for "network" distance calculations
            - network_weight : str, optional
                Edge attribute used as shortest-path weight when ``distance_metric='network'``

        For `proximity_method="fixed_radius"`:
            - radius : float, required
              Maximum connection distance between layers
            - distance_metric : str, default "euclidean"
              Distance metric ("euclidean", "manhattan", "network")
            - network_gdf : geopandas.GeoDataFrame, optional
              Network for "network" distance calculations
            - network_weight : str, optional
                Edge attribute used as shortest-path weight when ``distance_metric='network'``

    Returns
    -------
    tuple[dict[str, geopandas.GeoDataFrame], dict[tuple[str, str, str], geopandas.GeoDataFrame]] or networkx.Graph
        If `as_nx` is False, returns a tuple containing:

        - nodes_dict: The original input nodes_dict (unchanged)
        - edges_dict: Dictionary where keys are edge type tuples of the form
          (source_layer_name, "is_nearby", target_layer_name) and values are GeoDataFrames
          of the generated directed edges. Each unique tuple represents a distinct edge type
          in the heterogeneous graph, enabling differentiation between relationships across
          different node type combinations.

        If `as_nx` is True, returns a NetworkX graph object containing all nodes from all
        layers and the generated directed inter-layer edges, forming a heterogeneous graph
        structure where different node types are connected through proximity-based relationships.

    Raises
    ------
    ValueError
        If `nodes_dict` contains fewer than two layers.
        If `proximity_method` is not "knn" or "fixed_radius".
        If `proximity_method` is "fixed_radius" but `radius` is not provided in kwargs.
        If unknown node types are provided via ``source_node_types`` or ``target_node_types``.

    See Also
    --------
    knn_graph : Generate a k-nearest neighbour graph.
    fixed_radius_graph : Generate a fixed-radius graph.

    Notes
    -----
    - All generated edges are directed from source layer to target layer
    - The relation type for all generated edges is fixed as "is_nearby", creating a
      new edge type that bridges different node types in heterogeneous graphs
    - Each ordered pair of node layers generates a distinct edge type, enabling the
      construction of rich heterogeneous graph structures with multiple relationship
      types between different domain entities
    - Edge weights and geometries are calculated based on the chosen `distance_metric`
    - Each ordered pair of layers generates a separate edge GeoDataFrame
    - Self-connections (layer to itself) are not created
    - The resulting structure is ideal for heterogeneous graph neural networks,
      multi-layer network analysis, and cross-domain spatial relationship modeling
    """
    if len(nodes_dict) < 2:
        msg = "`nodes_dict` needs at least two layers"
        raise ValueError(msg)

    if proximity_method.lower() not in {"knn", "fixed_radius"}:
        msg = "proximity_method must be 'knn' or 'fixed_radius'"
        raise ValueError(msg)

    edge_dict = {}
    method = proximity_method.lower()
    metric_name = str(kwargs.get("distance_metric", "euclidean"))
    net_gdf = kwargs.get("network_gdf")
    net_weight_raw = kwargs.get("network_weight")
    net_weight: str | None = net_weight_raw if isinstance(net_weight_raw, str) else None

    # Extract param
    param = float(kwargs.get("k", 1)) if method == "knn" else float(kwargs["radius"])

    node_order = tuple(nodes_dict.keys())
    node_set = set(node_order)

    source_types = _normalize_layer_types(source_node_types, "source", node_order, node_set)
    target_types = _normalize_layer_types(target_node_types, "target", node_order, node_set)

    for src_type in source_types:
        for dst_type in target_types:
            if src_type == dst_type:
                continue

            src_gdf = nodes_dict[src_type]
            dst_gdf = nodes_dict[dst_type]

            _, edges_gdf = _directed_graph(
                src_gdf=src_gdf,
                dst_gdf=dst_gdf,
                distance_metric=metric_name,
                method=method,
                param=param,
                as_nx=False,
                network_gdf=net_gdf,
                network_weight=net_weight,
                return_nodes=False,
            )

            # Strip the ("src", id) and ("dst", id) wrappers to return original IDs
            if not edges_gdf.empty:
                src_ids = [s[1] for s in edges_gdf.index.get_level_values(0)]
                dst_ids = [d[1] for d in edges_gdf.index.get_level_values(1)]
                edges_gdf.index = pd.MultiIndex.from_arrays(
                    [src_ids, dst_ids], names=["source", "target"]
                )
            edge_dict[(src_type, "is_nearby", dst_type)] = edges_gdf

    if as_nx:
        return gdf_to_nx(nodes=nodes_dict, edges=edge_dict, multigraph=multigraph, directed=True)
    return nodes_dict, edge_dict


def group_nodes(
    polygons_gdf: gpd.GeoDataFrame,
    points_gdf: gpd.GeoDataFrame,
    *,
    distance_metric: Literal["euclidean", "manhattan", "network"] = "euclidean",
    network_gdf: gpd.GeoDataFrame | None = None,
    network_weight: str | None = None,
    predicate: str = "covered_by",
    as_nx: bool = False,
) -> tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]] | nx.Graph:
    r"""
    Create a heterogeneous graph linking polygon zones to contained points.

    This function builds a bipartite relation between polygon and point features by
    connecting each polygon to the points that satisfy a spatial containment
    predicate (default: "covered_by" so boundary points are included). It follows
    city2graph heterogeneous graph conventions and reuses the proximity helpers for
    computing edge weights and geometries according to the chosen distance metric.

    Parameters
    ----------
    polygons_gdf : geopandas.GeoDataFrame
        GeoDataFrame of polygonal features representing zones. CRS must match
        `points_gdf`. Original attributes and geometries are preserved in the resulting
        polygon nodes.
    points_gdf : geopandas.GeoDataFrame
        GeoDataFrame of point features to be associated with the polygons. CRS must
        match `polygons_gdf`. Original attributes and geometries are preserved in the
        resulting point nodes.
    distance_metric : {"euclidean", "manhattan", "network"}, default "euclidean"
        Metric used for edge weights and geometries. Euclidean produces straight
        line segments, Manhattan produces L-shaped polylines, and Network traces polylines
        along the provided `network_gdf` and uses shortest-path distances.
    network_gdf : geopandas.GeoDataFrame, optional
        Required when `distance_metric="network"`. Must share the same CRS as the inputs.
    network_weight : str, optional
        Edge attribute in `network_gdf` supplying network path weights. Defaults to
        geometry length when omitted.
    predicate : str, default "covered_by"
        Spatial predicate used to determine containment in a vectorized spatial join
        (e.g., "covered_by", "within", "contains", "intersects"). The default includes
        points on polygon boundaries.
    as_nx : bool, default False
        If False, return heterogeneous GeoDataFrame dictionaries. If True, return a
        typed heterogeneous NetworkX graph built with `gdf_to_nx`.

    Returns
    -------
    tuple[dict[str, GeoDataFrame], dict[tuple[str, str, str], GeoDataFrame]] or Graph
        (nodes_dict, edges_dict) (tuple of dicts)
            Returned when `as_nx=False`. `nodes_dict` is {"polygon": polygons_gdf, "point":
            points_gdf} with original indices, attributes, and geometries. `edges_dict` maps a
            typed edge key to an edges GeoDataFrame whose index is a MultiIndex of
            (polygon_id, point_id) and includes at least weight and geometry columns. The edge key
            has the form ("polygon", relation, "point"), where relation is derived from
            predicate (e.g., covered_by -> "covers", within -> "contains").
        G (networkx.Graph)
            Returned when `as_nx=True`. A heterogeneous graph with node_type in nodes and
            a typed edge_type reflecting the relation derived from predicate. Graph metadata
            includes crs and is_hetero=True.

    Notes
    -----
    - CRS must be present and identical for both inputs. For network metric, the
      network's CRS must also match.
    - Boundary points are included by default via `predicate="covered_by"`.
    - Distance calculations and edge geometries reuse internal helpers
      (_prepare_nodes, _distance_matrix, _add_edges) to ensure consistency with other
      proximity functions.
    """
    relation = _relation_from_predicate(predicate)
    metric = DistanceMetric(distance_metric, network_gdf, network_weight)

    # Validate CRS
    poly_crs = polygons_gdf.crs
    pt_crs = points_gdf.crs
    if not poly_crs or not pt_crs:
        msg = f"Both inputs must have a CRS (got polygons_gdf.crs={poly_crs}, points_gdf.crs={pt_crs})"
        raise ValueError(msg)

    validate_gdf({"polygon": polygons_gdf, "point": points_gdf}, None, allow_empty=True)

    if poly_crs != pt_crs:
        msg = f"CRS mismatch between inputs: {poly_crs} != {pt_crs}"
        raise ValueError(msg)

    metric.validate(poly_crs)

    if polygons_gdf.empty or points_gdf.empty:
        return _group_nodes_empty_result(polygons_gdf, points_gdf, relation, poly_crs, as_nx)

    pairs = _containment_pairs(points_gdf, polygons_gdf, predicate)
    if not pairs:
        return _group_nodes_empty_result(polygons_gdf, points_gdf, relation, poly_crs, as_nx)

    edges_gdf = _edges_gdf_from_pairs(polygons_gdf, points_gdf, pairs, metric)

    nodes_dict = {"polygon": polygons_gdf, "point": points_gdf}
    edges_dict = {("polygon", relation, "point"): edges_gdf}

    return (
        gdf_to_nx(nodes=nodes_dict, edges=edges_dict, directed=True)
        if as_nx
        else (nodes_dict, edges_dict)
    )


def contiguity_graph(
    gdf: gpd.GeoDataFrame,
    contiguity: str = "queen",
    *,
    distance_metric: str = "euclidean",
    network_gdf: gpd.GeoDataFrame | None = None,
    network_weight: str | None = None,
    as_nx: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph:
    r"""
    Generate a contiguity-based spatial graph from polygon geometries.

    This function creates a spatial graph where nodes represent polygons and edges
    connect spatially contiguous (adjacent) polygons based on Queen or Rook
    contiguity rules. It leverages libpysal's robust spatial weights functionality to
    accurately determine adjacency relationships, making it ideal for spatial analysis of
    administrative boundaries, urban morphology studies, land use patterns, and
    geographic network analysis.

    The function supports both Queen contiguity (polygons sharing edges or vertices)
    and Rook contiguity (polygons sharing only edges), providing flexibility for
    different spatial analysis requirements. Edge weights are calculated as
    distances between polygon centroids using the selected `distance_metric`. Supported metrics:

    - `euclidean` (default): straight-line distance; edge geometry is a direct
      centroid-to-centroid LineString.
    - `manhattan`: L1 distance; edge geometry is an L-shaped polyline (two segments)
      following an axis-aligned path between centroids.
    - `network`: shortest-path distance over `network_gdf` (a line network in the
      same CRS); edge geometry is the polyline path traced along the network.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input GeoDataFrame containing polygon geometries. Must contain valid polygon
        geometries in the 'geometry' column. The index of this GeoDataFrame will be
        preserved as node identifiers in the output graph. All original attributes are
        maintained in the nodes output.
    contiguity : {"queen", "rook"}, default "queen"
        Type of spatial contiguity rule to apply for determining adjacency:

        - "queen": Polygons are considered adjacent if they share any boundary (edges or
          vertices). This is more inclusive and typically results in more connections.
        - "rook": Polygons are considered adjacent only if they share an edge (not just
          vertices). This is more restrictive and results in fewer connections.
    distance_metric : {"euclidean", "manhattan", "network"}, default "euclidean"
        Metric used to compute edge weights and geometries.
    network_gdf : geopandas.GeoDataFrame, optional
        Required when `distance_metric='network'`. A line-based network whose CRS matches `gdf`.
    network_weight : str, optional
        Edge attribute in `network_gdf` to use for shortest-path weights. Defaults to
        geometry length when omitted.
    as_nx : bool, default False
        Output format control. If True, returns a NetworkX Graph object with spatial
        attributes. If False, returns a tuple of GeoDataFrames for nodes and edges,
        compatible with other city2graph functions.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame] or networkx.Graph
        When `as_nx=False` (default), returns `(nodes_gdf, edges_gdf)` as GeoDataFrames.
        When `as_nx=True`, returns a NetworkX Graph with spatial attributes and metadata.

    Raises
    ------
    TypeError
        If `gdf` is not a geopandas.GeoDataFrame instance.
    ValueError
        If `contiguity` is not one of {"queen", "rook"}.
        If `gdf` contains geometries that are not polygons (Point, LineString, etc.).
        If `gdf` contains invalid or corrupt polygon geometries.
        If libpysal fails to create spatial weights matrix.

    See Also
    --------
    libpysal.weights.Queen : Spatial weights based on Queen contiguity.
    libpysal.weights.Rook : Spatial weights based on Rook contiguity.
    knn_graph : Generate k-nearest neighbor graphs from point data.
    delaunay_graph : Generate Delaunay triangulation graphs from point data.
    fixed_radius_graph : Generate fixed-radius proximity graphs.
    gabriel_graph : Generate Gabriel graphs from point data.
    relative_neighborhood_graph : Generate relative neighborhood graphs.
    waxman_graph : Generate probabilistic Waxman graphs.
    """
    _validate_contiguity_input(gdf, contiguity)
    metric = DistanceMetric(distance_metric, network_gdf, network_weight)
    metric.validate(gdf.crs)

    if gdf.empty:
        return _empty_contiguity_result(gdf, contiguity, distance_metric=metric.name, as_nx=as_nx)

    weights = _create_spatial_weights(gdf, contiguity)
    edges = _generate_contiguity_edges(weights)

    # Build graph using GraphBuilder
    builder = GraphBuilder(gdf, metric)
    builder.prepare_nodes()

    # Convert edges from indices to node IDs
    # _generate_contiguity_edges returns indices (from gdf.index)
    # GraphBuilder expects node IDs which are also from gdf.index
    # So we can pass them directly
    builder.add_edges(edges)

    builder.G.graph["contiguity"] = contiguity
    builder.G.graph["distance_metric"] = metric.name

    return builder.to_output(as_nx)


# ============================================================================
# INTERNAL HELPERS (Heterogeneous / Directed)
# ============================================================================


@dataclass(slots=True)
class DirectedGraphContext:
    """
    Container for precomputed artefacts required by directed builders.

    Packing the intermediate data keeps downstream helper signatures small and
    ensures that repeated operations derive from a consistent snapshot.
    """

    src_gdf: gpd.GeoDataFrame
    dst_gdf: gpd.GeoDataFrame
    src_coords: npt.NDArray[np.floating]
    dst_coords: npt.NDArray[np.floating]
    src_ids: list[Any]
    dst_ids: list[Any]
    unique_src_ids: list[tuple[str, Any]]
    unique_dst_ids: list[tuple[str, Any]]
    edges: list[tuple[Any, Any]]
    dm: npt.NDArray[np.floating] | None
    metric: DistanceMetric


def _directed_graph(
    *,
    src_gdf: gpd.GeoDataFrame,
    dst_gdf: gpd.GeoDataFrame,
    distance_metric: str,
    method: str,
    param: float,
    as_nx: bool,
    network_gdf: gpd.GeoDataFrame | None = None,
    network_weight: str | None = None,
    return_nodes: bool = True,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph:
    """
    Build source-to-target proximity edges between two GeoDataFrames.

    The helper centralises CRS validation, computes the directed edges via the
    requested method, and then produces either GeoDataFrame or NetworkX outputs.

    Parameters
    ----------
    src_gdf, dst_gdf : geopandas.GeoDataFrame
        Source and destination layers to be connected.
    distance_metric : str
        Metric name as understood by :class:`DistanceMetric`.
    method : str
        ``"knn"`` or ``"radius"`` depending on the proximity strategy.
    param : float
        Method-specific parameter (``k`` or distance threshold).
    as_nx : bool
        If True, return a NetworkX graph instead of GeoDataFrames.
    network_gdf : geopandas.GeoDataFrame, optional
        Supporting network data required for ``network`` metric.
    network_weight : str, optional
        Edge attribute used as weight when ``distance_metric`` equals ``"network"``.
        Defaults to geometry-derived lengths.
    return_nodes : bool, default True
        When False, only the edge GeoDataFrame is produced.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame] or networkx.Graph
        Directed graph representation matching the requested format.
    """
    if src_gdf.crs != dst_gdf.crs:
        msg = "CRS mismatch between source and target GeoDataFrames"
        raise ValueError(msg)

    metric = DistanceMetric(distance_metric, network_gdf, network_weight)
    metric.validate(src_gdf.crs)

    src_coords = np.column_stack([src_gdf.geometry.centroid.x, src_gdf.geometry.centroid.y])
    dst_coords = np.column_stack([dst_gdf.geometry.centroid.x, dst_gdf.geometry.centroid.y])
    src_ids = list(src_gdf.index)
    dst_ids = list(dst_gdf.index)

    edges, dm = _directed_edges(
        src_coords, dst_coords, src_ids, dst_ids, metric=metric, method=method, param=param
    )

    # Prepare namespaced IDs
    unique_src_ids = [("src", sid) for sid in src_ids]
    unique_dst_ids = [("dst", did) for did in dst_ids]

    context = DirectedGraphContext(
        src_gdf=src_gdf,
        dst_gdf=dst_gdf,
        src_coords=src_coords,
        dst_coords=dst_coords,
        src_ids=src_ids,
        dst_ids=dst_ids,
        unique_src_ids=unique_src_ids,
        unique_dst_ids=unique_dst_ids,
        edges=edges,
        dm=dm,
        metric=metric,
    )

    if not as_nx:
        return _directed_graph_gdf(context, return_nodes=return_nodes)

    return _directed_graph_nx(context)


def _directed_graph_gdf(
    context: DirectedGraphContext, *, return_nodes: bool
) -> tuple[gpd.GeoDataFrame | None, gpd.GeoDataFrame]:
    """
    Build directed graph outputs expressed as GeoDataFrames.

    The helper mirrors the tuple-of-GeoDataFrames contract used by the public
    API so that heterogeneous flows can reuse the same downstream consumers.

    Parameters
    ----------
    context : DirectedGraphContext
        Precomputed values shared across heterogeneous builders.
    return_nodes : bool
        If False, only the edge GeoDataFrame is produced.

    Returns
    -------
    tuple[geopandas.GeoDataFrame | None, geopandas.GeoDataFrame]
        Directed node and edge layers, or ``(None, edges)`` when nodes are skipped.
    """
    combined_coords = np.vstack([context.src_coords, context.dst_coords])
    combined_ids = context.unique_src_ids + context.unique_dst_ids

    # Map edges to namespaced IDs
    src_map = dict(zip(context.src_ids, context.unique_src_ids, strict=False))
    dst_map = dict(zip(context.dst_ids, context.unique_dst_ids, strict=False))
    namespaced_edges = [(src_map[u], dst_map[v]) for u, v in context.edges]

    # Construct a dummy builder to use its _compute_edge_data
    dummy_builder = GraphBuilder(context.src_gdf, context.metric)
    dummy_builder.coords = combined_coords
    dummy_builder.node_ids = combined_ids
    dummy_builder.dm = context.dm

    weights, geoms = dummy_builder._compute_edge_data(namespaced_edges)

    edges_gdf = gpd.GeoDataFrame(
        {"weight": weights},
        geometry=geoms,
        crs=context.src_gdf.crs,
        index=pd.MultiIndex.from_tuples(namespaced_edges, names=["source", "target"]),
    )

    if not return_nodes:
        return None, edges_gdf

    # Construct nodes GDF
    src_nodes = context.src_gdf.copy()
    src_nodes.index = pd.MultiIndex.from_tuples(
        [("src", i) for i in src_nodes.index],
        names=["layer", context.src_gdf.index.name or "node_id"],
    )
    src_nodes["node_type"] = "src"
    src_nodes["_original_index"] = context.src_gdf.index

    dst_nodes = context.dst_gdf.copy()
    dst_nodes.index = pd.MultiIndex.from_tuples(
        [("dst", i) for i in dst_nodes.index],
        names=["layer", context.dst_gdf.index.name or "node_id"],
    )
    dst_nodes["node_type"] = "dst"
    dst_nodes["_original_index"] = context.dst_gdf.index

    nodes_gdf = pd.concat([src_nodes, dst_nodes])
    return nodes_gdf, edges_gdf


def _directed_graph_nx(context: DirectedGraphContext) -> nx.Graph:
    """
    Build directed graph outputs expressed as a NetworkX graph.

    This variant shares the same context as :func:`_directed_graph_gdf` but
    keeps the richer NetworkX representation intact for graph analytics.

    Parameters
    ----------
    context : DirectedGraphContext
        Precomputed values shared across heterogeneous builders.

    Returns
    -------
    networkx.Graph
        Directed NetworkX graph with node metadata and geometry-aware edges.
    """
    src_builder = GraphBuilder(context.src_gdf, context.metric, directed=True)
    src_builder.prepare_nodes()
    dst_builder = GraphBuilder(context.dst_gdf, context.metric, directed=True)
    dst_builder.prepare_nodes()

    src_relabel = dict(zip(context.src_ids, context.unique_src_ids, strict=False))
    dst_relabel = dict(zip(context.dst_ids, context.unique_dst_ids, strict=False))

    src_G = nx.relabel_nodes(src_builder.G, src_relabel, copy=False)
    dst_G = nx.relabel_nodes(dst_builder.G, dst_relabel, copy=False)

    nx.set_node_attributes(
        src_G, dict(zip(context.unique_src_ids, context.src_ids, strict=False)), "_original_index"
    )
    nx.set_node_attributes(src_G, "src", "node_type")
    nx.set_node_attributes(
        dst_G, dict(zip(context.unique_dst_ids, context.dst_ids, strict=False)), "_original_index"
    )
    nx.set_node_attributes(dst_G, "dst", "node_type")

    G = nx.compose(src_G, dst_G)

    relabeled_edges = [(src_relabel[u], dst_relabel[v]) for u, v in context.edges]

    # Compute edge data
    combined_coords = np.vstack([context.src_coords, context.dst_coords])
    combined_ids = context.unique_src_ids + context.unique_dst_ids

    dummy_builder = GraphBuilder(context.src_gdf, context.metric)
    dummy_builder.coords = combined_coords
    dummy_builder.node_ids = combined_ids
    dummy_builder.dm = context.dm
    dummy_builder.G = G

    dummy_builder.add_edges(relabeled_edges)

    orig_edge_index = {
        (u, v): (G.nodes[u]["_original_index"], G.nodes[v]["_original_index"]) for u, v in G.edges()
    }
    nx.set_edge_attributes(G, orig_edge_index, "_original_edge_index")

    return G


def _directed_edges(
    src_coords: npt.NDArray[np.floating],
    dst_coords: npt.NDArray[np.floating],
    src_ids: list[int],
    dst_ids: list[int],
    *,
    metric: DistanceMetric,
    method: str,
    param: float,
) -> tuple[list[tuple[int, int]], npt.NDArray[np.floating] | None]:
    """
    Generate directed edges from source to destination nodes.

    Supports both k-nearest neighbour and fixed-radius strategies while also
    accommodating Euclidean, Manhattan, and network distances.

    Parameters
    ----------
    src_coords, dst_coords : numpy.typing.NDArray[np.floating]
        Coordinate arrays for source and destination geometries.
    src_ids, dst_ids : list[int]
        Identifiers aligned with ``src_coords`` and ``dst_coords`` rows.
    metric : DistanceMetric
        Prepared metric wrapper that decides how distances are computed.
    method : str
        Either ``"knn"`` or ``"radius"`` to choose the selection logic.
    param : float
        Number of neighbours (``knn``) or distance threshold (``radius``).

    Returns
    -------
    tuple[list[tuple[int, int]], numpy.typing.NDArray[np.floating] or None]
        Pair of selected edges and the optional dense distance matrix.
    """
    if metric.name == "network":
        # Compute network distances for all src+dst points
        combined_coords = np.vstack([src_coords, dst_coords])
        dm_full = metric.matrix(combined_coords)

        src_n = len(src_coords)
        dst_n = len(dst_coords)
        d_sub = dm_full[:src_n, src_n : src_n + dst_n]
        finite = np.isfinite(d_sub)

        if method == "knn":
            k = int(param)
            order = np.argsort(d_sub, axis=1)
            rows = np.arange(src_n)[:, None]
            ranks = np.empty_like(order)
            ranks[rows, order] = np.arange(dst_n)[None, :]
            sel_mask = (ranks < k) & finite
            i_idx, j_idx = np.where(sel_mask)
        else:  # radius
            i_idx, j_idx = np.where(finite & (d_sub <= param))

        edges = list(zip((src_ids[i] for i in i_idx), (dst_ids[j] for j in j_idx), strict=True))
        return edges, dm_full

    # Euclidean / Manhattan
    nn_metric = "cityblock" if metric.name == "manhattan" else "euclidean"
    if method == "knn":
        k = int(param)
        n_neigh = min(k, len(dst_coords))
        nn = NearestNeighbors(n_neighbors=n_neigh, metric=nn_metric).fit(dst_coords)
        _, idxs = nn.kneighbors(src_coords)
        edges = [(src_ids[i], dst_ids[j]) for i, neigh in enumerate(idxs) for j in neigh]
    else:  # radius
        nn = NearestNeighbors(radius=param, metric=nn_metric).fit(dst_coords)
        idxs = nn.radius_neighbors(src_coords, return_distance=False)
        edges = [(src_ids[i], dst_ids[j]) for i, neigh in enumerate(idxs) for j in neigh]

    return edges, None


def _relation_from_predicate(predicate: str | None) -> str:
    """
    Map a spatial predicate to a canonical relation label.

    This keeps downstream heterogeneous graphs consistent regardless of the
    specific predicate wording used by the caller.

    Parameters
    ----------
    predicate : str or None
        Spatial predicate such as ``"within"`` or ``"covered_by"``.

    Returns
    -------
    str
        Canonicalised relation label used as the edge key.
    """
    pred = (predicate or "covered_by").lower()
    return {"covered_by": "covers", "within": "contains", "contains": "contains"}.get(pred, pred)


def _group_nodes_empty_result(
    polygons_gdf: gpd.GeoDataFrame,
    points_gdf: gpd.GeoDataFrame,
    relation: str,
    crs: object,
    as_nx: bool,
) -> tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]] | nx.Graph:
    """
    Return an empty heterogeneous structure.

    The helper ensures that callers always receive correctly typed containers
    even when no polygon-to-point relations satisfy the predicate.

    Parameters
    ----------
    polygons_gdf, points_gdf : geopandas.GeoDataFrame
        Polygon and point layers that would normally be connected.
    relation : str
        Canonical relation label (e.g., ``"contains"``).
    crs : object
        Coordinate reference system shared by both layers.
    as_nx : bool
        If True, convert the empty structure into a NetworkX graph.

    Returns
    -------
    tuple[dict[str, geopandas.GeoDataFrame], dict[tuple[str, str, str], geopandas.GeoDataFrame]] or networkx.DiGraph
        Either ``(nodes_dict, edges_dict)`` or an empty directed NetworkX graph.
    """
    edge_key = ("polygon", relation, "point")
    idx = pd.MultiIndex.from_tuples([], names=[polygons_gdf.index.name, points_gdf.index.name])
    empty_edges = gpd.GeoDataFrame(
        {"weight": pd.Series(dtype=float)},
        geometry=gpd.GeoSeries(dtype="geometry"),
        crs=crs,
        index=idx,
    )
    nodes_dict: dict[str, gpd.GeoDataFrame] = {"polygon": polygons_gdf, "point": points_gdf}
    edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame] = {edge_key: empty_edges}
    return (
        gdf_to_nx(nodes=nodes_dict, edges=edges_dict, directed=True)
        if as_nx
        else (nodes_dict, edges_dict)
    )


def _edges_gdf_from_pairs(
    polygons_gdf: gpd.GeoDataFrame,
    points_gdf: gpd.GeoDataFrame,
    pairs: list[tuple[Any, Any]],
    metric: DistanceMetric,
) -> gpd.GeoDataFrame:
    """
    Build an edges GeoDataFrame from ``(polygon_id, point_id)`` pairs.

    The function reuses :class:`GraphBuilder` to guarantee consistent weight
    and geometry calculations regardless of the metric configuration.

    Parameters
    ----------
    polygons_gdf, points_gdf : geopandas.GeoDataFrame
        Input layers that produced the candidate pairs.
    pairs : list[tuple[Any, Any]]
        Iterable of polygon/point index pairs that should form edges.
    metric : DistanceMetric
        Metric wrapper used to derive weights and geometries.

    Returns
    -------
    geopandas.GeoDataFrame
        Edge GeoDataFrame with ``weight`` and ``geometry`` columns. The frame is
        empty when ``pairs`` does not contain any matches.
    """
    # Build a unified temporary nodes GeoDataFrame using polygon centroids and point coordinates
    poly_centroids = polygons_gdf.geometry.centroid
    poly_nodes = gpd.GeoDataFrame(
        {"geometry": poly_centroids}, geometry="geometry", crs=polygons_gdf.crs
    )
    pt_nodes = gpd.GeoDataFrame(
        {"geometry": points_gdf.geometry}, geometry="geometry", crs=points_gdf.crs
    )

    # Namespace node IDs
    poly_index = pd.MultiIndex.from_tuples([("poly", i) for i in polygons_gdf.index])
    pt_index = pd.MultiIndex.from_tuples([("pt", i) for i in points_gdf.index])
    poly_nodes.index = poly_index
    pt_nodes.index = pt_index
    temp_nodes = pd.concat([poly_nodes, pt_nodes])

    # Use GraphBuilder
    builder = GraphBuilder(temp_nodes, metric, directed=True)
    builder.prepare_nodes()

    # Build namespaced edge list
    ns_edge_list = [(("poly", u), ("pt", v)) for u, v in pairs]

    builder.add_edges(ns_edge_list)

    # Extract edge records
    records: list[dict[str, Any]] = []
    index_tuples: list[tuple[Any, Any]] = []
    for u_ns, v_ns, data in builder.G.edges(data=True):
        u_orig = u_ns[1]
        v_orig = v_ns[1]
        index_tuples.append((u_orig, v_orig))
        records.append({"weight": data.get("weight", np.nan), "geometry": data.get("geometry")})

    edge_index = pd.MultiIndex.from_tuples(
        index_tuples,
        names=[polygons_gdf.index.name, points_gdf.index.name],
    )
    return gpd.GeoDataFrame(
        pd.DataFrame(records, index=edge_index)[["weight"]],
        geometry=[rec["geometry"] for rec in records],
        crs=polygons_gdf.crs,
    )


def _containment_pairs(
    points_gdf: gpd.GeoDataFrame,
    polygons_gdf: gpd.GeoDataFrame,
    predicate: str,
) -> list[tuple[Any, Any]]:
    """
    Return ``(polygon_id, point_id)`` pairs using a robust spatial join.

    The extra handling preserves stable indices even when GeoDataFrames have
    custom names or when ``geopandas.sjoin`` introduces helper columns.

    Parameters
    ----------
    points_gdf : geopandas.GeoDataFrame
        Point layer that provides candidate members.
    polygons_gdf : geopandas.GeoDataFrame
        Polygon layer tested for containment relations.
    predicate : str
        Spatial predicate understood by :func:`geopandas.sjoin`.

    Returns
    -------
    list[tuple[Any, Any]]
        Matched polygon/point index pairs.
    """
    predicate_lc = (predicate or "covered_by").lower()
    id_col = polygons_gdf.index.name or "index"
    polys = polygons_gdf.reset_index()

    joined = gpd.sjoin(points_gdf, polys, how="inner", predicate=predicate_lc)
    if joined.empty:
        return []

    poly_ids_series = (
        polys.loc[joined["index_right"], id_col]
        if "index_right" in joined.columns
        else joined[id_col]
    )

    point_ids = joined.index.to_list()
    poly_ids = poly_ids_series.to_list()
    return list(zip(poly_ids, point_ids, strict=False))


def _validate_contiguity_input(gdf: gpd.GeoDataFrame, contiguity: str) -> None:
    """
    Lightweight validation for the contiguity graph public API.

    It guards expensive contiguity operations with fast, descriptive feedback
    so callers can correct their inputs before computations begin.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input polygon layer expected by :func:`contiguity_graph`.
    contiguity : str
        Contiguity type, restricted to ``"queen"`` or ``"rook"``.

    Raises
    ------
    TypeError
        If ``gdf`` is not a GeoDataFrame instance.
    ValueError
        If ``contiguity`` is not an accepted keyword.
    """
    if not isinstance(gdf, gpd.GeoDataFrame):
        msg = (
            f"Input must be a GeoDataFrame, got {type(gdf).__name__}. "
            "Please provide a valid GeoDataFrame with polygon geometries."
        )
        raise TypeError(msg)
    if contiguity not in {"queen", "rook"}:
        msg = "Invalid contiguity type: must be 'queen' or 'rook'"
        raise ValueError(msg)


def _create_spatial_weights(
    gdf: gpd.GeoDataFrame,
    contiguity: str,
) -> libpysal.weights.W:
    """
    Create a libpysal spatial weights matrix for polygon contiguity.

    The wrapper keeps all libpysal-specific details in one place and provides
    early validation of supported contiguity keywords.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Polygon layer defining adjacency.
    contiguity : str
        Either ``"queen"`` or ``"rook"``.

    Returns
    -------
    libpysal.weights.W
        Spatial weights object keyed by polygon indices.
    """
    # Validation handled by _validate_contiguity_input

    ids = list(gdf.index)
    if contiguity.lower() == "queen":
        return libpysal.weights.Queen.from_dataframe(gdf, ids=ids)
    return libpysal.weights.Rook.from_dataframe(gdf, ids=ids)


def _generate_contiguity_edges(
    weights: libpysal.weights.W,
) -> list[EdgePair]:
    """
    Extract adjacency relationships from a spatial weights matrix.

    Converting the libpysal neighbour mapping into a deduplicated edge list
    keeps the rest of the pipeline agnostic to libpysal internals.

    Parameters
    ----------
    weights : libpysal.weights.W
        Spatial weights whose neighbours will be converted into edges.

    Returns
    -------
    list[EdgePair]
        Unique undirected edge tuples implied by the weights object.
    """
    return list(
        {tuple(sorted((src, nbr))) for src, nbrs in weights.neighbors.items() for nbr in nbrs}
    )


def _empty_contiguity_result(
    gdf: gpd.GeoDataFrame,
    contiguity: str,
    *,
    distance_metric: str = "euclidean",
    as_nx: bool,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph:
    """
    Create an empty-but-typed result matching :func:`contiguity_graph` outputs.

    Returning typed empties simplifies downstream consumers that expect CRS,
    metadata, and column consistency even when no polygons remain.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Template polygon GeoDataFrame.
    contiguity : str
        Requested contiguity type.
    distance_metric : str, default "euclidean"
        Metric recorded on the returned graph metadata.
    as_nx : bool
        If True, return an empty NetworkX graph; otherwise return GeoDataFrames.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame] or networkx.Graph
        Empty structure that mirrors the public API contract.
    """
    if as_nx:
        empty_graph = nx.Graph()
        empty_graph.graph["crs"] = gdf.crs
        empty_graph.graph["contiguity"] = contiguity
        empty_graph.graph["distance_metric"] = distance_metric
        return empty_graph

    empty_nodes = gpd.GeoDataFrame(
        columns=gdf.columns.tolist(),
        crs=gdf.crs,
        index=gdf.index[:0],
    )
    empty_edges = gpd.GeoDataFrame(
        columns=["weight", "geometry"],
        crs=gdf.crs,
    )
    return empty_nodes, empty_edges
