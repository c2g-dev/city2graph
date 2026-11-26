"""Compute origin-destination costs on weighted spatial networks."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

import math
import numbers
from typing import Literal

import numpy as np
import pandas as pd
from scipy import sparse

from ._network import GraphInput
from ._network import Points
from ._network import PreparedNetwork
from ._network import _Connections
from ._od import _pairs

__all__ = ["PreparedNetwork", "network_cost_matrix", "network_costs"]


def _route(
    origins: Points | _Connections,
    destinations: Points | _Connections | None,
    graph: GraphInput | PreparedNetwork | None,
    weight: str | None,
    directed: bool | None,
    threshold: float | None,
    k: int | None,
    connect_to: Literal["node", "edge"],
    max_connection_distance: float,
    connect_node_types: str | Sequence[str] | None,
) -> tuple[_Connections, _Connections, np.ndarray, np.ndarray, np.ndarray]:
    """
    Connect the inputs and compute their qualifying routes.

    Prepared graphs can reuse origin attachments across queries.

    Parameters
    ----------
    origins : ndarray
        Connected source vertices in original point order.
    destinations : ndarray
        Connected target vertices in original point order.
    graph : networkx.Graph or spatial graph
        Network supplying arcs and node coordinates.
    weight : str or None
        Edge cost attribute, or None for geometry length.
    directed : bool or None
        Traversal direction override, inferred from NetworkX when omitted.
    threshold : float or None
        Inclusive network cost cutoff.
    k : int or None
        Maximum number of destinations retained per origin.
    connect_to : str
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
        msg = "Supply a graph or PreparedNetwork"
        raise ValueError(msg)
    limit = math.inf if threshold is None else threshold
    if not limit >= 0:
        msg = "threshold must be non-negative"
        raise ValueError(msg)
    if k is not None and (isinstance(k, bool) or not isinstance(k, numbers.Integral) or k <= 0):
        msg = "k must be a positive integer"
        raise ValueError(msg)
    if isinstance(graph, PreparedNetwork):
        if (
            weight is not None
            or directed is not None
            or connect_node_types is not None
            or connect_to != "node"
        ):
            msg = "Set graph options on PreparedNetwork when preparing the graph"
            raise ValueError(msg)
        network = graph
    else:
        network = PreparedNetwork(
            graph,
            weight=weight,
            directed=directed,
            connect_to=connect_to,
            connect_node_types=connect_node_types,
        )

    def connect(points: Points | _Connections) -> _Connections:
        """
        Resolve raw or preconnected query points.

        A tighter limit may discard a previously prepared attachment.

        Parameters
        ----------
        points : GeoSeries or GeoDataFrame
            Spatial points to connect to the network.

        Returns
        -------
        _Connections
            Input labels, connected feature indices, offsets and distances.
        """
        if isinstance(points, _Connections):
            if points.network is not network:
                msg = "Connected points belong to a different PreparedNetwork"
                raise ValueError(msg)
            if max_connection_distance != math.inf and np.any(
                (points.feature >= 0) & (points.distance > max_connection_distance)
            ):
                feature = points.feature.copy()
                feature[points.distance > max_connection_distance] = -1
                return _Connections(network, points.labels, feature, points.along, points.distance)
            return points
        return network.connect(points, max_connection_distance)

    if not max_connection_distance >= 0:
        msg = "max_connection_distance must be non-negative"
        raise ValueError(msg)
    start = connect(origins)
    end = start if destinations is None else connect(destinations)
    rows, cols, costs = _pairs(start, end, limit, k, destinations is None)
    return start, end, rows, cols, costs


def network_costs(
    origins: Points | _Connections,
    destinations: Points | _Connections | None = None,
    graph: GraphInput | PreparedNetwork | None = None,
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
        Points in graph CRS, or the result of PreparedNetwork.connect. Omitting
        destinations routes among origins. With k set, the same row is excluded.
    graph : GeoDataFrame, tuple, NetworkX graph or PreparedNetwork
        Edges, (nodes, edges), a pair of typed layer dictionaries, or a network.
        Set graph options on the constructor when reusing a prepared network.
    weight : str, optional
        Non-negative numeric edge attribute. Defaults to planar geometry length.
    directed : bool, optional
        Follow edge direction. Inferred for NetworkX and false for GeoDataFrames.
    threshold : float, optional
        Inclusive upper bound in edge weight units.
    k : int, optional
        Maximum destinations per origin. Ties follow destination input order.
    connect_to : {"node", "edge"}
        Nearest node or projection onto a line with proportional traversal cost.
    max_connection_distance : float
        Maximum straight-line distance to each endpoint's network connection.
    connect_node_types : str or sequence of str, optional
        Eligible node layers for snapping. Both ends of an eligible edge must
        belong to these layers. All graph layers remain available for traversal.
    include_connection_distance : bool
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
    origins: Points | _Connections,
    destinations: Points | _Connections | None = None,
    graph: GraphInput | PreparedNetwork | None = None,
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
        Points in graph CRS, or the result of PreparedNetwork.connect. Omitting
        destinations routes among origins. With k set, the same row is excluded.
    graph : GeoDataFrame, tuple, NetworkX graph or PreparedNetwork
        Edges, (nodes, edges), a pair of typed layer dictionaries, or a network.
        Set graph options on the constructor when reusing a prepared network.
    weight : str, optional
        Non-negative numeric edge attribute. Defaults to planar geometry length.
    directed : bool, optional
        Follow edge direction. Inferred for NetworkX and false for GeoDataFrames.
    threshold : float, optional
        Inclusive upper bound in edge weight units.
    k : int, optional
        Maximum destinations per origin. Ties follow destination input order.
    connect_to : {"node", "edge"}
        Nearest node or projection onto a line with proportional traversal cost.
    max_connection_distance : float
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
