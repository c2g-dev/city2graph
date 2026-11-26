"""Recover path geometries with NetworkX-compatible predecessor ordering."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Hashable
    from collections.abc import Iterator
    from collections.abc import Sequence

    from scipy import sparse

import math
from typing import Any
from typing import Literal

import networkx as nx
import numpy as np
import numpy.typing as npt

from ._sparse import distance_batches
from ._sparse import sparse_adjacency

# Unbounded searches from many origins reuse one sparse matrix; NetworkX remains
# the reference for distances and for the paths chosen among equal-length routes.


def _first_predecessor_path(
    source: Hashable, target: Hashable, predecessors: dict[Any, list[Any]]
) -> list[Any] | None:
    """
    Return the path selected by the first NetworkX predecessors.

    The first predecessor of each node is the one that reached it at its final
    distance, which is the path NetworkX builds from strict improvements.

    Parameters
    ----------
    source : hashable
        Search origin.
    target : hashable
        Search destination.
    predecessors : dict
        Predecessor lists from :func:`networkx.dijkstra_predecessor_and_distance`.

    Returns
    -------
    list or None
        Nodes from source to target, or None for an unreached target.
    """
    if target not in predecessors:
        return None
    path = [target]
    while path[-1] != source:
        path.append(predecessors[path[-1]][0])
    path.reverse()
    return path


class NetworkXSearch:
    """
    Single-origin search computed by NetworkX.

    The search retains the predecessor lists and lengths of one Dijkstra
    traversal so that distances and paths share the same traversal.

    Parameters
    ----------
    graph : networkx.Graph
        Network to search.
    source : hashable
        Search origin.
    weight : str
        Edge weight attribute.
    cutoff : float, optional
        Maximum explored distance; nodes at the cutoff are included.
    """

    def __init__(
        self, graph: nx.Graph, source: Hashable, weight: str, cutoff: float | None
    ) -> None:
        """
        Run one Dijkstra traversal and retain its predecessor tree.

        Nodes beyond ``cutoff`` are not reached and have no predecessors.

        Parameters
        ----------
        graph : networkx.Graph
            Network to search.
        source : hashable
            Search origin.
        weight : str
            Edge weight attribute.
        cutoff : float, optional
            Maximum explored distance; nodes at the cutoff are included.
        """
        self.source = source
        self.predecessors, self.lengths = nx.dijkstra_predecessor_and_distance(
            graph, source, cutoff=cutoff, weight=weight
        )

    def distances(self, targets: Sequence[Hashable]) -> npt.NDArray[np.float64]:
        """
        Return distances to targets, with ``inf`` for unreached nodes.

        Targets keep their order, and repeated targets repeat their distance.

        Parameters
        ----------
        targets : sequence of hashable
            Destination nodes.

        Returns
        -------
        numpy.ndarray
            Distance for each target in order.
        """
        return np.asarray([self.lengths.get(t, np.inf) for t in targets], dtype=float)

    def distance(self, target: Hashable) -> float:
        """
        Return the distance to one target.

        Targets outside the search, including those beyond its cutoff, are
        reported as unreached.

        Parameters
        ----------
        target : hashable
            Destination node.

        Returns
        -------
        float
            Shortest-path length, or ``inf`` when unreached.
        """
        length: float = self.lengths.get(target, math.inf)
        return length

    def paths(self, targets: Sequence[Hashable]) -> list[list[Any] | None]:
        """
        Return the first-predecessor path to each target.

        Paths start at the search origin and end at the target.

        Parameters
        ----------
        targets : sequence of hashable
            Destination nodes.

        Returns
        -------
        list
            Node path, or None for an unreached target, for each target.
        """
        return [_first_predecessor_path(self.source, t, self.predecessors) for t in targets]


class SparseNetwork:
    """
    Directed CSR adjacency built once from a NetworkX graph.

    Undirected edges are stored in both directions. Parallel edges keep their
    minimum weight, as NetworkX does when it searches a multigraph. Edges with
    infinite weight are omitted. Explicit zero weights remain edges.

    Parameters
    ----------
    graph : networkx.Graph
        Source network, which may be directed or a multigraph.
    weight : str
        Edge weight attribute; missing values count as 1, as in NetworkX.
    nodes : list
        Node order of the matrix.
    matrix : scipy.sparse.csr_matrix
        Weighted adjacency from row to column.
    has_infinite : bool
        Whether infinite-weight edges were omitted.
    integral : bool
        Whether every weight is an integer, so finite distances are integers.
    """

    def __init__(
        self,
        graph: nx.Graph,
        weight: str,
        nodes: list[Any],
        matrix: sparse.csr_matrix,
        has_infinite: bool,
        integral: bool,
    ) -> None:
        """
        Store the converted network and its node order.

        Use :meth:`from_graph` to build the matrix from a NetworkX graph.

        Parameters
        ----------
        graph : networkx.Graph
            Source network, retained for NetworkX path decisions.
        weight : str
            Edge weight attribute.
        nodes : list
            Node order of the matrix.
        matrix : scipy.sparse.csr_matrix
            Weighted adjacency from row to column.
        has_infinite : bool
            Whether infinite-weight edges were omitted.
        integral : bool
            Whether every weight is an integer, so finite distances are integers.
        """
        self.graph = graph
        self.weight = weight
        self.nodes = nodes
        self.index = {node: i for i, node in enumerate(nodes)}
        self.matrix = matrix
        self.has_infinite = has_infinite
        self.integral = integral
        self._incoming: tuple[list[int], list[int], list[float]] | None = None

    @classmethod
    def from_graph(cls, graph: nx.Graph, weight: str) -> SparseNetwork | None:
        """
        Convert a NetworkX graph, or return None if NetworkX must be kept.

        Weights must be all floats or all integers. NetworkX adds them with the
        same rounding as the sparse search and returns floats or integers in the
        same way; mixed or other numeric types keep NetworkX.

        Parameters
        ----------
        graph : networkx.Graph
            Network with non-negative weights.
        weight : str
            Edge weight attribute.

        Returns
        -------
        SparseNetwork or None
            Converted network, or None when the weights cannot be converted exactly.
        """
        nodes = list(graph)
        converted = sparse_adjacency(graph, weight, {node: i for i, node in enumerate(nodes)})
        if converted is None:
            return None
        return cls(graph, weight, nodes, *converted)

    def searches(
        self,
        sources: Sequence[Hashable],
        targets: Sequence[Hashable],
        cutoff: float | None = None,
    ) -> Iterator[tuple[SparseSearch, npt.NDArray[np.float64]]]:
        """
        Yield searches and target distances in source order.

        Sources are processed in batches so that temporary distance and
        predecessor arrays stay bounded.

        Parameters
        ----------
        sources : sequence of hashable
            Search origins; repeated origins are searched again.
        targets : sequence of hashable
            Destination nodes whose distances are returned for each origin.
        cutoff : float, optional
            Maximum explored distance; nodes at the cutoff are included.

        Yields
        ------
        tuple[SparseSearch, numpy.ndarray]
            Search for one origin and its distances to ``targets``.
        """
        source_index = np.asarray([self.index[s] for s in sources], dtype=np.int64)
        target_index = np.asarray([self.index[t] for t in targets], dtype=np.int64)
        limit = np.inf if cutoff is None else cutoff
        for batch, distances, predecessors in distance_batches(
            self.matrix, source_index, limit, paths=True
        ):
            assert predecessors is not None
            for offset, position in enumerate(range(batch.start, batch.stop)):
                search = SparseSearch(
                    self, sources[position], distances[offset], predecessors[offset], cutoff
                )
                yield search, distances[offset][target_index]

    def incoming(self) -> tuple[list[int], list[int], list[float]]:
        """
        Return incoming adjacency lists for predecessor checks.

        The lists are built on first use and reused by every search.

        Returns
        -------
        tuple[list[int], list[int], list[float]]
            CSR pointer, source node positions and weights by target node.
        """
        if self._incoming is None:
            reverse = self.matrix.transpose().tocsr()
            self._incoming = (
                reverse.indptr.tolist(),
                reverse.indices.tolist(),
                reverse.data.tolist(),
            )
        return self._incoming


class SparseSearch:
    """
    Single-origin search from a sparse batch, with NetworkX path parity.

    NetworkX keeps the first predecessor that reaches each node at its final
    distance. With non-negative weights, that is the settled predecessor with the
    smallest distance. When several predecessors share that distance, the choice
    depends on traversal order, so those paths are recomputed by NetworkX with
    the furthest affected target as the cutoff.

    Parameters
    ----------
    network : SparseNetwork
        Network that produced the search.
    source : hashable
        Search origin.
    distances : numpy.ndarray
        Distances from the origin to every network node.
    predecessors : numpy.ndarray
        SciPy predecessor positions for every network node.
    cutoff : float, optional
        Maximum distance explored by the search.
    """

    def __init__(
        self,
        network: SparseNetwork,
        source: Hashable,
        distances: npt.NDArray[np.float64],
        predecessors: npt.NDArray[np.int32],
        cutoff: float | None = None,
    ) -> None:
        """
        Retain one row of a batched search.

        The row arrays are views into the batch that produced them.

        Parameters
        ----------
        network : SparseNetwork
            Network that produced the search.
        source : hashable
            Search origin.
        distances : numpy.ndarray
            Distances from the origin to every network node.
        predecessors : numpy.ndarray
            SciPy predecessor positions for every network node.
        cutoff : float, optional
            Maximum distance explored by the search.
        """
        self.network = network
        self.source = source
        self.cutoff = cutoff
        self._distances = distances
        self._predecessors = predecessors
        self._source_index = network.index[source]
        self._unique: set[int] = set()
        self._fallback: tuple[float, dict[Any, list[Any]]] | None = None

    def distance(self, target: Hashable) -> float:
        """
        Return the distance to one target.

        Targets outside the search, including those beyond its cutoff, are
        reported as unreached.

        Parameters
        ----------
        target : hashable
            Destination node.

        Returns
        -------
        float
            Shortest-path length, or ``inf`` when unreached. Finite lengths on
            integer-weighted networks are integers, as in NetworkX.
        """
        length = float(self._distances[self.network.index[target]])
        if self.network.integral and length < math.inf:
            return int(length)
        return length

    def paths(self, targets: Sequence[Hashable]) -> list[list[Any] | None]:
        """
        Return the path NetworkX would select to each target.

        Targets with tied predecessors share one bounded NetworkX search.

        Parameters
        ----------
        targets : sequence of hashable
            Destination nodes.

        Returns
        -------
        list
            Node path, or None for an unreached target, for each target.
        """
        paths: list[list[Any] | None] = []
        pending: list[int] = []
        for position, target in enumerate(targets):
            path = self._sparse_path(self.network.index[target])
            if path is False:
                pending.append(position)
                paths.append(None)
            else:
                paths.append(path)
        if pending:
            bound = max(self.distance(targets[p]) for p in pending)
            predecessors = self._networkx_predecessors(bound)
            for position in pending:
                paths[position] = _first_predecessor_path(
                    self.source, targets[position], predecessors
                )
        return paths

    def _sparse_path(self, target: int) -> list[Any] | Literal[False] | None:
        """
        Follow SciPy predecessors when they match the NetworkX choice.

        Each node on the path must have exactly one incoming edge that reaches
        it at its final distance from a node no further than the SciPy
        predecessor.

        Parameters
        ----------
        target : int
            Destination node position.

        Returns
        -------
        list, None or bool
            Node path, None for an unreachable target, or False when NetworkX
            must decide the path.
        """
        distances = self._distances
        if distances[target] == np.inf:
            # Infinite-weight edges are omitted here but traversed by NetworkX.
            return False if self.network.has_infinite else None
        pointer, sources, weights = self.network.incoming()
        path = [target]
        node = target
        while node != self._source_index:
            previous = int(self._predecessors[node])
            if node not in self._unique:
                bound = distances[previous]
                reached = distances[node]
                tied = 0
                for j in range(pointer[node], pointer[node + 1]):
                    other = sources[j]
                    if (
                        other != node
                        and distances[other] <= bound
                        and distances[other] + weights[j] == reached
                    ):
                        tied += 1
                if tied != 1:
                    return False
                self._unique.add(node)
            path.append(previous)
            node = previous
        nodes = self.network.nodes
        return [nodes[i] for i in reversed(path)]

    def _networkx_predecessors(self, bound: float) -> dict[Any, list[Any]]:
        """
        Return NetworkX predecessors for nodes within ``bound``.

        A cutoff leaves the predecessor order of nodes within it unchanged.

        Parameters
        ----------
        bound : float
            Largest distance whose path is required.

        Returns
        -------
        dict
            NetworkX predecessor lists.
        """
        if self._fallback is None or self._fallback[0] < bound:
            cutoff = self.cutoff if bound == np.inf else bound
            predecessors, _ = nx.dijkstra_predecessor_and_distance(
                self.network.graph, self.source, cutoff=cutoff, weight=self.network.weight
            )
            self._fallback = (bound, predecessors)
        return self._fallback[1]
