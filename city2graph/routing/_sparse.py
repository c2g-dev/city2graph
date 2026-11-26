"""Shared sparse shortest-path primitives for spatial networks."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Sequence

    import networkx as nx

from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

from city2graph._processing import row_batches


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
