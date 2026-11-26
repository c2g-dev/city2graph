"""Select reachable origin-destination pairs on a prepared network."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scipy import sparse

import heapq

import numpy as np

from city2graph._processing import row_batches

from ._network import _Connections
from ._network import _fraction_cost
from ._sparse import closest_positions
from ._sparse import distance_batches
from ._sparse import minimum_distances


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
        with end.network._seed_lock:
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
