"""
Module for metapath-based graph operations.

This module provides functionality for adding metapath-derived edges to heterogeneous graphs,
supporting both weighted and structural metapaths.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

if TYPE_CHECKING:
    from collections.abc import Callable

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csgraph
from shapely.geometry import LineString

from city2graph.utils import gdf_to_nx
from city2graph.utils import nx_to_gdf
from city2graph.utils import validate_nx

logger = logging.getLogger(__name__)

__all__ = [
    "add_metapaths",
    "add_metapaths_by_weight",
]


def add_metapaths(
    graph: (
        tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]]
        | nx.Graph
        | nx.MultiGraph
        | None
    ) = None,
    nodes: dict[str, gpd.GeoDataFrame] | None = None,
    edges: dict[tuple[str, str, str], gpd.GeoDataFrame] | None = None,
    sequence: list[tuple[str, str, str]] | None = None,
    new_relation_name: str | None = None,
    edge_attr: str | list[str] | None = None,
    edge_attr_agg: str | object | None = "sum",
    directed: bool = False,
    trace_path: bool = False,
    multigraph: bool = False,
    as_nx: bool = False,
    **_: object,
) -> (
    nx.Graph
    | nx.MultiGraph
    | tuple[
        dict[str, gpd.GeoDataFrame],
        dict[tuple[str, str, str], gpd.GeoDataFrame],
    ]
):
    """
    Add metapath-derived edges to a heterogeneous graph.

    The operation multiplies typed adjacency tables to connect terminal node
    pairs and can aggregate additional numeric edge attributes along the way.

    Parameters
    ----------
    graph : tuple or networkx.Graph or networkx.MultiGraph, optional
        Heterogeneous graph input expressed as typed GeoDataFrame dictionaries or
        a city2graph-compatible NetworkX graph.
    nodes : dict[str, geopandas.GeoDataFrame], optional
        Dictionary of node GeoDataFrames.
    edges : dict[tuple[str, str, str], geopandas.GeoDataFrame], optional
        Dictionary of edge GeoDataFrames.
    sequence : list[tuple[str, str, str]]
        Sequence of metapath specifications; every edge type is a
        ``(src_type, relation, dst_type)`` tuple and the path must contain at
        least two steps.
    new_relation_name : str, optional
        Target edge relation name for the new metapath edges.
        If None (default), edges are named ``metapath_0``.
    edge_attr : str | list[str] | None, optional
        Numeric edge attributes to aggregate along metapaths. When ``None``, only
        path weights are produced.
    edge_attr_agg : str | object | None, optional
        Aggregation strategy for ``edge_attr`` columns. Supported values are
        ``"sum"`` and ``"mean"`` (default ``"sum"``).
    directed : bool, optional
        Treat metapaths as directed when ``True``; otherwise both edge
        directions are accepted when available in the input graph.
    trace_path : bool, optional
        When ``True``, attempt to create traced geometries. Currently ignored but
        retained for API compatibility.
    multigraph : bool, optional
        When returning NetworkX data, build a ``networkx.MultiGraph`` if ``True``.
    as_nx : bool, optional
        Return the result as a NetworkX graph when ``True``.
    **_ : object
        Ignored placeholder for future keyword extensions.

    Returns
    -------
    tuple[dict[str, geopandas.GeoDataFrame], dict[tuple[str, str, str], geopandas.GeoDataFrame]] | networkx.Graph | networkx.MultiGraph
        The graph with metapath-derived edges.
        If as_nx is False (default), returns a tuple of node and edge GeoDataFrames.
        If as_nx is True, returns a NetworkX graph (Graph or MultiGraph).

    Notes
    -----
    Legacy scaffolding for path-tracing geometries has been removed because it
    was never executed. The trace_path argument is preserved for API
    compatibility but remains a no-op while straight-line geometries are
    generated for all metapath edges.
    """
    if trace_path:  # pragma: no cover
        logger.debug("trace_path option is not implemented; ignoring request.")

    if sequence is None:
        msg = "sequence must be provided"
        raise ValueError(msg)

    nodes_dict, edges_dict = _ensure_hetero_dict(graph, nodes, edges)

    if not sequence or not edges_dict:
        return _finalize_metapath_result(
            nodes_dict,
            edges_dict,
            as_nx,
            multigraph,
            directed,
            None,
        )

    edge_attrs = _normalize_edge_attrs(edge_attr)
    aggregation = _resolve_edge_attr_agg(edge_attr_agg)

    updated_edges = dict(edges_dict)
    metapath_metadata: dict[tuple[str, str, str], dict[str, object]] = {}

    edge_key, result_gdf, metadata_entry = _materialize_metapath(
        mp_index=0,
        metapath=sequence,
        nodes=nodes_dict,
        edges=edges_dict,
        directed=directed,
        edge_attrs=edge_attrs,
        aggregation=aggregation,
        new_relation_name=new_relation_name,
    )
    updated_edges[edge_key] = result_gdf
    metapath_metadata[edge_key] = metadata_entry

    return _finalize_metapath_result(
        nodes_dict,
        updated_edges,
        as_nx,
        multigraph,
        directed,
        metapath_metadata,
    )


def add_metapaths_by_weight(  # noqa: PLR0913
    graph: (
        tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]]
        | nx.Graph
        | nx.MultiGraph
        | None
    ) = None,
    nodes: dict[str, gpd.GeoDataFrame] | None = None,
    edges: dict[tuple[str, str, str], gpd.GeoDataFrame] | None = None,
    weight: str | None = None,
    threshold: float | None = None,
    new_relation_name: str | None = None,
    min_threshold: float = 0.0,
    edge_types: list[tuple[str, str, str]] | None = None,
    endpoint_type: str | None = None,
    directed: bool = False,
    multigraph: bool = False,
    as_nx: bool = False,
) -> (
    nx.Graph
    | nx.MultiGraph
    | tuple[
        dict[str, gpd.GeoDataFrame],
        dict[tuple[str, str, str], gpd.GeoDataFrame],
    ]
):
    """
    Connect nodes of a specific type if they are reachable within a cost threshold band.

    This function dynamically adds metapaths (edges) between nodes of a specified
    `endpoint_type` if they are reachable within a given cost band [`min_threshold`,
    `threshold`] based on edge weights (e.g., travel time). It uses Dijkstra's
    algorithm for path finding via `scipy.sparse.csgraph` for efficiency.

    Parameters
    ----------
    graph : tuple or networkx.Graph or networkx.MultiGraph, optional
        Input graph. Can be a tuple of (nodes_dict, edges_dict) or a NetworkX graph.
    nodes : dict[str, geopandas.GeoDataFrame], optional
        Dictionary of node GeoDataFrames.
    edges : dict[tuple[str, str, str], geopandas.GeoDataFrame], optional
        Dictionary of edge GeoDataFrames.
    weight : str
        The edge attribute to use as weight (e.g., 'travel_time').
    threshold : float
        The maximum cost threshold for connection.
    new_relation_name : str, optional
        Name of the new edge relation.
    min_threshold : float, default 0.0
        The minimum cost threshold for connection.
    edge_types : list[tuple[str, str, str]], optional
        List of edge types to consider for traversal. If None, all edges are used.
    endpoint_type : str, optional
        The node type to connect (e.g., 'building').
    directed : bool, default False
        If True, creates a directed graph for traversal.
    multigraph : bool, default False
        If True, returns a MultiGraph (only relevant if as_nx=True).
    as_nx : bool, default False
        If True, returns a NetworkX graph.

    Returns
    -------
    nx.Graph or nx.MultiGraph or tuple
        The graph with added metapaths. Format depends on `as_nx` parameter.
    """
    if weight is None:
        msg = "weight must be provided"
        raise ValueError(msg)
    if threshold is None:
        msg = "threshold must be provided"
        raise ValueError(msg)

    # Normalize input to graph tuple or nx graph
    if nodes is not None:
        if graph is not None:
            msg = "Cannot provide both 'graph' and 'nodes'/'edges'."
            raise ValueError(msg)
        graph = (nodes, edges or {})
    elif graph is None:
        msg = "Either 'graph' or 'nodes' (and optionally 'edges') must be provided."
        raise ValueError(msg)
    # Validate and resolve parameters
    params = _validate_and_resolve_parameters(
        new_relation_name, endpoint_type, min_threshold, threshold
    )
    if params is None:
        return _return_original_graph(graph)

    endpoint_type, target_edge_type_tuple = params

    # Prepare data for sparse matrix construction
    (
        _node_to_idx,
        idx_to_node,
        endpoint_indices,
        row_indices,
        col_indices,
        data,
        num_nodes,
    ) = _prepare_sparse_graph_data(graph, endpoint_type, edge_types, weight)

    if not data:
        return _return_original_graph(graph)

    if not endpoint_indices:
        logger.warning("No nodes of type '%s' found.", endpoint_type)
        return _return_original_graph(graph)

    # Build sparse matrix
    adj_matrix = sparse.csr_matrix((data, (row_indices, col_indices)), shape=(num_nodes, num_nodes))

    # Run Dijkstra's algorithm
    dist_matrix = csgraph.dijkstra(
        csgraph=adj_matrix,
        directed=directed or (isinstance(graph, nx.Graph) and graph.is_directed()),
        indices=endpoint_indices,
        limit=threshold,
    )

    # Extract metapath edges
    new_edges_data = _extract_metapath_edges(
        dist_matrix,
        endpoint_indices,
        idx_to_node,
        min_threshold,
        threshold,
        graph,
        directed,
        target_edge_type_tuple,
        weight,
    )

    # Construct and return result
    return _construct_metapath_result(
        graph,
        new_edges_data,
        as_nx,
        directed,
        multigraph,
        target_edge_type_tuple,
        weight,
        endpoint_type,
    )


# -----------------------------------------------------------------------------
# Helpers for add_metapaths
# -----------------------------------------------------------------------------


def _ensure_hetero_dict(
    graph: (
        tuple[
            dict[str, gpd.GeoDataFrame],
            dict[tuple[str, str, str], gpd.GeoDataFrame],
        ]
        | nx.Graph
        | nx.MultiGraph
        | None
    ) = None,
    nodes: dict[str, gpd.GeoDataFrame] | None = None,
    edges: dict[tuple[str, str, str], gpd.GeoDataFrame] | None = None,
) -> tuple[
    dict[str, gpd.GeoDataFrame],
    dict[tuple[str, str, str], gpd.GeoDataFrame],
]:
    """
    Normalize supported inputs to hetero GeoDataFrame dictionaries.

    Ensures callers can work with a predictable pair of hetero mappings.

    Parameters
    ----------
    graph : tuple or networkx.Graph or networkx.MultiGraph, optional
        Heterogeneous graph representation.
    nodes : dict[str, geopandas.GeoDataFrame], optional
        Dictionary of node GeoDataFrames.
    edges : dict[tuple[str, str, str], geopandas.GeoDataFrame], optional
        Dictionary of edge GeoDataFrames.

    Returns
    -------
    tuple[dict[str, geopandas.GeoDataFrame], dict[tuple[str, str, str], geopandas.GeoDataFrame]]
        Normalised node and edge dictionaries.
    """
    if graph is None and nodes is None:
        msg = "Either 'graph' or 'nodes' (and optionally 'edges') must be provided."
        raise ValueError(msg)

    if nodes is not None:
        if graph is not None:
            msg = "Cannot provide both 'graph' and 'nodes'/'edges'."
            raise ValueError(msg)
        return nodes, edges or {}

    if isinstance(graph, tuple):
        if len(graph) != 2:
            msg = "Graph tuple must contain (nodes_dict, edges_dict)"
            raise ValueError(msg)

        nodes_dict, raw_edges = graph

        if not isinstance(nodes_dict, dict):
            msg = "nodes_dict must be a dictionary"
            raise TypeError(msg)

        if raw_edges is None:
            empty_edges: dict[tuple[str, str, str], gpd.GeoDataFrame] = {}
            return nodes_dict, empty_edges

        if not isinstance(raw_edges, dict):
            msg = "edges_dict must be a dictionary"
            raise TypeError(msg)

        normalized_edges = cast(
            "dict[tuple[str, str, str], gpd.GeoDataFrame]",
            raw_edges,
        )

        return nodes_dict, normalized_edges

    if isinstance(graph, (nx.Graph, nx.MultiGraph, nx.DiGraph, nx.MultiDiGraph)):
        validate_nx(graph)
        nodes_data, edges_data = nx_to_gdf(graph)

        if not isinstance(nodes_data, dict):
            msg = "add_metapaths requires a heterogeneous graph with typed nodes"
            raise TypeError(msg)

        if edges_data is None:
            return nodes_data, {}

        if not isinstance(edges_data, dict):
            msg = "add_metapaths requires a heterogeneous graph with typed edges"
            raise TypeError(msg)

        normalized_edges_graph = cast(
            "dict[tuple[str, str, str], gpd.GeoDataFrame]",
            edges_data,
        )
        return nodes_data, normalized_edges_graph

    msg = "Unsupported graph input type for add_metapaths"
    raise TypeError(msg)


@dataclass(slots=True)
class _EdgeAttrAggregation:
    """
    Container describing how to reduce edge attributes along a metapath.

    Stores the callable reducers used during metapath aggregation.
    """

    tag: object
    row_reducer: Callable[[pd.DataFrame], pd.Series]
    group_reducer: str | Callable[[pd.Series], float]


def _resolve_edge_attr_agg(
    edge_attr_agg: str | object | None,
) -> _EdgeAttrAggregation:
    """
    Normalise the aggregation option for edge attributes.

    Translate the user supplied definition into reusable reducers.

    Parameters
    ----------
    edge_attr_agg : str | object | None
        Aggregation strategy requested by the caller. Supported values are
        ``"sum"``, ``"mean"``, a callable reducer, or ``None`` which defaults to
        ``"sum"``.

    Returns
    -------
    _EdgeAttrAggregation
        Resolved aggregation metadata containing callable reducers for row and
        group operations.

    Raises
    ------
    ValueError
        If a string option is supplied that is not ``"sum"`` or ``"mean"``.
    TypeError
        If the option is neither a recognised string, a callable, nor ``None``.
    """
    option = edge_attr_agg if edge_attr_agg is not None else "sum"

    if isinstance(option, str):
        normalized = option.lower()
        if normalized == "sum":
            return _EdgeAttrAggregation("sum", _row_reduce_sum, "sum")
        if normalized == "mean":
            return _EdgeAttrAggregation("mean", _row_reduce_mean, "mean")
        msg = f"Unsupported edge_attr_agg '{option}'"
        raise ValueError(msg)

    if callable(option):
        return _EdgeAttrAggregation(
            option,
            functools.partial(_row_reduce_callable, func=option),
            functools.partial(_group_reduce_callable, func=option),
        )

    msg = "edge_attr_agg must be 'sum', 'mean', a callable, or None"
    raise TypeError(msg)


def _row_reduce_sum(block: pd.DataFrame) -> pd.Series:
    """
    Return row-wise sums for a hop attribute block.

    Missing values are treated as zeros before summation.

    Parameters
    ----------
    block : pandas.DataFrame
        Normalised attribute columns for a single hop across multiple paths.

    Returns
    -------
    pandas.Series
        Row-wise sums with missing values treated as zeros.
    """
    numeric = block.apply(pd.to_numeric, errors="coerce")
    result = numeric.fillna(0.0).sum(axis=1)
    return cast("pd.Series", result)


def _row_reduce_mean(block: pd.DataFrame) -> pd.Series:
    """
    Return row-wise means while skipping missing values.

    Each row is reduced to the arithmetic mean of its valid entries.

    Parameters
    ----------
    block : pandas.DataFrame
        Normalised attribute columns for a single hop across multiple paths.

    Returns
    -------
    pandas.Series
        Row-wise means calculated with ``NaN`` values ignored.
    """
    numeric = block.apply(pd.to_numeric, errors="coerce")
    result = numeric.mean(axis=1, skipna=True)
    return cast("pd.Series", result)


def _row_reduce_callable(
    block: pd.DataFrame,
    *,
    func: Callable[[np.ndarray], float],
) -> pd.Series:
    """
    Apply a custom reducer to each row of a hop attribute block.

    Allows user supplied reducers to participate in per-path aggregation.

    Parameters
    ----------
    block : pandas.DataFrame
        Normalised attribute columns for a single hop across multiple paths.
    func : Callable[[numpy.ndarray], float]
        Callable that reduces the non-null numeric values in a row to a scalar.

    Returns
    -------
    pandas.Series
        Row-wise reductions produced by ``func``.
    """
    numeric = block.apply(pd.to_numeric, errors="coerce")
    # Avoid passing `func` as a keyword to DataFrame.apply since it clashes with
    # the DataFrame.apply(func=...) parameter name itself, causing
    # "multiple values for argument 'func'" TypeError. Use a lambda to forward
    # the user-supplied reducer to our row helper.
    result = numeric.apply(lambda row: _apply_callable_row(row, func=func), axis=1)
    return cast("pd.Series", result)


def _apply_callable_row(
    row: pd.Series,
    *,
    func: Callable[[np.ndarray], float],
) -> float:
    """
    Reduce a single hop row using ``func`` while handling empty inputs.

    Helper used by :func:`_row_reduce_callable` to evaluate user reducers.

    Parameters
    ----------
    row : pandas.Series
        Hop attribute values for a single metapath traversal.
    func : Callable[[numpy.ndarray], float]
        Callable that reduces numeric values to a scalar result.

    Returns
    -------
    float
        Reduced value for the row; ``NaN`` when all values are missing.
    """
    valid = row.dropna().to_numpy()
    if valid.size == 0:
        return float("nan")
    return float(func(valid))


def _group_reduce_callable(
    series: pd.Series,
    *,
    func: Callable[[np.ndarray], float],
) -> float:
    """
    Reduce grouped path values with ``func`` while ignoring missing data.

    Ensures custom reducers can be applied during terminal node aggregation.

    Parameters
    ----------
    series : pandas.Series
        Row reductions belonging to the same terminal node pair.
    func : Callable[[numpy.ndarray], float]
        Callable that combines numeric values into a scalar summary.

    Returns
    -------
    float
        Aggregated value for the group; ``NaN`` when the group is empty.
    """
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float(func(numeric.to_numpy()))


def _safe_linestring(start_geom: object, end_geom: object) -> LineString | None:
    """
    Safely build a ``LineString`` between two geometries when possible.

    Invalid or missing geometries result in ``None``.

    Parameters
    ----------
    start_geom : object
        Geometry for the source node of the metapath edge.
    end_geom : object
        Geometry for the destination node of the metapath edge.

    Returns
    -------
    shapely.geometry.LineString | None
        Straight segment between the input geometries, or ``None`` when either
        geometry is missing or invalid.
    """
    if start_geom is None or end_geom is None:
        return None
    if getattr(start_geom, "is_empty", False) or getattr(end_geom, "is_empty", False):
        return None
    try:
        return LineString([start_geom, end_geom])
    except (TypeError, ValueError):
        return None


def _normalize_edge_attrs(edge_attr: str | list[str] | None) -> list[str] | None:
    """
    Normalise edge attribute selection to a list.

    Produces predictable iterable input for downstream helpers.

    Parameters
    ----------
    edge_attr : str | list[str] | None
        User-supplied edge attribute selector, a single column name, a list of
        names, or ``None``.

    Returns
    -------
    list[str] | None
        Normalised list of attribute names, or ``None`` when no attributes were
        requested.
    """
    if edge_attr is None:
        return None
    if isinstance(edge_attr, str):
        return [edge_attr]
    return list(edge_attr)


def _materialize_metapath(
    *,
    mp_index: int,
    metapath: list[tuple[str, str, str]],
    nodes: dict[str, gpd.GeoDataFrame],
    edges: dict[tuple[str, str, str], gpd.GeoDataFrame],
    directed: bool,
    edge_attrs: list[str] | None,
    aggregation: _EdgeAttrAggregation,
    new_relation_name: str | None = None,
) -> tuple[tuple[str, str, str], gpd.GeoDataFrame, dict[str, object]]:
    """
    Materialise a single metapath into an aggregated GeoDataFrame.

    Collects hop data, aggregates attributes, and attaches straight geometries.

    Parameters
    ----------
    mp_index : int
        Index of the metapath.
    metapath : list[tuple[str, str, str]]
        List of edge types defining the metapath.
    nodes : dict[str, geopandas.GeoDataFrame]
        Dictionary of node GeoDataFrames.
    edges : dict[tuple[str, str, str], geopandas.GeoDataFrame]
        Dictionary of edge GeoDataFrames.
    directed : bool
        Whether the graph is directed.
    edge_attrs : list[str] or None
        List of edge attributes to aggregate.
    aggregation : _EdgeAttrAggregation
        Aggregation strategy.
    new_relation_name : str, optional
        Target edge relation name.

    Returns
    -------
    tuple
        Tuple containing the edge key, the result GeoDataFrame, and metadata.
    """
    if len(metapath) < 2:
        msg = "Each metapath must contain at least two edge types"
        raise ValueError(msg)

    start_type = metapath[0][0]
    end_type = metapath[-1][-1]

    if new_relation_name is not None:
        edge_key = (start_type, new_relation_name, end_type)
    else:
        edge_key = (start_type, f"metapath_{mp_index}", end_type)
    metadata = {
        "metapath_spec": tuple(metapath),
        "edge_attr": None if edge_attrs is None else tuple(edge_attrs),
        "edge_attr_agg": aggregation.tag,
    }

    # 1. Build canonical frames for each hop
    frames: list[pd.DataFrame] = []
    start_index_name = f"{start_type}_id"
    end_index_name = f"{end_type}_id"

    for step_idx, edge_type in enumerate(metapath):
        edge_gdf, reversed_lookup = _get_edge_frame(edges, edge_type, directed)

        # Update index names from the first/last hop
        if step_idx == 0:
            idx_name = edge_gdf.index.names[1 if reversed_lookup else 0]
            start_index_name = _normalise_index_name(idx_name, f"{start_type}_id")
        if step_idx == len(metapath) - 1:
            idx_name = edge_gdf.index.names[0 if reversed_lookup else 1]
            end_index_name = _normalise_index_name(idx_name, f"{end_type}_id")

        frame = _build_hop_frame(
            edge_gdf=edge_gdf,
            step_idx=step_idx,
            reversed_lookup=reversed_lookup,
            edge_attrs=edge_attrs,
        )
        if frame.empty:
            return (
                edge_key,
                _empty_metapath_gdf(
                    nodes, start_type, end_type, edge_attrs, start_index_name, end_index_name
                ),
                metadata,
            )
        frames.append(frame)

    # 2. Join hops
    joined = frames[0]
    for idx in range(1, len(frames)):
        joined = joined.merge(
            frames[idx],
            left_on=f"dst_{idx - 1}",
            right_on=f"src_{idx}",
            how="inner",
            copy=False,
        )
        # Drop intermediate join columns to save memory
        joined = joined.drop(columns=[f"dst_{idx - 1}", f"src_{idx}"], errors="ignore")

        if joined.empty:
            return (
                edge_key,
                _empty_metapath_gdf(
                    nodes, start_type, end_type, edge_attrs, start_index_name, end_index_name
                ),
                metadata,
            )

    # 3. Aggregate paths
    aggregated = _aggregate_paths(
        joined,
        step_count=len(frames),
        edge_attrs=edge_attrs,
        aggregation=aggregation,
        start_index_name=start_index_name,
        end_index_name=end_index_name,
    )

    if aggregated.empty:
        return (
            edge_key,
            _empty_metapath_gdf(
                nodes, start_type, end_type, edge_attrs, start_index_name, end_index_name
            ),
            metadata,
        )

    # 4. Attach geometry
    result = _attach_metapath_geometry(aggregated, nodes, start_type, end_type)
    return edge_key, result, metadata


def _get_edge_frame(
    edges: dict[tuple[str, str, str], gpd.GeoDataFrame],
    edge_type: tuple[str, str, str],
    directed: bool,
) -> tuple[gpd.GeoDataFrame, bool]:
    """
    Fetch the GeoDataFrame for an edge type, optionally using the reverse.

    Checks for the edge type in the dictionary. If not found and the graph is
    undirected, checks for the reverse edge type.

    Parameters
    ----------
    edges : dict[tuple[str, str, str], geopandas.GeoDataFrame]
        Dictionary of edge GeoDataFrames.
    edge_type : tuple[str, str, str]
        The edge type to fetch.
    directed : bool
        Whether the graph is directed.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, bool]
        The edge GeoDataFrame and a boolean indicating if it is reversed.
    """
    if edge_type in edges:
        return edges[edge_type], False

    if not directed:
        reverse_key = (edge_type[2], edge_type[1], edge_type[0])
        if reverse_key in edges:
            return edges[reverse_key], True

    msg = f"Edge type {edge_type} not found in edges dictionary"
    raise KeyError(msg)


def _build_hop_frame(
    *,
    edge_gdf: gpd.GeoDataFrame,
    step_idx: int,
    reversed_lookup: bool,
    edge_attrs: list[str] | None,
) -> pd.DataFrame:
    """
    Convert one hop into a canonical DataFrame used for joins.

    Extracts source and destination indices and optional attributes.

    Parameters
    ----------
    edge_gdf : geopandas.GeoDataFrame
        The edge GeoDataFrame for this hop.
    step_idx : int
        The index of the current step in the metapath.
    reversed_lookup : bool
        Whether the edge is being traversed in reverse.
    edge_attrs : list[str] or None
        List of edge attributes to include.

    Returns
    -------
    pandas.DataFrame
        Canonical DataFrame with source and destination columns.
    """
    if not isinstance(edge_gdf.index, pd.MultiIndex) or edge_gdf.index.nlevels < 2:
        msg = "Edge GeoDataFrame must have a two-level MultiIndex"
        raise ValueError(msg)

    src_level = 1 if reversed_lookup else 0
    dst_level = 0 if reversed_lookup else 1

    src_col = f"src_{step_idx}"
    dst_col = f"dst_{step_idx}"

    data = {
        src_col: edge_gdf.index.get_level_values(src_level).to_numpy(),
        dst_col: edge_gdf.index.get_level_values(dst_level).to_numpy(),
    }

    if edge_attrs:
        missing_attrs = [attr for attr in edge_attrs if attr not in edge_gdf.columns]
        if missing_attrs:
            msg = f"Edge attribute(s) {missing_attrs} missing in metapath steps"
            raise KeyError(msg)
        for attr in edge_attrs:
            data[f"{attr}_step{step_idx}"] = edge_gdf[attr].to_numpy()

    return pd.DataFrame(data)


def _aggregate_paths(
    combined: pd.DataFrame,
    *,
    step_count: int,
    edge_attrs: list[str] | None,
    aggregation: _EdgeAttrAggregation,
    start_index_name: str,
    end_index_name: str,
) -> pd.DataFrame:
    """
    Group joined paths into terminal node pairs with aggregated weights.

    Aggregates weights and optional attributes for each path.

    Parameters
    ----------
    combined : pandas.DataFrame
        DataFrame containing all joined paths.
    step_count : int
        Number of steps in the metapath.
    edge_attrs : list[str] or None
        List of edge attributes to aggregate.
    aggregation : _EdgeAttrAggregation
        Aggregation strategy for edge attributes.
    start_index_name : str
        Name of the start node index.
    end_index_name : str
        Name of the end node index.

    Returns
    -------
    pandas.DataFrame
        Aggregated DataFrame with weights.
    """
    src_col = "src_0"
    dst_col = f"dst_{step_count - 1}"

    # Prepare aggregation dictionary
    agg_map: dict[str, str | Callable[[pd.Series], float]] = {"weight": "sum"}

    # Base workload with path count (weight=1 for each path)
    workload_data = {
        "src": combined[src_col].to_numpy(),
        "dst": combined[dst_col].to_numpy(),
        "weight": np.ones(len(combined), dtype=float),
    }

    if edge_attrs:
        for attr in edge_attrs:
            # Collect columns for this attribute across all steps
            step_columns = [
                f"{attr}_step{i}"
                for i in range(step_count)
                if f"{attr}_step{i}" in combined.columns
            ]

            # Row reduction (e.g. sum of times along the path)
            block = combined[step_columns]
            workload_data[attr] = aggregation.row_reducer(block).to_numpy()
            agg_map[attr] = aggregation.group_reducer

    workload = pd.DataFrame(workload_data)

    # Group by terminal nodes and aggregate (e.g. sum of weights = number of paths)
    aggregated = workload.groupby(["src", "dst"], sort=False).agg(agg_map)

    if aggregated.empty:
        return _empty_metapath_frame(edge_attrs, start_index_name, end_index_name)

    aggregated.index = aggregated.index.set_names([start_index_name, end_index_name])
    return aggregated


def _empty_metapath_frame(
    edge_attrs: list[str] | None,
    start_index_name: str,
    end_index_name: str,
) -> pd.DataFrame:
    """
    Create an empty aggregation frame with a consistent schema.

    Ensures downstream consumers receive predictable column ordering and index
    names even when no metapath traversals are available.

    Parameters
    ----------
    edge_attrs : list[str] | None
        Edge attributes expected in the aggregated output.
    start_index_name : str
        MultiIndex level name representing the metapath start nodes.
    end_index_name : str
        MultiIndex level name representing the metapath end nodes.

    Returns
    -------
    pandas.DataFrame
        Empty DataFrame with the requested columns and index structure.
    """
    columns = ["weight"] + (edge_attrs or [])
    frame = pd.DataFrame(columns=columns)
    frame.index = pd.MultiIndex.from_tuples([], names=[start_index_name, end_index_name])
    return frame


def _empty_metapath_gdf(
    nodes: dict[str, gpd.GeoDataFrame],
    start_type: str,
    end_type: str,
    edge_attrs: list[str] | None,
    start_index_name: str,
    end_index_name: str,
) -> gpd.GeoDataFrame:
    """
    Create an empty GeoDataFrame placeholder for unmet metapaths.

    Ensures downstream consumers receive consistent structure even with no data.

    Parameters
    ----------
    nodes : dict[str, geopandas.GeoDataFrame]
        Mapping of node types to their GeoDataFrames.
    start_type : str
        Node type at the beginning of the metapath.
    end_type : str
        Node type at the end of the metapath.
    edge_attrs : list[str] | None
        Optional edge attributes expected in the result.
    start_index_name : str
        Name for the source level in the result index.
    end_index_name : str
        Name for the destination level in the result index.

    Returns
    -------
    geopandas.GeoDataFrame
        Empty GeoDataFrame matching the expected schema for the metapath.
    """
    start_nodes = nodes.get(start_type)
    end_nodes = nodes.get(end_type)

    crs = None
    if start_nodes is not None and start_nodes.crs:
        crs = start_nodes.crs
    elif end_nodes is not None:
        crs = end_nodes.crs

    base_frame = _empty_metapath_frame(edge_attrs, start_index_name, end_index_name)
    geometry = gpd.GeoSeries([], crs=crs)
    return gpd.GeoDataFrame(
        base_frame.assign(geometry=geometry),
        geometry="geometry",
        crs=crs,
    )


def _attach_metapath_geometry(
    aggregated: pd.DataFrame,
    nodes: dict[str, gpd.GeoDataFrame],
    start_type: str,
    end_type: str,
) -> gpd.GeoDataFrame:
    """
    Attach straight geometries between terminal node pairs.

    Generates straight-line links that connect the start and end node positions.

    Parameters
    ----------
    aggregated : pandas.DataFrame
        Aggregated metapath table produced by :func:`_reduce_metapath_paths`.
    nodes : dict[str, geopandas.GeoDataFrame]
        Mapping of node types to their GeoDataFrames.
    start_type : str
        Node type for the source nodes.
    end_type : str
        Node type for the destination nodes.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing aggregated metrics and straight-line geometries.

    Raises
    ------
    KeyError
        If node GeoDataFrames for ``start_type`` or ``end_type`` are missing.
    """
    start_nodes = nodes.get(start_type)
    end_nodes = nodes.get(end_type)

    if start_nodes is None or end_nodes is None:
        msg = f"Missing node GeoDataFrame for start '{start_type}' or end '{end_type}'"
        raise KeyError(msg)

    start_series = start_nodes.geometry.reindex(aggregated.index.get_level_values(0))
    end_series = end_nodes.geometry.reindex(aggregated.index.get_level_values(1))

    geometries = [
        _safe_linestring(start_geom, end_geom)
        for start_geom, end_geom in zip(start_series, end_series, strict=False)
    ]

    crs = start_nodes.crs or end_nodes.crs
    result = gpd.GeoDataFrame(aggregated, geometry=geometries, crs=crs)

    if "weight" in result.columns and not result.empty:
        weight_series = result["weight"]
        if weight_series.notna().all():
            rounded = weight_series.round()
            if np.allclose(weight_series.to_numpy(), rounded.to_numpy()):
                result["weight"] = rounded.astype(int)

    return result


def _normalise_index_name(raw_name: object, fallback: str) -> str:
    """
    Return a sensible index level name for merged metapaths.

    Provides deterministic naming when original indices are unnamed.

    Parameters
    ----------
    raw_name : object
        Original index level name extracted from a ``MultiIndex``.
    fallback : str
        Name to use when the source level is unnamed.

    Returns
    -------
    str
        Normalised index level name.
    """
    if isinstance(raw_name, str) and raw_name:
        return raw_name
    if raw_name is None:
        return fallback
    return str(raw_name)


def _finalize_metapath_result(
    nodes_dict: dict[str, gpd.GeoDataFrame],
    edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    as_nx: bool,
    multigraph: bool,
    directed: bool,
    metadata: dict[tuple[str, str, str], dict[str, object]] | None,
) -> (
    nx.Graph
    | nx.MultiGraph
    | tuple[
        dict[str, gpd.GeoDataFrame],
        dict[tuple[str, str, str], gpd.GeoDataFrame],
    ]
):
    """
    Return hetero dictionaries or convert to NetworkX with metadata.

    Handles the optional conversion to NetworkX while keeping metadata in sync.

    Parameters
    ----------
    nodes_dict : dict[str, geopandas.GeoDataFrame]
        Mapping of node types to their GeoDataFrames.
    edges_dict : dict[tuple[str, str, str], geopandas.GeoDataFrame]
        Mapping of edge types to their GeoDataFrames.
    as_nx : bool
        Whether to convert the result to a NetworkX graph.
    multigraph : bool
        Build a :class:`networkx.MultiGraph` when returning NetworkX data.
    directed : bool
        Produce a directed NetworkX graph if requested.
    metadata : dict[tuple[str, str, str], dict[str, object]] | None
        Metadata describing metapath-derived edges to attach to the graph.

    Returns
    -------
    networkx.Graph | networkx.MultiGraph | tuple[dict[str, geopandas.GeoDataFrame], dict[tuple[str, str, str], geopandas.GeoDataFrame]]
        Either the hetero GeoDataFrame dictionaries or, when ``as_nx`` is set,
        a NetworkX graph updated with the provided metadata.
    """
    if not as_nx:
        return nodes_dict, edges_dict

    nx_graph = gdf_to_nx(
        nodes=nodes_dict,
        edges=edges_dict,
        multigraph=multigraph,
        directed=directed,
    )

    if metadata:
        existing = nx_graph.graph.get("metapath_dict")
        if isinstance(existing, dict):
            existing.update(metadata)
        else:
            existing = dict(metadata)
        nx_graph.graph["metapath_dict"] = existing

    return nx_graph


# -----------------------------------------------------------------------------
# Helpers for add_metapaths_by_weight
# -----------------------------------------------------------------------------


def _validate_and_resolve_parameters(
    new_relation_name: str | None,
    endpoint_type: str | None,
    min_threshold: float,
    threshold: float,
) -> tuple[str, tuple[str, str, str]] | None:
    """
    Validate and resolve parameters for metapath connection.

    Validates parameter consistency and constructs the target edge type tuple
    from the provided parameters.

    Parameters
    ----------
    new_relation_name : str | None
        Target edge relation name.
    endpoint_type : str | None
        The node type to connect.
    min_threshold : float
        Minimum cost threshold.
    threshold : float
        Maximum cost threshold.

    Returns
    -------
    tuple[str, tuple[str, str, str]] | None
        Tuple of (endpoint_type, target_edge_type_tuple) if valid, None otherwise.
    """
    resolved_endpoint_type = endpoint_type
    relation_name = None
    target_edge_type_tuple = None

    if new_relation_name is not None:
        relation_name = new_relation_name

    if resolved_endpoint_type is None:
        logger.warning("endpoint_type not provided.")
        return None

    if target_edge_type_tuple is None:
        relation_name = relation_name or f"connected_within_{min_threshold}_{threshold}"
        target_edge_type_tuple = (resolved_endpoint_type, relation_name, resolved_endpoint_type)

    return resolved_endpoint_type, target_edge_type_tuple


def _return_original_graph(
    graph: (
        tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]]
        | nx.Graph
        | nx.MultiGraph
    ),
) -> (
    nx.Graph
    | nx.MultiGraph
    | tuple[
        dict[str, gpd.GeoDataFrame],
        dict[tuple[str, str, str], gpd.GeoDataFrame],
    ]
):
    """
    Return the original graph as-is.

    This function is only called when as_nx=False, so it just returns
    the graph without any conversion.

    Parameters
    ----------
    graph : tuple or networkx.Graph or networkx.MultiGraph
        Input graph in either GeoDataFrame tuple or NetworkX format.

    Returns
    -------
    nx.Graph or nx.MultiGraph or tuple
        The original graph unchanged.
    """
    # This function is only called with as_nx=False, so just return the graph as-is
    return graph


def _prepare_sparse_graph_data(
    graph: (
        tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]]
        | nx.Graph
        | nx.MultiGraph
    ),
    endpoint_type: str,
    edge_types: list[tuple[str, str, str]] | None,
    weight: str,
) -> tuple[
    dict[object, int],
    dict[int, object],
    list[int],
    list[int],
    list[int],
    list[float],
    int,
]:
    """
    Prepare sparse graph data by delegating to appropriate handler.

    Dispatches to GeoDataFrame or NetworkX specific handler based on input type.

    Parameters
    ----------
    graph : tuple or networkx.Graph or networkx.MultiGraph
        Input graph.
    endpoint_type : str
        Node type to identify as endpoints.
    edge_types : list[tuple[str, str, str]] or None
        Edge types to include in traversal.
    weight : str
        Edge attribute to use as weight.

    Returns
    -------
    tuple
        Seven-element tuple containing node mappings, indices, and edge data.
    """
    if isinstance(graph, tuple):
        return _prepare_graph_data_from_gdf(graph, endpoint_type, edge_types, weight)
    return _prepare_graph_data_from_nx(graph, endpoint_type, edge_types, weight)


def _prepare_graph_data_from_gdf(
    graph: tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]],
    endpoint_type: str,
    edge_types: list[tuple[str, str, str]] | None,
    weight: str,
) -> tuple[
    dict[object, int],
    dict[int, object],
    list[int],
    list[int],
    list[int],
    list[float],
    int,
]:
    """
    Build sparse graph data from GeoDataFrame tuple.

    Extracts node indices and edge weights from GeoDataFrame dictionaries.

    Parameters
    ----------
    graph : tuple[dict[str, GeoDataFrame], dict[tuple[str, str, str], GeoDataFrame]]
        Input graph as GeoDataFrame tuple.
    endpoint_type : str
        Node type to identify as endpoints.
    edge_types : list[tuple[str, str, str]] or None
        Edge types to include in traversal.
    weight : str
        Edge attribute to use as weight.

    Returns
    -------
    tuple
        Seven-element tuple containing node mappings, indices, and edge data.
    """
    nodes_dict, edges_dict = graph
    node_to_idx: dict[object, int] = {}
    idx_to_node: dict[int, object] = {}
    endpoint_indices: list[int] = []
    row_indices: list[int] = []
    col_indices: list[int] = []
    data: list[float] = []
    current_idx = 0

    # Build node index
    for n_type, gdf in nodes_dict.items():
        for node_id in gdf.index:
            key = (n_type, node_id)
            if key not in node_to_idx:
                node_to_idx[key] = current_idx
                idx_to_node[current_idx] = key
                current_idx += 1

            if n_type == endpoint_type:
                endpoint_indices.append(node_to_idx[key])

    # Collect edges
    for e_type, gdf in edges_dict.items():
        if edge_types and e_type not in edge_types:
            continue

        src_type, _, dst_type = e_type
        us = gdf.index.get_level_values(0)
        vs = gdf.index.get_level_values(1)
        ws = gdf[weight].to_numpy()

        for u, v, w in zip(us, vs, ws, strict=False):
            u_key = (src_type, u)
            v_key = (dst_type, v)
            u_idx = node_to_idx[u_key]
            v_idx = node_to_idx[v_key]
            w_val = float(w)

            row_indices.append(u_idx)
            col_indices.append(v_idx)
            data.append(w_val)

    return (
        node_to_idx,
        idx_to_node,
        endpoint_indices,
        row_indices,
        col_indices,
        data,
        current_idx,
    )


def _prepare_graph_data_from_nx(
    graph: nx.Graph | nx.MultiGraph,
    endpoint_type: str,
    edge_types: list[tuple[str, str, str]] | None,
    weight: str,
) -> tuple[
    dict[object, int],
    dict[int, object],
    list[int],
    list[int],
    list[int],
    list[float],
    int,
]:
    """
    Build sparse graph data from NetworkX graph.

    Extracts node indices and edge weights from NetworkX graph structure.

    Parameters
    ----------
    graph : networkx.Graph or networkx.MultiGraph
        Input NetworkX graph.
    endpoint_type : str
        Node type to identify as endpoints.
    edge_types : list[tuple[str, str, str]] or None
        Edge types to include in traversal.
    weight : str
        Edge attribute to use as weight.

    Returns
    -------
    tuple
        Seven-element tuple containing node mappings, indices, and edge data.
    """
    node_to_idx: dict[object, int] = {}
    idx_to_node: dict[int, object] = {}
    endpoint_indices: list[int] = []
    row_indices: list[int] = []
    col_indices: list[int] = []
    data: list[float] = []
    current_idx = 0

    # Build node index
    for n, d in graph.nodes(data=True):
        if n not in node_to_idx:
            node_to_idx[n] = current_idx
            idx_to_node[current_idx] = n
            current_idx += 1

        if d.get("node_type") == endpoint_type:
            endpoint_indices.append(node_to_idx[n])

    edges_iter = (
        graph.edges(data=True, keys=True) if graph.is_multigraph() else graph.edges(data=True)
    )

    min_weights: dict[tuple[int, int], float] = {}

    for edge in edges_iter:
        if graph.is_multigraph():
            u, v, _, d = edge
        else:
            u, v, d = edge

        if edge_types and d.get("edge_type") not in edge_types:
            continue

        w_val = d.get(weight)
        u_idx = node_to_idx[u]
        v_idx = node_to_idx[v]
        w_float = float(w_val)

        pair = (u_idx, v_idx)
        if pair not in min_weights or w_float < min_weights[pair]:
            min_weights[pair] = w_float

    for (u_idx, v_idx), w in min_weights.items():
        row_indices.append(u_idx)
        col_indices.append(v_idx)
        data.append(w)

    return (
        node_to_idx,
        idx_to_node,
        endpoint_indices,
        row_indices,
        col_indices,
        data,
        current_idx,
    )


def _extract_metapath_edges(
    dist_matrix: np.ndarray,
    endpoint_indices: list[int],
    idx_to_node: dict[int, object],
    min_threshold: float,
    threshold: float,
    graph: (
        tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]]
        | nx.Graph
        | nx.MultiGraph
    ),
    directed: bool,
    target_edge_type: tuple[str, str, str],
    weight: str,
) -> list[dict[str, Any]]:
    """
    Extract metapath edges from Dijkstra distance matrix.

    Filters distances and creates edge records for valid connections.

    Parameters
    ----------
    dist_matrix : numpy.ndarray
        Distance matrix from Dijkstra algorithm.
    endpoint_indices : list[int]
        Indices of endpoint nodes.
    idx_to_node : dict[int, object]
        Mapping from indices to node keys.
    min_threshold : float
        Minimum distance threshold.
    threshold : float
        Maximum distance threshold.
    graph : tuple or networkx.Graph or networkx.MultiGraph
        Original input graph for node attribute lookup.
    directed : bool
        Whether graph is directed.
    target_edge_type : tuple[str, str, str]
        Edge type for new metapath edges.
    weight : str
        Edge weight attribute name.

    Returns
    -------
    list[dict[str, Any]]
        List of edge data dictionaries.
    """
    new_edges_data = []
    endpoint_indices_arr = np.array(endpoint_indices)

    for i, start_idx in enumerate(endpoint_indices):
        dists = dist_matrix[i]
        dists_to_endpoints = dists[endpoint_indices_arr]

        valid_mask = (
            (dists_to_endpoints >= min_threshold)
            & (dists_to_endpoints <= threshold)
            & (dists_to_endpoints != np.inf)
        )
        valid_mask[i] = False

        valid_target_indices_in_endpoints = np.where(valid_mask)[0]

        if valid_target_indices_in_endpoints.size > 0:
            start_node_key = idx_to_node[start_idx]

            for target_idx_in_endpoints in valid_target_indices_in_endpoints:
                target_idx = endpoint_indices[target_idx_in_endpoints]
                dist = dists_to_endpoints[target_idx_in_endpoints]
                end_node_key = idx_to_node[target_idx]

                if isinstance(graph, tuple):
                    start_node = cast("tuple[str, object]", start_node_key)[1]
                    end_node = cast("tuple[str, object]", end_node_key)[1]
                else:
                    start_node = start_node_key
                    end_node = end_node_key

                start_orig = start_node
                end_orig = end_node

                if isinstance(graph, (nx.Graph, nx.MultiGraph)):
                    start_orig = graph.nodes[start_node].get("_original_index", start_node)
                    end_orig = graph.nodes[end_node].get("_original_index", end_node)

                orig_edge_index = (start_orig, end_orig)

                if not directed and start_orig > end_orig:  # type: ignore[operator]
                    orig_edge_index = (end_orig, start_orig)

                new_edges_data.append(
                    {
                        "u": start_node,
                        "v": end_node,
                        weight: float(dist),
                        "edge_type": target_edge_type,
                        "_original_edge_index": orig_edge_index,
                    }
                )
    return new_edges_data


def _add_edges_to_nx_graph(
    nx_graph: nx.Graph | nx.MultiGraph,
    new_edges_data: list[dict[str, Any]],
    target_edge_type: tuple[str, str, str],
    weight: str,
) -> nx.Graph | nx.MultiGraph:
    """
    Add metapath edges to a NetworkX graph.

    This function adds edges from the provided edge data list to an existing
    NetworkX graph and updates the graph's edge type registry.

    Parameters
    ----------
    nx_graph : networkx.Graph or networkx.MultiGraph
        NetworkX graph to add edges to.
    new_edges_data : list[dict[str, Any]]
        List of edge data dictionaries.
    target_edge_type : tuple[str, str, str]
        Edge type for new metapath edges.
    weight : str
        Edge weight attribute name.

    Returns
    -------
    networkx.Graph or networkx.MultiGraph
        Graph with edges added.
    """
    edges_to_add = [
        (
            e["u"],
            e["v"],
            {
                weight: e[weight],
                "edge_type": e["edge_type"],
                "_original_edge_index": e["_original_edge_index"],
            },
        )
        for e in new_edges_data
    ]
    nx_graph.add_edges_from(edges_to_add)

    if "edge_types" in nx_graph.graph and target_edge_type not in nx_graph.graph["edge_types"]:
        nx_graph.graph["edge_types"].append(target_edge_type)

    return nx_graph


def _create_edge_gdf_from_data(
    new_edges_data: list[dict[str, Any]],
    endpoint_gdf: gpd.GeoDataFrame,
    weight: str,
) -> gpd.GeoDataFrame:
    """
    Create GeoDataFrame for metapath edges.

    This function constructs a GeoDataFrame from edge data dictionaries,
    creating LineString geometries between endpoint nodes.

    Parameters
    ----------
    new_edges_data : list[dict[str, Any]]
        List of edge data dictionaries.
    endpoint_gdf : geopandas.GeoDataFrame
        GeoDataFrame of endpoint nodes for geometry creation.
    weight : str
        Edge weight attribute name.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing the new edges.
    """
    edges_df = pd.DataFrame(new_edges_data)

    u_geoms = endpoint_gdf.loc[edges_df["u"]].geometry.to_numpy()
    v_geoms = endpoint_gdf.loc[edges_df["v"]].geometry.to_numpy()

    geometries = [
        LineString([u.centroid, v.centroid]) for u, v in zip(u_geoms, v_geoms, strict=False)
    ]

    new_gdf = gpd.GeoDataFrame(
        edges_df[[weight, "edge_type"]],
        geometry=geometries,
        crs=endpoint_gdf.crs,
    )

    new_gdf.index = pd.MultiIndex.from_frame(edges_df[["u", "v"]])
    new_gdf.index.names = ["u", "v"]

    return new_gdf


def _construct_metapath_result(
    graph: (
        tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]]
        | nx.Graph
        | nx.MultiGraph
    ),
    new_edges_data: list[dict[str, Any]],
    as_nx: bool,
    directed: bool,
    multigraph: bool,
    target_edge_type: tuple[str, str, str],
    weight: str,
    endpoint_type: str,
) -> (
    nx.Graph
    | nx.MultiGraph
    | tuple[
        dict[str, gpd.GeoDataFrame],
        dict[tuple[str, str, str], gpd.GeoDataFrame],
    ]
):
    """
    Construct the result graph with metapath edges added.

    Adds new edges to the graph in NetworkX or GeoDataFrame format.

    Parameters
    ----------
    graph : tuple or networkx.Graph or networkx.MultiGraph
        Original input graph.
    new_edges_data : list[dict[str, Any]]
        List of new edge data to add.
    as_nx : bool
        Whether to return NetworkX format.
    directed : bool
        Whether graph is directed.
    multigraph : bool
        Whether to use MultiGraph format.
    target_edge_type : tuple[str, str, str]
        Edge type for new metapath edges.
    weight : str
        Edge weight attribute name.
    endpoint_type : str
        Node type of endpoints.

    Returns
    -------
    nx.Graph or nx.MultiGraph or tuple
        Graph with metapath edges added.
    """
    if as_nx:
        # Convert to NetworkX if needed
        if isinstance(graph, tuple):
            nodes_dict_in, edges_dict_in = graph
            nx_graph = gdf_to_nx(
                nodes_dict_in, edges_dict_in, directed=directed, multigraph=multigraph
            )
        else:
            nx_graph = graph

        # Add edges to NetworkX graph
        return _add_edges_to_nx_graph(nx_graph, new_edges_data, target_edge_type, weight)

    # Return as GeoDataFrame tuple
    if isinstance(graph, tuple):
        nodes_dict, edges_dict = graph
    else:
        nodes_dict, edges_dict = nx_to_gdf(graph)  # pragma: no cover

    if not new_edges_data:
        return nodes_dict, edges_dict

    # Create GeoDataFrame for new edges
    endpoint_gdf = nodes_dict[endpoint_type]
    new_gdf = _create_edge_gdf_from_data(new_edges_data, endpoint_gdf, weight)
    edges_dict[target_edge_type] = new_gdf

    return nodes_dict, edges_dict
