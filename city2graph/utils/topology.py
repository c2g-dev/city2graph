"""Graph topology and graph-operation utilities."""

# Standard library imports
import logging
import typing
import warnings
from collections import defaultdict
from collections.abc import Iterable
from itertools import combinations
from typing import Any
from typing import Literal

# Third-party imports
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon

from city2graph.base import GeoDataProcessor

from .conversion import _identify_source_target_cols
from .conversion import gdf_to_nx
from .conversion import nx_to_gdf

__all__ = [
    "canonicalize_edges",
    "clip_graph",
    "dual_graph",
    "remove_isolated_components",
    "symmetrize_edges",
]

logger = logging.getLogger("city2graph.utils")


def _safe_sort_key(value: object) -> tuple[str, str]:
    """
    Provide a deterministic sort key even for incomparable edge identifiers.

    Generates a tuple containing the type name and string representation of the
    value, allowing for consistent sorting of mixed types.

    Parameters
    ----------
    value : object
        The value to generate a sort key for.

    Returns
    -------
    tuple[str, str]
        A tuple of (type_name, repr_string) for sorting.
    """
    return (type(value).__name__, repr(value))


def _canonical_edge_pair(edge_a: object, edge_b: object) -> tuple[object, object]:
    """
    Return a deterministic ordering for an undirected edge pair.

    Sorts the two edge identifiers to ensure that the pair (u, v) is treated
    identically to (v, u), handling potential type comparison errors.

    Parameters
    ----------
    edge_a : object
        The first edge identifier.
    edge_b : object
        The second edge identifier.

    Returns
    -------
    tuple[object, object]
        The sorted pair of edge identifiers.
    """
    if edge_a == edge_b:
        return edge_a, edge_b
    try:
        return (
            (edge_a, edge_b)
            if typing.cast("Any", edge_a) <= typing.cast("Any", edge_b)
            else (edge_b, edge_a)
        )
    except TypeError:
        key_a = _safe_sort_key(edge_a)
        key_b = _safe_sort_key(edge_b)
        return (edge_a, edge_b) if key_a <= key_b else (edge_b, edge_a)


def _build_dual_edge_pairs(
    edge_ids: Iterable[object],
    u_values: Iterable[object],
    v_values: Iterable[object],
) -> list[tuple[object, object]]:
    """
    Compute unique dual-edge pairs given source/target node identifiers.

    Identifies pairs of edges that share a common node, effectively constructing
    the adjacency list for the dual graph.

    Parameters
    ----------
    edge_ids : Iterable[object]
        The identifiers of the edges.
    u_values : Iterable[object]
        The source node identifiers for each edge.
    v_values : Iterable[object]
        The target node identifiers for each edge.

    Returns
    -------
    list[tuple[object, object]]
        A list of unique pairs of adjacent edges.
    """
    adjacency: dict[object, set[object]] = defaultdict(set)
    for edge_id, u, v in zip(edge_ids, u_values, v_values, strict=False):
        adjacency[u].add(edge_id)
        adjacency[v].add(edge_id)

    pairs: set[tuple[Any, Any]] = set()
    for edges in adjacency.values():
        if len(edges) < 2:
            continue
        for edge_a, edge_b in combinations(edges, 2):
            pairs.add(_canonical_edge_pair(edge_a, edge_b))

    return sorted(
        pairs,
        key=lambda pair: (_safe_sort_key(pair[0]), _safe_sort_key(pair[1])),
    )


def _empty_dual_edge_gdf(crs: object, edge_id_col: str | None) -> gpd.GeoDataFrame:
    """
    Create an empty dual-edge GeoDataFrame with consistent index names.

    This helper ensures that even when no dual edges are produced, the resulting
    GeoDataFrame has the correct MultiIndex structure and column definitions.

    Parameters
    ----------
    crs : object
        The Coordinate Reference System.
    edge_id_col : str or None
        The name of the edge ID column.

    Returns
    -------
    geopandas.GeoDataFrame
        An empty GeoDataFrame for dual edges.
    """
    names = (
        [f"from_{edge_id_col}", f"to_{edge_id_col}"]
        if edge_id_col
        else ["from_edge_id", "to_edge_id"]
    )
    empty_index = pd.MultiIndex.from_arrays([[], []], names=names)
    return gpd.GeoDataFrame(geometry=[], crs=crs, index=empty_index)


def dual_graph(
    graph: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph | nx.MultiGraph,
    edge_id_col: str | None = None,
    keep_original_geom: bool = False,
    source_col: str | None = None,
    target_col: str | None = None,
    as_nx: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph | nx.MultiGraph:
    """
    Convert a primal graph represented by nodes and edges GeoDataFrames to its dual graph.

    In the dual graph, original edges become nodes and original nodes become edges connecting
    adjacent original edges.

    Parameters
    ----------
    graph : tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame] or networkx.Graph or networkx.MultiGraph
        A graph containing nodes and edges GeoDataFrames or a NetworkX graph of the primal graph.
    edge_id_col : str, optional
        The name of the column in the edges GeoDataFrame to be used as unique identifiers
        for dual graph nodes. If None, the index of the edges GeoDataFrame is used.
        Default is None.
    keep_original_geom : bool, default False
        If True, preserve the original geometry of the edges in a new column named
        'original_geometry' in the dual nodes GeoDataFrame.
    source_col : str, optional
        Name of the column or index level representing the source node ID in the edges GeoDataFrame.
        If provided, it overrides automatic detection.
    target_col : str, optional
        Name of the column or index level representing the target node ID in the edges GeoDataFrame.
        If provided, it overrides automatic detection.
    as_nx : bool, default False
        If True, return the dual graph as a NetworkX graph instead of GeoDataFrames.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame]
        A tuple containing the nodes and edges of the dual graph as GeoDataFrames.

        - Dual nodes GeoDataFrame: Nodes represent original edges. The geometry is the
          centroid of the original edge's geometry. The index is derived from `edge_id_col`
          or the original edge index.
        - Dual edges GeoDataFrame: Edges represent adjacency between original edges (i.e.,
          they shared a node in the primal graph). The geometry is a LineString connecting
          the centroids of the two dual nodes. The index is a MultiIndex of the connected
          dual node IDs.

    See Also
    --------
    segments_to_graph : Convert LineString segments to a graph structure.

    Examples
    --------
    >>> import geopandas as gpd
    >>> import pandas as pd
    >>> from shapely.geometry import Point, LineString
    >>> # Primal graph nodes
    >>> nodes = gpd.GeoDataFrame(
    ...     {"node_id": [0, 1, 2]},
    ...     geometry=[Point(0, 0), Point(1, 1), Point(1, 0)],
    ...     crs="EPSG:32633"
    ... ).set_index("node_id")
    >>> # Primal graph edges
    >>> edges = gpd.GeoDataFrame(
    ...     {"u": [0, 1], "v": [1, 2]},
    ...     geometry=[LineString([(0, 0), (1, 1)]), LineString([(1, 1), (1, 0)])],
    ...     crs="EPSG:32633"
    ... )
    >>> # Convert to dual graph
    >>> dual_nodes, dual_edges = dual_graph((nodes, edges))
    """
    # Validate input type and extract GeoDataFrames
    _nodes_gdf, edges_gdf = _validate_dual_graph_input(graph)

    # Ensure edges have a CRS
    if edges_gdf.crs is None:
        msg = "Edges GeoDataFrame must have a CRS."
        raise ValueError(msg)

    # Work on a copy to avoid modifying the input
    edges_clean = edges_gdf.copy()
    crs = edges_clean.crs

    # Handle empty edges case
    if edges_clean.empty:
        dual_nodes = gpd.GeoDataFrame(geometry=[], crs=crs)
        dual_edges = gpd.GeoDataFrame(geometry=[], crs=crs)
        return dual_nodes, dual_edges

    # edges_clean is guaranteed to be non-None and non-empty here
    assert edges_clean is not None
    assert not edges_clean.empty

    # 1. Create Dual Nodes
    # --------------------
    # Dual nodes are simply the centroids of the primal edges.
    # We preserve the original edge attributes in the dual nodes.
    dual_nodes = edges_clean.copy()
    if dual_nodes.crs.is_geographic:
        # Warn if using geographic CRS for centroid calculation
        warnings.warn(
            "Geometry is in a geographic CRS. Results from 'centroid' are likely incorrect. "
            "Use 'GeoSeries.to_crs()' to re-project geometries to a projected CRS before this operation.",
            UserWarning,
            stacklevel=2,
        )
    dual_nodes["geometry"] = dual_nodes.geometry.centroid

    if keep_original_geom:
        dual_nodes["original_geometry"] = edges_clean.geometry

    # Handle edge_id_col if provided
    if edge_id_col:
        if edge_id_col in dual_nodes.columns:
            dual_nodes = dual_nodes.set_index(edge_id_col)
        elif dual_nodes.index.name != edge_id_col:
            # If it's not a column and not the current index name, raise an error.
            msg = f"Column '{edge_id_col}' not found in edges GeoDataFrame."
            raise ValueError(msg)

    # 2. Create Dual Edges
    # --------------------
    # Dual edges connect dual nodes (primal edges) that share a primal node.
    # We can find these by looking at the start and end nodes of the primal edges.

    # We avoid reset_index() because it can fail if index names conflict with column names.
    u_values = None
    v_values = None

    # Identify source and target columns/indices
    u_values, v_values = _identify_source_target_cols(
        edges_clean, source_col=source_col, target_col=target_col
    )

    edge_pairs = _build_dual_edge_pairs(dual_nodes.index, u_values, v_values)

    if not edge_pairs:
        dual_edges = _empty_dual_edge_gdf(edges_clean.crs, edge_id_col)
        return (dual_nodes, dual_edges) if not as_nx else gdf_to_nx(dual_nodes, dual_edges)

    names = (
        [f"from_{edge_id_col}", f"to_{edge_id_col}"]
        if edge_id_col
        else ["from_edge_id", "to_edge_id"]
    )
    dual_index = pd.MultiIndex.from_tuples(edge_pairs, names=names)

    edge_ids_from = [pair[0] for pair in edge_pairs]
    edge_ids_to = [pair[1] for pair in edge_pairs]
    geom_series = dual_nodes.geometry
    p1 = geom_series.reindex(edge_ids_from).tolist()
    p2 = geom_series.reindex(edge_ids_to).tolist()
    geoms = [LineString([g1, g2]) for g1, g2 in zip(p1, p2, strict=False)]

    dual_edges = gpd.GeoDataFrame(
        geometry=geoms,
        crs=edges_clean.crs,
        index=dual_index,
    )

    if as_nx:
        return gdf_to_nx(dual_nodes, dual_edges)
    return dual_nodes, dual_edges


def canonicalize_edges(
    edges: gpd.GeoDataFrame,
    duplicates: Literal["first", "key", "error"] = "first",
) -> gpd.GeoDataFrame:
    """
    Canonicalize an edge GeoDataFrame to one deterministic row per undirected pair.

    Reorders the first two MultiIndex levels (source, target) of every
    non-self-loop edge into a deterministic canonical order, so reciprocal rows
    ``(u, v)`` and ``(v, u)`` — the typical shape of bidirectional roads exported
    from a directed source such as an OSMnx MultiDiGraph — map onto the same
    unordered key. Attributes and geometry of each row are left untouched; only
    the index values change (geometries are NOT reversed). Self-loops are
    returned unchanged.

    Rows that share the same unordered key after canonicalization are handled
    according to ``duplicates``.

    Parameters
    ----------
    edges : geopandas.GeoDataFrame
        Edge GeoDataFrame with a two-level (source, target) or three-level
        (source, target, key) MultiIndex. For heterogeneous graphs, call this
        function separately on each edge type's GeoDataFrame.
    duplicates : {"first", "key", "error"}, default "first"
        How to handle rows sharing the same unordered key after
        canonicalization:

        - ``"first"``: keep the first occurrence and drop the rest. For
          three-level input the key level participates in the comparison, so
          distinct parallel keys are preserved.
        - ``"key"``: keep all rows and (re)generate an integer key level per
          unordered pair, producing a valid three-level multigraph index.
        - ``"error"``: raise ``ValueError`` reporting the offending pairs.

    Returns
    -------
    geopandas.GeoDataFrame
        The canonicalized edge GeoDataFrame. Row order, columns, attributes,
        geometry, and CRS are preserved (apart from rows dropped by
        ``duplicates="first"``). Index level names are kept positionally:
        level 0 keeps its original name even where values were swapped.

    Raises
    ------
    ValueError
        If the index is not a MultiIndex with at least two levels, if
        ``duplicates`` is not a recognized option, or if duplicate unordered
        keys remain and ``duplicates="error"``.

    See Also
    --------
    symmetrize_edges : Inverse operation adding the reverse row of each edge.
    gdf_to_pyg : Convert GeoDataFrames to PyTorch Geometric objects.
    segments_to_graph : Convert LineString segments to a graph structure.

    Notes
    -----
    Node identifiers are ordered with a vectorized comparison when they are
    mutually comparable (e.g. all integers or all strings). For mixed,
    non-comparable identifier types the order falls back to first-appearance
    codes, which is deterministic for a given input but input-dependent.

    Examples
    --------
    >>> import geopandas as gpd
    >>> import pandas as pd
    >>> from shapely.geometry import LineString
    >>> index = pd.MultiIndex.from_tuples([(0, 1), (1, 0)], names=["u", "v"])
    >>> edges = gpd.GeoDataFrame(
    ...     {"name": ["ab", "ba"]},
    ...     geometry=[LineString([(0, 0), (1, 1)]), LineString([(1, 1), (0, 0)])],
    ...     index=index,
    ...     crs="EPSG:32633",
    ... )
    >>> canonicalize_edges(edges).index.tolist()
    [(0, 1)]
    >>> canonicalize_edges(edges, duplicates="key").index.tolist()
    [(0, 1, 0), (0, 1, 1)]
    """
    allowed = ("first", "key", "error")
    if duplicates not in allowed:
        msg = f"duplicates must be one of {allowed}, got {duplicates!r}."
        raise ValueError(msg)

    if not isinstance(edges.index, pd.MultiIndex) or edges.index.nlevels < 2:
        msg = (
            "Edge GeoDataFrame index must be a MultiIndex with at least "
            "two levels (source, target)."
        )
        raise ValueError(msg)

    if edges.empty:
        return edges.copy()

    src = np.asarray(edges.index.get_level_values(0), dtype=object)
    dst = np.asarray(edges.index.get_level_values(1), dtype=object)
    try:
        swap = src > dst
    except TypeError:
        # Mixed, non-comparable identifier types: order by first-appearance codes.
        codes = pd.factorize(np.concatenate([src, dst]), sort=False)[0]
        swap = codes[: len(src)] > codes[len(src) :]
    canon_src = np.where(swap, dst, src)
    canon_dst = np.where(swap, src, dst)

    names = list(edges.index.names)
    has_key_level = edges.index.nlevels >= 3
    pairs = pd.DataFrame({"u": canon_src, "v": canon_dst})
    if has_key_level:
        pairs["key"] = np.asarray(edges.index.get_level_values(2), dtype=object)

    result = edges.copy()

    if duplicates == "key":
        keys = pairs.groupby(["u", "v"], sort=False).cumcount().to_numpy()
        key_name = names[2] if has_key_level else "key"
        result.index = pd.MultiIndex.from_arrays(
            [canon_src, canon_dst, keys],
            names=[names[0], names[1], key_name],
        )
        return result

    duplicated_mask = pairs.duplicated(keep=False).to_numpy()
    if duplicates == "error" and duplicated_mask.any():
        dup_pairs = pairs.loc[duplicated_mask, ["u", "v"]].drop_duplicates()
        examples = ", ".join(f"({u}, {v})" for u, v in dup_pairs.head(3).itertuples(index=False))
        msg = (
            f"Duplicate undirected edges detected after canonicalization: "
            f"{int(duplicated_mask.sum())} row(s) across {len(dup_pairs)} "
            f"unordered pair(s), e.g. {examples}. Pass duplicates='first' to "
            f"keep one row per pair or duplicates='key' to keep all rows as "
            f"a multigraph."
        )
        raise ValueError(msg)

    index_arrays = [canon_src, canon_dst]
    if has_key_level:
        index_arrays.append(np.asarray(edges.index.get_level_values(2), dtype=object))
    result.index = pd.MultiIndex.from_arrays(index_arrays, names=names)
    if duplicates == "first":
        keep_mask = ~pairs.duplicated(keep="first").to_numpy()
        result = result[keep_mask]
    return result


def symmetrize_edges(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Symmetrize an edge GeoDataFrame by adding the reverse row of each edge.

    For every non-self-loop edge ``(u, v)`` whose reverse ``(v, u)`` is not
    already present, appends a row indexed ``(v, u)`` with the same attributes
    and a reversed geometry, so that each row's geometry starts at its source
    node. This makes neighbourhood queries on the MultiIndex complete (e.g.
    ``edges.xs(node, level=0)`` returns every neighbour of ``node``), at the
    cost of duplicating edge attributes and geometries. Self-loops and edges
    whose reverse already exists are left untouched, so the operation is
    idempotent. Three-level (source, target, key) multigraph indexes keep the
    key of the original row on the reverse row.

    This is the inverse operation of :func:`canonicalize_edges`.

    Parameters
    ----------
    edges : geopandas.GeoDataFrame
        Edge GeoDataFrame with a two-level (source, target) or three-level
        (source, target, key) MultiIndex. For heterogeneous graphs, call this
        function separately on each edge type's GeoDataFrame.

    Returns
    -------
    geopandas.GeoDataFrame
        The symmetrized edge GeoDataFrame. Original rows come first with
        columns, attributes, and CRS preserved; reverse rows are appended in
        the order of the rows they mirror. Index level names are kept
        positionally.

    Raises
    ------
    ValueError
        If the index is not a MultiIndex with at least two levels.

    See Also
    --------
    canonicalize_edges : Inverse operation collapsing reciprocal rows.
    gdf_to_pyg : Convert GeoDataFrames to PyTorch Geometric objects.

    Notes
    -----
    ``gdf_to_pyg(..., directed=False)`` rejects edge tables containing both
    ``(u, v)`` and ``(v, u)`` rows as ambiguous. Collapse a symmetrized table
    with :func:`canonicalize_edges` first, or pass ``directed=True`` to keep
    the reciprocal rows as directed edges.

    Examples
    --------
    >>> import geopandas as gpd
    >>> import pandas as pd
    >>> from shapely.geometry import LineString
    >>> index = pd.MultiIndex.from_tuples([(0, 1)], names=["u", "v"])
    >>> edges = gpd.GeoDataFrame(
    ...     {"name": ["ab"]},
    ...     geometry=[LineString([(0, 0), (1, 1)])],
    ...     index=index,
    ...     crs="EPSG:32633",
    ... )
    >>> symmetrize_edges(edges).index.tolist()
    [(0, 1), (1, 0)]
    """
    if not isinstance(edges.index, pd.MultiIndex) or edges.index.nlevels < 2:
        msg = (
            "Edge GeoDataFrame index must be a MultiIndex with at least "
            "two levels (source, target)."
        )
        raise ValueError(msg)

    if edges.empty:
        return edges.copy()

    src = np.asarray(edges.index.get_level_values(0), dtype=object)
    dst = np.asarray(edges.index.get_level_values(1), dtype=object)
    names = list(edges.index.names)

    reversed_arrays = [dst, src]
    reversed_arrays.extend(
        np.asarray(edges.index.get_level_values(level), dtype=object)
        for level in range(2, edges.index.nlevels)
    )
    reversed_index = pd.MultiIndex.from_arrays(reversed_arrays, names=names)

    # Self-loops reverse onto themselves, so the membership test alone keeps
    # them out of the appended rows.
    missing = (src != dst) & ~reversed_index.isin(edges.index)
    if not missing.any():
        return edges.copy()

    reverse_rows = edges[missing].copy()
    reverse_rows.index = reversed_index[missing]
    reverse_rows.geometry = reverse_rows.geometry.reverse()

    return gpd.GeoDataFrame(pd.concat([edges, reverse_rows]))


def _validate_dual_graph_input(
    graph: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph | nx.MultiGraph,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Validate and extract nodes and edges for dual graph conversion.

    This helper checks the input graph format and converts it to node and edge
    GeoDataFrames if necessary, ensuring a consistent starting point for dual graph creation.

    Parameters
    ----------
    graph : tuple or networkx.Graph
        The input graph representation.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame]
        The nodes and edges GeoDataFrames.
    """
    if not (
        isinstance(graph, (nx.Graph, nx.MultiGraph))
        or (isinstance(graph, tuple) and len(graph) == 2)
    ):
        msg = "Input `graph` must be a tuple of (nodes_gdf, edges_gdf) or a NetworkX graph."
        raise TypeError(msg)

    if isinstance(graph, (nx.Graph, nx.MultiGraph)):
        # If input is a NetworkX graph, convert it to GeoDataFrames
        nodes_gdf, edges_gdf = nx_to_gdf(graph, nodes=True, edges=True)
    else:
        # Input is guaranteed to be tuple[GeoDataFrame, GeoDataFrame] by type annotation
        nodes_gdf, edges_gdf = graph

    return nodes_gdf, edges_gdf


def _validate_graph_input(
    graph: gpd.GeoDataFrame | tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph | nx.MultiGraph,
) -> tuple[gpd.GeoDataFrame | None, gpd.GeoDataFrame, str]:
    """
    Validate graph input and extract nodes/edges GeoDataFrames.

    Converts various graph representations to a common format for internal processing.

    Parameters
    ----------
    graph : GeoDataFrame, tuple, or NetworkX graph
        Input graph in any supported city2graph format.

    Returns
    -------
    tuple[GeoDataFrame | None, GeoDataFrame, str]
        Tuple of (nodes_gdf, edges_gdf, input_type).
    """
    if isinstance(graph, (nx.Graph, nx.MultiGraph)):
        return *nx_to_gdf(graph, nodes=True, edges=True), "nx"
    if isinstance(graph, tuple) and len(graph) == 2:
        return graph[0], graph[1], "tuple"
    if isinstance(graph, gpd.GeoDataFrame):
        return None, graph, "gdf"
    msg = "Input must be GeoDataFrame, (nodes, edges) tuple, or NetworkX graph."
    raise TypeError(msg)


def _return_graph_output(
    nodes: gpd.GeoDataFrame | None,
    edges: gpd.GeoDataFrame,
    input_type: str,
    as_nx: bool,
) -> gpd.GeoDataFrame | tuple[gpd.GeoDataFrame | None, gpd.GeoDataFrame] | nx.Graph:
    """
    Return graph in format matching input type or as NetworkX if requested.

    Standardizes output format logic for graph manipulation functions.

    Parameters
    ----------
    nodes : GeoDataFrame or None
        Nodes GeoDataFrame, or None for edges-only graphs.
    edges : GeoDataFrame
        Edges GeoDataFrame.
    input_type : str
        Original input type: "gdf", "tuple", or "nx".
    as_nx : bool
        If True, return as NetworkX graph.

    Returns
    -------
    GeoDataFrame, tuple, or NetworkX graph
        Graph in the requested output format.
    """
    if as_nx or input_type == "nx":
        return gdf_to_nx(nodes=nodes, edges=edges) if nodes is not None else gdf_to_nx(edges=edges)
    return (nodes, edges) if input_type == "tuple" else edges


def _filter_nodes_by_edges(
    nodes: gpd.GeoDataFrame | None,
    edges: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame | None:
    """
    Filter nodes to only those connected to edges via MultiIndex.

    Extracts node IDs from edge MultiIndex and filters nodes accordingly.

    Parameters
    ----------
    nodes : GeoDataFrame or None
        Nodes GeoDataFrame to filter, or None.
    edges : GeoDataFrame
        Edges GeoDataFrame with MultiIndex for connectivity info.

    Returns
    -------
    GeoDataFrame or None
        Filtered nodes GeoDataFrame, or None if input was None.
    """
    if nodes is None:
        return None
    if nodes.empty or edges.empty:
        return nodes.iloc[0:0].copy()
    if isinstance(edges.index, pd.MultiIndex) and edges.index.nlevels >= 2:
        connected = set(edges.index.get_level_values(0)) | set(edges.index.get_level_values(1))
        return nodes[nodes.index.isin(connected)].copy()
    return nodes.copy()


def clip_graph(
    graph: gpd.GeoDataFrame | tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph | nx.MultiGraph,
    area: Polygon | MultiPolygon | gpd.GeoDataFrame | gpd.GeoSeries,
    keep_outer_neighbors: bool = False,
    as_nx: bool = False,
) -> gpd.GeoDataFrame | tuple[gpd.GeoDataFrame | None, gpd.GeoDataFrame] | nx.Graph:
    """
    Clip a graph to a specific area.

    Filters edges to those within (or intersecting) a polygon area.
    Nodes are filtered to include only those connected to remaining edges.

    Parameters
    ----------
    graph : GeoDataFrame, tuple, or NetworkX graph
        Input graph as edges GeoDataFrame, (nodes, edges) tuple, or NetworkX graph.
    area : Polygon, MultiPolygon, GeoDataFrame, or GeoSeries
        The area to clip to.
    keep_outer_neighbors : bool, default False
        If True, keeps segments that intersect the boundary.
    as_nx : bool, default False
        If True, return as NetworkX graph.

    Returns
    -------
    GeoDataFrame, tuple, or NetworkX graph
        Clipped graph in same format as input, or NetworkX if as_nx=True.
    """
    nodes, edges, input_type = _validate_graph_input(graph)

    if edges.empty:
        return _return_graph_output(nodes, edges, input_type, as_nx)

    # Align clip geometry CRS to edges CRS when area is geospatial tabular input.
    if isinstance(area, (gpd.GeoDataFrame, gpd.GeoSeries)):
        area_aligned = GeoDataProcessor.harmonize_crs(area, edges.crs, warn=False)
        clip_geom = (
            area_aligned.geometry.union_all()
            if isinstance(area_aligned, gpd.GeoDataFrame) and len(area_aligned) > 1
            else area_aligned.union_all()
            if isinstance(area_aligned, gpd.GeoSeries) and len(area_aligned) > 1
            else area_aligned.geometry.iloc[0]
            if isinstance(area_aligned, gpd.GeoDataFrame)
            else area_aligned.iloc[0]
        )
    else:
        clip_geom = area

    # Filter or clip edges
    clipped_edges = (
        edges[edges.geometry.intersects(clip_geom)].copy()
        if keep_outer_neighbors
        else gpd.clip(edges, clip_geom)
    )

    # Explode MultiLineStrings (created by clipping or existing)
    if not clipped_edges.empty and "MultiLineString" in clipped_edges.geometry.type.to_numpy():
        clipped_edges = clipped_edges.explode(index_parts=False)

    # For strict clipping, keep only nodes inside the boundary and edges whose
    # endpoints are both inside. This avoids retaining outside endpoints when
    # an edge intersects the boundary.
    if nodes is not None and not keep_outer_neighbors:
        clipped_nodes = nodes[nodes.geometry.intersects(clip_geom)].copy()
        if isinstance(clipped_edges.index, pd.MultiIndex) and clipped_edges.index.nlevels >= 2:
            in_bound_ids = set(clipped_nodes.index)
            clipped_edges = clipped_edges[
                clipped_edges.index.get_level_values(0).isin(in_bound_ids)
                & clipped_edges.index.get_level_values(1).isin(in_bound_ids)
            ].copy()
        filtered_nodes = _filter_nodes_by_edges(clipped_nodes, clipped_edges)
    else:
        filtered_nodes = _filter_nodes_by_edges(nodes, clipped_edges)

    return _return_graph_output(filtered_nodes, clipped_edges, input_type, as_nx)


def remove_isolated_components(
    graph: gpd.GeoDataFrame | tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph | nx.MultiGraph,
    as_nx: bool = False,
) -> gpd.GeoDataFrame | tuple[gpd.GeoDataFrame | None, gpd.GeoDataFrame] | nx.Graph:
    """
    Keep only the largest connected component of a graph.

    Identifies all connected components and retains only the largest one.

    Parameters
    ----------
    graph : GeoDataFrame, tuple, or NetworkX graph
        Input graph as edges GeoDataFrame, (nodes, edges) tuple, or NetworkX graph.
    as_nx : bool, default False
        If True, return as NetworkX graph.

    Returns
    -------
    GeoDataFrame, tuple, or NetworkX graph
        Graph with only largest component, in same format as input.
    """
    nodes, edges, input_type = _validate_graph_input(graph)

    if edges.empty:
        return _return_graph_output(nodes, edges, input_type, as_nx)

    # Build graph for component analysis
    try:
        nx_graph = gdf_to_nx(edges=edges)
    except (ValueError, TypeError, KeyError):
        return _return_graph_output(nodes, edges, input_type, as_nx)

    if nx_graph.number_of_nodes() == 0:
        return _return_graph_output(nodes, edges, input_type, as_nx)

    # Find largest component
    cc_func = nx.weakly_connected_components if nx_graph.is_directed() else nx.connected_components
    largest_cc = max(cc_func(nx_graph), key=len)
    subgraph = nx_graph.subgraph(largest_cc)

    # Get original edge indices
    edge_indices = [
        d["_original_edge_index"]
        for _, _, d in subgraph.edges(data=True)
        if "_original_edge_index" in d
    ]
    filtered_edges = edges.loc[edge_indices] if edge_indices else edges.iloc[0:0]

    return _return_graph_output(
        _filter_nodes_by_edges(nodes, filtered_edges), filtered_edges, input_type, as_nx
    )
