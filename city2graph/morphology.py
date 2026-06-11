"""
Module for creating morphological graphs from urban data.

This module provides comprehensive functionality for analyzing urban morphology
through graph representations, focusing on the relationships between place
spaces (buildings and their tessellations) and movement spaces (street segments).
It creates heterogeneous graphs that capture the complex spatial relationships
inherent in urban environments. Both GeoDataFrame and NetworkX objects can be
converted to PyTorch Geometric Data or HeteroData by functions from graph.py.

The module specializes in three types of spatial relationships:
1. Place-to-place: Adjacency relationships between building tessellations
2. Movement-to-movement: Topological connectivity between street segments
3. Place-to-movement: Interface relationships between place and movement spaces
"""

from __future__ import annotations

# Standard library imports
import logging
import math
import typing
import warnings

# Third-party imports
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely import STRtree
from shapely.creation import linestrings as sh_linestrings
from shapely.geometry import LineString
from shapely.geometry import Point

# Local imports
from .base import GeoDataProcessor
from .proximity import contiguity_graph
from .utils import create_tessellation
from .utils import dual_graph
from .utils import filter_graph_by_distance
from .utils import gdf_to_nx
from .utils import nx_to_gdf
from .utils import symmetrize_edges

if typing.TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry

# Public API definition
__all__ = [
    "morphological_graph",
    "movement_to_movement_graph",
    "place_to_movement_graph",
    "place_to_place_graph",
    "private_to_private_graph",
    "private_to_public_graph",
    "public_to_public_graph",
    "segments_to_graph",
]

# Module logger configuration
logger = logging.getLogger(__name__)
_SOURCE_BUILDING_INDEX_COL = "_source_building_index"


def _validate_duplicate_edges(duplicate_edges: bool, as_nx: bool) -> None:
    """
    Reject ``duplicate_edges`` combined with NetworkX output.

    Reciprocal (u, v) / (v, u) rows can only be represented in GeoDataFrame
    output, so the conflicting combination is rejected before any computation.

    Parameters
    ----------
    duplicate_edges : bool
        Whether the caller requested reciprocal edge rows.
    as_nx : bool
        Whether the caller requested NetworkX output.

    Raises
    ------
    ValueError
        If ``duplicate_edges`` is combined with ``as_nx=True``.
    """
    if duplicate_edges and as_nx:
        msg = (
            "duplicate_edges=True is not supported with as_nx=True: an "
            "undirected nx.Graph cannot hold reciprocal (u, v) and (v, u) "
            "edges. Use as_nx=False to get a symmetrized edge GeoDataFrame."
        )
        raise ValueError(msg)


def _symmetrize_edge_columns(
    edges: gpd.GeoDataFrame, from_col: str, to_col: str
) -> gpd.GeoDataFrame:
    """
    Symmetrize a column-based edge table by adding reverse rows.

    The morphology sub-functions return edge tables whose endpoints live in id
    columns rather than a MultiIndex, so the table is round-tripped through a
    (source, target) MultiIndex to reuse :func:`symmetrize_edges`.

    Parameters
    ----------
    edges : geopandas.GeoDataFrame
        Edge GeoDataFrame with source and target id columns.
    from_col : str
        Name of the source id column.
    to_col : str
        Name of the target id column.

    Returns
    -------
    geopandas.GeoDataFrame
        The symmetrized edge GeoDataFrame with the same column layout.
    """
    if edges.empty:
        return edges
    column_order = list(edges.columns)
    symmetrized = symmetrize_edges(edges.set_index([from_col, to_col])).reset_index()
    return symmetrized[column_order]


# ============================================================================
# MAIN MORPHOLOGICAL GRAPH FUNCTION
# ============================================================================


# ============================================================================
# PUBLIC API - MAIN FUNCTIONS
# ============================================================================


def morphological_graph(  # noqa: PLR0913
    buildings_gdf: gpd.GeoDataFrame,
    segments_gdf: gpd.GeoDataFrame,
    center_point: gpd.GeoSeries | gpd.GeoDataFrame | None = None,
    distance: float | None = None,
    clipping_buffer: float = math.inf,
    extent_buffer: float = 50.0,
    limit: gpd.GeoDataFrame | gpd.GeoSeries | BaseGeometry | None = None,
    primary_barrier_col: str | None = "barrier_geometry",
    contiguity: str = "queen",
    keep_buildings: bool = False,
    keep_segments: bool = True,
    tolerance: float = 1e-6,
    include_unenclosed_buildings: bool = False,
    as_nx: bool = False,
    duplicate_edges: bool = False,
) -> tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]] | nx.Graph:
    """
    Create a morphological graph from buildings and street segments.

    This function creates a comprehensive morphological graph that captures relationships
    between place spaces (building tessellations) and movement spaces (street segments).
    The graph includes three types of relationships: place-to-place adjacency,
    movement-to-movement connectivity, and place-to-movement interfaces.

    The 'place_id' for tessellation cells is derived from 'tess_id' (generated by
    `create_tessellation`) or assigned sequentially if 'tess_id' doesn't directly map.
    The 'movement_id' for street segments is taken directly from the index of `segments_gdf`.

    Parameters
    ----------
    buildings_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing building polygons. Should contain Polygon or MultiPolygon geometries.
    segments_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing street segments. Should contain LineString geometries.
    center_point : geopandas.GeoSeries or geopandas.GeoDataFrame, optional
        Center point(s) for spatial filtering. If provided with distance parameter,
        only segments within the specified distance will be included.
    distance : float, optional
        Maximum distance from ``center_point`` for spatial filtering. When
        specified, street segments beyond this shortest-path distance are
        removed and tessellation cells are kept only if their own distance via
        these segments does not exceed this value.
    clipping_buffer : float, default=math.inf
        Buffer distance to ensure adequate context for generating tessellation.
        Must be non-negative and not smaller than ``extent_buffer``.
    extent_buffer : float, default=50.0
        Maximum perpendicular access distance (a bounded catchment cap) from a
        street to a building/cell for it to be retained, and the maximum length
        of a ``faced_to`` connection. Unlike ``distance`` this access term is
        never added to the walking-network budget, so a building whose only
        nearby street is disconnected from the reachable network is not retained
        on the strength of a long straight-line access leg across barriers. Must
        be non-negative and not larger than ``clipping_buffer``.
    limit : geopandas.GeoDataFrame, geopandas.GeoSeries, shapely geometry, or None, optional
        Boundary passed to `momepy.enclosures` when creating enclosed
        tessellation. When None, `create_tessellation` computes one from the
        buildings and barriers.
    primary_barrier_col : str, optional
        Column name containing alternative geometry for movement spaces. If specified and exists,
        this geometry will be used instead of the main geometry column for tessellation barriers.
    contiguity : str, default="queen"
        Type of spatial contiguity for place-to-place connections.
        Must be either "queen" or "rook".
    keep_buildings : bool, default=False
        If True, preserves building information in the tessellation output.
    keep_segments : bool, default=True
        If True, preserves the original segment LineString geometry in a column
        named 'segment_geometry' in the movement nodes GeoDataFrame.
    tolerance : float, default=1e-6
        Buffer distance for movement geometries when creating place-to-movement connections.
        This parameter controls how close place spaces need to be to movement spaces
        to establish a connection.
    include_unenclosed_buildings : bool, default=False
        If True, buildings excluded by enclosed tessellation are added using
        barrier-free tessellation before distance filters are applied.
    as_nx : bool, default=False
        If True, convert the output to a NetworkX graph.
    duplicate_edges : bool, default=False
        If True, the undirected same-type edge GeoDataFrames —
        ("place", "touched_to", "place") and
        ("movement", "connected_to", "movement") — contain both (u, v) and (v, u)
        rows (roughly doubling their row count), so neighbourhood queries on
        the MultiIndex are complete. The heterogeneous
        ("place", "faced_to", "movement") edges are left unchanged because a
        reverse row would mix movement ids into the place index level.
        Incompatible with ``as_nx=True``.

    Returns
    -------
    tuple[dict[str, geopandas.GeoDataFrame], dict[tuple[str, str, str], geopandas.GeoDataFrame]] | networkx.Graph
        If as_nx is False (default), returns a tuple (nodes, edges) where:

        - nodes: Dictionary containing node GeoDataFrames with keys:
            - "place": Tessellation cells (place spaces)
            - "movement": Street segments (movement spaces)

        - edges: Dictionary containing edge GeoDataFrames with keys:
            - ("place", "touched_to", "place"): Adjacency between tessellation cells
            - ("movement", "connected_to", "movement"): Connectivity between street segments
            - ("place", "faced_to", "movement"): Interface between tessellation cells and street segments

        If as_nx is True, returns a NetworkX graph.

    Raises
    ------
    TypeError
        If buildings_gdf or segments_gdf are not GeoDataFrames.
    ValueError
        If contiguity parameter is not "queen" or "rook".
        If clipping_buffer is negative.
        If `duplicate_edges` is True together with `as_nx=True`.

    See Also
    --------
    place_to_place_graph : Create adjacency between place spaces.
    place_to_movement_graph : Create connections between place and movement spaces.
    movement_to_movement_graph : Create connectivity between movement spaces.

    Notes
    -----
    The function first filters the street network by `distance` (resulting in `segs`).
    A `segs_buffer` GeoDataFrame is also created for tessellation context, potentially
    filtered by `distance + clipping_buffer` or `distance` if `center_point` and
    `distance` are provided. This `segs_buffer` is used to create enclosures and
    tessellations.
    It then establishes three types of relationships:
    1. Place-to-place: Adjacency between tessellation cells (handled by place_to_place_graph)
    2. Movement-to-movement: Topological connectivity between street segments
    3. Place-to-movement: Spatial interfaces between tessellations and streets

    The output follows a heterogeneous graph structure suitable for network analysis
    of urban morphology.

    Examples
    --------
    >>> # Create morphological graph from buildings and segments
    >>> nodes, edges = morphological_graph(buildings_gdf, segments_gdf)
    >>> place_nodes = nodes['place']
    >>> movement_nodes = nodes['movement']
    """
    # Define fixed ID column names
    place_id_col = "place_id"
    movement_id_col = "movement_id"

    _validate_duplicate_edges(duplicate_edges, as_nx)
    _validate_input_gdfs(buildings_gdf, segments_gdf)

    # Validate contiguity parameter
    if contiguity not in {"queen", "rook"}:
        msg = "contiguity must be 'queen' or 'rook'"
        raise ValueError(msg)

    # Validate clipping_buffer and extent_buffer
    if clipping_buffer < 0:
        msg = "clipping_buffer cannot be negative."
        raise ValueError(msg)
    if extent_buffer < 0:
        msg = "extent_buffer cannot be negative."
        raise ValueError(msg)
    if clipping_buffer < extent_buffer:
        msg = "clipping_buffer must be greater than or equal to extent_buffer."
        raise ValueError(msg)

    if isinstance(buildings_gdf.index, pd.MultiIndex):
        buildings_gdf = buildings_gdf.copy()
        buildings_gdf.index = buildings_gdf.index.to_flat_index()

    # Ensure CRS consistency between buildings and segments
    segments_gdf = _ensure_crs_consistency(buildings_gdf, segments_gdf)

    segments_gdf = segments_gdf.copy()
    segments_gdf[movement_id_col] = segments_gdf.index

    # Compute the single-source reachability cost field once on the full network so that
    # streets, buildings and cells are all judged against the same metric on the same network.
    reachability_field: _ReachabilityField | None = None
    if center_point is not None and distance is not None and not segments_gdf.empty:
        _, full_segment_edges = segments_to_graph(segments_gdf)
        reachability_field = _network_reachability_field(full_segment_edges, center_point)

    segments_filtered, segments_buffer = _process_segments(
        segments_gdf,
        center_point,
        distance,
        clipping_buffer,
        field=reachability_field,
    )

    tessellation = _create_and_filter_tessellation(
        buildings_gdf,
        segments_buffer,
        segments_filtered,
        primary_barrier_col,
        distance,
        clipping_buffer,
        center_point,
        keep_buildings,
        place_id_col,
        include_unenclosed_buildings,
        extent_buffer,
        limit=limit,
        field=reachability_field,
    )

    # When a reachability budget is applied, prune place cells that face no
    # retained street so the induced subgraph contains no isolated place nodes.
    drop_isolated_place = center_point is not None and distance is not None

    nodes, edges = _build_morphological_layers(
        tessellation,
        segments_filtered,
        primary_barrier_col,
        contiguity,
        tolerance,
        place_id_col,
        movement_id_col,
        keep_segments,
        drop_isolated_place=drop_isolated_place,
        extent_buffer=extent_buffer,
    )

    if duplicate_edges:
        # Only same-node-type undirected edges can hold reverse rows; the
        # heterogeneous faced_to edges would mix id spaces across index levels.
        for edge_type in (
            ("place", "touched_to", "place"),
            ("movement", "connected_to", "movement"),
        ):
            edges[edge_type] = symmetrize_edges(edges[edge_type])

    return (nodes, edges) if not as_nx else gdf_to_nx(nodes, edges)


# ============================================================================
# PLACE TO PLACE GRAPH
# ============================================================================


def place_to_place_graph(
    place_gdf: gpd.GeoDataFrame,
    group_col: str | None = None,
    contiguity: str = "queen",
    as_nx: bool = False,
    duplicate_edges: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph:
    """
    Create edges between contiguous place polygons based on spatial adjacency.

    This function identifies spatial adjacency relationships between place polygons
    (e.g., tessellation cells) using either Queen or Rook contiguity criteria.
    Optionally groups connections within specified groups (e.g., enclosures).
    The input `place_gdf` is expected to have a 'place_id' column.

    Parameters
    ----------
    place_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing place space polygons. Must contain a 'place_id' column.
    group_col : str, optional
        Column name for grouping connections. Only polygons within the same group
        will be connected. If None, all polygons are considered as one group.
    contiguity : str, default="queen"
        Type of spatial contiguity to use. Must be either "queen" or "rook".
        Queen contiguity includes vertex neighbors, Rook includes only edge neighbors.
    as_nx : bool, default=False
        If True, convert the output to a NetworkX graph.
    duplicate_edges : bool, default=False
        If True, the edges GeoDataFrame contains both (u, v) and (v, u) rows
        for every undirected edge (roughly doubling the row count), with the
        'from_place_id' and 'to_place_id' values swapped on reverse rows.
        Incompatible with ``as_nx=True``.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame] or networkx.Graph
        If as_nx is False (default), returns a tuple (nodes, edges) where:

        - nodes is a geopandas.GeoDataFrame containing the place nodes.
        - edges is a geopandas.GeoDataFrame containing the adjacency connections.

        If as_nx is True, returns a networkx.Graph representing the place adjacency.

    Raises
    ------
    TypeError
        If place_gdf is not a GeoDataFrame.
    ValueError
        If contiguity not in {"queen", "rook"}, or if group_col doesn't exist.
        If `duplicate_edges` is True together with `as_nx=True`.

    See Also
    --------
    morphological_graph : Main function that creates comprehensive morphological graphs.
    place_to_movement_graph : Create connections between place and movement spaces.
    movement_to_movement_graph : Create connectivity between movement spaces.

    Notes
    -----
    The function uses libpysal's spatial weights to determine adjacency relationships.
    Edge geometries are created as LineStrings connecting polygon centroids.
    Self-connections and duplicate edges are automatically filtered out.
    The input place_gdf is expected to have a 'place_id' column.

    Examples
    --------
    >>> # Create place-to-place adjacency graph
    >>> nodes, edges = place_to_place_graph(tessellation_gdf)
    >>> # With grouping by enclosures
    >>> nodes, edges = place_to_place_graph(tessellation_gdf, group_col='enclosure_id')
    """
    # Input validation
    _validate_duplicate_edges(duplicate_edges, as_nx)
    _validate_single_gdf_input(place_gdf, "place_gdf")

    place_id_col = "place_id"

    # If not empty, require the ID column
    if not place_gdf.empty and place_id_col not in place_gdf.columns:
        msg = f"Expected ID column '{place_id_col}' not found in place_gdf."
        raise ValueError(msg)

    # Validate that the contiguity type is supported
    if contiguity not in {"queen", "rook"}:
        msg = "contiguity must be either 'queen' or 'rook'"
        raise ValueError(msg)

    # Validate that the group column exists if specified
    if group_col and group_col not in place_gdf.columns:
        msg = f"group_col '{group_col}' not found in place_gdf columns"
        raise ValueError(msg)

    # Handle empty or insufficient data: return empty edges GeoDataFrame
    if place_gdf.empty or len(place_gdf) < 2:
        return _return_empty_place_edges(place_gdf, group_col, as_nx)

    # Deduplicate based on place_id to avoid libpysal errors
    # This handles cases where place_gdf has been joined with other data (e.g. buildings)
    # resulting in multiple rows per place space. We only need unique place spaces for the graph.
    gdf_unique = place_gdf.drop_duplicates(subset=[place_id_col])

    # Set index to place_id so contiguity_graph returns edges with correct IDs
    # We must ensure the index is unique for libpysal
    gdf_indexed = gdf_unique.set_index(place_id_col)

    _, edges_gdf = contiguity_graph(
        gdf_indexed,
        contiguity=contiguity,
        as_nx=False,
    )

    if edges_gdf.empty:
        return _return_empty_place_edges(place_gdf, group_col, as_nx)

    # contiguity_graph returns edges with MultiIndex (source, target) containing the index values
    # (which are now place_ids). We reset index to get them as columns.
    edges_gdf = edges_gdf.reset_index()

    cols = edges_gdf.columns.tolist()
    cols[0] = "from_place_id"
    cols[1] = "to_place_id"
    edges_gdf.columns = cols

    # Add group column if applicable and filter edges
    if group_col:
        # Create a mapping from place_id to group
        # Use gdf_unique to ensure unique index
        id_to_group = gdf_unique.set_index(place_id_col)[group_col]

        # Map group values to edges
        edges_gdf[group_col] = edges_gdf["from_place_id"].map(id_to_group)
        to_group = edges_gdf["to_place_id"].map(id_to_group)

        # Filter edges where source and target are in the same group
        edges_gdf = edges_gdf[edges_gdf[group_col] == to_group].copy()

    else:
        edges_gdf["group"] = 0

    if as_nx:
        return gdf_to_nx(nodes=gdf_unique, edges=edges_gdf)

    if duplicate_edges:
        edges_gdf = _symmetrize_edge_columns(edges_gdf, "from_place_id", "to_place_id")

    return gdf_unique, edges_gdf


# ============================================================================
# PLACE TO MOVEMENT GRAPH
# ============================================================================


def place_to_movement_graph(
    place_gdf: gpd.GeoDataFrame,
    movement_gdf: gpd.GeoDataFrame,
    primary_barrier_col: str | None = None,
    tolerance: float = 1e-6,
    as_nx: bool = False,
    max_connection_distance: float = math.inf,
    duplicate_edges: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph:
    """
    Create edges between place polygons and nearby movement geometries.

    This function identifies spatial relationships between place spaces (tessellations)
    and movement spaces (street segments) by finding intersections between buffered movement
    geometries and place polygons.
    Input GDFs are expected to have 'place_id' and 'movement_id' columns respectively.

    Parameters
    ----------
    place_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing place space polygons. Expected to have a 'place_id' column.
    movement_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing movement space geometries (typically LineStrings).
        Expected to have a 'movement_id' column.
    primary_barrier_col : str, optional
        Column name for alternative movement geometry. If specified and exists,
        this geometry will be used instead of the main geometry column.
    tolerance : float, default=1e-6
        Buffer distance for movement geometries to detect proximity to place spaces.
    as_nx : bool, default=False
        If True, convert the output to a NetworkX graph.
    max_connection_distance : float, default=``math.inf``
        Maximum distance for the nearest-movement fallback connection. Place
        cells matched by the ``tolerance`` proximity query are unaffected; cells
        connected only by the fallback are dropped when their nearest movement
        geometry lies farther than this distance, preventing long star-shaped
        edges to distant streets.
    duplicate_edges : bool, default=False
        If True, the edges GeoDataFrame contains both (u, v) and (v, u) rows
        for every edge (roughly doubling the row count). Note that on reverse
        rows the 'place_id' column holds a movement id and vice versa, because
        the endpoints are swapped. Incompatible with ``as_nx=True``.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame] or networkx.Graph
        If as_nx is False (default), returns a tuple (nodes, edges) where:

        - nodes is a geopandas.GeoDataFrame containing the combined place and movement nodes.
        - edges is a geopandas.GeoDataFrame containing the edges between place and movement geometries.

        If as_nx is True, returns a networkx.Graph representing the place-to-movement connections.

    Raises
    ------
    TypeError
        If place_gdf or movement_gdf are not GeoDataFrames.
    ValueError
        If 'place_id' or 'movement_id' columns are missing from input GDFs.
        If `duplicate_edges` is True together with `as_nx=True`.

    See Also
    --------
    morphological_graph : Main function that creates comprehensive morphological graphs.
    place_to_place_graph : Create adjacency between place spaces.
    movement_to_movement_graph : Create connectivity between movement spaces.

    Notes
    -----
    Edge geometries are created as LineStrings connecting the centroids of
    place polygons and movement geometries. The function uses spatial joins
    to identify overlapping areas within the specified tolerance.
    Input GDFs are expected to have 'place_id' and 'movement_id' columns respectively.

    Examples
    --------
    >>> # Create place-to-movement interface graph
    >>> nodes, edges = place_to_movement_graph(tessellation_gdf, segments_gdf)
    >>> # With custom tolerance
    >>> nodes, edges = place_to_movement_graph(tessellation_gdf, segments_gdf, tolerance=2.0)
    """
    # Input validation
    _validate_duplicate_edges(duplicate_edges, as_nx)
    _validate_single_gdf_input(place_gdf, "place_gdf")
    _validate_single_gdf_input(movement_gdf, "movement_gdf")

    place_id_col = "place_id"
    movement_id_col = "movement_id"

    # Handle empty data: return empty edges GeoDataFrame
    if place_gdf.empty or movement_gdf.empty:
        empty_edges = _create_empty_edges_gdf(place_gdf.crs, place_id_col, movement_id_col)
        all_nodes = pd.concat([place_gdf, movement_gdf], ignore_index=True)

        return (
            (all_nodes, empty_edges) if not as_nx else gdf_to_nx(nodes=all_nodes, edges=empty_edges)
        )

    # Ensure required ID columns exist in the input GeoDataFrames
    if place_id_col not in place_gdf.columns:
        msg = f"Expected ID column '{place_id_col}' not found in place_gdf."
        raise ValueError(msg)
    if movement_id_col not in movement_gdf.columns:
        msg = f"Expected ID column '{movement_id_col}' not found in movement_gdf."
        raise ValueError(msg)

    # Ensure CRS consistency between place and movement GeoDataFrames
    movement_gdf = _ensure_crs_consistency(place_gdf, movement_gdf)

    # Determine which geometry to use for the query
    query_geometry = (
        movement_gdf[primary_barrier_col]
        if primary_barrier_col and primary_barrier_col in movement_gdf.columns
        else movement_gdf.geometry
    )

    # Use spatial index query with 'dwithin' predicate for efficient proximity search
    # This avoids creating expensive buffer geometries
    # indices[0] -> index of the geometry passed to query (movement_gdf)
    # indices[1] -> index of the geometry in the sindex (place_gdf)
    movement_indices, place_indices = place_gdf.sindex.query(
        query_geometry,
        predicate="dwithin",
        distance=tolerance,
    )

    # Create a DataFrame from the indices
    joined = pd.DataFrame(
        {
            place_id_col: place_gdf.iloc[place_indices][place_id_col].to_numpy(),
            movement_id_col: movement_gdf.iloc[movement_indices][movement_id_col].to_numpy(),
        }
    )

    # Drop duplicate pairs of (place_id, movement_id)
    joined = joined.drop_duplicates()
    joined = _connect_unmatched_place_to_nearest_movement(
        joined,
        place_gdf,
        movement_gdf,
        query_geometry,
        place_id_col,
        movement_id_col,
        max_connection_distance,
    )

    edges_gdf = _create_place_movement_edges(
        joined,
        place_gdf,
        movement_gdf,
        place_id_col,
        movement_id_col,
    )

    nodes_gdf = pd.concat([place_gdf, movement_gdf], ignore_index=True)

    if as_nx:
        return gdf_to_nx(nodes=nodes_gdf, edges=edges_gdf)

    if duplicate_edges:
        edges_gdf = _symmetrize_edge_columns(edges_gdf, "place_id", "movement_id")

    return nodes_gdf, edges_gdf


def _connect_unmatched_place_to_nearest_movement(
    joined: pd.DataFrame,
    place_gdf: gpd.GeoDataFrame,
    movement_gdf: gpd.GeoDataFrame,
    query_geometry: gpd.GeoSeries,
    place_id_col: str,
    movement_id_col: str,
    max_distance: float = math.inf,
) -> pd.DataFrame:
    """
    Add one nearest movement edge for place polygons missed by the proximity query.

    The regular ``dwithin`` query captures true face relationships. Fallback
    tessellation cells can be raw building footprints, so they may not touch a
    street barrier. Those otherwise isolated cells are connected to their nearest
    movement segment to keep them integrated in the morphology graph, but only when
    that segment lies within ``max_distance``. Cells whose nearest segment is
    farther are left unconnected so a caller's isolated-node pruning can remove
    them instead of forcing a long star-shaped edge.

    Parameters
    ----------
    joined : pd.DataFrame
        DataFrame of proximity query results linking place and movement geometries.
    place_gdf : gpd.GeoDataFrame
        GeoDataFrame of place geometries (tessellation cells or building footprints).
    movement_gdf : gpd.GeoDataFrame
        GeoDataFrame of movement geometries (street segments).
    query_geometry : gpd.GeoSeries
        GeoSeries used to find nearest movement segments for missing place geometries.
    place_id_col : str
        Column name identifying place geometries.
    movement_id_col : str
        Column name identifying movement geometries.
    max_distance : float, default ``math.inf``
        Maximum place-to-movement distance for a fallback connection to be added.

    Returns
    -------
    pd.DataFrame
        DataFrame containing augmented connections for isolated place geometries.
    """
    connected_place_ids = set(joined[place_id_col]) if not joined.empty else set()
    unmatched_place = place_gdf.loc[
        ~place_gdf[place_id_col].isin(connected_place_ids),
    ].drop_duplicates(subset=[place_id_col])
    if unmatched_place.empty:
        return joined

    movement_query = gpd.GeoSeries(query_geometry.to_numpy(), crs=movement_gdf.crs)
    if movement_query.empty:
        return joined

    place_nearest_indices, movement_nearest_indices = movement_query.sindex.nearest(
        unmatched_place.geometry,
        return_all=False,
    )
    if len(place_nearest_indices) == 0:
        return joined

    # Drop fallback pairs whose nearest movement segment is beyond the cap so that
    # only genuinely street-facing cells receive a fallback faced_to edge.
    if not math.isinf(max_distance):
        matched_place = unmatched_place.geometry.to_numpy()[place_nearest_indices]
        matched_movement = movement_query.to_numpy()[movement_nearest_indices]
        gaps = gpd.GeoSeries(matched_place, crs=movement_gdf.crs).distance(
            gpd.GeoSeries(matched_movement, crs=movement_gdf.crs),
            align=False,
        )
        within = (gaps <= max_distance).to_numpy()
        place_nearest_indices = place_nearest_indices[within]
        movement_nearest_indices = movement_nearest_indices[within]
        if len(place_nearest_indices) == 0:
            return joined

    nearest = pd.DataFrame(
        {
            place_id_col: unmatched_place.iloc[place_nearest_indices][place_id_col].to_numpy(),
            movement_id_col: movement_gdf.iloc[movement_nearest_indices][
                movement_id_col
            ].to_numpy(),
        },
    )
    return pd.concat([joined, nearest], ignore_index=True).drop_duplicates()


# ============================================================================
# MOVEMENT TO MOVEMENT GRAPH
# ============================================================================


def movement_to_movement_graph(
    movement_gdf: gpd.GeoDataFrame,
    as_nx: bool = False,
    duplicate_edges: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph:
    """
    Create edges between connected movement segments based on topological connectivity.

    This function identifies topological connections between movement space geometries
    (typically street segments) using the dual graph approach to find segments
    that share endpoints or connection points.
    The function automatically creates a unique identifier for each row if needed.

    Parameters
    ----------
    movement_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing movement space geometries (typically LineString).
    as_nx : bool, default=False
        If True, convert the output to a NetworkX graph.
    duplicate_edges : bool, default=False
        If True, the edges GeoDataFrame contains both (u, v) and (v, u) rows
        for every undirected edge (roughly doubling the row count), with the
        'from_movement_id' and 'to_movement_id' values swapped on reverse rows.
        Incompatible with ``as_nx=True``.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame] or networkx.Graph
        If as_nx is False (default), returns a tuple (nodes, edges) where:

        - nodes is a geopandas.GeoDataFrame containing the movement nodes.
        - edges is a geopandas.GeoDataFrame containing the topological connections.

        If as_nx is True, returns a networkx.Graph representing the movement connectivity.

    Raises
    ------
    TypeError
        If movement_gdf is not a GeoDataFrame.
    ValueError
        If `duplicate_edges` is True together with `as_nx=True`.

    See Also
    --------
    morphological_graph : Main function that creates comprehensive morphological graphs.
    place_to_place_graph : Create adjacency between place spaces.
    place_to_movement_graph : Create connections between place and movement spaces.

    Notes
    -----
    The function uses the dual graph approach where each LineString becomes a node,
    and edges represent topological connections between segments. Edge geometries
    are created as LineStrings connecting the centroids of connected segments.

    Examples
    --------
    >>> # Create movement-to-movement connectivity graph
    >>> nodes, edges = movement_to_movement_graph(segments_gdf)
    >>> # Convert to NetworkX format
    >>> graph = movement_to_movement_graph(segments_gdf, as_nx=True)
    """
    # Input validation
    _validate_duplicate_edges(duplicate_edges, as_nx)
    _validate_single_gdf_input(movement_gdf, "movement_gdf")

    # Handle empty or insufficient data: return empty edges GeoDataFrame
    if movement_gdf.empty or len(movement_gdf) < 2:
        empty_edges = _create_empty_edges_gdf(
            movement_gdf.crs, "from_movement_id", "to_movement_id"
        )

        return (
            (movement_gdf, empty_edges)
            if not as_nx
            else gdf_to_nx(nodes=movement_gdf, edges=empty_edges)
        )

    # Create a copy to avoid modifying the original
    movement_gdf_copy = movement_gdf.copy()

    if "movement_id" in movement_gdf_copy.columns:
        edge_id_col = "movement_id"

    else:
        edge_id_col = "_edge_id"
        movement_gdf_copy[edge_id_col] = [str(idx) for idx in movement_gdf_copy.index]

    # Convert movement_gdf to a graph
    segment_nodes, segment_edges = segments_to_graph(movement_gdf_copy)

    _, dual_edges = dual_graph(
        (segment_nodes, segment_edges),
        edge_id_col=edge_id_col,
        keep_original_geom=True,
    )

    # Preserve the original MultiIndex structure and data types
    if isinstance(dual_edges.index, pd.MultiIndex):
        # Rename the MultiIndex levels for clarity and consistency
        dual_edges.index.names = ["from_movement_id", "to_movement_id"]

    # Reset index to ensure it is a regular DataFrame
    dual_edges = dual_edges.reset_index()

    if as_nx:
        return gdf_to_nx(nodes=movement_gdf, edges=dual_edges)

    if duplicate_edges:
        dual_edges = _symmetrize_edge_columns(dual_edges, "from_movement_id", "to_movement_id")

    return movement_gdf, dual_edges


# ============================================================================
# SEGMENTS TO GRAPH
# ============================================================================


def segments_to_graph(
    segments_gdf: gpd.GeoDataFrame,
    multigraph: bool = True,
    directed: bool = True,
    as_nx: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph | nx.MultiGraph:
    r"""
    Convert a GeoDataFrame of LineString segments into a graph structure.

    This function takes a GeoDataFrame of LineStrings and processes it into a
    topologically explicit graph representation, consisting of a GeoDataFrame of
    unique nodes (the endpoints of the lines) and a GeoDataFrame of edges.

    The resulting nodes GeoDataFrame contains unique points representing the start
    and end points of the input line segments. The edges GeoDataFrame is a copy
    of the input, but with a new MultiIndex (`from_node_id`, `to_node_id`) that
    references the IDs in the new nodes GeoDataFrame. If `multigraph` is True
    (the default), an additional index level (`edge_key`) distinguishes multiple
    edges between the same pair of nodes.

    Parameters
    ----------
    segments_gdf : geopandas.GeoDataFrame
        A GeoDataFrame where each row represents a line segment, and the
        'geometry' column contains LineString objects.
    multigraph : bool, default True
        If True, supports multiple edges between the same pair of nodes by
        adding an `edge_key` level to the MultiIndex. Real-world segment data
        (e.g. from OSMnx) commonly contains parallel edges, so this is the
        default. If False, duplicate node pairs raise a ``ValueError``.

        .. versionchanged:: 0.4
            The default changed from ``False`` to ``True``, and
            ``multigraph=False`` now raises on duplicate node pairs instead
            of silently returning a duplicated MultiIndex.
    directed : bool, default True
        If True, edge orientation follows the LineString draw order: the
        first coordinate becomes `from_node_id` and the last becomes
        `to_node_id`. If False, each non-self-loop edge is canonicalized to
        an unordered ``(min, max)`` node-id order, so the same road drawn in
        opposite directions yields parallel edges of one unordered pair
        instead of reciprocal ``(u, v)`` / ``(v, u)`` rows. Geometries are
        left unchanged; only the index order is normalized. Use
        ``directed=False`` when the output feeds an undirected pipeline such
        as ``gdf_to_pyg(..., directed=False)``.
    as_nx : bool, default False
        If True, returns a NetworkX graph instead of a tuple of GeoDataFrames.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame]
        A tuple containing two GeoDataFrames:

        - nodes_gdf: A GeoDataFrame of unique nodes (Points), indexed by `node_id`.
        - edges_gdf: A GeoDataFrame of edges (LineStrings), with a MultiIndex
          mapping to the `node_id` in `nodes_gdf`. If `multigraph` is True,
          the index includes a third level (`edge_key`) for duplicate connections.

    Raises
    ------
    ValueError
        If ``multigraph=False`` and the segments contain duplicate node pairs
        (after canonicalization when ``directed=False``).

    See Also
    --------
    city2graph.canonicalize_edges : Collapse reciprocal/parallel edge rows.

    Examples
    --------
    >>> import geopandas as gpd
    >>> from shapely.geometry import LineString
    >>> # Create a GeoDataFrame of line segments
    >>> segments = gpd.GeoDataFrame(
    ...     {"road_name": ["A", "B"]},
    ...     geometry=[LineString([(0, 0), (1, 1)]), LineString([(1, 1), (1, 0)])],
    ...     crs="EPSG:32633"
    ... )
    >>> # Convert to graph representation
    >>> nodes_gdf, edges_gdf = segments_to_graph(segments)
    >>> print(nodes_gdf)
    >>> print(edges_gdf)
    node_id  geometry
    0        POINT (0 0)
    1        POINT (1 1)
    2        POINT (1 0)
                                             road_name   geometry
    from_node_id to_node_id edge_key
    0            1          0                A           LINESTRING (0 0, 1 1)
    1            2          0                B           LINESTRING (1 1, 1 0)

    >>> # Duplicate connections become parallel edges with distinct keys
    >>> segments_with_duplicates = gpd.GeoDataFrame(
    ...     {"road_name": ["A", "B", "C"]},
    ...     geometry=[LineString([(0, 0), (1, 1)]),
    ...               LineString([(0, 0), (1, 1)]),
    ...               LineString([(1, 1), (1, 0)])],
    ...     crs="EPSG:32633"
    ... )
    >>> nodes_gdf, edges_gdf = segments_to_graph(segments_with_duplicates)
    >>> print(edges_gdf.index.names)
    ['from_node_id', 'to_node_id', 'edge_key']
    """
    processor = GeoDataProcessor()

    # Validate input
    segments_clean = processor.validate_gdf(segments_gdf, ["LineString"])

    if segments_clean is None or segments_clean.empty:
        empty_nodes = gpd.GeoDataFrame(columns=["geometry"], crs=segments_gdf.crs)
        empty_nodes.index.name = "node_id"
        empty_edges = gpd.GeoDataFrame(
            columns=["geometry"],
            crs=segments_gdf.crs,
            index=pd.MultiIndex.from_arrays([[], []], names=["from_node_id", "to_node_id"]),
        )
        if as_nx:
            return gdf_to_nx(nodes=empty_nodes, edges=empty_edges, multigraph=multigraph)
        return empty_nodes, empty_edges

    # Extract coordinates
    start_coords = processor.extract_coordinates(segments_clean, start=True)
    end_coords = processor.extract_coordinates(segments_clean, start=False)

    # Create unique nodes
    all_coords = pd.concat([start_coords, end_coords]).drop_duplicates()
    coord_to_id = {coord: i for i, coord in enumerate(all_coords)}

    # Create nodes GeoDataFrame efficiently using gpd.points_from_xy
    coords_array = all_coords.to_numpy()
    x_coords = [coord[0] for coord in coords_array]
    y_coords = [coord[1] for coord in coords_array]

    # Create nodes GeoDataFrame with unique node IDs
    nodes_gdf = gpd.GeoDataFrame(
        {
            "node_id": range(len(all_coords)),
            "geometry": gpd.points_from_xy(x_coords, y_coords),
        },
        crs=segments_clean.crs,
    ).set_index("node_id", drop=True)

    # Create edges with MultiIndex
    from_ids = start_coords.map(coord_to_id)
    to_ids = end_coords.map(coord_to_id)

    if not directed:
        # Canonicalize each edge to an unordered (min, max) node-id order so
        # reverse-drawn duplicate segments share one unordered pair. The
        # geometries themselves are left unchanged.
        from_arr = from_ids.to_numpy()
        to_arr = to_ids.to_numpy()
        from_ids = pd.Series(np.minimum(from_arr, to_arr), index=from_ids.index)
        to_ids = pd.Series(np.maximum(from_arr, to_arr), index=to_ids.index)

    edges_gdf = segments_clean.copy()

    if multigraph:
        # For multigraph, handle potential duplicate node pairs by adding edge keys
        edge_pairs_df = pd.DataFrame({"from_id": from_ids, "to_id": to_ids})
        edge_keys = edge_pairs_df.groupby(["from_id", "to_id"]).cumcount()

        edges_gdf.index = pd.MultiIndex.from_arrays(
            [from_ids, to_ids, edge_keys],
            names=["from_node_id", "to_node_id", "edge_key"],
        )
    else:
        duplicated = pd.DataFrame({"from_id": from_ids, "to_id": to_ids}).duplicated()
        if duplicated.any():
            msg = (
                f"Found {int(duplicated.sum())} duplicate node pair(s) with "
                f"multigraph=False. Pass multigraph=True (the default) to keep "
                f"them as parallel edges, or deduplicate the segments first."
            )
            raise ValueError(msg)
        edges_gdf.index = pd.MultiIndex.from_arrays(
            [from_ids, to_ids],
            names=["from_node_id", "to_node_id"],
        )

    if as_nx:
        return gdf_to_nx(nodes=nodes_gdf, edges=edges_gdf, multigraph=multigraph)
    return nodes_gdf, edges_gdf


# ============================================================================
# VALIDATION AND CRS HELPERS
# ============================================================================


def _validate_input_gdfs(buildings_gdf: gpd.GeoDataFrame, segments_gdf: gpd.GeoDataFrame) -> None:
    """
    Validate the primary input GeoDataFrames for the morphological graph function.

    This function ensures that the provided buildings and segments inputs are both
    GeoDataFrames and that their geometries are of the expected types (Polygons for
    buildings, LineStrings for segments). It serves as a critical initial check
    to prevent errors in downstream processing.

    Parameters
    ----------
    buildings_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing building polygons.
    segments_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing street segments.

    Raises
    ------
    TypeError
        If either input is not a GeoDataFrame.
    ValueError
        If the geometries are not of the expected types.

    See Also
    --------
    _validate_single_gdf_input : Validate a single GeoDataFrame.
    """
    if not isinstance(buildings_gdf, gpd.GeoDataFrame):
        msg = "buildings_gdf must be a GeoDataFrame"
        raise TypeError(msg)
    if not isinstance(segments_gdf, gpd.GeoDataFrame):
        msg = "segments_gdf must be a GeoDataFrame"
        raise TypeError(msg)

    if not buildings_gdf.empty:
        building_geom_types = buildings_gdf.geometry.geom_type.unique()
        if not all(geom_type in {"Polygon", "MultiPolygon"} for geom_type in building_geom_types):
            msg = (
                f"buildings_gdf must contain only Polygon or MultiPolygon geometries. "
                f"Found: {', '.join(building_geom_types)}"
            )
            raise ValueError(msg)

    if not segments_gdf.empty:
        # Assuming LineString is required for operations like dual_graph
        segment_geom_types = segments_gdf.geometry.geom_type.unique()
        if not all(geom_type == "LineString" for geom_type in segment_geom_types):
            msg = (
                f"segments_gdf must contain only LineString geometries. "
                f"Found: {', '.join(segment_geom_types)}"
            )
            raise ValueError(msg)


def _validate_single_gdf_input(
    gdf: gpd.GeoDataFrame,
    gdf_name: str,
) -> None:
    """
    Validate that a single input is a GeoDataFrame.

    This is a simple utility function to ensure that an input object is of type
    geopandas.GeoDataFrame, raising a TypeError with a descriptive message if it
    is not. It is used for validating individual geospatial inputs in various
    functions.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The GeoDataFrame to validate.
    gdf_name : str
        The name of the GeoDataFrame, used in the error message.

    Raises
    ------
    TypeError
        If the input is not a GeoDataFrame.

    See Also
    --------
    _validate_input_gdfs : Validate both buildings and segments GeoDataFrames.
    """
    if not isinstance(gdf, gpd.GeoDataFrame):
        msg = f"{gdf_name} must be a GeoDataFrame"
        raise TypeError(msg)


def _ensure_crs_consistency(
    target_gdf: gpd.GeoDataFrame,
    source_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Ensure that the source GeoDataFrame has the same CRS as the target.

    This function checks if the Coordinate Reference System (CRS) of the source
    GeoDataFrame matches that of the target. If they do not match, it reprojects
    the source to the target's CRS and issues a warning. This is essential for
    ensuring that spatial operations between the two GeoDataFrames are valid.

    Parameters
    ----------
    target_gdf : geopandas.GeoDataFrame
        The GeoDataFrame with the target CRS.
    source_gdf : geopandas.GeoDataFrame
        The GeoDataFrame to check and potentially reproject.

    Returns
    -------
    geopandas.GeoDataFrame
        The source GeoDataFrame, reprojected to the target CRS if necessary.

    Warns
    -----
    RuntimeWarning
        If a CRS mismatch is detected and reprojection is performed.
    UserWarning
        If the resolved CRS is geographic (degrees). Distance parameters
        (``distance``, ``extent_buffer``, ``tolerance``) are interpreted as
        metric and ``.centroid`` results are unreliable on such a CRS.
    """
    if source_gdf.crs != target_gdf.crs:
        warnings.warn(
            "CRS mismatch detected, reprojecting",
            RuntimeWarning,
            stacklevel=3,
        )  # Warn user
        source_gdf = source_gdf.to_crs(target_gdf.crs)

    if target_gdf.crs is not None and target_gdf.crs.is_geographic:
        warnings.warn(
            "Geometry is in a geographic CRS. Distance parameters "
            "(distance, extent_buffer, tolerance) are treated as metric and "
            "'.centroid' results are unreliable. Re-project to a projected CRS "
            "with 'GeoSeries.to_crs()' before building the morphological graph.",
            UserWarning,
            stacklevel=3,
        )
    return source_gdf


# ============================================================================
# ORCHESTRATION HELPERS (for morphological_graph)
# ============================================================================


def _process_segments(
    segments_gdf: gpd.GeoDataFrame,
    center_point: gpd.GeoSeries | gpd.GeoDataFrame | None,
    distance: float | None,
    clipping_buffer: float,
    field: _ReachabilityField | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Process segments: create graph, filter by distance, and create buffer context.

    This function converts the input segments GeoDataFrame to a graph representation,
    filters it by network distance if a center point and distance are provided,
    and creates a buffered version of the segments for tessellation context.

    Parameters
    ----------
    segments_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing street segments.
    center_point : geopandas.GeoSeries or geopandas.GeoDataFrame or None
        Center point(s) for spatial filtering.
    distance : float or None
        Maximum distance from ``center_point`` for spatial filtering.
    clipping_buffer : float
        Buffer distance to ensure adequate context for generating tessellation.
    field : _ReachabilityField or None, optional
        A precomputed cost field shared across node types. When supplied it is
        reused for the segment filter so the field is computed only once on the
        full network; when ``None`` the field is computed here from the segments.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame]
        A tuple containing:
        - segments_filtered: Segments filtered by distance (or all segments if no filter).
        - segments_buffer: Segments used for tessellation context (buffered filter).
    """
    # Convert segments to a graph representation for efficient filtering.
    if not segments_gdf.empty:
        segment_nodes, segment_edges = segments_to_graph(segments_gdf)
        segments_graph = gdf_to_nx(nodes=segment_nodes, edges=segment_edges)
    else:
        segment_nodes, segment_edges = segments_to_graph(segments_gdf)
        segments_graph = nx.Graph()

    # Filter segments by network distance for the final graph if center_point and distance are
    # provided. The same reachability cost field is reused for tessellation cells and buildings,
    # so every node type is judged against a single distance metric on the same network.
    if center_point is not None and distance is not None and not segments_gdf.empty:
        seg_field = (
            field if field is not None else _network_reachability_field(segment_edges, center_point)
        )
        segments_filtered = _segments_within_network_distance(
            segment_edges,
            center_point,
            distance,
            field=seg_field,
        )
    else:
        segments_filtered = segment_edges

    if center_point is not None and distance is not None and not segments_gdf.empty:
        if not math.isinf(clipping_buffer):
            # Finite clipping_buffer: use distance + clipping_buffer for segments_buffer radius
            buffer_radius = distance + clipping_buffer
            segments_buffer_graph = filter_graph_by_distance(
                segments_graph,
                center_point,
                buffer_radius,
            )
            segments_buffer = nx_to_gdf(segments_buffer_graph, nodes=False, edges=True)
        else:  # clipping_buffer is math.inf
            # Fallback to 'distance' as radius for segments_buffer
            segments_buffer_graph = filter_graph_by_distance(
                segments_graph,
                center_point,
                distance,
            )
            segments_buffer = nx_to_gdf(segments_buffer_graph, nodes=False, edges=True)
    else:
        # No center_point or no distance, so segments_buffer is not filtered by distance
        segments_buffer = segment_edges

    return segments_filtered, segments_buffer


def _create_and_filter_tessellation(  # noqa: PLR0913
    buildings_gdf: gpd.GeoDataFrame,
    segments_buffer: gpd.GeoDataFrame,
    segments_filtered: gpd.GeoDataFrame,
    primary_barrier_col: str | None,
    distance: float | None,
    clipping_buffer: float,
    center_point: gpd.GeoSeries | gpd.GeoDataFrame | None,
    keep_buildings: bool,
    place_id_col: str,
    include_unenclosed_buildings: bool = False,
    extent_buffer: float = math.inf,
    limit: gpd.GeoDataFrame | gpd.GeoSeries | BaseGeometry | None = None,
    field: _ReachabilityField | None = None,
) -> gpd.GeoDataFrame:
    """
    Create tessellation and apply spatial filters.

    This function creates a tessellation from buildings and barriers, renames the ID column,
    and filters the tessellation based on adjacency to segments and network distance.

    The wider ``segments_buffer`` only provides tessellation context (barriers and
    a coarse adjacency gate); building and cell retention are judged solely
    against ``segments_filtered`` (the reachable isochrone streets) so that
    retention and the ``faced_to`` edges that follow share a single street set.

    Parameters
    ----------
    buildings_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing building polygons.
    segments_buffer : geopandas.GeoDataFrame
        Buffered segments used for tessellation context (barriers and adjacency).
    segments_filtered : geopandas.GeoDataFrame
        Reachable isochrone segments used for adjacency and all retention checks.
    primary_barrier_col : str or None
        Column name containing alternative geometry for movement spaces.
    distance : float or None
        Maximum distance from ``center_point`` for spatial filtering.
    clipping_buffer : float
        Buffer distance to ensure adequate context for generating tessellation.
    center_point : geopandas.GeoSeries or geopandas.GeoDataFrame or None
        Center point(s) for spatial filtering.
    keep_buildings : bool
        If True, preserves building information in the tessellation output.
    place_id_col : str
        Name of the place ID column.
    include_unenclosed_buildings : bool, default=False
        If True, buildings excluded by enclosed tessellation are added using
        barrier-free tessellation before distance filters are applied.
    extent_buffer : float, default ``math.inf``
        Maximum perpendicular access distance from a street to a building/cell
        centroid for it to be retained by the network-distance filters.
    limit : geopandas.GeoDataFrame, geopandas.GeoSeries, shapely geometry, or None, optional
        Boundary passed to `create_tessellation` for enclosed tessellation.
    field : _ReachabilityField or None, optional
        Shared reachability cost field computed once on the full network. When
        provided it is reused for both the building and tessellation network
        filters so they share a single metric; when ``None`` each filter computes
        its own field from the segments it is given.

    Returns
    -------
    geopandas.GeoDataFrame
        The created and filtered tessellation GeoDataFrame.
    """
    barriers = _prepare_barriers(segments_buffer, primary_barrier_col)
    # Create tessellation based on buildings and prepared barriers
    tessellation_kwargs = {}
    if limit is not None:
        tessellation_kwargs["limit"] = limit
    tessellation = create_tessellation(
        buildings_gdf,
        primary_barriers=None if barriers.empty else barriers,  # Use barriers if available
        **tessellation_kwargs,
    )

    tessellation = tessellation.rename(columns={"tess_id": place_id_col})

    # Ensure the fixed place ID column exists, creating a sequential ID if necessary
    if place_id_col not in tessellation.columns:
        tessellation[place_id_col] = range(len(tessellation))  # Assign sequential place IDs

    # Retention is judged solely against the reachable isochrone streets so that
    # cell/building acceptance and the faced_to edges share a single street set.
    distance_segments = segments_filtered

    eligible_buildings = _filter_buildings_by_network_distance(
        buildings_gdf,
        distance_segments,
        center_point,
        distance,
        extent_buffer,
        field=field,
    )

    if include_unenclosed_buildings and not barriers.empty:
        tessellation = _include_unenclosed_building_tessellation(
            tessellation,
            eligible_buildings,
            place_id_col,
        )

    # Determine max_distance for filtering tessellation adjacent to segments
    max_distance_for_adj_filter = distance + clipping_buffer if distance is not None else math.inf

    tessellation = _filter_adjacent_tessellation(
        tessellation,
        segments_filtered,  # Use 'segments_filtered' (final graph segments) for adjacency check
        max_distance=max_distance_for_adj_filter,  # Max distance for adjacency
    )

    # Further filter tessellation by network distance if center_point and distance are specified
    if center_point is not None and distance is not None:
        tessellation = _filter_tessellation_by_network_distance(
            tessellation,
            distance_segments,
            center_point,
            distance,  # Max network distance
            extent_buffer,
            field=field,
        )

    # Optionally preserve building information by joining tessellation with buildings
    if keep_buildings:
        tessellation = _add_building_info(tessellation, eligible_buildings)

    return tessellation


def _include_unenclosed_building_tessellation(
    tessellation: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    place_id_col: str,
) -> gpd.GeoDataFrame:
    """
    Append fallback cells for buildings missed by the enclosed tessellation.

    Buildings that fall outside every enclosed tessellation cell are represented
    by their own footprint as a fallback cell, tagged with a synthetic place id
    and a source-building reference so they can still join the morphological graph.

    Parameters
    ----------
    tessellation : geopandas.GeoDataFrame
        The enclosed tessellation to augment.
    buildings_gdf : geopandas.GeoDataFrame
        Buildings eligible for inclusion, used to find the unenclosed ones.
    place_id_col : str
        Name of the place ID column.

    Returns
    -------
    geopandas.GeoDataFrame
        The tessellation with fallback cells appended, or the original when none
        are missing.
    """
    missing_buildings = _buildings_without_tessellation(tessellation, buildings_gdf)
    if missing_buildings.empty:
        return tessellation

    tessellation = tessellation.copy()
    if "enclosure_index" in tessellation.columns:
        tessellation["enclosure_index"] = tessellation["enclosure_index"].astype(str)

    fallback = gpd.GeoDataFrame(
        {place_id_col: missing_buildings.index.astype(str)},
        geometry=missing_buildings.geometry.to_numpy(),
        crs=missing_buildings.crs,
    )
    fallback[_SOURCE_BUILDING_INDEX_COL] = missing_buildings.index.to_numpy()
    fallback[place_id_col] = "fallback_" + fallback[place_id_col].astype(str)
    fallback["enclosure_index"] = "fallback"

    return gpd.GeoDataFrame(
        pd.concat([tessellation, fallback], ignore_index=True, sort=False),
        geometry=tessellation.geometry.name,
        crs=tessellation.crs,
    )


def _filter_buildings_by_network_distance(
    buildings_gdf: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
    center_point: gpd.GeoSeries | gpd.GeoDataFrame | Point | None,
    max_distance: float | None,
    extent_buffer: float = math.inf,
    field: _ReachabilityField | None = None,
) -> gpd.GeoDataFrame:
    """
    Filter buildings by network distance from a centre point.

    Buildings are kept when their centroid is reachable within ``max_distance``
    network distance *and* within ``extent_buffer`` access distance of a street
    on the shared reachability cost field. Filtering is skipped when no centre or
    distance budget is supplied, returning the buildings unchanged.

    Parameters
    ----------
    buildings_gdf : geopandas.GeoDataFrame
        Buildings to filter.
    segments : geopandas.GeoDataFrame
        Street segments used to build the reachability network.
    center_point : geopandas.GeoSeries or geopandas.GeoDataFrame or shapely.geometry.Point or None
        Origin point(s) for the reachability computation.
    max_distance : float or None
        Maximum network distance for a building to be retained.
    extent_buffer : float, default ``math.inf``
        Maximum perpendicular access distance from a street to the building.
    field : _ReachabilityField or None, optional
        A precomputed cost field shared across node types. When ``None`` the field
        is computed from ``segments`` and ``center_point``.

    Returns
    -------
    geopandas.GeoDataFrame
        The reachable subset of ``buildings_gdf``.
    """
    if buildings_gdf.empty or segments.empty or center_point is None or max_distance is None:
        return buildings_gdf.copy()

    keep_ilocs = _geometry_ilocs_within_network_distance(
        buildings_gdf.geometry.centroid,
        segments,
        center_point,
        max_distance,
        extent_buffer,
        field=field,
    )
    return buildings_gdf.iloc[keep_ilocs].copy()


def _buildings_without_tessellation(
    tessellation: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Identify buildings not covered by any tessellation cell.

    A spatial join flags buildings whose footprint intersects a tessellation
    cell; those that intersect none are returned as the unenclosed buildings.

    Parameters
    ----------
    tessellation : geopandas.GeoDataFrame
        The tessellation whose coverage is tested.
    buildings_gdf : geopandas.GeoDataFrame
        Buildings to test against the tessellation.

    Returns
    -------
    geopandas.GeoDataFrame
        The subset of buildings that intersect no tessellation cell.
    """
    if buildings_gdf.empty:
        return buildings_gdf.copy()
    if tessellation.empty:
        return buildings_gdf.copy()

    covered = gpd.sjoin(
        buildings_gdf[[buildings_gdf.geometry.name]],
        tessellation[[tessellation.geometry.name]],
        how="inner",
        predicate="intersects",
    )
    return buildings_gdf.loc[~buildings_gdf.index.isin(covered.index.unique())].copy()


def _build_morphological_layers(
    tessellation: gpd.GeoDataFrame,
    segments_filtered: gpd.GeoDataFrame,
    primary_barrier_col: str | None,
    contiguity: str,
    tolerance: float,
    place_id_col: str,
    movement_id_col: str,
    keep_segments: bool = True,
    drop_isolated_place: bool = False,
    extent_buffer: float = math.inf,
) -> tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]]:
    """
    Build the node and edge layers for the morphological graph.

    This function orchestrates the creation of place-to-place, movement-to-movement,
    and place-to-movement graphs and organizes them into a heterogeneous graph structure.

    Parameters
    ----------
    tessellation : geopandas.GeoDataFrame
        GeoDataFrame containing tessellation cells (place nodes).
    segments_filtered : geopandas.GeoDataFrame
        GeoDataFrame containing filtered segments (movement nodes).
    primary_barrier_col : str or None
        Column name containing alternative geometry for movement spaces.
    contiguity : str
        Type of spatial contiguity for place-to-place connections.
    tolerance : float
        Buffer distance for movement geometries when creating place-to-movement connections.
    place_id_col : str
        Name of the place ID column.
    movement_id_col : str
        Name of the movement ID column.
    keep_segments : bool, default True
        If True, preserves the original segment LineString geometry in a column
        named 'segment_geometry' in the movement nodes GeoDataFrame.
    drop_isolated_place : bool, default False
        If True, place cells with no place-to-movement (faced_to) connection
        are removed, along with the place-to-place edges that reference them.
        This guarantees an induced subgraph free of isolated place nodes and is
        enabled when a reachability budget (``center_point`` and ``distance``) is
        applied.
    extent_buffer : float, default ``math.inf``
        Maximum length of a nearest-movement fallback ``faced_to`` connection. A
        place cell that touches no street within ``tolerance`` and whose
        nearest street lies farther than ``extent_buffer`` receives no fallback
        edge, so ``drop_isolated_place`` can remove it instead of forcing a
        long star-shaped connection to a distant street.

    Returns
    -------
    tuple[dict[str, geopandas.GeoDataFrame], dict[tuple[str, str, str], geopandas.GeoDataFrame]]
        A tuple containing:
        - nodes: Dictionary with keys "place" and "movement" containing node GeoDataFrames
        - edges: Dictionary with relationship type keys containing edge GeoDataFrames
    """
    # Determine group_col for place_to_place_graph
    group_col_for_priv_priv: str | None = "enclosure_index"
    if group_col_for_priv_priv not in tessellation.columns:
        if not tessellation.empty:
            logger.warning(
                "Column '%s' not found in tessellation. "
                "Place-to-place graph will not use grouping.",
                group_col_for_priv_priv,
            )
        group_col_for_priv_priv = None

    _, place_to_place_edges = place_to_place_graph(
        tessellation,
        group_col=group_col_for_priv_priv,
        contiguity=contiguity,
    )

    _, movement_to_movement_edges = movement_to_movement_graph(
        segments_filtered,
    )

    _, place_to_movement_edges = place_to_movement_graph(
        tessellation,
        segments_filtered,
        primary_barrier_col=primary_barrier_col,
        tolerance=tolerance,
        max_connection_distance=extent_buffer,
    )

    # Log warning if no place-movement connections found
    if place_to_movement_edges.empty:
        logger.warning("No place to movement connections found")

    # Drop place cells that face no retained street so the induced subgraph
    # contains no isolated place nodes. Adjacency (touched_to) edges that
    # reference removed cells are pruned to keep the layers consistent.
    if drop_isolated_place and not tessellation.empty:
        connected_place_ids = (
            set(place_to_movement_edges[place_id_col].unique())
            if not place_to_movement_edges.empty
            else set()
        )
        keep_mask = tessellation[place_id_col].isin(connected_place_ids)
        if not keep_mask.all():
            tessellation = tessellation.loc[keep_mask].copy()
            if not place_to_place_edges.empty:
                place_to_place_edges = place_to_place_edges.loc[
                    place_to_place_edges["from_place_id"].isin(connected_place_ids)
                    & place_to_place_edges["to_place_id"].isin(connected_place_ids)
                ].copy()

    # Prepare place nodes with Point geometry (centroids)
    # Preserve original tessellation geometry in tessellation_geometry column
    place_nodes = tessellation.copy()
    place_nodes["tessellation_geometry"] = place_nodes.geometry
    # Convert geometry to centroid for place nodes
    place_nodes["geometry"] = place_nodes.geometry.centroid

    # Prepare movement nodes with Point geometry (centroids)
    # Optionally preserve original segment geometry in segment_geometry column
    movement_nodes = segments_filtered.copy()
    if keep_segments:
        movement_nodes["segment_geometry"] = movement_nodes.geometry
    # Convert geometry to centroid for movement nodes
    movement_nodes["geometry"] = movement_nodes.geometry.centroid

    nodes = {
        "place": _set_node_index(place_nodes, place_id_col),
        "movement": _set_node_index(movement_nodes, movement_id_col),
    }

    # Organize edges into a dictionary with relationship types as keys
    edges = {
        ("place", "touched_to", "place"): _set_edge_index(
            place_to_place_edges,
            "from_place_id",
            "to_place_id",
        ),
        ("movement", "connected_to", "movement"): _set_edge_index(
            movement_to_movement_edges,
            "from_movement_id",
            "to_movement_id",
        ),
        ("place", "faced_to", "movement"): _set_edge_index(
            place_to_movement_edges,
            "place_id",
            "movement_id",
        ),
    }

    return nodes, edges


# ============================================================================
# DATA PREPARATION HELPERS
# ============================================================================


def _prepare_barriers(
    segments: gpd.GeoDataFrame,
    geom_col: str | None,
) -> gpd.GeoDataFrame:
    """
    Prepare the barrier geometries for tessellation.

    This function selects the appropriate geometry column from the segments
    GeoDataFrame to be used as barriers in the tessellation process. If an
    alternative geometry column is specified and exists, it is used; otherwise,
    the default geometry column is used.

    Parameters
    ----------
    segments : geopandas.GeoDataFrame
        The street segments GeoDataFrame.
    geom_col : str, optional
        The name of an alternative geometry column to use for the barriers.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing the prepared barrier geometries.
    """
    if geom_col and geom_col in segments.columns and geom_col != "geometry":
        return gpd.GeoDataFrame(
            segments.drop(columns=["geometry"]),
            geometry=segments[geom_col],
            crs=segments.crs,
        )
    return segments.copy()


def _add_building_info(
    tessellation: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Add building information to tessellation.

    This function associates each tessellation cell with buildings whose
    representative points fall inside the cell. Fallback cells with a recorded
    source building index are matched exactly. It adds a new column
    `building_geometry` to the tessellation, containing the geometry of the
    matched building.

    Parameters
    ----------
    tessellation : geopandas.GeoDataFrame
        The tessellation GeoDataFrame to which building information will be added.
    buildings : geopandas.GeoDataFrame
        The GeoDataFrame containing building geometries.

    Returns
    -------
    geopandas.GeoDataFrame
        The tessellation GeoDataFrame with an added `building_geometry` column.
    """
    if tessellation.empty or buildings.empty:
        joined = tessellation.copy()
        joined["building_geometry"] = gpd.GeoSeries(
            [None] * len(joined), index=joined.index, crs=buildings.crs
        )
        return joined.drop(columns=[_SOURCE_BUILDING_INDEX_COL], errors="ignore")

    building_columns = [col for col in buildings.columns if col != buildings.geometry.name]
    points = gpd.GeoDataFrame(
        buildings[building_columns].copy(),
        geometry=buildings.geometry.representative_point(),
        crs=buildings.crs,
    )
    joined = gpd.sjoin(tessellation, points, how="left", predicate="contains")

    if "index_right" not in joined.columns:
        joined["index_right"] = None

    if _SOURCE_BUILDING_INDEX_COL in joined.columns:
        source_index = joined[_SOURCE_BUILDING_INDEX_COL]
        joined["index_right"] = source_index.combine_first(joined["index_right"])
        source_mask = source_index.notna()
        for column in building_columns:
            joined.loc[source_mask, column] = (
                source_index.loc[source_mask]
                .map(
                    buildings[column],
                )
                .to_numpy()
            )

    building_geom_map = buildings.geometry.to_dict()
    joined["building_geometry"] = gpd.GeoSeries(
        joined["index_right"].map(building_geom_map),
        index=joined.index,
        crs=buildings.crs,
    )

    return joined.drop(columns=["index_right", _SOURCE_BUILDING_INDEX_COL], errors="ignore")


# ============================================================================
# SPATIAL FILTERING HELPERS
# ============================================================================


def _filter_adjacent_tessellation(
    tessellation: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
    max_distance: float = math.inf,
) -> gpd.GeoDataFrame:
    """
    Filter tessellation cells to include only those adjacent to segments.

    This function filters a tessellation GeoDataFrame to retain only the cells
    that are within a specified maximum Euclidean distance of the provided street
    segments. If the tessellation is grouped by enclosures, the filtering is
    performed independently for each enclosure, considering only the segments
    that intersect that enclosure.

    Parameters
    ----------
    tessellation : geopandas.GeoDataFrame
        The tessellation GeoDataFrame to filter.
    segments : geopandas.GeoDataFrame
        The street segments to measure distance against.
    max_distance : float, default math.inf
        The maximum distance for a tessellation cell to be considered adjacent.

    Returns
    -------
    geopandas.GeoDataFrame
        The filtered tessellation GeoDataFrame.
    """
    # If tessellation is empty, return an empty GeoDataFrame with the same structure
    if tessellation.empty:
        return tessellation.copy()

    # If max_distance is infinite, no filtering is needed based on distance
    if math.isinf(max_distance):
        return tessellation.copy()

    # Check if 'enclosure_index' column exists for grouped processing
    enclosure_col = "enclosure_index" if "enclosure_index" in tessellation.columns else None

    # List to store filtered parts of tessellation
    filtered_parts: list[gpd.GeoDataFrame] = []

    groups = (
        tessellation.groupby(enclosure_col) if enclosure_col is not None else [(None, tessellation)]
    )

    # Build the segment spatial index once so each enclosure only examines its
    # candidate segments instead of scanning the whole network.
    segments_sindex = segments.sindex
    all_segments_union = segments.union_all()

    # Iterate over each enclosure group
    for _, group in groups:
        # Geometry of the current enclosure
        enclosure_geom = group.union_all()

        # Segments intersecting this enclosure, found via the spatial index.
        candidate_pos = segments_sindex.query(enclosure_geom, predicate="intersects")
        segment_union_in_enclosure = (
            all_segments_union
            if len(candidate_pos) == 0
            else segments.iloc[candidate_pos].union_all()
        )

        # Centroids of cells in this group
        centroids_in_group = group.geometry.centroid

        # Distances to segments in this enclosure
        distances_in_group = centroids_in_group.distance(segment_union_in_enclosure)

        # Filter cells in group by distance
        filtered_group = group.loc[distances_in_group <= max_distance]

        # Add filtered group to list
        if not filtered_group.empty:
            filtered_parts.append(filtered_group)

    if not filtered_parts:
        return tessellation.iloc[0:0].copy()

    # Concatenate all filtered parts into a single GeoDataFrame
    return gpd.GeoDataFrame(pd.concat(filtered_parts), crs=tessellation.crs)


def _filter_tessellation_by_network_distance(
    tessellation: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
    center_point: gpd.GeoSeries | gpd.GeoDataFrame | Point,
    max_distance: float,
    extent_buffer: float = math.inf,
    field: _ReachabilityField | None = None,
) -> gpd.GeoDataFrame:
    """
    Filter tessellation by network distance from a center point.

    This function filters a tessellation GeoDataFrame to include only those cells
    that are within a specified network distance from a given `center_point` and
    within ``extent_buffer`` access distance of a street. It constructs a spatial
    graph from the street segments and tests each cell centroid against the
    shared reachability cost field with two independent caps.

    Parameters
    ----------
    tessellation : geopandas.GeoDataFrame
        The tessellation GeoDataFrame to filter.
    segments : geopandas.GeoDataFrame
        The street segments GeoDataFrame used to build the network for distance calculations.
    center_point : shapely.geometry.Point or geopandas.GeoSeries or geopandas.GeoDataFrame
        The geographic center point(s) from which to calculate network distances.
    max_distance : float
        The maximum network distance (e.g., in meters) for a tessellation cell to be included.
    extent_buffer : float, default ``math.inf``
        Maximum perpendicular access distance from a street to a cell centroid.
    field : _ReachabilityField or None, optional
        A precomputed cost field shared across node types. When ``None`` the field
        is computed from ``segments`` and ``center_point``.

    Returns
    -------
    geopandas.GeoDataFrame
        The filtered tessellation GeoDataFrame, containing only cells within the specified
        network distance from the center point.
    """
    # Return a copy of tessellation if it or segments are empty
    if tessellation.empty or segments.empty:
        return tessellation.copy()

    keep_ilocs = _geometry_ilocs_within_network_distance(
        tessellation.geometry.centroid,
        segments,
        center_point,
        max_distance,
        extent_buffer,
        field=field,
    )

    # Return the subset of the original tessellation corresponding to the kept ilocs
    return tessellation.iloc[keep_ilocs].copy()


def _extract_center_geometry(
    center_point: gpd.GeoSeries | gpd.GeoDataFrame | Point,
) -> Point:
    """
    Extract a single representative Point from a centre input.

    Accepts the several centre representations used across the module and
    returns the first geometry as a plain Point for downstream snapping.

    Parameters
    ----------
    center_point : geopandas.GeoSeries or geopandas.GeoDataFrame or shapely.geometry.Point
        The centre input to normalise.

    Returns
    -------
    shapely.geometry.Point
        The first point geometry of the input.
    """
    if isinstance(center_point, gpd.GeoDataFrame):
        return typing.cast("Point", center_point.geometry.iloc[0])
    if isinstance(center_point, gpd.GeoSeries):
        return typing.cast("Point", center_point.iloc[0])
    return center_point


class _ReachabilityField(typing.NamedTuple):
    """
    Single-source reachability cost field shared across all node types.

    The field is computed once from a centre snapped onto the street network and
    is reused to derive the acceptance of street segments, buildings and
    tessellation cells alike, so every node type is judged against the same
    reachability metric on the same network.

    The node identities of ``endpoint_lengths`` and ``edge_records`` are the exact
    endpoint coordinate tuples assigned by :func:`gdf_to_nx` (which keys implicit
    nodes by their coordinate), so segment endpoints can be matched directly
    against the cost field without a separate rounding scheme. ``edge_tree`` is an
    STRtree over ``edge_records`` geometries (aligned by position) used to prune
    the edges that need examining for a given destination geometry.
    """

    endpoint_lengths: dict[typing.Hashable, float]
    edge_records: list[tuple[typing.Hashable, typing.Hashable, LineString, float]]
    source_edge: tuple[typing.Hashable, typing.Hashable, LineString, float]
    source_along: float
    source_access: float
    edge_tree: STRtree


def _network_reachability_field(
    segments: gpd.GeoDataFrame,
    center_point: gpd.GeoSeries | gpd.GeoDataFrame | Point,
) -> _ReachabilityField | None:
    """
    Compute the single-source reachability cost field for a street network.

    The centre point is snapped onto its nearest network edge (capturing the
    last leg as an along-edge plus access projection), and a single shortest
    path computation yields the cumulative cost to reach every network node.
    This cost field is the common basis from which both street and cell
    acceptance are derived.

    Parameters
    ----------
    segments : geopandas.GeoDataFrame
        Street segments used to build the network.
    center_point : geopandas.GeoSeries or geopandas.GeoDataFrame or shapely.geometry.Point
        Origin point(s) for the reachability computation.

    Returns
    -------
    _ReachabilityField or None
        The computed cost field, or None when the network has no usable edges
        or the centre cannot be snapped onto it.
    """
    graph = gdf_to_nx(edges=segments)
    _ensure_graph_edge_lengths(graph)
    edge_records = _network_edge_records(graph)
    if not edge_records:
        return None

    source_id = "__city2graph_center__"
    graph = graph.copy()
    source_edge, source_along, source_access = _connect_point_to_nearest_edge(
        graph,
        source_id,
        _extract_center_geometry(center_point),
        edge_records,
    )
    if source_id not in graph:
        logger.warning("Source node for distance filtering not found in graph.")
        return None

    endpoint_lengths = nx.single_source_dijkstra_path_length(
        graph,
        source_id,
        weight="length",
    )
    return _ReachabilityField(
        endpoint_lengths=endpoint_lengths,
        edge_records=edge_records,
        source_edge=source_edge,
        source_along=source_along,
        source_access=source_access,
        edge_tree=STRtree([record[2] for record in edge_records]),
    )


def _geometry_ilocs_within_network_distance(
    geometries: gpd.GeoSeries,
    segments: gpd.GeoDataFrame,
    center_point: gpd.GeoSeries | gpd.GeoDataFrame | Point,
    max_distance: float,
    extent_buffer: float = math.inf,
    field: _ReachabilityField | None = None,
) -> list[int]:
    """
    Return the integer positions of geometries reachable within the budget caps.

    Each geometry (typically a cell or building centroid) is tested against the
    shared reachability cost field with two independent caps: the network
    distance from the centre to the projection foot must be within
    ``max_distance`` and the perpendicular access distance from the foot to the
    geometry must be within ``extent_buffer``. The access term is never folded
    into the network budget, so a geometry whose only nearby street is
    disconnected (forcing a long straight-line access leg across barriers) is not
    retained on the strength of a distant reachable street.

    Only edges whose perpendicular access is within ``extent_buffer`` can satisfy
    the access cap, so the field's STRtree restricts the per-geometry scan to that
    neighbourhood instead of every network edge.

    Parameters
    ----------
    geometries : geopandas.GeoSeries
        Geometries whose reachability is evaluated, in their original order.
    segments : geopandas.GeoDataFrame
        Street segments used to build the reachability network. Ignored when a
        precomputed ``field`` is supplied.
    center_point : geopandas.GeoSeries or geopandas.GeoDataFrame or shapely.geometry.Point
        Origin point(s) for the reachability computation.
    max_distance : float
        Maximum network distance from the centre to the projection foot.
    extent_buffer : float, default ``math.inf``
        Maximum perpendicular access distance from a street to the geometry.
    field : _ReachabilityField or None, optional
        A precomputed cost field shared across node types. When ``None`` the
        field is computed from ``segments`` and ``center_point``, reproducing the
        standalone behaviour.

    Returns
    -------
    list[int]
        Integer positions (``iloc``) of the reachable geometries.
    """
    if field is None:
        field = _network_reachability_field(segments, center_point)
    if field is None:
        return []

    return [
        iloc
        for iloc, geometry in enumerate(geometries)
        if _reachable_within_caps(
            geometry,
            field.endpoint_lengths,
            _candidate_edge_records(geometry, field, extent_buffer),
            field.source_edge,
            field.source_along,
            field.source_access,
            max_distance,
            extent_buffer,
        )
    ]


def _candidate_edge_records(
    point: Point,
    field: _ReachabilityField,
    extent_buffer: float,
) -> list[tuple[typing.Hashable, typing.Hashable, LineString, float]]:
    """
    Restrict the network edges worth testing for a single destination point.

    Only edges whose perpendicular access distance is within ``extent_buffer`` can
    satisfy the access cap, so the field's STRtree returns just the edges in that
    neighbourhood. When ``extent_buffer`` is infinite no spatial restriction is
    possible and every edge is returned, preserving the exhaustive behaviour.

    Parameters
    ----------
    point : shapely.geometry.Point
        The destination point being tested.
    field : _ReachabilityField
        The shared cost field whose STRtree indexes the edge geometries.
    extent_buffer : float
        Maximum perpendicular access distance from a street to the point.

    Returns
    -------
    list[tuple[typing.Hashable, typing.Hashable, shapely.geometry.LineString, float]]
        The subset of ``field.edge_records`` near ``point``.
    """
    if math.isinf(extent_buffer):
        return field.edge_records
    candidate_ilocs = field.edge_tree.query(point, predicate="dwithin", distance=extent_buffer)
    return [field.edge_records[iloc] for iloc in candidate_ilocs]


def _ensure_graph_edge_lengths(graph: nx.Graph) -> None:
    """
    Populate a ``length`` weight on every graph edge in place.

    Edges that already carry a valid ``length`` are left untouched; otherwise the
    length is taken from the edge geometry, or from the straight-line distance
    between the endpoint positions when no geometry is available.

    Parameters
    ----------
    graph : networkx.Graph
        Graph whose edges are annotated with a ``length`` weight in place.
    """
    positions = nx.get_node_attributes(graph, "pos")
    for from_node, to_node, data in graph.edges(data=True):
        length = data.get("length")
        if length is not None and not pd.isna(length):
            continue

        geometry = data.get("geometry")
        if geometry is not None:
            data["length"] = geometry.length
            continue

        from_pos = positions.get(from_node)
        to_pos = positions.get(to_node)
        if from_pos is not None and to_pos is not None:
            data["length"] = Point(from_pos).distance(Point(to_pos))


def _network_edge_records(
    graph: nx.Graph,
) -> list[tuple[typing.Hashable, typing.Hashable, LineString, float]]:
    """
    Collect the traversable edges of a network as reusable records.

    Each record bundles the endpoint node ids, the edge geometry and its length;
    zero-length edges and edges with undetermined geometry are skipped so the
    records can drive both snapping and distance evaluation.

    Parameters
    ----------
    graph : networkx.Graph
        The street network graph to enumerate.

    Returns
    -------
    list[tuple[typing.Hashable, typing.Hashable, shapely.geometry.LineString, float]]
        One ``(from_node, to_node, geometry, length)`` tuple per usable edge.
    """
    records = []
    positions = nx.get_node_attributes(graph, "pos")
    for from_node, to_node, data in graph.edges(data=True):
        geometry = data.get("geometry")
        if geometry is None:
            from_pos = positions.get(from_node)
            to_pos = positions.get(to_node)
            if from_pos is None or to_pos is None:
                continue
            geometry = LineString([from_pos, to_pos])

        length = data.get("length")
        if length is None or pd.isna(length):
            length = geometry.length
        if length <= 0:
            continue
        records.append((from_node, to_node, geometry, float(length)))
    return records


def _connect_point_to_nearest_edge(
    graph: nx.Graph,
    point_node_id: typing.Hashable,
    point: Point,
    edge_records: list[tuple[typing.Hashable, typing.Hashable, LineString, float]],
) -> tuple[tuple[typing.Hashable, typing.Hashable, LineString, float], float, float]:
    """
    Attach a point to the network through its nearest edge.

    A temporary node is added for the point and connected to both endpoints of
    the nearest edge, weighting each connection by the perpendicular access plus
    the along-edge distance so the last leg is reflected in the cost field.

    Parameters
    ----------
    graph : networkx.Graph
        Graph that the point node is added to in place.
    point_node_id : typing.Hashable
        Identifier used for the temporary point node.
    point : shapely.geometry.Point
        The point to connect to the network.
    edge_records : list[tuple[typing.Hashable, typing.Hashable, shapely.geometry.LineString, float]]
        Candidate edges produced by :func:`_network_edge_records`.

    Returns
    -------
    tuple[tuple[typing.Hashable, typing.Hashable, shapely.geometry.LineString, float], float, float]
        The chosen edge record, the along-edge distance and the access distance.
    """
    edge_record = min(
        edge_records,
        key=lambda record: point.distance(record[2]),
    )
    from_node, to_node, geometry, length = edge_record
    along, access_distance = _projected_edge_access(point, geometry)
    graph.add_node(point_node_id, pos=(point.x, point.y))
    graph.add_edge(point_node_id, from_node, length=access_distance + along)
    graph.add_edge(point_node_id, to_node, length=access_distance + max(length - along, 0.0))
    return edge_record, along, access_distance


def _reachable_within_caps(
    point: Point,
    endpoint_lengths: dict[typing.Hashable, float],
    edge_records: list[tuple[typing.Hashable, typing.Hashable, LineString, float]],
    source_edge: tuple[typing.Hashable, typing.Hashable, LineString, float],
    source_along: float,
    source_access: float,
    max_distance: float,
    extent_buffer: float,
) -> bool:
    """
    Decide whether a point is reachable within both the network and access caps.

    For every reachable edge the point is projected onto its foot, splitting the
    last leg into two independently-bounded terms: the network cost to reach the
    foot (the centre endpoint costs plus the along-edge distance) and the
    perpendicular access distance from the foot to the point. The point is kept
    when some edge holds the network cost within ``max_distance`` *and* the
    access distance within ``extent_buffer``. Keeping the access term out of the
    network budget prevents a straight-line access leg that crosses barriers from
    being mistaken for walkable network distance.

    Parameters
    ----------
    point : shapely.geometry.Point
        The destination point to test.
    endpoint_lengths : dict[typing.Hashable, float]
        Shortest-path cost from the centre to each reachable network node.
    edge_records : list[tuple[typing.Hashable, typing.Hashable, shapely.geometry.LineString, float]]
        Candidate edges to examine, typically the subset near ``point`` returned
        by :func:`_candidate_edge_records`.
    source_edge : tuple[typing.Hashable, typing.Hashable, shapely.geometry.LineString, float]
        The edge onto which the centre was snapped.
    source_along : float
        Along-edge distance of the centre projection on ``source_edge``.
    source_access : float
        Access distance from the centre to ``source_edge``.
    max_distance : float
        Maximum network distance from the centre to the projection foot.
    extent_buffer : float
        Maximum perpendicular access distance from the foot to ``point``.

    Returns
    -------
    bool
        ``True`` when at least one edge satisfies both caps, ``False`` otherwise.
    """
    for edge_record in edge_records:
        from_node, to_node, geometry, length = edge_record
        from_length = endpoint_lengths.get(from_node)
        to_length = endpoint_lengths.get(to_node)
        if from_length is None and to_length is None:
            continue

        along, access_distance = _projected_edge_access(point, geometry)
        if access_distance > extent_buffer:
            continue

        network_cost = math.inf
        if edge_record is source_edge:
            network_cost = min(network_cost, source_access + abs(along - source_along))
        if from_length is not None:
            network_cost = min(network_cost, from_length + along)
        if to_length is not None:
            network_cost = min(network_cost, to_length + max(length - along, 0.0))
        if network_cost <= max_distance:
            return True
    return False


def _projected_edge_access(point: Point, line: LineString) -> tuple[float, float]:
    """
    Project a point onto a line and measure the access geometry.

    The projection captures how far along the line the nearest point lies and how
    far the point sits from the line, which together form the last-leg access.

    Parameters
    ----------
    point : shapely.geometry.Point
        The point to project.
    line : shapely.geometry.LineString
        The line to project onto.

    Returns
    -------
    tuple[float, float]
        The along-edge distance and the perpendicular access distance.
    """
    along = float(line.project(point))
    projected = line.interpolate(along)
    return along, point.distance(projected)


def _segments_within_network_distance(
    segments: gpd.GeoDataFrame,
    center_point: gpd.GeoSeries | gpd.GeoDataFrame | Point,
    max_distance: float,
    field: _ReachabilityField | None = None,
) -> gpd.GeoDataFrame:
    """
    Filter street segments by the shared network reachability cost field.

    A segment is retained when its nearest reachable endpoint lies within
    ``max_distance`` on the same cost field used for buildings and tessellation
    cells. This keeps segments that straddle the budget boundary (their reachable
    portion is within budget) instead of dropping them on a binary wholly-in/out
    test, and ensures street and cell acceptance derive from a single
    reachability metric.

    Segment endpoints are looked up directly against ``field.endpoint_lengths``,
    whose keys are the exact endpoint coordinate tuples assigned by
    :func:`gdf_to_nx`. This is the same identity scheme :func:`segments_to_graph`
    uses, so no separate coordinate-rounding strategy is needed.

    Parameters
    ----------
    segments : geopandas.GeoDataFrame
        Street segments to filter. All columns are preserved on the kept rows.
        Ignored when a precomputed ``field`` is supplied.
    center_point : geopandas.GeoSeries or geopandas.GeoDataFrame or shapely.geometry.Point
        Origin point(s) for the reachability computation.
    max_distance : float
        Maximum network distance for a segment to be retained.
    field : _ReachabilityField or None, optional
        A precomputed cost field shared across node types. When ``None`` the field
        is computed from ``segments`` and ``center_point``.

    Returns
    -------
    geopandas.GeoDataFrame
        The subset of ``segments`` reachable within ``max_distance``.
    """
    if segments.empty:
        return segments.copy()

    if field is None:
        field = _network_reachability_field(segments, center_point)
    if field is None:
        return segments.iloc[0:0].copy()

    keep_ilocs = [
        iloc
        for iloc, geometry in enumerate(segments.geometry)
        if _segment_min_reachable_cost(geometry, field.endpoint_lengths) <= max_distance
    ]
    return segments.iloc[keep_ilocs].copy()


def _segment_min_reachable_cost(
    geometry: LineString,
    endpoint_lengths: dict[typing.Hashable, float],
) -> float:
    """
    Return the cheapest reachable cost among a segment's endpoints.

    The reachable portion of a segment is bounded by its cheaper endpoint, so the
    minimum endpoint cost determines whether any part of the segment falls within
    the reachability budget. Endpoints are matched by their exact coordinate
    tuple, the node identity used by the reachability graph.

    Parameters
    ----------
    geometry : shapely.geometry.LineString
        The segment geometry to evaluate.
    endpoint_lengths : dict[typing.Hashable, float]
        Shortest-path cost from the centre to each reachable network node, keyed
        by the node's exact ``(x, y)`` coordinate tuple.

    Returns
    -------
    float
        The minimum endpoint cost, or ``math.inf`` when neither endpoint is
        reachable.
    """
    coordinates = list(geometry.coords)
    if not coordinates:
        return math.inf
    start = endpoint_lengths.get(coordinates[0], math.inf)
    end = endpoint_lengths.get(coordinates[-1], math.inf)
    return min(start, end)


# ============================================================================
# EDGE CREATION HELPERS
# ============================================================================


def _create_place_movement_edges(
    joined: gpd.GeoDataFrame,
    place_gdf: gpd.GeoDataFrame,
    movement_gdf: gpd.GeoDataFrame,
    place_id_col: str,
    movement_id_col: str,
) -> gpd.GeoDataFrame:
    """
    Create edge geometries between place and movement spaces.

    This function creates LineString geometries connecting the centroids of
    intersecting place and movement spaces.

    Parameters
    ----------
    joined : geopandas.GeoDataFrame
        DataFrame containing intersecting place and movement IDs.
    place_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing place space polygons.
    movement_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing movement space geometries.
    place_id_col : str
        Name of the place ID column.
    movement_id_col : str
        Name of the movement ID column.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing the created edges with LineString geometries.
    """
    joined_with_geom = joined.copy()

    if joined_with_geom.empty:
        # Create empty GeoDataFrame with required columns and geometry
        return gpd.GeoDataFrame(
            joined_with_geom,
            geometry=gpd.GeoSeries([], crs=place_gdf.crs),
            crs=place_gdf.crs,
        )

    # Filter to only relevant IDs to avoid computing centroids for the entire dataset
    relevant_place_ids = joined_with_geom[place_id_col].unique()
    relevant_movement_ids = joined_with_geom[movement_id_col].unique()

    place_subset = place_gdf[place_gdf[place_id_col].isin(relevant_place_ids)]
    movement_subset = movement_gdf[movement_gdf[movement_id_col].isin(relevant_movement_ids)]

    # Compute centroids only for the subset
    place_centroids_map = (
        place_subset.drop_duplicates(subset=[place_id_col])
        .set_index(place_id_col)
        .geometry.centroid
    )
    movement_centroids_map = (
        movement_subset.drop_duplicates(subset=[movement_id_col])
        .set_index(movement_id_col)
        .geometry.centroid
    )

    place_centroids = place_centroids_map.loc[joined_with_geom[place_id_col]].reset_index(
        drop=True,
    )
    movement_centroids = movement_centroids_map.loc[joined_with_geom[movement_id_col]].reset_index(
        drop=True,
    )

    # Extract coordinates and ensure 2D array shape
    place_coords = np.array(list(zip(place_centroids.x, place_centroids.y, strict=True)))
    movement_coords = np.array(list(zip(movement_centroids.x, movement_centroids.y, strict=True)))

    # Ensure coords are 2D by reshaping if needed
    place_coords = place_coords.reshape(-1, 2)
    movement_coords = movement_coords.reshape(-1, 2)

    # Stack coordinates for LineString creation
    line_coords = np.stack((place_coords, movement_coords), axis=1)
    joined_with_geom["geometry"] = list(sh_linestrings(line_coords))

    # Convert the DataFrame with edge geometries to a GeoDataFrame
    return gpd.GeoDataFrame(joined_with_geom, geometry="geometry", crs=place_gdf.crs)


# ============================================================================
# UTILITY HELPERS
# ============================================================================


def _return_empty_place_edges(
    place_gdf: gpd.GeoDataFrame,
    group_col: str | None,
    as_nx: bool,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph:
    """
    Return an empty place-to-place graph with the correct structure.

    This helper function creates an empty edge GeoDataFrame with the appropriate
    columns and returns it along with the input nodes, either as a tuple or
    as a NetworkX graph depending on the `as_nx` parameter.

    Parameters
    ----------
    place_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing place space polygons.
    group_col : str, optional
        Column name for grouping connections.
    as_nx : bool
        If True, convert the output to a NetworkX graph.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame] | networkx.Graph
        Empty graph structure.
    """
    group_cols = [group_col] if group_col else ["group"]
    empty_edges = _create_empty_edges_gdf(
        place_gdf.crs,
        "from_place_id",
        "to_place_id",
        group_cols,
    )
    return (place_gdf, empty_edges) if not as_nx else gdf_to_nx(nodes=place_gdf, edges=empty_edges)


def _create_empty_edges_gdf(
    crs: str | int | None,
    from_col: str,  # Name for the 'from' node ID column
    to_col: str,  # Name for the 'to' node ID column
    extra_cols: list[str] | None = None,  # Optional list of additional column names
) -> gpd.GeoDataFrame:
    """
    Create an empty edges GeoDataFrame with specified column structure.

    This helper function generates an empty GeoDataFrame suitable for representing
    graph edges, ensuring it has the correct column names for 'from' and 'to'
    node IDs, and optionally additional columns, along with a geometry column
    and a specified Coordinate Reference System (CRS). This is useful for
    initializing empty edge GeoDataFrames when no connections are found or
    when setting up a new graph structure.

    Parameters
    ----------
    crs : str, int, or None
        Coordinate reference system.
    from_col : str
        Name for the 'from' node ID column.
    to_col : str
        Name for the 'to' node ID column.
    extra_cols : list[str], optional
        Optional list of additional column names.

    Returns
    -------
    geopandas.GeoDataFrame
        Empty GeoDataFrame with specified columns.
    """
    # Initialize list of column names with 'from' and 'to' ID columns
    columns = [from_col, to_col]
    # Add any extra columns if provided
    if extra_cols:
        columns.extend(extra_cols)
    # Add the 'geometry' column name (standard for GeoDataFrames)
    columns.append("geometry")

    # Create an empty GeoDataFrame with the defined columns, geometry column, and CRS
    return gpd.GeoDataFrame(columns=columns, geometry="geometry", crs=crs)


def _set_node_index(gdf: gpd.GeoDataFrame, col: str) -> gpd.GeoDataFrame:
    """
    Set GeoDataFrame index using a specified column, if it exists.

    This function attempts to set the index of a GeoDataFrame to a specified column.
    If the GeoDataFrame is empty, it safely returns an empty GeoDataFrame with an
    empty index. Otherwise, it sets the specified column as the new index, dropping
    the column from the DataFrame's columns.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input GeoDataFrame.
    col : str
        Column name to use as index.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with index set if column exists.

    See Also
    --------
    _set_edge_index : Set multi-index for edge GeoDataFrames.

    Examples
    --------
    >>> indexed_gdf = _set_node_index(gdf, 'node_id')
    """
    if gdf.empty:
        # For an empty GDF, set an empty index.
        # Attempting to set_index with a non-existent column `col` would error.
        # If `col` is the intended index name, it can be assigned after.
        return gdf.set_index(pd.Index([]))  # Safest for empty

    return gdf.set_index(col, drop=True)


def _set_edge_index(
    gdf: gpd.GeoDataFrame,
    from_col: str,
    to_col: str,
) -> gpd.GeoDataFrame:
    """
    Set multi-index for edge GeoDataFrame.

    This function sets a MultiIndex on an edge GeoDataFrame using specified
    'from' and 'to' column names. This is crucial for representing graph edges
    where each edge is uniquely identified by its source and target nodes.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The edge GeoDataFrame to modify.
    from_col : str
        The name of the column to be used as the first level of the MultiIndex
        (representing the source node ID).
    to_col : str
        The name of the column to be used as the second level of the MultiIndex
        (representing the target node ID).

    Returns
    -------
    geopandas.GeoDataFrame
        The GeoDataFrame with the new MultiIndex applied.
    """
    return gdf.set_index([from_col, to_col])


# ============================================================================
# DEPRECATED ALIASES
# ============================================================================


def private_to_private_graph(
    place_gdf: gpd.GeoDataFrame,
    group_col: str | None = None,
    contiguity: str = "queen",
    as_nx: bool = False,
    duplicate_edges: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph:
    """
    Create edges between contiguous place polygons (deprecated).

    .. deprecated::
        Use :func:`place_to_place_graph` instead.

    Parameters
    ----------
    place_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing place polygons (e.g., tessellation cells).
    group_col : str, optional
        Column name used to group connections (e.g., enclosures).
    contiguity : str, default "queen"
        Contiguity criterion, either "queen" or "rook".
    as_nx : bool, default False
        If True, return a NetworkX graph instead of GeoDataFrames.
    duplicate_edges : bool, default False
        If True, include both directions of each undirected edge.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame] or networkx.Graph
        Nodes and edges GeoDataFrames, or a NetworkX graph if ``as_nx=True``.
    """
    warnings.warn(
        "private_to_private_graph is deprecated; use place_to_place_graph",
        DeprecationWarning,
        stacklevel=2,
    )
    return place_to_place_graph(
        place_gdf,
        group_col=group_col,
        contiguity=contiguity,
        as_nx=as_nx,
        duplicate_edges=duplicate_edges,
    )


def private_to_public_graph(
    place_gdf: gpd.GeoDataFrame,
    movement_gdf: gpd.GeoDataFrame,
    primary_barrier_col: str | None = None,
    tolerance: float = 1e-6,
    as_nx: bool = False,
    max_connection_distance: float = math.inf,
    duplicate_edges: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph:
    """
    Create edges between place polygons and movement geometries (deprecated).

    .. deprecated::
        Use :func:`place_to_movement_graph` instead.

    Parameters
    ----------
    place_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing place polygons (e.g., tessellation cells).
    movement_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing movement geometries (e.g., street segments).
    primary_barrier_col : str, optional
        Column containing alternative geometries to use for movement spaces.
    tolerance : float, default 1e-6
        Buffer distance for spatial joins between places and movements.
    as_nx : bool, default False
        If True, return a NetworkX graph instead of GeoDataFrames.
    max_connection_distance : float, default math.inf
        Maximum distance for connecting unmatched places to movements.
    duplicate_edges : bool, default False
        If True, include both directions of each undirected edge.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame] or networkx.Graph
        Nodes and edges GeoDataFrames, or a NetworkX graph if ``as_nx=True``.
    """
    warnings.warn(
        "private_to_public_graph is deprecated; use place_to_movement_graph",
        DeprecationWarning,
        stacklevel=2,
    )
    return place_to_movement_graph(
        place_gdf,
        movement_gdf,
        primary_barrier_col=primary_barrier_col,
        tolerance=tolerance,
        as_nx=as_nx,
        max_connection_distance=max_connection_distance,
        duplicate_edges=duplicate_edges,
    )


def public_to_public_graph(
    movement_gdf: gpd.GeoDataFrame,
    as_nx: bool = False,
    duplicate_edges: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph:
    """
    Create edges between connected movement segments (deprecated).

    .. deprecated::
        Use :func:`movement_to_movement_graph` instead.

    Parameters
    ----------
    movement_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing movement geometries (e.g., street segments).
    as_nx : bool, default False
        If True, return a NetworkX graph instead of GeoDataFrames.
    duplicate_edges : bool, default False
        If True, include both directions of each undirected edge.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame] or networkx.Graph
        Nodes and edges GeoDataFrames, or a NetworkX graph if ``as_nx=True``.
    """
    warnings.warn(
        "public_to_public_graph is deprecated; use movement_to_movement_graph",
        DeprecationWarning,
        stacklevel=2,
    )
    return movement_to_movement_graph(
        movement_gdf,
        as_nx=as_nx,
        duplicate_edges=duplicate_edges,
    )
