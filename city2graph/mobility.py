"""
Mobility / OD matrix utilities.

This module introduces the public function ``od_matrix_to_graph`` which
converts Origin-Destination (OD) data (adjacency matrices or edge lists)
into spatial graph representations used throughout the city2graph
ecosystem.

Notes
-----
This module includes a complete implementation of ``od_matrix_to_graph``:
input validation, zone alignment, conversion to a canonical edgelist,
thresholding and self-loop handling, optional geometry creation, and an
optional NetworkX export path.

Examples
--------
See the function docstring for usage examples with adjacency matrices,
NumPy arrays and edge lists (single/multi weight columns).
"""

from __future__ import annotations

import logging
import numbers
import warnings
from typing import TYPE_CHECKING
from typing import Literal
from typing import cast

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import get_coordinates
from shapely import is_empty
from shapely import is_missing
from shapely import linestrings

from city2graph._processing import row_batches

from .base import GeoDataProcessor

# Use city2graph converters for compatibility across the stack
from .utils import gdf_to_nx

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping

    import networkx as nx
    from numpy.typing import NDArray

__all__ = ["od_matrix_to_graph"]

# Logger for informational summaries and errors (warnings used for data quality)
logger = logging.getLogger(__name__)


def od_matrix_to_graph(  # noqa: PLR0913 (public API requires many parameters)
    od_data: pd.DataFrame | np.ndarray,
    zones_gdf: gpd.GeoDataFrame,
    zone_id_col: str | None = None,
    *,
    matrix_type: Literal["edgelist", "adjacency"] = "edgelist",
    source_col: str = "source",
    target_col: str = "target",
    weight_cols: list[str] | None = None,
    threshold: float | None = None,
    threshold_col: str | None = None,
    include_self_loops: bool = False,
    compute_edge_geometry: bool = True,
    directed: bool = True,
    as_nx: bool | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph | nx.DiGraph:
    """
    Convert OD data (edge list or adjacency matrix) into graph structures.

    Creates spatially-aware graphs from OD data following city2graph's
    GeoDataFrame-first design. Supports adjacency matrices (DataFrame or
    ndarray) and edgelists with one or multiple numeric weight columns.
    By default, this function returns a pair of GeoDataFrames representing
    nodes and edges. When ``directed=False``, the output is undirected: for
    each unordered pair {u, v}, the edge weight equals the sum of directed
    weights in both directions (u->v plus v->u). When a threshold is provided
    in undirected mode, it is applied after this summation. By default edges
    are directed and thresholded with the rule weight >= threshold (or, when
    no threshold provided, strictly > 0). Optionally, it can return a NetworkX
    graph when ``as_nx=True``.

    Parameters
    ----------
    od_data : pandas.DataFrame | numpy.ndarray
        * When ``matrix_type='adjacency'``: a square DataFrame whose
          index & columns are zone IDs, or a square ndarray whose ordering
          matches ``zones_gdf``.
        * When ``matrix_type='edgelist'``: a DataFrame containing origin,
          destination and one or more numeric flow columns.
    zones_gdf : geopandas.GeoDataFrame
        GeoDataFrame of zones. Must contain unique identifiers in
        ``zone_id_col``.
    zone_id_col : str, optional
        Name of the zone ID column in ``zones_gdf`` (required in this
        initial skeleton; automatic inference may be added later).
    matrix_type : {'edgelist','adjacency'}, default 'edgelist'
        Declares how to interpret ``od_data``.
    source_col, target_col : str, default 'source','target'
        Column names for origins / destinations when using an edge list.
    weight_cols : Sequence[str] | None
        Edge list weight (flow) columns to preserve. A single column acts as
        the canonical weight. If multiple columns are provided a
        ``threshold_col`` must be designated. If a requested column is named
        ``weight``, it must be the primary column to avoid overwriting its values.
        Rename it before selecting a different primary column.
    threshold : float, optional
        Minimum flow retained (>=) applied to ``threshold_col``. Undirected
        graphs apply the cutoff to the sum of reciprocal flows. When
        ``matrix_type='adjacency'`` and ``threshold <= 0`` (for example,
        ``threshold=0``), zero-valued cells are also retained as edges, which
        produces a dense graph of :math:`O(N^2)` edges for large matrices.
    threshold_col : str, optional
        Column among ``weight_cols`` used for thresholding & canonical weight
        (required when ``len(weight_cols) > 1`` in full implementation).
    include_self_loops : bool, default False
        Keep flows where origin == destination (defaults drop when False).
    compute_edge_geometry : bool, default True
        Whether to build LineString geometries from zone centroids.
    directed : bool, default True
        Whether to build a directed graph. If False, reciprocal edges are
        merged by summing their weights (and all provided weight columns).
    as_nx : bool, optional
        If True, final output will be an NetworkX graph (``nx.DiGraph`` when
        ``directed=True``; otherwise ``nx.Graph``). If not explicitly set,
        defaults to False (GeoDataFrame output).

        .. deprecated::
            ``as_nx`` is deprecated and will be removed in a future release.
            Use :func:`gdf_to_nx` on the returned GeoDataFrames instead to
            obtain a NetworkX graph.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame] or networkx.Graph or networkx.DiGraph
        The graph representation in the requested format:

        *   When ``as_nx=False`` (default): Returns a tuple ``(nodes, edges)`` of
            GeoDataFrames. The nodes GeoDataFrame index is aligned with the zone
            identifier. The edges GeoDataFrame uses a pandas MultiIndex on
            (source_id, target_id).
        *   When ``as_nx=True``: Returns a NetworkX graph. A ``networkx.DiGraph``
            is returned if ``directed=True``, otherwise a ``networkx.Graph``.
    """
    if as_nx is not None:
        warnings.warn(
            "`as_nx` is deprecated and will be removed in a future release. "
            "Use `gdf_to_nx()` on the returned GeoDataFrames instead to "
            "obtain a NetworkX graph.",
            DeprecationWarning,
            stacklevel=2,
        )
    as_nx = bool(as_nx)

    # --- Validation (Task 2) ------------------------------------------------
    _validate_od_inputs(
        od_data,
        zones_gdf,
        zone_id_col=zone_id_col,
        matrix_type=matrix_type,
        source_col=source_col,
        target_col=target_col,
        weight_cols=weight_cols,
        threshold=threshold,
        threshold_col=threshold_col,
    )

    # --- Conversion to canonical edgelist ----------------------------------
    edge_df = _build_canonical_edgelist(
        od_data,
        zones_gdf,
        zone_id_col=zone_id_col,
        matrix_type=matrix_type,
        source_col=source_col,
        target_col=target_col,
        weight_cols=weight_cols,
        threshold=threshold,
        directed=directed,
        threshold_col=threshold_col,
        include_self_loops=include_self_loops,
    )

    # --- Spatial assembly (Task 5) -----------------------------------------
    nodes_gdf, edges_gdf = _assemble_od_frames(
        edge_df,
        zones_gdf,
        zone_id_col=zone_id_col,
        compute_edge_geometry=compute_edge_geometry,
    )

    # --- Output selection (Task 6) -----------------------------------------
    if not as_nx:
        # GeoDataFrame-first API: return (nodes, edges)
        logger.info("Created graph with %d nodes and %d edges", len(nodes_gdf), len(edges_gdf))
        return nodes_gdf, edges_gdf

    G = gdf_to_nx(
        nodes=nodes_gdf, edges=edges_gdf, keep_geom=compute_edge_geometry, directed=directed
    )
    logger.info(
        "Created graph with %d nodes and %d edges", G.number_of_nodes(), G.number_of_edges()
    )
    return G


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _validate_od_inputs(
    od_data: pd.DataFrame | np.ndarray,
    zones_gdf: gpd.GeoDataFrame,
    *,
    zone_id_col: str | None,
    matrix_type: Literal["edgelist", "adjacency"],
    source_col: str,
    target_col: str,
    weight_cols: list[str] | None,
    threshold: float | None,
    threshold_col: str | None,
) -> None:
    """
    Validate all inputs of :func:`od_matrix_to_graph`.

    Runs the zone-frame checks, CRS warnings, threshold type check and the
    ``matrix_type``-specific validation of ``od_data`` before any conversion
    takes place.

    Parameters
    ----------
    od_data : pandas.DataFrame | numpy.ndarray
        OD data as an edge list or adjacency matrix.
    zones_gdf : geopandas.GeoDataFrame
        GeoDataFrame of zones with unique identifiers.
    zone_id_col : str | None
        Name of the zone ID column in ``zones_gdf``.
    matrix_type : {'edgelist', 'adjacency'}
        Declares how to interpret ``od_data``.
    source_col : str
        Column name for origins when using an edge list.
    target_col : str
        Column name for destinations when using an edge list.
    weight_cols : list[str] | None
        Edge list weight (flow) columns to preserve.
    threshold : float | None
        Minimum flow retained, applied to the canonical weight.
    threshold_col : str | None
        Column among ``weight_cols`` used for thresholding.

    Returns
    -------
    None
        This function validates input and raises on failure.
    """
    _validate_zones_gdf(zones_gdf, zone_id_col)
    GeoDataProcessor.warn_crs_issues(
        zones_gdf,
        missing_message="zones_gdf has no CRS set; outputs will have undefined CRS (requirement 3.1)",
        geographic_message=(
            "Geographic CRS detected; distance/length measures may be inaccurate (requirement 3.5)"
        ),
    )

    # Validate matrix_type and that threshold is numeric if provided (Req 2.7)
    if threshold is not None and not isinstance(threshold, numbers.Number):
        msg = "threshold must be numeric (int or float)"
        raise ValueError(msg)

    validators: Mapping[str, Callable[..., None]] = {
        "adjacency": _validate_adjacency_data,
        "edgelist": _validate_edgelist_data,
    }
    if matrix_type not in validators:
        _msg = "matrix_type must be 'edgelist' or 'adjacency'"
        raise ValueError(_msg)
    validators[matrix_type](
        od_data,
        zones_gdf=zones_gdf,
        source_col=source_col,
        target_col=target_col,
        weight_cols=weight_cols,
    )
    if matrix_type == "edgelist":
        # Edgelist specifics: validate primary/threshold relationships
        # weight_cols is required for edgelist inputs (validated above)
        assert weight_cols is not None
        _validate_weights_threshold(
            cast("pd.DataFrame", od_data),
            weight_cols=weight_cols,
            _threshold=threshold,
            threshold_col=threshold_col,
        )


def _build_canonical_edgelist(
    od_data: pd.DataFrame | np.ndarray,
    zones_gdf: gpd.GeoDataFrame,
    *,
    zone_id_col: str | None,
    matrix_type: Literal["edgelist", "adjacency"],
    source_col: str,
    target_col: str,
    weight_cols: list[str] | None,
    threshold: float | None,
    directed: bool,
    threshold_col: str | None,
    include_self_loops: bool,
) -> pd.DataFrame:
    """
    Convert validated OD data into a canonical edgelist.

    Dispatches on ``matrix_type`` and the concrete ``od_data`` container
    (edge list, adjacency DataFrame or adjacency ndarray), aligning zone
    labels, merging reciprocal flows when ``directed=False``, applying
    thresholding, and normalising to canonical ``source``/``target``/``weight``
    columns. Guarantees the canonical columns exist even for empty results.

    Parameters
    ----------
    od_data : pandas.DataFrame | numpy.ndarray
        OD data as an edge list or adjacency matrix.
    zones_gdf : geopandas.GeoDataFrame
        GeoDataFrame of zones with unique identifiers.
    zone_id_col : str | None
        Name of the zone ID column in ``zones_gdf``.
    matrix_type : {'edgelist', 'adjacency'}
        Declares how to interpret ``od_data``.
    source_col : str
        Column name for origins when using an edge list.
    target_col : str
        Column name for destinations when using an edge list.
    weight_cols : list[str] | None
        Edge list weight (flow) columns to preserve.
    threshold : float | None
        Minimum flow retained, or the default positive-flow cutoff.
    directed : bool
        Whether to keep directed edges or merge reciprocal pairs before
        thresholding.
    threshold_col : str | None
        Column among ``weight_cols`` used for thresholding.
    include_self_loops : bool
        Keep flows where origin == destination.

    Returns
    -------
    pandas.DataFrame
        Edgelist with canonical ``source``, ``target`` and ``weight`` columns.
    """
    if matrix_type == "edgelist":
        # Filter to zones and aggregate duplicates first
        aligned = _align_edgelist_zones(
            cast("pd.DataFrame", od_data),
            zones_gdf=zones_gdf,
            zone_id_col=zone_id_col,
            source_col=source_col,
            target_col=target_col,
        )
        # Normalize: thresholding, self-loops policy, canonical columns
        edge_df = _normalize_edgelist(
            aligned,
            source_col=source_col,
            target_col=target_col,
            weight_cols=weight_cols if weight_cols is not None else [],
            threshold=threshold,
            directed=directed,
            threshold_col=threshold_col,
            include_self_loops=include_self_loops,
        )
    elif matrix_type == "adjacency" and isinstance(od_data, pd.DataFrame):
        # Align labels with zones
        adj = _align_adjacency_zones(
            od_data,
            zones_gdf=zones_gdf,
            zone_id_col=zone_id_col,
        )
        edge_df = _adjacency_to_edgelist(
            adj,
            include_self_loops=include_self_loops,
            threshold=threshold,
            directed=directed,
        )
    elif matrix_type == "adjacency" and isinstance(od_data, np.ndarray):
        zone_ids = _align_numpy_array_zones(
            od_data,
            zones_gdf=zones_gdf,
            zone_id_col=zone_id_col,
        )
        edge_df = _adjacency_to_edgelist(
            od_data,
            zone_ids,
            include_self_loops=include_self_loops,
            threshold=threshold,
            directed=directed,
        )

    # Ensure canonical columns exist even if empty result
    if edge_df.empty:
        edge_df = _empty_edgeframe(
            include_extra_weights=(matrix_type == "edgelist" and bool(weight_cols)),
            extra_weights=(weight_cols or []),
        )
    return edge_df


def _assemble_od_frames(
    edge_df: pd.DataFrame,
    zones_gdf: gpd.GeoDataFrame,
    *,
    zone_id_col: str | None,
    compute_edge_geometry: bool,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Assemble the spatial node and edge frames from a canonical edgelist.

    Aligns the node index with the zone identifier, creates edge geometries
    from zone centroids when requested, and converts the edge frame to a
    (source, target) MultiIndex.

    Parameters
    ----------
    edge_df : pandas.DataFrame
        Canonical edgelist with ``source``/``target``/``weight`` columns.
    zones_gdf : geopandas.GeoDataFrame
        GeoDataFrame of zones with unique identifiers.
    zone_id_col : str | None
        Name of the zone ID column in ``zones_gdf``.
    compute_edge_geometry : bool
        Whether to build LineString geometries from zone centroids.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame]
        The ``(nodes, edges)`` GeoDataFrames.
    """
    # Nodes: set index aligned with zone identifier (column or original index)
    nodes_gdf = (
        zones_gdf.set_index(zone_id_col, drop=False).copy()
        if zone_id_col is not None
        else zones_gdf.copy()
    )

    # Edges: create geometry, then convert to MultiIndex (source,target)
    edges_gdf = _create_edge_geometries(
        edge_df,
        zones_gdf,
        zone_id_col=zone_id_col,
        source_col="source",
        target_col="target",
        compute_edge_geometry=compute_edge_geometry,
    )

    # Convert to MultiIndex using the node identifiers and drop source/target columns
    if not edges_gdf.empty:
        mi = pd.MultiIndex.from_arrays(
            [edges_gdf["source"].to_numpy(), edges_gdf["target"].to_numpy()],
            names=["source", "target"],
        )
        edges_gdf = edges_gdf.drop(columns=["source", "target"])  # keep canonical weight/attrs
        edges_gdf.index = mi
        # Ensure CRS preserved (GeoPandas keeps it on GeoDataFrame)

    return nodes_gdf, edges_gdf


def _validate_zones_gdf(zones_gdf: gpd.GeoDataFrame, zone_id_col: str | None) -> None:
    """
    Validate the zones GeoDataFrame structure.

    Ensures the presence of a valid identifier column and basic integrity
    constraints (non-null, unique IDs) before downstream processing.

    Parameters
    ----------
    zones_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing zone geometries and attributes.
    zone_id_col : str | None
        Column name holding unique zone identifiers. Must be provided and
        present in ``zones_gdf``.

    Returns
    -------
    None
        This function validates input and raises on failure.
    """
    # Ensure zones_gdf is a GeoDataFrame (public API contracts/tests)
    if not isinstance(zones_gdf, gpd.GeoDataFrame):
        msg = "zones_gdf must be a GeoDataFrame"
        raise TypeError(msg)

    # Accept either an explicit zone id column or the DataFrame index
    if zone_id_col is not None:
        if zone_id_col not in zones_gdf.columns:
            msg = f"zone_id_col '{zone_id_col}' not found in zones_gdf columns"
            raise ValueError(msg)
        ids = zones_gdf[zone_id_col]
    else:
        ids = zones_gdf.index
    if ids.isna().any():
        msg = "zone_id_col contains null values"
        raise ValueError(msg)

    if not ids.is_unique:
        msg = "zone_id_col values must be unique"
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# Data normalization helpers
#   - _adjacency_to_edgelist: fast vectorized conversion using NumPy
#   - _normalize_edgelist: multi-attribute aggregation and thresholding
#   - _apply_threshold: common filtering semantics
# ---------------------------------------------------------------------------
def _extract_array_and_ids(
    adjacency: pd.DataFrame | np.ndarray,
    zone_ids: pd.Index | None,
) -> tuple[np.ndarray, pd.Index]:
    """
    Extract a numeric ndarray and matching zone ids from input.

    Accepts either a labeled DataFrame (preferred) or a raw ndarray with
    externally provided zone identifiers for consistent downstream mapping.

    Parameters
    ----------
    adjacency : pandas.DataFrame | numpy.ndarray
        Adjacency matrix as DataFrame (labels in index/columns) or ndarray.
    zone_ids : pandas.Index | None
        Zone identifiers when ``adjacency`` is an ndarray.

    Returns
    -------
    tuple[numpy.ndarray, pandas.Index]
        Tuple of (array, zone_ids).
    """
    if isinstance(adjacency, pd.DataFrame):
        ids = pd.Index(adjacency.index)
        arr = adjacency.to_numpy(dtype=float, copy=False)
        return arr, ids

    arr = np.asarray(adjacency, dtype=float)
    # Upstream ensures zone_ids is provided for ndarray inputs
    return arr, cast("pd.Index", zone_ids)


def _build_adjacency_mask(
    arr: np.ndarray,
    *,
    include_self_loops: bool,
    threshold: float | None,
) -> NDArray[np.bool_]:
    """
    Build a boolean mask selecting matrix entries to keep.

    The mask encodes the default thresholding semantics used throughout the
    module: when ``threshold`` is ``None`` entries strictly greater than 0 are
    selected, otherwise entries greater than or equal to the provided
    ``threshold`` are selected. When ``include_self_loops`` is ``False``, the
    diagonal is set to ``False`` to drop self-loops.

    Parameters
    ----------
    arr : numpy.ndarray
        Square adjacency matrix of weights to be filtered.
    include_self_loops : bool
        Whether to keep diagonal entries (self-loops). When ``False``, the
        diagonal is removed from the mask.
    threshold : float or None
        Inclusive minimum value to retain. If ``None``, values must be
        strictly greater than 0 to be kept.

    Returns
    -------
    numpy.ndarray of bool
        Boolean mask with the same shape as ``arr`` indicating entries that
        pass the thresholding and self-loop policy.
    """
    mask: NDArray[np.bool_] = (arr > 0) if threshold is None else (arr >= threshold)
    if include_self_loops:
        return mask.astype(bool, copy=False)
    # Drop diagonal in-place for clarity
    mask = mask.astype(bool, copy=False)
    np.fill_diagonal(mask, val=False)
    return mask


def _apply_threshold(series: pd.Series, *, threshold: float | None) -> pd.Series:
    """
    Return boolean mask for threshold semantics.

    Encodes the default behavior of dropping zeros when no threshold is
    provided, otherwise applying an inclusive cutoff (>= threshold).

    Parameters
    ----------
    series : pandas.Series
        Series of numeric values to filter.
    threshold : float | None
        Minimum value to retain. When ``None``, values strictly greater than
        zero are kept.

    Returns
    -------
    pandas.Series
        Boolean mask indicating which values pass the threshold.
    """
    return series > 0 if threshold is None else series >= threshold


def _empty_edgeframe(*, include_extra_weights: bool, extra_weights: list[str]) -> pd.DataFrame:
    """
    Construct a canonical empty edge DataFrame with appropriate columns.

    This helper is used when upstream processing yields no edges (for example,
    after thresholding or when no valid zone pairs remain). It guarantees a
    consistent schema for downstream consumers by returning the expected
    columns even in the absence of data.

    Parameters
    ----------
    include_extra_weights : bool
        Whether to include additional weight columns beyond the canonical
        'weight'.
    extra_weights : list[str]
        Names of the extra weight columns to include when requested.

    Returns
    -------
    pandas.DataFrame
        Empty DataFrame with columns ['source','target','weight',*extra_weights].
    """
    cols = ["source", "target", "weight"]
    if include_extra_weights:
        # Preserve provided order
        cols.extend(c for c in extra_weights if c != "weight")
    return pd.DataFrame(columns=cols)


def _canonical_zone_ranks(zone_ids: pd.Index) -> NDArray[np.intp] | None:
    """
    Compute canonical lexicographic string ranks for zone identifiers.

    Matches the undirected endpoint ordering (string comparison with
    encounter-order tie-breaking for distinct IDs sharing a string label).
    Returns ``None`` when ``zone_ids`` is already in canonical order.

    Parameters
    ----------
    zone_ids : pandas.Index
        Unique zone identifiers.

    Returns
    -------
    numpy.ndarray of intp or None
        Rank array aligned with ``zone_ids``, or ``None`` if already ordered.
    """
    id_str = zone_ids.astype(str).to_numpy()
    order = np.argsort(id_str, kind="stable")
    if np.array_equal(order, np.arange(len(zone_ids))):
        return None
    ranks = np.empty(len(zone_ids), dtype=np.intp)
    ranks[order] = np.arange(len(zone_ids), dtype=np.intp)
    return ranks


def _undirected_adjacency_to_edgelist(
    arr: np.ndarray,
    ids: pd.Index,
    *,
    nans: int,
    include_self_loops: bool,
    threshold: float | None,
) -> tuple[pd.DataFrame, bool]:
    """
    Build an undirected edgelist directly from upper-triangle reciprocal sums.

    Sums ``arr[i, j] + arr[j, i]`` (and retains ``arr[i, i]`` on the diagonal
    when ``include_self_loops`` is enabled) in bounded row batches, applies the
    threshold in NumPy before constructing a DataFrame, and avoids full-matrix
    edgelist expansion and pandas groupby overhead.

    Parameters
    ----------
    arr : numpy.ndarray
        Square float adjacency matrix.
    ids : pandas.Index
        Zone identifiers aligned with matrix axes.
    nans : int
        Count of NaN entries detected across ``arr``.
    include_self_loops : bool
        Whether to evaluate diagonal entries.
    threshold : float | None
        Inclusive minimum summed flow retained; ``None`` keeps strictly
        positive totals.

    Returns
    -------
    tuple[pandas.DataFrame, bool]
        Undirected canonical edgelist and whether any negative entry was seen.
    """
    n = len(arr)
    ranks = _canonical_zone_ranks(ids)
    negative = False
    parts: list[pd.DataFrame] = []
    tril_k = -1 if include_self_loops else 0

    for batch in row_batches(n, n):
        r0 = int(batch.start)
        if not negative:
            negative = bool((arr[batch] < 0).any())

        upper = arr[batch, r0:]
        lower_t = arr[r0:, batch].T
        if nans:
            upper = np.nan_to_num(upper, nan=0.0)
            lower_t = np.nan_to_num(lower_t, nan=0.0)

        combined = upper + lower_t
        batch_rows = len(combined)
        if include_self_loops:
            diag = np.arange(batch_rows)
            combined[diag, diag] = upper[diag, diag]

        mask = (combined > 0) if threshold is None else (combined >= threshold)
        mask[np.tril_indices(batch_rows, k=tril_k)] = False

        i, j = np.where(mask)
        if len(i) == 0:
            continue

        gi = i + r0
        gj = j + r0
        if ranks is not None:
            swap = ranks[gi] > ranks[gj]
            src_idx = np.where(swap, gj, gi)
            tgt_idx = np.where(swap, gi, gj)
        else:
            src_idx = gi
            tgt_idx = gj

        parts.append(
            pd.DataFrame(
                {
                    "source": ids.take(src_idx),
                    "target": ids.take(tgt_idx),
                    "weight": combined[i, j],
                }
            )
        )

    if not parts:
        empty = pd.DataFrame({"source": ids[:0], "target": ids[:0], "weight": np.empty(0)})
        return empty, negative
    return pd.concat(parts, ignore_index=True), negative


def _adjacency_to_edgelist(
    adjacency: pd.DataFrame | np.ndarray,
    zone_ids: pd.Index | None = None,
    *,
    include_self_loops: bool = False,
    threshold: float | None = None,
    directed: bool = True,
) -> pd.DataFrame:
    """
    Convert an adjacency matrix into a canonical edgelist DataFrame.

    Produces a long-form table of edges from a square matrix by applying
    thresholding and (optionally) dropping diagonal entries. When
    ``directed=False``, sums reciprocal entries directly over the upper
    triangle in bounded batches before applying the threshold.

    Parameters
    ----------
    adjacency : DataFrame | ndarray
        Square matrix of flows. If DataFrame, index/columns are zone IDs and
        override ``zone_ids``. If ndarray, ``zone_ids`` must be provided and
        correspond to row/column order (validated upstream).
    zone_ids : pd.Index | None
        Zone identifiers in the order matching the matrix when ``adjacency``
        is an ndarray.
    include_self_loops : bool
        Whether to keep diagonal elements subject to thresholding.
    threshold : float | None
        Minimum flow to keep. None keeps only positive totals.
    directed : bool, default True
        Whether to preserve directed edges or sum reciprocal pairs.

    Returns
    -------
    DataFrame
        Edge list with columns ['source', 'target', 'weight'].
    """
    arr, ids = _extract_array_and_ids(adjacency, zone_ids)

    nans = sum(int(np.isnan(arr[batch]).sum()) for batch in row_batches(len(arr), len(arr)))
    if nans:
        warnings.warn(
            f"Adjacency contains {nans} NaN values; treating as 0 (requirement 2.3)",
            UserWarning,
            stacklevel=2,
        )

    if not directed:
        edgelist, negative = _undirected_adjacency_to_edgelist(
            arr,
            ids,
            nans=nans,
            include_self_loops=include_self_loops,
            threshold=threshold,
        )
        if negative:
            warnings.warn(
                "Adjacency contains negative weights; keeping values (requirement 5.6)",
                UserWarning,
                stacklevel=2,
            )
        return edgelist

    negative = False
    parts = []
    for batch in row_batches(len(arr), len(arr)):
        block = np.nan_to_num(arr[batch], nan=0.0) if nans else arr[batch]
        if not negative:
            negative = bool((block < 0).any())
        mask = _build_adjacency_mask(block, include_self_loops=True, threshold=threshold)
        if not include_self_loops:
            rows = np.arange(len(block))
            mask[rows, rows + batch.start] = False
        i, j = np.where(mask)
        if len(i) == 0:
            continue
        parts.append(
            pd.DataFrame(
                {"source": ids.take(i + batch.start), "target": ids.take(j), "weight": block[i, j]}
            )
        )
    if negative:
        warnings.warn(
            "Adjacency contains negative weights; keeping values (requirement 5.6)",
            UserWarning,
            stacklevel=2,
        )
    if not parts:
        return pd.DataFrame({"source": ids[:0], "target": ids[:0], "weight": np.empty(0)})
    return pd.concat(parts, ignore_index=True)


def _coerce_weight_columns(
    df: pd.DataFrame,
    weight_cols: list[str],
) -> pd.DataFrame:
    """
    Coerce weight columns to numeric with NaN->0 and warnings.

    Converts specified columns to numeric dtype, treating pre-existing NaNs
    as zeros and warning when negative values are encountered.

    Parameters
    ----------
    df : pandas.DataFrame
        Input edge list DataFrame.
    weight_cols : list[str]
        Names of columns to coerce to numeric.

    Returns
    -------
    pandas.DataFrame
        DataFrame with specified columns coerced to numeric (NaNs filled where
        originally present) and warnings emitted for negatives.
    """
    coerced = df.copy()
    for col in weight_cols:
        original = coerced[col]
        original_nans = int(pd.isna(original).sum())
        s = pd.to_numeric(original, errors="coerce")
        n_total = len(s)
        after_nans = int(pd.isna(s).sum())

        if n_total > 0 and after_nans == n_total:
            msg = f"Column '{col}' could not be coerced to numeric (all values are non-numeric)"
            raise ValueError(msg)

        # Treat pre-existing NaNs as 0 with a warning (Req 2.3),
        # but if coercion introduced new NaNs (after_nans > original_nans) -> error (Req 5.10/5.11)
        if after_nans > original_nans:
            msg = f"Column '{col}' contains non-numeric values that cannot be coerced to numeric"
            raise ValueError(msg)
        if original_nans:
            warnings.warn(
                f"Column '{col}' has {original_nans} NaN values; treating as 0 (requirement 2.3)",
                UserWarning,
                stacklevel=2,
            )
            # Only fill positions that were originally NaN
            s = s.mask(pd.isna(original), 0)

        if (s < 0).any():
            warnings.warn(
                f"Column '{col}' contains negative weights; keeping values (requirement 5.6)",
                UserWarning,
                stacklevel=2,
            )
        coerced[col] = s
    return coerced


def _aggregate_edgelist(
    df: pd.DataFrame,
    *,
    source_col: str,
    target_col: str,
    weight_cols: list[str],
) -> pd.DataFrame:
    """
    Aggregate duplicate (source, target) pairs by summing weight columns.

    Groups edges by origin and destination, summing all provided weight
    columns to remove duplicates in a single pass.

    Parameters
    ----------
    df : pandas.DataFrame
        Edge list DataFrame.
    source_col, target_col : str
        Column names for origin/destination identifiers.
    weight_cols : list[str]
        Weight columns to sum during aggregation.

    Returns
    -------
    pandas.DataFrame
        Aggregated edge list.
    """
    return df.groupby([source_col, target_col], as_index=False)[weight_cols].sum()


def _resolve_primary(weight_cols: list[str], threshold_col: str | None) -> str:
    """
    Return the primary weight column used for thresholding and canonical weight.

    When multiple weight columns exist, this selects the one used for
    thresholding and populating the canonical 'weight' field.

    Parameters
    ----------
    weight_cols : list[str]
        Candidate weight columns.
    threshold_col : str | None
        Preferred column to use if multiple weights are present.

    Returns
    -------
    str
        The selected primary weight column name.
    """
    # Upstream validator ensures either single weight or a valid threshold_col
    if len(weight_cols) == 1:
        return weight_cols[0]
    return cast("str", threshold_col)


def _normalize_edgelist(
    edgelist_df: pd.DataFrame,
    *,
    source_col: str,
    target_col: str,
    weight_cols: list[str],
    threshold: float | None,
    directed: bool,
    threshold_col: str | None,
    include_self_loops: bool,
) -> pd.DataFrame:
    """
    Normalize an edgelist with optional multi-attribute weights.

    Coerces weights to numeric, removes self-loops when excluded, merges
    duplicate (and, when ``directed=False``, reciprocal) pairs in a single
    aggregation pass, applies thresholding, and returns canonical columns
    with a primary 'weight' field.

    Parameters
    ----------
    edgelist_df : pandas.DataFrame
        Input edge list DataFrame containing at least the source and target
        columns and the specified weight columns.
    source_col : str
        Column name for origin identifiers in ``edgelist_df``.
    target_col : str
        Column name for destination identifiers in ``edgelist_df``.
    weight_cols : list[str]
        Names of numeric weight (flow) columns to process and preserve.
    threshold : float | None
        Inclusive threshold applied to the primary weight column; when
        ``None``, only positive totals are retained.
    directed : bool
        Whether to keep directed edges or merge reciprocal pairs before
        thresholding.
    threshold_col : str | None
        Name of the primary weight column used for thresholding when multiple
        weights are provided; if a single weight column exists, this may be
        ``None``.
    include_self_loops : bool
        Whether to retain edges where source equals target; defaults to False.

    Returns
    -------
    pandas.DataFrame
        Canonical edge list with columns ['source', 'target', 'weight', ...],
        where 'weight' mirrors the chosen primary weight column.
    """
    coerced = _coerce_weight_columns(edgelist_df, weight_cols)

    # Remove self-loops before aggregation unless explicitly included
    if not include_self_loops:
        coerced = coerced.loc[coerced[source_col] != coerced[target_col]]

    # Select primary column for thresholding and canonical weight
    primary = _resolve_primary(weight_cols, threshold_col)

    # Zero-only rows cannot change sums or survive a strictly positive cutoff
    if threshold is None or threshold > 0:
        nonzero = (
            coerced[primary] != 0
            if len(weight_cols) == 1
            else (coerced[weight_cols] != 0).any(axis=1)
        )
        coerced = coerced.loc[nonzero]

    if coerced.empty:
        agg_df = coerced
    else:
        if not directed:
            coerced = _canonicalize_undirected_endpoints(coerced, source_col, target_col)
        agg_df = _aggregate_edgelist(
            coerced, source_col=source_col, target_col=target_col, weight_cols=weight_cols
        )
        agg_df = agg_df.loc[_apply_threshold(agg_df[primary], threshold=threshold)]

    # Build canonical columns and rename source/target for consistency
    out = agg_df.rename(columns={source_col: "source", target_col: "target"})
    out["weight"] = out[primary]  # canonical weight mirrors primary

    # Ensure canonical ordering: source, target, weight, then other weight columns
    # Keep only specified weight columns (MVP drops other non-weight attributes per 6.6)
    remaining_weights = [c for c in weight_cols if c != "weight"]

    # 'weight' already mirrors primary; still preserve all requested columns
    cols = ["source", "target", "weight", *remaining_weights]
    return out.loc[:, cols]


def _validate_adjacency_data(
    od_data: pd.DataFrame | np.ndarray,
    *,
    zones_gdf: gpd.GeoDataFrame,
    **kwargs: object,
) -> None:
    """
    Validate adjacency style OD data (square, labels/order).

    Checks shape, labeling, and compatibility with the provided zones when
    using ndarray inputs.

    Parameters
    ----------
    od_data : pandas.DataFrame | numpy.ndarray
        Adjacency data.
    zones_gdf : geopandas.GeoDataFrame
        Zones for validating shape/ordering when ndarray is used.

    Returns
    -------
    None
        This function validates input and raises on failure.

    Other Parameters
    ----------------
    **kwargs : object
        Additional keyword arguments ignored by this validator.
    """
    _ = kwargs  # ignore extra keyword arguments intentionally
    if isinstance(od_data, pd.DataFrame):
        checks = [
            (od_data.shape[0] == od_data.shape[1], "Adjacency DataFrame must be square"),
            (
                od_data.index.equals(od_data.columns),
                "Adjacency DataFrame index and columns must match exactly",
            ),
            (
                od_data.index.is_unique and od_data.columns.is_unique,
                "Adjacency DataFrame index and columns must be unique",
            ),
        ]
        for ok, msg in checks:
            if not ok:
                raise ValueError(msg)

    elif isinstance(od_data, np.ndarray):
        if not (od_data.ndim == 2 and od_data.shape[0] == od_data.shape[1]):
            msg = "Adjacency ndarray must be 2D square"
            raise ValueError(msg)

        if od_data.shape[0] != len(zones_gdf):
            msg = "Adjacency ndarray size must match number of zones in zones_gdf"
            raise ValueError(msg)

        warnings.warn(
            "Assuming ndarray row/column ordering matches zones_gdf order (requirement 5.7)",
            UserWarning,
            stacklevel=2,
        )

    else:
        msg = "For matrix_type='adjacency', od_data must be a pandas DataFrame or numpy ndarray"
        raise TypeError(msg)


def _validate_edgelist_data(
    od_data: pd.DataFrame,
    *,
    source_col: str,
    target_col: str,
    weight_cols: list[str] | None,
    **kwargs: object,
) -> None:
    """
    Validate edgelist structural requirements (required cols, weights present).

    Ensures the presence of required columns and weight fields prior to
    normalization and thresholding.

    Parameters
    ----------
    od_data : pandas.DataFrame
        Edge list data.
    source_col, target_col : str
        Column names for origin/destination identifiers.
    weight_cols : list[str] | None
        Names of weight columns that must exist.

    Returns
    -------
    None
        This function validates input and raises on failure.

    Other Parameters
    ----------------
    **kwargs : object
        Additional keyword arguments ignored by this validator.
    """
    _ = kwargs  # ignore extra keyword arguments intentionally
    if not isinstance(od_data, pd.DataFrame):
        msg = "For matrix_type='edgelist', od_data must be a pandas DataFrame"
        raise TypeError(msg)

    missing_basic = [c for c in (source_col, target_col) if c not in od_data.columns]
    if missing_basic:
        msg = f"Edgelist DataFrame missing required columns: {', '.join(missing_basic)}"
        raise ValueError(msg)

    if not weight_cols:
        msg = "weight_cols must be provided (at least one weight column) for edgelist input"
        raise ValueError(msg)

    missing_w = [c for c in weight_cols if c not in od_data.columns]
    if missing_w:
        msg = f"weight_cols contain columns not present in edgelist: {', '.join(missing_w)}"
        raise ValueError(msg)


def _validate_weights_threshold(
    _od_edgelist: pd.DataFrame,
    *,
    weight_cols: list[str],
    _threshold: float | None,
    threshold_col: str | None,
) -> None:
    """
    Validate weight columns and threshold semantics (subset of full spec).

    Confirms that a valid primary column is designated when multiple weights
    are supplied and that provided names are consistent.

    Parameters
    ----------
    _od_edgelist : pandas.DataFrame
        Edge list data (unused here; included for signature symmetry).
    weight_cols : list[str] | None
        Weight columns provided by the caller.
    _threshold : float | None
        Threshold value.
    threshold_col : str | None
        Name of the column used for thresholding when multiple weights exist.

    Returns
    -------
    None
        This function validates input and raises on failure.
    """
    # Validate threshold_col selection and unify coercion using shared helper
    if len(weight_cols) > 1 and (threshold_col is None or threshold_col not in weight_cols):
        msg = "When multiple weight_cols are provided a valid threshold_col must be specified"
        raise ValueError(msg)
    if len(weight_cols) == 1 and (threshold_col is not None and threshold_col not in weight_cols):
        msg = "threshold_col not in weight_cols"
        raise ValueError(msg)

    if "weight" in weight_cols and _resolve_primary(weight_cols, threshold_col) != "weight":
        msg = (
            "When weight_cols contains 'weight', threshold_col must be 'weight'; "
            "rename the input weight column to use a different primary weight"
        )
        raise ValueError(msg)

    # Do not coerce here; coercion happens once in _normalize_edgelist


# ---------------------------------------------------------------------------
# Zone alignment and mapping helpers
# ---------------------------------------------------------------------------
def _align_adjacency_zones(
    adjacency_df: pd.DataFrame,
    *,
    zones_gdf: gpd.GeoDataFrame,
    zone_id_col: str | None,
) -> pd.DataFrame:
    """
    Align an adjacency DataFrame to the provided zones.

    Reindexes the matrix to the intersection of zone identifiers and warns
    about labels that are missing on either side.

    Parameters
    ----------
    adjacency_df : pandas.DataFrame
        Square adjacency matrix with zone ID index/columns.
    zones_gdf : geopandas.GeoDataFrame
        Zones GeoDataFrame.
    zone_id_col : str
        Column name holding unique zone identifiers in ``zones_gdf``.

    Returns
    -------
    pandas.DataFrame
        Sub-matrix aligned to overlapping zone IDs in both inputs.
    """
    matrix_ids = pd.Index(adjacency_df.index)

    zones_ids = (
        pd.Index(zones_gdf[zone_id_col]) if zone_id_col is not None else pd.Index(zones_gdf.index)
    )
    common = matrix_ids.intersection(zones_ids)
    if common.empty:
        msg = "No overlapping zone IDs between adjacency matrix and zones_gdf"
        logger.error(msg)
        raise ValueError(msg)

    missing_in_matrix = zones_ids.difference(matrix_ids)
    missing_in_zones = matrix_ids.difference(zones_ids)
    if len(missing_in_matrix) > 0:
        warnings.warn(
            f"{len(missing_in_matrix)} zone IDs in zones_gdf not present in adjacency; they will be isolated nodes",
            UserWarning,
            stacklevel=2,
        )
    if len(missing_in_zones) > 0:
        warnings.warn(
            f"{len(missing_in_zones)} labels in adjacency not present in zones_gdf; related edges will be dropped",
            UserWarning,
            stacklevel=2,
        )

    # Reindex to common ids (preserve the order from adjacency_df to keep consistency with provided matrix)
    return adjacency_df.reindex(index=common, columns=common)


def _align_numpy_array_zones(
    adjacency_array: np.ndarray,
    *,
    zones_gdf: gpd.GeoDataFrame,
    zone_id_col: str | None,
) -> pd.Index:
    """
    Validate ndarray size and return zone ids in ``zones_gdf`` order.

    Ensures the array size matches the number of zones and yields the
    identifier sequence in the same order as the zones GeoDataFrame.

    Parameters
    ----------
    adjacency_array : numpy.ndarray
        Square adjacency array.
    zones_gdf : geopandas.GeoDataFrame
        Zones GeoDataFrame.
    zone_id_col : str
        Column name holding unique zone identifiers in ``zones_gdf``.

    Returns
    -------
    pandas.Index
        Zone identifiers in the same order as ``zones_gdf``.
    """
    # Upstream validation already checks shape and size; here we just warn about assumed ordering
    warnings.warn(
        (
            "Assuming ndarray row/column ordering matches zones_gdf order (requirement 5.7); "
            f"ndarray rows={adjacency_array.shape[0]}"
        ),
        UserWarning,
        stacklevel=2,
    )

    return cast(
        "pd.Index", pd.Index(zones_gdf[zone_id_col] if zone_id_col is not None else zones_gdf.index)
    )


def _align_edgelist_zones(
    edgelist_df: pd.DataFrame,
    *,
    zones_gdf: gpd.GeoDataFrame,
    zone_id_col: str | None,
    source_col: str,
    target_col: str,
) -> pd.DataFrame:
    """
    Filter edgelist to valid zones (no aggregation here).

    Drops edges referencing unknown zone identifiers while preserving
    duplicates for later aggregation.

    Parameters
    ----------
    edgelist_df : pandas.DataFrame
        Input edgelist containing at least ``source_col`` and ``target_col``.
    zones_gdf : geopandas.GeoDataFrame
        Zones with ``zone_id_col`` present.
    zone_id_col : str
        Column name holding unique zone identifiers.
    source_col, target_col : str
        Column names for origin/destination identifiers in ``edgelist_df``.

    Returns
    -------
    pandas.DataFrame
        Filtered edgelist containing only valid zone IDs. Duplicate
        ``(source, target)`` rows are preserved here and handled downstream.
    """
    valid_ids = set(
        (zones_gdf[zone_id_col] if zone_id_col is not None else zones_gdf.index).tolist()
    )
    before = len(edgelist_df)
    mask_valid = edgelist_df[source_col].isin(valid_ids) & edgelist_df[target_col].isin(valid_ids)
    filtered = edgelist_df.loc[mask_valid].copy()
    dropped = before - len(filtered)
    if dropped > 0:
        warnings.warn(
            f"Dropped {dropped} edges referencing unknown zone IDs (requirement 3.6)",
            UserWarning,
            stacklevel=2,
        )

    if filtered.empty:
        msg = "No overlapping zone IDs between edgelist and zones_gdf"
        logger.error(msg)
        raise ValueError(msg)

    return filtered


# ---------------------------------------------------------------------------
# Spatial geometry helpers (centroids and edge geometries)
# ---------------------------------------------------------------------------
def _create_edge_geometries(
    edges_df: pd.DataFrame,
    zones_gdf: gpd.GeoDataFrame,
    *,
    zone_id_col: str | None,
    source_col: str = "source",
    target_col: str = "target",
    compute_edge_geometry: bool = True,
) -> gpd.GeoDataFrame:
    """
    Create LineString geometries connecting zone centroids for each edge.

    For each row in ``edges_df`` the function looks up the centroid of the
    origin and destination zones and constructs a ``shapely.geometry.LineString``
    between them. If ``compute_edge_geometry`` is ``False`` or the input is
    empty, a GeoDataFrame is returned with a ``geometry`` column containing
    ``None`` for each row. Edges with unknown zone IDs or missing centroids are
    dropped with a warning.

    Parameters
    ----------
    edges_df : pandas.DataFrame
        Canonical edgelist containing at least ``source_col`` and
        ``target_col`` columns, and optionally weight attributes.
    zones_gdf : geopandas.GeoDataFrame
        GeoDataFrame of zone geometries used to compute centroids.
    zone_id_col : str or None
        Name of the identifier column in ``zones_gdf``. If ``None``, the index
        of ``zones_gdf`` is used as the identifier space.
    source_col : str, default 'source'
        Column name in ``edges_df`` holding origin zone identifiers.
    target_col : str, default 'target'
        Column name in ``edges_df`` holding destination zone identifiers.
    compute_edge_geometry : bool, default True
        Whether to compute LineString geometries. When ``False``, no geometry
        is computed and ``None`` is stored in the geometry column.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with the same non-geometry columns as ``edges_df`` and a
        ``geometry`` column containing LineStrings connecting origin and
        destination centroids. The CRS is inherited from ``zones_gdf``.
    """
    e = edges_df.copy()
    if not compute_edge_geometry or e.empty:
        geom = gpd.GeoSeries([None] * len(e), crs=zones_gdf.crs)
        return gpd.GeoDataFrame(e, geometry=geom, crs=zones_gdf.crs)

    # Filter out missing or empty zone centroids once per zone
    centroids = GeoDataProcessor.compute_centroids(zones_gdf)
    zone_ids = (
        pd.Index(zones_gdf[zone_id_col]) if zone_id_col is not None else pd.Index(zones_gdf.index)
    )
    centroid_arr = centroids.to_numpy()
    valid_centroids = ~is_missing(centroid_arr) & ~is_empty(centroid_arr)
    valid_ids = zone_ids[valid_centroids]

    src_idx = valid_ids.get_indexer(e[source_col])
    tgt_idx = valid_ids.get_indexer(e[target_col])

    missing_any = (src_idx < 0) | (tgt_idx < 0)
    n_missing = int(missing_any.sum())
    if n_missing:
        warnings.warn(
            f"Dropping {n_missing} edges with unknown zone IDs or missing centroid(s) (requirement 3.6)",
            UserWarning,
            stacklevel=2,
        )
        keep = ~missing_any
        e = e.loc[keep].copy()
        src_idx = src_idx[keep]
        tgt_idx = tgt_idx[keep]

    if e.empty:
        return gpd.GeoDataFrame(e, geometry=gpd.GeoSeries([], crs=zones_gdf.crs), crs=zones_gdf.crs)

    valid_coords = get_coordinates(centroid_arr[valid_centroids])
    line_coords = np.empty((len(src_idx), 2, 2), dtype=np.float64)
    line_coords[:, 0, :] = valid_coords[src_idx]
    line_coords[:, 1, :] = valid_coords[tgt_idx]
    geom = gpd.GeoSeries(linestrings(line_coords), crs=zones_gdf.crs)
    return gpd.GeoDataFrame(e.reset_index(drop=True), geometry=geom, crs=zones_gdf.crs)


def _can_have_string_label_collision(src: pd.Series, tgt: pd.Series) -> bool:
    """
    Check whether distinct endpoint values can share the same string form.

    Homogeneous numeric or explicit string dtypes have injective string
    representations for non-null values, so mixed-type tie-breaking is not
    needed.

    Parameters
    ----------
    src : pandas.Series
        Origin identifier column.
    tgt : pandas.Series
        Destination identifier column.

    Returns
    -------
    bool
        ``True`` if two unequal values may stringify identically.
    """
    if src.dtype != tgt.dtype:
        return True
    dtype = src.dtype
    return not (
        pd.api.types.is_integer_dtype(dtype)
        or pd.api.types.is_float_dtype(dtype)
        or isinstance(dtype, pd.StringDtype)
    )


def _canonicalize_undirected_endpoints(
    edges: pd.DataFrame,
    source_col: str,
    target_col: str,
) -> pd.DataFrame:
    """
    Order each row's endpoints canonically for undirected aggregation.

    Orders endpoints by string representation so that reciprocal flows map to
    the same ``(source_col, target_col)`` pair. Distinct IDs with identical
    string representations use a shared encounter-order tie-breaker.

    Parameters
    ----------
    edges : pandas.DataFrame
        Edge list containing ``source_col`` and ``target_col``.
    source_col : str
        Origin column name.
    target_col : str
        Destination column name.

    Returns
    -------
    pandas.DataFrame
        Shallow copy with canonically ordered ``source_col`` and ``target_col``.
    """
    src = edges[source_col]
    tgt = edges[target_col]
    s_str = src.astype(str)
    t_str = tgt.astype(str)

    keep = s_str <= t_str
    if _can_have_string_label_collision(src, tgt):
        same_label = s_str.eq(t_str) & src.ne(tgt)
        if same_label.any():
            source_dup = src.loc[same_label]
            target_dup = tgt.loc[same_label]
            ranks, _ = pd.factorize(pd.concat([source_dup, target_dup], ignore_index=True))
            keep.loc[same_label] = ranks[: len(source_dup)] < ranks[len(source_dup) :]

    src_norm = src.where(keep, tgt)
    tgt_norm = tgt.where(keep, src)
    return edges.assign(**{source_col: src_norm, target_col: tgt_norm})
