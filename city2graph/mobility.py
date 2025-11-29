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
from shapely.geometry import LineString

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
    as_nx: bool = False,
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
        ``threshold_col`` must be designated in the full implementation.
    threshold : float, optional
        Minimum flow retained (>=) applied to ``threshold_col`` (future logic).
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
    as_nx : bool, default False
        If True, final output will be an NetworkX graph (``nx.DiGraph`` when
        ``directed=True``; otherwise ``nx.Graph``).

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
    # --- Validation (Task 2) ------------------------------------------------
    _validate_zones_gdf(zones_gdf, zone_id_col)
    _validate_crs(zones_gdf)

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

    # --- Conversion to canonical edgelist ----------------------------------
    # For undirected mode, postpone thresholding until after symmetrization
    post_sum_threshold = threshold if not directed else None

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
            # In undirected mode, thresholding is applied later
            threshold=None if not directed else threshold,
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
            # In undirected mode, thresholding is applied later
            threshold=None if not directed else threshold,
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
            # In undirected mode, thresholding is applied later
            threshold=None if not directed else threshold,
        )

    # Ensure canonical columns exist even if empty result
    if edge_df.empty:
        edge_df = _empty_edgeframe(
            include_extra_weights=(matrix_type == "edgelist" and bool(weight_cols)),
            extra_weights=(weight_cols or []),
        )

    # If undirected, symmetrize by merging reciprocal edges and summing weights
    if not directed and not edge_df.empty:
        # Sum canonical 'weight' and any provided additional weight columns
        sum_cols = ["weight"]
        if matrix_type == "edgelist" and weight_cols:
            # Ensure we sum the explicitly requested weight columns as well
            # (they are already present in edge_df)
            for c in weight_cols:
                if c not in sum_cols:
                    sum_cols.append(c)

        edge_df = _symmetrize_edges(edge_df, sum_cols=sum_cols)

        # Apply threshold after summation when requested
        if post_sum_threshold is not None:
            edge_df = edge_df.loc[_apply_threshold(edge_df["weight"], threshold=post_sum_threshold)]

    # --- Spatial assembly (Task 5) -----------------------------------------
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
def _warn_and_clean_adjacency(arr: np.ndarray) -> np.ndarray:
    """
    Replace NaNs with 0 and warn on data quality issues (NaNs, negatives).

    Sanitizes adjacency arrays by converting missing values to zeros while
    preserving negative values, emitting warnings to flag potential issues.

    Parameters
    ----------
    arr : numpy.ndarray
        Adjacency matrix values to sanitize.

    Returns
    -------
    numpy.ndarray
        Array with NaNs replaced by 0.0. Negative values are preserved.
    """
    n_nans = int(np.isnan(arr).sum())
    if n_nans:
        warnings.warn(
            f"Adjacency contains {n_nans} NaN values; treating as 0 (requirement 2.3)",
            UserWarning,
            stacklevel=2,
        )
        arr = np.nan_to_num(arr, nan=0.0)

    if (arr < 0).any():
        warnings.warn(
            "Adjacency contains negative weights; keeping values (requirement 5.6)",
            UserWarning,
            stacklevel=2,
        )
    return arr


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
        cols.extend(extra_weights)
    return pd.DataFrame(columns=cols)


def _adjacency_to_edgelist(
    adjacency: pd.DataFrame | np.ndarray,
    zone_ids: pd.Index | None = None,
    *,
    include_self_loops: bool = False,
    threshold: float | None = None,
) -> pd.DataFrame:
    """
    Convert an adjacency matrix into a canonical edgelist DataFrame.

    Produces a long-form table of directed edges from a square matrix by
    applying thresholding and (optionally) dropping diagonal entries.

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
        Minimum flow to keep. None => drop zeros by default.

    Returns
    -------
    DataFrame
        Edge list with columns ['source', 'target', 'weight'].
    """
    arr, ids = _extract_array_and_ids(adjacency, zone_ids)

    # Clean and build mask
    arr = _warn_and_clean_adjacency(arr)
    mask = _build_adjacency_mask(arr, include_self_loops=include_self_loops, threshold=threshold)

    # Vectorized extraction of (i,j,weight)
    i, j = np.where(mask)

    return pd.DataFrame(
        {
            "source": ids.take(i),
            "target": ids.take(j),
            "weight": arr[i, j],
        }
    )


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
    threshold_col: str | None,
    include_self_loops: bool,
) -> pd.DataFrame:
    """
    Normalize an edgelist with optional multi-attribute weights.

    Coerces weights to numeric, aggregates duplicates, applies thresholding,
    and returns canonical columns with a primary 'weight' field.

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
        ``None``, zeros are dropped by default.
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
    # Coerce and aggregate in focused helpers
    coerced = _coerce_weight_columns(edgelist_df, weight_cols)
    agg_df = _aggregate_edgelist(
        coerced, source_col=source_col, target_col=target_col, weight_cols=weight_cols
    )

    # Remove self-loops unless explicitly included
    if not include_self_loops:
        agg_df = agg_df.loc[agg_df[source_col] != agg_df[target_col]]

    # Select primary column for thresholding and canonical weight
    primary = _resolve_primary(weight_cols, threshold_col)

    # Apply threshold
    mask = _apply_threshold(agg_df[primary], threshold=threshold)
    filtered = agg_df.loc[mask].copy()

    # Build canonical columns and rename source/target for consistency
    out = filtered.rename(columns={source_col: "source", target_col: "target"})
    out.insert(2, "weight", out[primary])  # canonical weight mirrors primary

    # Ensure canonical ordering: source, target, weight, then other weight columns
    # Keep only specified weight columns (MVP drops other non-weight attributes per 6.6)
    remaining_weights = [c for c in weight_cols]

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
def _validate_crs(zones_gdf: gpd.GeoDataFrame) -> None:
    """
    Warn about CRS issues and ensure CRS info is preserved.

    Emits warnings for missing CRS and geographic CRS to signal potential
    issues with distance-based computations.

    Parameters
    ----------
    zones_gdf : geopandas.GeoDataFrame
        Zones GeoDataFrame whose CRS will be validated.

    Returns
    -------
    None
        This function validates input and raises on failure.
    """
    crs = zones_gdf.crs
    if crs is None:
        warnings.warn(
            "zones_gdf has no CRS set; outputs will have undefined CRS (requirement 3.1)",
            UserWarning,
            stacklevel=2,
        )
        return

    # pyproj.CRS exposes is_geographic when available
    if getattr(crs, "is_geographic", False):
        warnings.warn(
            "Geographic CRS detected; distance/length measures may be inaccurate (requirement 3.5)",
            UserWarning,
            stacklevel=2,
        )


def _compute_centroids(zones_gdf: gpd.GeoDataFrame) -> gpd.GeoSeries:
    """
    Compute centroids for zones preserving CRS.

    Calculates feature centroids and explicitly propagates the CRS to the
    resulting GeoSeries for consistency.

    Parameters
    ----------
    zones_gdf : geopandas.GeoDataFrame
        Zones GeoDataFrame whose feature centroids will be computed.

    Returns
    -------
    geopandas.GeoSeries
        GeoSeries indexed like ``zones_gdf`` with the same CRS.
    """
    # GeoPandas returns a GeoSeries with the same CRS as input geometry
    centroids = zones_gdf.geometry.centroid
    # Explicitly carry CRS (some versions keep it automatically, we enforce)
    centroids.set_crs(zones_gdf.crs, allow_override=True, inplace=True)
    return centroids


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

    # Centroid lookup by ID
    centroids = _compute_centroids(zones_gdf)
    if zone_id_col is not None:
        centroids.index = pd.Index(zones_gdf[zone_id_col])

    # Map ids -> centroids via dict for clarity
    lookup = centroids.to_dict()
    src_pts = e[source_col].map(lookup)
    tgt_pts = e[target_col].map(lookup)

    missing_any = src_pts.isna() | tgt_pts.isna()
    n_missing = int(missing_any.sum())
    if n_missing:
        warnings.warn(
            f"Dropping {n_missing} edges with unknown zone IDs or missing centroid(s) (requirement 3.6)",
            UserWarning,
            stacklevel=2,
        )
        e = e.loc[~missing_any].copy()
        src_pts = src_pts.loc[~missing_any]
        tgt_pts = tgt_pts.loc[~missing_any]

    if e.empty:
        return gpd.GeoDataFrame(e, geometry=gpd.GeoSeries([], crs=zones_gdf.crs), crs=zones_gdf.crs)

    lines = [
        LineString([a, b]) for a, b in zip(src_pts.to_numpy(), tgt_pts.to_numpy(), strict=True)
    ]
    geom = gpd.GeoSeries(lines, crs=zones_gdf.crs)
    return gpd.GeoDataFrame(e.reset_index(drop=True), geometry=geom, crs=zones_gdf.crs)


def _symmetrize_edges(edges: pd.DataFrame, *, sum_cols: list[str]) -> pd.DataFrame:
    """
    Merge reciprocal directed edges into undirected edges by summing weights.

    For each unordered pair {u, v}, produce a single edge with ``source`` <= ``target``
    (based on string representation ordering) and sum the provided ``sum_cols`` across
    both directions. Self-loops (u == v) are preserved and included as-is.

    Parameters
    ----------
    edges : pandas.DataFrame
        Canonical directed edgelist containing 'source', 'target', and sum_cols.
    sum_cols : list[str]
        Columns to sum when merging reciprocal edges. Must include 'weight'.

    Returns
    -------
    pandas.DataFrame
        Undirected edgelist with merged weights and canonical columns.
    """
    edges_work = edges.copy()

    # Normalize (u, v) ordering deterministically; keep self-loops unchanged
    # Use astype(str) to ensure consistent comparison across dtypes
    s_str = edges_work["source"].astype(str)
    t_str = edges_work["target"].astype(str)

    # mask where source string is lexicographically <= target string
    keep = s_str <= t_str
    src_norm = edges_work["source"].where(keep, edges_work["target"])  # swap when needed
    tgt_norm = edges_work["target"].where(keep, edges_work["source"])  # swap when needed
    edges_work["_u"] = src_norm
    edges_work["_v"] = tgt_norm

    # Group by normalized unordered pair and sum all requested columns
    group_cols = ["_u", "_v"]
    agg = edges_work.groupby(group_cols, as_index=False)[sum_cols].sum()

    # Rename normalized cols back to canonical names
    agg = agg.rename(columns={"_u": "source", "_v": "target"})

    # Ensure column order: source, target, weight, then any others in sum_cols order
    other_cols = [c for c in sum_cols if c != "weight"]
    ordered_cols = ["source", "target", "weight", *other_cols]

    # Preserve any remaining non-summed columns (if present) by left-joining original
    # But to keep the function focused and deterministic, we return only canonical columns
    return agg.loc[:, ordered_cols]
