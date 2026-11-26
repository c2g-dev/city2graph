"""Spatial tessellation utilities."""

from __future__ import annotations

import logging
import typing
from functools import partial
from typing import TYPE_CHECKING
from typing import Any

import geopandas as gpd
import momepy
import numpy as np
import pandas as pd
import shapely

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry

__all__ = ["create_tessellation"]

logger = logging.getLogger("city2graph.utils")

_COARSE_GRID_SIZE: float = 1e-3
_JITTER_MAGNITUDE: float = 0.01


def create_tessellation(
    geometry: gpd.GeoDataFrame | gpd.GeoSeries,
    primary_barriers: gpd.GeoDataFrame | gpd.GeoSeries | None = None,
    shrink: float = 0.4,
    segment: float = 0.5,
    threshold: float = 0.05,
    n_jobs: int = -1,
    limit: gpd.GeoDataFrame | gpd.GeoSeries | BaseGeometry | None = None,
    **kwargs: object,
) -> gpd.GeoDataFrame:
    """
    Create tessellations from given geometries, with optional barriers.

    This function generates either morphological or enclosed tessellations based on
    the input geometries. If `primary_barriers` are provided, it creates an
    enclosed tessellation; otherwise, it generates a morphological tessellation.

    Parameters
    ----------
    geometry : geopandas.GeoDataFrame or geopandas.GeoSeries
        The geometries (typically building footprints) to tessellate around.
    primary_barriers : geopandas.GeoDataFrame or geopandas.GeoSeries, optional
        Geometries (typically road network) to use as barriers for enclosed
        tessellation. If provided, `momepy.enclosed_tessellation` is used.
        Default is None.
    shrink : float, default 0.4
        The distance to shrink the geometry for the skeleton endpoint generation.
        Passed to `momepy.morphological_tessellation` or `momepy.enclosed_tessellation`.
    segment : float, default 0.5
        The segment length for discretizing the geometry. Passed to
        `momepy.morphological_tessellation` or `momepy.enclosed_tessellation`.
    threshold : float, default 0.05
        The threshold for snapping skeleton endpoints to the boundary. Only used
        for enclosed tessellation.
    n_jobs : int, default -1
        The number of jobs to use for parallel processing. -1 means using all
        available processors. Only used for enclosed tessellation.
    limit : geopandas.GeoDataFrame, geopandas.GeoSeries, shapely geometry, or None, optional
        Boundary passed to `momepy.enclosures` for enclosed tessellation. When
        None, a buffered (100 m) union of input geometry and barriers is
        computed and the enclosures are clipped to it.
    **kwargs : object, optional
        Additional keyword arguments passed to the underlying `momepy`
        tessellation function.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing the tessellation cells as polygons. Each cell
        has a unique `tess_id`.

    Raises
    ------
    ValueError
        If `primary_barriers` are not provided and the geometry is in a
        geographic CRS (e.g., EPSG:4326), as morphological tessellation
        requires a projected CRS.

    See Also
    --------
    momepy.morphological_tessellation : Generate morphological tessellation.
    momepy.enclosed_tessellation : Generate enclosed tessellation.

    Examples
    --------
    >>> import geopandas as gpd
    >>> from shapely.geometry import Polygon
    >>> # Create some building footprints
    >>> buildings = gpd.GeoDataFrame(
    ...     geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
    ...               Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])],
    ...     crs="EPSG:32633"
    ... )
    >>> # Generate morphological tessellation
    >>> tessellation = create_tessellation(buildings)
    >>> print(tessellation.head())  # doctest: +NORMALIZE_WHITESPACE
                                                geometry  tess_id
    0     POLYGON ((2.6 0.4, 0.4 0.4, 0.4 2.6, 2.6 0.4))        0
    1  POLYGON ((0.4 2.6, 0.4 2.6, 2.6 2.6, 2.6 0.4, ...        1

    >>> # Generate enclosed tessellation with roads as barriers
    >>> from shapely.geometry import LineString
    >>> roads = gpd.GeoDataFrame(
    ...     geometry=[LineString([(0, -1), (3, -1)]), LineString([(1.5, -1), (1.5, 4)])],
    ...     crs="EPSG:32633"
    ... )
    >>> enclosed_tess = create_tessellation(buildings, primary_barriers=roads)
    >>> print(sorted(enclosed_tess.columns))
    ['enclosure_index', 'geometry', 'tess_id']
    """
    if geometry.empty:
        if primary_barriers is not None:
            return _create_empty_tessellation(geometry.crs)
        # Morphological tessellation has no enclosures.
        return gpd.GeoDataFrame(
            columns=["geometry", "tess_id"],
            geometry="geometry",
            crs=geometry.crs,
        )

    if primary_barriers is not None:
        return _create_enclosed_tessellation(
            geometry,
            primary_barriers,
            limit,
            shrink,
            segment,
            threshold,
            n_jobs,
            **kwargs,
        )

    return _create_morphological_tessellation(
        geometry,
        shrink,
        segment,
    )


def _polygonal_tessellation(tessellation: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Return only polygonal tessellation cells, salvaging GeometryCollections.

    Degenerate or overlapping footprints can make momepy emit GeometryCollection
    cells: when its own coverage simplification fails it warns and returns the
    raw dense partition, line work and all. Downstream consumers such as
    libpysal contiguity weights only accept (Multi)Polygon geometries, so
    collections are replaced by the union of their polygonal parts and any
    remaining non-polygonal or empty rows are dropped.

    Parameters
    ----------
    tessellation : geopandas.GeoDataFrame
        Raw tessellation returned by momepy.

    Returns
    -------
    geopandas.GeoDataFrame
        The polygonal subset of ``tessellation``.
    """
    collections = tessellation.geom_type == "GeometryCollection"
    if collections.any():
        logger.warning(
            "Tessellation produced %d GeometryCollection cell(s); keeping only "
            "their polygonal parts.",
            int(collections.sum()),
        )
        tessellation = tessellation.copy()
        tessellation.loc[collections, tessellation.geometry.name] = tessellation.loc[
            collections
        ].geometry.apply(
            lambda geom: shapely.unary_union(
                [part for part in geom.geoms if part.geom_type in ("Polygon", "MultiPolygon")],
            ),
        )

    keep = tessellation.geometry.notna() & ~tessellation.geometry.is_empty
    keep &= tessellation.geom_type.isin(["Polygon", "MultiPolygon"])
    if not keep.all():
        tessellation = tessellation.loc[keep].copy()
    return tessellation


def _degenerate_enclosures(
    tessellation: gpd.GeoDataFrame,
    enclosures: gpd.GeoDataFrame,
    tolerance: float = 1.05,
) -> list[Any]:
    """
    Return the indices of enclosures whose cells fail to partition them.

    A valid enclosed tessellation partitions its enclosure exactly, so the
    cells must neither overlap each other nor leave the enclosure uncovered.
    Both symptoms mean ``shapely.voronoi_polygons`` degenerated: overlapping
    cells contain the same building points and explode downstream spatial
    joins, while a cell that lost area no longer contains its own building and
    silently drops it. The three ratios catch distinct degeneracies: cells that
    each span (nearly) the whole enclosure overfill it, cells that overlap
    inside a partially covered enclosure exceed their own union, and cells
    collapsed to slivers or line work cover too little of the enclosure.

    Parameters
    ----------
    tessellation : geopandas.GeoDataFrame
        Raw ``momepy.enclosed_tessellation`` output with an ``enclosure_index``
        column referring to ``enclosures`` index values.
    enclosures : geopandas.GeoDataFrame
        The enclosures the tessellation was generated from.
    tolerance : float, default 1.05
        Allowed ratio between the summed cell area, the cells' union area and
        the enclosure area before an enclosure is reported as degenerate.

    Returns
    -------
    list[Any]
        The ``enclosure_index`` values of degenerate enclosures.
    """
    if tessellation.empty or "enclosure_index" not in tessellation.columns:
        return []
    grouped = tessellation.geometry.groupby(tessellation["enclosure_index"])
    cell_areas = grouped.apply(lambda cells: cells.area.sum())
    union_areas = grouped.apply(lambda cells: cells.union_all().area)
    enclosure_areas = enclosures.geometry.area.reindex(cell_areas.index)
    # NaN enclosure areas (unmatched indices) compare as False and are kept.
    broken = (
        (cell_areas > enclosure_areas * tolerance)
        | (cell_areas > union_areas * tolerance)
        | (union_areas * tolerance < enclosure_areas)
    )
    return list(cell_areas.index[broken])


def _jitter_hash_unit(coords: np.ndarray, salt: float) -> np.ndarray:
    """
    Map coordinate pairs to deterministic pseudo-random values in [0, 1).

    The value is a pure function of the coordinates and the salt, so repeated
    calls with the same input produce identical output. This keeps the
    geometry jitter reproducible across runs and consistent for vertices
    shared between geometries.

    Parameters
    ----------
    coords : numpy.ndarray
        Array of shape (n, 2) with x/y coordinates.
    salt : float
        Salt distinguishing independent channels (e.g. x- and y-offsets).

    Returns
    -------
    numpy.ndarray
        One value in [0, 1) per coordinate pair, identical for identical
        input pairs.
    """
    return np.abs(np.sin(coords[:, 0] * 12.9898 + coords[:, 1] * 78.233 + salt) * 43758.5453) % 1.0


def _shift_jittered_coordinates(coords: np.ndarray, magnitude: float) -> np.ndarray:
    """
    Offset an (n, 2) coordinate array by the deterministic jitter.

    Both offset channels derive from :func:`_jitter_hash_unit`, so the shift
    is reproducible and identical for identical coordinates.

    Parameters
    ----------
    coords : numpy.ndarray
        Array of shape (n, 2) with x/y coordinates.
    magnitude : float
        Maximum absolute offset per axis in map units.

    Returns
    -------
    numpy.ndarray
        The offset coordinates.
    """
    dx = (_jitter_hash_unit(coords, 0.0) - 0.5) * 2.0 * magnitude
    dy = (_jitter_hash_unit(coords, 1.0) - 0.5) * 2.0 * magnitude
    return coords + np.column_stack([dx, dy])


def _jitter_geometry(
    geometry: gpd.GeoDataFrame | gpd.GeoSeries,
    magnitude: float = _JITTER_MAGNITUDE,
) -> gpd.GeoDataFrame | gpd.GeoSeries:
    """
    Displace every vertex by a deterministic sub-centimetre offset.

    The offset is a pure function of the vertex coordinates, so identical
    vertices move identically: rings stay closed, shared party walls stay
    coincident, and repeated runs produce identical output. Used by the
    tessellation retry ladder to break the exact collinearity that makes
    ``shapely.voronoi_polygons`` degenerate on rectilinear footprints.

    Parameters
    ----------
    geometry : geopandas.GeoDataFrame or geopandas.GeoSeries
        Geometries to jitter.
    magnitude : float, default _JITTER_MAGNITUDE
        Maximum absolute offset per axis in map units.

    Returns
    -------
    geopandas.GeoDataFrame or geopandas.GeoSeries
        A copy with jittered vertex coordinates.
    """
    shift = partial(_shift_jittered_coordinates, magnitude=magnitude)
    jittered = geometry.copy()
    if isinstance(jittered, gpd.GeoDataFrame):
        jittered.geometry = shapely.transform(geometry.geometry, shift)
    else:
        jittered = gpd.GeoSeries(
            shapely.transform(geometry, shift),
            index=geometry.index,
            crs=geometry.crs,
        )
    return jittered


def _classify_tessellation_failure(error: Exception) -> str | None:
    """
    Classify a tessellation failure for the retry ladder.

    Known failures are the errors the retry ladder in
    :func:`_run_tessellation_with_retries` recovers from: the ``ValueError``
    momepy raises when an enclosure produces nothing to concatenate, the
    ``TypeError`` reporting an incorrect geometry type from the voronoi
    construction, and any ``GEOSException``. Anything else must propagate.

    Parameters
    ----------
    error : Exception
        The exception raised by ``momepy.enclosed_tessellation``.

    Returns
    -------
    str or None
        ``"empty"``, ``"geometry_type"`` or ``"geos"`` for known failures, or
        ``None`` for unknown errors that must propagate.
    """
    if isinstance(error, ValueError):
        return "empty" if "No objects to concatenate" in str(error) else None
    if isinstance(error, TypeError):
        return "geometry_type" if "incorrect geometry type" in str(error) else None
    return "geos" if isinstance(error, shapely.errors.GEOSException) else None


def _next_retry_overrides(
    kind: str,
    overrides: dict[str, object],
    kwargs: dict[str, object],
) -> dict[str, object] | None:
    """
    Return the next retry overrides for a classified tessellation failure.

    A coarser ``grid_size`` repairs the underlying voronoi partition, so it
    is tried first; the jitter rung then breaks the exact collinearity that no
    change of precision can fix. A ``grid_size`` pinned by the caller in
    ``kwargs`` (or already applied via ``overrides``) is never overridden, and
    the momepy concat failure has no remedy at all.

    Parameters
    ----------
    kind : str
        The failure kind from :func:`_classify_tessellation_failure`.
    overrides : dict[str, object]
        Retry overrides already in effect for the failed attempt.
    kwargs : dict[str, object]
        The caller-supplied momepy options (used to respect pinned values).

    Returns
    -------
    dict[str, object] or None
        The overrides for the next attempt, or ``None`` when the ladder is
        exhausted and the unit must degrade to an empty tessellation.
    """
    if kind != "empty" and "_jitter" not in overrides:
        if "grid_size" not in kwargs and "grid_size" not in overrides:
            return {**overrides, "grid_size": _COARSE_GRID_SIZE}
        # The final jitter rung replaces the earlier workaround instead of
        # stacking on it: snapping to the coarse grid would re-align the
        # jittered coordinates and reintroduce the degeneracy.
        return {"_jitter": True}
    return None


def _log_tessellation_retry(
    kind: str,
    error: Exception,
    added_option: str,
) -> None:
    """
    Log the retry decision for a known tessellation failure.

    The wording mirrors the failure kind and the option added by the next
    rung so operators can follow the ladder in the logs.

    Parameters
    ----------
    kind : str
        The failure kind from :func:`_classify_tessellation_failure`.
    error : Exception
        The failure that triggered the retry.
    added_option : str
        The option the next rung adds: ``"grid_size"`` or ``"_jitter"``.
    """
    if added_option == "_jitter":
        logger.warning(
            "Tessellation stayed degenerate (%s); retrying with jittered geometry.",
            error,
        )
    elif kind == "geos":
        logger.warning(
            "Tessellation hit a GEOS topology error (%s); retrying with coarser grid_size=1e-3.",
            error,
        )
    else:
        logger.warning(
            "Tessellation hit an incorrect geometry type (%s); retrying with grid_size=1e-3.",
            error,
        )


def _log_tessellation_degradation(error: Exception) -> None:
    """
    Log a known terminal failure before degrading the unit to an empty result.

    The warning message depends on the failure kind: the momepy concat
    ``ValueError`` keeps its historical wording, while geometry-type and GEOS
    failures report the exhausted retry ladder.

    Parameters
    ----------
    error : Exception
        The known failure (see :func:`_classify_tessellation_failure`) that
        exhausted the retry ladder.
    """
    if isinstance(error, ValueError):
        logger.warning("Momepy could not generate tessellation, returning empty GeoDataFrame.")
        return
    logger.warning(
        "Tessellation still failed at coarser precision; returning empty "
        "tessellation for this unit.",
    )


def _log_repair_retry_failure(error: Exception) -> None:
    """
    Log an exhausted overlap-repair retry.

    Unlike the main ladder, a failed repair retry does not degrade the unit
    to empty output: the original cells are kept and only the degenerate
    enclosures are dropped by :func:`_repair_or_drop_degenerate_enclosures`.

    Parameters
    ----------
    error : Exception
        The known failure that exhausted the repair retry.
    """
    logger.warning(
        "Tessellation retry at coarser precision failed (%s); keeping the original cells.",
        error,
    )


def _run_tessellation_with_retries(
    run_tessellation: typing.Callable[..., gpd.GeoDataFrame],
    kwargs: dict[str, object],
    overrides: dict[str, object] | None = None,
    log_exhausted: typing.Callable[[Exception], None] = _log_tessellation_degradation,
) -> tuple[gpd.GeoDataFrame | None, dict[str, object]]:
    """
    Run momepy enclosed tessellation through the known-failure retry ladder.

    ``shapely.voronoi_polygons`` can fail numerically inside an enclosure at
    momepy's default snapping precision; depending on the options this
    surfaces as a ``TypeError`` reporting an incorrect geometry type, a
    ``GEOSException``, or as a silently degenerate partition (handled
    separately by :func:`_repair_or_drop_degenerate_enclosures`). Each known
    failure escalates to the next rung chosen by
    :func:`_next_retry_overrides` — the coarser grid first, then jittered
    geometry — until the options are exhausted and the unit degrades to an
    empty tessellation. A caller-pinned ``grid_size`` is never overridden;
    only unknown errors propagate.

    Parameters
    ----------
    run_tessellation : Callable[..., geopandas.GeoDataFrame]
        Closure invoking ``momepy.enclosed_tessellation``; keyword arguments
        are applied as retry overrides on top of the caller-supplied options.
    kwargs : dict[str, object]
        The caller-supplied momepy options (used to respect pinned values).
    overrides : dict[str, object], optional
        Retry overrides to start the ladder from (e.g. the coarser
        ``grid_size`` for the overlap repair); the first attempt runs with
        them applied.
    log_exhausted : Callable[[Exception], None], optional
        Warning emitted when the ladder is exhausted, receiving the terminal
        failure. Defaults to :func:`_log_tessellation_degradation`.

    Returns
    -------
    tuple[geopandas.GeoDataFrame or None, dict[str, object]]
        ``(tessellation, overrides)`` where ``tessellation`` is ``None`` when
        momepy definitively failed and the caller should return an empty
        tessellation, and ``overrides`` records the retry options that
        produced the result.
    """
    overrides = dict(overrides or {})
    while True:
        try:
            return run_tessellation(**overrides), overrides
        except (ValueError, TypeError, shapely.errors.GEOSException) as error:
            kind = _classify_tessellation_failure(error)
            if kind is None:
                raise
            next_overrides = _next_retry_overrides(kind, overrides, kwargs)
            if next_overrides is None:
                log_exhausted(error)
                return None, overrides
            added_option = next(key for key in next_overrides if key not in overrides)
            _log_tessellation_retry(kind, error, added_option)
            overrides = next_overrides


def _repair_or_drop_degenerate_enclosures(
    tessellation: gpd.GeoDataFrame,
    enclosures: gpd.GeoDataFrame,
    run_tessellation: typing.Callable[..., gpd.GeoDataFrame],
    kwargs: dict[str, object],
    overrides: dict[str, object],
) -> gpd.GeoDataFrame:
    """
    Validate a tessellation and retry or drop degenerate enclosures.

    A degenerate voronoi partition can come back without any error at all, so
    the cells are validated against each other
    and against the enclosure areas before they can poison downstream spatial
    joins: overlapping cells claim the same buildings, and cells that lost area
    silently drop the building they were built around. A broken
    partition is retried once with a coarser ``grid_size`` (unless one is
    already in effect) and then once with deterministically jittered geometry
    (which breaks the exact collinearity that degenerates the voronoi
    partition of rectilinear footprints); when a retry itself fails with a
    known error the original cells are kept, and enclosures that stay
    degenerate have their cells dropped with a warning.

    Parameters
    ----------
    tessellation : geopandas.GeoDataFrame
        Raw ``momepy.enclosed_tessellation`` output.
    enclosures : geopandas.GeoDataFrame
        The enclosures the tessellation was generated from.
    run_tessellation : Callable[..., geopandas.GeoDataFrame]
        Closure invoking ``momepy.enclosed_tessellation`` with overrides.
    kwargs : dict[str, object]
        The caller-supplied momepy options (used to respect pinned values).
    overrides : dict[str, object]
        Retry overrides already applied to produce ``tessellation``.

    Returns
    -------
    geopandas.GeoDataFrame
        The validated tessellation, with degenerate enclosures repaired or
        removed.
    """
    broken = _degenerate_enclosures(tessellation, enclosures)
    if broken and "grid_size" not in kwargs and "grid_size" not in overrides:
        logger.warning(
            "Tessellation produced overlapping or incomplete cells in %d enclosure(s); "
            "retrying with grid_size=1e-3.",
            len(broken),
        )
        retried, overrides = _run_tessellation_with_retries(
            run_tessellation,
            kwargs,
            overrides={**overrides, "grid_size": _COARSE_GRID_SIZE},
            log_exhausted=_log_repair_retry_failure,
        )
        if retried is not None:
            tessellation = retried
            broken = _degenerate_enclosures(tessellation, enclosures)
    if broken and "_jitter" not in overrides:
        logger.warning(
            "Tessellation kept overlapping or incomplete cells in %d enclosure(s); "
            "retrying with jittered geometry.",
            len(broken),
        )
        # Jitter replaces the earlier workarounds (see _next_retry_overrides):
        # the coarse grid would re-align the jittered coordinates.
        retried, _ = _run_tessellation_with_retries(
            run_tessellation,
            kwargs,
            overrides={"_jitter": True},
            log_exhausted=_log_repair_retry_failure,
        )
        if retried is not None:
            tessellation = retried
            broken = _degenerate_enclosures(tessellation, enclosures)
    if broken:
        dropped = tessellation["enclosure_index"].isin(broken)
        logger.warning(
            "Dropping %d enclosure(s) whose tessellation cells overlap or leave gaps "
            "instead of partitioning the enclosure; %d cell(s) removed and their "
            "buildings degrade to footprint fallback cells.",
            len(broken),
            int(dropped.sum()),
        )
        tessellation = tessellation.loc[~dropped]
    return tessellation


def _run_enclosed_tessellation(
    *,
    geometry: gpd.GeoDataFrame | gpd.GeoSeries,
    enclosures: gpd.GeoDataFrame,
    shrink: float,
    segment: float,
    threshold: float,
    n_jobs: int,
    kwargs: dict[str, object],
    **extra_kwargs: object,
) -> gpd.GeoDataFrame:
    """
    Run momepy enclosed tessellation with retry-specific option overrides.

    The retry ladder passes only override options into this helper. This
    function merges those options with the caller-supplied momepy options and
    handles the internal ``_jitter`` marker before dispatching to momepy.

    Parameters
    ----------
    geometry : geopandas.GeoDataFrame or geopandas.GeoSeries
        The geometries to tessellate around.
    enclosures : geopandas.GeoDataFrame
        The enclosures to tessellate within.
    shrink : float
        The distance to shrink the geometry.
    segment : float
        The segment length for discretizing the geometry.
    threshold : float
        The threshold for snapping skeleton endpoints.
    n_jobs : int
        The number of jobs to use for parallel processing.
    kwargs : dict[str, object]
        Caller-supplied momepy options.
    **extra_kwargs : object
        Retry overrides, such as ``grid_size`` or the internal ``_jitter``
        marker.

    Returns
    -------
    geopandas.GeoDataFrame
        The enclosed tessellation.
    """
    options = {**kwargs, **extra_kwargs}
    tess_geometry = _jitter_geometry(geometry) if options.pop("_jitter", False) else geometry
    return momepy.enclosed_tessellation(
        geometry=tess_geometry,
        enclosures=enclosures,
        shrink=shrink,
        segment=segment,
        threshold=threshold,
        n_jobs=n_jobs,
        **options,
    )


def _create_enclosed_tessellation(
    geometry: gpd.GeoDataFrame | gpd.GeoSeries,
    primary_barriers: gpd.GeoDataFrame | gpd.GeoSeries,
    limit: gpd.GeoDataFrame | gpd.GeoSeries | BaseGeometry | None,
    shrink: float,
    segment: float,
    threshold: float,
    n_jobs: int,
    **kwargs: object,
) -> gpd.GeoDataFrame:
    """
    Create enclosed tessellation.

    This helper generates enclosed tessellations using momepy, handling the
    creation of enclosures from barriers and managing potential errors during
    generation.

    Parameters
    ----------
    geometry : geopandas.GeoDataFrame or geopandas.GeoSeries
        The geometries to tessellate around.
    primary_barriers : geopandas.GeoDataFrame or geopandas.GeoSeries
        Geometries to use as barriers.
    limit : geopandas.GeoDataFrame, geopandas.GeoSeries, shapely geometry, or None
        Boundary passed to `momepy.enclosures`. If None, a buffered union of
        geometry and barriers is derived and the enclosures are clipped to it.
    shrink : float
        The distance to shrink the geometry.
    segment : float
        The segment length for discretizing the geometry.
    threshold : float
        The threshold for snapping skeleton endpoints.
    n_jobs : int
        The number of jobs to use for parallel processing.
    **kwargs : object
        Additional keyword arguments passed to momepy.

    Returns
    -------
    geopandas.GeoDataFrame
        The enclosed tessellation.
    """
    derived_limit = limit is None
    if derived_limit:
        limit = _compute_enclosure_limit(geometry, primary_barriers)

    # The derived limit is a non-convex buffered union that may contain holes;
    # polygonization turns those holes into faces outside the limit, so they
    # must be clipped away. User-supplied limits keep clip=False because
    # momepy's clip requires a polygonal limit and clipping would change the
    # documented semantics of an explicit ``limit``.
    enclosures = momepy.enclosures(
        primary_barriers=primary_barriers,
        limit=limit,
        additional_barriers=None,
        enclosure_id="eID",
        clip=derived_limit and limit is not None,
    )

    if not enclosures.empty:
        run_enclosed_tessellation = partial(
            _run_enclosed_tessellation,
            geometry=geometry,
            enclosures=enclosures,
            shrink=shrink,
            segment=segment,
            threshold=threshold,
            n_jobs=n_jobs,
            kwargs=kwargs,
        )
        tessellation, overrides = _run_tessellation_with_retries(
            run_enclosed_tessellation,
            kwargs,
        )
        if tessellation is None:
            # ``momepy.enclosed_tessellation`` crashes with "No objects to
            # concatenate" when no enclosure holds two or more buildings,
            # although every single-building enclosure is a perfectly valid
            # cell; recover those units instead of degrading them to empty.
            tessellation = _recover_tessellation_without_splits(geometry, enclosures)
            if tessellation is None:
                return _create_empty_tessellation(geometry.crs)
        else:
            tessellation = _repair_or_drop_degenerate_enclosures(
                tessellation,
                enclosures,
                run_enclosed_tessellation,
                kwargs,
                overrides,
            )
    else:
        tessellation = _create_empty_tessellation(geometry.crs)

    tessellation = _polygonal_tessellation(tessellation)
    if tessellation.empty:
        return _create_empty_tessellation(geometry.crs)

    tessellation["tess_id"] = [
        f"{i}_{j}"
        for i, j in zip(tessellation["enclosure_index"], tessellation.index, strict=False)
    ]
    return tessellation.reset_index(drop=True)


def _recover_tessellation_without_splits(
    geometry: gpd.GeoDataFrame | gpd.GeoSeries,
    enclosures: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame | None:
    """
    Recover the tessellation after momepy's empty-concatenation crash.

    ``momepy.enclosed_tessellation`` voronoi-partitions only enclosures that
    intersect two or more buildings and crashes with "No objects to
    concatenate" when none exist, although single-building and empty
    enclosures are perfectly valid cells. When that precondition holds (and
    at least one polygonal building sits in an enclosure) the equivalent
    output is built directly; otherwise ``None`` signals that the failure was
    genuine and the unit must degrade to an empty tessellation.

    Parameters
    ----------
    geometry : geopandas.GeoDataFrame or geopandas.GeoSeries
        Building footprints passed to the tessellation.
    enclosures : geopandas.GeoDataFrame
        Enclosure polygons produced by ``momepy.enclosures``.

    Returns
    -------
    geopandas.GeoDataFrame or None
        The recovered tessellation, or ``None`` when recovery does not apply.
    """
    polygonal = geometry.geometry.notna() & geometry.geometry.geom_type.isin(
        ["Polygon", "MultiPolygon"],
    )
    buildings = geometry[polygonal]
    if buildings.empty:
        return None
    enclosure_positions, _ = buildings.sindex.query(
        enclosures.geometry,
        predicate="intersects",
    )
    if len(enclosure_positions) == 0:
        return None
    _, counts = np.unique(enclosure_positions, return_counts=True)
    if (counts > 1).any():
        # Some enclosure did need splitting, so the crash came from elsewhere.
        return None
    logger.warning(
        "Momepy could not tessellate because no enclosure holds two or more "
        "buildings; assigning single-building enclosures as cells directly.",
    )
    return _enclosure_cells_without_splits(buildings, enclosures)


def _enclosure_cells_without_splits(
    geometry: gpd.GeoDataFrame | gpd.GeoSeries,
    enclosures: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Build the enclosed tessellation when no enclosure needs splitting.

    Equivalent of the ``momepy.enclosed_tessellation`` output for the case its
    own implementation crashes on: every enclosure intersecting exactly one
    building becomes that building's cell (indexed by the building), and
    enclosures without buildings keep momepy's negative-index convention.

    Parameters
    ----------
    geometry : geopandas.GeoDataFrame or geopandas.GeoSeries
        Building footprints.
    enclosures : geopandas.GeoDataFrame
        Enclosure polygons produced by ``momepy.enclosures``.

    Returns
    -------
    geopandas.GeoDataFrame
        One cell per enclosure with the ``enclosure_index`` column set,
        matching the raw momepy output schema.
    """
    enclosure_positions, building_positions = geometry.sindex.query(
        enclosures.geometry,
        predicate="intersects",
    )
    cell_index = np.full(len(enclosures), None, dtype=object)
    cell_index[enclosure_positions] = geometry.geometry.index.to_numpy()[building_positions]
    empty_mask = np.array([index is None for index in cell_index])
    cell_index[empty_mask] = range(-int(empty_mask.sum()), 0)
    cells = gpd.GeoDataFrame(
        {"enclosure_index": enclosures.index},
        geometry=enclosures.geometry.to_numpy(),
        index=pd.Index(cell_index),
        crs=enclosures.crs,
    )
    # Match the raw momepy output column order (geometry first).
    return cells[["geometry", "enclosure_index"]]


def _compute_enclosure_limit(
    geometry: gpd.GeoDataFrame | gpd.GeoSeries,
    primary_barriers: gpd.GeoDataFrame | gpd.GeoSeries,
    buffer: float = 100,
) -> BaseGeometry | None:
    """
    Compute a buffered-union limit for momepy enclosures.

    The limit needs to include both buildings and street barriers so buildings
    near the outer street loops are not excluded from enclosed tessellation.
    It follows the built fabric (buffered union) rather than a convex hull:
    a hull leaves a vast outermost enclosure whose Voronoi cells stretch from
    street-front buildings deep into empty land as needle-shaped artifacts.

    Parameters
    ----------
    geometry : gpd.GeoDataFrame | gpd.GeoSeries
        Building geometries used to compute boundaries.
    primary_barriers : gpd.GeoDataFrame | gpd.GeoSeries
        Street geometries forming enclosures.
    buffer : float, default 100
        Distance in map units to buffer each geometry before the union.

    Returns
    -------
    BaseGeometry | None
        A single geometry representing the enclosure limit, or None if no valid
        geometries exist. The result may contain holes (block interiors farther
        than ``buffer`` from any street or building); callers must clip the
        enclosures to the limit so those holes do not become enclosures.
    """
    geometries = [
        geom
        for source in (geometry.geometry, primary_barriers.geometry)
        for geom in source
        if geom is not None and not geom.is_empty
    ]
    if not geometries:
        return None

    return gpd.GeoSeries(geometries, crs=geometry.crs).buffer(buffer).union_all()


def _create_empty_tessellation(crs: Any) -> gpd.GeoDataFrame:  # noqa: ANN401
    """
    Create an empty enclosed-tessellation GeoDataFrame.

    This helper generates a properly structured but empty GeoDataFrame when no
    tessellation can be generated, matching the column schema of non-empty
    enclosed tessellations so downstream consumers see a uniform shape.

    Parameters
    ----------
    crs : Any
        The Coordinate Reference System.

    Returns
    -------
    geopandas.GeoDataFrame
        An empty tessellation GeoDataFrame with the columns ``geometry``,
        ``enclosure_index`` and ``tess_id``.
    """
    return gpd.GeoDataFrame(
        columns=["geometry", "enclosure_index", "tess_id"],
        geometry="geometry",
        crs=crs,
    )


def _create_morphological_tessellation(
    geometry: gpd.GeoDataFrame | gpd.GeoSeries,
    shrink: float,
    segment: float,
) -> gpd.GeoDataFrame:
    """
    Create morphological tessellation.

    This helper generates morphological tessellations using momepy, which relies
    on the geometry itself without external barriers.

    Parameters
    ----------
    geometry : geopandas.GeoDataFrame or geopandas.GeoSeries
        The geometries to tessellate around.
    shrink : float
        The distance to shrink the geometry.
    segment : float
        The segment length for discretizing the geometry.

    Returns
    -------
    geopandas.GeoDataFrame
        The morphological tessellation.
    """
    tessellation = momepy.morphological_tessellation(
        geometry=geometry,
        clip="bounding_box",
        shrink=shrink,
        segment=segment,
    )

    tessellation["tess_id"] = tessellation.index
    return tessellation
