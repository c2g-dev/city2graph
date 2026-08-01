"""
Data Loading and Processing Module.

This module provides comprehensive functionality for loading and processing geospatial
data from various sources, with specialized support for Overture Maps data. It handles
data validation, coordinate reference system management, and geometric processing
operations commonly needed for urban network analysis.
"""

# Standard library imports
import io
import json
import subprocess
import warnings
from pathlib import Path

# Third-party imports
import geopandas as gpd
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from overturemaps.core import ALL_RELEASES
from pyproj import CRS
from shapely import get_coordinates
from shapely import get_point
from shapely import get_type_id
from shapely import get_x
from shapely import get_y
from shapely import is_empty
from shapely import set_coordinates
from shapely.geometry import LineString
from shapely.geometry import MultiLineString
from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon
from shapely.ops import substring

from .utils import clip_graph

# Public API definition
__all__ = ["get_boundaries", "load_overture_data", "process_overture_segments"]

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Valid Overture Maps data types for validation
VALID_OVERTURE_TYPES = {
    "address",
    "bathymetry",
    "building",
    "building_part",
    "division",
    "division_area",
    "division_boundary",
    "place",
    "segment",
    "connector",
    "infrastructure",
    "land",
    "land_cover",
    "land_use",
    "water",
}

# Standard coordinate reference system
WGS84_CRS = "EPSG:4326"


def load_overture_data(
    area: list[float] | Polygon | MultiPolygon | gpd.GeoSeries | gpd.GeoDataFrame | None = None,
    place_name: str | None = None,
    types: list[str] | None = None,
    output_dir: str = ".",
    prefix: str = "",
    save_to_file: bool = True,
    return_data: bool = True,
    release: str | None = None,
    connect_timeout: float | None = None,
    request_timeout: float | None = None,
    use_stac: bool = True,
    **kwargs: bool,
) -> dict[str, gpd.GeoDataFrame]:
    """
    Load data from Overture Maps using the CLI tool and optionally save to GeoJSON files.

    This function downloads geospatial data from Overture Maps for a specified area
    and data types. It can save the data to GeoJSON files and/or return it as
    GeoDataFrames.

    Parameters
    ----------
    area : list[float], Polygon, MultiPolygon, GeoSeries, or GeoDataFrame, optional
        The area of interest. Can be:
        - A bounding box as [min_lon, min_lat, max_lon, max_lat] in WGS84
        - A Shapely Polygon/MultiPolygon (assumed to be in WGS84)
        - A GeoSeries or GeoDataFrame with CRS info (will be reprojected to WGS84 if needed)
        Mutually exclusive with place_name.
    place_name : str, optional
        Name of a place to geocode (e.g., "Liverpool, UK"). Uses Nominatim to retrieve
        the boundary polygon. Mutually exclusive with area.
    types : list[str], optional
        List of Overture data types to download. If None, downloads all available types.

        Available types:

        | Type | Description |
        |------|-------------|
        | ``address`` | Represents a physical place through a series of attributes (street number, etc). |
        | ``bathymetry`` | Derived vectorized bathymetric data products from ETOPO1 and GLOBathy. |
        | ``building`` | The most basic form of a building feature; geometry is the outer footprint. |
        | ``building_part`` | A single part of a building (e.g. 3D part); associated with a parent building. |
        | ``connector`` | Point feature connecting segments in the transportation network. |
        | ``division`` | Represents an official/non-official organization of people (country, city, etc). |
        | ``division_area`` | Captures the shape of the land/maritime area belonging to a division. |
        | ``division_boundary`` | Represents a shared border between two division features. |
        | ``infrastructure`` | Features such as communication towers, lines, piers, and bridges. |
        | ``land`` | Physical representations of land surfaces derived from OSM Coastlines. |
        | ``land_cover`` | Derived from ESA WorldCover high-resolution optical Earth observation data. |
        | ``land_use`` | Classifications of the human use of a section of land (from OSM landuse). |
        | ``place`` | Points of interest: schools, businesses, hospitals, landmarks, etc. |
        | ``segment`` | LineString feature representing paths for travel (road, rail, water). |
        | ``water`` | Physical representations of inland and ocean marine surfaces. |

        For more information, see the [Overture Maps documentation](https://docs.overturemaps.org/schema/).
    output_dir : str, default "."
        Directory where GeoJSON files will be saved.
    prefix : str, default ""
        Prefix to add to output filenames.
    save_to_file : bool, default True
        Whether to save downloaded data to GeoJSON files.
    return_data : bool, default True
        Whether to return the data as GeoDataFrames.
    release : str, optional
        Overture Maps release version to use (e.g., '2024-11-13.0'). If None, uses the
        default release from the CLI tool. Must be a valid release from the overturemaps
        library's ALL_RELEASES list.
    connect_timeout : float, optional
        Socket connection timeout in seconds. If None, uses the AWS SDK default value
        (typically 1 second).
    request_timeout : float, optional
        Socket read timeout in seconds (Windows and macOS only). If None, uses the AWS SDK
        default value (typically 3 seconds). This option is ignored on non-Windows,
        non-macOS systems.
    use_stac : bool, default True
        Whether to use Overture's STAC-geoparquet catalog to speed up queries. If False,
        data will be read normally without the STAC optimization.
    **kwargs
        Additional keyword arguments:
        - keep_outer_neighbors (bool, default False): Whether to keep segments that
          partially intersect the boundary (True) or only those fully within (False).

    Returns
    -------
    dict[str, geopandas.GeoDataFrame]
        Dictionary mapping data type names to their corresponding GeoDataFrames.

    Raises
    ------
    ValueError
        If invalid data types are specified or if an invalid release version is provided.
    subprocess.CalledProcessError
        If the Overture Maps CLI command fails.

    See Also
    --------
    process_overture_segments : Process segments from Overture Maps.

    Examples
    --------
    >>> # Download building and segment data for a bounding box
    >>> bbox = [-74.01, 40.70, -73.99, 40.72]  # Manhattan area
    >>> data = load_overture_data(bbox, types=['building', 'segment'])
    >>> buildings = data['building']
    >>> segments = data['segment']

    >>> # Download with a specific release version
    >>> data = load_overture_data(bbox, types=['building'], release='2024-11-13.0')

    >>> # Download with custom timeout settings
    >>> data = load_overture_data(
    ...     bbox,
    ...     types=['building'],
    ...     connect_timeout=5.0,
    ...     request_timeout=10.0
    ... )

    >>> # Download without STAC optimization
    >>> data = load_overture_data(bbox, types=['building'], use_stac=False)
    """
    # Validate area/place_name mutual exclusion
    if (area is None) == (place_name is None):
        msg = "Exactly one of 'area' or 'place_name' must be provided"
        raise ValueError(msg)

    if place_name is not None:
        area = get_boundaries(place_name).geometry.iloc[0]

    # Validate input parameters
    types = types or list(VALID_OVERTURE_TYPES)
    invalid_types = [t for t in types if t not in VALID_OVERTURE_TYPES]
    if invalid_types:
        msg = f"Invalid types: {invalid_types}"
        raise ValueError(msg)

    # Validate release parameter if provided
    if release is not None and ALL_RELEASES is not None and release not in ALL_RELEASES:
        msg = f"Invalid release: {release}. Valid releases are: {', '.join(ALL_RELEASES)}"
        raise ValueError(msg)

    # Prepare area and bounding box
    bbox_str, clip_geom = _prepare_area_and_bbox(area)

    # Create output directory if needed
    if save_to_file:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Download and process each data type
    result = {}
    for data_type in types:
        gdf = _download_and_process_type(
            data_type,
            bbox_str,
            output_dir,
            prefix,
            save_to_file,
            return_data,
            clip_geom,
            release,
            connect_timeout,
            request_timeout,
            use_stac,
            keep_outer_neighbors=kwargs.get("keep_outer_neighbors", False),
        )
        if return_data:
            result[data_type] = gdf

    return result


def get_boundaries(place_name: str, user_agent: str = "city2graph") -> gpd.GeoDataFrame:
    """
    Retrieve polygon boundary for a place using Nominatim geocoding.

    Uses the Nominatim geocoding service to find the geographic boundary
    of a named place (city, country, region) and returns it as a GeoDataFrame.

    Parameters
    ----------
    place_name : str
        Name of the place to geocode (e.g., "Liverpool, UK").
    user_agent : str, default "city2graph"
        User agent string for Nominatim API.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with polygon geometry and place_name property.

    Raises
    ------
    ValueError
        If place is not found or returns non-polygon geometry.

    Examples
    --------
    >>> boundary = get_boundaries("Liverpool, UK")
    >>> data = load_overture_data(area=boundary.geometry.iloc[0], types=['building'])
    """
    # Get all geocoding results (not just the first one)
    locations = Nominatim(user_agent=user_agent).geocode(
        place_name, geometry="geojson", exactly_one=False
    )

    if not locations:
        msg = f"Place not found: '{place_name}'"
        raise ValueError(msg)

    # Find the first result with a Polygon or MultiPolygon geometry
    geojson = None
    for location in locations:
        geom = location.raw.get("geojson")
        if geom is not None and geom.get("type") in ("Polygon", "MultiPolygon"):
            geojson = geom
            break

    if geojson is None:
        msg = f"No polygon boundary for '{place_name}'. Try an administrative region."
        raise ValueError(msg)

    return gpd.GeoDataFrame.from_features(
        [{"type": "Feature", "geometry": geojson, "properties": {"place_name": place_name}}],
        crs=WGS84_CRS,
    )


def process_overture_segments(
    segments_gdf: gpd.GeoDataFrame,
    get_barriers: bool = True,
    connectors_gdf: gpd.GeoDataFrame | None = None,
    threshold: float = 1.0,
) -> gpd.GeoDataFrame:
    """
    Process segments from Overture Maps to be split by connectors and extract barriers.

    This function processes road segments by splitting them at connector points and
    optionally generates barrier geometries based on level rules. It also performs
    endpoint clustering to snap nearby endpoints together.

    Parameters
    ----------
    segments_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing road segments with LineString geometries.
        Expected to have 'connectors' and 'level_rules' columns.
    get_barriers : bool, default True
        Whether to generate barrier geometries from level rules.
    connectors_gdf : geopandas.GeoDataFrame, optional
        GeoDataFrame containing connector information. If provided, segments
        will be split at connector positions.
    threshold : float, default 1.0
        Distance threshold for endpoint clustering in the same units as the CRS.

    Returns
    -------
    geopandas.GeoDataFrame
        Processed segments with additional columns:
        - 'split_from', 'split_to': Split positions if segments were split
        - 'length': Length of each segment
        - 'barrier_geometry': Passable geometry if get_barriers=True

    See Also
    --------
    load_overture_data : Load data from Overture Maps.

    Examples
    --------
    >>> # Process segments with connector splitting
    >>> processed = process_overture_segments(
    ...     segments_gdf,
    ...     connectors_gdf=connectors_gdf,
    ...     threshold=1.0
    ... )
    >>> # Access barrier geometries for routing
    >>> barriers = processed['barrier_geometry']
    """
    if segments_gdf.empty:
        return segments_gdf

    # Warn if CRS is geographic or missing
    if segments_gdf.crs is None or segments_gdf.crs == WGS84_CRS:
        warnings.warn(
            "Segments GeoDataFrame has no CRS or is in WGS84 (EPSG:4326). "
            "Projected CRS is recommended for accurate length calculation and processing.",
            UserWarning,
            stacklevel=2,
        )

    # Initialize result and ensure required columns exist
    result_gdf = segments_gdf.copy()
    if "level_rules" not in result_gdf.columns:
        result_gdf["level_rules"] = ""
    else:
        result_gdf["level_rules"] = result_gdf["level_rules"].fillna("")

    # Split segments at connector positions
    result_gdf = _split_segments_at_connectors(result_gdf, connectors_gdf)

    # Cluster endpoints to snap nearby points together
    if connectors_gdf is not None and not connectors_gdf.empty:
        result_gdf = _cluster_segment_endpoints(result_gdf, threshold)

    # Add segment length
    result_gdf["length"] = result_gdf.geometry.length

    # Generate barrier geometries if requested
    if get_barriers:
        result_gdf["barrier_geometry"] = _generate_barrier_geometries(result_gdf)

    return result_gdf


def _prepare_area_and_bbox(
    area: list[float] | Polygon | MultiPolygon | gpd.GeoSeries | gpd.GeoDataFrame,
) -> tuple[str, Polygon | MultiPolygon | None]:
    """
    Prepare area input and convert to bbox string and clipping geometry.

    This function processes area input to create a bounding box string for API queries
    and optionally a clipping geometry for precise spatial filtering.

    Parameters
    ----------
    area : list[float], Polygon, MultiPolygon, GeoSeries, or GeoDataFrame
        The area of interest. Can be:
        - A bounding box as [min_lon, min_lat, max_lon, max_lat] in WGS84
        - A Shapely Polygon/MultiPolygon (assumed to be in WGS84)
        - A GeoSeries or GeoDataFrame with CRS info (will be reprojected to WGS84 if needed)

    Returns
    -------
    tuple[str, Polygon | MultiPolygon | None]
        Tuple containing bbox string and optional clipping geometry (in WGS84).

    See Also
    --------
    _download_and_process_type : Uses this function for area preparation.

    Examples
    --------
    >>> bbox = [-74.1, 40.7, -74.0, 40.8]
    >>> bbox_str, clip_geom = _prepare_area_and_bbox(bbox)
    >>> bbox_str
    '-74.1,40.7,-74.0,40.8'
    """
    # Handle GeoDataFrame/GeoSeries - extract geometry and reproject if needed
    if isinstance(area, (gpd.GeoDataFrame, gpd.GeoSeries)):
        wgs84_crs = CRS.from_epsg(4326)
        if area.crs is not None and CRS.from_user_input(area.crs) != wgs84_crs:
            area = area.to_crs(wgs84_crs)
        # Extract the first geometry
        area = area.geometry.iloc[0] if isinstance(area, gpd.GeoDataFrame) else area.iloc[0]

    if isinstance(area, (Polygon, MultiPolygon)):
        bbox_str = ",".join(str(round(c, 10)) for c in area.bounds)
        clip_geom = area
    else:
        bbox_str = ",".join(str(float(b)) for b in area)
        clip_geom = None

    return bbox_str, clip_geom


def _download_and_process_type(  # noqa: PLR0912, PLR0913, C901
    data_type: str,
    bbox_str: str,
    output_dir: str,
    prefix: str,
    save_to_file: bool,
    return_data: bool,
    clip_geom: Polygon | MultiPolygon | None,
    release: str | None = None,
    connect_timeout: float | None = None,
    request_timeout: float | None = None,
    use_stac: bool = True,
    keep_outer_neighbors: bool = False,
) -> gpd.GeoDataFrame:
    """
    Download and process a single data type from Overture Maps.

    This function handles the download and processing of a specific data type
    from Overture Maps, including optional clipping and file saving.

    Parameters
    ----------
    data_type : str
        Type of data to download (e.g., 'building', 'transportation').
    bbox_str : str
        Bounding box string for the API query.
    output_dir : str
        Directory to save output files.
    prefix : str
        Prefix for output filenames.
    save_to_file : bool
        Whether to save data to file.
    return_data : bool
        Whether to return the data.
    clip_geom : Polygon, MultiPolygon, or None
        Optional geometry for precise clipping.
    release : str or None
        Overture Maps release version to use.
    connect_timeout : float, optional
        Socket connection timeout in seconds. If None, uses the AWS SDK default value
        (typically 1 second).
    request_timeout : float, optional
        Socket read timeout in seconds (Windows and macOS only). If None, uses the AWS SDK
        default value (typically 3 seconds). This option is ignored on non-Windows,
        non-macOS systems.
    use_stac : bool, default True
        Whether to use Overture's STAC-geoparquet catalog to speed up queries. If False,
        data will be read normally without the STAC optimization.
    keep_outer_neighbors : bool, default False
        Whether to keep segments that partially intersect the boundary.

    Returns
    -------
    gpd.GeoDataFrame
        Processed geospatial data.

    See Also
    --------
    get_overture_data : Main function using this helper.

    Examples
    --------
    >>> gdf = _download_and_process_type('building', '-74.1,40.7,-74.0,40.8',
    ...                                  './data', 'nyc', True, True, None, '2024-11-13.0')
    """
    output_path = Path(output_dir) / f"{prefix}{data_type}.geojson"
    needs_postprocessing = clip_geom is not None or data_type == "segment"

    # Build and execute download command
    cmd = ["overturemaps", "download", f"--bbox={bbox_str}", "-f", "geojson", f"--type={data_type}"]
    if release:
        cmd.extend(["-r", release])
    if connect_timeout is not None:
        cmd.extend(["--connect-timeout", str(connect_timeout)])
    if request_timeout is not None:
        cmd.extend(["--request-timeout", str(request_timeout)])
    if not use_stac:
        cmd.append("--no-stac")
    if save_to_file:
        cmd.extend(["-o", str(output_path)])

    result = subprocess.run(cmd, check=True, capture_output=not save_to_file, text=True)

    output_exists = save_to_file and output_path.exists()
    if not return_data and not (output_exists and needs_postprocessing):
        return gpd.GeoDataFrame(geometry=[], crs=WGS84_CRS)

    # Load and clip data if needed
    if output_exists:
        gdf = gpd.read_file(output_path, encoding="utf-8")
    elif not save_to_file and result.stdout and isinstance(result.stdout, str):
        # Parse GeoJSON from subprocess stdout when not saving to file
        # The CLI may output warning messages before the GeoJSON, so we need to
        # find where the actual JSON starts (either { for object or [ for array)
        stdout = result.stdout
        json_start = -1
        for i, char in enumerate(stdout):
            if char in "{[":
                json_start = i
                break
        gdf = (
            gpd.read_file(io.StringIO(stdout[json_start:]))
            if json_start >= 0
            else gpd.GeoDataFrame(geometry=[], crs=WGS84_CRS)
        )
    else:
        gdf = gpd.GeoDataFrame(geometry=[], crs=WGS84_CRS)

    if clip_geom is not None and not gdf.empty:
        clip_gdf = gpd.GeoDataFrame(geometry=[clip_geom], crs=WGS84_CRS)
        # Handle CRS conversion and clipping safely, including for mock objects in tests

        # Only proceed if both CRS are real CRS objects or strings (but not Mock objects)
        clip_crs_valid = isinstance(clip_gdf.crs, (CRS, str, type(None)))
        gdf_crs_valid = isinstance(gdf.crs, (CRS, str, type(None)))

        if clip_crs_valid and gdf_crs_valid:
            if clip_gdf.crs != gdf.crs:
                clip_gdf = clip_gdf.to_crs(gdf.crs)

            if data_type == "segment":
                # For segments, use topological subsetting to preserve network integrity
                gdf = clip_graph(
                    gdf, clip_gdf.geometry.iloc[0], keep_outer_neighbors=keep_outer_neighbors
                )
            else:
                # For visual features (buildings, etc.), use geometric clipping
                gdf = gpd.clip(gdf, clip_gdf)

    # Filter non-LineString geometries for segments
    if data_type == "segment" and not gdf.empty:
        # Keep only LineStrings and explode MultiLineString
        lines = gdf[gdf.geometry.type == "LineString"]
        multi = gdf[gdf.geometry.type == "MultiLineString"]
        if not multi.empty:
            exploded = multi.explode(index_parts=False)
            lines = pd.concat([lines, exploded])
        gdf = lines.reset_index(drop=True)

    if output_exists and needs_postprocessing:
        gdf.to_file(output_path, driver="GeoJSON")

    return gdf


def _split_segments_at_connectors(
    segments_gdf: gpd.GeoDataFrame,
    connectors_gdf: gpd.GeoDataFrame | None,
) -> gpd.GeoDataFrame:
    """
    Split segments at connector positions.

    This function splits road segments at connector positions to create
    a more detailed network representation suitable for graph analysis.

    Parameters
    ----------
    segments_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing road segments.
    connectors_gdf : geopandas.GeoDataFrame or None
        GeoDataFrame containing connector information.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with segments split at connector positions.

    Examples
    --------
    >>> segments = gpd.GeoDataFrame({'geometry': [LineString([(0,0), (1,1)])]})
    >>> connectors = gpd.GeoDataFrame({'id': ['c1']})
    >>> split_segments = _split_segments_at_connectors(segments, connectors)
    """
    if connectors_gdf is None or connectors_gdf.empty:
        return segments_gdf

    valid_connector_ids = set(connectors_gdf["id"])
    connector_values = segments_gdf.get(
        "connectors",
        pd.Series("", index=segments_gdf.index, dtype=object),
    )
    parsed_connectors = connector_values.map(_parse_connector_records)

    # Normalize all connector records at once, retaining positional row IDs so
    # duplicate/non-unique input indexes do not affect ordering.
    connector_rows = pd.DataFrame(
        {
            "_segment_pos": np.arange(len(segments_gdf)),
            "_connector": parsed_connectors.to_numpy(),
        }
    ).explode("_connector")
    connector_rows = connector_rows[
        connector_rows["_connector"].map(lambda value: isinstance(value, dict))
    ]

    positions = pd.Series(
        [[0.0, 1.0] for _ in range(len(segments_gdf))],
        index=pd.RangeIndex(len(segments_gdf)),
        dtype=object,
    )
    if not connector_rows.empty:
        normalized = pd.json_normalize(connector_rows["_connector"])
        normalized["_segment_pos"] = connector_rows["_segment_pos"].to_numpy()
        if {"connector_id", "at"} <= set(normalized.columns):
            normalized = normalized[
                normalized["connector_id"].isin(valid_connector_ids) & normalized["at"].notna()
            ]
            normalized["at"] = normalized["at"].map(float)
            grouped_positions = normalized.groupby("_segment_pos", sort=False)["at"].agg(
                lambda values: sorted({0.0, *values, 1.0})
            )
            positions.loc[grouped_positions.index] = grouped_positions

    # Convert position lists into one interval table. This duplicates source
    # rows in a single indexed selection and preserves source/part ordering.
    bounds = positions.explode().rename("_split_from").to_frame()
    bounds["_segment_pos"] = bounds.index
    bounds["_part"] = bounds.groupby("_segment_pos", sort=False).cumcount()
    bounds["_split_to"] = bounds.groupby("_segment_pos", sort=False)["_split_from"].shift(-1)
    intervals = bounds[bounds["_split_to"].notna()].copy()
    interval_counts = intervals.groupby("_segment_pos", sort=False)["_part"].transform("size")
    split_mask = interval_counts.gt(1).to_numpy()

    result_gdf = segments_gdf.iloc[intervals["_segment_pos"].to_numpy()].copy()
    result_gdf = result_gdf.reset_index(drop=True)

    if split_mask.any():
        split_intervals = intervals.loc[split_mask]
        split_starts = split_intervals["_split_from"].astype(float).to_numpy()
        split_ends = split_intervals["_split_to"].astype(float).to_numpy()
        source_geometries = result_gdf.geometry.to_numpy()
        split_geometries = [
            substring(geometry, start, end, normalized=True)
            for geometry, start, end in zip(
                source_geometries[split_mask],
                split_starts,
                split_ends,
                strict=True,
            )
        ]
        source_geometries[split_mask] = split_geometries
        result_gdf.geometry = gpd.GeoSeries(
            source_geometries,
            index=result_gdf.index,
            crs=segments_gdf.crs,
        )

        if "split_from" not in result_gdf:
            result_gdf["split_from"] = np.nan
        if "split_to" not in result_gdf:
            result_gdf["split_to"] = np.nan
        result_gdf.loc[split_mask, "split_from"] = split_starts
        result_gdf.loc[split_mask, "split_to"] = split_ends

        source_positions = intervals.loc[split_mask, "_segment_pos"].to_numpy()
        original_ids = (
            segments_gdf["id"].to_numpy() if "id" in segments_gdf else segments_gdf.index.to_numpy()
        )
        part_numbers = intervals.loc[split_mask, "_part"].to_numpy() + 1
        result_gdf.loc[split_mask, "id"] = [
            f"{original_ids[source_pos]}_{part_number}"
            for source_pos, part_number in zip(source_positions, part_numbers, strict=True)
        ]

        valid_parts = np.ones(len(result_gdf), dtype=bool)
        valid_parts[split_mask] = [
            geometry is not None and not geometry.is_empty for geometry in split_geometries
        ]
        result_gdf = result_gdf.loc[valid_parts].reset_index(drop=True)

    return result_gdf


def _parse_connector_records(value: object) -> list[object]:
    """
    Normalize a connector value to a list of records.

    JSON-like string values are decoded permissively to match the Overture
    values accepted by the existing processing path.

    Parameters
    ----------
    value : object
        Connector data represented as a list, JSON-like string, mapping, or null value.

    Returns
    -------
    list[object]
        Parsed connector records. Invalid values produce an empty list.
    """
    if isinstance(value, list):
        return value
    if value is None or value == "":
        return []
    try:
        parsed = json.loads(str(value).replace("'", '"').replace("None", "null"))
    except (json.JSONDecodeError, AttributeError):
        return []
    if isinstance(parsed, list):
        return parsed
    return [parsed] if parsed else []


def _cluster_segment_endpoints(
    segments_gdf: gpd.GeoDataFrame,
    threshold: float,
) -> gpd.GeoDataFrame:
    """
    Cluster segment endpoints to snap nearby points together.

    This function performs spatial clustering of segment endpoints to snap
    nearby points together, improving network connectivity.

    Parameters
    ----------
    segments_gdf : gpd.GeoDataFrame
        GeoDataFrame containing road segments.
    threshold : float
        Distance threshold for clustering endpoints.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with adjusted segment endpoints.

    See Also
    --------
    process_overture_segments : Main function using this helper.

    Examples
    --------
    >>> segments = gpd.GeoDataFrame({'geometry': [LineString([(0,0), (1,1)])]})
    >>> clustered = _cluster_segment_endpoints(segments, 0.1)
    """
    result_gdf = segments_gdf.copy()
    geometries = result_gdf.geometry.to_numpy()
    line_mask = (get_type_id(geometries) == 1) & ~is_empty(geometries)
    line_positions = np.flatnonzero(line_mask)
    if not len(line_positions):
        return result_gdf

    line_geometries = geometries[line_positions]
    start_points = get_point(line_geometries, 0)
    end_points = get_point(line_geometries, -1)
    endpoint_x = np.column_stack((get_x(start_points), get_x(end_points))).ravel()
    endpoint_y = np.column_stack((get_y(start_points), get_y(end_points))).ravel()

    endpoints_df = pd.DataFrame({"x": endpoint_x, "y": endpoint_y})
    endpoints_df["bin_x"] = (endpoints_df["x"] / threshold).round().astype(int)
    endpoints_df["bin_y"] = (endpoints_df["y"] / threshold).round().astype(int)
    centroid_coordinates = (
        endpoints_df.groupby(["bin_x", "bin_y"], sort=False)[["x", "y"]]
        .transform("mean")
        .to_numpy()
        .reshape(-1, 2, 2)
    )

    coordinates, geometry_indexes = get_coordinates(line_geometries, return_index=True)
    first_coordinate = np.flatnonzero(np.r_[True, geometry_indexes[1:] != geometry_indexes[:-1]])
    last_coordinate = np.r_[first_coordinate[1:] - 1, len(coordinates) - 1]
    coordinates[first_coordinate, :2] = centroid_coordinates[:, 0, :]
    coordinates[last_coordinate, :2] = centroid_coordinates[:, 1, :]

    updated_geometries = set_coordinates(line_geometries.copy(), coordinates)
    geometries[line_positions] = updated_geometries
    result_gdf.geometry = gpd.GeoSeries(
        geometries,
        index=result_gdf.index,
        crs=segments_gdf.crs,
    )

    return result_gdf


def _generate_barrier_geometries(segments_gdf: gpd.GeoDataFrame) -> gpd.GeoSeries:
    """
    Generate barrier geometries from level rules.

    This function processes level rules to create barrier geometries that
    represent passable portions of road segments.

    Parameters
    ----------
    segments_gdf : gpd.GeoDataFrame
        GeoDataFrame containing segments with level rules.

    Returns
    -------
    gpd.GeoSeries
        Series of barrier geometries.

    See Also
    --------
    _parse_level_rules : Parse level rules from string.
    _create_barrier_geometry : Create geometry from intervals.

    Examples
    --------
    >>> segments = gpd.GeoDataFrame({'level_rules': [''], 'geometry': [LineString([(0,0), (1,1)])]})
    >>> barriers = _generate_barrier_geometries(segments)
    """
    level_rules = segments_gdf.get(
        "level_rules",
        pd.Series("", index=segments_gdf.index, dtype=object),
    )
    parsed_rules = level_rules.map(_parse_level_rules)
    full_barrier = parsed_rules.map(lambda value: value == "full_barrier").to_numpy()
    partial_barrier = parsed_rules.map(
        lambda value: isinstance(value, list) and bool(value)
    ).to_numpy()

    barrier_geometries = segments_gdf.geometry.to_numpy().copy()
    barrier_geometries[full_barrier] = None
    partial_positions = np.flatnonzero(partial_barrier)
    barrier_geometries[partial_positions] = [
        (
            geometry
            if geometry is None or geometry.is_empty
            else _create_barrier_geometry(geometry, parsed_rules.iloc[position])
        )
        for position, geometry in zip(
            partial_positions,
            barrier_geometries[partial_positions],
            strict=True,
        )
    ]

    return gpd.GeoSeries(
        barrier_geometries,
        index=segments_gdf.index,
        crs=segments_gdf.crs,
    )


def _parse_level_rules(level_rules_str: str) -> list[tuple[float, float]] | str:
    """
    Parse level rules string and extract barrier intervals.

    This function parses JSON-formatted level rules to extract barrier intervals
    that define restricted access areas along road segments.

    Parameters
    ----------
    level_rules_str : str
        JSON string containing level rules data.

    Returns
    -------
    list[tuple[float, float]] or str
        List of barrier intervals as (start, end) tuples, or "full_barrier" string.

    See Also
    --------
    _generate_barrier_geometries : Main function using this parser.

    Examples
    --------
    >>> rules = '[{"value": 1, "between": [0.2, 0.8]}]'
    >>> intervals = _parse_level_rules(rules)
    >>> intervals
    [(0.2, 0.8)]
    """
    if not level_rules_str:
        return []

    try:
        if isinstance(level_rules_str, list):
            rules_data = level_rules_str
        else:
            rules_data = json.loads(str(level_rules_str).replace("'", '"').replace("None", "null"))
    except (json.JSONDecodeError, AttributeError):
        return []

    barrier_intervals = []
    for rule in rules_data:
        if not isinstance(rule, dict) or rule.get("value") == 0:
            continue

        between = rule.get("between")
        if between is None:
            return "full_barrier"

        if isinstance(between, list) and len(between) == 2:
            start, end = float(between[0]), float(between[1])
            barrier_intervals.append((start, end))

    return barrier_intervals


def _create_barrier_geometry(
    geometry: LineString,
    barrier_intervals: list[tuple[float, float]],
) -> LineString | MultiLineString | None:
    """
    Create barrier geometry from intervals.

    This function creates passable geometry by removing barrier intervals
    from the original geometry, resulting in accessible road segments.

    Parameters
    ----------
    geometry : LineString
        Original road segment geometry.
    barrier_intervals : list[tuple[float, float]]
        Partial barrier intervals.

    Returns
    -------
    LineString, MultiLineString, or None
        Passable geometry after removing barriers, or None if fully blocked.

    See Also
    --------
    _calculate_passable_intervals : Calculate complement of barrier intervals.

    Examples
    --------
    >>> from shapely.geometry import LineString
    >>> geom = LineString([(0, 0), (1, 0)])
    >>> barriers = [(0.2, 0.8)]
    >>> passable = _create_barrier_geometry(geom, barriers)
    """
    # Calculate passable intervals (complement of barrier intervals)
    passable_intervals = _calculate_passable_intervals(barrier_intervals)

    if not passable_intervals:
        return None

    # Create geometry parts from passable intervals
    parts = []
    for start_pct, end_pct in passable_intervals:
        part = substring(geometry, start_pct, end_pct, normalized=True)

        if part and not part.is_empty:
            parts.append(part)

    if len(parts) == 1:
        return parts[0]
    return MultiLineString(parts)


def _calculate_passable_intervals(
    barrier_intervals: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """
    Calculate passable intervals as complement of barrier intervals.

    This function computes the passable portions of a segment by finding
    the complement of barrier intervals within the [0, 1] range.

    Parameters
    ----------
    barrier_intervals : list[tuple[float, float]]
        List of barrier intervals as (start, end) tuples.

    Returns
    -------
    list[tuple[float, float]]
        List of passable intervals as (start, end) tuples.

    See Also
    --------
    _create_barrier_geometry : Main function using this calculation.

    Examples
    --------
    >>> barriers = [(0.2, 0.4), (0.6, 0.8)]
    >>> passable = _calculate_passable_intervals(barriers)
    >>> passable
    [(0.0, 0.2), (0.4, 0.6), (0.8, 1.0)]
    """
    sorted_intervals = sorted(barrier_intervals)
    passable_intervals = []
    current = 0.0

    for start, end in sorted_intervals:
        if start > current:
            passable_intervals.append((current, start))
        current = max(current, end)

    if current < 1.0:
        passable_intervals.append((current, 1.0))

    return passable_intervals
