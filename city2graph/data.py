"""
Data Loading and Processing Module.

This module provides comprehensive functionality for loading and processing geospatial
data from various sources, with specialized support for Overture Maps data. It handles
data validation, coordinate reference system management, and geometric processing
operations commonly needed for urban network analysis.
"""

# Standard library imports
import json
import subprocess
from pathlib import Path

# Third-party imports
import geopandas as gpd
import pandas as pd
from overturemaps.core import ALL_RELEASES
from pyproj import CRS
from shapely.geometry import LineString
from shapely.geometry import MultiLineString
from shapely.geometry import Polygon
from shapely.ops import substring

# Public API definition
__all__ = ["load_overture_data", "process_overture_segments"]

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
    area: list[float] | Polygon,
    types: list[str] | None = None,
    output_dir: str = ".",
    prefix: str = "",
    save_to_file: bool = True,
    return_data: bool = True,
    release: str | None = None,
    connect_timeout: float | None = None,
    request_timeout: float | None = None,
    use_stac: bool = True,
) -> dict[str, gpd.GeoDataFrame]:
    """
    Load data from Overture Maps using the CLI tool and optionally save to GeoJSON files.

    This function downloads geospatial data from Overture Maps for a specified area
    and data types. It can save the data to GeoJSON files and/or return it as
    GeoDataFrames.

    Parameters
    ----------
    area : list[float] or Polygon
        The area of interest. Can be either a bounding box as [min_lon, min_lat, max_lon, max_lat]
        or a Polygon geometry.
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
        )
        if return_data:
            result[data_type] = gdf

    return result


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


def _prepare_area_and_bbox(area: list[float] | Polygon) -> tuple[str, Polygon | None]:
    """
    Prepare area input and convert to bbox string and clipping geometry.

    This function processes area input to create a bounding box string for API queries
    and optionally a clipping geometry for precise spatial filtering.

    Parameters
    ----------
    area : list[float] or Polygon
        The area of interest. Can be either a bounding box as [min_lon, min_lat, max_lon, max_lat]
        or a Polygon geometry.

    Returns
    -------
    tuple[str, Polygon or None]
        Tuple containing bbox string and optional clipping geometry.

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
    if isinstance(area, Polygon):
        # Convert to WGS84 if needed
        area_wgs84 = (
            area.to_crs(WGS84_CRS) if hasattr(area, "crs") and area.crs != WGS84_CRS else area
        )
        bbox_str = ",".join(str(round(c, 10)) for c in area_wgs84.bounds)
        clip_geom = area_wgs84
    else:
        bbox_str = ",".join(str(float(b)) for b in area)
        clip_geom = None

    return bbox_str, clip_geom


def _download_and_process_type(
    data_type: str,
    bbox_str: str,
    output_dir: str,
    prefix: str,
    save_to_file: bool,
    return_data: bool,
    clip_geom: Polygon | None,
    release: str | None = None,
    connect_timeout: float | None = None,
    request_timeout: float | None = None,
    use_stac: bool = True,
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
    clip_geom : Polygon or None
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

    subprocess.run(cmd, check=True, capture_output=not save_to_file, text=True)

    if not return_data:
        return gpd.GeoDataFrame(geometry=[], crs=WGS84_CRS)

    # Load and clip data if needed
    gdf = (
        gpd.read_file(output_path)
        if save_to_file and output_path.exists()
        else gpd.GeoDataFrame(geometry=[], crs=WGS84_CRS)
    )

    if clip_geom is not None and not gdf.empty:
        clip_gdf = gpd.GeoDataFrame(geometry=[clip_geom], crs=WGS84_CRS)
        # Handle CRS conversion and clipping safely, including for mock objects in tests

        # Only proceed if both CRS are real CRS objects or strings (but not Mock objects)
        clip_crs_valid = isinstance(clip_gdf.crs, (CRS, str, type(None)))
        gdf_crs_valid = isinstance(gdf.crs, (CRS, str, type(None)))

        if clip_crs_valid and gdf_crs_valid:
            if clip_gdf.crs != gdf.crs:
                clip_gdf = clip_gdf.to_crs(gdf.crs)
            gdf = gpd.clip(gdf, clip_gdf)

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

    See Also
    --------
    _extract_connector_positions : Extract connector positions from segments.

    Examples
    --------
    >>> segments = gpd.GeoDataFrame({'geometry': [LineString([(0,0), (1,1)])]})
    >>> connectors = gpd.GeoDataFrame({'id': ['c1']})
    >>> split_segments = _split_segments_at_connectors(segments, connectors)
    """
    if connectors_gdf is None or connectors_gdf.empty:
        return segments_gdf

    valid_connector_ids = set(connectors_gdf["id"])
    split_segments = []

    for _, segment in segments_gdf.iterrows():
        positions = _extract_connector_positions(segment, valid_connector_ids)
        split_parts = _create_segment_splits(segment, positions)
        split_segments.extend(split_parts)

    return gpd.GeoDataFrame(split_segments, crs=segments_gdf.crs).reset_index(drop=True)


def _extract_connector_positions(segment: pd.Series, valid_connector_ids: set[str]) -> list[float]:
    """
    Extract valid connector positions from a segment.

    This function parses connector information from a segment and returns
    the positions of valid connectors along the segment.

    Parameters
    ----------
    segment : pd.Series
        Series containing segment data with connector information.
    valid_connector_ids : set[str]
        Set of valid connector IDs to filter by.

    Returns
    -------
    list[float]
        List of connector positions along the segment (0.0 to 1.0).

    See Also
    --------
    _split_segments_at_connectors : Main function using this helper.

    Examples
    --------
    >>> segment = pd.Series({'connectors': '[{"connector_id": "c1", "at": 0.5}]'})
    >>> valid_ids = {'c1'}
    >>> positions = _extract_connector_positions(segment, valid_ids)
    [0.0, 0.5, 1.0]
    """
    connectors_val = segment.get("connectors", "")
    if not connectors_val:
        return [0.0, 1.0]

    # Parse connector data safely
    if isinstance(connectors_val, list):
        connectors_data = connectors_val
    else:
        try:
            connectors_data = json.loads(
                str(connectors_val).replace("'", '"').replace("None", "null")
            )
        except (json.JSONDecodeError, AttributeError):
            connectors_data = []

    # Ensure connectors_data is a list
    if not isinstance(connectors_data, list):
        connectors_data = [connectors_data] if connectors_data else []

    # Extract positions from valid connectors
    positions = [
        float(conn["at"])
        for conn in connectors_data
        if (
            isinstance(conn, dict)
            and conn.get("connector_id") in valid_connector_ids
            and "at" in conn
        )
    ]

    # Return sorted unique positions with start and end
    return sorted({0.0, *positions, 1.0})


def _create_segment_splits(segment: pd.Series, positions: list[float]) -> list[pd.Series]:
    """
    Create split segments from position list.

    This function takes a segment and a list of split positions and creates
    multiple segment parts based on those positions.

    Parameters
    ----------
    segment : pd.Series
        Original segment to be split.
    positions : list[float]
        List of positions along the segment where splits should occur.

    Returns
    -------
    list[pd.Series]
        List of split segment parts.

    See Also
    --------
    _split_segments_at_connectors : Main function using this helper.

    Examples
    --------
    >>> segment = pd.Series({'geometry': LineString([(0,0), (1,1)])})
    >>> positions = [0.0, 0.5, 1.0]
    >>> splits = _create_segment_splits(segment, positions)
    """
    if len(positions) <= 2:
        return [segment]

    original_id = segment.get("id", segment.name)
    split_parts = []

    for i in range(len(positions) - 1):
        start_pct, end_pct = positions[i], positions[i + 1]

        part_geom = substring(segment.geometry, start_pct, end_pct, normalized=True)

        if part_geom and not part_geom.is_empty:
            new_segment = segment.copy()
            new_segment.geometry = part_geom
            new_segment["split_from"] = start_pct
            new_segment["split_to"] = end_pct
            new_segment["id"] = f"{original_id}_{i + 1}" if len(positions) > 2 else original_id
            split_parts.append(new_segment)

    return split_parts


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
    # Extract all endpoints
    endpoints_data = []
    for idx, geom in segments_gdf.geometry.items():
        if isinstance(geom, LineString) and len(geom.coords) >= 2:
            coords = list(geom.coords)
            endpoints_data.append((idx, "start", coords[0]))
            endpoints_data.append((idx, "end", coords[-1]))

    # Create DataFrame for clustering
    endpoints_df = pd.DataFrame(
        [
            {"seg_id": idx, "pos": pos, "x": coord[0], "y": coord[1]}
            for idx, pos, coord in endpoints_data
        ],
    )

    # Perform spatial clustering using binning
    endpoints_df["bin_x"] = (endpoints_df["x"] / threshold).round().astype(int)
    endpoints_df["bin_y"] = (endpoints_df["y"] / threshold).round().astype(int)

    # Calculate cluster centroids
    centroids = endpoints_df.groupby(["bin_x", "bin_y"])[["x", "y"]].mean()
    endpoints_df = endpoints_df.merge(centroids, on=["bin_x", "bin_y"], suffixes=("", "_new"))

    # Create coordinate lookup
    coord_lookup = {
        (row["seg_id"], row["pos"]): (row["x_new"], row["y_new"])
        for _, row in endpoints_df.iterrows()
    }

    # Update segment geometries
    result_gdf = segments_gdf.copy()
    for idx, row in result_gdf.iterrows():
        if isinstance(row.geometry, LineString) and len(row.geometry.coords) >= 2:
            coords = list(row.geometry.coords)
            start_coord = coord_lookup.get((idx, "start"), coords[0])
            end_coord = coord_lookup.get((idx, "end"), coords[-1])
            result_gdf.loc[idx, "geometry"] = LineString([start_coord, *coords[1:-1], end_coord])

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
    barrier_geometries = []

    for _, row in segments_gdf.iterrows():
        level_rules_str = row.get("level_rules", "")
        geometry = row.geometry

        # Parse level rules
        barrier_intervals = _parse_level_rules(level_rules_str)

        # Generate barrier geometry
        barrier_geom = _create_barrier_geometry(geometry, barrier_intervals)
        barrier_geometries.append(barrier_geom)

    return gpd.GeoSeries(barrier_geometries, crs=segments_gdf.crs)


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
    barrier_intervals: list[tuple[float, float]] | str,
) -> LineString | MultiLineString | None:
    """
    Create barrier geometry from intervals.

    This function creates passable geometry by removing barrier intervals
    from the original geometry, resulting in accessible road segments.

    Parameters
    ----------
    geometry : LineString
        Original road segment geometry.
    barrier_intervals : list[tuple[float, float]] or str
        Barrier intervals or "full_barrier" indicator.

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
    if barrier_intervals == "full_barrier":
        return None

    if not barrier_intervals:
        return geometry

    # Ensure barrier_intervals is a list of tuples
    assert isinstance(barrier_intervals, list)

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
