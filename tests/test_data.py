"""Tests for the data module - refactored version focusing on public API only."""

import importlib.util
import subprocess
from pathlib import Path
from typing import Optional
from unittest.mock import Mock
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
import pytest
import requests
from requests.adapters import HTTPAdapter
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import Polygon

from tests.helpers import make_connectors_gdf
from tests.helpers import make_segments_gdf

spec = importlib.util.spec_from_file_location("data_module", "city2graph/data.py")
assert spec is not None
data_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(data_module)
VALID_OVERTURE_TYPES = data_module.VALID_OVERTURE_TYPES
WGS84_CRS = data_module.WGS84_CRS
load_overture_data = data_module.load_overture_data
process_overture_segments = data_module.process_overture_segments
_requests_session = data_module._requests_session
load_transportation_networks_data = data_module.load_transportation_networks_data
_get_available_transportation_networks = data_module._get_available_transportation_networks
_get_transportation_networks_data = data_module._get_transportation_networks_data
_strip_metadata = data_module._strip_metadata
_parse_tntp_from_lines = data_module._parse_tntp_from_lines
_parse_network_file = data_module._parse_network_file
_parse_flow_file = data_module._parse_flow_file
_parse_trips_file = data_module._parse_trips_file
_parse_nodes_file = data_module._parse_nodes_file
TransportationNetworkData = data_module.TransportationNetworkData


# Tests for constants and basic functionality
def test_valid_overture_types_constant() -> None:
    """Test that VALID_OVERTURE_TYPES contains expected types."""
    expected_types = {
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
    assert expected_types == VALID_OVERTURE_TYPES


def test_wgs84_crs_constant() -> None:
    """Test that WGS84_CRS is correctly defined."""
    assert WGS84_CRS == "EPSG:4326"


# Tests for load_overture_data function
@patch("city2graph.data.subprocess.run")
@patch("city2graph.data.gpd.read_file")
@patch("city2graph.data.Path.mkdir")
def test_load_overture_data_with_bbox_list(
    mock_mkdir: Mock,
    mock_read_file: Mock,
    mock_subprocess: Mock,
    test_bbox: list[float],
) -> None:
    """Test load_overture_data with bounding box as list."""
    # Setup
    types = ["building", "segment"]

    # Mock GeoDataFrame
    mock_gdf = Mock(spec=gpd.GeoDataFrame)
    mock_gdf.empty = False
    mock_read_file.return_value = mock_gdf

    # Execute
    result = load_overture_data(test_bbox, types=types, output_dir="test_dir")

    # Verify
    assert len(result) == 2
    assert "building" in result
    assert "segment" in result
    mock_mkdir.assert_called_once()
    assert mock_subprocess.call_count == 2


@patch("city2graph.data.subprocess.run")
@patch("city2graph.data.gpd.read_file")
@patch("city2graph.data.Path.mkdir")
def test_load_overture_data_with_polygon(
    mock_mkdir: Mock,
    mock_read_file: Mock,
    mock_subprocess: Mock,
    test_polygon: Polygon,
) -> None:
    """Test load_overture_data with Polygon area."""
    # Setup
    mock_gdf = Mock(spec=gpd.GeoDataFrame)
    mock_gdf.empty = False
    mock_read_file.return_value = mock_gdf

    # Execute
    result = load_overture_data(test_polygon, types=["building"])

    # Verify
    assert "building" in result
    mock_subprocess.assert_called_once()
    mock_mkdir.assert_called()  # Verify directory creation


@patch("city2graph.data.subprocess.run")
@patch("city2graph.data.gpd.read_file")
@patch("city2graph.data.gpd.clip")
@patch("city2graph.data.Path.exists")
def test_load_overture_data_with_polygon_clipping(
    mock_exists: Mock,
    mock_clip: Mock,
    mock_read_file: Mock,
    mock_subprocess: Mock,
    test_polygon: Polygon,
) -> None:
    """Test that polygon areas are properly clipped."""
    # Setup
    mock_gdf = Mock(spec=gpd.GeoDataFrame)
    mock_gdf.empty = False
    mock_gdf.crs = "EPSG:3857"
    mock_read_file.return_value = mock_gdf
    mock_exists.return_value = True

    mock_clipped_gdf = Mock(spec=gpd.GeoDataFrame)
    mock_clip.return_value = mock_clipped_gdf

    # Execute
    result = load_overture_data(test_polygon, types=["building"])

    # Verify
    mock_clip.assert_called_once()
    mock_subprocess.assert_called()  # Verify subprocess was called
    assert result["building"] == mock_clipped_gdf


def test_load_overture_data_invalid_types(test_bbox: list[float]) -> None:
    """Test that invalid types raise ValueError."""
    invalid_types = ["building", "invalid_type", "another_invalid"]

    with pytest.raises(ValueError, match="Invalid types: \\['invalid_type', 'another_invalid'\\]"):
        load_overture_data(test_bbox, types=invalid_types)


@patch("city2graph.data.subprocess.run")
@patch("city2graph.data.gpd.read_file")
def test_load_overture_data_default_types(
    mock_read_file: Mock,
    mock_subprocess: Mock,
    test_bbox: list[float],
) -> None:
    """Test that all valid types are used when types=None."""
    mock_gdf = Mock(spec=gpd.GeoDataFrame)
    mock_gdf.empty = False
    mock_read_file.return_value = mock_gdf

    result = load_overture_data(test_bbox, types=None, save_to_file=False)

    # Should have all valid types
    assert len(result) == len(VALID_OVERTURE_TYPES)
    for data_type in VALID_OVERTURE_TYPES:
        assert data_type in result

    # Verify subprocess was called for each type
    assert mock_subprocess.call_count == len(VALID_OVERTURE_TYPES)


@patch("city2graph.data.subprocess.run")
def test_load_overture_data_save_to_file_false(
    mock_subprocess: Mock,
    test_bbox: list[float],
) -> None:
    """Test load_overture_data with save_to_file=False."""
    result = load_overture_data(
        test_bbox,
        types=["building"],
        save_to_file=False,
        return_data=False,
    )

    # Should return empty dict when return_data=False
    assert result == {}

    # Subprocess should be called without output file
    mock_subprocess.assert_called_once()
    args = mock_subprocess.call_args[0][0]
    assert "-o" not in args


@patch("city2graph.data.subprocess.run")
@patch("city2graph.data.gpd.read_file")
def test_load_overture_data_return_data_false(
    mock_read_file: Mock,
    mock_subprocess: Mock,
    test_bbox: list[float],
) -> None:
    """Test load_overture_data with return_data=False."""
    result = load_overture_data(test_bbox, types=["building"], return_data=False)

    assert result == {}
    mock_read_file.assert_not_called()
    mock_subprocess.assert_called()  # Should still call subprocess to generate files


@patch("city2graph.data.subprocess.run")
def test_load_overture_data_subprocess_error(mock_subprocess: Mock, test_bbox: list[float]) -> None:
    """Test that subprocess errors are propagated."""
    mock_subprocess.side_effect = subprocess.CalledProcessError(1, "overturemaps")

    with pytest.raises(subprocess.CalledProcessError):
        load_overture_data(test_bbox, types=["building"])


@patch("city2graph.data.subprocess.run")
@patch("city2graph.data.gpd.read_file")
def test_load_overture_data_with_prefix(
    mock_read_file: Mock,
    mock_subprocess: Mock,
    test_bbox: list[float],
) -> None:
    """Test load_overture_data with filename prefix."""
    prefix = "test_prefix_"

    mock_gdf = Mock(spec=gpd.GeoDataFrame)
    mock_gdf.empty = False
    mock_read_file.return_value = mock_gdf

    load_overture_data(test_bbox, types=["building"], prefix=prefix)

    # Check that the output path includes the prefix
    args = mock_subprocess.call_args[0][0]
    output_index = args.index("-o") + 1
    output_path = args[output_index]
    assert prefix in output_path


@patch("city2graph.data.subprocess.run")
@patch("city2graph.data.gpd.read_file")
@patch("city2graph.data.Path.exists")
def test_load_overture_data_file_not_exists(
    mock_exists: Mock,
    mock_read_file: Mock,
    mock_subprocess: Mock,
    test_bbox: list[float],
) -> None:
    """Test behavior when output file doesn't exist."""
    mock_exists.return_value = False

    result = load_overture_data(test_bbox, types=["building"])

    # Should return empty GeoDataFrame when file doesn't exist
    mock_read_file.assert_not_called()
    assert "building" in result
    mock_subprocess.assert_called()  # Should still attempt to generate files


# Tests for process_overture_segments function
def test_process_overture_segments_empty_input() -> None:
    """Test process_overture_segments with empty GeoDataFrame."""
    empty_gdf = gpd.GeoDataFrame(geometry=[], crs=WGS84_CRS)
    result = process_overture_segments(empty_gdf)

    assert result.empty
    assert result.crs == WGS84_CRS


def test_process_overture_segments_basic() -> None:
    """Test basic functionality of process_overture_segments with local data."""
    # Create a minimal segments GeoDataFrame locally to avoid fixture dependency
    segments_gdf = make_segments_gdf(
        ids=["s1", "s2"],
        geoms_or_coords=[[(0, 0), (1, 0)], [(1, 0), (2, 0)]],
        level_rules="",
        crs=WGS84_CRS,
    )

    result = process_overture_segments(segments_gdf, get_barriers=False)

    # Should have length column
    assert "length" in result.columns
    assert all(result["length"] > 0)

    # Should preserve original data
    assert len(result) >= len(segments_gdf)
    assert "id" in result.columns


def test_process_overture_segments_with_connectors() -> None:
    """Test process_overture_segments with connectors."""
    connectors_data = '[{"connector_id": "c1", "at": 0.25}, {"connector_id": "c2", "at": 0.75}]'
    segments_gdf = make_segments_gdf(
        ids=["seg1"],
        geoms_or_coords=[[(0, 0), (4, 0)]],
        connectors=connectors_data,
        level_rules="",
        crs=WGS84_CRS,
    )
    connectors_gdf = make_connectors_gdf(ids=["c1", "c2"], coords=[(1, 0), (3, 0)], crs=WGS84_CRS)

    result = process_overture_segments(
        segments_gdf,
        connectors_gdf=connectors_gdf,
        get_barriers=False,
    )

    # Should have split columns for segments that were split
    split_segments = result[result["id"].str.contains("_")]
    if not split_segments.empty:
        assert "split_from" in result.columns
        assert "split_to" in result.columns


def test_process_overture_segments_with_barriers() -> None:
    """Test process_overture_segments with barrier generation."""
    segments_gdf = make_segments_gdf(
        ids=["seg1"],
        geoms_or_coords=[[(0, 0), (1, 1)]],
        level_rules="",
        crs=WGS84_CRS,
    )
    result = process_overture_segments(segments_gdf, get_barriers=True)

    # Should have barrier_geometry column
    assert "barrier_geometry" in result.columns


def test_process_overture_segments_missing_level_rules() -> None:
    """Test process_overture_segments with missing level_rules column."""
    segments_gdf = make_segments_gdf(
        ids=["seg1"],
        geoms_or_coords=[[(0, 0), (1, 1)]],
        level_rules=None,
        crs=WGS84_CRS,
    )

    # This should work - the function should handle missing level_rules gracefully
    result = process_overture_segments(segments_gdf)
    assert "level_rules" in result.columns
    assert result["level_rules"].iloc[0] == ""


def test_process_overture_segments_with_threshold() -> None:
    """Test process_overture_segments with custom threshold."""
    connectors_data = '[{"connector_id": "c1", "at": 0.5}]'
    segments_gdf = make_segments_gdf(
        ids=["seg1", "seg2"],
        geoms_or_coords=[[(0, 0), (1, 1)], [(1.1, 1.1), (2, 2)]],
        connectors=[connectors_data, connectors_data],
        level_rules="",
        crs=WGS84_CRS,
    )
    connectors_gdf = make_connectors_gdf(ids=["c1"], coords=[(1, 1)], crs=WGS84_CRS)

    result = process_overture_segments(
        segments_gdf,
        connectors_gdf=connectors_gdf,
        threshold=2.0,
    )

    # Should process without errors
    assert "length" in result.columns


def test_process_overture_segments_no_connectors() -> None:
    """Test process_overture_segments with None connectors."""
    segments_gdf = make_segments_gdf(
        ids=["s1", "s2"],
        geoms_or_coords=[[(0, 0), (1, 0)], [(1, 0), (2, 0)]],
        level_rules="",
        crs=WGS84_CRS,
    )
    result = process_overture_segments(segments_gdf, connectors_gdf=None)

    # Should not perform endpoint clustering
    assert len(result) == len(segments_gdf)


def test_process_overture_segments_empty_connectors() -> None:
    """Test process_overture_segments with empty connectors GeoDataFrame."""
    segments_gdf = make_segments_gdf(
        ids=["s1", "s2"],
        geoms_or_coords=[[(0, 0), (1, 0)], [(1, 0), (2, 0)]],
        level_rules="",
        crs=WGS84_CRS,
    )
    empty_connectors = make_connectors_gdf(ids=[], coords=[], crs=WGS84_CRS)
    result = process_overture_segments(segments_gdf, connectors_gdf=empty_connectors)

    # Should not perform splitting or clustering
    assert len(result) == len(segments_gdf)


def test_process_overture_segments_invalid_connector_data() -> None:
    """Test process_overture_segments with invalid connector JSON."""
    segments_gdf = make_segments_gdf(
        ids=["seg1"],
        geoms_or_coords=[[(0, 0), (1, 1)]],
        connectors="invalid_json",
        level_rules="",
        crs=WGS84_CRS,
    )

    result = process_overture_segments(segments_gdf)

    # Should handle invalid JSON gracefully
    assert len(result) == 1


def test_process_overture_segments_malformed_connectors() -> None:
    """Test process_overture_segments with malformed connector data."""
    segments_gdf = make_segments_gdf(
        ids=["seg1"],
        geoms_or_coords=[[(0, 0), (1, 1)]],
        connectors='{"invalid": "structure"}',
        level_rules="",
        crs=WGS84_CRS,
    )
    connectors_gdf = make_connectors_gdf(ids=["x"], coords=[(0.5, 0.5)], crs=WGS84_CRS)
    result = process_overture_segments(segments_gdf, connectors_gdf=connectors_gdf)

    # Should handle malformed data gracefully
    assert len(result) == 1


def test_process_overture_segments_invalid_level_rules() -> None:
    """Test process_overture_segments with invalid level rules JSON."""
    segments_gdf = make_segments_gdf(
        ids=["seg1"],
        geoms_or_coords=[[(0, 0), (1, 1)]],
        level_rules="invalid_json",
        crs=WGS84_CRS,
    )

    result = process_overture_segments(segments_gdf, get_barriers=True)

    # Should handle invalid JSON gracefully
    assert "barrier_geometry" in result.columns


def test_process_overture_segments_complex_level_rules() -> None:
    """Test process_overture_segments with complex level rules."""
    level_rules = '[{"value": 1, "between": [0.1, 0.3]}, {"value": 1, "between": [0.7, 0.9]}]'
    segments_gdf = make_segments_gdf(
        ids=["seg1"],
        geoms_or_coords=[[(0, 0), (1, 1)]],
        level_rules=level_rules,
        crs=WGS84_CRS,
    )

    result = process_overture_segments(segments_gdf, get_barriers=True)

    # Should create barrier geometry
    assert "barrier_geometry" in result.columns
    assert result["barrier_geometry"].iloc[0] is not None


def test_process_overture_segments_full_barrier() -> None:
    """Test process_overture_segments with full barrier level rules."""
    level_rules = '[{"value": 1}]'  # No "between" means full barrier
    segments_gdf = make_segments_gdf(
        ids=["seg1"],
        geoms_or_coords=[[(0, 0), (1, 1)]],
        level_rules=level_rules,
        crs=WGS84_CRS,
    )

    result = process_overture_segments(segments_gdf, get_barriers=True)

    # Should create None barrier geometry for full barriers
    assert "barrier_geometry" in result.columns
    assert result["barrier_geometry"].iloc[0] is None


def test_process_overture_segments_zero_value_rules() -> None:
    """Test process_overture_segments with zero value level rules."""
    level_rules = '[{"value": 0, "between": [0.2, 0.8]}]'
    segments_gdf = make_segments_gdf(
        ids=["seg1"],
        geoms_or_coords=[[(0, 0), (1, 1)]],
        level_rules=level_rules,
        crs=WGS84_CRS,
    )

    result = process_overture_segments(segments_gdf, get_barriers=True)

    # Zero value rules should be ignored
    assert "barrier_geometry" in result.columns
    # Should return original geometry since no barriers
    barrier_geom = result["barrier_geometry"].iloc[0]
    assert barrier_geom is not None


def test_process_overture_segments_segment_splitting() -> None:
    """Test that segments are properly split at connector positions."""
    connectors_data = (
        '[{"connector_id": "conn1", "at": 0.0}, '
        '{"connector_id": "conn2", "at": 0.5}, '
        '{"connector_id": "conn3", "at": 1.0}]'
    )

    segments_gdf = make_segments_gdf(
        ids=["seg1"],
        geoms_or_coords=[[(0, 0), (2, 2)]],
        connectors=connectors_data,
        level_rules="",
        crs=WGS84_CRS,
    )

    connectors_gdf = make_connectors_gdf(
        ids=["conn1", "conn2", "conn3"],
        coords=[(0, 0), (1, 1), (2, 2)],
        crs=WGS84_CRS,
    )

    result = process_overture_segments(segments_gdf, connectors_gdf=connectors_gdf)

    # Should create multiple segments
    assert len(result) > 1
    # Should have split information
    split_segments = result[result["id"].str.contains("_")]
    assert not split_segments.empty


def test_process_overture_segments_endpoint_clustering() -> None:
    """Test endpoint clustering functionality."""
    # Create segments with nearby endpoints
    segments_gdf = make_segments_gdf(
        ids=["seg1", "seg2"],
        geoms_or_coords=[[(0, 0), (1, 1)], [(1.1, 1.1), (2, 2)]],
        level_rules="",
        crs=WGS84_CRS,
    )

    connectors_gdf = make_connectors_gdf(ids=["conn1"], coords=[(1, 1)], crs=WGS84_CRS)

    result = process_overture_segments(
        segments_gdf,
        connectors_gdf=connectors_gdf,
        threshold=0.5,  # Large enough to cluster nearby points
    )

    # Should process without errors
    assert len(result) >= len(segments_gdf)


def test_process_overture_segments_level_rules_handling() -> None:
    """Test process_overture_segments level_rules column handling."""
    # Test with None values in level_rules
    segments_gdf = make_segments_gdf(
        ids=["seg1"],
        geoms_or_coords=[[(0, 0), (1, 1)]],
        level_rules=[None],
        crs=WGS84_CRS,
    )

    result = process_overture_segments(segments_gdf)
    assert result["level_rules"].iloc[0] == ""


# Integration tests
def test_load_and_process_integration() -> None:
    """Test integration between load_overture_data and process_overture_segments."""
    # Create mock data that resembles real Overture data
    segments_gdf = make_segments_gdf(
        ids=["seg1", "seg2"],
        geoms_or_coords=[[(0, 0), (1, 1)], [(1, 1), (2, 2)]],
        connectors=[
            '[{"connector_id": "conn1", "at": 0.0}]',
            '[{"connector_id": "conn2", "at": 1.0}]',
        ],
        level_rules="",
        crs=WGS84_CRS,
    )

    # Process the segments
    result = process_overture_segments(segments_gdf)

    # Should have all expected columns
    expected_columns = ["id", "connectors", "level_rules", "geometry", "length", "barrier_geometry"]
    for col in expected_columns:
        assert col in result.columns


@patch("city2graph.data.subprocess.run")
@patch("city2graph.data.gpd.read_file")
@patch("city2graph.data.Path.exists")
def test_real_world_scenario_simulation(
    mock_exists: Mock,
    mock_read_file: Mock,
    mock_subprocess: Mock,
    test_bbox: list[float],
) -> None:
    """Test a scenario that simulates real-world usage without external fixtures."""
    # Build local realistic-like GeoDataFrames
    segments_gdf = make_segments_gdf(
        ids=["seg1", "seg2"],
        geoms_or_coords=[[(0, 0), (1, 1)], [(1, 1), (2, 2)]],
        connectors=[
            '[{"connector_id": "conn1", "at": 0.25}]',
            '[{"connector_id": "conn2", "at": 0.75}]',
        ],
        level_rules="",
        crs=WGS84_CRS,
    )
    connectors_gdf = make_connectors_gdf(
        ids=["conn1", "conn2"],
        coords=[(0.25, 0.25), (1.75, 1.75)],
        crs=WGS84_CRS,
    )

    # Mock the file reading to return our local data
    mock_read_file.side_effect = [segments_gdf, connectors_gdf]
    mock_exists.return_value = True

    # Simulate loading data
    data = load_overture_data(test_bbox, types=["segment", "connector"])

    # Process the segments
    processed_segments = process_overture_segments(
        data["segment"],
        connectors_gdf=data["connector"],
    )
    assert not processed_segments.empty
    assert "barrier_geometry" in processed_segments.columns

    # Verify mocks were called appropriately
    mock_subprocess.assert_called()
    assert "length" in processed_segments.columns


# Additional edge case tests for comprehensive coverage
def test_process_overture_segments_with_non_dict_connector() -> None:
    """Test process_overture_segments with non-dict connector data."""
    segments_gdf = make_segments_gdf(
        ids=["seg1"],
        geoms_or_coords=[[(0, 0), (1, 1)]],
        connectors='["not_a_dict"]',
        level_rules="",
        crs=WGS84_CRS,
    )

    result = process_overture_segments(segments_gdf)
    # Should handle non-dict data gracefully
    assert len(result) == 1


def test_process_overture_segments_with_non_dict_level_rule() -> None:
    """Test process_overture_segments with non-dict level rule data."""
    segments_gdf = make_segments_gdf(
        ids=["seg1"],
        geoms_or_coords=[[(0, 0), (1, 1)]],
        level_rules='["not_a_dict"]',
        crs=WGS84_CRS,
    )

    result = process_overture_segments(segments_gdf, get_barriers=True)
    # Should handle non-dict data gracefully
    assert "barrier_geometry" in result.columns


def test_process_overture_segments_with_short_between_array() -> None:
    """Test process_overture_segments with short between array in level rules."""
    level_rules = '[{"value": 1, "between": [0.5]}]'  # Only one element
    segments_gdf = make_segments_gdf(
        ids=["seg1"],
        geoms_or_coords=[[(0, 0), (1, 1)]],
        level_rules=level_rules,
        crs=WGS84_CRS,
    )

    result = process_overture_segments(segments_gdf, get_barriers=True)
    # Should handle short between array gracefully
    assert "barrier_geometry" in result.columns


def test_process_overture_segments_with_empty_geometry() -> None:
    """Test process_overture_segments with empty geometry."""
    # Create an empty LineString
    empty_geom = LineString()
    segments_gdf = make_segments_gdf(
        ids=["seg1"],
        geoms_or_coords=[empty_geom],
        level_rules="",
        crs=WGS84_CRS,
    )

    result = process_overture_segments(segments_gdf, get_barriers=True)
    # Should handle empty geometry gracefully
    assert "barrier_geometry" in result.columns


def test_process_overture_segments_with_overlapping_barriers() -> None:
    """Test process_overture_segments with overlapping barrier intervals."""
    level_rules = '[{"value": 1, "between": [0.1, 0.5]}, {"value": 1, "between": [0.3, 0.7]}]'
    segments_gdf = make_segments_gdf(
        ids=["seg1"],
        geoms_or_coords=[[(0, 0), (4, 4)]],
        level_rules=level_rules,
        crs=WGS84_CRS,
    )

    result = process_overture_segments(segments_gdf, get_barriers=True)
    # Should handle overlapping barriers correctly
    assert "barrier_geometry" in result.columns


def test_process_overture_segments_with_touching_barriers() -> None:
    """Test process_overture_segments with touching barrier intervals."""
    level_rules = '[{"value": 1, "between": [0.0, 0.3]}, {"value": 1, "between": [0.3, 0.6]}]'
    segments_gdf = make_segments_gdf(
        ids=["seg1"],
        geoms_or_coords=[[(0, 0), (4, 4)]],
        level_rules=level_rules,
        crs=WGS84_CRS,
    )

    result = process_overture_segments(segments_gdf, get_barriers=True)
    # Should handle touching barriers correctly
    assert "barrier_geometry" in result.columns


def test_process_overture_segments_with_full_coverage_barriers() -> None:
    """Test process_overture_segments with barriers covering full segment."""
    level_rules = '[{"value": 1, "between": [0.0, 1.0]}]'
    segments_gdf = make_segments_gdf(
        ids=["seg1"],
        geoms_or_coords=[[(0, 0), (4, 4)]],
        level_rules=level_rules,
        crs=WGS84_CRS,
    )

    result = process_overture_segments(segments_gdf, get_barriers=True)
    # Should return None for full coverage barriers
    assert "barrier_geometry" in result.columns
    assert result["barrier_geometry"].iloc[0] is None


@patch("city2graph.data.subprocess.run")
@patch("city2graph.data.gpd.read_file")
@patch("city2graph.data.Path.mkdir")
def test_load_overture_data_comprehensive_all_types(
    mock_mkdir: Mock,
    mock_read_file: Mock,
    mock_subprocess: Mock,
) -> None:
    """Test load_overture_data with all types (types=None)."""
    # mock_mkdir is set up by @patch decorator but not called in this test with save_to_file=False
    _ = mock_mkdir  # Acknowledge the parameter

    # Mock GeoDataFrame
    mock_gdf = Mock(spec=gpd.GeoDataFrame)
    mock_gdf.empty = False
    mock_read_file.return_value = mock_gdf

    # Test with all types (types=None)
    bbox = [-74.01, 40.70, -73.99, 40.72]
    result = load_overture_data(bbox, types=None, save_to_file=False)

    # Should call subprocess for all valid types
    assert mock_subprocess.call_count == len(VALID_OVERTURE_TYPES)
    assert len(result) == len(VALID_OVERTURE_TYPES)


def test_process_overture_segments_with_non_linestring_endpoints() -> None:
    """Test endpoint clustering with non-LineString geometries."""
    # Mix LineString and Point geometries
    segments_gdf = make_segments_gdf(
        ids=["seg1", "seg2"],
        geoms_or_coords=[LineString([(0, 0), (1, 1)]), Point(2, 2)],
        level_rules="",
        crs=WGS84_CRS,
    )

    connectors_gdf = make_connectors_gdf(ids=["conn1"], coords=[(1, 1)], crs=WGS84_CRS)

    result = process_overture_segments(
        segments_gdf,
        connectors_gdf=connectors_gdf,
        threshold=1.0,
    )

    # Should process without errors
    assert len(result) == len(segments_gdf)


def test_process_overture_segments_with_short_linestring() -> None:
    """Test endpoint clustering with LineString having insufficient coordinates."""
    # Create a degenerate LineString (same start and end point)
    invalid_geom = LineString([(0, 0), (0, 0)])  # Degenerate but valid
    segments_gdf = make_segments_gdf(
        ids=["seg1", "seg2"],
        geoms_or_coords=[LineString([(0, 0), (1, 1)]), invalid_geom],
        level_rules="",
        crs=WGS84_CRS,
    )

    connectors_gdf = make_connectors_gdf(ids=["conn1"], coords=[(1, 1)], crs=WGS84_CRS)

    result = process_overture_segments(
        segments_gdf,
        connectors_gdf=connectors_gdf,
        threshold=1.0,
    )

    # Should process without errors
    assert len(result) == len(segments_gdf)


def test_requests_session_default_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Session should include retry adapter and custom user agent."""
    session = _requests_session()
    try:
        assert session.headers["User-Agent"] == "tntp-loader/1.0"
        assert "Authorization" not in session.headers
        adapter = session.adapters["https://"]
        assert isinstance(adapter, HTTPAdapter)
        assert adapter.max_retries.total == 5
    finally:
        session.close()

def test_load_transportation_networks_data_happy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify helper wiring and return conversion for transportation networks loader."""
    mock_session = Mock()
    network_df = pd.DataFrame({"capacity": [100]})
    with (
        patch.object(data_module, "_requests_session", return_value=mock_session) as mock_session_factory,
        patch.object(data_module, "_get_available_transportation_networks", return_value=["TestNet"]) as mock_available,
        patch.object(data_module, "_get_transportation_networks_data") as mock_get_data,
    ):
        mock_get_data.return_value = TransportationNetworkData(
            network=network_df,
            trips=None,
            nodes=None,
            flow=None,
        )

        result = load_transportation_networks_data(
            "TestNet",
            load_trips=False,
            load_nodes=False,
            load_flow=False,
        )

        mock_session_factory.assert_called_once()
        mock_available.assert_called_once_with(mock_session)
        mock_get_data.assert_called_once_with(
            session=mock_session,
            network_name="TestNet",
            output_dir=None,
            save_to_file=False,
            load_network=True,
            load_trips=False,
            load_nodes=False,
            load_flow=False,
            download_if_missing=True,
            best_effort=True,
        )

    assert set(result.keys()) == {"network", "trips", "nodes", "flow"}
    pd.testing.assert_frame_equal(result["network"], network_df)
    assert result["trips"] is None


def test_load_transportation_networks_data_unknown_network() -> None:
    """Unknown network names should raise early."""
    mock_session = Mock()
    with (
        patch.object(data_module, "_requests_session", return_value=mock_session),
        patch.object(data_module, "_get_available_transportation_networks", return_value=["OtherNet"]),
    ):
        with pytest.raises(ValueError, match="Network 'MissingNet' not found"):
            load_transportation_networks_data("MissingNet")


def test_load_transportation_networks_data_requires_output_dir() -> None:
    """Saving to disk without output directory is invalid."""
    mock_session = Mock()
    with (
        patch.object(data_module, "_requests_session", return_value=mock_session),
        patch.object(data_module, "_get_available_transportation_networks", return_value=["TestNet"]),
    ):
        with pytest.raises(ValueError, match="output_dir must be specified"):
            load_transportation_networks_data("TestNet", save_to_file=True, output_dir=None)


class _FakeResponse:
    """Minimal response object for session.get mocks."""

    def __init__(
        self,
        *,
        status_code: int = 200,
        json_data: Optional[list[dict[str, str]]] = None,
        text: str = "",
        ok: Optional[bool] = None,
    ) -> None:
        self.status_code = status_code
        self._json = json_data or []
        self.text = text
        self.ok = ok if ok is not None else status_code < 400

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self) -> list[dict[str, str]]:
        return self._json


def test_get_available_transportation_networks_filters_valid_dirs() -> None:
    """Only directories with *_net.tntp files should be returned."""
    session = Mock()
    base_api = "https://api.github.com/repos/bstabler/TransportationNetworks/contents"

    def _get(url: str, timeout: int) -> _FakeResponse:
        if url == base_api:
            return _FakeResponse(
                json_data=[
                    {"name": "NetA", "type": "dir"},
                    {"name": "_ignore", "type": "dir"},
                    {"name": "README.md", "type": "file"},
                    {"name": "NetB", "type": "dir"},
                ]
            )
        if "NetA_net.tntp" in url:
            return _FakeResponse(status_code=200)
        if "NetB_net.tntp" in url:
            return _FakeResponse(status_code=404)
        if url == f"{base_api}/NetB":
            return _FakeResponse(
                json_data=[{"name": "alt_net.tntp", "type": "file"}],
                ok=True,
            )
        raise AssertionError(f"Unexpected URL {url}")

    session.get.side_effect = _get

    result = _get_available_transportation_networks(session)
    assert result == ["NetA", "NetB"]


def test_get_transportation_networks_data_prefers_local_files(
    tmp_path: Path,
) -> None:
    """Existing local files should be used without hitting the network."""
    session = Mock()
    sample_path = tmp_path / "TestNet_net.tntp"
    sample_path.write_text("1 2 3", encoding="utf-8")
    df = pd.DataFrame({"value": [1]})
    with patch.object(data_module, "_parse_tntp_from_lines", return_value=df) as mock_parser:
        data = _get_transportation_networks_data(
            session=session,
            network_name="TestNet",
            output_dir=tmp_path,
            save_to_file=False,
            load_network=True,
            load_trips=False,
            load_nodes=False,
            load_flow=False,
            download_if_missing=True,
            best_effort=True,
        )

    session.get.assert_not_called()
    mock_parser.assert_called_once()
    assert isinstance(data.network, pd.DataFrame)
    assert data.trips is None


@patch("city2graph.data._parse_tntp_from_lines")
def test_get_transportation_networks_data_handles_404(
    mock_parser: Mock,
) -> None:
    """404 remote responses should yield None without parsing."""
    session = Mock()
    response = _FakeResponse(status_code=404)
    session.get.return_value = response

    data = _get_transportation_networks_data(
        session=session,
        network_name="MissingNet",
        output_dir=None,
        save_to_file=False,
        load_network=True,
        load_trips=False,
        load_nodes=False,
        load_flow=False,
        download_if_missing=True,
        best_effort=True,
    )

    mock_parser.assert_not_called()
    assert data.network is None


def test_strip_metadata_removes_preamble_and_comments() -> None:
    """Metadata markers and comment lines are removed."""
    lines = [
        "this is metadata",
        "<END OF METADATA>",
        "1, 2, 3 ~ inline comment",
        "   ",
        "~ full comment",
        "4;5;6",
    ]
    cleaned = _strip_metadata(lines)
    assert cleaned == ["1, 2, 3", "4;5;6"]


def test_parse_tntp_from_lines_network() -> None:
    """Network parser should coerce numeric columns and add defaults."""
    lines = [
        "header",
        "<END OF METADATA>",
        "~ comment",
        "1,2,1000,1.5,12,0.15,60,30,0,3",
    ]

    df = _parse_tntp_from_lines(lines, "network")

    assert {"init_node", "term_node", "capacity", "length", "link_type"}.issubset(df.columns)
    assert df.at[0, "init_node"] == 1
    assert df.at[0, "capacity"] == pytest.approx(1000)
    assert df.at[0, "link_type"] == 3


def test_parse_tntp_from_lines_trips_aggregates_duplicates() -> None:
    """Trip matrices should aggregate duplicate OD pairs."""
    lines = [
        "<END OF METADATA>",
        "Origin 1",
        "2 : 10.0  3 : 5",
        "2 : 2",
        "Origin 2",
        "1 : 7",
    ]

    df = _parse_tntp_from_lines(lines, "trips")
    df = df.sort_values(["origin", "destination"]).reset_index(drop=True)
    expected = pd.DataFrame(
        {
            "origin": [1, 1, 2],
            "destination": [2, 3, 1],
            "demand": [12.0, 5.0, 7.0],
        }
    )
    pd.testing.assert_frame_equal(df, expected)


def test_parse_tntp_from_lines_flow_table() -> None:
    """Flow parser should capture volume and cost columns."""
    lines = [
        "<END OF METADATA>",
        "From To Volume Cost",
        "1 2 10 1.5",
        "3 4 5.5 2.25",
    ]

    df = _parse_tntp_from_lines(lines, "flow")
    expected = pd.DataFrame(
        {
            "origin": [1, 3],
            "destination": [2, 4],
            "volume": [10.0, 5.5],
            "cost": [1.5, 2.25],
        }
    )
    pd.testing.assert_frame_equal(df, expected)


def test_parse_tntp_from_lines_nodes_casts_types() -> None:
    """Node parser should coerce coordinates to numeric types."""
    lines = [
        "<END OF METADATA>",
        "node,x,y",
        "1,10.5,20.1",
        "2  30.2  40.3",
    ]

    df = _parse_tntp_from_lines(lines, "nodes")
    assert df.at[0, "node"] == 1
    assert df.at[1, "x"] == pytest.approx(30.2)


def test_parse_tntp_from_lines_unknown_type() -> None:
    """Unsupported data types should raise ValueError."""
    with pytest.raises(ValueError):
        _parse_tntp_from_lines(["<END OF METADATA>", "foo"], "unknown")


def test_parse_network_file_skips_headers() -> None:
    """Header rows and non-numeric prefixes should be ignored."""
    lines = [
        "Init,Term,Capacity",
        "1, 2 , 1000 ; 1.5",
        "3 4 2000 2.0 10",
    ]
    df = _parse_network_file(lines)
    assert len(df) == 2
    assert set(df.columns) >= {"init_node", "term_node"}
    assert df.at[0, "term_node"] == 2


def test_parse_flow_file_skips_headers() -> None:
    """Flow parser should ignore header rows and coerce numeric values."""
    lines = [
        "From To Volume Cost",
        "1 2 10 1.5",
        "3 4 5.5 2.25",
    ]
    df = _parse_flow_file(lines)
    assert len(df) == 2
    assert set(df.columns) == {"origin", "destination", "volume", "cost"}
    assert df.at[1, "volume"] == pytest.approx(5.5)


def test_parse_trips_file_parses_pairs() -> None:
    """Matrix files should yield origin-destination-demand rows."""
    lines = [
        "Origin 1",
        "2 : 10.0  3 : 5",
        "Origin 2",
        "1 : 7",
    ]
    df = _parse_trips_file(lines)
    assert len(df) == 3
    assert {"origin", "destination", "demand"} == set(df.columns)


def test_parse_trips_file_requires_origin() -> None:
    """Matrix parser should complain when no origin section is present."""
    with pytest.raises(ValueError):
        _parse_trips_file(["2 : 1.0"])


def test_parse_nodes_file_skips_invalid_rows() -> None:
    """Nodes parser should ignore malformed lines."""
    lines = [
        "header",
        "1, 10, 20",
        "two, 30, 40",
        "3  50  60",
    ]
    df = _parse_nodes_file(lines)
    assert len(df) == 2
    assert df.at[1, "node"] == 3
