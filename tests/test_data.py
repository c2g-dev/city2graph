"""Tests for the data module - refactored version focusing on public API only."""

import subprocess
from unittest.mock import Mock
from unittest.mock import patch

import geopandas as gpd
import pytest
from shapely.geometry import LineString
from shapely.geometry import MultiLineString
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import Polygon

from city2graph import data as data_module
from tests.helpers import make_connectors_gdf
from tests.helpers import make_segments_gdf

VALID_OVERTURE_TYPES = data_module.VALID_OVERTURE_TYPES
WGS84_CRS = data_module.WGS84_CRS
load_overture_data = data_module.load_overture_data
process_overture_segments = data_module.process_overture_segments
get_boundaries = data_module.get_boundaries


# ============================================================================
# TESTS FOR CONSTANTS
# ============================================================================


class TestConstants:
    """Test module-level constants."""

    def test_valid_overture_types_constant(self) -> None:
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

    def test_wgs84_crs_constant(self) -> None:
        """Test that WGS84_CRS is correctly defined."""
        assert WGS84_CRS == "EPSG:4326"


# ============================================================================
# TESTS FOR load_overture_data FUNCTION
# ============================================================================


class TestLoadOvertureData:
    """Tests for the load_overture_data function."""

    @patch("city2graph.data.subprocess.run")
    @patch("city2graph.data.gpd.read_file")
    @patch("city2graph.data.Path.mkdir")
    def test_with_bbox_list(
        self,
        mock_mkdir: Mock,
        mock_read_file: Mock,
        mock_subprocess: Mock,
        test_bbox: list[float],
    ) -> None:
        """Test load_overture_data with bounding box as list."""
        types = ["building", "segment"]
        mock_gdf = Mock(spec=gpd.GeoDataFrame)
        mock_gdf.empty = False
        mock_read_file.return_value = mock_gdf

        result = load_overture_data(test_bbox, types=types, output_dir="test_dir")

        assert len(result) == 2
        assert "building" in result
        assert "segment" in result
        mock_mkdir.assert_called_once()
        assert mock_subprocess.call_count == 2

    @patch("city2graph.data.subprocess.run")
    @patch("city2graph.data.gpd.read_file")
    @patch("city2graph.data.Path.mkdir")
    def test_with_polygon(
        self,
        mock_mkdir: Mock,
        mock_read_file: Mock,
        mock_subprocess: Mock,
        test_polygon: Polygon,
    ) -> None:
        """Test load_overture_data with Polygon area."""
        mock_gdf = Mock(spec=gpd.GeoDataFrame)
        mock_gdf.empty = False
        mock_read_file.return_value = mock_gdf

        result = load_overture_data(test_polygon, types=["building"])

        assert "building" in result
        mock_subprocess.assert_called_once()
        mock_mkdir.assert_called()

    @patch("city2graph.data.subprocess.run")
    @patch("city2graph.data.gpd.read_file")
    @patch("city2graph.data.gpd.clip")
    @patch("city2graph.data.Path.exists")
    def test_with_polygon_clipping(
        self,
        mock_exists: Mock,
        mock_clip: Mock,
        mock_read_file: Mock,
        mock_subprocess: Mock,
        test_polygon: Polygon,
    ) -> None:
        """Test that polygon areas are properly clipped."""
        mock_gdf = Mock(spec=gpd.GeoDataFrame)
        mock_gdf.empty = False
        mock_gdf.crs = "EPSG:3857"
        mock_read_file.return_value = mock_gdf
        mock_exists.return_value = True
        mock_clipped_gdf = Mock(spec=gpd.GeoDataFrame)
        mock_clip.return_value = mock_clipped_gdf

        result = load_overture_data(test_polygon, types=["building"])

        mock_clip.assert_called_once()
        mock_subprocess.assert_called()
        assert result["building"] == mock_clipped_gdf

    @patch("city2graph.data.subprocess.run")
    @patch("city2graph.data.gpd.read_file")
    @patch("city2graph.data.Path.mkdir")
    def test_with_multipolygon(
        self,
        mock_mkdir: Mock,
        mock_read_file: Mock,
        mock_subprocess: Mock,
    ) -> None:
        """Test load_overture_data with MultiPolygon area."""
        poly1 = Polygon([(-74.01, 40.70), (-73.99, 40.70), (-73.99, 40.72), (-74.01, 40.72)])
        poly2 = Polygon([(-74.05, 40.75), (-74.03, 40.75), (-74.03, 40.77), (-74.05, 40.77)])
        test_multipolygon = MultiPolygon([poly1, poly2])

        mock_gdf = Mock(spec=gpd.GeoDataFrame)
        mock_gdf.empty = False
        mock_read_file.return_value = mock_gdf

        result = load_overture_data(test_multipolygon, types=["building"])

        assert "building" in result
        mock_subprocess.assert_called_once()
        mock_mkdir.assert_called()

    def test_invalid_types(self, test_bbox: list[float]) -> None:
        """Test that invalid types raise ValueError."""
        invalid_types = ["building", "invalid_type", "another_invalid"]

        with pytest.raises(
            ValueError, match=r"Invalid types: \['invalid_type', 'another_invalid'\]"
        ):
            load_overture_data(test_bbox, types=invalid_types)

    def test_invalid_release(self, test_bbox: list[float]) -> None:
        """Test that invalid release raises ValueError."""
        invalid_release = "2099-99-99.0"

        with (
            patch.object(
                data_module, "ALL_RELEASES", ["2025-08-20.0", "2025-08-20.1", "2025-09-24.0"]
            ),
            pytest.raises(
                ValueError,
                match="Invalid release: 2099-99-99.0. Valid releases are: 2025-08-20.0, 2025-08-20.1, 2025-09-24.0",
            ),
        ):
            load_overture_data(test_bbox, types=["building"], release=invalid_release)

    @patch("city2graph.data.subprocess.run")
    @patch("city2graph.data.gpd.read_file")
    def test_valid_release(
        self,
        mock_read_file: Mock,
        mock_subprocess: Mock,
        test_bbox: list[float],
    ) -> None:
        """Test that valid release is accepted without raising errors."""
        valid_release = "2025-08-20.1"
        mock_gdf = Mock(spec=gpd.GeoDataFrame)
        mock_gdf.empty = False
        mock_read_file.return_value = mock_gdf

        with patch.object(
            data_module, "ALL_RELEASES", ["2025-08-20.0", "2025-08-20.1", "2025-09-24.0"]
        ):
            result = load_overture_data(test_bbox, types=["building"], release=valid_release)

            assert "building" in result
            mock_subprocess.assert_called_once()
            call_args = mock_subprocess.call_args[0][0]
            assert "-r" in call_args
            assert valid_release in call_args

    @patch("city2graph.data.subprocess.run")
    @patch("city2graph.data.gpd.read_file")
    def test_default_types(
        self,
        mock_read_file: Mock,
        mock_subprocess: Mock,
        test_bbox: list[float],
    ) -> None:
        """Test that all valid types are used when types=None."""
        mock_gdf = Mock(spec=gpd.GeoDataFrame)
        mock_gdf.empty = False
        mock_read_file.return_value = mock_gdf

        result = load_overture_data(test_bbox, types=None, save_to_file=False)

        assert len(result) == len(VALID_OVERTURE_TYPES)
        for data_type in VALID_OVERTURE_TYPES:
            assert data_type in result
        assert mock_subprocess.call_count == len(VALID_OVERTURE_TYPES)

    @patch("city2graph.data.subprocess.run")
    def test_save_to_file_false(
        self,
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

        assert result == {}
        mock_subprocess.assert_called_once()
        args = mock_subprocess.call_args[0][0]
        assert "-o" not in args

    @patch("city2graph.data.subprocess.run")
    @patch("city2graph.data.gpd.read_file")
    def test_segment_geometry_filtering(
        self,
        mock_read_file: Mock,
        test_bbox: list[float],
    ) -> None:
        """Test that non-LineString geometries are filtered from segments."""
        # Create mixed geometry segments
        line = LineString([(0, 0), (1, 1)])
        point = Point(0.5, 0.5)
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])
        multi_line = MultiLineString([[(0, 0), (1, 1)], [(1, 1), (2, 2)]])

        mock_gdf = gpd.GeoDataFrame(
            {
                "id": ["line", "point", "poly", "multi"],
                "geometry": [line, point, poly, multi_line],
            },
            crs=WGS84_CRS,
        )
        mock_read_file.return_value = mock_gdf

        result = load_overture_data(test_bbox, types=["segment"])

        assert "segment" in result
        segments = result["segment"]

        # Should contain line and exploded multi-line (2 parts) -> total 3
        # Point and Polygon should be gone
        assert len(segments) == 3
        assert all(isinstance(geom, LineString) for geom in segments.geometry)
        assert "point" not in segments["id"].values
        assert "poly" not in segments["id"].values
        # Exploded multilinestring parts share the original ID
        assert len(segments[segments["id"] == "multi"]) == 2

    @patch("city2graph.data.subprocess.run")
    @patch("city2graph.data.gpd.read_file")
    def test_return_data_false(
        self,
        mock_read_file: Mock,
        mock_subprocess: Mock,
        test_bbox: list[float],
    ) -> None:
        """Test load_overture_data with return_data=False."""
        result = load_overture_data(test_bbox, types=["building"], return_data=False)

        assert result == {}
        mock_read_file.assert_not_called()
        mock_subprocess.assert_called()

    @patch("city2graph.data.subprocess.run")
    def test_subprocess_error(self, mock_subprocess: Mock, test_bbox: list[float]) -> None:
        """Test that subprocess errors are propagated."""
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, "overturemaps")

        with pytest.raises(subprocess.CalledProcessError):
            load_overture_data(test_bbox, types=["building"])

    @patch("city2graph.data.subprocess.run")
    @patch("city2graph.data.gpd.read_file")
    def test_with_prefix(
        self,
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

        args = mock_subprocess.call_args[0][0]
        output_index = args.index("-o") + 1
        output_path = args[output_index]
        assert prefix in output_path

    @patch("city2graph.data.subprocess.run")
    @patch("city2graph.data.gpd.read_file")
    def test_with_connect_timeout(
        self,
        mock_read_file: Mock,
        mock_subprocess: Mock,
        test_bbox: list[float],
    ) -> None:
        """Test load_overture_data with connect_timeout parameter."""
        mock_gdf = Mock(spec=gpd.GeoDataFrame)
        mock_gdf.empty = False
        mock_read_file.return_value = mock_gdf

        load_overture_data(test_bbox, types=["building"], connect_timeout=5.0)

        args = mock_subprocess.call_args[0][0]
        assert "--connect-timeout" in args
        assert "5.0" in args

    @patch("city2graph.data.subprocess.run")
    @patch("city2graph.data.gpd.read_file")
    def test_with_request_timeout(
        self,
        mock_read_file: Mock,
        mock_subprocess: Mock,
        test_bbox: list[float],
    ) -> None:
        """Test load_overture_data with request_timeout parameter."""
        mock_gdf = Mock(spec=gpd.GeoDataFrame)
        mock_gdf.empty = False
        mock_read_file.return_value = mock_gdf

        load_overture_data(test_bbox, types=["building"], request_timeout=10.0)

        args = mock_subprocess.call_args[0][0]
        assert "--request-timeout" in args
        assert "10.0" in args

    @patch("city2graph.data.subprocess.run")
    @patch("city2graph.data.gpd.read_file")
    def test_with_no_stac(
        self,
        mock_read_file: Mock,
        mock_subprocess: Mock,
        test_bbox: list[float],
    ) -> None:
        """Test load_overture_data with use_stac=False parameter."""
        mock_gdf = Mock(spec=gpd.GeoDataFrame)
        mock_gdf.empty = False
        mock_read_file.return_value = mock_gdf

        load_overture_data(test_bbox, types=["building"], use_stac=False)

        args = mock_subprocess.call_args[0][0]
        assert "--no-stac" in args

    @patch("city2graph.data.subprocess.run")
    @patch("city2graph.data.gpd.read_file")
    def test_with_stac_true(
        self,
        mock_read_file: Mock,
        mock_subprocess: Mock,
        test_bbox: list[float],
    ) -> None:
        """Test load_overture_data with use_stac=True (default)."""
        mock_gdf = Mock(spec=gpd.GeoDataFrame)
        mock_gdf.empty = False
        mock_read_file.return_value = mock_gdf

        load_overture_data(test_bbox, types=["building"], use_stac=True)

        args = mock_subprocess.call_args[0][0]
        assert "--no-stac" not in args

    @patch("city2graph.data.subprocess.run")
    @patch("city2graph.data.gpd.read_file")
    @patch("city2graph.data.Path.exists")
    def test_file_not_exists(
        self,
        mock_exists: Mock,
        mock_read_file: Mock,
        mock_subprocess: Mock,
        test_bbox: list[float],
    ) -> None:
        """Test behavior when output file doesn't exist."""
        mock_exists.return_value = False

        result = load_overture_data(test_bbox, types=["building"])

        mock_read_file.assert_not_called()
        assert "building" in result
        mock_subprocess.assert_called()

    @patch("city2graph.data.subprocess.run")
    @patch("city2graph.data.gpd.read_file")
    def test_with_place_name(
        self,
        mock_read: Mock,
        mock_subprocess: Mock,
    ) -> None:
        """Test load_overture_data with place_name parameter."""
        with patch.object(data_module, "Nominatim") as mock_nominatim:
            mock_nominatim.return_value.geocode.return_value = [
                Mock(
                    raw={
                        "geojson": {
                            "type": "Polygon",
                            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                        },
                    },
                )
            ]
            mock_gdf = Mock(spec=gpd.GeoDataFrame)
            mock_gdf.empty = False
            mock_read.return_value = mock_gdf

            result = load_overture_data(place_name="Liverpool", types=["building"])
            assert "building" in result
            mock_subprocess.assert_called()

    def test_mutual_exclusion(self, test_bbox: list[float]) -> None:
        """Test that area and place_name are mutually exclusive."""
        with pytest.raises(ValueError, match="Exactly one"):
            load_overture_data(area=test_bbox, place_name="Liverpool")
        with pytest.raises(ValueError, match="Exactly one"):
            load_overture_data()

    @patch("city2graph.data.subprocess.run")
    @patch("city2graph.data.gpd.read_file")
    @patch("city2graph.data.Path.mkdir")
    def test_with_geodataframe_non_wgs84(
        self,
        mock_mkdir: Mock,
        mock_read_file: Mock,
        mock_subprocess: Mock,
    ) -> None:
        """Test load_overture_data with GeoDataFrame having non-WGS84 CRS."""
        _ = mock_mkdir
        poly = Polygon([(-2.99, 53.40), (-2.98, 53.40), (-2.98, 53.41), (-2.99, 53.41)])
        area_gdf = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:27700")

        mock_gdf = Mock(spec=gpd.GeoDataFrame)
        mock_gdf.empty = False
        mock_read_file.return_value = mock_gdf

        result = load_overture_data(area=area_gdf, types=["building"])

        assert "building" in result
        mock_subprocess.assert_called_once()

    @patch("city2graph.data.subprocess.run")
    @patch("city2graph.data.gpd.read_file")
    @patch("city2graph.data.Path.mkdir")
    def test_with_geoseries_non_wgs84(
        self,
        mock_mkdir: Mock,
        mock_read_file: Mock,
        mock_subprocess: Mock,
    ) -> None:
        """Test load_overture_data with GeoSeries having non-WGS84 CRS."""
        _ = mock_mkdir
        poly = Polygon([(-2.99, 53.40), (-2.98, 53.40), (-2.98, 53.41), (-2.99, 53.41)])
        area_series = gpd.GeoSeries([poly], crs="EPSG:27700")

        mock_gdf = Mock(spec=gpd.GeoDataFrame)
        mock_gdf.empty = False
        mock_read_file.return_value = mock_gdf

        result = load_overture_data(area=area_series, types=["building"])

        assert "building" in result
        mock_subprocess.assert_called_once()

    @patch("city2graph.data.subprocess.run")
    def test_stdout_json_parsing(self, mock_subprocess: Mock) -> None:
        """Test JSON parsing from subprocess stdout with prefix text."""
        geojson_str = '{"type": "FeatureCollection", "features": []}'
        mock_subprocess.return_value = Mock(
            stdout=f"Some warning message\n{geojson_str}",
            returncode=0,
        )

        bbox = [-74.01, 40.70, -73.99, 40.72]
        result = load_overture_data(bbox, types=["building"], save_to_file=False)

        assert "building" in result
        mock_subprocess.assert_called()

    @patch("city2graph.data.subprocess.run")
    @patch("city2graph.data.gpd.read_file")
    @patch("city2graph.data.Path.mkdir")
    def test_comprehensive_all_types(
        self,
        mock_mkdir: Mock,
        mock_read_file: Mock,
        mock_subprocess: Mock,
    ) -> None:
        """Test load_overture_data with all types (types=None)."""
        _ = mock_mkdir
        mock_gdf = Mock(spec=gpd.GeoDataFrame)
        mock_gdf.empty = False
        mock_read_file.return_value = mock_gdf

        bbox = [-74.01, 40.70, -73.99, 40.72]
        result = load_overture_data(bbox, types=None, save_to_file=False)

        assert mock_subprocess.call_count == len(VALID_OVERTURE_TYPES)
        assert len(result) == len(VALID_OVERTURE_TYPES)


# ============================================================================
# TESTS FOR process_overture_segments FUNCTION
# ============================================================================


class TestProcessOvertureSegments:
    """Tests for the process_overture_segments function."""

    def test_empty_input(self) -> None:
        """Test process_overture_segments with empty GeoDataFrame."""
        empty_gdf = gpd.GeoDataFrame(geometry=[], crs=WGS84_CRS)
        result = process_overture_segments(empty_gdf)

        assert result.empty
        assert result.crs == WGS84_CRS

    def test_basic(self) -> None:
        """Test basic functionality of process_overture_segments."""
        segments_gdf = make_segments_gdf(
            ids=["s1", "s2"],
            geoms_or_coords=[[(0, 0), (1, 0)], [(1, 0), (2, 0)]],
            level_rules="",
            crs=WGS84_CRS,
        )

        result = process_overture_segments(segments_gdf, get_barriers=False)

        assert "length" in result.columns
        assert all(result["length"] > 0)
        assert len(result) >= len(segments_gdf)
        assert "id" in result.columns

    def test_with_connectors(self) -> None:
        """Test process_overture_segments with connectors."""
        connectors_data = '[{"connector_id": "c1", "at": 0.25}, {"connector_id": "c2", "at": 0.75}]'
        segments_gdf = make_segments_gdf(
            ids=["seg1"],
            geoms_or_coords=[[(0, 0), (4, 0)]],
            connectors=connectors_data,
            level_rules="",
            crs=WGS84_CRS,
        )
        connectors_gdf = make_connectors_gdf(
            ids=["c1", "c2"], coords=[(1, 0), (3, 0)], crs=WGS84_CRS
        )

        result = process_overture_segments(
            segments_gdf,
            connectors_gdf=connectors_gdf,
            get_barriers=False,
        )

        split_segments = result[result["id"].str.contains("_")]
        if not split_segments.empty:
            assert "split_from" in result.columns
            assert "split_to" in result.columns

    def test_with_barriers(self) -> None:
        """Test process_overture_segments with barrier generation."""
        segments_gdf = make_segments_gdf(
            ids=["seg1"],
            geoms_or_coords=[[(0, 0), (1, 1)]],
            level_rules="",
            crs=WGS84_CRS,
        )
        result = process_overture_segments(segments_gdf, get_barriers=True)

        assert "barrier_geometry" in result.columns

    def test_missing_level_rules(self) -> None:
        """Test process_overture_segments with missing level_rules column."""
        segments_gdf = make_segments_gdf(
            ids=["seg1"],
            geoms_or_coords=[[(0, 0), (1, 1)]],
            level_rules=None,
            crs=WGS84_CRS,
        )

        result = process_overture_segments(segments_gdf)
        assert "level_rules" in result.columns
        assert result["level_rules"].iloc[0] == ""

    def test_with_threshold(self) -> None:
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

        assert "length" in result.columns

    def test_no_connectors(self) -> None:
        """Test process_overture_segments with None connectors."""
        segments_gdf = make_segments_gdf(
            ids=["s1", "s2"],
            geoms_or_coords=[[(0, 0), (1, 0)], [(1, 0), (2, 0)]],
            level_rules="",
            crs=WGS84_CRS,
        )
        result = process_overture_segments(segments_gdf, connectors_gdf=None)

        assert len(result) == len(segments_gdf)

    def test_empty_connectors(self) -> None:
        """Test process_overture_segments with empty connectors GeoDataFrame."""
        segments_gdf = make_segments_gdf(
            ids=["s1", "s2"],
            geoms_or_coords=[[(0, 0), (1, 0)], [(1, 0), (2, 0)]],
            level_rules="",
            crs=WGS84_CRS,
        )
        empty_connectors = make_connectors_gdf(ids=[], coords=[], crs=WGS84_CRS)
        result = process_overture_segments(segments_gdf, connectors_gdf=empty_connectors)

        assert len(result) == len(segments_gdf)

    def test_invalid_connector_data(self) -> None:
        """Test process_overture_segments with invalid connector JSON."""
        segments_gdf = make_segments_gdf(
            ids=["seg1"],
            geoms_or_coords=[[(0, 0), (1, 1)]],
            connectors="invalid_json",
            level_rules="",
            crs=WGS84_CRS,
        )

        result = process_overture_segments(segments_gdf)
        assert len(result) == 1

    def test_malformed_connectors(self) -> None:
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

        assert len(result) == 1

    def test_invalid_level_rules(self) -> None:
        """Test process_overture_segments with invalid level rules JSON."""
        segments_gdf = make_segments_gdf(
            ids=["seg1"],
            geoms_or_coords=[[(0, 0), (1, 1)]],
            level_rules="invalid_json",
            crs=WGS84_CRS,
        )

        result = process_overture_segments(segments_gdf, get_barriers=True)
        assert "barrier_geometry" in result.columns

    def test_complex_level_rules(self) -> None:
        """Test process_overture_segments with complex level rules."""
        level_rules = '[{"value": 1, "between": [0.1, 0.3]}, {"value": 1, "between": [0.7, 0.9]}]'
        segments_gdf = make_segments_gdf(
            ids=["seg1"],
            geoms_or_coords=[[(0, 0), (1, 1)]],
            level_rules=level_rules,
            crs=WGS84_CRS,
        )

        result = process_overture_segments(segments_gdf, get_barriers=True)

        assert "barrier_geometry" in result.columns
        assert result["barrier_geometry"].iloc[0] is not None

    def test_full_barrier(self) -> None:
        """Test process_overture_segments with full barrier level rules."""
        level_rules = '[{"value": 1}]'
        segments_gdf = make_segments_gdf(
            ids=["seg1"],
            geoms_or_coords=[[(0, 0), (1, 1)]],
            level_rules=level_rules,
            crs=WGS84_CRS,
        )

        result = process_overture_segments(segments_gdf, get_barriers=True)

        assert "barrier_geometry" in result.columns
        assert result["barrier_geometry"].iloc[0] is None

    def test_zero_value_rules(self) -> None:
        """Test process_overture_segments with zero value level rules."""
        level_rules = '[{"value": 0, "between": [0.2, 0.8]}]'
        segments_gdf = make_segments_gdf(
            ids=["seg1"],
            geoms_or_coords=[[(0, 0), (1, 1)]],
            level_rules=level_rules,
            crs=WGS84_CRS,
        )

        result = process_overture_segments(segments_gdf, get_barriers=True)

        assert "barrier_geometry" in result.columns
        barrier_geom = result["barrier_geometry"].iloc[0]
        assert barrier_geom is not None

    def test_segment_splitting(self) -> None:
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

        assert len(result) > 1
        split_segments = result[result["id"].str.contains("_")]
        assert not split_segments.empty

    def test_endpoint_clustering(self) -> None:
        """Test endpoint clustering functionality."""
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
            threshold=0.5,
        )

        assert len(result) >= len(segments_gdf)

    def test_level_rules_handling(self) -> None:
        """Test process_overture_segments level_rules column handling."""
        segments_gdf = make_segments_gdf(
            ids=["seg1"],
            geoms_or_coords=[[(0, 0), (1, 1)]],
            level_rules=[None],
            crs=WGS84_CRS,
        )

        result = process_overture_segments(segments_gdf)
        assert result["level_rules"].iloc[0] == ""

    def test_with_non_dict_connector(self) -> None:
        """Test process_overture_segments with non-dict connector data."""
        segments_gdf = make_segments_gdf(
            ids=["seg1"],
            geoms_or_coords=[[(0, 0), (1, 1)]],
            connectors='["not_a_dict"]',
            level_rules="",
            crs=WGS84_CRS,
        )

        result = process_overture_segments(segments_gdf)
        assert len(result) == 1

    def test_with_non_dict_level_rule(self) -> None:
        """Test process_overture_segments with non-dict level rule data."""
        segments_gdf = make_segments_gdf(
            ids=["seg1"],
            geoms_or_coords=[[(0, 0), (1, 1)]],
            level_rules='["not_a_dict"]',
            crs=WGS84_CRS,
        )

        result = process_overture_segments(segments_gdf, get_barriers=True)
        assert "barrier_geometry" in result.columns

    def test_with_short_between_array(self) -> None:
        """Test process_overture_segments with short between array in level rules."""
        level_rules = '[{"value": 1, "between": [0.5]}]'
        segments_gdf = make_segments_gdf(
            ids=["seg1"],
            geoms_or_coords=[[(0, 0), (1, 1)]],
            level_rules=level_rules,
            crs=WGS84_CRS,
        )

        result = process_overture_segments(segments_gdf, get_barriers=True)
        assert "barrier_geometry" in result.columns

    def test_with_empty_geometry(self) -> None:
        """Test process_overture_segments with empty geometry."""
        empty_geom = LineString()
        segments_gdf = make_segments_gdf(
            ids=["seg1"],
            geoms_or_coords=[empty_geom],
            level_rules="",
            crs=WGS84_CRS,
        )

        result = process_overture_segments(segments_gdf, get_barriers=True)
        assert "barrier_geometry" in result.columns

    def test_with_overlapping_barriers(self) -> None:
        """Test process_overture_segments with overlapping barrier intervals."""
        level_rules = '[{"value": 1, "between": [0.1, 0.5]}, {"value": 1, "between": [0.3, 0.7]}]'
        segments_gdf = make_segments_gdf(
            ids=["seg1"],
            geoms_or_coords=[[(0, 0), (4, 4)]],
            level_rules=level_rules,
            crs=WGS84_CRS,
        )

        result = process_overture_segments(segments_gdf, get_barriers=True)
        assert "barrier_geometry" in result.columns

    def test_with_touching_barriers(self) -> None:
        """Test process_overture_segments with touching barrier intervals."""
        level_rules = '[{"value": 1, "between": [0.0, 0.3]}, {"value": 1, "between": [0.3, 0.6]}]'
        segments_gdf = make_segments_gdf(
            ids=["seg1"],
            geoms_or_coords=[[(0, 0), (4, 4)]],
            level_rules=level_rules,
            crs=WGS84_CRS,
        )

        result = process_overture_segments(segments_gdf, get_barriers=True)
        assert "barrier_geometry" in result.columns

    def test_with_full_coverage_barriers(self) -> None:
        """Test process_overture_segments with barriers covering full segment."""
        level_rules = '[{"value": 1, "between": [0.0, 1.0]}]'
        segments_gdf = make_segments_gdf(
            ids=["seg1"],
            geoms_or_coords=[[(0, 0), (4, 4)]],
            level_rules=level_rules,
            crs=WGS84_CRS,
        )

        result = process_overture_segments(segments_gdf, get_barriers=True)
        assert "barrier_geometry" in result.columns
        assert result["barrier_geometry"].iloc[0] is None

    def test_with_non_linestring_endpoints(self) -> None:
        """Test endpoint clustering with non-LineString geometries."""
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

        # Point geometry should NOT be filtered out (filtering moved to load_overture_data)
        assert len(result) == 2
        assert result.iloc[0]["id"] == "seg1"

    def test_with_short_linestring(self) -> None:
        """Test endpoint clustering with LineString having insufficient coordinates."""
        invalid_geom = LineString([(0, 0), (0, 0)])
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

        assert len(result) == len(segments_gdf)

    def test_connectors_as_list(self) -> None:
        """Test process_overture_segments when connectors column is already a list."""
        segments_gdf = make_segments_gdf(
            ids=["seg1"],
            geoms_or_coords=[[(0, 0), (2, 0)]],
            connectors='[{"connector_id": "c1", "at": 0.5}]',
            level_rules="",
            crs=WGS84_CRS,
        )
        connectors_gdf = make_connectors_gdf(ids=["c1"], coords=[(1, 0)], crs=WGS84_CRS)

        result = process_overture_segments(segments_gdf, connectors_gdf=connectors_gdf)

        assert len(result) >= 1

    def test_connectors_json_decode_error(self) -> None:
        """Test process_overture_segments with JSON that causes decode error."""
        segments_gdf = make_segments_gdf(
            ids=["seg1"],
            geoms_or_coords=[[(0, 0), (1, 1)]],
            connectors="{malformed json without quotes}",
            level_rules="",
            crs=WGS84_CRS,
        )
        connectors_gdf = make_connectors_gdf(ids=["c1"], coords=[(0.5, 0.5)], crs=WGS84_CRS)

        result = process_overture_segments(segments_gdf, connectors_gdf=connectors_gdf)

        assert len(result) == 1

    def test_level_rules_as_list(self) -> None:
        """Test process_overture_segments when level_rules is already a list."""
        segments_gdf = make_segments_gdf(
            ids=["seg1"],
            geoms_or_coords=[[(0, 0), (4, 4)]],
            level_rules='[{"value": 1, "between": [0.1, 0.5]}]',
            crs=WGS84_CRS,
        )

        result = process_overture_segments(segments_gdf, get_barriers=True)

        assert "barrier_geometry" in result.columns

    def test_connectors_single_dict(self) -> None:
        """Test process_overture_segments when connectors parses to a single dict."""
        segments_gdf = make_segments_gdf(
            ids=["seg1"],
            geoms_or_coords=[[(0, 0), (2, 0)]],
            connectors='{"connector_id": "c1", "at": 0.5}',
            level_rules="",
            crs=WGS84_CRS,
        )
        connectors_gdf = make_connectors_gdf(ids=["c1"], coords=[(1, 0)], crs=WGS84_CRS)

        result = process_overture_segments(segments_gdf, connectors_gdf=connectors_gdf)

        assert len(result) >= 1


# ============================================================================
# TESTS FOR get_boundaries FUNCTION
# ============================================================================


class TestGetBoundaries:
    """Tests for the get_boundaries function."""

    def test_polygon(self) -> None:
        """Test polygon boundary retrieval."""
        with patch.object(data_module, "Nominatim") as mock_nominatim:
            mock_nominatim.return_value.geocode.return_value = [
                Mock(
                    raw={
                        "geojson": {
                            "type": "Polygon",
                            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                        },
                    },
                )
            ]
            result = get_boundaries("Liverpool")
            assert result.geometry.geom_type.iloc[0] == "Polygon"
            assert result.crs == WGS84_CRS

    def test_not_found(self) -> None:
        """Test error when place not found."""
        with patch.object(data_module, "Nominatim") as mock_nominatim:
            mock_nominatim.return_value.geocode.return_value = []
            with pytest.raises(ValueError, match="Place not found"):
                get_boundaries("NonexistentPlace12345")

    def test_point_error(self) -> None:
        """Test error when Point geometry returned."""
        with patch.object(data_module, "Nominatim") as mock_nominatim:
            mock_nominatim.return_value.geocode.return_value = [
                Mock(
                    raw={"geojson": {"type": "Point", "coordinates": [0, 0]}},
                )
            ]
            with pytest.raises(ValueError, match="No polygon boundary"):
                get_boundaries("123 Main St")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests for the data module."""

    def test_load_and_process_integration(self) -> None:
        """Test integration between load_overture_data and process_overture_segments."""
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

        result = process_overture_segments(segments_gdf)

        expected_columns = [
            "id",
            "connectors",
            "level_rules",
            "geometry",
            "length",
            "barrier_geometry",
        ]
        for col in expected_columns:
            assert col in result.columns

    @patch("city2graph.data.subprocess.run")
    @patch("city2graph.data.gpd.read_file")
    @patch("city2graph.data.Path.exists")
    def test_real_world_scenario_simulation(
        self,
        mock_exists: Mock,
        mock_read_file: Mock,
        mock_subprocess: Mock,
        test_bbox: list[float],
    ) -> None:
        """Test a scenario that simulates real-world usage."""
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

        mock_read_file.side_effect = [segments_gdf, connectors_gdf]
        mock_exists.return_value = True

        data = load_overture_data(test_bbox, types=["segment", "connector"])

        processed_segments = process_overture_segments(
            data["segment"],
            connectors_gdf=data["connector"],
        )
        assert not processed_segments.empty
        assert "barrier_geometry" in processed_segments.columns

        mock_subprocess.assert_called()
        assert "length" in processed_segments.columns
