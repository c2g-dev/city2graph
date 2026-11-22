"""Tests for the data module - refactored version focusing on public API only."""

import subprocess
from pathlib import Path
from typing import BinaryIO
from typing import cast
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
from tests.city2graph.helpers import make_connectors_gdf
from tests.city2graph.helpers import make_segments_gdf

VALID_OVERTURE_TYPES = data_module.VALID_OVERTURE_TYPES
WGS84_CRS = data_module.WGS84_CRS
load_overture_data = data_module.load_overture_data
process_overture_segments = data_module.process_overture_segments
get_boundaries = data_module.get_boundaries


@pytest.fixture
def test_bbox() -> list[float]:
    """Standard test bounding box."""
    return [-74.01, 40.70, -73.99, 40.72]


@pytest.fixture
def test_polygon() -> Polygon:
    """Standard test polygon."""
    return Polygon([(-74.01, 40.70), (-73.99, 40.70), (-73.99, 40.72), (-74.01, 40.72)])


class UnreachableReleases:
    """Stand-in for ``ALL_RELEASES`` when the STAC catalogue cannot be fetched.

    overturemaps resolves the proxy on iteration and raises a bare ``Exception``
    when the catalogue is unreachable.
    """

    def __iter__(self) -> object:
        """Raise as overturemaps does when the catalogue cannot be fetched.

        Returns
        -------
        object
            Never returns; always raises.

        Raises
        ------
        Exception
            Always, mirroring the bare exception overturemaps raises.
        """
        msg = "Could not fetch STAC catalog: <urlopen error>"
        raise Exception(msg)  # noqa: TRY002 - mirrors the bare exception overturemaps raises


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

    @pytest.mark.parametrize("prefix", [b"", b"Warning\n", b"x" * 70000])
    def test_spooled_stdout_preserves_features(
        self, prefix: bytes, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Real JSON is loaded after bounded prefix stripping and temp cleanup."""
        paths = []
        payload = (
            b'{"type":"FeatureCollection","features":[{"type":"Feature",'
            b'"properties":{"name":"sample"},"geometry":{"type":"Point","coordinates":[1,2]}}]}'
        )

        def download(_cmd: object, **kwargs: object) -> None:
            stream = cast("BinaryIO", kwargs["stdout"])
            paths.append(Path(stream.name))
            stream.write(prefix + payload)

        monkeypatch.setattr("city2graph.data.subprocess.run", download)
        result = load_overture_data([0, 0, 3, 3], types=["building"], save_to_file=False)
        frame = result["building"]
        assert frame["name"].tolist() == ["sample"]
        assert frame.geometry.iloc[0].equals(Point(1, 2))
        assert all(not path.exists() for path in paths)

    @pytest.mark.parametrize("as_frame", [False, True])
    def test_download_uses_every_feature_in_polygon_area(
        self, as_frame: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Download the union, preserving a gap between disjoint polygons."""
        polygons = gpd.GeoSeries(
            [
                Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                Polygon([(4, 0), (6, 0), (6, 2), (4, 2)]),
            ],
            crs=WGS84_CRS,
        )
        area = gpd.GeoDataFrame(geometry=polygons) if as_frame else polygons
        features = gpd.GeoDataFrame(
            {"name": ["left", "gap", "right"]},
            geometry=[Point(1, 1), Point(3, 1), Point(5, 1)],
            crs=WGS84_CRS,
        )
        calls: list[list[str]] = []

        def download(cmd: list[str], **kwargs: object) -> None:
            calls.append(cmd)
            stream = cast("BinaryIO", kwargs["stdout"])
            stream.write(features.to_json().encode())

        monkeypatch.setattr("city2graph.data.subprocess.run", download)
        result = load_overture_data(area, types=["place"], save_to_file=False)
        assert "--bbox=0.0,0.0,6.0,2.0" in calls[0]
        assert set(result["place"]["name"]) == {"left", "right"}

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
    @patch("city2graph.data.gpd.clip")
    @patch("city2graph.data.Path.exists")
    def test_polygon_clipping_rewrites_saved_file_when_not_returning_data(
        self,
        mock_exists: Mock,
        mock_clip: Mock,
        mock_read_file: Mock,
        mock_subprocess: Mock,
        test_polygon: Polygon,
    ) -> None:
        """Processed polygon queries should overwrite the saved GeoJSON on disk."""
        raw_gdf = gpd.GeoDataFrame(
            {"id": ["raw"]},
            geometry=[Polygon([(0, 0), (2, 0), (2, 2), (0, 0)])],
            crs=WGS84_CRS,
        )
        clipped_gdf = gpd.GeoDataFrame(
            {"id": ["clipped"]},
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])],
            crs=WGS84_CRS,
        )
        clipped_gdf.to_file = Mock()
        mock_exists.return_value = True
        mock_read_file.return_value = raw_gdf
        mock_clip.return_value = clipped_gdf

        result = load_overture_data(
            test_polygon,
            types=["building"],
            return_data=False,
            prefix="processed_",
        )

        assert result == {}
        clipped_gdf.to_file.assert_called_once()
        output_path = clipped_gdf.to_file.call_args.args[0]
        assert output_path.name == "processed_building.geojson"
        assert clipped_gdf.to_file.call_args.kwargs == {"driver": "GeoJSON"}
        mock_subprocess.assert_called_once()

    @patch("city2graph.data.subprocess.run")
    @patch("city2graph.data.gpd.read_file")
    @patch("city2graph.data.clip_graph")
    @patch("city2graph.data.Path.exists")
    def test_segment_polygon_clipping_uses_topological_subset(
        self,
        mock_exists: Mock,
        mock_clip_graph: Mock,
        mock_read_file: Mock,
        mock_subprocess: Mock,
        test_polygon: Polygon,
    ) -> None:
        """Segment clipping should use clip_graph instead of geometric clipping."""
        segment_gdf = gpd.GeoDataFrame(
            {"id": ["seg1"]},
            geometry=[LineString([(0, 0), (1, 1)])],
            crs=WGS84_CRS,
        )
        clipped_gdf = segment_gdf.copy()
        mock_exists.return_value = True
        mock_read_file.return_value = segment_gdf
        mock_clip_graph.return_value = clipped_gdf

        result = load_overture_data(test_polygon, types=["segment"])

        mock_clip_graph.assert_called_once()
        mock_subprocess.assert_called_once()
        assert result["segment"].equals(clipped_gdf)

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

    def test_empty_types_does_not_download_every_type(self, tmp_path: Path) -> None:
        """An explicit empty selection avoids geocoding, downloads and file writes."""
        output = tmp_path / "unused"
        with (
            patch("city2graph.data.get_boundaries") as geocode,
            patch("city2graph.data.subprocess.run") as download,
        ):
            geocode.return_value = gpd.GeoDataFrame(
                geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], crs=WGS84_CRS
            )
            result = load_overture_data(place_name="Example", types=[], output_dir=str(output))

        assert result == {}
        geocode.assert_not_called()
        download.assert_not_called()
        assert not output.exists()

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
                match=r"Invalid release: 2099-99-99.0. Valid releases are: 2025-08-20.0, 2025-08-20.1, 2025-09-24.0",
            ),
        ):
            load_overture_data(test_bbox, types=["building"], release=invalid_release)

    @patch("city2graph.data.subprocess.run")
    @patch("city2graph.data.gpd.read_file")
    def test_release_validation_skipped_when_catalogue_unreachable(
        self,
        mock_read_file: Mock,
        mock_subprocess: Mock,
        test_bbox: list[float],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """An unreachable release catalogue warns instead of rejecting the release.

        From overturemaps 1.0 ``ALL_RELEASES`` resolves lazily over the network,
        so listing it can raise. That must not turn a validation step into an
        unhandled connection error.
        """
        mock_gdf = Mock(spec=gpd.GeoDataFrame)
        mock_gdf.empty = False
        mock_read_file.return_value = mock_gdf

        with (
            patch.object(data_module, "ALL_RELEASES", UnreachableReleases()),
            caplog.at_level("WARNING"),
        ):
            result = load_overture_data(test_bbox, types=["building"], release="2025-08-20.1")

        assert "building" in result
        assert "skipping validation" in caplog.text
        mock_subprocess.assert_called_once()

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

    @patch("city2graph.data.Path.exists")
    @patch("city2graph.data.subprocess.run")
    @patch("city2graph.data.gpd.read_file")
    def test_segment_geometry_filtering(
        self,
        mock_read_file: Mock,
        mock_subprocess: Mock,
        mock_exists: Mock,
        test_bbox: list[float],
    ) -> None:
        """Test that non-LineString geometries are filtered from segments."""
        mock_exists.return_value = True
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

        mock_subprocess.assert_called()
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

    @pytest.mark.parametrize("use_stac", [False, True])
    @patch("city2graph.data.subprocess.run")
    def test_download_cli_options(
        self, run: Mock, test_bbox: list[float], tmp_path: Path, use_stac: bool
    ) -> None:
        """Forward integer timeouts and the STAC flag to the Overture CLI."""
        load_overture_data(
            test_bbox,
            types=["building"],
            output_dir=str(tmp_path),
            return_data=False,
            connect_timeout=5.0,
            request_timeout=10.5,
            use_stac=use_stac,
        )
        run.assert_called_once()
        args = run.call_args.args[0]
        assert args[args.index("--connect_timeout") + 1] == "5"
        assert args[args.index("--request_timeout") + 1] == "10"
        assert ("--no-stac" in args) == (not use_stac)

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
        assert result["id"].tolist() == segments_gdf["id"].tolist()
        assert result.geometry.equals(segments_gdf.geometry)

    def test_with_connectors(self) -> None:
        """Test process_overture_segments with connectors."""
        connectors_data = '[{"connector_id": "c1", "at": 0.25}, {"connector_id": "c2", "at": 0.75}]'
        segments_gdf = make_segments_gdf(
            ids=["seg1"],
            geoms_or_coords=[[(0, 0), (4, 0)]],
            connectors=connectors_data,
            level_rules="",
            crs="EPSG:27700",
        )
        connectors_gdf = make_connectors_gdf(
            ids=["c1", "c2"], coords=[(1, 0), (3, 0)], crs="EPSG:27700"
        )

        result = process_overture_segments(
            segments_gdf,
            connectors_gdf=connectors_gdf,
            get_barriers=False,
        )

        assert result["id"].tolist() == ["seg1_1", "seg1_2", "seg1_3"]
        assert result["split_from"].tolist() == [0.0, 0.25, 0.75]
        assert result["split_to"].tolist() == [0.25, 0.75, 1.0]
        assert result["length"].tolist() == pytest.approx([1.0, 2.0, 1.0])
        assert result.geometry.union_all().equals(segments_gdf.geometry.iloc[0])
        assert result.crs == segments_gdf.crs

    @pytest.mark.parametrize(
        ("level_rules", "passable_intervals"),
        [
            pytest.param("", [(0, 10)], id="no-rules"),
            pytest.param("invalid_json", [(0, 10)], id="invalid-json"),
            pytest.param('["not_a_dict"]', [(0, 10)], id="non-record"),
            pytest.param('[{"value": 1, "between": [0.5]}]', [(0, 10)], id="incomplete-interval"),
            pytest.param('[{"value": 0, "between": [0.2, 0.8]}]', [(0, 10)], id="ground-level"),
            pytest.param('[{"value": 1}]', [], id="whole-line"),
            pytest.param('[{"value": 1, "between": [0, 1]}]', [], id="full-interval"),
            pytest.param([{"value": 1, "between": [0.1, 0.5]}], [(0, 1), (5, 10)], id="list"),
            pytest.param(
                '[{"value": 1, "between": [0.1, 0.3]}, {"value": 1, "between": [0.7, 0.9]}]',
                [(0, 1), (3, 7), (9, 10)],
                id="disjoint",
            ),
            pytest.param(
                '[{"value": 1, "between": [0.1, 0.5]}, {"value": 1, "between": [0.3, 0.7]}]',
                [(0, 1), (7, 10)],
                id="overlapping",
            ),
            pytest.param(
                '[{"value": 1, "between": [0, 0.3]}, {"value": 1, "between": [0.3, 0.6]}]',
                [(6, 10)],
                id="touching",
            ),
        ],
    )
    def test_level_rules_determine_passable_geometry(
        self, level_rules: object, passable_intervals: list[tuple[int, int]]
    ) -> None:
        """Subtract the union of off-ground intervals from the original segment."""
        segments = make_segments_gdf(["road"], [[(0, 0), (10, 0)]], crs="EPSG:3857")
        segments["level_rules"] = [level_rules]

        result = process_overture_segments(segments, get_barriers=True)

        expected_parts = [LineString([(start, 0), (end, 0)]) for start, end in passable_intervals]
        geometry = result["barrier_geometry"].iloc[0]
        if not expected_parts:
            assert geometry is None
        elif len(expected_parts) == 1:
            assert geometry.equals_exact(expected_parts[0], tolerance=1e-9)
        else:
            assert geometry.equals_exact(MultiLineString(expected_parts), tolerance=1e-9)

    @pytest.mark.parametrize("level_rules", [None, [None]], ids=["missing-column", "null-value"])
    def test_missing_level_rules(self, level_rules: list[str | None] | None) -> None:
        """Normalize missing and null rules to empty strings."""
        segments = make_segments_gdf(
            None, [[(0, 0), (1, 0)]], level_rules=level_rules, crs="EPSG:3857"
        )

        result = process_overture_segments(segments, get_barriers=True)

        assert result["level_rules"].tolist() == [""]
        assert result["barrier_geometry"].tolist() == segments.geometry.tolist()

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

    @pytest.mark.parametrize("empty_table", [False, True], ids=["none", "empty-table"])
    def test_absent_connectors_leave_geometry_intact(self, empty_table: bool) -> None:
        """Keep source IDs and geometries when there are no join candidates."""
        segments = make_segments_gdf(None, [[(0, 0), (1, 0)], [(1, 0), (3, 0)]], crs="EPSG:3857")
        connectors = make_connectors_gdf([], [], crs="EPSG:3857") if empty_table else None

        result = process_overture_segments(segments, connectors_gdf=connectors, get_barriers=False)

        assert result["id"].tolist() == segments["id"].tolist()
        assert result.geometry.equals(segments.geometry)
        assert result["length"].tolist() == [1.0, 2.0]

    @pytest.mark.parametrize(
        ("connector_records", "expected_intervals"),
        [
            pytest.param("invalid_json", [(0, 2)], id="invalid-json"),
            pytest.param('{"invalid": "structure"}', [(0, 2)], id="missing-fields"),
            pytest.param('["not_a_dict"]', [(0, 2)], id="non-record"),
            pytest.param([{"connector_id": "c1", "at": 0.5}], [(0, 1), (1, 2)], id="list"),
            pytest.param('{"connector_id": "c1", "at": 0.5}', [(0, 1), (1, 2)], id="json-record"),
        ],
    )
    def test_connector_formats(
        self, connector_records: object, expected_intervals: list[tuple[int, int]]
    ) -> None:
        """Split at valid connectors and retain the whole line for invalid records."""
        segments = make_segments_gdf(["road"], [[(0, 0), (2, 0)]], crs="EPSG:3857")
        segments["connectors"] = [connector_records]
        connectors = make_connectors_gdf(["c1"], [(1, 0)], crs="EPSG:3857")

        result = process_overture_segments(segments, connectors_gdf=connectors, threshold=0.01)

        assert result["id"].tolist() == (
            ["road_1", "road_2"] if len(expected_intervals) == 2 else ["road"]
        )
        assert result.geometry.tolist() == [
            LineString([(start, 0), (end, 0)]) for start, end in expected_intervals
        ]

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

        assert result["id"].tolist() == ["seg1_1", "seg1_2"]
        assert result["split_from"].tolist() == [0.0, 0.5]
        assert result["split_to"].tolist() == [0.5, 1.0]
        assert result.geometry.tolist() == [
            LineString([(0, 0), (1, 1)]),
            LineString([(1, 1), (2, 2)]),
        ]

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

    def test_endpoint_clustering_ignores_all_non_linestring_geometries(self) -> None:
        """Endpoint clustering returns non-LineString-only frames unchanged."""
        segments_gdf = make_segments_gdf(
            ids=["point"],
            geoms_or_coords=[Point(2, 2)],
            level_rules="",
            crs="EPSG:3857",
        )
        connectors_gdf = make_connectors_gdf(
            ids=["unused"],
            coords=[(1, 1)],
            crs="EPSG:3857",
        )

        result = process_overture_segments(
            segments_gdf,
            connectors_gdf=connectors_gdf,
            get_barriers=False,
        )

        assert result.geometry.tolist() == segments_gdf.geometry.tolist()

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

    @pytest.mark.parametrize("bad_at", ["bad", "nan", "inf", -0.25, 1.25])
    def test_invalid_connector_position_cannot_create_a_road_part(
        self, bad_at: str | float
    ) -> None:
        """Keep the valid split and ignore an unusable linear reference."""
        segments = gpd.GeoDataFrame(
            {
                "id": ["road"],
                "connectors": [
                    [
                        {"connector_id": "valid", "at": "0.5"},
                        {"connector_id": "bad", "at": bad_at},
                    ]
                ],
            },
            geometry=[LineString([(0, 0), (8, 0)])],
            crs="EPSG:3857",
        )
        connectors = make_connectors_gdf(
            ids=["valid", "bad"], coords=[(4, 0), (6, 0)], crs="EPSG:3857"
        )
        result = process_overture_segments(
            segments, connectors_gdf=connectors, get_barriers=False, threshold=0.01
        )
        assert result["split_from"].tolist() == [0.0, 0.5]
        assert result["split_to"].tolist() == [0.5, 1.0]
        assert result.geometry.tolist() == [
            LineString([(0, 0), (4, 0)]),
            LineString([(4, 0), (8, 0)]),
        ]

    def test_split_barriers_keep_their_position_on_the_original_road(self) -> None:
        """A rule's source fractions must be translated to each child segment."""
        segments = make_segments_gdf(
            ids=["road"],
            geoms_or_coords=[[(0, 0), (10, 0)]],
            connectors='[{"connector_id": "middle", "at": 0.5}]',
            level_rules='[{"value": 1, "between": [0.3, 0.7]}]',
            crs="EPSG:3857",
        )
        connectors = make_connectors_gdf(ids=["middle"], coords=[(5, 0)], crs="EPSG:3857")
        result = process_overture_segments(segments, connectors_gdf=connectors, threshold=0.01)

        assert result["id"].tolist() == ["road_1", "road_2"]
        assert result["barrier_geometry"].tolist() == [
            LineString([(0, 0), (3, 0)]),
            LineString([(7, 0), (10, 0)]),
        ]

    def test_vectorized_splitting_preserves_order_attributes_and_geometry(self) -> None:
        """Batch splitting preserves exact row semantics and source attributes."""
        segments_gdf = gpd.GeoDataFrame(
            {
                "id": ["split", "unchanged"],
                "connectors": [
                    [
                        {"connector_id": "valid", "at": 0.75},
                        {"connector_id": "valid", "at": 0.25},
                        {"connector_id": "missing", "at": 0.5},
                    ],
                    [],
                ],
                "level_rules": ["", ""],
                "road_class": ["primary", "residential"],
            },
            geometry=[
                LineString([(0, 0), (4, 0)]),
                LineString([(10, 0), (12, 0)]),
            ],
            crs="EPSG:3857",
        )
        connectors_gdf = make_connectors_gdf(
            ids=["valid"],
            coords=[(1, 0)],
            crs="EPSG:3857",
        )

        result = process_overture_segments(
            segments_gdf,
            connectors_gdf=connectors_gdf,
            get_barriers=False,
            threshold=0.01,
        )

        assert result["id"].tolist() == ["split_1", "split_2", "split_3", "unchanged"]
        assert result["road_class"].tolist() == [
            "primary",
            "primary",
            "primary",
            "residential",
        ]
        assert result["split_from"].iloc[:3].tolist() == [0.0, 0.25, 0.75]
        assert result["split_to"].iloc[:3].tolist() == [0.25, 0.75, 1.0]
        assert result["split_from"].iloc[3:].isna().all()
        assert result.geometry.tolist() == [
            LineString([(0, 0), (1, 0)]),
            LineString([(1, 0), (3, 0)]),
            LineString([(3, 0), (4, 0)]),
            LineString([(10, 0), (12, 0)]),
        ]
        assert result.crs == segments_gdf.crs

    def test_vectorized_endpoint_snapping_preserves_interior_coordinates(self) -> None:
        """Endpoint replacement leaves every interior coordinate unchanged."""
        segments_gdf = gpd.GeoDataFrame(
            {
                "id": ["left", "right"],
                "connectors": [[], []],
                "level_rules": ["", ""],
            },
            geometry=[
                LineString([(0, 0), (0.5, 2), (1, 0)]),
                LineString([(1.2, 0), (1.5, 2), (2, 0)]),
            ],
            crs="EPSG:3857",
        )
        connectors_gdf = make_connectors_gdf(
            ids=["unused"],
            coords=[(0, 0)],
            crs="EPSG:3857",
        )

        result = process_overture_segments(
            segments_gdf,
            connectors_gdf=connectors_gdf,
            get_barriers=False,
            threshold=1.0,
        )

        assert list(result.geometry.iloc[0].coords) == [(0, 0), (0.5, 2), (1.1, 0)]
        assert list(result.geometry.iloc[1].coords) == [(1.1, 0), (1.5, 2), (2, 0)]

    def test_vectorized_barriers_preserve_indexes_and_null_geometries(self) -> None:
        """Barrier generation aligns custom indexes and retains null/empty values."""
        segments_gdf = gpd.GeoDataFrame(
            {
                "id": ["open", "missing", "empty", "blocked"],
                "level_rules": [
                    "",
                    "",
                    [{"value": 1, "between": [0.2, 0.4]}],
                    [{"value": 1}],
                ],
            },
            geometry=[
                LineString([(0, 0), (10, 0)]),
                None,
                LineString(),
                LineString([(20, 0), (30, 0)]),
            ],
            index=[10, 20, 30, 40],
            crs="EPSG:3857",
        )

        result = process_overture_segments(segments_gdf)

        assert result.index.tolist() == [10, 20, 30, 40]
        assert result["barrier_geometry"].loc[10] == segments_gdf.geometry.loc[10]
        assert result["barrier_geometry"].loc[20] is None
        assert result["barrier_geometry"].loc[30].is_empty
        assert result["barrier_geometry"].loc[40] is None

    def test_processing_does_not_iterate_over_geodataframe_rows(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The public post-processing path does not use GeoDataFrame.iterrows."""

        def fail_iterrows(_self: gpd.GeoDataFrame) -> None:
            msg = "GeoDataFrame.iterrows must not be used"
            raise AssertionError(msg)

        monkeypatch.setattr(gpd.GeoDataFrame, "iterrows", fail_iterrows)
        segments_gdf = gpd.GeoDataFrame(
            {
                "id": ["segment"],
                "connectors": [[{"connector_id": "valid", "at": 0.5}]],
                "level_rules": [[{"value": 1, "between": [0.2, 0.4]}]],
            },
            geometry=[LineString([(0, 0), (2, 0)])],
            crs="EPSG:3857",
        )
        connectors_gdf = make_connectors_gdf(
            ids=["valid"],
            coords=[(1, 0)],
            crs="EPSG:3857",
        )

        result = process_overture_segments(segments_gdf, connectors_gdf=connectors_gdf)

        assert len(result) == 2


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
