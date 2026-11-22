"""Tests for graph metadata serialisation and validation."""

import geopandas as gpd
import pytest
from shapely.geometry import Point

from city2graph.base import GeoDataProcessor
from city2graph.base import GraphMetadata


class TestGraphMetadata:
    """Test GraphMetadata class functionality."""

    def test_metadata_creation_and_conversion(self) -> None:
        """Test GraphMetadata creation, conversion, and validation."""
        metadata = GraphMetadata(crs="EPSG:4326", is_hetero=True)
        metadata.node_types = ["building", "road"]
        metadata.edge_types = [("building", "connects", "road")]

        # Test to_dict conversion
        result_dict = metadata.to_dict()
        assert result_dict["crs"] == "EPSG:4326"
        assert result_dict["is_hetero"] is True
        assert result_dict["node_types"] == ["building", "road"]

        # Test from_dict creation
        recreated = GraphMetadata.from_dict(result_dict)
        assert recreated.crs == metadata.crs
        assert recreated.is_hetero == metadata.is_hetero
        assert recreated.node_types == metadata.node_types
        assert recreated.edge_types == metadata.edge_types

    @pytest.mark.parametrize(
        ("invalid_data", "error_type", "error_match"),
        [
            ({"crs": 123.45, "is_hetero": False}, TypeError, "CRS must be str, int, dict"),
            ({"crs": "EPSG:4326", "is_hetero": "not_bool"}, TypeError, "is_hetero must be bool"),
        ],
    )
    def test_metadata_validation_errors(
        self,
        invalid_data: dict[str, object],
        error_type: type[Exception],
        error_match: str,
    ) -> None:
        """Test GraphMetadata validation catches invalid inputs."""
        with pytest.raises(error_type, match=error_match):
            GraphMetadata.from_dict(invalid_data)

    def test_metadata_valid_crs_types(self) -> None:
        """Test GraphMetadata accepts various valid CRS formats."""
        valid_crs_values = ["EPSG:4326", 4326, {"init": "epsg:4326"}, None]

        for crs_value in valid_crs_values:
            metadata = GraphMetadata.from_dict({"crs": crs_value, "is_hetero": False})
            assert metadata.crs == crs_value


@pytest.fixture
def all_invalid_geom_gdf(sample_crs: str) -> gpd.GeoDataFrame:
    """Fixture for GeoDataFrame with all invalid geometries."""
    return gpd.GeoDataFrame(
        {
            "id": [1, 2],
            "geometry": [None, Point(0, 0).buffer(0).buffer(-1)],  # All invalid
        },
        crs=sample_crs,
    )


@pytest.fixture
def invalid_geom_gdf(sample_crs: str) -> gpd.GeoDataFrame:
    """Fixture for GeoDataFrame with mixed valid/invalid geometries."""
    return gpd.GeoDataFrame(
        {
            "id": [1, 2, 3, 4],
            "geometry": [
                Point(0, 0),
                None,
                Point(0, 0).buffer(0).buffer(-1),
                Point(1, 1),
            ],  # Mixed
        },
        crs=sample_crs,
    )


class TestGeoDataProcessor:
    """Tests for the public GeoDataProcessor validation methods."""

    def test_geo_processor_edge_cases(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        empty_gdf: gpd.GeoDataFrame,
        invalid_geom_gdf: gpd.GeoDataFrame,
        all_invalid_geom_gdf: gpd.GeoDataFrame,
    ) -> None:
        """GeoDataProcessor should handle geometry filtering and allow-empty toggles."""
        processor = GeoDataProcessor()

        result = processor.validate_gdf(
            sample_buildings_gdf,
            expected_geom_types=["Polygon", "MultiPolygon"],
        )
        assert isinstance(result, gpd.GeoDataFrame)

        assert processor.validate_gdf(empty_gdf, allow_empty=True) is not None
        with pytest.raises(ValueError, match="GeoDataFrame cannot be empty"):
            processor.validate_gdf(empty_gdf, allow_empty=False)

        filtered = processor.validate_gdf(invalid_geom_gdf)
        assert filtered is not None
        assert len(filtered) < len(invalid_geom_gdf)

        with pytest.raises(ValueError, match="GeoDataFrame cannot be empty"):
            processor.validate_gdf(all_invalid_geom_gdf, allow_empty=False)
