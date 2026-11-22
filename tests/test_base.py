"""Tests for graph metadata serialisation and validation."""

import pytest

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
