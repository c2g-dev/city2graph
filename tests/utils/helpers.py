"""Shared helpers for utility submodule tests."""

import geopandas as gpd

from tests import helpers


class BaseGraphTest:
    """Base class for graph-related utility tests."""

    @staticmethod
    def assert_valid_gdf(gdf: gpd.GeoDataFrame, expected_empty: bool = False) -> None:
        """Delegate GeoDataFrame assertions to the shared test helpers."""
        helpers.assert_valid_gdf(gdf, expected_empty)

    @staticmethod
    def assert_crs_consistency(*gdfs: gpd.GeoDataFrame) -> None:
        """Delegate CRS assertions to the shared test helpers."""
        helpers.assert_crs_consistency(*gdfs)


class BaseConversionTest(BaseGraphTest):
    """Base class for GeoDataFrame and NetworkX conversion tests."""

    def assert_roundtrip_consistency(
        self,
        original_nodes: gpd.GeoDataFrame,
        original_edges: gpd.GeoDataFrame,
        converted_nodes: gpd.GeoDataFrame,
        converted_edges: gpd.GeoDataFrame,
    ) -> None:
        """Delegate round-trip assertions to the shared test helpers."""
        helpers.assert_roundtrip_consistency(
            original_nodes,
            original_edges,
            converted_nodes,
            converted_edges,
        )
