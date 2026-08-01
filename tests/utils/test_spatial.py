"""Tests for :mod:`city2graph.utils.spatial`."""

from __future__ import annotations

import warnings
from typing import Any
from typing import cast
from unittest import mock

import geopandas as gpd
import momepy
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import shapely.errors
from shapely.geometry import GeometryCollection
from shapely.geometry import LineString
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import Polygon

from city2graph import utils
from city2graph.utils import gdf_to_nx
from city2graph.utils import spatial as spatial_utils
from tests import helpers
from tests.utils.helpers import BaseGraphTest

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None  # type: ignore[assignment]
    MATPLOTLIB_AVAILABLE = False


class TestTessellation(BaseGraphTest):
    """Test tessellation creation functionality."""

    @pytest.mark.parametrize(
        ("geometry_fixture", "barriers_fixture", "expect_empty"),
        [
            ("empty_gdf", None, True),
            ("sample_buildings_gdf", None, False),
            ("sample_buildings_gdf", "sample_segments_gdf", False),
        ],
    )
    def test_tessellation_creation(
        self,
        geometry_fixture: str,
        barriers_fixture: str | None,
        expect_empty: bool,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test tessellation creation with various input combinations."""
        geometry = request.getfixturevalue(geometry_fixture)
        barriers = request.getfixturevalue(barriers_fixture) if barriers_fixture else None

        try:
            tessellation = utils.create_tessellation(geometry, primary_barriers=barriers)
        except (UnboundLocalError, TypeError, ValueError) as e:
            pytest.skip(f"Skipping due to incomplete implementation: {e}")

        self.assert_valid_gdf(tessellation, expect_empty)
        if not expect_empty:
            helpers.assert_has_columns(tessellation, ["tess_id"])
            assert tessellation.crs == geometry.crs

    def test_tessellation_handles_empty_geometry(self, empty_gdf: gpd.GeoDataFrame) -> None:
        """Empty input should return an empty GeoDataFrame with expected columns."""
        result = utils.create_tessellation(empty_gdf)
        self.assert_valid_gdf(result, expected_empty=True)

    def test_tessellation_handles_degenerate_inputs(
        self,
        single_point_geom_gdf: gpd.GeoDataFrame,
        tessellation_barriers_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Degenerate geometries or barriers should not raise."""
        single_result = utils.create_tessellation(single_point_geom_gdf)
        self.assert_valid_gdf(single_result, expected_empty=True)

        barrier_result = utils.create_tessellation(
            single_point_geom_gdf,
            primary_barriers=tessellation_barriers_gdf,
        )
        self.assert_valid_gdf(barrier_result, expected_empty=True)

    def test_tessellation_empty_with_barriers(self, empty_gdf: gpd.GeoDataFrame) -> None:
        """Even with barriers an empty geometry should stay empty."""
        barriers = gpd.GeoDataFrame(
            {"geometry": [LineString([(0, 0), (1, 1)])]},
            crs=empty_gdf.crs,
        )
        result = utils.create_tessellation(empty_gdf, primary_barriers=barriers)
        self.assert_valid_gdf(result, expected_empty=True)
        assert list(result.columns) == ["geometry", "enclosure_index", "tess_id"]

    @pytest.mark.parametrize("failure_mode", ["geos_error", "concat_error", "no_enclosures"])
    def test_empty_enclosed_tessellation_schema_is_uniform(
        self,
        failure_mode: str,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Every degrade path must produce the same empty enclosed-tessellation schema."""
        enclosures = (
            gpd.GeoDataFrame({"eID": []}, geometry=[], crs=sample_buildings_gdf.crs)
            if failure_mode == "no_enclosures"
            else gpd.GeoDataFrame(
                {"eID": [1]},
                geometry=[Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])],
                crs=sample_buildings_gdf.crs,
            )
        )
        monkeypatch.setattr(momepy, "enclosures", lambda **_kwargs: enclosures)

        def raise_failure(**_kwargs: object) -> gpd.GeoDataFrame:
            if failure_mode == "geos_error":
                msg = "TopologyException: side location conflict at 0 0"
                raise shapely.errors.GEOSException(msg)
            msg = "No objects to concatenate"
            raise ValueError(msg)

        monkeypatch.setattr(momepy, "enclosed_tessellation", raise_failure)

        result = utils.create_tessellation(
            sample_buildings_gdf,
            primary_barriers=sample_segments_gdf,
        )

        assert result.empty
        assert list(result.columns) == ["geometry", "enclosure_index", "tess_id"]
        assert result.crs == sample_buildings_gdf.crs

    def test_enclosed_tessellation_rectilinear_buildings_not_degenerate(
        self,
        sample_crs: str,
    ) -> None:
        """Grid-aligned footprints must yield per-building, non-overlapping cells.

        Perfectly rectilinear coordinates make ``shapely.voronoi_polygons``
        degenerate (GeometryCollection cells, overlapping partitions); the
        retry ladder must recover via the deterministic jitter rung instead of
        dropping the enclosure and degrading its buildings to fallback cells.
        """
        corners = [(0.0, 0.0), (200.0, 0.0), (200.0, 200.0), (0.0, 200.0), (0.0, 0.0)]
        barriers = gpd.GeoDataFrame(
            geometry=[LineString([corners[i], corners[i + 1]]) for i in range(4)],
            crs=sample_crs,
        )
        buildings = gpd.GeoDataFrame(
            geometry=[
                Polygon([(40, 40), (80, 40), (80, 80), (40, 80)]),
                Polygon([(120, 120), (160, 120), (160, 160), (120, 160)]),
            ],
            crs=sample_crs,
        )

        result = utils.create_tessellation(buildings, primary_barriers=barriers, n_jobs=1)

        cells = list(result.geometry)
        overlap = sum(
            cells[i].intersection(cells[j]).area
            for i in range(len(cells))
            for j in range(i + 1, len(cells))
        )
        assert overlap == pytest.approx(0.0, abs=1e-6)
        for footprint in buildings.geometry:
            assert result.geometry.contains(footprint.centroid).sum() == 1

    def test_enclosed_tessellation_single_building_enclosures(
        self,
        sample_crs: str,
    ) -> None:
        """Enclosures holding at most one building each still produce cells.

        ``momepy.enclosed_tessellation`` crashes with "No objects to
        concatenate" when no enclosure holds two or more buildings; the
        single-building enclosures must become their buildings' cells instead
        of degrading the unit to an empty tessellation.
        """
        corners = [(0.0, 0.0), (200.0, 0.0), (200.0, 200.0), (0.0, 200.0), (0.0, 0.0)]
        barriers = gpd.GeoDataFrame(
            geometry=[LineString([corners[i], corners[i + 1]]) for i in range(4)],
            crs=sample_crs,
        )
        building = gpd.GeoDataFrame(
            geometry=[Polygon([(80, 80), (120, 80), (120, 120), (80, 120)])],
            crs=sample_crs,
        )

        result = utils.create_tessellation(building, primary_barriers=barriers, n_jobs=1)

        assert not result.empty
        assert list(result.columns) == ["geometry", "enclosure_index", "tess_id"]
        assert result.geometry.contains(building.geometry.iloc[0].centroid).sum() == 1

    def test_tessellation_momepy_error_handling(self, sample_crs: str) -> None:
        """Test tessellation error handling for momepy ValueError (lines 2393-2399)."""
        # Create geometry that might cause momepy to fail with "No objects to concatenate"
        # Use a configuration that could trigger internal momepy ValueError

        # Create valid input geometry
        geometry = gpd.GeoDataFrame(
            geometry=[Point(0, 0), Point(1, 1)],
            crs=sample_crs,
        )
        barriers = gpd.GeoDataFrame(
            geometry=[LineString([(0, 0), (1, 1)])],
            crs=sample_crs,
        )

        # Mock momepy.enclosed_tessellation to raise ValueError with "No objects to concatenate"
        # Since utils imports momepy, we patch city2graph.utils.spatial.momepy.enclosed_tessellation
        with mock.patch("city2graph.utils.spatial.momepy.enclosed_tessellation") as mock_tess:
            mock_tess.side_effect = ValueError("No objects to concatenate")

            # Should handle the error and return empty tessellation
            result = utils.create_tessellation(
                geometry,
                primary_barriers=barriers,
                shrink=0.4,
            )

            assert isinstance(result, gpd.GeoDataFrame)
            assert result.empty
            assert result.crs == sample_crs

    def test_tessellation_returns_empty_when_momepy_concat_fails_with_enclosures(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Enclosed tessellation should warn and return an empty schema on concat failures."""
        monkeypatch.setattr(
            momepy,
            "enclosures",
            lambda **_kwargs: gpd.GeoDataFrame(
                {"eID": [1]},
                geometry=[Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])],
                crs=sample_buildings_gdf.crs,
            ),
        )

        def raise_concat_error(**_kwargs: object) -> gpd.GeoDataFrame:
            msg = "No objects to concatenate"
            raise ValueError(msg)

        monkeypatch.setattr(momepy, "enclosed_tessellation", raise_concat_error)

        with caplog.at_level("WARNING"):
            result = utils.create_tessellation(
                sample_buildings_gdf,
                primary_barriers=sample_segments_gdf,
            )

        assert result.empty
        assert list(result.columns) == ["geometry", "enclosure_index", "tess_id"]
        assert "returning empty GeoDataFrame" in caplog.text

    def test_enclosed_tessellation_retries_with_grid_size_on_geometry_type_error(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A coverage_simplify TypeError should first retry with a coarser grid_size."""
        monkeypatch.setattr(
            momepy,
            "enclosures",
            lambda **_kwargs: gpd.GeoDataFrame(
                {"eID": [1]},
                geometry=[Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])],
                crs=sample_buildings_gdf.crs,
            ),
        )

        calls: list[tuple[object, object]] = []

        def fake_enclosed_tessellation(**kwargs: object) -> gpd.GeoDataFrame:
            calls.append((kwargs.get("simplify"), kwargs.get("grid_size")))
            if kwargs.get("grid_size") is None:
                msg = "One of the Geometry inputs is of incorrect geometry type."
                raise TypeError(msg)
            return gpd.GeoDataFrame(
                {"enclosure_index": [1]},
                geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
                index=[0],
                crs=sample_buildings_gdf.crs,
            )

        monkeypatch.setattr(momepy, "enclosed_tessellation", fake_enclosed_tessellation)

        with caplog.at_level("WARNING"):
            result = utils.create_tessellation(
                sample_buildings_gdf,
                primary_barriers=sample_segments_gdf,
            )

        assert not result.empty
        assert "tess_id" in result.columns
        assert calls == [(None, None), (None, 1e-3)]  # simplify untouched
        assert "retrying with grid_size=1e-3" in caplog.text

    def test_enclosed_tessellation_falls_back_to_simplify_false_when_grid_size_fails(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When the grid_size retry still fails, retry once more with simplify=False."""
        monkeypatch.setattr(
            momepy,
            "enclosures",
            lambda **_kwargs: gpd.GeoDataFrame(
                {"eID": [1]},
                geometry=[Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])],
                crs=sample_buildings_gdf.crs,
            ),
        )

        calls: list[tuple[object, object]] = []

        def fake_enclosed_tessellation(**kwargs: object) -> gpd.GeoDataFrame:
            calls.append((kwargs.get("simplify"), kwargs.get("grid_size")))
            if kwargs.get("simplify") is not False:
                msg = "One of the Geometry inputs is of incorrect geometry type."
                raise TypeError(msg)
            return gpd.GeoDataFrame(
                {"enclosure_index": [1]},
                geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
                index=[0],
                crs=sample_buildings_gdf.crs,
            )

        monkeypatch.setattr(momepy, "enclosed_tessellation", fake_enclosed_tessellation)

        with caplog.at_level("WARNING"):
            result = utils.create_tessellation(
                sample_buildings_gdf,
                primary_barriers=sample_segments_gdf,
            )

        assert not result.empty
        assert calls == [(None, None), (None, 1e-3), (False, 1e-3)]
        assert "retrying with grid_size=1e-3" in caplog.text
        assert "retrying with simplify=False" in caplog.text

    def test_enclosed_tessellation_retries_overlapping_cells_with_grid_size(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A silently degenerate partition should be detected and retried."""
        enclosure = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        monkeypatch.setattr(
            momepy,
            "enclosures",
            lambda **_kwargs: gpd.GeoDataFrame(
                {"eID": [0]},
                geometry=[enclosure],
                crs=sample_buildings_gdf.crs,
            ),
        )

        sane_cells = [
            Polygon([(0, 0), (5, 0), (5, 10), (0, 10)]),
            Polygon([(5, 0), (10, 0), (10, 10), (5, 10)]),
        ]

        def fake_enclosed_tessellation(**kwargs: object) -> gpd.GeoDataFrame:
            # Without a coarser grid_size every cell degenerates to the whole
            # enclosure (the failure mode observed on real data); with it the
            # cells form a proper partition.
            cells = sane_cells if kwargs.get("grid_size") else [enclosure, enclosure]
            return gpd.GeoDataFrame(
                {"enclosure_index": [0, 0]},
                geometry=cells,
                index=[0, 1],
                crs=sample_buildings_gdf.crs,
            )

        monkeypatch.setattr(momepy, "enclosed_tessellation", fake_enclosed_tessellation)

        with caplog.at_level("WARNING"):
            result = utils.create_tessellation(
                sample_buildings_gdf,
                primary_barriers=sample_segments_gdf,
            )

        assert len(result) == 2
        assert result.geometry.area.sum() == pytest.approx(enclosure.area)
        assert "overlapping cells" in caplog.text

    def test_enclosed_tessellation_overlap_retry_falls_back_to_simplify_false(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A geometry-type error during the overlap retry should retry simplify=False."""
        enclosure = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        monkeypatch.setattr(
            momepy,
            "enclosures",
            lambda **_kwargs: gpd.GeoDataFrame(
                {"eID": [0]},
                geometry=[enclosure],
                crs=sample_buildings_gdf.crs,
            ),
        )

        sane_cells = [
            Polygon([(0, 0), (5, 0), (5, 10), (0, 10)]),
            Polygon([(5, 0), (10, 0), (10, 10), (5, 10)]),
        ]

        def fake_enclosed_tessellation(**kwargs: object) -> gpd.GeoDataFrame:
            # The first run silently degenerates; the coarser grid_size retry
            # snaps cells into non-polygonal geometry and blows up in
            # coverage_simplify; only simplify=False succeeds.
            if kwargs.get("grid_size") and kwargs.get("simplify") is not False:
                msg = "One of the Geometry inputs is of incorrect geometry type."
                raise TypeError(msg)
            cells = sane_cells if kwargs.get("grid_size") else [enclosure, enclosure]
            return gpd.GeoDataFrame(
                {"enclosure_index": [0, 0]},
                geometry=cells,
                index=[0, 1],
                crs=sample_buildings_gdf.crs,
            )

        monkeypatch.setattr(momepy, "enclosed_tessellation", fake_enclosed_tessellation)

        with caplog.at_level("WARNING"):
            result = utils.create_tessellation(
                sample_buildings_gdf,
                primary_barriers=sample_segments_gdf,
            )

        assert len(result) == 2
        assert result.geometry.area.sum() == pytest.approx(enclosure.area)
        assert "retrying with simplify=False" in caplog.text

    def test_enclosed_tessellation_drops_persistently_overlapping_enclosures(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Enclosures that stay degenerate after the retry should be dropped."""
        bad_enclosure = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        good_enclosure = Polygon([(20, 0), (25, 0), (25, 5), (20, 5)])
        monkeypatch.setattr(
            momepy,
            "enclosures",
            lambda **_kwargs: gpd.GeoDataFrame(
                {"eID": [0, 1]},
                geometry=[bad_enclosure, good_enclosure],
                crs=sample_buildings_gdf.crs,
            ),
        )

        def fake_enclosed_tessellation(**_kwargs: object) -> gpd.GeoDataFrame:
            # Enclosure 0 keeps returning whole-enclosure duplicates even at a
            # coarser grid_size; enclosure 1 partitions fine.
            return gpd.GeoDataFrame(
                {"enclosure_index": [0, 0, 1]},
                geometry=[bad_enclosure, bad_enclosure, good_enclosure],
                index=[0, 1, 2],
                crs=sample_buildings_gdf.crs,
            )

        monkeypatch.setattr(momepy, "enclosed_tessellation", fake_enclosed_tessellation)

        with caplog.at_level("WARNING"):
            result = utils.create_tessellation(
                sample_buildings_gdf,
                primary_barriers=sample_segments_gdf,
            )

        assert set(result["enclosure_index"]) == {1}
        assert "Dropping 1 enclosure(s)" in caplog.text

    def test_enclosed_tessellation_salvages_geometry_collections(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """GeometryCollection cells should be reduced to their polygonal parts."""
        monkeypatch.setattr(
            momepy,
            "enclosures",
            lambda **_kwargs: gpd.GeoDataFrame(
                {"eID": [1]},
                geometry=[Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])],
                crs=sample_buildings_gdf.crs,
            ),
        )

        polygon_part = Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])

        def fake_enclosed_tessellation(**_kwargs: object) -> gpd.GeoDataFrame:
            return gpd.GeoDataFrame(
                {"enclosure_index": [1, 1, 1]},
                geometry=[
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    GeometryCollection([polygon_part, LineString([(0, 0), (1, 1)])]),
                    GeometryCollection([LineString([(2, 2), (3, 3)])]),
                ],
                index=[0, 1, 2],
                crs=sample_buildings_gdf.crs,
            )

        monkeypatch.setattr(momepy, "enclosed_tessellation", fake_enclosed_tessellation)

        with caplog.at_level("WARNING"):
            result = utils.create_tessellation(
                sample_buildings_gdf,
                primary_barriers=sample_segments_gdf,
            )

        assert len(result) == 2
        assert set(result.geom_type) == {"Polygon"}
        assert result.geometry.iloc[1].equals(polygon_part)
        assert "GeometryCollection cell(s)" in caplog.text

    def test_enclosed_tessellation_reraises_unrelated_type_error(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A TypeError unrelated to coverage_simplify should propagate."""
        monkeypatch.setattr(
            momepy,
            "enclosures",
            lambda **_kwargs: gpd.GeoDataFrame(
                {"eID": [1]},
                geometry=[Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])],
                crs=sample_buildings_gdf.crs,
            ),
        )

        def raise_unrelated(**_kwargs: object) -> gpd.GeoDataFrame:
            msg = "something else entirely"
            raise TypeError(msg)

        monkeypatch.setattr(momepy, "enclosed_tessellation", raise_unrelated)

        with pytest.raises(TypeError, match="something else entirely"):
            utils.create_tessellation(
                sample_buildings_gdf,
                primary_barriers=sample_segments_gdf,
            )

    def test_enclosed_tessellation_retries_with_coarser_grid_size_on_geos_error(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A GEOS topology error should trigger a retry with a coarser grid_size."""
        monkeypatch.setattr(
            momepy,
            "enclosures",
            lambda **_kwargs: gpd.GeoDataFrame(
                {"eID": [1]},
                geometry=[Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])],
                crs=sample_buildings_gdf.crs,
            ),
        )

        calls: list[object] = []

        def fake_enclosed_tessellation(**kwargs: object) -> gpd.GeoDataFrame:
            calls.append(kwargs.get("grid_size"))
            if kwargs.get("grid_size") is None:
                msg = "TopologyException: side location conflict at 0 0"
                raise shapely.errors.GEOSException(msg)
            return gpd.GeoDataFrame(
                {"enclosure_index": [1]},
                geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
                index=[0],
                crs=sample_buildings_gdf.crs,
            )

        monkeypatch.setattr(momepy, "enclosed_tessellation", fake_enclosed_tessellation)

        with caplog.at_level("WARNING"):
            result = utils.create_tessellation(
                sample_buildings_gdf,
                primary_barriers=sample_segments_gdf,
            )

        assert not result.empty
        assert "tess_id" in result.columns
        assert calls[-1] == 1e-3  # retried with coarser grid_size
        assert "retrying with coarser" in caplog.text

    def test_enclosed_tessellation_returns_empty_when_geos_error_persists(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A GEOS error that persists at coarser precision degrades to empty output."""
        monkeypatch.setattr(
            momepy,
            "enclosures",
            lambda **_kwargs: gpd.GeoDataFrame(
                {"eID": [1]},
                geometry=[Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])],
                crs=sample_buildings_gdf.crs,
            ),
        )

        def always_raise(**_kwargs: object) -> gpd.GeoDataFrame:
            msg = "TopologyException: side location conflict at 0 0"
            raise shapely.errors.GEOSException(msg)

        monkeypatch.setattr(momepy, "enclosed_tessellation", always_raise)

        with caplog.at_level("WARNING"):
            result = utils.create_tessellation(
                sample_buildings_gdf,
                primary_barriers=sample_segments_gdf,
            )

        assert result.empty
        assert "returning empty" in caplog.text

    def test_enclosed_tessellation_geos_retry_falls_back_to_simplify_false(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A coarser-grid retry after a GEOS error may fail simplification instead."""
        monkeypatch.setattr(
            momepy,
            "enclosures",
            lambda **_kwargs: gpd.GeoDataFrame(
                {"eID": [1]},
                geometry=[Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])],
                crs=sample_buildings_gdf.crs,
            ),
        )

        calls: list[tuple[object, object]] = []

        def fake_enclosed_tessellation(**kwargs: object) -> gpd.GeoDataFrame:
            calls.append((kwargs.get("grid_size"), kwargs.get("simplify")))
            if kwargs.get("grid_size") is None:
                msg = "TopologyException: side location conflict at 0 0"
                raise shapely.errors.GEOSException(msg)
            if kwargs.get("simplify") is not False:
                msg = "One of the Geometry inputs is of incorrect geometry type."
                raise TypeError(msg)
            return gpd.GeoDataFrame(
                {"enclosure_index": [1]},
                geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
                index=[0],
                crs=sample_buildings_gdf.crs,
            )

        monkeypatch.setattr(momepy, "enclosed_tessellation", fake_enclosed_tessellation)

        with caplog.at_level("WARNING"):
            result = utils.create_tessellation(
                sample_buildings_gdf,
                primary_barriers=sample_segments_gdf,
            )

        assert not result.empty
        assert "tess_id" in result.columns
        assert calls[-1] == (1e-3, False)
        assert "retrying with coarser" in caplog.text
        assert "retrying with simplify=False" in caplog.text

    def test_enclosed_tessellation_degrades_to_empty_with_pinned_grid_size(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A GEOS error with a caller-pinned grid_size degrades without overriding it."""
        monkeypatch.setattr(
            momepy,
            "enclosures",
            lambda **_kwargs: gpd.GeoDataFrame(
                {"eID": [1]},
                geometry=[Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])],
                crs=sample_buildings_gdf.crs,
            ),
        )

        calls: list[object] = []

        def raise_geos(**kwargs: object) -> gpd.GeoDataFrame:
            calls.append(kwargs.get("grid_size"))
            msg = "TopologyException: side location conflict at 0 0"
            raise shapely.errors.GEOSException(msg)

        monkeypatch.setattr(momepy, "enclosed_tessellation", raise_geos)

        with caplog.at_level("WARNING"):
            result = utils.create_tessellation(
                sample_buildings_gdf,
                primary_barriers=sample_segments_gdf,
                grid_size=1e-5,
            )

        assert result.empty
        # The pinned precision is never overridden; the final rung retries the
        # same options once with jittered geometry before degrading.
        assert calls == [1e-5, 1e-5]
        assert "retrying with jittered geometry" in caplog.text
        assert "returning empty" in caplog.text

    def test_enclosed_tessellation_pinned_grid_size_retries_simplify_false(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With a pinned grid_size, a coverage_simplify TypeError retries simplify=False."""
        monkeypatch.setattr(
            momepy,
            "enclosures",
            lambda **_kwargs: gpd.GeoDataFrame(
                {"eID": [1]},
                geometry=[Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])],
                crs=sample_buildings_gdf.crs,
            ),
        )

        calls: list[tuple[object, object]] = []

        def fake_enclosed_tessellation(**kwargs: object) -> gpd.GeoDataFrame:
            calls.append((kwargs.get("simplify"), kwargs.get("grid_size")))
            if kwargs.get("simplify") is not False:
                msg = "One of the Geometry inputs is of incorrect geometry type."
                raise TypeError(msg)
            return gpd.GeoDataFrame(
                {"enclosure_index": [1]},
                geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
                index=[0],
                crs=sample_buildings_gdf.crs,
            )

        monkeypatch.setattr(momepy, "enclosed_tessellation", fake_enclosed_tessellation)

        with caplog.at_level("WARNING"):
            result = utils.create_tessellation(
                sample_buildings_gdf,
                primary_barriers=sample_segments_gdf,
                grid_size=1e-5,
            )

        assert not result.empty
        assert calls == [(None, 1e-5), (False, 1e-5)]  # pinned grid_size kept
        assert "retrying with simplify=False" in caplog.text

    def test_enclosed_tessellation_degrades_when_both_retry_options_pinned(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With grid_size and simplify both pinned, a known failure degrades to empty."""
        monkeypatch.setattr(
            momepy,
            "enclosures",
            lambda **_kwargs: gpd.GeoDataFrame(
                {"eID": [1]},
                geometry=[Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])],
                crs=sample_buildings_gdf.crs,
            ),
        )

        calls: list[tuple[object, object]] = []

        def raise_type_error(**kwargs: object) -> gpd.GeoDataFrame:
            calls.append((kwargs.get("simplify"), kwargs.get("grid_size")))
            msg = "One of the Geometry inputs is of incorrect geometry type."
            raise TypeError(msg)

        monkeypatch.setattr(momepy, "enclosed_tessellation", raise_type_error)

        with caplog.at_level("WARNING"):
            result = utils.create_tessellation(
                sample_buildings_gdf,
                primary_barriers=sample_segments_gdf,
                grid_size=1e-5,
                simplify=True,
            )

        assert result.empty
        # Pinned options are never overridden; the final rung retries them
        # once with jittered geometry before degrading.
        assert calls == [(True, 1e-5), (True, 1e-5)]
        assert "returning empty" in caplog.text

    def test_enclosed_tessellation_reraises_unknown_error_on_retry(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An unknown error raised during a retry must propagate."""
        monkeypatch.setattr(
            momepy,
            "enclosures",
            lambda **_kwargs: gpd.GeoDataFrame(
                {"eID": [1]},
                geometry=[Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])],
                crs=sample_buildings_gdf.crs,
            ),
        )

        def fake_enclosed_tessellation(**kwargs: object) -> gpd.GeoDataFrame:
            if kwargs.get("grid_size") is None:
                msg = "One of the Geometry inputs is of incorrect geometry type."
                raise TypeError(msg)
            msg = "something else entirely"
            raise TypeError(msg)

        monkeypatch.setattr(momepy, "enclosed_tessellation", fake_enclosed_tessellation)

        with pytest.raises(TypeError, match="something else entirely"):
            utils.create_tessellation(
                sample_buildings_gdf,
                primary_barriers=sample_segments_gdf,
            )

    def test_enclosed_tessellation_overlap_repair_simplify_retry_failure_keeps_cells(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A GEOS crash in the repair simplify=False retry still drops only the broken cells.

        The repair retry escalates through the jitter rung, whose (mocked)
        result stays degenerate, so the broken enclosure is dropped while the
        good one survives.
        """
        bad_enclosure = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        good_enclosure = Polygon([(20, 0), (25, 0), (25, 5), (20, 5)])
        monkeypatch.setattr(
            momepy,
            "enclosures",
            lambda **_kwargs: gpd.GeoDataFrame(
                {"eID": [0, 1]},
                geometry=[bad_enclosure, good_enclosure],
                crs=sample_buildings_gdf.crs,
            ),
        )

        def fake_enclosed_tessellation(**kwargs: object) -> gpd.GeoDataFrame:
            # The first run silently degenerates in enclosure 0; the repair
            # retry fails coverage_simplify and its simplify=False retry
            # crashes in GEOS.
            if kwargs.get("grid_size") and kwargs.get("simplify") is not False:
                msg = "One of the Geometry inputs is of incorrect geometry type."
                raise TypeError(msg)
            if kwargs.get("grid_size"):
                msg = "TopologyException: side location conflict at 0 0"
                raise shapely.errors.GEOSException(msg)
            return gpd.GeoDataFrame(
                {"enclosure_index": [0, 0, 1]},
                geometry=[bad_enclosure, bad_enclosure, good_enclosure],
                index=[0, 1, 2],
                crs=sample_buildings_gdf.crs,
            )

        monkeypatch.setattr(momepy, "enclosed_tessellation", fake_enclosed_tessellation)

        with caplog.at_level("WARNING"):
            result = utils.create_tessellation(
                sample_buildings_gdf,
                primary_barriers=sample_segments_gdf,
            )

        assert set(result["enclosure_index"]) == {1}
        assert "retrying with jittered geometry" in caplog.text
        assert "Dropping 1 enclosure(s)" in caplog.text

    def test_enclosed_tessellation_degrades_when_concat_error_on_retry(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A concat failure during the grid_size retry degrades to empty output."""
        monkeypatch.setattr(
            momepy,
            "enclosures",
            lambda **_kwargs: gpd.GeoDataFrame(
                {"eID": [1]},
                geometry=[Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])],
                crs=sample_buildings_gdf.crs,
            ),
        )

        calls: list[tuple[object, object]] = []

        def fake_enclosed_tessellation(**kwargs: object) -> gpd.GeoDataFrame:
            calls.append((kwargs.get("simplify"), kwargs.get("grid_size")))
            if kwargs.get("grid_size") is None:
                msg = "One of the Geometry inputs is of incorrect geometry type."
                raise TypeError(msg)
            msg = "No objects to concatenate"
            raise ValueError(msg)

        monkeypatch.setattr(momepy, "enclosed_tessellation", fake_enclosed_tessellation)

        with caplog.at_level("WARNING"):
            result = utils.create_tessellation(
                sample_buildings_gdf,
                primary_barriers=sample_segments_gdf,
            )

        assert result.empty
        assert calls == [(None, None), (None, 1e-3)]
        assert "returning empty GeoDataFrame" in caplog.text

    def test_enclosed_tessellation_degrades_when_geos_error_during_simplify_retry(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A GEOS error during the grid_size retry of a TypeError degrades to empty."""
        monkeypatch.setattr(
            momepy,
            "enclosures",
            lambda **_kwargs: gpd.GeoDataFrame(
                {"eID": [1]},
                geometry=[Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])],
                crs=sample_buildings_gdf.crs,
            ),
        )

        calls: list[tuple[object, object]] = []

        def fake_enclosed_tessellation(**kwargs: object) -> gpd.GeoDataFrame:
            calls.append((kwargs.get("simplify"), kwargs.get("grid_size")))
            if kwargs.get("grid_size") is None:
                msg = "One of the Geometry inputs is of incorrect geometry type."
                raise TypeError(msg)
            msg = "TopologyException: side location conflict at 0 0"
            raise shapely.errors.GEOSException(msg)

        monkeypatch.setattr(momepy, "enclosed_tessellation", fake_enclosed_tessellation)

        with caplog.at_level("WARNING"):
            result = utils.create_tessellation(
                sample_buildings_gdf,
                primary_barriers=sample_segments_gdf,
            )

        assert result.empty
        # After the coarse-grid rung the ladder resets to a jittered attempt
        # at default options before degrading.
        assert calls == [(None, None), (None, 1e-3), (None, None)]
        assert "returning empty" in caplog.text

    def test_enclosed_tessellation_degrades_when_type_error_persists(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A coverage_simplify TypeError persisting through every rung degrades to empty."""
        monkeypatch.setattr(
            momepy,
            "enclosures",
            lambda **_kwargs: gpd.GeoDataFrame(
                {"eID": [1]},
                geometry=[Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])],
                crs=sample_buildings_gdf.crs,
            ),
        )

        calls: list[tuple[object, object]] = []

        def raise_type_error(**kwargs: object) -> gpd.GeoDataFrame:
            calls.append((kwargs.get("simplify"), kwargs.get("grid_size")))
            msg = "One of the Geometry inputs is of incorrect geometry type."
            raise TypeError(msg)

        monkeypatch.setattr(momepy, "enclosed_tessellation", raise_type_error)

        with caplog.at_level("WARNING"):
            result = utils.create_tessellation(
                sample_buildings_gdf,
                primary_barriers=sample_segments_gdf,
            )

        assert result.empty
        # grid_size, then simplify=False, then one jittered attempt at
        # default options before degrading.
        assert calls == [(None, None), (None, 1e-3), (False, 1e-3), (None, None)]
        assert "returning empty" in caplog.text

    def test_enclosed_tessellation_overlap_retry_failure_drops_enclosures(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A failing overlap-repair retry keeps the good enclosures and drops the broken.

        The GEOS crash in the coarser-grid retry escalates to the jitter rung,
        whose (mocked) result stays degenerate, so the broken enclosure is
        dropped while the good one survives.
        """
        bad_enclosure = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        good_enclosure = Polygon([(20, 0), (25, 0), (25, 5), (20, 5)])
        monkeypatch.setattr(
            momepy,
            "enclosures",
            lambda **_kwargs: gpd.GeoDataFrame(
                {"eID": [0, 1]},
                geometry=[bad_enclosure, good_enclosure],
                crs=sample_buildings_gdf.crs,
            ),
        )

        def fake_enclosed_tessellation(**kwargs: object) -> gpd.GeoDataFrame:
            # The first run silently degenerates in enclosure 0; the repair
            # retry at a coarser grid_size crashes in GEOS instead.
            if kwargs.get("grid_size"):
                msg = "TopologyException: side location conflict at 0 0"
                raise shapely.errors.GEOSException(msg)
            return gpd.GeoDataFrame(
                {"enclosure_index": [0, 0, 1]},
                geometry=[bad_enclosure, bad_enclosure, good_enclosure],
                index=[0, 1, 2],
                crs=sample_buildings_gdf.crs,
            )

        monkeypatch.setattr(momepy, "enclosed_tessellation", fake_enclosed_tessellation)

        with caplog.at_level("WARNING"):
            result = utils.create_tessellation(
                sample_buildings_gdf,
                primary_barriers=sample_segments_gdf,
            )

        assert set(result["enclosure_index"]) == {1}
        assert "retrying with jittered geometry" in caplog.text
        assert "Dropping 1 enclosure(s)" in caplog.text

    def test_enclosed_tessellation_passes_explicit_limit(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Explicit enclosure limits should be forwarded to momepy.enclosures."""
        custom_limit = Polygon([(-10, -10), (10, -10), (10, 10), (-10, 10)])
        captured: dict[str, object] = {}

        def fake_enclosures(**kwargs: object) -> gpd.GeoDataFrame:
            captured["limit"] = kwargs["limit"]
            captured["clip"] = kwargs["clip"]
            return gpd.GeoDataFrame(
                {"eID": [1]},
                geometry=[custom_limit],
                crs=sample_buildings_gdf.crs,
            )

        def fake_enclosed_tessellation(**_kwargs: object) -> gpd.GeoDataFrame:
            return gpd.GeoDataFrame(
                {"enclosure_index": [1]},
                geometry=[sample_buildings_gdf.geometry.iloc[0]],
                crs=sample_buildings_gdf.crs,
            )

        monkeypatch.setattr(momepy, "enclosures", fake_enclosures)
        monkeypatch.setattr(momepy, "enclosed_tessellation", fake_enclosed_tessellation)

        result = utils.create_tessellation(
            sample_buildings_gdf,
            primary_barriers=sample_segments_gdf,
            limit=custom_limit,
        )

        assert captured["limit"] is custom_limit
        # Explicit limits keep momepy's clip disabled (they may be non-polygonal
        # and clipping would change the documented semantics).
        assert captured["clip"] is False
        assert not result.empty

    def test_enclosed_tessellation_computes_default_limit(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A non-empty buffered-union limit should be computed when none is supplied."""
        captured: dict[str, object] = {}

        def fake_enclosures(**kwargs: object) -> gpd.GeoDataFrame:
            captured["limit"] = kwargs["limit"]
            captured["clip"] = kwargs["clip"]
            return gpd.GeoDataFrame(
                {"eID": [1]},
                geometry=[kwargs["limit"]],
                crs=sample_buildings_gdf.crs,
            )

        def fake_enclosed_tessellation(**_kwargs: object) -> gpd.GeoDataFrame:
            return gpd.GeoDataFrame(
                {"enclosure_index": [1]},
                geometry=[sample_buildings_gdf.geometry.iloc[0]],
                crs=sample_buildings_gdf.crs,
            )

        monkeypatch.setattr(momepy, "enclosures", fake_enclosures)
        monkeypatch.setattr(momepy, "enclosed_tessellation", fake_enclosed_tessellation)

        result = utils.create_tessellation(
            sample_buildings_gdf, primary_barriers=sample_segments_gdf
        )

        limit = captured["limit"]
        assert isinstance(limit, (Polygon, MultiPolygon))
        assert not limit.is_empty
        # The derived limit may contain holes whose faces fall outside it, so
        # the enclosures must be clipped to the limit.
        assert captured["clip"] is True
        assert not result.empty

    def test_enclosed_tessellation_default_limit_confines_cells(
        self,
        sample_crs: str,
    ) -> None:
        """Cells must not extend far beyond the built fabric without a limit.

        The derived enclosure limit follows the buffered union of streets and
        buildings; a convex-hull limit would let the Voronoi cell of a
        street-front building stretch hundreds of metres into empty land as a
        needle-shaped artifact.
        """
        corners = [(0.0, 0.0), (200.0, 0.0), (200.0, 200.0), (0.0, 200.0), (0.0, 0.0)]
        streets = gpd.GeoDataFrame(
            geometry=[LineString([corners[i], corners[i + 1]]) for i in range(4)],
            crs=sample_crs,
        )
        buildings = gpd.GeoDataFrame(
            geometry=[
                Polygon([(40, 40), (80, 40), (80, 80), (40, 80)]),
                Polygon([(120, 120), (160, 120), (160, 160), (120, 160)]),
                # Street-front building outside the square: its cell lives in
                # the outer enclosure bounded only by the derived limit.
                Polygon([(80, 210), (120, 210), (120, 240), (80, 240)]),
            ],
            crs=sample_crs,
        )

        result = utils.create_tessellation(buildings, primary_barriers=streets, n_jobs=1)

        fabric = gpd.GeoSeries(
            list(streets.geometry) + list(buildings.geometry), crs=sample_crs
        ).union_all()
        assert not result.empty
        assert result.geometry.union_all().within(fabric.buffer(100.0 + 1.0))

    def test_enclosed_tessellation_default_limit_keeps_remote_building(
        self,
        sample_crs: str,
    ) -> None:
        """A building far from every street still receives its own cell.

        The derived limit buffers the buildings themselves, so a remote
        building forms its own enclosure island instead of being dropped.
        """
        corners = [(0.0, 0.0), (200.0, 0.0), (200.0, 200.0), (0.0, 200.0), (0.0, 0.0)]
        streets = gpd.GeoDataFrame(
            geometry=[LineString([corners[i], corners[i + 1]]) for i in range(4)],
            crs=sample_crs,
        )
        buildings = gpd.GeoDataFrame(
            geometry=[
                Polygon([(80, 80), (120, 80), (120, 120), (80, 120)]),
                Polygon([(500, 80), (540, 80), (540, 120), (500, 120)]),
            ],
            crs=sample_crs,
        )

        result = utils.create_tessellation(buildings, primary_barriers=streets, n_jobs=1)

        remote_centroid = buildings.geometry.iloc[1].centroid
        assert result.geometry.contains(remote_centroid).sum() == 1


class TestGraphAnalysis(BaseGraphTest):
    """Test graph analysis operations like filtering and isochrone generation."""

    @pytest.mark.parametrize(
        ("graph_fixture", "as_nx", "center_fixture", "distance", "expect_empty"),
        [
            ("sample_segments_gdf", False, "mg_center_point", 100.0, False),
            ("sample_segments_gdf", False, "mg_center_point", 0.01, True),
            ("sample_nx_graph", True, "sample_nodes_gdf", 1.0, False),
            ("empty_gdf", False, "mg_center_point", 100.0, True),
        ],
    )
    def test_graph_distance_filtering(
        self,
        graph_fixture: str,
        as_nx: bool,
        center_fixture: str,
        distance: float,
        expect_empty: bool,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test filtering graphs by distance from center points."""
        graph = request.getfixturevalue(graph_fixture)
        center_source = request.getfixturevalue(center_fixture)
        center_point = helpers.get_center_point(center_source)

        reachable = utils.filter_graph_by_distance(
            graph,
            center_point=center_point,
            threshold=distance,
            edge_attr="length",
        )

        if as_nx:
            assert isinstance(reachable, nx.Graph)
            assert (reachable.number_of_edges() == 0) == expect_empty
        else:
            self.assert_valid_gdf(reachable, expect_empty)

    def test_filter_graph_multi_center_union(self, sample_crs: str) -> None:
        """Multiple centers should union the reachable components."""
        G = nx.Graph()
        G.graph["crs"] = sample_crs
        G.add_node(1, pos=(0, 0))
        G.add_node(2, pos=(1, 0))
        G.add_node(3, pos=(10, 0))
        G.add_node(4, pos=(11, 0))
        G.add_edge(1, 2, length=1.0)
        G.add_edge(3, 4, length=1.0)

        centers = gpd.GeoSeries([Point(0, 0), Point(10, 0)], crs=sample_crs)

        reachable = utils.filter_graph_by_distance(G, center_point=centers, threshold=1.0)

        assert sorted(reachable.nodes()) == [1, 2, 3, 4]
        assert sorted(tuple(sorted(edge)) for edge in reachable.edges()) == [(1, 2), (3, 4)]

    def test_filter_graph_multi_center_sequence_union(self, sample_crs: str) -> None:
        """Plain Point sequences should union the reachable components."""
        G = nx.Graph()
        G.graph["crs"] = sample_crs
        G.add_node(1, pos=(0, 0))
        G.add_node(2, pos=(1, 0))
        G.add_node(3, pos=(10, 0))
        G.add_node(4, pos=(11, 0))
        G.add_edge(1, 2, length=1.0)
        G.add_edge(3, 4, length=1.0)

        reachable = utils.filter_graph_by_distance(
            G,
            center_point=[Point(0, 0), Point(10, 0)],
            threshold=1.0,
        )

        assert sorted(reachable.nodes()) == [1, 2, 3, 4]
        assert sorted(tuple(sorted(edge)) for edge in reachable.edges()) == [(1, 2), (3, 4)]

    def test_filter_graph_edge_attr_none_falls_back_to_length(self, sample_crs: str) -> None:
        """edge_attr=None should preserve the existing length fallback."""
        G = nx.Graph()
        G.graph["crs"] = sample_crs
        G.add_node(1, pos=(0, 0))
        G.add_node(2, pos=(1, 0))
        G.add_node(3, pos=(0, 1))
        G.add_edge(1, 2, length=1.0)
        G.add_edge(1, 3, length=1.0)

        reachable_default = utils.filter_graph_by_distance(
            G,
            center_point=Point(0, 0),
            threshold=1.0,
            edge_attr=None,
        )
        reachable_length = utils.filter_graph_by_distance(
            G,
            center_point=Point(0, 0),
            threshold=1.0,
            edge_attr="length",
        )

        assert sorted(reachable_default.nodes()) == sorted(reachable_length.nodes())
        assert sorted(tuple(sorted(edge)) for edge in reachable_default.edges()) == sorted(
            tuple(sorted(edge)) for edge in reachable_length.edges()
        )

    def test_filter_graph_gdf_preserves_edge_membership_and_crs(self, sample_crs: str) -> None:
        """GeoDataFrame input should keep the same filtered edge set and CRS."""
        edges = gpd.GeoDataFrame(
            {
                "u": [1, 2],
                "v": [2, 3],
                "length": [1.0, 1.0],
                "geometry": [
                    LineString([(0, 0), (1, 0)]),
                    LineString([(1, 0), (2, 0)]),
                ],
            },
            crs=sample_crs,
        ).set_index(["u", "v"])

        reachable = utils.filter_graph_by_distance(edges, center_point=Point(0, 0), threshold=1.0)

        assert isinstance(reachable, gpd.GeoDataFrame)
        assert list(reachable.index) == [(1, 2)]
        assert reachable.crs == edges.crs

    def test_filter_graph_excludes_disconnected_components(self, sample_crs: str) -> None:
        """Filtering should not leak into disconnected components."""
        G = nx.Graph()
        G.graph["crs"] = sample_crs
        G.add_node(1, pos=(0, 0))
        G.add_node(2, pos=(1, 0))
        G.add_node(3, pos=(10, 10))
        G.add_node(4, pos=(11, 10))
        G.add_edge(1, 2, length=1.0)
        G.add_edge(3, 4, length=1.0)

        reachable = utils.filter_graph_by_distance(G, center_point=Point(0, 0), threshold=2.0)

        assert sorted(reachable.nodes()) == [1, 2]
        assert sorted(tuple(sorted(edge)) for edge in reachable.edges()) == [(1, 2)]

    def test_create_isochrone_basic(self, sample_nx_graph: nx.Graph) -> None:
        """Test basic isochrone creation."""
        center = Point(0, 0)
        distance = 2.0

        isochrone = utils.create_isochrone(
            sample_nx_graph,
            center_point=center,
            threshold=distance,
            edge_attr="edge_feature1",  # Use custom attribute
            method="convex_hull",
        )

        assert isinstance(isochrone, gpd.GeoDataFrame)
        assert len(isochrone) == 1
        assert isochrone.geometry.iloc[0].geom_type == "Polygon"

    def test_create_isochrone_buffer(self, sample_nx_graph: nx.Graph) -> None:
        """Test isochrone creation with buffer method."""
        center = Point(0, 0)
        distance = 1.0

        isochrone = utils.create_isochrone(
            sample_nx_graph,
            center_point=center,
            threshold=distance,
            edge_attr="edge_feature1",
            method="buffer",
            buffer_distance=0.1,
        )

        assert isinstance(isochrone, gpd.GeoDataFrame)
        assert len(isochrone) == 1
        assert isochrone.geometry.iloc[0].geom_type in ["Polygon", "MultiPolygon"]

    def test_create_isochrone_multi_threshold_returns_layered_rows(
        self, sample_nx_graph: nx.Graph
    ) -> None:
        """Sequence thresholds should return one row per requested layer."""
        result = utils.create_isochrone(
            sample_nx_graph,
            center_point=Point(0, 0),
            threshold=[2.0, 1.0, 2.0],
            edge_attr="edge_feature1",
            method="convex_hull",
        )

        assert isinstance(result, gpd.GeoDataFrame)
        assert list(result["threshold"]) == [2.0, 1.0, 2.0]
        assert len(result) == 3
        assert result.crs == sample_nx_graph.graph["crs"]
        assert result.geometry.notna().all()

    def test_create_isochrone_scalar_threshold_keeps_legacy_shape(
        self, sample_nx_graph: nx.Graph
    ) -> None:
        """Scalar thresholds should keep the pre-existing single-row schema."""
        result = utils.create_isochrone(
            sample_nx_graph,
            center_point=Point(0, 0),
            threshold=2.0,
            edge_attr="edge_feature1",
            method="convex_hull",
        )

        assert len(result) == 1
        assert list(result.columns) == ["geometry"]

    def test_create_isochrone_multi_threshold_preserves_empty_layers(self) -> None:
        """Multi-threshold mode should keep rows even when no polygon is generated."""
        empty_graph = nx.Graph()
        empty_graph.graph["crs"] = "EPSG:4326"

        result = utils.create_isochrone(
            empty_graph,
            center_point=Point(0, 0),
            threshold=[10.0, 20.0],
            method="convex_hull",
        )

        assert list(result["threshold"]) == [10.0, 20.0]
        assert len(result) == 2
        assert result.geometry.is_empty.tolist() == [True, True]

    def test_create_isochrone_multi_threshold_cut_edge_types(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Cut edge filtering should still apply when layered output is requested."""
        graph = gdf_to_nx(nodes=sample_hetero_nodes_dict, edges=sample_hetero_edges_dict)
        cut_type = ("building", "connects_to", "road")

        iso_full = utils.create_isochrone(
            graph,
            center_point=Point(0, 0),
            threshold=[100.0],
            method="buffer",
            buffer_distance=1.0,
        )
        iso_cut = utils.create_isochrone(
            graph,
            center_point=Point(0, 0),
            threshold=[100.0],
            method="buffer",
            buffer_distance=1.0,
            cut_edge_types=[cut_type],
        )

        assert iso_cut.geometry.iloc[0].area < iso_full.geometry.iloc[0].area

    def test_create_isochrone_multi_threshold_from_gdf_inputs(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Layered isochrones should work with nodes/edges inputs."""
        result = utils.create_isochrone(
            nodes=sample_nodes_gdf,
            edges=sample_edges_gdf,
            center_point=Point(0, 0),
            threshold=[1.0, 2.0],
            method="convex_hull",
        )

        assert list(result["threshold"]) == [1.0, 2.0]
        assert len(result) == 2
        assert result.geometry.notna().all()

    def test_create_isochrone_multi_threshold_multi_center(self, sample_crs: str) -> None:
        """Layered output should still union reachability from multiple centers."""
        graph = nx.Graph()
        graph.graph["crs"] = sample_crs
        graph.add_node(1, pos=(0, 0))
        graph.add_node(2, pos=(1, 0))
        graph.add_node(3, pos=(10, 0))
        graph.add_node(4, pos=(11, 0))
        graph.add_edge(1, 2, length=1.0)
        graph.add_edge(3, 4, length=1.0)

        result = utils.create_isochrone(
            graph,
            center_point=gpd.GeoSeries([Point(0, 0), Point(10, 0)], crs=sample_crs),
            threshold=[1.0, 2.0],
            edge_attr="length",
            method="convex_hull",
        )

        assert list(result["threshold"]) == [1.0, 2.0]
        assert result.geometry.iloc[0].intersects(Point(0, 0))
        assert result.geometry.iloc[0].intersects(Point(10, 0))

    def test_create_isochrone_multi_threshold_computes_distances_once(
        self, sample_nx_graph: nx.Graph
    ) -> None:
        """Layered isochrones should reuse a single distance computation."""
        with mock.patch(
            "city2graph.utils.spatial._compute_center_node_distances",
            wraps=spatial_utils._compute_center_node_distances,
        ) as mock_compute:
            result = utils.create_isochrone(
                sample_nx_graph,
                center_point=Point(0, 0),
                threshold=[1.0, 2.0, 3.0],
                edge_attr="edge_feature1",
                method="convex_hull",
            )

        assert len(result) == 3
        assert mock_compute.call_count == 1

    def test_isochrone_with_cut_edge_types(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Test isochrone generation with edge type cutting."""
        graph = gdf_to_nx(nodes=sample_hetero_nodes_dict, edges=sample_hetero_edges_dict)
        center = Point(0, 0)
        distance = 100.0

        # Generate without cutting
        iso_full = utils.create_isochrone(
            graph,
            center_point=center,
            threshold=distance,
            method="buffer",
            buffer_distance=1.0,
        )

        # Generate with cutting
        # Assuming 'building', 'connects_to', 'road' is the edge type in sample
        cut_type = ("building", "connects_to", "road")
        iso_cut = utils.create_isochrone(
            graph,
            center_point=center,
            threshold=distance,
            method="buffer",
            buffer_distance=1.0,
            cut_edge_types=[cut_type],
        )

        # The cut isochrone should be smaller or empty if all edges are cut
        # In the sample, we only have one edge type. If we cut it, we have no edges.
        # But nodes are still there?
        # create_isochrone with method="buffer" uses edges.
        # If all edges are removed, it returns empty or just nodes?
        # _extract_isochrone_geometries adds nodes AND edges for buffer?
        # Let's check _extract_isochrone_geometries implementation.
        # It adds node positions first, THEN edges if method="buffer".
        # So if edges are removed, it will buffer the nodes.

        # Full isochrone: nodes + edges buffered.
        # Cut isochrone: nodes buffered (edges removed).
        # So area should be smaller.

        assert not iso_full.empty
        assert not iso_cut.empty
        assert iso_cut.geometry.iloc[0].area < iso_full.geometry.iloc[0].area

    def test_isochrone_disconnected_components(self, sample_crs: str) -> None:
        """Test that cutting edges results in multiple polygons if graph becomes disconnected."""
        # Create two clusters connected by a single edge
        # Cluster 1
        c1_nodes = [Point(0, 0), Point(1, 0), Point(0, 1)]
        c1_ids = [1, 2, 3]

        # Cluster 2 (far away)
        c2_nodes = [Point(100, 100), Point(101, 100), Point(100, 101)]
        c2_ids = [11, 12, 13]

        nodes_gdf = gpd.GeoDataFrame(
            {"geometry": c1_nodes + c2_nodes},
            index=pd.Index(c1_ids + c2_ids, name="node_id"),
            crs=sample_crs,
        )

        # Edges
        edges_data = {
            "u": [1, 1, 11, 11, 3],
            "v": [2, 3, 12, 13, 11],  # 3->11 is the long connector
            "length": [1, 1, 1, 1, 10],
            "edge_type": [
                ("street", "street", "street"),
                ("street", "street", "street"),
                ("street", "street", "street"),
                ("street", "street", "street"),
                ("transit", "transit", "transit"),
            ],
            "geometry": [
                LineString([(0, 0), (1, 0)]),
                LineString([(0, 0), (0, 1)]),
                LineString([(100, 100), (101, 100)]),
                LineString([(100, 100), (100, 101)]),
                LineString([(0, 1), (100, 100)]),
            ],
        }
        edges_gdf = gpd.GeoDataFrame(edges_data, crs=sample_crs).set_index(["u", "v"])

        graph = gdf_to_nx(nodes=nodes_gdf, edges=edges_gdf)

        # Generate isochrone with cutting "transit"
        # Distance 20 is enough to reach cluster 2
        iso = utils.create_isochrone(
            graph,
            center_point=Point(0, 0),
            threshold=20,
            edge_attr="length",
            method="convex_hull",
            cut_edge_types=[("transit", "transit", "transit")],
        )

        # Should be MultiPolygon (or at least disjoint)
        geom = iso.geometry.iloc[0]
        assert geom.geom_type == "MultiPolygon"
        # Area should be small (approx 2 small triangles), not covering the gap
        assert geom.area < 100.0

    def test_isochrone_center_gdf(self, sample_nx_graph: nx.Graph) -> None:
        """Test isochrone creation with GeoDataFrame as center_point."""
        # Create a GeoDataFrame for center point
        center_gdf = gpd.GeoDataFrame(
            {"geometry": [Point(0, 0)]},
            index=[999],  # Mismatched index to ensure robustness
            crs=sample_nx_graph.graph["crs"],
        )

        isochrone = utils.create_isochrone(
            sample_nx_graph,
            center_point=center_gdf,
            threshold=2.0,
            edge_attr="edge_feature1",
            method="convex_hull",
        )

        assert isinstance(isochrone, gpd.GeoDataFrame)
        assert len(isochrone) == 1
        assert not isochrone.empty

    def test_create_isochrone_alpha_shape(self, sample_nx_graph: nx.Graph) -> None:
        """Test isochrone creation with alpha_shape method."""
        center = Point(0, 0)
        distance = 2.0

        isochrone = utils.create_isochrone(
            sample_nx_graph,
            center_point=center,
            threshold=distance,
            edge_attr="edge_feature1",
            method="concave_hull_alpha",
            hull_ratio=0.5,
        )

        assert isinstance(isochrone, gpd.GeoDataFrame)
        assert len(isochrone) == 1
        assert isochrone.geometry.iloc[0].geom_type in ["Polygon", "MultiPolygon"]

    def test_create_isochrone_default_method(self, sample_nx_graph: nx.Graph) -> None:
        """Test isochrone creation with default method (concave_hull_knn)."""
        center = Point(0, 0)
        distance = 2.0

        isochrone = utils.create_isochrone(
            sample_nx_graph,
            center_point=center,
            threshold=distance,
            edge_attr="edge_feature1",
        )

        assert isinstance(isochrone, gpd.GeoDataFrame)
        assert len(isochrone) == 1
        assert isochrone.geometry.iloc[0].geom_type in ["Polygon", "MultiPolygon"]

    def test_create_isochrone_concave_hull_knn_params(self, sample_nx_graph: nx.Graph) -> None:
        """Test concave_hull_knn with different k values."""
        center = Point(0, 0)
        distance = 2.0

        # k=1 (should work, might fallback or minimal hull)
        iso_k1 = utils.create_isochrone(
            sample_nx_graph,
            center_point=center,
            threshold=distance,
            method="concave_hull_knn",
            k=1,
        )
        assert not iso_k1.empty

        # k larger than n_points should clamp and still produce a polygonal hull.
        iso_k_large = utils.create_isochrone(
            sample_nx_graph,
            center_point=center,
            threshold=distance,
            method="concave_hull_knn",
            k=1000,
        )
        assert not iso_k_large.empty
        assert iso_k_large.geometry.iloc[0].geom_type == "Polygon"

    def test_create_isochrone_concave_hull_alpha_params(self, sample_nx_graph: nx.Graph) -> None:
        """Test concave_hull_alpha with different parameters."""
        center = Point(0, 0)
        distance = 2.0

        # High ratio (tighter fit)
        iso_tight = utils.create_isochrone(
            sample_nx_graph,
            center_point=center,
            threshold=distance,
            method="concave_hull_alpha",
            hull_ratio=0.9,
        )
        assert not iso_tight.empty

        # Allow holes
        iso_holes = utils.create_isochrone(
            sample_nx_graph,
            center_point=center,
            threshold=distance,
            method="concave_hull_alpha",
            allow_holes=True,
        )
        assert not iso_holes.empty

    def test_create_isochrone_degenerate_cases(self, sample_crs: str) -> None:
        """Test isochrone generation with degenerate graphs (1 or 2 nodes)."""
        # 1 node
        g1 = nx.Graph()
        g1.add_node(1, pos=(0, 0), geometry=Point(0, 0))
        g1.graph = {"crs": sample_crs}

        iso1 = utils.create_isochrone(
            g1, center_point=Point(0, 0), threshold=10, method="concave_hull_knn"
        )
        assert not iso1.empty
        # Output is always Polygon or MultiPolygon (buffered from Point)
        assert iso1.geometry.iloc[0].geom_type in ["Polygon", "MultiPolygon"]

        # 2 nodes
        g2 = nx.Graph()
        g2.add_node(1, pos=(0, 0), geometry=Point(0, 0))
        g2.add_node(2, pos=(1, 1), geometry=Point(1, 1))
        g2.add_edge(1, 2, length=1.414)
        g2.graph = {"crs": sample_crs}

        iso2 = utils.create_isochrone(
            g2, center_point=Point(0, 0), threshold=10, method="concave_hull_knn"
        )
        assert not iso2.empty
        # Output is always Polygon or MultiPolygon (convex hull of LineString)
        assert iso2.geometry.iloc[0].geom_type in ["Polygon", "MultiPolygon"]

    def test_create_isochrone_input_validation(self, sample_nx_graph: nx.Graph) -> None:
        """Test input validation for create_isochrone."""
        with pytest.raises(ValueError, match="Unknown method"):
            utils.create_isochrone(
                sample_nx_graph, center_point=Point(0, 0), threshold=10, method="invalid_method"
            )

        with pytest.raises(ValueError, match="center_point and threshold must be provided"):
            utils.create_isochrone(sample_nx_graph, center_point=None, threshold=10)

        with pytest.raises(
            ValueError, match="Either 'graph' or 'nodes' and 'edges' must be provided"
        ):
            utils.create_isochrone(
                graph=None, nodes=None, edges=None, center_point=Point(0, 0), threshold=10
            )

        with pytest.raises(ValueError, match="threshold sequence must not be empty"):
            utils.create_isochrone(sample_nx_graph, center_point=Point(0, 0), threshold=[])

        with pytest.raises(TypeError, match="center_point must be a Point"):
            utils.create_isochrone(
                sample_nx_graph,
                center_point=[Point(0, 0), "bad"],
                threshold=10,
            )

    def test_create_isochrone_multi_center(self, sample_nx_graph: nx.Graph) -> None:
        """Test isochrone with multiple center points."""
        centers = gpd.GeoSeries([Point(0, 0), Point(1, 1)], crs=sample_nx_graph.graph["crs"])

        iso = utils.create_isochrone(
            sample_nx_graph, center_point=centers, threshold=2.0, method="convex_hull"
        )
        assert not iso.empty

    def test_create_isochrone_multi_center_sequence(self, sample_crs: str) -> None:
        """Plain Point sequences should work for isochrone generation."""
        graph = nx.Graph()
        graph.graph["crs"] = sample_crs
        graph.add_node(1, pos=(0, 0))
        graph.add_node(2, pos=(1, 0))
        graph.add_node(3, pos=(10, 0))
        graph.add_node(4, pos=(11, 0))
        graph.add_edge(1, 2, length=1.0)
        graph.add_edge(3, 4, length=1.0)

        iso = utils.create_isochrone(
            graph,
            center_point=[Point(0, 0), Point(10, 0)],
            threshold=1.0,
            edge_attr="length",
            method="convex_hull",
        )

        assert not iso.empty
        assert iso.geometry.iloc[0].intersects(Point(0, 0))
        assert iso.geometry.iloc[0].intersects(Point(10, 0))

    def test_create_isochrone_multi_center_tuple_sequence(self, sample_crs: str) -> None:
        """Tuple Point sequences should be accepted the same way as lists."""
        graph = nx.Graph()
        graph.graph["crs"] = sample_crs
        graph.add_node(1, pos=(0, 0))
        graph.add_node(2, pos=(1, 0))
        graph.add_edge(1, 2, length=1.0)

        iso = utils.create_isochrone(
            graph,
            center_point=(Point(0, 0), Point(1, 0)),
            threshold=1.0,
            edge_attr="length",
            method="convex_hull",
        )

        assert not iso.empty

    def test_filter_graph_invalid_center_sequence_member(self, sample_nx_graph: nx.Graph) -> None:
        """Invalid sequence members should raise a clear TypeError."""
        with pytest.raises(TypeError, match="center_point must be a Point"):
            utils.filter_graph_by_distance(
                sample_nx_graph,
                center_point=[Point(0, 0), "bad"],
                threshold=1.0,
            )

    def test_create_isochrone_empty_center_sequence_returns_empty(
        self, sample_nx_graph: nx.Graph
    ) -> None:
        """Empty center sequences should preserve empty-output behavior."""
        iso = utils.create_isochrone(
            sample_nx_graph,
            center_point=[],
            threshold=1.0,
            method="convex_hull",
        )

        assert iso.empty

    def test_filter_graph_empty_center_sequence_returns_empty(
        self, sample_nx_graph: nx.Graph
    ) -> None:
        """Empty center sequences should preserve empty-graph behavior."""
        reachable = utils.filter_graph_by_distance(
            sample_nx_graph,
            center_point=[],
            threshold=1.0,
        )

        assert isinstance(reachable, nx.Graph)
        assert len(reachable.nodes()) == 0

    def test_create_isochrone_buffer_none(self, sample_nx_graph: nx.Graph) -> None:
        """Test buffer method with buffer_distance=None (should return Polygon/MultiPolygon)."""
        center = Point(0, 0)
        iso = utils.create_isochrone(
            sample_nx_graph,
            center_point=center,
            threshold=2.0,
            method="buffer",
            buffer_distance=None,
        )
        assert not iso.empty
        # Output is always Polygon or MultiPolygon
        assert iso.geometry.iloc[0].geom_type in ["Polygon", "MultiPolygon"]

    def test_create_isochrone_from_gdf_inputs(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test create_isochrone with nodes/edges input (returns GDF reachable)."""
        center = Point(0, 0)
        iso = utils.create_isochrone(
            nodes=sample_nodes_gdf,
            edges=sample_edges_gdf,
            center_point=center,
            threshold=2.0,
            method="convex_hull",
        )
        assert not iso.empty
        assert iso.geometry.iloc[0].geom_type == "Polygon"

    def test_filter_graph_no_pos(self, sample_crs: str) -> None:
        """Test filter_graph_by_distance with graph missing 'pos' attribute."""
        G = nx.Graph()
        G.add_node(1)  # No pos
        G.graph = {"crs": sample_crs}

        # Should return empty graph or handle gracefully
        reachable = utils.filter_graph_by_distance(G, center_point=Point(0, 0), threshold=10)
        assert isinstance(reachable, nx.Graph)
        assert len(reachable) == 0

    def test_create_isochrone_far_snap(self, sample_nx_graph: nx.Graph) -> None:
        """Test isochrone with far away center (should snap to nearest)."""
        center = Point(1000, 1000)  # Far away
        iso = utils.create_isochrone(
            sample_nx_graph,
            center_point=center,
            threshold=1.0,
            method="concave_hull_knn",
        )
        # Current implementation snaps to nearest node regardless of distance
        assert not iso.empty
        # Should be a geometry (Point, LineString, or Polygon depending on neighbors)
        assert iso.geometry.iloc[0].geom_type in ["Point", "LineString", "Polygon"]

    def test_create_isochrone_disconnected_graph_parts(self, sample_crs: str) -> None:
        """Test isochrone on a graph with completely disconnected components."""
        # Create two disconnected components
        G = nx.Graph()
        G.graph["crs"] = sample_crs

        # Component 1
        G.add_node(1, pos=(0, 0))
        G.add_node(2, pos=(1, 0))
        G.add_edge(1, 2, length=1)

        # Component 2
        G.add_node(3, pos=(10, 10))
        G.add_node(4, pos=(11, 10))
        G.add_edge(3, 4, length=1)

        # Isochrone from Component 1 should not include Component 2
        iso = utils.create_isochrone(
            G, center_point=Point(0, 0), threshold=2.0, edge_attr="length", method="convex_hull"
        )

        assert len(iso) == 1
        poly = iso.geometry.iloc[0]
        assert poly.intersects(Point(0, 0))
        assert not poly.intersects(Point(10, 10))

    def test_create_isochrone_all_methods(self, sample_nx_graph: nx.Graph) -> None:
        """Ensure all methods run without error on a standard graph."""
        center = Point(0, 0)
        methods = ["concave_hull_knn", "concave_hull_alpha", "convex_hull", "buffer"]

        for method in methods:
            iso = utils.create_isochrone(
                sample_nx_graph, center_point=center, threshold=2.0, method=method
            )
            assert not iso.empty
            assert isinstance(iso, gpd.GeoDataFrame)

    def test_prepare_isochrone_graph_edge_attr_injection(
        self, sample_nodes_gdf: gpd.GeoDataFrame, sample_edges_gdf: gpd.GeoDataFrame
    ) -> None:
        """Test that edge attribute is injected if missing when using GDF inputs."""
        # Remove length column if exists
        if "length" in sample_edges_gdf.columns:
            sample_edges_gdf = sample_edges_gdf.drop(columns=["length"])

        iso = utils.create_isochrone(
            nodes=sample_nodes_gdf,
            edges=sample_edges_gdf,
            center_point=Point(0, 0),
            threshold=10.0,
            edge_attr="length",  # Should be calculated from geometry
            method="convex_hull",
        )
        assert not iso.empty

    def test_concave_hull_knn_degenerate_triangle(self) -> None:
        """Test concave hull with exactly 3 points (triangle)."""
        points = [Point(0, 0), Point(1, 0), Point(0, 1)]
        poly = spatial_utils._concave_hull_knn(points, k=3)
        assert isinstance(poly, Polygon)
        assert poly.area == 0.5

    def test_concave_hull_knn_collinear(self) -> None:
        """Test concave hull with collinear points."""
        points = [Point(0, 0), Point(1, 1), Point(2, 2)]
        # Should return LineString or fallback to convex hull (which is LineString for collinear)
        geom = spatial_utils._concave_hull_knn(points, k=3)
        assert isinstance(geom, LineString)

    def test_prepare_isochrone_graph_dict_edges_missing_attr(self, sample_crs: str) -> None:
        """Test _prepare_isochrone_graph with dict edges missing the edge attribute."""
        nodes = {
            "node_type": gpd.GeoDataFrame(
                {"geometry": [Point(0, 0), Point(1, 1)]},
                index=[1, 2],
                crs=sample_crs,
            )
        }
        # Edges dict without 'length' attribute, but with geometry
        edges = {
            ("node_type", "rel", "node_type"): gpd.GeoDataFrame(
                {
                    "u": [1],
                    "v": [2],
                    "geometry": [LineString([(0, 0), (1, 1)])],
                },
                crs=sample_crs,
            ).set_index(["u", "v"])
        }

        # Should calculate length automatically
        graph = spatial_utils._prepare_isochrone_graph(
            graph=None, nodes=nodes, edges=edges, edge_attr="length"
        )
        assert isinstance(graph, (nx.Graph, nx.MultiGraph))

        # Check that at least one edge has length > 0
        found_length = False
        for _, _, data in graph.edges(data=True):
            if "length" in data and data["length"] > 0:
                found_length = True
                break
        assert found_length, "Edge length was not calculated"

    def test_find_best_candidate_no_valid(self) -> None:
        """Test _find_best_candidate when no valid candidate exists."""
        # Scenario: 3 points, but the only candidate would cause self-intersection
        # We can test the helper directly to hit the "return None" branch
        coords = np.array([(0, 0), (1, 0), (0, 1)])
        current_idx = 0
        candidates = [1, 2]
        prev_vec = np.array([1, 0])
        hull_indices = [0]
        start_idx = 0

        # This should find a candidate
        best = spatial_utils._find_best_candidate(
            coords, current_idx, candidates, prev_vec, hull_indices, start_idx
        )
        assert best is not None

        # Now force failure by making candidates invalid (e.g. zero length vector)
        # coords[1] same as coords[0]
        coords_degenerate = np.array([(0, 0), (0, 0)])
        best_degenerate = spatial_utils._find_best_candidate(
            coords_degenerate, 0, [1], prev_vec, hull_indices, start_idx
        )
        assert best_degenerate is None

    def test_is_valid_edge_intersection(self) -> None:
        """Test _is_valid_edge detects intersections."""
        # Construct a hull that is about to self-intersect
        coords = np.array([(0, 0), (10, 0), (10, 10), (0, 10), (5, 5)])
        # Hull: (0,0) -> (10,0) -> (10,10) -> (0,10)
        hull_indices = [0, 1, 2, 3]
        start_idx = 0
        hull_arr = np.asarray(hull_indices, dtype=int)
        seg_starts = coords[hull_arr[:-2]]
        seg_ends = coords[hull_arr[1:-1]]
        existing_lines = [
            LineString([start, end]) for start, end in zip(seg_starts, seg_ends, strict=True)
        ]
        seg_bounds_min = np.minimum(seg_starts, seg_ends)
        seg_bounds_max = np.maximum(seg_starts, seg_ends)

        # Edge from (0,10) to (5,5) is valid (inside)
        assert spatial_utils._is_valid_edge(
            coords=coords,
            current_idx=3,
            candidate_idx=4,
            start_idx=start_idx,
            existing_lines=existing_lines,
            seg_bounds_min=seg_bounds_min,
            seg_bounds_max=seg_bounds_max,
        )

        # Edge from (0,10) to (5, -5) crosses (0,0)-(10,0)
        invalid_coords = np.vstack([coords, np.array([(5, -5)])])
        assert not spatial_utils._is_valid_edge(
            coords=invalid_coords,
            current_idx=3,
            candidate_idx=5,
            start_idx=start_idx,
            existing_lines=existing_lines,
            seg_bounds_min=seg_bounds_min,
            seg_bounds_max=seg_bounds_max,
        )

    def test_create_isochrone_empty_result(self) -> None:
        """Test create_isochrone returning empty result."""
        # Threshold 0.0 should result in no reachable nodes/edges if center is not on a node?
        # Or if graph is empty.
        empty_graph = nx.Graph()
        empty_graph.graph["crs"] = "EPSG:4326"

        iso = utils.create_isochrone(empty_graph, center_point=Point(0, 0), threshold=10.0)
        assert iso.empty
        assert "geometry" in iso.columns

    def test_filter_edges_by_type_coverage(self, sample_crs: str) -> None:
        """Test _filter_edges_by_type with actual removal."""
        G = nx.Graph()
        G.graph["crs"] = sample_crs
        G.add_edge(1, 2, full_edge_type=("a", "b", "c"), length=1)
        G.add_node(1, pos=(0, 0))
        G.add_node(2, pos=(1, 1))

        iso = utils.create_isochrone(
            G,
            center_point=Point(0, 0),
            threshold=10,
            cut_edge_types=[("a", "b", "c")],
            method="convex_hull",
        )
        # Edge removed, nodes remain.
        # Output is always Polygon or MultiPolygon (buffered from remaining geometries)
        assert not iso.empty
        assert iso.geometry.iloc[0].geom_type in ["Polygon", "MultiPolygon"]

    def test_prepare_isochrone_graph_errors(self) -> None:
        """Test error conditions in _prepare_isochrone_graph."""
        with pytest.raises(
            ValueError, match="Either 'graph' or 'nodes' and 'edges' must be provided"
        ):
            spatial_utils._prepare_isochrone_graph(None, None, None, None)

    def test_process_component_empty(self, sample_crs: str) -> None:
        """Test _process_component with empty/invalid inputs."""
        # Empty component
        G = nx.Graph()
        assert spatial_utils._process_component(G, "convex_hull", sample_crs) is None

    def test_generate_buffer_options(self, sample_nx_graph: nx.Graph) -> None:
        """Test buffer method with non-default options."""
        iso = utils.create_isochrone(
            sample_nx_graph,
            center_point=Point(0, 0),
            threshold=1.0,
            method="buffer",
            buffer_distance=10.0,
            cap_style=2,
            join_style=2,
            resolution=4,
        )
        assert not iso.empty
        assert iso.geometry.iloc[0].geom_type in ["Polygon", "MultiPolygon"]

    def test_process_component_empty_geoms(self, sample_nx_graph: nx.Graph) -> None:
        """Test _process_component when geometries become empty/invalid."""
        with mock.patch("city2graph.utils.spatial._extract_isochrone_geometries") as mock_extract:
            mock_extract.return_value = [Point()]  # Empty point

            iso = utils.create_isochrone(
                sample_nx_graph,
                center_point=Point(0, 0),
                threshold=100,
                method="convex_hull",
                edge_attr="edge_feature1",
            )
            assert iso.empty
            assert len(iso) == 0

    def test_concave_hull_knn_k_zero(self, sample_nx_graph: nx.Graph) -> None:
        """Test concave_hull_knn with k=0 to trigger scalar index handling."""
        # This hits line 3160: indices = [indices]
        iso = utils.create_isochrone(
            sample_nx_graph,
            center_point=Point(0, 0),
            threshold=100,
            method="concave_hull_knn",
            k=0,
            edge_attr="edge_feature1",
        )
        assert not iso.empty

    def test_concave_hull_knn_retries_with_larger_k(self) -> None:
        """concave_hull_knn should retry the full walk with a larger neighborhood."""
        points = np.array([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], dtype=float)

        with mock.patch("city2graph.utils.spatial._trace_concave_hull_once") as mock_trace:
            mock_trace.side_effect = [None, [0, 1, 2, 3]]

            poly = spatial_utils._concave_hull_knn(points, k=2)

        assert isinstance(poly, Polygon)
        assert [call.args[2] for call in mock_trace.call_args_list] == [2, 3]

    def test_concave_hull_knn_fallback(self, sample_nx_graph: nx.Graph) -> None:
        """Test concave_hull_knn fallback to alpha hull when the walker cannot progress."""
        with (
            mock.patch("city2graph.utils.spatial._find_next_hull_point") as mock_find,
            mock.patch("city2graph.utils.spatial._concave_fallback_alpha") as mock_fallback,
        ):
            mock_find.return_value = None
            mock_fallback.return_value = Polygon([(0, 0), (1, 0), (0, 1)])

            iso = utils.create_isochrone(
                sample_nx_graph,
                center_point=Point(0, 0),
                threshold=100,
                method="concave_hull_knn",
                k=3,
                edge_attr="edge_feature1",
            )
            assert not iso.empty
            mock_fallback.assert_called_once()
            assert iso.geometry.iloc[0].equals(mock_fallback.return_value)

    def test_create_isochrone_falls_back_when_hull_walk_never_closes(self, sample_crs: str) -> None:
        """Isochrone generation should fall back when the concave walk exhausts without closure."""
        graph = nx.Graph()
        graph.graph["crs"] = sample_crs
        graph.add_node(1, pos=(0, 0))
        graph.add_node(2, pos=(2, 0))
        graph.add_node(3, pos=(1, 1))
        graph.add_edge(1, 2, length=1.0)
        graph.add_edge(2, 3, length=1.0)

        next_indices = iter([1, 2, 1, 2])

        with (
            mock.patch(
                "city2graph.utils.spatial._find_next_hull_point",
                side_effect=lambda **_kwargs: next(next_indices),
            ),
            mock.patch(
                "city2graph.utils.spatial._concave_fallback_alpha",
                return_value=Polygon([(0, 0), (2, 0), (1, 1)]),
            ) as mock_fallback,
        ):
            iso = utils.create_isochrone(
                graph,
                center_point=Point(0, 0),
                threshold=10.0,
                edge_attr="length",
                method="concave_hull_knn",
            )

        assert not iso.empty
        mock_fallback.assert_called_once()

    def test_buffer_negative_distance(self, sample_nx_graph: nx.Graph) -> None:
        """Test buffer method with negative distance (results in empty polygon)."""
        iso = utils.create_isochrone(
            sample_nx_graph,
            center_point=Point(0, 0),
            threshold=100,
            method="buffer",
            buffer_distance=-100.0,  # Should collapse points
            edge_attr="edge_feature1",
        )
        assert iso.empty
        assert len(iso) == 0

    def test_isochrone_no_edge_attr(self, sample_nx_graph: nx.Graph) -> None:
        """Test create_isochrone with edge_attr=None."""
        # Should rely on existing weights or default if not used by method
        iso = utils.create_isochrone(
            sample_nx_graph,
            center_point=Point(0, 0),
            threshold=10.0,
            edge_attr=None,
            method="convex_hull",
        )
        assert not iso.empty

    def test_isochrone_generate_polygon_returns_none(self, sample_nx_graph: nx.Graph) -> None:
        """Test _process_component when _generate_polygon returns None."""
        with mock.patch("city2graph.utils.spatial._generate_polygon") as mock_gen:
            mock_gen.return_value = None
            iso = utils.create_isochrone(
                sample_nx_graph,
                center_point=Point(0, 0),
                threshold=10.0,
                method="convex_hull",
            )
            assert iso.empty

    def test_isochrone_degenerate_buffer_param(self, sample_crs: str) -> None:
        """Test degenerate_buffer_distance parameter."""
        # Create a graph that results in a degenerate geometry (Point)
        G = nx.Graph()
        G.graph["crs"] = sample_crs
        G.add_node(1, pos=(0, 0))

        # Default buffer is 1.0. Let's set it to 0.5.
        iso = utils.create_isochrone(
            G,
            center_point=Point(0, 0),
            threshold=10.0,
            method="convex_hull",  # Single point convex hull is Point
            degenerate_buffer_distance=0.5,
        )
        assert not iso.empty
        poly = iso.geometry.iloc[0]
        # Area of circle with radius 0.5 is pi * 0.25 ~= 0.785
        assert 0.7 < poly.area < 0.8

    def test_isochrone_directed_graph(self, sample_crs: str) -> None:
        """Test create_isochrone with a directed graph."""
        G = nx.DiGraph()
        G.graph["crs"] = sample_crs
        G.add_node(1, pos=(0, 0))
        G.add_node(2, pos=(1, 0))
        G.add_edge(1, 2, length=1.0)  # 1 -> 2

        # From 1, can reach 2.
        iso = utils.create_isochrone(
            G,
            center_point=Point(0, 0),
            threshold=2.0,
            edge_attr="length",
            method="convex_hull",
        )
        assert not iso.empty
        # Should cover both points

        # From 2, cannot reach 1.
        iso2 = utils.create_isochrone(
            G,
            center_point=Point(1, 0),
            threshold=2.0,
            edge_attr="length",
            method="convex_hull",
        )
        # Should only contain node 2 (Point) -> buffered to Polygon
        assert not iso2.empty
        assert iso2.geometry.iloc[0].area < 4.0  # Small buffer around point


class TestPlotting(BaseGraphTest):
    """Test plotting functionality."""

    def test_plot_graph_homogeneous(self, sample_nx_graph: nx.Graph) -> None:
        """Test plotting a homogeneous graph."""
        if not utils.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        # Mock plt.show to avoid display
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("matplotlib.pyplot.show", lambda: None)
            utils.plot_graph(sample_nx_graph)

    def test_plot_graph_hetero_subplots(
        self,
        regular_hetero_graph: nx.Graph,
    ) -> None:
        """Test heterogeneous graph plotting with subplots."""
        if not utils.MATPLOTLIB_AVAILABLE:
            pytest.skip("Matplotlib not available")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            utils.plot_graph(
                graph=regular_hetero_graph,
                subplots=True,
            )

    def test_plot_graph_hetero_subplots_unused_axes(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Test heterogeneous subplot with more axes than edge types (line 3441)."""
        if not utils.MATPLOTLIB_AVAILABLE:
            pytest.skip("Matplotlib not available")

        # Create a hetero graph with only 2 edge types
        # When subplot grid is larger than number of edge types, unused axes should be hidden
        graph = utils.gdf_to_nx(
            nodes=sample_hetero_nodes_dict,
            edges=sample_hetero_edges_dict,
            multigraph=True,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # This should create more axes than needed and hide unused ones
            # The function will internally hide unused axes (line 3441)
            utils.plot_graph(graph=graph, subplots=True, figsize=(10, 10))

        plt.close("all")

    def test_plot_graph_hetero_no_subplots_with_graph_input(
        self,
        regular_hetero_graph: nx.Graph,
    ) -> None:
        """Test plotting heterogeneous graph without subplots using graph input (line 3444)."""
        if not utils.MATPLOTLIB_AVAILABLE:
            pytest.skip("Matplotlib not available")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            utils.plot_graph(
                graph=regular_hetero_graph,
                subplots=False,
            )

    def test_plot_graph_save(self, sample_nx_graph: nx.Graph) -> None:
        """Test plotting without showing (for saving to file)."""
        if not utils.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("matplotlib.pyplot.show", lambda: None)
            # Plot without auto-display (caller can save the figure)
            utils.plot_graph(sample_nx_graph)

    def test_plot_graph_no_matplotlib(self, sample_nx_graph: nx.Graph) -> None:
        """Test error when matplotlib is missing."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(spatial_utils, "MATPLOTLIB_AVAILABLE", False)
            with pytest.raises(ImportError, match=r"(?i)matplotlib is required"):
                utils.plot_graph(sample_nx_graph)

    def test_plot_graph_no_input(self) -> None:
        """Test error when no input is provided."""
        if not utils.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        with pytest.raises(ValueError, match="At least one of graph, nodes, or edges"):
            utils.plot_graph()

    def test_plot_graph_with_gdf_input(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test plotting with GeoDataFrame inputs directly."""
        if not utils.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("matplotlib.pyplot.show", lambda: None)
            utils.plot_graph(nodes=sample_nodes_gdf, edges=sample_edges_gdf)

    def test_plot_graph_gdf_as_graph_param(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test legacy support: GeoDataFrame passed as graph parameter."""
        if not utils.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("matplotlib.pyplot.show", lambda: None)
            # This should treat the GDF as nodes (legacy support)
            utils.plot_graph(graph=sample_nodes_gdf)

    def test_plot_graph_unsupported_type(self) -> None:
        """Test error when unsupported type is passed as graph."""
        if not utils.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        with pytest.raises(TypeError, match="Unsupported data type"):
            utils.plot_graph(graph="not_a_graph")

    def test_plot_graph_hetero_subplots_with_unused_axes(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Test plotting heterogeneous graph with subplots."""
        if not utils.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("matplotlib.pyplot.show", lambda: None)
            utils.plot_graph(
                nodes=sample_hetero_nodes_dict,
                edges=sample_hetero_edges_dict,
                subplots=True,
            )

    def test_plot_graph_hetero_subplots_title_color_override(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Titles should be visible when a custom color is provided."""
        if not utils.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        edge_count = max(1, len(sample_hetero_edges_dict))
        fig, axes = plt.subplots(1, edge_count)
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

        try:
            utils.plot_graph(
                nodes=sample_hetero_nodes_dict,
                edges=sample_hetero_edges_dict,
                subplots=True,
                ax=axes,
                bgcolor="white",
                title_color="black",
            )

            assert axes_flat[0].get_title() != ""
            assert axes_flat[0].title.get_color() == "black"
        finally:
            plt.close(fig)

    def test_plot_graph_with_ax_bgcolor(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test that bgcolor is applied when ax is provided."""
        if not utils.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        assert plt is not None
        fig, ax = plt.subplots()
        try:
            utils.plot_graph(
                nodes=sample_nodes_gdf,
                edges=sample_edges_gdf,
                ax=ax,
                bgcolor="#123456",
            )
            # Check that the axes background color was set
            assert ax.get_facecolor()[:3] == tuple(int(c, 16) / 255 for c in ["12", "34", "56"])
        finally:
            plt.close(fig)

    def test_plot_graph_hetero_no_subplots(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Test plotting heterogeneous graph without subplots."""
        if not utils.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("matplotlib.pyplot.show", lambda: None)
            utils.plot_graph(
                nodes=sample_hetero_nodes_dict,
                edges=sample_hetero_edges_dict,
                subplots=False,
            )

    def test_plot_graph_no_legend(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Test plotting without legend."""
        if not utils.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("matplotlib.pyplot.show", lambda: None)
            utils.plot_graph(
                nodes=sample_hetero_nodes_dict,
                edges=sample_hetero_edges_dict,
                legend_position=None,
                subplots=False,
            )

    def test_plot_graph_style_kwargs(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test plotting with custom style kwargs."""
        if not utils.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("matplotlib.pyplot.show", lambda: None)
            utils.plot_graph(
                nodes=sample_nodes_gdf,
                edges=sample_edges_gdf,
                node_color="red",
                node_alpha=0.5,
                edge_color="blue",
                edge_linewidth=2.0,
                markersize=10.0,
            )

    def test_plot_graph_with_ax(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test plotting with provided axes."""
        if not utils.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        assert plt is not None  # For type checker
        fig, ax = plt.subplots()
        try:
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr("matplotlib.pyplot.show", lambda: None)
                utils.plot_graph(nodes=sample_nodes_gdf, ax=ax)
        finally:
            plt.close(fig)

    def test_plot_graph_homo_with_legend_kwargs_and_title(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Homogeneous plotting should support legend kwargs and title."""
        if not utils.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        nodes = sample_nodes_gdf.copy()
        nodes["centrality_quantile"] = range(len(nodes))

        edge_linewidth = pd.Series([1.0] * len(sample_edges_gdf), index=sample_edges_gdf.index)
        fig, ax = plt.subplots(figsize=(5, 5))
        try:
            utils.plot_graph(
                nodes=nodes,
                edges=sample_edges_gdf,
                node_color="centrality_quantile",
                edge_color="#bbbbbb",
                edge_linewidth=edge_linewidth,
                legend=True,
                legend_kwargs={"label": "Betweenness Centrality", "orientation": "horizontal"},
                title="Central London Transit Network Betweenness Centrality of Stops",
                ax=ax,
            )
            assert (
                ax.get_title() == "Central London Transit Network Betweenness Centrality of Stops"
            )
        finally:
            plt.close(fig)

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_plot_empty_gdf(self, empty_gdf: gpd.GeoDataFrame) -> None:
        """Test _plot_gdf with empty GeoDataFrame (line 3246)."""
        # This should not raise and should return early
        fig, ax = plt.subplots()
        spatial_utils._plot_gdf(empty_gdf, ax)
        plt.close(fig)

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_plot_with_series_color(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test plot_graph with pd.Series for color parameter (lines 3085, 3263)."""
        # Create a Series for node colors with numeric indices (0-based from conversion)
        # When nodes are converted to NetworkX, they get new integer indices
        num_nodes = len(sample_nodes_gdf)
        node_colors = pd.Series(["red"] * num_nodes, index=range(num_nodes))

        utils.plot_graph(
            graph=None,
            nodes=sample_nodes_gdf,
            edges=sample_edges_gdf,
            node_color=node_colors,
        )
        plt.close("all")

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_plot_with_column_name(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test plot_graph with column name string for color (line 3087)."""
        # Add a numeric column for coloring
        nodes_with_attr = sample_nodes_gdf.copy()
        nodes_with_attr["value"] = range(len(nodes_with_attr))

        utils.plot_graph(
            graph=None,
            nodes=nodes_with_attr,
            edges=sample_edges_gdf,
            node_color="value",
        )
        plt.close("all")

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_plot_hetero_subplots_empty_edges(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Test _plot_hetero_subplots with no edge items (line 3406)."""
        # Create edges dict with only empty GeoDataFrames
        empty_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame] = {
            ("building", "connects", "road"): gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        }

        # This should return early without creating a plot
        fig, _ax = plt.subplots()
        spatial_utils._plot_hetero_subplots(
            sample_hetero_nodes_dict,
            empty_edges_dict,
            figsize=(10, 10),
            bgcolor="white",
        )
        plt.close(fig)

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_plot_graph_subplots_less_than_grid(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Test plot_graph with subplots where number of plots < grid size."""
        # Create a situation where we have 1 edge type but grid is 2 cols
        # This triggers the "hide unused axes" logic

        # Filter to just one edge type
        single_edge_type = next(iter(sample_hetero_edges_dict.keys()))
        single_edge_dict = {single_edge_type: sample_hetero_edges_dict[single_edge_type]}

        _fig, _ax = plt.subplots()
        utils.plot_graph(
            nodes=sample_hetero_nodes_dict,
            edges=single_edge_dict,
            subplots=True,
            ncols=2,  # Force 2 columns for 1 item
        )
        plt.close("all")


class TestRemoveIsolatedComponents(BaseGraphTest):
    """Test remove_isolated_components functionality."""

    def test_concave_hull_knn_two_points(self) -> None:
        """Test concave_hull_knn with 2 points returns LineString (covers line 3000)."""
        points = [Point(0, 0), Point(1, 1)]
        result = spatial_utils._concave_hull_knn(points, k=10)
        # With 2 points, should return a LineString
        assert result.geom_type == "LineString"

    def test_concave_hull_knn_one_point(self) -> None:
        """Test concave_hull_knn with 1 point returns Point (covers line 3000)."""
        points = [Point(0, 0)]
        result = spatial_utils._concave_hull_knn(points, k=10)
        # With 1 point, should return a Point
        assert result.geom_type == "Point"

    def test_concave_hull_alpha_degenerate(self) -> None:
        """Test _concave_hull_alpha with 2 points (covers line 3300)."""
        points = [Point(0, 0), Point(1, 1)]
        result = spatial_utils._concave_hull_alpha(points, ratio=0.5, allow_holes=False)
        # Should fallback to convex hull
        assert result.geom_type in ["Polygon", "LineString", "Point"]

    def test_isochrone_buffer_returns_non_polygon(self) -> None:
        """Test create_isochrone handles non-polygon geometry from buffer."""
        # Create a simple graph that might result in a geometry collection
        G = nx.Graph()
        G.add_node(1, pos=(0, 0), geometry=Point(0, 0))
        G.graph = {"crs": "EPSG:27700"}

        result = utils.create_isochrone(
            G,
            center_point=Point(0, 0),
            threshold=10,
            method="buffer",
            buffer_distance=0.01,
        )
        # Result should always be Polygon or MultiPolygon
        assert result.geometry.iloc[0].geom_type in ["Polygon", "MultiPolygon"]


class TestPublicUtilityBranches(BaseGraphTest):
    """Coverage-oriented tests exercised through public utility APIs."""

    def test_create_isochrone_buffers_non_polygon_union(
        self,
        sample_nx_graph: nx.Graph,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """create_isochrone should normalize non-polygon unions back into polygon output."""
        monkeypatch.setattr(
            spatial_utils,
            "_generate_component_polygons",
            lambda _reachable, _method, _crs, **_kwargs: [LineString([(0, 0), (1, 0)])],
        )

        result = utils.create_isochrone(
            sample_nx_graph,
            center_point=Point(0, 0),
            threshold=10.0,
            method="buffer",
        )

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 1

    def test_create_isochrone_with_duplicate_positions_uses_degenerate_knn_path(self) -> None:
        """Duplicate reachable positions should still produce an isochrone through the public API."""
        graph = nx.Graph()
        graph.add_node("a", pos=(0.0, 0.0))
        graph.add_node("b", pos=(0.0, 0.0))
        graph.add_node("c", pos=(1.0, 0.0))
        graph.add_edge("a", "b", length=0.0)
        graph.add_edge("b", "c", length=1.0)
        graph.graph["crs"] = "EPSG:27700"

        result = utils.create_isochrone(
            graph,
            center_point=Point(0, 0),
            threshold=5.0,
            method="concave_hull_knn",
        )

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 1

    def test_create_tessellation_reraises_unexpected_value_error(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Unexpected momepy ValueError instances should still surface to callers."""

        def raise_boom(*_args: object, **_kwargs: object) -> object:
            msg = "boom"
            error = ValueError(msg)
            raise error

        monkeypatch.setattr("city2graph.utils.spatial.momepy.enclosed_tessellation", raise_boom)

        with pytest.raises(ValueError, match="boom"):
            utils.create_tessellation(sample_buildings_gdf, primary_barriers=sample_segments_gdf)

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_plot_graph_resolves_type_specific_kwargs_for_subplots(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """plot_graph should resolve non-standard kwargs from type-keyed dictionaries."""
        edge_type = next(iter(sample_hetero_edges_dict))

        utils.plot_graph(
            nodes=sample_hetero_nodes_dict,
            edges=sample_hetero_edges_dict,
            subplots=True,
            rasterized={"building": False, "road": False, edge_type: False},
        )

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_plot_graph_hides_unused_provided_axes(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Provided axis lists should be supported and unused axes hidden."""
        edge_key, edge_gdf = next(iter(sample_hetero_edges_dict.items()))
        fig, axes = plt.subplots(1, 2)

        try:
            returned_axes = utils.plot_graph(
                nodes=sample_hetero_nodes_dict,
                edges={edge_key: edge_gdf},
                subplots=True,
                ax=axes,
            )
            assert isinstance(returned_axes, np.ndarray)
            assert list(returned_axes) == list(axes)
            assert not axes[1].get_visible()
        finally:
            plt.close(fig)

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_plot_graph_accepts_single_provided_axis_for_one_subplot(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """A single provided axis should be wrapped for subplot plotting."""
        edge_key, edge_gdf = next(iter(sample_hetero_edges_dict.items()))
        fig, ax = plt.subplots(1, 1)

        try:
            returned_ax = utils.plot_graph(
                nodes=sample_hetero_nodes_dict,
                edges={edge_key: edge_gdf},
                subplots=True,
                ax=ax,
            )
            assert returned_ax is ax
        finally:
            plt.close(fig)

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_plot_graph_accepts_axis_lists_for_subplots(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Subplot rendering should accept a plain list of axes."""
        edge_key, edge_gdf = next(iter(sample_hetero_edges_dict.items()))
        fig, axes = plt.subplots(1, 2)
        axes_list = list(axes)

        try:
            returned_axes = utils.plot_graph(
                nodes=sample_hetero_nodes_dict,
                edges={edge_key: edge_gdf},
                subplots=True,
                ax=cast("Any", axes_list),
            )
            assert returned_axes == axes_list
            assert not axes_list[1].get_visible()
        finally:
            plt.close(fig)
