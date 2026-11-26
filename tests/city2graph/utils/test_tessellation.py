"""Tests for :mod:`city2graph.utils.tessellation`."""

from __future__ import annotations

from typing import cast
from unittest import mock

import geopandas as gpd
import momepy
import pytest
import shapely.errors
from shapely.geometry import GeometryCollection
from shapely.geometry import LineString
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import Polygon

from city2graph import utils
from tests.city2graph import helpers


@pytest.fixture
def single_point_geom_gdf(sample_crs: str) -> gpd.GeoDataFrame:
    """Fixture for single point geometry that might cause tessellation issues."""
    return gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs=sample_crs)


@pytest.fixture
def tessellation_barriers_gdf(sample_crs: str) -> gpd.GeoDataFrame:
    """Fixture for barriers in tessellation testing."""
    return gpd.GeoDataFrame(
        {"geometry": [LineString([(0, 0), (1, 1)])]},
        crs=sample_crs,
    )


@pytest.fixture
def single_enclosure(
    monkeypatch: pytest.MonkeyPatch, sample_buildings_gdf: gpd.GeoDataFrame
) -> None:
    """Supply one boundary while testing tessellation retries and fallbacks."""
    enclosure = gpd.GeoDataFrame(
        {"eID": [1]},
        geometry=[Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])],
        crs=sample_buildings_gdf.crs,
    )
    monkeypatch.setattr(momepy, "enclosures", mock.Mock(return_value=enclosure))


class TestCreateTessellation:
    """Tests for ``create_tessellation``."""

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

        tessellation = utils.create_tessellation(geometry, primary_barriers=barriers)

        helpers.assert_valid_gdf(tessellation, expect_empty)
        if not expect_empty:
            helpers.assert_has_columns(tessellation, ["tess_id"])
            assert tessellation.crs == geometry.crs

    def test_tessellation_handles_degenerate_inputs(
        self,
        single_point_geom_gdf: gpd.GeoDataFrame,
        tessellation_barriers_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Degenerate geometries or barriers should not raise."""
        single_result = utils.create_tessellation(single_point_geom_gdf)
        helpers.assert_valid_gdf(single_result, expected_empty=True)

        barrier_result = utils.create_tessellation(
            single_point_geom_gdf,
            primary_barriers=tessellation_barriers_gdf,
        )
        helpers.assert_valid_gdf(barrier_result, expected_empty=True)

    def test_tessellation_empty_with_barriers(self, empty_gdf: gpd.GeoDataFrame) -> None:
        """Even with barriers an empty geometry should stay empty."""
        barriers = gpd.GeoDataFrame(
            {"geometry": [LineString([(0, 0), (1, 1)])]},
            crs=empty_gdf.crs,
        )
        result = utils.create_tessellation(empty_gdf, primary_barriers=barriers)
        helpers.assert_valid_gdf(result, expected_empty=True)
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
        # Since utils imports momepy, we patch city2graph.utils.tessellation.momepy.enclosed_tessellation
        with mock.patch("city2graph.utils.tessellation.momepy.enclosed_tessellation") as mock_tess:
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

    @pytest.mark.usefixtures("single_enclosure")
    def test_tessellation_returns_empty_when_momepy_concat_fails_with_enclosures(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Enclosed tessellation should warn and return an empty schema on concat failures."""

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

    @pytest.mark.usefixtures("single_enclosure")
    def test_enclosed_tessellation_retries_with_grid_size_on_geometry_type_error(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A geometry-type TypeError should first retry with a coarser grid_size."""
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
        # ``simplify`` is never injected: momepy >=1.0 rejects it as a no-op.
        assert calls == [(None, None), (None, 1e-3)]
        assert "retrying with grid_size=1e-3" in caplog.text

    @pytest.mark.usefixtures("single_enclosure")
    def test_enclosed_tessellation_falls_back_to_jitter_when_grid_size_fails(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When the grid_size retry still fails, retry once more with jittered geometry.

        The jitter rung replaces the coarse grid rather than stacking on it, so
        the third attempt reaches momepy with the original options and can only
        be told apart from the first by the geometry it receives.
        """
        calls: list[tuple[object, object]] = []
        geometries: list[gpd.GeoDataFrame] = []

        def fake_enclosed_tessellation(**kwargs: object) -> gpd.GeoDataFrame:
            calls.append((kwargs.get("simplify"), kwargs.get("grid_size")))
            geometries.append(cast("gpd.GeoDataFrame", kwargs.get("geometry")))
            if len(calls) < 3:
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
        assert calls == [(None, None), (None, 1e-3), (None, None)]
        assert not geometries[2].geometry.geom_equals(geometries[0].geometry).all()
        assert "retrying with grid_size=1e-3" in caplog.text
        assert "retrying with jittered geometry" in caplog.text

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
        assert "overlapping or incomplete cells" in caplog.text

    def test_enclosed_tessellation_retries_cells_that_undercover_the_enclosure(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Disjoint cells that leave the enclosure uncovered must also be retried.

        Collapsed cells do not overlap, so the area checks against the enclosure
        and against the cells' own union both pass; only the coverage check
        catches them. Such a cell no longer contains the building it was built
        around, which would silently drop that building downstream.
        """
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
        collapsed_cells = [
            Polygon([(0, 0), (1, 0), (1, 10), (0, 10)]),
            Polygon([(9, 0), (10, 0), (10, 10), (9, 10)]),
        ]

        def fake_enclosed_tessellation(**kwargs: object) -> gpd.GeoDataFrame:
            cells = sane_cells if kwargs.get("grid_size") else collapsed_cells
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
        assert "overlapping or incomplete cells" in caplog.text

    def test_enclosed_tessellation_overlap_retry_falls_back_to_jitter(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A geometry-type error during the overlap retry should escalate to jitter."""
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
        calls: list[object] = []

        def fake_enclosed_tessellation(**kwargs: object) -> gpd.GeoDataFrame:
            # The first run silently degenerates; the coarser grid_size retry
            # snaps cells into non-polygonal geometry and raises; only the
            # jittered attempt partitions the enclosure.
            calls.append(kwargs.get("grid_size"))
            if kwargs.get("grid_size"):
                msg = "One of the Geometry inputs is of incorrect geometry type."
                raise TypeError(msg)
            cells = sane_cells if len(calls) > 1 else [enclosure, enclosure]
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
        assert calls == [None, 1e-3, None]
        assert result.geometry.area.sum() == pytest.approx(enclosure.area)
        assert "overlapping or incomplete cells" in caplog.text
        assert "retrying with jittered geometry" in caplog.text

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

    @pytest.mark.usefixtures("single_enclosure")
    def test_enclosed_tessellation_salvages_geometry_collections(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """GeometryCollection cells should be reduced to their polygonal parts."""
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

    @pytest.mark.usefixtures("single_enclosure")
    def test_enclosed_tessellation_reraises_unrelated_type_error(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A TypeError that does not report an incorrect geometry type should propagate."""

        def raise_unrelated(**_kwargs: object) -> gpd.GeoDataFrame:
            msg = "something else entirely"
            raise TypeError(msg)

        monkeypatch.setattr(momepy, "enclosed_tessellation", raise_unrelated)

        with pytest.raises(TypeError, match="something else entirely"):
            utils.create_tessellation(
                sample_buildings_gdf,
                primary_barriers=sample_segments_gdf,
            )

    @pytest.mark.usefixtures("single_enclosure")
    def test_enclosed_tessellation_retries_with_coarser_grid_size_on_geos_error(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A GEOS topology error should trigger a retry with a coarser grid_size."""
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

    @pytest.mark.usefixtures("single_enclosure")
    def test_enclosed_tessellation_returns_empty_when_geos_error_persists(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A GEOS error that persists at coarser precision degrades to empty output."""

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

    @pytest.mark.usefixtures("single_enclosure")
    def test_enclosed_tessellation_degrades_to_empty_with_pinned_grid_size(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A GEOS error with a caller-pinned grid_size degrades without overriding it."""
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

    @pytest.mark.usefixtures("single_enclosure")
    def test_enclosed_tessellation_pinned_grid_size_retries_with_jitter(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With a pinned grid_size, a geometry-type TypeError skips to the jitter rung."""
        calls: list[tuple[object, object]] = []

        def fake_enclosed_tessellation(**kwargs: object) -> gpd.GeoDataFrame:
            calls.append((kwargs.get("simplify"), kwargs.get("grid_size")))
            if len(calls) < 2:
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
        # The pinned grid_size is kept and simplify is never injected.
        assert calls == [(None, 1e-5), (None, 1e-5)]
        assert "retrying with jittered geometry" in caplog.text

    @pytest.mark.usefixtures("single_enclosure")
    def test_enclosed_tessellation_reraises_unknown_error_on_retry(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An unknown error raised during a retry must propagate."""

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

    def test_enclosed_tessellation_overlap_repair_jitter_failure_keeps_cells(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A repair retry that exhausts the ladder keeps the original cells.

        The repair escalates coarse grid -> jitter, and the jitter attempt
        crashes too, so no replacement cells exist. The original ones are kept
        and the broken enclosure is dropped while the good one survives.
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

        calls: list[object] = []

        def fake_enclosed_tessellation(**kwargs: object) -> gpd.GeoDataFrame:
            # The first run silently degenerates in enclosure 0; the repair
            # retry at a coarser grid raises, and the jitter attempt that
            # follows crashes in GEOS, exhausting the repair ladder.
            calls.append(kwargs.get("grid_size"))
            if kwargs.get("grid_size"):
                msg = "One of the Geometry inputs is of incorrect geometry type."
                raise TypeError(msg)
            if len(calls) > 1:
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
        assert calls == [None, 1e-3, None]
        assert "retrying with jittered geometry" in caplog.text
        assert "keeping the original cells" in caplog.text
        assert "Dropping 1 enclosure(s)" in caplog.text

    @pytest.mark.parametrize(
        ("retry_error", "expected_grid_sizes"),
        [
            pytest.param(ValueError("No objects to concatenate"), [None, 1e-3], id="empty-retry"),
            pytest.param(
                shapely.errors.GEOSException("TopologyException: side location conflict at 0 0"),
                [None, 1e-3, None],
                id="geos-retry",
            ),
            pytest.param(
                TypeError("One of the Geometry inputs is of incorrect geometry type."),
                [None, 1e-3, None],
                id="type-error-retry",
            ),
        ],
    )
    def test_enclosed_tessellation_retry_exhaustion(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
        retry_error: Exception,
        expected_grid_sizes: list[float | None],
    ) -> None:
        """Return a valid empty tessellation after the available retries fail."""
        enclosures = gpd.GeoDataFrame(
            {"eID": [1]},
            geometry=[Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])],
            crs=sample_buildings_gdf.crs,
        )
        first_error = TypeError("One of the Geometry inputs is of incorrect geometry type.")
        tessellate = mock.Mock(side_effect=[first_error, retry_error, first_error])
        monkeypatch.setattr(momepy, "enclosures", mock.Mock(return_value=enclosures))
        monkeypatch.setattr(momepy, "enclosed_tessellation", tessellate)

        result = utils.create_tessellation(
            sample_buildings_gdf, primary_barriers=sample_segments_gdf
        )

        assert result.empty
        assert result.crs == sample_buildings_gdf.crs
        assert [
            call.kwargs.get("grid_size") for call in tessellate.call_args_list
        ] == expected_grid_sizes
        assert all(call.kwargs.get("simplify") is None for call in tessellate.call_args_list)
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

        monkeypatch.setattr(
            "city2graph.utils.tessellation.momepy.enclosed_tessellation", raise_boom
        )

        with pytest.raises(ValueError, match="boom"):
            utils.create_tessellation(sample_buildings_gdf, primary_barriers=sample_segments_gdf)
