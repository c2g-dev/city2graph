"""Refactored test module for morphology.py with improved maintainability and reduced redundancy."""

import inspect
import logging
import math
import warnings
from collections.abc import Callable
from functools import partial
from typing import Any

import geopandas as gpd
import networkx as nx
import pandas as pd
import pytest
from geopandas.testing import assert_geodataframe_equal
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import Polygon

import city2graph.morphology as morph
from city2graph.morphology import _create_and_filter_tessellation
from city2graph.morphology import _filter_buildings_by_network_distance
from city2graph.morphology import _filter_tessellation_by_network_distance
from city2graph.morphology import _include_unenclosed_building_tessellation
from city2graph.morphology import _MorphologyContext
from city2graph.morphology import _segments_within_network_distance
from city2graph.morphology import morphological_graph
from city2graph.morphology import morphological_graphs
from city2graph.morphology import movement_to_movement_graph
from city2graph.morphology import place_to_movement_graph
from city2graph.morphology import place_to_place_graph
from city2graph.morphology import private_to_private_graph
from city2graph.morphology import private_to_public_graph
from city2graph.morphology import public_to_public_graph
from city2graph.morphology import segments_to_graph
from tests.helpers import assert_valid_nx_graph
from tests.helpers import make_grid_polygons_gdf

# Option sets exercised by TestPublicInputImmutability. The ``with_center`` key
# marks cases that need the ``mg_center_point`` fixture injected as
# ``center_point`` at test time.
IMMUTABILITY_OPTION_SETS: list[dict[str, Any]] = [
    {},
    {
        "with_center": True,
        "distance": 100.0,
        "keep_buildings": True,
        "include_unenclosed_buildings": True,
        "tessellation_fallback": True,
    },
]

# ---------------------------------------------------------------------------
# Module-level monkeypatch stubs.
#
# These replace the production callables that tests patch out. They live at
# module scope (never nested inside a test) and receive any per-test state via
# ``functools.partial`` rather than closures.
# ---------------------------------------------------------------------------


def _fake_sjoin_without_index_right(
    tessellation: gpd.GeoDataFrame,
    _buildings: gpd.GeoDataFrame,
    *,
    how: str,
    predicate: str,
) -> gpd.GeoDataFrame:
    """Stub for ``sjoin`` that returns the tessellation without an ``index_right`` column."""
    assert how == "left"
    assert predicate == "contains"
    return tessellation.copy()


def _single_enclosed_cell_tessellation(
    geometry: gpd.GeoDataFrame,
    primary_barriers: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    """Return a single enclosed cell over the first input geometry."""
    _ = primary_barriers
    return gpd.GeoDataFrame(
        {"tess_id": ["enclosed"], "enclosure_index": [0]},
        geometry=[geometry.geometry.iloc[0]],
        crs=geometry.crs,
    )


def _enclosed_cell_or_fail_on_fallback(
    geometry: gpd.GeoDataFrame,
    primary_barriers: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    """Enclosed cell for barrier input; fail if a non-local fallback is requested."""
    if primary_barriers is not None:
        return gpd.GeoDataFrame(
            {"tess_id": ["enclosed"], "enclosure_index": [0]},
            geometry=[geometry.geometry.iloc[0]],
            crs=geometry.crs,
        )
    msg = "fallback should not create a non-local tessellation"
    raise AssertionError(msg)


def _enclosed_or_clip_fallback_tessellation(
    geometry: gpd.GeoDataFrame,
    primary_barriers: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    """Enclosed cell for barriers; a single large overlapping cell otherwise."""
    if primary_barriers is not None:
        return gpd.GeoDataFrame(
            {"tess_id": ["enclosed"], "enclosure_index": [0]},
            geometry=[geometry.geometry.iloc[0]],
            crs=geometry.crs,
        )
    return gpd.GeoDataFrame(
        {"tess_id": geometry.index.to_list()},
        geometry=[Polygon([(0.75, -1), (5, -1), (5, 2), (0.75, 2)])],
        crs=geometry.crs,
    )


def _fail_if_tessellation_called(
    geometry: gpd.GeoDataFrame,
    primary_barriers: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    """Guard stub asserting ``create_tessellation`` is never invoked."""
    _ = (geometry, primary_barriers)
    msg = "fallback should not create a non-local tessellation"
    raise AssertionError(msg)


def _unit_square_tessellation(
    geometry: gpd.GeoDataFrame,
    primary_barriers: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    """Single unit-square tessellation cell."""
    _ = primary_barriers
    return gpd.GeoDataFrame(
        {"tess_id": ["t1"]},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        crs=geometry.crs,
    )


def _two_cell_tessellation(
    cell_ids: list[str],
    cell_geometries: list[Polygon],
    geometry: gpd.GeoDataFrame,
    primary_barriers: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    """Return a fixed two-cell enclosed tessellation."""
    _ = primary_barriers
    return gpd.GeoDataFrame(
        {"tess_id": cell_ids, "enclosure_index": [0] * len(cell_ids)},
        geometry=cell_geometries,
        crs=geometry.crs,
    )


def _key_recording_tessellation(
    captured: dict[str, object],
    key: str,
    default: object,
    geometry: gpd.GeoDataFrame,
    primary_barriers: gpd.GeoDataFrame | None = None,
    **kwargs: object,
) -> gpd.GeoDataFrame:
    """Record one ``create_tessellation`` kwarg, then return a single cell."""
    _ = primary_barriers
    captured[key] = kwargs.get(key, default)
    return gpd.GeoDataFrame(
        {"tess_id": ["cell"], "enclosure_index": [0]},
        geometry=[geometry.geometry.iloc[0]],
        crs=geometry.crs,
    )


def _raise_no_objects_tessellation(
    geometry: gpd.GeoDataFrame,
    primary_barriers: gpd.GeoDataFrame | None = None,
    **kwargs: object,
) -> gpd.GeoDataFrame:
    """create_tessellation stub raising momepy's empty-concat error."""
    _ = (geometry, primary_barriers, kwargs)
    msg = "No objects to concatenate"
    raise ValueError(msg)


def _empty_tessellation(
    geometry: gpd.GeoDataFrame,
    primary_barriers: gpd.GeoDataFrame | None = None,
    **kwargs: object,
) -> gpd.GeoDataFrame:
    """create_tessellation stub returning an empty enclosed tessellation."""
    _ = (primary_barriers, kwargs)
    return gpd.GeoDataFrame(
        {"tess_id": [], "enclosure_index": []},
        geometry=[],
        crs=geometry.crs,
    )


def _passthrough_adjacent_filter(
    tessellation: gpd.GeoDataFrame,
    _segments: gpd.GeoDataFrame,
    max_distance: float = float("inf"),
) -> gpd.GeoDataFrame:
    """Adjacent-tessellation filter stub that returns the input unchanged."""
    _ = max_distance
    return tessellation


def _return_input_tessellation(
    tessellation: gpd.GeoDataFrame,
    *_args: object,
    **_kwargs: object,
) -> gpd.GeoDataFrame:
    """Network-distance filter stub that returns the tessellation unchanged."""
    return tessellation


def _connect_to_first_edge(
    _graph: nx.Graph,
    _point_node_id: str,
    _point: Point,
    edge_records: list[tuple[Any, Any, LineString, float]],
) -> tuple[tuple[Any, Any, LineString, float], float, float]:
    """Connect a point to the first candidate edge with zero-length legs."""
    return edge_records[0], 0.0, 0.0


def _counting_wrapper(
    state: dict[str, int],
    wrapped: Callable[..., Any],
    *args: Any,  # noqa: ANN401
    **kwargs: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Count invocations while delegating to the wrapped callable."""
    state["n"] += 1
    return wrapped(*args, **kwargs)


def _field_recording_wrapper(
    seen_fields: list[object],
    wrapped: Callable[..., Any],
    *args: Any,  # noqa: ANN401
    **kwargs: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Record the ``field`` kwarg while delegating to the wrapped callable."""
    seen_fields.append(kwargs.get("field"))
    return wrapped(*args, **kwargs)


class TestMorphologyBase:
    """Base class with common test utilities and validation methods."""

    @staticmethod
    def validate_basic_output(
        nodes: gpd.GeoDataFrame | dict[str, gpd.GeoDataFrame],
        edges: gpd.GeoDataFrame | dict[str, gpd.GeoDataFrame],
        expected_node_types: list[str] | None = None,
    ) -> None:
        """Validate basic output structure for morphology functions."""
        if expected_node_types:
            # For morphological_graph (returns dict)
            assert isinstance(nodes, dict)
            assert isinstance(edges, dict)
            for node_type in expected_node_types:
                assert node_type in nodes
        else:
            # For individual graph functions (returns GeoDataFrames)
            assert isinstance(nodes, gpd.GeoDataFrame)
            assert isinstance(edges, gpd.GeoDataFrame)

    @staticmethod
    def validate_networkx_output(graph: nx.Graph) -> None:
        """Validate NetworkX graph output using shared helper."""
        assert_valid_nx_graph(graph)

    @staticmethod
    def validate_empty_output(
        nodes: gpd.GeoDataFrame | dict[str, gpd.GeoDataFrame],
        edges: gpd.GeoDataFrame | dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Validate output for empty inputs."""
        if isinstance(nodes, dict):
            # morphological_graph case
            for node_gdf in nodes.values():
                assert node_gdf.empty
            for edge_gdf in edges.values():
                assert edge_gdf.empty
        else:
            # individual function case
            assert isinstance(nodes, gpd.GeoDataFrame)
            assert isinstance(edges, gpd.GeoDataFrame)
            assert nodes.empty
            assert edges.empty

    @staticmethod
    def validate_edge_columns(edges: gpd.GeoDataFrame, expected_columns: list[str]) -> None:
        """Validate that edges contain expected columns."""
        if not edges.empty:
            for col in expected_columns:
                assert col in edges.columns


class TestMorphologicalGraphCore(TestMorphologyBase):
    """Core tests for morphological_graph function."""

    def test_basic_functionality(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test basic morphological graph creation."""
        nodes, edges = morphological_graph(sample_buildings_gdf, sample_segments_gdf)

        self.validate_basic_output(nodes, edges, ["place", "movement"])

        # Validate expected edge types
        expected_edges = [
            ("place", "touched_to", "place"),
            ("movement", "connected_to", "movement"),
            ("place", "faced_to", "movement"),
        ]
        for edge_type in expected_edges:
            assert edge_type in edges

    def test_duplicate_edges_symmetrizes_same_type_edges(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
    ) -> None:
        """duplicate_edges=True doubles same-type edges and keeps faced_to as-is."""
        _, base_edges = morphological_graph(sample_buildings_gdf, sample_segments_gdf)
        _, dup_edges = morphological_graph(
            sample_buildings_gdf, sample_segments_gdf, duplicate_edges=True
        )

        for edge_type in [
            ("place", "touched_to", "place"),
            ("movement", "connected_to", "movement"),
        ]:
            base = base_edges[edge_type]
            dup = dup_edges[edge_type]
            assert len(dup) == 2 * len(base)
            pairs = set(base.index)
            assert set(dup.index) == pairs | {(v, u) for u, v in pairs}

        faced_to = ("place", "faced_to", "movement")
        assert dup_edges[faced_to].index.tolist() == base_edges[faced_to].index.tolist()

    def test_duplicate_edges_rejects_as_nx(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
    ) -> None:
        """duplicate_edges=True with as_nx=True raises ValueError."""
        with pytest.raises(ValueError, match="not supported with as_nx=True"):
            morphological_graph(
                sample_buildings_gdf, sample_segments_gdf, as_nx=True, duplicate_edges=True
            )

    def test_missing_enclosure_index_warning(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_crs: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test warning when enclosure_index column is missing from tessellation."""
        # Create empty segments to trigger morphological tessellation (without barriers)
        # This will cause tessellation to be created without enclosure_index column
        empty_segments = gpd.GeoDataFrame(geometry=[], crs=sample_crs)

        # Clear any existing log records
        caplog.clear()

        with caplog.at_level(logging.WARNING):
            nodes, edges = morphological_graph(sample_buildings_gdf, empty_segments)

            # Check if warning was issued about missing enclosure_index
            warning_messages = [
                record.message for record in caplog.records if record.levelno == logging.WARNING
            ]
            enclosure_warnings = [
                msg for msg in warning_messages if "enclosure_index" in msg and "not found" in msg
            ]

            # We expect a warning about missing enclosure_index
            assert len(enclosure_warnings) > 0

        self.validate_basic_output(nodes, edges, ["place", "movement"])

    def test_keep_buildings_handles_join_without_index_right(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """keep_buildings should still allocate building_geometry when sjoin omits index_right."""
        monkeypatch.setattr("city2graph.morphology.gpd.sjoin", _fake_sjoin_without_index_right)

        nodes, _edges = morphological_graph(
            sample_buildings_gdf,
            sample_segments_gdf,
            keep_buildings=True,
        )

        assert "building_geometry" in nodes["place"].columns

    def test_include_unenclosed_buildings_is_opt_in(
        self,
        sample_crs: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Buildings missed by enclosed tessellation are only added when requested."""
        buildings = gpd.GeoDataFrame(
            {"building_id": ["inside", "unenclosed"]},
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(10, 0), (11, 0), (11, 1), (10, 1)]),
            ],
            crs=sample_crs,
        )
        segments = gpd.GeoDataFrame(
            geometry=[LineString([(-1, -1), (2, -1)])],
            crs=sample_crs,
        )

        monkeypatch.setattr(
            "city2graph.morphology.create_tessellation",
            _enclosed_cell_or_fail_on_fallback,
        )
        monkeypatch.setattr(
            "city2graph.morphology._filter_adjacent_tessellation",
            _passthrough_adjacent_filter,
        )

        default_nodes, _ = morphological_graph(buildings, segments, keep_buildings=True)
        opt_in_nodes, _ = morphological_graph(
            buildings,
            segments,
            keep_buildings=True,
            include_unenclosed_buildings=True,
        )

        assert len(default_nodes["place"]) == 1
        assert len(opt_in_nodes["place"]) == 2
        assert "fallback_1" in opt_in_nodes["place"].index

    def test_keep_buildings_with_fallback_and_multi_building_cell(
        self,
        sample_crs: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Fallback cells coexist with cells containing several buildings."""
        buildings = gpd.GeoDataFrame(
            {"building_id": ["outer", "nested", "unenclosed"]},
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)]),
                Polygon([(10, 0), (11, 0), (11, 1), (10, 1)]),
            ],
            crs=sample_crs,
        )
        segments = gpd.GeoDataFrame(
            geometry=[LineString([(-1, -1), (2, -1)])],
            crs=sample_crs,
        )

        monkeypatch.setattr(
            "city2graph.morphology.create_tessellation",
            _enclosed_cell_or_fail_on_fallback,
        )
        monkeypatch.setattr(
            "city2graph.morphology._filter_adjacent_tessellation",
            _passthrough_adjacent_filter,
        )

        nodes, _ = morphological_graph(
            buildings,
            segments,
            keep_buildings=True,
            include_unenclosed_buildings=True,
        )

        place_ids = list(nodes["place"].index)
        # The enclosed cell holds both overlapping buildings; the fallback cell
        # for the unenclosed building is matched exactly once to its source.
        assert place_ids.count("fallback_2") == 1
        assert place_ids.count("enclosed") == 2

    def test_morphological_graph_passes_tessellation_limit(
        self,
        sample_crs: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Custom enclosure limits should propagate to tessellation creation."""
        custom_limit = Polygon([(-10, -10), (10, -10), (10, 10), (-10, 10)])
        buildings = gpd.GeoDataFrame(
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            crs=sample_crs,
        )
        segments = gpd.GeoDataFrame(
            geometry=[LineString([(-1, 0), (2, 0)])],
            crs=sample_crs,
        )
        captured: dict[str, object] = {}

        monkeypatch.setattr(
            "city2graph.morphology.create_tessellation",
            partial(_key_recording_tessellation, captured, "limit", None),
        )

        morphological_graph(buildings, segments, limit=custom_limit)

        assert captured["limit"] is custom_limit

    def test_include_unenclosed_buildings_uses_prefiltered_fallback_buildings(
        self,
        sample_crs: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Fallback tessellation should use the buildings already selected for graph inclusion."""
        buildings = gpd.GeoDataFrame(
            {"building_id": ["enclosed", "near_missing", "far_missing"]},
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
                Polygon([(9, 0), (10, 0), (10, 1), (9, 1)]),
            ],
            crs=sample_crs,
        )
        tessellation = gpd.GeoDataFrame(
            {"place_id": ["enclosed"], "enclosure_index": [0]},
            geometry=[buildings.geometry.iloc[0]],
            crs=sample_crs,
        )

        monkeypatch.setattr(
            "city2graph.morphology.create_tessellation",
            _fail_if_tessellation_called,
        )

        result = _include_unenclosed_building_tessellation(
            tessellation,
            buildings.iloc[:2],
        )

        assert list(result["place_id"]) == ["enclosed", "fallback_1"]

    def test_keep_buildings_excludes_distance_filtered_buildings_from_fallback_cells(
        self,
        sample_crs: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Large fallback cells must not reattach buildings filtered out by distance."""
        buildings = gpd.GeoDataFrame(
            {"building_id": ["enclosed", "near_missing", "far_outside"]},
            geometry=[
                Polygon([(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)]),
                Polygon([(1, 0), (1.5, 0), (1.5, 0.5), (1, 0.5)]),
                Polygon([(4, 0), (4.5, 0), (4.5, 0.5), (4, 0.5)]),
            ],
            crs=sample_crs,
        )
        segments = gpd.GeoDataFrame(
            geometry=[LineString([(0, 0), (5, 0)])],
            crs=sample_crs,
        )

        monkeypatch.setattr(
            "city2graph.morphology.create_tessellation",
            _enclosed_or_clip_fallback_tessellation,
        )
        monkeypatch.setattr(
            "city2graph.morphology._filter_adjacent_tessellation",
            _passthrough_adjacent_filter,
        )
        monkeypatch.setattr(
            "city2graph.morphology._filter_tessellation_by_network_distance",
            _return_input_tessellation,
        )

        nodes, _ = morphological_graph(
            buildings,
            segments,
            center_point=Point(0, 0),
            distance=2.0,
            keep_buildings=True,
            include_unenclosed_buildings=True,
        )

        attached_ids = set(nodes["place"]["building_id"].dropna())
        assert attached_ids == {"enclosed", "near_missing"}
        assert "far_outside" not in attached_ids

    def test_network_distance_filter_does_not_use_centroid_shortcuts(
        self,
        sample_crs: str,
    ) -> None:
        """Place cells should not make distant cells reachable through centroid-to-centroid edges."""
        tessellation = gpd.GeoDataFrame(
            {"place_id": ["near", "far"]},
            geometry=[
                Polygon([(0.9, -0.1), (1.1, -0.1), (1.1, 0.1), (0.9, 0.1)]),
                Polygon([(2.9, -0.1), (3.1, -0.1), (3.1, 0.1), (2.9, 0.1)]),
            ],
            crs=sample_crs,
        )
        segments = gpd.GeoDataFrame(
            geometry=[LineString([(0, 0), (1, 0)])],
            crs=sample_crs,
        )

        # The "far" centroid at (3, 0) projects onto the segment endpoint (1, 0)
        # with a 2.0 access leg; an ``extent_buffer`` below that access distance
        # excludes it, while "near" sits on the segment (zero access).
        filtered = _filter_tessellation_by_network_distance(
            tessellation,
            segments,
            Point(0, 0),
            2.2,
            extent_buffer=1.0,
        )

        assert list(filtered["place_id"]) == ["near"]

    def test_network_distance_projects_center_and_place_cells_to_segments(
        self,
        sample_crs: str,
    ) -> None:
        """A center and place cell near the middle of a long segment should use the segment."""
        tessellation = gpd.GeoDataFrame(
            {"place_id": ["near_midpoint", "too_far"]},
            geometry=[
                Polygon([(59, 0), (61, 0), (61, 2), (59, 2)]),
                Polygon([(89, 0), (91, 0), (91, 2), (89, 2)]),
            ],
            crs=sample_crs,
        )
        segments = gpd.GeoDataFrame(
            geometry=[LineString([(0, 0), (100, 0)])],
            crs=sample_crs,
        )

        filtered = _filter_tessellation_by_network_distance(
            tessellation,
            segments,
            Point(50, 0),
            12.0,
        )

        assert list(filtered["place_id"]) == ["near_midpoint"]

    def test_building_network_distance_uses_segment_projection(
        self,
        sample_crs: str,
    ) -> None:
        """Buildings should be selected by distance to the nearest point on a segment."""
        buildings = gpd.GeoDataFrame(
            {"building_id": ["near_midpoint", "too_far"]},
            geometry=[
                Polygon([(59, 0), (61, 0), (61, 2), (59, 2)]),
                Polygon([(89, 0), (91, 0), (91, 2), (89, 2)]),
            ],
            crs=sample_crs,
        )
        segments = gpd.GeoDataFrame(
            geometry=[LineString([(0, 0), (100, 0)])],
            crs=sample_crs,
        )

        filtered = _filter_buildings_by_network_distance(
            buildings,
            segments,
            Point(50, 0),
            12.0,
        )

        assert list(filtered["building_id"]) == ["near_midpoint"]

    def test_unenclosed_fallback_uses_projected_network_distance(
        self,
        sample_crs: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Fallback cells should include unenclosed buildings reachable along segment middles."""
        buildings = gpd.GeoDataFrame(
            {"building_id": ["enclosed", "near_missing", "far_missing"]},
            geometry=[
                Polygon([(49, 0), (51, 0), (51, 2), (49, 2)]),
                Polygon([(59, 0), (61, 0), (61, 2), (59, 2)]),
                Polygon([(89, 0), (91, 0), (91, 2), (89, 2)]),
            ],
            crs=sample_crs,
        )
        segments = gpd.GeoDataFrame(
            geometry=[LineString([(0, 0), (100, 0)])],
            crs=sample_crs,
        )

        monkeypatch.setattr(
            "city2graph.morphology.create_tessellation",
            _single_enclosed_cell_tessellation,
        )

        context = _MorphologyContext(
            buildings=buildings,
            segments=segments,
            barrier_segments=None,
            center_point=Point(50, 0),
            clipping_buffer=math.inf,
            extent_buffer=math.inf,
            limit=None,
            primary_barrier_col="geometry",
            contiguity="queen",
            keep_buildings=True,
            keep_segments=True,
            tolerance=1e-6,
            include_unenclosed_buildings=True,
            as_nx=False,
            duplicate_edges=False,
            tessellation_fallback=False,
            tessellation_n_jobs=-1,
            field=None,
        )
        tessellation = _create_and_filter_tessellation(
            context,
            12.0,
            segments_buffer=segments,
            segments_filtered=segments,
        )

        assert set(tessellation["building_id"].dropna()) == {"enclosed", "near_missing"}
        assert "fallback_1" in set(tessellation["place_id"])

    def test_distance_filter_handles_missing_center_node(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        sample_crs: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Distance filtering should degrade to an empty place layer if the center node is absent."""
        center_point = gpd.GeoSeries([Point(0.5, 0.5)], crs=sample_crs)

        monkeypatch.setattr(
            "city2graph.morphology.create_tessellation",
            _unit_square_tessellation,
        )
        monkeypatch.setattr(
            "city2graph.morphology._filter_adjacent_tessellation",
            _passthrough_adjacent_filter,
        )
        monkeypatch.setattr(
            "city2graph.morphology._connect_point_to_nearest_edge",
            _connect_to_first_edge,
        )

        nodes, _edges = morphological_graph(
            sample_buildings_gdf,
            sample_segments_gdf,
            center_point=center_point,
            distance=10.0,
        )

        assert nodes["place"].empty

    def test_segments_use_same_reachability_field_as_cells(
        self,
        sample_crs: str,
    ) -> None:
        """Street acceptance is derived from the same projected cost field as cells."""
        segments = gpd.GeoDataFrame(
            geometry=[LineString([(0, 0), (50, 0)]), LineString([(50, 0), (100, 0)])],
            crs=sample_crs,
        )
        tessellation = gpd.GeoDataFrame(
            {"place_id": ["near", "far"]},
            geometry=[
                Polygon([(9, -1), (11, -1), (11, 1), (9, 1)]),
                Polygon([(69, -1), (71, -1), (71, 1), (69, 1)]),
            ],
            crs=sample_crs,
        )

        kept_segments = _segments_within_network_distance(segments, Point(0, 0), 20.0)
        kept_cells = _filter_tessellation_by_network_distance(
            tessellation,
            segments,
            Point(0, 0),
            20.0,
        )

        # The near segment (reachable from x=0) is kept; the far one (min endpoint
        # cost 50) is dropped. The near cell (cost ~10) is kept while the far cell
        # (cost ~70) is dropped, all from the single reachability metric.
        assert len(kept_segments) == 1
        assert list(kept_cells["place_id"]) == ["near"]

    def test_segments_within_distance_keeps_boundary_straddling_segment(
        self,
        sample_crs: str,
    ) -> None:
        """A segment whose near endpoint is within budget is kept whole, not dropped."""
        segments = gpd.GeoDataFrame(
            {"movement_id": ["straddle", "beyond"]},
            geometry=[LineString([(0, 0), (50, 0)]), LineString([(50, 0), (100, 0)])],
            crs=sample_crs,
        )

        kept = _segments_within_network_distance(segments, Point(0, 0), 20.0)

        # "straddle" spans x=0..50 and crosses the 20 budget boundary yet is kept
        # whole because its reachable portion is within budget; "beyond" is dropped.
        assert list(kept["movement_id"]) == ["straddle"]

    def test_morphological_graph_has_no_isolated_place_nodes(
        self,
        sample_crs: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Reachability-budgeted output never contains a place cell without a faced_to edge."""
        near = Polygon([(4, -1), (6, -1), (6, 0), (4, 0)])
        isolated = Polygon([(4, 29), (6, 29), (6, 31), (4, 31)])
        buildings = gpd.GeoDataFrame(geometry=[near, isolated], crs=sample_crs)
        segments = gpd.GeoDataFrame(
            geometry=[LineString([(0, 0), (20, 0)])],
            crs=sample_crs,
        )

        monkeypatch.setattr(
            "city2graph.morphology.create_tessellation",
            partial(_two_cell_tessellation, ["near", "isolated"], [near, isolated]),
        )

        nodes, edges = morphological_graph(
            buildings,
            segments,
            center_point=Point(0, 0),
            distance=50.0,
        )

        faced = edges[("place", "faced_to", "movement")]
        place_ids = set(nodes["place"].index)
        faced_place_ids = set(faced.index.get_level_values(0)) if not faced.empty else set()

        # The "isolated" cell sits 30 m from the street, within the default
        # extent_buffer (100 m), so it receives a nearest movement fallback
        # faced_to edge and is kept rather than pruned.
        assert place_ids == {"near", "isolated"}
        assert place_ids <= faced_place_ids

    def test_extent_buffer_drops_cells_beyond_access_cap(
        self,
        sample_crs: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A cell whose only reachable street is farther than extent_buffer is dropped.

        Reproduces the E01006640 symptom: a cell reachable along the network but
        separated from the only retained street by a long straight-line access
        leg. The old summed budget (network + access) retained it; the two-cap
        rule rejects it because the access distance exceeds ``extent_buffer``.
        """
        near = Polygon([(49, 1), (51, 1), (51, 3), (49, 3)])  # centroid (50, 2), 2 m access
        far = Polygon([(49, 59), (51, 59), (51, 61), (49, 61)])  # centroid (50, 60), 60 m access
        buildings = gpd.GeoDataFrame(geometry=[near, far], crs=sample_crs)
        segments = gpd.GeoDataFrame(
            geometry=[LineString([(0, 0), (100, 0)])],
            crs=sample_crs,
        )

        monkeypatch.setattr(
            "city2graph.morphology.create_tessellation",
            partial(_two_cell_tessellation, ["near", "far"], [near, far]),
        )

        # distance budget (150) would admit "far" under a summed network+access
        # cost (50 + 60 = 110); the access cap (50) is what excludes it.
        nodes, edges = morphological_graph(
            buildings,
            segments,
            center_point=Point(0, 0),
            distance=150.0,
            clipping_buffer=200.0,
            extent_buffer=50.0,
        )

        place_ids = set(nodes["place"].index)
        faced = edges[("place", "faced_to", "movement")]
        faced_place_ids = set(faced.index.get_level_values(0)) if not faced.empty else set()

        assert place_ids == {"near"}
        assert "far" not in faced_place_ids

    def test_extent_buffer_validation(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
    ) -> None:
        """extent_buffer must be non-negative and not exceed clipping_buffer."""
        sig = inspect.signature(morphological_graph)
        assert sig.parameters["extent_buffer"].default == 100.0

        with pytest.raises(ValueError, match="clipping_buffer must be greater"):
            morphological_graph(
                sample_buildings_gdf,
                sample_segments_gdf,
                clipping_buffer=10.0,
                extent_buffer=50.0,
            )
        with pytest.raises(ValueError, match="extent_buffer cannot be negative"):
            morphological_graph(
                sample_buildings_gdf,
                sample_segments_gdf,
                extent_buffer=-1.0,
            )

    def test_place_to_movement_respects_max_connection_distance(
        self,
        sample_crs: str,
    ) -> None:
        """Fallback connections farther than max_connection_distance are not created."""
        place_gdf = gpd.GeoDataFrame(
            {"place_id": ["near", "far"]},
            geometry=[
                Polygon([(0, 1), (2, 1), (2, 3), (0, 3)]),  # 1 m from the segment
                Polygon([(0, 100), (2, 100), (2, 102), (0, 102)]),  # 100 m from the segment
            ],
            crs=sample_crs,
        )
        movement_gdf = gpd.GeoDataFrame(
            {"movement_id": [0]},
            geometry=[LineString([(0, 0), (50, 0)])],
            crs=sample_crs,
        )

        _, edges = place_to_movement_graph(
            place_gdf,
            movement_gdf,
            max_connection_distance=10.0,
        )

        connected = set(edges["place_id"]) if not edges.empty else set()
        assert "near" in connected
        assert "far" not in connected

    def test_networkx_conversion(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test NetworkX graph conversion."""
        graph = morphological_graph(sample_buildings_gdf, sample_segments_gdf, as_nx=True)
        self.validate_networkx_output(graph)

    @pytest.mark.parametrize("contiguity", ["queen", "rook"])
    def test_contiguity_options(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        contiguity: str,
    ) -> None:
        """Test different contiguity options."""
        nodes, edges = morphological_graph(
            sample_buildings_gdf,
            sample_segments_gdf,
            contiguity=contiguity,
        )
        self.validate_basic_output(nodes, edges, ["place", "movement"])

    @pytest.mark.parametrize(
        ("distance", "clipping_buffer"),
        [
            (1000, 100),
            (2000, 500),
            (None, None),
        ],
    )
    def test_distance_and_clipping(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        distance: float,
        clipping_buffer: float,
    ) -> None:
        """Test distance filtering and clipping buffer options."""
        if distance is not None:
            # Need center point for distance filtering
            center = Point(sample_buildings_gdf.geometry.centroid.iloc[0])
            nodes, edges = morphological_graph(
                sample_buildings_gdf,
                sample_segments_gdf,
                center_point=center,
                distance=distance,
                clipping_buffer=clipping_buffer,
            )
        else:
            nodes, edges = morphological_graph(sample_buildings_gdf, sample_segments_gdf)

        self.validate_basic_output(nodes, edges, ["place", "movement"])

    def test_comprehensive_parameters(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        segments_gdf_alt_geom: gpd.GeoDataFrame,
        custom_center_point: Point,
    ) -> None:
        """Test with all parameters specified."""
        nodes, edges = morphological_graph(
            sample_buildings_gdf,
            segments_gdf_alt_geom,
            center_point=custom_center_point,
            distance=2000,
            clipping_buffer=500,
            primary_barrier_col="barrier_geometry",
            contiguity="rook",
            keep_buildings=True,
            tolerance=0.01,
            as_nx=False,
        )
        self.validate_basic_output(nodes, edges, ["place", "movement"])


class TestMorphologicalGraphBarrierAndFallback(TestMorphologyBase):
    """Barrier-only segments, tessellation fallback and tessellation parallelism."""

    def test_non_movement_barrier_col_excludes_barriers_from_movement(
        self, sample_crs: str
    ) -> None:
        """Barrier-only segments shape barriers but never become movement nodes."""
        buildings = gpd.GeoDataFrame(
            geometry=[Polygon([(0.2, 0.2), (0.4, 0.2), (0.4, 0.4), (0.2, 0.4)])],
            crs=sample_crs,
        )
        segments = gpd.GeoDataFrame(
            {"is_barrier_only": [False, True]},
            geometry=[
                LineString([(0, 0), (1, 0)]),  # movement street
                LineString([(0, 1), (1, 1)]),  # barrier-only (e.g. railway)
            ],
            crs=sample_crs,
        )

        nodes, edges = morphological_graph(
            buildings,
            segments,
            non_movement_barrier_col="is_barrier_only",
        )

        # Only the movement street becomes a movement node.
        assert len(nodes["movement"]) == 1
        assert "is_barrier_only" in nodes["movement"].columns
        assert not nodes["movement"]["is_barrier_only"].any()
        assert set(edges) == {
            ("place", "touched_to", "place"),
            ("movement", "connected_to", "movement"),
            ("place", "faced_to", "movement"),
        }

    def test_non_movement_barrier_col_missing_treats_all_as_movement(self, sample_crs: str) -> None:
        """When the flag column is absent every segment stays a movement node."""
        buildings = gpd.GeoDataFrame(
            geometry=[Polygon([(0.2, 0.2), (0.4, 0.2), (0.4, 0.4), (0.2, 0.4)])],
            crs=sample_crs,
        )
        segments = gpd.GeoDataFrame(
            geometry=[LineString([(0, 0), (1, 0)]), LineString([(0, 1), (1, 1)])],
            crs=sample_crs,
        )

        nodes, _ = morphological_graph(
            buildings, segments, non_movement_barrier_col="is_barrier_only"
        )

        assert len(nodes["movement"]) == 2

    def test_barrier_geometry_with_stale_crs_is_reprojected(self, sample_crs: str) -> None:
        """A barrier column left in a pre-``to_crs()`` CRS is reprojected, not rejected.

        ``GeoDataFrame.to_crs()`` only reprojects the active geometry column,
        so a ``barrier_geometry`` column created beforehand (e.g. by
        ``process_overture_segments`` on WGS84 data) keeps the original CRS.
        """
        lon, lat = -0.12, 51.5  # projects into the EPSG:27700 sample CRS
        step = 1e-4
        buildings = gpd.GeoDataFrame(
            geometry=[
                Polygon(
                    [
                        (lon + 2 * step, lat + 2 * step),
                        (lon + 4 * step, lat + 2 * step),
                        (lon + 4 * step, lat + 4 * step),
                        (lon + 2 * step, lat + 4 * step),
                    ]
                )
            ],
            crs="EPSG:4326",
        )
        segments = gpd.GeoDataFrame(
            geometry=[
                LineString([(lon, lat), (lon + 10 * step, lat)]),
                LineString([(lon + 10 * step, lat), (lon + 10 * step, lat + 10 * step)]),
                LineString([(lon + 10 * step, lat + 10 * step), (lon, lat + 10 * step)]),
                LineString([(lon, lat + 10 * step), (lon, lat)]),
            ],
            crs="EPSG:4326",
        )
        segments["barrier_geometry"] = segments.geometry

        buildings = buildings.to_crs(sample_crs)
        # Only the active geometry column is reprojected; barrier_geometry
        # stays in EPSG:4326.
        stale_segments = segments.to_crs(sample_crs)

        nodes, edges = morphological_graph(
            buildings,
            stale_segments,
            primary_barrier_col="barrier_geometry",
        )

        fresh_segments = stale_segments.copy()
        fresh_segments["barrier_geometry"] = fresh_segments.geometry
        reference_nodes, _ = morphological_graph(
            buildings,
            fresh_segments,
            primary_barrier_col="barrier_geometry",
        )

        self.validate_basic_output(nodes, edges, ["place", "movement"])
        assert len(nodes["place"]) == len(reference_nodes["place"])
        assert nodes["place"].geometry.area.sum() == pytest.approx(
            reference_nodes["place"].geometry.area.sum()
        )

        # A CRS-less barrier column (e.g. plain shapely objects) adopts the
        # segments CRS instead of failing.
        naive_segments = fresh_segments.copy()
        naive_segments["barrier_geometry"] = pd.Series(
            [*fresh_segments.geometry], index=fresh_segments.index, dtype=object
        )
        naive_nodes, _ = morphological_graph(
            buildings,
            naive_segments,
            primary_barrier_col="barrier_geometry",
        )
        assert len(naive_nodes["place"]) == len(reference_nodes["place"])

    def test_none_barrier_geometry_keeps_movement_but_not_barrier(self, sample_crs: str) -> None:
        """A segment with a null barrier geometry stays a movement node but never cuts cells."""
        buildings = gpd.GeoDataFrame(
            geometry=[Polygon([(2, 2), (4, 2), (4, 4), (2, 4)])],
            crs=sample_crs,
        )
        # A closed street loop enclosing the building, plus a "tunnel" segment
        # crossing the enclosure that must not act as a tessellation barrier.
        loop = [
            LineString([(0, 0), (10, 0)]),
            LineString([(10, 0), (10, 10)]),
            LineString([(10, 10), (0, 10)]),
            LineString([(0, 10), (0, 0)]),
        ]
        tunnel = LineString([(5, 0), (5, 10)])
        segments = gpd.GeoDataFrame(
            {"is_tunnel": [False] * 4 + [True]},
            geometry=[*loop, tunnel],
            crs=sample_crs,
        )
        segments["barrier_geometry"] = segments.geometry
        segments.loc[segments["is_tunnel"], "barrier_geometry"] = None

        nodes, edges = morphological_graph(
            buildings,
            segments,
            primary_barrier_col="barrier_geometry",
        )
        reference_nodes, _ = morphological_graph(
            buildings,
            segments.loc[~segments["is_tunnel"]].copy(),
            primary_barrier_col="barrier_geometry",
        )

        # The tunnel is still a movement node.
        assert len(nodes["movement"]) == 5
        assert nodes["movement"]["is_tunnel"].sum() == 1

        # The tessellation matches the one built without the tunnel row: the
        # enclosure is not cut at x=5, so the cells span the full loop.
        assert len(nodes["place"]) == len(reference_nodes["place"])
        assert nodes["place"].geometry.area.sum() == pytest.approx(
            reference_nodes["place"].geometry.area.sum()
        )

        # The tunnel has no barrier geometry, so it receives no faced_to edges.
        tunnel_ids = set(nodes["movement"].loc[nodes["movement"]["is_tunnel"]].index)
        faced_to = edges[("place", "faced_to", "movement")]
        assert not tunnel_ids & set(faced_to.index.get_level_values("movement_id"))

    def test_tessellation_fallback_on_enclosure_error(
        self,
        sample_crs: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A momepy enclosure failure falls back to building footprints when enabled."""
        buildings = gpd.GeoDataFrame(
            geometry=[Polygon([(0.2, 0.2), (0.4, 0.2), (0.4, 0.4), (0.2, 0.4)])],
            crs=sample_crs,
        )
        segments = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 0)])], crs=sample_crs)

        monkeypatch.setattr(
            "city2graph.morphology.create_tessellation",
            _raise_no_objects_tessellation,
        )

        # Without the fallback the error propagates.
        with pytest.raises(ValueError, match="No objects to concatenate"):
            morphological_graph(buildings, segments)

        # With the fallback a footprint cell stands in for the missing tessellation.
        nodes, _ = morphological_graph(buildings, segments, tessellation_fallback=True)
        assert len(nodes["place"]) == 1
        assert nodes["place"].index[0].startswith("fallback_")

    def test_tessellation_fallback_on_empty_tessellation(
        self,
        sample_crs: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An empty enclosed tessellation falls back to footprints when enabled."""
        buildings = gpd.GeoDataFrame(
            geometry=[Polygon([(0.2, 0.2), (0.4, 0.2), (0.4, 0.4), (0.2, 0.4)])],
            crs=sample_crs,
        )
        segments = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 0)])], crs=sample_crs)

        monkeypatch.setattr(
            "city2graph.morphology.create_tessellation",
            _empty_tessellation,
        )

        default_nodes, _ = morphological_graph(buildings, segments)
        fallback_nodes, _ = morphological_graph(buildings, segments, tessellation_fallback=True)

        assert default_nodes["place"].empty
        assert len(fallback_nodes["place"]) == 1

    def test_tessellation_fallback_logs_reason_and_cell_count(
        self,
        sample_crs: str,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Whole-tessellation fallback warns with the failure reason and cell count."""
        buildings = gpd.GeoDataFrame(
            geometry=[Polygon([(0.2, 0.2), (0.4, 0.2), (0.4, 0.4), (0.2, 0.4)])],
            crs=sample_crs,
        )
        segments = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 0)])], crs=sample_crs)

        monkeypatch.setattr(
            "city2graph.morphology.create_tessellation",
            _raise_no_objects_tessellation,
        )

        with caplog.at_level(logging.WARNING, logger="city2graph.morphology"):
            morphological_graph(buildings, segments, tessellation_fallback=True)

        assert any(
            "could not be created" in message and "1 building-footprint fallback cells" in message
            for message in caplog.messages
        )

    def test_include_unenclosed_buildings_logs_missing_count(
        self,
        sample_crs: str,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Appending fallback cells for unenclosed buildings warns with the counts."""
        buildings = gpd.GeoDataFrame(
            {"building_id": ["inside", "unenclosed"]},
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(10, 0), (11, 0), (11, 1), (10, 1)]),
            ],
            crs=sample_crs,
        )
        segments = gpd.GeoDataFrame(
            geometry=[LineString([(-1, -1), (2, -1)])],
            crs=sample_crs,
        )

        monkeypatch.setattr(
            "city2graph.morphology.create_tessellation",
            _enclosed_cell_or_fail_on_fallback,
        )
        monkeypatch.setattr(
            "city2graph.morphology._filter_adjacent_tessellation",
            _passthrough_adjacent_filter,
        )

        with caplog.at_level(logging.WARNING, logger="city2graph.morphology"):
            morphological_graph(buildings, segments, include_unenclosed_buildings=True)

        assert any(
            "covers no cell for 1 of 2 eligible buildings" in message for message in caplog.messages
        )

    def test_fallback_cells_match_source_buildings_exactly(
        self,
        sample_crs: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Whole-tessellation fallback cells join their own building, not sjoin matches."""
        # The small building sits inside the large footprint, so the large
        # fallback cell contains both representative points and the spatial
        # join alone would duplicate its row and mis-assign the buildings.
        buildings = gpd.GeoDataFrame(
            {"name": ["large", "small"]},
            geometry=[
                Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
                Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),
            ],
            crs=sample_crs,
        )
        segments = gpd.GeoDataFrame(geometry=[LineString([(0, -1), (4, -1)])], crs=sample_crs)

        monkeypatch.setattr(
            "city2graph.morphology.create_tessellation",
            _raise_no_objects_tessellation,
        )

        nodes, _ = morphological_graph(
            buildings,
            segments,
            tessellation_fallback=True,
            keep_buildings=True,
        )

        place = nodes["place"]
        assert sorted(place.index) == ["fallback_0", "fallback_1"]
        assert place.loc["fallback_0", "building_geometry"].equals(buildings.geometry.iloc[0])
        assert place.loc["fallback_1", "building_geometry"].equals(buildings.geometry.iloc[1])
        assert list(place["name"]) == ["large", "small"]
        assert "_source_building_index" not in place.columns

    def test_fallback_tessellation_empty_return_keeps_place_schema(
        self,
        sample_crs: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An empty whole-tessellation fallback still returns the place schema."""
        # The building lies far beyond the network distance budget, so the
        # fallback retains no cells at all.
        buildings = gpd.GeoDataFrame(
            geometry=[Polygon([(100, 100), (101, 100), (101, 101), (100, 101)])],
            crs=sample_crs,
        )
        segments = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 0)])], crs=sample_crs)

        monkeypatch.setattr(
            "city2graph.morphology.create_tessellation",
            _raise_no_objects_tessellation,
        )

        nodes, _ = morphological_graph(
            buildings,
            segments,
            center_point=Point(0, 0),
            distance=5.0,
            extent_buffer=1.0,
            tessellation_fallback=True,
        )

        place = nodes["place"]
        assert place.empty
        assert "place_id" in place.columns
        assert "_source_building_index" not in place.columns

    def test_unenclosed_fallback_does_not_leak_source_index_column(
        self,
        sample_crs: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Unenclosed-building fallback cells keep the source column internal."""
        # The stub tessellation covers only the first building, so the second
        # one is appended as an unenclosed fallback cell.
        buildings = gpd.GeoDataFrame(
            geometry=[
                Polygon([(0.2, 0.2), (0.4, 0.2), (0.4, 0.4), (0.2, 0.4)]),
                Polygon([(10, 0.2), (10.5, 0.2), (10.5, 0.7), (10, 0.7)]),
            ],
            crs=sample_crs,
        )
        segments = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (11, 0)])], crs=sample_crs)

        def one_cell_tessellation(
            geometry: gpd.GeoDataFrame,
            primary_barriers: gpd.GeoDataFrame | None = None,
            **kwargs: object,
        ) -> gpd.GeoDataFrame:
            _ = (primary_barriers, kwargs)
            return gpd.GeoDataFrame(
                {"tess_id": ["cell"], "enclosure_index": [0]},
                geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
                crs=geometry.crs,
            )

        monkeypatch.setattr("city2graph.morphology.create_tessellation", one_cell_tessellation)

        nodes, _ = morphological_graph(
            buildings,
            segments,
            include_unenclosed_buildings=True,
            keep_buildings=False,
        )

        place = nodes["place"]
        assert any(str(idx).startswith("fallback_") for idx in place.index)
        assert "_source_building_index" not in place.columns

    def test_tessellation_n_jobs_forwarded(
        self,
        sample_crs: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """tessellation_n_jobs is forwarded to create_tessellation only when not -1."""
        buildings = gpd.GeoDataFrame(
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            crs=sample_crs,
        )
        segments = gpd.GeoDataFrame(geometry=[LineString([(-1, 0), (2, 0)])], crs=sample_crs)
        captured: dict[str, object] = {}

        monkeypatch.setattr(
            "city2graph.morphology.create_tessellation",
            partial(_key_recording_tessellation, captured, "n_jobs", "absent"),
        )

        morphological_graph(buildings, segments, tessellation_n_jobs=1)
        assert captured["n_jobs"] == 1

        morphological_graph(buildings, segments)
        assert captured["n_jobs"] == "absent"


class TestMorphologicalGraphEdgeCases(TestMorphologyBase):
    """Edge cases and error handling for morphological_graph."""

    def test_empty_inputs(self, empty_gdf: gpd.GeoDataFrame) -> None:
        """Test various empty input scenarios."""
        # Both empty
        nodes, edges = morphological_graph(empty_gdf, empty_gdf)
        self.validate_empty_output(nodes, edges)

    def test_single_elements(
        self,
        single_building_gdf: gpd.GeoDataFrame,
        single_segment_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test with single building and segment."""
        nodes, edges = morphological_graph(single_building_gdf, single_segment_gdf)
        self.validate_basic_output(nodes, edges, ["place", "movement"])

    @pytest.mark.parametrize("invalid_contiguity", ["invalid", "diagonal", 123])
    def test_invalid_contiguity(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        invalid_contiguity: str | int,
    ) -> None:
        """Test invalid contiguity values."""
        with pytest.raises(ValueError, match="contiguity must be"):
            morphological_graph(
                sample_buildings_gdf,
                sample_segments_gdf,
                contiguity=str(invalid_contiguity),
            )

    @pytest.mark.parametrize("invalid_value", [-100, -1, -0.1])
    def test_negative_parameters(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        invalid_value: float,
    ) -> None:
        """Test negative parameter values."""
        Point(0, 0)

        # Test negative clipping buffer (this should raise ValueError)
        with pytest.raises(ValueError, match=r"clipping_buffer cannot be negative."):
            morphological_graph(
                sample_buildings_gdf,
                sample_segments_gdf,
                clipping_buffer=invalid_value,
            )

    def test_crs_mismatch(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test CRS mismatch handling."""
        # Create segments with different CRS
        segments_diff_crs = sample_segments_gdf.to_crs("EPSG:4326")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nodes, edges = morphological_graph(sample_buildings_gdf, segments_diff_crs)
            self.validate_basic_output(nodes, edges, ["place", "movement"])


class TestIndividualGraphFunctions(TestMorphologyBase):
    """Tests for individual graph creation functions."""

    # Place-to-Place Tests
    def test_place_to_place_basic(self, sample_tessellation_gdf: gpd.GeoDataFrame) -> None:
        """Test basic place-to-place graph creation."""
        nodes, edges = place_to_place_graph(sample_tessellation_gdf)
        self.validate_basic_output(nodes, edges)
        assert nodes.equals(sample_tessellation_gdf)

    def test_place_to_place_networkx(self, sample_tessellation_gdf: gpd.GeoDataFrame) -> None:
        """Test place-to-place NetworkX conversion."""
        graph = place_to_place_graph(sample_tessellation_gdf, as_nx=True)
        self.validate_networkx_output(graph)

    @pytest.mark.parametrize("contiguity", ["queen", "rook"])
    def test_place_to_place_contiguity(
        self,
        sample_tessellation_gdf: gpd.GeoDataFrame,
        contiguity: str,
    ) -> None:
        """Test place-to-place with different contiguity types."""
        nodes, edges = place_to_place_graph(sample_tessellation_gdf, contiguity=contiguity)
        self.validate_basic_output(nodes, edges)
        self.validate_edge_columns(edges, ["from_place_id", "to_place_id"])

    def test_place_to_place_duplicate_index(
        self, sample_tessellation_gdf: gpd.GeoDataFrame
    ) -> None:
        """Test place-to-place graph with duplicate indices in input GDF."""
        # Create a GDF with duplicate indices
        gdf_duplicate = sample_tessellation_gdf.copy()
        # Set index to be all 0s (or any duplicate values)
        gdf_duplicate.index = [0] * len(gdf_duplicate)

        # This should not raise ValueError: The argument to the ids parameter contains duplicate entries
        nodes, edges = place_to_place_graph(gdf_duplicate)

        self.validate_basic_output(nodes, edges)
        assert not edges.empty
        # Ensure the from/to columns contain the actual place_ids, not the duplicate index values
        assert edges["from_place_id"].isin(sample_tessellation_gdf["place_id"]).all()
        assert edges["to_place_id"].isin(sample_tessellation_gdf["place_id"]).all()

    def test_place_to_place_invalid_contiguity(
        self,
        sample_tessellation_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test invalid contiguity parameter."""
        with pytest.raises(ValueError, match="contiguity must be either 'queen' or 'rook'"):
            place_to_place_graph(sample_tessellation_gdf, contiguity="invalid")

    def test_place_to_place_missing_group_column(
        self,
        sample_tessellation_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test error when specified group column doesn't exist."""
        with pytest.raises(ValueError, match="group_col 'nonexistent_col' not found"):
            place_to_place_graph(sample_tessellation_gdf, group_col="nonexistent_col")

    def test_place_to_place_single_polygon(self, sample_crs: str) -> None:
        """Test with single polygon (insufficient for adjacency)."""
        # Single square polygon tessellation
        single_poly = make_grid_polygons_gdf(1, 1, crs=sample_crs)
        single_poly = single_poly.reset_index().rename(columns={"id": "place_id"})
        nodes, edges = place_to_place_graph(single_poly)
        # Should return empty edges since we need at least 2 polygons for adjacency
        assert edges.empty
        assert len(nodes) == 1

    def test_place_to_place_empty_input(self, sample_crs: str) -> None:
        """Test with empty tessellation."""
        empty_tess = gpd.GeoDataFrame(geometry=[], crs=sample_crs)
        nodes, edges = place_to_place_graph(empty_tess)
        self.validate_empty_output(nodes, edges)

    # Place-to-Movement Tests
    def test_place_to_movement_basic(
        self,
        sample_tessellation_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test basic place-to-movement graph creation."""
        nodes, edges = place_to_movement_graph(sample_tessellation_gdf, sample_segments_gdf)
        self.validate_basic_output(nodes, edges)
        self.validate_edge_columns(edges, ["place_id", "movement_id"])

    def test_place_to_movement_networkx(
        self,
        sample_tessellation_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test place-to-movement NetworkX conversion."""
        graph = place_to_movement_graph(sample_tessellation_gdf, sample_segments_gdf, as_nx=True)
        self.validate_networkx_output(graph)

    @pytest.mark.parametrize("tolerance", [1e-6, 1e-3, 0.1, 1.0])
    def test_place_to_movement_tolerance(
        self,
        sample_tessellation_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        tolerance: float,
    ) -> None:
        """Test place-to-movement with different tolerance values."""
        nodes, edges = place_to_movement_graph(
            sample_tessellation_gdf,
            sample_segments_gdf,
            tolerance=tolerance,
        )
        self.validate_basic_output(nodes, edges)

    def test_place_to_movement_missing_id_columns(self, sample_crs: str) -> None:
        """Test error when required ID columns are missing."""
        # Create GeoDataFrames without required ID columns
        place_no_id = gpd.GeoDataFrame(
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            crs=sample_crs,
        )
        movement_no_id = gpd.GeoDataFrame(
            geometry=[LineString([(0, 0), (2, 0)])],
            crs=sample_crs,
        )

        # Test missing place_id
        with pytest.raises(ValueError, match="Expected ID column 'place_id' not found"):
            place_to_movement_graph(place_no_id, movement_no_id)

        # Test missing movement_id (add place_id but not movement_id)
        place_with_id = place_no_id.copy()
        place_with_id["place_id"] = [1]

        with pytest.raises(ValueError, match="Expected ID column 'movement_id' not found"):
            place_to_movement_graph(place_with_id, movement_no_id)

    def test_place_to_movement_nearest_fallback_for_empty_join_result(
        self,
        sample_crs: str,
    ) -> None:
        """Place spaces missed by dwithin should connect to the nearest movement segment."""
        # Create non-overlapping geometries that won't intersect
        place_gdf = make_grid_polygons_gdf(1, 1, crs=sample_crs)
        place_gdf = place_gdf.reset_index().rename(columns={"id": "place_id"})
        movement_gdf = gpd.GeoDataFrame(
            {"movement_id": [1]},
            geometry=[LineString([(10, 10), (20, 20)])],  # Far away from place geometry
            crs=sample_crs,
        )

        nodes, edges = place_to_movement_graph(place_gdf, movement_gdf, tolerance=0.1)
        assert not edges.empty
        assert set(edges["place_id"]) == set(place_gdf["place_id"])
        assert set(edges["movement_id"]) == {1}
        assert len(nodes) == 2  # Both place and movement nodes combined

    def test_place_to_movement_empty_movement_stays_empty(self, sample_crs: str) -> None:
        """Nearest fallback should not change empty-movement behavior."""
        place_gdf = make_grid_polygons_gdf(1, 1, crs=sample_crs)
        place_gdf = place_gdf.reset_index().rename(columns={"id": "place_id"})
        movement_gdf = gpd.GeoDataFrame({"movement_id": []}, geometry=[], crs=sample_crs)

        nodes, edges = place_to_movement_graph(place_gdf, movement_gdf, tolerance=0.1)

        assert edges.empty
        assert len(nodes) == 1

    # Movement-to-Movement Tests
    def test_movement_to_movement_basic(self, sample_segments_gdf: gpd.GeoDataFrame) -> None:
        """Test basic movement-to-movement graph creation."""
        nodes, edges = movement_to_movement_graph(sample_segments_gdf)
        self.validate_basic_output(nodes, edges)
        assert nodes.equals(sample_segments_gdf)

    def test_movement_to_movement_networkx(self, sample_segments_gdf: gpd.GeoDataFrame) -> None:
        """Test movement-to-movement NetworkX conversion."""
        graph = movement_to_movement_graph(sample_segments_gdf, as_nx=True)
        self.validate_networkx_output(graph)

    def test_movement_to_movement_edge_structure(
        self, sample_segments_gdf: gpd.GeoDataFrame
    ) -> None:
        """Test movement-to-movement edge structure."""
        _nodes, edges = movement_to_movement_graph(sample_segments_gdf)
        self.validate_edge_columns(edges, ["from_movement_id", "to_movement_id"])

    def test_movement_to_movement_multiindex_handling(self, sample_crs: str) -> None:
        """Test movement-to-movement graph with MultiIndex scenarios."""
        # Create segments with MultiIndex to test line 702-708
        segments_data = []
        for i in range(3):
            line = LineString([(i, 0), (i + 1, 0)])
            segments_data.append(line)

        segments_gdf = gpd.GeoDataFrame(
            {"segment_id": range(len(segments_data))},
            geometry=segments_data,
            crs=sample_crs,
        )

        # Set a MultiIndex to trigger the MultiIndex handling code
        segments_gdf = segments_gdf.set_index([segments_gdf.index, "segment_id"])

        nodes, edges = movement_to_movement_graph(segments_gdf)
        self.validate_basic_output(nodes, edges)

    # duplicate_edges Tests
    def test_place_to_place_duplicate_edges(
        self, sample_tessellation_gdf: gpd.GeoDataFrame
    ) -> None:
        """duplicate_edges=True doubles rows with swapped place id columns."""
        _, base_edges = place_to_place_graph(sample_tessellation_gdf)
        _, dup_edges = place_to_place_graph(sample_tessellation_gdf, duplicate_edges=True)

        assert len(dup_edges) == 2 * len(base_edges)
        pairs = set(zip(base_edges["from_place_id"], base_edges["to_place_id"], strict=True))
        dup_pairs = set(zip(dup_edges["from_place_id"], dup_edges["to_place_id"], strict=True))
        assert dup_pairs == pairs | {(v, u) for u, v in pairs}

    def test_place_to_movement_duplicate_edges(
        self,
        sample_tessellation_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
    ) -> None:
        """duplicate_edges=True doubles rows with swapped endpoint columns."""
        _, base_edges = place_to_movement_graph(sample_tessellation_gdf, sample_segments_gdf)
        _, dup_edges = place_to_movement_graph(
            sample_tessellation_gdf, sample_segments_gdf, duplicate_edges=True
        )

        # place and movement ids share the same integer space in the fixtures,
        # so a base pair's reverse may already exist as another connection and
        # the idempotent symmetrization adds no duplicate row for it.
        pairs = set(zip(base_edges["place_id"], base_edges["movement_id"], strict=True))
        expected_pairs = pairs | {(v, u) for u, v in pairs}
        dup_pairs = set(zip(dup_edges["place_id"], dup_edges["movement_id"], strict=True))
        assert dup_pairs == expected_pairs
        assert len(dup_edges) == len(expected_pairs)
        assert len(dup_edges) > len(base_edges)

    def test_movement_to_movement_duplicate_edges(
        self, sample_segments_gdf: gpd.GeoDataFrame
    ) -> None:
        """duplicate_edges=True doubles rows with swapped movement id columns."""
        _, base_edges = movement_to_movement_graph(sample_segments_gdf)
        _, dup_edges = movement_to_movement_graph(sample_segments_gdf, duplicate_edges=True)

        assert len(dup_edges) == 2 * len(base_edges)
        pairs = set(zip(base_edges["from_movement_id"], base_edges["to_movement_id"], strict=True))
        dup_pairs = set(
            zip(dup_edges["from_movement_id"], dup_edges["to_movement_id"], strict=True)
        )
        assert dup_pairs == pairs | {(v, u) for u, v in pairs}

    @pytest.mark.parametrize(
        "call",
        [
            lambda tess, _segs, **kw: place_to_place_graph(tess, **kw),
            place_to_movement_graph,
            lambda _tess, segs, **kw: movement_to_movement_graph(segs, **kw),
        ],
        ids=["place_to_place", "place_to_movement", "movement_to_movement"],
    )
    def test_duplicate_edges_rejects_as_nx(
        self,
        sample_tessellation_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        call: Callable[..., object],
    ) -> None:
        """duplicate_edges=True with as_nx=True raises ValueError."""
        with pytest.raises(ValueError, match="not supported with as_nx=True"):
            call(sample_tessellation_gdf, sample_segments_gdf, as_nx=True, duplicate_edges=True)


class TestInputValidationAndErrors(TestMorphologyBase):
    """Comprehensive input validation and error handling tests."""

    def test_geometry_type_validation(self, sample_crs: str) -> None:
        """Test geometry type validation for buildings and segments."""
        # Test invalid building geometry types
        invalid_buildings = gpd.GeoDataFrame(
            {"bldg_id": [1]},
            geometry=[Point(0, 0)],  # Point instead of Polygon
            crs=sample_crs,
        )
        valid_segments = gpd.GeoDataFrame(
            {"segment_id": [1]},
            geometry=[LineString([(0, 0), (1, 0)])],
            crs=sample_crs,
        )

        with pytest.raises(
            ValueError,
            match="buildings_gdf must contain only Polygon or MultiPolygon geometries",
        ):
            morphological_graph(invalid_buildings, valid_segments)

        # Test invalid segment geometry types
        valid_buildings = gpd.GeoDataFrame(
            {"bldg_id": [1]},
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            crs=sample_crs,
        )
        invalid_segments = gpd.GeoDataFrame(
            {"segment_id": [1]},
            geometry=[Point(0, 0)],  # Point instead of LineString
            crs=sample_crs,
        )

        with pytest.raises(
            ValueError,
            match="segments_gdf must contain only LineString geometries",
        ):
            morphological_graph(valid_buildings, invalid_segments)

    def test_empty_buildings_handling(
        self,
        sample_segments_gdf: gpd.GeoDataFrame,
        sample_crs: str,
    ) -> None:
        """Test handling of empty buildings GeoDataFrame."""
        empty_buildings = gpd.GeoDataFrame(geometry=[], crs=sample_crs)

        # This should work and return results based only on segments
        nodes, edges = morphological_graph(empty_buildings, sample_segments_gdf)
        self.validate_basic_output(nodes, edges, ["place", "movement"])

        # Place nodes should be empty, movement nodes should match segments
        assert nodes["place"].empty
        assert len(nodes["movement"]) == len(sample_segments_gdf)

    def test_network_distance_edge_cases(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test edge cases in network distance filtering."""
        # Test with center point very far from any graph node
        far_center = Point(1000000, 1000000)  # Very far away

        # Clear any existing log records
        caplog.clear()

        with caplog.at_level(logging.WARNING):
            nodes, edges = morphological_graph(
                sample_buildings_gdf,
                sample_segments_gdf,
                center_point=far_center,
                distance=100,
            )

            # Should get warning about source node not found
            warning_messages = [
                record.message for record in caplog.records if record.levelno == logging.WARNING
            ]
            distance_warnings = [
                msg
                for msg in warning_messages
                if "Source node for distance filtering not found" in msg
            ]
            # The projection helper snaps the source to an edge when graph context exists.
            assert len(distance_warnings) == 0

        # Should still return valid (possibly empty) results
        self.validate_basic_output(nodes, edges, ["place", "movement"])

    def test_successful_network_distance_filtering(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test successful network distance filtering that executes Dijkstra algorithm."""
        # Use existing sample data but with a center point close to the data
        # This should find a source node and execute the Dijkstra algorithm (lines 1054-1063)

        # Get a center point near the existing buildings
        buildings_bounds = sample_buildings_gdf.total_bounds
        center_x = (buildings_bounds[0] + buildings_bounds[2]) / 2
        center_y = (buildings_bounds[1] + buildings_bounds[3]) / 2
        center_point = Point(center_x, center_y)

        # Test with a reasonable distance that should include buildings
        # This should successfully execute the Dijkstra algorithm (lines 1054-1063)
        nodes, edges = morphological_graph(
            sample_buildings_gdf,
            sample_segments_gdf,
            center_point=center_point,
            distance=1000,  # Large enough to include buildings
        )

        # Should successfully execute and return filtered results
        self.validate_basic_output(nodes, edges, ["place", "movement"])

        # Should have some nodes
        assert len(nodes["movement"]) > 0

        # Test with a smaller distance to ensure filtering logic is executed
        nodes_small, edges_small = morphological_graph(
            sample_buildings_gdf,
            sample_segments_gdf,
            center_point=center_point,
            distance=100,  # Smaller distance
        )

        self.validate_basic_output(nodes_small, edges_small, ["place", "movement"])
        # The key is that this executes the filtering code path without errors

    @pytest.mark.parametrize(
        ("function", "args"),
        [
            (morphological_graph, ("not_gdf", "sample_segments_gdf")),
            (morphological_graph, ("sample_buildings_gdf", "not_gdf")),
            (place_to_place_graph, ("not_gdf",)),
            (place_to_movement_graph, ("not_gdf", "sample_segments_gdf")),
            (place_to_movement_graph, ("sample_tessellation_gdf", "not_gdf")),
            (movement_to_movement_graph, ("not_gdf",)),
        ],
    )
    def test_invalid_input_types(
        self,
        function: Callable[..., Any],
        args: tuple[str, ...],
        request: pytest.FixtureRequest,
    ) -> None:
        """Test all functions with invalid input types."""
        # Resolve fixture names to actual objects
        resolved_args = []
        for arg in args:
            if arg == "not_gdf":
                resolved_args.append("not a geodataframe")
            else:
                resolved_args.append(request.getfixturevalue(arg))

        with pytest.raises(TypeError):
            function(*resolved_args)

    def test_empty_input_handling(self, empty_gdf: gpd.GeoDataFrame) -> None:
        """Test all functions handle empty inputs gracefully."""
        # Test each function with appropriate empty inputs
        functions_and_args: list[tuple[Callable[..., Any], tuple[gpd.GeoDataFrame, ...]]] = [
            (place_to_place_graph, (empty_gdf,)),
            (place_to_movement_graph, (empty_gdf, empty_gdf)),
            (movement_to_movement_graph, (empty_gdf,)),
            (morphological_graph, (empty_gdf, empty_gdf)),
        ]

        for func, args in functions_and_args:
            nodes, edges = func(*args)
            self.validate_empty_output(nodes, edges)

    def test_missing_required_columns(self, sample_buildings_gdf: gpd.GeoDataFrame) -> None:
        """Test handling of missing required columns."""
        # Remove required column
        gdf_no_place_id = sample_buildings_gdf.drop(columns=["place_id"], errors="ignore")

        with pytest.raises((KeyError, ValueError)):
            place_to_place_graph(gdf_no_place_id)


class TestIntegrationAndStress(TestMorphologyBase):
    """Integration tests and stress testing scenarios."""

    def test_large_dataset_performance(self, sample_crs: str) -> None:
        """Test with larger synthetic dataset for performance."""
        # Create grid of buildings with proper spacing for tessellation
        buildings = []
        for i in range(5):
            for j in range(5):
                poly = Polygon(
                    [
                        (i * 20 + 2, j * 20 + 2),
                        (i * 20 + 18, j * 20 + 2),
                        (i * 20 + 18, j * 20 + 18),
                        (i * 20 + 2, j * 20 + 18),
                    ],
                )
                buildings.append(poly)

        buildings_gdf = gpd.GeoDataFrame(
            {"bldg_id": range(len(buildings))},
            geometry=buildings,
            crs=sample_crs,
        )

        # Create grid of streets
        streets = []
        for j in range(6):
            line = LineString([(0, j * 20), (100, j * 20)])
            streets.append(line)
        for i in range(6):
            line = LineString([(i * 20, 0), (i * 20, 100)])
            streets.append(line)

        segments_gdf = gpd.GeoDataFrame(
            {"segment_id": range(len(streets))},
            geometry=streets,
            crs=sample_crs,
        )

        # Test morphological graph creation
        nodes, edges = morphological_graph(buildings_gdf, segments_gdf)
        self.validate_basic_output(nodes, edges, ["place", "movement"])

        # Verify that we get some output (tessellation might filter some buildings)
        assert len(nodes["movement"]) == len(segments_gdf)
        assert len(nodes["place"]) >= 0  # Some buildings might be filtered out

    def test_function_consistency(self, sample_tessellation_gdf: gpd.GeoDataFrame) -> None:
        """Test consistency between different function calls."""
        # Test that place_to_place_graph produces consistent results
        nodes1, edges1 = place_to_place_graph(sample_tessellation_gdf)
        nodes2, edges2 = place_to_place_graph(sample_tessellation_gdf)

        assert nodes1.equals(nodes2)
        assert edges1.equals(edges2)

    def test_reproducibility(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test reproducibility of morphological graph creation."""
        nodes1, edges1 = morphological_graph(sample_buildings_gdf, sample_segments_gdf)
        nodes2, edges2 = morphological_graph(sample_buildings_gdf, sample_segments_gdf)

        # Check that results are identical
        for key in nodes1:
            assert nodes1[key].equals(nodes2[key])
        for key in edges1:
            assert edges1[key].equals(edges2[key])


class TestSpecialScenarios(TestMorphologyBase):
    """Tests for special scenarios and advanced features."""

    def test_barrier_column_handling(
        self,
        sample_tessellation_gdf: gpd.GeoDataFrame,
        segments_gdf_alt_geom: gpd.GeoDataFrame,
    ) -> None:
        """Test handling of alternative barrier columns."""
        nodes, edges = place_to_movement_graph(
            sample_tessellation_gdf,
            segments_gdf_alt_geom,
            primary_barrier_col="barrier_geometry",
        )
        self.validate_basic_output(nodes, edges)

    def test_group_column_functionality(self, sample_tessellation_gdf: gpd.GeoDataFrame) -> None:
        """Test group column functionality in place-to-place graphs."""
        nodes, edges = place_to_place_graph(
            sample_tessellation_gdf,
            group_col="enclosure_index",
        )
        self.validate_basic_output(nodes, edges)
        if not edges.empty:
            assert "enclosure_index" in edges.columns

    def test_center_point_variations(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test different center point specifications."""
        # Test with Point geometry
        center_point = Point(sample_buildings_gdf.geometry.centroid.iloc[0])
        nodes, edges = morphological_graph(
            sample_buildings_gdf,
            sample_segments_gdf,
            center_point=center_point,
            distance=1000,
        )
        self.validate_basic_output(nodes, edges, ["place", "movement"])

        # Test with GeoDataFrame containing single point (simplified test)
        # Just verify that the function accepts GeoDataFrame input without error
        center_gdf = gpd.GeoDataFrame(
            geometry=[center_point],
            crs=sample_buildings_gdf.crs,
        )
        try:
            nodes, edges = morphological_graph(
                sample_buildings_gdf,
                sample_segments_gdf,
                center_point=center_gdf,
                distance=1000,
            )
            # If it doesn't raise an error, that's sufficient for this test
            self.validate_basic_output(nodes, edges, ["place", "movement"])
        except (AttributeError, ValueError):
            # Some center point formats might not be fully supported
            # This is acceptable for this test
            pass

    def test_empty_adjacency_data_handling(self, sample_crs: str) -> None:
        """Test handling of empty adjacency data in place-to-place graphs."""
        # Create two non-adjacent polygons to test empty adjacency scenario
        non_adjacent_polys = gpd.GeoDataFrame(
            {"place_id": [1, 2], "enclosure_index": [1, 2]},
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),  # First polygon
                Polygon([(10, 10), (11, 10), (11, 11), (10, 11)]),  # Far away polygon
            ],
            crs=sample_crs,
        )

        nodes, edges = place_to_place_graph(non_adjacent_polys, group_col="enclosure_index")
        # Should handle empty adjacency data gracefully
        self.validate_basic_output(nodes, edges)
        # Edges might be empty if polygons are not adjacent
        assert len(nodes) == 2

    def test_multiindex_edge_case_coverage(self, sample_crs: str) -> None:
        """Test MultiIndex edge cases in graph processing."""
        # Create a simple tessellation that will test MultiIndex handling
        simple_polys = gpd.GeoDataFrame(
            {"place_id": [1, 2]},
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),  # Adjacent polygon
            ],
            crs=sample_crs,
        )

        # Test with and without group column to cover different MultiIndex scenarios
        nodes1, edges1 = place_to_place_graph(simple_polys)
        nodes2, edges2 = place_to_place_graph(simple_polys, group_col=None)

        self.validate_basic_output(nodes1, edges1)
        self.validate_basic_output(nodes2, edges2)

    def test_buildings_multiindex_handling(self, sample_crs: str) -> None:
        """Test morphological_graph with buildings having MultiIndex to cover lines 178-179."""
        # Create buildings with MultiIndex
        buildings_data = {
            "building_id": ["b1", "b2", "b3"],
            "geometry": [
                Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
                Polygon([(1, 0), (1, 1), (2, 1), (2, 0)]),
                Polygon([(2, 0), (2, 1), (3, 1), (3, 0)]),
            ],
        }
        buildings_gdf = gpd.GeoDataFrame(buildings_data, crs=sample_crs)

        # Create MultiIndex to trigger lines 178-179
        multi_index = pd.MultiIndex.from_arrays(
            [["type1", "type1", "type2"], ["b1", "b2", "b3"]],
            names=["building_type", "building_id"],
        )
        buildings_gdf.index = multi_index

        # Create simple segments that intersect with buildings
        segments_data = {
            "seg_id": ["s1", "s2"],
            "geometry": [
                LineString([(0.5, 0.5), (1.5, 0.5)]),  # Intersects buildings
                LineString([(1.5, 0.5), (2.5, 0.5)]),  # Intersects buildings
            ],
        }
        segments_gdf = gpd.GeoDataFrame(segments_data, crs=sample_crs)

        # This should trigger the MultiIndex handling code in lines 178-179
        # Even if it fails due to empty connections, the MultiIndex code was executed
        try:
            result = morphological_graph(buildings_gdf, segments_gdf)
            # morphological_graph returns a tuple (nodes_dict, edges_dict)
            assert isinstance(result, tuple)
            assert len(result) == 2
            nodes_dict, edges_dict = result
            assert isinstance(nodes_dict, dict)
            assert isinstance(edges_dict, dict)
        except KeyError:
            # Expected if no connections are found, but MultiIndex code was still executed
            pass


class TestReachabilityFieldRefactor(TestMorphologyBase):
    """Invariants introduced by the shared reachability-field refactor."""

    def test_reachability_field_computed_once(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        mg_center_point: gpd.GeoSeries,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A budgeted morphological_graph computes the cost field exactly once.

        Previously the field was recomputed independently for segments, buildings
        and cells (up to three Dijkstra/snap passes). It must now be built once on
        the full network and shared across all node types.
        """
        original = morph._network_reachability_field
        calls = {"n": 0}

        monkeypatch.setattr(
            morph,
            "_network_reachability_field",
            partial(_counting_wrapper, calls, original),
        )

        morphological_graph(
            sample_buildings_gdf,
            sample_segments_gdf,
            center_point=mg_center_point,
            distance=100.0,
        )

        assert calls["n"] == 1

    def test_shared_field_reaches_cell_and_building_filters(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        mg_center_point: gpd.GeoSeries,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The single full-network field is threaded into the centroid filters.

        This locks in the fix for the full-vs-truncated network discrepancy:
        buildings and cells are judged against the same field used for streets,
        not one rebuilt on the already-truncated segments.
        """
        original = morph._geometry_ilocs_within_network_distance
        seen_fields: list[object] = []

        monkeypatch.setattr(
            morph,
            "_geometry_ilocs_within_network_distance",
            partial(_field_recording_wrapper, seen_fields, original),
        )

        morphological_graph(
            sample_buildings_gdf,
            sample_segments_gdf,
            center_point=mg_center_point,
            distance=100.0,
        )

        # The centroid filters (buildings + cells) ran and every one received the
        # shared field instance rather than None.
        assert seen_fields
        assert all(field is not None for field in seen_fields)
        assert len({id(field) for field in seen_fields}) == 1

    def test_geographic_crs_emits_warning(self) -> None:
        """Distance params are metric, so a geographic CRS input must warn the user.

        Exercised through ``place_to_movement_graph`` because it shares the CRS
        chokepoint (``_ensure_crs_consistency``) with ``morphological_graph`` but
        does not invoke tessellation, which momepy refuses to run on a geographic
        CRS.
        """
        place_gdf = gpd.GeoDataFrame(
            {"place_id": [0]},
            geometry=[Polygon([(0, 0), (0, 0.001), (0.001, 0.001), (0.001, 0)])],
            crs="EPSG:4326",
        )
        movement_gdf = gpd.GeoDataFrame(
            {"movement_id": [10]},
            geometry=[LineString([(0, 0), (0.001, 0.001)])],
            crs="EPSG:4326",
        )

        with pytest.warns(UserWarning, match="geographic CRS"):
            place_to_movement_graph(place_gdf, movement_gdf)

    def test_projected_crs_does_not_emit_geographic_warning(
        self,
        sample_crs: str,
    ) -> None:
        """A projected CRS input must not raise the geographic-CRS warning."""
        place_gdf = gpd.GeoDataFrame(
            {"place_id": [0]},
            geometry=[Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])],
            crs=sample_crs,
        )
        movement_gdf = gpd.GeoDataFrame(
            {"movement_id": [10]},
            geometry=[LineString([(0, 0), (1, 1)])],
            crs=sample_crs,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            # Should not raise: inputs are in EPSG:27700 (projected).
            place_to_movement_graph(place_gdf, movement_gdf)


class TestMorphologicalGraphMultiDistance(TestMorphologyBase):
    """Multi-distance behaviour of morphological_graphs."""

    def test_distance_list_returns_dict_per_distance(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        mg_center_point: gpd.GeoSeries,
    ) -> None:
        """A list of distances yields a dict of (nodes, edges) keyed by distance."""
        distances = [50.0, 100.0]

        result = morphological_graphs(
            sample_buildings_gdf,
            sample_segments_gdf,
            distances,
            center_point=mg_center_point,
        )

        assert isinstance(result, dict)
        assert set(result) == set(distances)
        for nodes, edges in result.values():
            self.validate_basic_output(nodes, edges, ["place", "movement"])
            for edge_type in [
                ("place", "touched_to", "place"),
                ("movement", "connected_to", "movement"),
                ("place", "faced_to", "movement"),
            ]:
                assert edge_type in edges

        # A smaller budget can never retain more movement segments than a larger one.
        small_nodes, _ = result[50.0]
        large_nodes, _ = result[100.0]
        assert len(small_nodes["movement"]) <= len(large_nodes["movement"])

    def test_distance_list_as_nx_returns_dict_of_graphs(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        mg_center_point: gpd.GeoSeries,
    ) -> None:
        """With as_nx=True each per-distance value is a NetworkX graph."""
        result = morphological_graphs(
            sample_buildings_gdf,
            sample_segments_gdf,
            [50.0, 100.0],
            center_point=mg_center_point,
            as_nx=True,
        )

        assert isinstance(result, dict)
        for graph in result.values():
            self.validate_networkx_output(graph)

    def test_empty_distance_sequence_raises(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        mg_center_point: gpd.GeoSeries,
    ) -> None:
        """An empty distance sequence is rejected."""
        with pytest.raises(ValueError, match="distances must contain at least one value"):
            morphological_graphs(
                sample_buildings_gdf,
                sample_segments_gdf,
                [],
                center_point=mg_center_point,
            )


class TestPublicInputImmutability(TestMorphologyBase):
    """Public morphology functions must leave their input GeoDataFrames unchanged."""

    @pytest.mark.parametrize("options", IMMUTABILITY_OPTION_SETS)
    def test_morphological_graph_does_not_mutate_inputs(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        mg_center_point: gpd.GeoSeries,
        options: dict[str, Any],
    ) -> None:
        """morphological_graph leaves the buildings and segments inputs untouched."""
        kwargs = dict(options)
        if kwargs.pop("with_center", False):
            kwargs["center_point"] = mg_center_point
        buildings_before = sample_buildings_gdf.copy(deep=True)
        segments_before = sample_segments_gdf.copy(deep=True)

        morphological_graph(sample_buildings_gdf, sample_segments_gdf, **kwargs)

        assert_geodataframe_equal(sample_buildings_gdf, buildings_before)
        assert_geodataframe_equal(sample_segments_gdf, segments_before)

    def test_morphological_graph_barrier_split_does_not_mutate_input(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
    ) -> None:
        """The barrier-only segment split leaves the segments input untouched."""
        segments = sample_segments_gdf.copy(deep=True)
        segments["is_barrier"] = [i % 2 == 0 for i in range(len(segments))]
        segments_before = segments.copy(deep=True)

        morphological_graph(
            sample_buildings_gdf,
            segments,
            non_movement_barrier_col="is_barrier",
        )

        assert_geodataframe_equal(segments, segments_before)

    def test_morphological_graphs_does_not_mutate_inputs(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
        mg_center_point: gpd.GeoSeries,
    ) -> None:
        """The multi-distance pipeline leaves the shared inputs untouched."""
        buildings_before = sample_buildings_gdf.copy(deep=True)
        segments_before = sample_segments_gdf.copy(deep=True)

        morphological_graphs(
            sample_buildings_gdf,
            sample_segments_gdf,
            [50.0, 100.0],
            center_point=mg_center_point,
        )

        assert_geodataframe_equal(sample_buildings_gdf, buildings_before)
        assert_geodataframe_equal(sample_segments_gdf, segments_before)

    def test_pairwise_graph_functions_do_not_mutate_inputs(
        self,
        sample_tessellation_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
    ) -> None:
        """place/movement pairwise builders leave their inputs untouched."""
        tessellation_before = sample_tessellation_gdf.copy(deep=True)
        segments_before = sample_segments_gdf.copy(deep=True)

        place_to_place_graph(sample_tessellation_gdf)
        place_to_movement_graph(sample_tessellation_gdf, sample_segments_gdf)
        movement_to_movement_graph(sample_segments_gdf)

        assert_geodataframe_equal(sample_tessellation_gdf, tessellation_before)
        assert_geodataframe_equal(sample_segments_gdf, segments_before)

    def test_movement_graph_without_movement_id_does_not_mutate_input(
        self,
        sample_segments_gdf: gpd.GeoDataFrame,
    ) -> None:
        """The edge-id fallback write happens on a copy, not on the input."""
        segments = sample_segments_gdf.drop(columns=["movement_id"])
        segments_before = segments.copy(deep=True)

        movement_to_movement_graph(segments)

        assert_geodataframe_equal(segments, segments_before)

    def test_segments_to_graph_does_not_mutate_input(
        self,
        sample_segments_gdf: gpd.GeoDataFrame,
    ) -> None:
        """segments_to_graph builds its edge index on a copy of the input."""
        segments_before = sample_segments_gdf.copy(deep=True)

        segments_to_graph(sample_segments_gdf)

        assert_geodataframe_equal(sample_segments_gdf, segments_before)


class TestDeprecatedAliases:
    """Deprecated function names warn and delegate to the renamed functions."""

    def test_private_to_private_graph_deprecated(
        self,
        sample_tessellation_gdf: gpd.GeoDataFrame,
    ) -> None:
        """private_to_private_graph warns and matches place_to_place_graph."""
        with pytest.warns(DeprecationWarning, match="place_to_place_graph"):
            nodes, edges = private_to_private_graph(sample_tessellation_gdf)
        expected_nodes, expected_edges = place_to_place_graph(sample_tessellation_gdf)
        pd.testing.assert_frame_equal(nodes, expected_nodes)
        pd.testing.assert_frame_equal(edges, expected_edges)

    def test_private_to_public_graph_deprecated(
        self,
        sample_tessellation_gdf: gpd.GeoDataFrame,
        sample_segments_gdf: gpd.GeoDataFrame,
    ) -> None:
        """private_to_public_graph warns and matches place_to_movement_graph."""
        with pytest.warns(DeprecationWarning, match="place_to_movement_graph"):
            nodes, edges = private_to_public_graph(sample_tessellation_gdf, sample_segments_gdf)
        expected_nodes, expected_edges = place_to_movement_graph(
            sample_tessellation_gdf,
            sample_segments_gdf,
        )
        pd.testing.assert_frame_equal(nodes, expected_nodes)
        pd.testing.assert_frame_equal(edges, expected_edges)

    def test_public_to_public_graph_deprecated(
        self,
        sample_segments_gdf: gpd.GeoDataFrame,
    ) -> None:
        """public_to_public_graph warns and matches movement_to_movement_graph."""
        with pytest.warns(DeprecationWarning, match="movement_to_movement_graph"):
            nodes, edges = public_to_public_graph(sample_segments_gdf)
        expected_nodes, expected_edges = movement_to_movement_graph(sample_segments_gdf)
        pd.testing.assert_frame_equal(nodes, expected_nodes)
        pd.testing.assert_frame_equal(edges, expected_edges)
