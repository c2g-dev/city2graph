"""Tests for deprecated public API parameters.

Phase 1 of the ``as_nx`` deprecation replaces the ``as_nx: bool = False``
default with a ``None`` sentinel so that only explicit use triggers a
``DeprecationWarning``. The behaviour under test is therefore threefold:

1. Passing ``as_nx`` explicitly warns, whether the value is True or False.
2. Omitting ``as_nx`` warns not at all, including when the builder delegates
   to another deprecated builder internally.
3. The warning is attributed to the caller's frame rather than to
   city2graph's own source, so ``stacklevel`` stays correct.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING
from typing import Any

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from city2graph.mobility import od_matrix_to_graph
from city2graph.morphology import morphological_graph
from city2graph.morphology import movement_to_movement_graph
from city2graph.morphology import place_to_movement_graph
from city2graph.morphology import place_to_place_graph
from city2graph.morphology import segments_to_graph
from city2graph.proximity import bridge_nodes
from city2graph.proximity import contiguity_graph
from city2graph.proximity import delaunay_graph
from city2graph.proximity import euclidean_minimum_spanning_tree
from city2graph.proximity import fixed_radius_graph
from city2graph.proximity import gabriel_graph
from city2graph.proximity import group_nodes
from city2graph.proximity import knn_graph
from city2graph.proximity import relative_neighborhood_graph
from city2graph.proximity import waxman_graph
from city2graph.transportation import travel_summary_graph
from tests.helpers import make_grid_polygons_gdf
from tests.helpers import make_points_simple
from tests.helpers import make_poly_points_pair
from tests.helpers import make_segments_gdf
from tests.test_transportation import dict_to_con

if TYPE_CHECKING:
    from collections.abc import Callable

DEPRECATION_MATCH = "`as_nx` is deprecated"

# --- Input builders -------------------------------------------------------
# Each builder returns the positional/keyword arguments for one deprecated
# public function, excluding `as_nx` itself, which the tests supply.


def _points_args() -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Build arguments for the point-based proximity generators."""
    return (make_points_simple([(0, 0), (1, 0), (0, 1), (1, 1)]),), {}


def _polygons_args() -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Build arguments for ``contiguity_graph``."""
    return (make_grid_polygons_gdf(2, 2),), {}


def _place_gdf() -> gpd.GeoDataFrame:
    """Build a minimal tessellation carrying the required ``place_id`` column."""
    places = make_grid_polygons_gdf(2, 2).reset_index(drop=True)
    places["place_id"] = range(len(places))
    return places


def _movement_gdf() -> gpd.GeoDataFrame:
    """Build a minimal two-segment movement network with a ``movement_id`` column."""
    movements = make_segments_gdf(
        ["s1", "s2"],
        [[(0.0, 0.0), (2.0, 0.0)], [(2.0, 0.0), (2.0, 2.0)]],
        crs="EPSG:27700",
    )
    movements["movement_id"] = movements["id"]
    return movements


def _od_args() -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Build arguments for ``od_matrix_to_graph`` in edge-list form."""
    zones = gpd.GeoDataFrame(
        {"zone_id": ["A", "B", "C"]},
        geometry=[Point(0, 0), Point(1, 0), Point(0, 1)],
        crs="EPSG:27700",
    )
    edgelist = pd.DataFrame(
        {"source": ["A", "B"], "target": ["B", "C"], "flow": [3, 4]},
    )
    return (edgelist, zones), {
        "zone_id_col": "zone_id",
        "matrix_type": "edgelist",
        "weight_cols": ["flow"],
    }


def _bridge_args() -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Build arguments for ``bridge_nodes`` with two point layers."""
    nodes_dict = {
        "a": make_points_simple([(0, 0), (1, 0)]),
        "b": make_points_simple([(0, 1), (1, 1)]),
    }
    return (nodes_dict,), {"k": 1}


def _group_args() -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Build arguments for ``group_nodes``."""
    return make_poly_points_pair(), {}


# (id, function, argument builder) for every builder that gained the sentinel.
# ``travel_summary_graph`` needs a DuckDB connection and is covered separately.
DEPRECATED_BUILDERS: list[tuple[str, Callable[..., Any], Callable[[], Any]]] = [
    ("knn_graph", knn_graph, lambda: (_points_args()[0], {"k": 1})),
    ("delaunay_graph", delaunay_graph, _points_args),
    ("gabriel_graph", gabriel_graph, _points_args),
    ("relative_neighborhood_graph", relative_neighborhood_graph, _points_args),
    (
        "euclidean_minimum_spanning_tree",
        euclidean_minimum_spanning_tree,
        _points_args,
    ),
    ("fixed_radius_graph", fixed_radius_graph, lambda: (_points_args()[0], {"radius": 5.0})),
    (
        "waxman_graph",
        waxman_graph,
        lambda: (_points_args()[0], {"beta": 0.5, "r0": 5.0, "seed": 0}),
    ),
    ("bridge_nodes", bridge_nodes, _bridge_args),
    ("group_nodes", group_nodes, _group_args),
    ("contiguity_graph", contiguity_graph, _polygons_args),
    (
        "morphological_graph",
        morphological_graph,
        lambda: ((_place_gdf(), _movement_gdf()), {}),
    ),
    ("place_to_place_graph", place_to_place_graph, lambda: ((_place_gdf(),), {})),
    (
        "place_to_movement_graph",
        place_to_movement_graph,
        lambda: ((_place_gdf(), _movement_gdf()), {}),
    ),
    (
        "movement_to_movement_graph",
        movement_to_movement_graph,
        lambda: ((_movement_gdf(),), {}),
    ),
    ("segments_to_graph", segments_to_graph, lambda: ((_movement_gdf(),), {})),
    ("od_matrix_to_graph", od_matrix_to_graph, _od_args),
]

BUILDER_PARAMS = [
    pytest.param(func, builder, id=name) for name, func, builder in DEPRECATED_BUILDERS
]


class TestAsNxDeprecation:
    """Verify the ``as_nx`` deprecation warning fires exactly when intended."""

    @pytest.mark.parametrize(("func", "build_args"), BUILDER_PARAMS)
    @pytest.mark.parametrize("as_nx", [True, False])
    def test_explicit_as_nx_warns(
        self,
        func: Callable[..., Any],
        build_args: Callable[[], Any],
        as_nx: bool,
    ) -> None:
        """Passing ``as_nx`` explicitly warns, for both True and False."""
        args, kwargs = build_args()
        with pytest.warns(DeprecationWarning, match=DEPRECATION_MATCH):
            func(*args, **kwargs, as_nx=as_nx)

    @pytest.mark.parametrize(("func", "build_args"), BUILDER_PARAMS)
    def test_default_as_nx_is_silent(
        self,
        func: Callable[..., Any],
        build_args: Callable[[], Any],
    ) -> None:
        """Omitting ``as_nx`` emits no deprecation warning.

        This covers builders that delegate to another deprecated builder
        internally: a user who never touches ``as_nx`` must not be warned
        about a call they did not make.
        """
        args, kwargs = build_args()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            func(*args, **kwargs)
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert deprecations == [], [str(w.message) for w in deprecations]

    @pytest.mark.parametrize(("func", "build_args"), BUILDER_PARAMS)
    def test_warning_points_at_caller(
        self,
        func: Callable[..., Any],
        build_args: Callable[[], Any],
    ) -> None:
        """``stacklevel`` attributes the warning to the caller, not to city2graph."""
        args, kwargs = build_args()
        with pytest.warns(DeprecationWarning, match=DEPRECATION_MATCH) as record:
            func(*args, **kwargs, as_nx=False)
        assert record[0].filename == __file__

    def test_travel_summary_graph_deprecation(
        self,
        sample_gtfs_dict: dict[str, pd.DataFrame],
    ) -> None:
        """``travel_summary_graph`` warns on explicit use and stays silent by default."""
        con = dict_to_con(sample_gtfs_dict)

        with pytest.warns(DeprecationWarning, match=DEPRECATION_MATCH) as record:
            travel_summary_graph(con, as_nx=False)
        assert record[0].filename == __file__

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            travel_summary_graph(con)
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert deprecations == [], [str(w.message) for w in deprecations]
