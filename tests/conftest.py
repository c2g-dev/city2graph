"""Shared pytest fixtures for city2graph tests."""

import typing
import zipfile
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import duckdb
import geopandas as gpd
import networkx as nx
import pandas as pd
import pytest
from shapely.geometry import LineString
from shapely.geometry import Point

from city2graph import TransportData
from city2graph.transportation import load_gtfs
from tests.helpers import GtfsConnectionFactory
from tests.helpers import make_nx_graph
from tests.helpers import temporary_gtfs_zip

TESTS_DIR = Path(__file__).parent
TEST_DATA_DIR = TESTS_DIR / "data"
DEFAULT_SAMPLE_CRS = "EPSG:27700"
WGS84_CRS = "EPSG:4326"


@pytest.fixture
def osm_response() -> dict[str, Any]:
    """Provide a small synthetic Overpass response for public loader tests."""
    return {
        "osm3s": {"timestamp_osm_base": "2026-11-02T12:00:00Z"},
        "elements": [
            {
                "type": "relation",
                "id": 42,
                "tags": {
                    "type": "route",
                    "route": "bus",
                    "ref": "R",
                    "name": "Test route",
                    "public_transport:version": "2",
                },
                "members": [
                    {"type": "node", "ref": 1, "role": "stop"},
                    {"type": "node", "ref": 11, "role": "platform"},
                    {"type": "node", "ref": 2, "role": "stop_exit_only"},
                    {"type": "way", "ref": 3, "role": ""},
                ],
            },
            {"type": "node", "id": 1, "lat": 51, "lon": 0, "tags": {"name": "A"}},
            {"type": "node", "id": 11, "lat": 51.0001, "lon": 0, "tags": {"name": "A"}},
            {"type": "node", "id": 2, "lat": 51, "lon": 0.01, "tags": {"name": "B"}},
            {"type": "way", "id": 3, "geometry": [{"lat": 51, "lon": 0}, {"lat": 51, "lon": 0.01}]},
        ],
    }


@pytest.fixture
def gtfs_connection(tmp_path: Path) -> Iterator[GtfsConnectionFactory]:
    """Load independent GTFS feeds and close every connection after the test."""
    with ExitStack() as connections:
        counter = 0

        def load(tables: dict[str, pd.DataFrame]) -> duckdb.DuckDBPyConnection:
            nonlocal counter
            path = tmp_path / f"feed-{counter}.zip"
            counter += 1
            with zipfile.ZipFile(path, "w") as archive:
                for name, frame in tables.items():
                    data = frame.drop(columns="geometry", errors="ignore").copy()
                    for column in data.select_dtypes(include="bool"):
                        data[column] = data[column].astype(int)
                    archive.writestr(f"{name}.txt", data.to_csv(index=False))
            return connections.enter_context(load_gtfs(path))

        yield load


EdgeTableData = dict[str, object]


def _edge_multi_index(
    data: EdgeTableData,
    source_column: str,
    target_column: str,
    *,
    names: tuple[str, str] | None = None,
) -> pd.MultiIndex:
    """Build a two-level MultiIndex from edge endpoint columns."""
    index_names = names or (source_column, target_column)
    return pd.MultiIndex.from_arrays(
        [
            pd.Index(typing.cast("list[Any]", data[source_column])),
            pd.Index(typing.cast("list[Any]", data[target_column])),
        ],
        names=index_names,
    )


def _edge_gdf(
    data: EdgeTableData,
    *,
    source_column: str,
    target_column: str,
    crs: str,
    index_names: tuple[str, str] | None = None,
) -> gpd.GeoDataFrame:
    """Create an edge GeoDataFrame with a standard two-level MultiIndex."""
    return gpd.GeoDataFrame(
        data,
        index=_edge_multi_index(
            data,
            source_column,
            target_column,
            names=index_names,
        ),
        crs=crs,
    )


def _load_test_geojson(path: Path, *, target_crs: str) -> gpd.GeoDataFrame:
    """Load a GeoJSON test fixture and normalize it to the requested CRS."""
    try:
        return gpd.read_file(path).to_crs(target_crs)
    except OSError as exc:
        if not path.exists():
            pytest.fail(
                f"GeoJSON file not found at {path}. Make sure it exists under '{TEST_DATA_DIR.name}'.",
            )
        pytest.fail(f"Failed to load {path.name} from {path}: {exc}")


def _base_gtfs_zip_files() -> dict[str, str]:
    """Return a minimal valid GTFS feed as raw CSV file contents."""
    return {
        "agency.txt": (
            "agency_id,agency_name,agency_url,agency_timezone\n"
            "1,Test Agency,http://test.com,America/New_York\n"
        ),
        "stops.txt": (
            "stop_id,stop_name,stop_lat,stop_lon\n"
            "stop1,Stop 1,40.7128,-74.0060\n"
            "stop2,Stop 2,40.7589,-73.9851\n"
            "stop3,Stop 3,40.7505,-73.9934\n"
        ),
        "routes.txt": (
            "route_id,agency_id,route_short_name,route_long_name,route_type\n"
            "route1,1,1,Test Route 1,3\n"
        ),
        "trips.txt": (
            "route_id,service_id,trip_id\nroute1,service1,trip1\nroute1,service1,trip2\n"
        ),
        "stop_times.txt": (
            "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
            "trip1,08:00:00,08:00:00,stop1,1\n"
            "trip1,08:05:00,08:05:00,stop2,2\n"
            "trip1,08:10:00,08:10:00,stop3,3\n"
            "trip2,09:00:00,09:00:00,stop1,1\n"
            "trip2,09:05:00,09:05:00,stop2,2\n"
            "trip2,09:10:00,09:10:00,stop3,3\n"
        ),
        "calendar.txt": (
            "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\n"
            "service1,1,1,1,1,1,0,0,20240101,20241231\n"
        ),
    }


def _base_gtfs_dict() -> dict[str, pd.DataFrame]:
    """Return a canonical in-memory GTFS feed used across transportation tests."""
    return {
        "stops": gpd.GeoDataFrame(
            {
                "stop_id": ["stop1", "stop2", "stop3"],
                "stop_name": ["Stop 1", "Stop 2", "Stop 3"],
                "stop_lat": [40.7128, 40.7589, 40.7505],
                "stop_lon": [-74.0060, -73.9851, -73.9934],
                "geometry": [
                    Point(-74.0060, 40.7128),
                    Point(-73.9851, 40.7589),
                    Point(-73.9934, 40.7505),
                ],
            },
            crs=WGS84_CRS,
        ),
        "routes": pd.DataFrame(
            {
                "route_id": ["route1"],
                "agency_id": ["1"],
                "route_short_name": ["1"],
                "route_long_name": ["Test Route 1"],
                "route_type": [3],
            },
        ),
        "trips": pd.DataFrame(
            {
                "route_id": ["route1", "route1"],
                "service_id": ["service1", "service1"],
                "trip_id": ["trip1", "trip2"],
            },
        ),
        "stop_times": pd.DataFrame(
            {
                "trip_id": ["trip1", "trip1", "trip1", "trip2", "trip2", "trip2"],
                "arrival_time": [
                    "08:00:00",
                    "08:05:00",
                    "08:10:00",
                    "09:00:00",
                    "09:05:00",
                    "09:10:00",
                ],
                "departure_time": [
                    "08:00:00",
                    "08:05:00",
                    "08:10:00",
                    "09:00:00",
                    "09:05:00",
                    "09:10:00",
                ],
                "stop_id": ["stop1", "stop2", "stop3", "stop1", "stop2", "stop3"],
                "stop_sequence": [1, 2, 3, 1, 2, 3],
            },
        ),
        "calendar": pd.DataFrame(
            {
                "service_id": ["service1"],
                "monday": [True],
                "tuesday": [True],
                "wednesday": [True],
                "thursday": [True],
                "friday": [True],
                "saturday": [False],
                "sunday": [False],
                "start_date": ["20240101"],
                "end_date": ["20241231"],
            },
        ),
    }


@pytest.fixture(scope="session")
def sample_crs() -> str:
    """Coordinate Reference System fixture."""
    return DEFAULT_SAMPLE_CRS


@pytest.fixture
def sample_nodes_gdf(sample_crs: str) -> gpd.GeoDataFrame:
    """Fixture for a homogeneous nodes GeoDataFrame."""
    data = {
        "node_id": [1, 2, 3, 4],
        "feature1": [10.0, 20.0, 30.0, 40.0],
        "label1": [0, 1, 0, 1],
        "geometry": [Point(0, 0), Point(1, 1), Point(0, 1), Point(1, 0)],
    }
    return gpd.GeoDataFrame(data, crs=sample_crs).set_index("node_id")


@pytest.fixture
def sample_edges_gdf(sample_crs: str) -> gpd.GeoDataFrame:
    """Fixture for a homogeneous edges GeoDataFrame."""
    data = {
        "edge_id": ["e1", "e2", "e3", "e4"],
        "source_id": [1, 1, 2, 3],
        "target_id": [2, 3, 4, 4],
        "edge_feature1": [0.5, 0.8, 1.2, 2.5],
        "geometry": [
            LineString([(0, 0), (1, 1)]),
            LineString([(0, 0), (0, 1)]),
            LineString([(1, 1), (1, 0)]),
            LineString([(0, 1), (1, 0)]),
        ],
    }
    return _edge_gdf(
        data,
        source_column="source_id",
        target_column="target_id",
        crs=sample_crs,
    )


@pytest.fixture
def sample_hetero_nodes_dict(sample_crs: str) -> dict[str, gpd.GeoDataFrame]:
    """Fixture for a dictionary of heterogeneous nodes GeoDataFrames."""
    buildings_data = {
        "building_id": ["b1", "b2", "b3"],
        "b_feat1": [100.0, 150.0, 120.0],
        "b_label": [1, 0, 1],
        "geometry": [Point(10, 10), Point(11, 11), Point(10, 11)],
    }
    buildings_gdf = gpd.GeoDataFrame(buildings_data, crs=sample_crs).set_index(
        "building_id",
    )

    roads_data = {
        "road_id": ["r1", "r2"],
        "r_feat1": [5.5, 6.0],
        "r_label": [0, 0],
        "length": [100.0, 120.0],  # Add length column for tests
        "geometry": [Point(10, 12), Point(12, 12)],
    }
    roads_gdf = gpd.GeoDataFrame(roads_data, crs=sample_crs).set_index("road_id")

    return {"building": buildings_gdf, "road": roads_gdf}


@pytest.fixture
def sample_hetero_edges_dict(
    sample_crs: str,
) -> dict[tuple[str, str, str], gpd.GeoDataFrame]:
    """Fixture for a dictionary of heterogeneous edges GeoDataFrames."""
    connections_data = {
        "building_id": ["b1", "b2", "b3"],
        "road_id": ["r1", "r1", "r2"],
        "conn_feat1": [1.0, 2.0, 3.0],
        "geometry": [
            LineString([(10, 10), (10, 12)]),
            LineString([(11, 11), (10, 12)]),
            LineString([(10, 11), (12, 12)]),
        ],
    }
    connections_gdf = _edge_gdf(
        connections_data,
        source_column="building_id",
        target_column="road_id",
        crs=sample_crs,
    )

    road_links_data = {
        "source_road_id": ["r1"],
        "target_road_id": ["r2"],
        "link_feat1": [0.7],
        "geometry": [
            LineString([(10, 12), (12, 12)]),
        ],
    }
    road_links_gdf = _edge_gdf(
        road_links_data,
        source_column="source_road_id",
        target_column="target_road_id",
        crs=sample_crs,
    )

    return {
        ("building", "connects_to", "road"): connections_gdf,
        ("road", "links_to", "road"): road_links_gdf,
    }


@pytest.fixture
def sample_nx_graph(sample_crs: str) -> nx.Graph:
    """Fixture for a NetworkX graph."""
    return make_nx_graph(
        nx.Graph,
        nodes=[
            (1, {"feature1": 10.0, "label1": 0, "pos": (0, 0), "geometry": Point(0, 0)}),
            (2, {"feature1": 20.0, "label1": 1, "pos": (1, 1), "geometry": Point(1, 1)}),
            (3, {"feature1": 30.0, "label1": 0, "pos": (0, 1), "geometry": Point(0, 1)}),
        ],
        edges=[
            (1, 2, {"edge_feature1": 0.5, "geometry": LineString([(0, 0), (1, 1)])}),
            (1, 3, {"edge_feature1": 0.8, "geometry": LineString([(0, 0), (0, 1)])}),
        ],
        graph_attrs={"crs": sample_crs, "is_hetero": False},
    )


# ============================================================================
# Helper Functions for Creating and Modifying Test Data
# ============================================================================


@pytest.fixture
def empty_gdf(sample_crs: str) -> gpd.GeoDataFrame:
    """Return an empty GeoDataFrame for testing."""
    return gpd.GeoDataFrame(geometry=[], crs=sample_crs)


@pytest.fixture
def geojson_data_path() -> Path:
    """Path to the directory containing GeoJSON test files."""
    return TEST_DATA_DIR


@pytest.fixture
def sample_buildings_gdf(
    geojson_data_path: Path,
    sample_crs: str,
) -> gpd.GeoDataFrame:
    """Load sample building data from GeoJSON."""
    gdf = _load_test_geojson(geojson_data_path / "sample_buildings.geojson", target_crs=sample_crs)
    if "bldg_id" not in gdf.columns:
        gdf["bldg_id"] = gdf.index.astype(str) + "_bldg"
    if "place_id" in gdf.columns:
        gdf = gdf.drop(columns=["place_id"])
    return gdf


@pytest.fixture
def sample_segments_gdf(
    geojson_data_path: Path,
    sample_crs: str,
) -> gpd.GeoDataFrame:
    """Load sample segment data from GeoJSON."""
    gdf = _load_test_geojson(geojson_data_path / "sample_segments.geojson", target_crs=sample_crs)
    if "seg_id" not in gdf.columns:
        gdf["seg_id"] = gdf.index.astype(str) + "_seg"
    gdf["movement_id"] = gdf.index
    return gdf


@pytest.fixture
def mg_center_point(
    sample_segments_gdf: gpd.GeoDataFrame,
    sample_crs: str,
) -> gpd.GeoSeries:
    """Return a center point for filtering, e.g., centroid of the first segment."""
    if not sample_segments_gdf.empty:
        return gpd.GeoSeries(
            [sample_segments_gdf.geometry.iloc[0].centroid],
            crs=sample_crs,
        )
    return gpd.GeoSeries(
        [Point(0, 0)],
        crs=sample_crs,
    )  # Default if segments are empty


@pytest.fixture
def sample_nodes_gdf_alt_crs(sample_nodes_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return a nodes GeoDataFrame with an alternative CRS."""
    return sample_nodes_gdf.to_crs("EPSG:4326")


# Data module fixtures


# Transportation module fixtures
@pytest.fixture
def sample_gtfs_zip() -> typing.Generator[str, None, None]:
    """Create a sample GTFS zip file for testing."""
    yield from temporary_gtfs_zip(_base_gtfs_zip_files())


@pytest.fixture
def sample_gtfs_dict() -> dict[str, pd.DataFrame]:
    """Create a sample GTFS dictionary for testing."""
    return _base_gtfs_dict()


# Additional fixtures for enhanced test coverage


@pytest.fixture
def regular_hetero_graph(sample_crs: str) -> nx.Graph:
    """Fixture for regular (non-MultiGraph) heterogeneous graph."""
    graph = nx.Graph()  # Not MultiGraph
    graph.add_node(1, pos=(0, 0), node_type="building")
    graph.add_node(2, pos=(1, 1), node_type="road")
    graph.add_edge(1, 2, edge_type=("building", "connects", "road"))
    graph.graph = {
        "crs": sample_crs,
        "is_hetero": True,
        "node_types": ["building", "road"],
        "edge_types": [("building", "connects", "road")],
    }
    return graph


@pytest.fixture
def simple_nx_graph(sample_crs: str) -> nx.Graph:
    """Fixture for simple NetworkX graph for dual graph testing."""
    graph = nx.Graph()
    graph.add_node(1, pos=(0, 0))
    graph.add_node(2, pos=(1, 1))
    graph.add_edge(1, 2)
    graph.graph = {"crs": sample_crs, "is_hetero": False}
    return graph


@pytest.fixture
def transport_input() -> TransportData:
    """Provide synthetic transport transport input for public API tests."""
    return {
        "routes": pd.DataFrame([{"route_id": "001", "route_short_name": "R", "route_type": 3}]),
        "stops": gpd.GeoDataFrame(
            {"stop_id": ["001", "002", "003"], "stop_name": ["A", "B", "C"]},
            geometry=[Point(0, 51), Point(0.01, 51), Point(0.02, 51)],
            crs=4326,
        ),
        "patterns": gpd.GeoDataFrame(
            {
                "pattern_id": ["local", "express"],
                "route_id": ["001", "001"],
                "direction_id": [0, 0],
            },
            geometry=[LineString([(0, 51), (0.01, 51), (0.02, 51)])] * 2,
            crs=4326,
        ),
        "stop_sequences": pd.DataFrame(
            {
                "pattern_id": ["local"] * 3 + ["express"] * 2,
                "stop_id": ["001", "002", "003", "001", "003"],
                "stop_sequence": [0, 1, 2, 0, 1],
                "travel_time_secs": [0, 300, 300, 0, 480],
            }
        ),
    }


@pytest.fixture
def agency() -> pd.DataFrame:
    """Provide synthetic transport agency for public API tests."""
    return pd.DataFrame(
        [
            {
                "agency_name": "Synthetic Operator",
                "agency_url": "https://example.org",
                "agency_timezone": "Europe/London",
            }
        ]
    )


@pytest.fixture
def service_dates() -> pd.DataFrame:
    """Provide synthetic transport service dates for public API tests."""
    return pd.DataFrame([{"service_id": "weekday", "date": "20261102", "exception_type": 1}])


@pytest.fixture
def departures() -> pd.DataFrame:
    """Provide synthetic transport departures for public API tests."""
    return pd.DataFrame(
        {
            "pattern_id": ["local", "express"],
            "service_id": ["weekday"] * 2,
            "departure_time": ["08:00:00", "08:05:00"],
        }
    )
