"""Shared pytest fixtures for city2graph tests."""

import importlib.util
import tempfile
import typing
import zipfile
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import pandas as pd
import pytest
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import Polygon

TESTS_DIR = Path(__file__).parent
TEST_DATA_DIR = TESTS_DIR / "data"
DEFAULT_SAMPLE_CRS = "EPSG:27700"
WGS84_CRS = "EPSG:4326"

# Import city2graph modules at the top level to avoid PLC0415
try:
    from city2graph.graph import gdf_to_pyg
    from city2graph.utils import gdf_to_nx
    from city2graph.utils import nx_to_gdf
except ImportError:
    # These imports may fail if torch is not available
    gdf_to_pyg = typing.cast("Any", None)
    gdf_to_nx = typing.cast("Any", None)
    nx_to_gdf = typing.cast("Any", None)

TORCH_AVAILABLE = all(
    importlib.util.find_spec(module_name) is not None
    for module_name in ("torch", "torch_geometric")
)

# Pytest skipif marker for tests requiring torch
requires_torch = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch or PyTorch Geometric is not available.",
)

GraphEdge = tuple[Any, Any, dict[str, Any]] | tuple[Any, Any, Any, dict[str, Any]]
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


def _build_nx_graph(
    graph_cls: type[nx.Graph],
    *,
    nodes: list[tuple[Any, dict[str, Any]]],
    edges: list[GraphEdge],
    graph_attrs: dict[str, Any] | None = None,
) -> nx.Graph:
    """Create a NetworkX graph from declarative node and edge specs."""
    graph = graph_cls()
    for node_id, attrs in nodes:
        graph.add_node(node_id, **attrs)
    for edge in edges:
        if len(edge) == 4:
            source, target, key, attrs = edge
            graph.add_edge(source, target, key=key, **attrs)
            continue
        source, target, attrs = edge
        graph.add_edge(source, target, **attrs)
    graph.graph.update(graph_attrs or {})
    return graph


def _require_pyg_converter() -> typing.Callable[..., object]:
    """Return the GeoDataFrame to PyG converter or skip if unavailable."""
    if gdf_to_pyg is None:
        pytest.skip("PyTorch or PyTorch Geometric is not available.")
    return gdf_to_pyg


def _require_nx_converters() -> tuple[Any, Any]:
    """Return GeoDataFrame/NetworkX converters or skip if unavailable."""
    if gdf_to_nx is None or nx_to_gdf is None:
        pytest.skip("Graph conversion helpers are not available.")
    return gdf_to_nx, nx_to_gdf


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


def _temporary_gtfs_zip(files: dict[str, str]) -> typing.Generator[str, None, None]:
    """Create a temporary GTFS zip archive and yield its path."""
    with (
        tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_file,
        zipfile.ZipFile(temp_file.name, "w") as archive,
    ):
        for filename, contents in files.items():
            archive.writestr(filename, contents)
    try:
        yield temp_file.name
    finally:
        Path(temp_file.name).unlink(missing_ok=True)


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


@pytest.fixture
def od_zones_gdf() -> gpd.GeoDataFrame:
    """Small zones GeoDataFrame for OD/mobility tests.

    Provides three point-like zones with ids 'A','B','C' in WGS84, matching
    the previous local helper used in mobility tests.
    """
    return gpd.GeoDataFrame(
        {
            "zone_id": ["A", "B", "C"],
            "value": [1, 2, 3],
            "geometry": [Point(0, 0), Point(1, 0), Point(1, 1)],
        },
        crs=WGS84_CRS,
    )


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


# --- Contiguity graph fixtures (shared) ---
@pytest.fixture
def sample_polygons_gdf(sample_crs: str) -> gpd.GeoDataFrame:
    """Four squares: A touches B and C; D is isolated."""
    polygons = [
        Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),  # A
        Polygon([(2, 0), (4, 0), (4, 2), (2, 2)]),  # B
        Polygon([(0, 2), (2, 2), (2, 4), (0, 4)]),  # C
        Polygon([(5, 5), (7, 5), (7, 7), (5, 7)]),  # D (isolated)
    ]
    return gpd.GeoDataFrame(
        {
            "polygon_id": ["A", "B", "C", "D"],
            "area": [4.0, 4.0, 4.0, 4.0],
            "use": ["res", "com", "park", "ind"],
            "geometry": polygons,
        },
        crs=sample_crs,
    ).set_index("polygon_id")


@pytest.fixture
def l_shaped_polygons_gdf(sample_crs: str) -> gpd.GeoDataFrame:
    """Two L-shapes sharing a vertex only (Queen yes, Rook no)."""
    polys = [
        Polygon([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]),
        Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
    ]
    return gpd.GeoDataFrame({"id": ["L1", "L2"], "geometry": polys}, crs=sample_crs).set_index("id")


@pytest.fixture
def single_polygon_gdf(sample_crs: str) -> gpd.GeoDataFrame:
    """Single polygon (no adjacency possible)."""
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    return gpd.GeoDataFrame(
        {"id": ["only"], "attr": ["x"], "geometry": [poly]},
        crs=sample_crs,
    ).set_index("id")


@pytest.fixture
def empty_polygons_gdf(sample_crs: str) -> gpd.GeoDataFrame:
    """Empty polygons GeoDataFrame with CRS."""
    return gpd.GeoDataFrame(columns=["id", "geometry"], crs=sample_crs).set_index("id")


@pytest.fixture
def mixed_geometry_gdf(sample_crs: str) -> gpd.GeoDataFrame:
    """Mixed geometry types (Polygon, Point, LineString)."""
    geoms = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), Point(2, 2), LineString([(3, 3), (4, 4)])]
    return gpd.GeoDataFrame({"id": ["p", "pt", "ln"], "geometry": geoms}, crs=sample_crs).set_index(
        "id",
    )


@pytest.fixture
def invalid_geometry_gdf(sample_crs: str) -> gpd.GeoDataFrame:
    """Self-intersecting invalid polygon (bow-tie)."""
    bad = Polygon([(0, 0), (2, 0), (0, 2), (2, 2)])
    return gpd.GeoDataFrame({"id": ["bad"], "geometry": [bad]}, crs=sample_crs).set_index("id")


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
        "source_road_id": ["r1", "r2"],
        "target_road_id": ["r2", "r1"],
        "link_feat1": [0.7, 0.7],
        "geometry": [
            LineString([(10, 12), (12, 12)]),
            LineString([(12, 12), (10, 12)]),
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
    return _build_nx_graph(
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


@pytest.fixture
def sample_pyg_data(sample_nodes_gdf: gpd.GeoDataFrame) -> object:
    """Fixture for a sample PyG Data object."""
    return _require_pyg_converter()(sample_nodes_gdf)


@pytest.fixture
def sample_pyg_hetero_data(
    sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
) -> object:
    """Fixture for a sample PyG HeteroData object."""
    if not sample_hetero_edges_dict:
        dummy_edge_gdf = _edge_gdf(
            {
                "source_id": ["b1"],
                "target_id": ["r1"],
                "geometry": [LineString([(0, 0), (1, 1)])],
            },
            source_column="source_id",
            target_column="target_id",
            crs=sample_hetero_nodes_dict["building"].crs,
        )
        sample_hetero_edges_dict = {("building", "connects_to", "road"): dummy_edge_gdf}

    return _require_pyg_converter()(sample_hetero_nodes_dict, sample_hetero_edges_dict)


@pytest.fixture
def empty_edges_gdf(sample_crs: str) -> gpd.GeoDataFrame:
    """Fixture for an empty edges GeoDataFrame."""
    return gpd.GeoDataFrame(
        columns=["source_id", "target_id", "geometry"],
        crs=sample_crs,
    ).set_index(["source_id", "target_id"])


@pytest.fixture
def edges_dict_with_empty(
    sample_crs: str,
) -> dict[tuple[str, str, str], gpd.GeoDataFrame]:
    """Fixture for an edges dictionary with an empty GeoDataFrame."""
    empty_conn_gdf = gpd.GeoDataFrame(
        columns=["b_id", "r_id", "geometry"],
        crs=sample_crs,
    ).set_index(["b_id", "r_id"])
    return {("building", "connects", "road"): empty_conn_gdf}


@pytest.fixture
def sample_nx_graph_no_crs(sample_nx_graph: nx.Graph) -> nx.Graph:
    """Return a NetworkX graph with no CRS information."""
    return modify_nx_graph_remove_attr(sample_nx_graph, "crs")


@pytest.fixture
def sample_nx_graph_no_pos(sample_crs: str) -> nx.Graph:
    """Fixture for a NetworkX graph with no position data."""
    return _build_nx_graph(
        nx.Graph,
        nodes=[
            (1, {"feature1": 10.0, "label1": 0}),
            (2, {"feature1": 20.0, "label1": 1}),
        ],
        edges=[(1, 2, {"edge_feature1": 0.5})],
        graph_attrs={"crs": sample_crs, "is_hetero": False},
    )


@pytest.fixture
def custom_center_point() -> gpd.GeoSeries:
    """Return a custom center point for testing."""
    point = Point(-2.9879004, 53.4062724)
    return gpd.GeoSeries([point], crs=WGS84_CRS).to_crs(epsg=27700)


# ============================================================================
# Helper Functions for Creating and Modifying Test Data
# ============================================================================


def _create_gdf(
    geometries: list[Any],
    attributes_list: list[dict[str, Any]],
    crs: str = "EPSG:27700",
) -> gpd.GeoDataFrame:
    """
    Create a GeoDataFrame with proper handling of empty cases.

    Args:
        geometries: list of shapely geometry objects
        attributes_list: list of dictionaries with attributes for each geometry
        crs: coordinate reference system
    """
    if not geometries:
        columns = ["geometry"]
        if attributes_list and isinstance(attributes_list[0], dict) and attributes_list[0]:
            columns.extend(attributes_list[0].keys())
        # Remove duplicates while preserving order
        seen = set()
        unique_cols = []
        for item in columns:
            if item not in seen:
                seen.add(item)
                unique_cols.append(item)
        return gpd.GeoDataFrame(columns=unique_cols, geometry="geometry", crs=crs)

    return gpd.GeoDataFrame(attributes_list, geometry=geometries, crs=crs)


def modify_gdf_remove_column(gdf: gpd.GeoDataFrame, column: str) -> gpd.GeoDataFrame:
    """Remove a column from a GeoDataFrame copy."""
    gdf_copy = gdf.copy()
    if column in gdf_copy.columns:
        del gdf_copy[column]
    return gdf_copy


def modify_gdf_remove_crs(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Remove CRS from a GeoDataFrame copy."""
    gdf_copy = gdf.copy()
    gdf_copy.crs = None
    return gdf_copy


def modify_gdf_change_crs(gdf: gpd.GeoDataFrame, new_crs: str) -> gpd.GeoDataFrame:
    """Change CRS of a GeoDataFrame copy."""
    gdf_copy = gdf.copy()
    if not gdf_copy.empty:
        gdf_copy = gdf_copy.to_crs(new_crs)
    else:
        gdf_copy.crs = new_crs
    return gdf_copy


def modify_nx_graph_remove_attr(graph: nx.Graph, attr: str) -> nx.Graph:
    """Remove an attribute from a NetworkX graph copy."""
    graph_copy = graph.copy()
    if hasattr(graph_copy, "graph") and attr in graph_copy.graph:
        del graph_copy.graph[attr]
    return graph_copy


def modify_gdf_single_item(gdf: gpd.GeoDataFrame, index: int = 0) -> gpd.GeoDataFrame:
    """Create a single-item GeoDataFrame from an existing one."""
    if not gdf.empty and len(gdf) > index:
        return gdf.iloc[[index]].copy()
    return gdf.copy()


def modify_gdf_translate(
    gdf: gpd.GeoDataFrame,
    xoff: float,
    yoff: float,
) -> gpd.GeoDataFrame:
    """Translate geometries in a GeoDataFrame copy."""
    gdf_copy = gdf.copy()
    if not gdf_copy.empty:
        gdf_copy.geometry = gdf_copy.geometry.translate(xoff=xoff, yoff=yoff)
    return gdf_copy


def create_coincident_nodes_gdf(sample_crs: str) -> gpd.GeoDataFrame:
    """Create a GeoDataFrame with coincident nodes for testing edge cases."""
    data = {
        "node_id": [1, 2, 3],
        "feature1": [10.0, 20.0, 30.0],
        "geometry": [Point(0, 0), Point(0, 0), Point(1, 1)],  # Two coincident points
    }
    return gpd.GeoDataFrame(data, crs=sample_crs).set_index("node_id")


# --- General Fixtures ---
@pytest.fixture
def empty_gdf(sample_crs: str) -> gpd.GeoDataFrame:
    """Return an empty GeoDataFrame for testing."""
    return _create_gdf([], [], crs=sample_crs)


@pytest.fixture
def geojson_data_path() -> Path:
    """Path to the directory containing GeoJSON test files."""
    return TEST_DATA_DIR


# --- Core Data Fixtures from GeoJSON files ---
@pytest.fixture
def sample_buildings_gdf(
    geojson_data_path: Path,
    sample_crs: str,
) -> gpd.GeoDataFrame:
    """Load sample building data from GeoJSON."""
    gdf = _load_test_geojson(geojson_data_path / "sample_buildings.geojson", target_crs=sample_crs)
    if "bldg_id" not in gdf.columns:
        gdf["bldg_id"] = gdf.index.astype(str) + "_bldg"
    if "private_id" in gdf.columns:
        gdf = gdf.drop(columns=["private_id"])
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
    gdf["public_id"] = gdf.index
    return gdf


@pytest.fixture
def sample_tessellation_gdf(
    sample_buildings_gdf: gpd.GeoDataFrame,
    sample_crs: str,
) -> gpd.GeoDataFrame:
    """Return a simple tessellation-like GDF from sample_buildings_gdf for testing p2p and p2pub.

    This is a placeholder for actual tessellation output.
    It assigns 'private_id' and a dummy 'enclosure_index'.
    """
    if sample_buildings_gdf.empty:
        return _create_gdf(
            [],
            [{"private_id": None, "enclosure_index": None}],
            crs=sample_crs,
        )

    # For simplicity, let's assume each building corresponds to one tessellation cell
    # In reality, tessellation is more complex.
    tess_polys = sample_buildings_gdf.geometry.tolist()
    attrs = [
        {"private_id": i, "enclosure_index": 0}  # Dummy enclosure index
        for i in range(len(sample_buildings_gdf))
    ]
    return _create_gdf(tess_polys, attrs, crs=sample_crs)


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


# --- Fixtures for single-item GDFs for edge case testing ---
@pytest.fixture
def single_building_gdf(sample_buildings_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return a GDF with a single building polygon."""
    return modify_gdf_single_item(sample_buildings_gdf, 0)


@pytest.fixture
def single_segment_gdf(sample_segments_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return a GDF with a single segment line."""
    return modify_gdf_single_item(sample_segments_gdf, 0)


@pytest.fixture
def single_tessellation_cell_gdf(
    sample_tessellation_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Return a GDF with a single tessellation cell."""
    return modify_gdf_single_item(sample_tessellation_gdf, 0)


# --- Fixtures for Error/Edge Cases ---
@pytest.fixture
def private_gdf_no_private_id(
    sample_crs: str,
) -> gpd.GeoDataFrame:  # Represents a tessellation-like input
    """Synthetic private polygons GDF (like tessellation) lacking 'private_id'."""
    return _create_gdf(
        [Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])],
        [{"some_other_id": 100}],
        crs=sample_crs,
    )


@pytest.fixture
def segments_no_public_id_gdf(sample_crs: str) -> gpd.GeoDataFrame:
    """Synthetic segments GDF lacking 'public_id'."""
    return _create_gdf(
        [LineString([(0, 0), (1, 1)])],
        [{"other_id": 10}],
        crs=sample_crs,
    )


@pytest.fixture
def segments_gdf_far_away(sample_segments_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Sample segments translated far away, for testing no priv_pub connections."""
    return modify_gdf_translate(sample_segments_gdf, xoff=100000, yoff=100000)


@pytest.fixture
def segments_gdf_no_crs(sample_segments_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Sample segments GDF with CRS removed."""
    return modify_gdf_remove_crs(sample_segments_gdf)


@pytest.fixture
def segments_gdf_alt_crs(sample_segments_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Sample segments GDF with an alternative CRS."""
    return modify_gdf_change_crs(sample_segments_gdf, "EPSG:4326")


@pytest.fixture
def nodes_gdf_no_crs(sample_nodes_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return a nodes GeoDataFrame with no CRS."""
    return modify_gdf_remove_crs(sample_nodes_gdf)


@pytest.fixture
def sample_nodes_gdf_alt_crs(sample_nodes_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return a nodes GeoDataFrame with an alternative CRS."""
    return modify_gdf_change_crs(sample_nodes_gdf, "EPSG:4326")


@pytest.fixture
def segments_gdf_alt_geom(sample_segments_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Sample segments GDF with an alternative 'barrier_geometry' column."""
    gdf = sample_segments_gdf.copy()
    if not gdf.empty:
        # Create simple circular buffers around centroids as alternative geometries
        alt_geoms = [pt.buffer(0.2) for pt in gdf.geometry.centroid]
        gdf["barrier_geometry"] = gpd.GeoSeries(alt_geoms, crs=gdf.crs)
    else:
        # Ensure the column exists even if empty, with correct dtype and CRS
        gdf["barrier_geometry"] = gpd.GeoSeries([], dtype="geometry", crs=gdf.crs)
    return gdf


@pytest.fixture
def segments_gdf_with_custom_barrier(
    sample_segments_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Sample segments GDF (first segment) with a 'custom_barrier' column."""
    if sample_segments_gdf.empty:
        # Handle empty input: create the column with correct dtype and CRS
        segments_gdf_copy = sample_segments_gdf.copy()
        segments_gdf_copy["custom_barrier"] = gpd.GeoSeries(
            [],
            dtype="geometry",
            crs=segments_gdf_copy.crs,
        )
        return segments_gdf_copy

    segments_gdf = modify_gdf_single_item(sample_segments_gdf, 0)
    # Create a plausible barrier polygon based on the first segment's bounds
    first_segment_bounds = segments_gdf.geometry.iloc[0].bounds
    minx, miny, maxx, maxy = first_segment_bounds
    # Create a slightly larger bounding box as the barrier
    barrier_poly = Polygon(
        [
            (minx - 1, miny - 1),
            (minx - 1, maxy + 1),
            (maxx + 1, maxy + 1),
            (maxx + 1, miny - 1),
        ],
    )
    segments_gdf["custom_barrier"] = gpd.GeoSeries(
        [barrier_poly],
        crs=segments_gdf.crs,
    )
    return segments_gdf


@pytest.fixture
def buildings_invalid_geom_gdf(sample_crs: str) -> gpd.GeoDataFrame:
    """Synthetic buildings GDF with invalid (LineString) geometry."""
    lines = [LineString([(0, 0), (1, 1)])]
    attrs = [{"bldg_id": "InvalidLineGeom"}]  # Ensure bldg_id for consistency
    return _create_gdf(lines, attrs, crs=sample_crs)


@pytest.fixture
def segments_invalid_geom_gdf(sample_crs: str) -> gpd.GeoDataFrame:
    """Synthetic segments GDF with invalid (Polygon) geometry."""
    polys = [Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])]
    attrs = [{"seg_id": "InvalidPolyGeom"}]  # Ensure seg_id for consistency
    return _create_gdf(polys, attrs, crs=sample_crs)


@pytest.fixture
def not_a_gdf() -> pd.DataFrame:
    """Return a regular DataFrame for testing error cases."""
    return pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})


# --- Fixtures for validate_nx testing ---


@pytest.fixture
def sample_nx_multigraph(sample_crs: str) -> nx.MultiGraph:
    """Fixture for a NetworkX MultiGraph."""
    return _build_nx_graph(
        nx.MultiGraph,
        nodes=[
            (1, {"feature1": 10.0, "label1": 0, "pos": (0, 0), "geometry": Point(0, 0)}),
            (2, {"feature1": 20.0, "label1": 1, "pos": (1, 1), "geometry": Point(1, 1)}),
        ],
        edges=[
            (1, 2, 0, {"edge_feature1": 0.5, "geometry": LineString([(0, 0), (1, 1)])}),
            (1, 2, 1, {"edge_feature1": 0.8, "geometry": LineString([(0, 0), (1, 1)])}),
        ],
        graph_attrs={"crs": sample_crs, "is_hetero": False},
    )


@pytest.fixture
def sample_nx_digraph(sample_crs: str) -> nx.DiGraph:
    """Fixture for a NetworkX DiGraph."""
    return _build_nx_graph(
        nx.DiGraph,
        nodes=[
            (1, {"feature1": 10.0, "pos": (0, 0), "geometry": Point(0, 0)}),
            (2, {"feature1": 20.0, "pos": (1, 1), "geometry": Point(1, 1)}),
        ],
        edges=[(1, 2, {"edge_feature1": 0.5, "geometry": LineString([(0, 0), (1, 1)])})],
        graph_attrs={"crs": sample_crs, "is_hetero": False},
    )


@pytest.fixture
def sample_nx_multidigraph(sample_crs: str) -> nx.MultiDiGraph:
    """Fixture for a NetworkX MultiDiGraph."""
    return _build_nx_graph(
        nx.MultiDiGraph,
        nodes=[
            (1, {"feature1": 10.0, "pos": (0, 0), "geometry": Point(0, 0)}),
            (2, {"feature1": 20.0, "pos": (1, 1), "geometry": Point(1, 1)}),
        ],
        edges=[
            (1, 2, 0, {"edge_feature1": 0.5, "geometry": LineString([(0, 0), (1, 1)])}),
            (1, 2, 1, {"edge_feature1": 0.8, "geometry": LineString([(0, 0), (1, 1)])}),
        ],
        graph_attrs={"crs": sample_crs, "is_hetero": False},
    )


@pytest.fixture
def nx_graph_missing_graph_attr() -> nx.Graph:
    """Fixture for a NetworkX graph missing the graph attribute dictionary."""
    graph = nx.Graph()
    graph.add_node(1, pos=(0, 0))
    graph.add_node(2, pos=(1, 1))
    graph.add_edge(1, 2)
    # Remove the graph attribute entirely
    delattr(graph, "graph")
    return graph


@pytest.fixture
def nx_graph_non_dict_graph_attr() -> nx.Graph:
    """Fixture for a NetworkX graph with non-dict graph attribute."""
    graph = nx.Graph()
    graph.add_node(1, pos=(0, 0))
    graph.add_node(2, pos=(1, 1))
    graph.add_edge(1, 2)
    graph.graph = "not_a_dict"  # Invalid type
    return graph


@pytest.fixture
def nx_graph_missing_is_hetero() -> nx.Graph:
    """Fixture for a NetworkX graph missing is_hetero metadata."""
    graph = nx.Graph()
    graph.add_node(1, pos=(0, 0))
    graph.add_node(2, pos=(1, 1))
    graph.add_edge(1, 2)
    graph.graph = {"crs": "EPSG:27700"}  # Missing is_hetero
    return graph


@pytest.fixture
def nx_graph_missing_crs() -> nx.Graph:
    """Fixture for a NetworkX graph missing crs metadata."""
    graph = nx.Graph()
    graph.add_node(1, pos=(0, 0))
    graph.add_node(2, pos=(1, 1))
    graph.add_edge(1, 2)
    graph.graph = {"is_hetero": False}  # Missing crs
    return graph


@pytest.fixture
def nx_graph_no_pos_no_geom() -> nx.Graph:
    """Fixture for a NetworkX graph with nodes missing pos and geometry attributes."""
    graph = nx.Graph()
    graph.add_node(1, feature1=10.0)  # No pos or geometry
    graph.add_node(2, feature1=20.0)  # No pos or geometry
    graph.add_edge(1, 2)
    graph.graph = {"is_hetero": False, "crs": "EPSG:27700"}
    return graph


@pytest.fixture
def nx_graph_with_pos_only() -> nx.Graph:
    """Fixture for a NetworkX graph with nodes having only pos attributes."""
    graph = nx.Graph()
    graph.add_node(1, pos=(0, 0), feature1=10.0)
    graph.add_node(2, pos=(1, 1), feature1=20.0)
    graph.add_edge(1, 2)
    graph.graph = {"is_hetero": False, "crs": "EPSG:27700"}
    return graph


@pytest.fixture
def nx_graph_with_geometry_only() -> nx.Graph:
    """Fixture for a NetworkX graph with nodes having only geometry attributes."""
    graph = nx.Graph()
    graph.add_node(1, geometry=Point(0, 0), feature1=10.0)
    graph.add_node(2, geometry=Point(1, 1), feature1=20.0)
    graph.add_edge(1, 2)
    graph.graph = {"is_hetero": False, "crs": "EPSG:27700"}
    return graph


@pytest.fixture
def nx_hetero_graph_valid() -> nx.Graph:
    """Fixture for a valid heterogeneous NetworkX graph."""
    graph = nx.Graph()
    graph.add_node(1, node_type="building", pos=(0, 0), geometry=Point(0, 0))
    graph.add_node(2, node_type="road", pos=(1, 1), geometry=Point(1, 1))
    graph.add_edge(1, 2, edge_type="connects_to")
    graph.graph = {
        "is_hetero": True,
        "crs": "EPSG:27700",
        "node_types": ["building", "road"],
        "edge_types": [("building", "connects_to", "road")],
    }
    return graph


@pytest.fixture
def nx_hetero_graph_missing_node_types() -> nx.Graph:
    """Fixture for a heterogeneous graph missing node_types metadata."""
    graph = nx.Graph()
    graph.add_node(1, node_type="building", pos=(0, 0))
    graph.add_node(2, node_type="road", pos=(1, 1))
    graph.add_edge(1, 2, edge_type="connects_to")
    graph.graph = {
        "is_hetero": True,
        "crs": "EPSG:27700",
        "edge_types": [("building", "connects_to", "road")],
        # Missing node_types
    }
    return graph


@pytest.fixture
def nx_hetero_graph_empty_node_types() -> nx.Graph:
    """Fixture for a heterogeneous graph with empty node_types."""
    graph = nx.Graph()
    graph.add_node(1, node_type="building", pos=(0, 0))
    graph.add_node(2, node_type="road", pos=(1, 1))
    graph.add_edge(1, 2, edge_type="connects_to")
    graph.graph = {
        "is_hetero": True,
        "crs": "EPSG:27700",
        "node_types": [],  # Empty list
        "edge_types": [("building", "connects_to", "road")],
    }
    return graph


@pytest.fixture
def nx_hetero_graph_missing_edge_types() -> nx.Graph:
    """Fixture for a heterogeneous graph missing edge_types metadata."""
    graph = nx.Graph()
    graph.add_node(1, node_type="building", pos=(0, 0))
    graph.add_node(2, node_type="road", pos=(1, 1))
    graph.add_edge(1, 2, edge_type="connects_to")
    graph.graph = {
        "is_hetero": True,
        "crs": "EPSG:27700",
        "node_types": ["building", "road"],
        # Missing edge_types
    }
    return graph


@pytest.fixture
def nx_hetero_graph_missing_node_type_attr() -> nx.Graph:
    """Fixture for a heterogeneous graph with nodes missing node_type attribute."""
    graph = nx.Graph()
    graph.add_node(1, pos=(0, 0))  # Missing node_type
    graph.add_node(2, node_type="road", pos=(1, 1))
    graph.add_edge(1, 2, edge_type="connects_to")
    graph.graph = {
        "is_hetero": True,
        "crs": "EPSG:27700",
        "node_types": ["building", "road"],
        "edge_types": [("building", "connects_to", "road")],
    }
    return graph


@pytest.fixture
def nx_hetero_graph_missing_edge_type_attr() -> nx.Graph:
    """Fixture for a heterogeneous graph with edges missing edge_type attribute."""
    graph = nx.Graph()
    graph.add_node(1, node_type="building", pos=(0, 0))
    graph.add_node(2, node_type="road", pos=(1, 1))
    graph.add_edge(1, 2)  # Missing edge_type
    graph.graph = {
        "is_hetero": True,
        "crs": "EPSG:27700",
        "node_types": ["building", "road"],
        "edge_types": [("building", "connects_to", "road")],
    }
    return graph


@pytest.fixture
def invalid_input_string() -> str:
    """Fixture for invalid input (string instead of graph)."""
    return "not_a_graph"


@pytest.fixture
def invalid_input_dict() -> dict[str, str]:
    """Fixture for invalid input (dict instead of graph)."""
    return {"not": "a_graph"}


@pytest.fixture
def invalid_input_list() -> list[int]:
    """Fixture for invalid input (list instead of graph)."""
    return [1, 2, 3]


@pytest.fixture
def p2pub_private_single_cell(sample_crs: str) -> gpd.GeoDataFrame:
    """Return a single private cell for private_to_public tests."""
    poly = Polygon([(0.4, 0.4), (0.4, 0.6), (0.6, 0.6), (0.6, 0.4)])
    return _create_gdf([poly], [{"private_id": 0}], crs=sample_crs)


@pytest.fixture
def p2pub_public_single_segment(sample_crs: str) -> gpd.GeoDataFrame:
    """Return a single public segment for private_to_public tests."""
    line = LineString([(0.5, 0.3), (0.5, 0.7)])
    return _create_gdf([line], [{"public_id": 10}], crs=sample_crs)


@pytest.fixture
def p2p_isolated_polys_gdf(sample_crs: str) -> gpd.GeoDataFrame:
    """Return a GDF with polygons that are not contiguous, for testing private_to_private_graph."""
    polys = [
        Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
        Polygon([(10, 10), (10, 11), (11, 11), (11, 10)]),
        Polygon([(20, 20), (20, 21), (21, 21), (21, 20)]),
    ]
    attrs = [{"private_id": i} for i in range(len(polys))]
    return _create_gdf(polys, attrs, crs=sample_crs)


@pytest.fixture
def segments_gdf_with_multiindex_public_id(sample_crs: str) -> gpd.GeoDataFrame:
    """Return a GDF with segments and a MultiIndex for public_id, for public_to_public_graph."""
    data = {
        "seg_id": ["s1", "s2", "s3"],
        "geometry": [
            LineString([(0, 0), (1, 0)]),
            LineString([(1, 0), (2, 0)]),
            LineString([(2, 0), (3, 0)]),
        ],
    }
    gdf = gpd.GeoDataFrame(data, crs=sample_crs)
    # Create a dummy MultiIndex for public_id
    gdf["public_id"] = pd.MultiIndex.from_tuples(
        [("A", 1), ("B", 2), ("C", 3)],
        names=["type", "idx"],
    )
    return gdf


@pytest.fixture
def empty_hetero_nodes_dict() -> dict[str, gpd.GeoDataFrame]:
    """Fixture for an empty dictionary of heterogeneous nodes GeoDataFrames."""
    return {}


@pytest.fixture
def empty_hetero_edges_dict() -> dict[tuple[str, str, str], gpd.GeoDataFrame]:
    """Fixture for an empty dictionary of heterogeneous edges GeoDataFrames."""
    return {}


@pytest.fixture
def empty_nodes_gdf(sample_crs: str) -> gpd.GeoDataFrame:
    """Return an empty nodes GeoDataFrame for testing edge cases."""
    return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=sample_crs)


@pytest.fixture
def single_node_gdf(sample_crs: str) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame with a single node for testing edge cases."""
    data = {
        "node_id": [1],
        "feature1": [10.0],
        "geometry": [Point(0, 0)],
    }
    return gpd.GeoDataFrame(data, crs=sample_crs).set_index("node_id")


@pytest.fixture
def two_nodes_gdf(sample_crs: str) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame with two nodes for testing edge cases."""
    data = {
        "node_id": [1, 2],
        "feature1": [10.0, 20.0],
        "geometry": [Point(0, 0), Point(1, 1)],
    }
    return gpd.GeoDataFrame(data, crs=sample_crs).set_index("node_id")


@pytest.fixture
def coincident_nodes_gdf(sample_crs: str) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame with coincident nodes for testing edge cases."""
    return create_coincident_nodes_gdf(sample_crs)


@pytest.fixture
def network_gdf_with_pos(sample_crs: str) -> gpd.GeoDataFrame:
    """Return a network GeoDataFrame with proper pos attributes for testing."""
    data = {
        "edge_id": ["e1", "e2"],
        "source_id": [1, 2],
        "target_id": [2, 3],
        "geometry": [
            LineString([(0, 0), (1, 1)]),
            LineString([(1, 1), (2, 2)]),
        ],
    }
    edges_gdf = _edge_gdf(
        data,
        source_column="source_id",
        target_column="target_id",
        crs=sample_crs,
    )
    graph_to_nx, nx_to_gdf_converter = _require_nx_converters()
    graph = graph_to_nx(edges=edges_gdf)
    _, edges_with_pos = nx_to_gdf_converter(graph, nodes=True, edges=True)

    return edges_with_pos


@pytest.fixture
def network_gdf_no_pos(sample_crs: str) -> gpd.GeoDataFrame:
    """Return a network GeoDataFrame without pos attributes for testing error cases."""
    data = {
        "edge_id": ["e1"],
        "source_id": [1],
        "target_id": [2],
        "geometry": [LineString([(0, 0), (1, 1)])],
    }
    return _edge_gdf(
        data,
        source_column="source_id",
        target_column="target_id",
        crs=sample_crs,
    )


# Data module fixtures


@pytest.fixture
def test_bbox() -> list[float]:
    """Standard test bounding box."""
    return [-74.01, 40.70, -73.99, 40.72]


@pytest.fixture
def test_polygon() -> Polygon:
    """Standard test polygon."""
    return Polygon([(-74.01, 40.70), (-73.99, 40.70), (-73.99, 40.72), (-74.01, 40.72)])


# Transportation module fixtures
@pytest.fixture
def sample_gtfs_zip() -> typing.Generator[str, None, None]:
    """Create a sample GTFS zip file for testing."""
    yield from _temporary_gtfs_zip(_base_gtfs_zip_files())


@pytest.fixture
def sample_gtfs_zip_with_shapes() -> typing.Generator[str, None, None]:
    """Create a sample GTFS zip file with shapes for testing."""
    files = _base_gtfs_zip_files()
    files.update(
        {
            "stops.txt": (
                "stop_id,stop_name,stop_lat,stop_lon\n"
                "stop1,Stop 1,40.7128,-74.0060\n"
                "stop2,Stop 2,40.7589,-73.9851\n"
            ),
            "trips.txt": "route_id,service_id,trip_id,shape_id\nroute1,service1,trip1,shape1\n",
            "stop_times.txt": (
                "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
                "trip1,08:00:00,08:00:00,stop1,1\n"
                "trip1,08:05:00,08:05:00,stop2,2\n"
            ),
            "shapes.txt": (
                "shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence\n"
                "shape1,40.7128,-74.0060,1\n"
                "shape1,40.7300,-74.0000,2\n"
                "shape1,40.7589,-73.9851,3\n"
            ),
        },
    )
    yield from _temporary_gtfs_zip(files)


@pytest.fixture
def empty_gtfs_zip() -> typing.Generator[str, None, None]:
    """Create an empty GTFS zip file for testing."""
    yield from _temporary_gtfs_zip({})


@pytest.fixture
def gtfs_zip_invalid_coords() -> typing.Generator[str, None, None]:
    """Create a GTFS zip file with invalid coordinates for testing."""
    yield from _temporary_gtfs_zip(
        {
            "stops.txt": (
                "stop_id,stop_name,stop_lat,stop_lon\n"
                "stop1,Stop 1,40.7128,-74.0060\n"
                "stop2,Stop 2,invalid,invalid\n"
                "stop3,Stop 3,,\n"
            ),
            "agency.txt": (
                "agency_id,agency_name,agency_url,agency_timezone\n"
                "1,Test Agency,http://test.com,America/New_York\n"
            ),
            "calendar.txt": (
                "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\n"
                "service1,1,1,1,1,1,0,0,20240101,20241231\n"
            ),
        },
    )


@pytest.fixture
def sample_gtfs_dict() -> dict[str, pd.DataFrame]:
    """Create a sample GTFS dictionary for testing."""
    return _base_gtfs_dict()


@pytest.fixture
def sample_gtfs_dict_with_exceptions(
    sample_gtfs_dict: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Create a sample GTFS dictionary with calendar exceptions."""
    gtfs_dict = sample_gtfs_dict.copy()

    # Add calendar_dates for exceptions
    calendar_dates_df = pd.DataFrame(
        {
            "service_id": ["service1", "service1"],
            "date": ["20240101", "20240102"],
            "exception_type": [2, 1],  # 2 = remove service, 1 = add service
        },
    )

    gtfs_dict["calendar_dates"] = calendar_dates_df
    return gtfs_dict


@pytest.fixture
def not_a_pyg_object() -> str:
    """Fixture for an object that is not a PyG Data or HeteroData object."""
    return "not_a_pyg_object"


# Additional fixtures for enhanced test coverage
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


@pytest.fixture
def multiindex_nodes_gdf(sample_crs: str) -> gpd.GeoDataFrame:
    """Fixture for nodes GeoDataFrame with MultiIndex."""
    return gpd.GeoDataFrame(
        {
            "geometry": [Point(0, 0), Point(1, 1)],
        },
        index=pd.MultiIndex.from_tuples(
            [("type1", 0), ("type1", 1)],
            names=["node_type", "node_id"],
        ),
        crs=sample_crs,
    )


@pytest.fixture
def single_name_index_nodes_gdf(sample_crs: str) -> gpd.GeoDataFrame:
    """Fixture for nodes GeoDataFrame with single-level named index."""
    return gpd.GeoDataFrame(
        {
            "geometry": [Point(0, 0), Point(1, 1)],
        },
        index=pd.Index([0, 1], name="single_name"),
        crs=sample_crs,
    )


@pytest.fixture
def simple_edges_gdf(sample_crs: str) -> gpd.GeoDataFrame:
    """Fixture for simple edges GeoDataFrame."""
    return gpd.GeoDataFrame(
        {
            "geometry": [LineString([(0, 0), (1, 1)])],
        },
        crs=sample_crs,
    )


@pytest.fixture
def directed_multigraph_edges_gdf(sample_crs: str) -> gpd.GeoDataFrame:
    """Fixture for edges suitable for directed multigraph testing."""
    return gpd.GeoDataFrame(
        {
            "geometry": [LineString([(0, 0), (1, 1)])],
        },
        crs=sample_crs,
    )


@pytest.fixture
def hetero_edges_with_multiindex(
    sample_crs: str,
) -> dict[tuple[str, str, str], gpd.GeoDataFrame]:
    """Fixture for heterogeneous edges with MultiIndex."""
    return {
        ("type1", "connects", "type2"): gpd.GeoDataFrame(
            {
                "geometry": [LineString([(0, 0), (1, 1)])],
            },
            index=pd.MultiIndex.from_tuples([("a", "b")], names=["from", "to"]),
            crs=sample_crs,
        ),
    }


@pytest.fixture
def graph_missing_crs() -> nx.Graph:
    """Fixture for NetworkX graph missing CRS metadata."""
    graph = nx.Graph()
    graph.add_node(1, pos=(0, 0))
    graph.add_edge(1, 2)
    graph.graph = {"is_hetero": False}  # Missing crs
    return graph


@pytest.fixture
def hetero_graph_no_node_types(sample_crs: str) -> nx.Graph:
    """Fixture for heterogeneous graph missing node_types."""
    graph = nx.Graph()
    graph.add_node(1, pos=(0, 0))
    graph.add_edge(1, 2)
    graph.graph = {"crs": sample_crs, "is_hetero": True}
    return graph


@pytest.fixture
def hetero_graph_no_edge_types(sample_crs: str) -> nx.Graph:
    """Fixture for heterogeneous graph missing edge_types."""
    graph = nx.Graph()
    graph.add_node(1, pos=(0, 0))
    graph.add_edge(1, 2)
    graph.graph = {"crs": sample_crs, "is_hetero": True, "node_types": ["type1"]}
    return graph


@pytest.fixture
def graph_no_pos_geom(sample_crs: str) -> nx.Graph:
    """Fixture for graph with nodes missing pos/geometry."""
    graph = nx.Graph()
    graph.add_node(1)  # No pos or geometry
    graph.add_edge(1, 2)
    graph.graph = {"crs": sample_crs, "is_hetero": False}
    return graph


@pytest.fixture
def hetero_graph_no_node_type(sample_crs: str) -> nx.Graph:
    """Fixture for heterogeneous graph with nodes missing node_type."""
    graph = nx.Graph()
    graph.add_node(1, pos=(0, 0))  # No node_type
    graph.add_edge(1, 2)
    graph.graph = {
        "crs": sample_crs,
        "is_hetero": True,
        "node_types": ["type1"],
        "edge_types": [("type1", "connects", "type1")],
    }
    return graph


@pytest.fixture
def hetero_graph_no_edge_type(sample_crs: str) -> nx.Graph:
    """Fixture for heterogeneous graph with edges missing edge_type."""
    graph = nx.Graph()
    graph.add_node(1, pos=(0, 0), node_type="type1")
    graph.add_node(2, pos=(1, 1), node_type="type1")
    graph.add_edge(1, 2)  # No edge_type
    graph.graph = {
        "crs": sample_crs,
        "is_hetero": True,
        "node_types": ["type1"],
        "edge_types": [("type1", "connects", "type1")],
    }
    return graph


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
def empty_hetero_graph(sample_crs: str) -> nx.Graph:
    """Fixture for heterogeneous graph with no edges."""
    graph = nx.Graph()
    graph.add_node(1, pos=(0, 0), node_type="building")
    graph.graph = {
        "crs": sample_crs,
        "is_hetero": True,
        "node_types": ["building"],
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
def graph_with_edge_index_names(sample_crs: str) -> nx.Graph:
    """Fixture for graph with edge_index_names set to None."""
    graph = nx.Graph()
    graph.add_node(1, pos=(0, 0))
    graph.add_edge(1, 2)
    graph.graph = {"crs": sample_crs, "is_hetero": False, "edge_index_names": None}
    return graph


@pytest.fixture
def nodes_dict_bad_keys(sample_crs: str) -> dict[int, gpd.GeoDataFrame]:
    """Fixture for nodes dict with invalid keys."""
    return {123: gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs=sample_crs)}


@pytest.fixture
def edges_dict_bad_tuple(sample_crs: str) -> dict[str, gpd.GeoDataFrame]:
    """Fixture for edges dict with invalid tuple keys."""
    return {"not_a_tuple": gpd.GeoDataFrame({"geometry": []}, crs=sample_crs)}


@pytest.fixture
def edges_dict_bad_elements(
    sample_crs: str,
) -> dict[tuple[int, str, str], gpd.GeoDataFrame]:
    """Fixture for edges dict with invalid tuple elements."""
    return {
        (123, "connects", "type2"): gpd.GeoDataFrame({"geometry": []}, crs=sample_crs),
    }


@pytest.fixture
def nodes_non_dict_for_hetero(sample_crs: str) -> gpd.GeoDataFrame:
    """Fixture for non-dict nodes when edges is dict."""
    return gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs=sample_crs)


@pytest.fixture
def edges_dict_for_hetero(
    sample_crs: str,
) -> dict[tuple[str, str, str], gpd.GeoDataFrame]:
    """Fixture for edges dict for heterogeneous testing."""
    return {
        ("type1", "connects", "type2"): gpd.GeoDataFrame(
            {"geometry": []},
            crs=sample_crs,
        ),
    }


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
def hetero_multigraph_with_original_indices(sample_crs: str) -> nx.MultiGraph:
    """Fixture for heterogeneous MultiGraph with _original_edge_index attributes."""
    graph = nx.MultiGraph()
    graph.add_node(1, pos=(0, 0), node_type="building")
    graph.add_node(2, pos=(1, 1), node_type="road")
    # Add edge with _original_edge_index attribute to trigger line 695-696
    graph.add_edge(
        1,
        2,
        key=0,
        edge_type=("building", "connects", "road"),
        _original_edge_index=("custom", "index", "key"),
    )
    graph.graph = {
        "crs": sample_crs,
        "is_hetero": True,
        "node_types": ["building", "road"],
        "edge_types": [("building", "connects", "road")],
    }
    return graph


@pytest.fixture
def hetero_graph_with_original_indices(sample_crs: str) -> nx.Graph:
    """Fixture for heterogeneous Graph with _original_edge_index attributes."""
    graph = nx.Graph()
    graph.add_node(1, pos=(0, 0), node_type="building")
    graph.add_node(2, pos=(1, 1), node_type="road")
    # Add edge with _original_edge_index attribute to trigger line 711
    graph.add_edge(
        1,
        2,
        edge_type=("building", "connects", "road"),
        _original_edge_index=("custom", "index"),
    )
    graph.graph = {
        "crs": sample_crs,
        "is_hetero": True,
        "node_types": ["building", "road"],
        "edge_types": [("building", "connects", "road")],
    }
    return graph


@pytest.fixture
def empty_hetero_multigraph(sample_crs: str) -> nx.MultiGraph:
    """Fixture for empty heterogeneous MultiGraph to test empty edge handling."""
    graph = nx.MultiGraph()
    graph.add_node(1, pos=(0, 0), node_type="building")
    graph.graph = {
        "crs": sample_crs,
        "is_hetero": True,
        "node_types": ["building"],
        "edge_types": [
            ("building", "connects", "road"),
        ],  # Edge type exists but no actual edges
    }
    return graph


@pytest.fixture
def duplicate_segments_gdf(sample_crs: str) -> gpd.GeoDataFrame:
    """Fixture for duplicate segments test data."""
    return gpd.GeoDataFrame(
        {
            "geometry": [
                LineString([(0, 0), (1, 1)]),
                LineString([(0, 0), (1, 1)]),
            ],  # Duplicates
            "road_type": ["primary", "secondary"],
        },
        crs=sample_crs,
    )


@pytest.fixture
def simple_nodes_dict_type1(sample_crs: str) -> dict[str, gpd.GeoDataFrame]:
    """Fixture for simple nodes dict with type1."""
    return {"type1": gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs=sample_crs)}


@pytest.fixture
def simple_edges_dict_type1_type2(
    sample_crs: str,
) -> dict[tuple[str, str, str], gpd.GeoDataFrame]:
    """Fixture for simple edges dict connecting type1 to type2."""
    return {
        ("type1", "connects", "type2"): gpd.GeoDataFrame(
            {
                "geometry": [LineString([(0, 0), (1, 1)])],
            },
            index=pd.MultiIndex.from_tuples([("a", "b")], names=["from", "to"]),
            crs=sample_crs,
        ),
    }


@pytest.fixture
def gtfs_dict_add_service(
    sample_gtfs_dict: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Create a sample GTFS dictionary where calendar_dates adds service."""
    gtfs_dict = sample_gtfs_dict.copy()

    # Modify calendar to have no service on test date initially
    calendar_df = gtfs_dict["calendar"].copy()
    calendar_df["monday"] = [False]  # 20240101 is a Monday
    gtfs_dict["calendar"] = calendar_df

    # Add calendar_dates that adds service on the test date
    calendar_dates_df = pd.DataFrame(
        {
            "service_id": ["service1"],
            "date": ["20240101"],
            "exception_type": [1],  # 1 = add service
        },
    )

    gtfs_dict["calendar_dates"] = calendar_dates_df
    return gtfs_dict


@pytest.fixture
def gtfs_dict_remove_service(
    sample_gtfs_dict: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Create a sample GTFS dictionary where calendar_dates removes service."""
    gtfs_dict = sample_gtfs_dict.copy()

    # Ensure calendar has service on test date initially
    calendar_df = gtfs_dict["calendar"].copy()
    calendar_df["monday"] = [True]  # 20240101 is a Monday
    gtfs_dict["calendar"] = calendar_df

    # Add calendar_dates that removes service on the test date
    calendar_dates_df = pd.DataFrame(
        {
            "service_id": ["service1"],
            "date": ["20240101"],
            "exception_type": [2],  # 2 = remove service
        },
    )

    gtfs_dict["calendar_dates"] = calendar_dates_df
    return gtfs_dict


@pytest.fixture
def sample_gtfs_dict_with_frequencies(
    sample_gtfs_dict: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Create a sample GTFS dictionary with a frequencies table."""
    gtfs_dict = sample_gtfs_dict.copy()

    # Add frequencies.txt: trip1 runs every 600 seconds (10 min) from 08:00 to 09:00
    # That gives floor((09:00:00 - 08:00:00) / 600) = 6 departures
    frequencies_df = pd.DataFrame(
        {
            "trip_id": ["trip1"],
            "start_time": ["08:00:00"],
            "end_time": ["09:00:00"],
            "headway_secs": [600],
        },
    )
    gtfs_dict["frequencies"] = frequencies_df
    return gtfs_dict


@pytest.fixture
def sample_gtfs_dict_with_shapes(
    sample_gtfs_dict: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Create a sample GTFS dictionary with shapes data and shape_id on trips."""
    gtfs_dict = sample_gtfs_dict.copy()

    # Add shape_id to trips
    trips_df = gtfs_dict["trips"].copy()
    trips_df["shape_id"] = ["shape1", "shape1"]
    gtfs_dict["trips"] = trips_df

    # Add shapes.txt with 3 intermediate points between stops
    shapes_df = pd.DataFrame(
        {
            "shape_id": ["shape1"] * 5,
            "shape_pt_lat": [40.7128, 40.7250, 40.7400, 40.7505, 40.7589],
            "shape_pt_lon": [-74.0060, -74.0010, -73.9960, -73.9934, -73.9851],
            "shape_pt_sequence": [1, 2, 3, 4, 5],
        },
    )
    gtfs_dict["shapes"] = shapes_df

    return gtfs_dict
