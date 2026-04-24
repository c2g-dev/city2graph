"""Tests for the networks module (OSM and SUMO file import)."""

# ruff: noqa: D101, D102, D103

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import geopandas as gpd
import networkx as nx
import pytest
from shapely.geometry import LineString
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).parent.parent))

from city2graph.networks import _build_sumo_edges
from city2graph.networks import _build_sumo_nodes
from city2graph.networks import _network_gdfs_to_nx
from city2graph.networks import _parse_sumo_location
from city2graph.networks import _parse_sumo_shape
from city2graph.networks import _safe_int
from city2graph.networks import _sumo_xy_to_lonlat
from city2graph.networks import load_sumo_network

TESTS_DIR = Path(__file__).parent
DATA_DIR = TESTS_DIR / "data"
SAMPLE_OSM = DATA_DIR / "sample.osm"
SAMPLE_NET = DATA_DIR / "sample.net.xml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_nodes_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"node_id": ["A", "B"], "junction_type": ["priority", "priority"],
         "geometry": [Point(-0.1, 51.5), Point(-0.09, 51.51)]},
        geometry="geometry",
        crs="EPSG:4326",
    ).set_index("node_id")


def _make_edges_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "from_id": ["A", "B"],
            "to_id": ["B", "C"],
            "edge_id": ["AB", "BC"],
            "geometry": [
                LineString([(-0.1, 51.5), (-0.09, 51.51)]),
                LineString([(-0.09, 51.51), (-0.08, 51.52)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:4326",
    ).set_index(["from_id", "to_id"])


# ---------------------------------------------------------------------------
# Unit tests: helpers
# ---------------------------------------------------------------------------


class TestSafeInt:
    def test_valid(self) -> None:
        assert _safe_int("3") == 3

    def test_none(self) -> None:
        assert _safe_int(None) is None

    def test_invalid(self) -> None:
        assert _safe_int("abc") is None


class TestParseSumoLocation:
    def test_none_element(self) -> None:
        ox, oy, proj = _parse_sumo_location(None)
        assert ox == 0.0
        assert oy == 0.0
        assert proj is None

    def test_no_projection(self) -> None:
        elem = ET.fromstring(
            '<location netOffset="10.0,20.0" projParameter="!"/>'
        )
        ox, oy, proj = _parse_sumo_location(elem)
        assert ox == 10.0
        assert oy == 20.0
        assert proj is None

    def test_with_projection(self) -> None:
        elem = ET.fromstring(
            '<location netOffset="0,0" projParameter="+proj=utm +zone=30 +ellps=WGS84"/>'
        )
        _, _, proj = _parse_sumo_location(elem)
        assert proj is not None
        assert "+proj=utm" in proj

    def test_malformed_offset_defaults(self) -> None:
        elem = ET.fromstring('<location netOffset="bad" projParameter="!"/>')
        ox, oy, proj = _parse_sumo_location(elem)
        assert ox == 0.0
        assert oy == 0.0


class TestSumoXyToLonLat:
    def test_no_projection_passthrough(self) -> None:
        lon, lat = _sumo_xy_to_lonlat(
            -0.1, 51.5, offset_x=0.0, offset_y=0.0, proj_string=None
        )
        assert lon == pytest.approx(-0.1)
        assert lat == pytest.approx(51.5)

    def test_offset_applied(self) -> None:
        # sumo coords = projected - offset → projected = sumo + offset
        lon, lat = _sumo_xy_to_lonlat(
            5.0, 10.0, offset_x=1.0, offset_y=2.0, proj_string=None
        )
        assert lon == pytest.approx(6.0)
        assert lat == pytest.approx(12.0)

    def test_invalid_proj_falls_back(self) -> None:
        # Should log a warning and return offset-corrected values without raising.
        lon, lat = _sumo_xy_to_lonlat(
            0.0, 0.0, offset_x=1.0, offset_y=2.0, proj_string="+proj=invalid_xyz"
        )
        assert isinstance(lon, float)
        assert isinstance(lat, float)


class TestParseSumoShape:
    def test_valid_shape(self) -> None:
        geom = _parse_sumo_shape(
            "-0.1,51.5 -0.09,51.51",
            offset_x=0.0,
            offset_y=0.0,
            proj_string=None,
        )
        assert isinstance(geom, LineString)
        assert len(geom.coords) == 2

    def test_empty_shape_returns_none(self) -> None:
        geom = _parse_sumo_shape("", offset_x=0.0, offset_y=0.0, proj_string=None)
        assert geom is None

    def test_single_point_returns_none(self) -> None:
        geom = _parse_sumo_shape(
            "-0.1,51.5", offset_x=0.0, offset_y=0.0, proj_string=None
        )
        assert geom is None

    def test_malformed_pair_skipped(self) -> None:
        # One valid pair + one bad → None (fewer than 2 coords)
        geom = _parse_sumo_shape(
            "-0.1,51.5 bad", offset_x=0.0, offset_y=0.0, proj_string=None
        )
        assert geom is None


class TestNetworkGdfsToNx:
    def test_directed(self) -> None:
        G = _network_gdfs_to_nx(_make_nodes_gdf(), _make_edges_gdf(), directed=True)
        assert isinstance(G, nx.DiGraph)
        # A and B come from nodes_gdf; C is added implicitly by the (B, C) edge
        assert {"A", "B"}.issubset(set(G.nodes))
        assert ("A", "B") in G.edges

    def test_undirected(self) -> None:
        G = _network_gdfs_to_nx(_make_nodes_gdf(), _make_edges_gdf(), directed=False)
        assert isinstance(G, nx.Graph)
        assert not isinstance(G, nx.DiGraph)

    def test_crs_stored(self) -> None:
        G = _network_gdfs_to_nx(_make_nodes_gdf(), _make_edges_gdf(), directed=True)
        assert "EPSG:4326" in G.graph["crs"]

    def test_node_geometry_attribute(self) -> None:
        G = _network_gdfs_to_nx(_make_nodes_gdf(), _make_edges_gdf(), directed=True)
        assert isinstance(G.nodes["A"]["geometry"], Point)


# ---------------------------------------------------------------------------
# Unit tests: SUMO XML builders
# ---------------------------------------------------------------------------


class TestBuildSumoNodes:
    def _root(self) -> ET.Element:
        return ET.fromstring(
            """<net>
                <junction id="A" type="priority" x="-0.1" y="51.5"/>
                <junction id="B" type="priority" x="-0.09" y="51.51"/>
                <junction id=":int" type="internal" x="0" y="0"/>
            </net>"""
        )

    def test_internal_junctions_excluded(self) -> None:
        gdf = _build_sumo_nodes(
            self._root(), offset_x=0.0, offset_y=0.0, proj_string=None
        )
        assert ":int" not in gdf.index
        assert "A" in gdf.index
        assert "B" in gdf.index

    def test_geometry_is_point(self) -> None:
        gdf = _build_sumo_nodes(
            self._root(), offset_x=0.0, offset_y=0.0, proj_string=None
        )
        assert isinstance(gdf.loc["A", "geometry"], Point)

    def test_crs(self) -> None:
        gdf = _build_sumo_nodes(
            self._root(), offset_x=0.0, offset_y=0.0, proj_string=None
        )
        assert gdf.crs.to_epsg() == 4326

    def test_empty_net_returns_empty_gdf(self) -> None:
        root = ET.fromstring("<net/>")
        gdf = _build_sumo_nodes(root, offset_x=0.0, offset_y=0.0, proj_string=None)
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 0


class TestBuildSumoEdges:
    def _root_and_nodes(self) -> tuple[ET.Element, gpd.GeoDataFrame]:
        xml = """<net>
            <junction id="A" type="priority" x="-0.1" y="51.5"/>
            <junction id="B" type="priority" x="-0.09" y="51.51"/>
            <edge id="AB" from="A" to="B" priority="1" type="highway.residential">
                <lane id="AB_0" index="0" speed="13.89" length="100.0"
                      shape="-0.1,51.5 -0.09,51.51"/>
            </edge>
            <edge id=":internal" from=":j" to="B">
                <lane id=":internal_0" index="0" speed="5" length="10" shape=""/>
            </edge>
        </net>"""
        root = ET.fromstring(xml)
        nodes = _build_sumo_nodes(root, offset_x=0.0, offset_y=0.0, proj_string=None)
        return root, nodes

    def test_internal_edges_excluded(self) -> None:
        root, nodes = self._root_and_nodes()
        gdf = _build_sumo_edges(
            root, offset_x=0.0, offset_y=0.0, proj_string=None, nodes_gdf=nodes
        )
        assert ":internal" not in gdf.get("edge_id", gdf.reset_index().get("edge_id", []))

    def test_edge_indexed_by_from_to(self) -> None:
        root, nodes = self._root_and_nodes()
        gdf = _build_sumo_edges(
            root, offset_x=0.0, offset_y=0.0, proj_string=None, nodes_gdf=nodes
        )
        assert ("A", "B") in gdf.index

    def test_lane_attributes_aggregated(self) -> None:
        root, nodes = self._root_and_nodes()
        gdf = _build_sumo_edges(
            root, offset_x=0.0, offset_y=0.0, proj_string=None, nodes_gdf=nodes
        )
        assert gdf.loc[("A", "B"), "speed_ms"] == pytest.approx(13.89)
        assert gdf.loc[("A", "B"), "length_m"] == pytest.approx(100.0)
        assert gdf.loc[("A", "B"), "num_lanes"] == 1

    def test_geometry_is_linestring(self) -> None:
        root, nodes = self._root_and_nodes()
        gdf = _build_sumo_edges(
            root, offset_x=0.0, offset_y=0.0, proj_string=None, nodes_gdf=nodes
        )
        assert isinstance(gdf.loc[("A", "B"), "geometry"], LineString)

    def test_fallback_straight_line_when_no_shape(self) -> None:
        xml = """<net>
            <junction id="A" type="priority" x="-0.1" y="51.5"/>
            <junction id="B" type="priority" x="-0.09" y="51.51"/>
            <edge id="AB" from="A" to="B">
                <lane id="AB_0" index="0" speed="10" length="50" shape=""/>
            </edge>
        </net>"""
        root = ET.fromstring(xml)
        nodes = _build_sumo_nodes(root, offset_x=0.0, offset_y=0.0, proj_string=None)
        gdf = _build_sumo_edges(
            root, offset_x=0.0, offset_y=0.0, proj_string=None, nodes_gdf=nodes
        )
        assert isinstance(gdf.loc[("A", "B"), "geometry"], LineString)


# ---------------------------------------------------------------------------
# Integration tests: load_sumo_network
# ---------------------------------------------------------------------------


class TestLoadSumoNetwork:
    def test_returns_tuple_by_default(self) -> None:
        result = load_sumo_network(SAMPLE_NET)
        assert isinstance(result, tuple)
        nodes_gdf, edges_gdf = result
        assert isinstance(nodes_gdf, gpd.GeoDataFrame)
        assert isinstance(edges_gdf, gpd.GeoDataFrame)

    def test_nodes_indexed_by_junction_id(self) -> None:
        nodes_gdf, _ = load_sumo_network(SAMPLE_NET)
        assert nodes_gdf.index.name == "node_id"
        assert "A" in nodes_gdf.index
        assert "B" in nodes_gdf.index
        assert "C" in nodes_gdf.index

    def test_internal_nodes_excluded(self) -> None:
        nodes_gdf, _ = load_sumo_network(SAMPLE_NET)
        assert not any(str(idx).startswith(":") for idx in nodes_gdf.index)

    def test_internal_edges_excluded(self) -> None:
        _, edges_gdf = load_sumo_network(SAMPLE_NET)
        index_tuples = list(edges_gdf.index)
        assert not any(str(f).startswith(":") or str(t).startswith(":") for f, t in index_tuples)

    def test_nodes_crs(self) -> None:
        nodes_gdf, _ = load_sumo_network(SAMPLE_NET)
        assert nodes_gdf.crs.to_epsg() == 4326

    def test_edges_crs(self) -> None:
        _, edges_gdf = load_sumo_network(SAMPLE_NET)
        assert edges_gdf.crs.to_epsg() == 4326

    def test_nodes_geometry_is_point(self) -> None:
        nodes_gdf, _ = load_sumo_network(SAMPLE_NET)
        assert all(isinstance(g, Point) for g in nodes_gdf.geometry)

    def test_edges_geometry_is_linestring(self) -> None:
        _, edges_gdf = load_sumo_network(SAMPLE_NET)
        assert all(isinstance(g, LineString) for g in edges_gdf.geometry if g is not None)

    def test_directed_true_preserves_all_edges(self) -> None:
        _, edges_gdf = load_sumo_network(SAMPLE_NET, directed=True)
        # Fixture has AB, BC, BA → all three should be present
        index_tuples = {(f, t) for f, t in edges_gdf.index}
        assert ("A", "B") in index_tuples
        assert ("B", "A") in index_tuples

    def test_directed_false_deduplicates(self) -> None:
        _, edges_gdf = load_sumo_network(SAMPLE_NET, directed=False)
        index_tuples = list(edges_gdf.index)
        # No pair should appear in both directions
        pairs = {(f, t) for f, t in index_tuples}
        for f, t in pairs:
            assert (t, f) not in pairs or f == t

    def test_as_nx_directed(self) -> None:
        G = load_sumo_network(SAMPLE_NET, as_nx=True, directed=True)
        assert isinstance(G, nx.DiGraph)
        assert G.number_of_nodes() > 0

    def test_as_nx_undirected(self) -> None:
        G = load_sumo_network(SAMPLE_NET, as_nx=True, directed=False)
        assert isinstance(G, nx.Graph)
        assert not isinstance(G, nx.DiGraph)

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_sumo_network("nonexistent.net.xml")

    def test_invalid_xml(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.net.xml"
        bad_file.write_text("not valid xml <<<")
        with pytest.raises(ValueError, match="Failed to parse"):
            load_sumo_network(bad_file)

    def test_edge_attributes_present(self) -> None:
        _, edges_gdf = load_sumo_network(SAMPLE_NET)
        assert "num_lanes" in edges_gdf.columns
        assert "speed_ms" in edges_gdf.columns
        assert "length_m" in edges_gdf.columns
        assert "priority" in edges_gdf.columns

    def test_multi_lane_speed_averaged(self) -> None:
        # Fixture edge AB has 2 lanes both at 13.89 m/s
        _, edges_gdf = load_sumo_network(SAMPLE_NET, directed=True)
        assert edges_gdf.loc[("A", "B"), "speed_ms"] == pytest.approx(13.89)
        assert edges_gdf.loc[("A", "B"), "num_lanes"] == 2


# ---------------------------------------------------------------------------
# Integration tests: load_osm_network
# ---------------------------------------------------------------------------


class TestLoadOsmNetwork:
    @pytest.fixture()
    def osm_result(self) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        from city2graph.networks import load_osm_network

        return load_osm_network(SAMPLE_OSM)  # type: ignore[return-value]

    def test_returns_tuple(
        self, osm_result: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
    ) -> None:
        assert isinstance(osm_result, tuple)
        assert len(osm_result) == 2

    def test_nodes_is_geodataframe(
        self, osm_result: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
    ) -> None:
        nodes_gdf, _ = osm_result
        assert isinstance(nodes_gdf, gpd.GeoDataFrame)

    def test_edges_is_geodataframe(
        self, osm_result: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
    ) -> None:
        _, edges_gdf = osm_result
        assert isinstance(edges_gdf, gpd.GeoDataFrame)

    def test_nodes_crs(
        self, osm_result: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
    ) -> None:
        nodes_gdf, _ = osm_result
        assert nodes_gdf.crs.to_epsg() == 4326

    def test_edges_crs(
        self, osm_result: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
    ) -> None:
        _, edges_gdf = osm_result
        assert edges_gdf.crs.to_epsg() == 4326

    def test_nodes_geometry_is_point(
        self, osm_result: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
    ) -> None:
        nodes_gdf, _ = osm_result
        assert all(isinstance(g, Point) for g in nodes_gdf.geometry)

    def test_edges_index_is_u_v(
        self, osm_result: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
    ) -> None:
        _, edges_gdf = osm_result
        assert edges_gdf.index.names == ["u", "v"]

    def test_undirected_no_reverse_duplicates(self) -> None:
        from city2graph.networks import load_osm_network

        _, edges_gdf = load_osm_network(SAMPLE_OSM, directed=False)
        pairs = list(edges_gdf.index)
        for u, v in pairs:
            assert (v, u) not in pairs or u == v

    def test_as_nx_returns_digraph_when_directed(self) -> None:
        from city2graph.networks import load_osm_network

        G = load_osm_network(SAMPLE_OSM, as_nx=True, directed=True)
        assert isinstance(G, nx.DiGraph)

    def test_as_nx_returns_graph_when_undirected(self) -> None:
        from city2graph.networks import load_osm_network

        G = load_osm_network(SAMPLE_OSM, as_nx=True, directed=False)
        assert isinstance(G, nx.Graph)
        assert not isinstance(G, nx.DiGraph)

    def test_file_not_found(self) -> None:
        from city2graph.networks import load_osm_network

        with pytest.raises(FileNotFoundError):
            load_osm_network("nonexistent.osm")
