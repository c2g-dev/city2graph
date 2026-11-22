"""Tests for :mod:`city2graph.utils.conversion`."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import geopandas as gpd
import networkx as nx
import pandas as pd
import pytest
import rustworkx as rx
from geopandas.testing import assert_geodataframe_equal
from shapely.geometry import LineString
from shapely.geometry import Point

from city2graph import utils
from city2graph.utils import NxConverter
from city2graph.utils import gdf_to_nx
from city2graph.utils import nx_to_gdf
from tests.city2graph import helpers
from tests.city2graph.helpers import make_nx_graph

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def sample_nx_graph_no_crs(sample_nx_graph: nx.Graph) -> nx.Graph:
    """Return an independent graph without CRS metadata."""
    graph = sample_nx_graph.copy()
    del graph.graph["crs"]
    return graph


@pytest.fixture
def sample_nx_graph_no_pos(sample_crs: str) -> nx.Graph:
    """Fixture for a NetworkX graph with no position data."""
    return make_nx_graph(
        nx.Graph,
        nodes=[
            (1, {"feature1": 10.0, "label1": 0}),
            (2, {"feature1": 20.0, "label1": 1}),
        ],
        edges=[(1, 2, {"edge_feature1": 0.5})],
        graph_attrs={"crs": sample_crs, "is_hetero": False},
    )


@pytest.fixture
def not_a_gdf() -> pd.DataFrame:
    """Return a regular DataFrame for testing error cases."""
    return pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})


@pytest.fixture
def sample_nx_multigraph(sample_crs: str) -> nx.MultiGraph:
    """Fixture for a NetworkX MultiGraph."""
    return make_nx_graph(
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
def simple_nodes_dict_type1(sample_crs: str) -> dict[str, gpd.GeoDataFrame]:
    """Fixture for simple nodes dict with type1."""
    return {"type1": gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs=sample_crs)}


class TestGdfToNx:
    """Tests for ``gdf_to_nx``."""

    def test_nullable_attributes_keep_pandas_scalar_boxing(self) -> None:
        """Lazy attribute records preserve nullable payloads across batches."""
        nodes = gpd.GeoDataFrame(
            {"value": pd.array([1, None] * 135, dtype="Int64")},
            geometry=[Point(i, 0) for i in range(270)],
            crs=3857,
        )
        edges = gpd.GeoDataFrame(geometry=[], index=pd.MultiIndex.from_arrays([[], []]), crs=3857)
        graph = gdf_to_nx(nodes=nodes, edges=edges)
        values = [data["value"] for _, data in graph.nodes(data=True)]
        assert values == [1, None] * 135
        assert nodes["value"].isna().sum() == 135

    @pytest.mark.parametrize(
        ("input_type", "gdf_fixture"),
        [("edges_only", "sample_edges_gdf"), ("hetero_edges_only", "sample_hetero_edges_dict")],
    )
    def test_edges_only_conversion(
        self,
        input_type: str,
        gdf_fixture: str,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test conversion with only edge data provided."""
        gdf = request.getfixturevalue(gdf_fixture)

        if input_type == "edges_only":
            graph = gdf_to_nx(edges=gdf)
            assert isinstance(graph, nx.Graph)
            assert graph.number_of_edges() == len(gdf)
            assert graph.number_of_nodes() > 0  # Nodes created from edge endpoints
        else:  # hetero_edges_only
            graph = gdf_to_nx(edges=gdf)
            assert isinstance(graph, nx.Graph)
            # Heterogeneous edges dict without nodes results in empty graph
            assert graph.number_of_edges() == 0

    def test_geometryless_edges_mapped_by_index(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Edges without geometry are mapped via the (source, target) MultiIndex."""
        edges = sample_edges_gdf.set_geometry(
            gpd.GeoSeries([None] * len(sample_edges_gdf), index=sample_edges_gdf.index),
            crs=sample_edges_gdf.crs,
        )
        # An edge referencing an unknown node id cannot be mapped and is dropped
        unknown = edges.iloc[[0]].copy()
        unknown.index = pd.MultiIndex.from_tuples([(1, 99)], names=edges.index.names)
        edges = pd.concat([edges, unknown])

        with caplog.at_level(logging.WARNING, logger="city2graph.utils"):
            graph = gdf_to_nx(nodes=sample_nodes_gdf, edges=edges, keep_geom=False)

        assert graph.number_of_edges() == len(sample_edges_gdf)
        original_edges = {
            (graph.nodes[u]["_original_index"], graph.nodes[v]["_original_index"])
            for u, v in graph.edges()
        }
        assert original_edges == set(sample_edges_gdf.index)
        assert "Could not find nodes for 1 edges by endpoint ID" in caplog.text

    def test_unknown_endpoint_does_not_fall_back_to_geometry(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A matching line cannot override an unknown endpoint ID."""
        nodes = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(1, 0)], index=["a", "b"], crs=3857)
        edges = gpd.GeoDataFrame(
            geometry=[LineString([(0, 0), (1, 0)])],
            crs=3857,
            index=pd.MultiIndex.from_tuples([("a", "missing")]),
        )
        graph = gdf_to_nx(nodes, edges)
        assert graph.number_of_edges() == 0
        assert "Could not find nodes for 1 edges by endpoint ID" in caplog.text

    def test_geometryless_edges_without_multiindex_are_dropped(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Edges without geometry and without a (u, v) MultiIndex cannot be mapped."""
        flat_edges = sample_edges_gdf.reset_index(drop=True)
        edges = flat_edges.set_geometry(
            gpd.GeoSeries([None] * len(flat_edges), index=flat_edges.index),
            crs=flat_edges.crs,
        )

        with caplog.at_level(logging.WARNING, logger="city2graph.utils"):
            graph = gdf_to_nx(nodes=sample_nodes_gdf, edges=edges, keep_geom=False)

        assert graph.number_of_edges() == 0
        assert "MultiIndex is required" in caplog.text

    def test_directed_multigraph_conversion(
        self,
        simple_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Directed + multigraph flags should produce a MultiDiGraph."""
        converter = NxConverter(directed=True, multigraph=True)
        graph = converter.gdf_to_nx(nodes=None, edges=simple_edges_gdf)
        assert isinstance(graph, nx.MultiDiGraph)

    @pytest.mark.parametrize(
        ("nodes_arg", "edges_arg", "error_type", "error_match"),
        [
            (None, None, ValueError, "Either nodes or edges must be provided"),
            ("not_a_gdf", "sample_edges_gdf", TypeError, "Input must be a GeoDataFrame"),
            (
                "sample_hetero_nodes_dict",
                "sample_edges_gdf",
                TypeError,
                "If nodes is a dict, edges must also be a dict",
            ),
            (
                "sample_nodes_gdf_alt_crs",
                "sample_edges_gdf",
                ValueError,
                "All GeoDataFrames must have the same CRS",
            ),
        ],
    )
    def test_conversion_error_handling(
        self,
        nodes_arg: str | None,
        edges_arg: str | None,
        error_type: type[Exception],
        error_match: str,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test proper error handling for invalid conversion inputs."""
        nodes = request.getfixturevalue(nodes_arg) if nodes_arg else None
        edges = request.getfixturevalue(edges_arg) if edges_arg else None

        with pytest.raises(error_type, match=error_match):
            gdf_to_nx(nodes=nodes, edges=edges)

    def test_multiindex_nodes_roundtrip(
        self,
        multiindex_nodes_gdf: gpd.GeoDataFrame,
        simple_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """MultiIndex node metadata should be preserved in graph metadata."""
        graph = gdf_to_nx(nodes=multiindex_nodes_gdf, edges=simple_edges_gdf)
        assert graph.graph["node_index_names"] == ["node_type", "node_id"]

    def test_gdf_to_nx_edges_none(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test gdf_to_nx raises ValueError when edges is None."""
        converter = NxConverter()
        with pytest.raises(ValueError, match="Edges GeoDataFrame cannot be None"):
            converter.gdf_to_nx(nodes=sample_nodes_gdf, edges=None)

    def test_heterogeneous_validation_errors(
        self,
        nodes_dict_bad_keys: dict[int, gpd.GeoDataFrame],
        edges_dict_bad_tuple: dict[str, gpd.GeoDataFrame],
        simple_nodes_dict_type1: dict[str, gpd.GeoDataFrame],
        edges_dict_bad_elements: dict[tuple[int, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Ensure validation catches malformed heterogeneous inputs."""
        with pytest.raises(TypeError, match="Node type keys must be strings"):
            gdf_to_nx(nodes=nodes_dict_bad_keys, edges=None)

        with pytest.raises(TypeError, match="Edge type keys must be tuples"):
            gdf_to_nx(nodes=simple_nodes_dict_type1, edges=edges_dict_bad_tuple)

        with pytest.raises(TypeError, match="All elements in edge type tuples must be strings"):
            gdf_to_nx(nodes=simple_nodes_dict_type1, edges=edges_dict_bad_elements)


class TestNxToGdf:
    """Tests for ``nx_to_gdf``."""

    def test_homogeneous_roundtrip_conversion(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test roundtrip conversion preserves data integrity for homogeneous graphs."""
        graph = gdf_to_nx(sample_nodes_gdf, sample_edges_gdf)
        nodes_converted, edges_converted = nx_to_gdf(graph)

        assert_geodataframe_equal(nodes_converted[sample_nodes_gdf.columns], sample_nodes_gdf)
        assert_geodataframe_equal(edges_converted[sample_edges_gdf.columns], sample_edges_gdf)

    def test_heterogeneous_roundtrip_conversion(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Test roundtrip conversion for heterogeneous graphs."""
        graph = gdf_to_nx(
            nodes=sample_hetero_nodes_dict,
            edges=sample_hetero_edges_dict,
            multigraph=True,
        )
        nodes_dict_converted, edges_dict_converted = nx_to_gdf(graph)

        assert isinstance(nodes_dict_converted, dict)
        assert isinstance(edges_dict_converted, dict)
        assert sample_hetero_nodes_dict.keys() == nodes_dict_converted.keys()
        assert sample_hetero_edges_dict.keys() == edges_dict_converted.keys()

        for node_type, original in sample_hetero_nodes_dict.items():
            assert_geodataframe_equal(nodes_dict_converted[node_type][original.columns], original)
        for edge_type, original in sample_hetero_edges_dict.items():
            assert_geodataframe_equal(edges_dict_converted[edge_type][original.columns], original)

    @pytest.mark.parametrize("offset", [0.0, 0.001])
    @pytest.mark.parametrize("multigraph", [False, True])
    def test_endpoint_ids_override_geometry(self, offset: float, multigraph: bool) -> None:
        """Explicit IDs distinguish colocated nodes and tolerate offset line geometry."""
        nodes = gpd.GeoDataFrame(
            geometry=[Point(0, 0), Point(0, 0), Point(1, 0)],
            index=["a", "b", "c"],
            crs=3857,
        )
        edges = gpd.GeoDataFrame(
            {"weight": [7.0]},
            geometry=[LineString([(offset, 0), (1, 0)])],
            crs=3857,
            index=pd.MultiIndex.from_tuples([("a", "c")], names=["u", "v"]),
        )
        graph = gdf_to_nx(nodes, edges, multigraph=multigraph, directed=True)
        assert [
            (graph.nodes[u]["_original_index"], graph.nodes[v]["_original_index"])
            for u, v in graph.edges()
        ] == [("a", "c")]
        _, restored = nx_to_gdf(graph)
        assert isinstance(restored, gpd.GeoDataFrame)
        assert restored.geometry.iloc[0].equals(edges.geometry.iloc[0])

    def test_nx_to_gdf_requires_nodes_or_edges(self, simple_nx_graph: nx.Graph) -> None:
        """Requesting neither nodes nor edges should raise."""
        with pytest.raises(ValueError, match="Must request at least one of nodes or edges"):
            nx_to_gdf(simple_nx_graph, nodes=False, edges=False)

    def test_node_index_names_survive_roundtrip(
        self,
        single_name_index_nodes_gdf: gpd.GeoDataFrame,
        simple_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Roundtrips should respect single-level index names."""
        graph = gdf_to_nx(nodes=single_name_index_nodes_gdf, edges=simple_edges_gdf)
        nodes_back, _ = nx_to_gdf(graph)
        assert isinstance(nodes_back, gpd.GeoDataFrame)
        assert nodes_back.index.name == "single_name"

        graph.graph["node_index_names"] = None
        nodes_back, _ = nx_to_gdf(graph)
        assert isinstance(nodes_back, gpd.GeoDataFrame)
        assert nodes_back.index.name is None

    @pytest.mark.parametrize(
        ("graph_fixture", "expect_crs", "expect_geom"),
        [
            ("sample_nx_graph", True, True),
            ("sample_nx_graph_no_crs", False, True),
            ("sample_nx_graph_no_pos", True, True),
        ],
    )
    def test_nx_to_gdf_variants(
        self,
        graph_fixture: str,
        expect_crs: bool,
        expect_geom: bool,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test NetworkX to GDF conversion with different graph properties."""
        graph = request.getfixturevalue(graph_fixture)
        nodes, edges = nx_to_gdf(graph)

        # These are homogeneous graphs, so ensure they return GeoDataFrames not dicts
        assert isinstance(nodes, gpd.GeoDataFrame)
        assert isinstance(edges, gpd.GeoDataFrame)

        if expect_geom:
            assert "geometry" in nodes.columns
            assert "geometry" in edges.columns

        if expect_crs:
            assert nodes.crs is not None
            assert edges.crs is not None
        else:
            assert nodes.crs is None
            assert edges.crs is None

    def test_nx_to_gdf_handles_empty_edges(self, sample_crs: str) -> None:
        """Graphs with no edges should still return empty edge GeoDataFrames."""
        graph = nx.Graph()
        graph.add_node(1, pos=(0, 0), geometry=Point(0, 0))
        graph.graph = {"crs": sample_crs, "is_hetero": False}

        nodes_gdf, edges_gdf = nx_to_gdf(graph)
        helpers.assert_valid_gdf(nodes_gdf)
        helpers.assert_valid_gdf(edges_gdf, expected_empty=True)

    def test_nx_to_gdf_multigraph_edges(self, sample_crs: str) -> None:
        """Multigraph attributes should be preserved after conversion."""
        graph = nx.MultiGraph()
        graph.add_node(1, pos=(0, 0), geometry=Point(0, 0))
        graph.add_node(2, pos=(1, 1), geometry=Point(1, 1))
        graph.add_edge(1, 2, key=0, weight=1.0, geometry=LineString([(0, 0), (1, 1)]))
        graph.graph = {"crs": sample_crs, "is_hetero": False}

        _, edges_gdf = nx_to_gdf(graph)
        helpers.assert_valid_gdf(edges_gdf)
        assert isinstance(edges_gdf, gpd.GeoDataFrame)
        assert "weight" in edges_gdf.columns

    def test_nx_to_gdf_multiindex_edges(self, sample_crs: str) -> None:
        """List-based stored edge indices should be normalized during reconstruction."""
        graph = nx.MultiGraph()
        graph.add_node(1, pos=(0, 0), geometry=Point(0, 0))
        graph.add_node(2, pos=(1, 1), geometry=Point(1, 1))
        graph.add_edge(1, 2, key=0, geometry=LineString([(0, 0), (1, 1)]))
        graph.add_edge(1, 2, key=1, geometry=LineString([(0, 0), (1, 1)]))
        graph.graph = {"crs": sample_crs, "is_hetero": False}

        for u, v, k, attrs in graph.edges(data=True, keys=True):
            attrs["_original_edge_index"] = [u, v, k]

        _, edges_gdf = nx_to_gdf(graph)
        helpers.assert_valid_gdf(edges_gdf)
        assert len(edges_gdf) == 2

    def test_nx_to_gdf_single_coord_attr(self, sample_crs: str) -> None:
        """Missing pos attributes can be populated from alternative coordinate keys."""
        graph = nx.Graph()
        graph.add_node(1, coords=[0.0, 1.0])
        graph.add_node(2, coords=(2.0, 3.0))
        graph.graph = {"crs": sample_crs, "is_hetero": False}

        nodes_gdf = nx_to_gdf(graph, nodes=True, edges=False, set_missing_pos_from=("coords",))
        helpers.assert_valid_gdf(nodes_gdf)
        assert len(nodes_gdf) == 2

    @pytest.mark.parametrize("keep_geom", [True, False])
    def test_heterogeneous_geometryless_edges(self, keep_geom: bool) -> None:
        """Typed edges use endpoint IDs even when all geometries are missing."""
        nodes = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(1, 0)], index=["a", "b"], crs=3857)
        edges = gpd.GeoDataFrame(
            {"weight": [7.0]},
            geometry=[None],
            crs=3857,
            index=pd.MultiIndex.from_tuples([("a", "b")], names=["source", "target"]),
        )
        edge_type = ("place", "connects", "place")
        graph = gdf_to_nx({"place": nodes}, {edge_type: edges}, keep_geom=keep_geom, directed=True)
        assert graph.number_of_edges() == 1
        _, restored = nx_to_gdf(graph)
        assert restored[edge_type].index.tolist() == [("a", "b")]
        assert restored[edge_type]["weight"].tolist() == [7.0]
        assert edges.geometry.isna().all()

    def test_heterogeneous_edge_processing(
        self,
        regular_hetero_graph: nx.Graph,
        empty_hetero_graph: nx.Graph,
    ) -> None:
        """nx_to_gdf should return dictionaries for heterogeneous graphs."""
        nodes_dict, edges_dict = nx_to_gdf(regular_hetero_graph)
        assert isinstance(nodes_dict, dict)
        assert isinstance(edges_dict, dict)

        nodes_dict, edges_dict = nx_to_gdf(empty_hetero_graph)
        assert isinstance(edges_dict[("building", "connects", "road")], gpd.GeoDataFrame)
        assert edges_dict[("building", "connects", "road")].empty

    def test_empty_nodes_in_conversion(self, sample_crs: str) -> None:
        """Test conversion with completely empty nodes (line 995)."""
        # Create edges but with no nodes that will result in empty node records
        edges_gdf = gpd.GeoDataFrame(
            geometry=[LineString([(0, 0), (1, 1)])],
            crs=sample_crs,
        )

        # Create a graph and then remove all node attributes to trigger empty records
        converter = NxConverter()
        graph = converter.gdf_to_nx(edges=edges_gdf)

        # Manually clear node data to simulate empty records scenario
        for node in graph.nodes():
            # Keep only pos to trigger the empty records path
            graph.nodes[node].clear()
            graph.nodes[node]["pos"] = (0, 0)

        # Convert back - this should handle empty node records
        nodes_gdf, _edges_gdf_out = converter.nx_to_gdf(graph)
        assert isinstance(nodes_gdf, gpd.GeoDataFrame)

    def test_build_edge_index_empty(self) -> None:
        """Retain the legacy empty-index branch bypassed by public empty-graph handling."""
        converter = NxConverter()
        result = converter._build_edge_index([], None)
        assert isinstance(result, pd.Index)
        assert len(result) == 0

    def test_empty_graph_node_reconstruction(self, sample_crs: str) -> None:
        """Test nx_to_gdf with graph that has nodes without attributes (line 996)."""
        # Create a minimal graph with nodes that have no custom attributes
        # This will result in empty records when reconstructing
        G = nx.Graph()
        G.add_node(0, pos=(0, 0))
        G.add_node(1, pos=(1, 1))
        G.graph = {"crs": sample_crs, "is_hetero": False, "node_index_names": None}

        # Convert to GDF - should handle empty records gracefully
        nodes_gdf, _edges_gdf = utils.nx_to_gdf(G)
        assert isinstance(nodes_gdf, gpd.GeoDataFrame)
        assert len(nodes_gdf) == 2
        assert "geometry" in nodes_gdf.columns

    def test_edge_iteration_fallback(self, sample_crs: str) -> None:
        """Test edge iteration handles unexpected edge format (line 1200)."""
        # Create a graph and manually add malformed edge to test the fallback
        converter = NxConverter()
        graph = nx.Graph()
        graph.add_node(0, pos=(0, 0))
        graph.add_node(1, pos=(1, 1))
        graph.add_edge(0, 1, geometry=LineString([(0, 0), (1, 1)]))
        graph.graph = {"crs": sample_crs, "is_hetero": False, "edge_index_names": None}

        # The actual iteration path that could hit line 1200 is very defensive
        # It's hard to trigger without modifying internal NetworkX behavior
        # Convert to GDF to exercise the edge reconstruction path
        _, edges_gdf = converter.nx_to_gdf(graph)
        assert isinstance(edges_gdf, gpd.GeoDataFrame)

    def test_nx_to_gdf_multiindex_nodes(self, sample_crs: str) -> None:
        """Test nx_to_gdf with MultiIndex nodes."""
        G = nx.Graph()
        # Add nodes with tuple indices
        G.add_node((0, "a"), pos=(0, 0), _original_index=(0, "a"))
        G.add_node((1, "b"), pos=(1, 1), _original_index=(1, "b"))
        G.graph = {"crs": sample_crs, "is_hetero": False, "node_index_names": ["id", "type"]}

        nodes_gdf, _ = utils.nx_to_gdf(G)
        assert isinstance(nodes_gdf, gpd.GeoDataFrame)
        assert isinstance(nodes_gdf.index, pd.MultiIndex)
        assert nodes_gdf.index.names == ["id", "type"]
        assert len(nodes_gdf) == 2

    @pytest.mark.parametrize(
        ("names", "node_id", "expected"),
        [
            pytest.param("index_name", 7, pd.Index([7], name="index_name"), id="string-name"),
            pytest.param(
                ["a", "b"],
                (7, "x"),
                pd.MultiIndex.from_tuples([(7, "x")], names=["a", "b"]),
                id="multiple-names",
            ),
            pytest.param(None, 7, pd.Index([7]), id="unnamed"),
        ],
    )
    def test_nx_to_gdf_legacy_index_names(
        self,
        names: str | list[str] | None,
        node_id: int | tuple[int, str],
        expected: pd.Index,
        sample_crs: str,
    ) -> None:
        """Conversion restores index names stored by older NetworkX graphs."""
        graph = nx.Graph(crs=sample_crs, is_hetero=False, node_index_names=names)
        graph.add_node(0, pos=(0, 0), _original_index=node_id)
        nodes, edges = utils.nx_to_gdf(graph)
        assert isinstance(nodes, gpd.GeoDataFrame)
        assert isinstance(edges, gpd.GeoDataFrame)
        pd.testing.assert_index_equal(nodes.index, expected)
        assert nodes.geometry.iloc[0].equals(Point(0, 0))
        assert edges.empty

    def test_nx_to_gdf_empty_graph_no_nodes(self, sample_crs: str) -> None:
        """Test nx_to_gdf with a completely empty graph (no nodes)."""
        G = nx.Graph()
        G.graph = {"crs": sample_crs, "is_hetero": False}
        nodes_gdf, edges_gdf = utils.nx_to_gdf(G)
        assert isinstance(nodes_gdf, gpd.GeoDataFrame)
        assert isinstance(edges_gdf, gpd.GeoDataFrame)
        assert nodes_gdf.empty
        assert edges_gdf.empty
        pd.testing.assert_index_equal(nodes_gdf.index, pd.Index([]))
        assert nodes_gdf.crs == sample_crs
        assert "geometry" in nodes_gdf.columns

    @pytest.mark.parametrize("all_empty", [True, False])
    @pytest.mark.parametrize("multiindex", [True, False])
    def test_empty_heterogeneous_node_types(self, all_empty: bool, multiindex: bool) -> None:
        """Declared empty types retain their CRS and index structure on round trips."""
        index = (
            pd.MultiIndex.from_arrays([[], []], names=["district", "id"])
            if multiindex
            else pd.Index([], name="id")
        )
        empty = gpd.GeoDataFrame(geometry=[], index=index, crs=3857)
        populated = gpd.GeoDataFrame(
            geometry=[Point(0, 0)], index=pd.Index([1], name="id"), crs=3857
        )
        nodes = {"empty": empty, "other": empty if all_empty else populated}
        restored, edges = nx_to_gdf(gdf_to_nx(nodes, {}))
        assert set(restored) == set(nodes)
        assert edges == {}
        for name, original in nodes.items():
            assert len(restored[name]) == len(original)
            assert restored[name].crs == original.crs
            assert restored[name].index.names == original.index.names
            assert isinstance(restored[name].index, type(original.index))

    def test_nx_to_gdf_pos_from_xy_attributes(self) -> None:
        """Node x/y attributes reconstruct indexed point geometries and CRS."""
        G = nx.Graph()
        G.add_node(1, x=0.0, y=0.0)
        G.add_node(2, x=1.0, y=1.0)
        G.add_edge(1, 2)
        G.graph = {"crs": "EPSG:27700", "is_hetero": False}

        # Should populate 'pos' from x/y when calling nx_to_gdf
        nodes_gdf = nx_to_gdf(G, nodes=True, edges=False)
        assert isinstance(nodes_gdf, gpd.GeoDataFrame)
        assert list(nodes_gdf.index) == [1, 2]
        assert nodes_gdf.geometry.tolist() == [Point(0, 0), Point(1, 1)]
        assert nodes_gdf.crs == "EPSG:27700"


class TestValidateGdf:
    """Tests for ``validate_gdf``."""

    @pytest.mark.parametrize(
        ("nodes_fixture", "edges_fixture", "should_error", "error_match"),
        [
            # Success cases
            ("sample_nodes_gdf", "sample_edges_gdf", False, None),
            ("sample_nodes_gdf", None, False, None),
            (None, "sample_edges_gdf", False, None),
            ("empty_gdf", "sample_edges_gdf", False, None),
            # Error cases
            ("not_a_gdf", "sample_edges_gdf", True, "Input must be a GeoDataFrame"),
            (
                "sample_nodes_gdf_alt_crs",
                "sample_edges_gdf",
                True,
                "All GeoDataFrames must have the same CRS",
            ),
            (
                "sample_nodes_gdf",
                "edges_dict_for_hetero",
                True,
                "If edges is a dict, nodes must also be a dict or None",
            ),
            (
                "simple_nodes_dict_type1",
                "sample_edges_gdf",
                True,
                "If nodes is a dict, edges must also be a dict or None",
            ),
        ],
    )
    def test_gdf_validation(
        self,
        nodes_fixture: str | None,
        edges_fixture: str | None,
        should_error: bool,
        error_match: str | None,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test GeoDataFrame validation with various input combinations."""
        nodes = request.getfixturevalue(nodes_fixture) if nodes_fixture else None
        edges = request.getfixturevalue(edges_fixture) if edges_fixture else None

        if should_error:
            with pytest.raises((TypeError, ValueError), match=error_match):
                utils.validate_gdf(nodes, edges)
        else:
            utils.validate_gdf(nodes, edges)  # Should not raise

    def test_validation_edge_cases(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        empty_gdf: gpd.GeoDataFrame,
        segments_invalid_geom_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test validation handles edge cases properly."""
        # Empty edges should be allowed
        utils.validate_gdf(sample_nodes_gdf, empty_gdf)

        # Invalid geometries should be handled with warning
        utils.validate_gdf(sample_nodes_gdf, segments_invalid_geom_gdf)


class TestValidateNx:
    """Tests for ``validate_nx``."""

    @pytest.mark.parametrize(
        ("graph_fixture", "should_error", "error_match"),
        [
            ("sample_nx_graph", False, None),
            ("sample_nx_multigraph", False, None),
            ("not_a_gdf", True, "Input must be a NetworkX Graph or MultiGraph"),
        ],
    )
    def test_nx_validation(
        self,
        graph_fixture: str,
        should_error: bool,
        error_match: str | None,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test NetworkX graph validation."""
        graph = request.getfixturevalue(graph_fixture)

        if should_error:
            with pytest.raises(TypeError, match=error_match):
                utils.validate_nx(graph)
        else:
            utils.validate_nx(graph)  # Should not raise

    @pytest.mark.parametrize(
        ("graph_fixture", "error_match"),
        [
            ("graph_missing_crs", "Graph metadata is missing required key"),
            (
                "hetero_graph_no_node_types",
                "Heterogeneous graph metadata is missing 'node_types'",
            ),
            (
                "hetero_graph_no_edge_types",
                "Heterogeneous graph metadata is missing 'edge_types'",
            ),
            ("graph_no_pos_geom", "All nodes must have a 'pos' or 'geometry' attribute"),
            (
                "hetero_graph_no_node_type",
                "All nodes in a heterogeneous graph must have a 'node_type' attribute",
            ),
            (
                "hetero_graph_no_edge_type",
                "All edges in a heterogeneous graph must have an 'edge_type' attribute",
            ),
        ],
    )
    def test_validate_nx_rejects_invalid_graphs(
        self,
        graph_fixture: str,
        error_match: str,
        request: pytest.FixtureRequest,
    ) -> None:
        """validate_nx should surface detailed errors."""
        graph = request.getfixturevalue(graph_fixture)
        with pytest.raises(ValueError, match=error_match):
            utils.validate_nx(graph)

    def test_validate_nx_basic_structure_errors(self) -> None:
        """Graphs with missing metadata or topology should raise informative errors."""
        empty_graph = nx.Graph()
        with pytest.raises(ValueError, match="Graph has no nodes"):
            utils.validate_nx(empty_graph)

        no_edges_graph = nx.Graph()
        no_edges_graph.add_node(1)
        with pytest.raises(ValueError, match="Graph has no edges"):
            utils.validate_nx(no_edges_graph)

        incomplete_graph = nx.Graph()
        incomplete_graph.add_node(1, pos=(0, 0))
        incomplete_graph.add_edge(1, 2)
        delattr(incomplete_graph, "graph")
        with pytest.raises(ValueError, match="Graph is missing 'graph' attribute dictionary"):
            utils.validate_nx(incomplete_graph)

    def test_validate_nx_pos_from_xy(self, sample_crs: str) -> None:
        """Test validate_nx creates pos from x and y attributes (line 2016)."""
        # Create a graph with x, y but no pos
        G = nx.Graph()
        G.add_node(1, x=10.0, y=20.0)
        G.add_node(2, x=30.0, y=40.0)
        G.add_edge(1, 2, geometry=LineString([(10, 20), (30, 40)]))
        G.graph = {"crs": sample_crs, "is_hetero": False}

        # This should set pos from x and y
        utils.validate_nx(G)

        assert "pos" in G.nodes[1]
        assert "pos" in G.nodes[2]
        assert G.nodes[1]["pos"] == (10.0, 20.0)
        assert G.nodes[2]["pos"] == (30.0, 40.0)


class TestRxToNx:
    """Tests for ``rx_to_nx``."""

    @pytest.mark.parametrize(
        "create_graph",
        [
            lambda: nx.Graph(name="undirected"),
            lambda: nx.DiGraph(name="directed"),
            lambda: nx.MultiGraph(name="multi_undirected"),
            lambda: nx.MultiDiGraph(name="multi_directed"),
        ],
    )
    def test_nx_rx_roundtrip(self, create_graph: Callable[[], nx.Graph]) -> None:
        """Test full roundtrip conversion preserves structure and attributes."""
        # Setup complex graph
        G = create_graph()
        G.graph["crs"] = "EPSG:4326"
        G.add_node("a", color="red", size=10)
        G.add_node(1, color="blue")

        # Add edges with attributes
        G.add_edge("a", 1, weight=0.5, type="road")

        # If multi-graph, add another edge between same nodes
        if G.is_multigraph():
            G.add_edge("a", 1, weight=0.8, type="path")

        # Convert to rustworkx
        rx_graph = utils.nx_to_rx(G)

        # Verify RX structure
        assert rx_graph.num_nodes() == G.number_of_nodes()
        assert rx_graph.num_edges() == G.number_of_edges()
        assert rx_graph.attrs["crs"] == "EPSG:4326"

        # Convert back to NetworkX
        G_restored = utils.rx_to_nx(rx_graph)

        # Verify restored graph
        assert nx.utils.graphs_equal(G, G_restored)

    @pytest.mark.parametrize("graph_cls", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph])
    def test_roundtrip_arbitrary_and_reserved_attribute_keys(
        self, graph_cls: type[nx.Graph]
    ) -> None:
        """Restore arbitrary attribute keys, including constructor args and internal metadata keys."""
        graph = graph_cls()
        graph.add_edge(("a", 0), ("b", 1))
        edge_id = (("a", 0), ("b", 1), 0) if graph.is_multigraph() else (("a", 0), ("b", 1))
        graph.nodes[("a", 0)]["node_for_adding"] = "node label"
        graph.nodes[("a", 0)]["__nx_node_id__"] = "user node id"
        graph.nodes[("b", 1)][42] = "integer node attribute key"
        graph.nodes[("b", 1)]["__nx_node_id__"] = None
        graph.edges[edge_id]["key"] = "serialized edge label"
        graph.edges[edge_id]["u_of_edge"] = "source label"
        graph.edges[edge_id][7] = "integer edge attribute key"
        graph.edges[edge_id]["__nx_edge_key__"] = "user edge key"

        rx_graph = utils.nx_to_rx(graph)
        restored = utils.rx_to_nx(rx_graph)
        restored_again = utils.rx_to_nx(rx_graph)

        assert type(restored) is graph_cls
        assert nx.utils.graphs_equal(restored, graph)
        assert nx.utils.graphs_equal(restored_again, graph)

    def test_rx_to_nx_raw_multigraph_without_edge_keys(self) -> None:
        """Raw rustworkx multigraphs without __nx_edge_key__ assign distinct keys."""
        rx_graph = rx.PyGraph(multigraph=True)
        i1 = rx_graph.add_node({"label": "n1"})
        i2 = rx_graph.add_node({"label": "n2"})
        rx_graph.add_edge(i1, i2, {"weight": 1.0})
        rx_graph.add_edge(i1, i2, None)

        nx_graph = utils.rx_to_nx(rx_graph)

        assert isinstance(nx_graph, nx.MultiGraph)
        assert nx_graph.number_of_edges() == 2
        assert nx_graph.edges[i1, i2, 0] == {"weight": 1.0}
        assert nx_graph.edges[i1, i2, 1] == {}

    def test_rx_to_nx_raw_input(self) -> None:
        """Test converting a raw rustworkx graph (no __nx_node_id__)."""
        rx_G = rx.PyGraph(multigraph=False)
        rx_G.attrs = {"test": "attr"}

        # Add nodes with raw payload (not dict) and dict payload
        idx1 = rx_G.add_node("raw_payload")
        idx2 = rx_G.add_node({"attr": "value"})

        rx_G.add_edge(idx1, idx2, {"weight": 0.5})

        if rx_G.multigraph:
            rx_G.add_edge(idx1, idx2, {"weight": 0.8})

        # Convert
        nx_G = utils.rx_to_nx(rx_G)

        assert isinstance(nx_G, nx.Graph)
        assert nx_G.graph["test"] == "attr"

        # Check nodes - should use integer indices since no __nx_node_id__
        assert idx1 in nx_G.nodes
        assert nx_G.nodes[idx1]["payload"] == "raw_payload"
        assert nx_G.nodes[idx2]["attr"] == "value"

        # Check edge
        assert nx_G.has_edge(idx1, idx2)
        assert nx_G.edges[idx1, idx2]["weight"] == 0.5

    def test_rx_to_nx_raw_edge_payload(self) -> None:
        """Test converting RX graph with non-dict edge payloads."""
        rx_G = rx.PyGraph(multigraph=False)
        i1 = rx_G.add_node(None)
        i2 = rx_G.add_node(None)
        rx_G.add_edge(i1, i2, "edge_label")

        nx_G = utils.rx_to_nx(rx_G)
        assert nx_G.edges[i1, i2]["payload"] == "edge_label"

    @pytest.mark.parametrize("graph_cls", [nx.MultiGraph, nx.MultiDiGraph])
    def test_repeated_conversion_preserves_input(self, graph_cls: type[nx.MultiGraph]) -> None:
        """Repeated conversions retain keys without consuming source attributes."""
        graph = graph_cls()
        graph.add_edge("a", "b", key="road-1", weight=2)
        graph.add_edge("a", "b", key="road-2", weight=5)
        source = utils.nx_to_rx(graph)
        before = {
            key: (u, v, attrs.copy()) for key, (u, v, attrs) in source.edge_index_map().items()
        }
        first = utils.rx_to_nx(source)
        second = utils.rx_to_nx(source)
        assert nx.utils.graphs_equal(first, graph)
        assert nx.utils.graphs_equal(second, graph)
        assert source.edge_index_map() == before
