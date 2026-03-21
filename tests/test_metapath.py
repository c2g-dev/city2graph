"""Tests for the metapath module."""

from __future__ import annotations

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import Polygon

from city2graph.metapath import add_metapaths
from city2graph.metapath import add_metapaths_by_weight
from city2graph.utils import gdf_to_nx

METAPATH = [("building", "connects_to", "road"), ("road", "links_to", "road")]
RESULT_KEY = ("building", "metapath_0", "road")

WeightGraphData = tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]]


@pytest.fixture
def sample_weight_graph_data() -> tuple[
    dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]
]:
    """Small heterogeneous graph tailored for add_metapaths_by_weight tests."""
    buildings = gpd.GeoDataFrame(
        {
            "geometry": [Point(0, 0), Point(10, 0), Point(20, 0)],
            "node_type": "building",
        },
        index=[1, 2, 3],
        crs="EPSG:4326",
    )

    streets = gpd.GeoDataFrame(
        {
            "geometry": [Point(0, 1), Point(10, 1), Point(20, 1)],
            "node_type": "street",
        },
        index=[101, 102, 103],
        crs="EPSG:4326",
    )

    nodes_dict = {"building": buildings, "street": streets}

    b_s_edges = gpd.GeoDataFrame(
        {
            "weight": [1.0, 1.0, 1.0],
            "edge_type": "access",
            "geometry": [
                LineString([(0, 0), (0, 1)]),
                LineString([(10, 0), (10, 1)]),
                LineString([(20, 0), (20, 1)]),
            ],
        },
        index=pd.MultiIndex.from_tuples([(1, 101), (2, 102), (3, 103)]),
        crs="EPSG:4326",
    )

    s_b_edges = gpd.GeoDataFrame(
        {
            "weight": [1.0, 1.0, 1.0],
            "edge_type": "access",
            "geometry": [
                LineString([(0, 1), (0, 0)]),
                LineString([(10, 1), (10, 0)]),
                LineString([(20, 1), (20, 0)]),
            ],
        },
        index=pd.MultiIndex.from_tuples([(101, 1), (102, 2), (103, 3)]),
        crs="EPSG:4326",
    )

    s_s_edges = gpd.GeoDataFrame(
        {
            "weight": [10.0, 10.0],
            "edge_type": "road",
            "geometry": [
                LineString([(0, 1), (10, 1)]),
                LineString([(10, 1), (20, 1)]),
            ],
        },
        index=pd.MultiIndex.from_tuples([(101, 102), (102, 103)]),
        crs="EPSG:4326",
    )

    s_s_edges_rev = gpd.GeoDataFrame(
        {
            "weight": [10.0, 10.0],
            "edge_type": "road",
            "geometry": [
                LineString([(10, 1), (0, 1)]),
                LineString([(20, 1), (10, 1)]),
            ],
        },
        index=pd.MultiIndex.from_tuples([(102, 101), (103, 102)]),
        crs="EPSG:4326",
    )

    edges_dict = {
        ("building", "access", "street"): b_s_edges,
        ("street", "access", "building"): s_b_edges,
        ("street", "road", "street"): s_s_edges,
        ("street", "road_rev", "street"): s_s_edges_rev,
    }

    return nodes_dict, edges_dict


class TestMetapaths:
    """Test metapath addition functionality."""

    @pytest.mark.parametrize("agg_mode", ["sum", "mean", "callable", "callable_all_nan"])
    def test_add_metapaths_basic_aggregations(
        self,
        agg_mode: str,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Exercise aggregation choices: sum, mean, callable, and all-NaN callable path."""
        travel_time_values = {
            ("building", "connects_to", "road"): [10.0, 20.0, 30.0],
            ("road", "links_to", "road"): [5.0, 15.0],
        }
        edges_with_attr = {k: v.copy() for k, v in sample_hetero_edges_dict.items()}
        for ek, vals in travel_time_values.items():
            if ek in edges_with_attr:
                edges_with_attr[ek]["travel_time"] = vals

        if agg_mode == "sum":
            nodes_out, edges_out = add_metapaths(
                (sample_hetero_nodes_dict, edges_with_attr),
                sequence=METAPATH,
                edge_attr="travel_time",
                edge_attr_agg="sum",
            )
            assert nodes_out is sample_hetero_nodes_dict
            result = edges_out[RESULT_KEY]
            assert "weight" in result.columns
            assert pd.api.types.is_integer_dtype(result["weight"].dtype)
            assert "travel_time" in result.columns
            assert "geometry" in result.columns
        elif agg_mode == "mean":
            _, edges_out = add_metapaths(
                (sample_hetero_nodes_dict, edges_with_attr),
                sequence=METAPATH,
                edge_attr="travel_time",
                edge_attr_agg="mean",
            )
            assert (edges_out[RESULT_KEY]["travel_time"] > 0).all()
        elif agg_mode == "callable":
            _, edges_out = add_metapaths(
                (sample_hetero_nodes_dict, edges_with_attr),
                sequence=METAPATH,
                edge_attr="travel_time",
                edge_attr_agg=np.nanmax,
            )
            assert all(
                isinstance(v, float) for v in edges_out[RESULT_KEY]["travel_time"].to_numpy()
            )
        else:
            # callable_all_nan
            edges_nan_attr = {k: v.copy() for k, v in sample_hetero_edges_dict.items()}
            for ek in list(edges_nan_attr):
                edges_nan_attr[ek]["travel_time"] = np.nan
            _, edges_out = add_metapaths(
                (sample_hetero_nodes_dict, edges_nan_attr),
                sequence=METAPATH,
                edge_attr="travel_time",
                edge_attr_agg=np.nanmax,
            )
            result = edges_out[RESULT_KEY]
            if not result.empty:
                assert result["travel_time"].isna().all()

    def test_add_metapaths_edge_attr_list(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Passing ``edge_attr`` as a list should aggregate each requested column."""
        edges_with_attr = {k: v.copy() for k, v in sample_hetero_edges_dict.items()}
        for gdf in edges_with_attr.values():
            gdf["travel_time"] = 1.0

        _, edges_out = add_metapaths(
            (sample_hetero_nodes_dict, edges_with_attr),
            sequence=METAPATH,
            edge_attr=["travel_time"],
            edge_attr_agg="sum",
        )

        result = edges_out[RESULT_KEY]
        assert "travel_time" in result.columns
        if not result.empty:
            assert (result["travel_time"] > 0).all()

    @pytest.mark.parametrize("mode", ["as_nx", "networkx_input", "metadata_merge"])
    def test_add_metapaths_return_formats(
        self,
        mode: str,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cover return as NetworkX, accepting NX input, and metadata merge behavior."""
        edges_with_attr = {k: v.copy() for k, v in sample_hetero_edges_dict.items()}
        for ek in list(edges_with_attr):
            edges_with_attr[ek]["travel_time"] = 1.0

        if mode == "as_nx":
            g = add_metapaths(
                (sample_hetero_nodes_dict, edges_with_attr),
                sequence=METAPATH,
                edge_attr="travel_time",
                edge_attr_agg="sum",
                trace_path=True,
                as_nx=True,
                multigraph=True,
            )
            assert isinstance(g, nx.MultiGraph)
            assert RESULT_KEY in g.graph.get("metapath_dict", {})
        elif mode == "networkx_input":
            hetero_graph = gdf_to_nx(
                nodes=sample_hetero_nodes_dict, edges=sample_hetero_edges_dict, multigraph=True
            )
            _, edges_out = add_metapaths(hetero_graph, sequence=METAPATH)
            assert RESULT_KEY in edges_out
        else:

            def fake_gdf_to_nx(*_a: object, **_k: object) -> nx.Graph:
                g = nx.MultiGraph()
                g.graph["metapath_dict"] = {"legacy": {"tag": 1}}
                return g

            monkeypatch.setattr("city2graph.metapath.gdf_to_nx", fake_gdf_to_nx)
            g = add_metapaths(
                (sample_hetero_nodes_dict, sample_hetero_edges_dict),
                sequence=METAPATH,
                as_nx=True,
                multigraph=True,
            )
            assert isinstance(g, nx.MultiGraph)
            assert "legacy" in g.graph["metapath_dict"]

    def test_add_metapaths_index_name_fallback(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Unnamed hop indices should fall back to sensible identifiers."""
        edges_no_names = {k: v.copy() for k, v in sample_hetero_edges_dict.items()}
        for gdf in edges_no_names.values():
            gdf.index = gdf.index.set_names([None, None])

        _, edges_out = add_metapaths(
            (sample_hetero_nodes_dict, edges_no_names),
            sequence=METAPATH,
        )

        result = edges_out[RESULT_KEY]
        assert result.index.names == ["building_id", "road_id"]

    def test_add_metapaths_nx_edges_none(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When ``nx_to_gdf`` returns ``None`` for edges, it should normalise to {}."""
        hetero_graph = gdf_to_nx(
            nodes=sample_hetero_nodes_dict,
            edges=sample_hetero_edges_dict,
            multigraph=True,
        )

        monkeypatch.setattr(
            "city2graph.metapath.nx_to_gdf",
            lambda _g: (sample_hetero_nodes_dict, None),
        )

        nodes_out, edges_out = add_metapaths(hetero_graph, sequence=METAPATH)
        assert nodes_out is sample_hetero_nodes_dict
        assert edges_out == {}

    def test_add_metapaths_nx_edges_wrong_type(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Non-dict edge data from ``nx_to_gdf`` should raise ``TypeError``."""
        hetero_graph = gdf_to_nx(
            nodes=sample_hetero_nodes_dict,
            edges=sample_hetero_edges_dict,
            multigraph=True,
        )

        monkeypatch.setattr(
            "city2graph.metapath.nx_to_gdf",
            lambda _g: (sample_hetero_nodes_dict, [1, 2, 3]),
        )

        with pytest.raises(TypeError, match="typed edges"):
            add_metapaths(hetero_graph, sequence=METAPATH)

    @pytest.mark.parametrize("early", ["empty_metapaths", "empty_edges", "raw_edges_none"])
    def test_add_metapaths_input_normalization(
        self,
        early: str,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Validate early-return normalization paths for empty inputs and None edges."""
        if early == "empty_metapaths":
            nodes_out, edges_out = add_metapaths(
                (sample_hetero_nodes_dict, sample_hetero_edges_dict), sequence=[]
            )
            assert nodes_out is sample_hetero_nodes_dict
            assert edges_out is sample_hetero_edges_dict
        elif early == "empty_edges":
            nodes_out, edges_out = add_metapaths((sample_hetero_nodes_dict, {}), sequence=METAPATH)
            assert nodes_out is sample_hetero_nodes_dict
            assert edges_out == {}
        else:
            nodes_out, edges_out = add_metapaths(
                (sample_hetero_nodes_dict, None), sequence=METAPATH
            )
            assert nodes_out is sample_hetero_nodes_dict
            assert edges_out == {}

    @pytest.mark.parametrize(
        "direction_case", ["directed_true", "reverse_lookup", "edge_type_missing_directed"]
    )
    def test_add_metapaths_edge_direction_and_lookup(
        self,
        direction_case: str,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Check directed lookup, reverse fallback, and directed missing-edge error."""
        if direction_case == "directed_true":
            _, edges_out = add_metapaths(
                (sample_hetero_nodes_dict, sample_hetero_edges_dict),
                sequence=METAPATH,
                directed=True,
            )
            assert RESULT_KEY in edges_out
        elif direction_case == "reverse_lookup":
            mp_rev = [("road", "connects_to", "building"), ("building", "connects_to", "road")]
            _, edges_rev = add_metapaths(
                (sample_hetero_nodes_dict, sample_hetero_edges_dict),
                sequence=mp_rev,
                directed=False,
            )
            assert ("road", "metapath_0", "road") in edges_rev
        else:
            bad_mp = [("x", "y", "z"), ("z", "y", "x")]
            with pytest.raises(KeyError, match=r"Edge type .* not found"):
                add_metapaths(
                    (sample_hetero_nodes_dict, sample_hetero_edges_dict),
                    sequence=bad_mp,
                    directed=True,
                )

    def test_add_metapaths_undirected_deduplicates_mirrored_paths(self) -> None:
        """Undirected metapaths should not materialize mirrored terminal pairs twice."""
        buildings = gpd.GeoDataFrame(
            {
                "geometry": [Point(0, 0), Point(10, 0)],
                "node_type": "building",
            },
            index=[1, 2],
            crs="EPSG:4326",
        )
        streets = gpd.GeoDataFrame(
            {
                "geometry": [Point(0, 1), Point(10, 1)],
                "node_type": "street",
            },
            index=[101, 102],
            crs="EPSG:4326",
        )
        nodes = {"building": buildings, "street": streets}
        edges = {
            ("building", "access", "street"): gpd.GeoDataFrame(
                {
                    "edge_type": "access",
                    "geometry": [
                        LineString([(0, 0), (0, 1)]),
                        LineString([(10, 0), (10, 1)]),
                    ],
                },
                index=pd.MultiIndex.from_tuples([(1, 101), (2, 102)]),
                crs="EPSG:4326",
            ),
            ("street", "access", "building"): gpd.GeoDataFrame(
                {
                    "edge_type": "access",
                    "geometry": [
                        LineString([(0, 1), (0, 0)]),
                        LineString([(10, 1), (10, 0)]),
                    ],
                },
                index=pd.MultiIndex.from_tuples([(101, 1), (102, 2)]),
                crs="EPSG:4326",
            ),
            ("street", "road", "street"): gpd.GeoDataFrame(
                {
                    "edge_type": "road",
                    "geometry": [
                        LineString([(0, 1), (10, 1)]),
                        LineString([(10, 1), (0, 1)]),
                    ],
                },
                index=pd.MultiIndex.from_tuples([(101, 102), (102, 101)]),
                crs="EPSG:4326",
            ),
        }
        sequence = [
            ("building", "access", "street"),
            ("street", "road", "street"),
            ("street", "access", "building"),
        ]
        relation = ("building", "metapath_0", "building")

        _, undirected_edges = add_metapaths((nodes, edges), sequence=sequence, directed=False)
        undirected_pairs = set(undirected_edges[relation].index.tolist())
        assert undirected_pairs == {(1, 2)}
        assert undirected_edges[relation].loc[(1, 2), "weight"] == 1

        _, directed_edges = add_metapaths((nodes, edges), sequence=sequence, directed=True)
        directed_pairs = set(directed_edges[relation].index.tolist())
        assert directed_pairs == {(1, 2), (2, 1)}

    @pytest.mark.parametrize(
        "join_case", ["empty_hop", "disjoint", "nan_sources", "index_normalization"]
    )
    def test_add_metapaths_join_and_index_cases(
        self,
        join_case: str,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Cover empty hop, disjoint joins, NaN sources, and index name normalization."""
        if join_case == "empty_hop":
            edges_empty = {k: v.copy() for k, v in sample_hetero_edges_dict.items()}
            edges_empty[("road", "links_to", "road")] = (
                edges_empty[("road", "links_to", "road")].head(0).copy()
            )
            nodes = {"road": sample_hetero_nodes_dict["road"]}
            _, edges_out = add_metapaths((nodes, edges_empty), sequence=METAPATH)
            res = edges_out[RESULT_KEY]
            assert res.empty
            assert res.crs == sample_hetero_nodes_dict["road"].crs
        elif join_case == "disjoint":
            edges_disjoint = {k: v.copy() for k, v in sample_hetero_edges_dict.items()}
            rl = edges_disjoint[("road", "links_to", "road")].copy()
            rl.index = pd.MultiIndex.from_tuples(
                [("r10", "r11"), ("r11", "r10")], names=rl.index.names
            )
            edges_disjoint[("road", "links_to", "road")] = rl
            _, edges_out = add_metapaths(
                (sample_hetero_nodes_dict, edges_disjoint), sequence=METAPATH
            )
            assert edges_out[RESULT_KEY].empty
        elif join_case == "nan_sources":
            edges_nan = {k: v.copy() for k, v in sample_hetero_edges_dict.items()}
            con = edges_nan[("building", "connects_to", "road")].copy()
            con.index = pd.MultiIndex.from_arrays(
                [np.array([np.nan, np.nan, np.nan]), con.index.get_level_values(1)],
                names=con.index.names,
            )
            edges_nan[("building", "connects_to", "road")] = con
            _, edges_out = add_metapaths((sample_hetero_nodes_dict, edges_nan), sequence=METAPATH)
            assert edges_out[RESULT_KEY].empty
        else:
            edges_named = {k: v.copy() for k, v in sample_hetero_edges_dict.items()}
            e0 = edges_named[("building", "connects_to", "road")]
            e1 = edges_named[("road", "links_to", "road")]
            e0.index = pd.MultiIndex.from_tuples(list(e0.index), names=[1, 2])
            e1.index = pd.MultiIndex.from_tuples(list(e1.index), names=[3, 4])
            edges_named[("building", "connects_to", "road")] = e0
            edges_named[("road", "links_to", "road")] = e1
            _, edges_norm = add_metapaths(
                (sample_hetero_nodes_dict, edges_named), sequence=METAPATH
            )
            res_norm = edges_norm[RESULT_KEY]
            assert all(isinstance(n, str) for n in res_norm.index.names)

    @pytest.mark.parametrize("geom_case", ["geometry_fallback", "safe_linestring_error"])
    def test_add_metapaths_geometry_edge_cases(
        self,
        geom_case: str,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Exercise geometry fallbacks and exception-safe linestring creation."""
        if geom_case == "geometry_fallback":
            nodes_geom = {k: v.copy() for k, v in sample_hetero_nodes_dict.items()}
            nodes_geom["building"].loc["b1", "geometry"] = None
            nodes_geom["building"].loc["b2", "geometry"] = Point()
            nodes_geom["road"].loc["r1", "geometry"] = Polygon(
                [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
            )
            _, edges_out = add_metapaths((nodes_geom, sample_hetero_edges_dict), sequence=METAPATH)
            result = edges_out[RESULT_KEY].sort_index()
            assert result["geometry"].isna().any() or result.empty
        else:
            bad_nodes = {k: v.copy() for k, v in sample_hetero_nodes_dict.items()}
            b_first = bad_nodes["building"].index[0]
            bad_nodes["building"].loc[b_first, "geometry"] = Polygon(
                [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
            )
            _, edges_badgeom = add_metapaths(
                (bad_nodes, sample_hetero_edges_dict), sequence=METAPATH
            )
            res_badgeom = edges_badgeom[RESULT_KEY]
            assert res_badgeom["geometry"].isna().any() or res_badgeom.empty

    # --- Error cases ---
    def test_add_metapaths_tuple_length_error(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Tuple with invalid length should trigger ValueError in normalization."""
        with pytest.raises(ValueError, match="Graph tuple must contain"):
            add_metapaths(
                (sample_hetero_nodes_dict, sample_hetero_edges_dict, {}), sequence=METAPATH
            )

    def test_add_metapaths_nodes_dict_type_error(
        self,
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Non-dict nodes argument should raise TypeError."""
        with pytest.raises(TypeError, match="nodes_dict must be a dictionary"):
            add_metapaths(([1, 2, 3], sample_hetero_edges_dict), sequence=METAPATH)

    def test_add_metapaths_edges_dict_type_error(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Non-dict edges argument should raise TypeError."""
        with pytest.raises(TypeError, match="edges_dict must be a dictionary"):
            add_metapaths((sample_hetero_nodes_dict, [1, 2, 3]), sequence=METAPATH)

    def test_add_metapaths_nx_nodes_not_dict(self) -> None:
        """Homogeneous NetworkX graph should fail due to missing typed nodes."""
        g = nx.Graph()
        g.add_node(1, pos=(0.0, 0.0))
        g.add_node(2, pos=(1.0, 1.0))
        g.add_edge(1, 2)
        g.graph["is_hetero"] = False
        g.graph["crs"] = "EPSG:4326"
        with pytest.raises(TypeError, match="requires a heterogeneous graph with typed nodes"):
            add_metapaths(g, sequence=METAPATH)

    def test_add_metapaths_unsupported_input_type(self) -> None:
        """Unsupported input type should raise TypeError in _ensure_hetero_dict."""
        with pytest.raises(TypeError, match="Unsupported graph input type"):
            add_metapaths(12345, sequence=METAPATH)

    def test_add_metapaths_neither_graph_nor_nodes(self) -> None:
        """Test add_metapaths raises error when neither graph nor nodes provided."""
        with pytest.raises(ValueError, match="Either 'graph' or 'nodes'"):
            add_metapaths(sequence=METAPATH)

    def test_add_metapaths_sequence_none(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Test add_metapaths raises ValueError when sequence is None."""
        with pytest.raises(ValueError, match="sequence must be provided"):
            add_metapaths((sample_hetero_nodes_dict, sample_hetero_edges_dict), sequence=None)

    def test_add_metapaths_short_metapath_error(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Metapath shorter than two hops should raise ValueError."""
        with pytest.raises(ValueError, match="at least two edge types"):
            add_metapaths(
                (sample_hetero_nodes_dict, sample_hetero_edges_dict), sequence=[("a", "b", "c")]
            )

    def test_add_metapaths_invalid_index_error(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Edge frame without MultiIndex must raise ValueError in materialization."""
        edges_bad_index = {k: v.copy() for k, v in sample_hetero_edges_dict.items()}

        e0 = edges_bad_index[("building", "connects_to", "road")].reset_index(drop=True)
        edges_bad_index[("building", "connects_to", "road")] = e0
        with pytest.raises(ValueError, match="must have a two-level MultiIndex"):
            add_metapaths((sample_hetero_nodes_dict, edges_bad_index), sequence=METAPATH)

    def test_add_metapaths_by_weight_basic(self, sample_weight_graph_data: WeightGraphData) -> None:
        """Connect buildings within a threshold and materialize geometry."""
        nodes_dict, edges_dict = sample_weight_graph_data
        nodes_out, edges_out = add_metapaths_by_weight(
            (nodes_dict, edges_dict),
            endpoint_type="building",
            weight="weight",
            threshold=15.0,
            directed=True,
        )

        relation = ("building", "connected_within_0.0_15.0", "building")
        assert nodes_out is nodes_dict
        assert relation in edges_out

        new_edges = edges_out[relation]
        pairs = set(new_edges.index.tolist())
        assert pairs == {(1, 2), (2, 1), (2, 3), (3, 2)}
        assert new_edges.loc[(1, 2), "weight"] == pytest.approx(12.0)
        geom = new_edges.loc[(1, 2), "geometry"]
        assert isinstance(geom, LineString)
        assert list(geom.coords) == [(0.0, 0.0), (10.0, 0.0)]

    def test_add_metapaths_by_weight_undirected_deduplicates_pairs(
        self,
        sample_weight_graph_data: WeightGraphData,
    ) -> None:
        """Undirected weighted metapaths should emit each endpoint pair once."""
        nodes_dict, edges_dict = sample_weight_graph_data

        _, edges_out = add_metapaths_by_weight(
            (nodes_dict, edges_dict),
            endpoint_type="building",
            weight="weight",
            threshold=15.0,
            directed=False,
        )

        relation = ("building", "connected_within_0.0_15.0", "building")
        assert relation in edges_out

        new_edges = edges_out[relation]
        pairs = set(new_edges.index.tolist())
        assert pairs == {(1, 2), (2, 3)}
        assert (2, 1) not in pairs
        assert (3, 2) not in pairs
        assert new_edges.loc[(1, 2), "weight"] == pytest.approx(12.0)
        assert new_edges.loc[(2, 3), "weight"] == pytest.approx(12.0)

    def test_add_metapaths_by_weight_threshold_controls(
        self,
        sample_weight_graph_data: WeightGraphData,
    ) -> None:
        """Threshold and min_threshold bounds should gate which pairs are emitted."""
        nodes_dict, edges_dict = sample_weight_graph_data

        _, edges_threshold = add_metapaths_by_weight(
            (nodes_dict, edges_dict),
            endpoint_type="building",
            weight="weight",
            threshold=12.0,
            directed=True,
        )
        rel_max = ("building", "connected_within_0.0_12.0", "building")
        assert rel_max in edges_threshold
        assert (1, 2) in set(edges_threshold[rel_max].index.tolist())

        _, edges_filtered = add_metapaths_by_weight(
            (nodes_dict, edges_dict),
            endpoint_type="building",
            weight="weight",
            threshold=20.0,
            min_threshold=13.0,
            directed=True,
        )
        rel_min = ("building", "connected_within_13.0_20.0", "building")
        if rel_min in edges_filtered:
            assert edges_filtered[rel_min].empty

    def test_add_metapaths_by_weight_edge_types_and_custom_name(
        self,
        sample_weight_graph_data: WeightGraphData,
    ) -> None:
        """Ensure edge filters and friendly relation labels behave as expected."""
        nodes_dict, edges_dict = sample_weight_graph_data

        _, edges_custom = add_metapaths_by_weight(
            (nodes_dict, edges_dict),
            endpoint_type="building",
            weight="weight",
            threshold=15.0,
            new_relation_name="accessible",
        )
        custom_rel = ("building", "accessible", "building")
        assert custom_rel in edges_custom

        _, edges_filtered = add_metapaths_by_weight(
            (nodes_dict, edges_dict),
            endpoint_type="building",
            weight="weight",
            threshold=100.0,
            edge_types=[
                ("building", "access", "street"),
                ("street", "access", "building"),
            ],
        )
        filtered_rel = ("building", "connected_within_0.0_100.0", "building")
        assert filtered_rel not in edges_filtered

    def test_add_metapaths_by_weight_networkx_io(
        self, sample_weight_graph_data: WeightGraphData
    ) -> None:
        """Cover NetworkX round-trips both as input and output."""
        nodes_dict, edges_dict = sample_weight_graph_data

        nx_result = add_metapaths_by_weight(
            (nodes_dict, edges_dict),
            endpoint_type="building",
            weight="weight",
            threshold=15.0,
            as_nx=True,
        )
        assert isinstance(nx_result, nx.Graph)
        assert any(
            data.get("edge_type") == ("building", "connected_within_0.0_15.0", "building")
            for _, _, data in nx_result.edges(data=True)
        )

        hetero_graph = gdf_to_nx(nodes=nodes_dict, edges=edges_dict)
        nx_roundtrip = add_metapaths_by_weight(
            hetero_graph,
            endpoint_type="building",
            weight="weight",
            threshold=15.0,
            as_nx=True,
        )
        assert isinstance(nx_roundtrip, nx.Graph)
        assert any(
            data.get("edge_type") == ("building", "connected_within_0.0_15.0", "building")
            for _, _, data in nx_roundtrip.edges(data=True)
        )

    def test_add_metapaths_by_weight_networkx_multigraph_undirected_deduplicates(
        self,
        sample_weight_graph_data: WeightGraphData,
    ) -> None:
        """Undirected MultiGraph output should not include mirrored metapath duplicates."""
        nodes_dict, edges_dict = sample_weight_graph_data
        nx_multigraph = gdf_to_nx(nodes=nodes_dict, edges=edges_dict, multigraph=True)

        nx_result = add_metapaths_by_weight(
            nx_multigraph,
            endpoint_type="building",
            weight="weight",
            threshold=15.0,
            directed=False,
            as_nx=True,
            multigraph=True,
        )

        assert isinstance(nx_result, nx.MultiGraph)
        metapath_edges = [
            (u, v, data)
            for u, v, _, data in nx_result.edges(data=True, keys=True)
            if data.get("edge_type") == ("building", "connected_within_0.0_15.0", "building")
        ]
        endpoint_pairs = {
            tuple(
                sorted(
                    (
                        nx_result.nodes[u].get("_original_index", u),
                        nx_result.nodes[v].get("_original_index", v),
                    )
                )
            )
            for u, v, _ in metapath_edges
        }

        assert endpoint_pairs == {(1, 2), (2, 3)}
        assert len(metapath_edges) == 2

    def test_add_metapaths_by_weight_missing_endpoint(
        self,
        sample_weight_graph_data: WeightGraphData,
    ) -> None:
        """Unknown endpoint types should short-circuit and return originals."""
        nodes_dict, edges_dict = sample_weight_graph_data

        nodes_out, edges_out = add_metapaths_by_weight(
            (nodes_dict, edges_dict),
            endpoint_type="nonexistent",
            weight="weight",
            threshold=15.0,
        )

        assert nodes_out is nodes_dict
        assert len(edges_out) == len(edges_dict)

    def test_add_metapaths_missing_edge_attr_error(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Missing edge attribute at hop level should raise KeyError."""
        with pytest.raises(KeyError, match=r"Edge attribute\(s\)"):
            add_metapaths(
                (sample_hetero_nodes_dict, sample_hetero_edges_dict),
                sequence=METAPATH,
                edge_attr="missing_attr",
            )

    def test_add_metapaths_missing_join_attr_error(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Edge attribute missing in some steps should raise KeyError in join reduction."""
        edges_partial_attr = {k: v.copy() for k, v in sample_hetero_edges_dict.items()}
        edges_partial_attr[("building", "connects_to", "road")]["travel_time"] = [1.0] * len(
            edges_partial_attr[("building", "connects_to", "road")]
        )
        with pytest.raises(KeyError, match="missing in metapath steps"):
            add_metapaths(
                (sample_hetero_nodes_dict, edges_partial_attr),
                sequence=METAPATH,
                edge_attr="travel_time",
            )

    def test_add_metapaths_invalid_edge_attr_agg_string(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Unsupported string for edge_attr_agg must raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported edge_attr_agg"):
            add_metapaths(
                (sample_hetero_nodes_dict, sample_hetero_edges_dict),
                sequence=METAPATH,
                edge_attr_agg="median",
            )

    def test_add_metapaths_invalid_edge_attr_agg_type(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Non-string, non-callable edge_attr_agg must raise TypeError."""
        with pytest.raises(TypeError, match="edge_attr_agg must be"):
            add_metapaths(
                (sample_hetero_nodes_dict, sample_hetero_edges_dict),
                sequence=METAPATH,
                edge_attr_agg=123,
            )

    def test_add_metapaths_attach_geometry_missing_nodes(
        self,
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Missing node GeoDataFrame for start or end types should raise KeyError."""
        nodes_missing = {k: v for k, v in sample_hetero_nodes_dict.items() if k != "road"}
        with pytest.raises(KeyError, match="Missing node GeoDataFrame"):
            add_metapaths((nodes_missing, sample_hetero_edges_dict), sequence=METAPATH)

    def test_add_metapaths_by_weight_value_errors(
        self, sample_weight_graph_data: WeightGraphData
    ) -> None:
        """Test ValueError cases for add_metapaths_by_weight."""
        nodes_dict, edges_dict = sample_weight_graph_data

        # Missing weight parameter
        with pytest.raises(ValueError, match="weight must be provided"):
            add_metapaths_by_weight(
                (nodes_dict, edges_dict),
                endpoint_type="building",
                threshold=15.0,
            )

        # Missing threshold parameter
        with pytest.raises(ValueError, match="threshold must be provided"):
            add_metapaths_by_weight(
                (nodes_dict, edges_dict),
                endpoint_type="building",
                weight="weight",
            )

        # Both graph and nodes provided
        with pytest.raises(ValueError, match="Cannot provide both"):
            add_metapaths_by_weight(
                (nodes_dict, edges_dict),
                nodes=nodes_dict,
                endpoint_type="building",
                weight="weight",
                threshold=15.0,
            )

        # Neither graph nor nodes provided
        with pytest.raises(ValueError, match="Either 'graph' or 'nodes'"):
            add_metapaths_by_weight(
                endpoint_type="building",
                weight="weight",
                threshold=15.0,
            )

    def test_add_metapaths_by_weight_endpoint_type_none(
        self, sample_weight_graph_data: WeightGraphData
    ) -> None:
        """Test that endpoint_type=None returns original graph with warning."""
        nodes_dict, edges_dict = sample_weight_graph_data

        # endpoint_type is None - should return original graph
        nodes_out, edges_out = add_metapaths_by_weight(
            (nodes_dict, edges_dict),
            endpoint_type=None,
            weight="weight",
            threshold=15.0,
        )
        assert nodes_out is nodes_dict
        assert edges_out is edges_dict

    def test_add_metapaths_by_weight_empty_data(
        self, sample_weight_graph_data: WeightGraphData
    ) -> None:
        """Test add_metapaths_by_weight with no valid edges (empty data list)."""
        nodes_dict, edges_dict = sample_weight_graph_data

        # Filter out all edges by specifying non-matching edge types
        nodes_out, edges_out = add_metapaths_by_weight(
            (nodes_dict, edges_dict),
            endpoint_type="building",
            weight="weight",
            threshold=15.0,
            edge_types=[("nonexistent", "nonexistent", "nonexistent")],
        )
        # Should return original graph since no data for sparse matrix
        assert nodes_out is nodes_dict
        assert edges_out is edges_dict

    def test_add_metapaths_by_weight_multigraph_input(
        self, sample_weight_graph_data: WeightGraphData
    ) -> None:
        """Test add_metapaths_by_weight with MultiGraph input."""
        nodes_dict, edges_dict = sample_weight_graph_data

        # Convert to MultiGraph
        nx_multigraph = gdf_to_nx(nodes=nodes_dict, edges=edges_dict, multigraph=True)

        result = add_metapaths_by_weight(
            nx_multigraph,
            endpoint_type="building",
            weight="weight",
            threshold=15.0,
            as_nx=True,
            multigraph=True,
        )
        assert isinstance(result, (nx.Graph, nx.MultiGraph))

    def test_add_metapaths_by_weight_as_nx_from_tuple(
        self, sample_weight_graph_data: WeightGraphData
    ) -> None:
        """Test add_metapaths_by_weight with as_nx=True from tuple input."""
        nodes_dict, edges_dict = sample_weight_graph_data

        result = add_metapaths_by_weight(
            (nodes_dict, edges_dict),
            endpoint_type="building",
            weight="weight",
            threshold=15.0,
            as_nx=True,
            multigraph=False,
        )
        assert isinstance(result, nx.Graph)

    def test_add_metapaths_with_graph_and_nodes_error(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Test add_metapaths raises error when both graph and nodes provided."""
        with pytest.raises(ValueError, match="Cannot provide both"):
            add_metapaths(
                graph=(sample_hetero_nodes_dict, sample_hetero_edges_dict),
                nodes=sample_hetero_nodes_dict,
                sequence=METAPATH,
            )

    def test_add_metapaths_with_nodes_only(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Test add_metapaths with nodes parameter only (no graph)."""
        _, edges_out = add_metapaths(
            nodes=sample_hetero_nodes_dict,
            edges=sample_hetero_edges_dict,
            sequence=METAPATH,
        )
        assert RESULT_KEY in edges_out

    def test_add_metapaths_with_new_relation_name(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Test add_metapaths with custom new_relation_name."""
        _, edges_out = add_metapaths(
            (sample_hetero_nodes_dict, sample_hetero_edges_dict),
            sequence=METAPATH,
            new_relation_name="custom_path",
        )
        custom_key = ("building", "custom_path", "road")
        assert custom_key in edges_out

    def test_add_metapaths_trace_path_logging(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Test add_metapaths with trace_path=True triggers debug logging."""
        # trace_path=True should just log and continue without error
        _, edges_out = add_metapaths(
            (sample_hetero_nodes_dict, sample_hetero_edges_dict),
            sequence=METAPATH,
            trace_path=True,
        )
        assert RESULT_KEY in edges_out

    def test_add_metapaths_by_weight_nodes_param(
        self, sample_weight_graph_data: WeightGraphData
    ) -> None:
        """Test add_metapaths_by_weight using nodes/edges params instead of graph."""
        nodes_dict, edges_dict = sample_weight_graph_data

        # Use nodes parameter instead of graph
        nodes_out, edges_out = add_metapaths_by_weight(
            nodes=nodes_dict,
            edges=edges_dict,
            endpoint_type="building",
            weight="weight",
            threshold=15.0,
        )

        relation = ("building", "connected_within_0.0_15.0", "building")
        assert nodes_out is nodes_dict
        assert relation in edges_out

    def test_add_metapaths_by_weight_edge_type_filter_nx(
        self, sample_weight_graph_data: WeightGraphData
    ) -> None:
        """Test add_metapaths_by_weight edge_types filter with NetworkX input."""
        nodes_dict, edges_dict = sample_weight_graph_data

        # Convert to NetworkX (non-multigraph to hit the else branch in line 2716)
        nx_graph = gdf_to_nx(nodes=nodes_dict, edges=edges_dict, multigraph=False)

        # Filter to only use specific edge types
        result = add_metapaths_by_weight(
            nx_graph,
            endpoint_type="building",
            weight="weight",
            threshold=15.0,
            edge_types=[("building", "access", "street"), ("street", "access", "building")],
            as_nx=True,
        )
        assert isinstance(result, nx.Graph)
