"""Tests for mobility/OD matrix to graph, including undirected support."""

from typing import Any
from typing import Literal
from typing import cast

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString
from shapely.geometry import Point

from city2graph.mobility import PreparedNetwork
from city2graph.mobility import network_cost_matrix
from city2graph.mobility import network_costs
from city2graph.mobility import od_matrix_to_graph
from tests.city2graph.helpers import assert_as_nx_deprecation


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
        crs="EPSG:4326",
    )


class TestODMatrixToGraph:
    """Tests for od_matrix_to_graph function."""

    def test_geometry_creation_drops_edges_with_empty_centroids(self) -> None:
        """Keep unaffected flows when one valid zone ID has no spatial footprint."""
        zones = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(1, 0), Point()], crs="EPSG:3857")
        flows = pd.DataFrame({"source": [0, 0], "target": [1, 2], "flow": [3, 4]})
        with pytest.warns(UserWarning, match=r"Dropping 1 edges.*missing centroid"):
            nodes, edges = od_matrix_to_graph(flows, zones, weight_cols=["flow"])
        assert list(nodes.index) == [0, 1, 2]
        assert list(edges.index) == [(0, 1)]
        assert edges["weight"].tolist() == [3]

        _, nonspatial_edges = od_matrix_to_graph(
            flows, zones, weight_cols=["flow"], compute_edge_geometry=False
        )
        assert list(nonspatial_edges.index) == [(0, 1), (0, 2)]
        assert nonspatial_edges.geometry.isna().all()

    def test_adjacency_batch_boundary_and_input_ownership(self) -> None:
        """Diagonal removal and NaN handling are global across row batches."""
        n = 270
        zones = gpd.GeoDataFrame(geometry=[Point(i, 0) for i in range(n)], crs=3857)
        matrix = np.eye(n)
        matrix[0, 1] = np.nan
        matrix[256, 0] = 3
        matrix[269, 268] = 4
        original = matrix.copy()
        with pytest.warns(UserWarning, match="1 NaN"):
            _, edges = od_matrix_to_graph(matrix, zones, matrix_type="adjacency")
        assert list(edges.index) == [(256, 0), (269, 268)]
        assert edges["weight"].tolist() == [3, 4]
        np.testing.assert_array_equal(matrix, original)

    def test_edgelist_single_weight(self, od_zones_gdf: gpd.GeoDataFrame) -> None:
        """Single weight edgelist thresholding and zero filtering."""
        E = pd.DataFrame(
            {
                "source": ["A", "A", "B", "C"],
                "target": ["B", "C", "C", "A"],
                "flow": [5, 0, 2, 1],
            }
        )
        nodes, edges = od_matrix_to_graph(
            E,
            od_zones_gdf,
            zone_id_col="zone_id",
            matrix_type="edgelist",
            weight_cols=["flow"],
            threshold=1,
        )
        # zero weight and below threshold removed, expect edges >=1 excluding zero
        # edges use MultiIndex (source,target) and contain 'weight'
        assert "weight" in edges.columns
        assert getattr(edges.index, "nlevels", 1) == 2
        # nodes index aligns with original zone ids
        assert list(nodes.index) == list(od_zones_gdf["zone_id"])  # preserves order
        assert edges["weight"].to_dict() == {("A", "B"): 5, ("B", "C"): 2, ("C", "A"): 1}
        assert len(nodes) == 3

    def test_edgelist_multi_weight_threshold_col(self, od_zones_gdf: gpd.GeoDataFrame) -> None:
        """Multi-attribute edgelist with threshold_col filtering."""
        E = pd.DataFrame(
            {
                "s": ["A", "A", "B"],
                "t": ["B", "C", "C"],
                "w1": [5, 1, 0],
                "w2": [10, 0, 2],
            }
        )
        _nodes, edges = od_matrix_to_graph(
            E,
            od_zones_gdf,
            zone_id_col="zone_id",
            matrix_type="edgelist",
            source_col="s",
            target_col="t",
            weight_cols=["w1", "w2"],
            threshold_col="w1",
            threshold=2,
        )
        # threshold based on w1 keeps only first edge
        assert len(edges) == 1
        assert edges.iloc[0]["weight"] == 5
        assert "w2" in edges.columns

    def test_adjacency_dataframe(self, od_zones_gdf: gpd.GeoDataFrame) -> None:
        """Adjacency DataFrame conversion preserves weights >= threshold."""
        A = pd.DataFrame(
            [
                [0, 3, 0],
                [0, 0, 4],
                [1, 0, 0],
            ],
            index=od_zones_gdf["zone_id"],
            columns=od_zones_gdf["zone_id"],
        )
        nodes, edges = od_matrix_to_graph(
            A, od_zones_gdf, zone_id_col="zone_id", matrix_type="adjacency", threshold=1
        )
        assert len(nodes) == 3
        # edges where weight>=1
        assert edges["weight"].to_dict() == {("A", "B"): 3, ("B", "C"): 4, ("C", "A"): 1}

    def test_networkx_output(self, od_zones_gdf: gpd.GeoDataFrame) -> None:
        """NetworkX output contains expected nodes and edges."""
        A = np.array([[0, 2, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        G = cast(
            "nx.DiGraph",
            od_matrix_to_graph(
                A, od_zones_gdf, zone_id_col="zone_id", matrix_type="adjacency", as_nx=True
            ),
        )
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 2

    def test_networkx_output_without_edge_geometry(self, od_zones_gdf: gpd.GeoDataFrame) -> None:
        """Edges survive NetworkX conversion when compute_edge_geometry=False."""
        E = pd.DataFrame(
            {
                "source": ["A", "B"],
                "target": ["B", "C"],
                "flow": [5.0, 7.0],
            }
        )
        G = cast(
            "nx.DiGraph",
            od_matrix_to_graph(
                E,
                od_zones_gdf,
                zone_id_col="zone_id",
                matrix_type="edgelist",
                weight_cols=["flow"],
                compute_edge_geometry=False,
                as_nx=True,
            ),
        )
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 2
        # Endpoints map back to the zone ids and weights are preserved
        weights = {
            (G.nodes[u]["_original_index"], G.nodes[v]["_original_index"]): data["weight"]
            for u, v, data in G.edges(data=True)
        }
        assert weights == {("A", "B"): 5.0, ("B", "C"): 7.0}
        # No geometry attribute is stored on the edges
        assert all("geometry" not in data for _, _, data in G.edges(data=True))

    def test_undirected_sums_reciprocals_edgelist(self, od_zones_gdf: gpd.GeoDataFrame) -> None:
        """Undirected mode sums reciprocal edges for edgelist input."""
        E = pd.DataFrame(
            {
                "source": ["A", "B", "A", "C"],
                "target": ["B", "A", "C", "A"],
                "flow": [3, 2, 1, 4],
            }
        )
        _nodes, edges = od_matrix_to_graph(
            E,
            od_zones_gdf,
            zone_id_col="zone_id",
            matrix_type="edgelist",
            weight_cols=["flow"],
            directed=False,
        )
        # Expect two undirected edges: {A,B} with 3+2=5 and {A,C} with 1+4=5
        assert len(edges) == 2
        assert set(edges["weight"]) == {5}
        # Check canonical ordering source<=target lexicographically via MultiIndex
        for src, tgt in edges.index:
            assert str(src) <= str(tgt)

    def test_undirected_threshold_after_sum(self, od_zones_gdf: gpd.GeoDataFrame) -> None:
        """Thresholding is applied after summation in undirected mode."""
        # Pair A-B has 1 and 1 -> sum 2; threshold=2 keeps it. Pair A-C has 1 and 0 -> 1 drops.
        E = pd.DataFrame(
            {
                "source": ["A", "B", "A"],
                "target": ["B", "A", "C"],
                "flow": [1, 1, 1],
            }
        )
        _nodes, edges = od_matrix_to_graph(
            E,
            od_zones_gdf,
            zone_id_col="zone_id",
            matrix_type="edgelist",
            weight_cols=["flow"],
            directed=False,
            threshold=2,
        )
        assert len(edges) == 1
        src, tgt = edges.index[0]
        assert src == "A"
        assert tgt == "B"
        assert edges.iloc[0]["weight"] == 2

    def test_undirected_networkx_graph_output(self, od_zones_gdf: gpd.GeoDataFrame) -> None:
        """as_nx with directed=False returns an undirected Graph."""
        A = np.array([[0, 1, 0], [2, 0, 0], [0, 0, 0]], dtype=float)
        G = cast(
            "nx.Graph",
            od_matrix_to_graph(
                A,
                od_zones_gdf,
                zone_id_col="zone_id",
                matrix_type="adjacency",
                as_nx=True,
                directed=False,
            ),
        )
        # Expect 1 undirected edge between A and B
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 1

    def test_threshold_type_and_matrix_type_validation(
        self, od_zones_gdf: gpd.GeoDataFrame
    ) -> None:
        """Non-numeric threshold and invalid matrix_type should raise ValueError."""
        E = pd.DataFrame({"source": ["A"], "target": ["B"], "flow": [1]})
        # threshold must be numeric
        with pytest.raises(ValueError, match="threshold must be numeric"):
            od_matrix_to_graph(
                E,
                od_zones_gdf,
                zone_id_col="zone_id",
                matrix_type="edgelist",
                weight_cols=["flow"],
                threshold="x",  # type: ignore[arg-type]
            )

        # matrix_type must be valid
        with pytest.raises(ValueError, match="matrix_type must be"):
            od_matrix_to_graph(
                E,
                od_zones_gdf,
                zone_id_col="zone_id",
                matrix_type="oops",  # type: ignore[arg-type]
                weight_cols=["flow"],
            )

    def test_edgelist_missing_weight_cols_raises_and_multiple_weights_require_threshold_col(
        self, od_zones_gdf: gpd.GeoDataFrame
    ) -> None:
        """Edgelist without weights and multi-weight without threshold_col should fail."""
        # missing weight_cols
        E = pd.DataFrame({"source": ["A"], "target": ["B"]})
        with pytest.raises(ValueError, match="weight_cols must be provided"):
            od_matrix_to_graph(E, od_zones_gdf, zone_id_col="zone_id", matrix_type="edgelist")

        # multi-weight requires threshold_col
        E2 = pd.DataFrame({"s": ["A"], "t": ["B"], "w1": [1], "w2": [2]})
        with pytest.raises(ValueError, match="valid threshold_col"):
            od_matrix_to_graph(
                E2,
                od_zones_gdf,
                zone_id_col="zone_id",
                matrix_type="edgelist",
                source_col="s",
                target_col="t",
                weight_cols=["w1", "w2"],
            )

    def test_zone_id_col_none_uses_index_and_geometry_toggle(
        self, od_zones_gdf: gpd.GeoDataFrame
    ) -> None:
        """When zone_id_col is None, use index; geometry can be disabled."""
        zones = od_zones_gdf.set_index("zone_id")[["value", "geometry"]]
        E = pd.DataFrame({"source": ["A", "B"], "target": ["B", "A"], "flow": [1, 2]})
        nodes, edges = od_matrix_to_graph(
            E,
            zones,
            zone_id_col=None,
            matrix_type="edgelist",
            weight_cols=["flow"],
            compute_edge_geometry=False,
        )
        assert list(nodes.index) == ["A", "B", "C"]
        # geometry None when compute_edge_geometry=False
        assert edges.geometry.isna().all()

    def test_include_self_loops_adjacency_and_nan_negative(
        self, od_zones_gdf: gpd.GeoDataFrame
    ) -> None:
        """Adjacency conversion keeps self-loops when requested and handles NaN/negatives."""
        A = pd.DataFrame(
            [
                [1.0, np.nan, -1.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            index=od_zones_gdf["zone_id"],
            columns=od_zones_gdf["zone_id"],
        )
        _nodes, edges = od_matrix_to_graph(
            A,
            od_zones_gdf,
            zone_id_col="zone_id",
            matrix_type="adjacency",
            include_self_loops=True,
            threshold=0,
        )
        # expect at least (A,A) and (B,B); negative allowed but threshold=0 drops -1
        assert ("A", "A") in edges.index
        assert ("B", "B") in edges.index

    def test_align_edgelist_drops_unknown_and_no_overlap_raises(
        self, od_zones_gdf: gpd.GeoDataFrame
    ) -> None:
        """Unknown IDs are dropped; if nothing remains, ValueError is raised."""
        # One valid, one invalid edge -> should drop invalid
        E = pd.DataFrame({"source": ["A", "X"], "target": ["B", "A"], "flow": [1, 3]})
        _nodes, edges = od_matrix_to_graph(
            E,
            od_zones_gdf,
            zone_id_col="zone_id",
            matrix_type="edgelist",
            weight_cols=["flow"],
        )
        assert ("A", "B") in edges.index
        # All invalid -> raise
        E2 = pd.DataFrame({"source": ["X"], "target": ["Y"], "flow": [1]})
        with pytest.raises(ValueError, match="No overlapping zone IDs between edgelist"):
            od_matrix_to_graph(
                E2,
                od_zones_gdf,
                zone_id_col="zone_id",
                matrix_type="edgelist",
                weight_cols=["flow"],
            )

    def test_empty_after_threshold_yields_empty_edgeframe_schema(
        self, od_zones_gdf: gpd.GeoDataFrame
    ) -> None:
        """High threshold removes all edges but preserves schema columns."""
        E = pd.DataFrame({"source": ["A"], "target": ["B"], "w1": [1], "w2": [5]})
        _nodes, edges = od_matrix_to_graph(
            E,
            od_zones_gdf,
            zone_id_col="zone_id",
            matrix_type="edgelist",
            weight_cols=["w1", "w2"],
            threshold_col="w1",
            threshold=10,
        )
        # edges empty but columns present
        assert edges.empty
        assert {"weight", "w1", "w2"}.issubset(set(edges.columns))

    def test_undirected_with_self_loop_preserved(self, od_zones_gdf: gpd.GeoDataFrame) -> None:
        """In undirected mode, self-loops are preserved as-is after symmetrization."""
        E = pd.DataFrame({"source": ["A", "A"], "target": ["A", "B"], "flow": [2, 3]})
        _nodes, edges = od_matrix_to_graph(
            E,
            od_zones_gdf,
            zone_id_col="zone_id",
            matrix_type="edgelist",
            weight_cols=["flow"],
            include_self_loops=True,
            directed=False,
        )
        assert ("A", "A") in edges.index
        assert ("A", "B") in edges.index

    def test_validate_zones_gdf_errors(self, od_zones_gdf: gpd.GeoDataFrame) -> None:
        """zones_gdf validation: missing column and duplicate/NaN IDs raise."""
        E = pd.DataFrame({"source": ["A"], "target": ["B"], "flow": [1]})
        # missing zone_id_col
        with pytest.raises(ValueError, match=r"zone_id_col.*not found"):
            od_matrix_to_graph(
                E, od_zones_gdf, zone_id_col="missing", matrix_type="edgelist", weight_cols=["flow"]
            )

        # duplicate IDs
        zones_dup = od_zones_gdf.copy()
        zones_dup.loc[1, "zone_id"] = zones_dup.loc[0, "zone_id"]
        with pytest.raises(ValueError, match="zone_id_col values must be unique"):
            od_matrix_to_graph(
                E, zones_dup, zone_id_col="zone_id", matrix_type="edgelist", weight_cols=["flow"]
            )

    def test_validate_adjacency_dataframe_errors(self, od_zones_gdf: gpd.GeoDataFrame) -> None:
        """Adjacency validator raises for non-square and mismatched labels."""
        # non-square
        A_bad = pd.DataFrame([[0, 1, 0]], index=["A"], columns=["A", "B", "C"])
        with pytest.raises(ValueError, match="Adjacency DataFrame must be square"):
            od_matrix_to_graph(A_bad, od_zones_gdf, zone_id_col="zone_id", matrix_type="adjacency")

        # square but columns != index
        A = pd.DataFrame(np.zeros((3, 3)), index=["A", "B", "C"], columns=["A", "C", "B"])
        with pytest.raises(ValueError, match="index and columns must match exactly"):
            od_matrix_to_graph(A, od_zones_gdf, zone_id_col="zone_id", matrix_type="adjacency")

    def test_validate_edgelist_errors(self, od_zones_gdf: gpd.GeoDataFrame) -> None:
        """Edgelist validator: missing basic cols and weight col names."""
        # missing target column
        E_missing_target = pd.DataFrame({"source": ["A"], "flow": [1]})
        with pytest.raises(ValueError, match="missing required columns: target"):
            od_matrix_to_graph(
                E_missing_target,
                od_zones_gdf,
                zone_id_col="zone_id",
                matrix_type="edgelist",
                weight_cols=["flow"],
            )

        # weight_cols references missing column
        E_ok = pd.DataFrame({"source": ["A"], "target": ["B"], "flow": [1]})
        with pytest.raises(ValueError, match="weight_cols contain columns not present"):
            od_matrix_to_graph(
                E_ok,
                od_zones_gdf,
                zone_id_col="zone_id",
                matrix_type="edgelist",
                weight_cols=["flow", "w2"],
            )

        # threshold_col not in weight_cols
        with pytest.raises(ValueError, match="threshold_col not in weight_cols"):
            od_matrix_to_graph(
                E_ok,
                od_zones_gdf,
                zone_id_col="zone_id",
                matrix_type="edgelist",
                weight_cols=["flow"],
                threshold_col="wX",
            )

    def test_coerce_weight_columns_errors(self, od_zones_gdf: gpd.GeoDataFrame) -> None:
        """Coercion errors: all non-numeric and mixed invalid values raise."""
        # all non-numeric -> cannot coerce
        E1 = pd.DataFrame({"source": ["A"], "target": ["B"], "flow": ["x"]})
        with pytest.raises(ValueError, match="all values are non-numeric"):
            od_matrix_to_graph(
                E1,
                od_zones_gdf,
                zone_id_col="zone_id",
                matrix_type="edgelist",
                weight_cols=["flow"],
            )

        # mixed numeric and invalid -> introduces new NaNs
        E2 = pd.DataFrame({"source": ["A", "A"], "target": ["B", "B"], "flow": [1, "x"]})
        with pytest.raises(ValueError, match="contains non-numeric values"):
            od_matrix_to_graph(
                E2,
                od_zones_gdf,
                zone_id_col="zone_id",
                matrix_type="edgelist",
                weight_cols=["flow"],
            )

    def test_align_adjacency_zones_no_overlap_raises(self, od_zones_gdf: gpd.GeoDataFrame) -> None:
        """Alignment of adjacency with disjoint labels raises ValueError."""
        A = pd.DataFrame(np.eye(2), index=["X", "Y"], columns=["X", "Y"])
        with pytest.raises(ValueError, match="No overlapping zone IDs between adjacency"):
            od_matrix_to_graph(A, od_zones_gdf, zone_id_col="zone_id", matrix_type="adjacency")

    def test_missing_centroid_edges_dropped_in_geometry_creation(
        self, od_zones_gdf: gpd.GeoDataFrame
    ) -> None:
        """Edges referencing zones with missing geometry are dropped with a warning."""
        zones = od_zones_gdf.copy()
        zones.loc[1, "geometry"] = None  # make B missing
        E = pd.DataFrame({"source": ["A", "B"], "target": ["B", "A"], "flow": [1, 2]})
        _nodes, edges = od_matrix_to_graph(
            E,
            zones,
            zone_id_col="zone_id",
            matrix_type="edgelist",
            weight_cols=["flow"],
        )
        # Only A->B and B->A existed, but B centroid missing means all removed
        assert edges.empty

    # The following internal-helper tests were intentionally removed to keep
    # test coverage focused on the public API, per project guidance.

    def test_adjacency_all_zero_results_in_empty_edges(
        self, od_zones_gdf: gpd.GeoDataFrame
    ) -> None:
        """A zero adjacency produces an empty edge list."""
        A = pd.DataFrame(
            np.zeros((3, 3)), index=od_zones_gdf["zone_id"], columns=od_zones_gdf["zone_id"]
        )
        _nodes, edges = od_matrix_to_graph(
            A, od_zones_gdf, zone_id_col="zone_id", matrix_type="adjacency"
        )
        assert edges.empty

    def test_adjacency_ndarray_validator_errors(self, od_zones_gdf: gpd.GeoDataFrame) -> None:
        """Non-square and size-mismatch ndarray inputs raise ValueError."""
        # non-square
        arr_bad = np.zeros((2, 3))
        with pytest.raises(ValueError, match="Adjacency ndarray must be 2D square"):
            od_matrix_to_graph(
                arr_bad, od_zones_gdf, zone_id_col="zone_id", matrix_type="adjacency"
            )

        # square but wrong size vs zones
        arr_wrong = np.zeros((2, 2))
        with pytest.raises(ValueError, match="ndarray size must match"):
            od_matrix_to_graph(
                arr_wrong, od_zones_gdf, zone_id_col="zone_id", matrix_type="adjacency"
            )

    def test_edgelist_wrong_type_raises(self, od_zones_gdf: gpd.GeoDataFrame) -> None:
        """Passing non-DataFrame for edgelist raises TypeError."""
        with pytest.raises(TypeError, match="od_data must be a pandas DataFrame"):
            od_matrix_to_graph(
                np.array([[0, 1], [0, 0]]),
                od_zones_gdf,
                zone_id_col="zone_id",
                matrix_type="edgelist",
                weight_cols=["flow"],
            )

    def test_zones_gdf_wrong_type_raises(self) -> None:
        """zones_gdf must be a GeoDataFrame (TypeError path)."""
        zones_bad = pd.DataFrame({"zone_id": ["A"], "geometry": [(0, 0)]})
        E = pd.DataFrame({"source": ["A"], "target": ["A"], "flow": [1]})
        with pytest.raises(TypeError, match="zones_gdf must be a GeoDataFrame"):
            od_matrix_to_graph(
                E, zones_bad, zone_id_col="zone_id", matrix_type="edgelist", weight_cols=["flow"]
            )

    def test_weight_column_nan_and_negative_handling(self, od_zones_gdf: gpd.GeoDataFrame) -> None:
        """NaN values are filled with 0 (warns) and negatives retained (but dropped by threshold semantics)."""
        E = pd.DataFrame(
            {
                "source": ["A", "A", "B", "B"],
                "target": ["B", "C", "C", "A"],
                "flow": [np.nan, -1, 0.5, 0],
            }
        )
        _nodes, edges = od_matrix_to_graph(
            E, od_zones_gdf, zone_id_col="zone_id", matrix_type="edgelist", weight_cols=["flow"]
        )
        # NaN->0 then drop zeros; negative retained but threshold None drops values <=0
        assert edges["weight"].to_dict() == {("B", "C"): 0.5}

    ## Internal-helper tests removed per guidance.

    def test_align_adjacency_partial_overlap_warns_and_reindexes(
        self, od_zones_gdf: gpd.GeoDataFrame
    ) -> None:
        """Partial overlap produces warnings and returns sub-matrix aligned to common IDs."""
        # adjacency includes an extra label 'X' and misses 'C'
        A = pd.DataFrame(
            [[0, 2, 1], [0, 0, 0], [0, 0, 0]],
            index=["A", "B", "X"],
            columns=["A", "B", "X"],
        )
        _nodes, edges = od_matrix_to_graph(
            A, od_zones_gdf, zone_id_col="zone_id", matrix_type="adjacency"
        )
        # Only edges among A,B retained
        assert list(edges.index) == [("A", "B")]
        assert edges["weight"].tolist() == [2]

    def test_missing_crs_emits_warning(self, od_zones_gdf: gpd.GeoDataFrame) -> None:
        """When zones have no CRS, a warning is emitted and code continues."""
        zones = od_zones_gdf.copy()
        # Remove CRS in a version-robust way (GeoPandas requires allow_override when overriding)
        zones = zones.set_crs(None, allow_override=True)
        E = pd.DataFrame({"source": ["A"], "target": ["B"], "flow": [1]})
        nodes, _edges = od_matrix_to_graph(
            E,
            zones,
            zone_id_col="zone_id",
            matrix_type="edgelist",
            weight_cols=["flow"],
        )
        # No crash and nodes indexed by zone_id; nodes preserve all zones
        assert list(nodes.index) == list(zones["zone_id"])  # preserves order and all zones

    def test_zone_id_contains_nan_raises(self, od_zones_gdf: gpd.GeoDataFrame) -> None:
        """zones_gdf with NaN identifier should raise in validation."""
        zones = od_zones_gdf.copy()
        zones.loc[1, "zone_id"] = np.nan
        E = pd.DataFrame({"source": ["A"], "target": ["C"], "flow": [1]})
        with pytest.raises(ValueError, match="zone_id_col contains null values"):
            od_matrix_to_graph(
                E, zones, zone_id_col="zone_id", matrix_type="edgelist", weight_cols=["flow"]
            )

    def test_adjacency_wrong_type_raises(self, od_zones_gdf: gpd.GeoDataFrame) -> None:
        """Passing an unsupported type for adjacency matrix raises TypeError."""
        bad = [[0, 1, 0], [0, 0, 0], [0, 0, 0]]  # list, not DataFrame/ndarray
        bad_input: Any = bad
        with pytest.raises(TypeError, match="od_data must be a pandas DataFrame or numpy ndarray"):
            od_matrix_to_graph(
                bad_input,
                od_zones_gdf,
                zone_id_col="zone_id",
                matrix_type="adjacency",
            )

    def test_undirected_multi_weight_sums_all_columns(self, od_zones_gdf: gpd.GeoDataFrame) -> None:
        """Undirected mode sums canonical weight and additional weight columns."""
        E = pd.DataFrame(
            {
                "s": ["A", "B", "A", "C"],
                "t": ["B", "A", "C", "A"],
                "w1": [1, 2, 3, 4],
                "w2": [10, 20, 30, 40],
            }
        )
        _nodes, edges = od_matrix_to_graph(
            E,
            od_zones_gdf,
            zone_id_col="zone_id",
            matrix_type="edgelist",
            source_col="s",
            target_col="t",
            weight_cols=["w1", "w2"],
            threshold_col="w1",
            directed=False,
        )
        # two undirected edges: {A,B} sums (1+2) and (10+20); {A,C} sums (3+4) and (30+40)
        assert len(edges) == 2
        assert set(edges.columns).issuperset({"weight", "w1", "w2"})
        # Check sums
        sums = {tuple(idx): (row["w1"], row["w2"], row["weight"]) for idx, row in edges.iterrows()}
        assert sums[("A", "B")] == (3, 30, 3)
        assert sums[("A", "C")] == (7, 70, 7)

    def test_threshold_keeps_weights_equal_to_boundary(
        self, od_zones_gdf: gpd.GeoDataFrame
    ) -> None:
        """Thresholding is inclusive: weights equal to the threshold survive."""
        E = pd.DataFrame(
            {
                "source": ["A", "B"],
                "target": ["B", "C"],
                "flow": [2, 1],
            }
        )
        _nodes, edges = od_matrix_to_graph(
            E,
            od_zones_gdf,
            zone_id_col="zone_id",
            matrix_type="edgelist",
            weight_cols=["flow"],
            threshold=2,
        )
        # weight == threshold kept, weight below dropped
        assert list(edges.index) == [("A", "B")]
        assert edges.iloc[0]["weight"] == 2

    @pytest.mark.parametrize(
        ("options", "expected"),
        [
            ({}, {("A", "B"): 1}),
            ({"include_self_loops": True}, {("A", "A"): 3, ("A", "B"): 1}),
        ],
    )
    def test_directed_self_loop_weights(
        self,
        od_zones_gdf: gpd.GeoDataFrame,
        options: dict[str, Any],
        expected: dict[tuple[str, str], int],
    ) -> None:
        """The self-loop contributes its own flow only when requested."""
        flows = pd.DataFrame({"source": ["A", "A"], "target": ["A", "B"], "flow": [3, 1]})
        _, edges = od_matrix_to_graph(
            flows, od_zones_gdf, zone_id_col="zone_id", weight_cols=["flow"], **options
        )
        assert edges["weight"].to_dict() == expected

    def test_as_nx_metadata_follows_shared_conventions(
        self, od_zones_gdf: gpd.GeoDataFrame
    ) -> None:
        """as_nx output carries shared gdf_to_nx metadata and node conventions."""
        A = np.array([[0, 2, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        G = cast(
            "nx.DiGraph",
            od_matrix_to_graph(
                A, od_zones_gdf, zone_id_col="zone_id", matrix_type="adjacency", as_nx=True
            ),
        )
        assert str(G.graph["crs"]) == str(od_zones_gdf.crs)
        assert G.graph["is_hetero"] is False
        node_attrs = dict(G.nodes(data=True))
        assert {attrs["_original_index"] for attrs in node_attrs.values()} == {"A", "B", "C"}
        assert all("pos" in attrs for attrs in node_attrs.values())
        for _u, _v, attrs in G.edges(data=True):
            assert "weight" in attrs
            assert "_original_edge_index" in attrs

    def test_geographic_crs_emits_warning(self, od_zones_gdf: gpd.GeoDataFrame) -> None:
        """Zones in a geographic CRS trigger the accuracy warning."""
        E = pd.DataFrame({"source": ["A"], "target": ["B"], "flow": [1]})
        with pytest.warns(UserWarning, match="Geographic CRS detected"):
            od_matrix_to_graph(
                E,
                od_zones_gdf,
                zone_id_col="zone_id",
                matrix_type="edgelist",
                weight_cols=["flow"],
            )

    def test_adjacency_ndarray_with_zone_id_none_uses_index(
        self, od_zones_gdf: gpd.GeoDataFrame
    ) -> None:
        """NumPy adjacency with zone_id_col=None aligns rows to the zones index."""
        zones = od_zones_gdf.set_index("zone_id")[["value", "geometry"]]
        arr = np.array([[0, 5, 0], [0, 0, 2], [0, 0, 0]], dtype=float)
        nodes, edges = od_matrix_to_graph(arr, zones, matrix_type="adjacency")
        assert list(nodes.index) == ["A", "B", "C"]
        assert list(edges.index) == [("A", "B"), ("B", "C")]
        assert list(edges["weight"]) == [5.0, 2.0]

    def test_empty_zones_gdf_raises(self, od_zones_gdf: gpd.GeoDataFrame) -> None:
        """An empty zones GeoDataFrame leaves no overlapping zone IDs and raises."""
        empty_zones = od_zones_gdf.iloc[0:0]
        E = pd.DataFrame({"source": ["A"], "target": ["B"], "flow": [1]})
        with pytest.raises(ValueError, match="No overlapping zone IDs"):
            od_matrix_to_graph(
                E,
                empty_zones,
                zone_id_col="zone_id",
                matrix_type="edgelist",
                weight_cols=["flow"],
            )

    @pytest.mark.parametrize("directed", [True, False])
    @pytest.mark.parametrize("multiple", [True, False])
    @pytest.mark.parametrize("threshold", [None, 100.0])
    def test_canonical_weight_column(
        self,
        od_zones_gdf: gpd.GeoDataFrame,
        directed: bool,
        multiple: bool,
        threshold: float | None,
    ) -> None:
        """An input named weight yields a unique canonical output column."""
        data = pd.DataFrame(
            {"source": ["A", "B"], "target": ["B", "A"], "weight": [2.0, 3.0], "flow": [4.0, 6.0]}
        )
        _, edges = od_matrix_to_graph(
            data,
            od_zones_gdf,
            zone_id_col="zone_id",
            weight_cols=["weight", "flow"] if multiple else ["weight"],
            threshold_col="weight",
            directed=directed,
            threshold=threshold,
        )
        assert edges.columns.is_unique
        assert edges["weight"].tolist() == ([] if threshold else [2.0, 3.0] if directed else [5.0])
        if multiple:
            assert edges["flow"].tolist() == (
                [] if threshold else [4.0, 6.0] if directed else [10.0]
            )

    def test_conflicting_canonical_weight_is_rejected(self, od_zones_gdf: gpd.GeoDataFrame) -> None:
        """A different primary weight must not overwrite a requested weight column."""
        data = pd.DataFrame({"source": ["A"], "target": ["B"], "weight": [2.0], "flow": [4.0]})
        with pytest.raises(ValueError, match="rename the input weight column"):
            od_matrix_to_graph(
                data,
                od_zones_gdf,
                zone_id_col="zone_id",
                weight_cols=["weight", "flow"],
                threshold_col="flow",
            )
        assert data["weight"].tolist() == [2.0]

    def test_undirected_adjacency_unsorted_labels_nan_and_self_loops(self) -> None:
        """Undirected adjacency handles unsorted labels, NaNs, negatives and self-loops."""
        ids = ["C", "A", "B"]
        zones = gpd.GeoDataFrame(
            {"zone_id": ids},
            geometry=[Point(0, 0), Point(1, 0), Point(2, 0)],
            crs="EPSG:3857",
        )
        matrix = pd.DataFrame(
            [
                [2.0, 10.0, np.nan],
                [-4.0, 0.0, 5.0],
                [3.0, -5.0, 0.0],
            ],
            index=ids,
            columns=ids,
        )
        original = matrix.copy()

        with pytest.warns(UserWarning, match=r"1 NaN|negative weights"):
            _, edges = od_matrix_to_graph(
                matrix,
                zones,
                zone_id_col="zone_id",
                matrix_type="adjacency",
                directed=False,
                include_self_loops=True,
            )

        assert edges["weight"].to_dict() == {
            ("C", "C"): 2.0,
            ("A", "C"): 6.0,
            ("B", "C"): 3.0,
        }
        pd.testing.assert_frame_equal(matrix, original)

    def test_undirected_adjacency_explicit_zero_cutoff_retains_zero_pairs(
        self, od_zones_gdf: gpd.GeoDataFrame
    ) -> None:
        """Undirected adjacency with threshold=0 keeps all non-negative unordered pairs."""
        ids = ["A", "B", "C"]
        matrix = pd.DataFrame([[0, 10, 0], [-4, 0, 0], [0, 0, 0]], index=ids, columns=ids)

        with pytest.warns(UserWarning, match="negative weights"):
            _, edges = od_matrix_to_graph(
                matrix,
                od_zones_gdf,
                zone_id_col="zone_id",
                matrix_type="adjacency",
                directed=False,
                threshold=0.0,
            )

        assert edges["weight"].to_dict() == {("A", "B"): 6.0, ("A", "C"): 0.0, ("B", "C"): 0.0}

    def test_undirected_adjacency_batch_boundary_sums_reciprocals(self) -> None:
        """Reciprocal entries across row-batch boundaries are summed before thresholding."""
        n = 270
        zones = gpd.GeoDataFrame(geometry=[Point(i, 0) for i in range(n)], crs="EPSG:3857")
        matrix = np.zeros((n, n), dtype=float)
        matrix[0, 260] = 10.0
        matrix[260, 0] = -4.0
        matrix[268, 269] = 2.0
        matrix[269, 268] = -2.0

        with pytest.warns(UserWarning, match="negative weights"):
            _, edges = od_matrix_to_graph(
                matrix,
                zones,
                matrix_type="adjacency",
                directed=False,
                compute_edge_geometry=False,
            )
        with pytest.warns(UserWarning, match="negative weights"):
            _, empty_edges = od_matrix_to_graph(
                matrix,
                zones,
                matrix_type="adjacency",
                directed=False,
                threshold=8.0,
                compute_edge_geometry=False,
            )

        assert edges["weight"].to_dict() == {(0, 260): 6.0}
        assert empty_edges.empty

    def test_undirected_edgelist_all_zero_and_mixed_dtype_columns(self) -> None:
        """Zero-only edgelists return empty schemas and distinct column dtypes normalize."""
        zones = gpd.GeoDataFrame(
            {"zone_id": [1, "1", 2]},
            geometry=[Point(0, 0), Point(1, 0), Point(2, 0)],
            crs="EPSG:3857",
        )
        zero_flows = pd.DataFrame({"source": [1, 2], "target": [2, 1], "flow": [0, 0]})
        _, empty_edges = od_matrix_to_graph(
            zero_flows,
            zones,
            zone_id_col="zone_id",
            weight_cols=["flow"],
            directed=False,
        )
        assert empty_edges.empty
        assert {"weight", "flow"}.issubset(empty_edges.columns)

        mixed_col_dtypes = pd.DataFrame({"source": [1], "target": ["1"], "flow": [5]})
        _, mixed_edges = od_matrix_to_graph(
            mixed_col_dtypes,
            zones,
            zone_id_col="zone_id",
            weight_cols=["flow"],
            directed=False,
        )
        assert mixed_edges["weight"].tolist() == [5]
        assert set(mixed_edges.index[0]) == {1, "1"}

    def test_undirected_od_merges_reciprocal_mixed_type_zone_ids(self) -> None:
        """Distinct IDs can have the same string representation."""
        zones = gpd.GeoDataFrame(
            {"zone_id": [1, "1"]}, geometry=[Point(0, 0), Point(1, 0)], crs=3857
        )
        data = pd.DataFrame({"source": [1, "1"], "target": ["1", 1], "flow": [2, 3]})
        _, edges = od_matrix_to_graph(
            data,
            zones,
            zone_id_col="zone_id",
            weight_cols=["flow"],
            directed=False,
        )

        assert edges["weight"].tolist() == [5]
        assert set(edges.index[0]) == {1, "1"}

    @pytest.mark.parametrize("matrix_type", ["edgelist", "adjacency"])
    @pytest.mark.parametrize(("threshold", "expected"), [(None, [6.0]), (8.0, [])])
    def test_undirected_od_threshold_is_applied_after_signed_reciprocal_sum(
        self,
        od_zones_gdf: gpd.GeoDataFrame,
        matrix_type: Literal["edgelist", "adjacency"],
        threshold: float | None,
        expected: list[float],
    ) -> None:
        """A correction in the reverse direction contributes before thresholding."""
        if matrix_type == "edgelist":
            data = pd.DataFrame({"source": ["A", "B"], "target": ["B", "A"], "flow": [10, -4]})
        else:
            ids = ["A", "B", "C"]
            data = pd.DataFrame([[0, 10, 0], [-4, 0, 0], [0, 0, 0]], columns=ids, index=ids)

        with pytest.warns(UserWarning, match="negative weights"):
            _, edges = od_matrix_to_graph(
                data,
                od_zones_gdf,
                zone_id_col="zone_id",
                matrix_type=matrix_type,
                weight_cols=["flow"] if matrix_type == "edgelist" else None,
                threshold=threshold,
                directed=False,
            )

        assert edges["weight"].tolist() == expected

    def test_undirected_od_preserves_explicit_zero_cutoff(
        self, od_zones_gdf: gpd.GeoDataFrame
    ) -> None:
        """An explicit threshold of zero retains observed zero-flow pairs."""
        data = pd.DataFrame({"source": ["A"], "target": ["B"], "flow": [0]})
        _, edges = od_matrix_to_graph(
            data,
            od_zones_gdf,
            zone_id_col="zone_id",
            weight_cols=["flow"],
            threshold=0,
            directed=False,
        )
        assert list(edges.index) == [("A", "B")]
        assert edges["weight"].tolist() == [0]

    def test_od_as_nx_warning(self, od_zones_gdf: gpd.GeoDataFrame) -> None:
        """Explicit as_nx on the OD builder emits a caller warning."""
        flows = pd.DataFrame({"source": ["A"], "target": ["B"], "flow": [3]})
        assert_as_nx_deprecation(
            od_matrix_to_graph, flows, od_zones_gdf, zone_id_col="zone_id", weight_cols=["flow"]
        )


def points(*coords: tuple[float, float], labels: list[Any] | None = None) -> gpd.GeoSeries:
    """Build projected point inputs."""
    return gpd.GeoSeries([Point(xy) for xy in coords], index=labels, crs=3857)


@pytest.fixture
def streets() -> nx.MultiDiGraph:
    """Build an OSMnx-like graph with asymmetric parallel arcs and an isolated node."""
    graph = nx.MultiDiGraph(crs="EPSG:3857")
    for node, x in enumerate([0, 10, 20, 50]):
        graph.add_node(node, x=x, y=0)
    graph.add_edge(0, 1, time=10)
    graph.add_edge(0, 1, time=20)
    graph.add_edge(1, 0, time=30)
    graph.add_edge(1, 2, time=0)
    return graph


def test_zero_unreachable_and_labels(streets: nx.MultiDiGraph) -> None:
    """Explicit zeros survive CSR construction and duplicate labels stay distinct."""
    origins = points((10, 0), (50, 0), labels=["x", "x"])
    destinations = points((20, 0), (10, 0), labels=[9, 3])
    matrix = network_cost_matrix(origins, destinations, streets, weight="time", threshold=0)
    assert matrix.shape == (2, 2)
    assert matrix.indptr.tolist() == [0, 2, 2]
    assert matrix.data.tolist() == [0, 0]
    table = network_costs(origins, destinations, streets, weight="time", k=1, threshold=0)
    assert table.to_numpy().tolist() == [["x", 9, 0]]


def test_edge_cost_and_access(streets: nx.MultiDiGraph) -> None:
    """Reverse and forward partial traversals have different costs on the same line."""
    road = PreparedNetwork(streets, weight="time", connect_to="edge")
    origins = road.connect(points((2, 1), (8, 3), (40, 40)), max_connection_distance=3)
    table = network_costs(origins, points((5, 2)), road, k=1, include_connection_distance=True)
    np.testing.assert_allclose(table.cost, [3, 9])
    np.testing.assert_allclose(table.connection_distance, [3, 5])
    assert network_costs(origins, points((0, 0)), road, threshold=6.1).origin.tolist() == [0]


def test_within_and_one_way(streets: nx.MultiDiGraph) -> None:
    """Self exclusion is positional, and reaching the same vertex can cost zero."""
    origins = points((10, 0), (10, 0), (20, 0), (50, 0), labels=[8, 8, 4, 6])
    table = network_costs(origins, graph=streets, k=1, weight="time", threshold=0)
    assert table.to_numpy().tolist() == [[8, 8, 0], [8, 8, 0]]
    reverse = network_costs(points((18, 1)), points((12, 1)), streets, connect_to="edge")
    assert reverse.empty


def test_edge_frame_geometry_length() -> None:
    """An edge frame defaults to two-way travel and uses curved geometry length."""
    edges = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (0, 4), (3, 4)])], crs=3857)
    table = network_costs(points((3, 4)), points((0, 0)), edges)
    assert table.cost.tolist() == [7]


def test_heterogeneous_entry_layers() -> None:
    """Repeated node IDs in different layers stay distinct and transit remains traversable."""
    street = gpd.GeoDataFrame(geometry=points((0, 0), (10, 0)))
    stop = gpd.GeoDataFrame(geometry=points((1, 0), (9, 0)))

    def links(
        source: gpd.GeoDataFrame,
        target: gpd.GeoDataFrame,
        pairs: list[tuple[int, int]],
        times: list[int],
    ) -> gpd.GeoDataFrame:
        lines = [LineString([source.geometry.iloc[u], target.geometry.iloc[v]]) for u, v in pairs]
        return gpd.GeoDataFrame(
            {"time": times},
            geometry=lines,
            crs=3857,
            index=pd.MultiIndex.from_tuples(pairs, names=["u", "v"]),
        )

    edges = {
        ("street", "road", "street"): links(street, street, [(0, 1)], [15]),
        ("street", "board", "stop"): links(street, stop, [(0, 0)], [2]),
        ("stop", "ride", "stop"): links(stop, stop, [(0, 1)], [3]),
        ("stop", "alight", "street"): links(stop, street, [(1, 1)], [2]),
    }
    graph = ({"street": street, "stop": stop}, edges)
    table = network_costs(
        points((1, 0)),
        points((9, 0)),
        graph,
        weight="time",
        directed=True,
        connect_node_types="street",
        include_connection_distance=True,
    )
    assert table.cost.tolist() == [7]
    assert table.connection_distance.tolist() == [2]


@pytest.mark.parametrize("value", [-1, float("nan"), "2", None])
def test_invalid_cost(streets: nx.MultiDiGraph, value: object) -> None:
    """Validate every edge, even when another parallel edge is cheaper."""
    streets[0][1][1]["time"] = value
    with pytest.raises(ValueError, match="non-negative"):
        PreparedNetwork(streets, weight="time")


@pytest.mark.parametrize("reverse", [False, True])
def test_batched_costs_match_networkx(streets: nx.MultiDiGraph, reverse: bool) -> None:
    """Check both search directions and top-k pruning against NetworkX."""
    starts, ends = ([0, 1, 2, 3], [0, 2]) if reverse else ([0, 2], [0, 1, 2, 3])
    coords = [(0, 0), (10, 0), (20, 0), (50, 0)]
    origins = points(*(coords[i] for i in starts))
    destinations = points(*(coords[j] for j in ends))
    table = network_costs(origins, destinations, streets, weight="time", threshold=30, k=2)
    expected = []
    for row, source in enumerate(starts):
        lengths = nx.single_source_dijkstra_path_length(streets, source, cutoff=30, weight="time")
        ranked = sorted((lengths[node], col) for col, node in enumerate(ends) if node in lengths)
        expected.extend((row, col, cost) for cost, col in ranked[:2])
    assert list(table.itertuples(index=False, name=None)) == expected
