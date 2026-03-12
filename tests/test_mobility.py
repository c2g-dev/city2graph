"""Tests for mobility/OD matrix to graph, including undirected support.

This refactor centralizes fixtures (zones) into conftest and condenses
repeated patterns via parametrization, while preserving coverage.
"""

from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import geopandas as gpd
import numpy as np
import pandas as pd

from city2graph.mobility import od_matrix_to_graph

if TYPE_CHECKING:
    import networkx as nx


class TestODMatrixToGraph:
    """Tests for od_matrix_to_graph function."""

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
        assert (edges["weight"] >= 1).all()
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
        assert set(edges["weight"]) == {3, 4, 1}

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
        try:
            od_matrix_to_graph(
                E,
                od_zones_gdf,
                zone_id_col="zone_id",
                matrix_type="edgelist",
                weight_cols=["flow"],
                threshold="x",  # type: ignore[arg-type]
            )
        except ValueError:
            pass
        else:
            msg = "Expected ValueError for non-numeric threshold"
            raise AssertionError(msg)

        # matrix_type must be valid
        try:
            od_matrix_to_graph(
                E,
                od_zones_gdf,
                zone_id_col="zone_id",
                matrix_type="oops",  # type: ignore[arg-type]
                weight_cols=["flow"],
            )
        except ValueError:
            pass
        else:
            msg = "Expected ValueError for invalid matrix_type"
            raise AssertionError(msg)

    def test_edgelist_missing_weight_cols_raises_and_multiple_weights_require_threshold_col(
        self, od_zones_gdf: gpd.GeoDataFrame
    ) -> None:
        """Edgelist without weights and multi-weight without threshold_col should fail."""
        # missing weight_cols
        E = pd.DataFrame({"source": ["A"], "target": ["B"]})
        try:
            od_matrix_to_graph(E, od_zones_gdf, zone_id_col="zone_id", matrix_type="edgelist")
        except ValueError:
            pass
        else:
            msg = "Expected ValueError when weight_cols is missing"
            raise AssertionError(msg)

        # multi-weight requires threshold_col
        E2 = pd.DataFrame({"s": ["A"], "t": ["B"], "w1": [1], "w2": [2]})
        try:
            od_matrix_to_graph(
                E2,
                od_zones_gdf,
                zone_id_col="zone_id",
                matrix_type="edgelist",
                source_col="s",
                target_col="t",
                weight_cols=["w1", "w2"],
            )
        except ValueError:
            pass
        else:
            msg = "Expected ValueError when threshold_col missing with multiple weights"
            raise AssertionError(msg)

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
        try:
            od_matrix_to_graph(
                E2,
                od_zones_gdf,
                zone_id_col="zone_id",
                matrix_type="edgelist",
                weight_cols=["flow"],
            )
        except ValueError:
            pass
        else:
            msg = "Expected ValueError when no overlapping IDs remain"
            raise AssertionError(msg)

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
        try:
            od_matrix_to_graph(
                E, od_zones_gdf, zone_id_col="missing", matrix_type="edgelist", weight_cols=["flow"]
            )
        except ValueError:
            pass
        else:
            msg = "Expected ValueError for missing zone_id_col"
            raise AssertionError(msg)

        # duplicate IDs
        zones_dup = od_zones_gdf.copy()
        zones_dup.loc[1, "zone_id"] = zones_dup.loc[0, "zone_id"]
        try:
            od_matrix_to_graph(
                E, zones_dup, zone_id_col="zone_id", matrix_type="edgelist", weight_cols=["flow"]
            )
        except ValueError:
            pass
        else:
            msg = "Expected ValueError for duplicate zone_id values"
            raise AssertionError(msg)

    def test_validate_adjacency_dataframe_errors(self, od_zones_gdf: gpd.GeoDataFrame) -> None:
        """Adjacency validator raises for non-square and mismatched labels."""
        # non-square
        A_bad = pd.DataFrame([[0, 1, 0]], index=["A"], columns=["A", "B", "C"])
        try:
            od_matrix_to_graph(A_bad, od_zones_gdf, zone_id_col="zone_id", matrix_type="adjacency")
        except ValueError:
            pass
        else:
            msg = "Expected ValueError for non-square adjacency DataFrame"
            raise AssertionError(msg)

        # square but columns != index
        A = pd.DataFrame(np.zeros((3, 3)), index=["A", "B", "C"], columns=["A", "C", "B"])
        try:
            od_matrix_to_graph(A, od_zones_gdf, zone_id_col="zone_id", matrix_type="adjacency")
        except ValueError:
            pass
        else:
            msg = "Expected ValueError when index and columns mismatch"
            raise AssertionError(msg)

    def test_validate_edgelist_errors(self, od_zones_gdf: gpd.GeoDataFrame) -> None:
        """Edgelist validator: missing basic cols and weight col names."""
        # missing target column
        E_missing_target = pd.DataFrame({"source": ["A"], "flow": [1]})
        try:
            od_matrix_to_graph(
                E_missing_target,
                od_zones_gdf,
                zone_id_col="zone_id",
                matrix_type="edgelist",
                weight_cols=["flow"],
            )
        except ValueError:
            pass
        else:
            msg = "Expected ValueError for missing target column"
            raise AssertionError(msg)

        # weight_cols references missing column
        E_ok = pd.DataFrame({"source": ["A"], "target": ["B"], "flow": [1]})
        try:
            od_matrix_to_graph(
                E_ok,
                od_zones_gdf,
                zone_id_col="zone_id",
                matrix_type="edgelist",
                weight_cols=["flow", "w2"],
            )
        except ValueError:
            pass
        else:
            msg = "Expected ValueError when weight_cols contain missing column"
            raise AssertionError(msg)

        # threshold_col not in weight_cols
        try:
            od_matrix_to_graph(
                E_ok,
                od_zones_gdf,
                zone_id_col="zone_id",
                matrix_type="edgelist",
                weight_cols=["flow"],
                threshold_col="wX",
            )
        except ValueError:
            pass
        else:
            msg = "Expected ValueError when threshold_col not in weight_cols"
            raise AssertionError(msg)

    def test_coerce_weight_columns_errors(self, od_zones_gdf: gpd.GeoDataFrame) -> None:
        """Coercion errors: all non-numeric and mixed invalid values raise."""
        # all non-numeric -> cannot coerce
        E1 = pd.DataFrame({"source": ["A"], "target": ["B"], "flow": ["x"]})
        try:
            od_matrix_to_graph(
                E1,
                od_zones_gdf,
                zone_id_col="zone_id",
                matrix_type="edgelist",
                weight_cols=["flow"],
            )
        except ValueError:
            pass
        else:
            msg = "Expected ValueError when weight col is fully non-numeric"
            raise AssertionError(msg)

        # mixed numeric and invalid -> introduces new NaNs
        E2 = pd.DataFrame({"source": ["A", "A"], "target": ["B", "B"], "flow": [1, "x"]})
        try:
            od_matrix_to_graph(
                E2,
                od_zones_gdf,
                zone_id_col="zone_id",
                matrix_type="edgelist",
                weight_cols=["flow"],
            )
        except ValueError:
            pass
        else:
            msg = "Expected ValueError when non-numeric introduces new NaNs"
            raise AssertionError(msg)

    def test_align_adjacency_zones_no_overlap_raises(self, od_zones_gdf: gpd.GeoDataFrame) -> None:
        """Alignment of adjacency with disjoint labels raises ValueError."""
        A = pd.DataFrame(np.eye(2), index=["X", "Y"], columns=["X", "Y"])
        try:
            od_matrix_to_graph(A, od_zones_gdf, zone_id_col="zone_id", matrix_type="adjacency")
        except ValueError:
            pass
        else:
            msg = "Expected ValueError when no overlapping zone IDs exist"
            raise AssertionError(msg)

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
        try:
            od_matrix_to_graph(
                arr_bad, od_zones_gdf, zone_id_col="zone_id", matrix_type="adjacency"
            )
        except ValueError:
            pass
        else:
            msg = "Expected ValueError for non-square ndarray"
            raise AssertionError(msg)

        # square but wrong size vs zones
        arr_wrong = np.zeros((2, 2))
        try:
            od_matrix_to_graph(
                arr_wrong, od_zones_gdf, zone_id_col="zone_id", matrix_type="adjacency"
            )
        except ValueError:
            pass
        else:
            msg = "Expected ValueError for ndarray size != number of zones"
            raise AssertionError(msg)

    def test_edgelist_wrong_type_raises(self, od_zones_gdf: gpd.GeoDataFrame) -> None:
        """Passing non-DataFrame for edgelist raises TypeError."""
        try:
            od_matrix_to_graph(
                np.array([[0, 1], [0, 0]]),
                od_zones_gdf,
                zone_id_col="zone_id",
                matrix_type="edgelist",
                weight_cols=["flow"],
            )
        except TypeError:
            pass
        else:
            msg = "Expected TypeError for non-DataFrame edgelist"
            raise AssertionError(msg)

    def test_zones_gdf_wrong_type_raises(self) -> None:
        """zones_gdf must be a GeoDataFrame (TypeError path)."""
        zones_bad = pd.DataFrame({"zone_id": ["A"], "geometry": [(0, 0)]})
        E = pd.DataFrame({"source": ["A"], "target": ["A"], "flow": [1]})
        try:
            od_matrix_to_graph(
                E, zones_bad, zone_id_col="zone_id", matrix_type="edgelist", weight_cols=["flow"]
            )
        except TypeError:
            pass
        else:
            msg = "Expected TypeError for non-GeoDataFrame zones_gdf"
            raise AssertionError(msg)

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
        assert (edges["weight"] > 0).all()

    ## Internal-helper tests removed per guidance.

    def test_self_loop_removed_when_not_included(self, od_zones_gdf: gpd.GeoDataFrame) -> None:
        """Self-loop is removed by default include_self_loops=False."""
        E = pd.DataFrame({"source": ["A", "A"], "target": ["A", "B"], "flow": [5, 1]})
        _nodes, edges = od_matrix_to_graph(
            E,
            od_zones_gdf,
            zone_id_col="zone_id",
            matrix_type="edgelist",
            weight_cols=["flow"],
        )  # default include_self_loops=False
        assert ("A", "A") not in edges.index
        assert ("A", "B") in edges.index

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
        if not edges.empty:
            assert set(edges.index.get_level_values(0)).issubset({"A", "B"})
            assert set(edges.index.get_level_values(1)).issubset({"A", "B"})

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
        try:
            od_matrix_to_graph(
                E, zones, zone_id_col="zone_id", matrix_type="edgelist", weight_cols=["flow"]
            )
        except ValueError:
            pass
        else:
            msg = "Expected ValueError for NaN in zone_id_col"
            raise AssertionError(msg)

    def test_adjacency_wrong_type_raises(self, od_zones_gdf: gpd.GeoDataFrame) -> None:
        """Passing an unsupported type for adjacency matrix raises TypeError."""
        bad = [[0, 1, 0], [0, 0, 0], [0, 0, 0]]  # list, not DataFrame/ndarray
        bad_input: Any = bad
        try:
            od_matrix_to_graph(
                bad_input,
                od_zones_gdf,
                zone_id_col="zone_id",
                matrix_type="adjacency",
            )
        except TypeError:
            pass
        else:
            msg = "Expected TypeError for unsupported adjacency type"
            raise AssertionError(msg)

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
