"""Tests for :mod:`city2graph.utils.plotting`."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from city2graph import utils
from city2graph.utils import plotting as plotting_utils

if TYPE_CHECKING:
    from collections.abc import Iterator

    import networkx as nx

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None  # type: ignore[assignment]
    MATPLOTLIB_AVAILABLE = False


@pytest.fixture(autouse=True)
def close_figures() -> Iterator[None]:
    """Release figures even when a plotting assertion fails."""
    yield
    if plt is not None:
        plt.close("all")


class TestPlotGraph:
    """Tests for ``plot_graph``."""

    def test_plot_graph_no_matplotlib(self, sample_nx_graph: nx.Graph) -> None:
        """Test error when matplotlib is missing."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(plotting_utils, "MATPLOTLIB_AVAILABLE", False)
            with pytest.raises(ImportError, match=r"(?i)matplotlib is required"):
                utils.plot_graph(sample_nx_graph)

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_plot_graph_homogeneous(self, sample_nx_graph: nx.Graph) -> None:
        """Test plotting a homogeneous graph."""
        # Mock plt.show to avoid display
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("matplotlib.pyplot.show", lambda: None)
            ax = utils.plot_graph(sample_nx_graph)
        assert isinstance(ax, Axes)
        assert len(ax.collections) == 2
        assert ax.get_aspect() == 1

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    @pytest.mark.parametrize("as_graph", [True, False])
    def test_plot_each_edge_type_on_its_own_axis(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
        as_graph: bool,
    ) -> None:
        """NetworkX and GeoDataFrame inputs both draw the two relation types."""
        if as_graph:
            graph = utils.gdf_to_nx(
                sample_hetero_nodes_dict, sample_hetero_edges_dict, multigraph=True
            )
            axes = utils.plot_graph(graph=graph, subplots=True)
        else:
            axes = utils.plot_graph(
                nodes=sample_hetero_nodes_dict, edges=sample_hetero_edges_dict, subplots=True
            )
        assert isinstance(axes, np.ndarray)
        assert axes.size == len(sample_hetero_edges_dict)
        assert all(ax.collections for ax in axes.flat)

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    @pytest.mark.parametrize("subplots", [False, True])
    def test_plot_single_relation_graph(
        self, regular_hetero_graph: nx.Graph, subplots: bool
    ) -> None:
        """A single relationship renders its nodes and edge in either layout."""
        ax = utils.plot_graph(regular_hetero_graph, subplots=subplots)
        assert isinstance(ax, Axes)
        assert ax.collections

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_plot_graph_no_input(self) -> None:
        """Test error when no input is provided."""
        with pytest.raises(ValueError, match="At least one of graph, nodes, or edges"):
            utils.plot_graph()

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_plot_graph_with_gdf_input(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test plotting with GeoDataFrame inputs directly."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("matplotlib.pyplot.show", lambda: None)
            utils.plot_graph(nodes=sample_nodes_gdf, edges=sample_edges_gdf)

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_plot_graph_gdf_as_graph_param(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test legacy support: GeoDataFrame passed as graph parameter."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("matplotlib.pyplot.show", lambda: None)
            # This should treat the GDF as nodes (legacy support)
            utils.plot_graph(graph=sample_nodes_gdf)

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_plot_graph_unsupported_type(self) -> None:
        """Test error when unsupported type is passed as graph."""
        with pytest.raises(TypeError, match="Unsupported data type"):
            utils.plot_graph(graph="not_a_graph")

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_plot_graph_hetero_subplots_title_color_override(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Titles should be visible when a custom color is provided."""
        edge_count = max(1, len(sample_hetero_edges_dict))
        fig, axes = plt.subplots(1, edge_count)
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

        try:
            utils.plot_graph(
                nodes=sample_hetero_nodes_dict,
                edges=sample_hetero_edges_dict,
                subplots=True,
                ax=axes,
                bgcolor="white",
                title_color="black",
            )

            assert axes_flat[0].get_title() != ""
            assert axes_flat[0].title.get_color() == "black"
        finally:
            plt.close(fig)

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_plot_graph_with_ax_bgcolor(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test that bgcolor is applied when ax is provided."""
        assert plt is not None
        fig, ax = plt.subplots()
        try:
            utils.plot_graph(
                nodes=sample_nodes_gdf,
                edges=sample_edges_gdf,
                ax=ax,
                bgcolor="#123456",
            )
            # Check that the axes background color was set
            assert ax.get_facecolor()[:3] == tuple(int(c, 16) / 255 for c in ["12", "34", "56"])
        finally:
            plt.close(fig)

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    @pytest.mark.parametrize("position", ["upper left", None])
    def test_combined_heterogeneous_plot_legend(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
        position: str | None,
    ) -> None:
        """The combined plot displays a legend only when assigned a position."""
        ax = utils.plot_graph(
            nodes=sample_hetero_nodes_dict,
            edges=sample_hetero_edges_dict,
            subplots=False,
            legend_position=position,
        )
        assert isinstance(ax, Axes)
        assert ax.collections
        assert (ax.get_legend() is None) == (position is None)

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_plot_graph_style_kwargs(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test plotting with custom style kwargs."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("matplotlib.pyplot.show", lambda: None)
            utils.plot_graph(
                nodes=sample_nodes_gdf,
                edges=sample_edges_gdf,
                node_color="red",
                node_alpha=0.5,
                edge_color="blue",
                edge_linewidth=2.0,
                markersize=10.0,
            )

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_plot_graph_with_ax(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test plotting with provided axes."""
        assert plt is not None  # For type checker
        fig, ax = plt.subplots()
        try:
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr("matplotlib.pyplot.show", lambda: None)
                utils.plot_graph(nodes=sample_nodes_gdf, ax=ax)
        finally:
            plt.close(fig)

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_plot_graph_homo_with_legend_kwargs_and_title(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Homogeneous plotting should support legend kwargs and title."""
        nodes = sample_nodes_gdf.copy()
        nodes["centrality_quantile"] = range(len(nodes))

        edge_linewidth = pd.Series([1.0] * len(sample_edges_gdf), index=sample_edges_gdf.index)
        fig, ax = plt.subplots(figsize=(5, 5))
        try:
            utils.plot_graph(
                nodes=nodes,
                edges=sample_edges_gdf,
                node_color="centrality_quantile",
                edge_color="#bbbbbb",
                edge_linewidth=edge_linewidth,
                legend=True,
                legend_kwargs={"label": "Betweenness Centrality", "orientation": "horizontal"},
                title="Central London Transit Network Betweenness Centrality of Stops",
                ax=ax,
            )
            assert (
                ax.get_title() == "Central London Transit Network Betweenness Centrality of Stops"
            )
        finally:
            plt.close(fig)

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_plot_empty_gdf(self, empty_gdf: gpd.GeoDataFrame) -> None:
        """Test _plot_gdf with empty GeoDataFrame (line 3246)."""
        # This should not raise and should return early
        fig, ax = plt.subplots()
        returned = utils.plot_graph(nodes=empty_gdf, edges=empty_gdf, ax=ax)
        assert returned is ax
        assert not ax.collections
        plt.close(fig)

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_plot_with_series_color(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test plot_graph with pd.Series for color parameter (lines 3085, 3263)."""
        # Create a Series for node colors with numeric indices (0-based from conversion)
        # When nodes are converted to NetworkX, they get new integer indices
        num_nodes = len(sample_nodes_gdf)
        node_colors = pd.Series(["red"] * num_nodes, index=range(num_nodes))

        utils.plot_graph(
            graph=None,
            nodes=sample_nodes_gdf,
            edges=sample_edges_gdf,
            node_color=node_colors,
        )
        plt.close("all")

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_plot_with_column_name(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test plot_graph with column name string for color (line 3087)."""
        # Add a numeric column for coloring
        nodes_with_attr = sample_nodes_gdf.copy()
        nodes_with_attr["value"] = range(len(nodes_with_attr))

        utils.plot_graph(
            graph=None,
            nodes=nodes_with_attr,
            edges=sample_edges_gdf,
            node_color="value",
        )
        plt.close("all")

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_plot_hetero_subplots_empty_edges(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Test _plot_hetero_subplots with no edge items (line 3406)."""
        # Create edges dict with only empty GeoDataFrames
        empty_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame] = {
            ("building", "connects", "road"): gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        }

        # This should return early without creating a plot
        fig, _ax = plt.subplots()
        result = utils.plot_graph(
            nodes=sample_hetero_nodes_dict,
            edges=empty_edges_dict,
            subplots=True,
            figsize=(10, 10),
            bgcolor="white",
        )
        assert result is None
        plt.close(fig)

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_plot_graph_subplots_less_than_grid(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Test plot_graph with subplots where number of plots < grid size."""
        # Create a situation where we have 1 edge type but grid is 2 cols
        # This triggers the "hide unused axes" logic

        # Filter to just one edge type
        single_edge_type = next(iter(sample_hetero_edges_dict.keys()))
        single_edge_dict = {single_edge_type: sample_hetero_edges_dict[single_edge_type]}

        utils.plot_graph(
            nodes=sample_hetero_nodes_dict,
            edges=single_edge_dict,
            subplots=True,
            ncols=2,  # Force 2 columns for 1 item
        )
        plt.close("all")

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_plot_graph_resolves_type_specific_kwargs_for_subplots(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """plot_graph should resolve non-standard kwargs from type-keyed dictionaries."""
        edge_type = next(iter(sample_hetero_edges_dict))

        utils.plot_graph(
            nodes=sample_hetero_nodes_dict,
            edges=sample_hetero_edges_dict,
            subplots=True,
            rasterized={"building": False, "road": False, edge_type: False},
        )

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_plot_graph_hides_unused_provided_axes(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Provided axis lists should be supported and unused axes hidden."""
        edge_key, edge_gdf = next(iter(sample_hetero_edges_dict.items()))
        fig, axes = plt.subplots(1, 2)

        try:
            returned_axes = utils.plot_graph(
                nodes=sample_hetero_nodes_dict,
                edges={edge_key: edge_gdf},
                subplots=True,
                ax=axes,
            )
            assert isinstance(returned_axes, np.ndarray)
            assert list(returned_axes) == list(axes)
            assert not axes[1].get_visible()
        finally:
            plt.close(fig)

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_plot_graph_accepts_single_provided_axis_for_one_subplot(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """A single provided axis should be wrapped for subplot plotting."""
        edge_key, edge_gdf = next(iter(sample_hetero_edges_dict.items()))
        fig, ax = plt.subplots(1, 1)

        try:
            returned_ax = utils.plot_graph(
                nodes=sample_hetero_nodes_dict,
                edges={edge_key: edge_gdf},
                subplots=True,
                ax=ax,
            )
            assert returned_ax is ax
        finally:
            plt.close(fig)

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_plot_graph_accepts_axis_lists_for_subplots(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Subplot rendering should accept a plain list of axes."""
        edge_key, edge_gdf = next(iter(sample_hetero_edges_dict.items()))
        fig, axes = plt.subplots(1, 2)
        axes_list = list(axes)

        try:
            returned_axes = utils.plot_graph(
                nodes=sample_hetero_nodes_dict,
                edges={edge_key: edge_gdf},
                subplots=True,
                ax=cast("np.ndarray", axes_list),
            )
            assert returned_axes == axes_list
            assert not axes_list[1].get_visible()
        finally:
            plt.close(fig)
