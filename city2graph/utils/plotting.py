"""Graph plotting utilities."""

from __future__ import annotations

import math
import typing
from typing import TYPE_CHECKING
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    MATPLOTLIB_AVAILABLE = False

if TYPE_CHECKING:
    import matplotlib.axes

from .conversion import nx_to_gdf

__all__ = ["plot_graph"]

PLOT_DEFAULTS = {
    "node_color": "#22d3ee",
    "node_edgecolor": "none",
    "node_alpha": 0.8,
    "node_zorder": 2,
    "markersize": 4.0,
    "edge_color": "#ffffff",
    "edge_linewidth": 0.5,
    "edge_alpha": 0.3,
    "edge_zorder": 1,
    "title_color": "white",
}


def plot_graph(  # noqa: PLR0913, PLR0917
    graph: nx.Graph | nx.MultiGraph | None = None,
    nodes: gpd.GeoDataFrame | dict[str, gpd.GeoDataFrame] | None = None,
    edges: gpd.GeoDataFrame | dict[tuple[str, str, str], gpd.GeoDataFrame] | None = None,
    ax: matplotlib.axes.Axes | np.ndarray | None = None,
    bgcolor: str = "#000000",
    figsize: tuple[float, float] = (12, 12),
    subplots: bool = True,
    ncols: int | None = None,
    legend_position: str | None = "upper left",
    labelcolor: str = "white",
    title_color: str | None = None,
    node_color: str | float | pd.Series | dict[str, Any] | None = None,
    node_alpha: float | pd.Series | dict[str, Any] | None = None,
    node_zorder: int | pd.Series | dict[str, Any] | None = None,
    node_edgecolor: str | pd.Series | dict[str, Any] | None = None,
    markersize: float | pd.Series | dict[str, Any] | None = None,
    edge_color: str | float | pd.Series | dict[tuple[str, str, str], Any] | None = None,
    edge_linewidth: float | pd.Series | dict[tuple[str, str, str], Any] | None = None,
    edge_alpha: float | pd.Series | dict[tuple[str, str, str], Any] | None = None,
    edge_zorder: int | pd.Series | dict[tuple[str, str, str], Any] | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> matplotlib.axes.Axes | np.ndarray | None:
    """
    Plot a graph with a unified interface.

    This function provides a unified interface for plotting spatial network data,
    supporting both GeoDataFrame-based and NetworkX-based inputs. NetworkX graphs
    are automatically converted to GeoDataFrames before plotting. It can handle
    homogeneous and heterogeneous graphs with customizable styling.

    Parameters
    ----------
    graph : networkx.Graph or networkx.MultiGraph, optional
        The NetworkX graph to plot. If provided without nodes/edges, the function
        will convert it to GeoDataFrames before plotting.
    nodes : geopandas.GeoDataFrame or dict[str, geopandas.GeoDataFrame], optional
        Nodes to plot. Can be a single GeoDataFrame (homogeneous) or a dictionary
        mapping node type names to GeoDataFrames (heterogeneous).
    edges : geopandas.GeoDataFrame or dict[tuple[str, str, str], geopandas.GeoDataFrame], optional
        Edges to plot. Can be a single GeoDataFrame (homogeneous) or a dictionary
        mapping edge type tuples (src_type, rel_type, dst_type) to GeoDataFrames (heterogeneous).
    ax : matplotlib.axes.Axes or numpy.ndarray, optional
        The axes on which to plot. If None, a new figure and axes are created.
    bgcolor : str, default "#000000"
        Background color for the plot (Black theme).
    figsize : tuple[float, float], default (12, 12)
        Figure size as (width, height) in inches.
    subplots : bool, default True
        If True and the graph is heterogeneous, plot each node/edge type in a
        separate subplot. Uses 'ax' as array of subplots if provided.
    ncols : int, optional
        Number of columns (subplots per row) when plotting heterogeneous graphs
        with subplots=True. If None, defaults to min(3, number_of_edge_types).
    legend_position : str or None, default "upper left"
        Position of the legend for heterogeneous graphs. Common values include
        "upper left", "upper right", "lower left", "lower right", "center", etc.
        If None, no legend is displayed.
    labelcolor : str, default "white"
        Color of the legend text labels.
    title_color : str, optional
        Color for subplot titles when ``subplots=True``. Falls back to a white
        title on black backgrounds if not provided.
    node_color : str, float, pd.Series, or dict, optional
        Color for nodes. Can be a scalar, column name, Series, or a dictionary
        mapping node types to colors for heterogeneous graphs.
    node_alpha : float, pd.Series, or dict, optional
        Transparency for nodes (0.0-1.0). Can be a scalar, column name, Series,
        or a dictionary mapping node types to transparency values.
    node_zorder : int, pd.Series, or dict, optional
        Drawing order for nodes. Can be a scalar, column name, Series, or a
        dictionary mapping node types to zorder values.
    node_edgecolor : str, pd.Series, or dict, optional
        Color for node borders. Can be a scalar, column name, Series, or a
        dictionary mapping node types to edge colors.
    markersize : float, pd.Series, or dict, optional
        Size of the node markers. Can be a scalar, column name, Series, or a
        dictionary mapping node types to marker sizes.
    edge_color : str, float, pd.Series, or dict, optional
        Color for edges. Can be a scalar, column name, Series, or a dictionary
        mapping edge types to colors for heterogeneous graphs.
    edge_linewidth : float, pd.Series, or dict, optional
        Line width for edges. Can be a scalar, column name, Series, or a
        dictionary mapping edge types to line widths.
    edge_alpha : float, pd.Series, or dict, optional
        Transparency for edges (0.0-1.0). Can be a scalar, column name, Series,
        or a dictionary mapping edge types to transparency values.
    edge_zorder : int, pd.Series, or dict, optional
        Drawing order for edges. Can be a scalar, column name, Series, or a
        dictionary mapping edge types to zorder values.
    **kwargs : Any
        Additional keyword arguments passed to the GeoPandas plotting functions.

        Supports attribute-based styling where parameters can be specified as:

        - **Scalar values** (str/float): Applied uniformly to all geometries
        - **Column names** (str): If the string matches a column in the GeoDataFrame,
          that column's values are used for styling
        - **pd.Series**: Direct values for each geometry

        Other common options: etc.

    Returns
    -------
    matplotlib.axes.Axes or numpy.ndarray or None
        The axes object(s) used for plotting.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If no valid input is provided (all parameters are None).
    TypeError
        If the input data types are not supported.

    Examples
    --------
    >>> import geopandas as gpd
    >>> import pandas as pd
    >>> import networkx as nx
    >>> # Plot from NetworkX graph (automatically converted to GeoDataFrames)
    >>> G = nx.Graph()
    >>> G.add_node(0, pos=(0, 0))
    >>> G.add_edge(0, 1)
    >>> ax = plot_graph(graph=G)
    >>> # Plot from GeoDataFrames with scalar styling
    >>> plot_graph(nodes=nodes_gdf, edges=edges_gdf, node_color='red')
    >>> # Plot with attribute-based node colors (by column name)
    >>> plot_graph(nodes=nodes_gdf, edges=edges_gdf, node_color='building_type')
    >>> # Plot with pd.Series for edge linewidth
    >>> edge_widths = pd.Series([1.0, 2.0, 1.5], index=edges_gdf.index)
    >>> plot_graph(nodes=nodes_gdf, edges=edges_gdf, edge_linewidth=edge_widths)
    >>> # Plot heterogeneous graph
    >>> plot_graph(nodes=nodes_dict, edges=edges_dict)
    """
    if not MATPLOTLIB_AVAILABLE:
        msg = "Matplotlib is required for plotting functionality."
        raise ImportError(msg)

    # Input validation
    if graph is None and nodes is None and edges is None:
        msg = "At least one of graph, nodes, or edges must be provided"
        raise ValueError(msg)

    # Convert NetworkX graph to GeoDataFrames if provided
    nodes, edges = _normalize_graph_input(graph, nodes, edges)

    # Collect style arguments
    style_kwargs = {
        "node_color": node_color,
        "node_alpha": node_alpha,
        "node_zorder": node_zorder,
        "node_edgecolor": node_edgecolor,
        "markersize": markersize,
        "edge_color": edge_color,
        "edge_linewidth": edge_linewidth,
        "edge_alpha": edge_alpha,
        "edge_zorder": edge_zorder,
        "legend_position": legend_position,
        "labelcolor": labelcolor,
        "title_color": title_color,
        **kwargs,
    }

    # Handle heterogeneous subplots
    is_hetero = isinstance(nodes, dict) or isinstance(edges, dict)

    if subplots and is_hetero:
        return _plot_hetero_subplots(
            nodes, edges, figsize, bgcolor, ax=ax, ncols=ncols, **style_kwargs
        )

    # Setup figure and axes
    if ax is None:
        ax = _setup_plot_axes(figsize, bgcolor)
    elif not isinstance(ax, np.ndarray):
        # Apply bgcolor to provided axes
        ax.set_facecolor(bgcolor)
        ax.set_axis_off()
        if hasattr(ax, "figure") and ax.figure is not None:
            ax.figure.patch.set_facecolor(bgcolor)

    # GeoDataFrame-based plotting
    if is_hetero:
        _plot_hetero_graph(nodes, edges, ax, **style_kwargs)
    else:
        _plot_homo_graph(nodes, edges, ax, **style_kwargs)

    return ax


def _setup_plot_axes(
    figsize: tuple[float, float],
    bgcolor: str,
) -> matplotlib.axes.Axes:
    """
    Create and return a matplotlib axes configured for C2G plots.

    Centralizes figure styling for all C2G visualizations.

    Parameters
    ----------
    figsize : tuple[float, float]
        Width and height of the figure in inches.
    bgcolor : str
        Background color for figure and axes.

    Returns
    -------
    matplotlib.axes.Axes
        Configured axes instance.
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(bgcolor)
    ax.set_facecolor(bgcolor)
    ax.set_axis_off()
    return ax


def _normalize_graph_input(
    graph: nx.Graph | nx.MultiGraph | None,
    nodes: gpd.GeoDataFrame | dict[str, gpd.GeoDataFrame] | None,
    edges: gpd.GeoDataFrame | dict[tuple[str, str, str], gpd.GeoDataFrame] | None,
) -> tuple[
    gpd.GeoDataFrame | dict[str, gpd.GeoDataFrame] | None,
    gpd.GeoDataFrame | dict[tuple[str, str, str], gpd.GeoDataFrame] | None,
]:
    """
    Normalize graph input to GeoDataFrames.

    Converts various input formats into standardized GeoDataFrames.

    Parameters
    ----------
    graph : networkx.Graph or networkx.MultiGraph, optional
        NetworkX graph to convert.
    nodes : geopandas.GeoDataFrame or dict, optional
        Existing nodes data.
    edges : geopandas.GeoDataFrame or dict, optional
        Existing edges data.

    Returns
    -------
    tuple
        A tuple of (nodes, edges) as GeoDataFrames or dictionaries.
    """
    if graph is not None and nodes is None and edges is None:
        if isinstance(graph, (nx.Graph, nx.MultiGraph)):
            nodes, edges = nx_to_gdf(graph, nodes=True, edges=True)
        elif isinstance(graph, gpd.GeoDataFrame):
            nodes = graph
        else:
            msg = f"Unsupported data type for graph parameter: {type(graph)}"
            raise TypeError(msg)
    return nodes, edges


def _resolve_plot_parameter(
    gdf: gpd.GeoDataFrame,
    param_value: str | float | pd.Series | None,
    _param_name: str,
    default_value: Any,  # noqa: ANN401
) -> str | float | np.ndarray | pd.Series:
    """
    Resolve a plot parameter to a value usable by GeoPandas plot().

    Handles None, Series, column names, and scalar values.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing potential attribute columns.
    param_value : str, float, pd.Series, or None
        The parameter value to resolve.
    _param_name : str
        Name of the parameter (unused).
    default_value : Any
        Default value if param_value is None.

    Returns
    -------
    str, float, pd.Series, or np.ndarray
        Resolved parameter value.
    """
    if param_value is None:
        return typing.cast("str | float | np.ndarray | pd.Series", default_value)
    if isinstance(param_value, pd.Series):
        return param_value.to_numpy()
    if isinstance(param_value, str) and param_value in gdf.columns:
        return typing.cast("str | float | np.ndarray | pd.Series", gdf[param_value].to_numpy())
    return typing.cast("str | float | np.ndarray | pd.Series", param_value)


def _get_color_for_type(i: int) -> str | tuple[float, float, float, float]:
    """
    Get color for a specific type index using matplotlib's tab10 colormap.

    Uses cyclic indexing to support any number of types.

    Parameters
    ----------
    i : int
        The index of the current type being colored.

    Returns
    -------
    str or tuple
        A color from the tab10 colormap.
    """
    cmap = plt.get_cmap("tab10")
    return cmap(i % 10)


def _resolve_type_parameter(
    param: Any,  # noqa: ANN401
    type_key: str | tuple[str, str, str],
) -> Any:  # noqa: ANN401
    """
    Resolve a parameter that might be a dictionary keyed by type.

    Looks up value if param is a dict, otherwise returns param as-is.

    Parameters
    ----------
    param : Any
        The parameter value (scalar or dict).
    type_key : str or tuple
        The key identifying the node or edge type.

    Returns
    -------
    Any
        The resolved parameter value for the specific type.
    """
    return param.get(type_key) if isinstance(param, dict) else param


def _get_style_param(
    global_kwargs: dict[str, Any],
    name: str,
    type_key: str | tuple[str, str, str] | None,
) -> Any:  # noqa: ANN401
    """
    Get a style parameter value with optional type-specific lookup.

    Type-specific plotting options are stored as dictionaries keyed by node or
    edge type. This helper resolves those dictionaries when a type key is
    available and otherwise returns the raw parameter value.

    Parameters
    ----------
    global_kwargs : dict[str, Any]
        The kwargs passed to the main plot_graph function.
    name : str
        Parameter name to look up.
    type_key : str or tuple or None
        The key identifying the node or edge type.

    Returns
    -------
    Any
        The parameter value, or type-specific value if ``type_key`` is set.
    """
    val = global_kwargs.get(name)
    return _resolve_type_parameter(val, type_key) if type_key else val


def _or_default(val: Any, default: Any) -> Any:  # noqa: ANN401
    """
    Return a fallback value when the resolved style value is None.

    Plotting defaults should replace missing values while preserving explicit
    falsy values such as ``0`` or ``False``.

    Parameters
    ----------
    val : Any
        Value to check.
    default : Any
        Default value to use if ``val`` is None.

    Returns
    -------
    Any
        ``val`` if not None, otherwise ``default``.
    """
    return val if val is not None else default


def _resolve_style_kwargs(
    global_kwargs: dict[str, Any],
    type_key: str | tuple[str, str, str] | None,
    is_edge: bool,
    default_color: Any = None,  # noqa: ANN401
) -> dict[str, Any]:
    """
    Resolve style arguments for a specific graph element type.

    Extracts and resolves style parameters from kwargs, handling type-specific
    overrides if a type_key is provided.

    Parameters
    ----------
    global_kwargs : dict
        The kwargs passed to the main plot_graph function.
    type_key : str or tuple or None
        The key identifying the node or edge type. None for homogeneous graphs.
    is_edge : bool
        True if resolving for edges, False for nodes.
    default_color : Any, optional
        Fallback color if not specified in kwargs.

    Returns
    -------
    dict
        Resolved style arguments ready for _plot_gdf.
    """
    # Keys reserved for C2G-specific styling logic
    c2g_keys = {
        "node_color",
        "node_alpha",
        "node_zorder",
        "node_edgecolor",
        "markersize",
        "edge_color",
        "edge_linewidth",
        "edge_alpha",
        "edge_zorder",
        "legend_position",
        "labelcolor",
        "title_color",
        "legend",
        "legend_kwargs",
        "title",
    }

    # Start with all global kwargs, resolving potential type-specific dictionaries
    resolved = {}
    for k, v in global_kwargs.items():
        if k not in c2g_keys:
            resolved[k] = _resolve_type_parameter(v, type_key) if type_key else v

    if is_edge:
        resolved.update(
            {
                "linewidth": _or_default(
                    _get_style_param(global_kwargs, "edge_linewidth", type_key),
                    PLOT_DEFAULTS["edge_linewidth"],
                ),
                "alpha": _or_default(
                    _get_style_param(global_kwargs, "edge_alpha", type_key),
                    PLOT_DEFAULTS["edge_alpha"],
                ),
                "zorder": _or_default(
                    _get_style_param(global_kwargs, "edge_zorder", type_key),
                    PLOT_DEFAULTS["edge_zorder"],
                ),
                "color": _or_default(
                    _get_style_param(global_kwargs, "edge_color", type_key),
                    default_color,
                ),
            }
        )
    else:
        resolved.update(
            {
                "color": _or_default(
                    _get_style_param(global_kwargs, "node_color", type_key),
                    default_color or PLOT_DEFAULTS["node_color"],
                ),
                "alpha": _or_default(
                    _get_style_param(global_kwargs, "node_alpha", type_key),
                    PLOT_DEFAULTS["node_alpha"],
                ),
                "zorder": _or_default(
                    _get_style_param(global_kwargs, "node_zorder", type_key),
                    PLOT_DEFAULTS["node_zorder"],
                ),
                "edgecolor": _or_default(
                    _get_style_param(global_kwargs, "node_edgecolor", type_key),
                    PLOT_DEFAULTS["node_edgecolor"],
                ),
                "markersize": _or_default(
                    _get_style_param(global_kwargs, "markersize", type_key),
                    PLOT_DEFAULTS["markersize"],
                ),
            }
        )
    return resolved


def _plot_gdf(
    gdf: gpd.GeoDataFrame,
    ax: Any,  # noqa: ANN401
    **kwargs: Any,  # noqa: ANN401
) -> None:
    """
    Plot a GeoDataFrame with resolved parameters.

    Delegates to GeoPandas plot method after resolving style parameters.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The GeoDataFrame to plot.
    ax : matplotlib.axes.Axes
        The axes to plot on.
    **kwargs : Any
        Style parameters (color, alpha, linewidth, markersize, zorder, etc.).
    """
    if gdf.empty:
        return

    # Start with all kwargs as potential plot arguments
    plot_kwargs: dict[str, Any] = kwargs.copy()
    plot_kwargs["ax"] = ax

    param_defaults = {
        "color": PLOT_DEFAULTS["node_color"],
        "alpha": PLOT_DEFAULTS["node_alpha"],
        "linewidth": PLOT_DEFAULTS["edge_linewidth"],
        "markersize": PLOT_DEFAULTS["markersize"],
        "zorder": PLOT_DEFAULTS["node_zorder"],
        "edgecolor": PLOT_DEFAULTS["edge_color"],
    }

    for param_name, default_val in param_defaults.items():
        val = _resolve_plot_parameter(gdf, kwargs.get(param_name), param_name, default_val)
        if val is not None:
            if param_name == "color" and isinstance(val, (pd.Series, np.ndarray)):
                plot_kwargs["column"] = val
                plot_kwargs.pop("color", None)
            else:
                plot_kwargs[param_name] = val

    # Handle label separately to avoid it being resolved as a column name
    if "label" in kwargs:
        plot_kwargs["label"] = kwargs["label"]

    gdf.plot(**plot_kwargs)


def _plot_hetero_graph(
    nodes: dict[str, gpd.GeoDataFrame] | None,
    edges: dict[tuple[str, str, str], gpd.GeoDataFrame] | None,
    ax: Any,  # noqa: ANN401
    **kwargs: Any,  # noqa: ANN401
) -> None:
    """
    Plot heterogeneous graph with per-type styling and legend.

    Draws edges and nodes grouped by their semantic types.

    Parameters
    ----------
    nodes : dict[str, geopandas.GeoDataFrame], optional
        Mapping of node type names to GeoDataFrames.
    edges : dict[tuple[str, str, str], geopandas.GeoDataFrame], optional
        Mapping of edge type tuples to GeoDataFrames.
    ax : matplotlib.axes.Axes
        Axes instance for rendering.
    **kwargs : Any
        Additional styling arguments.
    """
    # Plot edges first
    if edges is not None and isinstance(edges, dict):
        for i, (edge_type, edge_gdf) in enumerate(edges.items()):
            default_color = _get_color_for_type(i)
            style_kwargs = _resolve_style_kwargs(
                kwargs,
                edge_type,
                is_edge=True,
                default_color=default_color,
            )
            _plot_gdf(edge_gdf, ax, label=str(edge_type), **style_kwargs)

    # Plot nodes
    if nodes is not None and isinstance(nodes, dict):
        for i, (node_type, node_gdf) in enumerate(nodes.items()):
            default_color = _get_color_for_type(i)
            style_kwargs = _resolve_style_kwargs(
                kwargs,
                node_type,
                is_edge=False,
                default_color=default_color,
            )
            _plot_gdf(node_gdf, ax, label=node_type, **style_kwargs)

    # Add legend for heterogeneous plots with sophisticated styling
    _style_legend(ax, **kwargs)


def _style_legend(ax: Any, **kwargs: Any) -> None:  # noqa: ANN401
    """
    Apply sophisticated styling to the legend.

    This helper configures the legend appearance, including position, frame,
    label styling, and handle adjustments for better readability.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to style the legend for.
    **kwargs : Any
        Additional styling arguments.
    """
    legend_position = kwargs.get("legend_position", "upper left")
    if legend_position is not None and ax.get_legend_handles_labels()[0]:
        legend = ax.legend(
            loc=legend_position,
            frameon=False,  # Remove frame/border
            labelcolor=kwargs.get("labelcolor", "white"),
            fontsize=8,
            handlelength=1.5,  # Shorter legend handles
            handleheight=1.2,
            handletextpad=0.5,  # Reduce space between handle and text
            borderpad=0.3,
            labelspacing=0.3,  # Reduce spacing between labels
            markerscale=0.5,  # Fixed marker scale in legend
        )
        # Make legend markers non-transparent and fixed size for better recognition
        for handle in legend.legend_handles:
            handle.set_alpha(1.0)
            # Set fixed marker size for point markers (nodes)
            if hasattr(handle, "set_markersize"):
                handle.set_markersize(5)  # Fixed size for all node markers
            if hasattr(handle, "set_sizes"):
                handle.set_sizes([25])  # Fixed size for scatter plot markers
            # Increase line width for line markers (edges)
            if hasattr(handle, "set_linewidth"):
                handle.set_linewidth(2.5)


def _plot_hetero_subplots(
    nodes: dict[str, gpd.GeoDataFrame] | None,
    edges: dict[tuple[str, str, str], gpd.GeoDataFrame] | None,
    figsize: tuple[float, float],
    bgcolor: str,
    ax: matplotlib.axes.Axes | np.ndarray | None = None,
    ncols: int | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> matplotlib.axes.Axes | np.ndarray | None:
    """
    Plot heterogeneous graph components in separate subplots.

    Creates a grid layout with one subplot per edge type.

    Parameters
    ----------
    nodes : dict[str, geopandas.GeoDataFrame], optional
        Mapping of node type names to GeoDataFrames.
    edges : dict[tuple[str, str, str], geopandas.GeoDataFrame], optional
        Mapping of edge type tuples to GeoDataFrames.
    figsize : tuple[float, float]
        Figure size (width, height) in inches.
    bgcolor : str
        Background color for figure and axes.
    ax : matplotlib.axes.Axes or numpy.ndarray, optional
        Existing axes to plot on. Can be a single axis or an array of axes.
    ncols : int, optional
        Number of columns in the subplot grid. If None, defaults to
        min(3, number_of_edge_types).
    **kwargs : Any
        Additional styling arguments.

    Returns
    -------
    matplotlib.axes.Axes or numpy.ndarray or None
        The axes object(s) used for plotting.
    """
    # Collect non-empty edge types to plot
    edge_items = [(k, v) for k, v in (edges or {}).items() if not v.empty]
    n_items = len(edge_items)
    if n_items == 0:
        return None

    returned_axes: matplotlib.axes.Axes | np.ndarray | None = None

    if ax is None:
        # Calculate grid layout
        cols = ncols if ncols is not None else min(3, n_items)
        cols = max(1, min(cols, n_items))  # Ensure cols is between 1 and n_items
        rows = math.ceil(n_items / cols)

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        fig.patch.set_facecolor(bgcolor)

        returned_axes = axes

        # Ensure axes is iterable
        axes_flat = [axes] if n_items == 1 else axes.flatten()
    else:
        # Use provided axes
        returned_axes = ax
        if hasattr(ax, "flatten"):
            axes_flat = ax.flatten()
        elif isinstance(ax, (list, tuple)):
            axes_flat = list(ax)
        else:
            axes_flat = [ax]

        # Apply bgcolor to provided axes' figure if available
        if (
            len(axes_flat) > 0
            and hasattr(axes_flat[0], "figure")
            and axes_flat[0].figure is not None
        ):
            axes_flat[0].figure.patch.set_facecolor(bgcolor)

    # Calculate total bounds for fixed extent
    xlim, ylim = _calculate_total_bounds(nodes, edges)

    # Plot each edge type
    for _, (subplot_ax, (edge_key, edge_gdf)) in enumerate(
        zip(axes_flat, edge_items, strict=False)
    ):
        subplot_ax.set_facecolor(bgcolor)
        subplot_ax.set_axis_off()
        subplot_ax.set_xlim(xlim)
        subplot_ax.set_ylim(ylim)

        # Get colors for this subplot
        colors = {
            "edge": None,
            "src": None,
            "dst": None,
        }

        _plot_hetero_subplot_item(subplot_ax, edge_key, edge_gdf, nodes, colors, **kwargs)

    # Hide unused axes
    for j in range(len(edge_items), len(axes_flat)):
        axes_flat[j].set_visible(False)

    return returned_axes


def _calculate_total_bounds(
    nodes: dict[str, gpd.GeoDataFrame] | None,
    edges: dict[tuple[str, str, str], gpd.GeoDataFrame] | None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Calculate total bounds for all nodes and edges with 5% padding.

    Computes the combined bounding box of all GeoDataFrames.

    Parameters
    ----------
    nodes : dict, optional
        Dictionary of node GeoDataFrames.
    edges : dict, optional
        Dictionary of edge GeoDataFrames.

    Returns
    -------
    tuple
        A tuple of ((minx, maxx), (miny, maxy)).
    """
    bounds = [float("inf"), float("inf"), float("-inf"), float("-inf")]

    for gdf_dict in [nodes, edges]:
        if gdf_dict:
            for gdf in gdf_dict.values():
                if not gdf.empty:
                    b = gdf.total_bounds
                    bounds[0] = min(bounds[0], b[0])
                    bounds[1] = min(bounds[1], b[1])
                    bounds[2] = max(bounds[2], b[2])
                    bounds[3] = max(bounds[3], b[3])

    dx, dy = bounds[2] - bounds[0], bounds[3] - bounds[1]
    pad_x, pad_y = dx * 0.05, dy * 0.05
    return (bounds[0] - pad_x, bounds[2] + pad_x), (bounds[1] - pad_y, bounds[3] + pad_y)


def _plot_hetero_subplot_item(
    ax: Any,  # noqa: ANN401
    edge_key: tuple[str, str, str],
    edge_gdf: gpd.GeoDataFrame,
    nodes: dict[str, gpd.GeoDataFrame] | None,
    colors: dict[str, Any],
    **kwargs: Any,  # noqa: ANN401
) -> None:
    """
    Plot a single heterogeneous subplot item.

    Renders an edge type and its connected source/target nodes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    edge_key : tuple
        The edge type tuple.
    edge_gdf : geopandas.GeoDataFrame
        The edge GeoDataFrame.
    nodes : dict, optional
        Dictionary of node GeoDataFrames.
    colors : dict
        Dictionary of fallback colors for this subplot.
    **kwargs : Any
        Additional arguments.
    """
    # Plot edges with resolved styling
    style_kwargs_edge = _resolve_style_kwargs(
        kwargs,
        edge_key,
        is_edge=True,
        default_color=colors.get("edge", PLOT_DEFAULTS["edge_color"]),
    )
    # Ensure alpha is at least 0.5 for visibility in subplots if not specified
    if kwargs.get("edge_alpha") is None:
        style_kwargs_edge["alpha"] = 0.5

    _plot_gdf(edge_gdf, ax, label=str(edge_key), **style_kwargs_edge)

    # Plot connected nodes
    src_type, _, dst_type = edge_key

    if nodes and src_type in nodes and not nodes[src_type].empty:
        style_kwargs_src = _resolve_style_kwargs(
            kwargs,
            src_type,
            is_edge=False,
            default_color=colors.get("src", PLOT_DEFAULTS["node_color"]),
        )
        _plot_gdf(nodes[src_type], ax, label=src_type, **style_kwargs_src)

    if nodes and dst_type in nodes and not nodes[dst_type].empty and dst_type != src_type:
        style_kwargs_dst = _resolve_style_kwargs(
            kwargs,
            dst_type,
            is_edge=False,
            default_color=colors.get("dst", PLOT_DEFAULTS["node_color"]),
        )
        _plot_gdf(nodes[dst_type], ax, label=dst_type, **style_kwargs_dst)

    # Set title
    title_color = kwargs.get("title_color")
    ax.set_title(
        f"{edge_key}",
        color=title_color if title_color is not None else PLOT_DEFAULTS["title_color"],
        fontsize=10,
    )


def _plot_homo_graph(
    nodes: gpd.GeoDataFrame | None,
    edges: gpd.GeoDataFrame | None,
    ax: Any,  # noqa: ANN401
    **kwargs: Any,  # noqa: ANN401
) -> None:
    """
    Plot homogeneous graph (edges first, then nodes on top).

    Renders a single node/edge GeoDataFrame pair on the provided axes.

    Parameters
    ----------
    nodes : geopandas.GeoDataFrame, optional
        Node geometries to render.
    edges : geopandas.GeoDataFrame, optional
        Edge geometries to render.
    ax : matplotlib.axes.Axes
        Target axes for rendering.
    **kwargs : Any
        Additional styling arguments.
    """
    title = kwargs.get("title")
    legend = kwargs.get("legend")
    legend_kwargs = kwargs.get("legend_kwargs")
    plot_kwargs = {k: v for k, v in kwargs.items() if k not in {"title", "legend", "legend_kwargs"}}

    # Plot edges first (in background)
    if edges is not None and isinstance(edges, gpd.GeoDataFrame):
        style_kwargs = _resolve_style_kwargs(plot_kwargs, None, is_edge=True)
        _plot_gdf(edges, ax, **style_kwargs)

    # Plot nodes on top
    if nodes is not None and isinstance(nodes, gpd.GeoDataFrame):
        style_kwargs = _resolve_style_kwargs(plot_kwargs, None, is_edge=False)
        if legend is not None:
            style_kwargs["legend"] = legend
        if legend_kwargs is not None:
            style_kwargs["legend_kwds"] = legend_kwargs
        _plot_gdf(nodes, ax, **style_kwargs)

    if title is not None:
        title_color = kwargs.get("title_color")
        ax.set_title(
            title,
            color=title_color if title_color is not None else PLOT_DEFAULTS["title_color"],
        )
