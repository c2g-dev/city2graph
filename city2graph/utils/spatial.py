"""
Deprecated compatibility shim for ``city2graph.utils.spatial``.

The utilities formerly defined here now live in focused submodules under
``city2graph.utils`` (``distance``, ``isochrone``, ``plotting``, and
``tessellation``) and remain re-exported from ``city2graph.utils`` and the
top-level ``city2graph`` package.
"""

from __future__ import annotations

import warnings

from .distance import filter_graph_by_distance
from .isochrone import create_isochrone
from .plotting import MATPLOTLIB_AVAILABLE as MATPLOTLIB_AVAILABLE  # noqa: PLC0414
from .plotting import plot_graph
from .tessellation import create_tessellation

warnings.warn(
    "city2graph.utils.spatial is deprecated and will be removed in a future release; "
    "import from city2graph, city2graph.utils, or the focused submodules "
    "(distance, isochrone, plotting, tessellation) instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "create_isochrone",
    "create_tessellation",
    "filter_graph_by_distance",
    "plot_graph",
]
