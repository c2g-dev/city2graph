"""
City2Graph: A comprehensive package for constructing graphs from geospatial datasets.

This package provides tools for converting geospatial data into graph representations
suitable for network analysis and graph neural networks. It supports various graph
types including proximity-based, morphological, and transportation networks.

Notes
-----
Main modules include:
- data : Loading and processing geospatial data from sources like Overture Maps
- graph : Converting between GeoDataFrames and PyTorch Geometric objects
- morphology : Creating morphological graphs from urban data
- proximity : Generating proximity-based graph networks
- transportation : Loading GTFS/GBFS data and creating transportation networks
- utils : Core utilities for graph conversion and validation

Author: Yuta Sato
"""

# Standard library imports
import contextlib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

# Import all public APIs from submodules
from .data import *  # noqa: F403
from .graph import *  # noqa: F403
from .metapath import *  # noqa: F403

# Explicit re-export to preserve typing information for mypy on public API
from .mobility import *  # noqa: F403
from .morphology import *  # noqa: F403
from .proximity import *  # noqa: F403
from .transportation import *  # noqa: F403
from .utils import *  # noqa: F403

# Package metadata
__author__ = "Yuta Sato"

# Version handling with graceful fallback
with contextlib.suppress(PackageNotFoundError):
    __version__ = version("city2graph")
