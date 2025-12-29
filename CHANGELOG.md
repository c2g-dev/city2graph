# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## 0.2.1 (2025-12-29)

### Added
- Added `node_geom_col` and `set_point_nodes` to `contiguity_graph` and `group_nodes` in `proximity.py` to specify the geometry column for nodes

### Changed
- Bumped `actions/setup-python` from 5 to 6
- Bumped `actions/checkout` from 4 to 6
- Bumped `actions/cache` from 4 to 5
- Bumped `astral-sh/setup-uv` from 5 to 7
- Separated metapath-related functions (e.g., `add_metapath`, `add_metapaths_by_weight`) from `graph.py` to a new module `metapath.py` for better code organization in https://github.com/c2g-dev/city2graph/pull/96

### Fixed
- Fixed module imports in tests to align with the new `metapath.py` structure
- Fixed linting errors and minor bugs

### Documentation
- Updated documentation with introduction for each module with table of available public APIs

## 0.2.0 (2025-12-10)

### Added
- Added `rustworkx` support for enhanced performance in graph operations
- Added `add_metapaths_by_weight` for weighted metapath addition per edge type
- Added `plot_graph` utility for unified graph visualization
- Added `keep_geom` parameter to graph conversion functions to choose whether to preserve geometries or not
- Added `source_node_types` and `target_node_types` parameters to `bridge_nodes` in `proximity.py`

### Changed
- Enhanced `create_isochrones` to support heterogeneous graphs with common weights.
- Refactored `proximity.py` to support `network_weight` for distance calculations
- Refactored `morphology.py` to include `segments_to_graph` migration
- Refactored `utils.py` for better code organization
- Moved core classes to `base.py` for improved package structure

### Fixed
- Fixed GitHub Actions workflow for documentation deployment
- Fixed `plot_graph` return types to optionally return axes or ndarray
- Fixed connector processing logic for Overture Maps to handle list attributes correctly
- Fixed type errors and implementation issues in `graph.py`

### Documentation
- Migrated documentation system from Sphinx to MkDocs
- Updated docstrings to support TeX formulas
- Added comprehensive description of available Overture Maps types


## 0.1.7 (2025-11-06)

### Added
- Added `cu130` for PyTorch support with CUDA 13.0

### Changed
- Updated minimum version requirement for `overturemaps` and `geopandas` as `>=0.17.0` and `>=1.1.1`, respectively
- Updated API parameters for `load_overture_data()`

### Fixed
NA

### Documentation
- Updated documentation version to 0.1.7


## 0.1.6 (2025-09-22)

### Added
- Added `add_metapath` by @yu-ta-sato in https://github.com/c2g-dev/city2graph/pull/43
- Added `set_missing_pos_from` with default of `("x", "y")` in `nx_to_gdf` in https://github.com/c2g-dev/city2graph/pull/43


### Changed
- Refactored test codes and adjusted sources by @yu-ta-sato in https://github.com/c2g-dev/city2graph/pull/44

### Fixed
- Set None as default for `edge_id_col` in `dual_graph` in https://github.com/c2g-dev/city2graph/pull/43

### Documentation
- Added examples of `add_metapaths` in https://city2graph.net/examples/adding_metapaths.ipynb


## 0.1.5 (2025-09-19)

### Added
- Added `contiguity_graph`
- Added `group_nodes`

### Changed
- Improved computation efficiency in `_add_edges`

### Fixed
- Fixed the issue [#30](https://github.com/c2g-dev/city2graph/issues/30)
- Fixed the issue [#31](https://github.com/c2g-dev/city2graph/issues/31)

### Documentation
- Added examples of `contiguity_graph` and `group_nodes` in https://city2graph.net/examples/generating_graphs_by_proximity.ipynb



## 0.1.4 (2025-09-16)

### Added
- Added `od_matrix_to_graph`

### Changed
- N/A

### Fixed
- N/A

### Documentation
- Added examples of `od_matrix_to_graph` in https://city2graph.net/examples/generating_graphs_from_od_matrix.ipynb

## 0.1.3 (2025-09-14)

### Added
- Added `contiguity_graph`

### Changed
- Updated dependent packages and tools

### Fixed
- Fixed issues in `_directed_graph`
  - [`#30`](https://github.com/c2g-dev/city2graph/issues/30)
  - [`#31`](https://github.com/c2g-dev/city2graph/issues/31)

### Documentation
- Added examples of `contiguity_graph` in https://city2graph.net/examples/generating_graphs_by_proximity.html

## 0.1.2 (2025-07-17)

### Added
- GitHub issue templates for bug reports and feature requests.
- Pull request template for better contribution workflow.
- Enhanced test coverage with improved test codes across all modules.
- New example notebooks in documentation including morphological graph examples.

### Changed
- Updated `morphological_graph()` function to accept MultiGraph inputs (e.g., from OSMnx) with bug fix.
- Enhanced `utils.py` module with improved compliance and functionality.
- Updated PyTorch dependencies to support newer CUDA versions (cu126, cu128).
- Improved documentation structure and content across multiple files.
- Updated uv dependency management configuration.

### Fixed
- Fixed edge index data types in `public_to_public_graph()` function.
- Fixed HTML title in documentation.
- Fixed CUDA version examples in documentation.
- Updated pre-commit configuration for better code quality.

### Documentation
- Added new badges and improved documentation presentation.
- Enhanced installation instructions with clearer CUDA support information.
- Updated example notebooks with more comprehensive demonstrations.
- Improved API documentation and descriptions.

## 0.1.1 (2025-07-12)

### Added
- Added conda-forge support.
- Added DOI badge and citation file reference for easier academic referencing.
- Improved documentation in `docs/source/index.rst` with clearer citation instructions and BibTeX example.

### Changed
- Minor formatting and content updates in documentation for clarity.

## 0.1.0 (2025-07-10)

### Changes

#### Core Features
- **Data Loading Module (`city2graph.data`)**: Comprehensive functionality for loading and processing geospatial data from various sources
  - Support for Overture Maps data integration
  - Data validation and coordinate reference system management
  - Geometric processing operations for urban network analysis
  - `load_overture_data()` and `process_overture_segments()` functions

- **Graph Conversion Module (`city2graph.graph`)**: Convert between GeoDataFrames and PyTorch Geometric objects
  - Seamless integration with Graph Neural Networks (GNNs)
  - Support for heterogeneous graph structures
  - PyTorch tensor conversion for machine learning workflows

- **Morphological Analysis Module (`city2graph.morphology`)**: Create morphological graphs from urban data
  - Private-to-private adjacency relationships between building tessellations
  - Public-to-public topological connectivity between street segments
  - Private-to-public interface relationships between private and public spaces
  - `morphological_graph()`, `private_to_private_graph()`, `private_to_public_graph()`, and `public_to_public_graph()` functions

- **Proximity Networks Module (`city2graph.proximity`)**: Generate graph networks based on spatial proximity relationships
  - Multiple proximity models (Euclidean, Manhattan, network-based distances)
  - Support for Delaunay triangulation, k-nearest neighbors, and radius-based networks
  - `bridge_nodes()` and other proximity-based graph generation functions

- **Transportation Networks Module (`city2graph.transportation`)**: Process GTFS data and create transportation networks
  - General Transit Feed Specification (GTFS) data processing
  - Public transit network representations
  - Origin-destination pair analysis
  - `get_od_pairs()`, `load_gtfs()`, and `travel_summary_graph()` functions

- **Utility Functions Module (`city2graph.utils`)**: Core utilities for graph conversion and validation
  - Graph conversion between different formats (NetworkX, GeoDataFrames, PyTorch Geometric)
  - Tessellation creation and dual graph operations
  - Distance filtering and validation utilities

#### Installation Options
- **Multiple PyTorch Installation Variants**: Support for different hardware configurations
  - Basic installation without PyTorch: `pip install city2graph`
  - CPU version: `pip install "city2graph[cpu]"`
  - CUDA support: `pip install "city2graph[cu118]"`, `pip install "city2graph[cu124]"`, `pip install "city2graph[cu126]"`, `pip install "city2graph[cu128]"`

#### Development Environment
- **Development Setup**: Comprehensive development environment using `uv`
  - Development dependencies including IPython, Jupyter, pytest, and testing tools
  - Jupyter kernel integration for interactive development
  - Pre-commit hooks and code formatting tools (isort, ruff)

- **Docker Support**: Complete Docker Compose setup
  - Jupyter notebook server with all dependencies pre-installed
  - GPU support when available
  - Mounted volumes for data and notebooks

#### Documentation and Examples
- **Comprehensive Documentation**: Detailed documentation available at https://city2graph.net
- **Example Notebooks**: Development notebook (`dev/dev.ipynb`) for testing and examples
- **API Documentation**: Complete docstring coverage for all public functions

#### Testing and Quality Assurance
- **Test Suite**: Comprehensive test coverage with pytest
  - Unit tests for all modules: `test_data.py`, `test_graph.py`, `test_morphology.py`, `test_proximity.py`, `test_transportation.py`, `test_utils.py`
  - Test data and utilities in `tests/data/` and `tests/utils/`
  - Code coverage reporting with codecov integration

- **Code Quality**:
  - Ruff linting and formatting
  - Type hints and static analysis
  - BSD-3-Clause license compliance

#### Dependencies
- **Core Dependencies**:
  - NetworkX ≥2.8 (graph operations)
  - OSMnx ≥2.0.3 (OpenStreetMap integration)
  - Shapely ≥2.1.0 (geometric operations)
  - GeoPandas >0.12.0 (geospatial data handling)
  - libpysal ≥4.12.1 (spatial analysis)
  - momepy (morphological analysis)
  - overturemaps (Overture Maps data)

- **Optional Dependencies**:
  - PyTorch ≥2.6.0 (machine learning backend)
  - PyTorch Geometric ≥2.6.1 (graph neural networks)
  - TorchVision ≥0.21.0 (computer vision utilities)

#### Platform Support
- **Python Version**: Requires Python ≥3.11, <4.0
- **Operating Systems**: macOS, Linux, Windows
- **Architecture**: CPU and GPU (CUDA) support

### Technical Details

#### Graph Types Supported
- **Morphological Graphs**: Buildings, streets, and land use relationships
- **Transportation Graphs**: Public transport networks (buses, trams, trains)
- **Proximity Graphs**: Spatial contiguity and distance-based relationships
- **Mobility Graphs**: Bike-sharing, migration, and pedestrian flow networks

#### Data Sources Integration
- **Overture Maps**: Direct integration with Overture Maps data
- **GTFS**: General Transit Feed Specification for public transport
- **OpenStreetMap**: Via OSMnx integration
- **Custom Geospatial Data**: Support for any GeoDataFrame input

#### Machine Learning Integration
- **PyTorch Geometric**: Native support for graph neural networks
- **Tensor Conversion**: Automatic conversion of geospatial data to PyTorch tensors
- **Heterogeneous Graphs**: Support for multi-type node and edge graphs

### Repository Structure
- **Main Package**: `city2graph/` - Core library modules
- **Tests**: `tests/` - Comprehensive test suite
- **Documentation**: `docs/` - Sphinx documentation source
- **Examples**: `dev/` - Development notebooks and examples
- **Docker**: `Dockerfile` and `docker-compose.yml` for containerized development

### Links
- **Documentation**: https://city2graph.net
- **PyPI Package**: https://pypi.org/project/city2graph/
- **GitHub Repository**: https://github.com/c2g-dev/city2graph
- **License**: BSD-3-Clause

---
