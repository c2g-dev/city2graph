# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## Unreleased

### Added

### Changed

### Deprecated

### Removed

### Fixed

### Documentation


## 1.0.0 (2026-07-31)

First stable release. The public API is now considered settled and will follow
semantic versioning: breaking changes are reserved for future major releases.

### Added
- Added `morphological_graphs()`, which builds morphological graphs for several distances in one shared pass. The expensive shared work, the reachability cost field and the enclosed tessellation, is computed once from the context of the largest distance and reused across all distances, so requesting every distance costs roughly one tessellation instead of one per distance. Because the tessellation context corresponds to the largest distance, results for smaller distances can differ slightly near the clipping boundary compared with calling `morphological_graph()` once per distance.
- Added `load_gbfs()` for loading GBFS JSON feeds from a local directory into an in-memory DuckDB connection, flattening station, bike, vehicle, vehicle-type, and feed structures into tables and materialising geometry columns from `lon`/`lat` fields where available.
- Added `non_movement_barrier_col`, `tessellation_fallback`, and `tessellation_n_jobs` parameters to `morphological_graph()` (and `morphological_graphs()`): `non_movement_barrier_col` supplies barrier geometries that constrain tessellation without becoming movement nodes, `tessellation_fallback` degrades failed enclosed tessellation to building footprints instead of raising, and `tessellation_n_jobs` controls tessellation parallelism.

### Changed
- **Breaking:** Removed the deprecated morphology aliases (see Removed below).
- **Breaking:** Changed the default `extent_buffer` of `morphological_graph()` from `50.0` to `100.0`. The perpendicular access cap between a street and a building or cell is now wider by default, so the default output retains more buildings and more `faced_to` edges. Pass `extent_buffer=50.0` explicitly to reproduce the previous default.
- **Breaking:** Raised the minimum Python version to 3.12. `momepy` 1.0, now the minimum supported version, requires Python 3.12 or newer, and keeping 3.11 would have meant shipping and testing two different tessellation implementations.
- **Breaking:** Required `momepy>=1.0.0`. Only its functional API is used, and 1.0 is the version the tessellation behaviour is tested against.
- **Breaking:** Removed `osmnx` from the runtime dependencies. It is never imported by the library; the example notebooks and the test suite still use it, so it moved to the `dev` dependency group. Install it alongside city2graph if you follow those examples.
- Declared `numpy`, `pandas`, and `pyproj` explicitly. All three are imported directly by the library but were previously relied on transitively through GeoPandas.
- Updated optional graph neural network dependencies to PyTorch 2.13, torchvision 0.28, and PyTorch Geometric 2.8 for CPU, CUDA 12.6, and CUDA 13.0 installs. CUDA 12.8 remains on PyTorch 2.11 and torchvision 0.26 because PyTorch no longer publishes CUDA 12.8 wheels past 2.11.
- Split `city2graph/utils.py` into a `city2graph/utils/` package with focused submodules for graph conversion, topology, and spatial operations. The public `city2graph.utils` surface is unchanged: every previously exported name is re-exported from the package, so `from city2graph.utils import ...` and `city2graph.<name>` continue to work.
- Consolidated shared CRS, centroid, validation, and graph-index handling into `city2graph/base.py`, so the mobility, morphology, and proximity builders now apply the same rules.
- Decomposed the long graph-building orchestrators (`od_matrix_to_graph()`, `segments_to_graph()`, `place_to_place_graph()`, `bridge_nodes()`, `plot_graph()`) into private validation, transformation, assembly, and output phase helpers. No public signature, return type, warning, or error message changed.
- Reworked `travel_summary_graph()` to produce its NetworkX output through the shared `gdf_to_nx()` path, so `as_nx=True` results now carry the same metadata and CRS conventions as the rest of the library.
- Narrowed the GTFS UDF registration in `load_gtfs()` from a blanket DuckDB error suppression to an existence pre-check, so genuine registration failures now propagate instead of being swallowed.
- Improved performance across the conversion and graph-building paths without changing behaviour: `gdf_to_pyg()` and `pyg_to_gdf()` map endpoints and serialise geometries with vectorised operations instead of Python loops, the homogeneous and heterogeneous PyG pipelines share one set of node and edge helpers, metapath materialisation is deferred with vectorised reducers, the proximity builders no longer allocate dense distance matrices, Overture segment post-processing is vectorised, concave-hull and tessellation-boundary construction is faster, and per-distance invariant work is hoisted out of the morphology multi-distance loop.
- Documented the frame ownership rule for the morphology module: public input GeoDataFrames are never mutated, caller-owned frames are copied at most once at the public boundary, and the public entry points are covered by input-immutability tests.
- Restricted the source distribution to the package, tests, and project metadata. Previous sdists shipped the full documentation tree (91 MB at 0.4.0, close to PyPI's 100 MB upload limit) along with repository configuration files; the sdist is now under 1 MB.

### Deprecated
- Deprecated the `as_nx` parameter of the graph builders. Passing `as_nx` explicitly, whether `True` or `False`, now emits a `DeprecationWarning` pointing at `gdf_to_nx()`, which will be the only supported route to a NetworkX graph once the parameter is removed in a future major release. The affected functions are `knn_graph()`, `delaunay_graph()`, `gabriel_graph()`, `relative_neighborhood_graph()`, `euclidean_minimum_spanning_tree()`, `fixed_radius_graph()`, `waxman_graph()`, `contiguity_graph()`, `bridge_nodes()`, `group_nodes()`, `morphological_graph()`, `place_to_place_graph()`, `place_to_movement_graph()`, `movement_to_movement_graph()`, `segments_to_graph()`, `od_matrix_to_graph()`, and `travel_summary_graph()`. Their `as_nx` default changed from `False` to a `None` sentinel so that the warning fires only on explicit use; the effective default, the return types, and the behaviour are all unchanged. To migrate, drop `as_nx` and pass the returned GeoDataFrames through `gdf_to_nx()`.

### Removed
- **Breaking:** Removed `private_to_private_graph()`, `private_to_public_graph()`, and `public_to_public_graph()`, deprecated in 0.4.0. Use `place_to_place_graph()`, `place_to_movement_graph()`, and `movement_to_movement_graph()` instead.
- Removed the unused Docker development environment and its support configuration.

### Fixed
- Fixed `gabriel_graph()` dropping valid edges. The disc test counted points within a closed disc and required exactly two, so any third node lying on the circle removed the edge. The criterion is now the exact open-disc test, `d(u,w)^2 + d(v,w)^2 < d(u,v)^2`, which keeps edges on cocircular configurations such as regular grids.
- Fixed `relative_neighborhood_graph()` on degenerate point configurations: the lune test now applies a relative floating-point tolerance so nodes at exactly the edge length no longer eliminate valid edges, and an exact Gabriel-disc pre-test rejects genuinely blocked candidates.
- Fixed `od_matrix_to_graph(compute_edge_geometry=False, as_nx=True)` returning a graph with no edges. Edge frames whose geometry column is entirely missing now bypass geometry validation and resolve their endpoints from the `(source, target)` MultiIndex, warning about unmappable edges instead of dropping them silently.
- Made enclosed tessellation substantially more robust. Invalid building footprints are repaired before any morphology computation, so one self-intersecting polygon no longer forces every building in the enclosure onto the footprint fallback. Null barrier geometries are ignored. The retry ladder is driven by a failure classifier and every known failure, whether a geometry-type `TypeError`, a GEOS topology error, or a momepy concatenation `ValueError`, now ends in a warning and an empty tessellation for the affected unit at whichever rung it occurs, rather than aborting the whole run; only unknown errors propagate. Tessellation is retried at a coarser precision and then with deterministically jittered geometry, and the overlap repair keeps the good cells and drops only the broken enclosures.
- Broadened the enclosed-tessellation degeneracy check. A tessellation must partition its enclosure, so cells are now validated against each other as well as against the enclosure: overlapping cells that stay within the enclosure area, and cells that collapse and leave the enclosure partly uncovered, are both detected and repaired. Previously only cells that overfilled the enclosure were caught, which let a degenerate partition silently drop the building a cell was built around.
- Fixed the timeout options passed to the Overture Maps CLI. `connect_timeout` and `request_timeout` were sent as `--connect-timeout`/`--request-timeout` with a fractional value, neither of which the CLI accepts, so any download with a timeout set failed outright. They are now sent as `--connect_timeout`/`--request_timeout` and rounded to whole seconds.
- Made Overture release validation resilient. From overturemaps 1.0 the release list resolves lazily over the network, so validating a `release` argument could raise a connection error instead of a `ValueError`. The catalogue is now fetched once and an unreachable catalogue downgrades to a warning.
- Unified the schema of empty enclosed-tessellation results so every path returns `[geometry, enclosure_index, tess_id]` instead of varying by failure mode.
- Unified fallback place-cell generation across the whole-tessellation and unenclosed-building fallbacks. Both now record the source building index and match exactly, fixing duplicated rows and mis-assigned buildings when one footprint contained several representative points, and the internal source-index column no longer leaks into the place nodes when `keep_buildings=False`.
- Guarded `get_od_pairs()` against an empty calendar table.
- Fixed a pandas chained-assignment warning raised when writing columns to the fallback-cell frame.

### Documentation
- Rebuilt the documentation landing page and the examples index with per-example thumbnails, refreshed logos, and a revised example order.
- Updated the preferred citation to the published *Computers, Environment and Urban Systems* article (doi: `10.1016/j.compenvurbsys.2026.102492`) across `CITATION.cff`, the README, and the documentation landing page. `preferred-citation` is now of type `article` and lists all four authors instead of citing the software release.
- Removed the outdated Zenodo software DOI from the README and `docs/llms.txt`. The *Computers, Environment and Urban Systems* article is now stated as the required citation across the README, the documentation landing page, the paper page, and `CITATION.cff`, whose abstract and `identifiers` entry carry the same instruction through to the Zenodo record.
- Repaired the executable docstring examples that produced wrong or impossible output: corrected stale GeoPandas/Shapely reprs, converted pseudo-output blocks to real doctest output in `segments_to_graph()`, `gdf_to_nx()`, and `nx_to_gdf()` (whose documented `full_edge_type` attribute is actually `edge_type`), made the `nx_to_pyg()` example pass its own CRS validation, and replaced the no-longer-available Overture release pinned in the `load_overture_data()` examples.
- Fixed an undefined variable in a plotting cell of the proximity-graphs example notebook, and repaired the broken workflow-table anchor on the landing page.
- Consolidated contributor guidance into `docs/contributing.md` as the single canonical development, testing, code-quality, and pull-request reference, with `README.md` and the pull request template pointing at it.
- Expanded `docs/llms.txt` and page metadata, and simplified the MkDocs setup.
- Clarified in the API documentation when `suppress_empty_error` actually fires: with barriers present, `create_tessellation()` degrades a momepy concatenation failure to an empty tessellation internally, so the `ValueError` conversion only applies on the barrier-free morphological path.
- Updated installation and security guidance for the current PyTorch and PyTorch Geometric support matrix.
- Updated release metadata and versioned documentation links for `v1.0.0`.


## 0.4.0 (2026-06-11)

### Added
- Added a `directed` parameter to `gdf_to_pyg()`. With the new default `directed=False`, each edge `(u, v)` is symmetrized by adding the reverse edge `(v, u)` (self-loops excluded, edge attributes duplicated) so PyTorch Geometric receives a proper undirected graph; `pyg_to_gdf()` deduplicates the symmetrized edges back to the original rows on reconstruction. For heterogeneous graphs, `directed` also accepts a complete dictionary mapping each edge type to its own directionality flag.
- Added a `reverse_edge_types` parameter to `gdf_to_pyg()` controlling undirected cross-type heterogeneous edges: `"auto"` (default) generates a `(dst_type, "rev_<relation>", src_type)` reverse edge store for message passing, a dict provides explicit mappings, and `None` raises a `ValueError` (strict mode). Generated reverse stores are skipped by `pyg_to_gdf()` reconstruction.
- Added a `multigraph` parameter to `gdf_to_pyg()` that promotes two-level edge indexes to a keyed `(source, target, key)` contract so parallel edges can be preserved; three-level edge indexes always keep their supplied keys, and `pyg_to_gdf()` round-trips them.
- Added a `directed` parameter to `nx_to_pyg()`. By default the NetworkX graph type decides: `Graph`/`MultiGraph` convert as undirected, `DiGraph`/`MultiDiGraph` as directed. `pyg_to_nx()` now returns `MultiGraph`/`MultiDiGraph` with preserved edge keys when the metadata or edge indexes carry keys, and restores original node labels for homogeneous graphs.
- Added `canonicalize_edges()` to collapse reciprocal `(u, v)` / `(v, u)` rows and parallel duplicates in edge GeoDataFrames, with `duplicates="first" | "key" | "error"` handling.
- Added `symmetrize_edges()` as the inverse of `canonicalize_edges()`: it appends the reverse row of every non-self-loop edge (with reversed geometry) so neighbourhood queries on the MultiIndex are complete. The operation is idempotent.
- Added a `duplicate_edges` parameter to the proximity generators (`knn_graph()`, `delaunay_graph()`, `gabriel_graph()`, `relative_neighborhood_graph()`, `euclidean_minimum_spanning_tree()`, `fixed_radius_graph()`, `waxman_graph()`, `contiguity_graph()`) and the morphology functions (`morphological_graph()`, `place_to_place_graph()`, `place_to_movement_graph()`, `movement_to_movement_graph()`) to optionally emit both `(u, v)` and `(v, u)` rows per undirected edge. Incompatible with `as_nx=True`.
- Added `extent_buffer`, `limit`, and `include_unenclosed_buildings` parameters to `morphological_graph()`: `extent_buffer` caps the perpendicular access distance from a street to a building/cell, `limit` passes an explicit enclosure boundary, and `include_unenclosed_buildings` keeps buildings outside any enclosure.
- Added a `max_connection_distance` parameter to `place_to_movement_graph()` to connect otherwise unmatched place polygons to their nearest movement geometry within a distance cap.
- Added a `limit` parameter to `create_tessellation()` forwarded to `momepy.enclosures`; when omitted, a buffered convex hull of the input geometry and barriers is computed so buildings near outer street loops are not dropped from enclosed tessellation.
- Added a `directed` parameter to `segments_to_graph()`; `directed=False` canonicalizes each edge to an unordered `(min, max)` node-id order so reverse-drawn duplicate segments become parallel edges of one unordered pair.

### Changed
- **Breaking:** `gdf_to_pyg()` now treats edges as undirected by default and validates them: edge tables containing both `(u, v)` and `(v, u)` rows, or parallel rows for the same unordered pair, raise a `ValueError` (use `canonicalize_edges()`, `multigraph=True`, or `directed=True` to resolve). Edge GeoDataFrames must have a MultiIndex with at least two levels (source, target).
- **Breaking:** Renamed the morphology terminology from "private"/"public" to "place"/"movement". Node keys are now `"place"` and `"movement"`; edge types are `("place", "touched_to", "place")`, `("movement", "connected_to", "movement")`, and `("place", "faced_to", "movement")`; identifier columns are `place_id`, `movement_id`, `from_place_id`/`to_place_id`, and `from_movement_id`/`to_movement_id`.
- **Breaking:** `segments_to_graph()` now defaults to `multigraph=True`, returning a three-level `(from_node_id, to_node_id, edge_key)` MultiIndex (and an `nx.MultiGraph` when `as_nx=True`). With `multigraph=False`, duplicate node pairs now raise a `ValueError` instead of silently returning a duplicated MultiIndex.
- `GraphMetadata` now records per-edge-type directionality, symmetrization provenance, multigraph flags, and generated reverse edge type mappings so `pyg_to_gdf()` and `pyg_to_nx()` can reconstruct the original input faithfully.
- `pyg_to_nx()` warns and falls back to an undirected graph when heterogeneous edge types mix directed and undirected metadata.
- Improved the undirected validation errors in `gdf_to_pyg()`: they now report the total number of affected node pairs with examples, explain the typical cause (reciprocal rows from directed sources such as OSMnx), and point to `canonicalize_edges()` as a remedy.

### Deprecated
- Deprecated `private_to_private_graph()`, `private_to_public_graph()`, and `public_to_public_graph()` in favor of `place_to_place_graph()`, `place_to_movement_graph()`, and `movement_to_movement_graph()`. The old names emit a `DeprecationWarning` and will be removed in a future major release.

### Fixed
- Fixed `segments_to_graph()` ignoring `as_nx=True` for empty inputs; it now returns an empty NetworkX graph instead of a tuple, and empty outputs carry properly named indexes.
- Fixed `segments_to_graph(multigraph=True, as_nx=True)` silently collapsing parallel edges by returning an `nx.Graph`; it now returns an `nx.MultiGraph`.
- Fixed a `UnicodeDecodeError` in `load_overture_data()` when reading back files saved with `save_to_file=True` by forcing UTF-8 encoding.
- Made enclosed tessellation more robust: boundary simplification failures (`TypeError` from `shapely.coverage_simplify` on degenerate footprints) retry once with `simplify=False`, and GEOS topology errors retry once with a coarser `grid_size` before returning an empty tessellation for the affected unit.
- Fixed `load_gtfs()` UDF registration against newer DuckDB releases by using explicit `duckdb.sqltype` signatures, and guarded `get_od_pairs()` against an empty calendar table.


## 0.3.1 (2026-03-21)

### Added
- Added regression coverage for undirected metapath deduplication in `add_metapaths()` and `add_metapaths_by_weight()` across GeoDataFrame and NetworkX graph outputs.

### Changed
- Refreshed the locked dependencies in `uv.lock`.

### Fixed
- Fixed undirected metapath materialization so mirrored traversals collapse into one path signature when `directed=False`.
- Fixed undirected weighted metapath extraction so endpoint pairs are emitted only once instead of duplicating reversed edges.

### Documentation
- Updated release metadata and versioned documentation links for `v0.3.1`.


## 0.3.0 (2026-03-15)

### Added
- Added DuckDB-backed GTFS loading through `load_gtfs()`, returning an in-memory database connection for SQL-first transit workflows.
- Added directed and frequency-aware options to `travel_summary_graph()` and directed OD-pair generation in `get_od_pairs()`.
- Added support for multiple center points and multi-threshold layered output in `create_isochrone()` and `filter_graph_by_distance()`.

### Changed
- Reworked the GTFS transportation pipeline to operate on DuckDB tables instead of materialized DataFrame dictionaries.
- Improved concave-hull isochrone generation defaults and internal performance for more stable polygon output.
- Switched `overturemaps` dependency sourcing to the published package release instead of a Git source.

### Fixed
- Fixed `clip_graph()` CRS alignment and strict clipping of out-of-boundary endpoints.
- Fixed `load_overture_data()` so clipped and post-processed outputs are written back to disk when saving files.
- Fixed native ID handling and empty building-join edge cases in `morphological_graph()`.
- Fixed heterogeneous PyG conversion edge handling, metapath empty-frame construction, and `plot_graph()` style kwargs forwarding.

### Documentation
- Updated release metadata, installation guidance, GTFS examples, and contributor docs for the `v0.3.0` release.


## 0.2.4 (2026-02-18)

### Added
- Added `get_boundaries` function to retrieve polygon boundaries using Nominatim geocoding.

### Changed
- Updated `load_overture_data` to support `place_name` parameter for automatic boundary retrieval via Nominatim geocoding.
- Updated PyTorch dependencies to address security vulnerabilities (CVE-2025-3730, CVE-2025-2953).
- Removed support for deprecated CUDA versions (`cu118`, `cu124`).
- Enforced `torch>=2.8.0` across all install variants.

### Fixed
- Fixed `GeometryTypeError` in `load_overture_data` by filtering out non-LineString geometries.

### Documentation
- Updated `SECURITY.md` to reflect supported versions and resolved vulnerabilities.


## 0.2.3 (2026-01-07)

### Added
- N/A

### Changed
- N/A

### Fixed
- Fixed `edge_feature_cols` logic in `gdf_to_pyg` for heterogeneous graphs
- Fixed `robots.txt` configuration

### Documentation
- Updated `robots.txt`


## 0.2.2 (2026-01-04)

### Added
- Added support for extracting additional node/edge attributes during graph reconstruction

### Changed
- N/A

### Fixed
- N/A

### Documentation
- Updated documents

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
