# City2Graph: Geospatial Graphs for Network Analysis and GNNs

[![City2Graph](https://city2graph.net/latest/assets/logos/social_preview_city2graph.png)](https://city2graph.net/latest/)

**City2Graph** is a Python library that turns buildings, streets, public
transport feeds, origin–destination matrices, zones, and points of interest
into spatial and heterogeneous graphs. It bridges
[GeoPandas](https://geopandas.org/), [NetworkX](https://networkx.org/), and
[PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) so the
same geospatial data can support network analysis, urban research, and Graph
Neural Networks (GNNs). See the
[documentation](https://city2graph.net/latest/) for installation, tutorials,
and the Python API reference.

[![PyPI version](https://badge.fury.io/py/city2graph.svg)](https://badge.fury.io/py/city2graph/) [![conda-forge Version](https://anaconda.org/conda-forge/city2graph/badges/version.svg)](https://anaconda.org/conda-forge/city2graph/) [![PyPI Downloads](https://static.pepy.tech/badge/city2graph)](https://pepy.tech/projects/city2graph) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15858845.svg)](https://doi.org/10.5281/zenodo.15858845) [![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/c2g-dev/city2graph/blob/main/LICENSE)
[![Platform](https://anaconda.org/conda-forge/city2graph/badges/platforms.svg
)](https://anaconda.org/conda-forge/city2graph) [![codecov](https://codecov.io/gh/c2g-dev/city2graph/graph/badge.svg?token=2R449G75Z0)](https://codecov.io/gh/c2g-dev/city2graph) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Features

[![City2Graph workflow from geospatial data to graph analysis](https://city2graph.net/latest/assets/figures/scope.png)](https://city2graph.net/latest/)

- **Morphology:** Graphs of buildings, streets, and tessellated urban fabric from OpenStreetMap and Overture Maps.
- **Transportation:** GTFS public transport and GBFS shared-mobility feeds loaded
  into DuckDB, with GTFS aggregated into stop-to-stop transit graphs.
- **Mobility:** Origin–destination matrices and flow data — migration, bike-sharing, pedestrian counts — as weighted spatial graphs.
- **Proximity and Contiguity:** KNN, Delaunay, Gilbert, and Waxman graphs plus queen/rook contiguity, under Euclidean, Manhattan, or network distances.
- **Heterogeneous Graphs and Metapaths:** Multiple node and edge types in one graph, with metapath-derived edges composing relations across them.
- **GNN-ready Tensors:** Round-trip conversion between GeoDataFrames, NetworkX, and PyTorch Geometric `Data`/`HeteroData`.

## Installation

### Using pip

#### Basic Installation

The simplest way to install City2Graph is via pip:

```bash
pip install city2graph
```

This installs the core functionality without PyTorch and PyTorch Geometric.

#### With PyTorch (CPU)

If you need the Graph Neural Networks functionality, install with the `cpu` option:

```bash
pip install "city2graph[cpu]"
```

This will install PyTorch and PyTorch Geometric with CPU support, suitable for development and small-scale processing.

#### With PyTorch + CUDA (GPU)

For GPU acceleration, you can install City2Graph with a specific CUDA version extra. For example, for CUDA 13.0:

```bash
pip install "city2graph[cu130]"
```

Supported CUDA versions are `cu126`, `cu128`, and `cu130`.
The `cpu`, `cu126`, and `cu130` extras use PyTorch 2.12 or newer. Because
PyTorch 2.12 no longer publishes CUDA 12.8 wheels, `cu128` uses PyTorch 2.11.

### Using conda

#### Basic Installation

You can also install City2Graph using conda from conda-forge:

```bash
conda install -c conda-forge city2graph
```

This installs the core functionality without PyTorch and PyTorch Geometric.

#### With PyTorch (CPU)

To use PyTorch and PyTorch Geometric with City2Graph installed from conda-forge, you need to manually add these libraries to your environment:

```bash
# Install city2graph
conda install -c conda-forge city2graph

# Then install PyTorch and PyTorch Geometric
conda install -c conda-forge pytorch pytorch_geometric
```

#### With PyTorch + CUDA (GPU)

For GPU support, you should select the appropriate PyTorch variant by specifying the version and CUDA build string. For example, to install PyTorch 2.12.0 with CUDA 13.0 support:

```bash
# Install city2graph
conda install -c conda-forge city2graph

# Then install PyTorch with CUDA support
conda install -c conda-forge pytorch=2.12.0=*cuda130*
conda install -c conda-forge pytorch_geometric
```

You can browse available CUDA-enabled builds on the [conda-forge PyTorch files page](https://anaconda.org/conda-forge/pytorch/files) and substitute the desired version and CUDA variant in your install command. Make sure that the versions of PyTorch and PyTorch Geometric you install are compatible with each other and with your system.

**⚠️ Important:** conda is not officially supported by PyTorch and PyTorch Geometric anymore, and only conda-forge distributions are available for them. We recommend using pip or uv for the most streamlined installation experience if you need PyTorch functionality.

## For Development

See the [Contributing Guide](docs/contributing.md) for the canonical development
setup, testing, code quality, documentation, and pull request instructions.

## Citation

If you use City2Graph in your research, please cite it as follows:

```bibtex
@software{sato2025city2graph,
  title = {City2Graph: Transform geospatial relations into graphs for spatial network analysis and Graph Neural Networks},
  author = {Sato, Yuta},
  year = {2025},
  url = {https://github.com/c2g-dev/city2graph},
  doi = {10.5281/zenodo.15858845},
}
```

You can also use the DOI to cite a specific version: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15858845.svg)](https://doi.org/10.5281/zenodo.15858845)

Alternatively, you can find the citation information in the [CITATION.cff](CITATION.cff) file in this repository, which follows the Citation File Format standard.

## Contributing

Contributions are welcome. The [Contributing Guide](docs/contributing.md)
contains the complete development and quality requirements.

## Documentation

City2Graph uses **MkDocs** for current documentation (v0.2.0+) and keeps **Sphinx** for legacy releases (v0.1.0–v0.1.7).

- **Legacy tags** (`v0.1.*`): Read the Docs builds `docs/source` via Sphinx.
- **Everything else** (branches / newer tags): Read the Docs builds via MkDocs (`mkdocs.yml`).

This is controlled in `.readthedocs.yaml` using `READTHEDOCS_VERSION_TYPE` and `READTHEDOCS_VERSION_NAME`.

[![GeoGraphic Data Science Lab](https://github.com/user-attachments/assets/569b9550-9a48-461d-a408-18d7a5dfc78c)](https://www.liverpool.ac.uk/geographic-data-science/)
