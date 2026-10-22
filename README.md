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

[![PyPI version](https://badge.fury.io/py/city2graph.svg)](https://badge.fury.io/py/city2graph/) [![conda-forge Version](https://anaconda.org/conda-forge/city2graph/badges/version.svg)](https://anaconda.org/conda-forge/city2graph/) [![PyPI Downloads](https://static.pepy.tech/badge/city2graph)](https://pepy.tech/projects/city2graph) [![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.compenvurbsys.2026.102492-blue?logo=doi&logoColor=white)](https://doi.org/10.1016/j.compenvurbsys.2026.102492) [![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/c2g-dev/city2graph/blob/main/LICENSE)
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

City2Graph supports Python 3.12–3.14. The simplest way to install it is via pip:

```bash
pip install city2graph
```

This installs the core functionality without PyTorch and PyTorch Geometric.

#### With PyTorch (CPU)

If you need the Graph Neural Networks functionality, install with the `cpu` option:

```bash
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
pip install "city2graph[cpu]"
```

This will install PyTorch and PyTorch Geometric with CPU support, suitable for development and small-scale processing.

#### With PyTorch + CUDA (GPU)

Install PyTorch and torchvision from the matching official wheel index before
installing the City2Graph extra. For example, for CUDA 13.2 on Linux or Windows:

```bash
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu132
pip install "city2graph[cu132]"
```

Supported CUDA extras are `cu126`, `cu128`, `cu130`, and `cu132`. The current
lockfile uses PyTorch 2.14.1, torchvision 0.29.1, and PyG 2.8.1 for CPU and all
CUDA variants except `cu128`, which uses PyTorch 2.11.0 and torchvision 0.26.0.
For CUDA 12.8, install those older versions from the `cu128` index first.
An extra name alone does not select a CPU or CUDA wheel index in pip.

PyG's published 2.8 support matrix lists PyTorch through 2.12. City2Graph tests
its graph conversions with PyTorch 2.14.1; this does not establish compatibility
for optional PyG binary extensions. See the [installation guide](docs/installation.md)
for the full commands and compatibility notes.

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

For GPU support, select an available CUDA build from the
[conda-forge PyTorch files page](https://anaconda.org/conda-forge/pytorch/files).
For example, replace `PYTORCH_VERSION` and `CUDA_VARIANT` below with a version
and build variant listed there:

```bash
# Install city2graph
conda install -c conda-forge city2graph

# Then install PyTorch with CUDA support
conda install -c conda-forge "pytorch=PYTORCH_VERSION=*CUDA_VARIANT*"
conda install -c conda-forge pytorch_geometric
```

You can browse available CUDA-enabled builds on the [conda-forge PyTorch files page](https://anaconda.org/conda-forge/pytorch/files) and substitute the desired version and CUDA variant in your install command. Make sure that the versions of PyTorch and PyTorch Geometric you install are compatible with each other and with your system.

**⚠️ Important:** conda is not officially supported by PyTorch and PyTorch Geometric anymore, and only conda-forge distributions are available for them. We recommend using pip or uv for the most streamlined installation experience if you need PyTorch functionality.

## For Development

See the [Contributing Guide](docs/contributing.md) for the canonical development
setup, testing, code quality, documentation, and pull request instructions.

## Citation

City2Graph is described in a peer-reviewed article. **Any use of City2Graph in
research or software must cite the following paper**, published in *Computers,
Environment and Urban Systems*:

[![City2Graph paper in Computers, Environment and Urban Systems](https://city2graph.net/latest/assets/figures/CEUS_thumbnail.jpg)](https://doi.org/10.1016/j.compenvurbsys.2026.102492)

Sato, Y., Pietrostefani, E., Mahabir, R., & Arribas-Bel, D. (2026). City2Graph: A Python library for Heterogeneous Graph Neural Networks and spatial analysis in urban systems. *Computers, Environment and Urban Systems*, 130, 102492. [https://doi.org/10.1016/j.compenvurbsys.2026.102492](https://doi.org/10.1016/j.compenvurbsys.2026.102492)

```bibtex
@article{sato2026city2graph,
  title = {City2Graph: A Python library for Heterogeneous Graph Neural Networks and spatial analysis in urban systems},
  author = {Sato, Yuta and Pietrostefani, Elisabetta and Mahabir, Ron and Arribas-Bel, Daniel},
  journal = {Computers, Environment and Urban Systems},
  volume = {130},
  pages = {102492},
  year = {2026},
  issn = {0198-9715},
  doi = {10.1016/j.compenvurbsys.2026.102492},
  url = {https://www.sciencedirect.com/science/article/pii/S0198971526000943},
}
```

The same citation is recorded in the [CITATION.cff](CITATION.cff) file in this repository, which follows the Citation File Format standard.

## Contributing

Contributions are welcome. The [Contributing Guide](docs/contributing.md)
contains the complete development and quality requirements.

## Documentation

City2Graph uses **MkDocs** for current documentation (v0.2.0+) and keeps **Sphinx** for legacy releases (v0.1.0–v0.1.7).

- **Legacy tags** (`v0.1.*`): Read the Docs builds `docs/source` via Sphinx.
- **Everything else** (branches / newer tags): Read the Docs builds via MkDocs (`mkdocs.yml`).

This is controlled in `.readthedocs.yaml` using `READTHEDOCS_VERSION_TYPE` and `READTHEDOCS_VERSION_NAME`.

[![GeoGraphic Data Science Lab](https://github.com/user-attachments/assets/569b9550-9a48-461d-a408-18d7a5dfc78c)](https://www.liverpool.ac.uk/geographic-data-science/)
