# City2Graph: GeoAI with Graph Neural Networks (GNNs) and Spatial Network Analysis

[![City2Graph](http://city2graph.net/latest/assets/logos/social_preview_city2graph.png)](http://city2graph.net/latest/assets/logos/social_preview_city2graph.png)

**City2Graph** is a Python library for converting geospatial datasets into graph representations, providing an integrated interface for [GeoPandas](https://geopandas.org/), [NetworkX](https://networkx.org/), and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) across multiple domains (e.g. streets, transportations, OD matrices, POI proximities, etc.). It enables researchers and practitioners to seamlessly develop advanced GeoAI and geographic data science applications. For more information, please visit the [documentation](https://city2graph.net).

[![PyPI version](https://badge.fury.io/py/city2graph.svg)](https://badge.fury.io/py/city2graph/) [![conda-forge Version](https://anaconda.org/conda-forge/city2graph/badges/version.svg)](https://anaconda.org/conda-forge/city2graph/) [![PyPI Downloads](https://static.pepy.tech/badge/city2graph)](https://pepy.tech/projects/city2graph) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15858845.svg)](https://doi.org/10.5281/zenodo.15858845) [![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/c2g-dev/city2graph/blob/main/LICENSE)
[![Platform](https://anaconda.org/conda-forge/city2graph/badges/platforms.svg
)](https://anaconda.org/conda-forge/city2graph) [![codecov](https://codecov.io/gh/c2g-dev/city2graph/graph/badge.svg?token=2R449G75Z0)](https://codecov.io/gh/c2g-dev/city2graph) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Features

[![scope](http://city2graph.net/latest/assets/figures/scope.png)](http://city2graph.net/latest/assets/figures/scope.png)

- **Graph Construction for GeoAI:** Build graphs from diverse urban datasets, including buildings, streets, and land use, to power GeoAI and GNN applications.
- **Transportation Network Modeling:** Analyze public transport systems (buses, trams, trains) by constructing detailed transportation graphs with support of GTFS format.
- **Proximity and Contiguity Analysis:** Create graphs based on spatial proximity and adjacency for applications in urban planning and environmental analysis.
- **Mobility Flow Analysis:** Model and analyze urban mobility patterns from various data sources like bike-sharing, migration, and pedestrian flows.
- **PyTorch Geometric Integration:** Seamlessly convert geospatial data into PyTorch tensors for GNNs.

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

Supported CUDA versions are `cu118`, `cu124`, `cu126`, `cu128`, and `cu130`.

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

For GPU support, you should select the appropriate PyTorch variant by specifying the version and CUDA build string. For example, to install PyTorch 2.7.1 with CUDA 12.8 support:

```bash
# Install city2graph
conda install -c conda-forge city2graph

# Then install PyTorch with CUDA support
conda install -c conda-forge pytorch=2.7.1=*cuda128*
conda install -c conda-forge pytorch_geometric
```

You can browse available CUDA-enabled builds on the [conda-forge PyTorch files page](https://anaconda.org/conda-forge/pytorch/files) and substitute the desired version and CUDA variant in your install command. Make sure that the versions of PyTorch and PyTorch Geometric you install are compatible with each other and with your system.

**⚠️ Important:** conda is not officially supported by PyTorch and PyTorch Geometric anymore, and only conda-forge distributions are available for them. We recommend using pip or uv for the most streamlined installation experience if you need PyTorch functionality.

## For Development

If you want to contribute to City2Graph, you can set up a development environment using `uv`.

```bash
# Install uv if you haven't already done it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/c2g-dev/city2graph.git
cd city2graph

# Install development dependencies with a PyTorch variant (e.g., cpu or cu128)
uv sync --extra cpu --group dev
```

You can then run commands within the managed environment:

```bash
# Add IPython kernel for interactive development
uv run ipython kernel install --name "your-env-name" --user

# Or start Jupyter Notebook
uv run jupyter notebook
```

### Development Environment

The development dependencies include:
- `ipython`: Enhanced interactive Python shell with Jupyter kernel support
- `jupyter` and `notebook`: For running Jupyter notebooks with project-specific kernel
- `isort`: Code formatting tools
- `pytest` and `pytest-cov`: Testing tools

The Jupyter kernel installation ensures that when you start Jupyter notebooks, you can select the "city2graph" kernel which has access to all your project dependencies in the correct virtual environment.

### Using Docker Compose

Before using Docker Compose, ensure you have Docker and Docker Compose installed on your system:

```bash
# Check Docker installation
docker --version

# Check Docker Compose installation
docker compose version
```

If these commands don't work, you need to install Docker first:
- For macOS: Install [Docker Desktop](https://www.docker.com/products/docker-desktop)
- For Linux: Follow the [installation instructions](https://docs.docker.com/engine/install/) for your specific distribution
- For Windows: Install [Docker Desktop](https://www.docker.com/products/docker-desktop)

Once Docker is installed, clone the repository and start the containers:

```bash
# Clone the repository
git clone https://github.com/yu-ta-sato/city2graph.git
cd city2graph

# Build and run in detached mode
docker compose up -d

# Access Jupyter notebook at http://localhost:8888

# Stop containers when done
docker compose down
```

You can customize the services in the `docker-compose.yml` file according to your needs.

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

We welcome contributions to the City2Graph project! To contribute:

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/<your-name>/city2graph.git
   cd city2graph
   git remote add upstream https://github.com/c2g-dev/city2graph.git
   ```

2. **Set up the development environment:**
   ```bash
   uv sync --group dev --extra cpu
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make your changes and test:**
   ```bash
   # Run pre-commit checks
   uv run pre-commit run --all-files

   # Run tests
   uv run pytest --cov=city2graph --cov-report=html --cov-report=term
   ```

5. **Submit a pull request** with a clear description of your changes.

For detailed contributing guidelines, code style requirements, and documentation standards, please see our [Contributing Guide](docs/source/contributing.rst).

## Code Quality

We maintain strict code quality standards using:
- **Ruff**: For linting and formatting
- **mypy**: For static type checking
- **numpydoc**: For docstring style validation

All contributions must pass pre-commit checks before being merged.

## Documentation

City2Graph uses **MkDocs** for current documentation (v0.2.0+) and keeps **Sphinx** for legacy releases (v0.1.0–v0.1.7).

- **Legacy tags** (`v0.1.*`): Read the Docs builds `docs/source` via Sphinx.
- **Everything else** (branches / newer tags): Read the Docs builds via MkDocs (`mkdocs.yml`).

This is controlled in `.readthedocs.yaml` using `READTHEDOCS_VERSION_TYPE` and `READTHEDOCS_VERSION_NAME`.

[![GeoGraphic Data Science Lab](https://github.com/user-attachments/assets/569b9550-9a48-461d-a408-18d7a5dfc78c)](https://www.liverpool.ac.uk/geographic-data-science/)
