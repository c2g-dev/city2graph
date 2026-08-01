---
seo_title: "Install City2Graph: pip, PyTorch, CUDA, and conda"
description: "Install City2Graph with pip or conda-forge, choose optional PyTorch Geometric CPU or CUDA support, and verify the Python installation."
hide:
  - navigation
---

# Install City2Graph

City2Graph supports Python 3.11–3.14. Install the core package for geospatial
graph construction and network analysis, or choose a PyTorch extra when you
need PyTorch Geometric tensors and Graph Neural Networks.

## Choose an installation

| Goal | Recommended command | Includes PyTorch and PyG? |
| --- | --- | --- |
| Spatial graph construction and NetworkX analysis | `pip install city2graph` | No |
| GNN development on CPU | `pip install "city2graph[cpu]"` | Yes, CPU |
| GNN training with CUDA 12.6 | `pip install "city2graph[cu126]"` | Yes, GPU |
| GNN training with CUDA 12.8 | `pip install "city2graph[cu128]"` | Yes, GPU |
| GNN training with CUDA 13.0 | `pip install "city2graph[cu130]"` | Yes, GPU |
| Core package from conda-forge | `conda install -c conda-forge city2graph` | No |

## Using pip

### Core installation

Install the core package when you want to construct geospatial graphs, convert
them to NetworkX or rustworkx, and run spatial network analysis:

```bash
pip install city2graph
```

This keeps the installation smaller by excluding PyTorch and PyTorch Geometric.

### With PyTorch (CPU)

For PyTorch Geometric conversion and GNN development without a GPU, use:

```bash
pip install "city2graph[cpu]"
```

This installs PyTorch and PyTorch Geometric with CPU support.

### With PyTorch and CUDA (GPU)

Choose the extra that matches the CUDA wheel required by your environment. For
example, install CUDA 13.0 support with:

```bash
pip install "city2graph[cu130]"
```

Supported extras are `cu126`, `cu128`, and `cu130`. The `cpu`, `cu126`, and
`cu130` extras use PyTorch 2.12 or newer. Because PyTorch 2.12 no longer
publishes CUDA 12.8 wheels, `cu128` uses PyTorch 2.11.

## Using conda-forge

### Core installation

Install the core City2Graph package from conda-forge with:

```bash
conda install -c conda-forge city2graph
```

This does not install PyTorch or PyTorch Geometric.

### With PyTorch (CPU)

To use PyTorch Geometric with the conda-forge package, add the CPU dependencies
separately:

```bash
# Install City2Graph
conda install -c conda-forge city2graph

# Add PyTorch and PyTorch Geometric
conda install -c conda-forge pytorch pytorch_geometric
```

### With PyTorch and CUDA (GPU)

For GPU support, select the appropriate PyTorch version and CUDA build. For
example, install PyTorch 2.12.0 with CUDA 13.0 support with:

```bash
# Install City2Graph
conda install -c conda-forge city2graph

# Add PyTorch with CUDA support and PyTorch Geometric
conda install -c conda-forge pytorch=2.12.0=*cuda130*
conda install -c conda-forge pytorch_geometric
```

Browse the
[conda-forge PyTorch files page](https://anaconda.org/conda-forge/pytorch/files)
to select another supported version and CUDA variant. Ensure that PyTorch,
PyTorch Geometric, and the installed CUDA runtime are mutually compatible.

!!! warning
    PyTorch and PyTorch Geometric no longer officially support conda packages.
    Only conda-forge distributions are available. Prefer pip or uv when you
    need PyTorch functionality.

## Verify the installation

Check that City2Graph imports and report the installed version:

```bash
python -c "import city2graph as c2g; print(c2g.__version__)"
```

For a PyTorch-enabled installation, also verify that PyTorch Geometric is
available:

```bash
python -c "import torch, torch_geometric; print(torch.__version__, torch_geometric.__version__)"
```

If these commands fail, confirm that the active Python environment is the same
one in which City2Graph was installed.

## Core dependencies

The core installation includes:

- NetworkX and rustworkx
- GeoPandas, Shapely, and OSMnx
- DuckDB
- libpysal and momepy
- Overture Maps
- SciPy and geopy

PyTorch and PyTorch Geometric are required only for PyG conversion and graph
neural network workflows.
