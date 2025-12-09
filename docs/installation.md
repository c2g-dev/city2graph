---
description: Step-by-step guide to installing City2Graph via pip or conda, including instructions for PyTorch Geometric and CUDA support for GPU acceleration.
hide:
  - navigation
---

# Installation

## Using pip

### Standard Installation

The simplest way to install City2Graph is via pip:

```bash
pip install city2graph
```

This installs the core functionality without PyTorch and PyTorch Geometric.

### With PyTorch (CPU)

If you need the graph neural network functionality, install with the `cpu` option:

```bash
pip install "city2graph[cpu]"
```

This will install PyTorch and PyTorch Geometric with CPU support.

### With PyTorch + CUDA (GPU)

For GPU acceleration, you can install City2Graph with a specific CUDA version extra. For example, to install for CUDA 13.0:

```bash
pip install "city2graph[cu130]"
```

Supported CUDA versions are `cu118`, `cu124`, `cu126`, `cu128`, and `cu130`.

## Using conda-forge

### Basic Installation

You can also install City2Graph using conda from conda-forge:

```bash
conda install -c conda-forge city2graph
```

This installs the core functionality without PyTorch and PyTorch Geometric.

### With PyTorch (CPU)

To use PyTorch and PyTorch Geometric with City2Graph installed from conda-forge, you need to manually add these libraries to your environment:

```bash
# Install city2graph
conda install -c conda-forge city2graph

# Then install PyTorch and PyTorch Geometric
conda install -c conda-forge pytorch pytorch_geometric
```

### With PyTorch + CUDA (GPU)

For GPU support, you should select the appropriate PyTorch variant by specifying the version and CUDA build string. For example, to install PyTorch 2.7.1 with CUDA 12.8 support:

```bash
# Install city2graph
conda install -c conda-forge city2graph

# Then install PyTorch with CUDA support
conda install -c conda-forge pytorch=2.7.1=*cuda128*
conda install -c conda-forge pytorch_geometric
```

You can browse available CUDA-enabled builds on the [conda-forge PyTorch files page](https://anaconda.org/conda-forge/pytorch/files) and substitute the desired version and CUDA variant in your install command. Make sure that the versions of PyTorch and PyTorch Geometric you install are compatible with each other and with your system.

!!! warning
    conda is not officially supported by PyTorch and PyTorch Geometric anymore, and only conda-forge distributions are available for them. We recommend using pip or uv for the most streamlined installation experience if you need PyTorch functionality.

## Requirements

City2Graph requires the following packages:

* networkx
* shapely
* geopandas
* libpysal
* momepy
* overturemaps
* rustworkx

For graph neural network functionality, you'll also need:

* torch
* torch_geometric
