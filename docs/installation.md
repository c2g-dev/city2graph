---
seo_title: "Install City2Graph: pip, PyTorch, CUDA, and conda"
description: "Install City2Graph with pip or conda-forge, choose optional PyTorch Geometric CPU or CUDA support, and verify the Python installation."
hide:
  - navigation
---

# Install City2Graph

City2Graph supports Python 3.12–3.14. Install the core package for geospatial
graph construction and network analysis, or choose a PyTorch extra when you
need PyTorch Geometric tensors and Graph Neural Networks.

## Choose an installation

| Goal | Recommended command | Includes PyTorch and PyG? |
| --- | --- | --- |
| Spatial graph construction and NetworkX analysis | `pip install city2graph` | No |
| GNN development on CPU | Install CPU wheels, then `pip install "city2graph[cpu]"` | Yes, CPU |
| GNN training with CUDA 12.6 | Install CUDA 12.6 wheels, then `pip install "city2graph[cu126]"` | Yes, GPU |
| GNN training with CUDA 12.8 | Install CUDA 12.8 wheels, then `pip install "city2graph[cu128]"` | Yes, GPU |
| GNN training with CUDA 13.0 | Install CUDA 13.0 wheels, then `pip install "city2graph[cu130]"` | Yes, GPU |
| GNN training with CUDA 13.2 | Install CUDA 13.2 wheels, then `pip install "city2graph[cu132]"` | Yes, GPU |
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
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
pip install "city2graph[cpu]"
```

This installs PyTorch and PyTorch Geometric with CPU support.

### With PyTorch and CUDA (GPU)

On Linux or Windows, install PyTorch and torchvision from the official index
for the required CUDA variant, then install the matching City2Graph extra.
For CUDA 13.2:

```bash
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu132
pip install "city2graph[cu132]"
```

For CUDA 12.6 or 13.0, replace `cu132` in both commands with `cu126` or
`cu130`. CUDA 12.8 wheels are available only through PyTorch 2.11, so use:

```bash
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install "city2graph[cu128]"
```

The lockfile resolves the following stable versions. These are the tested
versions, rather than new minimum requirements for every dependency:

| Extra | PyTorch | torchvision | PyG |
| --- | --- | --- | --- |
| `cpu`, `cu126`, `cu130`, `cu132` | 2.14.1 | 0.29.1 | 2.8.1 |
| `cu128` | 2.11.0 | 0.26.0 | 2.8.1 |

pip does not read this repository's `tool.uv.sources` settings, and an extra
name alone cannot select a wheel index. Installing the matching PyTorch wheels
first ensures that pip retains the intended CPU or CUDA build. The CUDA
instructions apply to Linux and Windows; macOS uses its native PyTorch build.

For a source checkout managed with uv, the index selection is configured for
both torch and torchvision. Select exactly one extra, for example:

```bash
uv sync --extra cu132 --group dev
```

[PyG 2.8's published support matrix](https://github.com/pyg-team/pytorch_geometric/releases/tag/2.8.0)
lists PyTorch 2.9–2.12. City2Graph verifies its graph conversions with PyTorch
2.14.1, but optional PyG binary extensions require their own compatible wheels.
They are not needed for City2Graph's conversions and are not installed by these
extras. See the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/stable/install/installation.html)
before adding them.

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

For GPU support, choose a PyTorch version and CUDA build that are actually
available from conda-forge. Replace `PYTORCH_VERSION` and `CUDA_VARIANT` with
values from the files page linked below:

```bash
# Install City2Graph
conda install -c conda-forge city2graph

# Add PyTorch with CUDA support and PyTorch Geometric
conda install -c conda-forge "pytorch=PYTORCH_VERSION=*CUDA_VARIANT*"
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
- GeoPandas and Shapely
- DuckDB
- libpysal and momepy
- Overture Maps
- SciPy and geopy

PyTorch and PyTorch Geometric are required only for PyG conversion and graph
neural network workflows.
