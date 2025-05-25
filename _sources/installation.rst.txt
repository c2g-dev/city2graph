============
Installation
============

Standard Installation
-------------------

The simplest way to install city2graph is via pip:

.. code-block:: bash

    # Basic installation (without PyTorch)
    pip install city2graph

This installs the core functionality without PyTorch and PyTorch Geometric.

.. warning::
    Conda distributions are deprecated for PyTorch and PyTorch Geometric due to limited demand and compatibility issues. We recommend using pip or uv for the most reliable installation experience.

With PyTorch (CPU)
----------------------

If you need the graph neural network functionality, install with the torch option:

.. code-block:: bash

    # Install with PyTorch and PyTorch Geometric (CPU version)
    pip install "city2graph[torch]"

This will install PyTorch and PyTorch Geometric with CPU support.

.. note::
   The PyTorch Geometric extensions (pyg_lib, torch_scatter, etc.) are not included in the [torch] extra and must be installed separately as shown in the CUDA/GPU section below.

With PyTorch + CUDA (GPU)
-----------------------------------------------

For GPU acceleration with a specific CUDA version, we recommend installing PyTorch and PyTorch Geometric separately before installing city2graph:

Step 1: Install PyTorch with your desired CUDA version

.. code-block:: bash

    pip install torch=={TORCH_VERSION} --index-url https://download.pytorch.org/whl/{CUDA_VERSION}

Step 2: Install PyTorch Geometric and its CUDA dependencies

.. code-block:: bash

    pip install torch_geometric
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-{TORCH_VERSION}+{CUDA_VERSION}.html

Step 3: Install city2graph (it will detect the pre-installed PyTorch)

.. code-block:: bash

    pip install city2graph

Replace `{TORCH_VERSION}` with the desired PyTorch version (e.g., `'2.4.0'` or above) and `{CUDA_VERSION}` with your CUDA version (e.g., `'cu121'` for CUDA 12.1). You can find the appropriate versions on the `PyTorch website <https://pytorch.org/get-started/locally/>`_ and `PyTorch Geometric website <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>`_.

.. note::
   The core package of PyTorch Geometric (`torch_geometric`) is independent from CUDA or CPU. However, the extensions (`pyg_lib`, `torch_scatter`, etc.) are CUDA-specific.

Using uv
--------

If you're using uv for dependency management, you can install city2graph from the source:

Step 1: Clone the repository and install base dependencies

.. code-block:: bash

    git clone https://github.com/c2g-dev/city2graph.git
    cd city2graph
    uv sync

Step 2: Install with PyTorch support (optional)

.. code-block:: bash

    uv sync --group torch

or if you want to install with a specific PyTorch version:

.. code-block:: bash

    uv sync --group torch --index https://download.pytorch.org/whl/{CUDA_VERSION}

Step 3: Install PyG extensions for CUDA support (if needed)

.. code-block:: bash

    uv add pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv --index https://data.pyg.org/whl/torch-{TORCH_VERSION}+{CUDA_VERSION}.html

Step 4: Install development dependencies (optional)

.. code-block:: bash

    uv sync --group dev

Step 5: Run commands with uv

.. code-block:: bash

    uv run python your_script.py
    uv run jupyter notebook

.. note::
   uv handles dependency resolution more efficiently than Poetry and can install PyTorch Geometric extensions directly through index URLs.


Requirements
-----------

city2graph requires the following packages:

* networkx
* shapely
* geopandas
* libpysal
* momepy
* overturemaps

For graph neural network functionality, you'll also need:

* torch
* torch_geometric
* pyg_lib
* torch_scatter
* torch_sparse
* torch_cluster
* torch_spline_conv