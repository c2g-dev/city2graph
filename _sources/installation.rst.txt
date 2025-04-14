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
    Conda distributions are deprecated for PyTorch and PyTorch Geometric due to limited demand and compatibility issues. We recommend using pip or Poetry for the most reliable installation experience.

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

    pip install torch-geometric
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-{TORCH_VERSION}+{CUDA_VERSION}.html

Step 3: Install city2graph (it will detect the pre-installed PyTorch)

.. code-block:: bash

    pip install city2graph

Replace `{TORCH_VERSION}` with the desired PyTorch version (e.g., `'2.4.0'` or above) and `{CUDA_VERSION}` with your CUDA version (e.g., `'cu121'` for CUDA 12.1). You can find the appropriate versions on the `PyTorch website <https://pytorch.org/get-started/locally/>`_ and `PyTorch Geometric website <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>`_.

.. note::
   The core package of PyTorch Geometric (`torch_geometric`) is independent from CUDA or CPU. However, the extensions (`pyg_lib`, `torch_scatter`, etc.) are CUDA-specific.

Using Poetry
----------

If you're using Poetry for dependency management, the PyTorch Geometric extensions must be installed separately:

Step 1: Install the base package with Poetry

.. code-block:: bash

    poetry add city2graph

Step 2: Add the torch group (optional)

.. code-block:: bash

    poetry add torch torch-geometric --group torch

Step 3: Install PyG extensions outside of Poetry's dependency resolver

.. code-block:: bash

    poetry run pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu121.html

Step 4: Activate the Poetry environment

.. code-block:: bash

    poetry env activate

.. warning::
   PyTorch Geometric extensions (pyg_lib, torch_scatter, etc.) cannot be managed by Poetry's dependency resolver and must be installed separately with pip as shown above.


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