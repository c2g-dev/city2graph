============
Installation
============

Standard Installation
-------------------

The simplest way to install city2graph is via pip:

.. code-block:: bash

    # Basic installation (without PyTorch)
    pip install city2graph

This installs the core functionality without PyTorch and PyTorch Geometric, suitable for basic graph operations with networkx, shapely, geopandas, etc.

.. raw:: html

   <div class="conda-deprecation-warning">
     <p><strong>Warning:</strong> Conda distributions are deprecated for PyTorch and PyTorch Geometric due to limited demand and compatibility issues. We recommend using pip or Poetry for the most reliable installation experience.</p>
   </div>

With PyTorch (Optional)
----------------------

If you need the graph neural network functionality, install with the torch option:

.. code-block:: bash

    # Install with PyTorch and PyTorch Geometric (CPU version)
    pip install "city2graph[torch]"

This will install PyTorch and PyTorch Geometric with CPU support, suitable for development and small-scale processing.

With Specific CUDA Version (for GPU acceleration)
-----------------------------------------------

For GPU acceleration with a specific CUDA version, we recommend installing PyTorch and PyTorch Geometric separately before installing city2graph:

.. code-block:: bash

    # Step 1: Install PyTorch with your desired CUDA version
    # Visit https://pytorch.org/get-started/locally/ and select your preferences
    # Example for PyTorch 2.4.0 with CUDA 12.1:
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    # Step 2: Install PyTorch Geometric and its CUDA dependencies
    pip install torch-geometric
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
    
    # Step 3: Now install city2graph (it will detect the pre-installed PyTorch)
    pip install city2graph

For macOS users with Apple Silicon:

.. code-block:: bash

    # PyTorch for macOS uses MPS (Metal Performance Shaders) instead of CUDA
    pip install torch torchvision torchaudio
    pip install torch-geometric
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
    pip install city2graph

With Development or Documentation Dependencies
-------------------------------------------

You can install city2graph with additional optional dependencies:

.. code-block:: bash
    
    # With documentation dependencies
    pip install "city2graph[docs]"
    
    # With development dependencies
    pip install "city2graph[dev]"
    
    # With both documentation and development dependencies
    pip install "city2graph[docs,dev]"

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