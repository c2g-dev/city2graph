============
Installation
============

You can install city2graph with conda using the provided environment files:

.. code-block:: bash

    # For documentation
    conda env create -f docs/environment.yml
    
Or you can install the package directly from source with pip:

.. code-block:: bash

    # Basic installation
    pip install -e .
    
    # With documentation dependencies
    pip install -e ".[docs]"
    
    # With development dependencies
    pip install -e ".[dev]"
    
    # With both documentation and development dependencies
    pip install -e ".[docs,dev]"

Requirements
-----------

city2graph requires the following packages:

* networkx
* shapely
* geopandas
* torch_geometric
* momepy
* overturemaps