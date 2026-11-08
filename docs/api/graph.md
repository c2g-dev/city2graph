---
seo_title: "Graph Conversion API: GeoDataFrame, NetworkX, PyG - City2Graph"
description: "Convert spatial graphs between GeoDataFrames, NetworkX and PyTorch Geometric Data and HeteroData for GeoAI and graph neural network workflows in Python."
keywords: PyTorch Geometric, GeoDataFrame, NetworkX, graph conversion, HeteroData, Data, gdf_to_pyg, pyg_to_gdf, geospatial deep learning, graph neural networks
---

# Graph Module

The graph module provides functions for converting between different graph representations, including GeoDataFrames, NetworkX graphs, and PyTorch Geometric data objects.

Use it to move a spatial graph from mapping and network analysis into GeoAI workflows: `gdf_to_pyg` produces `Data` or `HeteroData` for graph neural network training, and `pyg_to_gdf` converts PyTorch Geometric objects back to GeoDataFrames for mapping. Heterogeneous graphs keep their node and edge types in both directions. The [metapath tutorial](../examples/add_metapaths.ipynb) and the [morphology tutorial](../examples/morphological_graph_from_overturemaps.ipynb) show the conversions in context.

::: city2graph.graph
    options:
      show_root_heading: false
      members:
        - gdf_to_pyg
        - nx_to_pyg
        - pyg_to_gdf
        - pyg_to_nx
        - is_torch_available
        - validate_pyg
