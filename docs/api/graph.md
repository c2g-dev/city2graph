# Graph Module

The graph module provides functions for converting between different graph representations, including GeoDataFrames, NetworkX graphs, and PyTorch Geometric data objects.

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
