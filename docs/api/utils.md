---
description: API reference for the Utils module. Utility functions for graph conversion between GeoDataFrame, NetworkX and rustworkx, validation, tessellation, isochrone creation, and plotting.
keywords: gdf_to_nx, nx_to_gdf, rustworkx, tessellation, isochrone, dual_graph, plot_graph, graph validation, spatial utilities
---

# Utils Module

The utils module provides core utility functions for graph conversion, validation, and visualization.

::: city2graph.utils
    options:
      show_root_heading: false
      members:
        - gdf_to_nx
        - nx_to_gdf
        - nx_to_rx
        - rx_to_nx
        - validate_gdf
        - validate_nx
        - dual_graph
        - filter_graph_by_distance
        - create_tessellation
        - create_isochrone
        - plot_graph
        - clip_graph
        - remove_isolated_components
