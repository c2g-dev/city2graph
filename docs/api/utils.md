---
seo_title: "Utils API: Isochrones, Tessellation, rustworkx - City2Graph"
description: "Utility functions for GeoDataFrame, NetworkX and rustworkx conversion, graph validation, tessellation, isochrone creation and plotting in City2Graph."
keywords: gdf_to_nx, nx_to_gdf, rustworkx, tessellation, isochrone, dual_graph, plot_graph, graph validation, spatial utilities, canonicalize_edges, symmetrize_edges
---

# Utils Module

The utils module provides core utility functions for graph conversion, validation, and visualization.

It converts between GeoDataFrames, NetworkX, and rustworkx, validates graph inputs, creates tessellations, and computes isochrones for accessibility analysis.

For buffer isochrones, see [Filling holes in buffer isochrones](../examples/isochrone_holes.md)
for the optional `fill_holes` setting, a before-and-after figure, and clipping guidance.

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
        - canonicalize_edges
        - symmetrize_edges
        - dual_graph
        - filter_graph_by_distance
        - create_tessellation
        - create_isochrone
        - plot_graph
        - clip_graph
        - remove_isolated_components
