# Utils Module

The utils module provides core utility functions for graph conversion, validation, and visualization.

## Conversion Functions

::: city2graph.utils
    options:
      show_root_heading: false
      members:
        - gdf_to_nx
        - nx_to_gdf
        - nx_to_rx
        - rx_to_nx

## Validation Functions

::: city2graph.utils
    options:
      show_root_heading: false
      members:
        - validate_gdf
        - validate_nx

## Graph Operations

::: city2graph.utils
    options:
      show_root_heading: false
      members:
        - dual_graph
        - filter_graph_by_distance

## Spatial Operations

::: city2graph.utils
    options:
      show_root_heading: false
      members:
        - create_tessellation
        - create_isochrone

## Visualization

::: city2graph.utils
    options:
      show_root_heading: false
      members:
        - plot_graph
