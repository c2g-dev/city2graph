---
description: API reference for the Proximity module. Generate spatial graphs using KNN, Delaunay, Gabriel, Waxman algorithms, contiguity relations, and node bridging for POI and zone analysis.
keywords: KNN graph, Delaunay triangulation, Gabriel graph, Waxman graph, contiguity, spatial proximity, bridge_nodes, group_nodes, POI analysis, spatial networks
---

# Proximity Module

The proximity module provides functions for generating proximity-based graph networks from spatial data. These algorithms create edges between geometries based on various spatial relationships.

::: city2graph.proximity
    options:
      show_root_heading: false
      members:
        - knn_graph
        - delaunay_graph
        - gabriel_graph
        - relative_neighborhood_graph
        - euclidean_minimum_spanning_tree
        - fixed_radius_graph
        - waxman_graph
        - bridge_nodes
        - group_nodes
        - contiguity_graph
