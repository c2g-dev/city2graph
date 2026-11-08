---
seo_title: "Proximity Graph API: KNN, Delaunay, Waxman - City2Graph"
description: "Generate KNN, Delaunay, Gabriel, Waxman and contiguity graphs, and bridge or group nodes, for points of interest and zones with City2Graph."
keywords: KNN graph, Delaunay triangulation, Gabriel graph, Waxman graph, contiguity, spatial proximity, bridge_nodes, group_nodes, POI analysis, spatial networks
---

# Proximity Module

The proximity module provides functions for generating proximity-based graph networks from spatial data. These algorithms create edges between geometries based on various spatial relationships.

Apply them to points of interest or polygonal zones under Euclidean, Manhattan, or network distances. Contiguity, bridging, and grouping functions connect layers into multi-relational graphs. The [proximity tutorial](../examples/generating_graphs_by_proximity.ipynb) demonstrates each algorithm on Tokyo points of interest.

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
