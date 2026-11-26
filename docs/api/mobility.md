---
seo_title: "OD Matrix to Graph API for Mobility Flows - City2Graph"
description: "Convert origin-destination edge lists and adjacency matrices into weighted spatial graphs for migration, commuting and bike-share flow analysis."
keywords: OD matrix, origin-destination, mobility flows, migration, commuting, human mobility, od_matrix_to_graph, flow networks, spatial interaction
---

# Mobility Module

The mobility module provides functions for processing Origin-Destination (OD) matrices and creating mobility networks for urban flow analysis.

`od_matrix_to_graph` accepts an edge list or an adjacency matrix together with zone geometries and returns a weighted spatial graph of migration, commuting, or bike-sharing flows. The [OD matrix tutorial](../examples/generating_graphs_from_od_matrix.ipynb) builds one from 2021 census migration data for England and Wales.

::: city2graph.mobility
    options:
      show_root_heading: false
      members:
        - od_matrix_to_graph
