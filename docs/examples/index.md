---
description: Python tutorials for urban network analysis: Heterogeneous GNNs, OpenStreetMap (OSM) integration, GTFS to Graph conversion, and Origin-Destination (OD) matrix processing for spatial AI.
keywords: City2Graph Examples, Graph Neural Networks, Metapath, Urban Morphology, GTFS, OD Matrix, Spatial Proximity, Urban Analysis, Python Tutorial
---

# Examples

This section provides examples of how to use City2Graph in various urban analysis scenarios.

| Example | Description |
| :--- | :--- |
| [Metapath Construction for Heterogeneous GNNs](add_metapaths.ipynb) | Demonstrates the workflow of metapath construction from a heterogeneous graph. The example creates a dual graph of streets, connects amenities, and materializes metapaths between amenities via streets for **Heterogeneous Graph Neural Networks (HGNNs)**. |
| [Morphological Graphs from Overture Maps & OpenStreetMap](morphological_graph_from_overturemaps.ipynb) | Create morphological graphs from **Overture Maps** and **OpenStreetMap (OSM)**, capturing relationships between public (streets) and private (tessellations/buildings) spaces for urban form analysis. |
| [GTFS to Public Transit Networks as Graphs](gtfs.ipynb) | Transforms **General Transit Feed Specification (GTFS)** schedules into intuitive graph representations for **public transport accessibility analysis** and multi-modal network visualization. |
| [OD Matrices to Mobility Networks as Graphs](generating_graphs_from_od_matrix.ipynb) | Demonstrates the conversion of **Origin-Destination (OD) Matrices** into mobility networks, supporting both edge-list and adjacency formats for modeling **human mobility flows**. |
| [Spatial Proximity Graphs](generating_graphs_by_proximity.ipynb) | Illustrates how to generate and visualize different spatial graph types based on proximity metrics (**KNN, Delaunay Triangulation, Gilbert, Waxman**), bridging and grouping nodes for **spatial connectivity**. |

## External Links

| Title | Author | Language | Type | Release |
| :--- | :--- | :--- | :--- | :--- |
| [City2Graph: Python package for spatial network analysis and GeoAI with GNNs](https://medium.com/@yuta.sato.now/city2graph-a-python-package-for-spatial-network-analysis-and-graph-neural-networks-gnns-bc943dd6d85e) | Yuta Sato | EN | Tutorial | Sep 23, 2025 |
| [I created a Python library that converts geospatial data into graph representations for heterogeneous GNNs](https://zenn.dev/yutasato/articles/9d7994dc53d378) | Yuta Sato | JA | Tutorial | Oct 14, 2025 |
| [Verifying Hiroshima Station Redevelopment with Network Science using city2graph](https://nttdocomo-developers.jp/entry/2025/12/22/090000_6_2) | Koki Eguchi | JA | Blog | Dec 22, 2025 |

*Note*: We welcome external examples! Please submit [a pull request](https://github.com/c2g-dev/city2graph/pulls) if you have an example to share.
