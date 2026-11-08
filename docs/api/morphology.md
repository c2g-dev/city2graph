---
seo_title: "Morphological Graph API for Urban Form - City2Graph"
description: "Build heterogeneous morphological graphs linking buildings, tessellation cells and street segments from Overture Maps or OpenStreetMap data in Python."
keywords: urban morphology, morphological graph, buildings, streets, tessellation, dual graph, momepy, urban form analysis, spatial structure
---

# Morphology Module

The morphology module provides functions for creating morphological graphs from urban form data, capturing spatial relationships between place spaces (e.g., tessellation cells) and movement spaces (e.g., street segments).

The input is a set of buildings and street segments from Overture Maps, OpenStreetMap, or any GIS layer. The output is a heterogeneous graph for urban form analysis, walkability studies, or graph neural network embeddings. See the [morphological graph tutorial](../examples/morphological_graph_from_overturemaps.ipynb) for a complete Liverpool example.

::: city2graph.morphology
    options:
      show_root_heading: false
      members:
        - morphological_graph
        - morphological_graphs
        - place_to_place_graph
        - place_to_movement_graph
        - movement_to_movement_graph
        - segments_to_graph
