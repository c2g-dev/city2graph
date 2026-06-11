---
description: API reference for the Morphology module. Create morphological graphs from urban form data capturing relationships between buildings, streets, tessellations, and movement spaces.
keywords: urban morphology, morphological graph, buildings, streets, tessellation, dual graph, momepy, urban form analysis, spatial structure
---

# Morphology Module

The morphology module provides functions for creating morphological graphs from urban form data, capturing spatial relationships between place spaces (e.g., tessellation cells) and movement spaces (e.g., street segments).

::: city2graph.morphology
    options:
      show_root_heading: false
      members:
        - morphological_graph
        - place_to_place_graph
        - place_to_movement_graph
        - movement_to_movement_graph
        - segments_to_graph

## Deprecated

The following aliases emit a `DeprecationWarning` and delegate to the renamed functions above:

- `private_to_private_graph` → `place_to_place_graph`
- `private_to_public_graph` → `place_to_movement_graph`
- `public_to_public_graph` → `movement_to_movement_graph`
