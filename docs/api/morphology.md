# Morphology Module

The morphology module provides functions for creating morphological graphs from urban fabric data, capturing spatial relationships between buildings, streets, and public spaces.

## Composite Graphs

::: city2graph.morphology
    options:
      show_root_heading: false
      members:
        - morphological_graph

## Component Graphs

::: city2graph.morphology
    options:
      show_root_heading: false
      members:
        - private_to_private_graph
        - private_to_public_graph
        - public_to_public_graph
        - segments_to_graph
