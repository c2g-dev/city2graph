---
seo_title: "Load Overture Maps Data in Python - City2Graph Data API"
description: "Load Overture Maps buildings, roads, places and boundaries as GeoDataFrames with City2Graph, ready for morphological and transport graph construction."
keywords: Overture Maps, load_overture_data, buildings, roads, places, segments, geospatial data, OpenStreetMap alternative, urban data
---

# Data Module

The data module provides functions for loading and processing geospatial data from various sources, with a focus on [Overture Maps](https://overturemaps.org/) data.

It returns buildings, road segments, places, and administrative boundaries for a place name or bounding box as GeoDataFrames, which feed directly into the morphology and transportation builders. The [morphology tutorial](../examples/morphological_graph_from_overturemaps.ipynb) loads Liverpool data this way.

Public transport loaders and GTFS generation are documented in the
[Transportation module](transportation.md).

::: city2graph.data
    options:
      show_root_heading: false
      members:
        - load_overture_data
        - process_overture_segments
        - get_boundaries
