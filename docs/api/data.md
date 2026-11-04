---
description: API reference for the Data module. Load and process geospatial data from Overture Maps, including buildings, roads, places, and administrative boundaries for graph construction.
keywords: Overture Maps, load_overture_data, buildings, roads, places, segments, geospatial data, OpenStreetMap alternative, urban data
---

# Data Module

The data module provides functions for loading and processing geospatial data from various sources, with a focus on [Overture Maps](https://overturemaps.org/) data.

Transport loaders accept `modes="bus"` or `modes=["bus", "train"]`.
OSM can supply stopping patterns; Overture requires caller-supplied routes and
stop sequences. Both use the shared [GTFS generation API](transportation.md).
Explicit operating conditions take priority; omitted fields use interpretable
source values, then documented assumptions. The
[input decision flow](../examples/creating_gtfs.md#2-decide-what-to-enter-and-what-to-leave-automatic)
shows what to supply for each condition and what happens when it is omitted.
OSM selection also applies to supplemental patterns. Missing source route names
can be repaired in `data`; malformed explicit values always raise, even with
`on_invalid="skip"`.

::: city2graph.data
    options:
      show_root_heading: false
      members:
        - load_osm_transport
        - load_overture_transport
        - load_overture_data
        - process_overture_segments
        - get_boundaries
