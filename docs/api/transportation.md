---
description: API reference for the Transportation module. Process GTFS schedules and create public transit network graphs for accessibility analysis and multi-modal routing.
keywords: GTFS, public transport, transit network, bus, train, tram, accessibility analysis, travel summary, multi-modal routing, load_gtfs
---

# Transportation Module

The transportation module provides functions for processing GTFS (General Transit Feed Specification) data and creating transportation network graphs.

::: city2graph.transportation
    options:
      show_root_heading: false
      members:
        - load_gtfs
        - get_od_pairs
        - travel_summary_graph
