---
description: API reference for loading GTFS public transport and GBFS shared-mobility feeds, deriving OD pairs, and creating travel-summary graphs.
---

# Transportation Module

The transportation module creates synthetic GTFS feeds from explicit stopping
patterns, reports their assumptions, saves and restores them, and loads
[GTFS](https://gtfs.org/documentation/schedule/reference/) public transport
archives and local
[GBFS](https://gbfs.org/) shared-mobility JSON feeds into DuckDB. It also
derives stop-to-stop origin–destination records and aggregates GTFS schedules
into transportation network graphs.

See the [tutorial on creating GTFS from OpenStreetMap](../examples/creating_gtfs.md) for a first feed and
the [generation reference](gtfs_generation.md) for field precedence, source-tag
interpretation, partial service tables and default values.

::: city2graph.transportation
    options:
      show_root_heading: false
      members:
        - TransportData
        - create_gtfs
        - get_gtfs_summary
        - save_gtfs
        - load_gtfs
        - load_osm_transport
        - load_gbfs
        - get_od_pairs
        - travel_summary_graph

::: city2graph.transportation.GTFSGenerationWarning
