---
seo_title: "GTFS and GBFS API: Transit Graphs, Synthetic GTFS - City2Graph"
description: "Load GTFS and GBFS feeds into DuckDB, generate synthetic GTFS from OpenStreetMap, derive OD pairs and build travel-summary graphs with City2Graph."
---

# Transportation Module

The transportation module creates synthetic GTFS feeds from explicit stopping
patterns, reports their assumptions, saves and restores them, and loads
[GTFS](https://gtfs.org/documentation/schedule/reference/) public transport
archives and local
[GBFS](https://gbfs.org/) shared-mobility JSON feeds into DuckDB. It also
derives stop-to-stop origin–destination records and aggregates GTFS schedules
into transportation network graphs.

Detailed `get_od_pairs()` rows always retain the actual departure and arrival
stops, chronological timestamps and trip-oriented geometry. Its `directed`
parameter remains accepted for call compatibility; `directed=False` no longer
relabels reverse trips with a canonical stop pair. For canonical undirected
edges that combine both trip directions, use `travel_summary_graph(directed=False)`.

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
