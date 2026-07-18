---
description: API reference for loading GTFS public transport and GBFS shared-mobility feeds, deriving OD pairs, and creating travel-summary graphs.
---

# Transportation Module

The transportation module loads
[GTFS](https://gtfs.org/documentation/schedule/reference/) public transport
archives and local
[GBFS](https://gbfs.org/) shared-mobility JSON feeds into DuckDB. It also
derives stop-to-stop origin–destination records and aggregates GTFS schedules
into transportation network graphs.

::: city2graph.transportation
    options:
      show_root_heading: false
      members:
        - load_gtfs
        - load_gbfs
        - get_od_pairs
        - travel_summary_graph
