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

For generation, supply only the operating conditions you want to control.
Missing values use interpretable source data, then documented assumptions.
See the [input decision flow](../examples/creating_gtfs.md#2-decide-what-to-enter-and-what-to-leave-automatic)
for field-by-field choices, including fixed departures and the distinction
between per-pattern and shared-group intervals.

::: city2graph.transportation
    options:
      show_root_heading: false
      members:
        - TransportData
        - create_gtfs
        - gtfs_generation_report
        - save_gtfs
        - load_gtfs
        - load_gbfs
        - get_od_pairs
        - travel_summary_graph

::: city2graph.transportation.GTFSGenerationWarning
