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

## Missing intermediate stop times

With `interpolate_stop_times=True`, `get_od_pairs()` and `travel_summary_graph()`
estimate missing intermediate clocks from the preceding departure and following arrival.
For A (08:00) → B → C (08:10), cumulative `shape_dist_traveled` values of 0, 3 and 4
give 08:07:30 at B. When any distance in the gap is missing or non-finite, decreases,
or the endpoints have equal distances, the interval is divided equally by stop count
(08:05 at B). Estimates use numeric `stop_sequence` order and round half-seconds up.

A one-sided intermediate clock supplies its counterpart when consistent with its neighbours.
Missing clocks at endpoints, exact timepoints and stops with pickup/drop-off windows
block interpolation across the gap. Extended hours such as `24:05:00` retain their service day.
The input `stop_times` table keeps its original values.

OD rows flag estimated clocks in `departure_interpolated` and `arrival_interpolated`.
Summary edges report their contribution to `frequency` in `interpolated_frequency`,
with the same service-day, headway and time-window weights.

The OD frame and summary edges expose `attrs["stop_time_diagnostics"]` (on a NetworkX
result, `graph.graph["stop_time_diagnostics"]`).
`candidate_legs` and `excluded_legs` count adjacent legs per active trip template
before service expansion and time-of-day filtering. Each exclusion contributes once
to `excluded_by_reason`, using the priority below. Missing clocks produce one warning per call.

| Reason | Meaning |
| --- | --- |
| `interpolation_disabled` | A required leg clock is unavailable and interpolation is off. |
| `required_time_missing` | A clock is missing at a trip endpoint or an exact timepoint. |
| `pickup_drop_off_window` | The stop specifies a pickup or drop-off window. |
| `inconsistent_times` | The clocks conflict or an OD clock cannot be parsed. |
| `missing_anchor` | A bounding time is unavailable for interpolation. |
| `nonpositive_duration` | The summary graph excludes a leg with zero or negative duration. |

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
