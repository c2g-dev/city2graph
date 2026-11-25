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

Pass `interpolate_stop_times=True` to `get_od_pairs()` or
`travel_summary_graph()` to estimate missing intermediate stop times. For example,
if a trip leaves A at 08:00 and reaches C at 08:10 via an untimed stop B,
both functions use 08:05 at B when no distances are supplied. With cumulative
distances of 0, 3 and 4 at A, B and C, they use 08:07:30. By default, both
legs are excluded because each lacks a clock needed to calculate its duration.

For a run of untimed stops, the interval between the preceding departure and
following arrival is divided in proportion to `shape_dist_traveled` when all
distances in that gap are finite, never decrease, and the final distance exceeds
the initial distance. Otherwise the entire interval is divided equally among
the adjacent legs. Stops are ordered by numeric `stop_sequence`. Estimated
times are rounded to the nearest second,
with half-seconds rounded up. If just one clock is missing at an intermediate
stop, its counterpart supplies the estimate when consistent with neighbouring
times. The original `stop_times` table retains all supplied values and gaps.

First and last stops, explicit `timepoint=1` stops and stops with a pickup or
drop-off window keep their missing clocks. Estimates cannot cross these gaps
or use reversed anchor times. A missing `timepoint` column permits interpolation
at eligible intermediate stops. A time supplied with `timepoint=0` keeps its
feed provenance. Times such as `24:05:00` belong to the original service day.

When interpolation is enabled, OD rows include the Boolean columns
`departure_interpolated` and `arrival_interpolated`. Summary edges include
`interpolated_frequency`, the part of `frequency` contributed by traversals
with either clock estimated. Both counts use the same service days, headways
and time bounds.

Read `result.attrs["stop_time_diagnostics"]` on the OD frame or the summary
edge frame, or `graph.graph["stop_time_diagnostics"]` on a NetworkX result.
`candidate_legs` counts adjacent legs once per active trip template, before
expanding service days and frequencies or applying time-of-day filters.
`excluded_legs` counts candidates excluded for time-related reasons.
`excluded_by_reason` contains the non-zero counts. Each excluded leg receives
one reason, taking the first applicable entry in the following table. A single
warning summarises the missing-time exclusions for that call.

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
