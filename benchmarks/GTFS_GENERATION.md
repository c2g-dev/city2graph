# GTFS generation validation

This comparison supports issue #242. Results are in
`results/gtfs-generation-2026-11-02.json`; the script is
`gtfs_generation.py`. It is separate from the offline pytest suite.

## Inputs and comparison design

The two reference archives are user-provided local datasets. Their SHA-256
hashes are recorded in every result. They are not redistributed here.

| Reference | Service date | Routes | Trips | Stop occurrences |
|---|---|---|---:|---:|
| MTA subway | 2025-11-19 | GS | 508 | 1,016 |
| MTA subway | 2025-11-19 | 7, 7X | 642 | 12,938 |
| North West | 2025-12-17 | 138633, 402831 (ferry) | 40 | 80 |
| North West | 2025-12-17 | 138614, 77 (tram and bus) | 439 | 10,640 |

All active trips on the selected routes are included, with complete stop
sequences. The script filters the large North West stop-time table in chunks.
The comparison uses both the complete service day and 07:00–10:00, with
directed graphs and the same date/window settings for both feeds.

The exact reconstruction preserves every stop occurrence, relative arrival and
departure offset and first departure. Its stopping patterns, graph edges,
weighted travel times and service counts matched the references in every
selected case. Generated stop times also survived save/load unchanged.

The approximation retains the reference stopping patterns and departure times,
but replaces running times with distance divided by an explicitly chosen speed.
Dwell time is explicitly set to zero. The selected speeds are illustrative,
not fitted operating estimates. Consequently, topology and full-day service
counts are controlled inputs; their agreement is **not evidence of independent
network or timetable recovery from OSM or Overture**.

| Approximation | Distance basis | Speed | Full-day leg-time MAE |
|---|---|---:|---:|
| GS | Reference stop-to-stop geodesic | 25 km/h | 10.00 s |
| GS | Selected Overture rail alignment | 25 km/h | 10.00 s |
| 7, 7X | Reference stop-to-stop geodesic | 30 km/h | 22.52 s |
| North West ferry | Reference stop-to-stop geodesic | 20 km/h | 677.00 s |
| North West ferry | Selected Overture water alignment | 20 km/h | 682.00 s |
| North West tram and bus | Reference stop-to-stop geodesic | 25 km/h | 42.20 s |

The ferry approximation is poor: a constant speed along the selected alignment
does not reproduce the reference journey durations. Differences between service
patterns, possible indirect journeys and operating assumptions need independent
investigation; these results do not identify a single cause. The package keeps
those assumptions explicit rather than treating infrastructure speed as a
validated public transport running time.

JSON results also include relative errors, edge precision/recall, frequency
errors, weakly connected components and matched reachable-pair shortest-path
errors. Paths contain only vehicle-service edges, without transfers or walking.
No universal accuracy threshold is asserted. These are selected route subsets,
not whole-region or whole-mode accuracy evaluations.

## Source snapshots and live checks

The small alignment snapshots in `data/gtfs-*-alignments.geojson` retain the
Overture segment IDs, source attribution and release `2026-10-21.0`. They were
selected explicitly by infrastructure name and checked against the reference
stop locations; coordinates were reversed explicitly for the opposite direction.
The snapshots postdate the 2025 feeds, so this is not a historical reconstruction
of exactly contemporaneous infrastructure.

- NYC: `IRT 42nd Street Line`, segment
  `dc6990f8-8339-4f96-80f9-4c087ba6230d`.
- Mersey: `Mersey Ferry`, segment
  `42ae7e64-f917-4409-acbc-b22eb3ec112b`.

The source metadata attributes these segments to OpenStreetMap. Retain that
attribution when reusing the snapshots; see the
[Overture attribution guidance](https://docs.overturemaps.org/attribution/) and
[OpenStreetMap copyright page](https://www.openstreetmap.org/copyright).
The GeoJSON data is source material, separate from the Python implementation.

A live `load_overture_transport` call successfully fetched the NYC source,
assembled two explicitly directed patterns and generated a feed. With a
07:00–10:00 period and 300-second headway, each direction contributed 36 expected
traversals; estimated leg time was 100 seconds.

In the 2026-11-02 comparison, live OSM queries were attempted against
`overpass-api.de` and `overpass.kumi.systems`. They timed out or returned HTTP 502.
Live OSM end-to-end verification was therefore unconfirmed in that run.
Offline public-loader tests cover
the request, member order, candidate ambiguity, supplementation and shared
generation, using fixed Overpass responses. This network limitation is recorded
in the results rather than reported as a successful live check.

## Reproduction

Run from the repository root. Supply the corresponding local reference archive:

```bash
uv run python benchmarks/gtfs_generation.py \
  --feed /path/to/gtfs_subway.zip --date 20251119 --routes GS \
  --speed-kph 25 --dwell-secs 0 \
  --alignments benchmarks/data/gtfs-nyc-alignments.geojson

uv run python benchmarks/gtfs_generation.py \
  --feed /path/to/itm_north_west_gtfs.zip --date 20251217 \
  --routes 138633 402831 --speed-kph 20 --dwell-secs 0 \
  --alignments benchmarks/data/gtfs-northwest-alignments.geojson
```

Omit `--alignments` to reproduce the stop-to-stop geodesic variants. For the
other two cases, use routes `7 7X` at 30 km/h, or `138614 77` at 25 km/h, on
their respective dates. Keep dwell at zero to match the recorded runs.
The script asserts exact reconstruction and roundtrip equivalence before
emitting its comparison JSON. It does not impose an approximation accuracy gate.

## simple-v2 scenario sensitivity (2026-11-03)

`results/gtfs-sensitivity-2026-11-03.json` records two additional comparisons,
using the existing MTA GS and North West ferry archives. No reference corpus was
collected, and no defaults were fitted. Each archive's SHA-256 and study date are
retained in the result. The exact reconstruction checks still pass before the
scenario results are emitted.

The sensitivity runs deduplicate reference trips by route, direction and ordered
stop IDs, retaining one stopping pattern per signature. They use geodesic leg
distances, zero additional dwell, a 06:00–22:00 departure window and equal shares
of each route-direction frequency among its selected patterns. The subway base
speed is 35 km/h and the ferry base speed is 20 km/h, the provisional model values.
The remaining topology is controlled by the reference; this is not independent
network recovery. Reference dates are explicit, so the synthetic reference week
is not substituted for these comparisons.

| Subset | Speed factor | Mean generated leg time | Leg-time MAE against reference |
|---|---:|---:|---:|
| GS | 0.75 | 95 s | 5 s |
| GS | 1.00 | 71 s | 19 s |
| GS | 1.25 | 57 s | 33 s |
| Ferry | 0.75 | 318 s | 597 s |
| Ferry | 1.00 | 238 s | 677 s |
| Ferry | 1.25 | 191 s | 724 s |

At each speed, the 10-, 15- and 30-minute route-direction scenarios give 192,
128 and 64 total section traversals for GS, and 384, 256 and 128 for the ferry
subset. Those are sums across directed sections, not passenger counts. Section
counts change inversely with headway while the leg times respond to speed.
Rounding to integer seconds can cause small departures from exact inverse speed
scaling. The JSON also reports matched-edge and in-vehicle shortest-path errors.

The ferry errors remain large across these scenarios. The smaller GS error at
one scenario does not validate that speed for other lines or justify fitting the
package default to it. These selected comparisons establish the need to inspect
source journey times and study-specific sensitivity; they do not establish global
empirical accuracy. The scenario factors are **not confidence intervals**.

Reproduce either case with `--sensitivity`, using `--speed-kph 35` for GS or
`--speed-kph 20` for the ferry, without `--alignments`. Explicit full-day reference
reconstruction and its counts remain separate from the synthetic frequency cases.

## simple-v3 multi-city audit and fixes (2026-11-03 to 2026-11-04)

The original audit is preserved in
[`results/gtfs-city-audit-2026-11-03.json`](results/gtfs-city-audit-2026-11-03.json).
It found four API defects: reusing loaded stop tables containing binary geometry,
collisions between explicit and generated trip IDs, rejection of numeric direction
codes `0.0` and `1.0`, and an unhandled overflow for infinite `route_type`.
Public API regression tests now cover all four, including both loaders,
date-specific trip IDs, disjoint frequency periods and report save/load equality.

[`results/gtfs-city-revalidation-2026-11-04.json`](results/gtfs-city-revalidation-2026-11-04.json)
records the rerun using the same reference archives and saved source responses.
Both reference archives and all eight OSM snapshots matched their original
SHA-256 hashes. No new live downloads were required.

| Input and scope | Revalidation outcome |
|---|---|
| MTA and North West GTFS, 14 routes | 98 cases passed, including exact reconstruction of 53 sampled trips |
| OSM, 84 relations across eight cities | 97 generation cases passed; 61 expected source-validation errors unchanged |
| OSM with explicit stop/order supplements | 38 cases passed |
| Overture NYC subway and Mersey ferry snapshots | 8 cases passed |
| Boundary cases using actual MTA stops | 9 generated successfully; 4 invalid inputs raised the expected `ValueError` |

The OSM cities were London, Paris, Berlin, Zurich, Lisbon, Tokyo, Sydney and
Hong Kong. Of the 84 unmodified relations, 41 generated successfully and 43 still
require supplementation or exclusion because of ambiguous stop/platform pairs,
non-point boarding locations or unsupported relation structure. These source
limitations are separate from the four repaired API defects. The explicit repair
cases use ordered PTv2 stop members and point locations, with labelled audit
placeholders for missing names; they do not validate real timetables.

All four defects were checked again using the MTA boundary cases. Existing
successful scenarios retained their counts; all table contents in 203 regenerated
GTFS archives matched the previous run. Expected source errors retained their IDs
and reasons. The full suite passed 1,311 tests with browser checks enabled, and all
31 changed executable lines were covered. Strict documentation build and
pre-commit checks also passed. The small, network-free reproductions can be run
with:

```bash
uv run python benchmarks/gtfs_generation_edge_cases.py
```

Six valid examples should generate feeds; the infinite-route-type example should
raise a `ValueError` identifying the table, row and column. The JSON records
outcomes rather than treating these diagnostic examples as a timetable accuracy
benchmark. The selected routes and cached snapshots do not establish whole-city
coverage or the empirical accuracy of assumed operating conditions.

## Wider world audit (2026-11-04)

[`results/gtfs-world-audit-2026-11-04.json`](results/gtfs-world-audit-2026-11-04.json)
records 39 additional cities across Europe, Asia, Africa, North America, South
America and Oceania. Their 252 selected OSM relations were downloaded with full
members and available parent-route metadata. The previous eight cities were also
replayed, bringing the OSM coverage to 47 cities and 336 selected relations.

| Manual validation | Cases | Result |
|---|---:|---|
| Additional 39 cities, individual and combined OSM inputs | 659 | 456 generated; 203 expected source-structure errors |
| Previous eight cities | 158 | 97 generated; 61 expected source-structure errors |
| Explicit PTv2 stop/order supplementation | 101 | All passed |
| Extended route-type supplements through the OSM loader | 9 | All passed |
| Actual Overture infrastructure from 12 cities | 72 | All passed |
| Official BART and HSL GTFS, 17 sampled routes | 102 | All passed after fixes |

The 1,101 manual cases include mode and pattern selection, mixed valid/invalid
relations, partial fixed departures, overnight windows, a full reference week,
source-tag overrides and report roundtrips. The 456 successful new-city OSM
archives retained identical GTFS table contents after the fixes. The original
eight-city outcomes, including exclusion IDs and reasons, were unchanged.

The official [HSL feed](https://www.hsl.fi/en/hsl/open-data) exposed three related
input-handling problems. Whitespace-only `parent_station` cells were treated as
unresolved identifiers. Extended route types such as `701`, `900` and `109` were
excluded by the corresponding mode selections. Extended tram and suburban rail
codes also fell through to the generic 20 km/h assumption. Empty-cell handling
now accepts whitespace-only optional cells, and one registry maps extended types
for both mode selection and provisional speeds. Original non-blank identifiers
and route-type codes remain unchanged. Missing required fields still raise errors.

With source timing offsets deliberately removed from the HSL samples, tram `900`
now uses the existing tram baseline of 25 km/h and suburban rail `109` uses the
existing rail baseline of 60 km/h. Exact reconstruction of 50 sampled BART/HSL
trips passed independently of these assumptions. The
[BART archive](https://www.bart.gov/schedules/developers/gtfs) and HSL archive,
their retrieval date and hashes are recorded in the JSON.

The Overture checks used real road segments from Amsterdam, Helsinki, Seoul,
Singapore, Cairo, Cape Town, San Francisco, Toronto, Bogotá, São Paulo,
Melbourne and Wellington. Services and stop positions on those segments were
explicit audit inputs. The separate GTFS checks retained reference route and
stop order. Source completeness and empirical timetable accuracy remain outside
this validation's scope.

These cases remain a manual audit, with no additions to the pytest suite. The
existing 1,311 tests passed with browser checks enabled, along with the strict
documentation build and pre-commit checks. Raw snapshots, generated archives and diagnostic scripts
are retained in the local evidence directory recorded in the result; they are
excluded from distribution. Acquisition failures and corrections to the audit
scripts are recorded separately from package defects.

## Internal refactor and initial public model (2026-11-04)

The current, unreleased generation specification is `simple-v1`. The historical
model labels and measurements above remain as recorded. The implementation now
has six modules under `city2graph._gtfs`, with dependencies following the order
documented in the [contributor guide](../docs/contributing.md). Resolved services
retain their adopted values, origins and frequency scope together. Stop ordering,
source conditions, geometry preparation and calendars are shared within each
generation call; reports reuse the transport edge aggregation without building
unused nodes.

[`results/gtfs-refactor-2026-11-04.json`](results/gtfs-refactor-2026-11-04.json)
records comparisons against a source snapshot taken before this refactor. All
1,101 cached world-audit cases were replayed with network connections blocked:
837 generated successfully and 264 retained their expected source-validation
errors. An additional 17 setup generations brought the comparison to 1,118
public API calls. These cases remain outside the normal test suite.

GTFS table columns, dtypes, values and generated IDs matched exactly. History,
exclusion reasons, warnings and exceptions also matched after accounting for the
current model label and fixing the retrieval clock. Caller-owned input tables
remained unchanged. Two Sydney reports differed only in floating-point edge
travel-time sums, by at most `5.7e-14` seconds. Twelve repeated reports from the
unchanged baseline reproduced those variations. Only that aggregate column uses
an absolute comparison tolerance of `1e-12` seconds; all other fields are compared
exactly. Saved arbitrary model labels and incomplete historical metadata retain
their recorded values through the public persistence tests.

The timing comparison used one warm-up followed by seven measured repetitions,
with generation and reporting timed separately. Before/after workers ran
sequentially after the other checks completed. Each worker also recorded its
peak resident memory, including imports and output verification. BART inputs
contain 1, 32 or 128 complete reference trips represented as patterns; the Kuala
Lumpur input is the saved ten-pattern OSM response.

| Input | Generation median, before → after (s) | Report median, before → after (s) | Peak memory change |
|---|---:|---:|---:|
| BART, 1 pattern | 0.1199 → 0.1201 | 0.0412 → 0.0362 | −0.28% |
| BART, 32 patterns | 0.1946 → 0.1417 | 0.0660 → 0.0618 | −0.23% |
| BART, 128 patterns | 0.4270 → 0.2052 | 0.1426 → 0.1370 | +0.30% |
| Kuala Lumpur, 10 patterns | 0.4636 → 0.3082 | 0.1059 → 0.1017 | +0.20% |

The 128-pattern generation median fell by 52.0%. No measured generation time,
report time or peak memory increased by more than 10%. These are local cached
input measurements, excluding network latency. Invocation profiling separately
confirmed one source-condition parse, one shape preparation and one report
service-day aggregation per call. The shared calendars were evaluated once;
exact BART timings avoided stop-coordinate construction. Kuala Lumpur coordinates
were prepared once for generation, and shared source stops were validated once
before the separate generation-boundary validation.

All 1,311 tests passed with `CITY2GRAPH_DOCS_SITE=site`, including browser checks.
Total package coverage was 97%; the JSON records coverage for each GTFS module.
The strict documentation build, all pre-commit hooks, explicit checks of the new
package files, wheel resource inspection and installed-wheel generation/report
save-load smoke test passed. Raw comparison outputs, profiling scripts and logs
are retained in the local evidence directory recorded in the JSON. These checks
retain the world audit's selected-route scope and do not establish timetable
accuracy or city-wide source completeness.
