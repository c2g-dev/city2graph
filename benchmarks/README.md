# Benchmarks

These commands measure public APIs and evaluate synthetic GTFS against supplied
reference data. Run them from the repository root after
`uv sync --extra cpu --group dev`. Results are JSONL on stdout; diagnostic messages
use stderr. Store measurements under the ignored `benchmarks/results/` directory.
Correctness tests live under `tests/` and have no timing thresholds.

## Performance measurements

```bash
mkdir -p benchmarks/results
uv run python -m benchmarks.scalability > benchmarks/results/spatial.jsonl
uv run python -m benchmarks.gtfs_performance > benchmarks/results/gtfs.jsonl
```

Both commands accept `--case`, `--size`, `--seed` (default 42), `--repeat`
(default 3), `--warmup` (default 1) and `--package-root`. An explicit size applies
to every selected case. Select a small case while developing a workload:

```bash
uv run python -m benchmarks.scalability --case weighted --size 100
uv run python -m benchmarks.gtfs_performance --case generate-fixed --size 10
```

Each case, size and measurement phase uses a fresh process with a fixed Python
hash seed. Input preparation and output cleanup are outside the timed operation.
Outputs stay alive until the timer or memory reading has completed.

| Metric | Definition |
| --- | --- |
| `samples_seconds`, `median_seconds` | Uninstrumented wall-clock samples after warm-up. |
| `traced_peak_bytes` | A separate, single operation measured with `tracemalloc`; includes Python allocations but excludes some native allocations. |
| `process_peak_rss_bytes` | A separate, uninstrumented process high-water mark, including imports and prepared inputs. This is total process memory, not incremental operation memory. |
| `output` | Workload-specific row, edge or archive-byte counts, checked across repetitions and phases. |

Run comparisons sequentially on an otherwise idle machine. OS caches, native
allocators, library versions and CPU state can affect measurements. The memory
phases do not include warm-up; the time phase reports warm operation performance.

## Workloads

Spatial point inputs use seeded NumPy samples in a unit square with EPSG:3857.

| Case | Input and operation | Default sizes |
| --- | --- | --- |
| `weighted` | Unit-cost chain; weighted metapaths with threshold 1 and all nodes as endpoints. | 1,000 / 2,000 / 4,000 nodes |
| `paths` | Layers `1, n, 1, n, 1`, fully connected between neighbours; directed metapath aggregation. | 100 / 200 / 400 branches |
| `network` | Road chain and 20 points on its first nodes; network kNN with `k=1`. | 500 / 1,000 / 2,000 road nodes |
| `waxman` | Seeded points; `beta=0.00001`, `r0=1`, seeded edge selection. | 1,000 / 2,000 / 4,000 points |
| `mst` | Seeded points; Manhattan minimum spanning tree. | 100 / 200 / 400 points |
| `gabriel` | Seeded points; Gabriel graph. | 1,000 / 2,000 / 4,000 points |
| `rng` | Seeded points; relative neighbourhood graph. | 1,000 / 2,000 / 4,000 points |

GTFS cases combine `generate`, `save`, `load` or `summary` with `fixed` or
`frequency`, for example `load-frequency`. Sizes are 10, 100 and 1,000 patterns,
each with its own route and the same 20 ordered stops. Seeded leg durations are
40–80 seconds, with zero dwell and an explicit UTC agency. The fixed service
date is 2000-01-03. Fixed trips leave at 08:00; frequency services run 06:00–22:00
with a 600-second headway. Saved feeds include metadata.

Generated input hashes cover the GTFS tables and conditions. Spatial hashes
identify the versioned input recipe, size and seed. Geometry and timetable
correctness are covered by the public API tests; output counts are a lightweight
check that measurements performed the same amount of work.

## Compare results

Use the same harness and dependency environment for both checkouts:

```bash
uv run python -m benchmarks.scalability --package-root /path/to/checkout > benchmarks/results/baseline.jsonl
uv run python -m benchmarks.scalability > benchmarks/results/current.jsonl
uv run python -m benchmarks.compare benchmarks/results/baseline.jsonl benchmarks/results/current.jsonl
```

Records contain a schema version, suite, case, parameters, input hashes,
environment, source commit, dirty flag, source/harness hashes, metrics and checks.
The package root is verified in each worker. Source hashes include uncommitted
Python and registry changes; a commit alone cannot identify a dirty checkout.

The comparison matches suite, case, complete parameters and input hashes.
Unmatched cases are reported as missing. Matched scalar metrics have absolute
differences and current/baseline ratios; a zero baseline has a null ratio.
Environment differences are reported alongside the comparison. No performance
thresholds or universal accuracy criteria are imposed. Retain both original
files for samples and non-scalar results.

## GTFS reference evaluation

Use the reference evaluation below for exact reconstruction, approximation and
sensitivity analysis. Raw inputs belong in the ignored `benchmarks/data/`
directory or outside this checkout. Evaluation results use the same record
envelope, with quality metrics instead of performance timings.

## Add or change a workload

Add a case to the appropriate case registry and provide a `Workload` containing
prepared inputs, the public API operation, an output description and cleanup.
Register long-lived resources with the preparation `ExitStack`; release returned
connections after measurement. Inputs must be reproducible and independent of
previous invocations. Update the input recipe and workload version when its
meaning changes, and document the size unit and fixed conditions here.

Add small correctness and CLI tests under `tests/`. Test result structure,
selection, cleanup and comparable conditions without asserting elapsed time or
memory thresholds. Keep raw feeds and generated measurement files out of Git.

## Reference evaluation

### Run an evaluation

```bash
uv run python -m benchmarks.gtfs_generation \
  --feed /path/to/reference.zip --date 20251119 --routes GS \
  --speed-kph 25 --dwell-secs 0 \
  --window 07:00:00 10:00:00 > benchmarks/results/reference.jsonl
```

Create the results directory first. The whole service day is always evaluated;
repeat `--window START END` to add time windows. Times after midnight use extended
hours, for example `--window 23:00:00 26:00:00`.

The reference selection supports `calendar.txt`, `calendar_dates.txt`, or both.
Date exceptions override the weekly calendar. All active trips on the selected
routes are retained with their complete stop sequences. Large stop-time tables
are filtered in chunks. Identifiers retain their original string values.

Exact reconstruction requires scheduled trips and complete arrival/departure
values. Selected frequency-based trips, missing stop times and empty selections
raise explanatory errors. Frequency-based synthetic generation is measured
separately by the performance suite.

### Comparisons and checks

Exact reconstruction retains stop occurrences, relative arrival/departure offsets
and first departure times. Before reporting success, the command checks all
selected stop IDs, sequences and arrival/departure times, graph edge coverage,
travel times and service counts, then checks saved/reloaded stop-time equality.
Time strings are normalised to extended `HH:MM:SS` notation for comparison.

Approximation retains the reference stopping patterns and departures while using
the requested speed and dwell assumptions to estimate journey times. Its topology
and whole-day counts are controlled inputs. The evaluation measures timing
sensitivity within those conditions; it does not establish independent network
recovery or timetable accuracy from OSM.

Reported metrics include edge precision/recall, matched-edge travel-time and
frequency errors, weakly connected components and shortest-path errors over
matched reachable pairs. Paths use vehicle-service edges without walking or
transfers. Empty windows report zero counts and null undefined metrics. Relative
errors use only matched edges with positive reference travel time; the result
records included and excluded edge counts. Absolute errors include zero-duration
edges. Unmatched edges are reflected in coverage metrics, not assigned invented
travel times.

### Optional alignments

Supply `--alignments /path/to/alignments.geojson` to use explicit route shapes.
The file needs a CRS and one geometry per `route_id` / `direction_id` combination
used by the selected patterns. The result hashes both the feed and alignment
file and records the distance method. Check source dates and suitability for
your study independently.

### Sensitivity analysis

Add `--sensitivity` to deduplicate reference trips by route, direction and ordered
stops, then vary speed and route-direction headway. Defaults are speed factors
`0.75 1 1.25`, headways `600 900 1800` seconds and service hours `06:00:00` to
`22:00:00`. Set these explicitly when comparisons require different conditions:

```bash
uv run python -m benchmarks.gtfs_generation \
  --feed /path/to/reference.zip --date 20251119 --routes GS \
  --speed-kph 25 --dwell-secs 0 --sensitivity \
  --speed-factors 0.8 1 1.2 --headways 300 600 \
  --service-start 05:00:00 --service-end 25:00:00
```

Patterns share each route-direction frequency equally. Sensitivity uses the
chosen dwell value and distance method and compares against whole-day reference
service. Results record every condition, mean leg time, section traversals and
comparison metrics. These are scenarios rather than calibrated estimates or
confidence intervals. Generalising to a city or mode requires suitable reference
coverage and independently justified operating assumptions.
