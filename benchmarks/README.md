# Scalability refactoring

These measurements support [#266](https://github.com/c2g-dev/city2graph/issues/266).
Correctness tests remain under `tests/`; benchmarks have no timing assertions and
are not part of pytest.

## Reproduce the measurements

Run from a development environment with the CPU extra installed:

```bash
uv run python benchmarks/scalability.py > current.jsonl
uv run python benchmarks/scalability.py --package-root /path/to/baseline-checkout > baseline.jsonl
```

Use the same environment and the baseline checkout at `8cffe2d`. Each case and
input size runs in a fresh process. Run the two commands sequentially on an idle
machine. A single case can be selected with `--case weighted --size 4000`.

The JSON lines include:

- Median elapsed time of three runs without instrumentation.
- Peak Python allocations from a separate `tracemalloc` run. This is not RSS and
  excludes some allocations made by native libraries.
- Process peak RSS, including imported libraries, inputs and outputs. On macOS
  the native RSS value is bytes; on Linux the script converts KiB to bytes.
- Maximum returned distance-array size, maximum intermediate join row count,
  network traversal count, complete-path node references and enriched edge count.
- Python, operating system and dependency versions, plus the imported package path.

Point samples use NumPy generator seed 42 and the unit square, with EPSG:3857.
Weighted metapaths use a unit-cost chain with every node as an endpoint and a
threshold of 1. Ordinary metapaths use layers of sizes `1, n, 1, n, 1` and complete
connections between adjacent layers. Network k-nearest neighbours use a road
chain and 20 points at its first 20 nodes, with `k=1`. Waxman uses `beta=0.00001`,
`r0=1` and seed 42. MST uses Manhattan distance. Gabriel and relative neighbourhood
measurements use the same seeded point samples.

## Compatibility and remaining costs

Public signatures, defaults and materialised return formats are unchanged. The
refactoring preserves node/edge identity and ordering, CRS, geometry orientation,
metadata, seeded selection and the original exact distance-boundary predicates.
Record batching retains pandas' nullable scalar conversions; feature tensors
continue to own their storage.

- Weighted shortest paths use source batches of at most 256 rows and approximately
  8 MiB per float64 distance array. At least one full distance row is necessary;
  the caller consuming a generator can briefly retain the preceding batch.
- Waxman consumes every original random draw in row order, including discarded
  diagonal and lower-triangle draws. Temporary matrices are bounded, but time is
  still quadratic and dense outputs still require quadratic storage.
- Directed metapath counts and provably exactly representable integer-valued sum/mean
  attributes can be aggregated after each hop. Numeric strings follow the same
  conversion as the original reducers. Floating-point cancellation, missing-value
  means, arbitrary reducers and undirected path-signature deduplication retain
  the existing enumerating implementation. Their path-count-dependent cost remains.
- Network neighbour selection retains one predecessor tree at a time and builds
  only selected geometries. Target layers share each source-layer traversal.
  Network MST selects scalar candidates before reconstructing tree geometry; some
  source traversals are repeated for selected tree edges to avoid retaining all
  predecessor trees.
- MST still retains quadratic scalar candidates for non-Euclidean metrics.
  Legacy candidate-set order is necessary to preserve equal-weight edge selection
  and output order. It no longer constructs a complete attributed NetworkX graph
  or a geometry for every candidate on finite inputs with unique node IDs.
  Duplicate IDs and non-finite inputs retain the legacy NetworkX path, including
  its validation behaviour. String-ID tie order remains dependent on
  Python's hash seed, as in the baseline.
- Morphology shares the nearest-node buffer graph and its distance field across
  requested distances. The separate edge-snapped field and existing shared base
  tessellation retain their original semantics. Building prefiltering and local
  enclosure retries are deliberately unchanged: reducing their context has not
  been shown to preserve boundary geometry.
- Concave-hull intersection arrays are batched. Its neighbour cache has an 8 MiB
  retention budget; query sizes survive eviction so ties are reproduced when an
  array is recomputed. These changes do not make every hull walk linear.
- Overture stdout is spooled to temporary files; GDAL receives a path rather than
  a complete Python GeoJSON string. Temporary files are removed even on failure.
  Saved filenames and formats remain unchanged.
- GTFS timestamp conversion works on unique time/date pairs and maps results back
  in original row order. OD cleaning and masks use row batches. Both APIs still
  return complete tables, whose storage remains proportional to output size.

Correctness changes, including PyG type selection in #253, are outside this
diff; fixes already merged into main are retained.

## Local results (2 November 2026)

Baseline `8cffe2d` and the refactoring used the same environment: Python 3.13.5,
NumPy 2.5.4, pandas 3.1.0 and GeoPandas 1.2.0 on macOS 26.1 arm64.
The final refactoring is based on main at `924f85b`. Weighted-metapath runs
receive a fresh edge dictionary each time, preserving the API's existing input
mutation without allowing one measurement to affect the next. The table shows
audit baseline → refactored values; MB is decimal.

| Workload | Median time (s) | Peak traced allocations (MB) | Process peak RSS (MB) |
| --- | ---: | ---: | ---: |
| Weighted metapaths, 4,000 nodes | 0.0813 → 0.0847 | 131.17 → 18.99 | 679.6 → 567.8 |
| Metapaths, 400 branches | 0.0070 → 0.0072 | 25.85 → 0.14 | 562.5 → 544.6 |
| Network kNN, 2,000 road nodes / 20 points | 0.2699 → 0.0541 | 34.97 → 2.67 | 586.7 → 550.8 |
| Waxman, 4,000 points | 0.2107 → 0.1803 | 449.94 → 34.72 | 1062.6 → 583.2 |
| Manhattan MST, 400 points | 0.6365 → 0.0774 | 55.99 → 21.08 | 688.5 → 582.7 |
| Gabriel, 4,000 points | 0.7276 → 0.2509 | 10.26 → 10.27 | 568.8 → 571.7 |
| Relative neighbourhood, 4,000 points | 1.4380 → 0.2953 | 8.43 → 8.43 | 560.8 → 563.6 |

The weighted distance array decreased from 128 MB to 8.192 MB. The largest
ordinary-metapath join decreased from 160,000 rows to 400. Network kNN used
20 traversals instead of 40, without complete path lists to all reachable nodes.
MST enriched 399 edges instead of 79,800. All 21 workload/size combinations
returned the same output edge counts.

Small ordinary-metapath inputs pay additional aggregation overhead: at 100
branches the median increased from 0.0040 to 0.0074 seconds, while traced peak
allocations fell from 1.67 MB to 0.09 MB. These measurements establish memory
and scaling improvements, not a guarantee that every call is faster.

Full results at all input sizes are in
[`results/scalability-2026-11-02.json`](results/scalability-2026-11-02.json).
Supplementary comparison against main at `924f85b` with `PYTHONHASHSEED=0`
matched all 378 cases, including exact geometry WKB comparisons. Exported
signatures were unchanged for all 47 public functions. The regression suite
also covers batch boundaries, seeded edge selection, tied MST edges, repeated
IDs, internal-looking attribute names, nullable attributes, input ownership,
cache eviction and shared-search validation of empty or invalid layers.

Validation of the final source state:

- `uv run pytest -q`: 1,079 passed, 11 skipped; 98% reported coverage.
- `uv run pre-commit run --all-files`: passed, including type and docstring checks.

The audit baseline was 963 passed, 11 skipped and 99% reported coverage.
The retained dense-matrix compatibility helpers are no longer reached by the
public proximity workloads, contributing to the lower coverage percentage;
tests do not call private helpers solely to raise coverage.
