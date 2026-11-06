"""Compare compatible cases in two versioned benchmark JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.common import SCHEMA_VERSION
from benchmarks.common import emit

# Records use workload-specific metrics and nested input conditions.


def read_results(path: Path) -> dict[str, dict[str, Any]]:
    """Index results by suite, case, parameters and input hashes."""
    records = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["schema_version"] != SCHEMA_VERSION:
            msg = f"Unsupported result schema in {path}"
            raise ValueError(msg)
        key = json.dumps(
            {field: row[field] for field in ("suite", "case", "parameters", "inputs")},
            sort_keys=True,
        )
        if key in records:
            msg = f"Duplicate benchmark case in {path}: {row['case']}"
            raise ValueError(msg)
        records[key] = row
    return records


def numeric_metrics(metrics: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """Flatten scalar metrics while leaving samples and labels intact in source files."""
    values = {}
    for name, value in metrics.items():
        key = f"{prefix}.{name}" if prefix else name
        if isinstance(value, dict):
            values.update(numeric_metrics(value, key))
        elif isinstance(value, list) and all(isinstance(item, dict) for item in value):
            for index, item in enumerate(value):
                values.update(numeric_metrics(item, f"{key}.{index}"))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            values[key] = float(value)
    return values


def compare_results(baseline: Path, current: Path) -> list[dict[str, Any]]:
    """Report matched deltas and all cases with incompatible or missing conditions."""
    before, after = read_results(baseline), read_results(current)
    results = []
    for key in sorted(before.keys() | after.keys()):
        identity = json.loads(key)
        if key not in before or key not in after:
            results.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    **identity,
                    "status": "missing_baseline" if key not in before else "missing_current",
                }
            )
            continue
        left, right = before[key], after[key]
        differences = [
            field
            for field in ("python", "platform", "machine", "dependencies", "python_hash_seed")
            if left["environment"].get(field) != right["environment"].get(field)
        ]
        a, b = numeric_metrics(left["metrics"]), numeric_metrics(right["metrics"])
        results.append(
            {
                "schema_version": SCHEMA_VERSION,
                **identity,
                "status": "matched",
                "environment_differences": differences,
                "metrics": {
                    name: {
                        "baseline": a[name],
                        "current": b[name],
                        "delta": b[name] - a[name],
                        "ratio": b[name] / a[name] if a[name] else None,
                    }
                    for name in sorted(a.keys() & b.keys())
                },
                "missing_metrics": {
                    "baseline": sorted(b.keys() - a.keys()),
                    "current": sorted(a.keys() - b.keys()),
                },
                "checks": {"baseline": left["checks"], "current": right["checks"]},
            }
        )
    return results


def main() -> None:
    """Print case comparisons without imposing performance thresholds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("current", type=Path)
    args = parser.parse_args()
    for result in compare_results(args.baseline, args.current):
        emit(result)


if __name__ == "__main__":
    main()
