"""Tests for benchmark cli."""

from __future__ import annotations

import json
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path

import pytest

from benchmarks import gtfs_performance
from benchmarks import scalability
from benchmarks.common import measure

pytestmark = pytest.mark.filterwarnings("ignore::city2graph.GTFSGenerationWarning")


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("suite", "case"),
    [
        *[("scalability", case) for case in scalability.CASES],
        *[("gtfs_performance", case) for case in gtfs_performance.CASES],
    ],
)
def test_public_workloads(suite: str, case: str) -> None:
    """All registered workloads produce stable outputs across measurement phases."""
    module = scalability if suite == "scalability" else gtfs_performance
    with ExitStack() as stack:
        workload = module.prepare(case, 20 if suite == "scalability" else 2, 42, stack)
        outputs = [
            measure(workload, phase, repeat=2, warmup=1)["output"]
            for phase in ("time", "allocations", "rss")
        ]
        assert outputs[0] == outputs[1] == outputs[2]
        assert workload.inputs


@pytest.mark.parametrize(
    ("suite", "case"), [("scalability", "weighted"), ("gtfs_performance", "load-fixed")]
)
def test_performance_cli(suite: str, case: str, tmp_path: Path) -> None:
    """Workers select a checkout and emit a comparable JSONL record."""
    command = [
        sys.executable,
        "-m",
        f"benchmarks.{suite}",
        "--case",
        case,
        "--size",
        "2",
        "--repeat",
        "1",
        "--warmup",
        "0",
        "--seed",
        "7",
        "--package-root",
        str(ROOT),
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
    row = json.loads(completed.stdout)
    assert row["case"] == case
    assert row["parameters"]["size"] == 2
    assert row["parameters"]["seed"] == 7
    assert row["environment"]["package_root"] == str(ROOT)
    assert row["environment"]["python_hash_seed"] == "7"
    assert len(row["environment"]["source_hash"]) == 64
    assert len(row["metrics"]["samples_seconds"]) == 1
    assert {"median_seconds", "traced_peak_bytes", "process_peak_rss_bytes"} <= row[
        "metrics"
    ].keys()
    path = tmp_path / "result.jsonl"
    path.write_text(completed.stdout)
    comparison = subprocess.run(
        [sys.executable, "-m", "benchmarks.compare", str(path), str(path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(comparison.stdout)["status"] == "matched"
