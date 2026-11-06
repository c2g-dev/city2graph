"""Exercise benchmark commands and reference checks without performance thresholds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString

import city2graph as c2g
from benchmarks import gtfs_generation
from benchmarks import gtfs_performance
from benchmarks import scalability
from benchmarks.common import measure
from benchmarks.compare import compare_results

pytestmark = pytest.mark.filterwarnings("ignore::city2graph.GTFSGenerationWarning")
ROOT = Path(__file__).parents[1]


def reference_feed(tmp_path: Path, **changes: pd.DataFrame) -> Path:
    """Create a scheduled reference with leading-zero IDs and an overnight trip."""
    data, services, agency = gtfs_performance.synthetic_input(1, 42, "fixed")
    data["routes"]["route_id"] = "001"
    data["patterns"]["route_id"] = "001"
    services["departure_time"] = "24:00:00"
    with c2g.create_gtfs(data, services, agency=agency, service_date="20000103") as con:
        path = c2g.save_gtfs(con, tmp_path / "input.zip", include_metadata=False, overwrite=True)
    with ZipFile(path) as archive:
        tables = {name: archive.read(name) for name in archive.namelist()}
    tables.update(
        {name + ".txt": frame.to_csv(index=False).encode() for name, frame in changes.items()}
    )
    with ZipFile(path, "w") as archive:
        for name, content in tables.items():
            archive.writestr(name, content)
    return path


def evaluation_args(path: Path) -> argparse.Namespace:
    """Specify every comparison condition independently of parser defaults."""
    return argparse.Namespace(
        feed=path,
        date="20000103",
        routes=["001"],
        speed_kph=25.0,
        dwell_secs=0.0,
        alignments=None,
        sensitivity=True,
        window=[["01:00:00", "02:00:00"], ["24:00:00", "26:00:00"]],
        speed_factors=[1.0],
        headways=[600],
        service_start="23:00:00",
        service_end="26:00:00",
    )


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


def test_reference_cli(tmp_path: Path) -> None:
    """Calendar-dates-only archives, overnight times and empty windows are supported."""
    archive = reference_feed(tmp_path)
    command = [
        sys.executable,
        "-m",
        "benchmarks.gtfs_generation",
        "--feed",
        str(archive),
        "--date",
        "20000103",
        "--routes",
        "001",
        "--speed-kph",
        "25",
        "--dwell-secs",
        "0",
        "--window",
        "01:00:00",
        "02:00:00",
        "--window",
        "24:00:00",
        "26:00:00",
        "--sensitivity",
        "--speed-factors",
        "1",
        "--headways",
        "600",
        "--service-start",
        "23:00:00",
        "--service-end",
        "26:00:00",
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
    row = json.loads(completed.stdout)
    assert all(row["checks"].values())
    empty = row["metrics"]["windows"]["01:00:00-02:00:00"]["exact"]
    assert empty["reference_edges"] == 0
    assert empty["travel_time_mae_sec"] is None
    assert row["metrics"]["windows"]["24:00:00-26:00:00"]["exact"]["reference_edges"] == 19
    assert row["parameters"]["service_start"] == "23:00:00"
    assert len(row["metrics"]["sensitivity"]) == 1


def test_external_alignment(tmp_path: Path) -> None:
    """Evaluate supplied shapes with a temporary synthetic GeoJSON input."""
    args = evaluation_args(reference_feed(tmp_path))
    args.alignments = tmp_path / "alignment.geojson"
    args.sensitivity = False
    gpd.GeoDataFrame(
        {"route_id": ["001"], "direction_id": [0]},
        geometry=[LineString([(-0.12, 51.5), (-0.10, 51.5)])],
        crs=4326,
    ).to_file(args.alignments, driver="GeoJSON")
    result = gtfs_generation.compare(args)
    assert result["parameters"]["distance_method"] == "shape"
    assert len(result["inputs"]["alignments"]) == 64
    assert all(result["checks"].values())


def test_calendar_exceptions(tmp_path: Path) -> None:
    """Date exceptions add and remove services from a weekly calendar."""
    path = reference_feed(tmp_path)
    with ZipFile(path) as archive:
        service = pd.read_csv(archive.open("trips.txt"), dtype=str).service_id.iloc[0]
    weekly = pd.DataFrame(
        [
            {
                "service_id": service,
                "start_date": "20000101",
                "end_date": "20000110",
                **dict.fromkeys(
                    ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"),
                    1,
                ),
            }
        ]
    )
    exceptions = pd.DataFrame([{"service_id": service, "date": "20000103", "exception_type": 2}])
    path = reference_feed(tmp_path, calendar=weekly, calendar_dates=exceptions)
    with pytest.raises(ValueError, match="No active reference trips"):
        gtfs_generation.reference_tables(path, "20000103", ["001"])
    weekly["monday"] = 0
    exceptions["exception_type"] = 1
    path = reference_feed(tmp_path, calendar=weekly, calendar_dates=exceptions)
    assert len(gtfs_generation.reference_tables(path, "20000103", ["001"])["trips"]) == 1


@pytest.mark.parametrize("kind", ["frequency", "missing_time"])
def test_unsupported_reference_inputs(tmp_path: Path, kind: str) -> None:
    """Unsupported reference semantics produce an actionable error."""
    path = reference_feed(tmp_path)
    with ZipFile(path) as archive:
        times = pd.read_csv(archive.open("stop_times.txt"), dtype=str)
    if kind == "frequency":
        frequencies = pd.DataFrame(
            [
                {
                    "trip_id": times.trip_id.iloc[0],
                    "start_time": "06:00:00",
                    "end_time": "22:00:00",
                    "headway_secs": 600,
                }
            ]
        )
        path = reference_feed(tmp_path, frequencies=frequencies)
        message = "selected trips use frequencies"
    else:
        times.loc[1, "arrival_time"] = ""
        path = reference_feed(tmp_path, stop_times=times)
        message = "complete arrival_time"
    with pytest.raises(ValueError, match=message):
        gtfs_generation.reference_tables(path, "20000103", ["001"])


def test_zero_duration_metrics() -> None:
    """Zero-duration edges retain absolute errors but have no relative denominator."""
    edges = gpd.GeoDataFrame(
        {"travel_time_sec": [0.0, 10.0], "frequency": [1.0, 1.0]},
        geometry=[None, None],
        index=pd.MultiIndex.from_tuples([("A", "B"), ("B", "C")]),
    )
    candidate = edges.copy()
    candidate["travel_time_sec"] = [5.0, 20.0]
    metrics = gtfs_generation.graph_metrics(edges, candidate)
    assert metrics["relative_error_edges"] == 1
    assert metrics["relative_error_excluded_edges"] == 1
    assert metrics["travel_time_mean_relative_error"] == 1.0
    assert metrics["travel_time_mae_sec"] == 7.5
    zero = gtfs_generation.graph_metrics(edges.iloc[:1], edges.iloc[:1])
    assert zero["travel_time_mean_relative_error"] is None
    assert zero["travel_time_mae_sec"] == 0
    json.dumps(zero, allow_nan=False)


def test_comparison_conditions(tmp_path: Path) -> None:
    """Different inputs cannot silently produce a before/after ratio."""
    baseline: dict[str, Any] = {
        "schema_version": 1,
        "suite": "test",
        "case": "example",
        "parameters": {"size": 2},
        "inputs": {"synthetic": "a"},
        "environment": {"python": "3.12"},
        "metrics": {"count": 0},
        "checks": {},
    }
    current = {**baseline, "inputs": {"synthetic": "b"}}
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text(json.dumps(baseline) + "\n")
    b.write_text(json.dumps(current) + "\n")
    assert {row["status"] for row in compare_results(a, b)} == {
        "missing_baseline",
        "missing_current",
    }
    current = {**baseline, "environment": {"python": "3.13"}, "metrics": {"count": 1}}
    b.write_text(json.dumps(current) + "\n")
    row = compare_results(a, b)[0]
    assert row["environment_differences"] == ["python"]
    assert row["metrics"]["count"] == {"baseline": 0, "current": 1, "delta": 1, "ratio": None}
    b.write_text(json.dumps(current) + "\n" + json.dumps(current) + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        compare_results(a, b)
