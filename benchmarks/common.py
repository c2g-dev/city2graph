"""Shared result records and isolated measurement of public API workloads."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import shutil
import statistics
import subprocess
import sys
import time
import tracemalloc
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping

# Benchmark records intentionally contain workload-specific JSON values.
# ruff: noqa: T201

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1


def digest(value: object) -> str:
    """Hash a canonical JSON input specification."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_digest(path: Path) -> str:
    """Hash the bytes of an external benchmark input."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def environment(package_root: Path) -> dict[str, Any]:
    """Identify the imported code, source tree and measurement environment."""

    def git(*args: str) -> str | None:
        result = subprocess.run(
            [shutil.which("git") or "/usr/bin/git", "-C", str(package_root), *args],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    versions: dict[str, str | None] = {}
    for name in (
        "city2graph",
        "numpy",
        "pandas",
        "geopandas",
        "shapely",
        "scipy",
        "networkx",
        "duckdb",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    sources = {
        str(path.relative_to(package_root)): file_digest(path)
        for path in sorted((package_root / "city2graph").rglob("*"))
        if path.suffix in {".py", ".json"}
    }
    status = git("status", "--porcelain")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "dependencies": versions,
        "package_root": str(package_root.resolve()),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "source_hash": digest(sources),
        "harness_hash": digest(
            {p.name: file_digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
        ),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
    }


def record(
    suite: str,
    case: str,
    parameters: dict[str, Any],
    inputs: dict[str, str],
    metrics: dict[str, Any],
    checks: dict[str, Any],
    package_root: Path = ROOT,
) -> dict[str, Any]:
    """Construct a versioned, strict-JSON benchmark record."""
    return {
        "schema_version": SCHEMA_VERSION,
        "suite": suite,
        "case": case,
        "recorded_at": datetime.now(UTC).isoformat(),
        "parameters": parameters,
        "inputs": inputs,
        "environment": environment(package_root),
        "metrics": metrics,
        "checks": checks,
    }


def emit(value: dict[str, Any]) -> None:
    """Write one complete JSON record to stdout."""
    print(json.dumps(value, allow_nan=False, sort_keys=True), flush=True)


def positive_int(value: str) -> int:
    """Parse a positive count for command-line arguments."""
    result = int(value)
    if result < 1:
        msg = "expected a positive integer"
        raise argparse.ArgumentTypeError(msg)
    return result


def nonnegative_int(value: str) -> int:
    """Parse a non-negative count or seed."""
    result = int(value)
    if result < 0:
        msg = "expected a non-negative integer"
        raise argparse.ArgumentTypeError(msg)
    return result


@dataclass
class Workload:
    """Prepared inputs, a public API operation and output ownership callbacks."""

    run: Callable[[], Any]
    describe: Callable[[Any], dict[str, int]]
    inputs: dict[str, str]
    release: Callable[[Any], None] = lambda _result: None


def measure(workload: Workload, phase: str, repeat: int, warmup: int) -> dict[str, Any]:
    """Measure one phase, excluding input preparation and output cleanup."""
    if phase == "time":
        for _ in range(warmup):
            result = workload.run()
            workload.release(result)
            del result
        samples = []
        outputs = []
        for _ in range(repeat):
            gc.collect()
            start = time.perf_counter()
            result = workload.run()
            samples.append(time.perf_counter() - start)
            try:
                outputs.append(workload.describe(result))
            finally:
                workload.release(result)
            del result
        if any(output != outputs[0] for output in outputs):
            msg = "Output counts changed between repetitions"
            raise ValueError(msg)
        return {
            "samples_seconds": samples,
            "median_seconds": statistics.median(samples),
            "output": outputs[0],
        }
    gc.collect()
    if phase == "allocations":
        tracemalloc.start()
    try:
        result = workload.run()
        if phase == "allocations":
            peak = tracemalloc.get_traced_memory()[1]
        else:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak = int(rss if sys.platform == "darwin" else rss * 1024)
    finally:
        if phase == "allocations":
            tracemalloc.stop()
    try:
        output = workload.describe(result)
    finally:
        workload.release(result)
    key = "traced_peak_bytes" if phase == "allocations" else "process_peak_rss_bytes"
    return {key: peak, "output": output}


def run_suite(
    suite: str,
    cases: Mapping[str, tuple[int, ...]],
    prepare: Callable[[str, int, int, ExitStack], Workload],
) -> None:
    """Run each case, size and measurement phase in a fresh Python process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=ROOT)
    parser.add_argument("--case", choices=cases)
    parser.add_argument("--size", type=positive_int)
    parser.add_argument("--repeat", type=positive_int, default=3)
    parser.add_argument("--warmup", type=nonnegative_int, default=1)
    parser.add_argument("--seed", type=nonnegative_int, default=42)
    parser.add_argument("--phase", choices=("time", "allocations", "rss"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    package_root = args.package_root.resolve()
    if not (package_root / "city2graph/__init__.py").is_file():
        parser.error("--package-root must contain city2graph/__init__.py")
    if args.phase:
        if args.case is None or args.size is None:
            parser.error("--phase requires --case and --size")
        sys.path.insert(0, str(package_root))
        with ExitStack() as stack:
            workload = prepare(args.case, args.size, args.seed, stack)
            imported = Path(str(sys.modules["city2graph"].__file__)).resolve()
            if not imported.is_relative_to(package_root / "city2graph"):
                msg = f"Unexpected package import: {imported}"
                raise ValueError(msg)
            emit(
                {
                    "inputs": workload.inputs,
                    "metrics": measure(workload, args.phase, args.repeat, args.warmup),
                }
            )
        return
    selected = {args.case: cases[args.case]} if args.case else cases
    for case, sizes in selected.items():
        for size in (args.size,) if args.size is not None else sizes:
            phases = []
            for phase in ("time", "allocations", "rss"):
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        f"benchmarks.{suite}",
                        "--package-root",
                        str(package_root),
                        "--case",
                        case,
                        "--size",
                        str(size),
                        "--repeat",
                        str(args.repeat),
                        "--warmup",
                        str(args.warmup),
                        "--seed",
                        str(args.seed),
                        "--phase",
                        phase,
                    ],
                    cwd=ROOT,
                    env={**os.environ, "PYTHONHASHSEED": str(args.seed)},
                    stdout=subprocess.PIPE,
                    text=True,
                    check=True,
                )
                phases.append(json.loads(completed.stdout))
            if any(
                p["inputs"] != phases[0]["inputs"]
                or p["metrics"]["output"] != phases[0]["metrics"]["output"]
                for p in phases
            ):
                msg = "Inputs or output counts changed between measurement phases"
                raise ValueError(msg)
            metrics = {key: value for p in phases for key, value in p["metrics"].items()}
            result = record(
                suite,
                case,
                {
                    "size": size,
                    "seed": args.seed,
                    "repeat": args.repeat,
                    "warmup": args.warmup,
                    "workload_version": 1,
                },
                phases[0]["inputs"],
                metrics,
                {"output_counts_stable": True},
                package_root,
            )
            result["environment"]["python_hash_seed"] = str(args.seed)
            emit(result)
