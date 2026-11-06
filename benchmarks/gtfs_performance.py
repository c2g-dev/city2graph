"""Measure GTFS generation, persistence and summaries using synthetic services."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from benchmarks.common import Workload
from benchmarks.common import digest
from benchmarks.common import run_suite

if TYPE_CHECKING:
    from contextlib import ExitStack

    import pandas as pd

    from city2graph import TransportData

# Workload imports follow package checkout selection in the measurement worker.
# ruff: noqa: PLC0415, ANN401

CASES = {
    f"{operation}-{service}": (10, 100, 1000)
    for operation in ("generate", "save", "load", "summary")
    for service in ("fixed", "frequency")
}
SERVICE_DATE = "20000103"
STOP_COUNT = 20


def synthetic_input(
    size: int, seed: int, service: str
) -> tuple[TransportData, pd.DataFrame, pd.DataFrame]:
    """Build reproducible routes sharing twenty ordered boarding locations."""
    import numpy as np
    import pandas as pd

    patterns = [f"p{i}" for i in range(size)]
    stops = [f"s{i}" for i in range(STOP_COUNT)]
    data: TransportData = {
        "routes": pd.DataFrame(
            {"route_id": patterns, "route_short_name": patterns, "route_type": 3}
        ),
        "stops": pd.DataFrame(
            {
                "stop_id": stops,
                "stop_name": stops,
                "stop_lat": 51.5,
                "stop_lon": np.linspace(-0.12, -0.10, STOP_COUNT),
            }
        ),
        "patterns": pd.DataFrame({"pattern_id": patterns, "route_id": patterns, "direction_id": 0}),
        "stop_sequences": pd.DataFrame(
            {
                "pattern_id": np.repeat(patterns, STOP_COUNT),
                "stop_id": stops * size,
                "stop_sequence": list(range(STOP_COUNT)) * size,
                "travel_time_secs": np.tile(
                    np.r_[0, np.random.default_rng(seed).integers(40, 81, STOP_COUNT - 1)], size
                ),
            }
        ),
    }
    services = pd.DataFrame({"pattern_id": patterns})
    if service == "fixed":
        services["departure_time"] = "08:00:00"
    elif service == "frequency":
        services["start_time"] = "06:00:00"
        services["end_time"] = "22:00:00"
        services["headway_secs"] = 600
    else:
        msg = f"Unknown service kind: {service}"
        raise ValueError(msg)
    agency = pd.DataFrame(
        [
            {
                "agency_name": "Benchmark",
                "agency_url": "https://example.org",
                "agency_timezone": "UTC",
            }
        ]
    )
    return data, services, agency


def prepare(case: str, size: int, seed: int, stack: ExitStack) -> Workload:
    """Prepare inputs and ensure each measured result has a defined owner."""
    import city2graph as c2g

    operation, service = case.split("-")
    data, services, agency = synthetic_input(size, seed, service)
    inputs = {
        "synthetic": digest(
            {
                "generator": "gtfs-v1",
                "date": SERVICE_DATE,
                "tables": {
                    name: cast("pd.DataFrame", frame).to_json(orient="split")
                    for name, frame in data.items()
                },
                "services": services.to_json(orient="split"),
                "agency": agency.to_json(orient="split"),
                "dwell_time_secs": 0,
            }
        )
    }

    def generate() -> Any:
        return c2g.create_gtfs(
            data, services, agency=agency, service_date=SERVICE_DATE, dwell_time_secs=0
        )

    def describe_feed(con: Any) -> dict[str, int]:
        return {
            table: int(con.table(table).count("*").fetchall()[0][0])
            for table in ("routes", "stops", "trips", "stop_times")
        }

    if operation == "generate":
        return Workload(generate, describe_feed, inputs, lambda con: con.close())
    con = stack.enter_context(generate())
    if operation == "summary":
        return Workload(
            lambda: c2g.get_gtfs_summary(con),
            lambda report: {name: len(frame) for name, frame in report.items()},
            inputs,
        )
    directory = Path(stack.enter_context(tempfile.TemporaryDirectory()))
    archive = directory / "feed.zip"
    if operation == "save":
        return Workload(
            lambda: c2g.save_gtfs(con, archive, overwrite=True),
            lambda path: {"archive_bytes": path.stat().st_size},
            inputs,
        )
    c2g.save_gtfs(con, archive)
    return Workload(
        lambda: c2g.load_gtfs(archive), describe_feed, inputs, lambda loaded: loaded.close()
    )


if __name__ == "__main__":
    run_suite("gtfs_performance", CASES, prepare)
