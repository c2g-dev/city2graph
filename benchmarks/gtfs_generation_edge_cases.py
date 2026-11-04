"""Reproduce four API issues found in the 2026-11-03 multi-city audit.

Run with ``uv run python benchmarks/gtfs_generation_edge_cases.py``. The small
inputs isolate failures first checked with real GTFS data; no network is used.
This diagnostic records outcomes rather than asserting that current bugs persist.
"""

from __future__ import annotations

import json
import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

import city2graph as c2g

if TYPE_CHECKING:
    from collections.abc import Callable

    import duckdb


def outcome(name: str, generate: Callable[[], duckdb.DuckDBPyConnection]) -> dict[str, object]:
    """Record the public API's outcome without modifying its validation."""
    try:
        with generate() as con:
            return {"case": name, "outcome": "generated", "trips": len(con.table("trips").df())}
    except (ValueError, OverflowError) as exc:
        return {"case": name, "outcome": type(exc).__name__, "message": str(exc)}


def main() -> None:
    """Run controls and minimal reproductions with temporary GTFS archives."""
    data: c2g.TransportData = {
        "routes": pd.DataFrame([{"route_id": "R", "route_short_name": "R", "route_type": 3}]),
        "stops": pd.DataFrame(
            {
                "stop_id": ["A", "B"],
                "stop_name": ["Audit A", "Audit B"],
                "stop_lat": [51.50, 51.51],
                "stop_lon": [-0.12, -0.12],
            }
        ),
        "patterns": pd.DataFrame(
            {"pattern_id": ["p0", "p1"], "route_id": ["R", "R"], "direction_id": [0, 1]}
        ),
        "stop_sequences": pd.DataFrame(
            {
                "pattern_id": ["p0", "p0", "p1", "p1"],
                "stop_id": ["A", "B", "B", "A"],
                "stop_sequence": [0, 1, 0, 1],
            }
        ),
    }
    agency = pd.DataFrame(
        [{"agency_name": "Audit", "agency_url": "https://example.org", "agency_timezone": "UTC"}]
    )

    def generate(
        supplied: c2g.TransportData, services: pd.DataFrame | None = None
    ) -> duckdb.DuckDBPyConnection:
        return c2g.create_gtfs(supplied, services, agency=agency, service_date="20261103")

    with warnings.catch_warnings(), tempfile.TemporaryDirectory() as directory:
        warnings.simplefilter("ignore", c2g.GTFSGenerationWarning)
        results = [outcome("control", lambda: generate(data))]
        with generate(data) as con:
            archive = c2g.save_gtfs(con, Path(directory) / "audit.zip")
        with c2g.load_gtfs(archive) as con:
            loaded: c2g.TransportData = {**data, "stops": con.table("stops").df()}
        results.append(outcome("reuse_loaded_stops", lambda: generate(loaded)))
        loaded["stops"] = loaded["stops"].drop(columns="geometry")
        results.append(outcome("reuse_without_geometry_control", lambda: generate(loaded)))

        floating: c2g.TransportData = {
            **data,
            "patterns": data["patterns"].astype({"direction_id": float}),
        }
        results.append(outcome("numeric_direction_codes", lambda: generate(floating)))
        departures = pd.DataFrame(
            [{"pattern_id": "p0", "trip_id": "service_1", "departure_time": "08:00:00"}]
        )
        results.append(outcome("generated_id_collision", lambda: generate(data, departures)))
        departures["trip_id"] = "explicit_trip"
        results.append(outcome("noncolliding_id_control", lambda: generate(data, departures)))

        invalid: c2g.TransportData = {
            **data,
            "routes": data["routes"].assign(route_type=float("inf")),
        }
        results.append(outcome("invalid_route_type_diagnostic", lambda: generate(invalid)))
    print(json.dumps(results, indent=2))  # noqa: T201 - diagnostic command output


if __name__ == "__main__":
    main()
