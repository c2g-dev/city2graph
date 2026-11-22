"""Fixtures shared by the library and documentation suites."""

from typing import Any

import pytest


@pytest.fixture
def osm_response() -> dict[str, Any]:
    """Provide a small synthetic Overpass response for public loader tests."""
    return {
        "osm3s": {"timestamp_osm_base": "2026-11-02T12:00:00Z"},
        "elements": [
            {
                "type": "relation",
                "id": 42,
                "tags": {
                    "type": "route",
                    "route": "bus",
                    "ref": "R",
                    "name": "Test route",
                    "public_transport:version": "2",
                },
                "members": [
                    {"type": "node", "ref": 1, "role": "stop"},
                    {"type": "node", "ref": 11, "role": "platform"},
                    {"type": "node", "ref": 2, "role": "stop_exit_only"},
                    {"type": "way", "ref": 3, "role": ""},
                ],
            },
            {"type": "node", "id": 1, "lat": 51, "lon": 0, "tags": {"name": "A"}},
            {"type": "node", "id": 11, "lat": 51.0001, "lon": 0, "tags": {"name": "A"}},
            {"type": "node", "id": 2, "lat": 51, "lon": 0.01, "tags": {"name": "B"}},
            {"type": "way", "id": 3, "geometry": [{"lat": 51, "lon": 0}, {"lat": 51, "lon": 0.01}]},
        ],
    }
