"""Tests for benchmark compare."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from typing import Any

import pytest

from benchmarks.compare import compare_results

if TYPE_CHECKING:
    from pathlib import Path


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
