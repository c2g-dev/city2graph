"""Shared fixtures for gtfs tests."""

import pandas as pd
import pytest


@pytest.fixture
def departures() -> pd.DataFrame:
    """Provide synthetic transport departures for public API tests."""
    return pd.DataFrame(
        {
            "pattern_id": ["local", "express"],
            "service_id": ["weekday"] * 2,
            "departure_time": ["08:00:00", "08:05:00"],
        }
    )
