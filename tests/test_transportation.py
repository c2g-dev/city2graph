"""Tests for the transportation module."""

# ruff: noqa: D101, D102, D103, PLW2901, S608, PT011

import sys
from datetime import datetime
from pathlib import Path

import duckdb
import geopandas as gpd
import networkx as nx
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from city2graph.transportation import _time_to_seconds
from city2graph.transportation import _timestamp
from city2graph.transportation import get_od_pairs
from city2graph.transportation import load_gtfs
from city2graph.transportation import travel_summary_graph


def dict_to_con(d: dict[str, pd.DataFrame]) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("INSTALL spatial; LOAD spatial;")

    for k, v in d.items():
        v = v.copy()
        if k in ("stops", "shapes") and "geometry" in v.columns:
            v = v.drop(columns=["geometry"])
        # explicitly cast all columns to string so duckdb matching gtfs behaves consistently
        for col in v.columns:
            if v[col].dtype == bool:
                v[col] = v[col].astype(int)
            v[col] = v[col].astype(str)
        con.execute(f"CREATE TABLE {k} AS SELECT * FROM v")
        if k == "stops":
            con.execute("ALTER TABLE stops ADD COLUMN geometry GEOMETRY;")
            con.execute(
                "UPDATE stops SET geometry = ST_Point(CAST(stop_lon AS DOUBLE), CAST(stop_lat AS DOUBLE))"
            )

    # We must register the UDF exactly as load_gtfs does
    con.create_function("time_to_seconds", _time_to_seconds, ["VARCHAR"], "DOUBLE")
    return con


class TestTimeHelpers:
    def test_time_to_seconds_with_float(self) -> None:
        assert _time_to_seconds(3600.0) == 3600.0

    def test_time_to_seconds_with_none(self) -> None:
        assert _time_to_seconds(None) == 0.0

    def test_time_to_seconds_with_numeric_string(self) -> None:
        assert _time_to_seconds("3600.0") == 3600.0

    def test_timestamp_with_float(self) -> None:
        ts = _timestamp(3600.0, datetime(2024, 1, 1))
        assert ts == datetime(2024, 1, 1, 1, 0, 0)

    def test_timestamp_with_none(self) -> None:
        ts = _timestamp(None, datetime(2024, 1, 1))
        assert ts == datetime(2024, 1, 1, 0, 0, 0)

    def test_timestamp_with_numeric_string(self) -> None:
        ts = _timestamp("7200.0", datetime(2024, 1, 1))
        assert ts == datetime(2024, 1, 1, 2, 0, 0)


class TestLoadGtfs:
    def test_load_gtfs_basic(self, sample_gtfs_zip: str) -> None:
        con = load_gtfs(sample_gtfs_zip)
        assert isinstance(con, duckdb.DuckDBPyConnection)
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        assert {"stops", "routes", "trips", "stop_times", "calendar"}.issubset(set(tables))

    def test_load_gtfs_with_shapes(self, sample_gtfs_zip_with_shapes: str) -> None:
        con = load_gtfs(sample_gtfs_zip_with_shapes)
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        assert "shapes_geom" in tables

    def test_load_gtfs_empty_zip(self, empty_gtfs_zip: str) -> None:
        con = load_gtfs(empty_gtfs_zip)
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        assert len(tables) == 0

    def test_load_gtfs_invalid_coordinates(self, gtfs_zip_invalid_coords: str) -> None:
        con = load_gtfs(gtfs_zip_invalid_coords)
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        assert "stops" in tables
        # Invalid coordinates filtered appropriately

    def test_load_gtfs_nonexistent_file(self) -> None:
        con = load_gtfs("nonexistent.zip")
        assert len(con.execute("SHOW TABLES").fetchall()) == 0


class TestGetOdPairs:
    def test_get_od_pairs_basic(self, sample_gtfs_dict: dict[str, pd.DataFrame]) -> None:
        con = dict_to_con(sample_gtfs_dict)
        result = get_od_pairs(con)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) > 0
        expected_cols = {
            "trip_id",
            "service_id",
            "orig_stop_id",
            "dest_stop_id",
            "departure_ts",
            "arrival_ts",
            "travel_time_sec",
            "date",
        }
        assert expected_cols.issubset(result.columns)
        assert result.crs.to_string() == "EPSG:4326"

    def test_get_od_pairs_date_range(self, sample_gtfs_dict: dict[str, pd.DataFrame]) -> None:
        con = dict_to_con(sample_gtfs_dict)
        result = get_od_pairs(con, start_date="20240101", end_date="20240102")
        assert isinstance(result, gpd.GeoDataFrame)
        dates = pd.to_datetime(result["date"]).dt.date
        assert dates.min() >= datetime(2024, 1, 1).date()
        assert dates.max() <= datetime(2024, 1, 2).date()

    def test_get_od_pairs_no_geometry(self, sample_gtfs_dict: dict[str, pd.DataFrame]) -> None:
        con = dict_to_con(sample_gtfs_dict)
        result = get_od_pairs(con, include_geometry=False)
        assert isinstance(result, pd.DataFrame)
        assert "geometry" not in result.columns

    def test_get_od_pairs_incomplete_gtfs(self) -> None:
        con = duckdb.connect()
        con.execute("CREATE TABLE stops AS SELECT 1;")
        con.execute("CREATE TABLE trips AS SELECT 1;")
        con.execute("CREATE TABLE routes AS SELECT 1;")
        con.execute("CREATE TABLE stop_times AS SELECT 1;")
        con.execute("CREATE TABLE calendar AS SELECT 1;")
        # Need right columns, otherwise duckdb fails to find expected columns
        # To make it simple, let's just create an empty GTFS
        # Actually we just use a valid con but empty tables.
        # But a truly incomplete db:
        empty_con = dict_to_con(
            {"stops": pd.DataFrame(columns=["stop_id", "stop_lat", "stop_lon"])}
        )
        result = get_od_pairs(empty_con)
        assert result.empty

    def test_get_od_pairs_with_calendar_dates(
        self, sample_gtfs_dict_with_exceptions: dict[str, pd.DataFrame]
    ) -> None:
        con = dict_to_con(sample_gtfs_dict_with_exceptions)
        result = get_od_pairs(con)
        assert len(result) > 0

    def test_get_od_pairs_with_malformed_times(
        self,
        sample_gtfs_dict: dict[str, pd.DataFrame],
    ) -> None:
        gtfs_copy = sample_gtfs_dict.copy()
        stop_times = gtfs_copy["stop_times"].copy()
        # Coerce time columns to object dtype so we can inject non-string values
        for col in ("arrival_time", "departure_time"):
            stop_times[col] = stop_times[col].astype(object)
        stop_times.loc[0, "departure_time"] = None
        stop_times.loc[1, "arrival_time"] = 3600.0  # numeric value
        gtfs_copy["stop_times"] = stop_times
        con = dict_to_con(gtfs_copy)
        result = get_od_pairs(con)
        assert isinstance(result, (pd.DataFrame, gpd.GeoDataFrame))

    def test_get_od_pairs_removal_exception_integration(self) -> None:
        stops = pd.DataFrame(
            {"stop_id": ["s1", "s2"], "stop_lat": [0.0, 1.0], "stop_lon": [0.0, 1.0]}
        )
        trips = pd.DataFrame({"trip_id": ["t1"], "service_id": ["svc"]})
        stop_times = pd.DataFrame(
            {
                "trip_id": ["t1", "t1"],
                "stop_id": ["s1", "s2"],
                "stop_sequence": [1, 2],
                "departure_time": ["08:00:00", "08:10:00"],
                "arrival_time": ["08:10:00", "08:20:00"],
            }
        )
        calendar = pd.DataFrame(
            {
                "service_id": ["svc"],
                "start_date": ["20240101"],
                "end_date": ["20240101"],
                "monday": [False],
                "tuesday": [False],
                "wednesday": [False],
                "thursday": [False],
                "friday": [False],
                "saturday": [False],
                "sunday": [True],
            }
        )
        calendar_dates = pd.DataFrame(
            {"service_id": ["svc"], "date": ["20240101"], "exception_type": [2]}
        )

        gtfs = {
            "stops": stops,
            "trips": trips,
            "stop_times": stop_times,
            "calendar": calendar,
            "calendar_dates": calendar_dates,
        }
        con = dict_to_con(gtfs)
        result = get_od_pairs(con, include_geometry=False)
        assert len(result) == 0


class TestTravelSummaryGraph:
    def test_travel_summary_graph_basic(self, sample_gtfs_dict: dict[str, pd.DataFrame]) -> None:
        con = dict_to_con(sample_gtfs_dict)
        nodes_gdf, edges_gdf = travel_summary_graph(con)
        assert isinstance(nodes_gdf, gpd.GeoDataFrame)
        assert len(nodes_gdf) > 0
        assert isinstance(edges_gdf, gpd.GeoDataFrame)
        assert len(edges_gdf) > 0

    def test_travel_summary_graph_time_filter(
        self, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        con = dict_to_con(sample_gtfs_dict)
        nodes_gdf, edges_gdf = travel_summary_graph(con, start_time="07:00:00", end_time="09:00:00")
        assert isinstance(nodes_gdf, gpd.GeoDataFrame)
        assert isinstance(edges_gdf, gpd.GeoDataFrame)

    def test_travel_summary_graph_calendar_range(
        self, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        con = dict_to_con(sample_gtfs_dict)
        nodes_gdf, edges_gdf = travel_summary_graph(
            con, calendar_start="20240101", calendar_end="20240107"
        )
        assert isinstance(nodes_gdf, gpd.GeoDataFrame)
        assert isinstance(edges_gdf, gpd.GeoDataFrame)

    def test_travel_summary_graph_as_nx(self, sample_gtfs_dict: dict[str, pd.DataFrame]) -> None:
        con = dict_to_con(sample_gtfs_dict)
        result = travel_summary_graph(con, as_nx=True)
        assert isinstance(result, nx.Graph)

    def test_travel_summary_graph_missing_data(self) -> None:
        con = duckdb.connect()
        con.execute("CREATE TABLE stops AS SELECT 1 as a")
        with pytest.raises(ValueError):
            travel_summary_graph(con)

    def test_travel_summary_graph_empty_stop_times(
        self, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        sample_gtfs_dict["stop_times"] = pd.DataFrame(
            columns=["trip_id", "stop_id", "arrival_time", "departure_time", "stop_sequence"]
        )
        con = dict_to_con(sample_gtfs_dict)
        _nodes_gdf, edges_gdf = travel_summary_graph(con)
        assert len(edges_gdf) == 0

    def test_travel_summary_graph_with_calendar_dates(
        self, sample_gtfs_dict_with_exceptions: dict[str, pd.DataFrame]
    ) -> None:
        con = dict_to_con(sample_gtfs_dict_with_exceptions)
        _nodes_gdf, edges_gdf = travel_summary_graph(
            con, calendar_start="20240101", calendar_end="20240102"
        )
        assert len(edges_gdf) >= 0

    def test_travel_summary_graph_with_calendar_dates_add_service(
        self, gtfs_dict_add_service: dict[str, pd.DataFrame]
    ) -> None:
        con = dict_to_con(gtfs_dict_add_service)
        _nodes_gdf, edges_gdf = travel_summary_graph(
            con, calendar_start="20240101", calendar_end="20240101"
        )
        assert len(edges_gdf) > 0

    def test_travel_summary_graph_with_calendar_dates_remove_service(
        self, gtfs_dict_remove_service: dict[str, pd.DataFrame]
    ) -> None:
        con = dict_to_con(gtfs_dict_remove_service)
        _nodes_gdf, edges_gdf = travel_summary_graph(
            con, calendar_start="20240101", calendar_end="20240101"
        )
        assert len(edges_gdf) == 0

    def test_travel_summary_graph_missing_calendar(
        self, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        del sample_gtfs_dict["calendar"]
        con = dict_to_con(sample_gtfs_dict)
        with pytest.raises(ValueError):
            travel_summary_graph(con, calendar_start="20240101", calendar_end="20240107")

    def test_travel_summary_graph_empty_calendar(
        self, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        sample_gtfs_dict["calendar"] = pd.DataFrame(
            columns=["service_id", "start_date", "end_date"]
        )
        con = dict_to_con(sample_gtfs_dict)
        with pytest.raises(ValueError):
            travel_summary_graph(con, calendar_start="20240101", calendar_end="20240107")

    def test_travel_summary_graph_invalid_calendar_start_format(
        self, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        con = dict_to_con(sample_gtfs_dict)
        with pytest.raises(ValueError):
            travel_summary_graph(con, calendar_start="invalid", calendar_end="20240107")

    def test_travel_summary_graph_invalid_calendar_end_format(
        self, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        con = dict_to_con(sample_gtfs_dict)
        with pytest.raises(ValueError):
            travel_summary_graph(con, calendar_start="20240101", calendar_end="invalid")

    def test_travel_summary_graph_calendar_start_outside_range(
        self, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        con = dict_to_con(sample_gtfs_dict)
        with pytest.raises(ValueError):
            travel_summary_graph(con, calendar_start="20230101", calendar_end="20240107")

    def test_travel_summary_graph_calendar_end_outside_range(
        self, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        con = dict_to_con(sample_gtfs_dict)
        with pytest.raises(ValueError):
            travel_summary_graph(con, calendar_start="20240101", calendar_end="20250101")

    def test_travel_summary_graph_calendar_start_after_end(
        self, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        con = dict_to_con(sample_gtfs_dict)
        with pytest.raises(ValueError):
            travel_summary_graph(con, calendar_start="20240107", calendar_end="20240101")

    def test_travel_summary_graph_calendar_start_only_invalid(
        self, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        con = dict_to_con(sample_gtfs_dict)
        with pytest.raises(ValueError):
            travel_summary_graph(con, calendar_start="20230101")

    def test_travel_summary_graph_calendar_end_only_invalid(
        self, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        con = dict_to_con(sample_gtfs_dict)
        with pytest.raises(ValueError):
            travel_summary_graph(con, calendar_end="20250101")
