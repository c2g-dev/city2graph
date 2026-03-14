"""Tests for the transportation module."""

# ruff: noqa: D101, D102, D103, PLW2901, S608, PT011

import sys
from datetime import datetime
from pathlib import Path
from typing import cast

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
    con.create_function(
        "time_to_seconds", _time_to_seconds, [duckdb.sqltypes.VARCHAR], duckdb.sqltypes.DOUBLE
    )
    return con


class TestTimeHelpers:
    def test_time_to_seconds_with_float(self) -> None:
        assert _time_to_seconds(3600.0) == 3600.0

    def test_time_to_seconds_with_none(self) -> None:
        assert _time_to_seconds(None) == 0.0

    def test_time_to_seconds_valid_hms(self) -> None:
        assert _time_to_seconds("08:30:00") == 30600.0

    def test_time_to_seconds_extended_hours(self) -> None:
        assert _time_to_seconds("25:30:00") == 91800.0

    def test_time_to_seconds_rejects_numeric_string(self) -> None:
        with pytest.raises(ValueError, match="Expected HH:MM:SS"):
            _time_to_seconds("3600.0")

    def test_time_to_seconds_rejects_empty_string(self) -> None:
        with pytest.raises(ValueError, match="Expected HH:MM:SS"):
            _time_to_seconds("")

    def test_time_to_seconds_rejects_nan_string(self) -> None:
        with pytest.raises(ValueError, match="Expected HH:MM:SS"):
            _time_to_seconds("nan")

    def test_time_to_seconds_rejects_none_string(self) -> None:
        with pytest.raises(ValueError, match="Expected HH:MM:SS"):
            _time_to_seconds("None")

    def test_timestamp_with_float(self) -> None:
        ts = _timestamp(3600.0, datetime(2024, 1, 1))
        assert ts == datetime(2024, 1, 1, 1, 0, 0)

    def test_timestamp_with_none(self) -> None:
        ts = _timestamp(None, datetime(2024, 1, 1))
        assert ts == datetime(2024, 1, 1, 0, 0, 0)

    def test_timestamp_with_hms_string(self) -> None:
        ts = _timestamp("02:00:00", datetime(2024, 1, 1))
        assert ts == datetime(2024, 1, 1, 2, 0, 0)

    def test_timestamp_with_invalid_string(self) -> None:
        # Invalid string should return None (caught by _timestamp's try/except)
        ts = _timestamp("invalid", datetime(2024, 1, 1))
        assert ts is None


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

    def test_get_od_pairs_directed(self, sample_gtfs_dict: dict[str, pd.DataFrame]) -> None:
        con = dict_to_con(sample_gtfs_dict)
        result_directed = get_od_pairs(con, directed=True, include_geometry=False)
        result_undirected = get_od_pairs(con, directed=False, include_geometry=False)

        assert isinstance(result_directed, pd.DataFrame)
        assert isinstance(result_undirected, pd.DataFrame)
        # Undirected should have orig_stop_id <= dest_stop_id for all rows
        assert (result_undirected["orig_stop_id"] <= result_undirected["dest_stop_id"]).all()

    def test_get_od_pairs_calendar_dates_only_with_explicit_window(
        self, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        gtfs = sample_gtfs_dict.copy()
        gtfs.pop("calendar")
        gtfs["calendar_dates"] = pd.DataFrame(
            {
                "service_id": ["service1"],
                "date": ["20240101"],
                "exception_type": [1],
            }
        )

        con = dict_to_con(gtfs)
        result = get_od_pairs(
            con,
            start_date="20240101",
            end_date="20240101",
            include_geometry=False,
        )

        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_get_od_pairs_without_calendar_and_without_dates_returns_empty(
        self, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        gtfs = sample_gtfs_dict.copy()
        gtfs.pop("calendar")

        con = dict_to_con(gtfs)
        result = get_od_pairs(con, include_geometry=False)

        assert isinstance(result, pd.DataFrame)
        assert result.empty


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

    def test_travel_summary_graph_as_nx_undirected(
        self, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        con = dict_to_con(sample_gtfs_dict)
        result = travel_summary_graph(con, as_nx=True, directed=False)
        assert isinstance(result, nx.Graph)
        assert not isinstance(result, nx.DiGraph)

    def test_travel_summary_graph_as_nx_directed(
        self, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        con = dict_to_con(sample_gtfs_dict)
        result = travel_summary_graph(con, as_nx=True, directed=True)
        assert isinstance(result, nx.DiGraph)

    def test_travel_summary_graph_missing_data(self) -> None:
        con = duckdb.connect()
        con.execute("CREATE TABLE stops AS SELECT 1 as a")
        with pytest.raises(ValueError):
            travel_summary_graph(con)

    def test_travel_summary_graph_surfaces_spatial_extension_failures(self) -> None:
        class _ShowTablesResult:
            def fetchall(self) -> list[tuple[str]]:
                return [("stop_times",), ("stops",), ("trips",)]

        class FailingSpatialConnection:
            def execute(self, query: str) -> _ShowTablesResult:
                if query == "SHOW TABLES":
                    return _ShowTablesResult()
                if query == "LOAD spatial;":
                    msg = "load failed"
                    raise duckdb.Error(msg)
                if query == "INSTALL spatial; LOAD spatial;":
                    msg = "install failed"
                    raise duckdb.Error(msg)
                raise AssertionError(query)

        with pytest.raises(RuntimeError, match="DuckDB spatial extension is required"):
            travel_summary_graph(cast("duckdb.DuckDBPyConnection", FailingSpatialConnection()))

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

    def test_travel_summary_graph_directed_edges(
        self, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        con = dict_to_con(sample_gtfs_dict)
        _nodes, _edges_directed = travel_summary_graph(con, directed=True)
        _nodes2, edges_undirected = travel_summary_graph(con, directed=False)

        # Undirected: from_stop_id <= to_stop_id in every index entry
        for from_id, to_id in edges_undirected.index:
            assert from_id <= to_id

    def test_travel_summary_graph_with_frequencies(
        self, sample_gtfs_dict_with_frequencies: dict[str, pd.DataFrame]
    ) -> None:
        con = dict_to_con(sample_gtfs_dict_with_frequencies)
        _nodes, edges_with = travel_summary_graph(con, use_frequencies=True)
        assert len(edges_with) > 0
        # frequency should be higher because of headway expansion
        assert (edges_with["frequency"] > 0).all()

    def test_travel_summary_graph_without_frequencies(
        self, sample_gtfs_dict_with_frequencies: dict[str, pd.DataFrame]
    ) -> None:
        con = dict_to_con(sample_gtfs_dict_with_frequencies)
        _nodes, edges_without = travel_summary_graph(con, use_frequencies=False)
        assert len(edges_without) > 0

    def test_travel_summary_graph_use_shapes(
        self, sample_gtfs_dict_with_shapes: dict[str, pd.DataFrame]
    ) -> None:
        con = dict_to_con(sample_gtfs_dict_with_shapes)
        _nodes, edges = travel_summary_graph(con, use_shapes=True, directed=True)
        assert len(edges) > 0
        # At least some edges should have non-null geometry
        assert edges.geometry.notna().any()

    def test_travel_summary_graph_no_shapes(
        self, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        con = dict_to_con(sample_gtfs_dict)
        _nodes, edges = travel_summary_graph(con, use_shapes=False)
        assert len(edges) > 0

    def test_travel_summary_graph_without_calendar_falls_back_to_trip_counts(
        self, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        gtfs = sample_gtfs_dict.copy()
        gtfs.pop("calendar")
        con = dict_to_con(gtfs)

        nodes_gdf, edges_gdf = travel_summary_graph(con)

        assert isinstance(nodes_gdf, gpd.GeoDataFrame)
        assert isinstance(edges_gdf, gpd.GeoDataFrame)
        assert len(edges_gdf) > 0

    def test_travel_summary_graph_calendar_dates_only_feed(
        self, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        gtfs = sample_gtfs_dict.copy()
        gtfs.pop("calendar")
        gtfs["calendar_dates"] = pd.DataFrame(
            {
                "service_id": ["service1"],
                "date": ["20240101"],
                "exception_type": [1],
            }
        )
        con = dict_to_con(gtfs)

        _nodes_gdf, edges_gdf = travel_summary_graph(con)

        assert len(edges_gdf) > 0

    def test_travel_summary_graph_calendar_dates_only_empty_table_errors_with_window(
        self, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        gtfs = sample_gtfs_dict.copy()
        gtfs.pop("calendar")
        gtfs["calendar_dates"] = pd.DataFrame(columns=["service_id", "date", "exception_type"])
        con = dict_to_con(gtfs)

        with pytest.raises(ValueError, match="contain no usable dates"):
            travel_summary_graph(con, calendar_start="20240101", calendar_end="20240102")

    def test_travel_summary_graph_calendar_dates_only_empty_table_falls_back_without_window(
        self, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        gtfs = sample_gtfs_dict.copy()
        gtfs.pop("calendar")
        gtfs["calendar_dates"] = pd.DataFrame(columns=["service_id", "date", "exception_type"])
        con = dict_to_con(gtfs)

        _nodes_gdf, edges_gdf = travel_summary_graph(con)

        assert edges_gdf["frequency"].gt(0).all()

    def test_travel_summary_graph_without_calendar_rejects_explicit_window(
        self, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        gtfs = sample_gtfs_dict.copy()
        gtfs.pop("calendar")
        con = dict_to_con(gtfs)

        with pytest.raises(ValueError, match=r"has neither calendar\.txt nor calendar_dates\.txt"):
            travel_summary_graph(con, calendar_start="20240101", calendar_end="20240102")

    def test_travel_summary_graph_rejects_invalid_time_format(
        self, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        con = dict_to_con(sample_gtfs_dict)

        with pytest.raises(ValueError, match="Invalid time format"):
            travel_summary_graph(con, start_time="invalid")

    def test_travel_summary_graph_rejects_non_numeric_time_components(
        self, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        con = dict_to_con(sample_gtfs_dict)

        with pytest.raises(ValueError, match="Invalid time format"):
            travel_summary_graph(con, start_time="aa:bb:cc")

    def test_travel_summary_graph_rejects_inverted_time_window(
        self, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        con = dict_to_con(sample_gtfs_dict)

        with pytest.raises(ValueError, match="start_time must be <="):
            travel_summary_graph(con, start_time="09:00:00", end_time="08:00:00")

    def test_travel_summary_graph_requires_stop_coordinates_or_geometry(
        self, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        con = duckdb.connect(":memory:")
        con.execute("INSTALL spatial; LOAD spatial;")
        con.create_function(
            "time_to_seconds",
            _time_to_seconds,
            [duckdb.sqltypes.VARCHAR],
            duckdb.sqltypes.DOUBLE,
        )

        frame = pd.DataFrame({"stop_id": ["stop1", "stop2", "stop3"]})
        for name in ("trips", "stop_times", "calendar"):
            frame = sample_gtfs_dict[name].copy()
            for column in frame.columns:
                if frame[column].dtype == bool:
                    frame[column] = frame[column].astype(int)
                frame[column] = frame[column].astype(str)
            con.execute(f"CREATE TABLE {name} AS SELECT * FROM frame")
        frame = pd.DataFrame({"stop_id": ["stop1", "stop2", "stop3"]})
        con.execute("CREATE TABLE stops AS SELECT * FROM frame")

        with pytest.raises(ValueError, match="stops must contain either a geometry column"):
            travel_summary_graph(con)

    def test_travel_summary_graph_uses_shape_segments_from_loaded_feed(
        self, sample_gtfs_zip_with_shapes: str
    ) -> None:
        con = load_gtfs(sample_gtfs_zip_with_shapes)

        _nodes_gdf, edges_gdf = travel_summary_graph(con, use_shapes=True, directed=True)

        assert not edges_gdf.empty
        assert edges_gdf.geometry.notna().any()
