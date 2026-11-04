"""Tests for the transportation module."""

import typing

# ruff: noqa: D101, D102, S608, PT011
from datetime import datetime
from pathlib import Path
from typing import cast

import duckdb
import geopandas as gpd
import networkx as nx
import pandas as pd
import pytest

import city2graph as c2g
from city2graph.transportation import _time_to_seconds
from city2graph.transportation import _timestamp
from city2graph.transportation import get_od_pairs
from city2graph.transportation import load_gbfs
from city2graph.transportation import load_gtfs
from city2graph.transportation import travel_summary_graph
from tests.helpers import GtfsConnectionFactory
from tests.helpers import temporary_gtfs_zip


@pytest.fixture
def empty_gtfs_zip() -> typing.Generator[str, None, None]:
    """Create an empty GTFS zip file for testing."""
    yield from temporary_gtfs_zip({})


@pytest.fixture
def gtfs_zip_invalid_coords() -> typing.Generator[str, None, None]:
    """Create a GTFS zip file with invalid coordinates for testing."""
    yield from temporary_gtfs_zip(
        {
            "stops.txt": (
                "stop_id,stop_name,stop_lat,stop_lon\n"
                "stop1,Stop 1,40.7128,-74.0060\n"
                "stop2,Stop 2,invalid,invalid\n"
                "stop3,Stop 3,,\n"
            ),
            "agency.txt": (
                "agency_id,agency_name,agency_url,agency_timezone\n"
                "1,Test Agency,http://test.com,America/New_York\n"
            ),
            "calendar.txt": (
                "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\n"
                "service1,1,1,1,1,1,0,0,20240101,20241231\n"
            ),
        },
    )


@pytest.fixture
def sample_gtfs_dict_with_exceptions(
    sample_gtfs_dict: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Create a sample GTFS dictionary with calendar exceptions."""
    gtfs_dict = sample_gtfs_dict.copy()

    # Add calendar_dates for exceptions
    calendar_dates_df = pd.DataFrame(
        {
            "service_id": ["service1", "service1"],
            "date": ["20240101", "20240102"],
            "exception_type": [2, 1],  # 2 = remove service, 1 = add service
        },
    )

    gtfs_dict["calendar_dates"] = calendar_dates_df
    return gtfs_dict


@pytest.fixture
def gtfs_dict_add_service(
    sample_gtfs_dict: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Create a sample GTFS dictionary where calendar_dates adds service."""
    gtfs_dict = sample_gtfs_dict.copy()

    # Modify calendar to have no service on test date initially
    calendar_df = gtfs_dict["calendar"].copy()
    calendar_df["monday"] = [False]  # 20240101 is a Monday
    gtfs_dict["calendar"] = calendar_df

    # Add calendar_dates that adds service on the test date
    calendar_dates_df = pd.DataFrame(
        {
            "service_id": ["service1"],
            "date": ["20240101"],
            "exception_type": [1],  # 1 = add service
        },
    )

    gtfs_dict["calendar_dates"] = calendar_dates_df
    return gtfs_dict


@pytest.fixture
def gtfs_dict_remove_service(
    sample_gtfs_dict: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Create a sample GTFS dictionary where calendar_dates removes service."""
    gtfs_dict = sample_gtfs_dict.copy()

    # Ensure calendar has service on test date initially
    calendar_df = gtfs_dict["calendar"].copy()
    calendar_df["monday"] = [True]  # 20240101 is a Monday
    gtfs_dict["calendar"] = calendar_df

    # Add calendar_dates that removes service on the test date
    calendar_dates_df = pd.DataFrame(
        {
            "service_id": ["service1"],
            "date": ["20240101"],
            "exception_type": [2],  # 2 = remove service
        },
    )

    gtfs_dict["calendar_dates"] = calendar_dates_df
    return gtfs_dict


@pytest.fixture
def sample_gtfs_dict_with_frequencies(
    sample_gtfs_dict: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Create a sample GTFS dictionary with a frequencies table."""
    gtfs_dict = sample_gtfs_dict.copy()

    # Add frequencies.txt: trip1 runs every 600 seconds (10 min) from 08:00 to 09:00
    # That gives floor((09:00:00 - 08:00:00) / 600) = 6 departures
    frequencies_df = pd.DataFrame(
        {
            "trip_id": ["trip1"],
            "start_time": ["08:00:00"],
            "end_time": ["09:00:00"],
            "headway_secs": [600],
        },
    )
    gtfs_dict["frequencies"] = frequencies_df
    return gtfs_dict


class _ShowTablesResult:
    def fetchall(self) -> list[tuple[str]]:
        return [("stop_times",), ("stops",), ("trips",)]


class _FunctionCountResult:
    def fetchall(self) -> list[tuple[int]]:
        return [(0,)]


class FailingUdfConnection:
    """Connection stub whose UDF registration fails after spatial support loads."""

    def execute(self, query: str) -> _ShowTablesResult | _FunctionCountResult | None:
        if query == "SHOW TABLES":
            return _ShowTablesResult()
        if query == "LOAD spatial;":
            return None
        if "duckdb_functions()" in query:
            return _FunctionCountResult()
        raise AssertionError(query)

    def create_function(self, *_args: object, **_kwargs: object) -> None:
        msg = "UDF registration failed"
        raise duckdb.Error(msg)


class TestTimeHelpers:
    """Retain legacy numeric/null compatibility branches outside the CSV loader contract."""

    def test_time_to_seconds_with_float(self) -> None:
        assert _time_to_seconds(3600.0) == 3600.0

    def test_time_to_seconds_with_none(self) -> None:
        assert _time_to_seconds(None) == 0.0

    def test_timestamp_with_float(self) -> None:
        ts = _timestamp(3600.0, datetime(2024, 1, 1))
        assert ts == datetime(2024, 1, 1, 1, 0, 0)

    def test_timestamp_with_none(self) -> None:
        ts = _timestamp(None, datetime(2024, 1, 1))
        assert ts == datetime(2024, 1, 1, 0, 0, 0)


class TestLoadGtfs:
    def test_load_gtfs_basic(self, sample_gtfs_zip: str) -> None:
        with load_gtfs(sample_gtfs_zip) as con:
            assert isinstance(con, duckdb.DuckDBPyConnection)
            tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
            assert {"stops", "routes", "trips", "stop_times", "calendar"}.issubset(set(tables))

    def test_load_gtfs_empty_zip(self, empty_gtfs_zip: str) -> None:
        with load_gtfs(empty_gtfs_zip) as con:
            tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
            assert len(tables) == 0

    def test_load_gtfs_invalid_coordinates(self, gtfs_zip_invalid_coords: str) -> None:
        with load_gtfs(gtfs_zip_invalid_coords) as con:
            tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
            assert "stops" in tables
            assert con.sql(
                "SELECT stop_id, geometry IS NULL FROM stops ORDER BY stop_id"
            ).fetchall() == [
                ("stop1", False),
                ("stop2", True),
                ("stop3", True),
            ]

    def test_load_gtfs_nonexistent_file(self) -> None:
        with load_gtfs("nonexistent.zip") as con:
            assert len(con.execute("SHOW TABLES").fetchall()) == 0


class TestLoadGbfs:
    def test_load_gbfs_is_public_top_level_api(self) -> None:
        assert c2g.load_gbfs is load_gbfs

    def test_load_gbfs_basic(self, tmp_path: Path) -> None:
        gbfs_dir = tmp_path / "gbfs"
        gbfs_dir.mkdir()
        (gbfs_dir / "station_information.json").write_text(
            """
            {
                "data": {
                    "stations": [
                        {
                            "station_id": "1",
                            "name": "Test Station",
                            "lat": 51.5,
                            "lon": -0.1,
                            "capacity": 10
                        }
                    ]
                }
            }
            """,
            encoding="utf-8",
        )
        with load_gbfs(gbfs_dir) as con:
            assert isinstance(con, duckdb.DuckDBPyConnection)
            tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
            assert "station_information" in tables
            rows = con.execute(
                "SELECT station_id, name, lat, lon, capacity FROM station_information"
            ).fetchall()
            assert rows == [("1", "Test Station", 51.5, -0.1, 10)]

    def test_load_gbfs_empty_directory(self, tmp_path: Path) -> None:
        gbfs_dir = tmp_path / "empty_gbfs"
        gbfs_dir.mkdir()
        with load_gbfs(gbfs_dir) as con:
            tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
            assert len(tables) == 0

    def test_load_gbfs_nonexistent_directory(self) -> None:
        with load_gbfs("nonexistent_gbfs") as con:
            tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
            assert len(tables) == 0


class TestGetOdPairs:
    def test_get_od_pairs_basic(
        self, gtfs_connection: GtfsConnectionFactory, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        con = gtfs_connection(sample_gtfs_dict)
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

    @pytest.mark.parametrize(
        ("departure", "arrival", "expected_departure", "expected_arrival"),
        [
            pytest.param(
                "08:30:00", "08:35:00", "2024-01-01 08:30", "2024-01-01 08:35", id="same-day"
            ),
            pytest.param(
                "25:30:00", "25:35:00", "2024-01-02 01:30", "2024-01-02 01:35", id="after-midnight"
            ),
        ],
    )
    def test_od_timestamps_preserve_gtfs_service_day(
        self,
        gtfs_connection: GtfsConnectionFactory,
        sample_gtfs_dict: dict[str, pd.DataFrame],
        departure: str,
        arrival: str,
        expected_departure: str,
        expected_arrival: str,
    ) -> None:
        """Valid GTFS times become absolute timestamps without losing the service day."""
        sample_gtfs_dict["stop_times"] = pd.DataFrame(
            {
                "trip_id": ["trip1", "trip1"],
                "stop_id": ["stop1", "stop2"],
                "stop_sequence": [1, 2],
                "departure_time": [departure, arrival],
                "arrival_time": [departure, arrival],
            }
        )
        result = get_od_pairs(
            gtfs_connection(sample_gtfs_dict),
            start_date="20240101",
            end_date="20240101",
            include_geometry=False,
        )
        assert len(result) == 1
        assert result.iloc[0]["departure_ts"] == pd.Timestamp(expected_departure)
        assert result.iloc[0]["arrival_ts"] == pd.Timestamp(expected_arrival)
        assert result["travel_time_sec"].tolist() == [300.0]
        assert result["date"].tolist() == ["2024-01-01"]

    def test_get_od_pairs_date_range(
        self, gtfs_connection: GtfsConnectionFactory, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        con = gtfs_connection(sample_gtfs_dict)
        result = get_od_pairs(con, start_date="20240101", end_date="20240102")
        assert isinstance(result, gpd.GeoDataFrame)
        dates = pd.to_datetime(result["date"]).dt.date
        assert dates.min() >= datetime(2024, 1, 1).date()
        assert dates.max() <= datetime(2024, 1, 2).date()

    def test_get_od_pairs_no_geometry(
        self, gtfs_connection: GtfsConnectionFactory, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        con = gtfs_connection(sample_gtfs_dict)
        result = get_od_pairs(con, include_geometry=False)
        assert isinstance(result, pd.DataFrame)
        assert "geometry" not in result.columns

    def test_get_od_pairs_incomplete_gtfs(self, gtfs_connection: GtfsConnectionFactory) -> None:
        """A feed without trips or stop times cannot produce OD pairs."""
        con = gtfs_connection({"stops": pd.DataFrame(columns=["stop_id", "stop_lat", "stop_lon"])})
        assert get_od_pairs(con).empty

    def test_get_od_pairs_with_calendar_dates(
        self,
        gtfs_connection: GtfsConnectionFactory,
        sample_gtfs_dict_with_exceptions: dict[str, pd.DataFrame],
    ) -> None:
        con = gtfs_connection(sample_gtfs_dict_with_exceptions)
        result = get_od_pairs(con)
        assert len(result) > 0

    def test_get_od_pairs_with_malformed_times(
        self,
        gtfs_connection: GtfsConnectionFactory,
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
        con = gtfs_connection(gtfs_copy)
        result = get_od_pairs(con)
        assert isinstance(result, (pd.DataFrame, gpd.GeoDataFrame))

    def test_get_od_pairs_removal_exception_integration(
        self, gtfs_connection: GtfsConnectionFactory
    ) -> None:
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
        con = gtfs_connection(gtfs)
        result = get_od_pairs(con, include_geometry=False)
        assert len(result) == 0

    def test_get_od_pairs_directed(
        self, gtfs_connection: GtfsConnectionFactory, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        con = gtfs_connection(sample_gtfs_dict)
        result_directed = get_od_pairs(con, directed=True, include_geometry=False)
        result_undirected = get_od_pairs(con, directed=False, include_geometry=False)

        assert isinstance(result_directed, pd.DataFrame)
        assert isinstance(result_undirected, pd.DataFrame)
        # Undirected should have orig_stop_id <= dest_stop_id for all rows
        assert (result_undirected["orig_stop_id"] <= result_undirected["dest_stop_id"]).all()

    def test_get_od_pairs_calendar_dates_only_with_explicit_window(
        self, gtfs_connection: GtfsConnectionFactory, sample_gtfs_dict: dict[str, pd.DataFrame]
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

        con = gtfs_connection(gtfs)
        result = get_od_pairs(
            con,
            start_date="20240101",
            end_date="20240101",
            include_geometry=False,
        )

        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_get_od_pairs_without_calendar_and_without_dates_returns_empty(
        self, gtfs_connection: GtfsConnectionFactory, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        gtfs = sample_gtfs_dict.copy()
        gtfs.pop("calendar")

        con = gtfs_connection(gtfs)
        result = get_od_pairs(con, include_geometry=False)

        assert isinstance(result, pd.DataFrame)
        assert result.empty


class TestTravelSummaryGraph:
    @pytest.mark.parametrize(
        "value",
        [
            pytest.param("3600.0", id="bare-seconds"),
            pytest.param("", id="blank"),
            pytest.param("nan", id="nan-string"),
            pytest.param("None", id="none-string"),
        ],
    )
    def test_summary_rejects_malformed_time_bounds(
        self,
        gtfs_connection: GtfsConnectionFactory,
        sample_gtfs_dict: dict[str, pd.DataFrame],
        value: str,
    ) -> None:
        """Public time filters require HH:MM:SS, including extended GTFS hours."""
        with pytest.raises(ValueError, match=r"Invalid time format\. Expected HH:MM:SS"):
            travel_summary_graph(gtfs_connection(sample_gtfs_dict), start_time=value)

    def test_travel_summary_graph_basic(
        self, gtfs_connection: GtfsConnectionFactory, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        con = gtfs_connection(sample_gtfs_dict)
        nodes_gdf, edges_gdf = travel_summary_graph(con)
        assert isinstance(nodes_gdf, gpd.GeoDataFrame)
        assert len(nodes_gdf) > 0
        assert isinstance(edges_gdf, gpd.GeoDataFrame)
        assert len(edges_gdf) > 0

    def test_travel_summary_graph_time_filter(
        self, gtfs_connection: GtfsConnectionFactory, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        con = gtfs_connection(sample_gtfs_dict)
        nodes_gdf, edges_gdf = travel_summary_graph(con, start_time="07:00:00", end_time="09:00:00")
        assert isinstance(nodes_gdf, gpd.GeoDataFrame)
        assert isinstance(edges_gdf, gpd.GeoDataFrame)

    def test_travel_summary_graph_calendar_range(
        self, gtfs_connection: GtfsConnectionFactory, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        con = gtfs_connection(sample_gtfs_dict)
        nodes_gdf, edges_gdf = travel_summary_graph(
            con, calendar_start="20240101", calendar_end="20240107"
        )
        assert isinstance(nodes_gdf, gpd.GeoDataFrame)
        assert isinstance(edges_gdf, gpd.GeoDataFrame)

    def test_travel_summary_graph_as_nx_undirected(
        self, gtfs_connection: GtfsConnectionFactory, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        con = gtfs_connection(sample_gtfs_dict)
        result = travel_summary_graph(con, as_nx=True, directed=False)
        assert isinstance(result, nx.Graph)
        assert not isinstance(result, nx.DiGraph)

    def test_travel_summary_graph_as_nx_directed(
        self, gtfs_connection: GtfsConnectionFactory, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        con = gtfs_connection(sample_gtfs_dict)
        result = travel_summary_graph(con, as_nx=True, directed=True)
        assert isinstance(result, nx.DiGraph)

    def test_travel_summary_graph_missing_data(self) -> None:
        with duckdb.connect() as con:
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
        self, gtfs_connection: GtfsConnectionFactory, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        sample_gtfs_dict["stop_times"] = pd.DataFrame(
            columns=["trip_id", "stop_id", "arrival_time", "departure_time", "stop_sequence"]
        )
        con = gtfs_connection(sample_gtfs_dict)
        _nodes_gdf, edges_gdf = travel_summary_graph(con)
        assert len(edges_gdf) == 0

    def test_travel_summary_graph_with_calendar_dates(
        self,
        gtfs_connection: GtfsConnectionFactory,
        sample_gtfs_dict_with_exceptions: dict[str, pd.DataFrame],
    ) -> None:
        con = gtfs_connection(sample_gtfs_dict_with_exceptions)
        _nodes_gdf, edges_gdf = travel_summary_graph(
            con, calendar_start="20240101", calendar_end="20240102"
        )
        assert set(edges_gdf.index) == {("stop1", "stop2"), ("stop2", "stop3")}
        assert edges_gdf["frequency"].tolist() == [2.0, 2.0]
        assert edges_gdf["travel_time_sec"].tolist() == [300.0, 300.0]

    def test_travel_summary_graph_with_calendar_dates_add_service(
        self, gtfs_connection: GtfsConnectionFactory, gtfs_dict_add_service: dict[str, pd.DataFrame]
    ) -> None:
        con = gtfs_connection(gtfs_dict_add_service)
        _nodes_gdf, edges_gdf = travel_summary_graph(
            con, calendar_start="20240101", calendar_end="20240101"
        )
        assert len(edges_gdf) > 0

    def test_travel_summary_graph_with_calendar_dates_remove_service(
        self,
        gtfs_connection: GtfsConnectionFactory,
        gtfs_dict_remove_service: dict[str, pd.DataFrame],
    ) -> None:
        con = gtfs_connection(gtfs_dict_remove_service)
        _nodes_gdf, edges_gdf = travel_summary_graph(
            con, calendar_start="20240101", calendar_end="20240101"
        )
        assert len(edges_gdf) == 0

    def test_travel_summary_graph_missing_calendar(
        self, gtfs_connection: GtfsConnectionFactory, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        del sample_gtfs_dict["calendar"]
        con = gtfs_connection(sample_gtfs_dict)
        with pytest.raises(ValueError):
            travel_summary_graph(con, calendar_start="20240101", calendar_end="20240107")

    def test_travel_summary_graph_empty_calendar(
        self, gtfs_connection: GtfsConnectionFactory, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        sample_gtfs_dict["calendar"] = pd.DataFrame(
            columns=["service_id", "start_date", "end_date"]
        )
        con = gtfs_connection(sample_gtfs_dict)
        with pytest.raises(ValueError):
            travel_summary_graph(con, calendar_start="20240101", calendar_end="20240107")

    def test_travel_summary_graph_invalid_calendar_start_format(
        self, gtfs_connection: GtfsConnectionFactory, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        con = gtfs_connection(sample_gtfs_dict)
        with pytest.raises(ValueError):
            travel_summary_graph(con, calendar_start="invalid", calendar_end="20240107")

    def test_travel_summary_graph_invalid_calendar_end_format(
        self, gtfs_connection: GtfsConnectionFactory, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        con = gtfs_connection(sample_gtfs_dict)
        with pytest.raises(ValueError):
            travel_summary_graph(con, calendar_start="20240101", calendar_end="invalid")

    def test_travel_summary_graph_calendar_start_outside_range(
        self, gtfs_connection: GtfsConnectionFactory, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        con = gtfs_connection(sample_gtfs_dict)
        with pytest.raises(ValueError):
            travel_summary_graph(con, calendar_start="20230101", calendar_end="20240107")

    def test_travel_summary_graph_calendar_end_outside_range(
        self, gtfs_connection: GtfsConnectionFactory, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        con = gtfs_connection(sample_gtfs_dict)
        with pytest.raises(ValueError):
            travel_summary_graph(con, calendar_start="20240101", calendar_end="20250101")

    def test_travel_summary_graph_calendar_start_after_end(
        self, gtfs_connection: GtfsConnectionFactory, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        con = gtfs_connection(sample_gtfs_dict)
        with pytest.raises(ValueError):
            travel_summary_graph(con, calendar_start="20240107", calendar_end="20240101")

    def test_travel_summary_graph_calendar_start_only_invalid(
        self, gtfs_connection: GtfsConnectionFactory, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        con = gtfs_connection(sample_gtfs_dict)
        with pytest.raises(ValueError):
            travel_summary_graph(con, calendar_start="20230101")

    def test_travel_summary_graph_calendar_end_only_invalid(
        self, gtfs_connection: GtfsConnectionFactory, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        con = gtfs_connection(sample_gtfs_dict)
        with pytest.raises(ValueError):
            travel_summary_graph(con, calendar_end="20250101")

    def test_travel_summary_graph_directed_edges(
        self, gtfs_connection: GtfsConnectionFactory, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        con = gtfs_connection(sample_gtfs_dict)
        _nodes, _edges_directed = travel_summary_graph(con, directed=True)
        _nodes2, edges_undirected = travel_summary_graph(con, directed=False)

        # Undirected: from_stop_id <= to_stop_id in every index entry
        for from_id, to_id in edges_undirected.index:
            assert from_id <= to_id

    def test_travel_summary_graph_with_frequencies(
        self,
        gtfs_connection: GtfsConnectionFactory,
        sample_gtfs_dict_with_frequencies: dict[str, pd.DataFrame],
    ) -> None:
        con = gtfs_connection(sample_gtfs_dict_with_frequencies)
        _nodes, edges_with = travel_summary_graph(con, use_frequencies=True)
        assert len(edges_with) > 0
        # frequency should be higher because of headway expansion
        assert (edges_with["frequency"] > 0).all()

    def test_travel_summary_graph_without_frequencies(
        self,
        gtfs_connection: GtfsConnectionFactory,
        sample_gtfs_dict_with_frequencies: dict[str, pd.DataFrame],
    ) -> None:
        con = gtfs_connection(sample_gtfs_dict_with_frequencies)
        _nodes, edges_without = travel_summary_graph(con, use_frequencies=False)
        assert len(edges_without) > 0

    def test_travel_summary_graph_without_calendar_falls_back_to_trip_counts(
        self, gtfs_connection: GtfsConnectionFactory, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        gtfs = sample_gtfs_dict.copy()
        gtfs.pop("calendar")
        con = gtfs_connection(gtfs)

        nodes_gdf, edges_gdf = travel_summary_graph(con)

        assert isinstance(nodes_gdf, gpd.GeoDataFrame)
        assert isinstance(edges_gdf, gpd.GeoDataFrame)
        assert len(edges_gdf) > 0

    def test_travel_summary_graph_calendar_dates_only_feed(
        self, gtfs_connection: GtfsConnectionFactory, sample_gtfs_dict: dict[str, pd.DataFrame]
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
        con = gtfs_connection(gtfs)

        _nodes_gdf, edges_gdf = travel_summary_graph(con)

        assert len(edges_gdf) > 0

    def test_travel_summary_graph_calendar_dates_only_empty_table_errors_with_window(
        self, gtfs_connection: GtfsConnectionFactory, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        gtfs = sample_gtfs_dict.copy()
        gtfs.pop("calendar")
        gtfs["calendar_dates"] = pd.DataFrame(columns=["service_id", "date", "exception_type"])
        con = gtfs_connection(gtfs)

        with pytest.raises(ValueError, match="contain no usable dates"):
            travel_summary_graph(con, calendar_start="20240101", calendar_end="20240102")

    def test_travel_summary_graph_calendar_dates_only_empty_table_falls_back_without_window(
        self, gtfs_connection: GtfsConnectionFactory, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        gtfs = sample_gtfs_dict.copy()
        gtfs.pop("calendar")
        gtfs["calendar_dates"] = pd.DataFrame(columns=["service_id", "date", "exception_type"])
        con = gtfs_connection(gtfs)

        _nodes_gdf, edges_gdf = travel_summary_graph(con)

        assert edges_gdf["frequency"].gt(0).all()

    def test_travel_summary_graph_without_calendar_rejects_explicit_window(
        self, gtfs_connection: GtfsConnectionFactory, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        gtfs = sample_gtfs_dict.copy()
        gtfs.pop("calendar")
        con = gtfs_connection(gtfs)

        with pytest.raises(ValueError, match=r"has neither calendar\.txt nor calendar_dates\.txt"):
            travel_summary_graph(con, calendar_start="20240101", calendar_end="20240102")

    def test_travel_summary_graph_rejects_invalid_time_format(
        self, gtfs_connection: GtfsConnectionFactory, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        con = gtfs_connection(sample_gtfs_dict)

        with pytest.raises(ValueError, match="Invalid time format"):
            travel_summary_graph(con, start_time="invalid")

    def test_travel_summary_graph_rejects_non_numeric_time_components(
        self, gtfs_connection: GtfsConnectionFactory, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        con = gtfs_connection(sample_gtfs_dict)

        with pytest.raises(ValueError, match="Invalid time format"):
            travel_summary_graph(con, start_time="aa:bb:cc")

    def test_travel_summary_graph_rejects_inverted_time_window(
        self, gtfs_connection: GtfsConnectionFactory, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        con = gtfs_connection(sample_gtfs_dict)

        with pytest.raises(ValueError, match="start_time must be <="):
            travel_summary_graph(con, start_time="09:00:00", end_time="08:00:00")

    def test_travel_summary_graph_requires_stop_coordinates_or_geometry(
        self, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        with duckdb.connect(":memory:") as con:
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

    def test_travel_summary_graph_as_nx_follows_shared_conventions(
        self, gtfs_connection: GtfsConnectionFactory, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        con = gtfs_connection(sample_gtfs_dict)
        graph = travel_summary_graph(con, as_nx=True)

        assert isinstance(graph, nx.Graph)
        assert str(graph.graph["crs"]) == "EPSG:4326"
        assert graph.graph["is_hetero"] is False
        assert all(isinstance(node_id, int) for node_id in graph.nodes)

        node_attrs = dict(graph.nodes(data=True))
        original_ids = {attrs["_original_index"] for attrs in node_attrs.values()}
        assert original_ids == {"stop1", "stop2", "stop3"}
        assert all("pos" in attrs for attrs in node_attrs.values())

        assert graph.number_of_edges() > 0
        for _u, _v, attrs in graph.edges(data=True):
            assert attrs["travel_time_sec"] > 0
            assert attrs["frequency"] > 0
            assert "_original_edge_index" in attrs

    def test_travel_summary_graph_as_nx_empty_edges(
        self, gtfs_connection: GtfsConnectionFactory, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        sample_gtfs_dict["stop_times"] = pd.DataFrame(
            columns=["trip_id", "stop_id", "arrival_time", "departure_time", "stop_sequence"]
        )
        con = gtfs_connection(sample_gtfs_dict)
        graph = travel_summary_graph(con, as_nx=True)

        assert isinstance(graph, nx.Graph)
        assert graph.number_of_nodes() == 3
        assert graph.number_of_edges() == 0

    def test_travel_summary_graph_as_nx_time_filtered_directed(
        self, gtfs_connection: GtfsConnectionFactory, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        con = gtfs_connection(sample_gtfs_dict)
        graph = travel_summary_graph(
            con,
            start_time="08:00:00",
            end_time="08:10:00",
            as_nx=True,
            directed=True,
            use_frequencies=False,
        )

        assert isinstance(graph, nx.DiGraph)
        assert graph.number_of_edges() == 2
        for _u, _v, attrs in graph.edges(data=True):
            assert attrs["travel_time_sec"] > 0
            assert attrs["frequency"] > 0

    def test_travel_summary_graph_instant_time_window(
        self, gtfs_connection: GtfsConnectionFactory, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        con = gtfs_connection(sample_gtfs_dict)
        nodes_gdf, edges_gdf = travel_summary_graph(con, start_time="08:00:00", end_time="08:00:00")

        assert len(nodes_gdf) == 3
        assert len(edges_gdf) == 0

    def test_travel_summary_graph_surfaces_udf_registration_failures(self) -> None:
        with pytest.raises(duckdb.Error, match="UDF registration failed"):
            travel_summary_graph(cast("duckdb.DuckDBPyConnection", FailingUdfConnection()))
