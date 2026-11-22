"""Tests for the transportation module."""

import copy
import io
import json
import typing
import urllib.error
import urllib.parse
import zipfile
from collections.abc import Callable
from collections.abc import Iterator
from contextlib import ExitStack
from datetime import date

# ruff: noqa: D101, D102, S608, PT011
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Literal
from typing import cast

import duckdb
import geopandas as gpd
import networkx as nx
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from shapely.geometry import LineString
from shapely.geometry import Point

import city2graph as c2g
from city2graph import TransportData
from city2graph.transportation import _time_to_seconds
from city2graph.transportation import _timestamp
from city2graph.transportation import get_od_pairs
from city2graph.transportation import load_gbfs
from city2graph.transportation import load_gtfs
from city2graph.transportation import travel_summary_graph
from tests.city2graph.helpers import assert_as_nx_deprecation
from tests.city2graph.helpers import temporary_gtfs_zip

GtfsConnectionFactory = Callable[[dict[str, pd.DataFrame]], duckdb.DuckDBPyConnection]


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
                "stop4,Stop 4,0,NaN\n"
                "stop5,Stop 5,Infinity,0\n"
                "stop6,Stop 6,91,0\n"
                "stop7,Stop 7,0,-181\n"
                "stop8,Stop 8,0,0\n"
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


@pytest.fixture
def sample_gtfs_dict() -> dict[str, pd.DataFrame]:
    """Return a canonical in-memory GTFS feed used across transportation tests."""
    return {
        "stops": gpd.GeoDataFrame(
            {
                "stop_id": ["stop1", "stop2", "stop3"],
                "stop_name": ["Stop 1", "Stop 2", "Stop 3"],
                "stop_lat": [40.7128, 40.7589, 40.7505],
                "stop_lon": [-74.0060, -73.9851, -73.9934],
                "geometry": [
                    Point(-74.0060, 40.7128),
                    Point(-73.9851, 40.7589),
                    Point(-73.9934, 40.7505),
                ],
            },
            crs="EPSG:4326",
        ),
        "routes": pd.DataFrame(
            {
                "route_id": ["route1"],
                "agency_id": ["1"],
                "route_short_name": ["1"],
                "route_long_name": ["Test Route 1"],
                "route_type": [3],
            },
        ),
        "trips": pd.DataFrame(
            {
                "route_id": ["route1", "route1"],
                "service_id": ["service1", "service1"],
                "trip_id": ["trip1", "trip2"],
            },
        ),
        "stop_times": pd.DataFrame(
            {
                "trip_id": ["trip1", "trip1", "trip1", "trip2", "trip2", "trip2"],
                "arrival_time": [
                    "08:00:00",
                    "08:05:00",
                    "08:10:00",
                    "09:00:00",
                    "09:05:00",
                    "09:10:00",
                ],
                "departure_time": [
                    "08:00:00",
                    "08:05:00",
                    "08:10:00",
                    "09:00:00",
                    "09:05:00",
                    "09:10:00",
                ],
                "stop_id": ["stop1", "stop2", "stop3", "stop1", "stop2", "stop3"],
                "stop_sequence": [1, 2, 3, 1, 2, 3],
            },
        ),
        "calendar": pd.DataFrame(
            {
                "service_id": ["service1"],
                "monday": [True],
                "tuesday": [True],
                "wednesday": [True],
                "thursday": [True],
                "friday": [True],
                "saturday": [False],
                "sunday": [False],
                "start_date": ["20240101"],
                "end_date": ["20241231"],
            },
        ),
    }


@pytest.fixture
def gtfs_connection(tmp_path: Path) -> Iterator[GtfsConnectionFactory]:
    """Load independent GTFS feeds and close every connection after the test."""
    with ExitStack() as connections:
        counter = 0

        def load(tables: dict[str, pd.DataFrame]) -> duckdb.DuckDBPyConnection:
            nonlocal counter
            path = tmp_path / f"feed-{counter}.zip"
            counter += 1
            with zipfile.ZipFile(path, "w") as archive:
                for name, frame in tables.items():
                    data = frame.drop(columns="geometry", errors="ignore").copy()
                    for column in data.select_dtypes(include="bool"):
                        data[column] = data[column].astype(int)
                    archive.writestr(f"{name}.txt", data.to_csv(index=False))
            return connections.enter_context(load_gtfs(path))

        yield load


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
    @pytest.mark.parametrize("failure", ["setup", "metadata", "metadata_table", "geometry"])
    def test_loading_failure_closes_connection(
        self,
        sample_gtfs_zip: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        failure: str,
    ) -> None:
        with zipfile.ZipFile(sample_gtfs_zip) as archive:
            files = {name: archive.read(name) for name in archive.namelist()}
        if failure == "metadata":
            files["city2graph_metadata.json"] = b'{"version": 2, "records": []}'
        elif failure == "metadata_table":
            files["city2graph_metadata.json"] = b'{"version": 1, "records": []}'
            files["_city2graph_metadata.txt"] = b"field\nexisting\n"
        elif failure == "geometry":
            files["stops.txt"] = b"stop_id,stop_name,stop_lat\nA,A,51.5\n"
        path = tmp_path / "feed.zip"
        with zipfile.ZipFile(path, "w") as archive:
            for name, content in files.items():
                archive.writestr(name, content)
        connection = duckdb.connect(":memory:")
        monkeypatch.setattr(
            "city2graph.transportation.duckdb.connect", lambda _database: connection
        )
        if failure == "setup":

            def unavailable(_con: duckdb.DuckDBPyConnection) -> None:
                raise RuntimeError

            monkeypatch.setattr("city2graph.transportation._ensure_graph_sql_support", unavailable)
        with pytest.raises((RuntimeError, ValueError, duckdb.Error)):
            c2g.load_gtfs(path)
        with pytest.raises(duckdb.ConnectionException):
            connection.execute("SELECT 1")

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
                ("stop4", True),
                ("stop5", True),
                ("stop6", True),
                ("stop7", True),
                ("stop8", False),
            ]

    def test_load_gtfs_requires_coordinate_columns_for_optional_locations(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "missing-coordinate-columns.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("stops.txt", "stop_id,location_type\ngeneric-node,3\n")

        with pytest.raises(ValueError, match="both stop_lon and stop_lat columns"):
            load_gtfs(path)

    def test_load_gtfs_keeps_optional_locations_with_blank_coordinates(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "blank-optional-coordinates.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(
                "stops.txt",
                "stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station\n"
                "station,Station,51,-1,1,\n"
                "platform,Platform,51,-1,0,station\n"
                "junction,Junction,,,3,station\n"
                "boarding-area,Boarding Area,,,4,platform\n",
            )

        with load_gtfs(path) as con:
            rows = con.execute(
                "SELECT stop_id, geometry IS NULL FROM stops ORDER BY stop_id"
            ).fetchall()
        assert rows == [
            ("boarding-area", True),
            ("junction", True),
            ("platform", False),
            ("station", False),
        ]

    def test_load_gtfs_nonexistent_file(self) -> None:
        with load_gtfs("nonexistent.zip") as con:
            assert len(con.execute("SHOW TABLES").fetchall()) == 0


class TestGetGtfsSummary:
    """Tests for the service report of a generated feed."""

    @pytest.mark.parametrize(("exact", "expected"), [(0, 2.5), (1, 3)])
    def test_report_expected_and_exact_departures(
        self, transport_input: TransportData, exact: int, expected: float
    ) -> None:
        services = pd.DataFrame([{"pattern_id": "local", "exact_times": exact}])
        with c2g.create_gtfs(
            transport_input,
            services,
            pattern_ids="local",
            start_time="08:00:00",
            end_time="08:25:00",
            headway_secs=600,
            service_date="20260105",
        ) as con:
            report = c2g.get_gtfs_summary(con)
            assert report["services"]["departures"].tolist() == [expected]
            assert set(report["edges"]["frequency"]) == {expected}
            con.execute("DELETE FROM _city2graph_metadata WHERE table_name='assumptions'")
            with pytest.raises(ValueError, match="no effective assumptions"):
                c2g.get_gtfs_summary(con)


class TestLoadGbfs:
    def test_bad_json_does_not_hide_valid_station_geometries(self, tmp_path: Path) -> None:
        (tmp_path / "bad.json").write_text("{invalid", encoding="utf-8")
        (tmp_path / "station_information.json").write_text(
            '{"data":{"stations":[{"station_id":"a","lat":40,"lon":-74}]}}',
            encoding="utf-8",
        )
        with load_gbfs(tmp_path) as con:
            assert con.execute(
                "SELECT station_id, ST_AsText(geometry) FROM station_information"
            ).fetchall() == [("a", "POINT (-74 40)")]

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
    @pytest.mark.parametrize(
        ("start", "end", "expected"),
        [
            (None, None, {"2024-01-01", "2024-01-03"}),
            ("20240102", None, {"2024-01-03"}),
            (None, "20240102", {"2024-01-01"}),
            ("20230101", "20250101", {"2024-01-01", "2024-01-03"}),
        ],
    )
    @pytest.mark.parametrize("with_calendar", [False, True])
    def test_inferred_date_bounds_and_exceptions(
        self,
        gtfs_connection: GtfsConnectionFactory,
        sample_gtfs_dict: dict[str, pd.DataFrame],
        start: str | None,
        end: str | None,
        expected: set[str],
        with_calendar: bool,
    ) -> None:
        gtfs = sample_gtfs_dict.copy()
        if with_calendar:
            gtfs["calendar"] = gtfs["calendar"].assign(start_date="20240101", end_date="20240102")
        else:
            gtfs.pop("calendar")
        gtfs["calendar_dates"] = pd.DataFrame(
            {
                "service_id": ["service1"] * 3,
                "date": ["20240101", "20240102", "20240103"],
                "exception_type": [1, 2, 1],
            }
        )
        with gtfs_connection(gtfs) as con:
            result = get_od_pairs(con, start_date=start, end_date=end, include_geometry=False)
            assert set(result["date"]) == expected

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

    @pytest.mark.parametrize("with_coordinates", [True, False])
    def test_get_od_pairs_with_unmaterialised_stop_geometry(
        self,
        gtfs_connection: GtfsConnectionFactory,
        sample_gtfs_dict: dict[str, pd.DataFrame],
        with_coordinates: bool,
    ) -> None:
        with gtfs_connection(sample_gtfs_dict) as con:
            con.execute("ALTER TABLE stops DROP COLUMN geometry")
            if not with_coordinates:
                con.execute("ALTER TABLE stops DROP COLUMN stop_lon")
                con.execute("ALTER TABLE stops DROP COLUMN stop_lat")
            first = get_od_pairs(con, "20240101", "20240101")
            second = get_od_pairs(con, "20240101", "20240101")

        assert not first.empty
        if with_coordinates:
            assert isinstance(first, gpd.GeoDataFrame)
            assert isinstance(second, gpd.GeoDataFrame)
            assert first.geometry.notna().all()
            assert second.geometry.notna().all()
            pd.testing.assert_frame_equal(
                first.drop(columns="geometry"), second.drop(columns="geometry")
            )
        else:
            assert "geometry" not in first.columns
            assert "geometry" not in second.columns

    def test_repeated_calls_handle_temp_stops_view_and_table_transitions(
        self,
        gtfs_connection: GtfsConnectionFactory,
        sample_gtfs_dict: dict[str, pd.DataFrame],
    ) -> None:
        """Repeated calls reuse a connection cleanly across temp view and temp table modes."""
        with gtfs_connection(sample_gtfs_dict) as con:
            # First creates TEMP VIEW _stops_for_graph (stops has geometry column).
            _, edges_view = travel_summary_graph(con)
            # Drop geometry so subsequent calls create TEMP TABLE _stops_for_graph.
            con.execute("ALTER TABLE stops DROP COLUMN geometry")
            _, edges_table_1 = travel_summary_graph(con)
            _, edges_table_2 = travel_summary_graph(con)
            od_1 = get_od_pairs(con, "20240101", "20240101")
            od_2 = get_od_pairs(con, "20240101", "20240101")

        assert len(edges_view) == len(edges_table_1) == len(edges_table_2)
        assert isinstance(od_1, gpd.GeoDataFrame)
        assert isinstance(od_2, gpd.GeoDataFrame)
        assert len(od_1) == len(od_2) > 0

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

    @pytest.mark.parametrize("column", ["departure_time", "arrival_time"])
    def test_get_od_pairs_all_missing_times_returns_an_empty_frame(
        self,
        gtfs_connection: GtfsConnectionFactory,
        sample_gtfs_dict: dict[str, pd.DataFrame],
        column: str,
    ) -> None:
        with gtfs_connection(sample_gtfs_dict) as con:
            con.execute(f"UPDATE stop_times SET {column} = NULL")
            result = get_od_pairs(con, "20240101", "20240101", include_geometry=False)
        assert result.empty
        assert pd.api.types.is_datetime64_any_dtype(result["departure_ts"])
        assert pd.api.types.is_datetime64_any_dtype(result["arrival_ts"])

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
        pd.testing.assert_frame_equal(result_directed, result_undirected)

    @pytest.mark.parametrize("include_geometry", [False, True])
    @pytest.mark.parametrize("directed", [False, True])
    def test_od_pair_keeps_actual_departure_stop_and_trip_times(
        self,
        gtfs_connection: GtfsConnectionFactory,
        sample_gtfs_dict: dict[str, pd.DataFrame],
        include_geometry: bool,
        directed: bool,
    ) -> None:
        sample_gtfs_dict["stop_times"] = pd.DataFrame(
            {
                "trip_id": ["trip1", "trip1"],
                "stop_id": ["stop2", "stop1"],
                "stop_sequence": [1, 2],
                "departure_time": ["08:00:00", "08:10:00"],
                "arrival_time": ["08:00:00", "08:10:00"],
            }
        )
        with gtfs_connection(sample_gtfs_dict) as con:
            actual = get_od_pairs(
                con, "20240101", "20240101", directed=directed, include_geometry=include_geometry
            )

        assert actual.loc[actual["orig_stop_id"] == "stop1"].empty
        assert len(actual.loc[actual["orig_stop_id"] == "stop2"]) == 1
        row = actual.iloc[0]
        assert (row["orig_stop_id"], row["dest_stop_id"]) == ("stop2", "stop1")
        assert row["departure_ts"] == pd.Timestamp("2024-01-01 08:00")
        assert row["arrival_ts"] == pd.Timestamp("2024-01-01 08:10")
        assert row["travel_time_sec"] == 600.0
        if include_geometry:
            assert isinstance(actual, gpd.GeoDataFrame)
            stops = sample_gtfs_dict["stops"].set_index("stop_id")
            assert list(row.geometry.coords) == [
                (stops.loc[stop_id, "stop_lon"], stops.loc[stop_id, "stop_lat"])
                for stop_id in ("stop2", "stop1")
            ]

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
    def test_undirected_summary_orients_geometry_to_its_stop_index(
        self, gtfs_connection: GtfsConnectionFactory, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        sample_gtfs_dict["stop_times"] = pd.DataFrame(
            {
                "trip_id": ["trip1", "trip1"],
                "stop_id": ["stop2", "stop1"],
                "stop_sequence": [1, 2],
                "departure_time": ["08:00:00", "08:10:00"],
                "arrival_time": ["08:00:00", "08:10:00"],
            }
        )
        with gtfs_connection(sample_gtfs_dict) as con:
            _, actual = travel_summary_graph(con, directed=False)
            _, directed = travel_summary_graph(con, directed=True)

        assert actual.index.tolist() == [("stop1", "stop2")]
        assert directed.index.tolist() == [("stop2", "stop1")]
        assert list(actual.iloc[0].geometry.coords) == list(directed.iloc[0].geometry.coords)[::-1]

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

    @pytest.mark.parametrize("prebuilt_geometry", [True, False], ids=["loaded", "numeric-fallback"])
    def test_travel_summary_graph_retains_missing_geometry_in_frames(
        self,
        gtfs_connection: GtfsConnectionFactory,
        sample_gtfs_dict: dict[str, pd.DataFrame],
        prebuilt_geometry: bool,
    ) -> None:
        stops = sample_gtfs_dict["stops"]
        stops.loc[stops["stop_id"] == "stop3", "stop_lat"] = 91.0
        with gtfs_connection(sample_gtfs_dict) as con:
            if not prebuilt_geometry:
                con.execute("ALTER TABLE stops DROP COLUMN geometry")
                con.execute("ALTER TABLE stops ALTER COLUMN stop_lat TYPE DOUBLE")
                con.execute("ALTER TABLE stops ALTER COLUMN stop_lon TYPE DOUBLE")
            nodes, edges = travel_summary_graph(
                con, calendar_start="20240101", calendar_end="20240101"
            )

        assert nodes.geometry.isna().to_dict() == {"stop1": False, "stop2": False, "stop3": True}
        assert edges.geometry.isna().to_dict() == {
            ("stop1", "stop2"): False,
            ("stop2", "stop3"): True,
        }
        assert edges["frequency"].to_dict() == {("stop1", "stop2"): 2, ("stop2", "stop3"): 2}

        graph = c2g.gdf_to_nx(nodes=nodes, edges=edges)
        assert {attrs["_original_index"] for _, attrs in graph.nodes(data=True)} == {
            "stop1",
            "stop2",
        }
        assert graph.number_of_edges() == 1

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

    @pytest.mark.parametrize("column", ["start_time", "end_time"])
    def test_travel_summary_graph_rejects_missing_frequency_boundary(
        self,
        gtfs_connection: GtfsConnectionFactory,
        sample_gtfs_dict_with_frequencies: dict[str, pd.DataFrame],
        column: str,
    ) -> None:
        with gtfs_connection(sample_gtfs_dict_with_frequencies) as con:
            con.execute(f"UPDATE frequencies SET {column} = NULL")
            with pytest.raises(ValueError, match="Invalid or overlapping frequency periods"):
                travel_summary_graph(con, use_frequencies=True)

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

    def test_travel_summary_as_nx_warning(
        self, gtfs_connection: GtfsConnectionFactory, sample_gtfs_dict: dict[str, pd.DataFrame]
    ) -> None:
        """Summary graph calls retain the as_nx warning contract."""
        assert_as_nx_deprecation(travel_summary_graph, gtfs_connection(sample_gtfs_dict))


@pytest.fixture
def overpass_response(
    osm_response: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    """Serve the test's OSM elements on every Overpass request."""
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: io.BytesIO(json.dumps(osm_response).encode()),
    )
    return osm_response


@pytest.fixture
def osm_services() -> pd.DataFrame:
    """One scheduled departure for the synthetic OSM pattern."""
    return pd.DataFrame(
        [{"pattern_id": "osm:relation:42", "service_id": "weekday", "departure_time": "08:00:00"}]
    )


@pytest.mark.filterwarnings("ignore::city2graph.GTFSGenerationWarning")
class TestLoadOsmTransport:
    @pytest.mark.parametrize(
        "case",
        [
            "local_ref",
            "platform_name",
            "near_way",
            "inside_polygon",
            "near_point",
            "far",
            "conflict",
            "untagged",
            "missing_geometry",
        ],
    )
    def test_tagged_stop_platform_pairing(
        self, overpass_response: dict[str, Any], case: str
    ) -> None:
        """Resolve adjacent tagged platforms without conflating station and track refs."""
        stop = overpass_response["elements"][1]
        platform = overpass_response["elements"][2]
        stop["tags"].update(public_transport="stop_position", ref="G01")
        platform["tags"] = {"public_transport": "platform", "ref": "2"}
        if case in {"local_ref", "platform_name"}:
            stop["tags"]["local_ref"] = "2"
            platform["tags"]["ref"] = "1;2"
            if case == "platform_name":
                platform["tags"]["name"] = "Station A platforms 1 and 2"
        elif case in {"near_way", "inside_polygon", "missing_geometry"}:
            platform.update(type="way")
            platform.pop("lat")
            platform.pop("lon")
            overpass_response["elements"][0]["members"][1]["type"] = "way"
            if case == "near_way":
                platform["geometry"] = [
                    {"lon": -0.001, "lat": 51.0001},
                    {"lon": 0.001, "lat": 51.0001},
                ]
            elif case == "inside_polygon":
                platform["geometry"] = [
                    {"lon": x, "lat": y}
                    for x, y in [
                        (-0.001, 50.999),
                        (0.001, 50.999),
                        (0.001, 51.001),
                        (-0.001, 51.001),
                        (-0.001, 50.999),
                    ]
                ]
        elif case == "far":
            platform["lat"] = 51.01
        elif case == "conflict":
            platform["tags"]["name"] = "Another station"
        elif case == "untagged":
            platform["tags"].pop("public_transport")
        if case in {"far", "missing_geometry"}:
            with pytest.raises(ValueError, match=r"missing stop_name|no usable geometry"):
                c2g.load_osm_transport(relation_ids=[42], service_date="20260105")
        else:
            with c2g.load_osm_transport(relation_ids=[42], service_date="20260105") as con:
                assert len(con.table("stops").df()) == 2
                assert con.table("stop_times").df().stop_id.tolist() == ["osm:node:1", "osm:node:2"]

    @pytest.mark.parametrize("level", ["station", "platform"])
    @pytest.mark.parametrize("ambiguous", [False, True])
    def test_stop_area_grouping_and_platform_only_stops(
        self,
        monkeypatch: pytest.MonkeyPatch,
        osm_response: dict[str, Any],
        level: Literal["station", "platform"],
        ambiguous: bool,
    ) -> None:
        """Keep a platform-only next station and consolidate only unique memberships."""
        relation = osm_response["elements"][0]
        # Station B is represented only by its area platform in this pattern.
        relation["members"] = [
            relation["members"][0],
            {"type": "way", "ref": 20, "role": "platform"},
        ]
        osm_response["elements"].extend(
            [
                {
                    "type": "way",
                    "id": 20,
                    "tags": {"name": "B", "public_transport": "platform"},
                    "geometry": [{"lon": 0.01, "lat": 51}, {"lon": 0.0101, "lat": 51}],
                },
                {
                    "type": "relation",
                    "id": 101,
                    "tags": {"name": "Station A", "public_transport": "stop_area"},
                    "members": [
                        {"type": "node", "ref": 1, "role": "stop"},
                        {"type": "node", "ref": 11, "role": "platform"},
                    ],
                },
                {
                    "type": "relation",
                    "id": 102,
                    "tags": {"name": "Station B", "public_transport": "stop_area"},
                    "members": [{"type": "way", "ref": 20, "role": "platform"}],
                },
            ]
        )
        if ambiguous:
            duplicate = copy.deepcopy(osm_response["elements"][-1])
            duplicate["id"] = 103
            osm_response["elements"].append(duplicate)
        queries = []

        def fetch(request: urllib.request.Request, **_kwargs: object) -> io.BytesIO:
            assert isinstance(request.data, bytes)
            query = urllib.parse.parse_qs(request.data.decode())["data"][0]
            queries.append(query)
            station_request = "->.boarding;" in query
            elements = [
                row
                for row in osm_response["elements"]
                if (row.get("tags", {}).get("public_transport") == "stop_area") == station_request
            ]
            return io.BytesIO(json.dumps({"elements": elements}).encode())

        monkeypatch.setattr("urllib.request.urlopen", fetch)
        with c2g.load_osm_transport(relation_ids=[42], stop_level=level, on_invalid="raise") as con:
            ids = con.table("stop_times").df().stop_id.tolist()
            assert ids == (
                ["osm:stop_area:101", "osm:way:20" if ambiguous else "osm:stop_area:102"]
                if level == "station"
                else ["osm:node:1", "osm:way:20"]
            )
            nodes, edges = c2g.travel_summary_graph(con)
            assert len(nodes) == 2
            assert len(edges) == 1
            assert len(queries) == (2 if level == "station" else 1)

    def test_platform_multipolygon_and_default_exclusions(
        self,
        overpass_response: dict[str, Any],
    ) -> None:
        """Read platform multipolygons and keep usable routes under default policy."""
        relation = overpass_response["elements"][0]
        relation["members"][1] = {"type": "relation", "ref": 200, "role": "platform"}
        overpass_response["elements"].extend(
            [
                {
                    "type": "relation",
                    "id": 200,
                    "tags": {"type": "multipolygon", "public_transport": "platform"},
                    "members": [{"type": "way", "ref": 201, "role": "outer"}],
                },
                {
                    "type": "way",
                    "id": 201,
                    "geometry": [
                        {"lon": x, "lat": y}
                        for x, y in [
                            (-0.001, 50.999),
                            (0.001, 50.999),
                            (0.001, 51.001),
                            (-0.001, 51.001),
                            (-0.001, 50.999),
                        ]
                    ],
                },
            ]
        )
        invalid = copy.deepcopy(relation)
        invalid["id"] = 43
        invalid["tags"].pop("public_transport:version")
        overpass_response["elements"].append(invalid)
        with pytest.warns(c2g.GTFSGenerationWarning, match="excluded 1 OSM relation"):
            con = c2g.load_osm_transport(relation_ids=[42, 43])
        with con:
            assert con.table("stop_times").df().stop_id.tolist() == ["osm:node:1", "osm:node:2"]
            assert c2g.get_gtfs_summary(con)["summary"].iloc[0]["excluded_relation_count"] == 1

    @pytest.mark.usefixtures("overpass_response")
    def test_reuse_loaded_stops_as_supplement(self, tmp_path: Path) -> None:
        with c2g.load_osm_transport(relation_ids=[42], service_date="20261103") as original:
            expected = original.table("stop_times").df()
            archive = c2g.save_gtfs(original, tmp_path / "osm.zip")
        with c2g.load_gtfs(archive) as restored:
            stops = restored.table("stops").df()
        before = stops.copy(deep=True)
        with c2g.load_osm_transport(
            relation_ids=[42], data={"stops": stops}, service_date="20261103"
        ) as regenerated:
            assert_frame_equal(expected, regenerated.table("stop_times").df())
        assert_frame_equal(before, stops)

    def test_daily_trip_ids_avoid_explicit_ids_and_preserve_shared_periods(
        self, overpass_response: dict[str, Any], tmp_path: Path
    ) -> None:
        second = copy.deepcopy(overpass_response["elements"][0])
        second["id"] = 43
        overpass_response["elements"][0]["tags"]["opening_hours"] = "Mo-Su 06:00-09:00,17:00-21:00"
        overpass_response["elements"].append(second)
        services = pd.DataFrame(
            [
                {"pattern_id": "osm:relation:42", "trip_id": "named", "headway_secs": 900},
                {
                    "pattern_id": "osm:relation:43",
                    "trip_id": "named:c2g:20261103",
                    "departure_time": "08:00:00",
                },
            ]
        )
        with c2g.load_osm_transport(
            services, relation_ids=[42, 43], service_date="20261103"
        ) as con:
            trips = con.table("trips").df()
            assert len(trips) == 2
            assert trips["trip_id"].is_unique
            assert "named:c2g:20261103" in set(trips["trip_id"])
            frequencies = con.table("frequencies").df()
            assert len(frequencies) == 2
            assert frequencies["trip_id"].eq(frequencies["trip_id"].iloc[0]).all()
            assert frequencies["trip_id"].iloc[0] != "named:c2g:20261103"
            report = c2g.get_gtfs_summary(con)
            assert report["services"]["departures"].sum() == 29
            archive = c2g.save_gtfs(con, tmp_path / "daily.zip")
        with c2g.load_gtfs(archive) as restored:
            restored_report = c2g.get_gtfs_summary(restored)
            assert restored_report.keys() == report.keys()
            for name, frame in report.items():
                assert_frame_equal(frame, restored_report[name])

    @pytest.mark.parametrize(
        "options",
        [
            {"modes": []},
            {"modes": "plane"},
            {"pattern_ids": []},
            {"on_invalid": "ignore"},
            {"headway_sec": 600},
            {"data": {"stop_sequences": pd.DataFrame([{"travel_time_sec": 600}])}},
            {"services": pd.DataFrame([{"headway_sec": 600}])},
        ],
    )
    def test_preflight_rejects_before_geocoding(
        self, monkeypatch: pytest.MonkeyPatch, options: dict[str, Any]
    ) -> None:
        def unexpected(*_args: object, **_kwargs: object) -> None:
            pytest.fail("Input validation must precede network access")

        monkeypatch.setattr("city2graph.data.get_boundaries", unexpected)
        monkeypatch.setattr("urllib.request.urlopen", unexpected)
        with pytest.raises(
            ValueError, match=r"modes|pattern_ids|on_invalid|headway|services|travel_time_sec"
        ):
            c2g.load_osm_transport(place_name="Liverpool, UK", **options)

    @pytest.mark.parametrize(("modes", "count"), [("bus", 1), (["bus", "train"], 2), (None, 2)])
    def test_modes_filter_query_and_result(
        self,
        monkeypatch: pytest.MonkeyPatch,
        osm_response: dict[str, Any],
        modes: str | list[str] | None,
        count: int,
    ) -> None:
        other = copy.deepcopy(osm_response["elements"][0])
        other["id"] = 43
        other["tags"]["route"] = "train"
        osm_response["elements"].append(other)
        queries = []

        def fetch(request: urllib.request.Request, **_kwargs: object) -> io.BytesIO:
            assert isinstance(request.data, bytes)
            queries.append(urllib.parse.parse_qs(request.data.decode())["data"][0])
            return io.BytesIO(json.dumps(osm_response).encode())

        monkeypatch.setattr("urllib.request.urlopen", fetch)
        services = pd.DataFrame([{"pattern_id": "osm:relation:43", "headway_secs": 600}])
        with c2g.load_osm_transport(
            services, relation_ids=[42, 43], modes=modes, service_date="20261103"
        ) as con:
            assert len(con.table("routes").df()) == count
            assert len(con.table("stops").df()) == 2
            report = c2g.get_gtfs_summary(con)
            assert len(report["services"]) == count
            assert "modes" in set(report["assumptions"]["field"])
        if modes == "bus":
            assert '["route"~"^(bus)$"]' in queries[0]
            other["tags"].pop("public_transport:version")
            with c2g.load_osm_transport(relation_ids=[42, 43], modes="bus") as con:
                assert len(con.table("routes").df()) == 1

    @pytest.mark.parametrize(
        ("mode", "tag", "route_type"),
        [
            ("tram", "light_rail", 0),
            ("light_rail", "tram", 0),
            ("subway", "subway", 1),
            ("train", "train", 2),
            ("trolleybus", "trolleybus", 11),
            ("ferry", "ferry", 4),
            ("monorail", "monorail", 12),
            ("funicular", "funicular", 7),
        ],
    )
    def test_supported_mode_groups(
        self,
        overpass_response: dict[str, Any],
        mode: str,
        tag: str,
        route_type: int,
    ) -> None:
        overpass_response["elements"][0]["tags"]["route"] = tag
        with c2g.load_osm_transport(relation_ids=[42], modes=mode) as con:
            assert con.table("routes").df()["route_type"].tolist() == [str(route_type)]
        with pytest.raises(ValueError, match="No matching OSM"):
            c2g.load_osm_transport(relation_ids=[42], modes="bus")

    @pytest.mark.parametrize(
        "problem", ["unversioned", "one_stop", "platform", "unnamed_stop", "unnamed_route"]
    )
    def test_invalid_relations_aggregate_skip_and_roundtrip(
        self,
        overpass_response: dict[str, Any],
        tmp_path: Path,
        problem: str,
    ) -> None:
        invalid = copy.deepcopy(overpass_response["elements"][0])
        invalid["id"] = 43
        if problem == "unversioned":
            invalid["tags"].pop("public_transport:version")
        elif problem == "one_stop":
            invalid["members"] = invalid["members"][:1]
        elif problem == "unnamed_route":
            invalid["tags"].pop("name")
            invalid["tags"].pop("ref")
        elif problem == "unnamed_stop":
            overpass_response["elements"].append({"type": "node", "id": 30, "lat": 51, "lon": 0.03})
            invalid["members"] = [
                {"type": "node", "ref": 30, "role": "stop"},
                {"type": "node", "ref": 2, "role": "stop"},
            ]
        else:
            invalid["members"][1]["ref"] = 9999
        second = copy.deepcopy(invalid)
        second["id"] = 44
        overpass_response["elements"].extend([invalid, second])
        with pytest.raises(
            ValueError, match=r"(?s)osm:relation:43.*osm:relation:44.*Usable relation_ids: \[42\]"
        ):
            c2g.load_osm_transport(relation_ids=[42, 43, 44], on_invalid="raise")
        with pytest.warns(c2g.GTFSGenerationWarning, match="excluded 2 OSM relations"):
            con = c2g.load_osm_transport(relation_ids=[42, 43, 44], on_invalid="skip")
        with con:
            report = c2g.get_gtfs_summary(con)
            rejected = report["assumptions"].query("field == 'relation' and kind == 'ignored'")
            assert set(rejected["record_id"]) == {"osm:relation:43", "osm:relation:44"}
            assert report["summary"].loc[0, "excluded_relation_count"] == 2
            assert rejected["reason"].str.contains("Excluded by").all()
            assert con.table("routes").df()["route_id"].tolist() == ["osm:relation:42"]
            path = c2g.save_gtfs(con, tmp_path / "selected.zip")
        with c2g.load_gtfs(path) as restored:
            restored_report = c2g.get_gtfs_summary(restored)
            assert restored_report.keys() == report.keys()
            for key, frame in report.items():
                assert_frame_equal(frame, restored_report[key])
        with pytest.raises(ValueError, match="no usable relations remain"):
            c2g.load_osm_transport(relation_ids=[43, 44], on_invalid="skip")
        if problem == "unnamed_stop":
            # Supplementing another field does not repair a required missing name.
            with pytest.raises(ValueError, match="stop_name"):
                c2g.load_osm_transport(
                    relation_ids=[42, 43, 44],
                    on_invalid="skip",
                    data={
                        "stops": pd.DataFrame(
                            [{"stop_id": "osm:node:30", "wheelchair_boarding": 0}]
                        )
                    },
                )

    def test_explicit_input_is_not_skipped_and_selection_precedes_validation(
        self, overpass_response: dict[str, Any]
    ) -> None:
        invalid = copy.deepcopy(overpass_response["elements"][0])
        invalid["id"] = 43
        invalid["tags"].pop("public_transport:version")
        overpass_response["elements"].append(invalid)
        with c2g.load_osm_transport(relation_ids=[42, 43], pattern_ids="osm:relation:42") as con:
            assert len(con.table("trips").df()) == 1
        with pytest.raises(ValueError, match="unknown IDs"):
            c2g.load_osm_transport(relation_ids=[42], pattern_ids="missing")
        with pytest.raises(ValueError, match="source conflict"):
            c2g.load_osm_transport(
                relation_ids=[42, 43],
                on_invalid="skip",
                data={"stops": pd.DataFrame([{"stop_id": "osm:node:1", "stop_name": "Different"}])},
            )
        repaired = pd.DataFrame(
            {
                "pattern_id": ["osm:relation:43"] * 2,
                "stop_id": ["osm:node:1", "osm:node:2"],
                "stop_sequence": [0, 1],
            }
        )
        with c2g.load_osm_transport(
            relation_ids=[42, 43], data={"stop_sequences": repaired}, overwrite_source=True
        ) as con:
            assert len(con.table("routes").df()) == 2

    @pytest.mark.parametrize(
        ("options", "expected"),
        [
            ({"modes": "bus"}, {"osm:relation:42"}),
            ({"pattern_ids": "osm:relation:42"}, {"osm:relation:42"}),
            ({"modes": ["bus", "train"]}, {"osm:relation:42", "custom"}),
            ({}, {"osm:relation:42", "custom"}),
            ({"pattern_ids": "custom"}, {"custom"}),
        ],
    )
    @pytest.mark.usefixtures("overpass_response")
    def test_supplemental_patterns_obey_selection(
        self,
        tmp_path: Path,
        options: dict[str, Any],
        expected: set[str],
    ) -> None:
        data = {
            "routes": pd.DataFrame(
                [{"route_id": "rail", "route_short_name": "Rail", "route_type": 2}]
            ),
            "patterns": pd.DataFrame(
                [{"pattern_id": "custom", "route_id": "rail", "direction_id": 0}]
            ),
            "stop_sequences": pd.DataFrame(
                {
                    "pattern_id": ["custom", "custom"],
                    "stop_sequence": [0, 1],
                    "stop_id": ["osm:node:1", "osm:node:2"],
                }
            ),
            "stops": gpd.GeoDataFrame(
                {
                    "stop_id": ["osm:node:1", "station"],
                    "stop_name": [None, "Station"],
                    "parent_station": ["station", None],
                    "location_type": [0, 1],
                },
                geometry=[None, Point(0, 51)],
                crs=4326,
            ),
        }
        before = {name: frame.copy(deep=True) for name, frame in data.items()}
        services = pd.DataFrame([{"pattern_id": "custom", "departure_time": "08:00:00"}])
        with c2g.load_osm_transport(
            services,
            relation_ids=[42],
            data=data,
            service_date="20261103",
            **options,
        ) as con:
            report = c2g.get_gtfs_summary(con)
            assert set(report["services"]["pattern_id"]) == expected
            assert set(con.table("stops").df()["stop_id"]) == {
                "osm:node:1",
                "osm:node:2",
                "station",
            }
            if "custom" not in expected:
                assert con.table("routes").df()["route_type"].tolist() == ["3"]
                assert (
                    report["assumptions"]
                    .query("field == 'pattern_ids'")["reason"]
                    .str.contains("custom")
                    .any()
                )
            output = c2g.save_gtfs(con, tmp_path / "selection.zip")
        with c2g.load_gtfs(output) as restored:
            restored_report = c2g.get_gtfs_summary(restored)
            assert restored_report.keys() == report.keys()
            for key, frame in report.items():
                assert_frame_equal(frame, restored_report[key])
        for name, frame in data.items():
            assert_frame_equal(frame, before[name])
        if "custom" not in expected:
            transfers = pd.DataFrame(
                [
                    {
                        "from_stop_id": "osm:node:1",
                        "to_stop_id": "osm:node:2",
                        "from_route_id": "rail",
                        "transfer_type": 0,
                    }
                ]
            )
            with pytest.raises(ValueError, match=r"transfers.from_route_id.*rail"):
                c2g.load_osm_transport(
                    relation_ids=[42],
                    data=data,
                    optional_tables={"transfers": transfers},
                    **options,
                )

    @pytest.mark.parametrize("problem", ["name", "sequence", "mode"])
    @pytest.mark.usefixtures("overpass_response")
    def test_invalid_custom_input_and_empty_effective_mode_selection(
        self,
        problem: str,
    ) -> None:
        if problem == "mode":
            with pytest.raises(ValueError, match="No matching OSM patterns"):
                c2g.load_osm_transport(
                    relation_ids=[42],
                    modes="bus",
                    overwrite_source=True,
                    data={
                        "routes": pd.DataFrame([{"route_id": "osm:relation:42", "route_type": 2}])
                    },
                )
            return
        data = {
            "routes": pd.DataFrame(
                [
                    {
                        "route_id": "custom",
                        "route_type": 3,
                        "route_short_name": None if problem == "name" else "R",
                    }
                ]
            ),
            "patterns": pd.DataFrame(
                [{"pattern_id": "custom", "route_id": "custom", "direction_id": 0}]
            ),
        }
        with pytest.raises(ValueError, match=r"(?s)custom.*supply"):
            c2g.load_osm_transport(relation_ids=[42], on_invalid="skip", data=data)

    def test_mode_excluded_ids_are_known_without_member_downloads(
        self,
        monkeypatch: pytest.MonkeyPatch,
        osm_response: dict[str, Any],
    ) -> None:
        # Overpass first returns candidate IDs/tags, then bodies and members only
        # for the mode selection. The bus therefore appears twice in the JSON.
        tags_only = [
            {"type": "relation", "id": 42, "tags": osm_response["elements"][0]["tags"]},
            {"type": "relation", "id": 43, "tags": {"type": "route", "route": "train"}},
        ]
        osm_response["elements"] = tags_only + osm_response["elements"]
        queries = []

        def fetch(request: urllib.request.Request, **_kwargs: object) -> io.BytesIO:
            assert isinstance(request.data, bytes)
            queries.append(urllib.parse.parse_qs(request.data.decode())["data"][0])
            return io.BytesIO(json.dumps(osm_response).encode())

        monkeypatch.setattr("urllib.request.urlopen", fetch)
        services = pd.DataFrame([{"pattern_id": "osm:relation:43", "headway_secs": 600}])
        with c2g.load_osm_transport(
            services,
            relation_ids=[42, 43],
            modes="bus",
            pattern_ids=["osm:relation:42", "osm:relation:43"],
            service_date="20261103",
        ) as con:
            assert con.table("routes").df()["route_id"].tolist() == ["osm:relation:42"]
            report = c2g.get_gtfs_summary(con)
            assert report["services"]["headway_secs"].tolist() == [900]
            assert (
                report["assumptions"]
                .query("field == 'excluded_pattern_ids'")["value"]
                .str.contains("43")
                .any()
            )
        assert ".candidates out tags;" in queries[0]
        assert 'rel.candidates["route"~"^(bus)$"]->.selected;' in queries[0]
        assert ".selected >>;" in queries[0]
        with pytest.raises(ValueError, match=r"unknown IDs.*99"):
            c2g.load_osm_transport(
                relation_ids=[42, 99], modes="bus", pattern_ids="osm:relation:99"
            )
        with pytest.raises(ValueError, match=r"unknown IDs.*99"):
            c2g.load_osm_transport(
                pd.DataFrame([{"pattern_id": "osm:relation:99"}]),
                relation_ids=[42],
                modes="bus",
            )
        with pytest.raises(ValueError, match="No matching OSM patterns"):
            c2g.load_osm_transport(
                relation_ids=[42, 43], modes="bus", pattern_ids="osm:relation:43"
            )

    def test_supplement_repairs_missing_route_name(
        self,
        overpass_response: dict[str, Any],
    ) -> None:
        overpass_response["elements"][0]["tags"].pop("name")
        overpass_response["elements"][0]["tags"].pop("ref")
        with c2g.load_osm_transport(
            relation_ids=[42],
            data={
                "routes": pd.DataFrame(
                    [
                        {"route_id": "osm:relation:42", "route_short_name": "R"},
                    ]
                )
            },
        ) as con:
            assert con.table("routes").df()["route_short_name"].tolist() == ["R"]

    @pytest.mark.parametrize(
        ("table", "row", "field"),
        [
            ("patterns", {"pattern_id": "osm:relation:43", "direction_id": 7}, "direction_id"),
            ("routes", {"route_id": "osm:relation:43", "route_type": -1}, "route_type"),
            ("routes", {"route_id": "osm:relation:43", "route_type": float("inf")}, "route_type"),
            ("routes", {"route_id": "osm:relation:43", "route_type": float("-inf")}, "route_type"),
            ("stops", {"stop_id": "bad", "stop_lat": 100}, "stop_lat"),
            ("stops", {"stop_id": "bad", "geometry": LineString([(0, 0), (1, 1)])}, "geometry"),
            (
                "stop_sequences",
                {"pattern_id": "osm:relation:43", "stop_sequence": 0, "pickup_type": 7},
                "pickup_type",
            ),
        ],
    )
    def test_invalid_supplement_cannot_be_skipped(
        self,
        monkeypatch: pytest.MonkeyPatch,
        table: str,
        row: dict[str, Any],
        field: str,
    ) -> None:
        def unexpected(*_args: object, **_kwargs: object) -> None:
            pytest.fail("Malformed explicit input must fail before any network access")

        monkeypatch.setattr("urllib.request.urlopen", unexpected)
        monkeypatch.setattr("city2graph.data.get_boundaries", unexpected)
        with pytest.raises(ValueError, match=rf"{table} row 0.*{field}"):
            c2g.load_osm_transport(
                place_name="Liverpool, UK",
                on_invalid="skip",
                data={table: pd.DataFrame([row])},
            )

    def test_group_headway_overrides_source_and_conditional_intervals(
        self, overpass_response: dict[str, Any]
    ) -> None:
        overpass_response["elements"][0]["tags"].update(
            {"interval": "10", "interval:conditional": "5 @ (Mo-Fr 07:00-09:00)"}
        )
        with c2g.load_osm_transport(
            relation_ids=[42], group_headway_secs=1200, service_date="20261103"
        ) as con:
            report = c2g.get_gtfs_summary(con)
            assert report["services"]["headway_secs"].tolist() == [1200]
            assert report["services"]["group_headway_secs"].tolist() == [1200]
            assert (
                report["assumptions"]
                .query("field == 'interval:conditional'")["kind"]
                .eq("ignored")
                .any()
            )
        with c2g.load_osm_transport(
            pd.DataFrame([{"headway_secs": 300}]),
            relation_ids=[42],
            group_headway_secs=1200,
            service_date="20261103",
        ) as con:
            assert c2g.get_gtfs_summary(con)["services"]["headway_secs"].tolist() == [300]

    @pytest.mark.parametrize(
        ("tags", "periods", "headway", "active", "ignored"),
        [
            ({}, [("06:00:00", "22:00:00")], 900, 1, False),
            ({"interval": "6"}, [("06:00:00", "22:00:00")], 360, 1, False),
            ({"interval": "00:10"}, [("06:00:00", "22:00:00")], 600, 1, False),
            ({"interval": "00:00:30"}, [("06:00:00", "22:00:00")], 30, 1, False),
            ({"opening_hours": "24/7"}, [("00:00:00", "24:00:00")], 900, 1, False),
            ({"opening_hours": "Mo-Fr 07:00-10:00"}, [("07:00:00", "10:00:00")], 900, 1, False),
            (
                {"opening_hours": "Mo,We,Fr 07:00-09:00,16:00-18:00; Tu,Th off"},
                [("07:00:00", "09:00:00"), ("16:00:00", "18:00:00")],
                900,
                1,
                False,
            ),
            ({"opening_hours": "Mo 22:00-02:00"}, [("22:00:00", "26:00:00")], 900, 1, False),
            ({"opening_hours": "Sa-Su 08:00-12:00"}, [("06:00:00", "22:00:00")], 900, 0, False),
            ({"opening_hours": "off"}, [("06:00:00", "22:00:00")], 900, 0, False),
            ({"opening_hours": "Mo,Mo 07:00-10:00"}, [("06:00:00", "22:00:00")], 900, 1, True),
            (
                {"opening_hours": "Mo-Fr 07:00-10:00; PH off"},
                [("06:00:00", "22:00:00")],
                900,
                1,
                True,
            ),
            ({"opening_hours": "Mo-Fr sunrise-sunset"}, [("06:00:00", "22:00:00")], 900, 1, True),
            (
                {"opening_hours": "Mo-Fr 07:00-10:00; Mo 08:00-09:00"},
                [("06:00:00", "22:00:00")],
                900,
                1,
                True,
            ),
            (
                {"opening_hours": "Mo 07:00-10:00,09:00-11:00"},
                [("06:00:00", "22:00:00")],
                900,
                1,
                True,
            ),
            ({"opening_hours": "Mo 25:00-26:00"}, [("06:00:00", "22:00:00")], 900, 1, True),
            ({"interval": "0"}, [("06:00:00", "22:00:00")], 900, 1, True),
            ({"interval": "00:90"}, [("06:00:00", "22:00:00")], 900, 1, True),
            ({"interval": "often"}, [("06:00:00", "22:00:00")], 900, 1, True),
            (
                {"interval": "10", "interval:conditional": "20 @ (PH 10:00-16:00)"},
                [("06:00:00", "22:00:00")],
                900,
                1,
                True,
            ),
            ({"duration": "00:30"}, [("06:00:00", "22:00:00")], 900, 1, False),
        ],
    )
    def test_default_operation_and_source_tags(
        self,
        overpass_response: dict[str, Any],
        tags: dict[str, str],
        periods: list[tuple[str, str]],
        headway: int,
        active: int,
        ignored: bool,
    ) -> None:
        overpass_response["elements"][0]["tags"].update(tags)
        with c2g.load_osm_transport(relation_ids=[42], service_date="20260105") as con:
            report = c2g.get_gtfs_summary(con)
            services = report["services"]
            assert list(zip(services["start_time"], services["end_time"], strict=True)) == periods
            assert set(services["headway_secs"]) == {headway}
            assert set(services["active_days"]) == {active}
            assert report["assumptions"]["kind"].eq("ignored").any() == ignored
            if not active:
                assert services["departures"].eq(0).all()
                assert report["edges"].empty
            if tags.get("interval") in {"6", "00:10", "00:00:30"}:
                assert set(report["assumptions"].query("field == 'headway_secs'")["kind"]) == {
                    "source"
                }

    @pytest.mark.parametrize(
        ("conditional", "periods", "headways", "departures"),
        [
            (
                "10 @ (Mo 07:00-09:00)",
                [("06:00:00", "07:00:00"), ("07:00:00", "09:00:00"), ("09:00:00", "22:00:00")],
                [900, 600, 900],
                68,
            ),
            (
                "10 @ (Su 23:00-08:00)",
                [("06:00:00", "08:00:00"), ("08:00:00", "22:00:00")],
                [600, 900],
                68,
            ),
            ("10 @ (Mo)", [("06:00:00", "22:00:00")], [600], 96),
            ("10 @ Mo", [("06:00:00", "22:00:00")], [600], 96),
            (
                "10 @ Mo 7:00-09:00",
                [("06:00:00", "07:00:00"), ("07:00:00", "09:00:00"), ("09:00:00", "22:00:00")],
                [900, 600, 900],
                68,
            ),
            (
                "10 @ Mo; 5 @ (Mo 07:00-09:00)",
                [("06:00:00", "07:00:00"), ("07:00:00", "09:00:00"), ("09:00:00", "22:00:00")],
                [600, 300, 600],
                108,
            ),
            (
                "10 @ (Su 23:00-32:00)",
                [("06:00:00", "08:00:00"), ("08:00:00", "22:00:00")],
                [600, 900],
                68,
            ),
            ("10 @ (Mo 07:00-09:00); 20 @ (Mo 08:00-10:00)", [("06:00:00", "22:00:00")], [900], 64),
            ("10 @ (Su 22:00-02:00); 20 @ (Mo 01:00-03:00)", [("06:00:00", "22:00:00")], [900], 64),
        ],
    )
    def test_conditional_intervals(
        self,
        overpass_response: dict[str, Any],
        conditional: str,
        periods: list[tuple[str, str]],
        headways: list[int],
        departures: int,
    ) -> None:
        overpass_response["elements"][0]["tags"].update(
            {"interval": "15", "interval:conditional": conditional}
        )
        with c2g.load_osm_transport(relation_ids=[42], service_date="20260105") as con:
            report = c2g.get_gtfs_summary(con)
            daily = report["services"]
            assert list(zip(daily["start_time"], daily["end_time"], strict=True)) == periods
            assert daily["headway_secs"].astype(int).tolist() == headways
            assert daily["departures"].sum() == departures
            if len(headways) == 1 and headways[0] == 900:
                assert (
                    report["assumptions"]
                    .query("kind == 'ignored'")["details"]
                    .str.contains("whole")
                    .all()
                )

    @pytest.mark.parametrize("explicit", [False, True])
    def test_duration_allocation_and_explicit_leg_priority(
        self,
        overpass_response: dict[str, Any],
        explicit: bool,
    ) -> None:
        overpass_response["elements"][0]["tags"]["duration"] = "00:12"
        overpass_response["elements"][0]["members"].insert(
            2, {"type": "node", "ref": 4, "role": "stop"}
        )
        overpass_response["elements"].append(
            {"type": "node", "id": 4, "lat": 51, "lon": 0.005, "tags": {"name": "Middle"}}
        )
        sequence = pd.DataFrame(
            {
                "pattern_id": ["osm:relation:42"] * 3,
                "stop_sequence": [0, 1, 2],
                "distance_m": [0, 1000, 2000],
                "dwell_time_secs": [0, 30, 0],
            }
        )
        if explicit:
            sequence["travel_time_secs"] = [0, 120, None]
        with c2g.load_osm_transport(data={"stop_sequences": sequence}, relation_ids=[42]) as con:
            rows = con.sql(
                "SELECT arrival_time, departure_time FROM stop_times ORDER BY stop_sequence LIMIT 3"
            ).fetchall()
            assert rows == [
                ("00:00:00", "00:00:00"),
                ("00:02:00", "00:02:30") if explicit else ("00:03:50", "00:04:20"),
                ("00:12:00", "00:12:00"),
            ]
            assumptions = c2g.get_gtfs_summary(con)["assumptions"]
            timed = assumptions.query("table_name == 'assumptions' and field == 'travel_time_secs'")
            assert "estimated" in set(timed["kind"])
            assert set(timed.query("kind == 'user'")["value"].astype(float)) <= {120}
        with c2g.load_osm_transport(relation_ids=[42], speed_kph=36) as con:
            ignored = c2g.get_gtfs_summary(con)["assumptions"].query("kind == 'ignored'")
            assert ignored["details"].str.contains("speed", case=False).any()

    def test_route_master_inheritance_and_shared_budget(
        self,
        overpass_response: dict[str, Any],
    ) -> None:
        second = copy.deepcopy(overpass_response["elements"][0])
        second["id"] = 43
        overpass_response["elements"].extend(
            [
                second,
                {
                    "type": "relation",
                    "id": 100,
                    "tags": {
                        "type": "route_master",
                        "route_master": "bus",
                        "operator": "Source operator",
                        "operator:website": "https://example.org",
                        "timezone": "Europe/London",
                        "opening_hours": "Mo 07:00-09:00",
                        "duration": "00:20",
                    },
                    "members": [{"type": "relation", "ref": 42}, {"type": "relation", "ref": 43}],
                },
            ]
        )
        with c2g.load_osm_transport(relation_ids=[42, 43]) as con:
            report = c2g.get_gtfs_summary(con)
            assert report["services"]["departures"].sum() == 8
            assert report["edges"]["frequency"].eq(8).all()
            assert set(report["services"]["headway_secs"]) == {1800}
            assert con.sql("SELECT agency_name FROM agency").fetchall() == [("Source operator",)]
            assert (
                report["assumptions"]
                .query("kind == 'ignored'")["details"]
                .str.contains("Parent duration")
                .any()
            )
            assert (
                report["assumptions"]
                .query("field == 'service_group_direction'")["kind"]
                .eq("estimated")
                .all()
            )
        overpass_response["elements"][-1]["tags"]["interval"] = "10"
        with c2g.load_osm_transport(relation_ids=[42, 43]) as con:
            assert c2g.get_gtfs_summary(con)["services"]["departures"].sum() == 24
        overpass_response["elements"][0]["tags"]["interval"] = "20"
        with c2g.load_osm_transport(relation_ids=[42, 43], service_date="20260105") as con:
            assert sorted(c2g.get_gtfs_summary(con)["services"]["departures"]) == [6, 12]

    @pytest.mark.parametrize(
        ("tags", "reason"),
        [
            ({"timezone": "Invalid/Zone", "website": "invalid"}, "Invalid source"),
            (
                {
                    "timezone": "UTC",
                    "operator": "Source operator",
                    "website": "https://example.org",
                },
                None,
            ),
        ],
    )
    def test_source_agency_metadata(
        self,
        overpass_response: dict[str, Any],
        agency: pd.DataFrame,
        tags: dict[str, str],
        reason: str | None,
    ) -> None:
        overpass_response["elements"][0]["tags"].update(tags)
        with c2g.load_osm_transport(relation_ids=[42]) as con:
            assumptions = c2g.get_gtfs_summary(con)["assumptions"]
            if reason:
                assert assumptions.query("kind == 'ignored'")["details"].str.contains(reason).all()
                assert con.sql("SELECT agency_timezone FROM agency").fetchall() == [
                    ("Europe/London",)
                ]
            else:
                assert con.sql("SELECT agency_name, agency_timezone FROM agency").fetchall() == [
                    ("Source operator", "UTC")
                ]
                assert set(
                    assumptions.query("table_name == 'assumptions' and field == 'agency_timezone'")[
                        "kind"
                    ]
                ) == {"source"}
        with c2g.load_osm_transport(relation_ids=[42], agency=agency) as con:
            assert con.sql("SELECT agency_name FROM agency").fetchall() == [
                (agency.iloc[0].agency_name,)
            ]

    def test_ambiguous_parent_and_invalid_duration(
        self,
        overpass_response: dict[str, Any],
    ) -> None:
        overpass_response["elements"][0]["tags"]["duration"] = "12"
        overpass_response["elements"].extend(
            [
                {
                    "type": "relation",
                    "id": identifier,
                    "tags": {"type": "route_master", "route_master": "bus", "interval": "1"},
                    "members": [{"type": "relation", "ref": 42}],
                }
                for identifier in (100, 101)
            ]
        )
        with c2g.load_osm_transport(relation_ids=[42]) as con:
            report = c2g.get_gtfs_summary(con)
            ignored = report["assumptions"].query("kind == 'ignored'")
            assert set(ignored["field"]) == {"route_master", "duration"}
            assert "12" in set(ignored["value"])
            assert report["services"]["headway_secs"].eq(900).all()

    def test_source_week_and_report_restore(
        self,
        overpass_response: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        overpass_response["elements"][0]["tags"].update(
            opening_hours="Mo 22:00-02:00; Tu-Su off", interval="10"
        )
        today = date.today()  # noqa: DTZ011 - generation uses the local calendar date
        with c2g.load_osm_transport(relation_ids=[42]) as con:
            report = c2g.get_gtfs_summary(con)
            days = pd.date_range(today, periods=7)
            assert report["services"]["date"].tolist() == days.strftime("%Y%m%d").tolist()
            assert report["services"]["departures"].tolist() == [
                24 if day.weekday() == 0 else 0 for day in days
            ]
            assert report["services"]["weekday"].iloc[0] == today.strftime("%A")
            path = c2g.save_gtfs(con, tmp_path / "source.zip")
            with c2g.load_gtfs(path) as restored:
                restored_report = c2g.get_gtfs_summary(restored)
                assert restored_report.keys() == report.keys()
                for key, frame in report.items():
                    assert_frame_equal(frame, restored_report[key])

    def test_source_precedence_and_opt_out(self, overpass_response: dict[str, Any]) -> None:
        overpass_response["elements"][0]["tags"].update(
            interval="10", opening_hours="Mo-Fr 07:00-10:00"
        )
        with c2g.load_osm_transport(
            relation_ids=[42], start_time="08:00:00", headway_secs=300, service_date="20260105"
        ) as con:
            services = c2g.get_gtfs_summary(con)["services"]
            assert services[["start_time", "end_time", "headway_secs"]].to_numpy().tolist() == [
                ["08:00:00", "10:00:00", 300]
            ]
        services = pd.DataFrame([{"start_time": "09:00:00", "headway_secs": 1200}])
        with c2g.load_osm_transport(
            services, relation_ids=[42], headway_secs=300, service_date="20260105"
        ) as con:
            assert c2g.get_gtfs_summary(con)["services"]["departures"].tolist() == [3]
        with c2g.load_osm_transport(
            relation_ids=[42], use_source_tags=False, service_date="20260105"
        ) as con:
            report = c2g.get_gtfs_summary(con)
            assert report["services"]["departures"].tolist() == [64]
            assert (
                report["assumptions"]
                .query("kind == 'ignored'")["details"]
                .str.contains("disabled")
                .all()
            )

    def test_explicit_calendar_and_source_weekdays(
        self,
        overpass_response: dict[str, Any],
        service_dates: pd.DataFrame,
    ) -> None:
        overpass_response["elements"][0]["tags"].update(
            interval="10", opening_hours="Sa-Su 08:00-10:00"
        )
        with c2g.load_osm_transport(relation_ids=[42], service_date="20000108") as con:
            assert c2g.get_gtfs_summary(con)["services"]["departures"].tolist() == [12]
        with c2g.load_osm_transport(relation_ids=[42], calendar_dates=service_dates) as con:
            report = c2g.get_gtfs_summary(con)
            assert report["services"]["departures"].tolist() == [0]
            assert report["edges"].empty

    def test_invalid_overrides_do_not_download(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fetch(*_args: object, **_kwargs: object) -> None:
            pytest.fail("Invalid explicit overrides should be rejected before downloading")

        monkeypatch.setattr("urllib.request.urlopen", fetch)
        with pytest.raises(ValueError, match="Invalid transport data") as error:
            c2g.load_osm_transport(
                relation_ids=[42], speed_kph=-1, headway_secs=0, start_time="bad"
            )
        assert all(
            field in str(error.value) for field in ("speed_kph", "headway_secs", "start_time")
        )

    def test_invalid_metadata_does_not_download(
        self, monkeypatch: pytest.MonkeyPatch, agency: pd.DataFrame, service_dates: pd.DataFrame
    ) -> None:
        def fetch(*_args: object, **_kwargs: object) -> None:
            pytest.fail("Invalid explicit metadata should be rejected before downloading")

        monkeypatch.setattr("urllib.request.urlopen", fetch)
        agency.loc[0, "agency_timezone"] = "Invalid/Zone"
        service_dates.loc[0, "date"] = "20260230"
        with pytest.raises(ValueError, match="invalid timezone") as error:
            c2g.load_osm_transport(relation_ids=[42], agency=agency, calendar_dates=service_dates)
        assert "invalid date" in str(error.value)

    def test_source_hours_overridden_and_conflicting_calendar(
        self, overpass_response: dict[str, Any]
    ) -> None:
        overpass_response["elements"][0]["tags"]["opening_hours"] = "off"
        second = copy.deepcopy(overpass_response["elements"][0])
        second["id"] = 43
        second["tags"]["opening_hours"] = "Mo 07:00-08:00"
        overpass_response["elements"].append(second)
        shared = pd.DataFrame([{"service_id": "shared"}])
        with c2g.load_osm_transport(shared, relation_ids=[42, 43]) as con:
            daily = c2g.get_gtfs_summary(con)["services"]
            assert daily["departures"].sum() == 4
            assert daily.loc[daily["pattern_id"].eq("osm:relation:42"), "departures"].eq(0).all()
            assert set(shared["service_id"]) == {"shared"}
        with c2g.load_osm_transport(
            relation_ids=[42], start_time="08:00:00", end_time="09:00:00"
        ) as con:
            report = c2g.get_gtfs_summary(con)
            assert report["services"]["departures"].tolist() == [4] * 7
            assert (
                report["assumptions"]
                .query("kind == 'ignored'")["details"]
                .str.contains("Explicit service windows")
                .any()
            )

    @pytest.mark.usefixtures("overpass_response")
    def test_place_selection_and_uniform_services(
        self,
        monkeypatch: pytest.MonkeyPatch,
        osm_services: pd.DataFrame,
        agency: pd.DataFrame,
        service_dates: pd.DataFrame,
    ) -> None:
        boundary = gpd.GeoDataFrame(geometry=[Point(0, 51).buffer(0.001)], crs=4326)
        monkeypatch.setattr("city2graph.data.get_boundaries", lambda _name: boundary)
        with c2g.load_osm_transport(
            osm_services.drop(columns="pattern_id"),
            agency=agency,
            place_name="Test place",
            route_ref="R",
            calendar_dates=service_dates,
            speed_kph=36,
            dwell_time_secs=0,
        ) as con:
            # The second stop lies outside the boundary but belongs to the selected route.
            assert con.sql("SELECT count(*) FROM stops").fetchall() == [(2,)]

    def test_loader_matches_direct_generator(
        self,
        monkeypatch: pytest.MonkeyPatch,
        osm_response: dict[str, Any],
        osm_services: pd.DataFrame,
        agency: pd.DataFrame,
        service_dates: pd.DataFrame,
    ) -> None:
        requests = []

        def fetch(request: object, **_kwargs: object) -> io.BytesIO:
            requests.append(request)
            return io.BytesIO(json.dumps(osm_response).encode())

        monkeypatch.setattr("urllib.request.urlopen", fetch)
        with c2g.load_osm_transport(
            osm_services,
            agency=agency,
            relation_ids=[42],
            calendar_dates=service_dates,
            speed_kph=36,
            dwell_time_secs=0,
        ) as con:
            data: c2g.TransportData = {
                "routes": pd.DataFrame(
                    [
                        {
                            "route_id": "osm:relation:42",
                            "route_short_name": "R",
                            "route_long_name": "Test route",
                            "route_type": 3,
                        }
                    ]
                ),
                "patterns": gpd.GeoDataFrame(
                    [
                        {
                            "pattern_id": "osm:relation:42",
                            "route_id": "osm:relation:42",
                            "direction_id": 0,
                        }
                    ],
                    geometry=[LineString([(0, 51), (0.01, 51)])],
                    crs=4326,
                ),
                "stops": gpd.GeoDataFrame(
                    {"stop_id": ["osm:node:1", "osm:node:2"], "stop_name": ["A", "B"]},
                    geometry=[Point(0, 51), Point(0.01, 51)],
                    crs=4326,
                ),
                "stop_sequences": pd.DataFrame(
                    {
                        "pattern_id": ["osm:relation:42"] * 2,
                        "stop_id": ["osm:node:1", "osm:node:2"],
                        "stop_sequence": [0, 1],
                        "pickup_type": [0, 1],
                        "drop_off_type": [0, 0],
                    }
                ),
            }
            with c2g.create_gtfs(
                data,
                osm_services,
                agency=agency,
                calendar_dates=service_dates,
                speed_kph=36,
                dwell_time_secs=0,
            ) as direct:
                for table in ("routes", "stop_times", "trips", "shapes"):
                    assert_frame_equal(con.table(table).df(), direct.table(table).df())
                assert (
                    con.sql(
                        "SELECT count(*) FROM _city2graph_metadata WHERE kind='source'"
                    ).fetchall()[0][0]
                    > 0
                )
                assert len(requests) == 2

    @pytest.mark.parametrize(
        "problem",
        ["empty", "remark", "network", "ambiguous", "platform", "missing_point", "unversioned"],
    )
    @pytest.mark.parametrize("on_invalid", ["raise", "skip"])
    def test_diagnostics(
        self,
        monkeypatch: pytest.MonkeyPatch,
        osm_response: dict[str, Any],
        osm_services: pd.DataFrame,
        agency: pd.DataFrame,
        service_dates: pd.DataFrame,
        problem: str,
        on_invalid: Literal["raise", "skip"],
    ) -> None:
        if problem == "unversioned":
            osm_response["elements"][0]["tags"].pop("public_transport:version")
        elif problem == "empty":
            osm_response["elements"] = []
        elif problem == "remark":
            osm_response["remark"] = "query timeout"
        elif problem == "ambiguous":
            other = copy.deepcopy(osm_response["elements"][0])
            other["id"] = 43
            other["tags"]["ref"] = "Another route"
            osm_response["elements"].append(other)
        elif problem == "platform":
            osm_response["elements"][2]["tags"].pop("name")
            osm_response["elements"][2]["lat"] = 51.01
        elif problem == "missing_point":
            del osm_response["elements"][1]["lat"]

        def fetch(*_args: object, **_kwargs: object) -> io.BytesIO:
            if problem == "network":
                msg = "connection failed"
                raise urllib.error.URLError(msg)
            return io.BytesIO(json.dumps(osm_response).encode())

        monkeypatch.setattr("urllib.request.urlopen", fetch)
        error = RuntimeError if problem in {"network", "remark"} else ValueError
        with pytest.raises(error):
            c2g.load_osm_transport(
                osm_services,
                agency=agency,
                area=[-0.01, 50.9, 0.02, 51.1],
                route_name="Test route",
                on_invalid=on_invalid,
                calendar_dates=service_dates,
                speed_kph=36,
                dwell_time_secs=0,
            )

    @pytest.mark.usefixtures("overpass_response")
    def test_explicit_override(
        self,
        osm_services: pd.DataFrame,
        agency: pd.DataFrame,
        service_dates: pd.DataFrame,
    ) -> None:
        supplement = {"stops": pd.DataFrame([{"stop_id": "osm:node:1", "stop_name": "Custom A"}])}
        with pytest.raises(ValueError, match="source conflict"):
            c2g.load_osm_transport(
                osm_services,
                agency=agency,
                relation_ids=[42],
                data=supplement,
                calendar_dates=service_dates,
                speed_kph=36,
                dwell_time_secs=0,
            )
        with c2g.load_osm_transport(
            osm_services,
            agency=agency,
            relation_ids=[42],
            data=supplement,
            overwrite_source=True,
            calendar_dates=service_dates,
            speed_kph=36,
            dwell_time_secs=0,
        ) as con:
            assert con.sql("SELECT stop_name FROM stops WHERE stop_id='osm:node:1'").fetchall() == [
                ("Custom A",)
            ]
            assert con.sql(
                "SELECT count(*) FROM _city2graph_metadata WHERE kind='override'"
            ).fetchall() == [(1,)]
