"""Public API tests for synthetic GTFS generation and saving."""

# ruff: noqa: D101, D102

import json
import warnings
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import cast
from zipfile import ZipFile

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from shapely.geometry import Point

import city2graph as c2g
from city2graph import TransportData

if TYPE_CHECKING:
    from collections.abc import Mapping

    import geopandas as gpd


pytestmark = pytest.mark.filterwarnings("ignore::city2graph.GTFSGenerationWarning")


class TestCreateGtfs:
    def test_reuse_loaded_stop_table(self, transport_input: TransportData, tmp_path: Path) -> None:
        with c2g.create_gtfs(transport_input, service_date="20261103") as original:
            expected = original.table("stop_times").df()
            archive = c2g.save_gtfs(original, tmp_path / "source.zip")
        with c2g.load_gtfs(archive) as loaded:
            stops = loaded.table("stops").df()
        assert isinstance(stops.iloc[0]["geometry"], (bytes, bytearray, memoryview))
        before = stops.copy(deep=True)
        transport_input["stops"] = stops
        with c2g.create_gtfs(transport_input, service_date="20261103") as regenerated:
            assert_frame_equal(expected, regenerated.table("stop_times").df())
            report = c2g.gtfs_generation_report(regenerated)
            archive = c2g.save_gtfs(regenerated, tmp_path / "regenerated.zip")
        with c2g.load_gtfs(archive) as restored:
            restored_report = c2g.gtfs_generation_report(restored)
            assert restored_report.keys() == report.keys()
            for name, frame in report.items():
                assert_frame_equal(frame, restored_report[name])
        assert_frame_equal(stops, before)

    @pytest.mark.parametrize("dtype", ["float64", "Float64", "Int64", "string"])
    def test_numeric_direction_codes(
        self,
        transport_input: TransportData,
        dtype: Literal["float64", "Float64", "Int64", "string"],
    ) -> None:
        transport_input["patterns"]["direction_id"] = pd.Series([0.0, 1.0]).astype(dtype)
        before = transport_input["patterns"].copy(deep=True)
        with c2g.create_gtfs(transport_input, service_date="20261103") as con:
            assert con.table("trips").df()["direction_id"].tolist() == ["0", "1"]
            report = c2g.gtfs_generation_report(con)["services"]
            assert report["direction_id"].tolist() == [0, 1]
            assert report["service_group_id"].nunique() == 2
            assert report["departures"].sum() == 128
        assert_frame_equal(before, transport_input["patterns"])

    @pytest.mark.parametrize("value", [0.5, 2, float("inf"), float("-inf")])
    def test_invalid_direction_codes(self, transport_input: TransportData, value: float) -> None:
        transport_input["patterns"]["direction_id"] = value
        with pytest.raises(ValueError, match=r"patterns row 0.*local.*direction_id"):
            c2g.create_gtfs(transport_input)

    @pytest.mark.parametrize("value", [float("inf"), float("-inf"), "inf", "-Infinity", "1e999"])
    def test_nonfinite_route_type(self, transport_input: TransportData, value: float | str) -> None:
        transport_input["routes"]["route_type"] = value
        with pytest.raises(ValueError, match=r"routes row 0.*001.*route_type"):
            c2g.create_gtfs(transport_input)

    def test_generated_trip_ids_avoid_all_explicit_ids(
        self, transport_input: TransportData, tmp_path: Path
    ) -> None:
        services = pd.DataFrame(
            [
                {"pattern_id": "local", "trip_id": "service_2", "departure_time": "08:00:00"},
                {"pattern_id": "local", "trip_id": "service_2:1", "departure_time": "09:00:00"},
            ]
        )
        before = services.copy(deep=True)
        with c2g.create_gtfs(transport_input, services, service_date="20261103") as con:
            trips = con.table("trips").df()
            assert trips["trip_id"].is_unique
            assert set(services["trip_id"]) < set(trips["trip_id"])
            report = c2g.gtfs_generation_report(con)
            assert report["services"].groupby("pattern_id")["departures"].sum().to_dict() == {
                "local": 2,
                "express": 32,
            }
            archive = c2g.save_gtfs(con, tmp_path / "identifiers.zip")
        with c2g.create_gtfs(transport_input, services, service_date="20261103") as repeated:
            assert_frame_equal(trips, repeated.table("trips").df())
        with c2g.load_gtfs(archive) as restored:
            restored_report = c2g.gtfs_generation_report(restored)
            assert restored_report.keys() == report.keys()
            for name, frame in report.items():
                assert_frame_equal(frame, restored_report[name])
        assert_frame_equal(services, before)

    def test_partial_overrides_retain_patterns_and_selection(
        self, transport_input: TransportData
    ) -> None:
        services = pd.DataFrame(
            [{"pattern_id": "local", "departure_time": "08:05:00", "trip_headsign": "Central"}]
        )
        before = {
            name: frame.copy(deep=True)
            for name, frame in cast("Mapping[str, pd.DataFrame]", transport_input).items()
        }
        with c2g.create_gtfs(transport_input, services, service_date="20261103") as con:
            report = c2g.gtfs_generation_report(con)
            rows = report["services"].set_index("pattern_id")
            assert rows["departures"].to_dict() == {"local": 1, "express": 32}
            assert pd.isna(rows.loc["local", "headway_secs"])
            assert rows.loc["express", "headway_secs"] == 1800
            assert con.sql(
                "SELECT trip_headsign FROM trips WHERE trip_headsign IS NOT NULL"
            ).fetchall() == [("Central",)]
        with c2g.create_gtfs(
            transport_input, services, pattern_ids="express", service_date="20261103"
        ) as con:
            rows = c2g.gtfs_generation_report(con)["services"]
            assert rows["pattern_id"].tolist() == ["express"]
            assert rows["departures"].tolist() == [64]
            assert rows.columns.tolist() == report["services"].columns.tolist()
            assert rows.dtypes.equals(report["services"].dtypes)
        for name, frame in before.items():
            assert_frame_equal(frame, cast("Mapping[str, pd.DataFrame]", transport_input)[name])

    @pytest.mark.parametrize(
        ("options", "count", "headway"),
        [
            pytest.param({}, 64, 1800, id="default-group"),
            pytest.param({"group_headway_secs": 900}, 64, 1800, id="explicit-group"),
            pytest.param({"headway_secs": 900}, 128, 900, id="per-pattern"),
        ],
    )
    def test_group_and_pattern_headways(
        self, transport_input: TransportData, options: dict[str, Any], count: int, headway: int
    ) -> None:
        with c2g.create_gtfs(transport_input, service_date="20261103", **options) as con:
            rows = c2g.gtfs_generation_report(con)["services"]
            assert rows["departures"].sum() == count
            assert rows["headway_secs"].eq(headway).all()
            assert str(rows["headway_secs"].dtype) == "Int64"
            assert (3600 / rows["headway_secs"]).sum() == count / 16
            if "headway_secs" not in options:
                assert rows["group_headway_secs"].eq(900).all()
                assert rows["group_pattern_count"].eq(2).all()

    def test_service_headway_overrides_group_default(self, transport_input: TransportData) -> None:
        """A per-service override leaves the other pattern's shared allocation intact."""
        services = pd.DataFrame([{"pattern_id": "local", "headway_secs": 600}])
        with c2g.create_gtfs(
            transport_input, services, group_headway_secs=900, service_date="20261103"
        ) as con:
            rows = c2g.gtfs_generation_report(con)["services"].set_index("pattern_id")
            assert rows["headway_secs"].to_dict() == {"local": 600, "express": 1800}

    @pytest.mark.parametrize(
        "options",
        [
            {"pattern_ids": []},
            {"pattern_ids": ""},
            {"pattern_ids": [1]},
            {"pattern_ids": "missing"},
            {"group_headway_secs": 0},
            {"group_headway_secs": 1.5},
            {"group_headway_secs": 900, "headway_secs": 900},
        ],
    )
    def test_invalid_selection_and_group_parameters(
        self, transport_input: TransportData, options: dict[str, object]
    ) -> None:
        with pytest.raises(ValueError, match=r"modes|pattern_ids|on_invalid|headway|services"):
            c2g.create_gtfs(transport_input, **options)  # type: ignore[arg-type]

    def test_unknown_service_column_and_sequence_context(
        self, transport_input: TransportData
    ) -> None:
        with pytest.raises(ValueError, match=r"headway_sec.*did you mean 'headway_secs'"):
            c2g.create_gtfs(transport_input, pd.DataFrame([{"headway_sec": 600}]))
        with pytest.raises(ValueError, match=r"services.pattern_id: unknown IDs"):
            c2g.create_gtfs(transport_input, pd.DataFrame([{"pattern_id": "missing"}]))
        transport_input["stop_sequences"]["stop_sequence"] = transport_input["stop_sequences"][
            "stop_sequence"
        ].astype(object)
        transport_input["stop_sequences"].loc[1, "stop_sequence"] = "second"
        with pytest.raises(
            ValueError, match=r"stop_sequences row 1.*pattern_id='local'.*stop_sequence"
        ):
            c2g.create_gtfs(transport_input)

    def test_selected_stops_retain_parent_and_optional_references(
        self, transport_input: TransportData
    ) -> None:
        stops = transport_input["stops"]
        parent = stops.iloc[[0]].copy().assign(stop_id="station", location_type=1)
        stops = stops.assign(parent_station="station")
        transport_input["stops"] = pd.concat([stops, parent], ignore_index=True)
        with c2g.create_gtfs(transport_input, pattern_ids="express") as con:
            assert "station" in set(con.table("stops").df()["stop_id"])
        transport_input["routes"] = pd.concat(
            [transport_input["routes"], transport_input["routes"].assign(route_id="other")],
            ignore_index=True,
        )
        transport_input["patterns"].loc[1, "route_id"] = "other"
        transfers = pd.DataFrame(
            [
                {
                    "from_stop_id": stops.iloc[0]["stop_id"],
                    "to_stop_id": stops.iloc[2]["stop_id"],
                    "from_route_id": "other",
                    "transfer_type": 0,
                }
            ]
        )
        with pytest.raises(ValueError, match=r"transfers.from_route_id: unknown routes IDs.*other"):
            c2g.create_gtfs(
                transport_input, pattern_ids="local", optional_tables={"transfers": transfers}
            )

    @pytest.mark.parametrize(
        ("mode", "speed", "headway"),
        [
            (0, 25, 1800),
            (1, 35, 1800),
            (2, 60, 1800),
            (3, 20, 1800),
            (4, 20, 1800),
            (5, 25, 1800),
            (6, 10, 1800),
            (7, 10, 1800),
            (11, 20, 1800),
            (12, 25, 1800),
            (999, 20, 1800),
        ],
    )
    def test_mode_defaults_and_report(
        self, transport_input: TransportData, mode: int, speed: int, headway: int
    ) -> None:
        transport_input["routes"]["route_type"] = mode
        transport_input["stop_sequences"] = transport_input["stop_sequences"].drop(
            columns="travel_time_secs"
        )
        before = {
            name: frame.copy(deep=True)
            for name, frame in cast("Mapping[str, pd.DataFrame]", transport_input).items()
        }
        with c2g.create_gtfs(transport_input) as con:
            report = c2g.gtfs_generation_report(con)
            assert set(report) == {"assumptions", "services", "edges"}
            services = report["services"]
            assert set(services["headway_secs"]) == {headway}
            assert set(services["start_time"]) == {"06:00:00"}
            assert set(services["end_time"]) == {"22:00:00"}
            assert set(services["active_days"]) == {1}
            assert set(services["departures"]) == {57600 / headway}
            assumptions = report["assumptions"]
            adopted_speed = assumptions[assumptions["field"].eq("speed_kph")]
            assert set(adopted_speed["value"].astype(float)) == {speed}
            assert set(adopted_speed["kind"]) == {"default"}
            assert set(assumptions[assumptions["field"].eq("trip_id")]["kind"]) == {"generated"}
            assert set(assumptions[assumptions["field"].eq("exact_times")]["kind"]) == {"default"}
            assert con.sql("SELECT DISTINCT date FROM calendar_dates ORDER BY date").fetchall() == [
                (f"202601{day:02d}",) for day in range(3, 10)
            ]
            assert len(services) == 14
            assert services.groupby("date")["departures"].sum().eq(64).all()
            assert con.sql("SELECT agency_timezone FROM agency").fetchall() == [("Europe/London",)]
            assert len(report["edges"]) == 3
            for name, frame in before.items():
                assert_frame_equal(frame, cast("Mapping[str, pd.DataFrame]", transport_input)[name])

    @pytest.mark.parametrize("patterns", [1, 2, 3])
    def test_default_rate_is_independent_of_pattern_count(
        self, transport_input: TransportData, patterns: int
    ) -> None:
        local = transport_input["stop_sequences"].query("pattern_id == 'local'")
        base = transport_input["patterns"].iloc[[0]]
        transport_input["patterns"] = pd.concat(
            [base.assign(pattern_id=f"p{i}") for i in range(patterns)], ignore_index=True
        )
        transport_input["stop_sequences"] = pd.concat(
            [local.assign(pattern_id=f"p{i}") for i in range(patterns)], ignore_index=True
        )
        with c2g.create_gtfs(transport_input) as con:
            report = c2g.gtfs_generation_report(con)
            assert report["services"].groupby("date")["departures"].sum().eq(64).all()
            assert report["edges"]["frequency"].to_numpy() == pytest.approx(448)
            assert set(report["services"]["headway_secs"]) == {900 * patterns}
        transport_input["patterns"]["service_group_id"] = transport_input["patterns"]["pattern_id"]
        with c2g.create_gtfs(transport_input, service_date="20260103") as con:
            assert c2g.gtfs_generation_report(con)["edges"]["frequency"].eq(64 * patterns).all()

    def test_summary_warning_and_explicit_case(
        self,
        transport_input: TransportData,
        departures: pd.DataFrame,
        agency: pd.DataFrame,
        service_dates: pd.DataFrame,
    ) -> None:
        with pytest.warns(c2g.GTFSGenerationWarning, match="assumptions") as caught:
            c2g.create_gtfs(transport_input).close()
        assert len(caught) == 1
        with warnings.catch_warnings(record=True) as explicit_warnings:
            warnings.simplefilter("always", c2g.GTFSGenerationWarning)
            c2g.create_gtfs(
                transport_input,
                departures,
                agency=agency,
                calendar_dates=service_dates,
                dwell_time_secs=0,
                distance_method="auto",
            ).close()
        assert not [
            w for w in explicit_warnings if issubclass(w.category, c2g.GTFSGenerationWarning)
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("error", c2g.GTFSGenerationWarning)
            with pytest.raises(c2g.GTFSGenerationWarning):
                c2g.create_gtfs(transport_input)

    @pytest.mark.parametrize("longitude", [-74.0, -30.0])
    def test_ambiguous_geography_uses_warned_utc(
        self,
        transport_input: TransportData,
        longitude: float,
    ) -> None:
        transport_input["stops"] = transport_input["stops"].copy()
        transport_input["stops"].loc[0, "geometry"] = Point(longitude, 40)
        with ExitStack() as connections:
            with pytest.warns(c2g.GTFSGenerationWarning):
                con = connections.enter_context(c2g.create_gtfs(transport_input))
            assert con.sql("SELECT agency_timezone FROM agency").fetchall() == [("UTC",)]
            assumptions = c2g.gtfs_generation_report(con)["assumptions"]
            zone = assumptions.query("field == 'agency_timezone'")
            assert set(zone["kind"]) == {"default"}
            assert zone["reason"].str.contains("timezone", case=False).all()

    def test_partial_services_and_global_overrides(self, transport_input: TransportData) -> None:
        transport_input["stop_sequences"] = transport_input["stop_sequences"].drop(
            columns="travel_time_secs"
        )
        transport_input["stop_sequences"]["distance_m"] = 1000
        transport_input["stop_sequences"].loc[1, "speed_kph"] = 72
        services = pd.DataFrame(
            [{"pattern_id": "local", "start_time": "08:30:00", "headway_secs": 300}]
        )
        with c2g.create_gtfs(
            transport_input,
            services,
            pattern_ids="local",
            start_time="08:00:00",
            end_time="09:00:00",
            headway_secs=600,
            speed_kph=36,
            dwell_time_secs=5,
            service_date="20261102",
        ) as con:
            report = c2g.gtfs_generation_report(con)
            assert report["services"]["departures"].tolist() == [6]
            assert report["services"]["start_time"].tolist() == ["08:30:00"]
            assert report["services"]["headway_secs"].tolist() == [300]
            assert set(
                report["assumptions"].query("field == 'speed_kph'")["value"].astype(float)
            ) == {36, 72}
            assert con.sql(
                "SELECT arrival_time, departure_time FROM stop_times ORDER BY stop_sequence"
            ).fetchall() == [
                ("00:00:00", "00:00:05"),
                ("00:00:55", "00:01:00"),
                ("00:02:40", "00:02:45"),
            ]
            assert con.sql("SELECT DISTINCT date FROM calendar_dates").fetchall() == [("20261102",)]

    def test_default_agency_preserves_route_agency_id(self, transport_input: TransportData) -> None:
        transport_input["routes"]["agency_id"] = "operator"
        with c2g.create_gtfs(transport_input) as con:
            assert con.sql("SELECT agency_id, agency_timezone FROM agency").fetchall() == [
                ("operator", "Europe/London")
            ]

    def test_fixed_departure_and_calendar_inference(
        self, transport_input: TransportData, service_dates: pd.DataFrame
    ) -> None:
        services = pd.DataFrame([{"pattern_id": "local", "departure_time": "08:00:00"}])
        with c2g.create_gtfs(
            transport_input,
            services,
            pattern_ids="local",
            calendar_dates=service_dates,
            headway_secs=600,
        ) as con:
            assert "frequencies" not in {row[0] for row in con.sql("SHOW TABLES").fetchall()}
            report = c2g.gtfs_generation_report(con)
            assert report["services"]["departures"].tolist() == [1]
            assert report["services"]["service_id"].tolist() == ["weekday"]
            assert report["services"]["headway_secs"].isna().all()
            assert "calendar_dates:0" in set(report["assumptions"]["record_id"])
        ambiguous = pd.concat([service_dates, service_dates.assign(service_id="other")])
        with pytest.raises(ValueError, match="explicit calendar is not unique"):
            c2g.create_gtfs(transport_input, calendar_dates=ambiguous)

    def test_shared_pattern_counts(self, transport_input: TransportData) -> None:
        local = transport_input["stop_sequences"].query("pattern_id == 'local'")
        transport_input["stop_sequences"] = pd.concat([local, local.assign(pattern_id="express")])
        with c2g.create_gtfs(
            transport_input,
            start_time="08:00:00",
            end_time="09:00:00",
            headway_secs=600,
            service_date="20260105",
        ) as con:
            report = c2g.gtfs_generation_report(con)
            assert report["services"]["departures"].tolist() == [6, 6]
            assert set(report["edges"]["frequency"]) == {12}

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
            report = c2g.gtfs_generation_report(con)
            assert report["services"]["departures"].tolist() == [expected]
            assert set(report["edges"]["frequency"]) == {expected}
            con.execute("DELETE FROM _city2graph_provenance WHERE table_name='assumptions'")
            with pytest.raises(ValueError, match="no effective assumptions"):
                c2g.gtfs_generation_report(con)

    def test_reversed_shape_and_zero_distance(self, transport_input: TransportData) -> None:
        transport_input["stop_sequences"] = transport_input["stop_sequences"].drop(
            columns="travel_time_secs"
        )
        patterns = cast("gpd.GeoDataFrame", transport_input["patterns"])
        patterns.geometry = patterns.geometry.reverse()
        with c2g.create_gtfs(transport_input) as con:
            assert (
                c2g.gtfs_generation_report(con)["assumptions"]
                .query("kind == 'estimated' and field == 'travel_time_secs'")["details"]
                .str.contains("geodesic stop-to-stop")
                .all()
            )
        stops = cast("gpd.GeoDataFrame", transport_input["stops"])
        stops.geometry = [stops.geometry.iloc[0]] * len(stops)
        with pytest.raises(ValueError, match="zero stop-to-stop distance"):
            c2g.create_gtfs(transport_input)

    @pytest.mark.parametrize("method", [None, "auto", "geodesic", "shape"])
    def test_missing_shape_distance_policy(
        self, transport_input: TransportData, method: str | None
    ) -> None:
        transport_input["patterns"] = transport_input["patterns"].drop(columns="geometry")
        transport_input["stop_sequences"] = transport_input["stop_sequences"].drop(
            columns="travel_time_secs"
        )
        if method == "shape":
            with pytest.raises(ValueError, match="missing/looping geometry"):
                c2g.create_gtfs(transport_input, distance_method=method)
            return
        with c2g.create_gtfs(transport_input, distance_method=method) as con:
            report = c2g.gtfs_generation_report(con)
            assert (
                report["assumptions"]
                .query("kind == 'estimated' and field == 'travel_time_secs'")["details"]
                .str.contains("geodesic stop-to-stop")
                .all()
            )

    @pytest.mark.parametrize(
        "options",
        [
            {"speed_kph": 0},
            {"dwell_time_secs": -1},
            {"headway_secs": 1.5},
            {"start_time": "bad"},
            {"end_time": "08:99:00"},
            {"start_time": "09:00:00", "end_time": "08:00:00"},
            {"service_date": "20260230"},
            {"service_date": "2000-01-03"},
            {"distance_method": "guess"},
        ],
    )
    def test_invalid_default_overrides(
        self, transport_input: TransportData, options: dict[str, object]
    ) -> None:
        with pytest.raises(ValueError, match="Invalid transport data"):
            c2g.create_gtfs(transport_input, **options)  # type: ignore[arg-type]

    def test_empty_services_and_conflicting_date(
        self, transport_input: TransportData, service_dates: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            c2g.create_gtfs(transport_input, pd.DataFrame())
        with pytest.raises(ValueError, match="cannot be combined"):
            c2g.create_gtfs(transport_input, service_date="20261102", calendar_dates=service_dates)
        with pytest.raises(ValueError, match="Specify pattern_id"):
            c2g.create_gtfs(transport_input, pd.DataFrame([{"trip_id": "explicit"}]))

    def test_local_express_roundtrip(
        self,
        transport_input: TransportData,
        agency: pd.DataFrame,
        service_dates: pd.DataFrame,
        departures: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        original = transport_input["stop_sequences"].copy(deep=True)
        with c2g.create_gtfs(
            transport_input,
            departures,
            agency=agency,
            calendar_dates=service_dates,
            dwell_time_secs=30,
        ) as con:
            rows = con.sql(
                "SELECT stop_id, arrival_time, departure_time FROM stop_times "
                "WHERE trip_id='service_0' ORDER BY CAST(stop_sequence AS INTEGER)"
            ).fetchall()
            assert rows == [
                ("001", "07:59:30", "08:00:00"),
                ("002", "08:05:00", "08:05:30"),
                ("003", "08:10:30", "08:11:00"),
            ]
            nodes, edges = c2g.travel_summary_graph(
                con, calendar_start="20261102", calendar_end="20261102", directed=True
            )
            assert len(nodes) == 3
            assert set(edges.index) == {("001", "002"), ("002", "003"), ("001", "003")}
            assert len(c2g.get_od_pairs(con, start_date="20261102", end_date="20261102")) == 3
            archive = c2g.save_gtfs(con, tmp_path / "synthetic.zip")
            with c2g.load_gtfs(
                archive, provenance_path=str(archive) + ".provenance.json"
            ) as restored:
                for table in (
                    "agency",
                    "routes",
                    "trips",
                    "stop_times",
                    "shapes",
                    "calendar_dates",
                    "_city2graph_provenance",
                ):
                    assert_frame_equal(con.table(table).df(), restored.table(table).df())
                _, after = c2g.travel_summary_graph(
                    restored, calendar_start="20261102", calendar_end="20261102", directed=True
                )
                assert_frame_equal(edges, after)
            assert_frame_equal(original, transport_input["stop_sequences"])

    @pytest.mark.parametrize("method", ["shape", "geodesic"])
    def test_estimated_times_and_provenance(
        self,
        transport_input: TransportData,
        agency: pd.DataFrame,
        service_dates: pd.DataFrame,
        departures: pd.DataFrame,
        method: str,
    ) -> None:
        transport_input["stop_sequences"] = transport_input["stop_sequences"].drop(
            columns="travel_time_secs"
        )
        with c2g.create_gtfs(
            transport_input,
            departures,
            agency=agency,
            calendar_dates=service_dates,
            speed_kph=36,
            dwell_time_secs=0,
            distance_method=method,
        ) as con:
            estimated = con.sql("SELECT * FROM _city2graph_provenance WHERE kind='estimated'").df()
            assert len(estimated) == 3
            assert estimated["details"].str.contains('"speed_kph": 36.0').all()
            assert con.sql(
                "SELECT arrival_time FROM stop_times WHERE trip_id='service_0' AND stop_id='002'"
            ).fetchall() == [("08:01:10",)]

    def test_loop_repeated_stop_offsets(
        self, transport_input: TransportData, agency: pd.DataFrame, service_dates: pd.DataFrame
    ) -> None:
        transport_input["patterns"] = pd.DataFrame(
            [{"pattern_id": "loop", "route_id": "001", "direction_id": 1}]
        )
        transport_input["stop_sequences"] = pd.DataFrame(
            {
                "pattern_id": ["loop"] * 3,
                "stop_id": ["001", "002", "001"],
                "stop_sequence": [0, 1, 2],
                "arrival_offset_secs": [0, 300, 630],
                "departure_offset_secs": [0, 330, 630],
            }
        )
        services = pd.DataFrame(
            [{"pattern_id": "loop", "service_id": "weekday", "departure_time": "24:00:00"}]
        )
        with c2g.create_gtfs(
            transport_input, services, agency=agency, calendar_dates=service_dates
        ) as con:
            assert con.sql(
                "SELECT arrival_time FROM stop_times ORDER BY stop_sequence"
            ).fetchall() == [("24:00:00",), ("24:05:00",), ("24:10:30",)]
            _, edges = c2g.travel_summary_graph(
                con, calendar_start="20261102", calendar_end="20261102", directed=True
            )
            assert set(edges.index) == {("001", "002"), ("002", "001")}

    @pytest.mark.parametrize(
        ("exact", "end", "lower", "upper", "expected"),
        [
            (0, "09:00:00", None, None, 6),
            (1, "09:00:00", None, None, 6),
            (0, "09:05:00", None, None, 6.5),
            (1, "09:05:00", None, None, 7),
            (0, "09:00:00", "08:15:00", "08:45:00", 2.5),
            (1, "09:00:00", "08:15:00", "08:45:00", 3),
            (0, "09:00:00", "09:00:00", "09:20:00", 0),
            (1, "09:00:00", "09:00:00", "09:20:00", 0),
            (1, "09:00:00", "08:50:00", "08:55:00", 1),
            (0, "09:00:00", "08:50:00", "08:55:00", 0),
        ],
    )
    def test_frequency_windows(
        self,
        transport_input: TransportData,
        agency: pd.DataFrame,
        service_dates: pd.DataFrame,
        exact: int,
        end: str,
        lower: str | None,
        upper: str | None,
        expected: float,
    ) -> None:
        services = pd.DataFrame(
            [
                {
                    "pattern_id": "local",
                    "service_id": "weekday",
                    "start_time": "08:00:00",
                    "end_time": end,
                    "headway_secs": 600,
                    "exact_times": exact,
                }
            ]
        )
        with c2g.create_gtfs(
            transport_input,
            services,
            agency=agency,
            calendar_dates=service_dates,
            dwell_time_secs=0,
        ) as con:
            _, edges = c2g.travel_summary_graph(
                con,
                calendar_start="20261102",
                calendar_end="20261102",
                start_time=lower,
                end_time=upper,
                directed=True,
            )
            actual = edges.loc[("001", "002"), "frequency"] if ("001", "002") in edges.index else 0
            assert actual == pytest.approx(expected)

    def test_multiple_periods_mixed_service(
        self, transport_input: TransportData, agency: pd.DataFrame, service_dates: pd.DataFrame
    ) -> None:
        services = pd.DataFrame(
            [
                {
                    "pattern_id": "local",
                    "service_id": "weekday",
                    "trip_id": "freq",
                    "start_time": "08:00:00",
                    "end_time": "09:00:00",
                    "headway_secs": 600,
                },
                {
                    "pattern_id": "local",
                    "service_id": "weekday",
                    "trip_id": "freq",
                    "start_time": "09:00:00",
                    "end_time": "10:00:00",
                    "headway_secs": 1200,
                },
                {
                    "pattern_id": "express",
                    "service_id": "weekday",
                    "trip_id": "fixed",
                    "departure_time": "08:00:00",
                },
            ]
        )
        with c2g.create_gtfs(
            transport_input,
            services,
            agency=agency,
            calendar_dates=service_dates,
            dwell_time_secs=0,
        ) as con:
            assert con.table("trips").count("*").fetchall() == [(2,)]
            _, edges = c2g.travel_summary_graph(
                con, calendar_start="20261102", calendar_end="20261102", directed=True
            )
            assert edges.loc[("001", "002"), "frequency"] == 9
            assert edges.loc[("001", "003"), "frequency"] == 1
            _, templates = c2g.travel_summary_graph(
                con,
                calendar_start="20261102",
                calendar_end="20261102",
                directed=True,
                use_frequencies=False,
            )
            assert templates.loc[("001", "002"), "frequency"] == 1

    @pytest.mark.parametrize(
        "problem",
        [
            "missing_table",
            "missing_column",
            "unknown_stop",
            "duplicate_sequence",
            "bad_crs",
            "invalid_time",
            "invalid_zone",
            "duplicate_trip",
            "bad_distance_method",
            "negative_travel",
            "missing_direction",
            "null_service",
        ],
    )
    def test_invalid_inputs(
        self,
        transport_input: TransportData,
        agency: pd.DataFrame,
        service_dates: pd.DataFrame | None,
        departures: pd.DataFrame,
        problem: str,
    ) -> None:
        dwell: float | None = 0
        method = "shape"
        if problem == "missing_table":
            del transport_input["patterns"]  # type: ignore[misc]
        elif problem == "missing_column":
            transport_input["stop_sequences"] = transport_input["stop_sequences"].drop(
                columns="stop_sequence"
            )
        elif problem == "unknown_stop":
            transport_input["stop_sequences"].loc[0, "stop_id"] = "missing"
        elif problem == "duplicate_sequence":
            transport_input["stop_sequences"].loc[1, "stop_sequence"] = 0
        elif problem == "bad_crs":
            transport_input["stops"] = cast("gpd.GeoDataFrame", transport_input["stops"]).set_crs(
                None, allow_override=True
            )
        elif problem == "invalid_time":
            departures.loc[0, "departure_time"] = "08:99:00"
        elif problem == "invalid_zone":
            agency.loc[0, "agency_timezone"] = "Not/AZone"
        elif problem == "duplicate_trip":
            departures["trip_id"] = "duplicate"
        elif problem == "bad_distance_method":
            method = "nearest"
        elif problem == "negative_travel":
            transport_input["stop_sequences"].loc[1, "travel_time_secs"] = -1
        elif problem == "missing_direction":
            transport_input["patterns"].loc[0, "direction_id"] = 3
        else:
            departures.loc[0, "service_id"] = None
        with pytest.raises(ValueError, match=r".+"):
            c2g.create_gtfs(
                transport_input,
                departures,
                agency=agency,
                calendar_dates=service_dates,
                dwell_time_secs=dwell,
                distance_method=method,
            )

    @pytest.mark.parametrize("headway", [0, -1, 1.5, float("inf")])
    def test_invalid_headway(
        self,
        transport_input: TransportData,
        agency: pd.DataFrame,
        service_dates: pd.DataFrame,
        headway: float,
    ) -> None:
        services = pd.DataFrame(
            [
                {
                    "pattern_id": "local",
                    "service_id": "weekday",
                    "start_time": "08:00:00",
                    "end_time": "09:00:00",
                    "headway_secs": headway,
                }
            ]
        )
        with pytest.raises(ValueError, match="headway"):
            c2g.create_gtfs(
                transport_input,
                services,
                agency=agency,
                calendar_dates=service_dates,
                dwell_time_secs=0,
            )

    def test_calendar_exceptions_and_optional_tables(
        self,
        transport_input: TransportData,
        agency: pd.DataFrame,
        departures: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        calendar = pd.DataFrame(
            [
                {
                    "service_id": "weekday",
                    "monday": 1,
                    "tuesday": 1,
                    "wednesday": 0,
                    "thursday": 0,
                    "friday": 0,
                    "saturday": 0,
                    "sunday": 0,
                    "start_date": "20261102",
                    "end_date": "20261108",
                }
            ]
        )
        exceptions = pd.DataFrame(
            [
                {"service_id": "weekday", "date": "20261103", "exception_type": 2},
                {"service_id": "weekday", "date": "20261108", "exception_type": 1},
            ]
        )
        optional = {
            "transfers": pd.DataFrame(
                [
                    {
                        "from_stop_id": "001",
                        "to_stop_id": "002",
                        "transfer_type": 2,
                        "min_transfer_time": 120,
                    }
                ]
            ),
            "pathways": pd.DataFrame(
                [
                    {
                        "pathway_id": "walk",
                        "from_stop_id": "001",
                        "to_stop_id": "002",
                        "pathway_mode": 1,
                        "is_bidirectional": 1,
                    }
                ]
            ),
            "levels": pd.DataFrame([{"level_id": "ground", "level_index": 0}]),
            "feed_info": pd.DataFrame(
                [
                    {
                        "feed_publisher_name": "Synthetic",
                        "feed_publisher_url": "https://example.org",
                        "feed_lang": "en",
                    }
                ]
            ),
            "attributions": pd.DataFrame(
                [{"organization_name": "Example source", "is_producer": 1}]
            ),
        }
        with c2g.create_gtfs(
            transport_input,
            departures,
            agency=agency,
            calendar=calendar,
            calendar_dates=exceptions,
            dwell_time_secs=0,
            optional_tables=optional,
        ) as con:
            _, edges = c2g.travel_summary_graph(
                con, calendar_start="20261102", calendar_end="20261108", directed=True
            )
            assert edges.frequency.eq(2).all()
            with c2g.load_gtfs(c2g.save_gtfs(con, tmp_path / "with-options.zip")) as restored:
                for name in optional:
                    assert_frame_equal(con.table(name).df(), restored.table(name).df())

    @pytest.mark.parametrize("basis", ["distance_m", "shape_dist_traveled"])
    def test_explicit_distances_and_projected_geometry(
        self,
        transport_input: TransportData,
        agency: pd.DataFrame,
        service_dates: pd.DataFrame,
        departures: pd.DataFrame,
        basis: str,
    ) -> None:
        sequence = transport_input["stop_sequences"].drop(columns="travel_time_secs")
        sequence[basis] = (
            [0, 1000, 1000, 0, 2000] if basis == "distance_m" else [0, 1000, 2000, 0, 2000]
        )
        sequence["speed_kph"] = [float("nan"), 36, float("nan"), float("nan"), float("nan")]
        transport_input["stop_sequences"] = sequence
        transport_input["stops"] = cast("gpd.GeoDataFrame", transport_input["stops"]).to_crs(3857)
        transport_input["patterns"] = cast("gpd.GeoDataFrame", transport_input["patterns"]).to_crs(
            3857
        )
        with c2g.create_gtfs(
            transport_input,
            departures,
            agency=agency,
            calendar_dates=service_dates,
            dwell_time_secs=0,
            speed_kph=36,
        ) as con:
            times = con.sql(
                "SELECT arrival_time FROM stop_times WHERE trip_id='service_0' ORDER BY stop_sequence"
            ).fetchall()
            assert times == [("08:00:00",), ("08:01:40",), ("08:03:20",)]

    def test_overlapping_frequency_periods(
        self,
        transport_input: TransportData,
        agency: pd.DataFrame,
        service_dates: pd.DataFrame,
    ) -> None:
        services = pd.DataFrame(
            [
                {
                    "pattern_id": "local",
                    "trip_id": "f",
                    "service_id": "weekday",
                    "start_time": "08:00:00",
                    "end_time": "09:00:00",
                    "headway_secs": 600,
                },
                {
                    "pattern_id": "local",
                    "trip_id": "f",
                    "service_id": "weekday",
                    "start_time": "08:30:00",
                    "end_time": "10:00:00",
                    "headway_secs": 300,
                },
            ]
        )
        with pytest.raises(ValueError, match="overlapping"):
            c2g.create_gtfs(
                transport_input,
                services,
                agency=agency,
                calendar_dates=service_dates,
                dwell_time_secs=0,
            )


class TestSaveGtfs:
    def test_recorded_values_survive_missing_group_metadata(
        self, transport_input: TransportData, tmp_path: Path
    ) -> None:
        with c2g.create_gtfs(transport_input) as con:
            expected = c2g.gtfs_generation_report(con)["services"]
            path = c2g.save_gtfs(con, tmp_path / "recorded.zip")
        sidecar = Path(str(path) + ".provenance.json")
        payload = json.loads(sidecar.read_text())
        payload["records"] = [
            row for row in payload["records"] if row["field"] != "group_headway_secs"
        ]
        for row in payload["records"]:
            if row["field"] == "model":
                row["value"] = "recorded-model"
            if row["source"] == "simple-v1":
                row["source"] = "recorded-model"
        sidecar.write_text(json.dumps(payload))
        with c2g.load_gtfs(path) as con:
            report = c2g.gtfs_generation_report(con)
            assert report["assumptions"].query("field == 'model'")["value"].tolist() == [
                "recorded-model"
            ]
            assert report["services"]["group_headway_secs"].isna().all()
            columns = ["date", "headway_secs", "departures"]
            assert_frame_equal(expected[columns], report["services"][columns])
            saved = c2g.save_gtfs(con, tmp_path / "recorded-again.zip")
        with c2g.load_gtfs(saved) as con:
            restored_report = c2g.gtfs_generation_report(con)
            assert restored_report.keys() == report.keys()
            for key, frame in report.items():
                assert_frame_equal(frame, restored_report[key])

    def test_missing_daily_history_retains_saved_counts(
        self,
        transport_input: TransportData,
        tmp_path: Path,
    ) -> None:
        with c2g.create_gtfs(transport_input, service_date="20000103", headway_secs=900) as con:
            path = c2g.save_gtfs(con, tmp_path / "legacy.zip")
        sidecar = Path(str(path) + ".provenance.json")
        payload = json.loads(sidecar.read_text())
        payload["records"] = [row for row in payload["records"] if row["field"] != "service_dates"]
        sidecar.write_text(json.dumps(payload))
        with c2g.load_gtfs(path) as restored:
            report = c2g.gtfs_generation_report(restored)
            assert report["assumptions"].query("field == 'model'")["value"].tolist() == [
                "simple-v1"
            ]
            assert report["services"]["date"].isna().all()
            assert report["services"]["departures"].tolist() == [64, 64]
            assert restored.sql("SELECT DISTINCT date FROM calendar_dates").fetchall() == [
                ("20000103",)
            ]
            saved = c2g.save_gtfs(restored, tmp_path / "legacy-again.zip")
        with c2g.load_gtfs(saved) as restored:
            restored_report = c2g.gtfs_generation_report(restored)
            assert restored_report.keys() == report.keys()
            for name, frame in report.items():
                assert_frame_equal(frame, restored_report[name])

    def test_report_roundtrip_and_automatic_provenance(
        self, transport_input: TransportData, tmp_path: Path
    ) -> None:
        with c2g.create_gtfs(transport_input) as con:
            before = c2g.gtfs_generation_report(con)
            path = c2g.save_gtfs(con, tmp_path / "feed.zip")
        with c2g.load_gtfs(path) as restored:
            after = c2g.gtfs_generation_report(restored)
            for name in before:
                assert_frame_equal(before[name], after[name])
        with (
            c2g.load_gtfs(path, provenance_path=None) as without,
            pytest.raises(ValueError, match="provenance is unavailable"),
        ):
            c2g.gtfs_generation_report(without)
        Path(str(path) + ".provenance.json").write_text("{bad", encoding="utf-8")
        with pytest.raises(ValueError, match="Expecting property name"):
            c2g.load_gtfs(path)

    def test_overwrite_and_exclusions(
        self,
        transport_input: TransportData,
        agency: pd.DataFrame,
        service_dates: pd.DataFrame,
        departures: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        with c2g.create_gtfs(
            transport_input,
            departures,
            agency=agency,
            calendar_dates=service_dates,
            dwell_time_secs=0,
        ) as con:
            c2g.travel_summary_graph(con)
            path = c2g.save_gtfs(con, tmp_path / "feed.zip")
            with ZipFile(path) as archive:
                assert "_city2graph_provenance.txt" not in archive.namelist()
                assert all(not name.startswith("_") for name in archive.namelist())
                assert "geometry" not in archive.read("stops.txt").decode().splitlines()[0]
            with pytest.raises(FileExistsError):
                c2g.save_gtfs(con, path)
            c2g.save_gtfs(con, path, overwrite=True, include_provenance=False)
            assert not Path(str(path) + ".provenance.json").exists()
            con.execute("CREATE TABLE fares (fare_id VARCHAR)")
            with pytest.raises(ValueError, match="Unsupported"):
                c2g.save_gtfs(con, tmp_path / "unsupported.zip")
            assert not (tmp_path / "unsupported.zip").exists()

    def test_external_feed_and_invalid_references(
        self, sample_gtfs_zip: str, tmp_path: Path
    ) -> None:
        with c2g.load_gtfs(sample_gtfs_zip) as con:
            # The legacy fixture deliberately omits agency metadata.
            if "agency" not in {row[0] for row in con.execute("SHOW TABLES").fetchall()}:
                con.execute(
                    "CREATE TABLE agency AS SELECT 'Test' AS agency_name, 'https://example.org' AS agency_url, "
                    "'Europe/London' AS agency_timezone"
                )
            con.execute("UPDATE trips SET route_id='unknown'")
            with pytest.raises(ValueError, match="unknown"):
                c2g.save_gtfs(con, tmp_path / "invalid.zip")

    @pytest.mark.parametrize(
        ("statement", "message"),
        [
            ("UPDATE stop_times SET arrival_time='07:00:00' WHERE stop_sequence='1'", "monotonic"),
            ("UPDATE stop_times SET stop_sequence='-1' WHERE stop_sequence='1'", "non-negative"),
            ("UPDATE stops SET stop_lat='91'", "stop_lat"),
            ("UPDATE stops SET location_type='1'", "boarding"),
            ("UPDATE routes SET route_short_name=NULL", "route_short_name"),
            ("UPDATE shapes SET shape_pt_lat='NaN'", "coordinate"),
            ("UPDATE calendar_dates SET date='20260230'", "invalid date"),
            ("UPDATE calendar_dates SET exception_type='3'", "exception_type"),
            ("UPDATE trips SET service_id='unregistered'", "unknown services"),
            ("DELETE FROM stop_times WHERE trip_id='service_0'", "without stop_times"),
        ],
    )
    def test_invalid_feed_is_not_written(
        self,
        transport_input: TransportData,
        agency: pd.DataFrame,
        service_dates: pd.DataFrame,
        departures: pd.DataFrame,
        tmp_path: Path,
        statement: str,
        message: str,
    ) -> None:
        with c2g.create_gtfs(
            transport_input,
            departures,
            agency=agency,
            calendar_dates=service_dates,
            dwell_time_secs=0,
        ) as con:
            if "location_type" in statement:
                con.execute("ALTER TABLE stops ADD COLUMN location_type VARCHAR")
            con.execute(statement)
            with pytest.raises(ValueError, match=message):
                c2g.save_gtfs(con, tmp_path / "invalid.zip")
            assert not (tmp_path / "invalid.zip").exists()

    def test_optional_transfer_requires_duration(
        self,
        transport_input: TransportData,
        agency: pd.DataFrame,
        service_dates: pd.DataFrame,
        departures: pd.DataFrame,
    ) -> None:
        with pytest.raises(ValueError, match="min_transfer_time"):
            c2g.create_gtfs(
                transport_input,
                departures,
                agency=agency,
                calendar_dates=service_dates,
                dwell_time_secs=0,
                optional_tables={
                    "transfers": pd.DataFrame(
                        [
                            {"from_stop_id": "001", "to_stop_id": "002", "transfer_type": 2},
                        ]
                    )
                },
            )

    def test_explicit_provenance_validation(
        self,
        transport_input: TransportData,
        agency: pd.DataFrame,
        service_dates: pd.DataFrame,
        departures: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        with c2g.create_gtfs(
            transport_input,
            departures,
            agency=agency,
            calendar_dates=service_dates,
            dwell_time_secs=0,
        ) as con:
            path = c2g.save_gtfs(con, tmp_path / "feed.zip")
        metadata = tmp_path / "invalid.json"
        metadata.write_text('{"version": 999, "records": []}', encoding="utf-8")
        with pytest.raises(ValueError, match="provenance"):
            c2g.load_gtfs(path, provenance_path=metadata)

    def test_preserves_blank_intermediate_stop_times(
        self,
        transport_input: TransportData,
        agency: pd.DataFrame,
        service_dates: pd.DataFrame,
        departures: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        with c2g.create_gtfs(
            transport_input,
            departures,
            agency=agency,
            calendar_dates=service_dates,
            dwell_time_secs=0,
        ) as con:
            con.execute(
                "UPDATE stop_times SET arrival_time=NULL, departure_time=NULL WHERE trip_id='service_0' AND stop_sequence='1'"
            )
            path = c2g.save_gtfs(con, tmp_path / "sparse.zip")
            with c2g.load_gtfs(path) as restored:
                assert restored.sql(
                    "SELECT count(*) FROM stop_times WHERE arrival_time IS NULL"
                ).fetchall() == [(1,)]
            con.execute("ALTER TABLE stop_times ADD COLUMN timepoint VARCHAR")
            con.execute("UPDATE stop_times SET timepoint='1' WHERE stop_sequence='1'")
            with pytest.raises(ValueError, match="exact timepoints"):
                c2g.save_gtfs(con, tmp_path / "invalid.zip")
