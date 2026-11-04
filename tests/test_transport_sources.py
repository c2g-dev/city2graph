"""Source adapters exercised exclusively through their public loaders."""

# ruff: noqa: D101, D102, D103

import copy
import io
import json
import urllib.error
import urllib.parse
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import cast

import geopandas as gpd
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from shapely.geometry import LineString
from shapely.geometry import Point

import city2graph as c2g

if TYPE_CHECKING:
    from collections.abc import Mapping


pytestmark = pytest.mark.filterwarnings("ignore::city2graph.GTFSGenerationWarning")


@pytest.fixture(autouse=True)
def fixed_overture_release(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("city2graph._gtfs.sources.ALL_RELEASES", ["2026-10-21.0"])


@pytest.fixture
def osm_response() -> dict[str, Any]:
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


@pytest.fixture
def osm_services() -> pd.DataFrame:
    return pd.DataFrame(
        [{"pattern_id": "osm:relation:42", "service_id": "weekday", "departure_time": "08:00:00"}]
    )


class TestLoadOsmTransport:
    def test_reuse_loaded_stops_as_supplement(
        self, monkeypatch: pytest.MonkeyPatch, osm_response: dict[str, Any], tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *_args, **_kwargs: io.BytesIO(json.dumps(osm_response).encode()),
        )
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
        self, monkeypatch: pytest.MonkeyPatch, osm_response: dict[str, Any], tmp_path: Path
    ) -> None:
        second = copy.deepcopy(osm_response["elements"][0])
        second["id"] = 43
        osm_response["elements"][0]["tags"]["opening_hours"] = "Mo-Su 06:00-09:00,17:00-21:00"
        osm_response["elements"].append(second)
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *_args, **_kwargs: io.BytesIO(json.dumps(osm_response).encode()),
        )
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
            report = c2g.gtfs_generation_report(con)
            assert report["services"]["departures"].sum() == 29
            archive = c2g.save_gtfs(con, tmp_path / "daily.zip")
        with c2g.load_gtfs(archive) as restored:
            restored_report = c2g.gtfs_generation_report(restored)
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
        with pytest.raises(ValueError, match=r"modes|pattern_ids|on_invalid|headway|services"):
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
            report = c2g.gtfs_generation_report(con)
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
        monkeypatch: pytest.MonkeyPatch,
        osm_response: dict[str, Any],
        mode: str,
        tag: str,
        route_type: int,
    ) -> None:
        osm_response["elements"][0]["tags"]["route"] = tag
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *_args, **_kwargs: io.BytesIO(json.dumps(osm_response).encode()),
        )
        with c2g.load_osm_transport(relation_ids=[42], modes=mode) as con:
            assert con.table("routes").df()["route_type"].tolist() == [str(route_type)]
        with pytest.raises(ValueError, match="No matching OSM"):
            c2g.load_osm_transport(relation_ids=[42], modes="bus")

    @pytest.mark.parametrize(
        "problem", ["unversioned", "one_stop", "platform", "unnamed_stop", "unnamed_route"]
    )
    def test_invalid_relations_aggregate_skip_and_roundtrip(
        self,
        monkeypatch: pytest.MonkeyPatch,
        osm_response: dict[str, Any],
        tmp_path: Path,
        problem: str,
    ) -> None:
        invalid = copy.deepcopy(osm_response["elements"][0])
        invalid["id"] = 43
        if problem == "unversioned":
            invalid["tags"].pop("public_transport:version")
        elif problem == "one_stop":
            invalid["members"] = invalid["members"][:1]
        elif problem == "unnamed_route":
            invalid["tags"].pop("name")
            invalid["tags"].pop("ref")
        elif problem == "unnamed_stop":
            osm_response["elements"].append({"type": "node", "id": 30, "lat": 51, "lon": 0.03})
            invalid["members"] = [
                {"type": "node", "ref": 30, "role": "stop"},
                {"type": "node", "ref": 2, "role": "stop"},
            ]
        else:
            invalid["members"][1]["ref"] = 2
        second = copy.deepcopy(invalid)
        second["id"] = 44
        osm_response["elements"].extend([invalid, second])
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *_args, **_kwargs: io.BytesIO(json.dumps(osm_response).encode()),
        )
        with pytest.raises(
            ValueError, match=r"(?s)osm:relation:43.*osm:relation:44.*Usable relation_ids: \[42\]"
        ):
            c2g.load_osm_transport(relation_ids=[42, 43, 44])
        with pytest.warns(c2g.GTFSGenerationWarning, match="excluded 2 OSM relations"):
            con = c2g.load_osm_transport(relation_ids=[42, 43, 44], on_invalid="skip")
        with con:
            report = c2g.gtfs_generation_report(con)
            rejected = report["assumptions"].query("field == 'relation' and kind == 'ignored'")
            assert set(rejected["record_id"]) == {"osm:relation:43", "osm:relation:44"}
            assert rejected["reason"].str.contains("Excluded by").all()
            assert con.table("routes").df()["route_id"].tolist() == ["osm:relation:42"]
            path = c2g.save_gtfs(con, tmp_path / "selected.zip")
        with c2g.load_gtfs(path) as restored:
            restored_report = c2g.gtfs_generation_report(restored)
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
        self, monkeypatch: pytest.MonkeyPatch, osm_response: dict[str, Any]
    ) -> None:
        invalid = copy.deepcopy(osm_response["elements"][0])
        invalid["id"] = 43
        invalid["tags"].pop("public_transport:version")
        osm_response["elements"].append(invalid)
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *_args, **_kwargs: io.BytesIO(json.dumps(osm_response).encode()),
        )
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
    def test_supplemental_patterns_obey_selection(
        self,
        monkeypatch: pytest.MonkeyPatch,
        osm_response: dict[str, Any],
        tmp_path: Path,
        options: dict[str, Any],
        expected: set[str],
    ) -> None:
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *_args, **_kwargs: io.BytesIO(json.dumps(osm_response).encode()),
        )
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
            report = c2g.gtfs_generation_report(con)
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
            restored_report = c2g.gtfs_generation_report(restored)
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
    def test_invalid_custom_input_and_empty_effective_mode_selection(
        self,
        monkeypatch: pytest.MonkeyPatch,
        osm_response: dict[str, Any],
        problem: str,
    ) -> None:
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *_args, **_kwargs: io.BytesIO(json.dumps(osm_response).encode()),
        )
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
            report = c2g.gtfs_generation_report(con)
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
        monkeypatch: pytest.MonkeyPatch,
        osm_response: dict[str, Any],
    ) -> None:
        osm_response["elements"][0]["tags"].pop("name")
        osm_response["elements"][0]["tags"].pop("ref")
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *_args, **_kwargs: io.BytesIO(json.dumps(osm_response).encode()),
        )
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
        self, monkeypatch: pytest.MonkeyPatch, osm_response: dict[str, Any]
    ) -> None:
        osm_response["elements"][0]["tags"].update(
            {"interval": "10", "interval:conditional": "5 @ (Mo-Fr 07:00-09:00)"}
        )
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *_args, **_kwargs: io.BytesIO(json.dumps(osm_response).encode()),
        )
        with c2g.load_osm_transport(
            relation_ids=[42], group_headway_secs=1200, service_date="20261103"
        ) as con:
            report = c2g.gtfs_generation_report(con)
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
            assert c2g.gtfs_generation_report(con)["services"]["headway_secs"].tolist() == [300]

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
        monkeypatch: pytest.MonkeyPatch,
        osm_response: dict[str, Any],
        tags: dict[str, str],
        periods: list[tuple[str, str]],
        headway: int,
        active: int,
        ignored: bool,
    ) -> None:
        osm_response["elements"][0]["tags"].update(tags)
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *_args, **_kwargs: io.BytesIO(json.dumps(osm_response).encode()),
        )
        with c2g.load_osm_transport(relation_ids=[42], service_date="20260105") as con:
            report = c2g.gtfs_generation_report(con)
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
            ("10 @ (Mo 07:00-09:00); 20 @ (Mo 08:00-10:00)", [("06:00:00", "22:00:00")], [900], 64),
            ("10 @ (Su 22:00-02:00); 20 @ (Mo 01:00-03:00)", [("06:00:00", "22:00:00")], [900], 64),
        ],
    )
    def test_conditional_intervals(
        self,
        monkeypatch: pytest.MonkeyPatch,
        osm_response: dict[str, Any],
        conditional: str,
        periods: list[tuple[str, str]],
        headways: list[int],
        departures: int,
    ) -> None:
        osm_response["elements"][0]["tags"].update(
            {"interval": "15", "interval:conditional": conditional}
        )
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *_a, **_k: io.BytesIO(json.dumps(osm_response).encode()),
        )
        with c2g.load_osm_transport(relation_ids=[42], service_date="20260105") as con:
            report = c2g.gtfs_generation_report(con)
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
        monkeypatch: pytest.MonkeyPatch,
        osm_response: dict[str, Any],
        explicit: bool,
    ) -> None:
        osm_response["elements"][0]["tags"]["duration"] = "00:12"
        osm_response["elements"][0]["members"].insert(2, {"type": "node", "ref": 4, "role": "stop"})
        osm_response["elements"].append(
            {"type": "node", "id": 4, "lat": 51, "lon": 0.005, "tags": {"name": "Middle"}}
        )
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *_a, **_k: io.BytesIO(json.dumps(osm_response).encode()),
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
            assumptions = c2g.gtfs_generation_report(con)["assumptions"]
            timed = assumptions.query("table_name == 'assumptions' and field == 'travel_time_secs'")
            assert "estimated" in set(timed["kind"])
            assert set(timed.query("kind == 'user'")["value"].astype(float)) <= {120}
        with c2g.load_osm_transport(relation_ids=[42], speed_kph=36) as con:
            ignored = c2g.gtfs_generation_report(con)["assumptions"].query("kind == 'ignored'")
            assert ignored["details"].str.contains("speed", case=False).any()

    def test_route_master_inheritance_and_shared_budget(
        self,
        monkeypatch: pytest.MonkeyPatch,
        osm_response: dict[str, Any],
    ) -> None:
        second = copy.deepcopy(osm_response["elements"][0])
        second["id"] = 43
        osm_response["elements"].extend(
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
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *_a, **_k: io.BytesIO(json.dumps(osm_response).encode()),
        )
        with c2g.load_osm_transport(relation_ids=[42, 43]) as con:
            report = c2g.gtfs_generation_report(con)
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
        osm_response["elements"][-1]["tags"]["interval"] = "10"
        with c2g.load_osm_transport(relation_ids=[42, 43]) as con:
            assert c2g.gtfs_generation_report(con)["services"]["departures"].sum() == 24
        osm_response["elements"][0]["tags"]["interval"] = "20"
        with c2g.load_osm_transport(relation_ids=[42, 43], service_date="20260105") as con:
            assert sorted(c2g.gtfs_generation_report(con)["services"]["departures"]) == [6, 12]

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
        monkeypatch: pytest.MonkeyPatch,
        osm_response: dict[str, Any],
        agency: pd.DataFrame,
        tags: dict[str, str],
        reason: str | None,
    ) -> None:
        osm_response["elements"][0]["tags"].update(tags)
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *_a, **_k: io.BytesIO(json.dumps(osm_response).encode()),
        )
        with c2g.load_osm_transport(relation_ids=[42]) as con:
            assumptions = c2g.gtfs_generation_report(con)["assumptions"]
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
        monkeypatch: pytest.MonkeyPatch,
        osm_response: dict[str, Any],
    ) -> None:
        osm_response["elements"][0]["tags"]["duration"] = "12"
        osm_response["elements"].extend(
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
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *_a, **_k: io.BytesIO(json.dumps(osm_response).encode()),
        )
        with c2g.load_osm_transport(relation_ids=[42]) as con:
            report = c2g.gtfs_generation_report(con)
            ignored = report["assumptions"].query("kind == 'ignored'")
            assert set(ignored["field"]) == {"route_master", "duration"}
            assert "12" in set(ignored["value"])
            assert report["services"]["headway_secs"].eq(900).all()

    def test_source_week_and_report_restore(
        self,
        monkeypatch: pytest.MonkeyPatch,
        osm_response: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        osm_response["elements"][0]["tags"].update(
            opening_hours="Mo 22:00-02:00; Tu-Su off", interval="10"
        )
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *_a, **_k: io.BytesIO(json.dumps(osm_response).encode()),
        )
        with c2g.load_osm_transport(relation_ids=[42]) as con:
            report = c2g.gtfs_generation_report(con)
            assert report["services"]["date"].tolist() == [
                f"202601{day:02d}" for day in range(3, 10)
            ]
            assert report["services"]["departures"].tolist() == [0, 0, 24, 0, 0, 0, 0]
            assert report["services"]["weekday"].iloc[0] == "Saturday"
            path = c2g.save_gtfs(con, tmp_path / "source.zip")
            with c2g.load_gtfs(path) as restored:
                restored_report = c2g.gtfs_generation_report(restored)
                assert restored_report.keys() == report.keys()
                for key, frame in report.items():
                    assert_frame_equal(frame, restored_report[key])

    def test_source_precedence_and_opt_out(
        self, monkeypatch: pytest.MonkeyPatch, osm_response: dict[str, Any]
    ) -> None:
        osm_response["elements"][0]["tags"].update(interval="10", opening_hours="Mo-Fr 07:00-10:00")
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *_args, **_kwargs: io.BytesIO(json.dumps(osm_response).encode()),
        )
        with c2g.load_osm_transport(
            relation_ids=[42], start_time="08:00:00", headway_secs=300, service_date="20260105"
        ) as con:
            services = c2g.gtfs_generation_report(con)["services"]
            assert services[["start_time", "end_time", "headway_secs"]].to_numpy().tolist() == [
                ["08:00:00", "10:00:00", 300]
            ]
        services = pd.DataFrame([{"start_time": "09:00:00", "headway_secs": 1200}])
        with c2g.load_osm_transport(
            services, relation_ids=[42], headway_secs=300, service_date="20260105"
        ) as con:
            assert c2g.gtfs_generation_report(con)["services"]["departures"].tolist() == [3]
        with c2g.load_osm_transport(
            relation_ids=[42], use_source_tags=False, service_date="20260105"
        ) as con:
            report = c2g.gtfs_generation_report(con)
            assert report["services"]["departures"].tolist() == [64]
            assert (
                report["assumptions"]
                .query("kind == 'ignored'")["details"]
                .str.contains("disabled")
                .all()
            )

    def test_explicit_calendar_and_source_weekdays(
        self,
        monkeypatch: pytest.MonkeyPatch,
        osm_response: dict[str, Any],
        service_dates: pd.DataFrame,
    ) -> None:
        osm_response["elements"][0]["tags"].update(interval="10", opening_hours="Sa-Su 08:00-10:00")
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *_args, **_kwargs: io.BytesIO(json.dumps(osm_response).encode()),
        )
        with c2g.load_osm_transport(relation_ids=[42], service_date="20000108") as con:
            assert c2g.gtfs_generation_report(con)["services"]["departures"].tolist() == [12]
        with c2g.load_osm_transport(relation_ids=[42], calendar_dates=service_dates) as con:
            report = c2g.gtfs_generation_report(con)
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
        self, monkeypatch: pytest.MonkeyPatch, osm_response: dict[str, Any]
    ) -> None:
        osm_response["elements"][0]["tags"]["opening_hours"] = "off"
        second = copy.deepcopy(osm_response["elements"][0])
        second["id"] = 43
        second["tags"]["opening_hours"] = "Mo 07:00-08:00"
        osm_response["elements"].append(second)
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *_args, **_kwargs: io.BytesIO(json.dumps(osm_response).encode()),
        )
        shared = pd.DataFrame([{"service_id": "shared"}])
        with c2g.load_osm_transport(shared, relation_ids=[42, 43]) as con:
            daily = c2g.gtfs_generation_report(con)["services"]
            assert daily["departures"].sum() == 4
            assert daily.loc[daily["pattern_id"].eq("osm:relation:42"), "departures"].eq(0).all()
            assert set(shared["service_id"]) == {"shared"}
        with c2g.load_osm_transport(
            relation_ids=[42], start_time="08:00:00", end_time="09:00:00"
        ) as con:
            report = c2g.gtfs_generation_report(con)
            assert report["services"]["departures"].tolist() == [4] * 7
            assert (
                report["assumptions"]
                .query("kind == 'ignored'")["details"]
                .str.contains("Explicit service windows")
                .any()
            )

    def test_place_selection_and_uniform_services(
        self,
        monkeypatch: pytest.MonkeyPatch,
        osm_response: dict[str, Any],
        osm_services: pd.DataFrame,
        agency: pd.DataFrame,
        service_dates: pd.DataFrame,
    ) -> None:
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *_args, **_kwargs: io.BytesIO(json.dumps(osm_response).encode()),
        )
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
                        "SELECT count(*) FROM _city2graph_provenance WHERE kind='source'"
                    ).fetchall()[0][0]
                    > 0
                )
                assert len(requests) == 1

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
            osm_response["elements"][2]["tags"]["name"] = "Unrelated"
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

    def test_explicit_override(
        self,
        monkeypatch: pytest.MonkeyPatch,
        osm_response: dict[str, Any],
        osm_services: pd.DataFrame,
        agency: pd.DataFrame,
        service_dates: pd.DataFrame,
    ) -> None:
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *_args, **_kwargs: io.BytesIO(json.dumps(osm_response).encode()),
        )
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
                "SELECT count(*) FROM _city2graph_provenance WHERE kind='override'"
            ).fetchall() == [(1,)]


class TestLoadOvertureTransport:
    def test_reuse_loaded_stops_and_numeric_directions(
        self, monkeypatch: pytest.MonkeyPatch, transport_input: c2g.TransportData, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "city2graph.data.load_overture_data",
            lambda **_kwargs: {
                "segment": gpd.GeoDataFrame(
                    {"id": ["s"]},
                    geometry=[LineString([(0, 51), (0.02, 51)])],
                    crs=4326,
                )
            },
        )
        with c2g.create_gtfs(transport_input, service_date="20261103") as original:
            expected = original.table("stop_times").df()
            archive = c2g.save_gtfs(original, tmp_path / "source.zip")
        with c2g.load_gtfs(archive) as loaded:
            transport_input["stops"] = loaded.table("stops").df()
        transport_input["patterns"]["direction_id"] = transport_input["patterns"][
            "direction_id"
        ].astype(float)
        with c2g.load_overture_transport(
            data=transport_input,
            area=[-0.01, 50.9, 0.03, 51.1],
            modes="bus",
            service_date="20261103",
        ) as con:
            assert_frame_equal(expected, con.table("stop_times").df())
            assert con.table("trips").df()["direction_id"].eq("0").all()

    @pytest.mark.parametrize("value", [float("inf"), float("-inf")])
    def test_nonfinite_route_type_precedes_network_access(
        self, monkeypatch: pytest.MonkeyPatch, transport_input: c2g.TransportData, value: float
    ) -> None:
        def unexpected(*_args: object, **_kwargs: object) -> None:
            pytest.fail("Invalid route_type must fail before any network access")

        monkeypatch.setattr("city2graph.data.get_boundaries", unexpected)
        monkeypatch.setattr("city2graph.data.load_overture_data", unexpected)
        transport_input["routes"]["route_type"] = value
        with pytest.raises(ValueError, match=r"routes row 0.*001.*route_type"):
            c2g.load_overture_transport(
                data=transport_input, place_name="Liverpool, UK", modes="bus"
            )

    @pytest.mark.parametrize(("modes", "count"), [("bus", 1), (["bus", "train"], 2), (None, 2)])
    def test_mode_selection_and_shared_stops(
        self,
        monkeypatch: pytest.MonkeyPatch,
        transport_input: c2g.TransportData,
        modes: str | list[str] | None,
        count: int,
    ) -> None:
        transport_input["routes"] = pd.concat(
            [
                transport_input["routes"],
                transport_input["routes"].assign(route_id="rail", route_type=2),
            ],
            ignore_index=True,
        )
        transport_input["patterns"].loc[1, "route_id"] = "rail"
        before = {
            name: frame.copy(deep=True)
            for name, frame in cast("Mapping[str, pd.DataFrame]", transport_input).items()
        }
        calls = []

        def fetch(**kwargs: object) -> dict[str, gpd.GeoDataFrame]:
            calls.append(kwargs)
            return {
                "segment": gpd.GeoDataFrame(
                    {"id": ["s"]}, geometry=[LineString([(0, 51), (0.02, 51)])], crs=4326
                )
            }

        monkeypatch.setattr("city2graph.data.load_overture_data", fetch)
        services = pd.DataFrame([{"pattern_id": "express", "headway_secs": 600}])
        with c2g.load_overture_transport(
            services,
            data=transport_input,
            area=[-0.01, 50.9, 0.03, 51.1],
            modes=modes,
            service_date="20261103",
        ) as con:
            assert len(con.table("routes").df()) == count
            assert len(c2g.gtfs_generation_report(con)["services"]) == count
            assert con.table("stops").df()["stop_id"].is_unique
        assert len(calls) == 1
        with c2g.load_overture_transport(
            data=transport_input,
            area=[-0.01, 50.9, 0.03, 51.1],
            modes=["bus", "train"],
            pattern_ids="express",
        ) as con:
            assert con.table("routes").df()["route_type"].tolist() == ["2"]
        for name, frame in before.items():
            assert_frame_equal(frame, cast("Mapping[str, pd.DataFrame]", transport_input)[name])

    @pytest.mark.parametrize(
        "options",
        [
            {"modes": []},
            {"modes": "plane"},
            {"modes": "ferry"},
            {"pattern_ids": "missing"},
            {"headway_sec": 600},
            {"services": pd.DataFrame([{"headway_sec": 600}])},
        ],
    )
    def test_mode_preflight_before_download(
        self,
        monkeypatch: pytest.MonkeyPatch,
        transport_input: c2g.TransportData,
        options: dict[str, Any],
    ) -> None:
        def unexpected(**_kwargs: object) -> None:
            pytest.fail("Invalid input must be rejected before downloading")

        monkeypatch.setattr("city2graph.data.load_overture_data", unexpected)
        with pytest.raises(ValueError, match=r"modes|pattern_ids|on_invalid|headway|services"):
            c2g.load_overture_transport(
                data=transport_input, area=[-0.01, 50.9, 0.03, 51.1], **options
            )

    def test_without_operating_parameters(
        self, monkeypatch: pytest.MonkeyPatch, transport_input: c2g.TransportData
    ) -> None:
        monkeypatch.setattr(
            "city2graph.data.load_overture_data",
            lambda **_kwargs: {
                "segment": gpd.GeoDataFrame(
                    {"id": ["s"]}, geometry=[LineString([(0, 51), (0.02, 51)])], crs=4326
                )
            },
        )
        transport_input["stop_sequences"] = transport_input["stop_sequences"].drop(
            columns="travel_time_secs"
        )
        with c2g.load_overture_transport(
            data=transport_input, area=[-0.01, 50.9, 0.03, 51.1]
        ) as con:
            report = c2g.gtfs_generation_report(con)
            assert report["services"].groupby("pattern_id")["departures"].sum().to_dict() == {
                "express": 224,
                "local": 224,
            }
            assert set(report["assumptions"].query("field == 'speed_kph'")["kind"]) == {"default"}

    def test_loader_matches_direct_generator(
        self,
        monkeypatch: pytest.MonkeyPatch,
        transport_input: c2g.TransportData,
        agency: pd.DataFrame,
        service_dates: pd.DataFrame,
        departures: pd.DataFrame,
    ) -> None:
        calls = []
        raw = {
            "segment": gpd.GeoDataFrame(
                {"id": ["segment-1"], "names": [{"primary": "Road"}]},
                geometry=[LineString([(0, 51), (0.01, 51), (0.02, 51)])],
                crs=4326,
            )
        }

        def load(**kwargs: object) -> dict[str, gpd.GeoDataFrame]:
            calls.append(kwargs)
            return raw

        monkeypatch.setattr("city2graph.data.load_overture_data", load)
        data = dict(cast("dict[str, pd.DataFrame]", transport_input))
        data["patterns"] = pd.DataFrame(data["patterns"].drop(columns="geometry"))
        data["patterns"]["segment_ids"] = [["segment-1"], ["segment-1"]]
        with (
            c2g.load_overture_transport(
                departures,
                agency=agency,
                data=data,
                area=[-0.01, 50.9, 0.03, 51.1],
                release="2026-10-21.0",
                route_name="Road",
                calendar_dates=service_dates,
                dwell_time_secs=0,
            ) as con,
            c2g.create_gtfs(
                transport_input,
                departures,
                agency=agency,
                calendar_dates=service_dates,
                dwell_time_secs=0,
            ) as direct,
        ):
            for table in ("trips", "stop_times", "shapes"):
                assert_frame_equal(con.table(table).df(), direct.table(table).df())
            assert calls[0]["save_to_file"] is False
            assert calls[0]["release"] == "2026-10-21.0"

    def test_missing_operating_inputs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        agency: pd.DataFrame,
        service_dates: pd.DataFrame,
        departures: pd.DataFrame,
    ) -> None:
        monkeypatch.setattr(
            "city2graph.data.load_overture_data",
            lambda **_kwargs: {
                "segment": gpd.GeoDataFrame(
                    {"id": ["s"]}, geometry=[LineString([(0, 0), (1, 1)])], crs=4326
                )
            },
        )
        with pytest.raises(ValueError, match="missing input table"):
            c2g.load_overture_transport(
                departures,
                agency=agency,
                data={},
                area=[0, 0, 1, 1],
                calendar_dates=service_dates,
                dwell_time_secs=0,
            )

    def test_empty_result(
        self, monkeypatch: pytest.MonkeyPatch, agency: pd.DataFrame, departures: pd.DataFrame
    ) -> None:
        monkeypatch.setattr("city2graph.data.load_overture_data", lambda **_kwargs: {})
        with pytest.raises(ValueError, match="No matching"):
            c2g.load_overture_transport(departures, agency=agency, data={}, area=[0, 0, 1, 1])

    @pytest.mark.parametrize("encoding", ["mapping", "json", "literal"])
    def test_selected_station_candidates(
        self,
        monkeypatch: pytest.MonkeyPatch,
        transport_input: c2g.TransportData,
        agency: pd.DataFrame,
        service_dates: pd.DataFrame,
        departures: pd.DataFrame,
        encoding: str,
    ) -> None:
        source = gpd.GeoDataFrame(
            {
                "id": ["a", "b"],
                "names": [{"primary": "A"}, {"primary": "B"}],
                "categories": [{"primary": "train_station"}] * 2,
            },
            geometry=[Point(0, 51), Point(0.01, 51)],
            crs=4326,
        )
        if encoding != "mapping":
            for field in ("names", "categories"):
                source[field] = source[field].map(json.dumps if encoding == "json" else str)
        calls = []

        def load(**kwargs: object) -> dict[str, gpd.GeoDataFrame]:
            calls.append(kwargs)
            return {
                "place": source,
                "segment": gpd.GeoDataFrame(
                    {"id": ["s"]}, geometry=[LineString([(0, 51), (0.01, 51)])], crs=4326
                ),
            }

        monkeypatch.setattr("city2graph.data.load_overture_data", load)
        data = {
            "routes": transport_input["routes"],
            "patterns": transport_input["patterns"].iloc[:1],
            "stop_sequences": transport_input["stop_sequences"].iloc[:2].copy(),
        }
        data["stop_sequences"]["stop_id"] = ["overture:a", "overture:b"]
        with c2g.load_overture_transport(
            departures.iloc[:1],
            agency=agency,
            data=data,
            area=[-0.01, 50.9, 0.03, 51.1],
            calendar_dates=service_dates,
            dwell_time_secs=0,
        ) as con:
            assert con.sql("SELECT stop_id FROM stops ORDER BY stop_id").fetchall() == [
                ("overture:a",),
                ("overture:b",),
            ]
            assert calls[0]["types"] == ["segment", "connector", "place"]
