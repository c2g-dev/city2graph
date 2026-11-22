"""Public API tests for GTFS archive I/O."""

# ruff: noqa: D101, D102

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING
from typing import cast
from zipfile import ZipFile

import duckdb
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import city2graph as c2g
from city2graph import TransportData

if TYPE_CHECKING:
    from collections.abc import Mapping


pytestmark = pytest.mark.filterwarnings("ignore::city2graph.GTFSGenerationWarning")


class TestSaveGtfs:
    @pytest.mark.parametrize("failure", ["setup", "metadata", "metadata_table", "geometry"])
    def test_loading_failure_closes_connection(
        self,
        transport_input: TransportData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        failure: str,
    ) -> None:
        with c2g.create_gtfs(transport_input, service_date="20000103") as con:
            path = c2g.save_gtfs(con, tmp_path / "feed.zip", include_metadata=False)
        with ZipFile(path) as archive:
            files = {name: archive.read(name) for name in archive.namelist()}
        if failure == "metadata":
            files["city2graph_metadata.json"] = b'{"version": 2, "records": []}'
        elif failure == "metadata_table":
            files["city2graph_metadata.json"] = b'{"version": 1, "records": []}'
            files["_city2graph_metadata.txt"] = b"field\nexisting\n"
        elif failure == "geometry":
            files["stops.txt"] = b"stop_id,stop_name,stop_lat\nA,A,51.5\n"
        with ZipFile(path, "w") as archive:
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

    def test_metadata_input_is_preserved(
        self,
        transport_input: TransportData,
        tmp_path: Path,
    ) -> None:
        transport_input["metadata"] = pd.DataFrame(
            [
                {
                    "table_name": "routes",
                    "record_id": "001",
                    "field": "route_type",
                    "kind": "source",
                    "source": "example",
                    "value": "3",
                    "details": "Recorded input",
                }
            ]
        )
        supplied = cast("Mapping[str, pd.DataFrame]", transport_input)
        before = {name: frame.copy(deep=True) for name, frame in supplied.items()}
        with c2g.create_gtfs(transport_input, service_date="20000103") as con:
            path = c2g.save_gtfs(con, tmp_path / "metadata.zip")
        for name, frame in before.items():
            assert_frame_equal(frame, supplied[name])
        with c2g.load_gtfs(path) as restored:
            recorded = restored.table("_city2graph_metadata").df()
        original = recorded.query("table_name == 'routes' and field == 'route_type'")
        assert original["source"].tolist() == ["example"]
        assert original["value"].tolist() == ["3"]

    def test_archive_copy_and_rename_preserves_summary(
        self, transport_input: TransportData, tmp_path: Path
    ) -> None:
        with c2g.create_gtfs(transport_input) as con:
            expected = c2g.get_gtfs_summary(con)
            path = c2g.save_gtfs(con, tmp_path / "original.zip")
        assert list(tmp_path.iterdir()) == [path]
        copied = tmp_path / "copy.zip"
        shutil.copyfile(path, copied)
        renamed = copied.rename(tmp_path / "renamed.zip")
        with c2g.load_gtfs(renamed) as con:
            actual = c2g.get_gtfs_summary(con)
            for name, frame in expected.items():
                assert_frame_equal(frame, actual[name])

    @pytest.mark.parametrize(
        "metadata", ["{bad", '{"version":2,"records":[]}', '{"version":1,"records":[{}]}']
    )
    def test_invalid_embedded_metadata(
        self, transport_input: TransportData, tmp_path: Path, metadata: str
    ) -> None:
        with c2g.create_gtfs(transport_input) as con:
            path = c2g.save_gtfs(con, tmp_path / "feed.zip", include_metadata=False)
        with ZipFile(path, "a") as archive:
            archive.writestr("city2graph_metadata.json", metadata)
        with pytest.raises(ValueError, match=r"Expecting property name|Unsupported metadata"):
            c2g.load_gtfs(path)
        with c2g.load_gtfs(path, include_metadata=False) as con:
            assert not c2g.get_od_pairs(con).empty

    def test_report_roundtrip_and_automatic_metadata(
        self, transport_input: TransportData, tmp_path: Path
    ) -> None:
        with c2g.create_gtfs(transport_input) as con:
            before = c2g.get_gtfs_summary(con)
            path = c2g.save_gtfs(con, tmp_path / "feed.zip")
        with c2g.load_gtfs(path) as restored:
            after = c2g.get_gtfs_summary(restored)
            for name in before:
                assert_frame_equal(before[name], after[name])
        with (
            c2g.load_gtfs(path, include_metadata=False) as without,
            pytest.raises(ValueError, match="metadata is unavailable"),
        ):
            c2g.get_gtfs_summary(without)
        sidecar = Path(str(path) + ".metadata.json")
        sidecar.write_text("{bad", encoding="utf-8")
        with c2g.load_gtfs(path) as restored:
            assert_frame_equal(before["summary"], c2g.get_gtfs_summary(restored)["summary"])

    @pytest.mark.parametrize("field", ["service_dates", "service_input", "excluded_relation_count"])
    def test_incomplete_generation_history_requires_regeneration(
        self, transport_input: TransportData, tmp_path: Path, field: str
    ) -> None:
        """Require the current daily records before constructing a generation summary."""
        with c2g.create_gtfs(transport_input) as con:
            path = c2g.save_gtfs(con, tmp_path / "feed.zip")
            with ZipFile(path) as archive:
                payload = json.loads(archive.read("city2graph_metadata.json"))
            c2g.save_gtfs(con, path, overwrite=True, include_metadata=False)
        payload["records"] = [row for row in payload["records"] if row["field"] != field]
        with ZipFile(path, "a") as archive:
            archive.writestr("city2graph_metadata.json", json.dumps(payload))
        with (
            c2g.load_gtfs(path) as restored,
            pytest.raises(ValueError, match=r"incomplete.*regenerate"),
        ):
            c2g.get_gtfs_summary(restored)

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
                assert "_city2graph_metadata.txt" not in archive.namelist()
                assert all(not name.startswith("_") for name in archive.namelist())
                assert "geometry" not in archive.read("stops.txt").decode().splitlines()[0]
            with pytest.raises(FileExistsError):
                c2g.save_gtfs(con, path)
            adjacent = Path(str(path) + ".metadata.json")
            adjacent.write_text("unrelated file")
            c2g.save_gtfs(con, path, overwrite=True, include_metadata=False)
            with ZipFile(path) as archive:
                assert "city2graph_metadata.json" not in archive.namelist()
            assert adjacent.read_text() == "unrelated file"
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
            ("UPDATE routes SET route_type='inf'", "route_type"),
            ("UPDATE routes SET route_type='-inf'", "route_type"),
            ("UPDATE routes SET route_type='NaN'", "route_type"),
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
