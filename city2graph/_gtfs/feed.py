"""Persist GTFS tables and provenance and build reports from recorded decisions."""

from __future__ import annotations

import json
import math
import tempfile
import warnings
import zipfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import duckdb
import pandas as pd

from .schema import _KEYS
from .schema import _MODEL
from .schema import _PROVENANCE
from .schema import _PROVENANCE_COLUMNS
from .schema import _REGISTRY
from .schema import _REQUIRED
from .schema import GTFSGenerationWarning
from .schema import TransportData
from .schema import _default
from .schema import _present
from .schema import _records
from .schema import _seconds
from .schema import _text_frame
from .schema import _time_to_seconds
from .validation import _fail
from .validation import _require
from .validation import _validate_tables

if TYPE_CHECKING:
    from collections.abc import Mapping

    import geopandas as gpd

_INCLUDE_PROVENANCE = bool(_default("include_provenance"))


_OVERWRITE = bool(_default("overwrite"))


def _input_provenance(data: Mapping[str, pd.DataFrame]) -> list[dict[str, object]]:
    """
    Record supplied fields without replacing source-adapter provenance.

    Preserve adapter-supplied history when it identifies the same input field.

    Parameters
    ----------
    data : Mapping[str, pd.DataFrame]
        Common transport input tables.

    Returns
    -------
    list[dict[str, object]]
        Records identifying supplied and source-derived fields.
    """
    records: list[dict[str, object]] = []
    keys = {
        "patterns": ["pattern_id"],
        "stop_sequences": ["pattern_id", "stop_sequence"],
        "services": ["trip_id"],
    }
    source_records = []
    if "provenance" in data:
        source = data["provenance"]
        issues: list[str] = []
        _require(source, _PROVENANCE_COLUMNS[:5], "provenance", issues)
        _fail(issues)
        source_records = _records(source)
    supplied = {(row["table_name"], row["record_id"], row["field"]) for row in source_records}
    for table, frame in data.items():
        if table == "provenance":
            continue
        fields = keys.get(table, _KEYS.get(table, []))
        for index, row in enumerate(_records(frame)):
            record_id = "|".join(str(row.get(key, index)) for key in fields) or str(index)
            for field, value in row.items():
                if (table, record_id, field) not in supplied and _present(value):
                    records.append(
                        {
                            "table_name": table,
                            "record_id": record_id,
                            "field": field,
                            "kind": "user",
                            "source": "input",
                            "value": str(value),
                            "details": "",
                        }
                    )
    records.extend(source_records)
    return records


def _connection(
    tables: dict[str, pd.DataFrame], provenance: pd.DataFrame
) -> duckdb.DuckDBPyConnection:
    """
    Initialise a feed through the same SQL and geometry support as analysis.

    Close partially initialised connections if table creation fails.

    Parameters
    ----------
    tables : dict[str, pd.DataFrame]
        GTFS tables represented as string-valued DataFrames.
    provenance : pd.DataFrame
        Source and assumption records; estimation records are appended during generation.

    Returns
    -------
    duckdb.DuckDBPyConnection
        Initialised in-memory GTFS database.
    """
    con = duckdb.connect(":memory:")
    try:
        _ensure_graph_sql_support(con)
        for name, frame in {**tables, _PROVENANCE: _text_frame(provenance)}.items():
            con.register("_input_frame", frame)
            con.execute(f'CREATE TABLE "{name}" AS SELECT * FROM _input_frame')  # noqa: S608
            con.unregister("_input_frame")
        con.execute("ALTER TABLE stops ADD COLUMN geometry GEOMETRY")
        con.execute(
            "UPDATE stops SET geometry = ST_Point(try_cast(stop_lon AS DOUBLE), try_cast(stop_lat AS DOUBLE))"
        )
    except Exception:
        con.close()
        raise
    return con


def save_gtfs(
    con: duckdb.DuckDBPyConnection,
    path: str | Path,
    *,
    include_provenance: bool = _INCLUDE_PROVENANCE,
    overwrite: bool = _OVERWRITE,
) -> Path:
    """
    Validate and save supported GTFS tables as a ZIP archive.

    Analysis scratch tables and geometry columns are excluded. Provenance is
    stored separately so external consumers receive a standard GTFS archive.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Connection returned by create_gtfs, a transport loader or load_gtfs.
    path : str or pathlib.Path
        Destination ZIP file. Its parent directory must exist.
    include_provenance : bool, default True
        Write ``<path>.provenance.json`` when provenance is present. load_gtfs
        restores an adjacent sidecar automatically unless explicitly disabled.
    overwrite : bool, default False
        Explicitly permit replacing existing output files.

    Returns
    -------
    pathlib.Path
        Path to the saved ZIP archive.

    Raises
    ------
    ValueError
        If the feed is incomplete or contains unsupported persistent tables.
    FileExistsError
        If an output already exists and overwrite is False.
    """
    path = Path(path)
    sidecar = Path(str(path) + ".provenance.json")
    names = {
        row[0]
        for row in con.execute(
            "SELECT table_name FROM duckdb_tables() WHERE NOT temporary"
        ).fetchall()
    }
    unsupported = names - _REQUIRED.keys() - {_PROVENANCE}
    if unsupported:
        msg = f"Unsupported GTFS tables: {sorted(unsupported)}"
        raise ValueError(msg)
    tables = {name: _text_frame(con.table(name).df()) for name in names if name in _REQUIRED}
    _validate_tables(tables)
    metadata = None
    if include_provenance and _PROVENANCE in names:
        metadata = json.dumps(
            {"version": 1, "records": _records(con.table(_PROVENANCE).df().fillna(""))},
            ensure_ascii=False,
            indent=2,
        )
    if not overwrite and (path.exists() or sidecar.exists()):
        msg = f"Output exists: {path} or {sidecar}"
        raise FileExistsError(msg)
    # Prepare both outputs completely before replacing either destination.
    with tempfile.TemporaryDirectory(dir=path.parent) as directory:
        temporary = Path(directory) / "feed.zip"
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(tables):
                archive.writestr(
                    f"{name}.txt", tables[name].to_csv(index=False, lineterminator="\n")
                )
        temporary_metadata = Path(directory) / "provenance.json"
        if metadata is not None:
            temporary_metadata.write_text(metadata, encoding="utf-8")
        temporary.replace(path)
        if metadata is not None:
            temporary_metadata.replace(sidecar)
        elif overwrite:
            sidecar.unlink(missing_ok=True)
    return path


def _restore_provenance(con: duckdb.DuckDBPyConnection, path: str | Path) -> None:
    """
    Restore explicitly selected provenance without inspecting adjacent files.

    Only an explicitly supplied metadata path is read.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Open DuckDB connection containing GTFS tables.
    path : str | Path
        Explicit provenance JSON path.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("version") != 1
        or not isinstance(payload.get("records"), list)
    ):
        msg = "Unsupported provenance file format"
        raise ValueError(msg)
    frame = pd.DataFrame(payload["records"], columns=_PROVENANCE_COLUMNS)
    con.register("_provenance_input", _text_frame(frame))
    con.execute("CREATE TABLE _city2graph_provenance AS SELECT * FROM _provenance_input")
    con.unregister("_provenance_input")


def _assumption(
    records: list[dict[str, object]],
    record_id: str,
    field: str,
    value: object,
    kind: str = "default",
    source: str = _MODEL,
    details: str = "",
) -> None:
    """
    Append one effective value independently of the original input history.

    Effective records are the stable input to the public generation report.

    Parameters
    ----------
    records : list of dict
        Provenance records to extend.
    record_id : str
        Generation, service or leg identifier.
    field : str
        Name of the adopted parameter.
    value : object
        Adopted value, serialised as text.
    kind : str
        User, source, default, generated, estimated or ignored classification.
    source : str
        Origin of the value.
    details : str
        Optional explanation.
    """
    if not details:
        entry = _REGISTRY["settings"].get(field, {})
        if field == "speed_kph" and kind == "default":
            details = str(_REGISTRY["speed_reason"])
        elif kind == "default":
            details = str(entry.get("reason", "Documented synthetic-model convention"))
        elif kind == "generated":
            details = "Deterministically generated metadata, not user-supplied information"
        elif kind == "user":
            details = "Explicit input takes precedence over source information and defaults"
        elif kind == "source":
            details = "Adopted from an interpretable source value; not independently verified"
    records.append(
        {
            "table_name": "assumptions",
            "record_id": record_id,
            "field": field,
            "kind": kind,
            "source": source,
            "value": str(value),
            "details": details,
        }
    )


def _source_conditions(data: TransportData) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Read adapter-supplied operating values without treating them as user input.

    Unsupported source values remain in provenance but cannot supply defaults.

    Parameters
    ----------
    data : TransportData
        Common input with optional source provenance.

    Returns
    -------
    dict
        Pattern identifiers mapped to supported source fields and records.
    """
    result: dict[str, dict[str, dict[str, Any]]] = {}
    provenance = data.get("provenance")
    if provenance is not None:
        selected = provenance.loc[
            provenance["table_name"].eq("operation_source") & provenance["kind"].eq("source")
        ]
        for row in _records(selected):
            result.setdefault(str(row["record_id"]), {})[str(row["field"])] = row
    return result


def _retrieval_assumptions(records: list[dict[str, object]]) -> None:
    """Expose source retrieval context alongside operating assumptions.

    Retrieval settings describe the input snapshot rather than modelled operation.

    Parameters
    ----------
    records : list of dict
        Original provenance to extend with readable retrieval context.
    """
    for source_record in list(records):
        if source_record["table_name"] == "source" and source_record["field"] == "request":
            metadata = json.loads(str(source_record["details"]))
            metadata["retrieved_at"] = source_record["value"]
            if isinstance(metadata.get("osm3s"), dict):
                metadata["source_timestamp"] = metadata["osm3s"].get("timestamp_osm_base")
            for field in (
                "release",
                "endpoint",
                "timeout",
                "use_source_tags",
                "overwrite_source",
                "retrieved_at",
                "source_timestamp",
                "modes",
                "pattern_ids",
                "on_invalid",
                "excluded_pattern_ids",
            ):
                if field in metadata and metadata[field] is not None:
                    _assumption(
                        records,
                        f"source:{source_record['source']}",
                        field,
                        metadata[field],
                        "source",
                        str(source_record["source"]),
                        "Recorded retrieval context; not an operating assumption",
                    )


def _report_provenance(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Read and validate history before report aggregation.

    Missing effective records fail before any analysis tables are prepared.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Open DuckDB connection containing GTFS tables.

    Returns
    -------
    pandas.DataFrame
        Independent generation history with missing cells represented as empty strings.
    """
    tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    if _PROVENANCE not in tables:
        msg = "Generation provenance is unavailable; restore the .provenance.json sidecar"
        raise ValueError(msg)
    provenance = con.table(_PROVENANCE).df().fillna("")
    effective = provenance[provenance["table_name"].eq("assumptions")]
    if effective.empty:
        msg = "Generation provenance has no effective assumptions; regenerate the feed"
        raise ValueError(msg)
    return cast("pd.DataFrame", provenance)


def _generation_report(
    con: duckdb.DuckDBPyConnection,
    provenance: pd.DataFrame,
    edges: gpd.GeoDataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Inspect effective assumptions, pattern services and shared-section counts.

    The report uses persisted generation records and survives GTFS save/load.
    Edge counts are directed and cover all active service days in the feed.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Generated feed, optionally restored with its provenance sidecar.
    provenance : pd.DataFrame
        Validated persisted generation history.
    edges : gpd.GeoDataFrame
        Directed edge counts over the resolved service calendar.

    Returns
    -------
    dict of str to pandas.DataFrame
        ``assumptions`` lists adopted values and ignored source information;
        ``services`` lists each date and service period with its activity and
        number of departures; ``edges`` gives combined stop-to-stop traversals.
        Legacy reports without daily provenance retain their original aggregate.
        Frequency-based departures are expected counts unless exact_times is 1.
        Service columns and their order are stable. Non-applicable frequency
        fields are null for fixed departures. Counts and seconds are numeric,
        integer fields are nullable Int64, and IDs, dates and clocks are strings.
        Missing legacy fields remain null rather than being reconstructed.

    Raises
    ------
    ValueError
        If generation provenance is unavailable or predates effective records.
    """
    effective = provenance[provenance["table_name"].eq("assumptions")]
    assumptions = provenance[
        provenance["table_name"].isin(["assumptions", "operation_source"])
        | provenance["kind"].isin(["estimated", "ignored"])
    ].reset_index(drop=True)
    assumptions["unit"] = assumptions["field"].map(
        lambda field: str(
            _REGISTRY["settings"]
            .get(field, {})
            .get(
                "unit",
                {
                    "speed_kph": "km/h",
                    "headway_secs": "s",
                    "travel_time_secs": "s",
                    "duration_secs": "s",
                }.get(field, ""),
            )
        )
    )
    assumptions["reason"] = assumptions["details"].map(_readable_reason)
    assumptions["evidence"] = assumptions.apply(
        lambda row: (
            str(
                _REGISTRY["settings"]
                .get(row["field"], {})
                .get("evidence", "Unverified modelling assumption")
            )
            if row["kind"] == "default" and row["source"] == _MODEL
            else {
                "user": "Explicit input; not independently verified",
                "source": "Source information; not independently verified",
                "estimated": "Derived estimate; inspect the recorded method",
                "generated": "Generated identifier, selection or metadata",
                "ignored": "Not adopted",
            }.get(row["kind"], "Unverified legacy assumption")
        ),
        axis=1,
    )
    counts = dict(con.sql("SELECT service_id, sc FROM _service_counts").fetchall())
    active_dates = set(
        con.sql("SELECT service_id, strftime(active_date, '%Y%m%d') FROM _active_dates").fetchall()
    )
    rows = []
    for scope, frame in effective[effective["record_id"].str.startswith("service:")].groupby(
        "record_id", sort=False
    ):
        row = dict(zip(frame["field"], frame["value"], strict=True))
        row["active_days"] = int(counts.get(row["service_id"], 0))
        if "headway_secs" in row:
            count = (_seconds(row["end_time"]) - _seconds(row["start_time"])) / float(
                row["headway_secs"]
            )
            if float(row["exact_times"]) == 1:
                count = math.ceil(count)
        else:
            count = 1.0
        row["record_id"] = scope
        if "service_dates" in row:
            for date in json.loads(row["service_dates"]):
                daily = {
                    **row,
                    "date": date,
                    "weekday": (
                        "Monday",
                        "Tuesday",
                        "Wednesday",
                        "Thursday",
                        "Friday",
                        "Saturday",
                        "Sunday",
                    )[datetime.strptime(date, "%Y%m%d").weekday()],
                }
                daily["active_days"] = int((row["service_id"], date) in active_dates)
                daily["departures"] = count * daily["active_days"]
                rows.append(daily)
        else:
            row["departures"] = count * row["active_days"]
            rows.append(row)
    service_frame = pd.DataFrame(rows)
    trip_columns = con.table("trips").df()[["trip_id", "route_id", "direction_id"]]
    service_frame = service_frame.merge(trip_columns, on="trip_id", how="left")
    service_frame = service_frame.merge(
        con.table("routes").df()[["route_id", "route_type"]], on="route_id", how="left"
    )
    columns = [
        "date",
        "weekday",
        "pattern_id",
        "route_id",
        "direction_id",
        "route_type",
        "service_id",
        "trip_id",
        "departure_time",
        "start_time",
        "end_time",
        "headway_secs",
        "group_headway_secs",
        "exact_times",
        "departures",
        "active_days",
        "service_group_id",
        "group_pattern_count",
        "frequency_scope",
        "service_dates",
        "record_id",
    ]
    service_frame = service_frame.reindex(columns=columns)
    integers = {
        "direction_id",
        "route_type",
        "headway_secs",
        "group_headway_secs",
        "exact_times",
        "active_days",
        "group_pattern_count",
    }
    for field in columns:
        if field in integers or field == "departures":
            service_frame[field] = pd.to_numeric(service_frame[field], errors="raise").astype(
                "Float64" if field == "departures" else "Int64"
            )
        else:
            service_frame[field] = service_frame[field].astype("string")
    return {"assumptions": assumptions, "services": service_frame, "edges": edges}


def _readable_reason(value: str) -> str:
    """Extract a concise explanation while retaining the full original details.

    Structured estimation details remain available in the original details column.

    Parameters
    ----------
    value : str
        Plain-text or JSON provenance details.

    Returns
    -------
    str
        Recorded reason, estimation method, or an explicit history gap.
    """
    if not value:
        return "Explanation was not recorded by the original generator"
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value
    if isinstance(parsed, dict):
        return str(
            parsed.get("reason") or parsed.get("method") or parsed.get("distance_basis") or value
        )
    return value


def _warn_generation(records: list[dict[str, object]]) -> None:
    """Summarise unverified supplementation once after successful validation.

    One warning covers all fallbacks and estimates; the report retains their detail.

    Parameters
    ----------
    records : list of dict
        Complete original and effective generation history.
    """
    fields = sorted(
        {
            str(row["field"])
            for row in records
            if row.get("kind") in {"default", "estimated"}
            and row.get("field") not in {"model", "distance_method", "direction_id"}
        }
    )
    ignored = {str(row["field"]) for row in records if row.get("kind") == "ignored"}
    if fields or ignored:
        message = "Generated GTFS uses unverified assumptions or estimates"
        if fields:
            message += " for " + ", ".join(fields)
        if ignored:
            message += "; source information was not adopted for " + ", ".join(sorted(ignored))
        excluded = sorted(
            {
                str(row["record_id"])
                for row in records
                if row.get("field") == "relation" and row.get("kind") == "ignored"
            }
        )
        if excluded:
            message += f"; excluded {len(excluded)} OSM relations: {', '.join(excluded)}"
        warnings.warn(
            message + ". Inspect gtfs_generation_report(con) for values, reasons and scope.",
            GTFSGenerationWarning,
            stacklevel=3,
            skip_file_prefixes=(str(Path(__file__).parents[1]),),
        )


def _ensure_graph_sql_support(con: duckdb.DuckDBPyConnection) -> None:
    """
    Ensure SQL features required by transport graph queries are available.

    The function loads DuckDB spatial support and registers the
    ``time_to_seconds`` scalar UDF when missing.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Open DuckDB connection.

    Returns
    -------
    None
        The function mutates runtime SQL capabilities in ``con``.

    Raises
    ------
    RuntimeError
        If the DuckDB spatial extension cannot be loaded or installed.
    """
    try:
        con.execute("LOAD spatial;")
    except duckdb.Error:
        try:
            con.execute("INSTALL spatial; LOAD spatial;")
        except duckdb.Error as error:
            msg = "DuckDB spatial extension is required to build transport graph geometries."
            raise RuntimeError(msg) from error

    registered = con.execute(
        "SELECT count(*) FROM duckdb_functions() WHERE function_name = 'time_to_seconds'"
    ).fetchall()
    if not registered or not registered[0][0]:
        con.create_function(
            "time_to_seconds",
            _time_to_seconds,
            [duckdb.sqltype("VARCHAR")],
            duckdb.sqltype("DOUBLE"),
        )
