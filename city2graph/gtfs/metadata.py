"""Record, validate and serialise GTFS source information and adopted conditions."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import pandas as pd

from .schema import _KEYS
from .schema import _METADATA
from .schema import _METADATA_COLUMNS
from .schema import _MODEL
from .schema import _REGISTRY
from .schema import GTFSGenerationWarning
from .schema import TransportData
from .schema import _present
from .schema import _records
from .schema import _text_frame
from .validation import _fail
from .validation import _require

if TYPE_CHECKING:
    from collections.abc import Mapping

    import duckdb


def _input_metadata(data: Mapping[str, pd.DataFrame]) -> list[dict[str, object]]:
    """
    Record supplied fields without replacing source-adapter metadata.

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
    if "metadata" in data:
        source = data["metadata"]
        issues: list[str] = []
        _require(source, _METADATA_COLUMNS[:5], "metadata", issues)
        _fail(issues)
        source_records = _records(source)
    supplied = {(row["table_name"], row["record_id"], row["field"]) for row in source_records}
    for table, frame in data.items():
        if table == "metadata":
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


def _restore_metadata(con: duckdb.DuckDBPyConnection, content: str | bytes) -> None:
    """
    Restore embedded metadata from its JSON content.

    The JSON content accompanies the GTFS tables in the same archive.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Open DuckDB connection containing GTFS tables.
    content : str or bytes
        Serialised metadata JSON.
    """
    payload = json.loads(content)
    if (
        not isinstance(payload, dict)
        or payload.get("version") != 1
        or not isinstance(payload.get("records"), list)
        or not all(
            isinstance(row, dict) and all(field in row for field in _METADATA_COLUMNS)
            for row in payload["records"]
        )
    ):
        msg = "Unsupported metadata file format"
        raise ValueError(msg)
    frame = pd.DataFrame(payload["records"], columns=_METADATA_COLUMNS)
    con.register("_metadata_input", _text_frame(frame))
    try:
        con.execute("CREATE TABLE _city2graph_metadata AS SELECT * FROM _metadata_input")
    finally:
        con.unregister("_metadata_input")


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
        Metadata records to extend.
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

    Unsupported source values remain in metadata but cannot supply defaults.

    Parameters
    ----------
    data : TransportData
        Common input with optional source metadata.

    Returns
    -------
    dict
        Pattern identifiers mapped to supported source fields and records.
    """
    result: dict[str, dict[str, dict[str, Any]]] = {}
    metadata = data.get("metadata")
    if metadata is not None:
        selected = metadata.loc[
            metadata["table_name"].eq("operation_source") & metadata["kind"].eq("source")
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
        Original metadata to extend with readable retrieval context.
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


def _report_metadata(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
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
    if _METADATA not in tables:
        msg = "Generation metadata is unavailable; load a GTFS archive with embedded history"
        raise ValueError(msg)
    metadata = con.table(_METADATA).df().fillna("")
    effective = metadata[metadata["table_name"].eq("assumptions")]
    if effective.empty:
        msg = "Generation metadata has no effective assumptions; regenerate the feed"
        raise ValueError(msg)
    required = {"generation": {"model", "excluded_relation_count"}}
    required.update(
        {
            scope: {"service_dates", "service_group_id", "group_pattern_count", "frequency_scope"}
            for scope in effective["record_id"].unique()
            if str(scope).startswith("service:")
        }
    )
    required.update(
        {
            f"pattern:{identifier}": {"service_input"}
            for identifier in effective.loc[effective["field"].eq("pattern_id"), "value"]
        }
    )
    recorded = effective.groupby("record_id", sort=False)["field"].agg(set).to_dict()
    for scope, fields in required.items():
        if missing := fields - recorded.get(scope, set()):
            msg = f"Generation metadata is incomplete for {scope}: missing {sorted(missing)}; regenerate the feed"
            raise ValueError(msg)
    return cast("pd.DataFrame", metadata)


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
    completed = {
        str(row["record_id"])
        for row in records
        if row.get("field") == "service_input" and row.get("value") == "completed"
    }
    if fields or ignored or completed:
        message = "Generated GTFS uses unverified assumptions or estimates"
        if fields:
            message += " for " + ", ".join(fields)
        if ignored:
            message += "; source information was not adopted for " + ", ".join(sorted(ignored))
        if completed:
            message += (
                f"; completed service rows for {len(completed)} patterns absent from services"
            )
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
            message + ". Inspect get_gtfs_summary(con) for values, reasons and scope.",
            GTFSGenerationWarning,
            stacklevel=3,
            skip_file_prefixes=(str(Path(__file__).parents[1]),),
        )


def _serialise_metadata(frame: pd.DataFrame) -> str:
    """Serialise generation records in the embedded metadata format.

    Validate required record fields before exporting the versioned JSON payload.

    Parameters
    ----------
    frame : pandas.DataFrame
        Source and effective-condition records.

    Returns
    -------
    str
        Versioned JSON content suitable for a GTFS ZIP archive.
    """
    issues: list[str] = []
    _require(frame, _METADATA_COLUMNS[:5], "metadata", issues)
    _fail(issues)
    return json.dumps(
        {"version": 1, "records": _records(frame.reindex(columns=_METADATA_COLUMNS).fillna(""))},
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )
