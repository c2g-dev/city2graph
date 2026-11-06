"""Initialise DuckDB feeds and persist validated GTFS tables as ZIP archives."""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

if TYPE_CHECKING:
    import pandas as pd

from .metadata import _serialise_metadata
from .schema import _METADATA
from .schema import _METADATA_FILENAME
from .schema import _REQUIRED
from .schema import _default
from .schema import _text_frame
from .schema import _time_to_seconds
from .validation import _validate_tables

_INCLUDE_METADATA = bool(_default("include_metadata"))
_OVERWRITE = bool(_default("overwrite"))


def _connection(
    tables: dict[str, pd.DataFrame], metadata: pd.DataFrame
) -> duckdb.DuckDBPyConnection:
    """
    Initialise a feed through the same SQL and geometry support as analysis.

    Close partially initialised connections if table creation fails.

    Parameters
    ----------
    tables : dict[str, pd.DataFrame]
        GTFS tables represented as string-valued DataFrames.
    metadata : pd.DataFrame
        Source and assumption records; estimation records are appended during generation.

    Returns
    -------
    duckdb.DuckDBPyConnection
        Initialised in-memory GTFS database.
    """
    con = duckdb.connect(":memory:")
    try:
        _ensure_graph_sql_support(con)
        for name, frame in {**tables, _METADATA: _text_frame(metadata)}.items():
            con.register("_input_frame", frame)
            try:
                con.execute(f'CREATE TABLE "{name}" AS SELECT * FROM _input_frame')  # noqa: S608
            finally:
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
    include_metadata: bool = _INCLUDE_METADATA,
    overwrite: bool = _OVERWRITE,
) -> Path:
    """
    Validate and save supported GTFS tables as a ZIP archive.

    Analysis scratch tables and geometry columns are excluded. Metadata is
    included as JSON so copying or renaming the archive preserves its history.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Connection returned by create_gtfs, a transport loader or load_gtfs.
    path : str or pathlib.Path
        Destination ZIP file. Its parent directory must exist.
    include_metadata : bool, default True
        Include ``city2graph_metadata.json`` when metadata is present.
        load_gtfs restores it automatically. Set False to export only GTFS tables.
    overwrite : bool, default False
        Explicitly permit replacing an existing archive.

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
    names = {
        row[0]
        for row in con.execute(
            "SELECT table_name FROM duckdb_tables() WHERE NOT temporary"
        ).fetchall()
    }
    unsupported = names - _REQUIRED.keys() - {_METADATA}
    if unsupported:
        msg = f"Unsupported GTFS tables: {sorted(unsupported)}"
        raise ValueError(msg)
    tables = {name: _text_frame(con.table(name).df()) for name in names if name in _REQUIRED}
    _validate_tables(tables)
    metadata = None
    if include_metadata and _METADATA in names:
        metadata = _serialise_metadata(con.table(_METADATA).df())
    if not overwrite and path.exists():
        msg = f"Output exists: {path}"
        raise FileExistsError(msg)
    # Prepare the complete archive before replacing the destination.
    with tempfile.TemporaryDirectory(dir=path.parent) as directory:
        temporary = Path(directory) / "feed.zip"
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(tables):
                archive.writestr(
                    f"{name}.txt", tables[name].to_csv(index=False, lineterminator="\n")
                )
            if metadata is not None:
                archive.writestr(_METADATA_FILENAME, metadata)
        temporary.replace(path)
    return path


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
