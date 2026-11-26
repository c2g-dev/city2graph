"""Bound temporary numeric work without changing public materialised outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterator


def row_batches(rows: int, columns: int) -> Iterator[slice]:
    """
    Generate bounded slices in original row order.

    A single row may exceed the byte budget; rows are never truncated.

    Parameters
    ----------
    rows : int
        Total number of rows.
    columns : int
        Number of float64 values per row.

    Yields
    ------
    slice
        Contiguous row slice containing at most 256 rows.
    """
    size = max(1, min(256, (8 * 1024 * 1024) // max(8 * columns, 1)))
    for start in range(0, rows, size):
        yield slice(start, min(start + size, rows))


def frame_records(frame: pd.DataFrame) -> Iterator[dict[Any, Any]]:
    """
    Yield independent attribute dictionaries.

    Bounded record batches preserve pandas scalar boxing, including nullable values,
    without retaining a full list of attributes.

    Parameters
    ----------
    frame : pandas.DataFrame
        Source table; its column order is retained.

    Yields
    ------
    dict[str, Any]
        Attributes for one row.
    """
    for batch in row_batches(len(frame), len(frame.columns)):
        yield from frame.iloc[batch].to_dict("records")


def column_or_level(df: pd.DataFrame, name: str) -> Any | None:  # noqa: ANN401
    """
    Get values from a column or index level by name.

    This helper abstracts the logic for retrieving data whether it resides in a
    DataFrame column or an index level, prioritizing columns.

    Parameters
    ----------
    df : pandas.DataFrame or geopandas.GeoDataFrame
        The DataFrame to search.
    name : str
        The name of the column or index level.

    Returns
    -------
    Any or None
        The values if found, otherwise None.
    """
    if name in df.columns:
        return df[name].to_numpy()
    if isinstance(df.index, pd.MultiIndex) and name in df.index.names:
        return df.index.get_level_values(name)
    # Handle standard Index with a name
    if df.index.name == name:
        return df.index.to_numpy()
    return None


def edge_endpoints(edges_df: pd.DataFrame) -> tuple[Any, Any] | None:
    """
    Read named endpoints or the first two levels of an edge index.

    A frame without endpoint identifiers can use geometry-based topology.

    Parameters
    ----------
    edges_df : pandas.DataFrame
        Spatial edge records in their original order.

    Returns
    -------
    tuple or None
        Source and target identifiers, or None for implicit nodes.
    """
    # 2. Implicit Candidates
    # List of (u_name, v_name) pairs to check
    candidates = [
        ("from_node_id", "to_node_id"),
        ("source_id", "target_id"),
        ("u", "v"),
        ("source", "target"),
    ]

    for u_name, v_name in candidates:
        u = column_or_level(edges_df, u_name)
        v = column_or_level(edges_df, v_name)
        if u is not None and v is not None:
            return u, v

    # 3. Fallback: MultiIndex position (first two levels)
    if isinstance(edges_df.index, pd.MultiIndex) and edges_df.index.nlevels >= 2:
        return edges_df.index.get_level_values(0), edges_df.index.get_level_values(1)

    return None
