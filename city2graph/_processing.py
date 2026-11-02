"""Bound temporary numeric work without changing public materialised outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pandas as pd


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
