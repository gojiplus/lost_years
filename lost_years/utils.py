"""Shared helpers for reading input frames and matching to life-table rows."""

import argparse
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import requests

from .datasets import TableUnavailableError

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

# Setup logger
logger = logging.getLogger(__name__)


def isstring(s: Any) -> bool:
    """Report whether ``s`` is a string.

    Args:
        s: Value to test.

    Returns:
        True when ``s`` is a ``str``.
    """
    return isinstance(s, str)


def column_exists(df: pd.DataFrame, col: str | None) -> bool:
    """Check the column name exists in the DataFrame.

    Args:
        df: Pandas DataFrame.
        col: Column name.

    Returns:
        bool: True if exists, False if not exists.

    """
    if col and (col not in df.columns):
        logger.warning("The specified column `%s` was not found in the input file", col)
        return False
    return True


def fixup_columns(cols: list[Any]) -> list[str]:
    """Replace index location column to name with `col` prefix.

    Args:
        cols: List of original columns

    Returns:
        List of column names

    """
    out_cols = []
    for col in cols:
        if isinstance(col, int):
            out_cols.append(f"col{col:d}")
        else:
            out_cols.append(col)
    return out_cols


def closest(
    lst: "list[float] | npt.NDArray[np.floating[Any]]",
    c: float,
    tolerance: float | None = None,
) -> float:
    """Find closest value in list or array.

    A missing target is rejected rather than matched. ``abs(x - nan)`` is
    ``nan`` and ``nan`` compares False against everything, so ``min`` used to
    fall through to the first element: a row with no age silently returned the
    life expectancy at age 0.

    ``tolerance`` bounds how far the answer may sit from the question. Without
    it, asking for the year 1900 or 2500 against a table that holds only 2022
    returns the 2022 figure with nothing to say it is not an answer.

    Args:
        lst: List of floats or numpy array
        c: Target value to find closest match for
        tolerance: Largest accepted distance between ``c`` and the match.
            None accepts any distance.

    Returns:
        Closest value in the list/array

    Raises:
        ValueError: If ``c`` is missing, if there is no non-missing candidate,
            or if the closest candidate is further than ``tolerance`` away.
    """
    if c is None or pd.isna(c):
        raise ValueError("cannot match a missing value")
    working_list: list[float] = lst if isinstance(lst, list) else lst.tolist()
    candidates = [v for v in working_list if v is not None and not pd.isna(v)]
    if not candidates:
        raise ValueError("no candidate values to match against")
    match = min(candidates, key=lambda v: abs(v - c))
    if tolerance is not None and abs(match - c) > tolerance:
        raise ValueError(
            f"closest available value {match} is more than {tolerance} from {c}"
        )
    return match


def download_file(url: str, local_path: str | Path | None = None) -> None:
    """Stream ``url`` to disk.

    Args:
        url: Source URL.
        local_path: Destination path. Defaults to the URL's last path segment.
    """
    match local_path:
        case None:
            local_path = Path(url.split("/")[-1])
        case str():
            local_path = Path(local_path)
        case _:
            pass  # Already a Path object

    # These are multi-megabyte life tables on slow public hosts; the timeout is
    # per-read, not for the whole transfer, so a generous value is safe.
    r = requests.get(url, timeout=60)
    with local_path.open("wb") as f:
        for chunk in r.iter_content(chunk_size=512 * 1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


COLUMN_FLAGS = {"age": "-a", "country": "-c", "sex": "-s", "year": "-y"}


def run_lookup_cli(
    argv: list[str],
    *,
    title: str,
    columns: tuple[str, ...],
    lookup: Callable[[pd.DataFrame, dict[str, str], argparse.Namespace], pd.DataFrame],
    output_default: str,
    output_help: str,
    add_options: Callable[[argparse.ArgumentParser], None] | None = None,
) -> int:
    """Run a per-source command: read a CSV, append life-table columns, write it.

    Args:
        argv: Command line arguments.
        title: Description shown by ``--help``.
        columns: Input columns the source matches on, in option order.
        lookup: Appends the source's columns, given the input, the column
            mapping and the parsed arguments.
        output_default: Output file when ``--output`` is not given.
        output_help: Help text for ``--output``.
        add_options: Adds the source's own options, before ``--output``.

    Returns:
        0 on success, -1 when a column is missing or the source's table has not
        been installed.
    """
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument("input", default=None, help="Input file")
    for col in columns:
        parser.add_argument(
            COLUMN_FLAGS[col],
            f"--{col}",
            default=col,
            help=f"Column name of {col} in the input file (default=`{col}`)",
        )
    if add_options is not None:
        add_options(parser)
    parser.add_argument("-o", "--output", default=output_default, help=output_help)

    args = parser.parse_args(argv)
    logger.debug(args)

    df = pd.read_csv(args.input)

    names = {col: getattr(args, col) for col in columns}
    for name in names.values():
        if not column_exists(df, name):
            logger.error("Column: `%s` not found in the input file", name)
            return -1

    try:
        result = lookup(df, names, args)
    except TableUnavailableError as exc:
        logger.error("%s", exc)
        return -1

    logger.info("Saving output to file: `%s`", args.output)
    result.columns = fixup_columns(result.columns.tolist())
    result.to_csv(args.output, index=False)

    return 0
