from beartype import beartype
from beartype.typing import Literal, Any
from pathlib import Path
import pandas as pd
import polars as pl

from dotools_py.logger import logger
from dotools_py.utils import convert_path, EmptyType

_Empty = EmptyType()



@beartype
def read_excel(
    path: str | Path,
    filename: str,
    sheet_name: str = "Sheet1",
    backend: Literal["pandas", "polars"] = "pandas",
    **kwargs
) -> pd.DataFrame:
    """Read Excel Sheet into a DataFrame.

    Parameters
    ----------
    path
        Directory containing the Excel Sheet.
    filename
        Name of the Excel Sheet file, including its extension
    sheet_name
        Name of the Sheet to read.
    backend
        Library to use for reading. If ``"polars"`` is selected and reading fails, Pandas is used as a fallback.
    **kwargs
        Additional arguments passed directly to
        `polars.read_excel <https://docs.pola.rs/api/python/stable/reference/api/polars.read_excel.html>`_
        or
        `pandas.read_excel <https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html>`_.

    Returns
    -------
    pd.DataFrame
        Returns a `pd.DataFrame` containing the content from the selected sheet.

    """
    input_path: Path = convert_path(path) / filename
    df: pl.DataFrame | pd.DataFrame | EmptyType = _Empty
    if backend == "polars":
        try:
            df: pl.DataFrame = pl.read_excel(source=input_path, sheet_name=sheet_name, **kwargs)
            df = df.to_pandas()
        except Exception as e:
            logger.warn(f"Error using polars backend falling back to pandas.\n{e}")

    if df is _Empty:
        df: pd.DataFrame = pd.read_excel(io=input_path, sheet_name=sheet_name, **kwargs)

    return df


@beartype
def read_csv(
    path: str | Path,
    filename: str,
    delimiter: str = ",",
    backend: Literal["pandas", "polars"] = "pandas",
    **kwargs
) -> pd.DataFrame:
    """Read comma separated files into a DataFrame.

    Parameters
    ----------
    path
        Directory containing the comma separated file.
    filename
        Name of the comma separated file.
    delimiter
        Character or regex pattern to treat as the delimiter.
    backend
        Library to use for reading. If ``"polars"`` is selected and reading fails, Pandas is used as a fallback.
    kwargs
        Additional arguments passed directly to
            `polars.read_csv <https://docs.pola.rs/api/python/stable/reference/api/polars.read_csv.html>`_
            or
            `pandas.read_csv <https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html>`_.

    Returns
    -------
    pd.DataFrame
        Returns a `pd.DataFrame` containing the content from the selected sheet.

    """
    input_path: Path = convert_path(path) / filename
    df: pl.DataFrame | pd.DataFrame | Any  = _Empty

    if backend == "polars":
        try:
            df: pl.DataFrame = pl.read_csv(source=input_path, separator=delimiter, **kwargs)
            df = df.to_pandas()
        except Exception as e:
            logger.warn(f"Error using polars backend falling back to pandas.\n{e}")

    if df is _Empty:
        df: pd.DataFrame | Any = pd.read_csv(input_path, sep=delimiter, iterator=False, **kwargs)

    return  df


@beartype
def read_parquet(
    path: str | Path,
    filename: str,
    backend: Literal["pandas", "polars"] = "pandas",
    **kwargs
) -> pd.DataFrame:
    """Read a parquet object into a DataFrame.

    Parameters
    ----------
    path
        Directory containing the comma separated file.
    filename
         Name of the parquet file.
    backend
        Library to use for reading. If ``"polars"`` is selected and reading fails, Pandas is used as a fallback.
    kwargs
        Additional arguments passed directly to
            `polars.read_parquet <https://docs.pola.rs/api/python/stable/reference/api/polars.read_parquet.html>`_
            or
            `pandas.read_parquet <https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html>`_.

    Returns
    -------
     pd.DataFrame
            Returns a `pd.DataFrame` containing the content from the selected sheet.

    """

    input_path: Path = convert_path(path) / filename
    df: pl.DataFrame | pd.DataFrame | Any = _Empty

    if backend == "polars":
        try:
            df: pl.DataFrame = pl.read_parquet(source=input_path, **kwargs)
        except Exception as e:
            logger.warn(f"Error using polars backend falling back to pandas.\n{e}")

    if df is _Empty:
        df: pd.DataFrame | Any = pd.read_parquet(input_path, **kwargs)

    return df

