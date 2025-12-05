from typing import Literal
from beartype import beartype
from pathlib import Path
import anndata as ad

from dotools_py.utils import convert_path, EmptyType

_Empty = EmptyType()


@beartype
def read_h5ad(
    path: str | Path,
    filename: str,
    **kwargs,
) -> ad.AnnData:
    """Read `.h5ad`-formatted hdf5 file.

    Parameters
    ----------
    path
        Directory with the H5AD file.
    filename
        Name of the H5AD file.
    kwargs
        Additional arguments pass to
        `ad.read_h5ad <https://anndata.readthedocs.io/en/stable/generated/anndata.io.read_h5ad.html>`_.

    Returns
    -------
    ad.AnnData
        Returns an `AnnData` Object.

    """
    return ad.read_h5ad(filename =  convert_path(path) / filename, **kwargs)


@beartype
def read_zarr(
    path: str | Path,
    filename: str,
    backend: Literal["anndata", "spatialdata"],
) -> ad.AnnData:
    """Read from a hierarchical Zarr array store into an AnnData Object.

    Parameters
    ----------
    path
        Directory with the Zarr.
    filename
        Name of the Zarr array.
    backend
        Library to use for reading. If ``"spatialdata"`` is selected an SpatialData Object is returned. Currently not
        implemented.

    Returns
    -------
    ad.AnnData
        Returns an `ad.AnnData` Object.

    """

    input_path: Path = convert_path(path) / filename
    adata: ad.AnnData | EmptyType = _Empty
    if backend == "spatialdata":
        raise NotImplementedError("Currently not implemented")

    if adata is _Empty:
        adata: ad.AnnData = ad.read_zarr(store=input_path)

    return adata




