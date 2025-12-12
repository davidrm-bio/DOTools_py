from typing import Literal
from beartype import beartype
from pathlib import Path
import anndata as ad
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import spatialdata as st

from dotools_py.utils import convert_path, EmptyType

_Empty = EmptyType()


@beartype
def read_h5ad(
    path: str | Path,
    filename: str = None,
    **kwargs,
) -> ad.AnnData:
    """Read `.h5ad`-formatted hdf5 file.

    Parameters
    ----------
    path
        Directory with the H5AD file.
    filename
        Name of the H5AD file. If not specified, assume that `path` contains the full path to the H5AD file.
    kwargs
        Additional arguments pass to
        `ad.read_h5ad <https://anndata.readthedocs.io/en/stable/generated/anndata.io.read_h5ad.html>`_.

    Returns
    -------
    ad.AnnData
        Returns an `AnnData` Object.

    """
    input_path: Path = convert_path(path) if filename is None else convert_path(path) / filename
    return  ad.read_h5ad(filename=input_path, **kwargs)


def read_zarr(
    path: str | Path,
    filename: str = None,
    backend: Literal["anndata", "spatialdata"] = "anndata",
) -> "ad.AnnData | st.SpatialData":
    """Read from a hierarchical Zarr array store into an AnnData Object.

    Parameters
    ----------
    path
        Directory with the Zarr.
    filename
        Name of the Zarr array. If not specified, assume that `path` contains the full path to the Zarr directory.
    backend
        Library to use for reading. If ``"spatialdata"`` is selected an SpatialData Object is returned. Currently not
        implemented.

    Returns
    -------
    ad.AnnData
        Returns an `ad.AnnData` Object.

    """
    input_path: Path = convert_path(path) if filename is None else convert_path(path) / filename
    if backend == "spatialdata":
        try:
            import spatialdata as st
            adata: ad.AnnData | EmptyType | st.SpatialData = _Empty
            adata = st.read_zarr(store=input_path)

        except ModuleNotFoundError:
            raise ModuleNotFoundError("spatialdata backend requires spatial data to be installed")
    else:
        adata: ad.AnnData | EmptyType = _Empty
        adata: ad.AnnData = ad.read_zarr(store=input_path)
    return adata


def read_10x_h5(
    path: str | Path,
    filename: str = None,
    **kwargs
) -> ad.AnnData:
    """Read 10x-Genomics-formatted hdf5 file.

    Parameters
    ----------
    path
        Directory with the HDF5 file.
    filename
        Name of the file.  If not specified, assume that `path` contains the full path to the HDF5 file.
    kwargs
        Additional arguments pass to `scanpy.read_10x_h5 <https://scanpy.readthedocs.io/en/stable/generated/scanpy.read_10x_h5.html#scanpy.read_10x_h5>`_

    Returns
    -------
    Returns an `AnnData` object.

    """
    import scanpy as sc
    input_path: Path = convert_path(path) if filename is None else convert_path(path) / filename
    return sc.read_10x_h5(input_path, **kwargs)


def read_10x_mtx(
    path: str | Path,
    **kwargs
)-> ad.AnnData:
    """Read 10x-Genomics-formatted mtx directory.

    Parameters
    ----------
    path
        Directory with the `.mtx` and `.tsv` file.
    kwargs
        Additional arguments pass to `scanpy.read_10x_mtx <https://scanpy.readthedocs.io/en/stable/generated/scanpy.read_10x_mtx.html>`_

    Returns
    -------
    Returns an `AnnData` object.

    """
    import scanpy as sc
    return sc.read_10x_mtx(convert_path(path), **kwargs)



def read_mtx(
    path: str | Path,
    filename: str = None,
    **kwargs
)-> ad.AnnData:
    """Read `.mtx` file.

    Parameters
    ----------
    path
         Directory with the `.mtx` file.
    filename
        Name of the `.mtx` file.  If not specified, assume that `path` contains the full path to the `.mtx` file.
    kwargs
        Additional arguments pass to `anndata.io.read_mtx <https://anndata.readthedocs.io/en/stable/generated/anndata.io.read_mtx.html#anndata.io.read_mtx>`_.

    Returns
    -------
    Returns an `AnnData` object.

    """
    input_path: Path = convert_path(path) if filename is None else convert_path(path) / filename
    return  ad.io.read_mtx(input_path, **kwargs)

