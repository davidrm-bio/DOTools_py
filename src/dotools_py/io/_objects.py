from typing import Literal
from beartype import beartype
from pathlib import Path
import anndata as ad

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import spatialdata as st

from dotools_py.utils import convert_path
from dotools_py.logger import logger
from dotools_py._custom_class import EmptyType

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
    return ad.read_h5ad(filename=input_path, **kwargs)


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
) -> ad.AnnData:
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
) -> ad.AnnData:
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
    return ad.io.read_mtx(input_path, **kwargs)


def _read_counts(
    path: str | Path,
    counts_file: str,
    library_id: str | None = None,
    **kwargs,
) -> tuple[ad.AnnData, str]:
    import scanpy as sc
    from h5py import File

    path = convert_path(path)
    if counts_file.endswith(".h5"):
        adata = read_10x_h5(path / counts_file, **kwargs)
        with File(path / counts_file, mode="r") as f:
            attrs = dict(f.attrs)
            if library_id is None:
                try:
                    lid = attrs.pop("library_ids")[0]
                    library_id = lid.decode("utf-8") if isinstance(lid, bytes) else str(lid)
                except ValueError:
                    raise KeyError("Unable to extract library id from attributes. Please specify one explicitly.")

            adata.uns["spatial"] = {library_id: {"metadata": {}}}  # can overwrite
            for key in ["chemistry_description", "software_version"]:
                if key not in attrs:
                    continue
                metadata = attrs[key].decode("utf-8") if isinstance(attrs[key], bytes) else attrs[key]
                adata.uns["spatial"][library_id]["metadata"][key] = metadata
        return adata, library_id

    if library_id is None:
        raise ValueError("Please explicitly specify library id.")

    if counts_file.endswith((".csv", ".txt")):
        adata = sc.read_text(path / counts_file, **kwargs)
    elif counts_file.endswith(".mtx.gz"):
        adata = read_10x_mtx(path, **kwargs)
    else:
        raise NotImplementedError("Not a valid input")

    adata.uns["spatial"] = {library_id: {"metadata": {}}}  # can overwrite
    return adata, library_id


def read_visium(
    path: str | Path,
    counts_file: str = "filtered_feature_bc_matrix.h5",
    library_id: str | None = None,
    load_images: bool = True,
    source_image_path: str | Path | None = None,
    **kwargs,
) -> ad.AnnData:
    """Read SpaceRanger output into AnnData Object.

    Adapted from `Squidpy <https://squidpy.readthedocs.io/en/stable/api/squidpy.read.visium.html#squidpy.read.visium>`_.

    :param path: Path to the folder containing the Visium files.
    :param counts_file: Name of the file to use as the count file.
    :param library_id: Identifier for the Visium library.
    :param load_images: Whether to load the image or not and save it in `adata.uns['spatial']`
    :param source_image_path: Path to the source image.
    :param kwargs: Additional arguments pass when reading the H5 file.
    :return: Returns an AnnData Object.

    """
    import json
    from PIL import Image
    import numpy as np
    import pandas as pd
    path = convert_path(path)
    adata, library_id = _read_counts(path, counts_file=counts_file, library_id=library_id, **kwargs)

    if not load_images:
        return adata

    # Load image
    adata.uns["spatial"][library_id]["images"] = {
        res: np.asarray(Image.open(path / f"spatial/tissue_{res}_image.png")) for res in ["hires", "lowres"]
    }

    # Load scale factors
    adata.uns["spatial"][library_id]["scalefactors"] = json.loads(
        (path / "spatial/scalefactors_json.json").read_bytes()
    )

    # Space Ranger versions use different file formats:
    #   - v1: tissue_positions.csv (no header)
    #   - v2: tissue_positions_list.csv (with header)
    #   - v3: tissue_positions.csv (with header)
    tissue_positions_file = (
        path / "spatial/tissue_positions.csv"
        if (path / "spatial/tissue_positions.csv").exists()
        else path / "spatial/tissue_positions_list.csv"
    )

    # Detect header by checking if first cell is 'barcode' (header) or a barcode value
    with open(tissue_positions_file) as f:
        first_cell = f.readline().split(",")[0].strip()
    has_header = first_cell.lower() == "barcode"

    coords = pd.read_csv(tissue_positions_file, header=0 if has_header else None, index_col=0)
    coords.columns = ["in_tissue", "array_row", "array_col", "pxl_col_in_fullres", "pxl_row_in_fullres"]
    coords.set_index(coords.index.astype(adata.obs.index.dtype), inplace=True)

    adata.obs = pd.merge(adata.obs, coords, how="left", left_index=True, right_index=True)
    adata.obsm["spatial"] = adata.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values
    adata.obs.drop(columns=["pxl_row_in_fullres", "pxl_col_in_fullres"], inplace=True)

    if source_image_path is not None:
        source_image_path = Path(source_image_path).absolute()
        if not source_image_path.exists():
            logger.warn(f"Path to the high-resolution tissue image `{source_image_path}` does not exist")
        adata.uns["spatial"][library_id]["metadata"]["source_image_path"] = str(source_image_path)

    return adata
