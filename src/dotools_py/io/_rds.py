import os.path
import uuid
import shutil
import subprocess
from pathlib import Path
from typing import Literal

import anndata as ad
import pandas as pd
import numpy as np
from scipy.sparse import issparse, csr_matrix

from dotools_py._utils import get_paths_utils, check_r_package
from dotools_py import logger
from dotools_py._custom_class import PathLike, InputError


def read_rds(
    path_rds: PathLike,
    path_h5ad: PathLike,
    batch_key: str = "batch",
) -> ad.AnnData | None:
    """Read Rds object with Seurat or SingleCellExperiment Object.

    .. note::
        When reading an RDS Object with counts and logcounts data, the logcounts will be returned in the
        `X` attribute, while the counts are returned as a layer.

    :param path_rds: Path to RDS file with SingleCellExperiment or SeuratObject.
    :param path_h5ad: Path to save AnnData Object.
    :param batch_key: Name in `obs` to save batch information.
    :return: Returns an `AnnData` Object or `None`. The AnnData can also be saved under `path_adata`.

    See Also
    --------
        :func:`dotools_py.io.save_rds`: Save an AnnData as  SingleCellExperiment or Seurat Object

    Example
    -------
    >>> import dotools_py as do
    >>> path_seurat = "/tmp/Seurat.rds"
    >>> path_adata = "/tmp/adata.h5ad"
    >>> adata = do.io.read_rds(path_rds=path_seurat, path_h5ad=path_adata)
    >>> adata
    AnnData object with n_obs × n_vars = 2801 × 18517
        obs: 'nCount_originalexp', 'nFeature_originalexp', 'batch', 'condition', 'n_genes_by_counts', 'log1p_n_genes_by_counts',
             'total_counts', 'log1p_total_counts', 'total_counts_mt', 'log1p_total_counts_mt', 'pct_counts_mt', 'total_counts_ribo',
             'log1p_total_counts_ribo', 'pct_counts_ribo', 'n_genes', 'n_counts', 'doublet_class', 'doublet_score', 'leiden',
             'cell_type', 'autoAnnot', 'celltypist_conf_score', 'annotation', 'annotation_recluster'
        var: 'mean', 'std', 'highly_variable', 'means', 'dispersions', 'dispersions_norm', 'highly_variable_nbatches', 'highly_variable_intersection', 'varm.PCs.V1', 'varm.PCs.V2', 'varm.PCs.V3', 'varm.PCs.V4', 'varm.PCs.V5', 'varm.PCs.V6', 'varm.PCs.V7', 'varm.PCs.V8', 'varm.PCs.V9', 'varm.PCs.V10', 'varm.PCs.V11', 'varm.PCs.V12', 'varm.PCs.V13', 'varm.PCs.V14', 'varm.PCs.V15', 'varm.PCs.V16', 'varm.PCs.V17', 'varm.PCs.V18', 'varm.PCs.V19', 'varm.PCs.V20', 'varm.PCs.V21', 'varm.PCs.V22', 'varm.PCs.V23', 'varm.PCs.V24', 'varm.PCs.V25', 'varm.PCs.V26', 'varm.PCs.V27', 'varm.PCs.V28', 'varm.PCs.V29', 'varm.PCs.V30', 'varm.PCs.V31', 'varm.PCs.V32', 'varm.PCs.V33', 'varm.PCs.V34', 'varm.PCs.V35', 'varm.PCs.V36', 'varm.PCs.V37', 'varm.PCs.V38', 'varm.PCs.V39', 'varm.PCs.V40', 'varm.PCs.V41', 'varm.PCs.V42', 'varm.PCs.V43', 'varm.PCs.V44', 'varm.PCs.V45', 'varm.PCs.V46', 'varm.PCs.V47', 'varm.PCs.V48', 'varm.PCs.V49', 'varm.PCs.V50'
        obsm: 'X_cca', 'X_pca', 'X_umap'
        layers: 'counts', 'logcounts'
        obsp: 'connectivities', 'distances'

    """
    check_r_package(["Seurat", "anndataR", "optparse", "remotes", "data.table"])

    rscript = get_paths_utils("_ReadWrite_RDS_anndataR.R")

    cmd = [
        "Rscript",
        rscript,
        "--input=" + str(path_rds),
        "--out=" + str(path_h5ad),
        "--operation=" + "read",
        "--batch_key=" + batch_key
    ]

    logger.info("Reading the RDS")
    subprocess.call(cmd)
    logger.info("Generating AnnData Object")
    adata = ad.read_h5ad(path_h5ad)

    # Connectivities
    if "snn" in adata.obsp.keys():
        adata.obsp["connectivities"] = adata.obsp["snn"].copy()
        del adata.obsp["snn"]
    if "nn" in adata.obsp.keys():
        adata.obsp["distances"] = adata.obsp["nn"].copy()
        del adata.obsp["nn"]

    # Rename orig.ident if present
    if "orig.ident" in list(adata.obs.columns):
        logger.info(f"Renaming orig.ident to {batch_key}")
        adata.obs[batch_key] = adata.obs["orig.ident"].copy()
        del adata.obs["orig.ident"]

    if "data" in adata.layers.keys():
        adata.layers["logcounts"] = adata.layers["data"].copy()
        del adata.layers["data"]
        adata.X = adata.layers["logcounts"].copy()
    else:
        adata.X  = adata.layers["counts"].copy()

    # Save the Updated Object
    adata.write(path_h5ad)
    logger.info("Done")
    return adata


def save_rds(
    path_rds: PathLike,
    batch_key: str = "batch",
    adata: ad.AnnData | None = None,
    path_h5ad: PathLike | None = None,
    out_type: Literal["seurat", "sce"] = "seurat",
) -> None:
    """Save AnnData as Seurat or SingleCellExperiment Object.

    :param path_rds: Path to save RDS Object.
    :param batch_key: Name in `obs` with batch information.
    :param adata: AnnData object
    :param path_h5ad:  Path to AnnData Object including the filename.
    :param out_type: Specify the type of object that the AnnData should be converted to.
    :return: Returns `None`. Generate an RDS file in `path_rds` containing the Seurat or SingleCellExperiment Object.

    See Also
    --------
        :func:`dotools_py.io.read_rds`: Read a SingleCellExperiment or Seurat Object save as RDS

    Example
    -------
    >>> import dotools_py as do
    >>> import os
    >>> adata = do.dt.example_10x_processed()
    >>> do.io.save_rds(path_rds="/tmp/Seurat.rds", adata=adata, out_type="seurat", batch_key="batch")
    >>> os.path.exists("/tmp/Seurat.rds")
    True

    Example (R)
    -----------

    .. code-block:: r

        seu <- readRDS("/tmp/Seurat.rds")
        seu

        Output:
            An object of class Seurat
            1851 features across 700 samples within 1 assay
            Active assay: RNA (1851 features, 191 variable features)
            2 layers present: counts, data
            3-dimensional reductions calculated: cca, pca, umap

    """
    import polars as pl

    check_r_package(["Seurat", "anndataR", "optparse", "remotes", "data.table"])

    rscript = get_paths_utils("_ReadWrite_RDS_anndataR.R")  # rscript = "/Users/david/Desktop/ICR/PycharmProjects/DOTools_py/src/dotools_py/util_scripts/_ReadWrite_RDS_anndataR.R"

    assert not (adata is not None and path_h5ad is not None), "Provide an AnnData or the path to an AnnData Object not both"
    assert out_type in ["seurat", "sce"], "Specify the object type for the RDS 'SingleCellExperiment' or 'SeuratObject''"
    object_type = "Seurat" if out_type == "seurat" else "SingleCellExperiment"

    if out_type == "seurat":
        if "counts" not in adata.layers.keys():
            raise InputError("Layer counts not found in adata.layers, but is required when out_type='seurat'")
        if "logcounts" not in adata.layers.keys():
            raise InputError("Layer logcounts not found in adata.layers, but is required when out_type='seurat'")


    tmp_path = None
    if adata is not None:  # If adata is provided, save in a tmp folder
        path_h5ad = Path("/tmp") / f"Convertion_{uuid.uuid4().hex}"
        path_h5ad.mkdir(parents=True, exist_ok=False)
        tmp_path = path_h5ad
        del adata.uns, adata.raw
        adata.write(path_h5ad / "adata.h5ad")
        path_h5ad = os.path.join(path_h5ad, 'adata.h5ad')

    input_folder = str(path_h5ad).split("/")
    input_folder = input_folder[:-1]
    input_folder = "/".join(input_folder)

    cmd = [
        "Rscript",
        rscript,
        "--input=" + str(path_h5ad),
        "--out=" + str(path_rds),
        "--type=" + object_type,
        "--operation=" + "write",
        "--batch_key=" + batch_key
    ]

    logger.info(f"Generating the {object_type}")
    subprocess.call(cmd)
    if not os.path.exists(path_rds):
        logger.warn("Error generating the RDS file")
        return None

    if tmp_path is not None:
        shutil.rmtree(tmp_path)

    # Remove tmp folder
    logger.info("Done")

    return None

