import os.path
import uuid
import shutil
import subprocess
from pathlib import Path
from typing import Literal

import anndata as ad
import pandas as pd
import numpy as np
from scipy.sparse import issparse

from dotools_py.utils import get_paths_utils
from dotools_py import logger


def read_rds(
    path_rds: str | Path,
    path_h5ad: str | Path,
    batch_key: str = "batch",
) -> ad.AnnData | None:
    """Read Rds object with Seurat or SingleCellExperiment Object.

    .. note::
        When reading an RDS Object with counts and logcounts data, the counts will be returned in the
        `X` attribute, while the logcounts are returned as a layer.

    :param path_rds: Path to RDS file with SingleCellExperiment or SeuratObject.
    :param path_h5ad: Path to save AnnData Object.
    :param batch_key: Name in `obs` to save batch information.
    :return: Returns an `AnnData` Object or `None`. The AnnData can also be saved under `path_adata`.

    See Also
    --------
        :func:`dotools_py.utility.save_rds`: Save an AnnData as  SingleCellExperiment or Seurat Object

    Example
    -------
    >>> import dotools_py as do
    >>> path_seurat = "/tmp/Seurat.rds"
    >>> path_adata = "/tmp/adata.h5ad"
    >>> adata = do.utility.read_rds(path_rds=path_seurat, path_h5ad=path_adata)
    >>> adata
    AnnData object with n_obs × n_vars = 700 × 1851
        obs: 'nCount_originalexp', 'nFeature_originalexp', 'batch', 'condition', 'n_genes_by_counts',
             'log1p_n_genes_by_counts', 'total_counts', 'log1p_total_counts', 'total_counts_mt',
             'log1p_total_counts_mt', 'pct_counts_mt', 'total_counts_ribo', 'log1p_total_counts_ribo',
             'pct_counts_ribo', 'n_genes', 'n_counts', 'doublet_class', 'doublet_score', 'leiden', 'cell_type',
             'autoAnnot', 'celltypist_conf_score', 'annotation', 'annotation_recluster', 'ident'
        var: 'highly_variable'
        uns: 'X_name'
        obsm: 'X_cca', 'X_pca', 'X_umap'
        layers: 'logcounts', 'counts'
        obsp: 'connectivities', 'distances'

    """
    import polars as pl
    import scipy.sparse as sp

    rscript = get_paths_utils("_ReadWrite_RDS.R")

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

    # Transfer missing information
    input_folder = str(path_rds).split("/")
    input_folder = input_folder[:-1]
    input_folder = "/".join(input_folder)

    # Variable Features
    try:
        hvg = pd.read_csv(os.path.join(input_folder, "VariableFeatures.csv")).set_index("Unnamed: 0")
        logger.info("Transferring HVGs")
        hvg_bool = [True if g in list(hvg["hvg"]) else False for g in adata.var_names]
        adata.var["highly_variable"] = hvg_bool
    except FileNotFoundError as e:
        logger.info(f"Problem transferring HVGs, {e}")

    # Connectivities
    try:
        connectivities = pl.read_csv(os.path.join(input_folder, "Connectivities.csv"), has_header=True,
                                     dtypes={bc: pl.Float64 for bc in adata.obs_names})
        connectivities = connectivities.to_pandas()
        if "" in connectivities.columns:
            del connectivities[""]  # Index
        if connectivities.shape[0] == connectivities.shape[1]:
            logger.info("Transferring connectivities")
            adata.obsp["connectivities"] = sp.csr_matrix(connectivities.values)
        else:
            logger.info("Problem transferring connectivities")
    except FileNotFoundError as e:
        logger.info(f"Problem transferring connectivities, {e}")

    # Distances
    try:
        distances = pl.read_csv(os.path.join(input_folder, "Distances.csv"), has_header=True,
                                dtypes={bc: pl.Float64 for bc in adata.obs_names})
        distances = distances.to_pandas()
        if "" in distances.columns:
            del distances[""]  # Index
        if distances.shape[0] == distances.shape[1]:
            logger.info("Transferring neighbor distances")
            adata.obsp["distances"] = sp.csr_matrix(distances.values)
        else:
            logger.info("Problem transferring neighbor distances")
    except FileNotFoundError as e:
        logger.info(f"Problem transferring distances, {e}")

    # Rename reductions
    logger.info("Renaming reductions")
    obsm_keys = [key for key in adata.obsm.keys()]
    for key in obsm_keys:
        new_key = "X_" + key.lower().replace(".", "_").replace("-", "_")
        adata.obsm[new_key] = adata.obsm[key].values
        del adata.obsm[key]

    # Rename orig.ident if present
    if "orig.ident" in list(adata.obs.columns):
        logger.info(f"Renaming orig.ident to {batch_key}")
        adata.obs[batch_key] = adata.obs["orig.ident"].copy()
        del adata.obs["orig.ident"]

    # Default is X with raw counts
    if issparse(adata.X):
        if all(np.array(adata.X.data) % 1 == 0):
            adata.layers["counts"] = adata.X.copy()
    else:
        if all(adata.X.flatten() % 1 == 0):
            adata.layers["counts"] = adata.X.copy()

    # Remove all intermediate files
    for f in ["Distances.csv", "Connectivities.csv", "VariableFeatures.csv"]:
        try:
            os.remove(os.path.join(input_folder, f))
        except FileNotFoundError:
            continue
    logger.info("Done")
    return adata


def save_rds(
    path_rds: str,
    batch_key: str = "batch",
    adata: ad.AnnData = None,
    path_h5ad: str = None,
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
        :func:`dotools_py.utility.read_rds`: Read a SingleCellExperiment or Seurat Object save as RDS

    Example
    -------
    >>> import dotools_py as do
    >>> import os
    >>> adata = do.dt.example_10x_processed()
    >>> do.utility.save_rds(path_rds="/tmp/Seurat.rds", adata=adata, object_type="seurat", batch_key="batch")
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

    rscript = get_paths_utils("_ReadWrite_RDS.R")

    assert not (adata is not None and path_h5ad is not None), "Provide an AnnData or the path to an AnnData Object not both"
    assert out_type in ["seurat", "sce"], "Specify the object type for the RDS 'SingleCellExperiment' or 'SeuratObject''"
    object_type = "SeuratObject" if out_type == "seurat" else "sce"

    tmp_path = None
    if adata is not None:  # If adata is provided, save in a tmp folder
        path_h5ad = Path("/tmp") / f"Convertion_{uuid.uuid4().hex}"
        path_h5ad.mkdir(parents=True, exist_ok=False)
        tmp_path = path_h5ad
        del adata.uns, adata.raw
        adata.write(path_h5ad / "adata.h5ad")
        path_h5ad = os.path.join(path_h5ad, 'adata.h5ad')
    else:
        adata = ad.read_h5ad(path_h5ad)

    input_folder = str(path_h5ad).split("/")
    input_folder = input_folder[:-1]
    input_folder = "/".join(input_folder)

    if "distances" in adata.obsp:
        df = pl.DataFrame(adata.obsp["distances"].toarray())
        df.columns = adata.obs_names
        df.write_csv(os.path.join(input_folder, "Distances.csv"))
    if "connectivities" in adata.obsp:
        # connectivities --> snn
        df = pl.DataFrame(adata.obsp["connectivities"].toarray())
        df.columns = adata.obs_names
        df.write_csv(os.path.join(input_folder, "Connectivities.csv"))
    if "highly_variable" in adata.var.columns:
        # HVGs
        hvg = adata.var.highly_variable
        hvg.to_csv(os.path.join(input_folder, "VariableFeatures.csv"))

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

    for f in ["Distances.csv", "Connectivities.csv", "VariableFeatures.csv"]:
        try:
            os.remove(os.path.join(input_folder, f))
        except FileNotFoundError:
            continue
    if tmp_path is not None:
        shutil.rmtree(tmp_path)

    # Remove tmp folder
    logger.info("Done")

    return None

