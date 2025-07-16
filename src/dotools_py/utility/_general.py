import os.path
from pathlib import Path
import uuid
import subprocess
from typing import Literal

import anndata as ad

from dotools_py import logger
from dotools_py.utils import get_paths_utils

def free_memory() -> None:
    """Garbage collector.

    :return:
    """
    import ctypes
    import gc

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass
    return None


def transfer_labels(
    adata_original: ad.AnnData,
    adata_subset: ad.AnnData,
    original_key: str,
    subset_key: str,
    original_labels: list,
    copy: bool = False,
) -> ad.AnnData | None:
    """Transfer annotation from a subset AnnData to an AnnData.

    :param adata_original: original AnnData.
    :param adata_subset: subsetted AnnData.
    :param original_key: obs column name in the original AnnData where new labels are added.
    :param subset_key: obs column name in the subsetted AnnData with the new labels.
    :param original_labels: list of labels in `original_key` to replace.
    :param copy: if set to True, returns the updated anndata
    :return: If `copy` is set to `True`, returns the original AnnData with the updated labels, otherwise returns `None`.
             The  original_labels in original_key will be updated with the labels in subset_key.
    """
    if copy:
        adata_original = adata_original.copy()
        adata_subset = adata_subset.copy()
    assert adata_subset.n_obs < adata_original.n_obs, "adata_subset is not a subset of adata_original"

    labels_original = [original_labels] if isinstance(original_labels, str) else original_labels
    adata_original.obs[original_key] = adata_original.obs[original_key].astype(str)
    adata_original.obs[original_key] = adata_original.obs[original_key].where(
        ~adata_original.obs[original_key].isin(labels_original),
        adata_original.obs.index.map(adata_subset.obs[subset_key]),
    )

    if copy:
        return adata_original
    else:
        return None


def read_rds(path_rds: str, path_adata: str) -> ad.AnnData:
    """Read Rds object with Seurat or SingleCellExperiment Object.

    :param path_rds: path to RDS file with SingleCellExperiment or SeuratObject.
    :param path_adata: path to save AnnData Object including the filename.
    :return: Returns an `AnnData` Object. The AnnData is also saved under `path_adata`.

    See Also
    --------
        :func:`dotools_py.utility.save_rds`: Save an AnnData as  SingleCellExperiment or Seurat Object

    Example
    -------
    >>> import dotools_py as do
    >>> path_rds = "/tmp/Seurat.rds"
    >>> path_adata = "/tmp/adata.h5ad"
    >>> adata = do.utility.read_rds(path_rds=path_rds, path_adata=path_adata)
    >>> adata
    AnnData object with n_obs × n_vars = 700 × 1851
        obs: 'orig.ident', 'nCount_originalexp', 'nFeature_originalexp', 'batch', 'condition', 'n_genes_by_counts', 'log1p_n_genes_by_counts', 'total_counts', 'log1p_total_counts', 'total_counts_mt', 'log1p_total_counts_mt', 'pct_counts_mt', 'total_counts_ribo', 'log1p_total_counts_ribo', 'pct_counts_ribo', 'n_genes', 'n_counts', 'doublet_class', 'doublet_score', 'leiden', 'cell_type', 'autoAnnot', 'celltypist_conf_score', 'annotation', 'annotation_recluster', 'ident'
        uns: 'X_name'
        obsm: 'X_CCA', 'X_PCA', 'X_UMAP'
        layers: 'logcounts'

    """
    rscript = get_paths_utils("_ReadWrite_RDS.R")

    cmd = [
        "Rscript",
        rscript,
        "--input=" + str(path_rds),
        "--out=" + str(path_adata),
        "--operation=" + "read",
    ]

    logger.info("Reading the RDS")
    subprocess.call(cmd)
    logger.info("Generating AnnData Object")
    adata = ad.read_h5ad(path_adata)
    return  adata

def save_rds(
    path_rds: str,
    adata: ad.AnnData = None,
    path_adata: str = None,
    object_type: Literal["SingleCellExperiment", "SeuratObject"] = "SeuratObject",
) -> None:
    """Read Rds object with Seurat or SingleCellExperiment Object.

    :param path_rds: Path to save RDS Object including filename.
    :param adata: AnnData object
    :param path_adata: Path to AnnData Object including the filename.
    :param object_type: Specify the type of object that the AnnData should be converted to.
    :return: Generate an RDS file in `path_rds` containing the Seurat or SingleCellExperiment Object.

    See Also
    --------
        :func:`dotools_py.utility.read_rds`: Read a SingleCellExperiment or Seurat Object save as RDS

    Example
    -------
    >>> import dotools_py as do
    >>> import os
    >>> adata = do.dt.example_10x_processed()
    >>> do.utility.save_rds(path_rds="/tmp/Seurat.rds", adata=adata, object_type="SeuratObject")
    >>> os.path.exists("/tmp/Seurat.rds")
    True

    """
    rscript = get_paths_utils("_ReadWrite_RDS.R")

    assert not (adata is not None and path_adata is not None), "Provide an AnnData or the path to an AnnData Object not both"
    assert object_type in ["SeuratObject", "SingleCellExperiment"], "Specify the object type for the RDS 'SingleCellExperiment' or 'SeuratObject''"

    if adata is not None:
        path_adata = Path("/tmp") / f"Convertion_{uuid.uuid4().hex}"
        path_adata.mkdir(parents=True, exist_ok=False)
        del adata.uns, adata.raw
        adata.write(path_adata / "adata.h5ad")
        path_adata = os.path.join(path_adata, 'adata.h5ad')

    cmd = [
        "Rscript",
        rscript,
        "--input=" + str(path_adata),
        "--out=" + str(path_rds) ,
        "--type=" + object_type,
        "--operation=" + "write",
    ]

    logger.info(f"Generating the {object_type}")
    subprocess.call(cmd)
    logger.info("Done")
    return None

