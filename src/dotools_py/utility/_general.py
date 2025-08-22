import os.path
from pathlib import Path
import uuid
import subprocess
from typing import Literal, Union
import gzip
import pickle

import anndata as ad
import pandas as pd
import polars as pl
import scipy.sparse as sp

from dotools_py import logger
from dotools_py.utils import get_paths_utils

HERE = Path(__file__).parent


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


def read_rds(path_rds: str | Path, path_adata: str | Path, batch_key: str = 'batch') -> ad.AnnData:
    """Read Rds object with Seurat or SingleCellExperiment Object.

    .. note::
        When reading an RDS Object with counts and logcounts data, the counts will be returned in the
        `X` attribute, while the logcounts are returned as a layer.

    :param path_rds: path to RDS file with SingleCellExperiment or SeuratObject.
    :param path_adata: path to save AnnData Object including the filename.
    :param batch_key: name in `obs` to save batch information.
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
    rscript = get_paths_utils("_ReadWrite_RDS.R")

    cmd = [
        "Rscript",
        rscript,
        "--input=" + str(path_rds),
        "--out=" + str(path_adata),
        "--operation=" + "read",
        "--batch_key=" + batch_key
    ]

    logger.info("Reading the RDS")
    subprocess.call(cmd)
    logger.info("Generating AnnData Object")
    adata = ad.read_h5ad(path_adata)

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
        connectivities = pl.read_csv(os.path.join(input_folder, "Connectivities.csv"), has_header=True, dtypes={bc: pl.Float64 for bc in adata.obs_names})
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
        distances = pl.read_csv(os.path.join(input_folder, "Distances.csv"), has_header=True, dtypes={bc: pl.Float64 for bc in adata.obs_names})
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
    if all(adata.X.data % 1 == 0):
        adata.layers["counts"] = adata.X.copy()

    # Remove all intermediate files
    for f in ["Distances.csv", "Connectivities.csv", "VariableFeatures.csv" ]:
        try:
            os.remove(os.path.join(input_folder, f))
        except FileNotFoundError:
            continue
    logger.info("Done")
    return  adata


def save_rds(
    path_rds: str,
    batch_key: str,
    adata: ad.AnnData = None,
    path_adata: str = None,
    object_type: Literal["SingleCellExperiment", "SeuratObject"] = "SeuratObject",
) -> None:
    """Save AnnData as Seurat or SingleCellExperiment Object.

    :param path_rds: Path to save RDS Object including filename.
    :param batch_key: Name in `obs` with batch information.
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
    >>> do.utility.save_rds(path_rds="/tmp/Seurat.rds", adata=adata, object_type="SeuratObject", batch_key="batch")
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
            3 dimensional reductions calculated: cca, pca, umap

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
    else:
        adata = ad.read_h5ad(path_adata)

    # Save intermediate files
    input_folder = str(path_adata).split("/")
    input_folder = input_folder[:-1]
    input_folder = "/".join(input_folder)

    # distances --> nn
    if "distances" in adata.obsp:
        df = pl.DataFrame(adata.obsp["distances"].toarray())
        df.columns = adata.obs_names
        df.write_csv(os.path.join(input_folder,  "Distances.csv"))
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
        "--input=" + str(path_adata),
        "--out=" + str(path_rds) ,
        "--type=" + object_type,
        "--operation=" + "write",
        "--batch_key=" + batch_key
    ]

    logger.info(f"Generating the {object_type}")
    subprocess.call(cmd)


    # Remove all intermediate files
    for f in ["Distances.csv", "Connectivities.csv","VariableFeatures.csv"]:
        try:
            os.remove(os.path.join(input_folder, f))
        except FileNotFoundError:
            continue

    logger.info("Done")
    return None


def add_gene_metadata(data: Union[pd.DataFrame, ad.AnnData],
                      gene_key: str,
                      species: Literal["mouse", "human"] = "mouse"
                      ) -> Union[pd.DataFrame, ad.AnnData]:
    """Add gene metadata to AnnData or DataFrame.

    Add gene metadata obtained from the GTF or Uniprot-database. This information includes,
    the gene biotype (e.g., protein-coding, lncRNA, etc.); the ENSEMBL gene ID and the subcellular location.

    :param data:  Annotated data matrix or pandas dataframe with for example results from differential gene expression analysis.
    :param gene_key: name of the key with gene names. If an AnnData is provided the .var name column name with gene names. If the gene names are in
                     `var_names`, specify `var_names`.
    :param species: the input species.
    :return:  Returns a dataframe or AnnData object. Three new columns will be set: `biotype`, `locations` and `gene_id`.

    Examples
    --------

    >>> import dotools_py as do
    >>> # AnnData Input
    >>> adata = do.dt.example_10x_processed()
    >>> adata = add_gene_metadata(adata, "var_names", "human")
    >>> adata.var[["biotype", "gene_id", "locations"]].head(5)
                           biotype          gene_id                locations
    ATP2A1-AS1          lncRNA  ENSG00000260442  Unreview status Uniprot
    STK17A      protein_coding  ENSG00000164543                  nucleus
    C19orf18    protein_coding  ENSG00000177025                 membrane
    TPP2        protein_coding  ENSG00000134900        nucleus,cytoplasm
    MFSD1       protein_coding  ENSG00000118855       membrane,cytoplasm
    >>>
    >>> # Dataframe Input
    >>> df = pd.DataFrame(["Acta2", "Tagln", "Ptprc", "Vcam1"], columns=["genes"])
    >>> df = add_gene_metadata(df, "genes")
    >>> df.head()
           genes         biotype          locations             gene_id
    0  Acta2  protein_coding          cytoplasm  ENSMUSG00000035783
    1  Tagln  protein_coding          cytoplasm  ENSMUSG00000032085
    2  Ptprc  protein_coding           membrane  ENSMUSG00000026395
    3  Vcam1  protein_coding  secreted,membrane  ENSMUSG00000027962


    """
    data_copy = data.copy()  # Changes will not be inplace

    assert species in ["mouse", "human"], "Not a valid species: use mouse or human"
    file = "MusMusculus_GeneMetadata.pickle.gz" if species == "mouse" else "MusMusculus_GeneMetadata.pickle.gz"
    with gzip.open(os.path.join(HERE, file), "rb") as pickle_file:
        database = pickle.load(pickle_file)

    if isinstance(data, pd.DataFrame):
        genes = data_copy[gene_key].tolist()
        data_copy["biotype"] = [database[g]["gene_type"] if g in database else "NaN" for g in genes]
        data_copy["locations"] = [",".join(database[g]["locations"]) if g in database else "NaN" for g in genes]
        data_copy["gene_id"] = [database[g]["gene_id"] if g in database else "NaN" for g in genes]
    elif isinstance(data_copy, ad.AnnData):
        genes = list(data_copy.var_names)  if gene_key == "var_names" else data_copy.var[gene_key].tolist()
        data_copy.var["biotype"] = [database[g]["gene_biotype"] if g in database else "NaN" for g in genes]
        data_copy.var["locations"] = [",".join(database[g]["locations"]) if g in database else "NaN" for g in genes]
        data_copy.var["gene_id"] = [database[g]["gene_id"] if g in database else "NaN" for g in genes]
    else:
        raise Exception("Not a valid input, provide a DataFrame or AnnData")

    return data_copy
