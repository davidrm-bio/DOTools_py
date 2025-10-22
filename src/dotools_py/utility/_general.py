import os.path
from pathlib import Path
import platform
from typing import Literal

import anndata as ad
import pandas as pd

from dotools_py.utility._language import RDSConverter

HERE = Path(__file__).parent


def free_memory() -> None:
    """Garbage collector.

    :return:
    """
    import ctypes
    import gc

    gc.collect()

    system = platform.system()

    if system == "Linux":
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    else:
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


def read_rds(
    path_rds: str | Path,
    path_h5ad: str | Path = None,
    filename_h5ad: str = "adata.h5ad",
    batch_key: str = "batch",
    rds_batch_key: str = "orig.ident",
) -> ad.AnnData | None:
    """Read Rds object with Seurat or SingleCellExperiment Object.

    .. note::
        When reading an RDS Object with counts and logcounts data, the counts will be returned in the
        `X` attribute, while the logcounts are returned as a layer.

    :param path_rds: Path to RDS file with SingleCellExperiment or SeuratObject.
    :param path_h5ad: Path to save AnnData Object.
    :param filename_h5ad: Name of the AnnData file.
    :param batch_key: Name in `obs` to save batch information.
    :param rds_batch_key: Name in the metadata column of the Seurat Object to save the batch information.
    :return: Returns an `AnnData` Object or `None`. The AnnData can also be saved under `path_adata`.

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
    converter = RDSConverter(input_obj=path_rds,
                             out_obj="anndata",
                             batch_key=batch_key,
                             rds_batch_key=rds_batch_key,
                             path=None,
                             filename=None,
                             get_anndata=True
                             )
    adata = converter.to_h5ad()

    if path_h5ad is not None:
        adata.write(Path(path_h5ad) / filename_h5ad)
    return adata


def save_rds(
    path_rds: str,
    filename_rds: str,
    batch_key: str = "batch",
    rds_batch_key: str = "orig.ident",
    adata: ad.AnnData = None,
    path_adata: str = None,
    out_type: Literal["seurat", "sce"] = "seurat",
) -> None:
    """Save AnnData as Seurat or SingleCellExperiment Object.

    :param path_rds: Path to save RDS Object.
    :param filename_rds: Name of the RDS file.
    :param batch_key: Name in `obs` with batch information.
    :param rds_batch_key: Name in the metadata column of the Seurat Object to save the batch information.
    :param adata: AnnData object
    :param path_adata:  Path to AnnData Object including the filename.
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

    if adata is None:
        adata = ad.read_h5ad(path_adata)

    converter = RDSConverter(input_obj=adata,
                             out_obj=out_type,
                             batch_key=batch_key,
                             rds_batch_key=rds_batch_key,
                             path=path_rds,
                             filename=filename_rds,
                             get_anndata=False
                             )
    converter.to_rds()
    return None


def add_gene_metadata(
    data: ad.AnnData | pd.DataFrame,
    gene_key: str,
    species: Literal["mouse", "human"] = "mouse"
) -> ad.AnnData | pd.DataFrame:
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
    import gzip
    import pickle

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
        genes = list(data_copy.var_names) if gene_key == "var_names" else data_copy.var[gene_key].tolist()
        data_copy.var["biotype"] = [database[g]["gene_type"] if g in database else "NaN" for g in genes]
        data_copy.var["locations"] = [",".join(database[g]["locations"]) if g in database else "NaN" for g in genes]
        data_copy.var["gene_id"] = [database[g]["gene_id"] if g in database else "NaN" for g in genes]
    else:
        raise Exception("Not a valid input, provide a DataFrame or AnnData")

    return data_copy
