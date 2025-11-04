import os.path
from pathlib import Path
import platform
from typing import Literal

import anndata as ad
import pandas as pd


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
