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



def create_report(
    log_file: str | Path,
) -> None:
    """Create a report file.

    This function takes a log_file that should have been set at the beginning of the session with
    `dotools_py.settings.set.set_kernel_logger` and add information regarding the session such as
    the machine characteristics and the version of the packages.

    :param log_file: Path to the log file
    :return: Returns None. The log file is updated with session information.

    Examples
    --------
    >>> import dotools_py as do
    >>> do.settings.set_kernel_logger('./History.log', overwrite=True)
    >>> adata = do.dt.example_10x_processed()
    >>> adata
    >>> do.utility.create_report("./History.log")
    >>> print(open("History.log").read())
    [CODE 2026-01-22 13:59:28.904757]
    >>> adata = do.dt.example_10x_processed()
    [CODE 2026-01-22 13:59:29.617186]
    >>> adata
    [OUTPUT 2026-01-22 13:59:29.619246]
    AnnData object with n_obs × n_vars = 700 × 1851
        obs: 'batch', 'condition', 'n_genes_by_counts', 'log1p_n_genes_by_counts', 'total_counts', 'log1p_total_counts', 'total_counts_mt', 'log1p_total_counts_mt', 'pct_counts_mt', 'total_counts_ribo', 'log1p_total_counts_ribo', 'pct_counts_ribo', 'n_genes', 'n_counts', 'doublet_class', 'doublet_score', 'leiden', 'cell_type', 'autoAnnot', 'celltypist_conf_score', 'annotation', 'annotation_recluster'
        var: 'mean', 'std', 'highly_variable', 'means', 'dispersions', 'dispersions_norm', 'highly_variable_nbatches', 'highly_variable_intersection'
        uns: 'annotation_colors', 'annotation_recluster_colors', 'batch_colors', 'hvg', 'leiden', 'leiden_colors', 'log1p', 'neighbors', 'pca', 'umap'
        obsm: 'X_CCA', 'X_pca', 'X_umap'
        varm: 'PCs'
        layers: 'counts', 'logcounts'
        obsp: 'connectivities', 'distances'
    ==================== Session Information ====================
    OS:macOS-26.2-arm64-arm-64bit
    Machine: arm64
    Processor: arm
    CPU cores (physical): 10
    CPU cores (logical): 10
    Total RAM (GB): 16.0
    Python version: 3.11.13
    -----
    anndata     0.11.4
    dotools_py  0.0.1
    pandas      2.3.2
    platform    1.0.8
    -----
    Cython                      3.1.4
    IPython                     9.5.0
    PIL                         11.3.0
    adjustText                  1.3.0
    altair                      6.0.0
    argparse                    1.1
    arrow                       1.3.0
    attr                        25.3.0
    attrs                       25.3.0
    beartype                    0.22.8
    charset_normalizer          3.4.3
    cloudpickle                 3.1.1
    comm                        0.2.3
    coverage                    7.11.0
    csv                         1.0
    ctypes                      1.1.0
    cycler                      0.12.1
    cython                      3.1.4
    dask                        2024.11.2
    dateutil                    2.9.0.post0
    decimal                     1.70
    decorator                   5.2.1
    defusedxml                  0.7.1
    deprecated                  1.2.18
    executing                   2.2.1
    h5py                        3.14.0
    idna                        3.10
    igraph                      0.11.9
    ipaddress                   1.0
    ipywidgets                  8.1.7
    jedi                        0.19.2
    jinja2                      3.1.6
    joblib                      1.5.2
    json                        2.0.9
    jsonpointer                 3.0.0
    jsonschema                  4.25.1
    kiwisolver                  1.4.9
    lark                        1.2.2
    leidenalg                   0.10.2
    llvmlite                    0.45.0
    logging                     0.5.1.2
    markupsafe                  3.0.2
    marshal                     4
    matplotlib                  3.10.6
    msgpack                     1.1.2
    narwhals                    2.5.0
    natsort                     8.4.0
    numba                       0.62.0
    numcodecs                   0.15.1
    numpy                       2.3.3
    packaging                   25.0
    parso                       0.8.5
    patsy                       1.0.1
    polars                      1.33.1
    prompt_toolkit              3.0.52
    psutil                      7.1.0
    pure_eval                   0.2.3
    pyarrow                     21.0.0
    pydot                       4.0.1
    pygments                    2.19.2
    pyparsing                   3.2.4
    pytz                        2025.2
    re                          2.2.1
    rfc3339_validator           0.1.4
    rfc3986_validator           0.1.1
    scanpy                      1.11.4
    scipy                       1.15.3
    seaborn                     0.13.2
    session_info                v1.0.1
    six                         1.17.0
    sklearn                     1.7.2
    socketserver                0.4
    sparse                      0.17.0
    sqlite3                     2.6.0
    stack_data                  0.6.3
    statsmodels                 0.14.5
    stdlib_list                 0.11.1
    sys                         3.11.13 (main, Jun  5 2025, 08:21:08) [Clang 14.0.6 ]
    tarfile                     0.9.0
    texttable                   1.7.0
    threadpoolctl               3.6.0
    tlz                         1.0.0
    toolz                       1.0.0
    torch                       2.8.0
    tqdm                        4.67.1
    traitlets                   5.14.3
    wcwidth                     0.2.13
    wrapt                       1.17.3
    yaml                        6.0.2
    zarr                        2.18.7
    zlib                        1.0
    -----
    Python 3.11.13 (main, Jun  5 2025, 08:21:08) [Clang 14.0.6 ]
    macOS-26.2-arm64-arm-64bit
    10 logical CPU cores, arm
    -----
    Session information updated at 2026-01-22 13:59
    =============================================================
    """
    import os
    import session_info
    import platform
    import psutil
    import io
    from contextlib import redirect_stdout

    if not os.path.exists(log_file):
        raise FileNotFoundError(f"{log_file} does not exist. Set kernel logger with do.settings.set_kernel_logger()")
    else:
        with open(log_file, "a") as f:
            f.write("\n\n")
            f.write("==================== Session Information ====================")
            f.write("\n")
            f.write("OS:" + platform.platform() + "\n")
            f.write("Machine: " + platform.machine()+ "\n")
            f.write("Processor: " + platform.processor()+ "\n")
            f.write("CPU cores (physical): " + str(psutil.cpu_count(logical=False))+ "\n")
            f.write("CPU cores (logical): " + str(psutil.cpu_count(logical=True))+ "\n")
            f.write("Total RAM (GB): " + str(round(psutil.virtual_memory().total / (1024 ** 3), 2))+ "\n")
            f.write("Python version: " + platform.python_version()+  "\n")
            f.write("\n\n")
            # Capture session_info.show() output
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                session_info.show(
                    na=False,
                    os=True,
                    cpu=True,
                    excludes=["backports"],
                    std_lib=True,
                    dependencies=True,
                    html=False,
                    jupyter=None,
                )
            f.write(buffer.getvalue())
            f.write("\n")
            f.write("=============================================================")
    return None







