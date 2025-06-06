import os
import shutil
import subprocess
import uuid
from datetime import date
from pathlib import Path
from typing import Union

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars
import scanpy as sc
import seaborn as sns

from dotools_py import logger
from dotools_py.utils import convert_path, get_paths_utils


def _qc_vln(
    adata: ad.AnnData,
    title: str = "ViolinPlots - Quality Metrics",
    path: [str, None] = None,
    filename: str = "ViolinPlots.png",
    stats: list = ("total_counts", "n_genes", "pct_counts_mt"),
    colors: Union[str, list] = "lightsteelblue",
) -> None:
    """Violin Plots showing basic QC stats.

    Generate ViolinPlots to show the distribution of total counts, number of genes and percentage of mitochondrial genes.

    :param adata: annotated dt matrix.
    :param title: zitle of the Plot.
    :param path: path to figure folder.
    :param filename: name of the file.
    :param stats: `.obs` column name to plot.
    :param colors: colors for the violinplots.
    :return:
    """
    if isinstance(stats, tuple):
        stats = list(stats)

    assert all(col in list(adata.obs.columns) for col in stats), "column name in col_obs missing in adata.obs"
    assert len(stats) == 3, "Expected 3 variables to plot: total_counts, n_genes_by_counts, pct_counts_mt"

    if isinstance(colors, str):
        colors = [colors]
    if len(colors) == 1:
        colors = colors * 3

    fig, axs = plt.subplots(1, 3, figsize=(5, 6))
    for idx in range(3):
        vln = sns.violinplot(adata.obs[stats[idx]], ax=axs[idx], color=colors[idx])
        vln.set_xticklabels([f"Median = {np.round(np.median(adata.obs[stats[idx]]), 1)}"], fontweight="bold")
        vln.set_title("")
    plt.suptitle(title, fontsize=30, fontweight="bold")

    if path is not None:
        plt.savefig(convert_path(path) / filename, bbox_inches="tight")
    return


def _filter_quantiles(
    adata: ad.AnnData,
    low: Union[int, None] = None,
    high: Union[int, None] = None,
) -> ad.AnnData:
    """Filter cells based on total nUMI counts using quantiles.

    :param adata: annotated dt matrix
    :param low: lower quantile
    :param high: upper qauntile
    :return: annotated dt matrix
    """
    counts = adata.obs["total_counts"]
    mask = np.ones(adata.n_obs, dtype=bool)
    if low:
        mask &= counts > np.percentile(counts, low)
    if high:
        mask &= counts < np.percentile(counts, high)
    return adata[mask, :].copy()


def _run_scdblfinder(
    adata: ad.AnnData,
    batch_key: Union[str, None] = None,
) -> None:
    """Find doublets.

    The inference is performed using `scDblFinder <https://github.com/plger/scDblFinder>`_ in R.

    :param adata: annotated anndata matrix
    :param batch_key: `.obs` column name with batch information. Required if the anndata contain more than 1 sample.
    :return:
    """
    logger.info("Finding Neotypic doublets")
    rscript = get_paths_utils("_run_scDblFinder.R")
    tmpdir_path = Path("/tmp") / f"scDblFinder_{uuid.uuid4().hex}"
    tmpdir_path.mkdir(parents=True, exist_ok=False)
    adata.write(tmpdir_path / "adata_tmp.h5ad")

    logger.info("Running scDblFinder")
    cmd = ["Rscript", rscript, "--input=" + str(tmpdir_path) + "/adata_tmp.h5ad", "--out=" + str(tmpdir_path) + "/"]
    if batch_key:
        cmd = cmd["--name=" + batch_key]
    subprocess.call(cmd)

    doublets = polars.read_csv(tmpdir_path / "scDblFinder_inference.csv", infer_schema_length=0)
    doublets = doublets.to_pandas()
    doublets = doublets.set_index(adata.obs_names)  # Avoid ImplicitModificationWarning
    adata.obs[["doublet_class", "doublet_score"]] = doublets.values
    shutil.rmtree(tmpdir_path)
    return


def _normalise(
    adata: ad.AnnData,
    n_reads: int = 10_000,
    max_val: Union[float, None] = None,
    scale: bool = True
) -> None:
    """Normalise raw counts.

    The input is an unnormalise anndata object. The dt in X will be log-normalise to 10,000 reads per cell.
    The returned anndata object will contain 3 layers:
    * counts: contains the raw unnormalised counts
    * logcounts: contains the log-normalise counts
    * scaled: contained the log-normalise counts scaled
    Additionally, the log-normalise counts will also be saved under the X attribute.

    :param adata: annData object
    :param n_reads: target number of reads per cell to normalise to. (Default  is **10,000**)
    :param max_val: maximum expression value after scaling. (Default is **None**)
    :param scale: whether to scale or not the dt. (Default is **True**)
    :return: log-normalise anndata object
    """
    adata.layers["counts"] = adata.X.copy()  # Save raw counts
    sc.pp.normalize_total(adata, target_sum=n_reads)
    sc.pp.log1p(adata)
    adata.layers["logcounts"] = adata.X.copy()

    if scale:
        logger.info("Scaling dt")
        sc.pp.scale(adata, zero_center=True, max_value=max_val)
        adata.layers["scaled"] = adata.X.copy()
        adata.X = adata.layers["logcounts"].copy()
    return


def _qc_scrna(
    adata: ad.AnnData,
    ids: str,
    qc_path: Union[str, None] = None,
    batch_key: Union[str, None] = None,
    min_genes_in_cell: int = 300,
    min_cells_with_genes: int = 5,
    cut_mt: int = 5,
    min_counts: Union[int, None] = None,
    max_counts: Union[int, None] = None,
    min_genes: Union[int, None] = None,
    max_genes: Union[int, None] = None,
    low_quantile: Union[int, None] = None,
    high_quantile: Union[int, None] = None,
    include_rbs: bool = True,
    remove_doublets: bool = False,
    metrics: bool = True,
) -> ad.AnnData:
    """Basic QC.

    :param adata: annotated dt matrix.
    :param ids: id or name for the dt.
    :param qc_path: path where to save the metric and the violin plots.
    :param min_genes_in_cell: minimum number of genes in a cell.
    :param min_cells_with_genes:  minimum number of cells expressing a gene.
    :param cut_mt: maximum number of mitochondrial content for cells.
    :param min_counts: minimum number of counts per cell.
    :param max_counts: maximum number of counts per cell.
    :param min_genes: minimum number of genes per cell.
    :param max_genes: maxinum number of genes per cell.
    :param low_quantile: low quantile to filter genes and counts.
    :param high_quantile: upper quantile to filter genes and counts.
    :param include_rbs: calculate stats for ribosomal genes.
    :param remove_doublets: remove doublets.
    :param metrics: whether to generate a metrics file or not.
    :return: annotated dt matrix
    """
    # Create a metrics file
    today = date.today().strftime("%y%m%d")
    metrics_filename = f"{today}_Metrics_{ids}.xlsx"
    df = pd.DataFrame([], columns=["QC_Step", "nCells", "nFeatures", "Comments"])
    df.loc[0] = ["Input_Shape", adata.shape[0], adata.shape[1], ""]

    # Compute Metrics
    mt_gene, ribo_gene = "mt-", ("rbs", "rpl")
    qc_metrics = ["mt", "ribo"] if include_rbs else ["mt"]
    adata.var["genenames"] = adata.var_names.str.lower()  # Generalise for any gene format
    adata.var["mt"] = adata.var["genenames"].str.startswith(mt_gene)  # Annotate mitochondria genes
    adata.var["ribo"] = adata.var["genenames"].str.startswith(ribo_gene)  # Annotate mitochondria genes
    sc.pp.calculate_qc_metrics(adata, qc_vars=qc_metrics, percent_top=None, log1p=True, inplace=True, parallel=True)

    # Vln Plots showing Metrics before qc
    _qc_vln(adata, title=f"PreQC for {ids}", path=qc_path, filename=f"Vln_PreQC_{ids}.svg")

    # Step 1 -
    logger.info("Remove Cells with low number of genes")
    sc.pp.filter_cells(adata, min_genes=min_genes_in_cell, inplace=True)
    df.loc[1] = ["Rm_poor_Cells", adata.shape[0], adata.shape[1], "Remove cells with low number of genes"]

    # Step 2 -
    logger.info("Remove Genes lowly expressed")
    sc.pp.filter_genes(adata, min_cells=min_cells_with_genes, inplace=True)
    df.loc[2] = [
        "Rm_low_Genes",
        adata.shape[0],
        adata.shape[1],
        f"Remove genes express in less than {min_cells_with_genes} cells",
    ]

    # Step 3 -
    logger.info("Remove cells with high MT-content")
    adata = adata[adata.obs.pct_counts_mt < cut_mt, :].copy()
    df.loc[3] = [
        "Rm_cell_high_MT",
        adata.shape[0],
        adata.shape[1],
        f"Remove cells with >{cut_mt}% of Mitochondrial genes",
    ]

    # Step 4 -
    logger.info("Remove cells based on nUMI counts")
    assert (min_counts is None) != (low_quantile is None), "Set min_count or low_quantile"
    assert (max_counts is None) != (high_quantile is None), "Set max_count or high_quantile"

    if min_counts is not None:
        sc.pp.filter_cells(adata, min_counts=min_counts)
    if max_counts is not None:
        sc.pp.filter_cells(adata, max_counts=max_counts)
    if min_genes is not None:
        sc.pp.filter_cells(adata, min_genes=min_genes)
    if max_genes is not None:
        sc.pp.filter_cells(adata, max_genes=max_genes)

    # Apply quantile-based filtering (conditionally)
    adata = _filter_quantiles(adata, low_quantile, high_quantile)
    df.loc[4] = [
        "Rm_Cells_nFeatures",
        adata.shape[0],
        adata.shape[1],
        "Remove cells based on nUMI counts and nFeatures",
    ]

    # Step 5 -
    if remove_doublets:
        adata.layers["counts"] = adata.X.copy()  # needed for scDblFinder
        _run_scdblfinder(adata, batch_key)
        n_doublets = adata.obs["doublet_class"].value_counts()["doublet"]
        adata = adata[adata.obs["doublet_class"] == "singlet"].copy()
        logger.info(f"Remove {n_doublets} doublets")
        df.loc[5] = ["Rm_doublets", adata.shape[0], adata.shape[1], "Remove neotypic doublets"]

    # Save Metrics File
    if metrics is True:
        df_plot = df.iloc[:, :-1].melt(id_vars="QC_Step")  # Exclude comments
        fig, axs = plt.subplots(1, 1, figsize=(5, 6))  # initializes figure and plots
        bp = sns.barplot(
            df_plot,
            hue="QC_Step",
            x="value",
            y="variable",
            order=["nCells", "nFeatures"],
            hue_order=list(df["QC_Step"]),
            palette="tab20",
            ax=axs,
        )
        for container in bp.containers:
            bp.bar_label(container)
        bp.set_title("")
        bp.set_ylabel("", fontsize=18)
        bp.set_xlabel("Counts", fontsize=18)
        bp.legend(title="QC_Step", fontsize=12, frameon=False, title_fontproperties={"weight": "bold", "size": 15})
        plt.savefig(os.path.join(qc_path, f"{today}_QC_Metrics{ids}.svg"), bbox_inches="tight")

        # Save Metric File
        df.to_excel(os.path.join(qc_path, metrics_filename), index=False)
    return adata


def importer_py(
    paths: list,
    ids: list,
    metadata: Union[dict, None] = None,
    batch_key: str = "batch",
    remove_doublets: bool = True,
    min_genes_in_cell: int = 300,
    min_cells_with_genes: int = 5,
    cut_mt: int = 5,
    n_reads: int = 10_000,
    min_counts: Union[int, None] = None,
    max_counts: Union[int, None] = None,
    min_genes: Union[int, None] = None,
    max_genes: Union[int, None] = None,
    low_quantile: Union[int, None] = None,
    high_quantile: Union[int, None] = None,
) -> ad.AnnData:
    """Quality control analysis for sc/snRNA.

    The input is a list with paths to H5 files generated with CellRanger, CellBender or STARsolo and a list with
    the batch name for each sample. A dictionary with extra metadata information can be provided. The order
    should always be mainted.

    For each sample a several quality and filtering steps are applied:
        * Filter genes express in low number of cells.
        * Filter cells with low number of genes.
        * Filter cells with high mitochondrial content. Recommended to use 5% for scRNA and 3% for snRNA.
        * Filter cells based on UMI and features. There are two modes:
            * Absolute filtering: set absolute values for the maximum and minimum number of UMI and features.
            * Quantile filtering: filter the top and/or quantile.
        * Remove doublets using `scDblFinder <https://github.com/plger/scDblFinder>`_.

    An ExcelSheet with stats on how many cells and features were removed in each step, and violin plots showing the
    distribution of `total_counts`, `n_genes` and `pct_mt_content` per  cell before and after the quality control will
    be generated. These files will be saved under the folder containing the H5 files.

    After the quality control, the dt will be log-normalised and scaled. Adiitionaly, the highly variable genes and PCA
    will be calculated.

    :param paths: list with the path to the H5 files.
    :param ids: list with the batch name for each sample.
    :param metadata: dictionary with metadata information.
    :param batch_key: key in `.obs` for the batch information.
    :param remove_doublets: if set to True, neotypic doublets will be removed.
    :param min_genes_in_cell: minimum number of genes per cell.
    :param min_cells_with_genes: minimum cells expressing a genes.
    :param n_reads: target sum after normalisation per cell.
    :param cut_mt: maximum percentage of mitochondrial genes per cell.
    :param min_counts:  minimum number of counts per cell.
    :param max_counts: maximum number of counts per cell.
    :param min_genes: minimum number of genes per cell.
    :param max_genes: maximum number of genes per cell.
    :param low_quantile: low quantile to filter cells based on counts.
    :param high_quantile: upper quantile to filter cells based on counts.
    :return: annotated dt matrix of shape `n_obs` x `n_vars` with all the samples concatenated.

    Example
    -------
    >>> import dotools_py as do
    >>> paths = ['/path/sample1', '/path/sample2']
    >>> batchname = ['sample1', 'sample2']
    >>> metadata = {'condition': ['WT', 'KO'],
    ...             'age': ['3m', '3m'],
    ...             }
    >>> adata = do.pp.importer_py(paths=paths,
    ...                           ids=batchname,
    ...                           metadata=metadata,
    ...                           batch_key='batch',
    ...                           remove_doublets=True,
    ...                           min_genes_in_cell=300,
    ...                           min_cells_with_genes=5,
    ...                           n_reads=10_000,
    ...                           cut_mt=5,
    ...                           high_quantile=95,
    ...                           min_counts=500
    ...                           )
    """
    # Checks
    assert isinstance(paths, list) and isinstance(ids, list), "Please provide a list of paths and ids"
    assert len(paths) == len(ids), f"Provided {len(paths)} paths and {len(ids)} ids"

    adata_dict = {}
    for idx, path in enumerate(paths):
        # Save QC Plots in the folder with raw dt
        qc_path = convert_path("/".join(path.split("/")[:-1]))

        logger.info(f"Reading {ids[idx]}")
        try:
            adata = sc.read_10x_h5(path)  # Works for 10x and CellBender and StarSolo?
        except IsADirectoryError:
            adata = sc.read_10x_mtx(path)  # Directory with .mtx and .tsv files

        adata.var_names_make_unique()

        # Add ID and Metadata
        adata.obs[batch_key] = ids[idx]
        if metadata:
            for key, value in metadata.items():
                adata.obs[key] = adata.obs[batch_key].map(dict(zip(ids, value, strict=False)))

        # Quality Control
        adata = _qc_scrna(
            adata=adata,
            ids=ids[idx],
            batch_key=batch_key,
            qc_path=qc_path,
            metrics=True,
            min_genes_in_cell=min_genes_in_cell,
            min_cells_with_genes=min_cells_with_genes,
            cut_mt=cut_mt,
            min_counts=min_counts,
            max_counts=max_counts,
            min_genes=min_genes,
            max_genes=max_genes,
            low_quantile=low_quantile,
            high_quantile=high_quantile,
            include_rbs=True,
            remove_doublets=remove_doublets,
        )

        # Vln Plots showing Metrics before qc
        _qc_vln(adata, title=f"PostQC for {ids[idx]}", path=qc_path, filename=f"Vln_PostQC_{ids[idx]}.svg")

        adata_dict[ids[idx]] = adata

    logger.info("Concatenating samples")
    adata_concat = ad.concat(
        adata_dict.values(), label=batch_key, keys=adata_dict.keys(), join="outer", index_unique="-", fill_value=0
    )
    logger.info("Normalisation of the expression")
    _normalise(adata_concat, n_reads=n_reads, scale=True)

    logger.info("Finding Highly Variable Genes shared across samples")
    sc.pp.highly_variable_genes(adata_concat, batch_key=batch_key)

    logger.info("Run PCA")
    sc.pp.pca(adata_concat, layer='scaled')
    return adata_concat
