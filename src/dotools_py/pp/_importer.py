import os
import shutil
import subprocess
import uuid
from datetime import date
from pathlib import Path
from typing import Literal

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dotools_py import logger
from dotools_py.utils import convert_path, get_paths_utils


def _qc_vln(
    adata: ad.AnnData,
    title: str = "ViolinPlots - Quality Metrics",
    path: str | None = None,
    filename: str = "ViolinPlots.png",
    stats: list = ("total_counts", "n_genes_by_counts", "pct_counts_mt"),
    colors: str | list = "lightsteelblue",
) -> None:
    """Violin Plots showing basic QC stats.

    Generate ViolinPlots to show the distribution of total counts, number of genes and percentage of
    mitochondrial genes.

    :param adata: annotated dt matrix.
    :param title: title of the Plot.
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

    fig, axs = plt.subplots(1, 3, figsize=(10, 6))
    for idx in range(3):
        vln = sns.violinplot(adata.obs[stats[idx]], ax=axs[idx], color=colors[idx])
        vln.set_xticklabels([f"Median = {np.floor(np.median(adata.obs[stats[idx]]))}"], fontweight="bold")
        vln.set_title("")
    plt.suptitle(title, fontsize=30, fontweight="bold")

    if path is not None:
        plt.savefig(convert_path(path) / filename, bbox_inches="tight")
        plt.close(fig)
    else:
        return


def _filter_quantiles(
    adata: ad.AnnData,
    low: int | None = None,
    high: int | None = None,
) -> ad.AnnData:
    """Filter cells based on total nUMI counts using quantiles.

    :param adata: annotated dt matrix
    :param low: lower quantile
    :param high: upper quantile
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
    batch_key: str | None = None,
) -> None:
    """Find doublets.

    The inference is performed using `scDblFinder <https://github.com/plger/scDblFinder>`_ in R.

    :param adata: annotated anndata matrix
    :param batch_key: `.obs` column name with batch information. Required if the anndata contain more than 1 sample.
    :return:
    """
    import polars

    logger.info("Finding Neotypic doublets")
    rscript = get_paths_utils("_run_scDblFinder.R")
    tmpdir_path = Path("/tmp") / f"scDblFinder_{uuid.uuid4().hex}"
    tmpdir_path.mkdir(parents=True, exist_ok=False)
    adata.write(tmpdir_path / "adata_tmp.h5ad")

    logger.info("Running scDblFinder")
    cmd = ["Rscript", rscript, "--input=" + str(tmpdir_path) + "/adata_tmp.h5ad", "--out=" + str(tmpdir_path) + "/"]
    if batch_key:
        cmd += ["--name=" + batch_key]
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
    max_val: float | None = None,
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
    import scanpy as sc
    adata.layers["counts"] = adata.X.copy()  # Save raw counts
    sc.pp.normalize_total(adata, target_sum=n_reads)
    sc.pp.log1p(adata)
    adata.layers["logcounts"] = adata.X.copy()

    if scale:
        logger.info("Scaling data")
        sc.pp.scale(adata, zero_center=True, max_value=max_val)
        adata.layers["scaled"] = adata.X.copy()
        adata.X = adata.layers["logcounts"].copy()
    return


def _qc_scrna(
    adata: ad.AnnData,
    ids: str,
    qc_path: str | Path | None = None,
    batch_key: str | None = None,
    min_genes_in_cell: int = 300,
    min_cells_with_genes: int = 5,
    cut_mt: int = 5,
    min_counts: int | None = None,
    max_counts: int | None = None,
    min_genes: int | None = None,
    max_genes: int | None = None,
    low_quantile: int | None = None,
    high_quantile: int | None = None,
    include_rbs: bool = True,
    remove_doublets: bool = False,
    doublet_tool: str = "scDblFinder",
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
    :param max_genes: maximum number of genes per cell.
    :param low_quantile: low quantile to filter genes and counts.
    :param high_quantile: upper quantile to filter genes and counts.
    :param include_rbs: calculate stats for ribosomal genes.
    :param remove_doublets: remove doublets.
    :param doublet_tool: doublet tool to use. Available scDblFinder, Scrublet and DoubletDetection.
    :param metrics: whether to generate a metrics file or not.
    :return: annotated dt matrix
    """
    import doubletdetection
    import scanpy as sc

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
    df.loc[1] = ["Rm_Cells_lowGenes", adata.shape[0], adata.shape[1], f"Remove cells with <{min_genes_in_cell} genes"]

    # Step 2 -
    logger.info("Remove Genes lowly expressed")
    sc.pp.filter_genes(adata, min_cells=min_cells_with_genes, inplace=True)
    df.loc[2] = [
        "Rm_Genes_lowCells",
        adata.shape[0],
        adata.shape[1],
        f"Remove genes express in less than {min_cells_with_genes} cells",
    ]

    # Step 3 -
    logger.info("Remove cells with high MT-content")
    adata = adata[adata.obs.pct_counts_mt < cut_mt, :].copy()
    df.loc[3] = [
        "Rm_Cell_HighMT",
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
        "Rm_Cells_nUMI_nGenes",
        adata.shape[0],
        adata.shape[1],
        f"Remove cells based on nUMI counts[Absolute (Min/Max): {min_counts}/{max_counts}, "
        f"Quantile (low/high): {low_quantile}/{high_quantile}] and nFeatures [Absolute (Min/Max): "
        f"{min_genes}/{max_genes}]",
    ]

    # Step 5 -
    if remove_doublets:
        if doublet_tool == "scDblFinder":
            adata.layers["counts"] = adata.X.copy()  # needed for scDblFinder
            _run_scdblfinder(adata, batch_key)
        elif doublet_tool == "Scrublet":
            sc.pp.scrublet(adata)
            adata.obs["doublet_class"] = adata.obs["predicted_doublet"].map({False: "singlet", True: "doublet"})
            del adata.obs["predicted_doublet"]
        elif doublet_tool == "DoubletDetection":
            clf = doubletdetection.BoostClassifier(
                n_iters=15,
                clustering_algorithm="leiden",
                standard_scaling=True,
                pseudocount=0.1,
                verbose=False,
                n_jobs=-1,
            )
            doublets = clf.fit(adata.X).predict()
            doublet_score = clf.doublet_score()
            mapped = np.full(doublets.shape, "singlet", dtype=object)
            mapped[doublets == 1.0] = "doublet"
            adata.obs["doublet_class"] = pd.Categorical(mapped, categories=["singlet", "doublet"])
            adata.obs["doublet_score"] = doublet_score
        else:
            raise Exception("Doublet detection tool available: scDblFinder, Scrublet and DoubletDetection")

        n_doublets = adata.obs["doublet_class"].value_counts()["doublet"]
        adata = adata[adata.obs["doublet_class"] == "singlet"].copy()
        logger.info(f"Remove {n_doublets} doublets")
        df.loc[5] = ["Rm_Doublets", adata.shape[0], adata.shape[1], f"Remove neotypic doublets using {doublet_tool}"]

    # Save Metrics File
    if metrics:
        from dotools_py.utils import make_grid_spec
        from dotools_py.utility import get_hex_colormaps
        import matplotlib.lines as mlines
        df_plot = df.iloc[:, :-1].melt(id_vars="QC_Step")  # Exclude comments

        fig, gs = make_grid_spec(
            None or (8, 5), nrows=1, ncols=2, wspace=0.7 / 6, width_ratios=[6 - (0.9 + 0) + 0, 0.9]
        )

        ax = fig.add_subplot(gs[0])

        bp = sns.barplot(df_plot, hue="QC_Step", x="value", y="variable",
                         order=["nCells", "nFeatures"], hue_order=list(df["QC_Step"]),
                         palette="tab10", ax=ax, legend=False)

        for container in bp.containers:
            bp.bar_label(container, fmt='{:,.0f}')
        bp.set_title("Summary Quality Control", fontdict={"weight": "bold"})
        bp.set_ylabel("", fontsize=18)
        bp.set_xlabel("Counts", fontsize=18)
        bp.set_yticklabels(bp.get_yticklabels(), rotation=90, va="center", fontdict={"weight": "bold"})

        axs_legend = fig.add_subplot(gs[1])
        colors_dict = dict(zip(list(df["QC_Step"]), get_hex_colormaps("tab10"), strict=False))
        handles = []
        for lab, c in colors_dict.items():
            handles.append(mlines.Line2D([0], [0], marker=".", color=c, lw=0, label=lab,
                                         markerfacecolor=c, markeredgecolor=None, markersize=18))

        legend = axs_legend.legend(handles=handles, frameon=False, loc="center left", ncols=1, title="",
                                   prop={"size": 12, "weight": "bold"})
        legend.get_title().set_fontweight("bold")
        legend.get_title().set_fontsize(12 + 2)
        axs_legend.tick_params(axis="both", left=False, labelleft=False, labelright=False, bottom=False,
                               labelbottom=False)
        axs_legend.spines[["right", "left", "top", "bottom"]].set_visible(False)
        axs_legend.grid(visible=False)
        plt.savefig(os.path.join(qc_path, f"{today}_QC_Metrics{ids}.svg"), bbox_inches="tight")
        plt.close(fig)

        # Save Metric File
        df.to_excel(os.path.join(qc_path, metrics_filename), index=False)
    return adata


def importer_py(
    paths: list,
    ids: list,
    metadata: dict | None = None,
    batch_key: str = "batch",
    remove_doublets: bool = True,
    doublet_tool: Literal["scDblFinder", "Scrublet", "DoubletDetection"] = "scDblFinder",
    min_genes_in_cell: int = 300,
    min_cells_with_genes: int = 5,
    cut_mt: int = 5,
    n_reads: int = 10_000,
    min_counts: int | None = None,
    max_counts: int | None = None,
    min_genes: int | None = None,
    max_genes: int | None = None,
    low_quantile: int | None = None,
    high_quantile: int | None = None,
) -> ad.AnnData:
    """Quality control analysis for sc/snRNA.

    The input is a list with paths to H5 files generated with
    `CellRanger <https://www.10xgenomics.com/support/software/cell-ranger/latest>`_,
    `Cellbender <https://cellbender.readthedocs.io/en/latest/>`_, or
    `STARsolo <https://github.com/alexdobin/STAR>`_. A list of batch names for each sample must also be provided.
    Optionally, a dictionary with additional metadata can be passed. The order of batch names and metadata must
    match the order of the file paths.

    For each sample, several quality and filtering steps are applied:

    - Filter genes expressed in a low number of cells.
    - Filter cells with a low number of genes.
    - Filter cells with high mitochondrial content (recommended: 5% for scRNA, 3% for snRNA).
    - Filter cells based on nUMI and features using two modes:
        1. **Absolute filtering**: Sets absolute values for min/max UMI and features.
        2. **Quantile filtering**: Filters top/lower quantiles.
    - Remove doublets using scDblFinder, Scrublet, or DoubletDetection.

    An Excel sheet summarizing how many cells/genes were removed at each step will be generated,
    along with violin plots showing the distribution of `total_counts`, `n_genes_by_counts`,
    and `pct_mt_content` before and after QC. These outputs will be saved in the folder containing the H5 files.

    After QC, the data will be log-normalized and scaled. Highly variable genes and PCA will also be computed.

    :param paths: list with the path to the H5 files.
    :param ids: list with the batch name for each sample.
    :param metadata: dictionary with metadata information.
    :param batch_key: key in `.obs` for the batch information.
    :param remove_doublets: if set to True, neotypic doublets will be removed.
    :param doublet_tool: doublet tool to use. Available scDblFinder, Scrublet and DoubletDetection.
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
    :return: Returns an Annotated data matrix of shape `n_obs` x `n_vars` with all the samples concatenated.

    Example
    -------
    >>> import dotools_py as do
    >>> paths = ["/path/sample1", "/path/sample2"]
    >>> batchname = ["sample1", "sample2"]
    >>> metadata = {
    ...     "condition": ["WT", "KO"],
    ...     "age": ["3m", "3m"],
    ... }
    >>> adata = do.pp.importer_py(
    ...     paths=paths,
    ...     ids=batchname,
    ...     metadata=metadata,
    ...     batch_key="batch",
    ...     remove_doublets=True,
    ...     min_genes_in_cell=300,
    ...     min_cells_with_genes=5,
    ...     n_reads=10_000,
    ...     cut_mt=5,
    ...     high_quantile=95,
    ...     min_counts=500,
    ... )
    """
    import scanpy as sc

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
            doublet_tool=doublet_tool,
        )

        # Vln Plots showing Metrics before qc
        _qc_vln(adata, title=f"PostQC for {ids[idx]}", path=qc_path, filename=f"Vln_PostQC_{ids[idx]}.svg")

        adata_dict[ids[idx]] = adata

    logger.info("Concatenating samples")
    adata_concat = ad.concat(
        adata_dict.values(), label=batch_key, keys=adata_dict.keys(), join="outer", index_unique="-", fill_value=0
    )
    logger.info("Normalisation of the expression")
    _normalise(adata_concat, n_reads=n_reads, scale=False)

    logger.info("Finding Highly Variable Genes shared across samples")
    sc.pp.highly_variable_genes(adata_concat, batch_key=batch_key)

    logger.info("Run PCA")
    hvg = adata_concat[:, adata_concat.var.highly_variable].copy()
    sc.pp.scale(hvg, zero_center=True)  # Scale only on HVGs to replicate Seurat Approach
    sc.pp.pca(hvg)  # PCA on Scaled HVGs
    adata_concat.obsm["X_pca"] = hvg.obsm["X_pca"].copy()  # Save in original object

    return adata_concat


def sctransform_normalise(adata: ad.AnnData, batch_key: str = None, layer: str = None) -> None:
    """Normalisation based on `SCTransform <https://github.com/satijalab/sctransform>`_.

    This function performs an alternative normalisation based on the SCTransform.

    :param adata: AnnData object with counts in `X`.
    :param batch_key: obs metadata with batch information.
    :param layer: layer to use.
    :return: Returns None. The input AnnData object will have two new layers containing the SCT counts and normalise data.

    Example
    ------
    >>> import dotools_py as do
    >>> adata = do.dt.example_10x_processed()
    >>> adata
    AnnData object with n_obs × n_vars = 700 × 1851
    obs: 'batch', 'condition', 'n_genes_by_counts', 'log1p_n_genes_by_counts', 'total_counts', 'log1p_total_counts',
         'total_counts_mt', 'log1p_total_counts_mt', 'pct_counts_mt', 'total_counts_ribo', 'log1p_total_counts_ribo',
         'pct_counts_ribo', 'n_genes', 'n_counts', 'doublet_class', 'doublet_score', 'leiden', 'cell_type',
         'autoAnnot', 'celltypist_conf_score', 'annotation', 'annotation_recluster'
    var: 'mean', 'std', 'highly_variable', 'means', 'dispersions', 'dispersions_norm', 'highly_variable_nbatches',
         'highly_variable_intersection'
    uns: 'annotation_colors', 'annotation_recluster_colors', 'batch_colors', 'hvg', 'leiden', 'leiden_colors', 'log1p',
         'neighbors', 'pca', 'umap'
    obsm: 'X_CCA', 'X_pca', 'X_umap'
    varm: 'PCs'
    layers: 'counts', 'logcounts'
    obsp: 'connectivities', 'distances'
    >>>
    >>> do.pp.sctransform_normalise(adata, batch_key="batch", layer="counts")
    >>> adata
    AnnData object with n_obs × n_vars = 700 × 1181
    obs: 'batch', 'condition', 'n_genes_by_counts', 'log1p_n_genes_by_counts', 'total_counts', 'log1p_total_counts',
         'total_counts_mt', 'log1p_total_counts_mt', 'pct_counts_mt', 'total_counts_ribo', 'log1p_total_counts_ribo',
         'pct_counts_ribo', 'n_genes', 'n_counts', 'doublet_class', 'doublet_score', 'leiden', 'cell_type',
         'autoAnnot', 'celltypist_conf_score', 'annotation', 'annotation_recluster'
    var: 'mean', 'std', 'highly_variable', 'means', 'dispersions', 'dispersions_norm', 'highly_variable_nbatches',
         'highly_variable_intersection', 'SCT_rm'
    obsm: 'SCT_rm'
    varm: 'PCs'
    layers: 'counts', 'logcounts', 'SCT_norm', 'SCT_counts'
    obsp: 'connectivities', 'distances'
    """
    from scipy import sparse
    import polars

    rscript = get_paths_utils("_run_SCTransform.R")
    tmpdir_path = Path("/tmp") / f"SCTransform_{uuid.uuid4().hex}"
    tmpdir_path.mkdir(parents=True, exist_ok=False)

    logger.info("Preparing to transfer to R")
    adata_copy = adata.copy()
    if layer is not None:
        adata.X = adata.layers[layer].copy()
    del adata.uns
    del adata.obsm

    if batch_key is not None:
        adata_copy.obs["batch"] = adata_copy.obs[batch_key].copy()
    else:
        adata_copy.obs["batch"] = "batch1"
    adata_copy.write(tmpdir_path / "adata_tmp.h5ad")

    logger.info("Running SCTransform in R")
    subprocess.call(["Rscript", rscript, "--input=" + str(tmpdir_path) + "/", "--out=" + str(tmpdir_path) + "/"])

    raw_counts = polars.read_csv(os.path.join(tmpdir_path, "SCTransform_raw.csv"), infer_schema_length=0)
    raw_counts = raw_counts.to_pandas().astype(float)
    raw_counts = raw_counts.set_index(adata.obs_names)

    norm_counts = polars.read_csv(os.path.join(tmpdir_path, "SCTransform_norm.csv"), infer_schema_length=0)
    norm_counts = norm_counts.to_pandas().astype(float)
    norm_counts = norm_counts.set_index(adata.obs_names)

    # Transfer genes not kept during normalisation to .obsm
    excluded_genes = [gene for gene in adata.var_names if gene not in norm_counts.columns]
    adata.var["SCT_rm"] = [True if gene in excluded_genes else False for gene in adata.var_names]
    adata.obsm["SCT_rm"] = adata[:, adata.var["SCT_rm"].values].X.toarray()
    adata = adata[:, ~adata.var["SCT_rm"].values]

    # Make sure we have the same order or barcodes and features
    norm_counts = norm_counts.reindex(index=adata.obs_names, columns=adata.var_names)
    raw_counts = raw_counts.reindex(index=adata.obs_names, columns=adata.var_names)

    adata.layers["SCT_norm"] = sparse.csr_matrix(norm_counts.values)
    adata.layers["SCT_counts"] = sparse.csr_matrix(raw_counts.values)
    return None
