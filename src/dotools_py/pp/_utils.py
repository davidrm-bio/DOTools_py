import os
from typing import Literal, Dict

from prelude_py import ad, np, pd, plt, sns

from dotools_py.logger import  logger
from dotools_py._utils import iterase_input, check_missing, sanitize_anndata, check_r_package, convert_path
from dotools_py.pl._plot_utils import save_plot
from dotools_py._custom_class import  PathLike


def _qc_vln(
    adata: ad.AnnData,
    title: str = "ViolinPlots - Quality Metrics",
    path: PathLike = None,
    filename: str = "ViolinPlots.png",
    stats: list = ("total_counts", "n_genes_by_counts", "pct_counts_mt"),
    colors: str = "lightsteelblue",
) -> None:
    """Violin Plots showing basic QC stats.

    Generate ViolinPlots to show the distribution of total counts, number of genes and percentage of
    mitochondrial genes.

    :param adata: annotated dt matrix.
    :param title: title of the Plot.
    :param path: path to figure folder.
    :param filename: name of the file.
    :param stats: `.obs` column name to plot.
    :param colors: colors for the violin plots.
    :return:
    """
    sanitize_anndata(adata)
    stats, colors = iterase_input(stats), iterase_input(colors)
    check_missing(adata, groups=stats)
    colors, ncols = colors * len(stats), len(stats)

    fig, axs = plt.subplots(1, ncols, figsize=(10, 6))
    for idx in range(ncols):
        vln = sns.violinplot(adata.obs[stats[idx]], ax=axs[idx], color=colors[idx])
        vln.set_xticklabels([f"Median = {np.floor(np.median(adata.obs[stats[idx]]))}"], fontweight="bold")
        vln.set_title("")
    plt.suptitle(title, fontsize=30, fontweight="bold")
    save_plot(path, filename)
    return plt.close()



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
    sanitize_anndata(adata)
    counts, mask = adata.obs["total_counts"], np.ones(adata.n_obs, dtype=bool)
    if low:
        mask &= counts > np.percentile(counts, low)
    if high:
        mask &= counts < np.percentile(counts, high)
    return adata[mask, :].copy()


def py_none_to_r(obj):
    from rpy2.robjects import r
    if obj is None:
        return r("NULL")  # evaluated at conversion time
    return obj



def _normalise(
    adata: ad.AnnData,
    n_reads: int = 10_000,
    log_data: bool = True
) -> None:
    """Normalize raw counts.

    The input is an unnormalize anndata object. The dt in X will be log-normalize to 10,000 reads per cell.
    The returned anndata object will contain 3 layers:
    * counts: contains the raw unnormalized counts
    * logcounts: contains the log-normalize counts
    * scaled: contained the log-normalize counts scaled
    Additionally, the log-normalize counts will also be saved under the X attribute.

    :param adata: annData object
    :param n_reads: target number of reads per cell to normalize to. (Default  is **10,000**)
    :param log_data: Whether to apply logarithm to the normalize data or not.
    :return: log-normalise anndata object
    """
    import scanpy as sc
    sc.pp.normalize_total(adata, target_sum=n_reads, inplace=True)
    if log_data:
        sc.pp.log1p(adata)
        adata.layers["logcounts"] = adata.X.copy()
    else:
        adata.layers["norm_counts"] = adata.X.copy()
    return


def _lower_strings(obj):
    if isinstance(obj, str):
        return obj.lower()
    elif isinstance(obj, list):
        return [_lower_strings(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_lower_strings(item) for item in obj)
    else:
        return obj


def _run_sc_dbl_finder(
    adata: ad.AnnData,
    batch_key: str,
    cluster_key: str,
    doublet_rate: float = None,
    scdblfinder_metric: Literal['merror', 'logloss', 'auc', 'aucpr'] = "logloss",
    random_state: int = 0,
) -> pd.DataFrame:
    """Detect doublets using scDblFinder

    :param adata: Annotated data matrix.
    :param batch_key: Column in `adata.obs` with batch information.
    :param cluster_key: Column in `adata.obs` with cluster information.
    :param doublet_rate: Doublet rate.
    :param scdblfinder_metric: Error metric to optimize during training (e.g. 'merror', 'logloss', 'auc', 'aucpr').
    :param random_state: Random seed
    :return: Returns a pandas DataFrame with the results of the double inference.
    """
    import anndata2ri
    from rpy2.robjects import r, conversion, globalenv, pandas2ri

    check_r_package("scDblFinder")
    none_converter = conversion.Converter("None converter")
    none_converter.py2rpy.register(type(None), py_none_to_r)
    adata_copy = adata.copy()
    del adata_copy.raw
    adata_copy.uns.clear()

    with conversion.localconverter(anndata2ri.converter + none_converter + pandas2ri.converter):
        r.assign("adata", adata_copy)
        r.assign("batch", batch_key)
        r.assign("cluster", cluster_key)
        r.assign("dbr", doublet_rate)
        r.assign("metric", scdblfinder_metric)
        r.assign("random_state", random_state)
        r(
            """
            library(scDblFinder)
            if (!suppressPackageStartupMessages(require(SingleCellExperiment))) {
                stop("R dependecy SingleCellExperiment not found.")
            }
            set.seed(random_state)
            sce <- as(adata, "SingleCellExperiment")
            sce <- scDblFinder(sce, samples = batch, clusters=cluster, dbr=dbr, metric=metric, verbose = F)
            df <- data.frame(
                scDblFinder.class = as.character(colData(sce)$scDblFinder.class),
                scDblFinder.score = as.numeric(colData(sce)$scDblFinder.score),
                stringsAsFactors = FALSE
            )
            """
        )
        doublets = globalenv["df"]
        r(
            """
            rm(adata, batch, cluster, dbr, metric, random_state, sce, df)
            gc()
            """
        )
    doublets = doublets.set_index(adata.obs_names)
    return doublets


def _run_ovrlpy(
    df: pd.DataFrame,
    batch_key: str,
    ovrlpy_report_path: PathLike,
    ovrlpy_keys: Dict,
    random_state: int = 0
) -> None:
    assert isinstance(df, pd.DataFrame), (
        "To run Ovrlpy (Detection of doublets in scSpatialTranscriptomics provide a DataFrame with X,Y,Z coordinates "
        "for features."
    )
    assert batch_key is None, "Ovrlpy cannot perform doublet detection across batches"
    assert ovrlpy_report_path is not None, "Provide path to save the report from the Ovrlpy inference"

    import ovrlpy
    import pickle
    available_cores = int(os.cpu_count() / 2)
    ovrlpy_keys = {} if ovrlpy_keys is None else ovrlpy_keys
    gene_key, x_key, y_key, z_key = (ovrlpy_keys.get("gene_key", "feature_name"),
                                     ovrlpy_keys.get("x_key", "x_location"),
                                     ovrlpy_keys.get("y_key", "y_location"),
                                     ovrlpy_keys.get("z_key", "z_location"))

    data = ovrlpy.Ovrlp(
        df, n_workers=available_cores, random_state=random_state, gene_key=gene_key,
        coordinate_keys=(x_key, y_key, z_key),
    )
    data.analyse()

    # Save results in the report folder
    logger.info("Generating Report")
    os.makedirs(ovrlpy_report_path, exist_ok=True)
    _ = ovrlpy.plot_pseudocells(data)
    plt.savefig(convert_path(ovrlpy_report_path) / "Overview_Ovrlpy.pdf", bbox_inches="tight")
    plt.close()
    _ = ovrlpy.plot_signal_integrity(data, signal_threshold=3)
    plt.savefig(convert_path(ovrlpy_report_path) / "Integrity_Ovrlpy.pdf", bbox_inches="tight")
    plt.close()

    doublets = data.detect_doublets(min_signal=3, integrity_sigma=2)

    fig, ax = plt.subplots()
    _scatter = ax.scatter(doublets["x"], doublets["y"], c=doublets["integrity"], s=0.2, cmap="viridis")
    _ = ax.set_aspect("equal")
    _ = fig.colorbar(_scatter, ax=ax)
    plt.savefig(convert_path(ovrlpy_report_path) / "DoubletsIntegrity_Ovrlpy.pdf", bbox_inches="tight")
    plt.close()

    with open(convert_path(ovrlpy_report_path) / "ObjectOvrlpy.pickle", "wb") as file:
        pickle.dump(data, file)
    doublets.write_csv(convert_path(ovrlpy_report_path) / "SummaryDoublets.csv")
    return None
