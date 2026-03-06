from pathlib import Path

import anndata as ad
import numpy as np

from dotools_py.utils import iterase_input, check_missing, save_plot, sanitize_anndata

import matplotlib.pyplot as plt
import seaborn as sns


def _qc_vln(
    adata: ad.AnnData,
    title: str = "ViolinPlots - Quality Metrics",
    path: str | Path | None = None,
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
    return plt.show()



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
