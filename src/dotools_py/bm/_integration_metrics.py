import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from numba import njit
from dotools_py.utils import check_missing, require_dependencies, iterase_input, make_grid_spec, convert_path, return_axis, save_plot
from dotools_py.utility._plotting import get_hex_colormaps
from dotools_py.get import subset as get_subset
from typing import Literal
from dotools_py.bm._helper import kBET, scib_silhouette_batch, scib_silhouette
from dotools_py.logger import logger


@njit
def _largest_component_fraction(labels):
    labels = np.sort(labels)

    max_count, current = 1, 1
    for i in range(1, labels.size):
        if labels[i] == labels[i - 1]:
            current += 1  # Go through the list and count the occurrences
        else:  # When the new one is different update the max_count
            if current > max_count:
                max_count = current
            current = 1
    if current > max_count:
        max_count = current
    return max_count / labels.size


def graph_connectivity(
    adata: ad.AnnData,
    annotation_key: str,
) -> np.floating:
    """Graph Connectivity.

    Quantify the connectivity of the subgraph per cell type. The final score is the average for all cell types.


    :param adata: Integrated annotated data matrix with neighborhood graph computed.
    :param annotation_key:  Column in `adata.obs` containing the group labels.
    :return: Returns a float number between 0 and 1 that represents the connectivity of the subgraph. Larger values represent a better batch removal.

    Examples
    --------
    >>> import dotools_py as do
    >>> adata = do.dt.example_10x_processed()
    >>> do.bm.graph_connectivity(adata, "annotation")
    Out[48]: np.float64(0.594)

    """
    assert "neighbors" in adata.uns, "Neighborhood graph is not in adata.uns"
    check_missing(adata, groups=annotation_key)

    results = []
    for group in adata.obs[annotation_key].unique():
        adata_group = get_subset(adata, obs_key=annotation_key, obs_groups=group)
        _, labels = connected_components(
            adata_group.obsp["connectivities"], connection="strong"
        )
        results.append(_largest_component_fraction(labels))
    return np.round(np.mean(results), 3)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# def ilisi_graph(
#     adata: ad.AnnData,
#     batch_key: str = "batch",
#     integration_type: Literal["embedding", "knn", "full"] = "knn",
#     use_rep: str = None,
#     n_neighbors: int = 90,
#     subsample: int = None,
#     scale: bool = True,
#     threads: int = 1,
# ) -> float:
#     """Integration LISI (iLISI) score.
#
#     Parameters
#     ----------
#     adata
#         Annotated data matrix.
#     batch_key
#         Column in adata.obs with batch information.
#     integration_type
#         Type of data integration. If set to `knn` it will take the neighborhood present in the object. If
#         set to `embedding` it will recompute the neighborhood based on `use_rep` and if set to `full` it will
#         recompute PCA use the PCA embedding for the neighborhood graph.
#     use_rep
#         Representation to use to compute neighborhood when `integration_type` is set to `embedding`.
#     n_neighbors
#         Number of nearest neighbors to compute lisi score. The initial neighborhood size that is used to compute
#         the shortest paths is 15.
#     subsample
#         Percentage of observations (integer between 0 and 100) to which lisi scoring should be subsampled.
#     scale
#         Re-scale output values between 0 and 1 (True/False)
#     threads
#         Number of cores to use for computation. If not specify will use half of the available cores.
#
#     Returns
#     -------
#     Returns the median iLISI scores per batch.
#
#     Examples
#     --------
#     >>> import dotools_py as do
#     >>> adata = do.dt.example_10x_processed()
#     >>> ilisi_graph(adata, batch_key="batch", integration_type="embedding", use_rep="X_CCA")
#
#     """
#     import scib
#     import multiprocessing
#
#     check_missing(adata, groups=batch_key)
#     threads = int(multiprocessing.cpu_count() / 2) if threads is None else int(threads)
#     integration_type = {"embedding": "emb", "knn": "knn", "full": "full"}[integration_type]
#
#     score = scib.metrics.ilisi_graph(
#         adata=adata,
#         batch_key=batch_key,
#         type_=integration_type,
#         use_rep=use_rep,
#         k0=n_neighbors,
#         subsample=subsample,
#         scale=scale,
#         n_cores=threads,
#         verbose=True
#     )
#
#     return score


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def kbet(
    adata: ad.AnnData,
    batch_key: str = "batch",
    annotation_key: str = "annotation",
    integration_type: Literal["embedding", "knn", "full"] = "knn",
    use_rep: str = None,
    scale: bool = True,
    get_data: bool = True,
) -> float | pd.DataFrame:
    """kBET score.

    Compute the average of k-nearest neighbor batch effect test (kBET) score per annotation.
    This is a wrapper function of the implementation by `Büttner et al. 2019 <https://www.nature.com/articles/s41592-018-0254-1>`_.
    kBET measures the bias of a batch variable in the kNN graph. Specifically, kBET is quantified as the average
    rejection rate of Chi-squared tests of local vs global batch label distributions.
    This means that smaller values indicate better batch mixing. By default, the original kBET score is scaled
    between 0 and 1 so that larger scores are associated with better batch mixing.


    :param adata: Annotated data matrix.
    :param batch_key: Column in adata.obs with batch information.
    :param annotation_key:  Column in adata.obs with cell type or cluster information.
    :param integration_type: Type of data integration. If set to `knn` it will take the neighborhood present in the object. If
                             set to `embedding` it will recompute the neighborhood based on `use_rep` and if set to `full` it will
                             recompute PCA use the PCA embedding for the neighborhood graph.
    :param use_rep:  Representation to use to compute neighborhood when `integration_type` is set to `embedding`.
    :param scale: Re-scale output values between 0 and 1 (True/False)
    :param get_data: If it is set to `True` it also returns a `pd.DataFrame` with kBET observed rejection rater per cluster
    :return: Returns de kBET score (average of kBET per label) based on observed rejection rate. If `get_data` is set to `True` it
            returns a `pd.DataFrame` with kBET observed rejection rater per cluster

    Examples
    --------
    >>> import dotools_py as do
    >>> adata = do.dt.example_10x_processed()
    >>> do.bm.kbet(adata, "batch", "annotation")  # Estimation of score per cell type
        Out[23]:
         cluster      kBET
    0    B_cells  1.000000
    1  Monocytes  1.000000
    2         NK  1.000000
    3    T_cells  0.323617
    4        pDC  1.000000
    >>> kbet(adata, "batch", "annotation", get_data=False)  # Estimation of score
    Out[24]: np.float64(0.13540425531914901)

    """
    integration_type = {"embedding": "emb", "knn": "knn", "full": "full"}[integration_type]

    scores = kBET(
        adata=adata,
        batch_key=batch_key,
        label_key=annotation_key,
        type_=integration_type,
        embed=use_rep,
        scaled=scale,
        return_df=get_data,
    )

    return scores


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

@require_dependencies([{"name": "scib"}])
def pcr_comparison(
    adata_pre: ad.AnnData,
    adata_post: ad.AnnData,
    covariate: str,
    use_rep: str,
    n_comps: int = 50,
    recompute_pca: bool = False,
    linreg_method: Literal["numpy", "sklearn"] = "numpy",
    scale: bool = True
) -> float:
    """Principal component regression score.

    Compare the explained variance before and after integration. Return either the difference of variance contribution
    before and after integration or a score between 0 and 1 (if `scaled` is set to `True`) with 0 if the variance contribution
    hasn’t changed. The larger the score, the more different the variance contributions
    are before and after integration.

    :param adata_pre: Annotated data matrix before the integration
    :param adata_post: Annotated data matrix after the integration
    :param covariate: Column in adata.obs to regress against
    :param use_rep: Embedding to use for principal component analysis. If `None`, use the full expression matrix (adata.X), otherwise use the embedding provided in adata_post.obsm[use_rep].
    :param n_comps: Number of principal components to compute
    :param recompute_pca: Whether to recompute PCA with default settings
    :param linreg_method: Method for computing the linear regression
    :param scale: If set to `True` scale score between 0 and 1.
    :return:  Returns the difference of variance contribution of PCR.

    Examples
    --------
    >>> import dotools_py as do
    >>> adata = do.dt.example_10x_processed()
    >>> adata_unintegrated = adata.copy()
    >>> del adata_unintegrated.obsm
    >>> do.bm.pcr_comparison(adata_pre=adata_unintegrated, adata_post=adata, covariate="batch", use_rep="X_CCA")
    Out[47]: np.float64(0.832)
    """
    import scib
    import multiprocessing
    threads = multiprocessing.cpu_count() / 2

    score = scib.metrics.pcr_comparison(
        adata_post=adata_post,
        adata_pre=adata_pre,
        covariate=covariate,
        embed=use_rep,
        n_comps=n_comps,
        linreg_method=linreg_method,
        recompute_pca=recompute_pca,
        scale=scale,
        verbose=False,
        n_threads=threads,
    )
    return np.round(score, 3)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def silhouette_batch(
    adata: ad.AnnData,
    batch_key: str,
    annotation_key: str,
    use_rep: str,
    metric: str = "euclidean",
    scale: bool = True,
    get_all: bool = False,
) -> float | tuple:
    """Batch ASW.

    Modified average silhouette width (ASW) of batch This metric measures the silhouette of a given batch.
    It assumes that a silhouette width close to 0 represents perfect overlap of the batches, thus the absolute value of
    the silhouette width is used to measure how well batches are mixed. If `scale` is set to `True`, the  absolute ASW
    per group is subtracted from 1 before averaging, so that 0 indicates suboptimal label representation and 1 indicates
    optimal label representation.

    :param adata: Annotated data matrix.
    :param batch_key: Column in adata.obs with batch information.
    :param annotation_key: Column in adata.obs with cell type or cluster information.
    :param use_rep: Column in adata.obsm with the embedding.
    :param metric: See `sklearn.silhouette_score <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html>`_
    :param scale: If set to `True`, scale the values between 0 and 1
    :param get_all: If set to `True` returns the silhouette score, the average silhouette score per cluster and all the silhouette scores.
    :return: Returns 1) the average width silhouette  2) the average silhouette score per cluster and 3) all silhouette scores if `get_all` is set to `True`, otherwise returns the average width silhouette (ASW).

    Examples
    --------
    >>> import dotools_py as do
    >>> adata = do.dt.example_10x_processed()
    >>> do.bm.silhouette_batch(adata, batch_key="batch", annotation_key="annotation", use_rep="X_CCA")
    Out[63]: np.float64(0.8107897347900055)
    >>> score, mean_score, all_scores = do.bm.silhouette_batch(adata, batch_key="batch", annotation_key="annotation", use_rep="X_CCA", get_all=True)
    >>> mean_score
                   silhouette_score
    group
    B_cells            0.795807
    Monocytes          0.603867
    NK                 0.878482
    T_cells            0.961296
    pDC                0.814496

    """

    score = scib_silhouette_batch(
        adata=adata,
        batch_key=batch_key,
        annotation_key=annotation_key,
        embed=use_rep,
        metric=metric,
        scale=scale,
        return_all=get_all,
    )
    return score


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def eval_integration(
    adata_post: ad.AnnData,
    adata_pre: ad.AnnData,
    batch_key: str,
    annotation_key: str,
    use_rep: str | list,

    figsize: tuple = (6, 5),
    ax: plt.Axes = None,
    palette: dict = None,
    cmap: str = "tab10",
    path: str = None,
    filename: str = None,

    title: str = None,
    title_fontsize: int = 15,
    title_legend: str = None,
    legend_fontsize: int = 12,
    show: bool = True,

    scale: bool = True,
    compute_metrics: Literal[
                         "GraphConnectivity", "kBET", "pcr_comparison", "silhouette_batch", "silhouette_global", "all"] | list = "all",
) -> pd.DataFrame | tuple[pd.DataFrame | dict[str, plt.Axes]]:
    """Evaluate the integration.

    This function calculate several metrics to evaluate how well the integration worked. The batch correction metrics
    values are scaled by default between 0 and 1, in which larger scores represent better batch removal. It will generate
    a barplot to summarize all the metrics.

    :param adata_post:  Annotated data matrix after integration.
    :param adata_pre: Annotated data matrix before integration.
    :param batch_key: Column in adata.obs with batch information.
    :param annotation_key: Column in adata.obs with clustering information.
    :param use_rep: Key(s) in adata.obsp with the embedding generated from the integration.
    :param figsize: Figure size, the format is (width, height).
    :param ax: Matplotlib axes to use for plotting. If not set, a new figure will be generated.
    :param palette: Dictionary with the embedding names (keys) and the color as values.
    :param cmap: Matplotlib colormap to use for the different embeddings.
    :param path: Path to the folder to save the figure.
    :param filename:  Name of file to use when saving the figure.
    :param title: Title for the figure.
    :param title_fontsize: Size of the title font.
    :param title_legend: Title of the legend.
    :param legend_fontsize: Size of the legend title font.
    :param show:  If set to `False`, returns a dictionary with the matplotlib axes.
    :param scale: If set to `True` scale score between 0 and 1.
    :param compute_metrics: List of the metrics to compute. Set to "all" to compute all metrics.
    :return: Returns a `pd.Dataframe` with the metrics for each embedding in `use_rep`. If show is set to `False` it returns a
            tuple with the first element being the DataFrame with the metrics and the second a dictionary with the matplotlib
            axes for the figure.

    Examples
    ---------

    Evaluate the metrics across different integration methods.

    .. plot::
        :context: close-figs

        import dotools_py as do
        adata = do.dt.example_10x_processed()
        adata_unintegrated = adata.copy()
        del adata_unintegrated.obsm
        database = do.bm.eval_integration(adata_post=adata, adata_pre=adata_unintegrated, batch_key="batch", annotation_key="annotation", use_rep=["X_CCA", "X_pca"], compute_metrics = ["GraphConnectivity", "silhouette_batch", "silhouette_global"] )

    """

    import seaborn as sns
    import matplotlib.lines as mlines

    use_rep = iterase_input(use_rep)

    compute_metrics = (
        ["GraphConnectivity", "kBET", "pcr_comparison", "silhouette_batch", "silhouette_global"]
        if compute_metrics == "all" else compute_metrics
    )

    database = {key: [] for key in compute_metrics}

    # database["kBET_annotation_key"] = kbet(adata=adata_post, batch_key=batch_key,annotation_key=annotation_key, integration_type="knn", use_rep=use_rep, scale=scale, get_data=True)
    # database["silhouette_batch_annotation_key"] = silhouette_batch(adata=adata_post,  batch_key=batch_key,annotation_key=annotation_key,  use_rep=use_rep, scale=scale, get_all=True)[1]
    # database["isolated_label_asw"] = scib.metrics.isolated_labels_asw(adata=adata_post, label_key=annotation_key, batch_key=batch_key, embed=use_rep, scale=scale, verbose=False)

    for rep in use_rep:
        logger.info(f"Computing metrics for {rep}")
        if "GraphConnectivity" in compute_metrics:
            database["GraphConnectivity"].append(graph_connectivity(adata_post, annotation_key))
        if "kBET" in compute_metrics:
            database["kBET"].append(
                kbet(adata=adata_post, batch_key=batch_key, annotation_key=annotation_key, integration_type="knn",
                     use_rep=rep, scale=scale, get_data=False)
            )
        if "pcr_comparison" in compute_metrics:
            database["pcr_comparison"].append(
                pcr_comparison(adata_post=adata_post, adata_pre=adata_pre, covariate=batch_key, use_rep=rep,
                               scale=scale)
            )
        if "silhouette_batch" in compute_metrics:
            database["silhouette_batch"].append(
                silhouette_batch(adata=adata_post, batch_key=batch_key, annotation_key=annotation_key, use_rep=rep,
                                 scale=scale)
            )
        if "silhouette_global" in compute_metrics:
            database["silhouette_global"].append(
                scib_silhouette(adata=adata_post, annotation_key=annotation_key, embed=rep, scale=scale)
            )

    database = pd.DataFrame.from_dict(database, orient="index", columns=use_rep)
    database = database.reset_index().melt(id_vars="index", var_name="embedding", value_name="value")

    #### Figure set up
    if len(get_hex_colormaps(cmap)) < len(use_rep):
        raise ValueError(f"{cmap} has {len(get_hex_colormaps(cmap))} but there are {len(use_rep)} embeddings to test")
    embedding_to_color = dict(zip(use_rep, get_hex_colormaps(cmap)))
    palette = palette if palette is not None else embedding_to_color
    height, width = figsize
    fig, gs = make_grid_spec(
        ax or (width, height), nrows=1, ncols=2, wspace=0.7 / width, width_ratios=[width - 1.5, 1.5])

    # Add MainPlot
    main_axis = fig.add_subplot(gs[0])
    sns.barplot(
        database, x="index", y="value", hue="embedding", ax=main_axis, legend=False, palette=palette,
    )
    main_axis.set_xlabel("Metrics")
    main_axis.set_ylabel("Value")
    main_axis.set_xticklabels(main_axis.get_xticklabels(), rotation=45, ha="right", fontweight="bold")
    main_axis.set_title(title, fontsize=title_fontsize, fontweight="bold")
    if scale:
        main_axis.set_ylim(0, 1)

    # Add Legend
    legend_axis = fig.add_subplot(gs[1])
    handles = [mlines.Line2D([0], [0], marker=".", color=c, lw=0, label=lab, markerfacecolor=c,
                             markeredgecolor=None, markersize=18) for lab, c in palette.items()]

    legend = legend_axis.legend(
        handles=handles, frameon=False, loc="center", ncols=1, title=title_legend,
        prop={"size": legend_fontsize, "weight": "bold"},
    )
    legend.get_title().set_fontweight("bold")
    legend.get_title().set_fontsize(legend_fontsize + 2)
    legend_axis.tick_params(
        axis="both", left=False, labelleft=False, labelright=False, bottom=False, labelbottom=False)
    legend_axis.spines[["right", "left", "top", "bottom"]].set_visible(False)
    legend_axis.grid(visible=False)
    save_plot(path, filename)
    if show:
        plt.tight_layout()
        plt.show()
        return database
    else:
        return database, return_axis(show, {"main_ax": main_axis, "legend_ax": legend_axis})
