from typing import Literal,Dict

from prelude_py import ad, plt, pd
from dotools_py._custom_class import PathLike
from dotools_py.logger import logger
from dotools_py.pl._Classes import MatrixPlot

def dotplot(
    adata: ad.AnnData,
    *,
    x_axis: str,
    features: str | list | dict,
    y_axis: str | None = None,
    xticks_order: list | None = None,
    yticks_order: list | None = None,
    layer: str | None = None,
    subset_adata: bool = False,
    logcounts: bool = True,
    expression_cutoff: float = 0.0,
    mean_only_expressed: bool = False,
    z_score: Literal["x_axis", "y_axis"] | None = None,
    standard_scale: Literal["x_axis", "y_axis"] | None = None,

    # Figure parameters
    figsize: tuple = (5, 6),
    ax: plt.Axes | None = None,
    swap_axes: bool = False,
    title: str = "",
    title_fontproperties: Dict[Literal["size", "weight"], str | int] | None = None,
    palette: str | None = None,
    cmap: str = "Reds",
    vmin: float | None = None,
    vcenter: float | None = None,
    vmax: float | None = None,

    color_legend_title: str = "LogMean(nUMI)\nin group",
    size_legend_title: str = "Fraction of cells\nin group (%)",
    feature_fontsize: float = 12,

    xticks_properties: dict | None = None,
    xticks_rotation: int = 45,
    cluster_x_axis: bool = False,
    yticks_properties: dict | None = None,
    yticks_rotation: int = 0,
    cluster_y_axis: bool = False,
    clustering_method: str = "complete",
    clustering_metric: str = "euclidean",
    var_group_offset: float = 0.75,
    var_group_rotation: float = 0,

    # IO
    path: PathLike | None = None,
    filename: str = "Heatmap.svg",
    show: bool = True,

    # Statistics
    add_stats: Literal["x_axis", "y_axis"] | None = None,
    test: Literal["wilcoxon", "t-test"] = "wilcoxon",
    correction_method: Literal["benjamini-hochberg", "bonferroni"] = "benjamini-hochberg",
    df_pvals: pd.DataFrame | None = None,
    stats_x_size: float | None = None,
    square_x_size: dict | None = None,
    pval_cutoff: float = 0.05,
    log2fc_cutoff: float = 0.0,

    # Fx specific
    smallest_dot: float | None = None,
    largest_dot: float | None = None,
    dot_min: float| None = None,
    dot_max: float| None = None,
    rect_height: float = 0.35,
    size_exponent: float| None = None,
    edge_color: str | None = None,
    edge_lw: float | None = None,
    **kwargs,
) -> dict | None:
    """Make a dot plot of the values of `adata.var_names` or `adata.obs`.

    There are two types of visualization:
        * Categorical dotplot: the ``features`` are grouped by ``x_axis``. The color represents the logarithmize mean
                               nUMI or the mean value for features in `adata.obs`. The size of the dots represents
                               the fraction of cells expressing the feature.
        * 2-Categorical dotplot: the ``features`` are groupbed by ``x_axis`` and ``y_axis``. For each feature the
                                ``x_axis`` categories will be duplicates to show how is the expression across the
                                ``y_axis`` categories.

    .. note::
        A feature is considered expressed if the expression value in the `adata` is above the specified threshold
        which is zero by default.

    :param adata: Annotated data matrix.
    :param x_axis: A valid key from `adata.obs` to groupby.
    :param features: A valid key from `adata.var_names` or a continuous value in `adata.obs` to show the mean values.
                    If `features` is a dictionary, then the key is used as label to group the values.
    :param y_axis: A valid additional key from `adata.obs` to groupby.
    :param xticks_order: Order of the categories in x_axis.
    :param yticks_order: Order of the categories in y_axis.
    :param layer: A valid key from `adata.layers` to be used for the mean expression calculations.
    :param subset_adata: Set to `True` to subset the AnnData if less `x_categories_order` and `y_categories_order`
                        are provided.
    :param logcounts: Set to `True` if the input is in log space.
    :param expression_cutoff: Cutoff used for binarizing the gene expression and determining the fraction of
                              cells expressing a given feature. A gene is expressed only if the expression value is greater
                              than this threshold.
    :param z_score: Apply Z-score transformation over the x-axis or y-axis.
    :param standard_scale: Apply MinMax normalisation over the x-axis or y-axis.
    :param figsize: Figure size, the format is (width, height).
    :param ax: Matplotlib axes to use for plotting. If not set, a new figure will be generated.
    :param swap_axes: Change the X and Y-axis.
    :param title: Title for the figure.
    :param title_fontproperties: Dictionary which should contain 'size' and 'weight' to define the fontsize and fontweight of the title of the figure.
    :param palette: String denoting matplotlib colormap.
    :param cmap: String denoting matplotlib colormap.
    :param vmin: The value representing the lower limit of the color scale.
    :param vcenter: The value representing the center of the color scale.
    :param vmax: The value representing the upper limit of the color scale.
    :param color_legend_title: Title for the colorbar.
    :param size_legend_title:  Title for the size legend.
    :param feature_fontsize: Size of the feature names when y-ticks is specified.
    :param xticks_properties: Dictionary which should contain 'size' and 'weight' to define the fontsize and fontweight of font of the x-axis.
    :param xticks_rotation:  Rotation of the x-ticks.
    :param cluster_x_axis: Hierarchically clustering the x-axis.
    :param yticks_properties:  Dictionary which should contain 'size' and 'weight' to define the fontsize and fontweight of the font of the y-axis.
    :param yticks_rotation: Rotation of the y-ticks.
    :param cluster_y_axis: Hierarchically clustering the y-axis.
    :param clustering_method: Linkage method to use for calculating clusters. See `scipy.cluster.hierarchy.linkage <https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage>`_.
    :param clustering_metric: Distance metric to use for the data. See `scipy.spatial.distance.pdist <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist>`_.
    :param var_group_offset: If `features` is a dictionary, set the rotation of the labels for each group.
    :param var_group_rotation:  Offset for the labels text in the brackets.
    :param path: Path to save plot.
    :param filename: Filename of the plot.
    :param show: If set to `False`, returns a dictionary with the matplotlib axes.
    :param add_stats: Add an asterisk (*) to indicate statistical significance.
    :param test: Statistical method to use.
    :param correction_method: Correction method for multiple testing.
    :param df_pvals: Dataframe with significant values with the same shape as the dotplot.
    :param stats_x_size: Size of the star.
    :param square_x_size: Dictionary specifying the `size` and thickness (`width`) of the squares.
    :param pval_cutoff: Cutoff for the p-value.
    :param log2fc_cutoff: Minimum value for the log2-foldchange to consider the feature to be differently regulated.
    :param smallest_dot: All expression levels with dot_min are plotted with this size.
    :param largest_dot: All expression levels with dot_max are plotted with this size.
    :param dot_min: If `None`, the minimum dot size is set to 0. If given, the value should be a number between 0 and 1.
                    All fractions smaller than dot_min are clipped to this value.
    :param dot_max:  If `None`, the maximum dot size is set to the maximum fraction value found. If given, the value
                    should be a number between 0 and 1. All Fractions larger than dot_max are clipped to this value.
    :param rect_height: Height of the boxes of the feature names in the dotplot when `y_axis` is provided
    :param size_exponent: Control the increase of dots.
    :param edge_color: Color of the edge of the dot.
    :param edge_lw: Thickness of the dot edges.
    :param kwargs: Additional arguments passed to `matplotlib.pyplot.scatter <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html>`_
    :return: Depending on ``show``, returns the plot if set to `True` or a dictionary with the axes

    Example
    -------
    Create a dotplot using a list of markers and a PBMC example dataset grouped by the
    cell type annotation

    .. plot::
        :context: close-figs

        import dotools_py as do
        adata = do.dt.example_10x_processed()
        markers = ['CD79A', 'CD4', 'CDK1']
        do.pl.dotplot(adata, x_axis='annotation',features=markers, figsize=(4, 3))

        # Add Statistical significance
        do.pl.dotplot(adata, x_axis='condition', features=markers, figsize=(6, 4), add_stats='x_axis')


    Create a dotplot grouping by two categorical metadata

    .. plot::
        :context: close-figs

        do.pl.dotplot(adata, x_axis='condition', features=markers, y_axis='annotation', figsize=(6, 4))

        # Add Statistical significance for groups with pvals < 0.05 and log2fc > 0.0
        # Note, the object is quite small, some groups cannot be tested for having one condition only.
        do.pl.dotplot(adata, x_axis='condition', features=markers, y_axis='annotation', figsize=(6, 4), add_stats='x_axis')

        axis =        do.pl.dotplot(adata, x_axis='condition', features=markers, y_axis='annotation', figsize=(6, 4), add_stats='x_axis', show=False)

    """
    # todo implement mean_only_expressed

    if subset_adata:
        if xticks_order is not None:
            adata = adata[adata.obs[x_axis].isin(xticks_order)]
            logger.warn(f"Subsetting anndata for {xticks_order}")
        if yticks_order is not None:
            adata = adata[adata.obs[y_axis].isin(yticks_order)]
            logger.warn(f"Subsetting anndata for {yticks_order}")
    else:
        if xticks_order is not None:
            assert len(xticks_order) == adata.obs[x_axis].nunique(), (
                f"Not all {x_axis} categories provided. Specify all or use subset_adata = True"
            )
        if yticks_order is not None:
            assert len(yticks_order) == adata.obs[y_axis].nunique(), (
                f"Not all {y_axis} categories provided. Specify all or use subset_adata = True"
            )

    if cmap is not None and palette is not None:
        raise ValueError("Cannot provide both cmap and palette")
    elif cmap is None and palette is None:
        cmap = "Reds"

    colormap = cmap if cmap is not None else palette

    if mean_only_expressed:
        raise NotImplementedError("mean_only_expressed is currently not available")


    plotter = MatrixPlot(
        adata=adata, x_axis=x_axis, features=features, y_axis=y_axis,
        xticks_order=xticks_order, yticks_order=yticks_order, layer=layer, logcounts=logcounts,
        figsize=figsize, ax=ax, swap_axes=swap_axes, title=title, title_fontproperties=title_fontproperties,
        palette=None, cmap=colormap, xticks_properties=xticks_properties, yticks_properties=yticks_properties,
        xticks_rotation=xticks_rotation, yticks_rotation=yticks_rotation, cluster_x_axis=cluster_x_axis,
        cluster_y_axis=cluster_y_axis, var_group_offset=var_group_offset, var_group_rotation=var_group_rotation,
        legend_size_title=size_legend_title, legend_title=color_legend_title, path=path, filename=filename,
        show=show, add_stats=add_stats, test=test, correction_method=correction_method, df_pvals=df_pvals,
        stats_x_size=stats_x_size, square_x_size=square_x_size, pval_cutoff=pval_cutoff, log2fc_cutoff=log2fc_cutoff,
        z_score=z_score, standard_scale=standard_scale, clustering_method=clustering_method,
        clustering_metric=clustering_metric, linewidth=0.1, vmin=vmin, vcenter=vcenter, vmax=vmax,
        annot_fontsize=feature_fontsize, annot_ratio=rect_height, expression_cutoff=expression_cutoff,
        dot_max=dot_max, dot_min=dot_min, size_exponent=size_exponent, largest_dot=largest_dot, smallest_dot=smallest_dot,
        norm=None, edge_color=edge_color, edge_lw=edge_lw, x_padding=None, y_padding=None, color_on="dot",
        **kwargs
    )

    return  plotter.dotplot()
