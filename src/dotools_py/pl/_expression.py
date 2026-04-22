from typing import Literal, Dict, Callable

import anndata as ad

import matplotlib.pyplot as plt
from matplotlib.colors import Colormap


from dotools_py.pl._Classes import BaseSeaborn
from dotools_py._custom_class import PathLike
from dotools_py.pl._plot_utils import save_plot, return_axis





def barplot(
    # Data
    adata: ad.AnnData,
    x_axis: str,
    feature: str,
    batch_key: str = "batch",
    hue: str = None,
    hue_order: list = None,
    layer: str = None,
    logcounts: bool =True,

    # Figure Parameters
    figsize: tuple[float, float] = (3, 4.2),
    palette: str  | dict | Colormap = "tab10",
    title: str = None,
    title_fontproperties: Dict[Literal["size", "weight"], str | int] = None,
    xticks_order: list = None,
    xticks_rotation: int = 45,
    ylabel: str = "LogMean(nUMI)",
    ylim_max: float = None,

    # Legend Parameters
    legend_title: str = None,
    legend_fontproperties: Dict[Literal["size", "weight"], str | int] = None,
    legend_ncols: int = 1,
    legend_loc: Literal["center left", "cemter right", "upper right", "upper left", "lower left", "lower right", "right", "lower center", "upper center", "center"] = 'center left',

    # IO
    path: PathLike = None,
    filename: str = "barplot.svg",
    show: bool = True,
    ax: plt.Axes = None,

    # Statistics
    reference: str = None,
    groups: str | list = None,
    groups_pvals: float | list = None,
    test: Literal["wilcoxon", "t-test", "kruskal", "anova", "logreg", "t-test_overestim_var"] = "wilcoxon",
    corr_method: Literal["benjamini-hochberg", "bonferroni"] = "benjamini-hochberg",
    line_offset: float = 0.05,
    txt_size: int = 13,
    txt: str = "p = ",

    # Fx Specific
    capsize: float = 0.1,
    marker_size: int = 6,
    estimator: Literal["logmean", "mean", "median"] | Callable = "logmean",
    **kwargs
)-> plt.Axes | dict | None:
    """Barplot with statistical significance.

    Show the average expression of features in `adata.var_names` or a continuous value in `adata.obs` along different
    categorical values and test for significance. The mean pseudo-bulk expression per sample will be plotted as dots.


    :param adata: Annotated data matrix.
    :param x_axis: Name of a categorical column in `adata.obs` to groupby.
    :param feature: A valid feature in `adata.var_names` or column in `adata.obs` with continuous values.
    :param batch_key: Name of a categorical column in `adata.obs` that contains the sample names.
    :param hue: Name of a second categorical column in `adata.obs` to use additionally to groupby.
    :param hue_order: List with orders for the categories in `hue`. If it is not set, the order will be inferred.
    :param layer: Name of the AnnData object layer that wants to be plotted. By default, `adata.X` is plotted.
                 If layer is set to a valid layer name, then the layer is plotted.
    :param logcounts: If set to `True`, the log-transformed mean will be shown (i.e, LogMean(nUMI)), otherwise the Mean(nUMI) is shown.
    :param figsize: Figure size, the format is (width, height).
    :param palette:  String denoting matplotlib colormap.  If not set, it will try to access `adata.uns[hue_colors | x_axis_colors]`, if not
                    the colormap `do.utility.tab30()` will be used. A dictionary with the categories available in `adata.obs[x_axis]` or `adata.obs[hue]`
                    if hue is not None can also be provided. The format is {category:color}.
    :param title: Title for the figure.
    :param title_fontproperties: Dictionary which should contain 'size' and 'weight' to define the fontsize and fontweight of the title of the figure.
    :param xticks_order: Order for the categories in `adata.obs[x_axis]`.
    :param xticks_rotation: Rotation of the X-axis ticks.
    :param ylabel: Label for the Y-axis.
    :param ylim_max: Set the maximum limit of the Y-axis to this value.
    :param legend_title: Title for the legend.
    :param legend_fontproperties: Dictionary which should contain 'size' and 'weight' to define the fontsize and fontweight of the title of the legend.
    :param legend_ncols:  Number of columns for the legend.
    :param legend_loc:  Location of the legend.
    :param path: Path to the folder to save the figure.
    :param filename: Name of file to use when saving the figure.
    :param show: If set to `False`, returns a dictionary with the matplotlib axes.
    :param ax: Matplotlib axes to use for plotting. If not set, a new figure will be generated.
    :param reference: Reference condition to use when testing for significance. When `hue` is set, the reference
                      condition correspond to the categories in `hue`. For each `x_axis` category the different hue
                      categories will be tested.
    :param groups: List of the name of the groups to test against.
    :param groups_pvals: If provided, these values will be plotted. If not set, the p-values will be estimated.
                        The order of the p-values should match the order of the `groups_cond` categories.
    :param test: Name of the method to test for significance.
    :param corr_method: Correction method for multiple testing.
    :param line_offset: Offset for the brackets draw to indicate significance. This offset represent a percentage.
    :param txt_size: Font size of the text indicating significance.
    :param txt: Text to include before the p-value. If not set, only the p-value is shown.
    :param capsize: Width of the `caps` on error bars, relative to bar spacing.
    :param marker_size: Radius of the markers, in points.
    :param estimator: Statistical function to estimate within each categorical bin. If set to `logmean` the mean
                     will be performed on the un-transformed logarithmize data. After calculating the mean,
                     the mean will be log1p transform if `logcounts` is set to `True`. It can also accept a custom function.
    :param kwargs: Other parameters are passed through to `sns.barplot <https://seaborn.pydata.org/generated/seaborn.barplot.html>`_.
    :return: Depending on ``show``, returns the plot if set to `True` or a dictionary with the axes.

    Example
    -------
    Create a barplot showing the mean expression of a given gene including the p-value to indicate if there is
    a significant statistical difference between groups.

    .. plot::
        :context: close-figs

        import dotools_py as do
        adata = do.dt.example_10x_processed()
        do.pl.barplot(adata,  'annotation', 'CD4', reference = 'pDC', groups=['B_cells'], xticks_rotation=45, ylim_max=2)

    Setting the `hue` argument allow to test across conditions for several groups.

    .. plot::
        :context: close-figs

        # Take only lymphoid cells
        lymphoid = adata[adata.obs['annotation'].isin(['T_cells', 'NK', 'B_cells'])].copy()
        do.pl.barplot(lymphoid,'annotation','CD4',  hue = 'condition', reference = 'healthy', groups=['disease'], hue_order=['healthy', 'disease'], xticks_rotation=45, figsize=(6, 4))

    Plot a continuous value in `adata.obs`.

    .. plot::
        :context: close-figs

        do.pl.barplot(adata,'annotation','total_counts', figsize=(6, 4), show=False)

    """

    plotter = BaseSeaborn(
        # Data and calculation
        adata=adata, feature=feature, batch_key=batch_key, x_axis=x_axis,  hue=hue, layer=layer,
        log1p_data=logcounts, pseudobulk = False,
        # Figure parameters
        figsize=figsize, ax=ax, cmap=palette,
        # Layout
        title=title, title_fontproperties=title_fontproperties,
        legend_title=legend_title, legend_properties=legend_fontproperties, legend_ncols=legend_ncols,
        legend_loc=legend_loc,
        xticks_order=xticks_order, xticks_properties={"rotation": xticks_rotation},
        hue_order=hue_order,
        # Statistics
        reference=reference, groups=groups, groups_pvals=groups_pvals, test=test,
        corr_method=corr_method, line_offset=line_offset, txt_size=txt_size, txt=txt
    )
    axis = plotter.barplot(
        estimator=estimator, capsize=capsize, marker_size=marker_size, ylabel=ylabel, ylim_max=ylim_max, **kwargs
    )
    axis = axis if len(axis) != 1 else axis["mainplot_ax"]

    save_plot(path, filename)
    return return_axis(show, axis=axis)




def boxplot(
    # Data
    adata: ad.AnnData,
    x_axis: str,
    feature: str,
    batch_key: str = "batch",
    hue: str = None,
    hue_order: list = None,
    layer: str = None,
    pseudobulk: bool = False,

    # Figure Parameters
    figsize: tuple[float, float] = (3, 4.2),
    palette: str  | dict | Colormap = "tab10",
    title: str = None,
    title_fontproperties: Dict[Literal["size", "weight"], str | int] = None,
    xticks_order: list = None,
    xticks_rotation: int = 45,
    ylabel: str = "Log(nUMI)",

    # Legend Parameters
    legend_title: str = None,
    legend_fontproperties: Dict[Literal["size", "weight"], str | int] = None,
    legend_ncols: int = 1,
    legend_loc: Literal["center left", "cemter right", "upper right", "upper left", "lower left", "lower right", "right", "lower center", "upper center", "center"] = 'center left',

    # IO
    path: PathLike = None,
    filename: str = "barplot.svg",
    show: bool = True,
    ax: plt.Axes = None,

    # Statistics
    reference: str = None,
    groups: str | list = None,
    groups_pvals: float | list = None,
    test: Literal["wilcoxon", "t-test", "kruskal", "anova", "logreg", "t-test_overestim_var"] = "wilcoxon",
    corr_method: Literal["benjamini-hochberg", "bonferroni"] = "benjamini-hochberg",
    line_offset: float = 0.05,
    txt_size: int = 13,
    txt: str = "p = ",

    # Fx Specific
    showfliers: bool = False,
    scatter: bool = False,
    marker_size: float = 2,
    **kwargs
) -> plt.Axes | dict | None:
    """Boxplot with statistical significance.

    Show the distribution of the  expression of `var_names` or a continuous value in `obs` along different categorical
    values and test for significance.

    :param adata: Annotated data matrix.
    :param x_axis: Name of a categorical column in `adata.obs` to groupby.
    :param feature: A valid feature in `adata.var_names` or column in `adata.obs` with continuous values.
    :param batch_key: Name of a categorical column in `adata.obs` that contains the sample names.
    :param hue: Name of a second categorical column in `adata.obs` to use additionally to groupby.
    :param hue_order: List with orders for the categories in `hue`. If it is not set, the order will be inferred.
    :param layer: Name of the AnnData object layer that wants to be plotted. By default, `adata.X` is plotted.
              If layer is set to a valid layer name, then the layer is plotted.
    :param pseudobulk: If set to `True` the distribution of the mean across samples will be plotted.
    :param figsize: Figure size, the format is (width, height).
    :param palette:  String denoting matplotlib colormap.  If not set, it will try to access `adata.uns[hue_colors | x_axis_colors]`, if not
                 the colormap `do.utility.tab30()` will be used. A dictionary with the categories available in `adata.obs[x_axis]` or `adata.obs[hue]`
                 if hue is not None can also be provided. The format is {category:color}.
    :param title: Title for the figure.
    :param title_fontproperties: Dictionary which should contain 'size' and 'weight' to define the fontsize and fontweight of the title of the figure.
    :param xticks_order: Order for the categories in `adata.obs[x_axis]`.
    :param xticks_rotation: Rotation of the X-axis ticks.
    :param ylabel: Label for the Y-axis.
    :param legend_title: Title for the legend.
    :param legend_fontproperties: Dictionary which should contain 'size' and 'weight' to define the fontsize and fontweight of the title of the legend.
    :param legend_ncols:  Number of columns for the legend.
    :param legend_loc:  Location of the legend.
    :param path: Path to the folder to save the figure.
    :param filename: Name of file to use when saving the figure.
    :param show: If set to `False`, returns a dictionary with the matplotlib axes.
    :param ax: Matplotlib axes to use for plotting. If not set, a new figure will be generated.
    :param reference: Reference condition to use when testing for significance. When `hue` is set, the reference
                   condition correspond to the categories in `hue`. For each `x_axis` category the different hue
                   categories will be tested.
    :param groups: List of the name of the groups to test against.
    :param groups_pvals: If provided, these values will be plotted. If not set, the p-values will be estimated.
                     The order of the p-values should match the order of the `groups_cond` categories.
    :param test: Name of the method to test for significance.
    :param corr_method: Correction method for multiple testing.
    :param line_offset: Offset for the brackets draw to indicate significance. This offset represent a percentage.
    :param txt_size: Font size of the text indicating significance.
    :param txt: Text to include before the p-value. If not set, only the p-value is shown.
    :param showfliers: Show the outliers beyond the caps.
    :param scatter: Plot the mean expression per sample on top of the boxplots plots.
    :param marker_size: Radius of the markers, in points.
    :param kwargs: Other parameters are passed through to `sns.boxplot <https://seaborn.pydata.org/generated/seaborn.boxplot.html>`_.
    :return: Depending on ``show``, returns the plot if set to `True` or a dictionary with the axes.

    Example
    -------
    Create a boxplot showing the expression of a given gene including the p-value to indicate if there is
    a significant statistical difference between groups.

    .. plot::
        :context: close-figs

        import dotools_py as do
        adata = do.dt.example_10x_processed()
        do.pl.boxplot(adata,  'annotation', 'CD4', reference = 'pDC', groups=['B_cells'], xticks_rotation=45, scatter=False)

    Setting the `hue` argument allow to test across conditions for several groups.

    .. plot::
        :context: close-figs

        # Take only lymphoid cells
        lymphoid = adata[adata.obs['annotation'].isin(['T_cells', 'NK', 'B_cells'])].copy()
        do.pl.boxplot(lymphoid, 'annotation', 'RPL11', hue = 'condition', reference = 'healthy', groups=['disease'], hue_order=['healthy', 'disease'], xticks_rotation=45, figsize=(6, 4), scatter=True)

    Plot a continuous value in `adata.obs`.

    .. plot::
        :context: close-figs

        do.pl.boxplot(adata,'annotation','total_counts', figsize=(6, 4), scatter=True)

    Plot over the sample level

    .. plot::
        :context: close-figs
        do.pl.boxplot(adata, "condition", "RPL11", batch_key="annotation", pseudobulk=True, scatter=True, marker_size=5)


    """

    plotter = BaseSeaborn(
        # Data and calculation
        adata=adata, feature=feature, batch_key=batch_key, x_axis=x_axis, hue=hue, layer=layer,
        log1p_data=True, pseudobulk=pseudobulk,
        # Figure parameters
        figsize=figsize, ax=ax, cmap=palette,
        # Layout
        title=title, title_fontproperties=title_fontproperties,
        legend_title=legend_title, legend_properties=legend_fontproperties, legend_ncols=legend_ncols,
        legend_loc=legend_loc,
        xticks_order=xticks_order, xticks_properties={"rotation": xticks_rotation},
        hue_order=hue_order,
        # Statistics
        reference=reference, groups=groups, groups_pvals=groups_pvals, test=test,
        corr_method=corr_method, line_offset=line_offset, txt_size=txt_size, txt=txt
    )
    axis = plotter.boxplot(showfliers=showfliers, scatter=scatter, marker_size=marker_size, ylabel=ylabel, **kwargs)
    axis = axis if len(axis) != 1 else axis["mainplot_ax"]

    save_plot(path, filename)
    return return_axis(show, axis=axis)




def violinplot(
    # Data
    adata: ad.AnnData,
    x_axis: str,
    feature: str,
    hue: str = None,
    hue_order: list = None,
    layer: str = None,

    # Figure Parameters
    figsize: tuple[float, float] = (3, 4.2),
    palette: str | dict | Colormap = "tab10",
    title: str = None,
    title_fontproperties: Dict[Literal["size", "weight"], str | int] = None,
    xticks_order: list = None,
    xticks_rotation: int = 45,
    ylabel: str = "Log(nUMI)",

    # Legend Parameters
    legend_title: str = None,
    legend_fontproperties: Dict[Literal["size", "weight"], str | int] = None,
    legend_ncols: int = 1,
    legend_loc: Literal[
        "center left", "cemter right", "upper right", "upper left", "lower left", "lower right", "right", "lower center", "upper center", "center"] = 'center left',

    # IO
    path: PathLike = None,
    filename: str = "barplot.svg",
    show: bool = True,
    ax: plt.Axes = None,

    # Statistics
    reference: str = None,
    groups: str | list = None,
    groups_pvals: float | list = None,
    test: Literal["wilcoxon", "t-test", "kruskal", "anova", "logreg", "t-test_overestim_var"] = "wilcoxon",
    corr_method: Literal["benjamini-hochberg", "bonferroni"] = "benjamini-hochberg",
    line_offset: float = 0.05,
    txt_size: int = 13,
    txt: str = "p = ",

    # Fx Specific
    scatter: bool = False,
    marker_size: int = 2,
    cut: float = 0,

    **kwargs
)-> plt.Axes | dict | None:
    """Violin plot with statistical significance.

    Show the distribution of the  expression of `var_names` or a continuous value in `obs` along different categorical
    values and test for significance.

    :param adata: Annotated data matrix.
    :param x_axis: Name of a categorical column in `adata.obs` to groupby.
    :param feature: A valid feature in `adata.var_names` or column in `adata.obs` with continuous values.
    :param hue: Name of a second categorical column in `adata.obs` to use additionally to groupby.
    :param hue_order: List with orders for the categories in `hue`. If it is not set, the order will be inferred.
    :param layer: Name of the AnnData object layer that wants to be plotted. By default, `adata.X` is plotted.
            If layer is set to a valid layer name, then the layer is plotted.
    :param figsize: Figure size, the format is (width, height).
    :param palette:  String denoting matplotlib colormap.  If not set, it will try to access `adata.uns[hue_colors | x_axis_colors]`, if not
               the colormap `do.utility.tab30()` will be used. A dictionary with the categories available in `adata.obs[x_axis]` or `adata.obs[hue]`
               if hue is not None can also be provided. The format is {category:color}.
    :param title: Title for the figure.
    :param title_fontproperties: Dictionary which should contain 'size' and 'weight' to define the fontsize and fontweight of the title of the figure.
    :param xticks_order: Order for the categories in `adata.obs[x_axis]`.
    :param xticks_rotation: Rotation of the X-axis ticks.
    :param ylabel: Label for the Y-axis.
    :param legend_title: Title for the legend.
    :param legend_fontproperties: Dictionary which should contain 'size' and 'weight' to define the fontsize and fontweight of the title of the legend.
    :param legend_ncols:  Number of columns for the legend.
    :param legend_loc:  Location of the legend.
    :param path: Path to the folder to save the figure.
    :param filename: Name of file to use when saving the figure.
    :param show: If set to `False`, returns a dictionary with the matplotlib axes.
    :param ax: Matplotlib axes to use for plotting. If not set, a new figure will be generated.
    :param reference: Reference condition to use when testing for significance. When `hue` is set, the reference
                 condition correspond to the categories in `hue`. For each `x_axis` category the different hue
                 categories will be tested.
    :param groups: List of the name of the groups to test against.
    :param groups_pvals: If provided, these values will be plotted. If not set, the p-values will be estimated.
                   The order of the p-values should match the order of the `groups_cond` categories.
    :param test: Name of the method to test for significance.
    :param corr_method: Correction method for multiple testing.
    :param line_offset: Offset for the brackets draw to indicate significance. This offset represent a percentage.
    :param txt_size: Font size of the text indicating significance.
    :param txt: Text to include before the p-value. If not set, only the p-value is shown.
    :param scatter: Plot the mean expression per sample on top of the boxplots plots.
    :param marker_size: Radius of the markers, in points.
    :param cut: Distance, in units of bandwidth, to extend the density past extreme datapoints. Set to 0 to limit the violin within the data range.
    :param kwargs: Other parameters are passed through to `sns.violinplot <https://seaborn.pydata.org/generated/seaborn.violinplot.html>`_.
    :return: Depending on ``show``, returns the plot if set to `True` or a dictionary with the axes.

    Example
    -------
    Create a violin plot showing the expression of a given gene including the p-value to indicate if there is
    a significant statistical difference between groups.

    .. plot::
        :context: close-figs

        import dotools_py as do
        adata = do.dt.example_10x_processed()
        do.pl.violinplot(adata,  'annotation', 'CD4', reference = 'pDC', groups=['B_cells'], xticks_rotation=45, scatter=True)

    Setting the `hue` argument allow to test across conditions for several groups.

    .. plot::
        :context: close-figs

        # Take only lymphoid cells
        lymphoid = adata[adata.obs['annotation'].isin(['T_cells', 'NK', 'B_cells'])].copy()
        do.pl.violinplot(lymphoid, 'annotation','CD4',  hue = 'condition',   reference = 'healthy', groups='disease', hue_order=['healthy', 'disease'], xticks_rotation=45, figsize=(6, 4))


    Plot a continuous value in `adata.obs`.

    .. plot::
        :context: close-figs

        do.pl.violinplot(adata,'annotation','total_counts', figsize=(6, 4), scatter=True)
    """

    plotter = BaseSeaborn(
        # Data and calculation
        adata=adata, feature=feature, batch_key=None, x_axis=x_axis, hue=hue, layer=layer,
        log1p_data=True, pseudobulk=False,
        # Figure parameters
        figsize=figsize, ax=ax, cmap=palette,
        # Layout
        title=title, title_fontproperties=title_fontproperties,
        legend_title=legend_title, legend_properties=legend_fontproperties, legend_ncols=legend_ncols,
        legend_loc=legend_loc,
        xticks_order=xticks_order, xticks_properties={"rotation": xticks_rotation},
        hue_order=hue_order,
        # Statistics
        reference=reference, groups=groups, groups_pvals=groups_pvals, test=test,
        corr_method=corr_method, line_offset=line_offset, txt_size=txt_size, txt=txt
    )
    axis = plotter.violinplot(scatter=scatter, marker_size=marker_size, cut=cut, ylabel=ylabel, **kwargs)
    axis = axis if len(axis) != 1 else axis["mainplot_ax"]

    save_plot(path, filename)
    return return_axis(show, axis=axis)









# @_doc_params(COMMON_ARGS=COMMON_EXPR_ARGS)
# def barplot(
#     # Data
#     adata: ad.AnnData,
#     x_axis: str,
#     feature: str,
#     batch_key: str = "batch",
#     hue: str = None,
#     hue_order: list = None,
#     layer: str = None,
#     logcounts: bool =True,
#
#     # Figure Parameters
#     figsize: tuple[float, float] = (3, 4.2),
#     palette: str  | dict | Colormap = "tab10",
#     title: str = None,
#     title_fontproperties: Dict[Literal["size", "weight"], str | int] = None,
#     xticks_order: list = None,
#     xticks_rotation: int = 45,
#     ylabel: str = "LogMean(nUMI)",
#     ylim_max: float = None,
#
#     # Legend Parameters
#     legend_title: str = None,
#     legend_fontproperties: Dict[Literal["size", "weight"], str | int] = None,
#     legend_ncols: int = 1,
#     legend_loc: Literal["center left", "cemter right", "upper right", "upper left", "lower left", "lower right", "right", "lower center", "upper center", "center"] = 'center left',
#
#     # IO
#     path: PathLike = None,
#     filename: str = "barplot.svg",
#     show: bool = True,
#     ax: plt.Axes = None,
#
#     # Statistics
#     reference: str = None,
#     groups: str | list = None,
#     groups_pvals: float | list = None,
#     test: Literal["wilcoxon", "t-test", "kruskal", "anova", "logreg", "t-test_overestim_var"] = "wilcoxon",
#     corr_method: Literal["benjamini-hochberg", "bonferroni"] = "benjamini-hochberg",
#     line_offset: float = 0.05,
#     txt_size: int = 13,
#     txt: str = "p = ",
#
#     # Fx Specific
#     capsize: float = 0.1,
#     marker_size: int = 6,
#     estimator: Literal["logmean", "mean", "median"] = "logmean",
#     **kwargs
# ) -> plt.Axes | dict | None:
#     """Barplot with statistical significance.
#
#     Show the average expression of features in `adata.var_names` or a continuous value in `adata.obs` along different
#     categorical values and test for significance. The mean pseudo-bulk expression per sample will be plotted as dots.
#
#     Parameters
#     ----------
#     {COMMON_ARGS}
#     batch_key:
#         Name of a categorical column in `adata.obs` that contains the sample names.
#     logcounts:
#         If set to `True`, consider that the values in `adata.X` or `adata.layers[layer]` if layer is set is log1p
#         transformed.
#     ylim_max:
#         Set the maximum limit of the Y-axis to this value.
#     capsize:
#         Width of the `caps` on error bars, relative to bar spacing.
#     marker_size:
#         Radius of the markers, in points.
#     estimator:
#         Statistical function to estimate within each categorical bin. If set to `LogMean` the mean will be performed
#         on the un-transformed logarithmize data. After calculating the mean, the mean will be log1p transform.
#     kwargs:
#         Other parameters are passed through to `sns.barplot <https://seaborn.pydata.org/generated/seaborn.barplot.html>`_.
#
#     Returns
#     -------
#     Depending on ``show``, returns the plot if set to `True` or a dictionary with the axes.
#
#     Example
#     -------
#     Create a barplot showing the mean expression of a given gene including the p-value to indicate if there is
#     a significant statistical difference between groups.
#
#     .. plot::
#         :context: close-figs
#
#         import dotools_py as do
#         adata = do.dt.example_10x_processed()
#         do.pl.barplot(adata,  'annotation', 'CD4', reference = 'pDC', groups=['B_cells'], xticks_rotation=45)
#
#     Setting the `hue` argument allow to test across conditions for several groups.
#
#     .. plot::
#         :context: close-figs
#
#         # Take only lymphoid cells
#         lymphoid = adata[adata.obs['annotation'].isin(['T_cells', 'NK', 'B_cells'])].copy()
#         do.pl.barplot(lymphoid,'annotation','CD4',  hue = 'condition', reference = 'healthy', groups=['disease'], hue_order=['healthy', 'disease'], xticks_rotation=45, figsize=(6, 4))
#
#     Plot a continuous value in `adata.obs`.
#
#     .. plot::
#         :context: close-figs
#
#         do.pl.barplot(adata,'annotation','total_counts', figsize=(6, 4))
#
#
#     """
#     import numpy as np
#
#     def log_estimator(values):
#         values = np.array(values, dtype=float)  # ensure numeric
#         if len(values) == 0:
#             return np.nan
#         return np.log1p(np.mean(np.expm1(values)))
#
#     plotter = BaseSeaborn(
#         adata=adata, x_axis=x_axis, feature=feature, batch_key=batch_key, xticks_order=xticks_order, hue=hue,
#         hue_order=hue_order, layer=layer, logcounts=logcounts, figsize=figsize, ax=ax, cmap=palette, show=show,
#         title=title, title_fontproperties=title_fontproperties,  xticks_properties={"rotation": xticks_rotation},
#         legend_properties=legend_fontproperties, path=path, filename=filename,
#     )
#
#     # Extract the data required for plotting
#     df = plotter.get_expression(keep=[x_axis, hue] if hue is not None else [x_axis])
#     df_batch = plotter.get_mean_expression()
#
#     # Create figure
#     nrows, ncols = (1, 1) if hue is None else (1, 2)
#     plotter.make_figure(nrows=nrows, ncols=ncols)
#     main_axis = plotter.fig.add_subplot(plotter.gs[0])
#     if all(feature in list(plotter.adata.obs.columns) for feature in plotter.feature):
#         estimator = "mean" if estimator == "logmean" else estimator
#         logger.warn("Feature in adata.obs but logcounts set to True, changing estimator to mean")
#
#     if estimator == "logmean" and logcounts ==False:
#         raise ValueError("If logcounts is set to `False`, estimator cannot be set to 'logmean' ")
#
#     if estimator == "logmean":
#         bp = sns.barplot(
#             df, x=plotter.x_axis, y="expr", estimator=log_estimator,
#             capsize=capsize, ax=main_axis, palette=plotter.cmap,
#             hue=plotter.hue, order=plotter.xticks_order, hue_order=plotter.hue_order, legend=False, **kwargs
#              )
#     else:
#         bp = sns.barplot(
#             df, x=plotter.x_axis, y="expr", estimator=estimator,
#             capsize=capsize, ax=main_axis, palette=plotter.cmap,
#             hue=plotter.hue, order=plotter.xticks_order, hue_order=plotter.hue_order, legend=False, **kwargs
#         )
#
#     sns.stripplot(
#         df_batch, x=plotter.x_axis, y="expr", alpha=0.75, color="k", s=marker_size, ax=bp, hue=plotter.hue,
#         hue_order=plotter.hue_order, order=plotter.xticks_order, dodge= True if hue else False, legend=False
#     )
#
#     # Statistical Testing
#     groups_cond = iterase_input(groups)
#     groups_pvals = iterase_input(groups_pvals)
#
#     if reference is not None and len(groups_cond) != 0:
#         if len(groups_pvals) == 0:
#             testing = TestData(
#                 data=adata, feature=feature, cond_key=x_axis if hue is None else hue, ctrl=reference,
#                 groups=groups_cond, category_key=None if hue is None else x_axis,
#                 category_order=None if hue is None else plotter.xticks_order, test=test, test_correction=corr_method
#             )
#             testing.run_test()
#             groups_pvals = testing.pvals  # Should be the same order as for StatsPlotter
#             del testing
#         stats_plotter = StatsPlotter(
#             bp, x_axis=x_axis, y_axis="expr", ctrl=reference, groups=groups_cond, pvals=groups_pvals, txt_size=txt_size,
#             txt=txt, kind="bar", line_offset=line_offset, hue=hue, hue_order=hue_order,
#         )
#         stats_plotter.plot_stats()
#         del stats_plotter
#
#     # Set the Layout
#     plotter.set_xticks(ax=bp)
#     plotter.legend(show=show, title=legend_title)
#     plotter.set_title(ax=bp)
#     bp.set_xlabel("")
#     bp.set_ylabel(ylabel, fontweight="bold")
#
#     if len(adata.obs[batch_key].unique()) > 2:
#         ymax = df_batch["expr"].max() + df_batch["expr"].max() * 0.1
#         ymax = ylim_max if ylim_max is not None else ymax
#         bp.set_ylim(0, ymax)
#
#     # Add Legend if hue is not None
#     if hue is not None:
#         axs_legend = plotter.fig.add_subplot(plotter.gs[1])
#         handles = []
#         for lab, c in plotter.cmap_dict.items():
#             handles.append(
#                 mlines.Line2D(
#                     [0], [0], marker=".", color=c, lw=0, label=lab, markerfacecolor=c, markeredgecolor=None,
#                     markersize=18
#                 ))
#
#         legend = axs_legend.legend(
#             handles=handles, frameon=False, loc=legend_loc, ncols=legend_ncols, title=legend_title,
#             prop={"size": plotter.legend_fontsize, "weight": plotter.legend_title_fontweight},
#         )
#         legend.get_title().set_fontweight("bold")
#         legend.get_title().set_fontsize(plotter.legend_fontsize + 2)
#         axs_legend.tick_params(
#             axis="both", left=False, labelleft=False, labelright=False, bottom=False, labelbottom=False)
#         axs_legend.spines[["right", "left", "top", "bottom"]].set_visible(False)
#         axs_legend.grid(visible=False)
#         plotter.dict_axis = {"mainplot_ax": bp, "legend_ax": axs_legend}
#     else:
#         plotter.dict_axis = bp
#
#     return plotter.saving_return_axis()
#
#
# @_doc_params(COMMON_ARGS=COMMON_EXPR_ARGS)
# def boxplot(
#     # Data
#     adata: ad.AnnData,
#     x_axis: str,
#     feature: str,
#     hue: str = None,
#     hue_order: list = None,
#     layer: str = None,
#
#     # Figure Parameters
#     figsize: tuple[float, float] = (3, 4.2),
#     palette: str  | dict | Colormap = "tab10",
#     title: str = None,
#     title_fontproperties: Dict[Literal["size", "weight"], str | int] = None,
#     xticks_order: list = None,
#     xticks_rotation: int = 45,
#     ylabel: str = "LogMean(nUMI)",
#
#     # Legend Parameters
#     legend_title: str = None,
#     legend_ncols: int = 1,
#     legend_fontproperties: Dict[Literal["size", "weight"], str | int] = None,
#     legend_loc: Literal["center left", "cemter right", "upper right", "upper left", "lower left", "lower right", "right", "lower center", "upper center", "center"] = 'center left',
#
#     # IO
#     path: PathLike = None,
#     filename: str = "barplot.svg",
#     show: bool = True,
#     ax: plt.Axes = None,
#
#     # Statistics
#     reference: str = None,
#     groups: str | list = None,
#     groups_pvals: float | list = None,
#     test: Literal["wilcoxon", "t-test", "kruskal", "anova", "logreg", "t-test_overestim_var"] = "wilcoxon",
#     corr_method: Literal["benjamini-hochberg", "bonferroni"] = "benjamini-hochberg",
#     line_offset: float = 0.05,
#     txt_size: int = 13,
#     txt: str = "p = ",
#
#     # Fx Specific
#     showfliers: bool = False,
#     scatter: bool = False,
#     marker_size: float = 2,
#     **kwargs
# ) -> plt.Axes | dict | None:
#     """Boxplot with statistical significance.
#
#     Show the distribution of the  expression of `var_names` or a continuous value in `obs` along different categorical
#     values and test for significance.
#
#     Parameters
#     ----------
#     {COMMON_ARGS}
#     showfliers:
#         Show the outliers beyond the caps.
#     scatter:
#          Plot the mean expression per sample on top of the boxplots plots.
#     marker_size:
#         Radius of the dots.
#     kwargs:
#         Other parameters are passed through to `sns.boxplot <https://seaborn.pydata.org/generated/seaborn.boxplot.html>`_.
#
#     Returns
#     -------
#     Depending on ``show``, returns the plot if set to `True` or a dictionary with the axes.
#
#     Example
#     -------
#     Create a boxplot showing the expression of a given gene including the p-value to indicate if there is
#     a significant statistical difference between groups.
#
#     .. plot::
#         :context: close-figs
#
#         import dotools_py as do
#         adata = do.dt.example_10x_processed()
#         do.pl.boxplot(adata,  'annotation', 'CD4', reference = 'pDC', groups=['B_cells'], xticks_rotation=45, scatter=False)
#
#     Setting the `hue` argument allow to test across conditions for several groups.
#
#     .. plot::
#         :context: close-figs
#
#         # Take only lymphoid cells
#         lymphoid = adata[adata.obs['annotation'].isin(['T_cells', 'NK', 'B_cells'])].copy()
#         do.pl.boxplot(lymphoid, 'annotation', 'RPL11', hue = 'condition', reference = 'healthy', groups=['disease'], hue_order=['healthy', 'disease'], xticks_rotation=45, figsize=(6, 4), scatter=True)
#
#     Plot a continuous value in `adata.obs`.
#
#     .. plot::
#         :context: close-figs
#
#         do.pl.boxplot(adata,'annotation','total_counts', figsize=(6, 4), scatter=True)
#
#     """
#
#     plotter = BaseSeaborn(
#         adata=adata, x_axis=x_axis, feature=feature, xticks_order=xticks_order, hue=hue,
#         hue_order=hue_order, layer=layer, figsize=figsize, ax=ax, cmap=palette, show=show,
#         title=title, title_fontproperties=title_fontproperties, xticks_properties={"rotation": xticks_rotation},
#         legend_properties=legend_fontproperties, path=path, filename=filename,
#     )
#
#     # Extract the data required for plotting
#     df = plotter.get_expression(keep=[x_axis, hue] if hue is not None else [x_axis])
#
#     # Create figure
#     nrows, ncols = (1, 1) if hue is None else (1, 2)
#     plotter.make_figure(nrows=nrows, ncols=ncols)
#     main_axis = plotter.fig.add_subplot(plotter.gs[0])
#
#     bx = sns.boxplot(
#         df, x=plotter.x_axis, y="expr", showfliers=showfliers, ax=main_axis, palette=plotter.cmap,
#         order=plotter.xticks_order, hue=plotter.hue, hue_order=plotter.hue_order, legend=False, **kwargs
#     )
#
#     if scatter:
#         sns.stripplot(
#             df, x=plotter.x_axis, y="expr", ax=bx, color="k", order=plotter.xticks_order,
#             hue=plotter.hue, hue_order=plotter.hue_order, legend=False, size=marker_size, dodge=True
#         )
#
#
#     # Statistical testing
#     groups_cond = iterase_input(groups)
#     groups_pvals = iterase_input(groups_pvals)
#
#     if reference is not None and len(groups_cond) != 0:
#         if len(groups_pvals) == 0:
#             testing = TestData(
#                 data=adata, feature=feature, cond_key=x_axis if hue is None else hue, ctrl=reference,
#                 groups=groups_cond, category_key=None if hue is None else x_axis,
#                 category_order=None if hue is None else plotter.xticks_order, test=test, test_correction=corr_method
#             )
#             testing.run_test()
#             groups_pvals = testing.pvals  # Should be the same order as for StatsPlotter
#             del testing
#
#         stats_plotter = StatsPlotter(
#             bx, x_axis=x_axis, y_axis="expr", ctrl=reference, groups=groups_cond, pvals=groups_pvals, txt_size=txt_size,
#             txt=txt, kind="box", line_offset=line_offset, hue=hue, hue_order=hue_order,
#         )
#         stats_plotter.plot_stats()
#         del stats_plotter
#
#     # Set the Layout
#     plotter.set_xticks(ax=bx)
#     plotter.set_title(ax=bx)
#     plotter.legend(show=show, title=legend_title)
#     bx.set_xlabel("")
#     bx.set_ylabel(ylabel, fontweight="bold")
#
#     # Add Legend if hue is not None
#     if hue is not None:
#         axs_legend = plotter.fig.add_subplot(plotter.gs[1])
#         handles = []
#         for lab, c in plotter.cmap_dict.items():
#             handles.append(
#                 mlines.Line2D(
#                     [0], [0], marker=".", color=c, lw=0, label=lab, markerfacecolor=c, markeredgecolor=None,
#                     markersize=18
#                 ))
#
#         legend = axs_legend.legend(
#             handles=handles, frameon=False, loc=legend_loc, ncols=legend_ncols, title=legend_title,
#             prop={"size": plotter.legend_fontsize, "weight": plotter.legend_title_fontweight},
#         )
#         legend.get_title().set_fontweight("bold")
#         legend.get_title().set_fontsize(plotter.legend_fontsize + 2)
#         axs_legend.tick_params(
#             axis="both", left=False, labelleft=False, labelright=False, bottom=False, labelbottom=False)
#         axs_legend.spines[["right", "left", "top", "bottom"]].set_visible(False)
#         axs_legend.grid(visible=False)
#         plotter.dict_axis = {"mainplot_ax": bx, "legend_ax": axs_legend}
#     else:
#         plotter.dict_axis = bx
#
#     return plotter.saving_return_axis()
#
#
# @_doc_params(COMMON_ARGS=COMMON_EXPR_ARGS)
# def violinplot(
#     # Data
#     adata: ad.AnnData,
#     x_axis: str,
#     feature: str,
#     hue: str = None,
#     hue_order: list = None,
#     layer: str = None,
#
#     # Figure Parameters
#     figsize: tuple[float, float] = (3, 4.2),
#     palette: str  | dict | Colormap = "tab10",
#     title: str = None,
#     title_fontproperties: Dict[Literal["size", "weight"], str | int] = None,
#     xticks_order: list = None,
#     xticks_rotation: int = 45,
#     ylabel: str = "LogMean(nUMI)",
#
#     # Legend Parameters
#     legend_title: str = None,
#     legend_fontproperties: Dict[Literal["size", "weight"], str | int] = None,
#     legend_ncols: int = 1,
#     legend_loc: Literal["center left", "cemter right", "upper right", "upper left", "lower left", "lower right", "right", "lower center", "upper center", "center"] = 'center left',
#
#     # IO
#     path: PathLike = None,
#     filename: str = "barplot.svg",
#     show: bool = True,
#     ax: plt.Axes = None,
#
#     # Statistics
#     reference: str = None,
#     groups: str | list = None,
#     groups_pvals: float | list = None,
#     test: Literal["wilcoxon", "t-test", "kruskal", "anova", "logreg", "t-test_overestim_var"] = "wilcoxon",
#     corr_method: Literal["benjamini-hochberg", "bonferroni"] = "benjamini-hochberg",
#     line_offset: float = 0.05,
#     txt_size: int = 13,
#     txt: str = "p = ",
#
#     # Fx Specific
#     scatter: bool = False,
#     marker_size: int = 2,
#     cut: float = 0,
#
#     **kwargs
# ) -> plt.Axes | dict | None:
#     """Violin plot with statistical significance.
#
#     Show the distribution of the  expression of `var_names` or a continuous value in `obs` along different categorical
#     values and test for significance.
#
#     Parameters
#     ----------
#     {COMMON_ARGS}
#     scatter:
#          Plot non-zero values as dots on top of the violin plots.
#     marker_size:
#         Radius of the dots.
#     cut:
#         Distance, in units of bandwidth, to extend the density past extreme datapoints.
#         Set to 0 to limit the violin within the data range.
#     kwargs:
#         Other parameters are passed through to `sns.violinplot <https://seaborn.pydata.org/generated/seaborn.violinplot.html>`_.
#
#     Returns
#     -------
#     Depending on ``show``, returns the plot if set to `True` or a dictionary with the axes.
#
#     Example
#     -------
#     Create a violin plot showing the expression of a given gene including the p-value to indicate if there is
#     a significant statistical difference between groups.
#
#     .. plot::
#         :context: close-figs
#
#         import dotools_py as do
#         adata = do.dt.example_10x_processed()
#         do.pl.violinplot(adata,  'annotation', 'CD4', reference = 'pDC', groups=['B_cells'], xticks_rotation=45, scatter=True)
#
#     Setting the `hue` argument allow to test across conditions for several groups.
#
#     .. plot::
#         :context: close-figs
#
#         # Take only lymphoid cells
#         lymphoid = adata[adata.obs['annotation'].isin(['T_cells', 'NK', 'B_cells'])].copy()
#         do.pl.violinplot(lymphoid,'annotation','CD4',  hue = 'condition',   reference = 'healthy', groups=['disease'], hue_order=['healthy', 'disease'], xticks_rotation=45, figsize=(6, 4))
#
#     Plot a continuous value in `adata.obs`.
#
#     .. plot::
#         :context: close-figs
#
#         do.pl.violinplot(adata,'annotation','total_counts', figsize=(6, 4), scatter=True)
#
#     """
#
#     plotter = BaseSeaborn(
#         adata=adata, x_axis=x_axis, feature=feature, xticks_order=xticks_order, hue=hue,
#         hue_order=hue_order, layer=layer, figsize=figsize, ax=ax, cmap=palette, show=show,
#         title=title, title_fontproperties=title_fontproperties, xticks_properties={"rotation": xticks_rotation},
#         legend_properties=legend_fontproperties, path=path, filename=filename,
#     )
#
#     # Extract the data required for plotting
#     df = plotter.get_expression(keep=[x_axis, hue] if hue is not None else [x_axis])
#
#     # Create figure
#     nrows, ncols = (1, 1) if hue is None else (1, 2)
#     plotter.make_figure(nrows=nrows, ncols=ncols)
#     main_axis = plotter.fig.add_subplot(plotter.gs[0])
#
#     vln = sns.violinplot(
#         df, x=plotter.x_axis, y="expr", ax=main_axis, palette=plotter.cmap, cut=cut,
#         order=plotter.xticks_order, hue=plotter.hue, hue_order=plotter.hue_order, legend=False, **kwargs
#     )
#     if scatter:
#         sns.stripplot(
#             df[df.expr != 0], x=plotter.x_axis, y="expr", ax=vln, color="k", order=plotter.xticks_order,
#             hue=plotter.hue, hue_order=plotter.hue_order, legend=False, size=marker_size, dodge=True
#         )
#
#
#     # Statistical testing
#     groups_cond = iterase_input(groups)
#     groups_pvals = iterase_input(groups_pvals)
#
#     if reference is not None and len(groups_cond) != 0:
#         if len(groups_pvals) == 0:
#             testing = TestData(
#                 data=adata, feature=feature, cond_key=x_axis if hue is None else hue, ctrl=reference,
#                 groups=groups_cond, category_key=None if hue is None else x_axis,
#                 category_order=None if hue is None else plotter.xticks_order, test=test, test_correction=corr_method
#             )
#             testing.run_test()
#             groups_pvals = testing.pvals  # Should be the same order as for StatsPlotter
#             del testing
#
#         stats_plotter = StatsPlotter(
#             vln, x_axis=x_axis, y_axis="expr", ctrl=reference, groups=groups_cond, pvals=groups_pvals, txt_size=txt_size,
#             txt=txt, kind="violin", line_offset=line_offset, hue=hue, hue_order=hue_order,
#         )
#         stats_plotter.plot_stats()
#         del stats_plotter
#
#     # Set the Layout
#     plotter.set_xticks(ax=vln)
#     plotter.set_title(ax=vln)
#     plotter.legend(show=show, title=legend_title)
#     vln.set_xlabel("")
#     vln.set_ylabel(ylabel, fontweight="bold")
#
#     # Add Legend if hue is not None
#     if hue is not None:
#         axs_legend = plotter.fig.add_subplot(plotter.gs[1])
#         handles = []
#         for lab, c in plotter.cmap_dict.items():
#             handles.append(
#                 mlines.Line2D(
#                     [0], [0], marker=".", color=c, lw=0, label=lab, markerfacecolor=c, markeredgecolor=None,
#                     markersize=18
#                 ))
#
#         legend = axs_legend.legend(
#             handles=handles, frameon=False, loc=legend_loc, ncols=legend_ncols, title=legend_title,
#             prop={"size": plotter.legend_fontsize, "weight": plotter.legend_title_fontweight},
#         )
#         legend.get_title().set_fontweight("bold")
#         legend.get_title().set_fontsize(plotter.legend_fontsize + 2)
#         axs_legend.tick_params(
#             axis="both", left=False, labelleft=False, labelright=False, bottom=False, labelbottom=False)
#         axs_legend.spines[["right", "left", "top", "bottom"]].set_visible(False)
#         axs_legend.grid(visible=False)
#         plotter.dict_axis = {"mainplot_ax": vln, "legend_ax": axs_legend}
#     else:
#         plotter.dict_axis = vln
#
#     return plotter.saving_return_axis()
