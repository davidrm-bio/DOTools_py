from typing import Literal, Dict
from pathlib import Path

import anndata as ad

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
import seaborn as sns

from dotools_py import logger
from dotools_py.pl._StatsPlotter import TestData, StatsPlotter
from dotools_py.utils import iterase_input, logmean

from dotools_py.pl._plot_utils import COMMON_EXPR_ARGS, _doc_params
from dotools_py.pl._Classes import BaseSeaborn


@_doc_params(COMMON_ARGS=COMMON_EXPR_ARGS)
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
    xticks_rotation: int = None,
    ylabel: str = "LogMean(nUMI)",
    ylim_max: int = None,

    # Legend Parameters
    legend_title: str = None,
    legend_fontproperties: Dict[Literal["size", "weight"], str | int] = None,
    legend_ncols: int = 1,
    legend_loc: Literal["center left", "cemter right", "upper right", "upper left", "lower left", "lower right", "right", "lower center", "upper center", "center"] = 'center left',

    # IO
    path: str | Path = None,
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
    estimator: Literal["logmean", "mean", "median"] = "logmean",
    **kwargs
) -> plt.Axes | dict | None:
    """Barplot with statistical significance.

    Show the average expression of features in `adata.var_names` or a continuous value in `adata.obs` along different
    categorical values and test for significance. The mean pseudo-bulk expression per sample will be plotted as dots.

    Parameters
    ----------
    {COMMON_ARGS}
    batch_key:
        Name of a categorical column in `adata.obs` that contains the sample names.
    logcounts:
        If set to `True`, consider that the values in `adata.X` or `adata.layers[layer]` if layer is set is log1p
        transformed.
    ylim_max:
        Set the maximum limit of the Y-axis to this value.
    capsize:
        Width of the `caps` on error bars, relative to bar spacing.
    marker_size:
        Radius of the markers, in points.
    estimator:
        Statistical function to estimate within each categorical bin. If set to `LogMean` the mean will be performed
        on the un-transformed logarithmize data. After calculating the mean, the mean will be log1p transform.
    kwargs:
        Other parameters are passed through to `sns.barplot <https://seaborn.pydata.org/generated/seaborn.barplot.html>`_.

    Returns
    -------
    Depending on ``show``, returns the plot if set to `True` or a dictionary with the axes.

    Example
    -------
    Create a barplot showing the mean expression of a given gene including the p-value to indicate if there is
    a significant statistical difference between groups.

    .. plot::
        :context: close-figs

        import dotools_py as do
        adata = do.dt.example_10x_processed()
        do.pl.barplot(adata,  'annotation', 'CD4', reference = 'pDC', groups=['B_cells'], xticks_rotation=45)

    Setting the `hue` argument allow to test across conditions for several groups.

    .. plot::
        :context: close-figs

        # Take only lymphoid cells
        lymphoid = adata[adata.obs['annotation'].isin(['T_cells', 'NK', 'B_cells'])].copy()
        do.pl.barplot(lymphoid,'annotation','CD4',  hue = 'condition', reference = 'healthy', groups=['disease'], hue_order=['healthy', 'disease'], xticks_rotation=45, figsize=(6, 4))

    Plot a continuous value in `adata.obs`.

    .. plot::
        :context: close-figs

        do.pl.barplot(adata,'annotation','total_counts', figsize=(6, 4))


    """
    import numpy as np

    def log_estimator(values):
        values = np.array(values, dtype=float)  # ensure numeric
        if len(values) == 0:
            return np.nan
        return np.log1p(np.mean(np.expm1(values)))

    plotter = BaseSeaborn(
        adata=adata, x_axis=x_axis, feature=feature, batch_key=batch_key, xticks_order=xticks_order, hue=hue,
        hue_order=hue_order, layer=layer, logcounts=logcounts, figsize=figsize, ax=ax, cmap=palette, show=show,
        title=title, title_fontproperties=title_fontproperties,  xticks_properties={"rotation": xticks_rotation},
        legend_properties=legend_fontproperties, path=path, filename=filename,
    )

    # Extract the data required for plotting
    df = plotter.get_expression(keep=[x_axis, hue] if hue is not None else [x_axis])
    df_batch = plotter.get_mean_expression()

    # Create figure
    nrows, ncols = (1, 1) if hue is None else (1, 2)
    plotter.make_figure(nrows=nrows, ncols=ncols)
    main_axis = plotter.fig.add_subplot(plotter.gs[0])
    if all(feature in list(plotter.adata.obs.columns) for feature in plotter.feature):
        estimator = "mean" if estimator == "logmean" else estimator
        logger.warn("Feature in adata.obs but logcounts set to True, changing estimator to mean")

    if estimator == "logmean":
        bp = sns.barplot(
            df, x=plotter.x_axis, y="expr", estimator=log_estimator,
            capsize=capsize, ax=main_axis, palette=plotter.cmap,
            hue=plotter.hue, order=plotter.xticks_order, hue_order=plotter.hue_order, legend=False, **kwargs
             )
    else:
        bp = sns.barplot(
            df, x=plotter.x_axis, y="expr", estimator=estimator,
            capsize=capsize, ax=main_axis, palette=plotter.cmap,
            hue=plotter.hue, order=plotter.xticks_order, hue_order=plotter.hue_order, legend=False, **kwargs
        )

    sns.stripplot(
        df_batch, x=plotter.x_axis, y="expr", alpha=0.75, color="k", s=marker_size, ax=bp, hue=plotter.hue,
        hue_order=plotter.hue_order, order=plotter.xticks_order, dodge= True if hue else False, legend=False
    )

    # Statistical Testing
    groups_cond = iterase_input(groups)
    groups_pvals = iterase_input(groups_pvals)

    if reference is not None and len(groups_cond) != 0:
        if len(groups_pvals) == 0:
            testing = TestData(
                data=adata, feature=feature, cond_key=x_axis if hue is None else hue, ctrl=reference,
                groups=groups_cond, category_key=None if hue is None else x_axis,
                category_order=None if hue is None else plotter.xticks_order, test=test, test_correction=corr_method
            )
            testing.run_test()
            groups_pvals = testing.pvals  # Should be the same order as for StatsPlotter

        stats_plotter = StatsPlotter(
            bp, x_axis=x_axis, y_axis="expr", ctrl=reference, groups=groups_cond, pvals=groups_pvals, txt_size=txt_size,
            txt=txt, kind="bar", line_offset=line_offset, hue=hue, hue_order=hue_order,
        )
        stats_plotter.plot_stats()
        del stats_plotter, testing

    # Set the Layout
    plotter.set_xticks(ax=bp)
    plotter.legend(show=show, title=legend_title)
    plotter.set_title(ax=bp)
    bp.set_xlabel("")
    bp.set_ylabel(ylabel, fontweight="bold")

    if len(adata.obs[batch_key].unique()) > 2:
        ymax = df_batch["expr"].max() + df_batch["expr"].max() * 0.1
        ymax = ylim_max if ylim_max is not None else ymax
        bp.set_ylim(0, ymax)

    # Add Legend if hue is not None
    if hue is not None:
        axs_legend = plotter.fig.add_subplot(plotter.gs[1])
        handles = []
        for lab, c in plotter.cmap_dict.items():
            handles.append(
                mlines.Line2D(
                    [0], [0], marker=".", color=c, lw=0, label=lab, markerfacecolor=c, markeredgecolor=None,
                    markersize=18
                ))

        legend = axs_legend.legend(
            handles=handles, frameon=False, loc=legend_loc, ncols=legend_ncols, title=legend_title,
            prop={"size": plotter.legend_fontsize, "weight": plotter.legend_title_fontweight},
        )
        legend.get_title().set_fontweight("bold")
        legend.get_title().set_fontsize(plotter.legend_fontsize + 2)
        axs_legend.tick_params(
            axis="both", left=False, labelleft=False, labelright=False, bottom=False, labelbottom=False)
        axs_legend.spines[["right", "left", "top", "bottom"]].set_visible(False)
        axs_legend.grid(visible=False)
        plotter.dict_axis = {"mainplot_ax": bp, "legend_ax": axs_legend}
    else:
        plotter.dict_axis = bp

    return plotter.saving_return_axis()


@_doc_params(COMMON_ARGS=COMMON_EXPR_ARGS)
def boxplot(
    # Data
    adata: ad.AnnData,
    x_axis: str,
    feature: str,
    hue: str = None,
    hue_order: list = None,
    layer: str = None,

    # Figure Parameters
    figsize: tuple[float, float] = (3, 4.2),
    palette: str  | dict | Colormap = "tab10",
    title: str = None,
    title_fontproperties: Dict[Literal["size", "weight"], str | int] = None,
    xticks_order: list = None,
    xticks_rotation: int = None,
    ylabel: str = "LogMean(nUMI)",

    # Legend Parameters
    legend_title: str = None,
    legend_ncols: int = 1,
    legend_fontproperties: Dict[Literal["size", "weight"], str | int] = None,
    legend_loc: Literal["center left", "cemter right", "upper right", "upper left", "lower left", "lower right", "right", "lower center", "upper center", "center"] = 'center left',

    # IO
    path: str | Path = None,
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
    **kwargs
) -> plt.Axes | dict | None:
    """Boxplot with statistical significance.

    Show the distribution of the  expression of `var_names` or a continuous value in `obs` along different categorical
    values and test for significance.

    Parameters
    ----------
    {COMMON_ARGS}
    showfliers:
        Show the outliers beyond the caps.
    kwargs:
        Other parameters are passed through to `sns.boxplot <https://seaborn.pydata.org/generated/seaborn.boxplot.html>`_.

    Returns
    -------
    Depending on ``show``, returns the plot if set to `True` or a dictionary with the axes.

    Example
    -------
    Create a boxplot showing the expression of a given gene including the p-value to indicate if there is
    a significant statistical difference between groups.

    .. plot::
        :context: close-figs

        import dotools_py as do
        adata = do.dt.example_10x_processed()
        do.pl.boxplot(adata,  'annotation', 'CD4', reference = 'pDC', groups=['B_cells'], xticks_rotation=45)

    Setting the `hue` argument allow to test across conditions for several groups.

    .. plot::
        :context: close-figs

        # Take only lymphoid cells
        lymphoid = adata[adata.obs['annotation'].isin(['T_cells', 'NK', 'B_cells'])].copy()
        do.pl.boxplot(lymphoid, 'annotation', 'RPL11', hue = 'condition', reference = 'healthy', groups=['disease'], hue_order=['healthy', 'disease'], xticks_rotation=45, figsize=(6, 4))

    Plot a continuous value in `adata.obs`.

    .. plot::
        :context: close-figs

        do.pl.boxplot(adata,'annotation','total_counts', figsize=(6, 4))

    """

    plotter = BaseSeaborn(
        adata=adata, x_axis=x_axis, feature=feature, xticks_order=xticks_order, hue=hue,
        hue_order=hue_order, layer=layer, figsize=figsize, ax=ax, cmap=palette, show=show,
        title=title, title_fontproperties=title_fontproperties, xticks_properties={"rotation": xticks_rotation},
        legend_properties=legend_fontproperties, path=path, filename=filename,
    )

    # Extract the data required for plotting
    df = plotter.get_expression(keep=[x_axis, hue] if hue is not None else [x_axis])

    # Create figure
    nrows, ncols = (1, 1) if hue is None else (1, 2)
    plotter.make_figure(nrows=nrows, ncols=ncols)
    main_axis = plotter.fig.add_subplot(plotter.gs[0])

    bx = sns.boxplot(
        df, x=plotter.x_axis, y="expr", showfliers=showfliers, ax=main_axis, palette=plotter.cmap,
        order=plotter.xticks_order, hue=plotter.hue, hue_order=plotter.hue_order, legend=False, **kwargs
    )

    # Statistical testing
    groups_cond = iterase_input(groups)
    groups_pvals = iterase_input(groups_pvals)

    if reference is not None and len(groups_cond) != 0:
        if len(groups_pvals) == 0:
            testing = TestData(
                data=adata, feature=feature, cond_key=x_axis if hue is None else hue, ctrl=reference,
                groups=groups_cond, category_key=None if hue is None else x_axis,
                category_order=None if hue is None else plotter.xticks_order, test=test, test_correction=corr_method
            )
            testing.run_test()
            groups_pvals = testing.pvals  # Should be the same order as for StatsPlotter

        stats_plotter = StatsPlotter(
            bx, x_axis=x_axis, y_axis="expr", ctrl=reference, groups=groups_cond, pvals=groups_pvals, txt_size=txt_size,
            txt=txt, kind="box", line_offset=line_offset, hue=hue, hue_order=hue_order,
        )
        stats_plotter.plot_stats()
        del stats_plotter, testing

    # Set the Layout
    plotter.set_xticks(ax=bx)
    plotter.set_title(ax=bx)
    plotter.legend(show=show, title=legend_title)
    bx.set_xlabel("")
    bx.set_ylabel(ylabel, fontweight="bold")

    # Add Legend if hue is not None
    if hue is not None:
        axs_legend = plotter.fig.add_subplot(plotter.gs[1])
        handles = []
        for lab, c in plotter.cmap_dict.items():
            handles.append(
                mlines.Line2D(
                    [0], [0], marker=".", color=c, lw=0, label=lab, markerfacecolor=c, markeredgecolor=None,
                    markersize=18
                ))

        legend = axs_legend.legend(
            handles=handles, frameon=False, loc=legend_loc, ncols=legend_ncols, title=legend_title,
            prop={"size": plotter.legend_fontsize, "weight": plotter.legend_title_fontweight},
        )
        legend.get_title().set_fontweight("bold")
        legend.get_title().set_fontsize(plotter.legend_fontsize + 2)
        axs_legend.tick_params(
            axis="both", left=False, labelleft=False, labelright=False, bottom=False, labelbottom=False)
        axs_legend.spines[["right", "left", "top", "bottom"]].set_visible(False)
        axs_legend.grid(visible=False)
        plotter.dict_axis = {"mainplot_ax": bx, "legend_ax": axs_legend}
    else:
        plotter.dict_axis = bx

    return plotter.saving_return_axis()


@_doc_params(COMMON_ARGS=COMMON_EXPR_ARGS)
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
    palette: str  | dict | Colormap = "tab10",
    title: str = None,
    title_fontproperties: Dict[Literal["size", "weight"], str | int] = None,
    xticks_order: list = None,
    xticks_rotation: int = None,
    ylabel: str = "LogMean(nUMI)",

    # Legend Parameters
    legend_title: str = None,
    legend_fontproperties: Dict[Literal["size", "weight"], str | int] = None,
    legend_ncols: int = 1,
    legend_loc: Literal["center left", "cemter right", "upper right", "upper left", "lower left", "lower right", "right", "lower center", "upper center", "center"] = 'center left',

    # IO
    path: str | Path = None,
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
    scatter_size: int = 2,
    cut: float = 0,

    **kwargs
) -> plt.Axes | dict | None:
    """Violin plot with statistical significance.

    Show the distribution of the  expression of `var_names` or a continuous value in `obs` along different categorical
    values and test for significance.

    Parameters
    ----------
    {COMMON_ARGS}
    scatter:
         Plot non-zero values as dots on top of the violin plots.
    scatter_size:
        Radius of the dots.
    cut:
        Distance, in units of bandwidth, to extend the density past extreme datapoints.
        Set to 0 to limit the violin within the data range.
    kwargs:
        Other parameters are passed through to `sns.violinplot <https://seaborn.pydata.org/generated/seaborn.violinplot.html>`_.

    Returns
    -------
    Depending on ``show``, returns the plot if set to `True` or a dictionary with the axes.

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
        do.pl.violinplot(lymphoid,'annotation','CD4',  hue = 'condition',   reference = 'healthy', groups=['disease'], hue_order=['healthy', 'disease'], xticks_rotation=45, figsize=(6, 4))

    Plot a continuous value in `adata.obs`.

    .. plot::
        :context: close-figs

        do.pl.violinplot(adata,'annotation','total_counts', figsize=(6, 4), scatter=True)

    """

    plotter = BaseSeaborn(
        adata=adata, x_axis=x_axis, feature=feature, xticks_order=xticks_order, hue=hue,
        hue_order=hue_order, layer=layer, figsize=figsize, ax=ax, cmap=palette, show=show,
        title=title, title_fontproperties=title_fontproperties, xticks_properties={"rotation": xticks_rotation},
        legend_properties=legend_fontproperties, path=path, filename=filename,
    )

    # Extract the data required for plotting
    df = plotter.get_expression(keep=[x_axis, hue] if hue is not None else [x_axis])

    # Create figure
    nrows, ncols = (1, 1) if hue is None else (1, 2)
    plotter.make_figure(nrows=nrows, ncols=ncols)
    main_axis = plotter.fig.add_subplot(plotter.gs[0])

    vln = sns.violinplot(
        df, x=plotter.x_axis, y="expr", ax=main_axis, palette=plotter.cmap, cut=cut,
        order=plotter.xticks_order, hue=plotter.hue, hue_order=plotter.hue_order, legend=False, **kwargs
    )
    if scatter:
        sns.stripplot(
            df[df.expr != 0], x=plotter.x_axis, y="expr", ax=vln, color="k", order=plotter.xticks_order,
            hue=plotter.hue, hue_order=plotter.hue_order, legend=False, size=scatter_size, dodge=True
        )


    # Statistical testing
    groups_cond = iterase_input(groups)
    groups_pvals = iterase_input(groups_pvals)

    if reference is not None and len(groups_cond) != 0:
        if len(groups_pvals) == 0:
            testing = TestData(
                data=adata, feature=feature, cond_key=x_axis if hue is None else hue, ctrl=reference,
                groups=groups_cond, category_key=None if hue is None else x_axis,
                category_order=None if hue is None else plotter.xticks_order, test=test, test_correction=corr_method
            )
            testing.run_test()
            groups_pvals = testing.pvals  # Should be the same order as for StatsPlotter

        stats_plotter = StatsPlotter(
            vln, x_axis=x_axis, y_axis="expr", ctrl=reference, groups=groups_cond, pvals=groups_pvals, txt_size=txt_size,
            txt=txt, kind="violin", line_offset=line_offset, hue=hue, hue_order=hue_order,
        )
        stats_plotter.plot_stats()
        del stats_plotter, testing

    # Set the Layout
    plotter.set_xticks(ax=vln)
    plotter.set_title(ax=vln)
    plotter.legend(show=show, title=legend_title)
    vln.set_xlabel("")
    vln.set_ylabel(ylabel, fontweight="bold")

    # Add Legend if hue is not None
    if hue is not None:
        axs_legend = plotter.fig.add_subplot(plotter.gs[1])
        handles = []
        for lab, c in plotter.cmap_dict.items():
            handles.append(
                mlines.Line2D(
                    [0], [0], marker=".", color=c, lw=0, label=lab, markerfacecolor=c, markeredgecolor=None,
                    markersize=18
                ))

        legend = axs_legend.legend(
            handles=handles, frameon=False, loc=legend_loc, ncols=legend_ncols, title=legend_title,
            prop={"size": plotter.legend_fontsize, "weight": plotter.legend_title_fontweight},
        )
        legend.get_title().set_fontweight("bold")
        legend.get_title().set_fontsize(plotter.legend_fontsize + 2)
        axs_legend.tick_params(
            axis="both", left=False, labelleft=False, labelright=False, bottom=False, labelbottom=False)
        axs_legend.spines[["right", "left", "top", "bottom"]].set_visible(False)
        axs_legend.grid(visible=False)
        plotter.dict_axis = {"mainplot_ax": vln, "legend_ax": axs_legend}
    else:
        plotter.dict_axis = vln

    return plotter.saving_return_axis()
