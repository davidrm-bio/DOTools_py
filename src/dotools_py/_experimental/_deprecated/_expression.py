from typing import Literal
from pathlib import Path
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.lines as mlines

import dotools_py.utility
from dotools_py import logger
from dotools_py.pl._StatsPlotter import StatsPlotter, TestData
from dotools_py.get import mean_expr
from dotools_py.get._generic import expr as get_expr

from dotools_py.utils import convert_path, sanitize_anndata, make_grid_spec


def barplot(
    adata: ad.AnnData,
    x_axis: str,
    feature: str,
    batch_key: str = "batch",
    order: list = None,
    hue: str = None,
    hue_order: list = None,
    layer: str = None,
    figsize: tuple = (3, 4.2),
    palette: str | list | dict = "tab10",
    capsize: float = 0.1,
    xtick_rotation: int = None,
    ctrl_cond: str = None,
    groups_cond: str | list = None,
    groups_pvals: list = None,
    title: str = None,
    path: str | Path = None,
    filename: str = None,
    title_fontproperties: dict = None,
    hue_legend_fontsize: int = 12,
    hue_legend_fontweight: float | str = None,
    hue_legend_cols: int = 1,
    hue_legend_title: str = '',
    hue_legend_loc: Literal["center left", "cemter right", "upper right", "upper left", "lower left", "lower right", "right", "lower center", "upper center", "center"] = 'center left',
    show: bool = True,
    marker_size: int = 6,
    ax: plt.Axes = None,
    logcounts: bool = True,
    estimator: str | None = "LogMean",
    test: Literal["wilcoxon", "t-test", "kruskal", "anova", "logreg", "t-test_overestim_var"] = "wilcoxon",
    corr_method: Literal["benjamini-hochberg", "bonferroni"] = "benjamini-hochberg",
    txt_size: int = 13,
    txt: str = "p = ",
    ylabel: str = "LogMean(nUMI)",
    line_offset: float = 0.05,
    ylim_max: float = None,
    **kwargs,
):
    """Barplot with stats.

    Show the average expression of `var_names` or a continuous value in `obs` along different categorical values
    and test for significance. The mean pseudo-bulk expression per sample will be plotted as dots.

    :param adata: annotated data matrix
    :param x_axis: categorical `obs` column to group-by.
    :param feature: feature in `var_name` or `obs`.
    :param batch_key: `obs` column with batch information.
    :param order: order for the x_axis categories.
    :param hue: categorical `obs` column to split by each group in x_axis.
    :param hue_order: order for the hue categories.
    :param layer: layer in the AnnData to use.
    :param figsize:  figure size.
    :param palette: dictionary or palette to use.
    :param capsize: width of the 'caps' on error bars, relative to bar spacing.
    :param xtick_rotation: rotation of the x-ticks.
    :param ctrl_cond: name of the ctrl condition in the x-ticks. When hue is set, the control correspond to the categories in hue.
                      for each x_axis category the different hue categories will be tested.
    :param groups_cond: list of the name of the groups to test in the x-ticks.
    :param groups_pvals: if provided, these values will be plotted. If not set, provide a list of the groups in the x-ticks
                         to test.
    :param title: title of the plot.
    :param path: path to save the figure.
    :param filename: name of the file.
    :param title_fontproperties: properties of the title text.
    :param hue_legend_fontsize: size of the legend text when hue is set.
    :param hue_legend_fontweight: fontweight of the legend text when hue is set.
    :param hue_legend_cols: number of columns for the legend when hue is set.
    :param hue_legend_loc: location of the legend when hue is set.
    :param hue_legend_title: title of the legend when hue is set.
    :param show: if set to False, return the axis.
    :param marker_size: size of the markers showing the pseudo-bulk mean expression.
    :param ax: matplotlib axis.
    :param logcounts: if set to True, assume input is log1p transformed
    :param estimator: estimator to calculate the mean expression. If set to LogMean assume log1p.
    :param test: name of the method to test for significance.
    :param corr_method: correction method for multiple testing.
    :param txt_size: size of the text indicating significance.
    :param txt: text for indicating significance. If not set, only the p-value is shown.
    :param ylabel: Y-axis label.
    :param line_offset: line offset for the stat
    :param ylim_max: set maximum Y limit.
    :param kwargs: additional arguments passed to `sns.barplot() <https://seaborn.pydata.org/generated/seaborn.barplot.html>`_
    :return: Depending on ``show``, returns the plot if set to `True` or a dictionary with the axes.

    Example
    -------

    .. plot::
        :context: close-figs

        import dotools_py as do
        adata = do.dt.example_10x_processed()
        do.pl.barplot(adata, 'annotation', 'CD4', ctrl_cond = 'T_cells', groups_cond=['B_cells'], xtick_rotation=45)
        # Take only lymphoid cells
        lymphoid = adata[adata.obs['annotation'].isin(['T_cells', 'NK', 'B_cells'])].copy()
        # When hue is set we test for each annotation healthy Vs disease
        do.pl.barplot(lymphoid, 'annotation', 'CD4', hue = 'condition',   ctrl_cond = 'healthy', groups_cond=['disease'], hue_order=['healthy', 'disease'], xtick_rotation=45, figsize=(6, 4))


    """
    # Checks
    sanitize_anndata(adata)
    groups_cond = [groups_cond] if isinstance(groups_cond, str) else groups_cond
    groups_pvals = [groups_pvals] if isinstance(groups_pvals, float) else groups_pvals
    feature = [feature] if isinstance(feature, str) else feature
    assert len(feature) == 1, "Only 1 feature can be plotted"
    feature = feature[0]

    def log_estimator(values):
        return np.log1p(np.mean(np.expm1(values)))

    groups_to_save = [x_axis, hue] if hue is not None else [x_axis]
    groups_to_save_batch = [x_axis, batch_key, hue] if hue is not None else [x_axis, batch_key]

    if feature in adata.var_names:
        df = get_expr(adata, feature, groups=groups_to_save, layer=layer)
        df_batch = mean_expr(adata, group_by=groups_to_save_batch, features=feature, layer=layer)
        df_batch.columns = ["gene"] + groups_to_save_batch + ["expr"]
    elif feature in list(adata.obs.columns):
        df = adata.obs[groups_to_save + [feature]]
        df.columns = groups_to_save + ["expr"]
        df_batch = adata.obs[[feature] + groups_to_save_batch]
        df_batch = df_batch.groupby(groups_to_save_batch).agg(np.mean).fillna(0).reset_index()
        df_batch["gene"] = feature
        df_batch.columns = groups_to_save_batch + ["expr", "gene"]
        if logcounts:
            logger.warn(f"Assumming Log-counts but {feature} is in adata.obs")
    else:
        raise ValueError(f"{feature} is not in adata.var_names or adata.obs")

    if hue is not None:
        hue_order = list(adata.obs[hue].unique()) if hue_order is None else hue_order
    order = list(adata.obs[x_axis].unique()) if order is None else order

    # Create figure
    if ax is None:
        if hue is not None:
            width, height = figsize  # Define figure layout
            fig, gs = make_grid_spec(
                ax or (width, height), nrows=1, ncols=2, wspace=0.7 / width, width_ratios=[width - (0.9 + 0) + 0, 0.9]
            )
            ax = fig.add_subplot(gs[0])

        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

    if estimator == "LogMean":
        bp = sns.barplot(
            df, x=x_axis, y="expr", estimator=log_estimator, capsize=capsize, ax=ax, palette=palette,
            hue=hue, hue_order=hue_order, order=order, legend=False, **kwargs)
    else:
        bp = sns.barplot(df, x=x_axis, y="expr", estimator=estimator, capsize=capsize, ax=ax, palette=palette,
                         hue=hue, order=order, hue_order=hue_order, legend=False, **kwargs)

    dodge =True if hue else False
    sns.stripplot(df_batch, x=x_axis, y="expr", alpha=0.75, color="k", s=marker_size, ax=bp, hue=hue,
                  hue_order=hue_order, order=order, dodge=dodge, legend=False)

    if ctrl_cond is not None and groups_cond is not None:
        if groups_pvals is None:
            if hue is not None:
                testing = TestData(data=adata,
                                   feature=feature,
                                   cond_key=hue,
                                   ctrl=ctrl_cond,
                                   groups=groups_cond,
                                   category_key=x_axis,
                                   category_order=order,
                                   test=test,
                                   test_correction=corr_method)
            else:
                testing = TestData(data=adata,
                                   feature=feature,
                                   cond_key=x_axis,
                                   ctrl=ctrl_cond,
                                   groups=groups_cond,
                                   category_key=None,
                                   category_order=None,
                                   test=test,
                                   test_correction=corr_method)
            testing.run_test()
            groups_pvals = testing.pvals  # Should be the same order as for StatsPlotter

        plotter = StatsPlotter(
            bp,
            x_axis=x_axis,
            y_axis="expr",
            ctrl=ctrl_cond,
            groups=groups_cond,
            pvals=groups_pvals,
            txt_size=txt_size,
            txt=txt,
            kind="bar",
            line_offset=line_offset,
            hue=hue,
            hue_order=hue_order,
        )
        plotter.plot_stats()

    if xtick_rotation is not None:
        bp.set_xticklabels(bp.get_xticklabels(), rotation=xtick_rotation, ha="right", va="top", fontweight="bold")
    else:
        bp.set_xticklabels(bp.get_xticklabels(), fontweight="bold")

    bp.set_xlabel("")
    bp.set_ylabel(ylabel, fontweight='bold')

    # Correct YLim in case it was cut
    if len(adata.obs[batch_key]) == 2:  # There are only 1 batch per condition
        pass
    else:
        ymax = df_batch["expr"].max() + df_batch["expr"].max() * 0.1
        ymax = ylim_max if ylim_max is None else ymax
        bp.set_ylim(0, ymax)

    title_fontproperties = {} if title_fontproperties is None else title_fontproperties
    title_size = title_fontproperties.get("size", 20)
    title_font = title_fontproperties.get("weight", "bold")

    # If hue is define we need a legend
    if hue is not None:
        axs_legend = fig.add_subplot(gs[1])
        handles = []

        if isinstance(palette, str):
            colors = dotools_py.utility.get_hex_colormaps(palette)
            colors_dict = dict(zip(hue_order, colors, strict=False))
        elif isinstance(palette, dict):
            colors_dict = palette
        else:
            raise Exception('palette can only be a string or dictionary')

        handles = []
        for lab, c in colors_dict.items():
            handles.append(
                mlines.Line2D(
                    [0], [0], marker=".", color=c, lw=0, label=lab, markerfacecolor=c, markeredgecolor=None,
                    markersize=18
                )
            )

        legend = axs_legend.legend(
            handles=handles,
            frameon=False,
            loc=hue_legend_loc,
            ncols=hue_legend_cols,
            title=hue_legend_title,
            prop={"size": hue_legend_fontsize, "weight": hue_legend_fontweight},
        )
        legend.get_title().set_fontweight("bold")
        legend.get_title().set_fontsize(hue_legend_fontsize + 2)
        axs_legend.tick_params(axis="both", left=False, labelleft=False, labelright=False, bottom=False,
                               labelbottom=False)
        axs_legend.spines[["right", "left", "top", "bottom"]].set_visible(False)
        axs_legend.grid(visible=False)


    if title is None:
        bp.set_title(feature, fontsize=title_size, fontweight=title_font)  # Title is the genename
    else:
        bp.set_title(title, fontsize=title_size, fontweight=title_font)
    if path is not None:  # If the path is provided we save it
        plt.savefig(convert_path(path) / filename, bbox_inches="tight")
    if show is False:  # if show is false we return the axes
        if hue is not None:
            bp = {'mainplot_ax': bp,
                  'legend_ax': axs_legend}
        return bp
    else:
        plt.tight_layout()
        return plt.show()


def boxplot(
    adata: ad.AnnData,
    x_axis: str,
    feature: str,
    order: list = None,
    hue: str = None,
    hue_order: list = None,
    layer: str = None,
    figsize: tuple = (3, 4.2),
    palette: str | list = "tab10",
    xtick_rotation: int = None,
    ctrl_cond: str = None,
    groups_cond: str | list = None,
    groups_pvals: list = None,
    title: str = None,
    path: str = None,
    filename: str = None,
    title_fontproperties: dict = None,
    hue_legend_fontsize: int = 12,
    hue_legend_fontweight: float | str = None,
    hue_legend_cols: int = 1,
    hue_legend_title: str = '',
    hue_legend_loc: Literal["center left", "cemter right", "upper right", "upper left", "lower left", "lower right", "right", "lower center", "upper center", "center"] = 'center left',
    show: bool = True,
    ax: plt.Axes = None,
    showfliers: bool = False,
    test: Literal["wilcoxon", "t-test", "kruskal", "anova", "logreg", "t-test_overestim_var"] = "wilcoxon",
    corr_method: Literal["benjamini-hochberg", "bonferroni"] = "benjamini-hochberg",
    txt_size: int = 13,
    txt: str = "p = ",
    ylabel="LogMean(nUMI)",
    line_offset: float = 0.05,
    **kwargs,
):
    """Boxplot with stats.

    Show the distribution of the  expression of `var_names` or a continuous value in `obs` along different categorical values
    and test for significance.

    :param adata: annotated data matrix
    :param x_axis: categorical `obs` column to groupby.
    :param feature: feature in `var_name` or `obs`.
    :param order: order for the x_axis categories.
    :param hue: categorical `obs` column to split by each group in x_axis.
    :param hue_order: order for the hue categories.
    :param layer: layer in the AnnData to use.
    :param figsize:  figure size.
    :param palette: dictionary or palette to use.
    :param xtick_rotation: rotation of the xticks.
    :param ctrl_cond: name of the ctrl condition in the xticks. When hue is set, the control correspond to the categories in hue.
                      for each x_axis category the different hue categories will be tested.
    :param groups_cond: list of the name of the groups to test in the xticks.
    :param groups_pvals: if provided, these values will be plotted. If not set, provide a list of the groups in the xticks
                      to test.
    :param title: title of the plot.
    :param path: path to save the figure.
    :param filename: name of the file.
    :param title_fontproperties: properties of the title text.
    :param hue_legend_fontsize: size of the legend text when hue is set.
    :param hue_legend_fontweight: fontweight of the legend text when hue is set.
    :param hue_legend_cols: number of columns for the legend when hue is set.
    :param hue_legend_loc: location of the legend when hue is set.
    :param hue_legend_title: title of the legend when hue is set.
    :param show: if set to False, return the axis.
    :param ax: matplotlib axis.
    :param showfliers: if set to False, the outliers of the boxplot are not shown.
    :param test: name of the method to test for significance. Available: ['wilcoxon', 't-test', 'kruskal', 'anova', 'logreg', 't-test_overestim_var'].
    :param corr_method: correction method for multiple testing. Available: ['benjamini-hochberg', 'bonferroni'].
    :param txt_size: size of the text indicating significance.
    :param txt: text for indicating significance. If not set, only the p-value is shown.
    :param ylabel: Y-axis label.
    :param line_offset: offset from the stats.
    :param kwargs: additional arguments passed to `sns.boxplot() <https://seaborn.pydata.org/generated/seaborn.boxplot.html>`_
    :return: Depending on ``show``, returns the plot if set to `True` or a dictionary with the axes.

    Example
    -------

    .. plot::
        :context: close-figs

        import dotools_py as do
        adata = do.dt.example_10x_processed()
        do.pl.boxplot(adata,  'annotation', 'CD4', ctrl_cond = 'pDC', groups_cond=['B_cells'], xtick_rotation=45)
        # Take only lymphoid cells
        lymphoid = adata[adata.obs['annotation'].isin(['T_cells', 'NK', 'B_cells'])].copy()
        # When hue is set we test for each annotation healthy Vs disease
        do.pl.boxplot(lymphoid, 'annotation', 'RPL11', hue = 'condition', ctrl_cond = 'healthy', groups_cond=['disease'], hue_order=['healthy', 'disease'], xtick_rotation=45, figsize=(6, 4))


    """
    # Checks
    sanitize_anndata(adata)
    groups_cond = [groups_cond] if isinstance(groups_cond, str) else groups_cond
    groups_pvals = [groups_pvals] if isinstance(groups_pvals, float) else groups_pvals
    feature = [feature] if isinstance(feature, str) else feature
    assert len(feature) == 1, "Only 1 feature can be plotted"
    feature = feature[0]

    groups_to_save = [x_axis, hue] if hue is not None else [x_axis]
    if feature in adata.var_names:
        df = get_expr(adata, feature, groups=groups_to_save, layer=layer)
    elif feature in list(adata.obs.columns):
        df = adata.obs[groups_to_save + [feature]]
        df.columns = groups_to_save + ["expr"]
    else:
        raise ValueError(f"{feature} is not in adata.var_names or adata.obs")

    if hue is not None:
        hue_order = list(adata.obs[hue].unique()) if hue_order is None else hue_order
    order = list(adata.obs[x_axis].unique()) if order is None else order

    # Create figure
    if ax is None:
        if hue is not None:
            width, height = figsize  # Define figure layout
            fig, gs = make_grid_spec(
                ax or (width, height), nrows=1, ncols=2, wspace=0.7 / width, width_ratios=[width - (0.9 + 0) + 0, 0.9]
            )
            ax = fig.add_subplot(gs[0])

        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

    bx = sns.boxplot(df, x=x_axis, y="expr", showfliers=showfliers, ax=ax, palette=palette,
                 order=order, hue=hue, hue_order=hue_order, legend=False, **kwargs)

    if ctrl_cond is not None and groups_cond is not None:
        if groups_pvals is None:
            if hue is not None:
                testing = TestData(data=adata,
                                   feature=feature,
                                   cond_key=hue,
                                   ctrl=ctrl_cond,
                                   groups=groups_cond,
                                   category_key=x_axis,
                                   category_order=order,
                                   test=test,
                                   test_correction=corr_method)
            else:
                testing = TestData(
                    adata,
                    feature=feature,
                    cond_key=x_axis,
                    ctrl=ctrl_cond,
                    groups=groups_cond,
                    category_order=None,
                    category_key=None,
                    test=test,
                    test_correction=corr_method,
                )
            testing.run_test()
            groups_pvals = testing.pvals

        plotter = StatsPlotter(
            bx,
            x_axis=x_axis,
            y_axis="expr",
            ctrl=ctrl_cond,
            groups=groups_cond,
            pvals=groups_pvals,
            txt_size=txt_size,
            txt=txt,
            kind="box",
            line_offset=line_offset,
            hue=hue,
            hue_order=hue_order
        )
        plotter.plot_stats()

    if xtick_rotation is not None:
        bx.set_xticklabels(bx.get_xticklabels(), rotation=xtick_rotation, ha="right", va="top", fontweight="bold")
    else:
        bx.set_xticklabels(bx.get_xticklabels(), fontweight="bold")
    bx.set_xlabel("")
    bx.set_ylabel(ylabel, fontweight="bold")

    title_fontproperties = {} if title_fontproperties is None else title_fontproperties
    title_size = title_fontproperties.get("size", 20)
    title_font = title_fontproperties.get("weight", "bold")

    if hue is not None:
        axs_legend = fig.add_subplot(gs[1])

        if isinstance(palette, str):
            colors = dotools_py.utility.get_hex_colormaps(palette)
            colors_dict = dict(zip(hue_order, colors, strict=False))
        elif isinstance(palette, dict):
            colors_dict = palette
        else:
            raise Exception('palette can only be a string or dictionary')

        handles = []
        for lab, c in colors_dict.items():
            handles.append(
                mlines.Line2D(
                    [0], [0], marker=".", color=c, lw=0, label=lab, markerfacecolor=c, markeredgecolor=None,
                    markersize=18
                )
            )

        legend = axs_legend.legend(
            handles=handles,
            frameon=False,
            loc=hue_legend_loc,
            ncols=hue_legend_cols,
            title=hue_legend_title,
            prop={"size": hue_legend_fontsize, "weight": hue_legend_fontweight},
        )
        legend.get_title().set_fontweight("bold")
        legend.get_title().set_fontsize(hue_legend_fontsize + 2)
        axs_legend.tick_params(axis="both", left=False, labelleft=False, labelright=False, bottom=False,
                               labelbottom=False)
        axs_legend.spines[["right", "left", "top", "bottom"]].set_visible(False)
        axs_legend.grid(visible=False)

    if title is None:
        bx.set_title(feature, fontsize=title_size, fontweight=title_font)  # Title is the genename
    else:
        bx.set_title(title, fontsize=title_size, fontweight=title_font)
    if path is not None:  # If the path is provided we save it
        plt.savefig(convert_path(path) / filename, bbox_inches="tight")
    if show is False:  # if show is false we return the axes
        if hue is not None:
            bx = {'mainplot_ax': bx,
                  'legend_ax': axs_legend}
        return bx
    else:
        plt.tight_layout()
        return plt.show()


def violin(
    adata: ad.AnnData,
    x_axis: str,
    feature: str,
    order: list = None,
    hue: str = None,
    hue_order: list = None,
    layer: str = None,
    figsize: tuple = (3, 4.2),
    palette: str | list = "tab10",
    xtick_rotation: int = None,
    ctrl_cond: str = None,
    groups_cond: str | list = None,
    groups_pvals: list = None,
    scatter: bool = False,
    scatter_size: int = 2,
    title: str = None,
    path: str = None,
    filename: str = None,
    title_fontproperties: dict = None,
    hue_legend_fontsize: int = 12,
    hue_legend_fontweight: float | str = None,
    hue_legend_cols: int = 1,
    hue_legend_title: str = '',
    hue_legend_loc: Literal["center left", "cemter right", "upper right", "upper left", "lower left", "lower right", "right", "lower center", "upper center", "center"] = 'center left',
    show: bool = True,
    ax: plt.Axes = None,
    cut: float = 0,
    test: Literal["wilcoxon", "t-test", "kruskal", "anova", "logreg", "t-test_overestim_var"] = "wilcoxon",
    corr_method: Literal["benjamini-hochberg", "bonferroni"] = "benjamini-hochberg",
    txt_size: int = 13,
    txt: str = "p = ",
    ylabel="LogMean(nUMI)",
    line_offset: float = 0.05,
    **kwargs,
):
    """Violinplot with stats.

    Show the distribution of the  expression of `var_names` or a continuous value in `obs` along different categorical values
    and test for significance.

    :param adata: annotated data matrix
    :param x_axis: categorical `obs` column to groupby.
    :param feature: feature in `var_name` or `obs`.
     :param order: order for the x_axis categories.
    :param hue: categorical `obs` column to split by each group in x_axis.
    :param hue_order: order for the hue categories.
    :param layer: layer in the AnnData to use.
    :param figsize:  figure size.
    :param palette: dictionary or palette to use.
    :param xtick_rotation: rotation of the xticks.
    :param ctrl_cond: name of the ctrl condition in the xticks. When hue is set, the control correspond to the categories in hue.
                      for each x_axis category the different hue categories will be tested.
    :param groups_cond: list of the name of the groups to test in the xticks
    :param groups_pvals: if provided, these values will be plotted. If not set, provide a list of the groups in the
                      xticks to test.
    :param scatter: Plot non-zero values as scatterplot on top of the violin plots.
    :param scatter_size: Size of the scatterplot dots.
    :param title: title of the plot.
    :param path: path to save the figure.
    :param filename: name of the file.
    :param title_fontproperties: properties of the title text.
    :param hue_legend_fontsize: size of the legend text when hue is set.
    :param hue_legend_fontweight: fontweight of the legend text when hue is set.
    :param hue_legend_cols: number of columns for the legend when hue is set.
    :param hue_legend_loc: location of the legend when hue is set.
    :param hue_legend_title: title of the legend when hue is set.
    :param show: if set to False, return the axis.
    :param ax: matplotlib axis.
    :param cut: distance in units of bandwidth, to extend the density past extreme datapoints. Set to 0 to limit the
             violin within the data range
    :param test: name of the method to test for significance. Available: ['wilcoxon', 't-test', 'kruskal', 'anova', 'logreg', 't-test_overestim_var'].
    :param corr_method: correction method for multiple testing. Available: ['benjamini-hochberg', 'bonferroni'].
    :param txt_size: size of the text indicating significance.
    :param txt: text for indicating significance. If not set, only the p-value is shown.
    :param ylabel: Y-axis label.
    :param line_offset: offset for the stat
    :param kwargs: additional arguments passed to `sns.barplot() <https://seaborn.pydata.org/generated/seaborn.violinplot.html>`_
    :return: Depending on ``show``, returns the plot if set to `True` or a dictionary with the axes.

    Example
    -------

    .. plot::
        :context: close-figs

        import dotools_py as do
        adata = do.dt.example_10x_processed()
        do.pl.violin(adata,  'annotation', 'CD4', ctrl_cond = 'pDC', groups_cond=['B_cells'], xtick_rotation=45, scatter=True)
        # Take only lymphoid cells
        lymphoid = adata[adata.obs['annotation'].isin(['T_cells', 'NK', 'B_cells'])].copy()
        # When hue is set we test for each annotation healthy Vs disease
        do.pl.violin(lymphoid,'annotation','CD4',  hue = 'condition',   ctrl_cond = 'healthy', groups_cond=['disease'], hue_order=['healthy', 'disease'], xtick_rotation=45, figsize=(6, 4))


    """

    # Checks
    sanitize_anndata(adata)
    groups_cond = [groups_cond] if isinstance(groups_cond, str) else groups_cond
    groups_pvals = [groups_pvals] if isinstance(groups_pvals, float) else groups_pvals
    feature = [feature] if isinstance(feature, str) else feature
    assert len(feature) == 1, "Only 1 feature can be plotted"
    feature = feature[0]

    groups_to_save = [x_axis, hue] if hue is not None else [x_axis]
    if feature in adata.var_names:
        df = get_expr(adata, feature, groups=groups_to_save, layer=layer)
    elif feature in list(adata.obs.columns):
        df = adata.obs[groups_to_save + [feature]]
        df.columns = groups_to_save + ["expr"]
    else:
        raise ValueError(f"{feature} is not in adata.var_names or adata.obs")

    if hue is not None:
        hue_order = list(adata.obs[hue].unique()) if hue_order is None else hue_order
    order = list(adata.obs[x_axis].unique()) if order is None else order

    # Create figure
    if ax is None:
        if hue is not None:
            width, height = figsize  # Define figure layout
            fig, gs = make_grid_spec(
                ax or (width, height), nrows=1, ncols=2, wspace=0.7 / width, width_ratios=[width - (0.9 + 0) + 0, 0.9]
            )
            ax = fig.add_subplot(gs[0])

        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

    vln = sns.violinplot(df, x=x_axis, y="expr", ax=ax, palette=palette, cut=cut,
                         order=order, hue=hue, hue_order=hue_order, legend=False, **kwargs)
    if scatter:
        sns.stripplot(df[df.expr != 0], x=x_axis,  y="expr", ax=vln, color="k", order=order, hue=hue, hue_order=hue_order, legend=False,
                      size=scatter_size, dodge=True, )

    if ctrl_cond is not None and groups_cond is not None:
        if groups_pvals is None:
            if hue is not None:
                testing = TestData(data=adata,
                                   feature=feature,
                                   cond_key=hue,
                                   ctrl=ctrl_cond,
                                   groups=groups_cond,
                                   category_order=order,
                                   category_key=x_axis,
                                   test=test,
                                   test_correction=corr_method)
            else:
                testing = TestData(
                    adata,
                    feature=feature,
                    cond_key=x_axis,
                    ctrl=ctrl_cond,
                    groups=groups_cond,
                    category_order=None,
                    category_key=None,
                    test=test,
                    test_correction=corr_method,
                )
            testing.run_test()
            groups_pvals = testing.pvals

        plotter = StatsPlotter(
            vln,
            x_axis=x_axis,
            y_axis="expr",
            ctrl=ctrl_cond,
            groups=groups_cond,
            pvals=groups_pvals,
            txt_size=txt_size,
            txt=txt,
            kind="violin",
            line_offset=line_offset,
            hue=hue,
            hue_order=hue_order
        )
        plotter.plot_stats()

    if xtick_rotation is not None:
        vln.set_xticklabels(vln.get_xticklabels(), rotation=xtick_rotation, ha="right", va="top", fontweight="bold")
    vln.set_xlabel("")
    vln.set_ylabel(ylabel, fontweight="bold")

    title_fontproperties = {} if title_fontproperties is None else title_fontproperties
    title_size = title_fontproperties.get("size", 20)
    title_font = title_fontproperties.get("weight", "bold")

    if hue is not None:
        axs_legend = fig.add_subplot(gs[1])
        handles = []

        if isinstance(palette, str):
            colors = dotools_py.utility.get_hex_colormaps(palette)
            colors_dict = dict(zip(hue_order, colors, strict=False))
        elif isinstance(palette, dict):
            colors_dict = palette
        else:
            raise Exception('palette can only be a string or dictionary')

        handles = []
        for lab, c in colors_dict.items():
            handles.append(
                mlines.Line2D(
                    [0], [0], marker=".", color=c, lw=0, label=lab, markerfacecolor=c, markeredgecolor=None,
                    markersize=18
                )
            )

        legend = axs_legend.legend(
            handles=handles,
            frameon=False,
            loc=hue_legend_loc,
            ncols=hue_legend_cols,
            title=hue_legend_title,
            prop={"size": hue_legend_fontsize, "weight": hue_legend_fontweight},
        )
        legend.get_title().set_fontweight("bold")
        legend.get_title().set_fontsize(hue_legend_fontsize + 2)
        axs_legend.tick_params(axis="both", left=False, labelleft=False, labelright=False, bottom=False,
                               labelbottom=False)
        axs_legend.spines[["right", "left", "top", "bottom"]].set_visible(False)
        axs_legend.grid(visible=False)

    if title is None:
        vln.set_title(feature, fontsize=title_size, fontweight=title_font)  # Title is the genename
    else:
        vln.set_title(title, fontsize=title_size, fontweight=title_font)
    if path is not None:  # If the path is provided we save it
        plt.savefig(convert_path(path) / filename, bbox_inches="tight")
    if show is False:  # if show is false we return the axes
        if hue is not None:
            vln = {'mainplot_ax': vln,
                  'legend_ax': axs_legend}
        return vln
    else:
        plt.tight_layout()
        return plt.show()
