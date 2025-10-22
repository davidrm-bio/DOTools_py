from typing import Literal, Union

import anndata as ad
import numpy as np
import pandas as pd
import scipy.stats

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patheffects as path_effects

from dotools_py.get._generic import expr as get_expr
from dotools_py.utility._plotting import get_hex_colormaps
from dotools_py.utils import make_grid_spec, logmean, logsem, save_plot, return_axis, sanitize_anndata
from adjustText import adjust_text


def lineplot(adata: ad.AnnData,
             x_axis: str,
             features: str | list,
             x_categories_order: list = None,
             hue: Union[str, Literal["features"]] = None,
             estimator: Literal["logmean", "mean"] = "logmean",
             figsize: tuple = (6, 5),
             ax: plt.Axes = None,
             palette: str | dict = "tab10",
             markersize: int = 8,
             ylim: tuple = None,
             ylabel: str = "LogMean(nUMI)",
             title: str = None,
             legend_title: str = None,
             legend_loc: Literal["right", "axis"] = "right",
             labels_repel: dict = None,
             xtick_rotation: int | None = None,
             show: bool = False,
             path: str = None,
             filename: str = "lineplot.svg",
             ):
    """Lineplot for AnnData features.

    :param adata: AnnData object.
    :param x_axis: Column in `obs` to group by.
    :param features: Feature in `var_names`. If one feature is provided, the hue argument can be used to group-by an
                    additional column in `obs`. If several features are provided, use 'features' in hue.
    :param x_categories_order: Order for the categories.
    :param hue: Additional column in `obs` to group-by when one feature is provided. Set to 'feature' when multiple features
                are provided.
    :param estimator: If set to `logmean`, the mean will be calculated after undoing the log. The returned mean expression
                     is also represented in log-space.
    :param figsize: Figure width x height.
    :param ax: Matplotlib axis
    :param palette: Name of a palette or a dictionary with colors for each category.
    :param markersize: Marker size.
    :param ylim: Set limit for Y-axis.
    :param ylabel: Name of the Y-axis.
    :param title: Title of the plot.
    :param legend_title: Title of the legend.
    :param legend_loc: Location from the legend. If set to `axis` labels will be added in the plot.
    :param labels_repel:  additional arguments pass to adjust_text.
    :param xtick_rotation: Rotation of the xticks.
    :param show: if set to False, return the axis.
    :param path: Path to the folder where the plot will be saved.
    :param filename: Name of the file.
    :return: Depending on ``show``, returns the plot if set to `True` or a dictionary with the axes.

    Example
    -------

    .. plot::
        :context: close-figs

        import dotools_py as do
        adata = do.dt.example_10x_processed()
        do.pl.lineplot(adata, 'condition', 'CD4', hue = 'annotation')
        # Plot several Genes
        do.pl.lineplot(adata, 'condition', ['CD4', 'CD79A'], hue = 'features')

    """
    sanitize_anndata(adata)

    features = [features] if isinstance(features, str) else features
    if len(features) > 1:
        assert hue == "features", "When multiple features are provided, use hue = 'features'"

    # Generate the data
    estimator = logmean if estimator == "logmean" else estimator
    sem_estimator = logsem if estimator == "logmean" else scipy.stats.sem
    markers = ["o", "s", "v", "^", "P", "X", "D", "<", ">"]
    markers = markers*5

    hue_arg = [] if (hue is None) or (hue == "features") else [hue]
    hue = "genes" if hue == "features" else hue
    groups = [x_axis] + [hue] if hue is not None else [x_axis]

    df = get_expr(adata, features=features, groups=[x_axis] + hue_arg)
    df_mean = df.groupby(groups).agg({"expr": estimator}).reset_index()
    df_sem = df.groupby(groups).agg({"expr": sem_estimator}).fillna(0).reset_index()
    df_sem.columns = groups + ["sem"]
    df = pd.merge(df_mean, df_sem, on=groups)
    if hue is None:
        hue = "tmp"
        df["tmp"] = "tmp"

    # Test for significance - TODO indicate significance with a discontinued line

    # Generate the plot
    width, height = figsize
    ncols, fig_kwargs = 1, {}
    if hue is not None and legend_loc == "right":
        fig_kwargs = {"wspace": 0.7 / width, "width_ratios":[width - (1.5 + 0) + 0, 1.5]}
        ncols = 2

    hue_groups = list(df[hue].unique())
    if isinstance(palette, str) or palette is None:
        colors = get_hex_colormaps(palette)
        palette = dict(zip(hue_groups, colors))

    fig, gs = make_grid_spec(ax or (width, height), nrows=1, ncols=ncols, **fig_kwargs)
    axs = fig.add_subplot(gs[0])

    handles = []
    text_list = []
    for idx, h in enumerate(hue_groups):
        sdf = df[df[hue] == h]

        if x_categories_order is not None:
            sdf[x_axis] = pd.Categorical(sdf[x_axis], categories=x_categories_order, ordered=True)
            sdf = sdf.sort_values(x_axis)
        axs.plot(sdf[x_axis], sdf["expr"],color=palette[h])
        axs.errorbar(sdf[x_axis], sdf["expr"], yerr=sdf["sem"], fmt=markers[idx], capsize=5, ecolor="k", color=palette[h],
                     markersize=markersize)
        if hue != "tmp":
            handles.append(mlines.Line2D([0], [0], marker=".", color=palette[h], lw=0, label=h, markerfacecolor=palette[h],
                                         markeredgecolor=None, markersize=15))
        if legend_loc == "axis":
            text = axs.text(len(sdf[x_axis]) -1 + 0.15, sdf["expr"].tail(1), h, color="black")
            text.set_path_effects([
                path_effects.Stroke(linewidth=1, foreground=palette[h]),  # Edge color
                path_effects.Normal()])

            text_list.append(text)
    if len(text_list) !=0:
        if labels_repel is None:
            labels_repel = {}
        adjust_text(text_list, ax=axs, expand_axes=True,  only_move= {"text": "y", "static": "y", "explode": "y", "pull": "y"}, **labels_repel)

    ticks_kwargs = {"fontweight": "bold", "fontsize": 12}
    if xtick_rotation is not None:
        ticks_kwargs.update({"rotation": xtick_rotation, "ha": "right", "va": "top"})

    axs.set_xticklabels(axs.get_xticklabels(), **ticks_kwargs)

    xlims = np.round(axs.get_xlim(), 2)
    ylims =  np.round(axs.get_ylim(), 2) if ylim is None else ylim
    axs.set_xlim(xlims[0] + np.sign(xlims[0]) * 0.25, xlims[1] + np.sign(xlims[1]) * 0.25)
    axs.set_ylim(0, ylims[1])
    if estimator == "mean" and ylabel == "LogMean(nUMI)":
        ylabel = "Mean(nUMI)"

    axs.set_ylabel(ylabel=ylabel)
    axs.set_xlabel("")

    if len(features) == 1 and title is None:
        title = features[0]

    axs.set_title(title)

    if ncols == 2 and legend_loc == "right" and len(handles) !=0:
        legend_axs = fig.add_subplot(gs[1])
        legend_axs.legend(handles=handles, frameon=False, loc="center left", ncols=1, title=legend_title)
        legend_axs.tick_params(axis="both", left=False, labelleft=False, labelright=False, bottom=False,
                               labelbottom=False)
        legend_axs.spines[["right", "left", "top", "bottom"]].set_visible(False)
        legend_axs.grid(visible=False)

    try:
        axis_dict = {"mainplot_ax": axs, "legend_ax": legend_axs}
    except NameError:
        axis_dict = axs

    save_plot(path, filename)
    return return_axis(show, axis_dict, tight=True)









