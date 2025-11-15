from typing import Literal, Union
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.stats

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patheffects as path_effects
from adjustText import adjust_text

from dotools_py.get._generic import expr as get_expr
from dotools_py.utility._plotting import get_hex_colormaps
from dotools_py.utils import make_grid_spec, logmean, logsem, save_plot, return_axis, sanitize_anndata, iterase_input, \
    check_missing


def lineplot(
    # Data
    adata: ad.AnnData,
    x_axis: str,
    features: str | list,
    hue: Union[str, Literal["features"]] = None,

    # Figure parameters
    figsize: tuple = (6, 5),
    ax: plt.Axes = None,
    palette: str | dict = "tab10",
    title: str = None,
    xticks_rotation: int | None = None,
    xticks_order: list = None,
    ylim: tuple[int, int] = None,
    ylabel: str = "LogMean(nUMI)",

    # Legend Parameters
    legend_title: str = None,
    legend_loc: Literal["right", "axis"] = "right",
    legend_repel: dict = None,

    # IO
    path: str | Path = None,
    filename: str = "lineplot.svg",
    show: bool = False,

    # Statistics
    estimator: Literal["logmean", "mean"] = "logmean",

    # Fx specific
    markersize: int = 8,
) -> plt.Axes | dict | None:
    """Lineplot for AnnData.

    :param adata: Annotated data matrix
    :param x_axis: Name of a categorical column in `adata.obs` to groupby.
    :param features:  A valid feature in `adata.var_names` or column in `adata.obs` with continuous values.
    :param hue: Name of a second categorical column in `adata.obs` to use additionally to groupby. If several `features`
                are provided, set to `features`.
    :param figsize: Figure size, the format is (width, height).
    :param ax: Matplotlib axes to use for plotting. If not set, a new figure will be generated.
    :param palette:  String denoting matplotlib colormap. A dictionary with the categories available in `adata.obs[x_axis]` or
                    `adata.obs[hue]` if hue is not None can also be provided. The format is {category:color}.
    :param title: Title for the figure.
    :param xticks_rotation: Rotation of the X-axis ticks.
    :param xticks_order: Order for the categories in `adata.obs[x_axis]`.
    :param ylim: Set limit for Y-axis.
    :param ylabel: Label for the Y-axis.
    :param legend_title: Title for the legend.
    :param legend_loc:  Location of the legend.
    :param legend_repel: Additional arguments pass to `adjust_text <https://adjusttext.readthedocs.io/en/latest/>_`.
    :param path: Path to the folder to save the figure.
    :param filename: Name of file to use when saving the figure.
    :param show: If set to `False`, returns a dictionary with the matplotlib axes.
    :param estimator: If set to `logmean`, the mean will be calculated after undoing the log. The returned mean expression
                     is also represented in log-space.
    :param markersize: Radius of the markers
    :return: Depending on ``show``, returns the plot if set to `True` or a dictionary with the axes.

    Example
    -------

    Plot the expression for a gene across several groups.

    .. plot::
        :context: close-figs

        import dotools_py as do
        adata = do.dt.example_10x_processed()
        do.pl.lineplot(adata, 'condition', 'CD4', hue = 'annotation')

    Plot the distribution of several genes at the same time.

    .. plot::
        :context: close-figs

        do.pl.lineplot(adata, 'condition', ['CD4', 'CD79A'], hue = 'features')

    """
    sanitize_anndata(adata)

    features = iterase_input(features)
    check_missing(adata, features=features, groups=x_axis)
    if len(features) > 1:
        assert hue == "features", "When multiple features are provided, use hue = 'features'"

    # Generate the data
    estimator = logmean if estimator == "logmean" else estimator
    sem_estimator = logsem if estimator == "logmean" else scipy.stats.sem
    markers = ["o", "s", "v", "^", "P", "X", "D", "<", ">"]
    markers = markers * 5

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

    # Generate the plot
    width, height = figsize
    ncols, fig_kwargs = 1, {}
    if hue is not None and legend_loc == "right":
        fig_kwargs = {"wspace": 0.7 / width, "width_ratios": [width - (1.5 + 0) + 0, 1.5]}
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

        if xticks_order is not None:
            sdf[x_axis] = pd.Categorical(sdf[x_axis], categories=xticks_order, ordered=True)
            sdf = sdf.sort_values(x_axis)
        axs.plot(sdf[x_axis], sdf["expr"], color=palette[h])
        axs.errorbar(sdf[x_axis], sdf["expr"], yerr=sdf["sem"], fmt=markers[idx], capsize=5, ecolor="k",
                     color=palette[h],
                     markersize=markersize)
        if hue != "tmp":
            handles.append(
                mlines.Line2D([0], [0], marker=".", color=palette[h], lw=0, label=h, markerfacecolor=palette[h],
                              markeredgecolor=None, markersize=15))
        if legend_loc == "axis":
            text = axs.text(len(sdf[x_axis]) - 1 + 0.15, sdf["expr"].tail(1), h, color="black")
            text.set_path_effects([
                path_effects.Stroke(linewidth=1, foreground=palette[h]),  # Edge color
                path_effects.Normal()])

            text_list.append(text)
    if len(text_list) != 0:
        legend_repel = {} if legend_repel is None else legend_repel
        adjust_text(text_list, ax=axs, expand_axes=True,
                    only_move={"text": "y", "static": "y", "explode": "y", "pull": "y"}, **legend_repel)

    ticks_kwargs = {"fontweight": "bold", "fontsize": 12}
    if xticks_rotation is not None:
        ticks_kwargs.update({"rotation": xticks_rotation, "ha": "right", "va": "top"})

    axs.set_xticklabels(axs.get_xticklabels(), **ticks_kwargs)

    xlims = np.round(axs.get_xlim(), 2)
    ylims = np.round(axs.get_ylim(), 2) if ylim is None else ylim
    axs.set_xlim(xlims[0] + np.sign(xlims[0]) * 0.25, xlims[1] + np.sign(xlims[1]) * 0.25)
    axs.set_ylim(0, ylims[1])
    if estimator == "mean" and ylabel == "LogMean(nUMI)":
        ylabel = "Mean(nUMI)"

    axs.set_ylabel(ylabel=ylabel)
    axs.set_xlabel("")

    if len(features) == 1 and title is None:
        title = features[0]

    axs.set_title(title)

    legend_axs = None
    if ncols == 2 and legend_loc == "right" and len(handles) != 0:
        legend_axs = fig.add_subplot(gs[1])
        legend_axs.legend(handles=handles, frameon=False, loc="center left", ncols=1, title=legend_title)
        legend_axs.tick_params(axis="both", left=False, labelleft=False, labelright=False, bottom=False,
                               labelbottom=False)
        legend_axs.spines[["right", "left", "top", "bottom"]].set_visible(False)
        legend_axs.grid(visible=False)

    if legend_axs is not None:
        axis_dict = {"mainplot_ax": axs, "legend_ax": legend_axs}
    else:
        axis_dict = axs

    save_plot(path, filename)
    return return_axis(show, axis_dict, tight=True)
