import sys
from typing import Literal, Union, Dict
from numpy.typing import NDArray
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.stats

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patheffects as path_effects
from matplotlib.patches import PathPatch
from adjustText import adjust_text

from dotools_py.get._generic import expr as get_expr
from dotools_py.utility._plotting import get_hex_colormaps
from dotools_py.utils import make_grid_spec, logmean, logsem, save_plot, return_axis, sanitize_anndata, iterase_input, \
    check_missing, draw_vertical_bracket


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


def _seaborn_kde_1d(
    data: pd.Series | NDArray,
    xs: NDArray,
    bw_adjust: float = 0.5,
    bw_method: str = "scott",
    weights: NDArray = None
) -> NDArray:
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(data, bw_method=bw_method, weights=weights)
    kde.set_bandwidth(kde.factor * bw_adjust)
    return kde(xs)



def ridgeplot(
    # Data
    adata: ad.AnnData,
    group_by: str,
    feature: str,
    layer: str = None,

    # Figure parameters
    figsize: tuple[int, int] = (6, 5),
    ax: plt.Axes = None,
    title: str = None,
    title_fontproperties: Dict[Literal["size", "weight"], str | int] = None,
    palette: str | dict = None,
    x_linspace: int = 500,
    alpha: float = 1,
    x_label: str = "Log(nUMI)",
    add_y_ticks: bool = True,

    # Statistics
    reference: str = None,
    groups: str | list = None,
    groups_pvals: list = None,
    test: Literal["wilcoxon", "t-test", "kruskal", "anova", "logreg", "t-test_overestim_var"] = "wilcoxon",
    corr_method: Literal["benjamini-hochberg", "bonferroni"] = "benjamini-hochberg",
    txt: str = "p = ",
    txt_size: int = 10,

    # IO
    path: str | Path = None,
    filename: str = "Ridgeplot.pdf",
    show: bool = True,

    # Fx specific
    bw_adjust: float = 0.5,
    bw_method: str = "scott",
    ridge_height: float = 0.8,
    ridge_spacing: float = 0.6,

) -> plt.Axes | dict | None:
    """Ridgeplot for AnnData.

    Represent in a ridgeplot the expression of a feature in `adata.var_names` or a
    continuous metadata in `adata.obs`.

    :param adata: Annotated data matrix.
    :param group_by: Column in `adata.obs` to group in the Y axis.
    :param feature: Valid key in `adata.var_names` or continuous metadata in `adata.obs`.
    :param layer: Layer in `adata.layers` to use.
    :param figsize:  Figure size, the format is (width, height).
    :param ax: Matplotlib axes to use for plotting. If not set, a new figure will be generated.
    :param title:  Title for the figure.
    :param title_fontproperties: Dictionary which should contain 'size' and 'weight' to define the fontsize and fontweight of the title of the figure.
    :param palette: Can be the name of a valid matplotlib colormap or a dictionary of the groups as keys and the colors as values. If set to `None` will extract the colors from `adata.uns[group_by_colors]`
    :param x_linspace: Number of points to generate for the x-axis.
    :param alpha: Transparency level of the object, where 0 is fully transparent and 1 is fully opaque.
    :param x_label: Name of the X axis label.
    :param add_y_ticks: If set to `True` the groups will be shown in the y-ticks, otherwise the Y-axis is removed and the labels are displayed inside the plot.
    :param reference: Reference condition to use when testing for significance.
    :param groups: List of the name of the groups to test against.
    :param groups_pvals: If provided, these values will be plotted. If not set, the p-values will be estimated. The order of the p-values should match the order of the `groups_cond` categories.
    :param test: Name of the method to test for significance.
    :param corr_method: Correction method for multiple testing.
    :param txt: Text to include before the p-value. If not set, only the p-value is shown.
    :param txt_size:  Font size of the text indicating significance.
    :param path: Path to the folder to save the figure.
    :param filename: Name of file to use when saving the figure.
    :param show: If set to `False`, returns a dictionary with the matplotlib axes.
    :param bw_adjust: Factor that multiplicatively scales the value chosen using `bw_method`. Increasing will make the curve smoother.
    :param bw_method: Method for determining the smoothing bandwidth to use; passed to `scipy.stats.gaussian_kde <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html>`_.
    :param ridge_height:  Scaling factor controlling the ridge height.
    :param ridge_spacing: Distance between consecutive ridges.
    :return: Depending on ``show``, returns the plot if set to `True` or a dictionary with the axes.

    Example
    --------
    Create a ridgeplot showing the expression of a given gene including the p-value to indicate if there is
    a significant statistical difference between groups.

    .. plot::
        :context: close-figs

        import dotools_py as do
        adata = do.dt.example_10x_processed()
        do.pl.ridgeplot(adata,  'annotation', 'CD4', reference = 'pDC', groups=['B_cells'])

    Plot a continuous value in `adata.obs`.

    .. plot::
        :context: close-figs

        do.pl.ridgeplot(adata,'condition','total_counts', reference = 'healthy', groups=['disease'], figsize=(6, 4), x_label="total_counts", title="", palette={"healthy":"sandybrown", "disease":"royalblue"})


    """

    sanitize_anndata(adata)
    catgs = adata.obs[group_by].unique().tolist()

    # Palette selection for group_by categories
    if isinstance(palette, str):
        palette = get_hex_colormaps(palette)
        palette = dict(zip(catgs, palette))
    elif isinstance(palette, dict):
        missing = [key for key in catgs if key not in palette]
        assert len(missing) == 0, f"{missing} is not present in palette"
    elif palette is None:
        assert group_by + "_colors" in adata.uns.keys(), f"palette is set to None, but adata.uns[{group_by}_colors] is not present"
        palette = dict(zip(adata.obs[group_by].cat.categories, adata.uns[group_by + "_colors"]))
    else:
        raise ValueError("Not a valid value for palette, set a colormap, dictionary or set to None")


    # # # # # # # # # #
    # Data preparation
    if feature in adata.var_names.tolist():
        df = get_expr(adata, feature, groups=group_by, layer=layer)
    elif feature in adata.obs.columns.tolist():
        df = adata.obs[[feature, group_by]].copy()
        df = df.rename(columns={feature: "expr"})
    else:
        raise ValueError(f"{feature} is not in adata.var_names nor adata.obs")

    xmin, xmax = df["expr"].min(), df["expr"].max()
    xs = np.linspace(xmin, xmax, x_linspace)

    # # # # # # # # #
    # Generate Plot
    fig, gs = make_grid_spec(ax or figsize, nrows=1, ncols=2, width_ratios=[2, 0.25], wspace=0.01)

    # Main Axes
    ax_ridge = fig.add_subplot(gs[0])
    positions = {}
    for i, g in enumerate(catgs):
        tmp = df[df[group_by] == g]["expr"]
        y0 = i * ridge_spacing
        try:
            ys = _seaborn_kde_1d(tmp.to_numpy(), xs, bw_adjust=bw_adjust, bw_method=bw_method)
        except np.linalg.LinAlgError:  # No variance, everything is the same value
            eps = 1e-5  # Add pseudo-count
            data_jittered = tmp.to_numpy() + np.random.normal(0, eps, len(tmp))
            ys = _seaborn_kde_1d(data_jittered, xs, bw_adjust=bw_adjust, bw_method=bw_method)
        ys = ys / ys.max() * ridge_height  # Normalize height so ridges are comparable

        ax_ridge.fill_between(
            xs, y0, ys + y0 , color=palette[g], alpha=alpha, linewidth=1.5, zorder=len(catgs) -i
        )
        ax_ridge.plot(xs, ys + y0, color="white", lw=1.5, zorder=len(catgs) -i, linestyle="-")
        if not add_y_ticks:
            ax_ridge.text(
                xmin, y0 + 0.4, g, ha="left", va="center", fontweight="bold",
                path_effects=[path_effects.Stroke(linewidth=3, foreground="white"), path_effects.Normal()]
            )
        positions[g] = y0

    tmp = list(positions.values())
    tmp = np.abs(tmp[0] - tmp[1]) / 2
    if add_y_ticks:
        y_ticks = [p + tmp for p in positions.values()]
        ax_ridge.set_yticks(y_ticks)
        ax_ridge.set_yticklabels(catgs)
    else:
        ax_ridge.set_yticks([])
        ax_ridge.spines[["top", "left", "right"]].set_visible(False)

    positions = {key:val+tmp for key, val in positions.items()}


    # Compute significance
    groups_cond = iterase_input(groups)
    groups_pvals = iterase_input(groups_pvals)
    if reference is not None and len(groups_cond) != 0:
        from dotools_py.pl._StatsPlotter import TestData
        if len(groups_pvals) == 0:
            testing = TestData(
                data=adata, feature=feature, cond_key=group_by, ctrl=reference, groups=groups_cond, category_key=None,
                category_order=None, test=test, test_correction=corr_method
            )
            testing.run_test()
            groups_pvals = testing.pvals  # Should be the same order as for StatsPlotter
            del testing
        groups_pvals = [float(p) for p in groups_pvals]
        groups_pvals = [
            str(np.round(p, 2)) if p > 0.05 else str(np.round(p, 4))
            if p > 0.009 else f"{sys.float_info.min if p == 0 else p:0.2e}"
            for p in groups_pvals
        ]

        ax_significance = fig.add_subplot(gs[1])
        ax_significance.set_ylim(ax_ridge.get_ylim())  # Set the same Y-axis to match

        # Add Significance
        sig_pos = [(positions[reference], positions[g]) for g in groups]
        x_pos, cont = 0, 0
        for x0, x1 in sig_pos:
            x_pos += .35
            patch_path = draw_vertical_bracket(
                y_start=x0, y_end=x1, x_left=x_pos - 0.1, x_right=x_pos,
            )
            patch = PathPatch(patch_path, linewidth=1, facecolor="none", edgecolor="black", clip_on=False, zorder=1)
            ax_significance.add_patch(patch)

            ax_significance.text(
                x_pos + x_pos / 4, (x0 + x1) / 2, f"{txt}{groups_pvals[cont]}", fontsize=txt_size, ha="center",
                va="center",
                rotation=270, rotation_mode="anchor",
            )
            cont += 1
        ax_significance.axis("off")  # Hide axes for clean display
        ax_significance.set_xlim(0, x_pos + .35)
        axis_dict = {"mainplot_ax": ax_ridge, "significance_ax": ax_significance}
    else:
        axis_dict = ax_ridge

    # Plot Layout
    title_fontproperties = {} if title_fontproperties is None else title_fontproperties
    title_size = title_fontproperties.get("size", 18)
    title_fontweight = title_fontproperties.get("weight","bold")
    ax_ridge.set_title(feature if title is None else title, fontsize=title_size, fontweight=title_fontweight)
    ax_ridge.set_xlabel(x_label)

    save_plot(path, filename)
    return return_axis(show, axis_dict, tight=True)

