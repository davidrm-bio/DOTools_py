from typing import Literal, Dict
from pathlib import Path

import anndata as ad
import matplotlib.colors
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from scipy.stats import zscore

from dotools_py.logger import logger
from dotools_py.tl import rank_genes_groups
from dotools_py.get import mean_expr
from dotools_py.utils import convert_path, sanitize_anndata


def make_grid_spec(
    ax_or_figsize,
    *,
    nrows: int,
    ncols: int,
    wspace: float = None,
    hspace: float = None,
    width_ratios: float | list = None,
    height_ratios: float | list = None,
):
    # Taken from Scanpy
    kw = {"wspace": wspace, "hspace": hspace, "width_ratios": width_ratios, "height_ratios": height_ratios}

    if isinstance(ax_or_figsize, tuple):
        fig = plt.figure(figsize=ax_or_figsize)
        return fig, gridspec.GridSpec(nrows, ncols, **kw)
    else:
        ax = ax_or_figsize
        ax.axis("off")
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
        return ax.figure, ax.get_subplotspec().subgridspec(nrows, ncols, **kw)


def check_colornorm(vmin = None, vmax = None, vcenter = None, norm = None):
    from matplotlib.colors import Normalize

    try:
        from matplotlib.colors import TwoSlopeNorm as DivNorm
    except ImportError:
        # matplotlib<3.2
        from matplotlib.colors import DivergingNorm as DivNorm

    if norm is not None:
        if (vmin is not None) or (vmax is not None) or (vcenter is not None):
            raise ValueError("Passing both norm and vmin/vmax/vcenter is not allowed.")
    else:
        if vcenter is not None:
            norm = DivNorm(vmin=vmin, vmax=vmax, vcenter=vcenter)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)

    return norm


def square_color(rgba: list) -> str:
    """Determine if the background is dark or clear and return black or white.

    :param rgba: list with rgba values
    :return: black or white
    """
    r, g, b = rgba[:3]  # ignore alpha
    # Convert from 0 to 1 float to 0–255 int
    r, g, b = [int(c * 255) for c in (r, g, b)]
    # Use brightness heuristic
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    return "black" if brightness > 128 else "white"


def small_squares(ax: plt.Axes, pos: list, color: list, size: float = 1, linewidth: float = 0.8, zorder: int = 20) -> None:
    """Add small squares.

    :param ax: matplotlib axis
    :param pos: list of positions
    :param color: list of colors
    :param size:  size of the square
    :param linewidth: linewith of the square
    :param zorder: location of the square
    :return: None
    """
    for idx, xy in enumerate(pos):
        x, y = xy
        margin = (1 - size) / 2
        rect = patches.Rectangle(
            (y + margin, x + margin),
            size,
            size,
            linewidth=linewidth,
            edgecolor=color[idx],
            facecolor="none",
            zorder=zorder,
        )
        if zorder == 0:
            rect.set_alpha(0)  # Hide square if they should be in the back, for the dotplot
        ax.add_patch(rect)
    return None


def heatmap(
    # Data
    adata: ad.AnnData,
    group_by: str | list,
    features: str | list,
    groups_order: list = None,
    layer: str = None,
    logcounts: bool = True,

    # Figure parameters
    figsize: tuple = (5, 6),
    ax: plt.Axes | None = None,
    swap_axes: bool = True,
    title: str = None,
    title_fontproperties: Dict[Literal["size", "weight"], str | int] = None,
    palette: str = "Reds",
    ticks_fontproperties: Dict[Literal["size", "weight"], str | int] = None,

    xticks_rotation: int = None,
    yticks_rotation: int = None,
    cluster_x_axis: bool = False,
    cluster_y_axis: bool = False,

    legend_title: str = "LogMean(nUMI)\nin group",

    # IO
    path: str | Path = None,
    filename: str = "Heatmap.svg",
    show: bool = True,

    # Statistics
    add_stats: bool = False,
    test: Literal["wilcoxon", "t-test"] = "wilcoxon",
    correction_method: Literal["benjamini-hochberg", "bonferroni"] = "benjamini-hochberg",
    df_pvals: pd.DataFrame = None,
    stats_x_size: float = None,
    square_x_size: dict = None,
    pval_cutoff: float = 0.05,
    log2fc_cutoff: float = 0.0,

    # Fx specific
    z_score: Literal["var", "group"] = None,  # x_axis is the group_by
    clustering_method: str = "complete",
    clustering_metric: str = "euclidean",
    linewidth: float = 0.1,
    vmin: float = 0.0,
    vcenter: float = None,
    vmax: float = None,
    square: bool = True,
    **kargs,
) -> dict | None:
    """Heatmap of the mean expression of genes across a groups.

    Generate a heatmap of showing the average nUMI for a set of genes in different groups. Differential gene
    expression analysis between the different groups can be performed.

    :param adata: Annotated data matrix.
    :param group_by:  Name of a categorical column in `adata.obs` to groupby.
    :param features: cA valid feature in `adata.var_names` or column in `adata.obs` with continuous values.
    :param groups_order: Order for the categories in `adata.obs[group_by]`.
    :param z_score: Apply z-score transformation.
    :param path: Path to the folder to save the figure.
    :param filename: Name of file to use when saving the figure.
    :param layer: Name of the AnnData object layer that wants to be plotted. By default `adata.X` is plotted.
                 If layer is set to a valid layer name, then the layer is plotted.
    :param swap_axes: Whether to swap the axes or not.
    :param palette: String denoting matplotlib colormap.
    :param title: Title for the figure.
    :param title_fontproperties: Dictionary which should contain 'size' and 'weight' to define the fontsize and
                                fontweight of the title of the figure.
    :param clustering_method: Linkage method to use for calculating clusters. See `scipy.cluster.hierarchy.linkage <https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage>`_.
    :param clustering_metric: Distance metric to use for the data. See `scipy.spatial.distance.pdist <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist>`_.
    :param cluster_x_axis: Hierarchically clustering the x-axis.
    :param cluster_y_axis: Hierarchically clustering the y-axis.
    :param ax: Matplotlib axes to use for plotting. If not set, a new figure will be generated.
    :param figsize: Figure size, the format is (width, height).
    :param linewidth: Linewidth for the border of cells.
    :param ticks_fontproperties: Dictionary which should contain 'size' and 'weight' to define the fontsize and fontweight of the font of the x/y axis.
    :param xticks_rotation: Rotation of the x-ticks.
    :param yticks_rotation: Rotations of the y-ticks.
    :param vmin: The value representing the lower limit of the color scale.
    :param vcenter: The value representing the center of the color scale.
    :param vmax: The value representing the upper limit of the color scale.
    :param legend_title: Title for the colorbar.
    :param add_stats: Add statistical annotation. Will add a square with an '*' in the center if the expression is significantly different in a group with respect to the others.
    :param df_pvals: Dataframe with the pvals. Should be `gene x group` or `group x gene` in case of swap_axes is `False`.
    :param stats_x_size: Scaling factor to control the size of the asterisk.
    :param square_x_size: Size and thickness of the square.
    :param test: Name of the method to test for significance.
    :param correction_method: Correction method for multiple testing.
    :param pval_cutoff: Cutoff for the p-value.
    :param log2fc_cutoff: Minimum cutoff for the log2FC.
    :param square: Whether to make the cell square or not.
    :param show: If set to `False`, returns a dictionary with the matplotlib axes.
    :param logcounts: whether the input is logcounts or not.
    :param kargs: Additional arguments pass to `sns.heatmap <https://seaborn.pydata.org/generated/seaborn.heatmap.html>`_.
    :return: Depending on ``show``, returns the plot if set to `True` or a dictionary with the axes.

    Example
    -------

    .. plot::
        :context: close-figs

        import dotools_py as do
        adata = do.dt.example_10x_processed()
        do.pl.heatmap(adata, 'annotation', ['CD4', 'CD79A'], add_stats=True)

    """
    import scanpy as sc
    from scipy.cluster.hierarchy import dendrogram, linkage

    # Checks
    sanitize_anndata(adata)
    features = [features] if isinstance(features, str) else features
    features = features if isinstance(features, list) else list(features)
    missing = [g for g in features if g not in adata.var_names]
    assert len(missing) == 0, f'{missing} features missing in the object'

    # Get Data for the Heatmap
    if all(item in list(adata.var_names) for item in features):
        if logcounts:
            df = mean_expr(
                adata, group_by=group_by, features=features, layer=layer, out_format="wide"
            )  # genes x groups (genes are the index)
        else:
            raise Exception("Not implemented, specified var_name value but logcounts is set to False")
    elif all(item in list(adata.obs.columns) for item in features):
        df = adata.obs[[group_by] + features].groupby(group_by).agg("mean")
    else:
        raise Exception("Provide features either var_names or obs.columns")

    # Hierarchical clustering
    new_index = (
        df.index[
            dendrogram(linkage(df.values, method=clustering_method, metric=clustering_metric), no_plot=True)["leaves"]
        ]
        if cluster_x_axis
        else features
    )

    new_columns = groups_order if groups_order is not None else list(df.columns)

    new_column = (
        df.columns[
            dendrogram(linkage(df.T.values, method=clustering_method, metric=clustering_metric), no_plot=True)["leaves"]
        ]
        if cluster_y_axis
        else new_columns
    )

    df = df.reindex(index=new_index, columns=new_column)

    # Layout
    if swap_axes:
        df = df.T

    # Compute Statistics
    annot_pvals = None
    if add_stats:
        if df_pvals is None:
            if all(item in list(adata.var_names) for item in features):
                rank_genes_groups(adata, groupby=group_by, method=test, tie_correct=True, corr_method=correction_method)
                table = sc.get.rank_genes_groups_df(
                    adata, group=None, pval_cutoff=pval_cutoff, log2fc_min=log2fc_cutoff
                )
                table_filt = table[table["names"].isin(features)]
            elif all(item in list(adata.obs.columns) for item in features):
                raise Exception('Not Implemented')
                # TODO Fix Bug
                tdf = adata.obs[[group_by] + features]
                tdata = ad.AnnData(tdf.iloc[:, 1:].values, obs=pd.DataFrame(tdf[group_by]), var=list(tdf.columns)[1:])
                tdata.var_names = tdata.var[0].copy()
                rank_genes_groups(
                    tdata,
                    groupby=group_by,
                    method=test,
                    tie_correct=True,
                    corr_method=correction_method,
                    logcounts=False,
                )
                table = sc.get.rank_genes_groups_df(
                    tdata, group=None, pval_cutoff=pval_cutoff, log2fc_min=log2fc_cutoff
                )
                table_filt = table[table["names"].isin(features)]

            # Dataframe with gene x groups with the pvals
            table_filt["group"] = table_filt["group"].str.replace("-", "_")  # Correction used in get_expr()
            df_pvals = pd.DataFrame([], index=df.index, columns=df.columns)
            for idx, row in table_filt.iterrows():
                if row["group"] in list(df.index):
                    df_pvals.loc[row["group"], row["names"]] = row["pvals_adj"]
                else:
                    df_pvals.loc[row["names"], row["group"]] = row["pvals_adj"]
            df_pvals[df_pvals.isna()] = 1
        else:
            if list(df.index)[0] in list(df_pvals.index):
                pass
            else:
                df_pvals = df_pvals.T
        # Replace pvals < 0.05 with an X
        annot_pvals = df_pvals.applymap(lambda x: "*" if x < pval_cutoff else "")

    # Data Transformation
    if z_score is not None:
        if z_score == "var":
            if features[0] in list(df.index):
                axis = 0
            else:
                axis = 1
        elif z_score == "group":
            if features[0] in list(df.index):
                axis = 1
            else:
                axis = 0
        else:
            raise Exception(f'{z_score} not a valid key for z_score, use "var" or "group"')

        if axis == 1:
            df = df.apply(lambda row: pd.Series(zscore(row), index=df.columns), axis = axis)  # z_score over the genes
        else:
            df = df.apply(zscore, axis=axis, result_type="expand")  # z_score over the genes

        if palette == "Reds":
            logger.warn(
                "Z-score set to True, but the cmap is Reds, setting to RdBu_r"
            )  # Make sure to use divergent colormap
            palette = "RdBu_r"
        if legend_title == "LogMean(nUMI)\nin group":
            legend_title = "Z-score"
        vmin, vcenter, vmax = round(df.min().min() * 20) / 20, 0.0, None

    # ------ Arguments for the layout -------------
    width, height = figsize if figsize is not None else (None, None)
    legends_width_spacer = 0.7 / width
    mainplot_width = width - (1.5 + 0)
    if height is None:
        height = len(adata.obs[group_by].cat.categories) * 0.37
        width = len(features) * len(adata.obs[group_by].cat.categories) * 0.37 + 0.8

    min_figure_height = max([0.35, height])
    cbar_legend_height = min_figure_height * 0.08
    sig_legend = min_figure_height * 0.27
    spacer_height = min_figure_height * 0.3
    height_ratios = [
        height - sig_legend - cbar_legend_height - spacer_height,
        sig_legend,
        spacer_height,
        cbar_legend_height,
    ]

    ticks_fontproperties = {} if ticks_fontproperties is None else ticks_fontproperties
    ticks_fontproperties = {"weight": ticks_fontproperties.get("weight", "bold"), "size": ticks_fontproperties.get("size", 13)}
    tick_weight = ticks_fontproperties["weight"]
    tick_size = ticks_fontproperties["size"]

    title_fontproperties = {} if title_fontproperties is None else title_fontproperties
    title_fontprop = {"weight": title_fontproperties.get("weight", "bold"), "size": title_fontproperties.get("size", 15)}
    # Parameters for colorbar
    vmin = 0.0 if vmin is None else vmin
    vmax = round(df.max().max() * 20) / 20 if vmax is None else vmax  # Normalise to round to 5 or 0
    colormap = plt.get_cmap(palette)
    normalize = check_colornorm(vmin=vmin, vmax=vmax, vcenter=vcenter)
    mappable = ScalarMappable(norm=normalize, cmap=colormap)
    mean_flat = df.T.values.flatten()
    color = colormap(normalize(mean_flat))
    color = [square_color(c) for c in color]

    # Parameter for stats
    square_x_size = {} if square_x_size is None else square_x_size
    square_x_size = {"width": square_x_size.get("weight", 1), "size": square_x_size.get("size", 0.8)}
    # stats_x_size = max(np.sqrt(height * width), 14) if stats_x_size is None else stats_x_size
    stats_x_size = min(width / df.shape[1], height / df.shape[1]) * 10 if stats_x_size is None else min(width / df.shape[1], height / df.shape[1]) * stats_x_size

    # Save the axis
    return_ax_dict = {}
    # ---------------------------------------

    # Generate figure
    fig, gs = make_grid_spec(
        ax or (width, height), nrows=1, ncols=2, wspace=legends_width_spacer, width_ratios=[mainplot_width + 0, 1.5]
    )
    main_ax = fig.add_subplot(gs[0])
    legend_ax = fig.add_subplot(gs[1])

    fig, legend_gs = make_grid_spec(legend_ax, nrows=4, ncols=1, height_ratios=height_ratios)
    color_legend_ax = fig.add_subplot(legend_gs[3])
    if add_stats:
        sig_ax = fig.add_subplot(legend_gs[2])

    # Add Main Plot
    hm = sns.heatmap(
        data=df,
        cmap=palette,
        ax=main_ax,
        linewidths=linewidth,
        cbar=False,
        annot_kws={"color": "black", "size": stats_x_size, "ha": "center", "va": "center", "fontfamily":'DejaVu Sans Mono'},
        annot=annot_pvals,
        fmt="s",
        square=square,
        **kargs,
    )

    # Add Legend
    matplotlib.colorbar.Colorbar(color_legend_ax, mappable=mappable, orientation="horizontal")
    color_legend_ax.set_title(legend_title, fontsize="small", fontweight="bold")
    color_legend_ax.xaxis.set_tick_params(labelsize="small")
    return_ax_dict["legend_ax"] = color_legend_ax

    # Significance Legend
    if add_stats:
        x, y = 0, 0.5
        sig_ax.scatter(x, y, s=500, facecolors="none", edgecolors="black", marker="s")
        sig_ax.text(x, y, "*", fontsize=18, ha="center", va="center", color="black", fontfamily='DejaVu Sans Mono')
        sig_ax.text(x + 0.03, y, "FDR < 0.05", fontsize=12, va="center", fontweight="bold")
        sig_ax.set_xlim(x - 0.02, x + 0.1)
        sig_ax.set_title("Significance", fontsize="small", fontweight="bold")
        plt.gca().set_aspect("equal")
        sig_ax.axis("off")  # Hide axes for clean display
        return_ax_dict["signifiance_ax"] = sig_ax

    # Modify layout from main plot
    hm.spines[["top", "right", "bottom", "left"]].set_visible(True)
    hm.set_xlabel("")
    hm.set_ylabel("")

    rotation_props_x, rotation_props_y = {"rotation": None}, {"rotation": None}
    rotation_props_x = (
        {"rotation": xticks_rotation, "va": "top", "ha": "right"} if xticks_rotation is not None else rotation_props_x
    )
    rotation_props_y = (
        {"rotation": yticks_rotation, "va": "top", "ha": "right"} if yticks_rotation is not None else rotation_props_y
    )
    hm.set_xticklabels(hm.get_xticklabels(), fontdict={"weight": tick_weight, "size": tick_size}, **rotation_props_x)
    hm.set_yticklabels(hm.get_yticklabels(), fontdict={"weight": tick_weight, "size": tick_size}, **rotation_props_y)
    hm.set_title(title, **title_fontprop)
    return_ax_dict["mainplot_ax"] = hm

    # Add Square around the Xs
    if add_stats:
        df_x = pd.DataFrame([], index=df.index, columns=df.columns)
        df_x[df_x.isna()] = "black"
        df_x = df.map(lambda x: square_color(colormap(normalize(x))))
        pos_rows, pos_cols = np.where(df_pvals < 0.05)
        pos = list(zip(pos_rows, pos_cols, strict=False))
        colors = [df_x.iloc[row, col] for row, col in pos]

        small_squares(
            hm,
            color=colors,
            pos=pos,
            size=square_x_size["size"],
            linewidth=square_x_size["width"],
        )

        # Now set colors manually on each annotation text base on the background
        for text, color in zip(hm.texts, df_x.values.flatten(), strict=False):
            text.set_color(color)

    if path is not None:
        plt.savefig(convert_path(path) / filename, bbox_inches="tight")
    if show:
        return plt.show()
    else:
        return return_ax_dict
