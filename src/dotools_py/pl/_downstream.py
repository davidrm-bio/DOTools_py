import sys
from pathlib import Path
from typing import Literal, Dict

import anndata as ad
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from dotools_py import logger
from dotools_py.get import mean_expr
from dotools_py.utility import generate_cmap
from dotools_py.utils import convert_path, format_terms_gsea, make_grid_spec, require_dependencies, sanitize_anndata, save_plot, return_axis


def correlation(
    # Data
    adata: ad.AnnData,
    group_by: str = "batch",

    # Figure parameters
    figsize: tuple = (3, 4),
    ax: plt.Axes | None = None,
    palette: str | list | LinearSegmentedColormap = "RdBu_r",
    ticks_size: int = 12,

    # IO
    path: str | None = None,
    filename: str = "Correlation.svg",
    show: bool = True,

    # Statistics
    method: Literal["spearman", "pearson", "kendall"] = "pearson",

    # Fx specific
    mode: Literal["colors", "letters"] = "letters",
    mask: Literal["upper", "lower"] = None,
    square: bool = True,
    linewidths: float = 0.1,
    linecolor: str = "black",
    annot: bool = True,
    annot_fontsize: int = 15,
    annot_color: str = "white",
    annot_kws: dict | None = None,

) -> plt.Axes | None:
    """Calculate correlation between samples.

    Calculates the pearson, spearman or kendall correlation between categorical metadata for
    all the genes and plot it using a heatmap representation. There are two modes:

    * `letters`: the color of the squares will be white and the correlation values will be colored based on a gradient
    * `colors`: the squares will be colored based on a gradient and the letters will be white.

    The gradient is defined based on the provided palette. The input is expected to be log-normalised data.

    :param adata: Annotated data matrix.
    :param group_by: Name of a categorical column in `adata.obs` to groupby.
    :param figsize:  Figure size, the format is (width, height).
    :param ax: Matplotlib axes to use for plotting. If not set, a new figure will be generated.
    :param palette: String denoting matplotlib colormap. If a list of colors is provided a colormap will be generated.
    :param ticks_size: Size of the X and Y axis ticks.
    :param path: Path to the folder to save the figure.
    :param filename: Name of file to use when saving the figure.
    :param show: If set to `False`, returns a dictionary with the matplotlib axes.
    :param method: Method to use to calculate correlation: Pearson, Spearman or Kendall.
    :param mode: Indicate how the correlation is represented.
    :param mask: If set to `lower` or `upper` hide the upper or lower triangle of the heatmap, respectively.
    :param square: If `True`, set the Axes aspect to “equal” so each cell will be square-shaped.
    :param linewidths: Width of the lines that divide each cell.
    :param linecolor: Color of the lines that divide each cell.
    :param annot: Add text with correlation values to the squares.
    :param annot_fontsize: Size of the font.
    :param annot_color: Color of the font. Will be ignored when mode is `letters`.
    :param annot_kws: Additional Keyword arguments for `matplotlib.axes.Axes.text() <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html>`_
    :return: Depending on ``show``, returns the plot if set to `True` or a dictionary with the axes.

    Example
    -------

    .. plot::
        :context: close-figs

        import dotools_py as do
        adata = do.dt.example_10x_processed()
        do.pl.correlation(adata, 'batch')


    """
    # Checks
    assert method in ["spearman", "pearson", "kendall"], "Not a valid method use spearman, pearson or kendall"
    assert mode in ["colors", "letters"], "Not a valid method use spearman, pearson or kendall"
    part_removed = mask

    # Extract the Average Expression
    df = mean_expr(adata, group_by=group_by, features=list(adata.var_names))  # All Genes
    df_pivot = df.pivot(index="gene", columns=group_by, values="expr")
    df_corr = df_pivot.corr(method=method)

    # Define mask
    if mask == "upper":
        mask = ~np.tril(df_corr.values).astype(bool)
    elif mask == "lower":
        mask = ~np.triu(df_corr.values).astype(bool)
    elif mask is None:
        mask = None
    else:
        raise Exception(f"'{mask}' is not a valid mask, specify 'upper', 'lower' or None ")

    # Define the colormap
    norm_palette = plt.Normalize(df_corr.min().min(), df_corr.max().min())
    if isinstance(palette, str):  # Assume is a cmap in matplotlib
        palette = plt.cm.get_cmap(palette)
    elif isinstance(palette, list):
        palette = generate_cmap(*palette)
    elif isinstance(palette, LinearSegmentedColormap):
        palette = palette
    else:
        raise Exception("Provide the name of a colormap, a list of colors or a custom colormap")

    white_cmap = ListedColormap(["white"])
    palette_cbar = white_cmap if mode == "letters" else palette

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    hm = sns.heatmap(
        df_corr, mask=mask, cmap=palette_cbar, square=square, linewidths=linewidths, annot=False, ax=ax,
        linecolor=linecolor, cbar=False,
    )

    # Remove the gridlines around masked data
    if mask is not None:
        mesh = hm.findobj()[0]
        nan_mask_flat = mask.ravel()
        edge_colors = np.array([(0, 0, 0, 0) if is_nan else (0, 0, 0, 1) for is_nan in nan_mask_flat])
        mesh.set_edgecolors(edge_colors)

    if mode == "letters" or annot:
        for i in range(df_corr.shape[0]):
            for j in range(df_corr.shape[1]):
                if mask is not None:
                    if mask[i, j]:
                        continue
                value = df_corr.iloc[i, j]
                color = palette(norm_palette(value)) if mode == "letters" else annot_color
                hm.text(
                    j + 0.5, i + 0.5, f"{value:.2f}", color=color, ha="center", va="center", fontsize=annot_fontsize,
                    weight="bold"
                )
    # Add Colorbar
    sm = ScalarMappable(norm=norm_palette, cmap=palette)
    sm.set_array([])  # Needed for colorbar to work

    annot_kws = {} if annot_kws is None else annot_kws
    annot_kws.update(
        {
            "orientation": annot_kws.get("orientation", "horizontal"),
            "fraction": annot_kws.get("fraction", 0.05),
            "pad": annot_kws.get("pad", 0.2),
            "shrink": annot_kws.get("shrink", 0.5),
        }
    )

    cbar = fig.colorbar(sm, ax=ax, **annot_kws)
    cbar.ax.set_title(f"Correlation {method}", fontdict={"size": 12})

    # Layout
    hm.set_xlabel("")
    hm.set_ylabel("")

    xticks, yticks = hm.get_xticks(), hm.get_yticks()
    (xtickslabels, ytickslabels) = (
        [label.get_text() for label in hm.get_xticklabels()],
        [label.get_text() for label in hm.get_yticklabels()],
    )
    if part_removed == "lower":
        remove_x, remove_y = [], []
        for row in range(mask.shape[0]):
            for col in range(mask.shape[1]):
                if mask[row, col]:
                    xtickslabels[col] = ""
                    ytickslabels[row] = ""
                    remove_x.append(xticks[col])
                    remove_y.append(yticks[row])
        xticks = [tick for tick in xticks if tick not in remove_x]
        yticks = [tick for tick in yticks if tick not in remove_y]
        xtickslabels = [label for label in xtickslabels if label != ""]
        ytickslabels = [label for label in ytickslabels if label != ""]

    hm.set_xticks(xticks)
    hm.set_xticklabels(xtickslabels, fontweight="bold", fontsize=ticks_size)
    hm.set_yticks(yticks)
    hm.set_yticklabels(ytickslabels, fontweight="bold", fontsize=ticks_size)

    save_plot(path, filename)
    return return_axis(show, axis=hm)



@require_dependencies([{"name": "scanpro"}])
def cell_composition(
    # Data
    adata: ad.AnnData,
    annot_key: str,
    condition_key: str,
    batch_key: str,
    annot_order: list | None = None,
    condition_order: list | None = None,
    subset_cells: list | None = None,

    # Figure Parameters
    figsize: tuple = (5, 6),
    ax: plt.Axes | None = None,
    title: str = "",
    title_fontproperties: Dict[Literal["size", "weight"], str | int] = None,

    xticks_rotation: int = None,
    legend_title: str = "",
    legend_fontproperties: Dict[Literal["size", "weight"], str | int] = None,
    legend_ncols: int = 1,

    # IO
    path: str | Path = None,
    filename: str = "Proportions.svg",
    show: bool = True,

    # Statistics
    covariates: list | None = None,
    pval_cutoff: float = 0.05,
    transform: str = "logit",

    # Fx specific
    sep: float = 0.5,
    bar_width: float = 0.2,
    linewidth: float = 0.9,
    add_total_ncell: bool = True,
    get_props: bool = False,
    **kwargs,
) -> None | pd.DataFrame | plt.Axes:
    """Stacked barplot showing changes in cell-type proportions.

    Generates a stacked barplot to show changes in cell-type proportions between different conditions. Significant changes
    in cell proportions between conditions will be tested with `scanpro <https://github.com/loosolab/scanpro>` and will be
    indicated by a discontinued line. The significant p-value/FDR will be indicated in the legend.

    :param adata: Annotated data matrix.
    :param annot_key: Name of a categorical column in `adata.obs` with the annotation to test for significant differences.
    :param condition_key:  Name of a categorical column in `adata.obs` to group by.
    :param batch_key:  Name of a categorical column in `adata.obs` with the batch information.
    :param annot_order: Order for the categories in `adata.obs[annot_key]`
    :param condition_order:  Order for the categories in `adata.obs[condition_key]`
    :param subset_cells: Only show a subset of groups in `adata.obs[annot_key]`. The test is applied over all cells.
    :param figsize: Figure size, the format is (width, height).
    :param ax: Matplotlib axes to use for plotting. If not set, a new figure will be generated.
    :param title: Title for the figure.
    :param title_fontproperties: Dictionary which should contain 'size' and 'weight' to define the fontsize and fontweight of the title of the figure.
    :param xticks_rotation: Order for the categories in `adata.obs[condition_key]`.
    :param legend_title:  Title for the legend.
    :param legend_fontproperties: Dictionary which should contain 'size' and 'weight' to define the fontsize and fontweight of the title of the legend.
    :param legend_ncols: Number of columns for the legend.
    :param path: Path to the folder to save the figure.
    :param filename:  Name of file to use when saving the figure.
    :param show: If set to `False`, returns a dictionary with the matplotlib axes.
    :param covariates: Additional covariates for the model. See `scanpro <https://scanpro.readthedocs.io/en/latest/API.html>`_.
    :param pval_cutoff: P-val/FDR cutoff.
    :param transform: Method of transformation of proportions.
    :param sep: Separation between bars.
    :param bar_width: Bars width.
    :param linewidth: Thickness of the lines connecting significant bars.
    :param add_total_ncell: Add the total number of cells in the dataset.
    :param get_props: If set to `True`, returns a dataframe with the cell proportions.
    :param kwargs: Additional arguments pass to `scanpro() <https://scanpro.readthedocs.io/en/latest/API.html#scanpro.scanpro.scanpro>`_.
    :return: Depending on ``show``, returns the plot if set to `True` or a dictionary with the axes.

    Example
    -------

     .. plot::
        :context: close-figs

        import dotools_py as do
        adata = do.dt.example_10x_processed()
        do.pl.cell_composition(adata, "annotation", "condition", "batch", condition_order=["healthy", "disease"], transform="arcsin")

    """
    ########################
    # Test for changes in cell population
    ########################
    from scanpro import scanpro

    transform = transform if batch_key is not None else "arcsin"
    adata = adata.copy()  # Do not modify input
    sanitize_anndata(adata)

    if annot_order is not None:
        assert all(x in annot_order for x in list(adata.obs[annot_key].cat.categories)), (
            "annotation  order is missing categories"
        )
        adata.obs[annot_key] = pd.Categorical(adata.obs[annot_key], categories=annot_order, ordered=True)

    if condition_order is not None:
        assert all(x in condition_order for x in list(adata.obs[condition_key].cat.categories)), (
            "condition order is missing categories"
        )
        adata.obs[condition_key] = pd.Categorical(adata.obs[condition_key], categories=condition_order, ordered=True)

    out = scanpro(
        adata,
        clusters_col=annot_key,
        conds_col=condition_key,
        samples_col=batch_key,
        covariates=covariates,
        transform=transform,
        **kwargs,
    )

    ########################
    # Set-Up, Get Data for plotting
    ########################
    subset_cells = subset_cells if subset_cells is not None else adata.obs[annot_key].cat.categories

    df = out.results.copy()
    pval_col = "adjusted_p_values" if "adjusted_p_values" in df.columns else "p_values"
    n_sig = len(df[df[pval_col] < 0.05])
    logger.info(f"There are {n_sig} populations with a significant change")

    df = df.loc[subset_cells, :]

    try:
        colors_dict = dict(zip(adata.obs[annot_key].cat.categories, adata.uns[annot_key + "_colors"], strict=False))
    except KeyError:
        tab20_colors = plt.cm.tab20.colors
        if len(adata.obs[annot_key].cat.categories) > 20:
            tab20_colors = tab20_colors * 3
        colors_dict = dict(zip(adata.obs[annot_key].cat.categories, tab20_colors, strict=False))
    colors_list = [colors_dict[ct] for ct in df.index]

    cond_keys = [f"mean_props_{cond}" for cond in adata.obs[condition_key].cat.categories]
    data_dict = {"bar_bottom": {}, "bar_height": {}, "pvals": list(df[pval_col])}

    for cond in cond_keys:
        tmp = np.zeros(len(df))
        for idx, prop in enumerate(list(df[cond])[:-1]):
            tmp[idx + 1] = prop + tmp[idx]
        data_dict["bar_bottom"][cond] = tmp
        data_dict["bar_height"][cond] = list(df[cond])

    ########################
    # Plotting
    ########################
    width, height = figsize  # Define figure layout
    fig, gs = make_grid_spec(
        ax or (width, height), nrows=1, ncols=2, wspace=0.7 / width, width_ratios=[width - (1.5 + 0) + 0, 1.5]
    )

    # Main Axis
    axs = fig.add_subplot(gs[0])
    xtick, xtext = [], []
    for x_pos, c in enumerate(cond_keys):
        x_pos = x_pos - sep * x_pos
        bars_obj = axs.bar(
            x_pos,
            data_dict["bar_height"][c],
            width=bar_width,
            bottom=data_dict["bar_bottom"][c],
            align="edge",
            zorder=2,
            color=colors_list,
        )
        xtick.append(x_pos + bar_width / 2)
        xtext.append(c.split("mean_props_")[-1])

        for i, padj in enumerate(data_dict["pvals"]):
            if padj < pval_cutoff:
                if x_pos / sep + 1 < len(cond_keys):
                    cond1 = c
                    cond2 = cond_keys[int(cond_keys.index(c) + 1)]
                    axs.plot(
                        [x_pos + bar_width, x_pos + 1 - sep],
                        [data_dict["bar_bottom"][cond1][i], data_dict["bar_bottom"][cond2][i]],
                        color="k",
                        linestyle="--",
                        zorder=1,
                        linewidth=linewidth,
                    )

                    axs.plot(
                        [x_pos + bar_width, x_pos + 1 - sep],
                        [
                            data_dict["bar_height"][cond1][i] + data_dict["bar_bottom"][cond1][i],
                            data_dict["bar_height"][cond2][i] + data_dict["bar_bottom"][cond2][i],
                        ],
                        color="k",
                        linestyle="--",
                        zorder=1,
                        linewidth=linewidth,
                    )

                for j, b in enumerate(bars_obj):
                    if i == j:
                        b.set_edgecolor("black")
                        b.set_linewidth(1)
                        b.set_zorder(3)

    if xticks_rotation is not None:
        xticks_prop = {"rotation": xticks_rotation, "ha": "right", "va": "top"}
    else:
        xticks_prop = {"rotation": xticks_rotation}
    axs.set_xticks(xtick, xtext, fontweight="bold", **xticks_prop)

    title_fontproperties = {} if title_fontproperties is None else title_fontproperties
    axs.set_title(title, fontsize=title_fontproperties.get("size", 15), fontweight=title_fontproperties.get("weight", "bold"))
    axs.set_ylabel("Proportions", fontweight="bold")

    # Legend Axis
    axs_legend = fig.add_subplot(gs[1])
    handles = []
    for lab, c in colors_dict.items():
        if lab not in subset_cells:
            continue

        pval = df.loc[lab, pval_col]
        if df.loc[lab, pval_col] < pval_cutoff:
            if pval > 0.05:
                pval = str(round(pval, 2))
            elif pval > 0.009:
                pval = str(round(pval, 4))
            else:
                if pval == 0:
                    pval = sys.float_info.min
                pval = f"{pval:0.2e}"
            txt = "FDR" if pval_col == "adjusted_p_values" else "p"
            lab = lab + f" ({txt} = " + str(pval) + ")"

        handles.append(
            mlines.Line2D(
                [0], [0], marker=".", color=c, lw=0, label=lab, markerfacecolor=c, markeredgecolor=None, markersize=18
            )
        )
    if add_total_ncell:
        handles.append(
            mlines.Line2D(
                [0],
                [0],
                marker=".",
                color="white",
                lw=0,
                label=f"nCells = {adata.n_obs:,}",
                markerfacecolor="white",
                markeredgecolor=None,
                markersize=18,
            )
        )

    legend_fontproperties = {} if legend_fontproperties is None else legend_fontproperties
    legend_fontsize = legend_fontproperties.get("size", 12)
    legend_fontweight = legend_fontproperties.get("weight", "bold")

    legend = axs_legend.legend(
        handles=handles,
        frameon=False,
        loc="center left",
        ncols=legend_ncols,
        title=legend_title,
        prop={"size": legend_fontsize, "weight": legend_fontweight},
    )
    legend.get_title().set_fontweight("bold")
    legend.get_title().set_fontsize(legend_fontsize + 2)

    # Remove Ticks, Grid, Spines
    axs_legend.tick_params(axis="both", left=False, labelleft=False, labelright=False, bottom=False, labelbottom=False)
    axs_legend.spines[["right", "left", "top", "bottom"]].set_visible(False)
    axs_legend.grid(visible=False)

    # Save if specified
    if path is not None:
        plt.savefig(convert_path(path) / filename, bbox_inches="tight")

    if show is True and get_props is False:  # True and False
        plt.tight_layout()
        return plt.show()
    elif show is True and get_props is True:  # True and True
        plt.tight_layout()
        plt.show()
        return df
    elif show is False and get_props is True:  # False and True
        return df, {"mainplot_ax": axs, "legend_ax": axs_legend}
    else:  # False and False
        return {"mainplot_ax": axs, "legend_ax": axs_legend}


def volcano_plot(
    # Data
    df: pd.DataFrame,
    lfc_col: str = "log2fc",
    pval_col: str = "padj",
    gene_col: str = "GeneName",

    # Figure Parameters
    figsize: tuple[int, int] = (7, 5),
    ax: plt.Axes = None,

    title: str = "",

    legend_loc: Literal["top", "bottom", "right"] = "right",
    legend_ncols: int = 1,

    # IO
    path: str | Path = None,
    filename: str = "Volcano.svg",
    show: bool = True,

    # Statistics
    pval_lim: float = 2e-10,
    lfc_lim: tuple = (-6, 6),
    lfc_cut: float = 0.25,
    pval_cut: float = 0.05,

    # Fx Specific
    mygenes: list | None = None,
    clean: bool = True,
    dot_size: float = 3,
    topn: int = 10,
    textprops: dict = None,
    **kwargs,
) -> dict | None:
    """Generate a volcano plot.

    Genes will be colored differently depending on the p-value (Pval) and logfoldchange (LFC):

    * Genes Pval < pval_cut & LFC > lfc_cut: Red.
    * Genes Pval < pval_cut & LFC < lfc_cut: Blue.
    * Genes Pval > pval_cut & LFC > lfc_cut: Green.
    * Genes Pval > pval_cut & LFC < lfc_cut: Gray.

    If no genes are provided (`mygenes`) the top 10 genes with highest and lowest LFC that are
    significant will be indicated.

    :param df: pandas dataframe with DGE. Should have at least 3 columns (Genes, Pvalue, Logfoldchange).
    :param lfc_col: name of the column that has the logfoldchanges.
    :param pval_col: name of the column that has the Pvals.
    :param gene_col: name of the column that has the gene names.
    :param path: path where to save the figure.
    :param filename: name of the file.
    :param pval_lim: Y-axis limit. Genes with a < p-value will be set to this value.
    :param lfc_lim: X-axis limit. Genes with a > LFC will be ignored.
    :param title: a text to add as the title of the plot.
    :param figsize: size of the plot.
    :param ax: matplotlib axis.
    :param legend_loc: location of the legend.
    :param legend_ncols: number of columns for the legend.
    :param lfc_cut: significance threshold for the LFC.
    :param pval_cut: significance threshold for the P-value.
    :param mygenes: list of genes to be annotated.
    :param clean: remove genes with Pval == 1 and LFC > lfc_lim.
    :param dot_size: size of the dots.
    :param topn: if mygenes is None. The top 10 positive and negative genes are plotted.
    :param textprops: properties of the gene labels (See `plt.text <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.text.html>`_)
    :param show: if set to true, return axis.
    :return: Depending on ``show``, returns the plot if set to `True` or a dictionary with the axes.

    Example
    -------

    .. plot::
        :context: close-figs

        import dotools_py as do
        adata = do.dt.example_10x_processed()
        do.tl.rank_genes_groups(adata,  'condition', method='wilcoxon', tie_correct=True, pts=True)
        table = do.get.dge_results(adata)
        table = table[table.group == 'disease']
        do.pl.volcano_plot(table, 'log2fc', 'padj', 'GeneName')

    """
    from adjustText import adjust_text

    dge = df.copy()  # Do not Modify input

    # Data Preparation # # #
    ## Replace Pvals and LFCs greater than limit with the limit
    dge[pval_col][dge[pval_col] < pval_lim] = pval_lim
    dge[lfc_col][dge[lfc_col] < lfc_lim[0]] = lfc_lim[0]
    dge[lfc_col][dge[lfc_col] > lfc_lim[1]] = lfc_lim[1]

    if clean: # Remove Genes with P adjusted == 1 (Not Informative)
        dge = dge[dge[pval_col] < 1]
        dge = dge[dge[lfc_col] > lfc_lim[0]]
        dge = dge[dge[lfc_col] < lfc_lim[1]]

    ## Define 4 Categories for colors
    pvals, lfcs, genes = dge[pval_col].to_numpy(), dge[lfc_col].to_numpy(), dge[gene_col].to_numpy()
    cat1 = np.where((pvals < pval_cut) & ((lfcs > lfc_cut) | (lfcs < -lfc_cut)))[0]  # Significant for both
    cat2 = np.where((pvals < pval_cut) & (lfcs > -lfc_cut) & (lfcs < lfc_cut))[0]  # Significant for padj
    cat3 = np.where((pvals > pval_cut) & ((lfcs > lfc_cut) | (lfcs < -lfc_cut)))[0]  # Significant for lfc
    cat4 = [val for val in range(len(lfcs)) if val not in list(np.concatenate((cat1, cat2, cat3)))] # not significant

    # Set figure layout # # #
    width, height = figsize
    loc_key = 0
    if legend_loc == "right":
        fig_args = {"nrows": 1, "ncols": 2, "width_ratios":[width - (1.5 + 0) + 0, 1.5], "wspace": 0.7/width}
        legend_loc = "center left"
    elif legend_loc in ["bottom", "top"]:
        loc_key = loc_key + 1 if legend_loc == "top" else loc_key
        height_ratios = [height - (1.5 + 0) + 0, 1] if legend_loc == "bottom" else [1, height - (1.5 + 0) + 0]
        fig_args = {"nrows": 2, "ncols": 1, "height_ratios": height_ratios, "hspace": 0.7 / height}
        legend_loc = "center" if legend_loc =="bottom" else "center"
        legend_ncols = 2 if legend_ncols == 1 else legend_ncols
    else:
        raise  NotImplementedError(f"{legend_loc} is not a valid key for legend_loc")

    ## Generate the plot
    fig, gs = make_grid_spec(ax or (width, height), **fig_args)

    ## Add Main Axis
    axs = fig.add_subplot(gs[0]) if loc_key == 0 else fig.add_subplot(gs[1])
    axs.scatter(lfcs[cat1], -np.log10(pvals[cat1]), color="tomato", alpha=0.75, label="Padj & log2FC", s=dot_size**2,
                rasterized=True)
    axs.scatter(lfcs[cat2], -np.log10(pvals[cat2]), color="royalblue", alpha=0.75, label="Padj", s=dot_size**2,
                rasterized=True)
    axs.scatter(lfcs[cat3], -np.log10(pvals[cat3]), color="limegreen", alpha=0.75, label="log2FC", s=dot_size**2,
                rasterized=True)
    axs.scatter(lfcs[cat4], -np.log10(pvals[cat4]), color="gainsboro", alpha=0.75, label="NS", s=dot_size**2, rasterized=True)

    # Add lfc/pvals significance lines
    axs.axhline(-np.log10(pval_cut), color="black", linestyle="--", alpha=0.5)
    axs.axvline(-lfc_cut, color="black", linestyle="--", alpha=0.5)
    axs.axvline(lfc_cut, color="black", linestyle="--", alpha=0.5)

    textprops = {} if textprops is None else textprops
    textprops = {"weight": textprops.get("weight", "bold"), "size": textprops.get("size", 13)}

    # Add text
    topPos = (dge[(dge[pval_col] < pval_cut) & (dge[lfc_col] > lfc_cut)]
              .sort_values(lfc_col, ascending=False)[gene_col]
              .head(topn)
              .tolist())
    topNeg = (dge[(dge[pval_col] < pval_cut) & (dge[lfc_col] < -lfc_cut)]
              .sort_values(lfc_col, ascending=True)[gene_col]
              .head(topn)
              .tolist())
    texts = []
    for x, y, l in zip(lfcs, pvals, genes, strict=False):
        if mygenes is None:
            if l in topPos:
                texts.append(plt.text(x, -np.log10(y), l, ha="center", va="center", fontdict=textprops))
            if l in topNeg:
                texts.append(plt.text(x, -np.log10(y), l, ha="center", va="center", fontdict=textprops))
        else:
            if l in mygenes:
                texts.append(plt.text(x, -np.log10(y), l, ha="center", va="center", fontdict=textprops))
    adjust_text(texts, arrowprops={"arrowstyle": "-", "color": "k", "lw": 0.5}, **kwargs)

    # Layout for Main Axis
    axs.spines[["top", "right"]].set_visible(False)
    axs.grid(False)
    axs.set_xlabel("Log2FC")
    axs.set_ylabel("-log10(FDR)")
    axs.set_title(title)

    ## Add Legend Axis
    legend_axs = fig.add_subplot(gs[1]) if loc_key == 0 else fig.add_subplot(gs[0])
    handles = []
    for c, lab in [("tomato", "Padj & log2FC"), ("royalblue", "Padj"),
                   ("limegreen", "log2FC"), ("gainsboro", "NS")]:
        handles.append(mlines.Line2D([0], [0], marker=".",  color=c, lw=0, label=lab, markerfacecolor=c,
                                     markeredgecolor=None, markersize=18))
    legend_axs.legend(handles=handles, frameon=False, loc=legend_loc, ncols=legend_ncols, title="")
    legend_axs.tick_params(axis="both", left=False, labelleft=False, labelright=False, bottom=False, labelbottom=False)
    legend_axs.spines[["right", "left", "top", "bottom"]].set_visible(False)
    legend_axs.grid(visible=False)
    legend_axs.patch.set_alpha(0.0)

    if path is not None:
        plt.savefig(convert_path(path) / filename, bbox_inches="tight")
    if not show:
        return {"mainplot_ax": axs, "legend_ax": legend_axs}
    else:
        return plt.show()


def split_bar_gsea(
    df: pd.DataFrame,
    term_col: str,
    col_split: str,
    cond_col: str,
    pos_cond: str,
    cutoff: int = 40,
    log10_transform: bool = True,
    figsize: tuple[int, int] = (12, 8),
    topn: float = 10,
    colors_pairs: list = ("sandybrown", "royalblue"),
    alpha_colors: float = 0.3,
    path: str | None = None,
    spacing: float = 5,
    txt_size: float = 12,
    filename: str = "SplitBar.svg",
    title: str = "Top 10 GO Terms in each Condition",
    show: bool = True,
) -> plt.Axes | None:
    """Split BarPlot for GO terms.

    This function generates a split barplot. This is a plot where the top 10 GO terms are shown, sorted based on a
    column `col_split`. Two conditions are shown at the same time. One condition is shown in the positive axis,
    while the other in the negative one. The condition to be shown as positive is set with `pos_col`.

    .. warning::
        Expected a filtered dataframe containing only significant Terms

    :param df: dataframe with the results of a gene set enrichment analysis.
    :param term_col: column in the dataframe that contains the terms.
    :param col_split: column in the dataframe that will be used to sort and split the plot.
    :param cond_col: column in the dataframe that contains the condition information.
    :param pos_cond: condition that will be shown in the positive side of the plot.
    :param cutoff: maximum number of characters per line.
    :param log10_transform: if col_split contains values between 0 and 1, assume they are pvals and apply a -log10 transformation.
    :param figsize: figure size.
    :param topn: how many terms are shown.
    :param path: path to save the plot.
    :param filename: filename for the plot.
    :param spacing: space to add between bars and origin. It is a percentage value, indicating that the bars start at 5 % of the maximum X axis value.
    :param txt_size: size of the go terms text.
    :param alpha_colors: alpha value for the colors of the bars.
    :param colors_pairs: colors for each condition (1st color --> negative axis; 2nd color --> positive axis).
    :param title: title of the plot.
    :param show: if False, the axis is return.
    :return: Depending on ``show``, returns the plot if set to `True` or a dictionary with the axes.

    Example
    -------

    .. plot::
        :context: close-figs

        import dotools_py as do
        adata = do.dt.example_10x_processed()
        do.tl.rank_genes_groups(adata,  'condition', method='wilcoxon', tie_correct=True, pts=True)
        table = do.get.dge_results(adata)
        table = table[table.group == 'disease']
        table_go = do.tl.go_analysis(table, 'GeneName', 'padj', 'log2fc', specie='Human', go_catgs = ['GO_Molecular_Function_2023', 'GO_Cellular_Component_2023', 'GO_Biological_Process_2023'])
        table_go = table_go[table_go['P-value'] < 0.25]
        do.pl.split_bar_gsea(table_go, 'Term', 'Combined Score', 'state', 'enriched', show=True)

    """
    if len(df[cond_col].unique()) != 2:
        if len(df[cond_col].unique()) > 2:
            assert len(df[cond_col].unique()) == 2, "Not implement - Only 1 or 2 conditions can be used"
        elif len(df[cond_col].unique()) == 1:
            logger.warn("!!! WARNING - There are no terms for one of the conditions")
        else:
            assert len(df[cond_col].unique()) == 2, "Not implement - Only 1 or 2 conditions can be used"

    logger.warn("!!! Assuming GO Terms are preprocessed (Only Significant terms included)")

    df = df.copy()  # Ensure we do not modify the input
    jdx = list(df.columns).index(cond_col)  # Get index of the condition column

    # Update the col_split values; Positive values for one condition and
    # negative for the other positive. The positive is set by the 'pos_cond' argument
    min_val, max_val = df[col_split].min(), df[col_split].max()
    is_pval = True if (min_val >= 0) and (max_val <= 1) else False
    if is_pval and log10_transform:
        logger.warn("Assuming col_split contains Pvals, apply -log10 transformation")
        df["-log10(Padj)"] = -np.log10(df[col_split])
        col_split = "-log10(Padj)"
        spacing = 0.5  # Correct spacing in case it was not specified
    df[col_split] = [
        val if df.iloc[idx, jdx] == pos_cond else -val for idx, val in enumerate(df[col_split])
    ]  # Set negative and positive values for each condition

    # Format the Terms
    df[term_col] = df[term_col].str.capitalize()  # Capitalise
    df = format_terms_gsea(df, term_col, cutoff)  # Split terms too long in several rows

    # Get the dataframe for the positive and negative axis
    df_pos = df[df[cond_col] == pos_cond].sort_values(col_split, ascending=False).head(int(topn))
    df_neg = df[df[cond_col] != pos_cond].sort_values(col_split).head(int(topn))

    # Check that the size of the dataframes is equal
    if len(df_pos) != len(df_neg):
        logger.warn("Different number of GO Terms in positive and negative axis, adding empty rows")
        logger.warn(f"Positive side has {len(df_pos)} and Negative side has {len(df_neg)}")
        missing_rows = topn - len(df_pos) if len(df_pos) < len(df_neg) else topn - len(df_neg)
        missing_rows_data = [np.nan for val in range(len(df_pos.columns))]
        missing_df = pd.DataFrame([missing_rows_data] * missing_rows, columns=list(df_pos.columns))
        missing_df[term_col] = ""
        missing_df[col_split] = 0
        if len(df_pos) > len(df_neg):
            df_neg = pd.concat([df_neg, missing_df])
        else:
            df_pos = pd.concat([df_pos, missing_df])

    spacing_unit = np.abs(df[col_split]).max() * spacing / 100
    # Actual Plot
    fig, axs = plt.subplots(1, 1, figsize=figsize)
    y_pos = range(int(topn))

    # Plot bars for "Down" condition (positive values) on the left side
    bars_down = axs.barh(
        y_pos,
        df_neg[col_split].sort_values(ascending=False),
        left=-spacing_unit,
        color=colors_pairs[0],
        align="center",
        alpha=alpha_colors,
    )

    # Plot bars for "Up" condition (negative values) on the right side
    bars_up = axs.barh(
        y_pos,
        df_pos[col_split].sort_values(),
        left=spacing_unit,
        color=colors_pairs[1],
        align="center",
        alpha=alpha_colors,
    )

    # Layout
    axs.spines[["left", "top", "right"]].set_visible(False)
    axs.set_yticks([])
    axs.set_xlim(-np.abs(df[col_split]).max(), np.abs(df[col_split]).max())
    axs.set_xlabel(col_split, fontsize=18)
    axs.set_title(title, fontsize=20)
    axs.grid(False)
    plt.vlines(0, -1, float(topn) - 0.5, color="k", lw=1)
    axs.set_ylim(-0.5, float(topn))

    # Add text labels for each bar (GO term name)
    for i, bar in enumerate(bars_up):
        # Add the GO term for "Up" bars (positive)
        axs.text(
            spacing_unit * 2,
            bar.get_y() + bar.get_height() / 2,
            df_pos.sort_values(col_split)[term_col].iloc[i],
            va="center",
            ha="left",
            color="k",
            fontweight="bold",
            fontsize=txt_size,
        )

    for i, bar in enumerate(bars_down):
        # Add the GO term for "Down" bars (negative)
        axs.text(
            -spacing_unit * 2,
            bar.get_y() + bar.get_height() / 2,
            df_neg.sort_values(col_split, ascending=False)[term_col].iloc[i],
            va="center",
            ha="right",
            color="k",
            fontweight="bold",
            fontsize=txt_size,
        )
    # Save Plot
    save_plot(path=path, filename=filename)
    return  return_axis(show, axs, tight=False)
    # If show is False, return axs
    #if not show:
    #    return axs
    #else:
    #    return plt.show()
