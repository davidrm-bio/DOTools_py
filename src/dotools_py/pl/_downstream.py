import sys
from pathlib import Path

import anndata as ad
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text

from dotools_py import logger
from dotools_py.utils import convert_path, format_terms_gsea, make_grid_spec, sanitize_anndata, require_dependencies
from typing import Union

@require_dependencies([{'name': 'scanpro'}])
def cell_props(
    adata: ad.AnnData,
    annot_key: str,
    cond_key: str,
    batch_key: str,
    annot_order: Union[list, None] = None,
    cond_order: Union[list, None] = None,
    covariates: Union[list, None] = None,
    subset_cells: Union[list, None] = None,
    pval_cutoff: float = 0.05,
    figsize: tuple = (5, 6),
    axis: Union[plt.Axes, None] = None,
    path: Union[Path, str] = None,
    filename: str = "Proportions.svg",
    legend_cols: int = 1,
    sep: float = 0.5,
    bar_width: float = 0.2,
    title: str = "",
    title_fontsize: int = 15,
    legend_fontsize: int = 12,
    legend_fontweight: Union[float, str] = None,
    show: bool = True,
    legend_title: str = "",
    add_total_ncell: bool = True,
    transform: str = "logit",
    linewidth: float = 0.9,
    get_props: bool = False,
    **kwargs,
) -> Union[None, pd.DataFrame, plt.Axes]:
    """Stacked barplot showing changes in celltype proportions.

    Make a stacked barplot to show changes in celltype proportions between different conditions. Significant changes
    in cell proportions between conditions will tested with `scanpro <https://github.com/loosolab/scanpro>` and will be
    indicated by a discontinued line. The significant p-value/FDR will be shown in the legend.

    :param adata: annotated data matrix.
    :param annot_key: `.obs` column name with cell type annotation.
    :param cond_key: `.obs` column name with condition information.
    :param batch_key: `.obs` column name with batch IDs.
    :param annot_order: `.obs` column name with sample information. If None or the datasets has no replicates, then
                        replicates will be simulated. Additional arguments can be provided to control simulations.
    :param cond_order: order for the conditions.
    :param covariates: additional covariates for the model.
    :param subset_cells: only show a subset of the celltypes. The test is applied over all cell type populations.
    :param pval_cutoff: pval/FDR cutoff.
    :param figsize: figure size.
    :param axis: matplotlib axis.
    :param path: path to save the figure.
    :param filename: name of the file.
    :param legend_cols: number of columns for the legend.
    :param sep: separation between bars.
    :param bar_width: bars width.
    :param title: title of the plot.
    :param title_fontsize: fontsize of the title.
    :param legend_fontsize: fontsize of the legend.
    :param legend_fontweight: fontweight of the legend.
    :param show: whether to return or not the matplotlib axis. To return the axis, set to False.
    :param legend_title: title for the legend.
    :param add_total_ncell: add the total number of cells in the dataset.
    :param transform: transformation applied to test for significant differences. Default logit, set to arcsin if simulations
                      are performed for more accurate results.
    :param linewidth: thickness of the lines connecting significant bars.
    :param get_props: get a dataframe with the proportions and pvals.
    :param kwargs: additional arguments pass to scanpro().
    :return: None, matplolib axis or dataframe with results of scanpro.
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

    if cond_order is not None:
        assert all(x in cond_order for x in list(adata.obs[cond_key].cat.categories)), (
            "condition order is missing categories"
        )
        adata.obs[cond_key] = pd.Categorical(adata.obs[cond_key], categories=cond_order, ordered=True)

    out = scanpro(
        adata,
        clusters_col=annot_key,
        conds_col=cond_key,
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

    cond_keys = [f"mean_props_{cond}" for cond in adata.obs[cond_key].cat.categories]
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
        axis or (width, height), nrows=1, ncols=2, wspace=0.7 / width, width_ratios=[width - (1.5 + 0) + 0, 1.5]
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
    axs.set_xticks(xtick, xtext, fontweight="bold")
    axs.set_title(title, fontsize=title_fontsize, fontweight="bold")

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

    legend = axs_legend.legend(
        handles=handles,
        frameon=False,
        loc="center left",
        ncols=legend_cols,
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
    dge: pd.DataFrame,
    lfc_col: str = "logfoldchanges",
    pval_col: str = "pvals_adj",
    gene_col: str = "names",
    fig_path: Union[str, None] = None,
    filename: str = "Volcano.svg",
    pval_lim: float = 2e-10,
    lfc_lim: tuple = (-10, 10),
    title: str = "",
    figsize: tuple[int, int] = (18, 9),
    mygenes: Union[list, None] = None,
    lfc_cut: float = 0.25,
    pval_cut: float = 0.05,
    clean: bool = True,
    dot_size: float = 2.5,
    topn: int = 10,
    textprops: dict = None,
    show: bool = False,
    **kwargs,
) -> Union[plt.Axes, None]:
    """Generate a volcano plot.

    Genes will be colored differently depending on the p-value (Pval) and logfoldchange (LFC):

    * Genes Pval < pval_cut & LFC > lfc_cut: Red.
    * Genes Pval < pval_cut & LFC < lfc_cut: Blue.
    * Genes Pval > pval_cut & LFC > lfc_cut: Green.
    * Genes Pval > pval_cut & LFC < lfc_cut: Gray.

    If no genes are provided (with the mygenes argument) the top 10 genes with highest and lowest LFC that are
    significant will be marked.

    :param dge: pandas dataframe with DGE. Should have at least 3 columns (Genes, Pvalue, Logfoldchange).
    :param lfc_col: name of the column that has the logfoldchanges.
    :param pval_col: name of the column that has the Pvals.
    :param gene_col: name of the column that has the gene names.
    :param fig_path: path where to save the figure.
    :param filename: name of the file.
    :param pval_lim: Y-axis limit. Genes with a < p-value will be set to this value.
    :param lfc_lim: X-axis limit. Genes with a > LFC will be ignored.
    :param title: a text to add as the title of the plot.
    :param figsize: size of the plot.
    :param lfc_cut: significance threshold for the LFC.
    :param pval_cut: significance threshold for the P-value.
    :param mygenes: list of genes to be annotated.
    :param clean: remove genes with Pval == 1 and LFC > lfc_lim.
    :param dot_size: size of the dots.
    :param topn: if mygenes is None. The top 10 positive and negative genes are plotted.
    :param textprops: properties of the gene labels (See `plt.text <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.text.html>`_)
    :param show: if set to true, return axis.
    :return: Volcano plot.
    """
    dge = dge.copy()  # Do not Modify input

    textprops = {} if textprops is None else textprops
    textprops = {'weight': textprops.get('weight', 'bold'),
                 'size': textprops.get('size', 13)}

    # Replace Pvals & LFC greater than limit to the limit
    dge[pval_col][dge[pval_col] < pval_lim] = pval_lim

    assert lfc_lim[0] < lfc_lim[1], f"{lfc_lim[0]} cannot be greater than {lfc_lim[1]}"
    dge[lfc_col][dge[lfc_col] < lfc_lim[0]] = lfc_lim[0]
    dge[lfc_col][dge[lfc_col] > lfc_lim[1]] = lfc_lim[1]

    if clean:
        # Remove Genes with P adjusted == 1 (Not Informative)
        dge = dge[dge[pval_col] < 1]
        dge = dge[dge[lfc_col] > lfc_lim[0]]
        dge = dge[dge[lfc_col] < lfc_lim[1]]

    # Define 3 Categories: LFC > lfc_cut; Pval < pval_cut & combination
    pvals = dge[pval_col].to_numpy()
    lfcs = dge[lfc_col].to_numpy()
    genes = dge[gene_col].to_numpy()
    cat1 = np.where((pvals < pval_cut) & ((lfcs > lfc_cut) | (lfcs < -lfc_cut)))
    cat2 = np.where((pvals < pval_cut) & (lfcs > -lfc_cut) & (lfcs < lfc_cut))
    cat3 = np.where((pvals > pval_cut) & ((lfcs > lfc_cut) | (lfcs < -lfc_cut)))

    # Generate Plot
    # Create scatter Plot
    fig, axs = plt.subplots(1, 1, figsize=figsize)
    axs.scatter(lfcs, -np.log10(pvals), color="grey", alpha=0.7, label="NS", s=dot_size**2, rasterized=True)
    axs.scatter(
        lfcs[cat1], -np.log10(pvals[cat1]), color="tomato", alpha=0.7, label="FDR & log2FC", s=dot_size**2, rasterized=True
    )
    axs.scatter(
        lfcs[cat2], -np.log10(pvals[cat2]), color="lightsteelblue", alpha=0.7, label="FDR", s=dot_size**2, rasterized=True
    )
    axs.scatter(
        lfcs[cat3], -np.log10(pvals[cat3]), color="limegreen", alpha=0.7, label="log2FC", s=dot_size**2, rasterized=True
    )
    axs.spines[["top", "right"]].set_visible(False)
    axs.grid(False)

    # Add significant lines
    axs.axhline(-np.log10(pval_cut), color="black", linestyle="--", alpha=0.8)
    axs.axvline(-lfc_cut, color="black", linestyle="--", alpha=0.8)
    axs.axvline(lfc_cut, color="black", linestyle="--", alpha=0.8)

    topPos = (
        dge[(dge[pval_col] < pval_cut) & (dge[lfc_col] > lfc_cut)]
        .sort_values(lfc_col, ascending=False)[gene_col]
        .head(topn)
        .tolist()
    )
    topNeg = (
        dge[(dge[pval_col] < pval_cut) & (dge[lfc_col] < -lfc_cut)]
        .sort_values(lfc_col, ascending=True)[gene_col]
        .head(topn)
        .tolist()
    )
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
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color="k", lw=0.5), **kwargs)

    # Add Axis labels, Legend, & Title
    axs.set_xlabel("Log2FC")
    axs.set_ylabel("-log10(FDR)")
    axs.set_title(title)
    axs.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False, ncols=2, markerscale=dot_size, prop={'weight': 'bold'})

    if fig_path is not None:
        plt.savefig(convert_path(fig_path) / filename, bbox_inches="tight")
    if not show:
        return axs
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
    path: Union[str, None] = None,
    spacing: float = 5,
    txt_size: float = 12,
    filename: str = "SplitBar.svg",
    title: str = "Top 10 GO Terms in each Condition",
    show: bool = True,
) -> Union[plt.Axes, None]:
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
    :return: None or the axis
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
    if path is not None:
        plt.savefig(convert_path(path) / filename, bbox_inches="tight")

    # If show is False, return axs
    if not show:
        return axs
    else:
        return plt.show()
