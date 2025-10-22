import os

import anndata as ad
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from adjustText import adjust_text

from dotools_py.get._generic import expr as get_expr
from dotools_py.utility import spine_format
from dotools_py.utils import convert_path, get_centroids, get_subplot_shape, remove_extra, sanitize_anndata


def embedding(
    adata: ad.AnnData,
    color: str | list,
    split_by: str | None = None,
    order_catgs: list = None,
    ncols: int = 4,
    title_font: dict = None,
    figsize: tuple = (6, 5),
    common_legend: bool = False,
    title: str = None,
    vmax: float | None = None,
    spacing: tuple = (0.3, 0.2),
    path: str | None = None,
    filename: str = "Umap.svg",
    show: bool = True,
    labels: str = None,
    labels_fontproporties: dict = None,
    labels_repel: dict = None,
    basis: str = "X_umap",
    ax: plt.Axes = None,
    **kwargs,
) -> plt.Axes | None:
    """Make Embedding Plot.

    This function builds on `sc.pl.embedding() <https://scanpy.readthedocs.io/en/latest/api/generated/scanpy.pl.embedding.html>`_
    and add extra functionalities like splitting by a categorical column in obs.

    :param adata: annotated data matrix.
    :param color: `.obs` column or `.var_names` value.
    :param split_by: categorical `.obs` column.
    :param order_catgs: order of the categories when splitting by a categorical column.
    :param ncols: number of columns per row.
    :param figsize: figure size (width, heigh) in inches.
    :param common_legend: set a common legend when plotting multiple values, it will automatically scale if plotting continuous values like
                          gene expression if vmax is not specified.
    :param title: title of the plot. Only used when 1 value is plotted. If 1 value is plotted splitting by categories, the title
                  will be the categories. If several values are plotted the title will be each value.
    :param title_font: font properties of the title for each subplot.
    :param vmax: maximum value for continuos data.
    :param spacing: spacing between subplots (height, width) padding between plots.
    :param show: when set to False the matplotlib axes will be returned.
    :param labels: `.obs` column name with categorical values to add to the plot.
    :param labels_fontproporties: font-properties for the labels.
    :param labels_repel: additional arguments pass to adjust_text.
    :param basis: embedding to use, default UMAP.
    :param ax: matplotlib axis.
    :param path: path to save plot.
    :param filename: filename of the plot.
    :param kwargs: additional parameters pass to `sc.pl.embedding() <https://scanpy.readthedocs.io/en/latest/api/generated/scanpy.pl.embedding.html>`_.
    :return: Depending on ``show``, returns the plot if set to `True` or a dictionary with the axes.
    """
    sanitize_anndata(adata)

    def _plot_labels_embedding(
        axis: plt.Axes, centroids: pd.DataFrame, fontweight: float | str, fontsize: float, fontoutline: float
    ) -> list:
        txts = []
        for label, row in centroids.iterrows():
            text = axis.text(
                row["x"],
                row["y"],
                label,
                weight=fontweight,
                fontsize=fontsize,
                verticalalignment="center",
                horizontalalignment="center",
                path_effects=[path_effects.withStroke(linewidth=fontoutline, foreground="w")],
            )
            txts.append(text)
        return txts

    # adata = adata.copy()  # We copy to not modify input

    # Labels is used when plotting inside the plot
    if labels is not None:
        labels_centroids = get_centroids(adata, labels, basis=basis)
        if labels_fontproporties is None:
            labels_fontproporties = {}

        labels_fontproporties.update(
            {
                "size": labels_fontproporties.get("size", 12),
                "weight": labels_fontproporties.get("weight", "bold"),
                "outline": labels_fontproporties.get("outline", 1.5),
            }
        )
        (labels_fontweight, labels_fontsize, labels_fontoutline) = (
            labels_fontproporties["weight"],
            labels_fontproporties["size"],
            labels_fontproporties["outline"],
        )

    # We consider that the input is always a list;
    color = [color] if isinstance(color, str) else color

    # Avoid problems with colors
    for c in color:
        if c in list(adata.obs.columns) and c + '_colors' in adata.uns.keys():
            if len(adata.obs[c].cat.categories) != len(adata.uns[c + '_colors']):
                del adata.uns[c + '_colors']

    # font-properties for the title
    if title_font is None:
        title_font = {}

    title_font.update({"size": title_font.get("size", 18), "weight": title_font.get("weight", "bold")})

    # When plotting only one thing, we can define the title
    if title is None and len(color) == 1:
        title = color[0]
    if labels_repel is None:
        labels_repel = {}

    if basis == "X_umap":
        txt_basis = "UMAP"
    elif basis == "X_spatial" or basis == "spatial":
        txt_basis = "SP"
    else:
        txt_basis = basis

    # If a .obs column is provided plot will have as many subplots as categories
    ncatgs = 1
    if split_by is not None:
        assert adata.obs[split_by].dtype == "category", "split_by is not a categorical column"
        ncatgs = len(adata.obs[split_by].unique())

        if order_catgs is not None:
            assert len(adata.obs[split_by].unique()) == len(order_catgs), (
                f"Number of categories provided != ccategories in {split_by}"
            )
            catgs = order_catgs
        else:
            catgs = adata.obs[split_by].unique()
        nrows, ncols, nExtra = get_subplot_shape(ncatgs, ncols)
    else:  # Otherwise, we have as many subplots as things to plot (len(colors))
        nrows, ncols, nExtra = get_subplot_shape(len(color), ncols)

    # Scale vmax when setting a common legend
    vmax_genes = vmax
    if vmax is None and common_legend is True:
        # We could be plotting different genes
        if len(color) > 1:
            genes = [val for val in color if val in adata.var_names]
            expr = get_expr(adata, features=genes, out_format="wide")  # Extract the expression
            vmax_genes = expr.apply(
                lambda x: np.percentile(x, 99.2), axis=0
            ).mean()  # the vmax is the mean of 99.2 percentile across genes
        else:  # We could also be plotting one gene splitting by categories
            if split_by is not None and color[0] in adata.var_names:

                def q99_2(x):
                    return x.quantile(0.992)

                # We plot one value but split by something
                expr = get_expr(adata, features=color[0], groups=split_by, out_format="wide")  # Extract the expression
                vmax_genes = (
                    expr.groupby(split_by).agg(q99_2).mean()[0]
                )  # the vmax is the mean of 99.2 percentile across categories

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Generate the Plot                                       #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    if ax is None:
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
        plt.subplots_adjust(hspace=spacing[0], wspace=spacing[1], left=0.1)  # Spacing between subplots
    else:
        axs = ax
    cat, cb_loc, cont = None, "right", 0
    # 1st Case; - We do not split by categories and only 1 thing is plotted
    if ncatgs == 1:
        if len(color) == 1:
            color = color[0]  # Color is always a list
            sc.pl.embedding(
                adata, basis=basis, color=color, ax=axs, vmax=vmax, show=False, **kwargs
            )  # Use embedding to generalise
            axs.set_title(title, fontdict=title_font)
            spine_format(axs, txt_basis)
            if labels is not None:
                texts = _plot_labels_embedding(
                    axs, labels_centroids, labels_fontweight, labels_fontsize, labels_fontoutline
                )
                adjust_text(texts, ax=axs, **labels_repel)

        # 2nd Case; We do not split by categories and multiple values are plotted
        else:
            axs = axs.flatten()
            for idx, val in enumerate(color):
                if common_legend:
                    # Remove the legend from all subplots except the last one per row
                    if (
                        cont != ncols - 1 and idx != len(color) - 1
                    ):  # We remove legend from all subplots except last column per row
                        if val in adata.obs.columns:  # Is color in .obs?
                            cat = adata.obs[val].dtype.name  # It can be continuous or categorical
                        if cat != "category":
                            # Is continuous --> Remove color bar
                            cb_loc = None
                        cont += 1
                    else:
                        # Entered when we are in the last column per row
                        cat, cb_loc, cont = None, "right", 0

                # If value to plot is a gene, update vmax (if common legend true) otherwise use the vmax provided by user
                vmax = vmax_genes if val in adata.var_names else vmax
                sc.pl.embedding(
                    adata, color=val, ax=axs[idx], colorbar_loc=cb_loc, basis=basis, vmax=vmax, show=False, **kwargs
                )  # use embedding to generalise
                spine_format(axs[idx], txt_basis)
                axs[idx].set_title(val, fontdict=title_font)
                remove_extra(nExtra, nrows, ncols, axs)
                if labels is not None:
                    texts = _plot_labels_embedding(
                        axs[idx], labels_centroids, labels_fontweight, labels_fontsize, labels_fontoutline
                    )
                    adjust_text(texts, ax=axs[idx], **labels_repel)

                # Never remove categorical when plotting several values without splitting by
                # if common_legend and cat == 'category':
                #    axs[idx].get_legend().remove()  # Remove legend for categorical values

    else:
        # 3rd Case Multiple Values are plotted and splitting by categories
        # 3rd case plot each category per row
        assert len(color) == 1, "Not Implemented"

        # 4th Case; One value is plotted splitting by categories
        color = color[0]  # color is always converted to list
        axs = axs.flatten()
        for idx in range(ncatgs):
            adata_subset = adata[adata.obs[split_by] == catgs[idx]]
            if common_legend:
                # Remove the legend from all subplots except the last one per row
                if cont != ncols - 1 and idx != ncatgs - 1:
                    if color in adata.obs.columns:  # Is color in .obs?
                        cat = adata.obs[color].dtype.name  # It can be continuous or categorical
                    if cat != "category":
                        # Is continuous --> Remove color bar
                        cb_loc = None
                    cont += 1
                else:
                    # Entered when we are in the last column per row
                    cat, cb_loc, cont = None, "right", 0

            # If value to plot is a gene, update vmax (if common legend true) otherwise use the vmax provided by user
            vmax = vmax_genes if color in adata.var_names else vmax

            sc.pl.embedding(
                adata_subset,
                basis=basis,
                color=color,
                ax=axs[idx],
                colorbar_loc=cb_loc,
                vmax=vmax,
                show=False,
                **kwargs,
            )  # embedding to generalise
            spine_format(axs[idx], txt_basis)
            remove_extra(nExtra, nrows, ncols, axs)

            if common_legend and cat == "category":
                try:
                    axs[idx].get_legend().remove()  # Remove legend for categorical values except last column
                except AttributeError:
                    pass
            if labels is not None:
                texts = _plot_labels_embedding(
                    axs[idx], labels_centroids, labels_fontweight, labels_fontsize, labels_fontoutline
                )
                adjust_text(texts, ax=axs[idx], **labels_repel)

            # Minimal Text when Splitting by categories
            axs[idx].set_title(catgs[idx], fontdict=title_font)
            fig.supylabel(color, fontsize=23, fontweight="bold")

    if path is not None:
        plt.savefig(os.path.join(path, filename), bbox_inches="tight")
    if show:
        return plt.show()
    else:
        return axs


def umap(
    adata: ad.AnnData,
    color: str,
    split_by: str | None = None,
    order_catgs: list = None,
    ncols: int = 4,
    title_font: dict = None,
    figsize: tuple = (6, 5),
    common_legend=False,
    title: str = None,
    vmax: float | None = None,
    spacing: tuple = (0.3, 0.2),
    path: str | None = None,
    filename: str = "Umap.svg",
    show: bool = True,
    labels: str = None,
    labels_fontproporties: dict = None,
    labels_repel: dict = None,
    ax: plt.Axes = None,
    **kwargs,
) -> None | plt.Axes:
    """Make UMAP Plot.

    This function builds on `sc.pl.embedding() <https://scanpy.readthedocs.io/en/latest/api/generated/scanpy.pl.embedding.html>`_ and add extra functionalities like
    splitting by a categorical column in `.obs`.

    :param adata: annotated data matrix.
    :param color: `.obs` column or `.var_names` value.
    :param split_by: categorical `.obs` column.
    :param order_catgs: order of the categories when splitting by a categorical column.
    :param ncols: number of columns per row.
    :param figsize: figure size (width, heigh) in Inches.
    :param common_legend: set a common legend when plotting multiple values, it will automatically scale if plotting continuous values like
                          gene expression if vmax is not specified.
    :param title: title of the plot. Only used when 1 value is plotted. If 1 value is plotted splitting by categories, the title
                  will be the categories. If several values are plotted the title will be each value.
    :param title_font: font properties of the title for each subplot.
    :param vmax: maximum value for continuos data.
    :param spacing: spacing between subplots (height, width) padding between plots.
    :param show: when set to False the matplotlib axes will be returned.
    :param labels: `.obs` column name with categorical values to add to the plot.
    :param labels_fontproporties: fontproperties for the labels.
    :param labels_repel: additional arguments pass to adjust_text.
    :param ax: matplotlib axis.
    :param path: path to save plot.
    :param filename: filename of the plot.
    :param kwargs: additional parameters pass to `sc.pl.embedding() <https://scanpy.readthedocs.io/en/latest/api/generated/scanpy.pl.embedding.html>`_
    :return: Depending on ``show``, returns the plot if set to `True` or a dictionary with the axes.

    Example
    -------
    Visualise a categorical column

    .. plot::
        :context: close-figs

        import dotools_py as do
        adata = do.dt.example_10x_processed()
        do.pl.umap(adata, 'annotation', split_by='condition', ncols=2, figsize=(9, 4), size=20)
        do.pl.umap(adata, 'CD4', split_by='condition',  size=50, labels='annotation', cmap='Reds')

    or the expression of a gene

    .. plot::
        :context: close-figs

        do.pl.umap(adata, 'CD4', split_by='condition',  size=50, labels='annotation', cmap='Reds')

    """
    axis = embedding(
        adata=adata,
        color=color,
        split_by=split_by,
        order_catgs=order_catgs,
        ncols=ncols,
        title_font=title_font,
        figsize=figsize,
        common_legend=common_legend,
        title=title,
        vmax=vmax,
        spacing=spacing,
        path=path,
        filename=filename,
        show=show,
        labels=labels,
        labels_fontproporties=labels_fontproporties,
        labels_repel=labels_repel,
        basis="X_umap",
        ax=ax,
        **kwargs,
    )

    if show:
        return plt.show()
    else:
        return axis


def split_embeddding(
    adata: ad.AnnData,
    split_by: str,
    ncols: int = 4,
    title_font: dict = None,
    path: str = None,
    filename: str = "UMAP.svg",
    figsize: tuple = (6, 5),
    basis: str = "X_umap",
    visium: bool = False,
    sp_size: float = 1.5,
    show: bool = True,
    **kwargs,
) -> plt.Axes | None:
    """Plot categorical data split in an embedding.

    This function takes an AnnData and a categorical column in obs and generate a plot of subplots  highlighting the
    different categories of the obs column.

    :param adata: annotated data matrix object.
    :param split_by: obs column with categorical values.
    :param ncols: number of subplots per row.
    :param title_font: properties of the title font for each subplot.
    :param path: path to save the plot.
    :param filename: filename of the plot.
    :param figsize: size of the figure.
    :param basis: embedding to use.
    :param visium: set to True if you anndata has visium data.
    :param sp_size: spot size when plotting visium data.
    :param show: if set to True returns axes.
    :param kwargs: additional arguments for `sc.pl.embedding() <https://scanpy.readthedocs.io/en/latest/api/generated/scanpy.pl.embedding.html>`_ or `sc.pl.spatial() <https://scanpy.readthedocs.io/en/latest/api/generated/scanpy.pl.spatial.html>`_ if visium is True.
    :return: Depending on ``show``, returns the plot if set to `True` or a dictionary with the axes.

    Example
    -------

    .. plot::
        :context: close-figs

        import dotools_py as do
        adata = do.dt.example_10x_processed()
        do.pl.split_embeddding(adata, 'annotation', ncols=3)

    """
    assert adata.obs[split_by].dtypes == "category", "Not a categorical column"
    sanitize_anndata(adata)

    if title_font is None:
        title_font = {}

    title_font.update({"size": title_font.get("size", 18), "weight": title_font.get("weight", "bold")})

    # Set-Up
    categories = adata.obs[split_by].cat.categories
    nrows, ncols, nextra = get_subplot_shape(len(categories), ncols)

    # Plotting
    figs, axs = plt.subplots(nrows, ncols, figsize=figsize)
    axs = axs.flatten()
    for idx, cat in enumerate(categories):
        if visium:
            sc.pl.spatial(adata, ax=axs[idx], groups=[cat], size=sp_size, show=False, **kwargs)
        else:
            sc.pl.embedding(
                adata, basis=basis, color=split_by, groups=[cat], ax=axs[idx], title=str(cat), show=False, **kwargs
            )
        axs[idx].set_title(cat, fontdict=title_font)
        axs[idx].get_legend().remove()
        spine_format(axs[idx])
        remove_extra(nextra, nrows, ncols, axs)
    if path is not None:
        plt.savefig(convert_path(path) / filename, bbox_inches="tight")
    if show:
        return plt.show()
    else:
        return axs
