import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from dotools_py.get import expr as get_expr
from dotools_py.utils import convert_path, get_subplot_shape, remove_extra, sanitize_anndata, spine_format


def layers(
    adata: ad.AnnData, color: str, layers: str, ncols: int = 4, normalise: bool = False, show: bool = True, **kwargs
) -> None | plt.Axes:
    """Plot several layers.

    Plot different layers in subplots. Useful for deconvolution analysis with celltype counts in layers.

    :param adata: annotated data matrix.
    :param color: var_names or obs column to plot.
    :param layers: layers to plot.
    :param ncols:  number of columns in the plot.
    :param normalise: do log-normalisation on the layers.
    :param show: if set to False, return axis.
    :param kwargs: additional arguments for `sc.pl.spatial <https://scanpy.readthedocs.io/en/latest/api/generated/scanpy.pl.spatial.html>`_.
    :return:  None or plt.axes.
    """
    import scanpy as sc
    adata = adata.copy()
    sanitize_anndata(adata)
    if normalise:
        for layer in tqdm(layers, desc="Normalised Layers"):
            sc.pp.normalize_total(adata, layer=layer)
            sc.pp.log1p(adata, layer=layer)
    nrows, ncols, extras = get_subplot_shape(len(layers), ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 8))
    axs = axs.flatten()
    for idx, ly in enumerate(layers):
        sc.pl.spatial(adata, color=color, ax=axs[idx], layer=ly, **kwargs)
        axs[idx].set_title(ly + "\n" + color)
        spine_format(axs[idx], "SP")
    remove_extra(extras, nrows, ncols, axs)
    if not show:
        return axs
    else:
        return None


def slides(
    adata: ad.AnnData,
    color: str,
    batch_key: str = "batch",
    ncols: int = 4,
    sp_size: float = 1.5,
    path: str = None,
    filename: str = "Spatial.svg",
    common_legend: bool = True,
    order: list = None,
    figsize: tuple = (15, 8),
    layer: str = None,
    img_key: str = "hires",
    title_fontsize: int = 15,
    title_fontweight: str = None,
    select_samples: list | str = None,
    show: bool = True,
    minimal_title: bool = True,
    vmax: float = None,
    verbose: bool = True,
    spacing: tuple = (0.3, 0.2),
    **kwargs,
) -> plt.Axes:
    """Plot multiple visium slides

    Plot a feature in var_names or a column from obs in multiple visium slides.

    :param adata: annotated data matrix.
    :param color:  var_names or obs column to plot.
    :param batch_key: obs column containing Batch/Sample Information. This column should have the same names system use
                    to save the spatial images in `adata.uns['spatial'].keys()`.
    :param ncols: number of subplots per row.
    :param sp_size: size of the dots.
    :param path: path to save the plot.
    :param filename: filename of the plot.
    :param common_legend: if set to true only the legend of the last column will be shown. Otherwise, the legend of
                          all the subplots will be shown.
    :param order: provide a list with the order of the slides to show. If not set the `obs_col` will be sorted.
    :param figsize: size of the subplots.
    :param layer: layer to use to plot dt. If not specified, `.X` will be used.
    :param img_key: image key to use for plotting (hires or lowres).
    :param title_fontsize: fontsize of the title for the subplots.
    :param title_fontweight: change fontweight of the title.
    :param select_samples: list with a subset of samplename that want to be plotted.
    :param show: if False, return axs.
    :param minimal_title: if set to true only the sample name will be shown as title, otherwise title + color
    :param vmax: maximum value for continus values (e.g., expression). If common legend is set to True and vmax
                 is not specified, it will be automatically computed taking the p99.2 expression value across
                 all subplots.
    :param verbose: show a progress bar when plotting multiple slides.
    :param kwargs: additional arguments for the function `scanpy.pl.spatial()`.
    :param spacing: spacing between subplots (height, width) padding between plots
    :return: a matplotlib axes object
    """
    import scanpy as sc
    # TODO Consider the case where we only have 1 sample
    sanitize_anndata(adata)

    if select_samples is not None:
        if type(select_samples) is str:
            select_samples = [select_samples]
        adata = adata[adata.obs[batch_key].isin(select_samples)].copy()
        adata.obs[batch_key] = pd.Categorical(adata.obs[batch_key].astype(str))

    # Define the number of rows base on the desired number of columns
    n_samples = len(adata.obs[batch_key].unique())
    nrows, ncols, extras = get_subplot_shape(n_samples, ncols)

    # Control the order of the samples
    show_order = order
    if order is None:
        show_order = sorted(adata.obs[batch_key].unique())  # If no order is provided sort the samples

    # Assume we plot non-categorical values
    cat, cb_loc, cont = None, "right", 0

    # Scale values if common legend
    if vmax is None and common_legend is True:
        if color in adata.var_names:
            expr = get_expr(adata, color)
            vmax = np.percentile(expr["expr"], 99.2)
        if color in adata.obs.columns and adata.obs[color].dtype.name != "category":
            vmax = np.percentile(adata.obs[color], 99.2)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    plt.subplots_adjust(hspace=spacing[0], wspace=spacing[1], left=0.05)  # Spacing between subplots
    axs = axs.flatten()
    for idx, sample in tqdm(enumerate(show_order), desc="Slide ", disable=not verbose, total=len(show_order)):
        sdata = adata[adata.obs[batch_key] == sample]

        if common_legend:
            # Remove the legend from all subplots except the last one per row
            if cont != ncols - 1 and idx != n_samples - 1:
                if color in adata.obs.columns:  # Is color in .obs?
                    cat = adata.obs[color].dtype.name  # It can be continuous or categorical
                if cat != "category":
                    # Is continuous --> Remove color bar
                    cb_loc = None
                cont += 1
            else:
                # Entered when we are in the last column per row
                cb_loc = "right"
                cont = 0
                cat = None

        # Main Plotting function, based on Scanpy
        sc.pl.spatial(
            sdata,
            ax=axs[idx],
            img_key=img_key,
            color=color,
            library_id=sample,
            size=sp_size,
            colorbar_loc=cb_loc,
            layer=layer,
            vmax=vmax,
            show=False,
            **kwargs,
        )

        if common_legend and cat == "category":
            axs[idx].get_legend().remove()  # Remove legend for categorical values

        # Modify axis
        title_color = "" if color is None else color
        """
        if color == None:
            title_color = ''
        else:
            title_color = color
        """
        if minimal_title:
            axs[idx].set_title(sample, fontsize=title_fontsize, fontweight=title_fontweight)
            # plt.suptitle(color, fontsize=23, fontweight='bold')
            fig.supylabel(color, fontsize=23, fontweight="bold")
        else:
            axs[idx].set_title(sample + "\n" + title_color, fontsize=title_fontsize, fontweight=title_fontweight)

        spine_format(axs[idx], txt="SP")
        remove_extra(extras, nrows, ncols, axs)  # Remove extra subplots
        if path is not None:
            plt.savefig(convert_path(path) / filename, bbox_inches="tight")
    if show:
        return None
    else:
        return axs
