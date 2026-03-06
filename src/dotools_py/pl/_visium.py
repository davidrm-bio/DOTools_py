from typing import Mapping
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from dotools_py.get import expr as get_expr
from dotools_py.utils import convert_path, get_subplot_shape, remove_extra, sanitize_anndata, spine_format, iterase_input
from dotools_py._custom_class import InputError

from dotools_py.pl._embeddings import embedding

def layers(
    adata: ad.AnnData, color: str, key_layers: str | list, ncols: int = 4, normalise: bool = False, show: bool = True, **kwargs
) -> None | plt.Axes:
    """Plot several layers.

    Plot different layers in subplots. Useful for deconvolution analysis with celltype counts in layers.

    :param adata: annotated data matrix.
    :param color: var_names or obs column to plot.
    :param key_layers: layers to plot.
    :param ncols:  number of columns in the plot.
    :param normalise: do log-normalization on the layers.
    :param show: if set to False, return axis.
    :param kwargs: additional arguments for `sc.pl.spatial <https://scanpy.readthedocs.io/en/latest/api/generated/scanpy.pl.spatial.html>`_.
    :return:  None or plt.axes.
    """
    import scanpy as sc
    adata = adata.copy()
    sanitize_anndata(adata)
    if normalise:
        for layer in tqdm(key_layers, desc="Normalised Layers"):
            sc.pp.normalize_total(adata, layer=layer)
            sc.pp.log1p(adata, layer=layer)
    nrows, ncols, extras = get_subplot_shape(len(key_layers), ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 8))
    axs = axs.flatten()
    for idx, ly in enumerate(key_layers):
        sc.pl.spatial(adata, color=color, ax=axs[idx], layer=ly, **kwargs)
        axs[idx].set_title(ly + "\n" + color)
        spine_format(axs[idx], "SP")
    remove_extra(extras, nrows, ncols, axs)
    if not show:
        return axs
    else:
        return None




def _check_spatial_data(
    uns: dict | Mapping,
    library_id: str = None
) -> tuple:
    spatial_mapping = uns.get("spatial", {})
    if library_id is None:
        if len(spatial_mapping) > 1:
            raise ValueError(
                f"Found multiple possible libraries in `adata.uns['spatial']`. Please specify one:"
                f"\n{uns['spatial'].keys()}"
            )
        elif len(spatial_mapping) == 1:
            library_id = next(iter(spatial_mapping.keys()))
        else:
            library_id = None
    spatial_data = spatial_mapping[library_id] if library_id is not None else None
    return library_id, spatial_data




def _check_img(
    spatial_data: Mapping | None,
    img: np.ndarray | None,
    img_key: None | str ,
    bw: bool = False,
) -> tuple[np.ndarray | None, str | None]:
    if img is None and spatial_data is not None and img_key is None:
        img_key = next(
            (k for k in ["hires", "lowres"] if k in spatial_data["images"]),
        )  # Throws StopIteration Error if keys not present
    if img is None and spatial_data is not None and img_key is not None:
        img = spatial_data["images"][img_key]
    if bw:
        img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    return img, img_key

def _check_spot_size(spatial_data: Mapping | None, spot_size: float | None) -> float:
    if spatial_data is None and spot_size is None:
        raise  ValueError(
            "When .uns['spatial'][library_id] does not exist, spot_size must be provided directly."
        )
    elif spot_size is None:
        return spatial_data["scalefactors"]["spot_diameter_fullres"]
    else:
        return spot_size

def _check_scale_factor(spatial_data: Mapping | None, img_key: str | None, scale_factor: float | None,) -> float:
    if scale_factor is not None:
        return scale_factor
    elif spatial_data is not None and img_key is not None:
        return spatial_data["scalefactors"][f"tissue_{img_key}_scalef"]
    else:
        return 1.0

def _check_crop_coord(crop_coord: tuple | None, scale_factor: float) -> tuple | None:
    if crop_coord is None:
        return None
    if len(crop_coord) != 4:
        raise ValueError(f"Invalid crop_coord of length {len(crop_coord)}(!=4)")
    crop_coord = tuple(float(c * scale_factor) for c in crop_coord)
    return crop_coord


def _check_na_color(
    na_color: None, img: np.ndarray | None = None
) :
    if na_color is None:
        na_color = (0.0, 0.0, 0.0, 0.0) if img is not None else "lightgray"
    return na_color


def _spatial(
    adata: ad.AnnData,
    color: str | list,
    library_id: str | Mapping = None,
    img: np.ndarray | None = None,
    img_key: str | None = "hires",
    bw: bool = False,
    spot_size: float = None,
    scale_factor: float = None,
    crop_coord: tuple = None,
    na_color: str | tuple = None,
    size: float = 1.5,
    basis: str = "spatial",
    alpha_img: float = 1,
    show: bool =True,
    **kwargs
):
    """
    Examples
    --------
    >>> import dotools_py as do
    >>> adata = do.io.read_visium("/Users/david/Downloads/PublicVisium10x")
    >>> spatial(adata, color="CDH5")
    """

    sanitize_anndata(adata)

    library_id, spatial_data = _check_spatial_data(adata.uns, library_id)
    img, img_key = _check_img(spatial_data, img, img_key, bw=bw)
    spot_size = _check_spot_size(spatial_data, spot_size)
    scale_factor = _check_scale_factor(spatial_data, img_key=img_key, scale_factor=scale_factor)
    crop_coord = _check_crop_coord(crop_coord, scale_factor)
    na_color = _check_na_color(na_color, img=img)

    cmap_img = "lightgrey" if bw else None
    circle_radius = size * scale_factor * spot_size * 0.5

    axs = embedding(
        adata,
        color=color,
        basis=basis,
        scale_factor=scale_factor,
        size=circle_radius,
        na_color=na_color,
        show=False,
        save=False,
        **kwargs,
    )
    axs = iterase_input(axs)

    for ax in axs:
        cur_coords = np.concatenate([ax.get_xlim(), ax.get_ylim()])
        if img is not None:
            ax.imshow(img, cmap=cmap_img, alpha=alpha_img)
        else:
            ax.set_aspect("equal")
            ax.invert_yaxis()
        if crop_coord is not None:
            ax.set_xlim(crop_coord[0], crop_coord[1])
            ax.set_ylim(crop_coord[3], crop_coord[2])
        else:
            ax.set_xlim(cur_coords[0], cur_coords[1])
            ax.set_ylim(cur_coords[3], cur_coords[2])
    if not show:
        return axs
    else:
        return plt.show()


def slides(
    adata: ad.AnnData,
    color: str | list,
    batch_key: str = "batch",
    ncols: int = 4,
    sp_size: float = 1.5,
    path: str | Path = None,
    filename: str = "Spatial.svg",
    common_expr: str | float | None = "p99.2",
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
) -> plt.Axes | None:
    """Plot visium slides.

    Plot a feature in var_names or a column from obs in one or multiple visium slides.

    :param adata: annotated data matrix.
    :param color:  var_names or obs column to plot. When multiple slides are available in the object, provide one feature.
    :param batch_key: obs column containing Batch/Sample Information. This column should have the same names system use
                    to save the spatial images in `adata.uns['spatial'].keys()`.
    :param ncols: number of subplots per row.
    :param sp_size: size of the dots.
    :param path: path to save the plot.
    :param filename: filename of the plot.
    :param common_expr: specify a float or a string in the form of 'p99.2' (percentile 99.2) to normalize expression for continuous values across multiple slides.
    :param order: provide a list with the order of the slides to show. If not set the `batch_key` will be sorted.
    :param figsize: size of the subplots.
    :param layer: layer to use to plot dt. If not specified, `.X` will be used.
    :param img_key: image key to use for plotting (hires or lowres).
    :param title_fontsize: fontsize of the title for the subplots.
    :param title_fontweight: change fontweight of the title.
    :param select_samples: list with a subset of samplename that want to be plotted.
    :param show: if False, return axs.
    :param minimal_title: if set to true only the sample name will be shown as title, otherwise title + color
    :param vmax: maximum value for continuous values (e.g., expression). If common expression is set to True and vmax
                 is not specified, it will be automatically computed taking the p99.2 expression value across
                 all subplots.
    :param verbose: show a progress bar when plotting multiple slides.
    :param kwargs: additional arguments for the function `scanpy.pl.spatial()`.
    :param spacing: spacing between subplots (height, width) padding between plots
    :return: a matplotlib axes object
    """
    sanitize_anndata(adata)

    if select_samples is not None:
        select_samples = iterase_input(select_samples)
        adata = adata[adata.obs[batch_key].isin(select_samples)].copy()
        adata.obs[batch_key] = pd.Categorical(adata.obs[batch_key].astype(str))

    # Define the number of rows base on the desired number of columns
    n_samples = len(adata.obs[batch_key].unique())
    nrows, ncols, extras = get_subplot_shape(n_samples, ncols)

    # Control the order of the samples
    show_order = order
    if order is None:
        show_order = adata.obs[batch_key].cat.categories.to_list()

    color = iterase_input(color)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    if n_samples == 1: # Case 1 - We have 1 sample
        axs = _spatial(
            adata=adata,
            color=color,
            size=sp_size,
            library_id=adata.obs[batch_key].unique().tolist()[0],
            layer=layer,
            vmax=vmax,
            show=False,
            figsize=figsize,
            **kwargs,
        )
    else:
        if len(color) != 1:
            raise InputError("When multiples slides are plotted, only one feature can be plotted")

        if vmax is None and common_expr is not None:
            if isinstance(common_expr, str):
                percentile = float(common_expr.replace("p",""))
                try:
                    expr = get_expr(adata, color)
                    vmax = np.percentile(expr["expr"], percentile)
                except ValueError:
                    vmax = np.percentile(adata.obs[color], percentile)
            else:
                vmax = common_expr

        fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
        plt.subplots_adjust(hspace=spacing[0], wspace=spacing[1], left=0.05)  # Spacing between subplots
        axs = axs.flatten()
        for idx, sample in tqdm(enumerate(show_order), desc="Slide ", disable=not verbose, total=len(show_order)):
            sdata = adata[adata.obs[batch_key] == sample]
            _spatial(
                sdata,
                ax=axs[idx],
                img_key=img_key,
                color=color,
                library_id=sample,
                size=sp_size,
                layer=layer,
                vmax=vmax,
                show=False,
                **kwargs,
            )
            # Modify axis
            title_color = "" if color is None else color
            if minimal_title:
                axs[idx].set_title(sample, fontsize=title_fontsize, fontweight=title_fontweight)
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
