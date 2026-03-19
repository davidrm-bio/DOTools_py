import re
from typing import TypeVar, Callable
from textwrap import indent
from matplotlib.axes import Axes
from matplotlib import axes
from dotools_py._custom_class import PathLike
from dotools_py._utils import convert_path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import functools
import anndata as ad
import pandas as pd
import numpy as np


COMMON_EXPR_ARGS = """\
adata:
    Annotated data matrix.
x_axis:
    Name of a categorical column in `adata.obs` to groupby.
feature:
    A valid feature in `adata.var_names` or column in `adata.obs` with continuous values.
hue:
    Name of a second categorical column in `adata.obs` to use additionally to groupby.
hue_order:
    List with orders for the categories in `hue`. If it is not set, the order will be inferred.
layer:
    Name of the AnnData object layer that wants to be plotted. By default `adata.X` is plotted. If layer is set to a
    valid layer name, then the layer is plotted.
figsize:
    Figure size, the format is (width, height).
ax:
    Matplotlib axes to use for plotting. If not set, a new figure will be generated.
palette:
    String denoting matplotlib colormap. A dictionary with the categories available in `adata.obs[x_axis]` or
    `adata.obs[hue]` if hue is not None can also be provided. The format is {category:color}.
title:
    Title for the figure.
title_fontproperties:
    Dictionary which should contain 'size' and 'weight' to define the fontsize and fontweight of the title of the
    figure.
xticks_order:
    Order for the categories in `adata.obs[x_axis]`.
xticks_rotation:
    Rotation of the X-axis ticks.
ylabel:
    Label for the Y-axis.
legend_title:
    Title for the legend.
legend_fontproperties:
    Dictionary which should contain 'size' and 'weight' to define the fontsize and fontweight of the title of the
    legend.
legend_ncols:
    Number of columns for the legend.
legend_loc:
    Location of the legend.
path:
    Path to the folder to save the figure.
filename:
    Name of file to use when saving the figure.
show:
    If set to `False`, returns a dictionary with the matplotlib axes.
reference:
    Reference condition to use when testing for significance. When `hue` is set, the reference condition correspond
    to the categories in `hue`. For each `x_axis` category the different hue categories will be tested.
groups:
     List of the name of the groups to test against.
groups_pvals:
    If provided, these values will be plotted. If not set, the p-values will be estimated. The order of the p-values
    should match the order of the `groups_cond` categories.
test:
    Name of the method to test for significance.
corr_method:
    Correction method for multiple testing.
line_offset:
    Offset for the brackets draw to indicate significance.
txt_size:
    Font size of the text indicating significance.
txt:
    Text to include before the p-value. If not set, only the p-value is shown.\
"""

_leading_whitespace_re = re.compile("(^[ ]*)(?:[^ \n])", re.MULTILINE)
T = TypeVar("T", bound=Callable | type)

def _doc_params(**replacements: str) -> Callable[[T], T]:
    def dec(obj: T) -> T:
        assert obj.__doc__
        assert "\t" not in obj.__doc__

        # The first line of the docstring is unindented,
        # so find indent size starting after it.
        start_line_2 = obj.__doc__.find("\n") + 1
        assert start_line_2 > 0, f"{obj.__name__} has single-line docstring."
        n_spaces = min(
            len(m.group(1))
            for m in _leading_whitespace_re.finditer(obj.__doc__[start_line_2:])
        )

        # The placeholder is already indented, so only indent subsequent lines
        indented_replacements = {
            k: indent(v, " " * n_spaces)[n_spaces:] for k, v in replacements.items()
        }
        obj.__doc__ = obj.__doc__.format_map(indented_replacements)
        return obj

    return dec

class _AxesSubplot(Axes, axes.SubplotBase):
    """Intersection between Axes and SubplotBase: Has methods of both."""


def save_plot(path: PathLike | None, filename: str) -> None:
    """Save a plot.

    :param path: Path to the folder where to save the plot.
    :param filename: Name of the file.
    :return: Returns None. If path is None, no plot is saved.
    """
    if path is not None:
        plt.savefig(convert_path(path) / filename, bbox_inches="tight")
    return None


def return_axis(show: bool, axis: dict | plt.Axes, tight: bool = True) -> None | plt.Axes:
    """Whether to return axis or not.

    :param show: Boolean to indicate if the axis is returned or not.
    :param axis: Dictionary of axis or axis.
    :param tight: Tight layout.
    :return: Returns None if show is True, otherwise returns the axis.
    """
    if show:
        if tight:
            plt.tight_layout()
        return plt.show()
    else:
        return axis



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
    """Adapted from Scanpy.

    :param ax_or_figsize: axes or figsize
    :param nrows: number of rows
    :param ncols: number of columns
    :param wspace: width space
    :param hspace: height space
    :param width_ratios: width ratio
    :param height_ratios: height ratio
    :return: Figure and matplotlib Axes
    """
    kw = dict(wspace=wspace, hspace=hspace, width_ratios=width_ratios, height_ratios=height_ratios)

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



def vector_friendly():
    """ Decorator to set Scanpy figure parameters in a vector-friendly way."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import scanpy as sc
            sc.set_figure_params(scanpy=False, vector_friendly=True)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def spine_format(axis: plt.Axes, txt: str = "UMAP", fontsize: int = 10) -> None:
    """Formatting the spines for Embeddings.

    Removes the top and right spines and set the x- and y-label for the left and bottom spine
    moving them to the corner.

    :param axis: matplotlib axes object.
    :param txt: text for the embedding.
    :param fontsize: size of the text.
    :return:
    """
    axis.spines[["right", "top"]].set_visible(False)
    axis.set_xlabel(txt + "1", loc="left", fontsize=fontsize, fontweight="bold")
    axis.set_ylabel(txt + "2", loc="bottom", fontsize=fontsize, fontweight="bold")
    return


def remove_extra(extras: int, nrows: int, ncols: int, axs: plt.Axes) -> None:
    """Hide the last subplots.

    :param extras: number of subplots to remove.
    :param nrows: number of rows of the plot.
    :param ncols: number of columns of the plot.
    :param axs: matplotlib axes object.
    :return:
    """
    if extras == 0:
        return None
    else:
        for check in range(nrows * ncols - extras, nrows * ncols):
            axs[check].set_visible(False)
        return None



def get_centroids(adata: ad.AnnData, cluster_key: str, basis: str = "X_umap") -> pd.DataFrame:
    """Get centroids for clusters in anndata object.

    :param adata: AnnData.
    :param cluster_key: obs column with categorical information.
    :param basis: embedding to use.
    :return: centroids as a panda dataframe.
    """
    all_pos = pd.DataFrame(adata.obsm[basis], columns=["x", "y"])
    all_pos["group"] = adata.obs[cluster_key].values
    return all_pos.groupby("group", observed=True).median().sort_index()



def get_subplot_shape(n_samples: int, ncols: int) -> tuple:
    """Compute the number of rows and columns to use for defining the figure base on a desired number of samples and columns.

    :param n_samples: number of samples to plot.
    :param ncols: number of columns to plot.
    :return: nrows, ncols, extras (extra subplots that should be hidden).
    """
    if n_samples < ncols:  # Correction
        ncols = n_samples  # Adjust plot if more cols than samples are specified
    nrows = int(np.ceil(n_samples / ncols))
    extras = nrows * ncols - n_samples  # For hiding empty subplots
    return nrows, ncols, extras



def draw_vertical_bracket(y_start, y_end, x_left=0, x_right=1, stem_length=0.2):
    import matplotlib.path as mpath

    verts = [
        (x_left, y_start),  # Start of bracket (bottom-left)
        (x_right, y_start),  # Horizontal stem right
        (x_right, y_end - stem_length),  # Vertical part up
        (x_left, y_end - stem_length)  # Horizontal back left
    ]
    codes = [mpath.Path.MOVETO, mpath.Path.LINETO, mpath.Path.LINETO, mpath.Path.LINETO]
    return mpath.Path(verts, codes)


def draw_bracket(x_start, x_end, y_bottom=0, y_top=1, stem_length=0.2):
    import matplotlib.path

    verts = [
        (x_start, y_bottom),  # Start of the bracket (bottom-left)
        (x_start, y_top),  # Vertical stem up
        (x_end - stem_length, y_top),  # Horizontal part
        (x_end - stem_length, y_bottom)  # Down to bottom-right
    ]
    codes = [matplotlib.path.Path.MOVETO, matplotlib.path.Path.LINETO,
             matplotlib.path.Path.LINETO, matplotlib.path.Path.LINETO]
    return matplotlib.path.Path(verts, codes)
