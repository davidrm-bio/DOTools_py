import re
from typing import TypeVar, Callable, Literal, NamedTuple
from typing_extensions import Self
from collections.abc import Sequence
from textwrap import indent
from matplotlib.axes import Axes
from matplotlib import axes
from dotools_py._custom_class import PathLike
from dotools_py._utils import convert_path
import matplotlib.gridspec as gridspec
import functools

from prelude_py import np, pd, ad, plt


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


def check_colornorm(vmin=None, vmax=None, vcenter=None, norm=None):
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


def small_squares(ax: plt.Axes, pos: list, color: list, size: float = 1, linewidth: float = 0.8,
                  zorder: int = 20) -> None:
    """Add small squares.

    :param ax: matplotlib axis
    :param pos: list of positions
    :param color: list of colors
    :param size:  size of the square
    :param linewidth: linewith of the square
    :param zorder: location of the square
    :return: None
    """
    import matplotlib.patches as patches

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



def fix_kwds(kwds_dict, **kwargs):
    """Merge the parameters into a single consolidated dictionary.

    Given a dictionary of plot parameters (`kwds_dict`) and a dict of `kwds`,
    this function prevents argument duplication errors.

    If `kwds_dict` an kwargs have the same key, only the value in `kwds_dict` is kept.

    Parameters
    ----------
    kwds_dict
        kwds dictionary
    kwargs

    Returns
    -------
    `kwds_dict` merged with `kwargs`

    Examples
    --------
    >>> def _example(**kwds):
    ...     return fix_kwds(kwds, key1="value1", key2="value2")
    >>> _example(key1="value10", key3="value3")
    {'key1': 'value10', 'key2': 'value2', 'key3': 'value3'}

    """
    kwargs.update(kwds_dict)

    return kwargs


class VarGroups(NamedTuple):
    labels: Sequence[str]
    """Var labels."""
    positions: Sequence[tuple[int, int]]
    """Var positions.

    Each item in the list should contain the start and end position that the bracket should cover.
    Eg. `[(0, 4), (5, 8)]` means that there are two brackets,
    one for the var_names (eg genes) in positions 0-4 and other for positions 5-8
    """

    @classmethod
    def validate(
        cls,
        labels: Sequence[str] | None,
        positions: Sequence[tuple[int, int]] | None
    ) -> Self | None:
        if labels is None and positions is None:
            return None
        if labels is None or positions is None:
            raise ValueError(
                "If var_group_labels or var_group_positions are given, both have to be given."
            )
        if len(labels) != len(positions):
            raise ValueError(
                f"var_group_labels and var_group_positions must have the same length."
                f"\nGot {len(labels)=} and {len(positions)=}."
            )
        return None if len(labels) == 0 else cls(labels, positions)

def _var_groups(var_names) -> tuple:
    """Adapated from scanpy, Normalize var_names."""
    from collections.abc import Mapping

    if not isinstance(var_names, Mapping):
        var_names = [var_names] if isinstance(var_names, str) else var_names
        return var_names, None
    if len(var_names) == 0:
        return [], None

    var_group_labels: list[str] = []
    var_names_seq: list[str] = []
    var_group_positions: list[tuple[int, int]] = []
    for label, vars_list in var_names.items():
        vars_list = [vars_list] if isinstance(vars_list, str) else vars_list
        start = len(var_names_seq)
        # use list() in case var_list is a numpy array or pandas series
        var_names_seq.extend(list(vars_list))
        var_group_labels.append(label)
        var_group_positions.append((start, start + len(vars_list) - 1))
    if not var_names_seq:
        msg = "No valid var_names were passed."
        raise ValueError(msg)
    return var_names_seq, VarGroups(var_group_labels, var_group_positions)

def _plot_var_groups_brackets(
    var_groups_ax,
    *,
    var_groups: VarGroups,
    left_adjustment: float = -0.3,
    right_adjustment: float = 0.3,
    rotation: float | None = None,
    orientation: Literal["top", "right"] = "top",
    wide: bool = False,
) -> None:
    """Draw brackets that represent groups of genes on the give axis.

    For best results, this axis is located on top of an image whose
    x axis contains gene names.

    The `var_groups_ax` should share the x axis with the main ax.

    E.g: `var_groups_ax=fig.add_subplot(axs[0, 0], sharex=dot_ax)`

    This function is used by dotplot, heatmap etc.

    Parameters
    ----------
    var_groups_ax
        In this axis the gene marks are drawn
    var_groups
        Group labels and positions
    left_adjustment
        adjustment to plot the bracket start slightly before or after the first gene position.
        If the value is negative the start is moved before.
    right_adjustment
        adjustment to plot the bracket end slightly before or after the last gene position
        If the value is negative the start is moved before.
    rotation
        rotation degrees for the labels. If not given, small labels (<4 characters) are not
        rotated, otherwise, they are rotated 90 degrees
    orientation
        location of the brackets. Either `top` or `right`

    Returns
    -------
    None

    """
    from matplotlib import patches
    from matplotlib.path import Path

    # get the 'brackets' coordinates as lists of start and end positions
    left = [x[0] + left_adjustment for x in var_groups.positions]
    right = [x[1] + right_adjustment for x in var_groups.positions]

    # verts and codes are used by PathPatch to make the brackets
    verts = []
    codes = []
    if orientation == "top":
        # rotate labels if any of them is longer than 4 characters
        if rotation is None:
            rotation = 90 if max([len(x) for x in var_groups.labels]) > 4 else 0
        for idx in range(len(left)):
            verts.append((left[idx], 0))  # lower-left
            verts.append((left[idx], 0.6))  # upper-left
            verts.append((right[idx], 0.6))  # upper-right
            verts.append((right[idx], 0))  # lower-right

            codes.append(Path.MOVETO)
            codes.append(Path.LINETO)
            codes.append(Path.LINETO)
            codes.append(Path.LINETO)

            group_x_center = left[idx] + float(right[idx] - left[idx]) / 2
            var_groups_ax.text(
                group_x_center,
                1.1,
                var_groups.labels[idx],
                ha="center",
                va="bottom",
                rotation=rotation,
            )
    else:
        top = left
        bottom = right
        for idx in range(len(top)):
            verts.append((0, top[idx]))  # upper-left
            verts.append((0.4 if wide else 0.15, top[idx]))  # upper-right
            verts.append((0.4 if wide else 0.15, bottom[idx]))  # lower-right
            verts.append((0, bottom[idx]))  # lower-left

            codes.append(Path.MOVETO)
            codes.append(Path.LINETO)
            codes.append(Path.LINETO)
            codes.append(Path.LINETO)

            diff = bottom[idx] - top[idx]
            group_y_center = top[idx] + float(diff) / 2
            # cut label to fit available space
            label = (
                f"{var_groups.labels[idx][: int(diff * 2)]}."
                if diff * 2 < len(var_groups.labels[idx])
                else var_groups.labels[idx]
            )
            var_groups_ax.text(
                1.1 if wide else 0.6,
                group_y_center,
                label,
                ha="right",
                va="center",
                rotation=270,
                fontsize="small",
            )

    path = Path(verts, codes)

    patch = patches.PathPatch(path, facecolor="none", lw=1.5)

    var_groups_ax.add_patch(patch)
    var_groups_ax.grid(visible=False)
    var_groups_ax.axis("off")
    # remove y ticks
    var_groups_ax.tick_params(axis="y", left=False, labelleft=False)
    # remove x ticks and labels
    var_groups_ax.tick_params(axis="x", bottom=False, labelbottom=False, labeltop=False)



def add_star_on_square(axis: plt.Axes, size: int = 20, d3: bool = False) -> None:
    """Draw an asterisk if the upper right corner of a rectangle patch.

    :param axis: Matplotlib axis.
    :param size: Size of the asterisk.
    :return: None
    """
    for patch in axis.patches:
        x0, y0 = patch.get_xy()
        if d3:  # 3D version needs to go reverse
            upper_right = (x0 + patch.get_width(), y0 + patch.get_height())
        else:
            upper_right = (x0 + patch.get_width(), y0)

        if patch.get_width() != patch.get_height():
            continue # Ignore patches that are not squares

        axis.text(
            upper_right[0], upper_right[1],
            "*",
            fontsize=size,
            color="black",
            ha="center",
            va="center",
            fontfamily='Helvetica'
        )
        patch.set_edgecolor("white")
    return None


