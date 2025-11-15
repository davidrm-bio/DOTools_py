import re
from typing import TYPE_CHECKING
from typing import TypeVar, Callable
from textwrap import indent
from matplotlib.axes import Axes
from matplotlib import axes


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
    shoudl match the order of the `groups_cond` categories.
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
