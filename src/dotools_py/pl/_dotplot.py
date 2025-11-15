from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad
import matplotlib.colorbar
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from scanpy._compat import old_positionals
from scanpy._settings import settings
from scanpy._utils import _doc_params, _empty
from scanpy.plotting._anndata import _prepare_dataframe
from scanpy.plotting._baseplot_class import BasePlot, doc_common_groupby_plot_args
from scanpy.plotting._docs import doc_common_plot_args, doc_show_save_ax, doc_vboundnorm
from scanpy.plotting._utils import (
    check_colornorm,
    fix_kwds,
    make_grid_spec,
    savefig_or_show,
)

from scanpy.plotting._anndata import VarGroups, _plot_var_groups_brackets
from dotools_py import logger
from dotools_py.get._generic import expr as get_expr
from dotools_py.utils import convert_path, sanitize_anndata, draw_bracket, save_plot, return_axis
from dotools_py.pl._heatmap import small_squares
from dotools_py.tl._get_stats import rank_genes_groups

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal

    import pandas as pd
    from anndata import AnnData
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap, Normalize
    from scanpy._utils import Empty
    from scanpy.plotting._utils import ColorLike, _AxesSubplot
    from typing_extensions import Self


########################################################################################################################
#
#     CORRECTION FOR THE DOTPLOT CLASS TO CONSIDER LOGCOUNTS
#
########################################################################################################################
def _var_groups(var_names) -> tuple:
    """Adapated from scanpy, Normalize var_names.

    """
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
            fontfamily='DejaVu Sans Mono'
        )
        patch.set_edgecolor("white")
    return None



def _dk(dendrogram: bool | str | None) -> str | None:
    """Convert the `dendrogram` parameter to a `dendrogram_key` parameter."""
    return None if isinstance(dendrogram, bool) else dendrogram


@_doc_params(common_plot_args=doc_common_plot_args)
class DotPlot(BasePlot):
    """Dotplot class.

    Allows the visualization of two values that are encoded as
    dot size and color. The size usually represents the fraction
    of cells (obs) that have a non-zero value for genes (var).

    For each var_name and each `groupby` category a dot is plotted.
    Each dot represents two values: mean expression within each category
    (visualized by color) and fraction of cells expressing the `var_name` in the
    category (visualized by the size of the dot). If `groupby` is not given,
    the dotplot assumes that all dt belongs to a single category.

    .. note::
       A gene is considered expressed if the expression value in the `adata` (or
       `adata.raw`) is above the specified threshold which is zero by default.

    An example of dotplot usage is to visualize, for multiple marker genes,
    the mean value and the percentage of cells expressing the gene
    across multiple clusters.

    Parameters
    ----------
    {common_plot_args}
    title
        Title for the figure
    expression_cutoff
        Expression cutoff that is used for binarizing the gene expression and
        determining the fraction of cells expressing given genes. A gene is
        expressed only if the expression value is greater than this threshold.
    mean_only_expressed
        If True, gene expression is averaged only over the cells
        expressing the given genes.
    standard_scale
        Whether or not to standardize that dimension between 0 and 1,
        meaning for each variable or group,
        subtract the minimum and divide each by its maximum.
    kwds
        Are passed to :func:`matplotlib.pyplot.scatter`.

    See Also
    --------
    :func:`~scanpy.pl.dotplot`: Simpler way to call DotPlot but with less options.
    :func:`~scanpy.pl.rank_genes_groups_dotplot`: to plot marker
        genes identified using the :func:`~scanpy.tl.rank_genes_groups` function.

    Examples
    --------
    >>> import scanpy as sc
    >>> adata = sc.datasets.pbmc68k_reduced()
    >>> markers = ["C1QA", "PSAP", "CD79A", "CD79B", "CST3", "LYZ"]
    >>> sc.pl.DotPlot(adata, markers, groupby="bulk_labels").show()

    Using var_names as dict:

    >>> markers = {{"T-cell": "CD3D", "B-cell": "CD79A", "myeloid": "CST3"}}
    >>> sc.pl.DotPlot(adata, markers, groupby="bulk_labels").show()

    """

    DEFAULT_SAVE_PREFIX = "dotplot_"

    # default style parameters
    DEFAULT_COLORMAP = "Reds"
    DEFAULT_COLOR_ON = "dot"
    DEFAULT_DOT_MAX = None
    DEFAULT_DOT_MIN = None
    DEFAULT_SMALLEST_DOT = 0.0
    DEFAULT_LARGEST_DOT = 200.0
    DEFAULT_DOT_EDGECOLOR = "black"
    DEFAULT_DOT_EDGELW = 0.2
    DEFAULT_SIZE_EXPONENT = 1.5

    # default legend parameters
    DEFAULT_SIZE_LEGEND_TITLE = "Fraction of cells\nin group (%)"
    DEFAULT_COLOR_LEGEND_TITLE = "Mean expression\nin group"
    DEFAULT_LEGENDS_WIDTH = 1.5  # inches
    DEFAULT_PLOT_X_PADDING = 0.8  # a unit is the distance between two x-axis ticks
    DEFAULT_PLOT_Y_PADDING = 1.0  # a unit is the distance between two y-axis ticks

    @old_positionals(
        "use_raw",
        "log",
        "num_categories",
        "categories_order",
        "title",
        "figsize",
        "gene_symbols",
        "var_group_positions",
        "var_group_labels",
        "var_group_rotation",
        "layer",
        "expression_cutoff",
        "mean_only_expressed",
        "standard_scale",
        "dot_color_df",
        "dot_size_df",
        "ax",
        "vmin",
        "vmax",
        "vcenter",
        "norm",
    )
    def __init__(
        self,
        adata: AnnData,
        var_names,
        groupby: str | Sequence[str],
        *,
        use_raw: bool | None = None,
        log: bool = False,
        num_categories: int = 7,
        categories_order: Sequence[str] | None = None,
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
        gene_symbols: str | None = None,
        var_group_positions: Sequence[tuple[int, int]] | None = None,
        var_group_labels: Sequence[str] | None = None,
        var_group_rotation: float | None = None,
        layer: str | None = None,
        expression_cutoff: float = 0.0,
        mean_only_expressed: bool = False,
        standard_scale: Literal["var", "group"] | None = None,
        dot_color_df: pd.DataFrame | None = None,
        dot_size_df: pd.DataFrame | None = None,
        ax: _AxesSubplot | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        vcenter: float | None = None,
        norm: Normalize | None = None,
        logcounts: bool = True,
        add_stats: bool = False,
        stats_type: Literal["square", "star"] = "star",
        **kwds,
    ) -> None:
        BasePlot.__init__(
            self,
            adata,
            var_names,
            groupby,
            use_raw=use_raw,
            log=log,
            num_categories=num_categories,
            categories_order=categories_order,
            title=title,
            figsize=figsize,
            gene_symbols=gene_symbols,
            var_group_positions=var_group_positions,
            var_group_labels=var_group_labels,
            var_group_rotation=var_group_rotation,
            layer=layer,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            vcenter=vcenter,
            norm=norm,
            **kwds,
        )

        # Prepare the plotting dataframe
        var_names = [var_names] if isinstance(var_names, str) else var_names
        adata_view = adata[:, adata.var_names.isin(var_names)]
        adata_view._sanitize()
        categories, obs_matrix = _prepare_dataframe(adata_view, var_names=var_names, groupby=groupby, layer=layer)

        # 1. compute fraction of cells having value > expression_cutoff
        obs_bool = obs_matrix > expression_cutoff

        if dot_size_df is None:
            dot_size_df = (
                obs_bool.groupby(level=0, observed=True).sum() / obs_bool.groupby(level=0, observed=True).count()
            )

        # 2. compute mean expression value
        if dot_color_df is None:
            if mean_only_expressed:
                if logcounts:  # Correction in case logcounts are provided
                    obs_matrix = np.expm1(obs_matrix)
                    dot_color_df = np.log1p(obs_matrix.mask(~obs_bool).groupby(level=0, observed=True).mean().fillna(0))
                else:
                    dot_color_df = obs_matrix.mask(~obs_bool).groupby(level=0, observed=True).mean().fillna(0)
            else:
                if logcounts:  # Correction in case logcounts are provided
                    obs_matrix = np.expm1(obs_matrix)
                    dot_color_df = np.log1p(obs_matrix.groupby(level=0, observed=True).mean())
                else:
                    dot_color_df = obs_matrix.groupby(level=0, observed=True).mean()

            # TODO Implement Z-score scaling for normal Dotplot
            # Scale the data
            if standard_scale == "group":
                dot_color_df = dot_color_df.sub(dot_color_df.min(1), axis=0)
                dot_color_df = dot_color_df.div(dot_color_df.max(1), axis=0).fillna(0)
            elif standard_scale == "var":
                dot_color_df -= dot_color_df.min(0)
                dot_color_df = (dot_color_df / dot_color_df.max(0)).fillna(0)
            elif standard_scale is None:
                pass
            else:
                logger.warn("Unknown type for standard_scale, ignored")
        else:
            assert dot_color_df.shape != dot_size_df.shape, "The dot_color_df and dot_size_df have different shape"

            # Correction in case of duplicate genes
            unique_var_names, unique_idx = np.unique(dot_color_df.columns, return_index=True)
            if len(unique_var_names) != len(self.var_names):
                dot_color_df = dot_color_df.iloc[:, unique_idx]

            dot_color_df = dot_color_df.loc[dot_size_df.index][dot_size_df.columns]

        # Save in self the dot_color_df and seld.dot_size_df
        self.dot_color_df, self.dot_size_df = (
            df.loc[
                categories_order if categories_order is not None else self.categories  # Remove self if it does not work
            ]
            for df in (dot_color_df, dot_size_df)
        )

        # Save standard_scale argument
        self.standard_scale = standard_scale

        # Set default style parameters
        self.cmap = self.DEFAULT_COLORMAP
        self.dot_max = self.DEFAULT_DOT_MAX
        self.dot_min = self.DEFAULT_DOT_MIN
        self.smallest_dot = self.DEFAULT_SMALLEST_DOT
        self.largest_dot = self.DEFAULT_LARGEST_DOT
        self.color_on = self.DEFAULT_COLOR_ON
        self.size_exponent = self.DEFAULT_SIZE_EXPONENT
        self.grid = False
        self.plot_x_padding = self.DEFAULT_PLOT_X_PADDING
        self.plot_y_padding = self.DEFAULT_PLOT_Y_PADDING

        self.dot_edge_color = self.DEFAULT_DOT_EDGECOLOR
        self.dot_edge_lw = self.DEFAULT_DOT_EDGELW

        # set legend defaults
        self.color_legend_title = self.DEFAULT_COLOR_LEGEND_TITLE
        self.size_title = self.DEFAULT_SIZE_LEGEND_TITLE
        self.legends_width = self.DEFAULT_LEGENDS_WIDTH
        self.show_size_legend = True
        self.show_colorbar = True
        self.add_stats = add_stats
        self.stats_type = stats_type
        return

    @old_positionals(
        "cmap",
        "color_on",
        "dot_max",
        "dot_min",
        "smallest_dot",
        "largest_dot",
        "dot_edge_color",
        "dot_edge_lw",
        "size_exponent",
        "grid",
        "x_padding",
        "y_padding",
    )
    def style(
        self,
        *,
        cmap: Colormap | str | None | Empty = _empty,
        color_on: Literal["dot", "square"] | Empty = _empty,
        dot_max: float | None | Empty = _empty,
        dot_min: float | None | Empty = _empty,
        smallest_dot: float | Empty = _empty,
        largest_dot: float | Empty = _empty,
        dot_edge_color: ColorLike | None | Empty = _empty,
        dot_edge_lw: float | None | Empty = _empty,
        size_exponent: float | Empty = _empty,
        grid: bool | Empty = _empty,
        x_padding: float | Empty = _empty,
        y_padding: float | Empty = _empty,
    ) -> Self:
        """\
        Modifies plot visual parameters.

        Parameters
        ----------
        cmap
            String denoting matplotlib color map.
        color_on
            By default the color map is applied to the color of the ``"dot"``.
            Optionally, the colormap can be applied to a ``"square"`` behind the dot,
            in which case the dot is transparent and only the edge is shown.
        dot_max
            If ``None``, the maximum dot size is set to the maximum fraction value found (e.g. 0.6).
            If given, the value should be a number between 0 and 1.
            All fractions larger than dot_max are clipped to this value.
        dot_min
            If ``None``, the minimum dot size is set to 0.
            If given, the value should be a number between 0 and 1.
            All fractions smaller than dot_min are clipped to this value.
        smallest_dot
            All expression fractions with `dot_min` are plotted with this size.
        largest_dot
            All expression fractions with `dot_max` are plotted with this size.
        dot_edge_color
            Dot edge color.
            When `color_on='dot'`, ``None`` means no edge.
            When `color_on='square'`, ``None`` means that
            the edge color is white for darker colors and black for lighter background square colors.
        dot_edge_lw
            Dot edge line width.
            When `color_on='dot'`, ``None`` means no edge.
            When `color_on='square'`, ``None`` means a line width of 1.5.
        size_exponent
            Dot size is computed as:
            fraction  ** size exponent and afterwards scaled to match the
            `smallest_dot` and `largest_dot` size parameters.
            Using a different size exponent changes the relative sizes of the dots
            to each other.
        grid
            Set to true to show grid lines. By default grid lines are not shown.
            Further configuration of the grid lines can be achieved directly on the
            returned ax.
        x_padding
            Space between the plot left/right borders and the dots center. A unit
            is the distance between the x ticks. Only applied when color_on = dot
        y_padding
            Space between the plot top/bottom borders and the dots center. A unit is
            the distance between the y ticks. Only applied when color_on = dot

        Returns
        -------
        :class:`~scanpy.pl.DotPlot`

        Examples
        --------
        >>> import scanpy as sc
        >>> adata = sc.datasets.pbmc68k_reduced()
        >>> markers = ['C1QA', 'PSAP', 'CD79A', 'CD79B', 'CST3', 'LYZ']

        Change color map and apply it to the square behind the dot

        >>> sc.pl.DotPlot(adata, markers, groupby='bulk_labels') \
        ...     .style(cmap='RdBu_r', color_on='square').show()

        Add edge to dots and plot a grid

        >>> sc.pl.DotPlot(adata, markers, groupby='bulk_labels') \
        ...     .style(dot_edge_color='black', dot_edge_lw=1, grid=True) \
        ...     .show()
        """
        super().style(cmap=cmap)

        if dot_max is not _empty:
            self.dot_max = dot_max
        if dot_min is not _empty:
            self.dot_min = dot_min
        if smallest_dot is not _empty:
            self.smallest_dot = smallest_dot
        if largest_dot is not _empty:
            self.largest_dot = largest_dot
        if color_on is not _empty:
            self.color_on = color_on
        if size_exponent is not _empty:
            self.size_exponent = size_exponent
        if dot_edge_color is not _empty:
            self.dot_edge_color = dot_edge_color
        if dot_edge_lw is not _empty:
            self.dot_edge_lw = dot_edge_lw
        if grid is not _empty:
            self.grid = grid
        if x_padding is not _empty:
            self.plot_x_padding = x_padding
        if y_padding is not _empty:
            self.plot_y_padding = y_padding

        return self

    def legend(
        self,
        *,
        show: bool | None = True,
        show_size_legend: bool | None = True,
        show_colorbar: bool | None = True,
        size_title: str | None = DEFAULT_SIZE_LEGEND_TITLE,
        colorbar_title: str | None = DEFAULT_COLOR_LEGEND_TITLE,
        width: float | None = DEFAULT_LEGENDS_WIDTH,
    ) -> Self:
        """\
        Configures dot size and the colorbar legends.

        Parameters
        ----------
        show
            Set to `False` to hide the default plot of the legends. This sets the
            legend width to zero, which will result in a wider main plot.
        show_size_legend
            Set to `False` to hide the dot size legend
        show_colorbar
            Set to `False` to hide the colorbar legend
        size_title
            Title for the dot size legend. Use '\\n' to add line breaks. Appears on top
            of dot sizes
        colorbar_title
            Title for the color bar. Use '\\n' to add line breaks. Appears on top of the
            color bar
        width
            Width of the legends area. The unit is the same as in matplotlib (inches).

        Returns
        -------
        :class:`~scanpy.pl.DotPlot`

        Examples
        --------
        Set color bar title:

        >>> import scanpy as sc
        >>> adata = sc.datasets.pbmc68k_reduced()
        >>> markers = {"T-cell": "CD3D", "B-cell": "CD79A", "myeloid": "CST3"}
        >>> dp = sc.pl.DotPlot(adata, markers, groupby="bulk_labels")
        >>> dp.legend(colorbar_title="log(UMI counts + 1)").show()
        """
        if not show:
            # turn of legends by setting width to 0
            self.legends_width = 0
        else:
            self.color_legend_title = colorbar_title
            self.size_title = size_title
            self.legends_width = width
            self.show_size_legend = show_size_legend
            self.show_colorbar = show_colorbar
        return self

    def _plot_size_legend(self, size_legend_ax: Axes):
        # for the dot size legend, use step between dot_max and dot_min
        # based on how different they are.
        diff = self.dot_max - self.dot_min
        if 0.3 < diff <= 0.6:
            step = 0.1
        elif diff <= 0.3:
            step = 0.05
        else:
            step = 0.2
        # a descending range that is afterwards inverted is used
        # to guarantee that dot_max is in the legend.
        size_range = np.arange(self.dot_max, self.dot_min, step * -1)[::-1]
        if self.dot_min != 0 or self.dot_max != 1:
            dot_range = self.dot_max - self.dot_min
            size_values = (size_range - self.dot_min) / dot_range
        else:
            size_values = size_range

        size = size_values**self.size_exponent
        size = size * (self.largest_dot - self.smallest_dot) + self.smallest_dot

        # plot size bar
        size_legend_ax.scatter(
            np.arange(len(size)) + 0.5,
            np.repeat(0, len(size)),
            s=size,
            color="gray",
            edgecolor="black",
            linewidth=self.dot_edge_lw,
            zorder=100,
        )

        size_legend_ax.set_xticks(np.arange(len(size)) + 0.5)
        labels = [f"{np.round((x * 100), decimals=0).astype(int)}" for x in size_range]
        size_legend_ax.set_xticklabels(labels, fontsize="small")

        # remove y ticks and labels
        size_legend_ax.tick_params(axis="y", left=False, labelleft=False, labelright=False)

        # remove surrounding lines
        size_legend_ax.spines["right"].set_visible(False)
        size_legend_ax.spines["top"].set_visible(False)
        size_legend_ax.spines["left"].set_visible(False)
        size_legend_ax.spines["bottom"].set_visible(False)
        size_legend_ax.grid(visible=False)

        ymax = size_legend_ax.get_ylim()[1]
        size_legend_ax.set_ylim(-1.05 - self.largest_dot * 0.003, 4)
        size_legend_ax.set_title(self.size_title, y=ymax + 0.45, size="small")

        xmin, xmax = size_legend_ax.get_xlim()
        size_legend_ax.set_xlim(xmin - 0.15, xmax + 0.5)

    def _plot_stat_legend(self, sig_legend_ax: Axes):
        x, y = 0, 0.5
        if self.stats_type == "square":
            sig_legend_ax.scatter(x, y, s=400, facecolors="none", edgecolors="black", marker="s")
        elif self.stats_type == "star":
            sig_legend_ax.scatter(x, y, s=400, facecolors="none", edgecolors="white", marker="s")
            sig_legend_ax.text(x, y, "*", fontsize=25, ha="center", va="center", color="black", fontfamily='DejaVu Sans Mono')

        sig_legend_ax.text(x + 0.03, y, "FDR < 0.05", fontsize=12, va="center", fontweight="bold")
        sig_legend_ax.set_xlim(x - 0.02, x + 0.1)

        if isinstance(self.groupby, list):
            if len(self.groupby) > 1:
                name = '_'.join(self.groupby)
            else:
                name = self.groupby[0]
        else:
            name = self.groupby
        sig_legend_ax.set_title(f"Significance over\n{name}", fontsize="small", fontweight="bold", pad=1, ha='center')
        plt.gca().set_aspect("equal")
        sig_legend_ax.axis("off")  # Hide axes for clean display

    def _plot_legend(self, legend_ax, return_ax_dict, normalize):
        # to maintain the fixed height size of the legends, a
        # spacer of variable height is added at the bottom.
        # The structure for the legends is:
        # first row: variable space to keep the other rows of
        #            the same size (avoid stretching)
        # second row: legend for dot size
        # third row: spacer to avoid color and size legend titles to overlap
        # fourth row: colorbar

        cbar_legend_height = self.min_figure_height * 0.08
        size_legend_height = self.min_figure_height * 0.27
        sig_legend = self.min_figure_height * 0.15
        spacer_height = self.min_figure_height * 0.3

        if self.add_stats:
            height_ratios = [
                self.height - size_legend_height - cbar_legend_height - spacer_height - sig_legend - spacer_height,
                sig_legend,
                spacer_height,
                size_legend_height,
                spacer_height,
                cbar_legend_height,
            ]

        else:
            height_ratios = [
                self.height - size_legend_height - cbar_legend_height - spacer_height,
                size_legend_height,
                spacer_height,
                cbar_legend_height,
            ]

        nrows = len(height_ratios)
        fig, legend_gs = make_grid_spec(legend_ax, nrows=nrows, ncols=1, height_ratios=height_ratios)

        if self.add_stats:
            sig_legend_ax = fig.add_subplot(legend_gs[2])
            self._plot_stat_legend(sig_legend_ax)
            return_ax_dict["significance_legend_ax"] = sig_legend_ax

            if self.show_size_legend:
                size_legend_ax = fig.add_subplot(legend_gs[3])
                self._plot_size_legend(size_legend_ax)
                return_ax_dict["size_legend_ax"] = size_legend_ax

            if self.show_colorbar:
                color_legend_ax = fig.add_subplot(legend_gs[5])

                self._plot_colorbar(color_legend_ax, normalize)
                return_ax_dict["color_legend_ax"] = color_legend_ax
        else:
            if self.show_size_legend:
                size_legend_ax = fig.add_subplot(legend_gs[1])
                self._plot_size_legend(size_legend_ax)
                return_ax_dict["size_legend_ax"] = size_legend_ax

            if self.show_colorbar:
                color_legend_ax = fig.add_subplot(legend_gs[3])

                self._plot_colorbar(color_legend_ax, normalize)
                return_ax_dict["color_legend_ax"] = color_legend_ax

    def _mainplot(self, ax: Axes):
        # work on a copy of the dataframes. This is to avoid changes
        # on the original dt frames after repetitive calls to the
        # DotPlot object, for example once with swap_axes and other without

        _color_df = self.dot_color_df.copy()
        _size_df = self.dot_size_df.copy()
        if self.var_names_idx_order is not None:
            _color_df = _color_df.iloc[:, self.var_names_idx_order]
            _size_df = _size_df.iloc[:, self.var_names_idx_order]

        if self.categories_order is not None:
            _color_df = _color_df.loc[self.categories_order, :]
            _size_df = _size_df.loc[self.categories_order, :]

        if self.are_axes_swapped:
            _size_df = _size_df.T
            _color_df = _color_df.T
        self.cmap = self.kwds.pop("cmap", self.cmap)

        normalize, dot_min, dot_max = self._dotplot(
            _size_df,
            _color_df,
            ax,
            cmap=self.cmap,
            color_on=self.color_on,
            dot_max=self.dot_max,
            dot_min=self.dot_min,
            standard_scale=self.standard_scale,
            edge_color=self.dot_edge_color,
            edge_lw=self.dot_edge_lw,
            smallest_dot=self.smallest_dot,
            largest_dot=self.largest_dot,
            size_exponent=self.size_exponent,
            grid=self.grid,
            x_padding=self.plot_x_padding,
            y_padding=self.plot_y_padding,
            vmin=self.vboundnorm.vmin,
            vmax=self.vboundnorm.vmax,
            vcenter=self.vboundnorm.vcenter,
            norm=self.vboundnorm.norm,
            **self.kwds,
        )

        self.dot_min, self.dot_max = dot_min, dot_max
        return normalize

    @staticmethod
    def _dotplot(
        dot_size: pd.DataFrame,
        dot_color: pd.DataFrame,
        dot_ax: Axes,
        *,
        cmap: Colormap | str | None,
        color_on: Literal["dot", "square"],
        dot_max: float | None,
        dot_min: float | None,
        standard_scale: Literal["var", "group"] | None,
        smallest_dot: float,
        largest_dot: float,
        size_exponent: float,
        edge_color: ColorLike | None,
        edge_lw: float | None,
        grid: bool,
        x_padding: float,
        y_padding: float,
        vmin: float | None,
        vmax: float | None,
        vcenter: float | None,
        norm: Normalize | None,
        **kwds,
    ):
        """\
        Makes a *dot plot* given two dt frames, one containing
        the doc size and other containing the dot color. The indices and
        columns of the dt frame are used to label the output image

        The dots are plotted using :func:`matplotlib.pyplot.scatter`. Thus, additional
        arguments can be passed.

        Parameters
        ----------
        dot_size
            Data frame containing the dot_size.
        dot_color
            Data frame containing the dot_color, should have the same,
            shape, columns and indices as dot_size.
        dot_ax
            matplotlib axis
        cmap
        color_on
        dot_max
        dot_min
        standard_scale
        smallest_dot
        edge_color
        edge_lw
        grid
        x_padding
        y_padding
            See `style`
        kwds
            Are passed to :func:`matplotlib.pyplot.scatter`.

        Returns
        -------
        matplotlib.colors.Normalize, dot_min, dot_max

        """
        assert dot_size.shape == dot_color.shape, (
            "please check that dot_size and dot_color dataframes have the same shape"
        )

        assert list(dot_size.index) == list(dot_color.index), (
            "please check that dot_size and dot_color dataframes have the same index"
        )

        assert list(dot_size.columns) == list(dot_color.columns), (
            "please check that the dot_size and dot_color dataframes have the same columns"
        )

        if standard_scale == "group":
            dot_color = dot_color.sub(dot_color.min(1), axis=0)
            dot_color = dot_color.div(dot_color.max(1), axis=0).fillna(0)
        elif standard_scale == "var":
            dot_color -= dot_color.min(0)
            dot_color = (dot_color / dot_color.max(0)).fillna(0)
        elif standard_scale is None:
            pass

        # make scatter plot in which
        # x = var_names
        # y = groupby category
        # size = fraction
        # color = mean expression

        # +0.5 in y and x to set the dot center at 0.5 multiples
        # this facilitates dendrogram and totals alignment for
        # matrixplot, dotplot and stackec_violin using the same coordinates.
        y, x = np.indices(dot_color.shape)
        y = y.flatten() + 0.5
        x = x.flatten() + 0.5
        frac = dot_size.values.flatten()
        mean_flat = dot_color.values.flatten()
        cmap = plt.get_cmap(cmap)
        if dot_max is None:
            dot_max = np.ceil(max(frac) * 10) / 10
        else:
            if dot_max < 0 or dot_max > 1:
                raise ValueError("`dot_max` value has to be between 0 and 1")
        if dot_min is None:
            dot_min = 0
        else:
            if dot_min < 0 or dot_min > 1:
                raise ValueError("`dot_min` value has to be between 0 and 1")

        if dot_min != 0 or dot_max != 1:
            # clip frac between dot_min and  dot_max
            frac = np.clip(frac, dot_min, dot_max)
            old_range = dot_max - dot_min
            # re-scale frac between 0 and 1
            frac = (frac - dot_min) / old_range

        size = frac**size_exponent
        # rescale size to match smallest_dot and largest_dot
        size = size * (largest_dot - smallest_dot) + smallest_dot
        normalize = check_colornorm(vmin, vmax, vcenter, norm)

        if color_on == "square":
            if edge_color is None:
                from seaborn.utils import relative_luminance

                # use either black or white for the edge color
                # depending on the luminance of the background
                # square color
                edge_color = []
                for color_value in cmap(normalize(mean_flat)):
                    lum = relative_luminance(color_value)
                    edge_color.append(".15" if lum > 0.408 else "w")

            edge_lw = 1.5 if edge_lw is None else edge_lw

            # first make a heatmap similar to `sc.pl.matrixplot`
            # (squares with the asigned colormap). Circles will be plotted
            # on top
            dot_ax.pcolor(dot_color.values, cmap=cmap, norm=normalize)
            for axis in ["top", "bottom", "left", "right"]:
                dot_ax.spines[axis].set_linewidth(1.5)
            kwds = fix_kwds(
                kwds,
                s=size,
                linewidth=edge_lw,
                facecolor="none",
                edgecolor=edge_color,
            )
            dot_ax.scatter(x, y, **kwds)
        else:
            edge_color = "none" if edge_color is None else edge_color
            edge_lw = 0.0 if edge_lw is None else edge_lw

            color = cmap(normalize(mean_flat))
            kwds = fix_kwds(
                kwds,
                s=size,
                color=color,
                linewidth=edge_lw,
                edgecolor=edge_color,
            )
            dot_ax.scatter(x, y, **kwds)

        y_ticks = np.arange(dot_color.shape[0]) + 0.5
        dot_ax.set_yticks(y_ticks)
        dot_ax.set_yticklabels([dot_color.index[idx] for idx, _ in enumerate(y_ticks)], minor=False)

        x_ticks = np.arange(dot_color.shape[1]) + 0.5
        dot_ax.set_xticks(x_ticks)
        dot_ax.set_xticklabels(
            [dot_color.columns[idx] for idx, _ in enumerate(x_ticks)],
            rotation=90,
            ha="center",
            minor=False,
        )
        dot_ax.tick_params(axis="both", labelsize="small")
        dot_ax.grid(visible=False)

        # to be consistent with the heatmap plot, is better to
        # invert the order of the y-axis, such that the first group is on
        # top
        dot_ax.set_ylim(dot_color.shape[0], 0)
        dot_ax.set_xlim(0, dot_color.shape[1])

        if color_on == "dot":
            # add padding to the x and y lims when the color is not in the square
            # default y range goes from 0.5 to num cols + 0.5
            # and default x range goes from 0.5 to num rows + 0.5, thus
            # the padding needs to be corrected.
            x_padding = x_padding - 0.5
            y_padding = y_padding - 0.5
            dot_ax.set_ylim(dot_color.shape[0] + y_padding, -y_padding)

            dot_ax.set_xlim(-x_padding, dot_color.shape[1] + x_padding)

        if grid:
            dot_ax.grid(visible=True, color="gray", linewidth=0.1)
            dot_ax.set_axisbelow(True)

        return normalize, dot_min, dot_max


@old_positionals(
    "use_raw",
    "log",
    "num_categories",
    "expression_cutoff",
    "mean_only_expressed",
    "cmap",
    "dot_max",
    "dot_min",
    "standard_scale",
    "smallest_dot",
    "title",
    "colorbar_title",
    "size_title",
    # No need to have backwards compat for > 16 positional parameters
)
@_doc_params(
    show_save_ax=doc_show_save_ax,
    common_plot_args=doc_common_plot_args,
    groupby_plots_args=doc_common_groupby_plot_args,
    vminmax=doc_vboundnorm,
)
def dotplot_scanpy(
    adata: AnnData,
    var_names,
    groupby: str | Sequence[str],
    *,
    use_raw: bool | None = None,
    log: bool = False,
    num_categories: int = 7,
    categories_order: Sequence[str] | None = None,
    expression_cutoff: float = 0.0,
    mean_only_expressed: bool = False,
    standard_scale: Literal["var", "group"] | None = None,
    title: str | None = None,
    colorbar_title: str | None = "LogMean expression\nin group",
    size_title: str | None = "Fraction of cells\nin group (%)",
    figsize: tuple[float, float] | None = None,
    dendrogram: bool | str = False,
    gene_symbols: str | None = None,
    var_group_positions: Sequence[tuple[int, int]] | None = None,
    var_group_labels: Sequence[str] | None = None,
    var_group_rotation: float | None = None,
    layer: str | None = None,
    swap_axes: bool | None = False,
    dot_color_df: pd.DataFrame | None = None,
    show: bool | None = None,
    save: str | bool | None = None,
    ax: _AxesSubplot | None = None,
    return_fig: bool | None = False,
    vmin: float | None = None,
    vmax: float | None = None,
    vcenter: float | None = None,
    norm: Normalize | None = None,
    # Style parameters
    cmap: Colormap | str | None = "Reds",
    dot_max: float | None = None,
    dot_min: float | None = None,
    smallest_dot: float = 0.0,
    logcounts: bool = True,
    add_stats: bool = False,
    stats_type: Literal["square", "star"] = "star",
    **kwds,
) -> DotPlot | dict | None:
    """\
    Makes a *dot plot* of the expression values of `var_names`.

    For each var_name and each `groupby` category a dot is plotted.
    Each dot represents two values: mean expression within each category
    (visualized by color) and fraction of cells expressing the `var_name` in the
    category (visualized by the size of the dot). If `groupby` is not given,
    the dotplot assumes that all dt belongs to a single category.

    note::
       A gene is considered expressed if the expression value in the `adata` (or
       `adata.raw`) is above the specified threshold which is zero by default.

    An example of dotplot usage is to visualize, for multiple marker genes,
    the mean value and the percentage of cells expressing the gene
    across  multiple clusters.

    This function provides a convenient interface to the :class:`~scanpy.pl.DotPlot`
    class. If you need more flexibility, you should use :class:`~scanpy.pl.DotPlot`
    directly.

    Parameters
    ----------
    {common_plot_args}
    {groupby_plots_args}
    size_title
        Title for the size legend. New line character (\\n) can be used.
    expression_cutoff
        Expression cutoff that is used for binarizing the gene expression and
        determining the fraction of cells expressing given genes. A gene is
        expressed only if the expression value is greater than this threshold.
    mean_only_expressed
        If True, gene expression is averaged only over the cells
        expressing the given genes.
    dot_max
        If ``None``, the maximum dot size is set to the maximum fraction value found
        (e.g. 0.6). If given, the value should be a number between 0 and 1.
        All fractions larger than dot_max are clipped to this value.
    dot_min
        If ``None``, the minimum dot size is set to 0. If given,
        the value should be a number between 0 and 1.
        All fractions smaller than dot_min are clipped to this value.
    smallest_dot
        All expression levels with `dot_min` are plotted with this size.
    {show_save_ax}
    {vminmax}
    kwds
        Are passed to :func:`matplotlib.pyplot.scatter`.

    Returns
    -------
    If `return_fig` is `True`, returns a :class:`~scanpy.pl.DotPlot` object,
    else if `show` is false, return axes dict

    See Also
    --------
    :class:`~scanpy.pl.DotPlot`: The DotPlot class can be used to control
        several visual parameters not available in this function.
    :func:`~scanpy.pl.rank_genes_groups_dotplot`: to plot marker genes
        identified using the :func:`~scanpy.tl.rank_genes_groups` function.

    Examples
    --------
    Create a dot plot using the given markers and the PBMC example dataset grouped by
    the category 'bulk_labels'.

    plot::
        :context: close-figs

        import scanpy as sc
        adata = sc.datasets.pbmc68k_reduced()
        markers = ['C1QA', 'PSAP', 'CD79A', 'CD79B', 'CST3', 'LYZ']
        sc.pl.dotplot(adata, markers, groupby='bulk_labels', dendrogram=True)

    Using var_names as dict:

    plot::
        :context: close-figs

        markers = {{'T-cell': 'CD3D', 'B-cell': 'CD79A', 'myeloid': 'CST3'}}
        sc.pl.dotplot(adata, markers, groupby='bulk_labels', dendrogram=True)

    Get DotPlot object for fine-tuning

    plot::
        :context: close-figs

        dp = sc.pl.dotplot(adata, markers, 'bulk_labels', return_fig=True)
        dp.add_totals().style(dot_edge_color='black', dot_edge_lw=0.5).show()

    The axes used can be obtained using the get_axes() method

    code-block:: python

        axes_dict = dp.get_axes()
        print(axes_dict)

    """
    # backwards compatibility: previous version of dotplot used `color_map`
    # instead of `cmap`
    cmap = kwds.pop("color_map", cmap)

    dp = DotPlot(
        adata,
        var_names,
        groupby,
        use_raw=use_raw,
        log=log,
        num_categories=num_categories,
        categories_order=categories_order,
        expression_cutoff=expression_cutoff,
        mean_only_expressed=mean_only_expressed,
        standard_scale=standard_scale,
        title=title,
        figsize=figsize,
        gene_symbols=gene_symbols,
        var_group_positions=var_group_positions,
        var_group_labels=var_group_labels,
        var_group_rotation=var_group_rotation,
        layer=layer,
        dot_color_df=dot_color_df,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        vcenter=vcenter,
        norm=norm,
        logcounts=logcounts,
        add_stats=add_stats,
        stats_type=stats_type,
        **kwds,
    )

    if dendrogram:
        dp.add_dendrogram(dendrogram_key=_dk(dendrogram))
    if swap_axes:
        dp.swap_axes()

    dp = dp.style(
        cmap=cmap,
        dot_max=dot_max,
        dot_min=dot_min,
        smallest_dot=smallest_dot,
        dot_edge_lw=kwds.pop("linewidth", _empty),
    ).legend(colorbar_title=colorbar_title, size_title=size_title)

    if return_fig:
        return dp
    else:
        dp.make_figure()
        savefig_or_show(DotPlot.DEFAULT_SAVE_PREFIX, show=show, save=save)
        show = settings.autoshow if show is None else show
        if not show:
            return dp.get_axes()


def dotplot(
    adata: ad.AnnData,
    x_axis: str,
    features: str | list,
    y_axis: str = None,
    layer: str | None = None,
    x_categories_order: list | None = None,
    y_categories_order: list | None = None,
    subset_adata: bool = False,
    logcounts: bool = True,
    expression_cutoff: float = 0.0,
    mean_only_expressed: bool = False,
    z_score: str | None = None,
    cmap: str = "Reds",
    vmax: float | None = None,
    vmin: float | None = None,
    vcenter: float | None = None,
    size_legend_title: str = "Fraction of cells\nin group (%)",
    color_legend_title: str = "LogMean(nUMI)\nin group",
    feature_fontsize: float = 15,
    xticks_rotation: float = 90,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (8, 4),
    path: str | Path | None = None,
    filename: str = "Dotplot.svg",
    smallest_dot: float = 0.0,
    largest_dot: float = 200.0,
    show: bool = True,
    swap_axes: bool = True,
    rect_height: float | None = None,
    size_exponent: float = 1.5,
    edge_lw: float = 0.2,
    edge_color: str = "black",
    dot_max: float | None = None,
    dot_min: float | None = None,
    add_stats: Literal["x_axis", "y_axis"] = None,
    stats_type: Literal["square", "star"] = "star",
    df_pvals: pd.DataFrame = None,
    square_x_size: dict = None,
    star_x_size: int = 25,
    test: Literal["wilcoxon", "t-test"] = "wilcoxon",
    correction_method: Literal["benjamini-hochberg", "bonferroni"] = "benjamini-hochberg",
    pval_cutoff: float = 0.05,
    log2fc_cutoff: float = 0.0,
    set_equal_aspect: bool = False,
    var_group_rotation: int = 0,
    var_group_offset: float = 0.75,
    **kwargs,
) -> dict | None:
    """Makes a 2d dotplot or 3d dotplot.

    There are two type of visualisation:
        * 2d dotplot: X_axis shows ``x_axis`` categories and Y_axis the ``features``. The color represents the logarithmize mean
                    nUMI and the size of the dots the fractions of cells expressing the feature.
        * 3d dotplot: X_axis shows ``x_axis`` categories and Y_axis shows ``y_axis`` categories. For each feature the
                      ``x_axis`` categories will be duplicated, to show how is the expressing across 2 categorical columns
                      in `.obs`. The color represents the logarithmize mean nUMI and the size the fraction of cells expressing the
                      feature.

    .. note::
        The 2d dotplot implementation allows for standard scaling while the 3d implementation allows for Z-score
        scaling.

    :param adata: annotated data matrix.
    :param x_axis: `.obs` column to group-by.
    :param features: `.var_names` to show mean values.
    :param y_axis: `.obs` column to group-by in the other axis.
    :param layer: layer of the AnnData to use.
    :param x_categories_order: order of the categories in x_axis.
    :param y_categories_order: order of the categories in y_axis.
    :param subset_adata: subset anndata if less x_categories_order and y_categories_order are provided.
    :param logcounts: the expression values provided are log-counts.
    :param expression_cutoff: expression cutoff used for binarizing the gene expression and determining the fraction of
                              cells expressing a given feature. A gene is expressed only if the expression value is greater
                              than this threshold.
    :param mean_only_expressed: if True, gene expression is averaged only over the cells expressing the given gene.
    :param z_score: apply z_score transformation. Possible values: x_axis, y_axis or None. Only can be used when y_axis is provided.
                    Use standard_scale (group or var) if only x_axis is specified.
    :param cmap: colormap.
    :param vmax: the value representing the upper limit of the color scale.
    :param vmin: the value representing the lower limit of the color scale.
    :param vcenter: the value representing the center of the color scale.
    :param size_legend_title: title for the size legend.
    :param color_legend_title: title for the colorbar.
    :param feature_fontsize: size of the feature names when y-ticks is specified.
    :param xticks_rotation: rotation of the x-ticks.
    :param ax: matplotlib axis.
    :param figsize: figure size.
    :param path: path to save plot.
    :param filename: filename of the plot.
    :param smallest_dot: all expression levels with `dot_min` are plotted with this size.
    :param largest_dot: all expression levels with `dot_max` are plotted with this size.
    :param show: return axis
    :param size_exponent: control the increase of dots.
    :param edge_lw: thickness of the dot edges.
    :param edge_color: dot edge color.
    :param dot_min: If `None`, the minimum dot size is set to 0. If given, the value should be a number between 0 and 1.
                    All fractions smaller than dot_min are clipped to this value.
    :param dot_max: If `None`, the maximum dot size is set to the maximum fraction value found. If given, the value
                    should be a number between 0 and 1. All Fractions larger than dot_max are clipped to this value.
    :param swap_axes: swap axis. Default is True to match the 3d dotplot arguments
    :param rect_height: height of the boxes of the features in 3d dotplot
    :param add_stats: add a square to indicate statistical significance. Indicate the x_axis to test for.
    :param stats_type: how to indicate significance. Square will add a square around the dot, while star will add an asterisk
                       in the top right side of the dotplot.
    :param df_pvals: dataframe with significant values. Not yet implemented
    :param square_x_size: dictionary specifying the size and thickness of the squares.
    :param star_x_size: size of the star.
    :param test: statistical method to use.
    :param correction_method: correction method for multiple testing.
    :param pval_cutoff: cutoff for the p-value.
    :param log2fc_cutoff: cutoff for the log2-foldchange
    :param set_equal_aspect: set equal ratio both axis.
    :param var_group_rotation: if var_names is a dictionary, set the rotation of the labels for each group.
    :param var_group_offset: offset for the labels text in the brackets
    :param kwargs: additional arguments passed to the `Dotplot class <https://scanpy.readthedocs.io/en/stable/api/generated/classes/scanpy.pl.DotPlot.html#scanpy.pl.DotPlot>`_
    :return: Depending on ``show``, returns the plot if set to `True` or a dictionary with the axes

    Example
    -------
    Create a 2d dotplot using a list of markers and a PBMC example dataset grouped by the
    cell type annotation

    .. plot::
        :context: close-figs

        import dotools_py as do
        adata = do.dt.example_10x_processed()
        markers = ['CD79A', 'CD4', 'CDK1']
        do.pl.dotplot(adata, 'annotation', markers, figsize=(4, 3))

        # Add Statistical significance
        do.pl.dotplot(adata, 'condition', markers, figsize=(6, 4), add_stats='x_axis', set_equal_aspect=True)


    Create a 3d dotplot grouping also by condition

    .. plot::
        :context: close-figs

        do.pl.dotplot(adata, 'condition', markers, 'annotation', figsize=(6, 4))

        # Add Statistical significance for groups with pvals < 0.05 and log2fc > 0.0
        # Note, the object is quite small, some groups cannot be tested for having one condition only.
        do.pl.dotplot(adata, 'condition', markers, 'annotation', figsize=(6, 4), add_stats='x_axis', set_equal_aspect=True)


    """

    # <editor-fold desc="Section 0  - Helper functions">
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #  Helper functions
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def make_grid_spec(
        ax_or_figsize,
        *,
        nrows: int,
        ncols: int,
        wspace: float = None,
        hspace: float = None,
        width_ratios: float = None,
        height_ratios: float = None,
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

    def add_size_legend(ax_size_legend: plt.Axes, dot_size_df: pd.DataFrame):
        # Calculate the size of the dots
        dot_max, dot_min = round(dot_size_df.max().max() * 20) / 20, 0
        dot_diff = dot_max - dot_min

        if 0.3 < dot_diff <= 0.6:
            dot_step = 0.1
        elif dot_diff <= 0.3:
            dot_step = 0.05
        else:
            dot_step = 0.2

        size_range = np.arange(dot_max, dot_min, dot_step * -1)[::-1]
        if dot_min != 0 or dot_max != 1:
            dot_range = dot_max - dot_min
            size_values = (size_range - dot_min) / dot_range
        else:
            size_values = size_range

        size = size_values**size_exponent
        size = size * (largest_dot - smallest_dot) + smallest_dot

        # plot size bar
        ax_size_legend.scatter(
            np.arange(len(size)) + 0.5,
            np.repeat(0, len(size)),
            s=size,
            color="gray",
            edgecolor="black",
            linewidth=0.2,
            zorder=100,
        )
        ax_size_legend.set_xticks(np.arange(len(size)) + 0.5)
        labels = [f"{np.round((x * 100), decimals=0).astype(int)}" for x in size_range]
        ax_size_legend.set_xticklabels(labels, fontsize="small")

        # remove y ticks and labels
        ax_size_legend.tick_params(axis="y", left=False, labelleft=False, labelright=False)

        # remove surrounding lines
        ax_size_legend.spines[["right", "left", "top", "bottom"]].set_visible(False)
        ax_size_legend.grid(visible=False)

        ymax = ax_size_legend.get_ylim()[1]
        ax_size_legend.set_ylim(-1.05 - largest_dot * 0.003, 4)
        ax_size_legend.set_title(size_legend_title, y=ymax + 0.45, size="small")

        xmin, xmax = ax_size_legend.get_xlim()
        ax_size_legend.set_xlim(xmin - 0.15, xmax + 0.5)


    def plot_stat_legend(group_stat, sig_legend_ax: Axes, stats_type: str):
        if isinstance(group_stat, list):
            if len(group_stat) > 1:
                name = '_'.join(group_stat)
            else:
                name = group_stat[0]
        else:
            name = group_stat
        x = 0
        y = 0.5
        if stats_type == "square":
            sig_legend_ax.scatter(x, y, s=400, facecolors="none", edgecolors="black", marker="s")
        elif stats_type == "star":
            sig_legend_ax.scatter(x, y, s=400, facecolors="none", edgecolors="white", marker="s")
            sig_legend_ax.text(x, y, "*", fontsize=25, ha="center", va="center", color="black", fontfamily='DejaVu Sans Mono')
        sig_legend_ax.text(x + 0.03, y, "FDR < 0.05", fontsize=12, va="center", fontweight="bold")
        sig_legend_ax.set_xlim(x - 0.02, x + 0.1)
        sig_legend_ax.set_title(f"Significance over\n{group_stat}", fontsize="small", fontweight="bold", pad=1, ha="center")
        plt.gca().set_aspect("equal")
        sig_legend_ax.axis("off")  # Hide axes for clean display


    def add_legend(legend_ax, return_ax_dict, normalize, df_size):
        width, height = figsize if figsize is not None else (None, None)

        if height is None:
            height = len(adata.obs[y_axis].cat.categories) * 0.37
            # width = len(features) * len(x_axis) * 0.37 + 0.8

        min_figure_height = max([0.35, height])
        cbar_legend_height = min_figure_height * 0.08
        size_legend_height = min_figure_height * 0.27
        spacer_height = min_figure_height * 0.3
        sig_legend = min_figure_height * 0.15

        if add_stats is not None:
            height_ratios = [
                height - size_legend_height - cbar_legend_height - spacer_height - sig_legend - spacer_height,
                sig_legend,
                spacer_height,
                size_legend_height,
                spacer_height,
                cbar_legend_height,
            ]

        else:
            height_ratios = [
                height - size_legend_height - cbar_legend_height - spacer_height,
                size_legend_height,
                spacer_height,
                cbar_legend_height,
            ]

        nrows = len(height_ratios)
        # Create the legend axis
        fig, legend_gs = make_grid_spec(legend_ax, nrows=nrows, ncols=1, height_ratios=height_ratios)

        if add_stats is not None:
            sig_legend_ax = fig.add_subplot(legend_gs[2])
            group_size = x_axis if add_stats == "x_axis" else y_axis
            plot_stat_legend(group_size, sig_legend_ax, stats_type)
            return_ax_dict["significance_legend_ax"] = sig_legend_ax

            # Add Size Legend
            size_legend_ax = fig.add_subplot(legend_gs[3])
            add_size_legend(size_legend_ax, df_size)
            return_ax_dict["size_legend_ax"] = size_legend_ax

            # Add Color Legend
            color_legend_ax = fig.add_subplot(legend_gs[5])
            colormap = plt.get_cmap(cmap)

            mappable = ScalarMappable(norm=normalize, cmap=colormap)
            matplotlib.colorbar.Colorbar(color_legend_ax, mappable=mappable, orientation="horizontal")
            color_legend_ax.set_title(color_legend_title, fontsize="small")
            color_legend_ax.xaxis.set_tick_params(labelsize="small")
            return_ax_dict["color_legend_ax"] = color_legend_ax

        else:
            # Add Size Legend
            size_legend_ax = fig.add_subplot(legend_gs[1])
            add_size_legend(size_legend_ax, df_size)
            return_ax_dict["size_legend_ax"] = size_legend_ax

            # Add Color Legend
            color_legend_ax = fig.add_subplot(legend_gs[3])
            colormap = plt.get_cmap(cmap)

            mappable = ScalarMappable(norm=normalize, cmap=colormap)
            matplotlib.colorbar.Colorbar(color_legend_ax, mappable=mappable, orientation="horizontal")
            color_legend_ax.set_title(color_legend_title, fontsize="small")
            color_legend_ax.xaxis.set_tick_params(labelsize="small")
            return_ax_dict["color_legend_ax"] = color_legend_ax

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

    def adjust_rect_height_to_text(dot_ax, rect, text_obj, padding=0.05):
        """Efficiently adjust rect_height so the text is fully enclosed and centered with exact padding."""
        renderer = dot_ax.figure.canvas.get_renderer()

        # Get bounding boxes in display coordinates
        text_bbox = text_obj.get_window_extent(renderer)

        # Convert padding from display to dt coordinates
        y_min, y_max = dot_ax.get_ylim()
        total_range = y_max - y_min

        # Calculate padding in dt coordinates
        padding_data = padding * total_range

        # Calculate height needed in dt coordinates
        text_height_display = text_bbox.height
        text_height_data = (text_height_display / renderer.height) * total_range

        required_height = text_height_data + 2 * padding_data

        # Update rectangle height
        rect.set_height(required_height)
        # rect.set_y(dot_ax.get_ylim()[1] + required_height)  # Reposition

        # Center the text vertically
        text_obj.set_position((text_obj.get_position()[0], rect.get_y() + rect.get_height() / 4))

        return rect

    # </editor-fold>

    # <editor-fold desc="Section 1  - Check which dotplot to use">
    adata = adata.copy()
    sanitize_anndata(adata)
    x_categories_order = [x_categories_order] if isinstance(x_categories_order, str) else x_categories_order
    y_categories_order = [y_categories_order] if isinstance(y_categories_order, str) else y_categories_order

    bracket_groups = None
    if isinstance(features, dict):
        features, bracket_groups = _var_groups(features)
        swap_axes = False  # Make sure the genes are in the X-axis
        #raise Exception("Not Implemented, provide a list")

    # Features should always be a list
    features = list(features) if isinstance(features, tuple) else features
    features = [features] if isinstance(features, str) else features

    if all(g for g in features if g in list(adata.var_names)):
        missing = [g for g in features if g not in list(adata.var_names)]
        if len(missing) != 0:
            raise KeyError(f"{missing} not in adata.var_names or adata.obs")
    elif all(c for c in features if c in list(adata.obs.columns)):
        missing = [g for g in features if g not in list(adata.obs.columns)]
        if len(missing) != 0:
            raise KeyError(f"{missing} not in adata.var_names or adata.obs")

    if subset_adata:
        # If subset is true and x_categories_order or y_categories_order is provided, subset by this
        # if not all categories are provided, subset
        if x_categories_order is not None:
            adata = adata[adata.obs[x_axis].isin(x_categories_order)]
            logger.warn(f"Subsetting anndata for {x_categories_order}")
        if y_categories_order is not None:
            adata = adata[adata.obs[y_axis].isin(y_categories_order)]
            logger.warn(f"Subsetting anndata for {x_categories_order}")
    else:
        if x_categories_order is not None:
            assert len(x_categories_order) == len(adata.obs[x_axis].cat.categories), (
                f"Not all {x_axis} categories provided. Specify all or use subset_adata = True"
            )
        if y_categories_order is not None:
            assert len(y_categories_order) == len(adata.obs[y_axis].cat.categories), (
                f"Not all {y_axis} categories provided. Specify all or use subset_adata = True"
            )

    # If no y_axis is provided, then use base dotplot from scanpy
    if y_axis is None:
        try:  # Use my modification to account for logcounts
            axis_dict = dotplot_scanpy(
                adata,
                groupby=x_axis,
                var_names=features,
                categories_order=x_categories_order,
                figsize=figsize,
                expression_cutoff=expression_cutoff,
                mean_only_expressed=mean_only_expressed,
                ax=ax,
                vmin=vmin,
                size_title=size_legend_title,
                colorbar_title=color_legend_title,
                cmap=cmap,
                vmax=vmax,
                layer=layer,
                vcenter=vcenter,
                show=False,
                smallest_dot=smallest_dot,
                logcounts=logcounts,
                swap_axes=swap_axes,
                add_stats = True if add_stats == "x_axis" else False,
                stats_type = stats_type,
                **kwargs,
            )
        except Exception as e:  # Fallback to scanpy
            print(e)
            if add_stats is not None:
                logger.warn('Error when plotting, falling back to base plot. Stats not displayed')
            axis_dict = sc.pl.dotplot(
                adata,
                groupby=x_axis,
                var_names=features,
                categories_order=x_categories_order,
                figsize=figsize,
                expression_cutoff=expression_cutoff,
                mean_only_expressed=mean_only_expressed,
                ax=ax,
                vmin=vmin,
                size_title=size_legend_title,
                colorbar_title=color_legend_title,
                cmap=cmap,
                vmax=vmax,
                layer=layer,
                vcenter=vcenter,
                show=False,  # Always return axis to modify layout
                smallest_dot=smallest_dot,
                swap_axes=swap_axes,
                **kwargs,
            )

        axis_dict["mainplot_ax"].spines[["top", "right"]].set_visible(True)
        axis_dict["mainplot_ax"].set_xticklabels(
            axis_dict["mainplot_ax"].get_xticklabels(), fontweight="bold", rotation=xticks_rotation
        )

        if bracket_groups is not None and y_axis is None:
            fig = plt.gcf()
            bbox_pos = axis_dict['mainplot_ax'].get_position()
            top_ax = fig.add_axes((bbox_pos.x0, bbox_pos.y1, bbox_pos.width, 0.1))  # Adjust

            _plot_var_groups_brackets(top_ax,
                                      var_groups=bracket_groups,
                                      left_adjustment=-0.3,
                                      right_adjustment=0.3,
                                      rotation=var_group_rotation,
                                      orientation='top',
                                      )
            # Adjust text position
            for text in top_ax.texts:
                x, y = text.get_position()
                text.set_position((x, var_group_offset))  # Move all text to new position

            top_ax.set_xlim(-0.5, len(features) - 0.3)
            top_ax.set_ylim(0, 1)
            axis_dict['var_group_ax'] = top_ax


    # </editor-fold>

    else:  # Yaxis is provided, we made a 3d dotplot
        # <editor-fold desc="Section 2 - Get expression">

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        #  Extract the Expression
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Correct to make sure we have a list of features
        features = [features] if isinstance(features, str) else features

        # TODO account for cases where features are in .obs and .var_names
        try:
            df_expr = get_expr(adata, features=features, groups=[x_axis, y_axis], layer=layer, out_format="wide")
        except KeyError:  # Assume features are in .obs
            df_expr = adata.obs[features + [x_axis, y_axis]]

        df_expr.set_index([x_axis, y_axis], inplace=True)  # Set MultiIndex

        # </editor-fold>

        # <editor-fold desc="Section 3  - Get expressing cells">
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        #  Compute fraction of cells having value > expression_cutoff
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        obs_bool = df_expr > expression_cutoff
        dot_size_df = (
            obs_bool.groupby(level=[x_axis, y_axis], observed=True).sum()
            / obs_bool.groupby(level=[x_axis, y_axis], observed=True).count()
        )

        # </editor-fold>

        # <editor-fold desc="Section 4 - Compute Mean Expression">
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        #  Compute Mean Expression
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


        if mean_only_expressed:  # Compute mean only considering cells expressing the gene
            if logcounts:  # in case we have logcounts, undo the log to not do mean in log space
                df_expr = np.expm1(df_expr)
                dot_color_df = np.log1p(
                    df_expr.mask(~obs_bool).groupby(level=[x_axis, y_axis], observed=True).mean().fillna(0)
                )
            else:
                dot_color_df = df_expr.mask(~obs_bool).groupby(level=[x_axis, y_axis], observed=True).mean().fillna(0)

        else:
            if logcounts:  # in case we have logcounts, undo the log to not do mean in log space
                df_expr = np.expm1(df_expr)
                dot_color_df = np.log1p(df_expr.groupby(level=[x_axis, y_axis], observed=True).mean())
            else:
                dot_color_df = df_expr.groupby(level=[x_axis, y_axis], observed=True).mean()

        # </editor-fold>

        # <editor-fold desc="Section 5 - Zscore">
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        #  Z-score transformation
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # TODO implement standard scaling for 3D dotplot
        if z_score == "x_axis":
            if logcounts:  # If we have logcounts we need to do the log-space for the mean and std
                dot_color_df = np.expm1(dot_color_df)
                dot_color_df = np.log1p(
                    dot_color_df.groupby(level=x_axis)
                    .apply(lambda x: (x - x.mean(axis=0)) / x.std(axis=0, ddof=0))
                    .fillna(0)
                )
            else:
                dot_color_df = (
                    dot_color_df.groupby(level=x_axis)
                    .apply(lambda x: (x - x.mean(axis=0)) / x.std(axis=0, ddof=0))
                    .fillna(0)
                )
            dot_color_df = dot_color_df.reset_index(level=0, drop=True)

        elif z_score == "y_axis":
            if logcounts:  # If we have logcounts we need to do the log-space for the mean and std
                dot_color_df = np.expm1(dot_color_df)
                dot_color_df = np.log1p(
                    dot_color_df.groupby(level=y_axis)
                    .apply(lambda x: (x - x.mean(axis=0)) / x.std(axis=0, ddof=0))
                    .fillna(0)
                )
            else:
                dot_color_df = (
                    dot_color_df.groupby(level=y_axis)
                    .apply(lambda x: (x - x.mean(axis=0)) / x.std(axis=0, ddof=0))
                    .fillna(0)
                )
            dot_color_df = dot_color_df.reset_index(level=0, drop=True)

        elif z_score is None:
            pass
        else:
            logger.warn("Option not recognise, Zscore not applied")

        # Correction for the legend
        if z_score is not None and color_legend_title == "LogMean(nUMI)\nin group":
            text = x_axis if z_score == "x_axis" else y_axis
            color_legend_title = f"Zscore over\n{text}"

        # </editor-fold>

        # <editor-fold desc="Section 6  - Sorting categories">
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        #  Reorder categories
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        _color_df = dot_color_df.copy()
        _size_df = dot_size_df.copy()
        _color_df = _color_df.unstack(level=x_axis).T.fillna(0)  # unstack for the plotting
        _size_df = _size_df.unstack(level=x_axis).T.fillna(0)  # unstack for the plotting

        if y_categories_order is not None:
            y_categories_order = y_categories_order[::-1]  # Reverse to start on the top
            print(_color_df)
            _color_df = _color_df.reindex(columns=y_categories_order)  # row x col --> Yaxis x Xaxis
            _size_df = _size_df.reindex(columns=y_categories_order)
            print(_color_df)

        if x_categories_order is not None:
            idx = pd.MultiIndex.from_product([_color_df.index.get_level_values(0).unique(), x_categories_order])
            _color_df = _color_df.reindex(index=idx)
            _size_df = _size_df.reindex(index=idx)

        # </editor-fold>

        # <editor-fold desc="Section 7  - SetUp Plot">
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Set-Up for the Plot
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        x, y = np.indices(_color_df.shape)  # get x and y ticks
        y = y.flatten() + 0.5  # add offset of 0.5
        x = x.flatten() + 0.5
        frac = _size_df.values.flatten()
        mean_flat = _color_df.values.flatten()

        # Define vmin and vmax if not provided
        vmin = 0.0 if vmin is None else vmin
        if mean_only_expressed:
            vmin = round(_color_df.min().min() * 20) / 20
        if z_score is not None:
            vmin = round(_color_df.min().min() * 20) / 20
            vcenter = vcenter if vcenter is not None else 0.0  # For Zscore values can be negative, set center to 0
            cmap = cmap if cmap != "Reds" else "RdBu_r"  # Because we have neg and pos, we use a divergent colormap

        vmax = round(_color_df.max().max() * 20) / 20 if vmax is None else vmax  # Normalise to round to 5 or 0
        normalize = check_colornorm(vmin=vmin, vmax=vmax, vcenter=vcenter)
        colormap = plt.get_cmap(cmap)
        color = colormap(normalize(mean_flat))

        if dot_max is None:
            dot_max = np.ceil(max(frac) * 10) / 10
        if dot_min is None:
            dot_min = 0
        if dot_min != 0 or dot_max != 1:
            # clip frac between dot_min and  dot_max
            frac = np.clip(frac, dot_min, dot_max)
            old_range = dot_max - dot_min
            # re-scale frac between 0 and 1
            frac = (frac - dot_min) / old_range

        # Figure layout
        width, height = figsize if figsize is not None else (None, None)
        if height is None:
            height = len(adata.obs[y_axis].cat.categories) * 0.37
            width = len(features) * len(adata.obs[y_axis].cat.categories) * 0.37 + 0.8

        legends_width_spacer = 0.7 / width
        mainplot_width = width - (1.5 + 0)
        axis_dict = {}
        # </editor-fold>

        # <editor-fold desc="Section 8  - Create Plot">
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Create Plot
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        fig, gs = make_grid_spec(
            ax or (width, height), nrows=1, ncols=2, wspace=legends_width_spacer, width_ratios=[mainplot_width + 0, 1.5]
        )

        # Add Legends Axis
        size_legend_ax = fig.add_subplot(gs[1])
        add_legend(size_legend_ax, axis_dict, normalize, _size_df)

        # Add Main Axis
        dot_ax = fig.add_subplot(gs[0])

        size = frac**size_exponent
        size = size * (largest_dot - smallest_dot) + smallest_dot

        edge_color = "none" if edge_color is None else edge_color
        edge_lw = 0.0 if edge_lw is None else edge_lw

        kwas = {"s": size, "color": color, "linewidth": edge_lw, "edgecolor": edge_color}
        dot_ax.axes.scatter(x, y, **kwas)

        # Layout for Axis
        x_padding, y_padding = 0.3, 0.3
        dot_ax.set_ylim(-y_padding + y_padding / 2, len(_color_df.columns) + y_padding)
        dot_ax.set_xlim(-x_padding, (len(_color_df.index) + x_padding))

        dot_ax.set_xticks(list(np.unique(x)))
        ticks_kws = {} if xticks_rotation == 90 else {"ha": "right", "va": "top"}

        xticklabels = list(_color_df.index.get_level_values(1).unique()) * len(features)
        yticklabels = list(_color_df.columns)

        dot_ax.set_xticklabels(xticklabels, fontweight="bold", rotation=xticks_rotation, **ticks_kws)
        dot_ax.set_yticks(list(np.unique(y)))
        dot_ax.set_yticklabels(yticklabels)

        # Add Boxes for features
        labels = list(_color_df.index.get_level_values(level=0).unique())
        xticks = np.sort(list(np.unique(x)))
        rect_width = len(xticks) / len(features)
        rectangle_positions = [
            (start, start + rect_width + x_padding)
            for start in np.arange(dot_ax.get_xlim()[0], dot_ax.get_xlim()[-1], rect_width)
        ]
        rectangle_positions = rectangle_positions[: len(features)]

        rect_height = rect_height if rect_height is not None else np.min((np.max([height * 0.25, 0.6]), 0.9))
        if feature_fontsize is None:
            feature_fontsize = 10 * (rect_height / 0.5)  # Adjust the divisor as needed
            feature_fontsize = max(min(feature_fontsize, 20), 10)  # Limit the font size between 10 and 20

        for i, (x_start, x_end) in enumerate(rectangle_positions):
            if i == 0:
                x_start -= x_padding  # Special case
            if len(features) == 1:
                rect_width += x_padding  # Special case

            rect = patches.Rectangle(
                (x_start + x_padding, dot_ax.get_ylim()[1]),
                rect_width + x_padding,
                rect_height,
                linewidth=1.5,
                edgecolor="black",
                facecolor="gainsboro",
            )
            rect.set_clip_on(False)  # Prevent clipping
            dot_ax.add_patch(rect)

            # Add line to split features in groups
            if i < len(features) - 1:
                dot_ax.axvline(x=x_end, color="k")

            # Control the size of the features
            range_ticks = xticks[np.where((xticks > x_start) & (xticks < x_end))]
            text = dot_ax.text(
                np.median(range_ticks),
                dot_ax.get_ylim()[1] + 0.2,
                labels[i],
                ha="center",
                va="bottom",
                fontsize=feature_fontsize,
                fontweight="bold",
                color="black",
            )
            # Apply path effects for outlining
            text.set_path_effects(
                [
                    path_effects.Stroke(linewidth=2, foreground="white"),  # Outline
                    path_effects.Normal(),  # Inner text
                ]
            )
            rect = adjust_rect_height_to_text(dot_ax, rect, text)

        # Adjust layout for the main axis
        dot_ax.spines[["top", "right"]].set_visible(True)
        for spine in dot_ax.spines.values():
            spine.set_linewidth(1.5)

        axis_dict["mainplot_ax"] = dot_ax
        # </editor-fold>


    # <editor-fold desc="Section 9 - Add significance to the dotplot">
    # Compute stats
    if add_stats is not None:
        if add_stats == 'y_axis':
            assert y_axis is not None, "Testing y_axis but argument is None"
        group_by = x_axis if add_stats == "x_axis" else y_axis
        alternative = x_axis if add_stats == "y_axis" else y_axis

        if df_pvals is None:
            features = [features] if isinstance(features, str) else features
            if y_axis is None:
                if all(item in list(adata.var_names) for item in features):
                    try:
                        rank_genes_groups(adata, groupby=group_by, method=test, tie_correct=True,
                                          corr_method=correction_method, layer=layer)
                        table = sc.get.rank_genes_groups_df(
                            adata, group=None, pval_cutoff=pval_cutoff, log2fc_min=log2fc_cutoff
                        )
                        table_filt = table[table["names"].isin(features)]

                        if len(table_filt) == 0:
                            logger.warn('No Significant group')
                    except Exception as e:
                        logger.warn(f'Error testing, {e}')
                        table_filt = pd.DataFrame([], columns=['group', 'names', 'scores', 'logfoldchanges', 'pvals',
                                                           'pvals_adj', 'pct_nz_group', 'pct_nz_reference'])

                elif all(item in list(adata.obs.columns) for item in features):
                    raise Exception('Not Implemented')
                else:
                    raise Exception('Not a valid input')

            else:
                if all(item in list(adata.var_names) for item in features):
                    table_filt = pd.DataFrame([])
                    for alt in adata.obs[alternative].unique():
                        sdata = adata[adata.obs[alternative] == alt].copy()
                        try:
                            rank_genes_groups(sdata, groupby=group_by, method=test, tie_correct=True,
                                              corr_method=correction_method, layer=layer)
                            stable = sc.get.rank_genes_groups_df(
                                sdata, group=None, pval_cutoff=pval_cutoff, log2fc_min=log2fc_cutoff
                            )
                        except Exception as e:
                            logger.warn(f'Error while testing: {e}')
                            # If there is only one condition in the group
                            stable = pd.DataFrame([], columns=['group', 'names', 'scores', 'logfoldchanges', 'pvals',
                                                               'pvals_adj', 'pct_nz_group', 'pct_nz_reference'])
                        stable_filt = stable[stable["names"].isin(features)]
                        stable_filt['group2'] = alt
                        table_filt = pd.concat([table_filt, stable_filt])

                    if len(table_filt) == 0:
                        logger.warn('No Significant group')
                elif all(item in list(adata.obs.columns) for item in features):
                    raise Exception('Not Implemented')
                else:
                    raise Exception('Not a valid input')

            # Dataframe with gene x groups with the pvals
            if y_axis is not None:
                table_filt["group2"] = table_filt["group2"].str.replace("-", "_")  # Correction used in get_expr()

            columns = [txt.get_text() for txt in axis_dict["mainplot_ax"].get_xticklabels()]
            index = [txt.get_text() for txt in axis_dict["mainplot_ax"].get_yticklabels()]

            if y_axis is not None:
                genes = np.repeat(features, 2)
                df_pvals = pd.DataFrame([], index=index, columns=pd.MultiIndex.from_tuples(
                    [(genes[idx], col) for idx, col in enumerate(columns)]))
            else:
                df_pvals = pd.DataFrame([], index=index, columns=columns)

            for idx, row in table_filt.iterrows():
                if y_axis is None:
                    if row["group"] in list(index):
                        df_pvals.loc[row["group"], row["names"]] = row["pvals_adj"]
                    else:
                        df_pvals.loc[row["names"], row["group"]] = row["pvals_adj"]
                else:
                    if row["group"] in list(index):
                        df_pvals.loc[row["group"], (row["names"], row["group2"])] = row["pvals_adj"]
                    else:
                        df_pvals.loc[row["group2"], (row["names"], row["group"])] = row["pvals_adj"]
            df_pvals[df_pvals.isna()] = 1
        else:
           raise Exception('Not Implemented')

        # Add Stats
        square_x_size = {} if square_x_size is None else square_x_size
        square_x_size = {"width": square_x_size.get("weight", 1), "size": square_x_size.get("size", 0.5)}

        tmp = axis_dict['mainplot_ax'].get_xticklabels()[0].get_text()
        if tmp not in columns:
            df_pvals = df_pvals.T

        pos_rows, pos_cols = np.where(df_pvals < 0.05)
        pos = list(zip(pos_rows, pos_cols, strict=False))
        colors = ['black'] * len(pos)

        small_squares(
            ax=axis_dict['mainplot_ax'],
            color=colors,
            pos=pos,
            size=square_x_size["size"],
            linewidth=square_x_size["width"],
            zorder=0, # Should be on the back
        )

        if stats_type == "square":
            pass
        elif stats_type == "star":
            d3 = False if y_axis is None else True
            add_star_on_square(axis_dict["mainplot_ax"], star_x_size, d3 =d3)
        else:
            raise Exception("Not a valid stats_type, use square or star")
    # </editor-fold>

    if set_equal_aspect:
        axis_dict['mainplot_ax'].set_aspect(set_equal_aspect)

    save_plot(path=path, filename=filename)
    return  return_axis(show, axis_dict, tight=True)

    #if show:
    #    plt.tight_layout()
    #    return plt.show()
    #else:
    #    return axis_dict
