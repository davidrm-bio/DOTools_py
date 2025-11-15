from pathlib import Path

import anndata as ad
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec

from dotools_py.utils import sanitize_anndata, iterase_input, convert_path
from dotools_py.pl._plot_utils import _AxesSubplot

from matplotlib.colors import Colormap
from collections.abc import Sequence
from matplotlib.axes import Axes
from matplotlib.figure import Figure



class BaseSeaborn:
    """
    Utility class to plot data from an AnnData Object using seaborn.
    """

    MIN_FIGURE_HEIGHT = 4.2
    DEFAULT_WSPACE = 0.0
    DEFAULT_LEGEND_WIDTH = 1.5
    DEFAULT_CMAP = "tab10"

    DEFAULT_TITLE_SIZE = 20
    DEFAULT_TITLE_FONTWEIGHT = "bold"

    DEFAULT_XTICKS_SIZE = 12
    DEFAULT_XTICKS_FONTWEIGHT = "bold"
    DEFAULT_XTICKS_ROTATION = None

    DEFAULT_LEGEND_TITLE_FONTSIZE = 12
    DEFAULT_LEGEND_TITLE_FONTWEIGHT = "bold"

    def __init__(
        self,
        adata: ad.AnnData,
        x_axis: str,
        feature: str,
        batch_key: str = None,
        xticks_order: list = None,
        hue: str = None,
        hue_order: list = None,
        layer: str = None,
        logcounts: bool = True,
        figsize: tuple = (3, 4.2),
        ax: plt.Axes = None,
        cmap: str | Colormap | dict = None,
        title: str = None,
        title_fontproperties: dict = None,
        xticks_properties: dict = None,
        legend_properties: dict = None,
        path: str | Path = None,
        filename: str = "figure.svg",
        show: bool = True
    ):
        """Initialize class.

        :param adata: Annotated data matrix.
        :param x_axis: Name of a categorical column in `adata.obs` to groupby.
        :param feature: A valid feature in `adata.var_names` or column in `adata.obs` with continuous values.
        :param batch_key: Name of a categorical column in `adata.obs` that contains the sample names.
        :param xticks_order: Order for the categories in `adata.obs[x_axis]`.
        :param hue: Name of a second categorical column in `adata.obs` to use additionally to groupby.
        :param hue_order: List with orders for the categories in `hue`. If it is not set, the order will be inferred.
        :param layer: Name of the AnnData object layer that wants to be plotted. The default `adata.X` is plotted. If
                     layer is set to a valid layer name, then the layer is plotted.
        :param logcounts: If set to `True`, consider that the values in `adata.X` or `adata.layers[layer]` if layer is
                         set is log1p transformed.
        :param figsize: Figure size, the format is (width, height).
        :param ax: Matplotlib axes to use for plotting. If not set, a new figure will be generated.
        :param cmap: String denoting matplotlib colormap. A dictionary with the categories available in
                    `adata.obs[x_axis]` or `adata.obs[hue]` if hue is not None can also be provided. The format is
                    {category:color}.
        :param title: Title for the figure.
        :param title_fontproperties: Dictionary which should contain 'size' and 'weight' to define the fontsize and
                                    fontweight of the title of the figure.
        :param xticks_properties: Dictionary which should contain 'size' and 'weight' to define the fontsize and
                                 fontweight of the xticks of the figure.
        :param legend_properties: Dictionary which should contain 'size' and 'weight' to define the fontsize and
                                 fontweight of the title of the legend.
        :param path: Path to the folder to save the figure.
        :param filename: Name of file to use when saving the figure.
        :param show: If set to `False`, returns a dictionary with the matplotlib axes.
        """
        from dotools_py.utility._plotting import get_hex_colormaps

        sanitize_anndata(adata)

        # Data Section
        self.adata = adata
        self.x_axis = x_axis
        self.feature = iterase_input(feature)  # We always assume we have a list
        self.batch_key = batch_key

        self.xticks_order = xticks_order if xticks_order is not None else list(adata.obs[x_axis].unique())
        self.hue = hue
        if hue is None:
            self.hue_order = None
        else:
            self.hue_order = hue_order if hue_order is not None else list(adata.obs[hue].unique())

        self.layer = layer
        self.logcounts = logcounts


        # Figure paramters
        self.figsize = figsize
        self.fig = None,
        self.gs = None
        self.width, self.height = figsize if figsize is not None else (None, None)
        self.ax = ax
        self.legends_width = self.DEFAULT_LEGEND_WIDTH
        self.cmap = self.DEFAULT_CMAP if cmap is None else cmap

        if isinstance(self.cmap, str):
            tmp = self.hue_order if hue_order is not None else self.xticks_order
            colors_dict = dict(zip(tmp, get_hex_colormaps(self.cmap), strict=False))
        elif isinstance(self.cmap, dict):
            colors_dict = self.cmap
        else:
            raise Exception('palette can only be a string or dictionary')

        self.cmap_dict = colors_dict

        # Title Properties
        self.title = title if title is not None else feature
        title_fontproperties = {} if title_fontproperties is None else title_fontproperties
        self.title_size = title_fontproperties.get("size", self.DEFAULT_TITLE_SIZE)
        self.title_fontweight = title_fontproperties.get("weight", self.DEFAULT_TITLE_FONTWEIGHT)

        # X-ticks Properties
        xticks_properties = {} if xticks_properties is None else xticks_properties
        self.xticks_fontsize = xticks_properties.get("size", self.DEFAULT_XTICKS_SIZE)
        self.xticks_fontweight = xticks_properties.get("weight", self.DEFAULT_XTICKS_FONTWEIGHT)
        rotation = xticks_properties.get("rotation", self.DEFAULT_XTICKS_ROTATION)
        self.rotation = {"rotation": rotation, "ha": "right", "va": "top"} if rotation is not None else {}

        # Legend Properties
        legend_properties = {} if legend_properties is None else legend_properties
        self.legend_title = legend_properties.get("size", self.DEFAULT_LEGEND_TITLE_FONTSIZE)
        self.legend_title_fontweight = legend_properties.get("weight", self.DEFAULT_LEGEND_TITLE_FONTWEIGHT)
        self.legend_fontsize = legend_properties.get("size", self.DEFAULT_LEGEND_TITLE_FONTSIZE - 2)

        # Saving
        self.path = path
        self.filename = filename
        self.show = show
        self.dict_axis = None

        return

    def make_figure(
        self,
        nrows: int = 1,
        ncols: int = 1
    ) -> None:
        """Generate figure.

        :param nrows: Number of rows.
        :param ncols: Number of Columns
        :return: Returns None.
        """
        mainplot_width = self.width - self.legends_width

        fig, gs = self.make_grid_spec(
            self.ax or (self.width, self.height),
            nrows=nrows, ncols=ncols, wspace=0.7 / self.width,
            width_ratios=[mainplot_width, self.legends_width] if ncols == 2 else [mainplot_width]
        )

        self.fig = fig
        self.gs = gs
        return None

    def legend(
        self,
        show: bool = False,
        width: float = 1.5,
        title: str = None,
    ) -> None:
        """Set legend parameters.

        :param show: If set to `False`, the legend is deactivated.
        :param width: width of the figure reserve for the legend.
        :param title: title of the legend.
        :return: Returns None.
        """
        if not show:
            # Deactivate legend by setting the width to 0
            self.legends_width = 0
        else:
            self.legend_title = title
            self.legends_width = width
        return None

    @staticmethod
    def make_grid_spec(
        ax_or_figsize: tuple[int, int] | _AxesSubplot,
        *,
        nrows: int,
        ncols: int,
        wspace: float | None = None,
        hspace: float | None = None,
        width_ratios: Sequence[float] | None = None,
        height_ratios: Sequence[float] | None = None,
    ) -> tuple[Figure, gridspec.GridSpecBase]:
        """Adapted from Scanpy"""

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

    def saving_return_axis(self) -> dict[str, Axes] | plt.Axes | None:
        """Return axis and save figure.

        :return: Returns a dictionary with the matplotlib axes or matplotlib axes if `show` is set to False.
        """
        if self.path is not None:
            plt.savefig(convert_path(self.path) / self.filename, bbox_inches="tight")
        if self.show:
            plt.tight_layout()
            return plt.show()
        else:
            return self.dict_axis


    def set_xticks(self, ax: plt.Axes) -> None:
        """Set properties for the xticks.

        :param ax: Matplotlib Axes.
        :return: Returns None.
        """
        ax.set_xticklabels(ax.get_xticklabels(), fontweight=self.xticks_fontweight, **self.rotation)
        return None


    def set_title(self, ax: plt.Axes) -> None:
        """Set the properties for the title.

        :param ax: Matplotlib Axes.
        :return: Returns None
        """
        ax.set_title(self.title, fontsize=self.title_size, fontweight=self.title_fontweight)
        return None

    def get_expression(self, keep: list) -> pd.DataFrame:
        """Get the expression.

        :param keep: Columns in `adata.obs` to keep.
        :return: Returns a DataFrame with the expression extracted from the AnnData object.
        """
        from dotools_py.get._generic import expr as get_expr
        keep = iterase_input(keep)
        if all(feature in list(self.adata.var_names) for feature in self.feature):
            df = get_expr(self.adata, self.feature, groups=keep, layer=self.layer)
        elif all(feature in list(self.adata.obs.columns) for feature in self.feature):
            df = self.adata.obs[keep + self.feature]
            # df["expr"] = df[self.feature[0]]
            df = df.rename(columns={self.feature[0]: "expr"})
        else:
            raise ValueError(f"{self.feature} needs to be in adata.var_names or adata.obs")
        return df


    def get_mean_expression(self) -> pd.DataFrame:
        """Get the mean expression.

        :return: Returns a DataFrame with the mean expression.
        """
        from dotools_py.get._generic import mean_expr as get_mean_expr

        hue = iterase_input(self.hue)
        group_by = [self.x_axis, self.batch_key] + hue

        if all(feature in list(self.adata.var_names) for feature in self.feature):
            df_mean = get_mean_expr(self.adata, group_by=group_by, features=self.feature, layer=self.layer)
        elif all(feature in list(self.adata.obs.columns) for feature in self.feature):
            df_mean = self.adata.obs[self.feature + group_by]
            df_mean = df_mean.groupby(group_by).agg(np.mean).fillna(0).reset_index()
            df_mean["gene"] = self.feature[0]
            df_mean["expr"] = df_mean[self.feature[0]]
        else:
            raise ValueError(f"{self.feature} is not in adata.var_names or adata.obs")
        return df_mean

    @staticmethod
    def log_estimator(values: np.ndarray):
        """Compute mean of log1p transform data.

        :param values: values to calculate the mean expression on.
        :return: Returns a numpy array with the mean expression log1p transform
        """
        return np.log1p(np.mean(np.expm1(values)))


# def _pseudobuk_mode(
#     # Data
#     adata: ad.AnnData,
#     x_axis: str,
#     feature: str,
#     batch_key: str,
#     hue: str,
#     layer: str = "counts",
#
#     # Statistics
#     ctrl_cond: str = None,
#     groups_cond: str | list = None,
#     groups_pvals: float | list = None,
#     test: Literal["edger", "deseq", "cluster_ttest"] = "edger",
#     mode: Literal["sum"] = "sum",
#
#     # Fx specific
#     design: str = "~condition",
#     pseudobulk: bool = False
#
# ):
#     from dotools_py.tl._Statistical import DGEAnalysis
#
#     if len(groups_pvals) == 0:
#         tester = DGEAnalysis(
#             data=adata,
#             group_by=x_axis,
#             batch_key=batch_key,
#             pseudobulk_mode=mode,
#             pseudobulk_groups=hue,
#             is_pseudobulk=pseudobulk,
#         )
#         tester._get_pseudobulk()
#
#         if test == "cluster_ttest":
#             tester.cluster_ttest(reference=ctrl_cond, groups=groups_cond, layer=layer)
#         elif test == "edger":
#             tester.edger(
#                 design=design, reference=ctrl_cond, groups=groups_cond, layer=layer)
#         elif test == "deseq":
#             tester.deseq(
#                 design=design, reference=ctrl_cond, groups=groups_cond, layer=layer)
#         else:
#             raise ValueError(f"{test} is not a valid test, use: ['edger', 'deseq', 'cluster_ttest']")
#
#         tester.get_dge()
#


