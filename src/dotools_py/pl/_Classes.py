import anndata as ad
import pandas as pd
import numpy as np
from typing import Literal, Callable

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

from dotools_py._utils import sanitize_anndata, iterase_input, x_is_raw_counts
from dotools_py._custom_class import  InputError
from dotools_py.pl._plot_utils import make_grid_spec
from dotools_py.utility import get_hex_colormaps
from matplotlib.colors import Colormap
from numpy.typing import NDArray
from dotools_py.pl._StatsPlotter import TestData, StatsPlotter
from dotools_py.logger import  logger
from dotools_py.get._generic import expr as get_expr
from dotools_py.get._generic import mean_expr as get_mean_expr


class BaseSeaborn:

    MIN_FIGURE_HEIGHT = 4.2
    DEFAULT_WSPACE = 0.0
    DEFAULT_LEGEND_WIDTH = 1.5
    DEFAULT_CMAP = "tab30"

    DEFAULT_TITLE_SIZE = 20
    DEFAULT_TITLE_FONTWEIGHT = "bold"

    DEFAULT_XTICKS_SIZE = 12
    DEFAULT_XTICKS_FONTWEIGHT = "bold"
    DEFAULT_XTICKS_ROTATION = None

    DEFAULT_LEGEND_TITLE_FONTSIZE = 12
    DEFAULT_LEGEND_TITLE_FONTWEIGHT = "bold"

    def __init__(
        self,
        # Data
        adata: ad.AnnData,
        x_axis: str,
        feature: str,
        batch_key: str | None = None,
        hue: str | None = None,
        layer: str | None = None,
        log1p_data: bool = True,
        pseudobulk: bool = False,

        # Figure Parameters
        figsize: tuple = (6, 5),
        ax: plt.Axes | None = None,
        cmap: str | Colormap | dict | None = None,

        # Layout
        xticks_order: list | None = None,
        xticks_properties: dict = None,
        hue_order: list | None = None,
        title: str = None,
        title_fontproperties: dict = None,
        legend_properties: dict = None,
        legend_title: str = None,
        legend_ncols: int = 1,
        legend_loc: str = None,

        # Statistics
        reference: str = None,
        groups: str | list = None,
        groups_pvals: float | list = None,
        test: Literal["wilcoxon", "t-test", "kruskal", "anova", "logreg", "t-test_overestim_var"] = "wilcoxon",
        corr_method: Literal["benjamini-hochberg", "bonferroni"] = "benjamini-hochberg",
        line_offset: float = 0.05,
        txt_size: int = 13,
        txt: str = "p = ",
    ):
        sanitize_anndata(adata)

        # Data Section
        self.adata = adata

        self.x_axis = x_axis
        self.hue = hue
        self.batch_key = batch_key

        self.feature = iterase_input(feature)
        self.layer = layer
        self.log1p_data = log1p_data
        self.pseudobulk = pseudobulk

        # Order for the Xticks
        self.xticks_order = xticks_order if xticks_order is not None else self._get_categories(x_axis)
        self.hue_order = hue_order if hue_order is not None else self._get_categories(hue)

        # Figure parameters
        self.figsize = figsize
        self.fig, self.gs = None, None
        self.width, self.height = figsize
        self.ax =ax
        self.legends_width = self.DEFAULT_LEGEND_WIDTH

        if self.hue is None:
            if self.x_axis + "_colors" in adata.uns.keys():
                _colormap = dict(zip(iterase_input(self.xticks_order), adata.uns[self.x_axis + "_colors"], strict=True))
            else:
                _colormap = self.DEFAULT_CMAP if cmap is None else cmap
        else:
            if self.hue + "_colors" in adata.uns.keys():
                _colormap = dict(zip(iterase_input(self.hue_order), adata.uns[self.hue + "_colors"], strict=True))
            else:
                _colormap = self.DEFAULT_CMAP if cmap is None else cmap
        self.cmap = _colormap

        colors_dict = None
        if hue is not None:
            if isinstance(self.cmap, str):
                list_colors = get_hex_colormaps(self.cmap)
                if len(list_colors) < len(iterase_input(self.hue_order)):
                    list_colors *=5
                colors_dict = dict(zip(iterase_input(self.hue_order), list_colors, strict=False))
            elif isinstance(self.cmap, dict):
                colors_dict = self.cmap
            else:
                raise  InputError("Currently palette only supports a string or dictionary")

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
        self.rotation = {"rotation": rotation} if rotation is not None else {}
        if rotation != 90:
            self.rotation["ha"] = "right"
            self.rotation["va"] = "top"

        # Legend Properties
        legend_properties = {} if legend_properties is None else legend_properties
        self.legend_title = legend_title
        self.legend_ncols = legend_ncols
        self.legend_title_fontsize = legend_properties.get("size", self.DEFAULT_LEGEND_TITLE_FONTSIZE)
        self.legend_title_fontweight = legend_properties.get("weight", self.DEFAULT_LEGEND_TITLE_FONTWEIGHT)
        self.legend_fontsize = legend_properties.get("size", self.DEFAULT_LEGEND_TITLE_FONTSIZE - 2)
        self.legend_loc = legend_loc

        # Saving
        self.dict_axis = {}

        # Statistics
        self.groups_cond = iterase_input(groups)
        self.groups_pvals = iterase_input(groups_pvals)
        self.reference = reference
        self.test = test
        self.corr_method = corr_method
        self.line_offset = line_offset
        self.txt_size = txt_size
        self.txt = txt


    # Utils for categorical plots
    def get_expression(self, keep: list) -> pd.DataFrame:
        keep = iterase_input(keep)
        if all(feature in list(self.adata.var_names) for feature in self.feature):
            df = get_expr(self.adata, self.feature, groups=keep, layer=self.layer)
        elif all(feature in list(self.adata.obs.columns) for feature in self.feature):
            df = self.adata.obs[keep + self.feature]
            df = df.rename(columns={self.feature[0]: "expr"})
        else:
            raise InputError(f"{self.feature} needs to be in adata.var_names or adata.obs")
        return df
    def get_mean_expr(self) -> pd.DataFrame:
        hue = iterase_input(self.hue)
        group_by = [self.x_axis, self.batch_key] + hue
        if all(feature in list(self.adata.var_names) for feature in self.feature):
            df_mean = get_mean_expr(self.adata, group_by=group_by, features=self.feature, layer=self.layer)
        elif all(feature in list(self.adata.obs.columns) for feature in self.feature):
            df_mean = self.adata.obs[self.feature + group_by]
            df_mean = df_mean.groupby(group_by).agg(np.mean).fillna(0).reset_index()
            df_mean["gene"] = self.feature[0]
            df_mean = df_mean.rename(columns={self.feature[0]: "expr"})
        else:
            raise InputError(f"{self.feature} is not in adata.var_names or adata.obs")
        return df_mean
    def make_figure(self, nrows: int = 1, ncols: int = 1) -> None:
        self.fig, self.gs = make_grid_spec(
            self.ax or (self.width, self.height), nrows=nrows, ncols=ncols, wspace=0.7/self.width,
            width_ratios =(
                [self.width - self.legends_width, self.legends_width] if ncols==2 else [self.width - self.legends_width]
            )
        )
        return  None


    @staticmethod
    def log_estimator_umi(values: NDArray) -> float:
        """Compute the mean of Log1p transformed data.

        :param values: Numpy array with values log1p transformed
        :return: Returns a `float` value representing the mean.
        """
        values = np.array(values, dtype=float)
        if values.shape[0] == 0:
            return np.nan
        return np.mean(np.expm1(values), dtype=float)
    @staticmethod
    def log_estimator_log_umi(values: NDArray) -> float:
        """Compute the mean of Log1p transformed data.

        :param values: Numpy array with values log1p transformed
        :return: Returns a `float` value representing the mean.
        """
        values = np.array(values, dtype=float)
        if values.shape[0] == 0:
            return np.nan
        return np.log1p(np.mean(np.expm1(values)), dtype=float)

    # Utils class
    def _get_categories(self, column: str | None) -> list | None:
        if column is None:
            return None
        else:
            return (
                list(self.adata.obs[column].cat.categories) if self.adata.obs[column].dtype.name == "category"
                else list(self.adata.obs[column].unique())
            )
    def _set_legend(self) -> None:
        # Add legend if hue is not None
        if self.hue is not None:
            axs_legend = self.fig.add_subplot(self.gs[1])
            handles = []
            for lab, c in self.cmap_dict.items():
                handles.append(
                    mlines.Line2D(
                        [0], [0], marker=".", color=c, lw=0, label=lab, markerfacecolor=c, markeredgecolor=None,
                        markersize=18
                    )
                )

            legend = axs_legend.legend(
                handles=handles, frameon=False, loc=self.legend_loc, ncols=self.legend_ncols, title=self.legend_title,
                prop={"size": self.legend_fontsize, "weight": self.legend_title_fontweight},
            )
            legend.get_title().set_fontweight("bold")
            legend.get_title().set_fontsize(self.legend_fontsize + 2)
            axs_legend.tick_params(
                axis="both", left=False, labelleft=False, labelright=False, bottom=False, labelbottom=False)
            axs_legend.spines[["right", "left", "top", "bottom"]].set_visible(False)
            axs_legend.grid(visible=False)
            self.dict_axis["legend_ax"] = axs_legend
        return  None
    def _check_ylabel(self, label: str) -> str:
        # Data can be metadata in adata.obs
        if label == "LogMean(nUMI)":
            if all(feature in list(self.adata.obs.columns) for feature in self.feature):
                logger.warn("ylabel == 'LogMean(nUMI)' but the feature is in adata.obs, setting to Value")
                label = "Value"
            if not self.log1p_data:
                label = "Mean(nUMI)"
        if label == "Log(nUMI)":
            if all(feature in list(self.adata.obs.columns) for feature in self.feature):
                logger.warn("ylabel == 'LogMean(nUMI)' but the feature is in adata.obs, setting to Value")
                label = "Value"
            if not self.log1p_data:
                label = "nUMI"
        return label
    def _set_common_layout(self, axis: plt.Axes, ylabel: str) -> None:
        # Layout for Ticks
        axis.set_xticklabels(axis.get_xticklabels(), fontweight=self.xticks_fontweight, **self.rotation)
        axis.set_xlabel("")
        axis.set_ylabel(self._check_ylabel(ylabel), fontweight="bold")
        axis.set_title(self.title, fontsize=self.title_size, fontweight=self.title_fontweight)
        return None
    def _add_statistics(self, axis, kind: Literal["violin", "box", "bar"] = None):
        if self.reference is not None and len(self.groups_cond) != 0:
            groups_pvals = self.groups_pvals
            if len(self.groups_pvals) == 0:
                testing = TestData(
                    data=self.adata, feature=self.feature[0], cond_key=self.x_axis if self.hue is None else self.hue,
                    ctrl=self.reference, groups=self.groups_cond, category_key=None if self.hue is None else self.x_axis,
                    category_order=None if self.hue is None else self.xticks_order, test=self.test,
                    test_correction=self.corr_method
                )
                testing.run_test()
                groups_pvals = testing.pvals
                del testing

            stats_plotter = StatsPlotter(
                axis, x_axis=self.x_axis, y_axis="expr", ctrl=self.reference, groups=self.groups_cond,
                pvals=iterase_input(groups_pvals), txt_size=self.txt_size, txt=self.txt, kind=kind,
                line_offset=self.line_offset, hue=self.hue, hue_order=self.hue_order,
            )
            stats_plotter.plot_stats()
            del stats_plotter
        return None

    # Plots
    def barplot(
        self,
        estimator: Literal["logmean", "mean", "median"] | Callable,
        capsize: float,
        marker_size: float,
        ylabel="LogMean(nUMI)",
        ylim_max:float = None,
        **kwargs,
    ) -> dict:
        # Set up the data
        if self.pseudobulk:
            raise InputError("Not implemented")
        else:
            df = self.get_expression(keep=[self.x_axis, self.hue] if self.hue is not None else [self.x_axis])
            df_batch = self.get_mean_expr()
            if not self.log1p_data:
                df_batch["expr"] = np.expm1(df_batch["expr"])


        # Set the estimator
        if all(feature in list(self.adata.obs.columns) for feature in self.feature):
            estimator = "mean" if estimator == "logmean" else estimator
            logger.warn("Feature in adata.obs but estimator is set to 'logmean', changing estimator to 'mean'")
        if estimator == "logmean":
            fx_est = self.log_estimator_log_umi if self.log1p_data else self.log_estimator_umi
            x_is_raw_counts(adata=self.adata, inverse=True, layer=self.layer)
        else:
            fx_est = estimator

        # Create figure
        nrows, ncols = (1, 1) if self.hue is None else (1, 2)
        self.make_figure(nrows=nrows, ncols=ncols)

        # Create Main Axes
        main_axis = self.fig.add_subplot(self.gs[0])
        bp = sns.barplot(
            df, x=self.x_axis, y="expr", estimator=fx_est, capsize=capsize, ax=main_axis, palette=self.cmap,
            hue=self.hue, order=self.xticks_order, hue_order=self.hue_order, legend=False, **kwargs
        )
        sns.stripplot(
            df_batch, x=self.x_axis, y="expr", alpha=0.75, color="k", s=marker_size, ax=bp, hue=self.hue,
            hue_order=self.hue_order, order=self.xticks_order, dodge=True if self.hue else False, legend=False
        )
        self._add_statistics(axis=bp, kind="bar")

        # Layout for title
        self._set_common_layout(axis=bp, ylabel=ylabel)

        # Correct ylim
        if self.adata.obs[self.batch_key].nunique() > 2:
            ymax = bp.get_ylim()[1]
            ymax = ylim_max if ylim_max is not None else ymax + ymax * 0.1
            ymin = 0
            if df_batch["expr"].min() < 0  and estimator != "logmean":
                ymin =  df_batch["expr"].min() + df_batch["expr"].min() * 0.1
            bp.set_ylim(ymin, ymax)

        # Add Legend
        self._set_legend()

        # Save Main Axis
        self.dict_axis["mainplot_ax"] = bp
        return  self.dict_axis

    def boxplot(
        self,
        showfliers: bool = False,
        scatter: bool = False,
        marker_size: float = 2,
        ylabel: str = "Log(nUMI)",
        **kwargs,
    ) -> dict:
        # Extract the data
        if self.pseudobulk:
            df = self.get_mean_expr()
        else:
            df = self.get_expression(keep=[self.x_axis,self.hue] if self.hue is not None else [self.x_axis])

        # Create figure
        nrows, ncols = (1, 1) if self.hue is None else (1, 2)
        self.make_figure(nrows=nrows, ncols=ncols)
        main_axis = self.fig.add_subplot(self.gs[0])

        bx = sns.boxplot(
            df, x=self.x_axis, y="expr", showfliers=showfliers, ax=main_axis, palette=self.cmap,
            order=self.xticks_order, hue=self.hue, hue_order=self.hue_order, legend=False, **kwargs
        )
        if scatter:
            sns.stripplot(
                df, x=self.x_axis, y="expr", ax=bx, color="k", order=self.xticks_order,
                hue=self.hue, hue_order=self.hue_order, legend=False, size=marker_size, dodge=True
            )
        self._add_statistics(bx, kind="box")

        # Layout for title
        self._set_common_layout(axis=bx, ylabel=ylabel)

        # Add Legend
        self._set_legend()

        # Save Main Axis
        self.dict_axis["mainplot_ax"] = bx
        return self.dict_axis


    def violinplot(
        self,
        scatter: bool =False,
        marker_size: int = 2,
        cut: float = 0,
        ylabel: str = "Log(nUMI)",
        **kwargs
    ) -> dict:
        df = self.get_expression(keep=[self.x_axis, self.hue] if self.hue is not None else [self.x_axis])

        nrows, ncols = (1, 1) if self.hue is None else (1, 2)
        self.make_figure(nrows=nrows, ncols=ncols)
        main_axis = self.fig.add_subplot(self.gs[0])

        vln = sns.violinplot(
            df, x=self.x_axis, y="expr", ax=main_axis, palette=self.cmap, cut=cut,
            order=self.xticks_order, hue=self.hue, hue_order=self.hue_order, legend=False, **kwargs
        )
        if scatter:
            sns.stripplot(
                df[df.expr != 0], x=self.x_axis, y="expr", ax=vln, color="k", order=self.xticks_order,
                hue=self.hue, hue_order=self.hue_order, legend=False, size=marker_size, dodge=True
            )
        self._add_statistics(vln, kind="violin")

        # Layout for title
        self._set_common_layout(axis=vln, ylabel=ylabel)

        # Add Legend
        self._set_legend()

        # Save Main Axis
        self.dict_axis["mainplot_ax"] = vln
        return self.dict_axis




# class BaseSeaborn:
#     """
#     Utility class to plot data from an AnnData Object using seaborn.
#     """
#
#     MIN_FIGURE_HEIGHT = 4.2
#     DEFAULT_WSPACE = 0.0
#     DEFAULT_LEGEND_WIDTH = 1.5
#     DEFAULT_CMAP = "tab10"
#
#     DEFAULT_TITLE_SIZE = 20
#     DEFAULT_TITLE_FONTWEIGHT = "bold"
#
#     DEFAULT_XTICKS_SIZE = 12
#     DEFAULT_XTICKS_FONTWEIGHT = "bold"
#     DEFAULT_XTICKS_ROTATION = None
#
#     DEFAULT_LEGEND_TITLE_FONTSIZE = 12
#     DEFAULT_LEGEND_TITLE_FONTWEIGHT = "bold"
#
#     def __init__(
#         self,
#         adata: ad.AnnData,
#         x_axis: str,
#         feature: str,
#         batch_key: str = None,
#         xticks_order: list = None,
#         hue: str = None,
#         hue_order: list = None,
#         layer: str = None,
#         logcounts: bool = True,
#         figsize: tuple = (3, 4.2),
#         ax: plt.Axes = None,
#         cmap: str | Colormap | dict = None,
#         title: str = None,
#         title_fontproperties: dict = None,
#         xticks_properties: dict = None,
#         legend_properties: dict = None,
#         path: PathLike = None,
#         filename: str = "figure.svg",
#         show: bool = True
#     ):
#         """Initialize class.
#
#         :param adata: Annotated data matrix.
#         :param x_axis: Name of a categorical column in `adata.obs` to groupby.
#         :param feature: A valid feature in `adata.var_names` or column in `adata.obs` with continuous values.
#         :param batch_key: Name of a categorical column in `adata.obs` that contains the sample names.
#         :param xticks_order: Order for the categories in `adata.obs[x_axis]`.
#         :param hue: Name of a second categorical column in `adata.obs` to use additionally to groupby.
#         :param hue_order: List with orders for the categories in `hue`. If it is not set, the order will be inferred.
#         :param layer: Name of the AnnData object layer that wants to be plotted. The default `adata.X` is plotted. If
#                      layer is set to a valid layer name, then the layer is plotted.
#         :param logcounts: If set to `True`, consider that the values in `adata.X` or `adata.layers[layer]` if layer is
#                          set is log1p transformed.
#         :param figsize: Figure size, the format is (width, height).
#         :param ax: Matplotlib axes to use for plotting. If not set, a new figure will be generated.
#         :param cmap: String denoting matplotlib colormap. A dictionary with the categories available in
#                     `adata.obs[x_axis]` or `adata.obs[hue]` if hue is not None can also be provided. The format is
#                     {category:color}.
#         :param title: Title for the figure.
#         :param title_fontproperties: Dictionary which should contain 'size' and 'weight' to define the fontsize and
#                                     fontweight of the title of the figure.
#         :param xticks_properties: Dictionary which should contain 'size' and 'weight' to define the fontsize and
#                                  fontweight of the xticks of the figure.
#         :param legend_properties: Dictionary which should contain 'size' and 'weight' to define the fontsize and
#                                  fontweight of the title of the legend.
#         :param path: Path to the folder to save the figure.
#         :param filename: Name of file to use when saving the figure.
#         :param show: If set to `False`, returns a dictionary with the matplotlib axes.
#         """
#         sanitize_anndata(adata)
#
#         # Data Section
#         self.adata = adata
#         self.x_axis = x_axis
#         self.feature = iterase_input(feature)  # We always assume we have a list
#         self.batch_key = batch_key
#
#         self.xticks_order = xticks_order if xticks_order is not None else list(adata.obs[x_axis].unique())
#         self.hue = hue
#         if hue is None:
#             self.hue_order = None
#         else:
#             self.hue_order = hue_order if hue_order is not None else list(adata.obs[hue].unique())
#
#         self.layer = layer
#         self.logcounts = logcounts
#
#
#         # Figure parameters
#         self.figsize = figsize
#         self.fig = None,
#         self.gs = None
#         self.width, self.height = figsize if figsize is not None else (None, None)
#         self.ax = ax
#         self.legends_width = self.DEFAULT_LEGEND_WIDTH
#         self.cmap = self.DEFAULT_CMAP if cmap is None else cmap
#
#         colors_dict = None  # Only used when hue is not None
#         if hue is not None:
#             if isinstance(self.cmap, str):
#                 list_colors = get_hex_colormaps(self.cmap)
#                 if len(list_colors) < len(self.hue_order):
#                     list_colors *= 5
#                 colors_dict = dict(zip(self.hue_order, get_hex_colormaps(self.cmap), strict=False))
#             elif isinstance(self.cmap, dict):
#                 colors_dict = self.cmap
#             else:
#                 raise Exception('palette can only be a string or dictionary')
#
#         self.cmap_dict = colors_dict
#
#         # Title Properties
#         self.title = title if title is not None else feature
#         title_fontproperties = {} if title_fontproperties is None else title_fontproperties
#         self.title_size = title_fontproperties.get("size", self.DEFAULT_TITLE_SIZE)
#         self.title_fontweight = title_fontproperties.get("weight", self.DEFAULT_TITLE_FONTWEIGHT)
#
#         # X-ticks Properties
#         xticks_properties = {} if xticks_properties is None else xticks_properties
#         self.xticks_fontsize = xticks_properties.get("size", self.DEFAULT_XTICKS_SIZE)
#         self.xticks_fontweight = xticks_properties.get("weight", self.DEFAULT_XTICKS_FONTWEIGHT)
#         rotation = xticks_properties.get("rotation", self.DEFAULT_XTICKS_ROTATION)
#         self.rotation = {"rotation": rotation, "ha": "right", "va": "top"} if rotation is not None else {}
#
#         # Legend Properties
#         legend_properties = {} if legend_properties is None else legend_properties
#         self.legend_title = legend_properties.get("size", self.DEFAULT_LEGEND_TITLE_FONTSIZE)
#         self.legend_title_fontweight = legend_properties.get("weight", self.DEFAULT_LEGEND_TITLE_FONTWEIGHT)
#         self.legend_fontsize = legend_properties.get("size", self.DEFAULT_LEGEND_TITLE_FONTSIZE - 2)
#
#         # Saving
#         self.path = path
#         self.filename = filename
#         self.show = show
#         self.dict_axis = None
#
#         return
#
#     def make_figure(
#         self,
#         nrows: int = 1,
#         ncols: int = 1
#     ) -> None:
#         """Generate figure.
#
#         :param nrows: Number of rows.
#         :param ncols: Number of Columns
#         :return: Returns None.
#         """
#         mainplot_width = self.width - self.legends_width
#
#         fig, gs = self.make_grid_spec(
#             self.ax or (self.width, self.height),
#             nrows=nrows, ncols=ncols, wspace=0.7 / self.width,
#             width_ratios=[mainplot_width, self.legends_width] if ncols == 2 else [mainplot_width]
#         )
#
#         self.fig = fig
#         self.gs = gs
#         return None
#
#     def legend(
#         self,
#         show: bool = False,
#         width: float = 1.5,
#         title: str = None,
#     ) -> None:
#         """Set legend parameters.
#
#         :param show: If set to `False`, the legend is deactivated.
#         :param width: width of the figure reserve for the legend.
#         :param title: title of the legend.
#         :return: Returns None.
#         """
#         if not show:
#             # Deactivate legend by setting the width to 0
#             self.legends_width = 0
#         else:
#             self.legend_title = title
#             self.legends_width = width
#         return None
#
#     @staticmethod
#     def make_grid_spec(
#         ax_or_figsize: tuple[int, int] | _AxesSubplot,
#         *,
#         nrows: int,
#         ncols: int,
#         wspace: float | None = None,
#         hspace: float | None = None,
#         width_ratios: Sequence[float] | None = None,
#         height_ratios: Sequence[float] | None = None,
#     ) -> tuple[Figure, gridspec.GridSpecBase]:
#         """Adapted from Scanpy"""
#
#         kw = dict(wspace=wspace, hspace=hspace, width_ratios=width_ratios, height_ratios=height_ratios)
#         if isinstance(ax_or_figsize, tuple):
#             fig = plt.figure(figsize=ax_or_figsize)
#             return fig, gridspec.GridSpec(nrows, ncols, **kw)
#         else:
#             ax = ax_or_figsize
#             ax.axis("off")
#             ax.set_frame_on(False)
#             ax.set_xticks([])
#             ax.set_yticks([])
#             return ax.figure, ax.get_subplotspec().subgridspec(nrows, ncols, **kw)
#
#     def saving_return_axis(self) -> dict[str, Axes] | plt.Axes | None:
#         """Return axis and save figure.
#
#         :return: Returns a dictionary with the matplotlib axes or matplotlib axes if `show` is set to False.
#         """
#         if self.path is not None:
#             plt.savefig(convert_path(self.path) / self.filename, bbox_inches="tight")
#         if self.show:
#             plt.tight_layout()
#             return plt.show()
#         else:
#             return self.dict_axis
#
#
#     def set_xticks(self, ax: plt.Axes) -> None:
#         """Set properties for the xticks.
#
#         :param ax: Matplotlib Axes.
#         :return: Returns None.
#         """
#         ax.set_xticklabels(ax.get_xticklabels(), fontweight=self.xticks_fontweight, **self.rotation)
#         return None
#
#     def set_title(self, ax: plt.Axes) -> None:
#         """Set the properties for the title.
#
#         :param ax: Matplotlib Axes.
#         :return: Returns None
#         """
#         ax.set_title(self.title, fontsize=self.title_size, fontweight=self.title_fontweight)
#         return None
#
#     def get_expression(self, keep: list) -> pd.DataFrame:
#         """Get the expression.
#
#         :param keep: Columns in `adata.obs` to keep.
#         :return: Returns a DataFrame with the expression extracted from the AnnData object.
#         """
#         from dotools_py.get._generic import expr as get_expr
#         keep = iterase_input(keep)
#         if all(feature in list(self.adata.var_names) for feature in self.feature):
#             df = get_expr(self.adata, self.feature, groups=keep, layer=self.layer)
#         elif all(feature in list(self.adata.obs.columns) for feature in self.feature):
#             df = self.adata.obs[keep + self.feature]
#             # df["expr"] = df[self.feature[0]]
#             df = df.rename(columns={self.feature[0]: "expr"})
#         else:
#             raise ValueError(f"{self.feature} needs to be in adata.var_names or adata.obs")
#         return df
#
#
#     def get_mean_expression(self) -> pd.DataFrame:
#         """Get the mean expression.
#
#         :return: Returns a DataFrame with the mean expression.
#         """
#         from dotools_py.get._generic import mean_expr as get_mean_expr
#
#         hue = iterase_input(self.hue)
#         group_by = [self.x_axis, self.batch_key] + hue
#
#         if all(feature in list(self.adata.var_names) for feature in self.feature):
#             df_mean = get_mean_expr(self.adata, group_by=group_by, features=self.feature, layer=self.layer)
#         elif all(feature in list(self.adata.obs.columns) for feature in self.feature):
#             df_mean = self.adata.obs[self.feature + group_by]
#             df_mean = df_mean.groupby(group_by).agg(np.mean).fillna(0).reset_index()
#             df_mean["gene"] = self.feature[0]
#             df_mean["expr"] = df_mean[self.feature[0]]
#         else:
#             raise ValueError(f"{self.feature} is not in adata.var_names or adata.obs")
#         return df_mean
#
#     @staticmethod
#     def log_estimator(values: np.ndarray):
#         """Compute mean of log1p transform data.
#
#         :param values: values to calculate the mean expression on.
#         :return: Returns a numpy array with the mean expression log1p transform
#         """
#         return np.log1p(np.mean(np.expm1(values)))

