import anndata as ad
import pandas as pd
import numpy as np
from typing import Literal, Callable, Dict

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as patches
from matplotlib.cm import ScalarMappable
import seaborn as sns
from scipy.stats import zscore

from dotools_py._utils import sanitize_anndata, iterase_input, x_is_raw_counts
from dotools_py._custom_class import  InputError, PathLike
from dotools_py.pl._plot_utils import make_grid_spec, return_axis, save_plot, check_colornorm, square_color, small_squares
from dotools_py.utility import get_hex_colormaps
from matplotlib.colors import Colormap
from numpy.typing import NDArray
from dotools_py.pl._StatsPlotter import TestData, StatsPlotter
from dotools_py.logger import  logger
from dotools_py.get._generic import expr as get_expr
from dotools_py.get._generic import mean_expr as get_mean_expr
from dotools_py.tl import rank_genes_groups

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
            if cmap is None:
                if self.x_axis + "_colors" in adata.uns.keys():
                    _colormap = dict(zip(iterase_input(self.xticks_order), adata.uns[self.x_axis + "_colors"], strict=True))
                else:
                    _colormap = self.DEFAULT_CMAP
            else:
                _colormap = cmap
        else:
            if cmap is None:
                if self.hue + "_colors" in adata.uns.keys():
                    _colormap = dict(zip(iterase_input(self.hue_order), adata.uns[self.hue + "_colors"], strict=True))
                else:
                    _colormap = self.DEFAULT_CMAP
            else:
                _colormap = cmap
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

    def lineplot(
        self,
    ) -> dict:
        ...





class MatrixPlot:
    MIN_FIGURE_HEIGHT = 4.2
    DEFAULT_WSPACE = 0.0
    DEFAULT_LEGEND_WIDTH = 1.5
    DEFAULT_ANNOT_WIDTH = 0.3
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
        adata: ad.AnnData,
        x_axis: str,
        features: str | list,
        y_axis: str | None = None,
        xticks_order: list | None = None,
        yticks_order: list | None = None,
        layer: str | None = None,
        logcounts: bool = True,

        # Figure parameters
        figsize: tuple = (5, 6),
        ax: plt.Axes | None = None,
        swap_axes: bool = True,
        title: str = "",
        title_fontproperties: Dict[Literal["size", "weight"], str | int] | None = None,
        palette: str = "Reds",

        xticks_properties: dict | None = None,
        yticks_properties: dict | None = None,
        xticks_rotation: int = 45,
        yticks_rotation: int = 0,
        cluster_x_axis: bool = False,
        cluster_y_axis: bool = False,

        legend_title: str = "LogMean(nUMI)\nin group",

        # IO
        path: PathLike | None = None,
        filename: str = "Heatmap.svg",
        show: bool = True,

        # Statistics
        add_stats: Literal["x_axis", "y_axis"] | None = None,
        test: Literal["wilcoxon", "t-test"] = "wilcoxon",
        correction_method: Literal["benjamini-hochberg", "bonferroni"] = "benjamini-hochberg",
        df_pvals: pd.DataFrame | None= None,
        stats_x_size: float | None  = None,
        square_x_size: dict | None  = None,
        pval_cutoff: float = 0.05,
        log2fc_cutoff: float = 0.0,

        # Fx specific
        z_score: Literal["x_axis", "y_axis"] | None  = None,
        clustering_method: str = "complete",
        clustering_metric: str = "euclidean",
        linewidth: float = 0.1,
        vmin: float | None = None,
        vcenter: float | None= None,
        vmax: float | None = None,
        annot_fontsize: float = 12,
        annot_ratio: float = 0.35,
        **kwargs,
    ):

        sanitize_anndata(adata)

        # region Data Section
        self.adata = adata
        self.x_axis, self.y_axis = x_axis, y_axis
        self.swap_axes = swap_axes

        features = iterase_input(features)
        self._check_missing(adata, features)
        self.features = features
        self.z_score = z_score

        self.layer = layer
        self.logcounts = logcounts
        # endregion


        # region Figure parameters
        self.figsize = figsize
        self.width, self.height = figsize
        self.fig, self.gs = None, None
        self.ax = ax

        self.legends_width = self.DEFAULT_LEGEND_WIDTH
        self.annot_width = self.DEFAULT_ANNOT_WIDTH

        min_figure_height = max([0.35, self.height])
        cbar_legend_height = min_figure_height * 0.08
        sig_legend = min_figure_height * 0.27
        spacer_height = min_figure_height * 0.3

        self.height_ratios_legend = [
            self.height - sig_legend - cbar_legend_height - spacer_height,
            sig_legend,
            spacer_height,
            cbar_legend_height,
        ]

        self.height_ratios_annot = [
            annot_ratio,
            self.height - annot_ratio,
        ]

        self.palette = palette
        # endregion

        # region Title properties
        self.title = title
        title_fontproperties = {} if title_fontproperties is None else title_fontproperties
        self.title_size = title_fontproperties.get("size", self.DEFAULT_TITLE_SIZE)
        self.title_fontweight = title_fontproperties.get("weight", self.DEFAULT_TITLE_FONTWEIGHT)
        # endregion


        # region Tick Properties
        # Order for the X and Y ticks | Order for features is based on input
        self.xticks_order = xticks_order if xticks_order is not None else self._get_categories(x_axis)
        self.yticks_order = yticks_order if yticks_order is not None else self._get_categories(y_axis)

        xticks_properties = {} if xticks_properties is None else xticks_properties
        self.xticks_fontsize = xticks_properties.get("size", self.DEFAULT_XTICKS_SIZE)
        self.xticks_fontweight = xticks_properties.get("weight", self.DEFAULT_XTICKS_FONTWEIGHT)
        self.rotation_props_x = {"rotation": xticks_rotation} if xticks_rotation is not None else {}
        if xticks_rotation != 90 and "ha" not in xticks_properties.keys():
            self.rotation_props_x["ha"] = "right"
            self.rotation_props_x["va"] = "top"

        yticks_properties = {} if yticks_properties is None else yticks_properties
        self.yticks_fontsize = yticks_properties.get("size", self.DEFAULT_XTICKS_SIZE)
        self.yticks_fontweight = yticks_properties.get("weight", self.DEFAULT_XTICKS_FONTWEIGHT)
        self.yticks_rotation = {"rotation": yticks_rotation} if yticks_rotation is not None else {}
        # endregion


        # region Legend Properties
        self.legend_title = legend_title
        # endregion


        # region Saving
        self.return_ax_dict = {}
        self.path = path
        self.filename = filename
        self.show = show
        # endregion


        # region Statistics
        self.add_stats = add_stats
        self.test = test
        self.correction_method = correction_method
        self.df_pvals = df_pvals

        square_x_size = {} if square_x_size is None else square_x_size
        square_x_size = {"width": square_x_size.get("weight", 1), "size": square_x_size.get("size", 0.8)}
        self.stats_x_size = stats_x_size
        self.square_x_size = square_x_size
        self.pval_cutoff = pval_cutoff
        self.log2fc_cutoff = log2fc_cutoff
        # endregion

        # region Fx specific
        self.cluster_x_axis = cluster_x_axis
        self.cluster_y_axis = cluster_y_axis

        self.clustering_method = clustering_method
        self.clustering_metric = clustering_metric

        self.linewidth = linewidth
        self.kwargs = kwargs
        self.vmin = vmin
        self.vmax = vmax
        self.vcenter = vcenter
        self.annot_fontsize = annot_fontsize
        # endregion


    # Data operations
    def _get_mean_expr(self) -> pd.DataFrame:
        x_axis, y_axis = iterase_input(self.x_axis), iterase_input(self.y_axis)
        group_by = x_axis + y_axis

        if all(feature in list(self.adata.var_names) for feature in self.features):
            df_mean = get_mean_expr(self.adata, group_by=group_by, features=self.features, layer=self.layer,
                                    logcounts=self.logcounts)
        elif all(feature in list(self.adata.obs.columns) for feature in self.features):
            df_mean = self.adata.obs[self.features + group_by]
            df_mean = df_mean.groupby(group_by).agg("mean").fillna(0).reset_index()
            df_mean = df_mean.melt(id_vars=group_by, var_name="gene", value_name="expr")
        else:
            raise InputError(f"{self.features} are not in adata.var_names or adata.obs")
        return df_mean

    def _scale(self, df: pd.DataFrame) -> pd.DataFrame:
        import scipy
        if self.y_axis is None:
            if self.z_score == "y_axis":
                df = df.apply(zscore, axis=0, result_type="expand")
            elif self.z_score == "x_axis":
                df = df.apply(lambda row: pd.Series(zscore(row), index=df.columns), axis=1)
            else:
                raise InputError(f'{self.z_score} not a valid key for z_score, use "x_axis" or "y_axis"')
        else:
            if self.z_score == "y_axis":
                df = df.groupby(level="gene", axis=1, group_keys=False).apply(
                    lambda x: x.apply(scipy.stats.zscore, axis=0)
                )
            elif self.z_score =="x_axis":
                _backup = df.columns
                df = df.groupby(level="gene", axis=1, group_keys=False).apply(
                    lambda x: pd.DataFrame(scipy.stats.zscore(x, axis=1), index=x.index, columns=x.columns)
                )
                df.columns = _backup

            else:
                raise InputError(f'{self.z_score} not a valid key for z_score, use "x_axis" or "y_axis"')

        if self.palette == "Reds":
            logger.warn("Z-score set to True, but the palette is Reds, setting to RdBu_r")  # Make sure to use divergent colormap
            self.palette = "RdBu_r"
        if self.legend_title == "LogMean(nUMI)\nin group":
            self.legend_title = "Z-score"
        df = df.fillna(0)
        return df

    def _get_categories(self, column: str | None) -> list | None:
        if column is None:
            return None
        else:
            return (
                list(self.adata.obs[column].cat.categories) if self.adata.obs[column].dtype.name == "category"
                else list(self.adata.obs[column].unique())
            )

    def _reindex(self, df: pd.DataFrame) -> tuple:
        from scipy.cluster.hierarchy import dendrogram, linkage

        # Sort based on categories
        if self.y_axis is None:
            new_index, new_cols = self.features, self.xticks_order
        else:
            new_index = self.yticks_order
            new_cols = pd.MultiIndex.from_product([self.features, self.xticks_order], names=["gene", self.x_axis])

        # Resort in case clustering is True
        new_index = (
            df.index[
                dendrogram(
                    linkage(df.values, method=self.clustering_method, metric=self.clustering_metric), no_plot=True
                )["leaves"]]
            if self.cluster_y_axis
            else new_index
        )

        new_cols = (
            df.columns[
                dendrogram(
                    linkage(df.T.values, method=self.clustering_method, metric=self.clustering_metric), no_plot=True
                )["leaves"]
            ]
            if self.cluster_x_axis
            else new_cols
        )

        return new_index, new_cols

    @staticmethod
    def _check_missing(adata: ad.AnnData, values: list):
        missing = [g for g in values if g not in adata.var_names]
        if len(missing) != 0:
            missing = [g for g in missing if g not in adata.obs.columns]
        assert len(missing) == 0, f'{missing} features missing in the object'

    def _compute_stats(self, df):
        import scanpy as sc
        if self.add_stats is not None:
            if self.add_stats == "y_axis" and self.y_axis is None:
                raise ValueError("Testing y_axis but argument is None")
            group_by = self.x_axis if self.add_stats == "x_axis" else self.y_axis
            alternative = self.x_axis if self.add_stats == "y_axis" else self.y_axis
            if self.df_pvals is None:
                features = iterase_input(self.features)

                if self.y_axis is None:
                    if all(item in list(self.adata.var_names) for item in features):
                        try:
                            rank_genes_groups(
                                self.adata, groupby=group_by, method=self.test, tie_correct=True,
                                corr_method=self.correction_method, layer=self.layer
                            )
                            table = sc.get.rank_genes_groups_df(
                                self.adata, group=None, pval_cutoff=self.pval_cutoff, log2fc_min=self.log2fc_cutoff
                            )
                            table_filt = table[table["names"].isin(features)]

                            if len(table_filt) == 0:
                                logger.warn("No significant groups")
                        except Exception as e:
                            logger.warn(f"Error testing, {e}")
                            table_filt = pd.DataFrame(
                                [], columns=[
                                    'group', 'names', 'scores', 'logfoldchanges', 'pvals', 'pvals_adj', 'pct_nz_group',
                                    'pct_nz_reference']
                            )
                    elif all(item in list(self.adata.obs.columns) for item in features):
                        raise NotImplementedError("Testing for features in adata.obs is not implemented")
                    else:
                        raise InputError("Not a valid input for testing")
                else:
                    if all(item in list(self.adata.var_names) for item in features):
                        table_filt = pd.DataFrame([])
                        for alt in self.adata.obs[alternative].unique():
                            sdata = self.adata[self.adata.obs[alternative] == alt].copy()
                            try:
                                rank_genes_groups(sdata, groupby=group_by, method=self.test, tie_correct=True,
                                                  corr_method=self.correction_method, layer=self.layer)
                                stable = sc.get.rank_genes_groups_df(
                                    sdata, group=None, pval_cutoff=self.pval_cutoff, log2fc_min=self.log2fc_cutoff
                                )
                            except Exception as e:
                                logger.warn(f'Error while testing: {e}')
                                stable = pd.DataFrame(
                                    [], columns=[
                                        'group', 'names', 'scores', 'logfoldchanges', 'pvals', 'pvals_adj',
                                        'pct_nz_group', 'pct_nz_reference']
                                )
                            stable_filt = stable[stable["names"].isin(features)]
                            stable_filt['group2'] = alt
                            table_filt = pd.concat([table_filt, stable_filt])
                        if len(table_filt) == 0:
                            logger.warn('No Significant group')
                    elif all(item in list(self.adata.obs.columns) for item in features):
                        raise NotImplementedError("Testing for features in adata.obs is not implemented")
                    else:
                        raise InputError("Not a valid input for testing")
            else:
                raise InputError("Not a valid input for testing")

            columns = df.columns
            index = df.index
            df_pvals = pd.DataFrame([], index=index, columns=columns)

            for idx, row in table_filt.iterrows():
                if self.y_axis is None:
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
            annot_pvals  = (df_pvals < self.pval_cutoff).replace({True: "*", False: ""})
            self.annot_pvals = annot_pvals
            self.df_pvals = df_pvals
        else:
            self.annot_pvals = None
        return None

    # Visualisation
    def make_figure(self, nrows: int = 1, ncols: int = 1, height_ratios: list | None = None, hspace: float | None = None) -> None:
        self.fig, self.gs = make_grid_spec(
            self.ax or (self.width, self.height), nrows=nrows, ncols=ncols, wspace=0.7/self.width,
            width_ratios =(
                [self.width - self.legends_width, self.legends_width] if ncols==2 else [self.width - self.legends_width]
            ),
            height_ratios=height_ratios, hspace=hspace,
        )
        return None

    def _create_colorbar(self, df, axis):
        from matplotlib.colorbar import Colorbar

        if self.z_score is None:
            vmin = 0.0 if self.vmin is None else self.vmin
            vmax = round(df.max().max() * 20) / 20 if self.vmax is None else self.vmax
            txt = ""
        else:
            vmin = round(df.min().min() * 20) / 20 if self.vmin is None else self.vmin
            vmax = round(df.max().max() * 20) / 20 if self.vmax is None else self.vmax

            if self.y_axis is None:
                n = self.x_axis if self.z_score == "x_axis" else "features"
            else:
                n = self.x_axis if self.z_score == "x_axis" else self.y_axis
            txt = f"\nacross {n}"

        self.vmin, self.vmax = vmin, vmax

        colormap = plt.get_cmap(self.palette)
        normalize = check_colornorm(vmin=self.vmin, vmax=self.vmax, vcenter=self.vcenter)
        mappable = ScalarMappable(norm=normalize, cmap=colormap)

        Colorbar(axis, mappable=mappable, orientation="horizontal")
        axis.set_title(self.legend_title + txt, fontsize="small", fontweight="bold")
        axis.xaxis.set_tick_params(labelsize="small")
        return axis

    # Main Fxs
    def heatmap(self):
        df = self._get_mean_expr()

        # stats_x_size = max(np.sqrt(height * width), 14) if stats_x_size is None else stats_x_size
        self.stats_x_size = min(self.width / df.shape[1], self.height / df.shape[1]) * 10 if self.stats_x_size is None else min(
            self.width / df.shape[1], self.height / df.shape[1]) * self.stats_x_size

        if self.y_axis is None:
            # Features x Categories
            df = df.pivot(index="gene", columns=self.x_axis, values="expr")

            # Apply Z-score scaling
            if self.z_score is not None:
                df = self._scale(df)

            # Reindex
            new_idx, new_cols = self._reindex(df)
            df = df.reindex(index=new_idx, columns=new_cols)

            # Swap Axes --> Categories x Features
            if self.swap_axes:
                df = df.T

            # Make figure
            self.make_figure(nrows=1, ncols=2)
            main_ax = self.fig.add_subplot(self.gs[0])
            legend_ax = self.fig.add_subplot(self.gs[1])
            annot_ax = None
        else:
            # XCategories x YCategories x Features
            df = df.pivot(index=self.y_axis, columns=["gene", self.x_axis], values="expr")

            if self.z_score is not None:
                df = self._scale(df)

            # Reindex
            new_idx, new_cols = self._reindex(df)
            df = df.reindex(index=new_idx, columns=new_cols)

            # Make Figure
            self.make_figure(nrows=2, ncols=2, height_ratios=self.height_ratios_annot, hspace=0)
            annot_ax = self.fig.add_subplot(self.gs[0, 0])
            main_ax = self.fig.add_subplot(self.gs[1, 0], sharex=annot_ax)
            legend_ax = self.fig.add_subplot(self.gs[:, 1])

        fig, legend_gs = make_grid_spec(legend_ax, nrows=4, ncols=1, height_ratios=self.height_ratios_legend)
        color_legend_ax = fig.add_subplot(legend_gs[3])

        if self.add_stats:
            sig_ax = fig.add_subplot(legend_gs[2])

        self._compute_stats(df)

        # Plot data
        # Add Legend
        self.return_ax_dict["legend_ax"] = self._create_colorbar(df=df, axis=color_legend_ax)

        hm = sns.heatmap(
            df,
            cmap=self.palette,
            ax=main_ax,
            linewidths=self.linewidth,
            cbar=False,
            annot_kws={"color": "black", "size": self.stats_x_size, "ha": "center", "va": "center", "fontfamily":'DejaVu Sans Mono'},
            annot=self.annot_pvals,
            fmt="s",
            square=False,
            vmin=self.vmin,
            vmax=self.vmax,
            center=self.vcenter,
            **self.kwargs,
        )

        hm.spines[["top", "right", "bottom", "left"]].set_visible(True)
        hm.set_xlabel("")
        hm.set_ylabel("")

        hm.set_xticklabels(
            df.columns.get_level_values(self.x_axis) if annot_ax is not None else df.columns,
            fontdict={"weight": self.xticks_fontweight, "size": self.xticks_fontsize}, **self.rotation_props_x)

        hm.set_yticklabels(
            hm.get_yticklabels(), fontdict={"weight": self.yticks_fontweight, "size": self.yticks_fontsize}, **self.yticks_rotation)
        hm.set_title(self.title, fontdict={"size": self.title_size, "weight":self.title_fontweight})

        self.return_ax_dict["mainplot_ax"] = hm

        # Annotation Axis
        if annot_ax is not None:
            annot_ax.set_xlim(0, df.shape[1])
            annot_ax.set_ylim(0, 1)
            annot_ax.axis("off")
            genes = df.columns.get_level_values("gene")

            start, current = 0, genes[0]
            for i in range(1, len(genes) + 1):
                if i == len(genes) or genes[i] != current:
                    width = i - start
                    rect = patches.Rectangle(
                        (start, 0), width,1, facecolor="lightgray", edgecolor="black",
                        clip_on=False, linewidth=hm.spines["right"].get_linewidth()
                    )
                    annot_ax.add_patch(rect)
                    annot_ax.text(
                        start + width / 2, 0.5, current, ha="center", va="center", fontsize=self.annot_fontsize,
                        fontweight="bold",
                    )
                    if i < len(genes):
                        hm.axvline(i, color="black", linewidth=hm.spines["right"].get_linewidth())
                        start, current = i, genes[i]
            self.return_ax_dict["annot_ax"] = annot_ax

        # Significance legend
        if self.add_stats:
            x, y = 0, 0.5
            sig_ax.scatter(x, y, s=500, facecolors="none", edgecolors="black", marker="s")
            sig_ax.text(x, y, "*", fontsize=18, ha="center", va="center", color="black", fontfamily='DejaVu Sans Mono')
            sig_ax.text(x + 0.03, y, "FDR < 0.05", fontsize=12, va="center", fontweight="bold")
            sig_ax.set_xlim(x - 0.02, x + 0.1)

            n = self.x_axis if self.add_stats == "x_axis" else self.y_axis
            txt = f"\nacross {n}"

            sig_ax.set_title("Significance" + txt, fontsize="small", fontweight="bold")
            plt.gca().set_aspect("equal")
            sig_ax.axis("off")  # Hide axes for clean display
            self.return_ax_dict["signifiance_ax"] = sig_ax

        if self.add_stats:
            colormap = plt.get_cmap(self.palette)
            normalize = check_colornorm(vmin=self.vmin, vmax=self.vmax, vcenter=self.vcenter)

            df_x = pd.DataFrame([], index=df.index, columns=df.columns)
            df_x[df_x.isna()] = "black"
            df_x = df.map(lambda x: square_color(colormap(normalize(x))))
            pos_rows, pos_cols = np.where(self.df_pvals < 0.05)
            pos = list(zip(pos_rows, pos_cols, strict=False))
            colors = [df_x.iloc[row, col] for row, col in pos]

            small_squares(
                hm,
                color=colors,
                pos=pos,
                size=self.square_x_size["size"],
                linewidth=self.square_x_size["width"],
            )

            # Now set colors manually on each annotation text base on the background
            for text, color in zip(hm.texts, df_x.values.flatten(), strict=False):
                text.set_color(color)

        save_plot(path = self.path, filename =self.filename)

        if self.show:
            return plt.show()
        else:
            return return_axis(self.show, self.return_ax_dict, tight=True)


