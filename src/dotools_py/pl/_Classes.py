from typing import Literal, Callable, Dict, Any

import matplotlib.lines as mlines
import matplotlib.patches as patches
from matplotlib.cm import ScalarMappable
import textwrap
from adjustText import adjust_text

from prelude_py import ad, pd, np, sns, plt

import networkx as nx

from scipy.stats import zscore
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from itertools import combinations


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
                    _colormap = dict(zip(iterase_input(self.xticks_order), get_hex_colormaps(self.DEFAULT_CMAP)[:len(iterase_input(self.xticks_order))], strict=True))
            else:
                _colormap = cmap
        else:
            if cmap is None:
                if self.hue + "_colors" in adata.uns.keys():
                    _colormap = dict(zip(iterase_input(self.hue_order), adata.uns[self.hue + "_colors"], strict=True))
                else:
                    _colormap = dict(zip(iterase_input(self.hue_order), get_hex_colormaps(self.DEFAULT_CMAP)[:len(iterase_input(self.hue_order))], strict=True))
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
            df = get_expr(self.adata, features=self.feature, groups=keep, layer=self.layer)
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
                axis=axis, x_axis=self.x_axis, y_axis="expr", ctrl=self.reference, groups=self.groups_cond,
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
                # annot_pvals  = (df_pvals < self.pval_cutoff).replace({True: "*", False: ""})
                #mask = df_pvals < self.pval_cutoff
                #annot_pvals = df_pvals.mask(mask, "*").mask(~mask, "")
            else:
                index_data, index_pvals = df.index, self.df_pvals.index
                columns_data, columns_pvals = df.columns, self.df_pvals.columns
                if all(v in index_pvals for v in index_data) and all(v in columns_pvals for v in columns_data):
                    df_pvals = self.df_pvals.reindex(index=index_data, columns=columns_data)
                elif all(v in columns_pvals for v in index_data) and all(v in index_pvals for v in columns_data):
                    df_pvals = self.df_pvals.T.reindex(index=index_data, columns=columns_data)
                else:
                    raise InputError(
                        f"df_pvals does not have the same columns and index."
                        f"\nExpected index: {index_data}\nExpected columns: {columns_data}"
                    )
            mask = df_pvals < self.pval_cutoff
            annot_pvals = df_pvals.mask(mask, "*").mask(~mask, "")
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
            yticklabels=1,
            xticklabels=1,
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


class DrawNetwork:
    DEFAULT_TITLE_SIZE = 20
    DEFAULT_TITLE_FONTWEIGHT = "bold"

    DEFAULT_LEGEND_TITLE_FONTSIZE = 12
    DEFAULT_LEGEND_TITLE_FONTWEIGHT = "bold"
    DEFAULT_LEGEND_WIDTH = 1.5

    DEFAULT_LABEL_SIZE = 10
    DEFAULT_LABEL_FONTWEIGHT = "bold"

    SHAPES = [
        "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*",
        "h", "H", "+", "x", "X", "D"
    ]

    DEFAULT_NX_LAYOUT = {"k": 0.35, "iterations": 200}

    def __init__(
        self,
        # Data
        df: pd.DataFrame,
        term_col: str,
        padj_col: str,
        score_col: str,
        genes_col: str,
        annot_col: str | None = None,
        direction_col: str | None = None,

        # Figure parameters
        figsize: tuple = (6, 5),
        palette: str | dict = "tab30",
        title: str | None = None,
        title_fontproperties: Dict[Literal["size", "weight"], str | int] | None = None,
        ax: plt.Axes | None = None,

        # Legend parameters
        legend_title: str | None = None,
        legend_properties: Dict[Literal["size", "weight"], str | int] | None = None,
        legend_ncols: int = 1,
        legend_loc: Literal[
            "center left", "cemter right", "upper right", "upper left", "lower left", "lower right", "right", "lower center", "upper center", "center"] = 'center left',

        # Fx specific
        shapes: dict | None = None,
        similarity: Literal["kappa", "overlap", "jaccard"] = "overlap",
        k_neighbors: int = 5,
        edge_threshold: float = 0.3,
        cluster_algorithm: Literal["hierarchical", "louvain", "connected_components", "leiden"] = "hierarchical",
        cluster_method: str = "complete",
        cluster_t: float = 0.7,
        resolution: float = 1,
        cluster_criterion: str = "distance",
        min_cluster_size: int = 3,
        representative_method: Literal["pval", "degree", "combined"] = "pval",
        max_significant_terms: int = 10,
        nx_layout: Any = nx.spring_layout,
        nx_layout_kwargs: Dict | None = None,
        labels_fontproperties: Dict[Literal["size", "weight"], str | int] | None = None,

        # Customise
        edge_color: str = "gray",
        edge_alpha: float = 0.25,
        textwrap_width: int = 25,
    ):

        # Data section
        df = df.copy()  # Create a copy of the input
        df["node_id"] = df.reset_index(drop=True).index  # Each row is a node
        if annot_col is None:
            df["no_annot_provided"] = "same"
            annot_col = "no_annot_provided"

        self.df = df
        self.term_col = term_col
        self.padj_col = padj_col
        self.score_col = score_col
        self.annot_col = annot_col
        self.direction_col = direction_col
        self.gene_col = genes_col

        # Figure parameters
        self.figsize = figsize
        self.fig, self.gs = None, None
        self.width, self.height = figsize
        self.ax = ax

        self.title = title
        title_fontproperties = {} if title_fontproperties is None else title_fontproperties
        self.title_size = title_fontproperties.get("size", self.DEFAULT_TITLE_SIZE)
        self.title_fontweight = title_fontproperties.get("weight", self.DEFAULT_TITLE_FONTWEIGHT)

        self.legend_title = legend_title
        self.legends_width = self.DEFAULT_LEGEND_WIDTH

        legend_properties = {} if legend_properties is None else legend_properties
        self.legend_title = legend_title
        self.legend_ncols = legend_ncols
        self.legend_title_fontsize = legend_properties.get("size", self.DEFAULT_LEGEND_TITLE_FONTSIZE)
        self.legend_title_fontweight = legend_properties.get("weight", self.DEFAULT_LEGEND_TITLE_FONTWEIGHT)
        self.legend_fontsize = legend_properties.get("size", self.DEFAULT_LEGEND_TITLE_FONTSIZE - 2)
        self.legend_loc = legend_loc

        labels_fontproperties = {} if labels_fontproperties is None else labels_fontproperties
        self.label_size = labels_fontproperties.get("size", self.DEFAULT_LABEL_SIZE)
        self.label_fontweight = labels_fontproperties.get("weight", self.DEFAULT_LABEL_FONTWEIGHT)

        # Define the palette
        if isinstance(palette, str):
            color_list = get_hex_colormaps(palette)
            assert df[annot_col].nunique() < len(color_list), "There are more categories than colors in df[annot_col]"
            palette = dict(zip(df[annot_col].unique(), color_list))
        elif isinstance(palette, dict):
            missing = [k for k in  df[annot_col].unique() if k not in palette]
            assert len(missing) == 0, f"{missing} is missing in palette"
        else:
            raise InputError("Not a valid palette input")
        self.palette = palette

        # IO
        self.return_axis = {}

        # Fx Specific
        if shapes is None:
            if direction_col is not None:
                assert df[direction_col].nunique() < len(self.SHAPES), f"There cannot be more than {len(self.SHAPES)} directions"
                shapes = dict(zip(df[direction_col].unique(), self.SHAPES[:df[direction_col].nunique()]))
            else:
                self.direction_col = "direction"
                df["direction"] = "same"
                shapes = {"same": "o"}
        else:
            if direction_col is None:
                raise InputError("shapes is provided but direction_col is None")
            else:
                assert df[direction_col].nunique() == len(
                    shapes), f"There are {len(shapes)} and {direction_col} has {df[direction_col].nunique()} values"
        self.shapes = shapes

        self.representative_method = representative_method
        self.cluster_algorithm = cluster_algorithm
        self.cluster_method = cluster_method
        self.cluster_t = cluster_t
        self.cluster_criterion = cluster_criterion
        self.min_cluster_size = min_cluster_size
        self.resolution = resolution
        self.similarity = similarity
        self.k_neighbors = k_neighbors
        self.edge_threshold = edge_threshold

        # Get df_filt and S
        self._preprocess()

        # Show the top N more representatives terms
        self.representatives = self.choose_representatives()
        self.representatives = self.representatives.sort_values(self.padj_col, ascending=True).head(max_significant_terms)

        #self.representatives = (self.df.sort_values(padj_col).groupby("clusters").first())
        self.G = None
        self.nx_layout = nx_layout

        if nx_layout is nx.spring_layout:
            self.nx_layout_kwargs = nx_layout_kwargs if nx_layout_kwargs is not None else self.DEFAULT_NX_LAYOUT
        else:
            self.nx_layout_kwargs = nx_layout_kwargs if nx_layout_kwargs is not None else {}
        logger.info(f"There are {len(self.representatives)} representatives terms")

        # Customize
        self.edge_color = edge_color
        self.edge_alpha = edge_alpha
        self.textwrap = textwrap_width

    # Compute Similarity
    @staticmethod
    def kappa(a: NDArray, b: NDArray) -> NDArray:
        N = len(a)
        A = np.sum((a == 1) & (b == 1))
        B = np.sum((a == 1) & (b == 0))
        C = np.sum((a == 0) & (b == 1))
        D = np.sum((a == 0) & (b == 0))
        po = (A + D) / N
        p_yes = ((A + B) / N) * ((A + C) / N)
        p_no = ((C + D) / N) * ((B + D) / N)
        pe = p_yes + p_no
        if pe == 1:
            return 1
        return (po - pe) / (1 - pe)

    @staticmethod
    def overlap(a, b):
        A = np.sum((a == 1) & (b == 1))
        na = np.sum(a)
        nb = np.sum(b)
        if min(na, nb) == 0:
            return 0
        return  A / min(na, nb)

    @staticmethod
    def jaccard(a, b):
        inter = np.sum((a == 1) & (b==1))
        union = np.sum((a==1) | (b==1))
        if union == 0:
            return 0
        return inter / union

    def compute_similarity(self, a, b):
        if self.similarity == "kappa":
            return self.kappa(a, b)
        elif self.similarity == "overlap":
            return self.overlap(a, b)
        elif self.similarity == "jaccard":
            return self.jaccard(a, b)
        else:
            raise ValueError("Error computing similarity")


    # Clustering of the terms
    @staticmethod
    def louvain_clustering(G):
        communities = nx.community.louvain_communities(G, weight="weight", seed=0)
        clusters = np.zeros(len(G), dtype=int)
        for cid, community in enumerate(communities):
            for node in community:
                clusters[node] = cid
        return clusters

    @staticmethod
    def connected_component_clustering(G):
        clusters = np.zeros(len(G), dtype=int)
        for cid, component in enumerate(nx.connected_components(G)):
            for node in component:
                clusters[node] = cid
        return clusters

    def leiden_clustering(self, G):
        import igraph as ig
        import leidenalg
        edges = list(G.edges())
        weights = [G[u][v]["weight"] for u, v in edges]
        g = ig.Graph()
        g.add_vertices(len(G))
        g.add_edges(edges)
        partition = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition, weights=weights, resolution_parameter=self.resolution,
        )
        clusters = np.zeros(len(G), dtype=int)
        for cid, community in enumerate(partition):
            for node in community:
                clusters[node] = cid
        return clusters


    # Utils
    def build_similarity_graph(self, S):
        G = nx.Graph()
        G.add_nodes_from(range(len(S)))
        for i in range(len(S)):
            neighbours = np.argsort(S[i])[::-1]
            added = 0
            for j in neighbours:
                if i == j:
                    continue
                if S[i, j] < self.edge_threshold:
                    continue
                G.add_edge(i, j, weight=S[i, j])
                added += 1
                if added == self.k_neighbors:
                    break
        return G


    def choose_representatives(self):
        degree = dict(self.similarity_graph.degree(weight="weight"))
        df = self.df.copy()
        df["degree"] = df.node_id.map(degree)
        representatives = []

        for _, group in df.groupby("clusters"):
            if self.representative_method == "pval":
                idx = group[self.padj_col].idxmin()
            elif self.representative_method == "degree":
                idx = group.degree.idxmax()
            else:
                score = (-np.log10(group[self.padj_col]) + group.degree)
                idx = score.idxmax()

            try:
                selected = df.loc[idx,].to_frame()
            except AttributeError:
                selected = df.loc[idx,]

            if self.term_col not in selected.columns:
                selected = selected.T
            selected = selected.head(1)
            representatives.append(selected)
        return pd.concat(representatives)


    def remove_small_components(self):
        components = list(nx.connected_components(self.similarity_graph))

        keep = set()
        for comp in components:
            if len(comp) >= self.min_cluster_size:
                keep.update(comp)
        keep = sorted(keep)

        self.df = self.df[self.df["node_id"].isin(keep)].copy()
        self.similarity_graph = self.similarity_graph.subgraph(keep).copy()
        self.S = self.S[np.ix_(keep, keep)]

        old_to_new = {old: new for new, old in enumerate(keep)}
        self.df["node_id"] = self.df["node_id"].map(old_to_new)
        self.similarity_graph = nx.relabel_nodes(self.similarity_graph, old_to_new)

    # Old Functions
    def remove_isolated_nodes(self):
        isolated = list(nx.isolates(self.similarity_graph))
        if not isolated:
            return
        keep = sorted(set(self.similarity_graph.nodes()) - set(isolated))
        self.df = self.df[self.df["node_id"].isin(keep)].copy()
        self.similarity_graph = self.similarity_graph.subgraph(keep).copy()
        self.S = self.S[np.ix_(keep, keep)]
        old_to_new = {old: new for new, old in enumerate(keep)}
        self.df["node_id"] = self.df["node_id"].map(old_to_new)
        self.similarity_graph = nx.relabel_nodes(self.similarity_graph, old_to_new)

    def merge_small_clusters(self):
        cluster_sizes = self.df["clusters"].value_counts()
        small_clusters = cluster_sizes[cluster_sizes < self.min_cluster_size].index
        for cluster in small_clusters:
            nodes = self.df.loc[self.df["clusters"] == cluster, "node_id"]
            weights = {}
            for node in nodes:
                for neighbour in self.similarity_graph.neighbors(node):
                    neighbour_cluster = self.df.loc[self.df.node_id == neighbour, "clusters"].iloc[0]
                    if neighbour_cluster == cluster:
                        continue
                    w = self.similarity_graph[node][neighbour]["weight"]
                    weights[neighbour_cluster] = (weights.get(neighbour_cluster, 0) + w)
            if len(weights):
                best = max(weights, key=weights.get)
                self.df.loc[self.df["clusters"] == cluster, "clusters"] = best

    @staticmethod
    def louvain_clustering_old(s: NDArray, threshold: float = 0.3, resolution: float = 1) -> NDArray:
        G = nx.Graph()
        n = len(s)
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if s[i, j] >= threshold:
                    G.add_edge(i, j, weight=float(s[i, j]))

        communities = nx.community.louvain_communities(G, weight="weight", resolution=resolution, seed=0)
        clusters = np.zeros(n, dtype=int)
        for cid, community in enumerate(communities, start=1):
            for node in community:
                clusters[node] = cid
        return  clusters

    @staticmethod
    def connected_component_clustering_old(s: NDArray, threshold: float = 0.3) -> NDArray:
        G = nx.Graph()
        n = len(s)
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if s[i, j] >= threshold:
                    G.add_edge(i, j)
        clusters = np.zeros(n, dtype=int)
        for cid, component in enumerate(nx.connected_components(G), start=1):
            for node in component:
                clusters[node] = cid
        return clusters

    # Main Fxs
    def _preprocess(self) -> None:
        term2genes = {}
        for _, row in self.df.iterrows():
            genes = row[self.gene_col]
            if isinstance(genes, str):
                if ";" in genes:
                    genes = genes.split(";")
                elif "," in genes:
                    genes = genes.split(",")
                else:
                    genes = genes.split()
                term2genes[row["node_id"]] = set(g.strip() for g in genes if g.strip())
            else:
                raise InputError(f"{self.gene_col} is not a string column")

        # Binary Matrix
        all_genes = sorted(set.union(*term2genes.values()))
        genes_idx = {g: idx for idx, g in enumerate(all_genes)}
        x = np.zeros((len(term2genes), len(all_genes)), dtype=np.uint8)
        for term, genes in term2genes.items():
            idx = [genes_idx[g] for g in genes]
            x[term, idx] = 1

        # Cluster base on similarity
        n = x.shape[0]
        s = np.eye(n)
        for i, j in combinations(range(n), 2):
            sim = self.compute_similarity(x[i], x[j])
            s[i, j], s[j, i] = sim, sim  # Similarity

        #distance = 1 - s

        # Clustering
        #if self.cluster_algorithm == "hierarchical":
        #    z = linkage(squareform(distance), method=self.cluster_method)
        #    clusters = fcluster(z, t=self.cluster_t, criterion=self.cluster_criterion)
        #elif self.cluster_algorithm == "louvain":
        #    clusters = self.louvain_clustering(s, threshold=self.cluster_t, resolution=self.resolution)
        #elif self.cluster_algorithm == "connected_components":
        #    clusters = self.connected_component_clustering(s, threshold=self.cluster_t)
        #else:
        #    raise InputError(f"{self.cluster_algorithm} is not a valid cluster_algorithm value")
        #self.df["cluster"] = clusters

        # Remove singleton
        #cluster_sizes = self.df["cluster"].value_counts()
        #valid_clusters = cluster_sizes[cluster_sizes >= self.min_cluster_size].index
        #self.df = self.df[self.df["cluster"].isin(valid_clusters)].copy()

        # Re-index nodes
        #old_to_new = dict(zip(self.df["node_id"], range(len(self.df))))
        #keep = self.df["node_id"].values
        #s = s[np.ix_(keep, keep)]
        #self.df["node_id"] = self.df["node_id"].map(old_to_new)

        self.S = s
        self.similarity_graph = self.build_similarity_graph(s)
        # self.remove_isolated_nodes()
        self.remove_small_components()

        # Clustering
        if self.cluster_algorithm == "hierarchical":
            distance = 1 - self.S
            z = linkage(squareform(distance), method=self.cluster_method)
            clusters = fcluster(z, t = self.cluster_t, criterion=self.cluster_criterion)
        elif self.cluster_algorithm == "louvain":
            clusters = self.louvain_clustering(self.similarity_graph)
        elif self.cluster_algorithm == "leiden":
            clusters = self.leiden_clustering(self.similarity_graph)
        elif self.cluster_algorithm == "connected_components":
            clusters = self.connected_component_clustering(self.similarity_graph)
        else:
            raise InputError(f"{self.cluster_algorithm} is not a valid clustering algorithm")

        self.df["clusters"] = clusters

        # self.merge_small_clusters()

        #keep = self.df["node_id"].values
        #self.S = self.S[np.ix_(keep, keep)]
        #self.similarity_graph = self.similarity_graph.subgraph(keep).copy()


        #old_to_new = {old: new for new, old in enumerate(keep)}
        #self.df["node_id"] = self.df["node_id"].map(old_to_new)
        #self.similarity_graph = nx.relabel_nodes(self.similarity_graph, old_to_new)
        return None

    def make_figure(self, nrows: int = 1, ncols: int = 2) -> None:
        self.fig, self.gs = make_grid_spec(
            self.ax or (self.width, self.height), nrows=nrows, ncols=ncols, wspace=0.7 / self.width,
            width_ratios=(
                [self.width - self.legends_width, self.legends_width] if ncols == 2 else [
                    self.width - self.legends_width]
            )
        )
        return None

    def draw_graph(self) -> Dict | None:

        # self.G = nx.Graph()

        self.G = self.similarity_graph.copy()

        # Add Nodes
        for _, row in self.df.iterrows():
            self.G.add_node(
                row["node_id"],
                term=row[self.term_col],
                cluster=row["clusters"],
                score=row[self.score_col],
                annot=row[self.annot_col],
                direction=row[self.direction_col]
            )

        # Add Edges
        #n = len(self.df)
        #for i in range(n):
        #    for j in range(i + 1, n):
        #        if self.S[i, j] >= 0.3:
        #           self.G.add_edge(i, j, weight=self.S[i, j])

        # Data
        scores = np.array([self.G.nodes[n]["score"] for n in self.G.nodes()])
        sizes = 80 + 600 * (scores - scores.min()) / (scores.max() - scores.min())
        # clusters = np.array([self.G.nodes[n]["cluster"] for n in self.G.nodes()])
        weights = [2 * self.G[u][v]["weight"] for u, v in self.G.edges()]
        pos = self.nx_layout(self.G, weight="weight", seed=0, **self.nx_layout_kwargs)

        # Make the plot
        if self.annot_col == "no_annot_provided":
            self.make_figure(nrows=1, ncols=1)
        else:
            self.make_figure(nrows=1, ncols=2)
        main_axis = self.fig.add_subplot(self.gs[0])

        # Plot Edges
        nx.draw_networkx_edges(self.G, pos, alpha=self.edge_alpha, width=weights, edge_color=self.edge_color,
                               ax=main_axis)

        # Plot Nodes
        for direction, marker in self.shapes.items():
            nodes = [n for n in self.G.nodes() if self.G.nodes[n]["direction"] == direction]
            colors = [self.palette[self.G.nodes[n]["annot"]] for n in nodes]
            node_sizes = [sizes[list(self.G.nodes()).index(n)] for n in nodes]
            nx.draw_networkx_nodes(
                self.G, pos, nodelist=nodes, node_color=colors, node_size=node_sizes,
                node_shape=marker, edgecolors="black", linewidths=0.5, ax=main_axis
            )

        # Labels from representatives
        labels = {
            row["node_id"]: textwrap.fill(row[self.term_col], width=self.textwrap)
            for _, row in self.representatives.iterrows()
        }
        texts = nx.draw_networkx_labels(
            self.G, pos, labels, font_size=self.label_size, font_weight=self.label_fontweight, verticalalignment="top",
            ax=main_axis
        )
        texts = list(texts.values())
        # Adjust labels to not overlap text or nodes
        x, y = [pos[n][0] for n in self.G.nodes()], [pos[n][1] for n in self.G.nodes()]

        adjust_text(
            texts, ax=main_axis, x=x, y=y, arrowprops=dict(arrowstyle="-", linestyle="--", color="k", lw=1.2, alpha=1),
            expand_text=(1.2, 1.5), expand_points=(1.5, 1.5), force_text=(0.5, 0.8), force_points=(0.2, 0.5),
        )

        main_axis.set_title(self.title, fontsize=self.title_size, fontweight=self.title_fontweight)
        # main_axis.axis("off")
        main_axis.spines[["top", "bottom", "left", "right"]].set_visible(False)
        main_axis.set(xticks=[], yticks=[], yticklabels=[], xticklabels=[])

        # Add Legend
        if self.annot_col != "no_annot_provided":
            legend_axis = self.fig.add_subplot(self.gs[1])

            legend_elements = []
            for b in self.palette:
                legend_elements.append(
                    mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor=self.palette[b],
                           markersize=10, label=b)
                )
            if len(self.shapes) > 1:
                for name, marker in self.shapes.items():
                    legend_elements.append(
                        mlines.Line2D(
                            [0], [0], marker=marker, linestyle="", color="black",
                            markerfacecolor="white", markersize=10, label=name)
                    )

            legend_axis.legend(handles=legend_elements, frameon=False)
            sns.move_legend(legend_axis, loc=self.legend_loc, ncols=self.legend_ncols, title=self.legend_title,
                            title_fontproperties={"size": self.legend_title_fontsize,
                                                  "weight": self.legend_title_fontweight},
                            fontsize=self.legend_fontsize, frameon=False)
            # legend_axis.axis("off")
            legend_axis.spines[["top", "bottom", "left", "right"]].set_visible(False)
            legend_axis.set(xticks=[], yticks=[], yticklabels=[], xticklabels=[])
            self.return_axis["legend_ax"] = legend_axis

        self.return_axis["main_ax"] = main_axis
        return



class SeabornDataframes:
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
        df: pd.DataFrame,
        x_axis: str,
        feature: str,
        batch_key: str | None = None,
        hue: str | None = None,

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
        # Data Section
        self.df = df

        self.x_axis = x_axis
        self.hue = hue
        self.batch_key = batch_key

        self.feature = iterase_input(feature)

        # Order for the Xticks
        self.xticks_order = xticks_order if xticks_order is not None else self._get_categories(x_axis)
        self.hue_order = hue_order if hue_order is not None else self._get_categories(hue)

        # Figure parameters
        self.figsize = figsize
        self.fig, self.gs = None, None
        self.width, self.height = figsize
        self.ax = ax
        self.legends_width = self.DEFAULT_LEGEND_WIDTH

        self.cmap = cmap

        colors_dict = None
        if hue is not None:
            if isinstance(self.cmap, str):
                list_colors = get_hex_colormaps(self.cmap)
                if len(list_colors) < len(iterase_input(self.hue_order)):
                    list_colors *= 5
                colors_dict = dict(zip(iterase_input(self.hue_order), list_colors, strict=False))
            elif isinstance(self.cmap, dict):
                colors_dict = self.cmap
            else:
                raise InputError("Currently palette only supports a string or dictionary")

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

    def _get_categories(self, column: str | None) -> list | None:
        if column is None:
            return None
        else:
            return (
                list(self.df[column].cat.categories) if self.df[column].dtype.name == "category"
                else list(self.df[column].unique())
            )

    def barplot(self):
        ...
    def boxplot(self):
        ...
    def violinplot(self):
        ...
    def lineplot(self):
        ...
    def heatmap(self):
        ...



