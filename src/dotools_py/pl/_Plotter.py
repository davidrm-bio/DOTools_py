import anndata as ad
#import matplotlib.pyplot as plt
import numpy as np
from typing import Literal

import pandas as pd
#from scipy.cluster.hierarchy import dendrogram, linkage
#from pathlib import Path

from dotools_py.utils import iterase_input, check_missing, sanitize_anndata
from dotools_py.get._generic import expr as get_expr
from dotools_py import logger
#import matplotlib.pyplot as plt
#from scipy.stats import zscore

#import dotools_py as do
from dotools_py.tl._get_stats import rank_genes_groups
from scanpy.get.get import rank_genes_groups_df


# Base class for plots like heatmap or dotplot

class MatrixDataGenerator:
    DEFAULT_ESTIMATOR = "mean"
    DEFAULT_EXPR_CUTOFF = 0

    def __init__(self,
                 adata: ad.AnnData,
                 x_axis: str,
                 features: str | list,
                 y_axis: str = None,
                 logcounts: bool = True,
                 layer: str = None,
                 estimator: str = None,  # TODO Allow Fx
                 mean_express_only: bool = False,
                 expression_cutoff: float = None,
                 z_score: Literal["x_axis", "y_axis"] = None,
                 minmax: Literal["x_axis", "y_axis"] = None,
                 test: Literal["wilcoxon", "t-test"] = "wilcoxon",
                 correction_method: Literal["benjamini-hochberg", "bonferroni"] = "benjamini-hochberg",
                 add_stats: Literal["x_axis", "y_axis"] = None,
                 df_pvals: pd.DataFrame = None,
                 pval_cutoff: float = 0.05,
                 lfc_cutoff: float = 0.25,
                 ):

        sanitize_anndata(adata)  # Convert string columns to categorical

        self.adata = adata
        self.x_axis = iterase_input(x_axis)  # Convert to list
        self.y_axis = iterase_input(y_axis)
        self.group_by = self.x_axis + self.y_axis
        self.features = iterase_input(features)  # Convert to list
        self.logcounts = logcounts  # Whether input is logcounts
        self.layer = layer
        self.estimator = estimator if estimator is not None else self.DEFAULT_ESTIMATOR
        self.mean_express_only = mean_express_only
        self.expression_cutoff = expression_cutoff if expression_cutoff is not None else self.DEFAULT_EXPR_CUTOFF
        self.z_score = z_score
        self.minmax = minmax
        self.test = test
        self.correction_method = correction_method
        self.add_stats = add_stats
        self.df_pvals = df_pvals
        self.pval_cutoff = pval_cutoff
        self.lfc_cutoff = lfc_cutoff

        assert not (
            self.z_score is not None and self.minmax is not None), "Specify either Z-score or MinMax Normalisation, not both"

        # Cheks
        check_missing(self.adata, groups=self.x_axis + self.y_axis, features=self.features)

    def get_expr_df(self):
        """Calculate the mean expression for the features per group.

        :return: The df_mean and df_expr attribute are initialise containing a dataframe in long format
                with the mean expression and expression.
        """
        if all(item in list(self.adata.var_names) for item in self.features):
            df_expr = get_expr(
                self.adata, features=self.features, groups=self.group_by, layer=self.layer, out_format="wide"
            )
            df_expr.set_index(self.group_by, inplace=True)  # Set Index

            # If mean_express_only is True, the mean is only over the cells with expr > expression_cutoff
            obs_bool = df_expr > self.expression_cutoff if self.mean_express_only else df_expr > float("-inf")

            if self.estimator != "mean":
                logger.warn(f"Estimator set to {self.estimator}, but mean will be used")

            # Case 1: Input are features in .var_names and are log-normalise
            if self.logcounts:
                df_expr = np.expm1(df_expr)
                df = np.log1p(
                    df_expr.mask(~obs_bool).groupby(level=self.group_by, observed=True).mean().fillna(0)
                )
            else:
                # Case 2: Input are features in .var_names and are not log-normalise
                df = df_expr.mask(~obs_bool).groupby(level=self.group_by, observed=True).mean().fillna(0)

        elif all(item in iterase_input(self.adata.obs.columns) for item in self.features):
            # Case 3: Input are features in .obs
            df_expr = self.adata.obs[self.features + self.group_by]
            df_expr.set_index(self.group_by, inplace=True)  # Set Index
            df = self.adata.obs[self.group_by + self.features].groupby(self.group_by).agg(self.estimator)
        else:
            raise Exception("Provide features either in var_names or obs.columns")

        self.df_mean = df.reset_index()
        self.df_expr = df_expr
        return None

    def get_pct_df(self):
        """Calculate the proportion of cells expressing a feature based on a cutoff.

        :return: The df_pct attribute is initialise containing a dataframe in long format with the percentage
                of cells expressing the features
        """
        try:
            obs_bool = self.df_expr > self.expression_cutoff
        except AttributeError as e:
            self.get_expr_df()
            obs_bool = self.df_expr > self.expression_cutoff

        df_pct = (
            obs_bool.groupby(level=self.group_by, observed=True).sum()
            / obs_bool.groupby(level=self.group_by, observed=True).count()
        )
        self.df_pct = df_pct
        return None

    def zscore_transform(self):
        """Perform Z-score transformation on the data.

        :return: The df_zscore attribute is initialise.
        """
        if self.z_score is not None:
            df_mean = self.df_mean.set_index(self.group_by)

            if len(self.y_axis) == 0:
                if self.z_score == "x_axis":
                    df_mean = (
                        df_mean.T
                        .apply(lambda x: (x - x.mean(axis=0)) / x.std(axis=0, ddof=0))  # Scale over the genes
                        .fillna(0).T
                    )
                else:
                    # If y_axis is written and y_axis is not define, then is done for each gene
                    df_mean = (
                        df_mean
                        .apply(lambda x: (x - x.mean(axis=0)) / x.std(axis=0, ddof=0))  # Scale over the genes
                        .fillna(0)
                    )
            else:
                z_score = self.y_axis if self.z_score == "x_axis" else self.x_axis
                df_mean = (
                    df_mean.groupby(level=z_score)
                    .apply(lambda x: (x - x.mean(axis=0)) / x.std(axis=0,
                                                                  ddof=0))  # Scale for each gene over the alternative axis
                    .fillna(0)
                )

                to_drop = [
                    i for i in range(len(df_mean.index.levels))
                    if i > 0 and df_mean.index.get_level_values(i).equals(df_mean.index.get_level_values(i - 1))
                ]
                if to_drop:
                    df_mean = df_mean.droplevel(to_drop)

            df_mean = df_mean.reset_index()
            self.df_zscore = df_mean
        else:
            logger.warn("Specify axis to apply Z-score")
        return None

    def minmax_transform(self):
        """Perform MinMax normalisation on the data.

        :return: The df_minmax attribute is initialise
        """
        if self.minmax is not None:
            df_mean = self.df_mean.set_index(self.group_by)

            if len(self.y_axis) == 0:
                if self.minmax == "x_axis":
                    # Scaling by x_axis would mean that we Scale for each x_axis group
                    df_mean = (
                        df_mean.T
                        .apply(lambda x: (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0)))
                        .fillna(0).T
                    )  # group x gene
                else:
                    # If y_axis is written and y_axis is not define, then is done for each gene
                    df_mean = (
                        df_mean
                        .apply(lambda x: (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0)))
                        .fillna(0)
                    )  # group x gene

            else:
                minmax = self.y_axis if self.minmax == "x_axis" else self.x_axis
                df_mean = (
                    df_mean.groupby(level=minmax)
                    .apply(lambda x: (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0)))
                    .fillna(0)
                )  # group/group2 x gene

            df_mean = df_mean.reset_index()
            self.df_minmax = df_mean
        else:
            logger.warn("Specify axis to perform minmax normalisation")
        return None

    def test_significance(self):
        if self.add_stats == 'y_axis':
            assert self.y_axis is not None, "Testing y_axis but argument is None"

        group_by = self.x_axis[0] if self.add_stats == "x_axis" else self.y_axis[0]

        if self.df_pvals is None:
            if len(self.y_axis) == 0:
                # We only group by the x_axis
                if all(item in list(self.adata.var_names) for item in self.features):
                    # All features are in adata.var_names
                    try:
                        rank_genes_groups(
                            self.adata, groupby=group_by, method=self.test, tie_correct=True,
                            corr_method=self.correction_method, layer=self.layer
                        )
                        table = rank_genes_groups_df(self.adata, group=None)
                        table = table[table["names"].isin(self.features)]
                        if len(table) == 0:
                            logger.warn("No significant group")
                    except ValueError as e:
                        logger.warn(f"Error testing, {e}")
                        table = pd.DataFrame([],
                                             columns=['group', 'names', 'scores', 'logfoldchanges', 'pvals',
                                                      'pvals_adj', 'pct_nz_group', 'pct_nz_reference']
                                             )
                elif all(item in list(self.adata.obs.columns) for item in self.features):
                    # All features are in adata.obs.columns
                    table = pd.DataFrame([], columns=['group', 'names', 'scores', 'logfoldchanges', 'pvals',
                                                      'pvals_adj', 'pct_nz_group', 'pct_nz_reference'])
                    logger.warn("Testing for .obs variables not implemented")
                else:
                    raise NotImplementedError("Provide either features in var_names or in obs, not both")
            else:
                alternative = self.x_axis[0] if self.add_stats == "y_axis" else self.y_axis[0]
                # We group by the x_axis and y_axis
                if all(item in list(self.adata.var_names) for item in self.features):
                    # All features are in adata.var_names
                    table = pd.DataFrame([])
                    for alt in self.adata.obs[alternative].unique():
                        sdata = self.adata[self.adata.obs[alternative] == alt].copy()
                        try:
                            rank_genes_groups(
                                sdata, groupby=group_by, method=self.test, tie_correct=True,
                                corr_method=self.correction_method, layer=self.layer
                            )
                            stable = rank_genes_groups_df(sdata, group=None)
                        except ValueError as e:
                            logger.warn(f"Error while testing: {e}")
                            stable = pd.DataFrame([],
                                                  columns=['group', 'names', 'scores', 'logfoldchanges', 'pvals',
                                                           'pvals_adj', 'pct_nz_group', 'pct_nz_reference', "group2"]
                                                  )
                        stable = stable[stable["names"].isin(self.features)]
                        stable['group2'] = alt
                        table = pd.concat([table, stable])
                    if len(table) == 0:
                        logger.warn('No Significant group')
                elif all(item in list(self.adata.obs.columns) for item in self.features):
                    # All features are in adata.obs.columns
                    table = pd.DataFrame([], columns=['group', 'names', 'scores', 'logfoldchanges', 'pvals',
                                                      'pvals_adj', 'pct_nz_group', 'pct_nz_reference', "group2"])
                    logger.warn("Testing for .obs variables not implemented")
                else:
                    raise NotImplementedError("Provide either features in var_names or in obs, not both")

            table["group"] = table["group"].str.replace("-", "_")  # TODO remove correction in get_expr()
            if len(self.y_axis) != 0:
                table["group2"] = table["group2"].str.replace("-", "_")

            self.df_pvals = table  # retain logfoldchanges and pvals
            self.pvals_groupby = group_by
        else:
            pass

        return None


# class MatrixPlotter:
#     DEFAULT_CMAP = "Reds"
#     DEFAULT_TITLE = "",
#     DEFAULT_XTICK_ROTATION = 90
#     DEFAULT_FIGSIZE = (5, 4)
#
#     def __init__(self,
#                  adata: ad.AnnData,
#                  x_axis: str,
#                  features: str | list,
#                  kind: Literal["dotplot", "heatmap"],
#                  z_score: bool = False,
#                  min_max: bool = False,
#                  cmap: str = None,
#                  vmax: float | None = None,
#                  vmin: float | None = None,
#                  vcenter: float | None = None,
#                  title: str | None = None,
#                  size_legend_title: str = "Fraction of cells\nin group (%)",
#                  color_legend_title: str = "LogMean(nUMI)\nin group",
#                  x_ticks_rotation: int = None,
#                  swap_axis: bool = True,
#                  axis: plt.Axes = None,
#                  figsize: tuple = None,
#                  path: str | Path = None,
#                  filename: str = "plot.pdf",
#                  show: bool = True,
#                  ):
#
#         self.adata = adata
#         self.x_axis = x_axis
#         self.y_axis = features
#         self.kind = kind
#         self.cmap = self.DEFAULT_CMAP if cmap is None else cmap
#         self.vmax = vmax
#         self.vmin = vmin
#         self.vcenter = vcenter
#         self.main_title = title
#         self.size_legend_title = size_legend_title
#         self.color_legend_title = color_legend_title
#         self.x_ticks_rotation = x_ticks_rotation
#         self.axis = axis
#         self.figsize = self.DEFAULT_FIGSIZE if figsize is None else figsize
#         self.width, self.height = self.figsize
#
#         # Saving Parameters
#         self.path = path
#         self.filename = filename
#         self.show = show
#         self.swap_axis = swap_axis
#
#     def make_figure(self):
#         legends_width_spacer = 0.7 / self.width
#         mainplot_width = self.width - (1.5 + 0)
#
#
#
#     def set_layout(self):
#         if self.z_score is not None:
#             if self.cmap == "Reds":
#                 logger.warn(
#                     "Z-score set to True, but the cmap is Reds, setting to RdBu_r"
#                 )  # Make sure to use divergent colormap
#             self.color_legend_title = "Z-score"
#             self.cmap = "RdBu_r"







# ##########################################
# def heatmap(
#     adata: ad.AnnData,
#     x_axis: str | list,
#     features: str | list,
#     groups_order: list = None,
#     z_score: Literal["x_axis", "y_axis"] = None,  # x_axis is the group_by
#     min_max: Literal["x_axis", "y_axis"] = None,
#     path: str = None,
#     filename: str = "Heatmap.svg",
#     layer: str = None,
#     swap_axes: bool = True,
#     cmap: str = "Reds",
#     title: str = None,
#     title_fontprop: dict = None,
#     clustering_method: str = "complete",
#     clustering_metric: str = "euclidean",
#     cluster_x_axis: bool = False,
#     cluster_y_axis: bool = False,
#     axs: plt.Axes  = None,
#     figsize: tuple = (5, 6),
#     linewidth: float = 0.1,
#     ticks_fontdict: dict = None,
#     xticks_rotation: int = None,
#     yticks_rotation: int = None,
#     vmin: float = 0.0,
#     vcenter: float = None,
#     vmax: float = None,
#     legend_title: str = "LogMean(nUMI)\nin group",
#     add_stats: bool = False,
#     df_pvals: pd.DataFrame = None,
#     stats_x_size: float = None,
#     square_x_size: dict = None,
#     test: Literal["wilcoxon", "t-test"] = "wilcoxon",
#     correction_method: Literal["benjamini-hochberg", "bonferroni"] = "benjamini-hochberg",
#     pval_cutoff: float = 0.05,
#     log2fc_cutoff: float = 0.0,
#     square: bool = True,
#     show: bool = True,
#     logcounts: bool = True,
#     **kargs,
# ) -> dict | None:
#     """Heatmap of the mean expression of genes across a groups.
#
#     Generate a heatmap of showing the average nUMI for a set of genes in different groups. Differential gene
#     expression analysis between the different groups can be performed.
#
#     :param adata: annotated data matrix.
#     :param group_by: obs column name with categorical values.
#     :param features: continuous value in var_names or obs.
#     :param groups_order: order for the categories in group_by
#     :param z_score: apply z-score transformation.
#     :param path: path to save the plot
#     :param filename: name of the file.
#     :param layer: layer to use.
#     :param swap_axes: whether to swap the axes or not.
#     :param cmap: colormap.
#     :param title: title for the main plot.
#     :param title_fontprop: font properties for the title (e.g., 'weight' and 'size').
#     :param clustering_method: clustering method to use when hierarchically clustering the x and y-axis.
#     :param clustering_metric: metric to use when hierarchically clustering the x and y-axis.
#     :param cluster_x_axis: hierarchically clustering the x-axis.
#     :param cluster_y_axis: hierarchically clustering the y-axis.
#     :param axs: matplotlib axis.
#     :param figsize: figure size.
#     :param linewidth: linewidth for the border of cells.
#     :param ticks_fontdict: font properties for the x and y ticks (e.g.,  'weight' and 'size').
#     :param xticks_rotation: rotation of the x-ticks.
#     :param yticks_rotation: rotations of the y-ticks.
#     :param vmin: minimum value.
#     :param vcenter: center value.
#     :param vmax: maximum value.
#     :param legend_title: title for the colorbar.
#     :param add_stats: add statistical annotation. Will add a square with an '*' in the center if the expression is significantly different in a group with respect to the others.
#     :param df_pvals: dataframe with the pvals. Should be gene x group or group x gene in case of swap_axes is False.
#     :param stats_x_size: scaling factor to control the size of the asterisk.
#     :param square_x_size: size and thickness of the square.
#     :param test: test to use for test for significance.
#     :param correction_method: multiple correction method to use.
#     :param pval_cutoff: cutoff for the p-value.
#     :param log2fc_cutoff: minimum cutoff for the log2FC.
#     :param square: whether to make the cell square or not.
#     :param show: if set to false return a dictionary with the axis.
#     :param logcounts: whether the input is logcounts or not.
#     :param kargs: additional arguments pass to `sns.heatmap() <https://seaborn.pydata.org/generated/seaborn.heatmap.html>`_.
#     :return: Depending on ``show``, returns the plot if set to `True` or a dictionary with the axes.
#
#     Example
#     -------
#
#     .. plot::
#         :context: close-figs
#
#         import dotools_py as do
#         adata = do.dt.example_10x_processed()
#         do.pl.heatmap(adata, 'annotation', ['CD4', 'CD79A'], add_stats=True)
#
#     """
#
#
#     x_axis = "annotation"
#     features = ["CD4", "CD79A"]
#     logcounts=True
#     layer = None
#     z_score = None
#     min_max = None
#     test = "wilcoxon"
#     correction_method = "benjamini-hochberg"
#     add_stats = True
#     pval_cutoff = 0.05
#     log2fc_cutoff = 0.25
#     x_groups_order = None
#
#     sanitize_anndata(adata)
#
#     data = MatrixDataGenerator(adata=adata,
#                                x_axis=x_axis,
#                                features=features,
#                                y_axis=None,  # TODO Allow
#                                logcounts=logcounts,
#                                layer=layer,
#                                estimator=None,  # Always Mean
#                                mean_express_only=False,
#                                expression_cutoff=None,
#                                z_score=z_score,
#                                minmax=min_max,
#                                test=test,
#                                correction_method=correction_method,
#                                add_stats="x_axis" if add_stats else None,
#                                df_pvals=None,
#                                pval_cutoff=pval_cutoff,
#                                lfc_cutoff=log2fc_cutoff,
#                                )
#
#     data.get_expr_df()  # Compute mean expression
#
#     if z_score is not None:
#         data.zscore_transform()
#         df_plot = data.df_zscore.copy()
#     elif min_max is not None:
#         data.minmax_transform()
#         df_plot = data.df_minmax.copy()
#     else:
#         df_plot = data.df_mean.copy()  # wide format
#
#     df_plot = (df_plot
#                .melt(id_vars=x_axis, var_name="genes", value_name="expr")
#                .pivot(index=x_axis, columns="genes", values="expr"))  # Convert to matrix format x_axis x features
#
#     # Get the Dataframe with pvals if we want significance
#     if add_stats:
#         data.test_significance()
#         df_pvals = data.df_pvals.copy()
#         df_pvals = df_pvals[["group", "names", "pvals_adj"]].pivot(
#             index="group", columns="names", values="pvals_adj"
#         ).fillna(0)
#     else:
#         df_pvals = pd.DataFrame(np.ones(df_plot.shape), index=df_plot.index, columns=df_plot.columns)
#
#
#     # Hierarchical clustering --> Set the order for the X and Y Axis
#     index_order = x_groups_order if x_groups_order is not None else list(adata.obs[x_axis].cat.categories)
#     new_index = (
#         df_plot.index[
#             dendrogram(linkage(df_plot.values, method=clustering_method, metric=clustering_metric), no_plot=True)["leaves"]
#         ]
#         if cluster_x_axis
#         else index_order  # Order from the object or user provided
#     )
#
#     new_columns = (
#         df_plot.columns[
#             dendrogram(linkage(df_plot.T.values, method=clustering_method, metric=clustering_metric), no_plot=True)["leaves"]
#         ]
#         if cluster_y_axis
#         else features # Keep the order from the input
#     )
#
#     df_plot = df_plot.reindex(index=new_index, columns=new_columns)
#     df_pvals = df_pvals.reindex(index=new_index, columns=new_columns)
#
#     if swap_axes:
#         df_plot = df_plot.T
#         df_pvals = df_pvals.T
#
#     annot_pvals = df_pvals.applymap(lambda x: "*" if x < pval_cutoff else "")
#
#
#     # Initialise Plotter Class
#
#
#
#
#
#     if cmap == "Reds":
#         logger.warn(
#             "Z-score set to True, but the cmap is Reds, setting to RdBu_r"
#         )  # Make sure to use divergent colormap
#         cmap = "RdBu_r"
#     if legend_title == "LogMean(nUMI)\nin group":
#         legend_title = "Z-score"
#     #vmin, vcenter, vmax = round(df.min().min() * 20) / 20, 0.0, None
#
