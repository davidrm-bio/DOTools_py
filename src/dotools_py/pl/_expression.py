import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from dotools_py import logger
from dotools_py.utils import convert_path, sanitize_anndata
from typing import Union
from dotools_py.tl import get_expr, mean_expr
from dotools_py.pl._StatsPlotter import TestData, StatsPlotter


def barplot(adata: ad.AnnData,
            feature: str,
            x_axis: str,
            batch_key: str = 'batch',
            layer: str = None,
            figsize: tuple = (5, 6),
            palette: Union[str, list] = 'tab10',
            capsize: float = 0.1,
            xtick_rotation: int = None,
            ctrl_cond: str = None,
            groups_cond: Union[str, list] = None,
            groups_pvals: list = None,
            title: str = None,
            path: str = None, filename: str = None,
            title_fontproperties: dict = {'weight': 'bold', 'size': 20},
            show: bool = True, marker_size: int = 5,
            ax: plt.Axes = None,
            logcounts: bool = True,
            estimator: Union[str, None] = 'LogMean',
            test: str = 'wilcoxon',
            corr_method: str = 'benjamini-hochberg',
            txt_size: int = 13,
            txt: str = 'p = ',
            ylabel='LogMean(nUMI)',
            **kwargs,
            ):
    """Barplot with stats.

    Show the average expression of `var_names` or a continuous value in `obs` along different categorical values
    and test for significance. The mean pseudo-bulk expression per sample will be plotted as dots.

    :param adata: annotated data matrix
    :param feature: feature in `var_name` or `obs`.
    :param x_axis: categorical `obs` column to groupby.
    :param batch_key: `obs` column with batch information.
    :param layer: layer in the AnnData to use.
    :param figsize:  figure size.
    :param palette: dictionary or palette to use.
    :param capsize: width of the 'caps' on error bars, relative to bar spacing.
    :param xtick_rotation: rotation of the xticks.
    :param ctrl_cond: name of the ctrl condition in the xticks.
    :param groups_cond: list of the name of the groups to test in the xticks.
    :param groups_pvals: if provided, these values will be plotted. If not set, provide a list of the groups in the xticks
                         to test.
    :param title: title of the plot.
    :param path: path to save the figure.
    :param filename: name of the file.
    :param title_fontproperties: properties of the title text.
    :param show: if set to False, return the axis.
    :param marker_size: size of the markers showing the pseudo-bulk mean expression.
    :param ax: matplotlib axis.
    :param logcounts: if set to True, assume input is log1p transformed
    :param estimator: estimator to calculate the mean expression. If set to LogMean assume log1p.
    :param test: name of the method to test for significance. Available: ['wilcoxon', 't-test', 'kruskal', 'anova', 'logreg', 't-test_overestim_var'].
    :param corr_method: correction method for multiple testing. Available: ['benjamini-hochberg', 'bonferroni'].
    :param txt_size: size of the text indicating significance.
    :param txt: text for indicating significance. If not set, only the p-value is shown.
    :param ylabel: Y-axis label.
    :param kwargs: additional arguments passed to `sns.barplot()`
    :return: matplotlib axis
    """
    # Checks
    sanitize_anndata(adata)
    groups_cond = [groups_cond] if isinstance(groups_cond, str) else groups_cond
    groups_pvals = [groups_pvals] if isinstance(groups_pvals, float) else groups_pvals
    feature = [feature] if isinstance(feature, str) else feature
    assert len(feature) == 1, 'Only 1 feature can be plotted'
    feature = feature[0]

    def log_estimator(values):
        return np.log1p(np.mean(np.expm1(values)))

    if feature in adata.var_names:
        df = get_expr(adata, feature, groups=x_axis, layer=layer)
        df_batch = mean_expr(adata, group_by=[x_axis, batch_key], features=feature, layer=layer)
        df_batch.columns = ['gene', x_axis, batch_key, 'expr']
    elif feature in list(adata.obs.columns):
        df = adata.obs[feature]
        df_batch = adata.obs[[feature, x_axis, batch_key]]
        df_batch = df_batch.groupby([x_axis, batch_key]).agg(np.mean).fillna(0).reset_index()
        df_batch['gene'] = feature
        df_batch.columns = [x_axis, batch_key, 'expr', 'gene']
        if logcounts:
            logger.warn(f'Assumming Log-counts but {feature} is in adata.obs')
    else:
        raise ValueError(f'{feature} is not in adata.var_names or adata.obs')

    # Create figure
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if estimator == 'LogMean':
        bp = sns.barplot(df, x=x_axis, y='expr', estimator=log_estimator,
                         capsize=capsize, ax=ax, palette=palette, **kwargs)
    else:
        bp = sns.barplot(df, x=x_axis, y='expr', estimator=estimator,
                         capsize=capsize, ax=ax, palette=palette, **kwargs)

    sns.stripplot(df_batch, x=x_axis, y='expr', alpha=0.75, color='k', s=marker_size, ax=ax)

    if ctrl_cond is not None and groups_cond is not None:
        if groups_pvals is None:
            testing = TestData(adata, feature, x_axis, ctrl_cond, groups_cond,
                               test, corr_method)
            testing.run_test()
            groups_pvals = testing.pvals

        plotter = StatsPlotter(bp, x_axis, 'expr', ctrl_cond, groups_cond,
                               txt_size, txt, groups_pvals, 'bar')
        plotter.plot_stats()

    if xtick_rotation is not None:
        bp.set_xticklabels(bp.get_xticklabels(), rotation=xtick_rotation, ha='right', va='top',
                           fontweight='bold')
    bp.set_xlabel('')
    bp.set_ylabel(ylabel)

    # TODO Remove the lower part from the errorbar

    try:
        title_size = title_fontproperties['size']
        title_font = title_fontproperties['weight']
    except ValueError:
        logger.warn('Setting title fontsize to 20 and fontweight to bold')
        title_size = 20
        title_font = 'bold'

    if title is None:
        bp.set_title(feature, fontsize=title_size, fontweight=title_font)  # Title is the genename
    else:
        bp.set_title(title, fontsize=title_size, fontweight=title_font)
    if path is not None:  # If the path is provided we save it
        plt.savefig(convert_path(path) / filename, bbox_inches='tight')
    if show is False:  # if show is false we return the axes
        return bp
    else:
        plt.tight_layout()
        return plt.show()


def boxplot(adata: ad.AnnData,
            feature: str,
            x_axis: str,
            layer: str = None,
            figsize: tuple = (5, 6),
            palette: Union[str, list] = 'tab10',
            xtick_rotation: int = None,
            ctrl_cond: str = None,
            groups_cond: Union[str, list] = None,
            groups_pvals: list = None,
            title: str = None,
            path: str = None, filename: str = None,
            title_fontproperties: dict = {'weight': 'bold', 'size': 20},
            show: bool = True,
            ax: plt.Axes = None,
            showfliers: bool = False,
            test: str = 'wilcoxon',
            corr_method: str = 'benjamini-hochberg',
            txt_size: int = 13,
            txt: str = 'p',
            ylabel='LogMean(nUMI)',
            **kwargs,
            ):
    """Boxplot with stats.

     Show the distribution of the  expression of `var_names` or a continuous value in `obs` along different categorical values
     and test for significance.

     :param adata: annotated data matrix
     :param feature: feature in `var_name` or `obs`.
     :param x_axis: categorical `obs` column to groupby.
     :param layer: layer in the AnnData to use.
     :param figsize:  figure size.
     :param palette: dictionary or palette to use.
     :param xtick_rotation: rotation of the xticks.
     :param ctrl_cond: name of the ctrl condition in the xticks
     :param groups_cond: list of the name of the groups to test in the xticks
     :param groups_pvals: if provided, these values will be plotted. If not set, provide a list of the groups in the xticks
                          to test.
     :param title: title of the plot.
     :param path: path to save the figure.
     :param filename: name of the file.
     :param title_fontproperties: properties of the title text.
     :param show: if set to False, return the axis.
     :param ax: matplotlib axis.
     :param showfliers: if set to False, the outliers of the boxplot are not shown.
     :param test: name of the method to test for significance. Available: ['wilcoxon', 't-test', 'kruskal', 'anova', 'logreg', 't-test_overestim_var'].
     :param corr_method: correction method for multiple testing. Available: ['benjamini-hochberg', 'bonferroni'].
     :param txt_size: size of the text indicating significance.
     :param txt: text for indicating significance. If not set, only the p-value is shown.
     :param ylabel: Y-axis label.
     :param kwargs: additional arguments passed to `sns.barplot()`
     :return: matplotlib axis
     """
    # Checks
    sanitize_anndata(adata)
    groups_cond = [groups_cond] if isinstance(groups_cond, str) else groups_cond
    groups_pvals = [groups_pvals] if isinstance(groups_pvals, float) else groups_pvals
    feature = [feature] if isinstance(feature, str) else feature
    assert len(feature) == 1, 'Only 1 feature can be plotted'
    feature = feature[0]

    if feature in adata.var_names:
        df = get_expr(adata, feature, groups=x_axis, layer=layer)
    elif feature in list(adata.obs.columns):
        df = adata.obs[feature]
    else:
        raise ValueError(f'{feature} is not in adata.var_names or adata.obs')

    # Create figure
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    bx = sns.boxplot(df, x=x_axis, y='expr', showfliers=showfliers, ax=ax, palette=palette, **kwargs)

    if ctrl_cond is not None and groups_cond is not None:
        if groups_pvals is None:
            testing = TestData(adata, feature, x_axis, ctrl_cond, groups_cond,
                               test, corr_method)
            testing.run_test()
            groups_pvals = testing.pvals

        plotter = StatsPlotter(bx, x_axis, 'expr', ctrl_cond, groups_cond,
                               txt_size, txt, groups_pvals, 'box')
        plotter.plot_stats()

    if xtick_rotation is not None:
        bx.set_xticklabels(bx.get_xticklabels(), rotation=xtick_rotation, ha='right', va='top',
                           fontweight='bold')
    bx.set_xlabel('')
    bx.set_ylabel(ylabel)

    try:
        title_size = title_fontproperties['size']
        title_font = title_fontproperties['weight']
    except ValueError:
        logger.warn('Setting title fontsize to 20 and fontweight to bold')
        title_size = 20
        title_font = 'bold'

    if title is None:
        bx.set_title(feature, fontsize=title_size, fontweight=title_font)  # Title is the genename
    else:
        bx.set_title(title, fontsize=title_size, fontweight=title_font)
    if path is not None:  # If the path is provided we save it
        plt.savefig(convert_path(path) / filename, bbox_inches='tight')
    if show is False:  # if show is false we return the axes
        return bx
    else:
        plt.tight_layout()
        return plt.show()


def violin(adata: ad.AnnData,
           feature: str,
           x_axis: str,
           layer: str = None,
           figsize: tuple = (5, 6),
           palette: Union[str, list] = 'tab10',
           xtick_rotation: int = None,
           ctrl_cond: str = None,
           groups_cond: Union[str, list] = None,
           groups_pvals: list = None,
           title: str = None,
           path: str = None, filename: str = None,
           title_fontproperties: dict = {'weight': 'bold', 'size': 20},
           show: bool = True,
           ax: plt.Axes = None,
           cut: float = 0,
           test: str = 'wilcoxon',
           corr_method: str = 'benjamini-hochberg',
           txt_size: int = 13,
           txt: str = 'p',
           ylabel='LogMean(nUMI)',
           **kwargs,
           ):
    """Violinplot with stats.

     Show the distribution of the  expression of `var_names` or a continuous value in `obs` along different categorical values
     and test for significance.

     :param adata: annotated data matrix
     :param feature: feature in `var_name` or `obs`.
     :param x_axis: categorical `obs` column to groupby.
     :param layer: layer in the AnnData to use.
     :param figsize:  figure size.
     :param palette: dictionary or palette to use.
     :param xtick_rotation: rotation of the xticks.
     :param ctrl_cond: name of the ctrl condition in the xticks
     :param groups_cond: list of the name of the groups to test in the xticks
     :param groups_pvals: if provided, these values will be plotted. If not set, provide a list of the groups in the
                          xticks to test.
     :param title: title of the plot.
     :param path: path to save the figure.
     :param filename: name of the file.
     :param title_fontproperties: properties of the title text.
     :param show: if set to False, return the axis.
     :param ax: matplotlib axis.
     :param cut: distance in units of bandwidth, to extend the density past extreme datapoints. Set to 0 to limit the
                 violin within the data range
     :param test: name of the method to test for significance. Available: ['wilcoxon', 't-test', 'kruskal', 'anova', 'logreg', 't-test_overestim_var'].
     :param corr_method: correction method for multiple testing. Available: ['benjamini-hochberg', 'bonferroni'].
     :param txt_size: size of the text indicating significance.
     :param txt: text for indicating significance. If not set, only the p-value is shown.
     :param ylabel: Y-axis label.
     :param kwargs: additional arguments passed to `sns.barplot()`
     :return: matplotlib axis
     """
    # Checks
    sanitize_anndata(adata)
    groups_cond = [groups_cond] if isinstance(groups_cond, str) else groups_cond
    groups_pvals = [groups_pvals] if isinstance(groups_pvals, float) else groups_pvals
    feature = [feature] if isinstance(feature, str) else feature
    assert len(feature) == 1, 'Only 1 feature can be plotted'
    feature = feature[0]

    if feature in adata.var_names:
        df = get_expr(adata, feature, groups=x_axis, layer=layer)
    elif feature in list(adata.obs.columns):
        df = adata.obs[feature]
    else:
        raise ValueError(f'{feature} is not in adata.var_names or adata.obs')

    # Create figure
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    vln = sns.violinplot(df, x=x_axis, y='expr', ax=ax, palette=palette, **kwargs)

    if ctrl_cond is not None and groups_cond is not None:
        if groups_pvals is None:
            testing = TestData(adata, feature, x_axis, ctrl_cond, groups_cond,
                               test, corr_method)
            testing.run_test()
            groups_pvals = testing.pvals

        plotter = StatsPlotter(vln, x_axis, 'expr', ctrl_cond, groups_cond,
                               txt_size, txt, groups_pvals, 'violin')
        plotter.plot_stats()

    if xtick_rotation is not None:
        vln.set_xticklabels(vln.get_xticklabels(), rotation=xtick_rotation, ha='right', va='top',
                            fontweight='bold')
    vln.set_xlabel('')
    vln.set_ylabel(ylabel)

    try:
        title_size = title_fontproperties['size']
        title_font = title_fontproperties['weight']
    except ValueError:
        logger.warn('Setting title fontsize to 20 and fontweight to bold')
        title_size = 20
        title_font = 'bold'

    if title is None:
        vln.set_title(feature, fontsize=title_size, fontweight=title_font)  # Title is the genename
    else:
        vln.set_title(title, fontsize=title_size, fontweight=title_font)
    if path is not None:  # If the path is provided we save it
        plt.savefig(convert_path(path) / filename, bbox_inches='tight')
    if show is False:  # if show is false we return the axes
        return vln
    else:
        plt.tight_layout()
        return plt.show()
