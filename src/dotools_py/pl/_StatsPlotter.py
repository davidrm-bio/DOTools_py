import sys
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.stats import shapiro
import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.collections import PolyCollection

from dotools_py import logger
from scipy.stats import ttest_ind, f_oneway, mannwhitneyu, kruskal


DEFAULT_TXT_SIZE = 13
DEFAULT_TXT = 'p'
DEFAULT_LINES_OFFSET = 0.05
DEFAULT_TEST = 'wilcoxon'
DEFAULT_MULTIPLE_TEST_CORRECTION = 'benjamini-hochberg'


class StatsPlotter:
    """Class to add statistics on bar, box or violin plots.

    This class add statistical annotations to bar, box and violin plots. A bracket will connect the control and tested
    condition and will indicate the p-value. The control and conditions to be tested should be in the x_axis.


    Parameters
    ----------
    axis
        matplotlib axis.
    x_axis
        name of the x-axis.
    y_axis
        name of the y-axis.
    ctrl
        name of the control condition. Expected to be present in the xticks.
    groups
        list of conditions in the xticks that have been tested.
    txt_size
        size of the text added.
    txt
        text to add before the p-value (e.g., p = ). If not set, only the p-value is added.
    pvals
        list of p-values for the conditions in groups. Expected to be in the same order.
    kind
        type of plot. Available: box, violin, bar.
    line_offset
        brackets are added in the highest y-value plus this offset


    See Also
    --------
        :class:`dotools_py.pl.TestData` - useful class to calculate statistics
    """
    def __init__(self,
                 axis: plt.Axes,
                 x_axis: str,
                 y_axis: str,
                 ctrl: str,
                 groups: list,
                 pvals: list,
                 txt_size: int = None,
                 txt: str = None,
                 kind: str = None,
                 line_offset: float = None
                 ):
        """Initialise.

        :param axis: matplotlib axis.
        :param x_axis: name of the x-axis.
        :param y_axis: name of the y-axis.
        :param ctrl: name of the control condition in x-axis.
        :param groups: list of names in the x-axis to add the stats for.
        :param txt_size: size of the text plotted.
        :param txt:  text to add before the p-value (e.g., p = ).
        :param pvals: list of p-values for the groups.
        :param kind: kind of plot: box, bar, violin.
        :param line_offset: offset from the bars/violin/boxplot for the stats.
        """

        if kind not in ['bar', 'box', 'violin']:
            raise NotImplemented(f'{kind} not implemented')

        self.axis = axis
        self.kind = kind

        self.x_axis = x_axis
        self.xticks = self.axis.get_xmajorticklabels()
        self.x_tick_pos = [_tick.get_position()[0] for _tick in self.xticks]
        self.x_ticks_labels = [_tick.get_text() for _tick in self.xticks]

        self.y_axis = y_axis
        self.yticks = self.axis.get_ymajorticklabels()
        self.y_ticks_pos = [_tick.get_position()[0] for _tick in self.yticks]
        self.y_ticks_labels = [_tick.get_text() for _tick in self.yticks]

        self.ctrl = ctrl
        self.groups = [groups] if isinstance(groups, str) else groups

        self.txt_size = DEFAULT_TXT_SIZE if txt_size is None else txt_size
        self.txt = DEFAULT_TXT if txt is None else txt
        self.line_offset = DEFAULT_LINES_OFFSET if line_offset is None else line_offset

        if pvals is not None:
            pvals = [float(p) for p in pvals]
            self.pvals = [
                str(np.round(p, 2)) if p > 0.05 else
                str(np.round(p, 4)) if p > 0.009 else
                '{:0.2e}'.format(sys.float_info.min if p == 0 else p)
                for p in pvals]
        else:
            self.pvals = pvals
        return

    def _get_height(self):
        """Calculate the heigh of bars, violins and boxs.

        :return: Self
        """
        # For bars (with capsize) and boxplots
        heights = {key: 0 for key in self.x_tick_pos}
        # ViolinPlots use Polycollection (Priority) and line2D(boxplot inside)
        if self.kind == 'violin':
            for _, pc in enumerate(self.axis.collections):
                if isinstance(pc, PolyCollection):
                    y_vals = pc.get_paths()[0].vertices[:, 1]  # The second column is the y-values
                    x_vals = int(pc.get_paths()[0].vertices[:, 0].mean())
                    if x_vals in heights:
                        heights[x_vals] = max(max(y_vals), heights[
                            x_vals])  # We expect X to be Categorical and have always pos 0, 1, 2, ...
        if self.kind in ['bar', 'box']:
            #  Bars with errorbars and boxplots (with/without outliers)
            for line in self.axis.lines:
                x_data, y_data = line.get_xdata(), line.get_ydata()
                for x, y in zip(x_data, y_data):
                    if x in heights:
                        heights[x] = max(heights[x], y)

            # Bars without errorbars
            try:
                for key, val in heights.items():
                    if val == 0:
                        for patch in self.axis.patches:
                            x = (patch.get_x() + patch.get_x() + patch.get_width()) / 2
                            if key == x:
                                y = patch.get_height()
                                heights[x] = max(heights[x], y)
            except AttributeError:
                pass

        self.heights = heights
        return

    def _get_pos_pairs(self):
        """Get x position and y heigh for the pairs tested.

        :return: Self
        """
        pairs_xpos, pairs_ypos = [], []
        for group in self.groups:
            # Position in X Axis [[x0_start, x0_end], [x1_start, x1_end]]
            xpair = [self.x_tick_pos[self.x_ticks_labels.index(self.ctrl)],
                     self.x_tick_pos[self.x_ticks_labels.index(group)]]
            pairs_xpos.append(xpair)
            # Position in Y Axis  [y0, y1]
            ypair = [self.heights[self.x_tick_pos[self.x_ticks_labels.index(self.ctrl)]],
                     self.heights[self.x_tick_pos[self.x_ticks_labels.index(group)]]]
            pairs_ypos.append(max(max(ypair), max(self.heights.values())))  # Start in the highest spot
        self.pairs_xpos = pairs_xpos
        self.pairs_ypos = pairs_ypos
        return

    def _get_offsets(self):
        """Get the offset to add the stats.

        :return: Self
        """
        pairs_offset = {key: round(val, 2) for key in self.groups for val in self.pairs_ypos}
        for key, val in pairs_offset.items():
            offset_added = self.line_offset
            new_pos = val + val * offset_added
            if new_pos in pairs_offset.values():
                cont = 0
                while new_pos in pairs_offset.values():
                    if cont == 100:
                        break
                    offset_added += 0.05
                    new_pos = val + val * offset_added
                    cont +=1
            pairs_offset[key] = new_pos
        self.heights_offset = list(pairs_offset.values())
        return

    def _draw_brackets(self):
        """Draw the brackets for the stats conecting ctrl and group.

        :return: Self
        """
        from matplotlib.path import Path

        rects = []
        for _stat in range(len(self.groups)):
            if len(self.groups) == 1:
                stem_length = (self.heights_offset[_stat] - np.max(list(self.heights.values()))) / 3
            else:
                try:
                    stem_length = (self.heights_offset[_stat+1] - self.heights_offset[_stat]) / 3
                except IndexError:
                    stem_length =  np.abs((self.heights_offset[_stat-1] - self.heights_offset[_stat]) / 3)

            verts = [(self.pairs_xpos[_stat][0], self.heights_offset[_stat] - stem_length),
                     (self.pairs_xpos[_stat][0], self.heights_offset[_stat]),
                     (self.pairs_xpos[_stat][1], self.heights_offset[_stat]),
                     (self.pairs_xpos[_stat][1], self.heights_offset[_stat] - stem_length),
                     ]
            codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]
            patch_path = Path(verts, codes)
            patch = PathPatch(patch_path, linewidth=1, facecolor='none', edgecolor='k', clip_on=False)
            rects.append(patch)
        self.brackets_patchs = rects
        return

    def _add_stats(self):
        """Plot the stats.

        :return: Self
        """
        for _stat, rect in enumerate(self.brackets_patchs):
            self.axis.add_patch(rect)

            # Add text in the center of the box
            txt_x = (self.pairs_xpos[_stat][0] + self.pairs_xpos[_stat][1]) / 2
            txt_y = self.heights_offset[_stat]
            self.axis.text(txt_x, txt_y, f'{self.txt}' + self.pvals[_stat], ha="center", va="bottom", fontsize=self.txt_size)

        bottom_y = self.axis.get_ylim()[0]
        top_y = max(self.heights_offset)
        buffer = np.abs(top_y) * 0.1
        self.axis.set_ylim(bottom_y, top_y + buffer)
        return

    def plot_stats(self):
        """Method to add the statistical annotation.

        :return: None
        """
        self._get_height()
        self._get_pos_pairs()
        self._get_offsets()
        self._draw_brackets()
        self._add_stats()


class TestData:
    """Class to perform test in AnnData or Pandas DataFrames.

    Class to perform statistical test between two or multiple conditions in an AnnData or pandas DataFrame (long format).
    Different statistical test can be used including: wilcoxon, t-test, kruskal, anova, logreg, t-test_overestim_var.
    Additionnally, different correction methods can be used for multiple testing (bonferroni and benjamini-hochberg)

    .. note::
        t-test_overestim_var and logreg is only available for AnnData input and anova and kruskal is only available
        for pandas dataframe


    Parameters
    ----------
    data
        annotated data matrix or pandas dataframe.
    feature
        var_name or obs column in the AnnData or column in the pandas dataframe to test.
    cond_key
        obs column or column in the dataframe with condition information.
    ctrl
        control condition.
    groups
        list of conditions
    test
        method to use for testing significance ('wilcoxon', 't-test', 'kruskal', 'anova', 'logreg', 't-test_overestim_var').
    test_correction
        correction method for multiple testing to use ('benjamini-hochberg', 'bonferroni')


    See Also
    --------
        :func:`dotools_py.pl.StatsPlotter`: class to plot the p-values in barplots, boxplots or violinplots

    """
    def __init__(self,
                 data,
                 feature,
                 cond_key,
                 ctrl,
                 groups,
                 test: str = None,
                 test_correction: str = None,
                 ):
        """Initialise,

        :param data: annotated data matrix or pandas dataframe.
        :param feature: var_names or obs column in the AnnData or column in the DataFrame.
        :param cond_key: column with condition information.
        :param ctrl: name of the control condition.
        :param groups: list of the alternative conditions to test against.
        :param test: method to use for testing. Available: ['wilcoxon', 't-test', 'kruskal', 'anova', 'logreg', 't-test_overestim_var'].
        :param test_correction: correction method to use. Available: ['benjamini-hochberg', 'bonferroni'].
        """

        assert isinstance(data, ad.AnnData) or isinstance(data, pd.DataFrame), 'Provide a DataFrame in long format or AnnData'
        self.data = data
        feature = [feature] if isinstance(feature, str) else feature
        assert len(feature) == 1, f'{len(feature)} features provided. Please provide only 1'
        self.key = feature[0]  # We only plot 1 feature
        if isinstance(data, pd.DataFrame):
            assert (cond_key in list(data.columns)), f'{cond_key} not in adata.obs or df.columns'
        if isinstance(data, ad.AnnData):
            assert (cond_key in list(data.obs.columns)), f'{cond_key} not in adata.obs or df.columns'
        self.cond_key = cond_key
        self.ctrl = ctrl
        self.groups = [groups] if isinstance(groups, str) else groups
        assert test in ['wilcoxon', 't-test', 'kruskal', 'anova', 'logreg', 't-test_overestim_var'], f'{test} not a valid test, use: "wilcoxon", "t-test", "kruskal", "anova", "logreg", "t-test_overestim_var"'
        self.test = test  # ['wilcoxon', 't-test', 'kruskal', 'anova', 'logreg', 't-test_overestim_var']
        assert test_correction in ['benjamini-hochberg', 'bonferroni'], f'{test_correction} not a valid test correction method, use: "benjamini-hochberg", "bonferroni"'
        self.test_corr = test_correction  # ['benjamini-hochberg', 'bonferroni']
        self.pvals = None
        self.correction = test_correction if test_correction is not None else DEFAULT_MULTIPLE_TEST_CORRECTION
        self.test = test if test is not None else DEFAULT_TEST

    def _test_adata(self):
        """Run test on AnnData.

        :return: Self
        """
        pvals = []
        if self.key in self.data.var_names:
            sc.tl.rank_genes_groups(self.data, groupby=self.cond_key, method=self.test, tie_correct=True,
                                    reference=self.ctrl, groups=self.groups, corr_method=self.test_corr)
            df = sc.get.rank_genes_groups_df(self.data, group=None)
            df = df[df['names'] == self.key]

            if len(self.groups) == 1:
                pvals += df['pvals_adj'].tolist()
            else:
                df.set_index('group', inplace=True)
                for group in self.groups:
                    pvals.append(df.loc[group, 'pvals_adj'])

        elif self.key in self.data.obs.columns:
            df_tmp = self.data.obs[[self.cond_key, self.key]]
            for group in self.groups:
                _, p = mannwhitneyu(df_tmp[df_tmp[self.cond_key] == self.ctrl][self.key],
                                    df_tmp[df_tmp[self.cond_key] == group][self.key],
                                    use_continuity=True,
                                    nan_policy='omit')
                pvals.append(p)
        else:
            raise Exception(f'{self.key} is not in adata.obs or adata.var_names')
        self.pvals = pvals
        return None


    def _test_df(self):
        """Run test on DataFrame.

        :return:
        """
        pvals = []

        if self.test in ['t-test', 'anova']:
            # Test for normality
            for group in self.groups + [self.ctrl]:
                _, p = shapiro(self.data[self.data[self.cond_key] == group][self.key])
                if p > 0.05:
                    new_test = 'wilcoxon' if self.test == 't-test' else 'anova'
                    logger.warn(f'Data does not follow normality but {self.test} was set, changing to {new_test}')
                    self.test = new_test
                    break

        if len(self.groups) == 1 and self.test in ['t-test', 'wilcoxon']:
            logger.warn(f'Running {self.test} but testing {len(self.groups)} conditions')

        for group in self.groups:
            x = self.data[self.data[self.cond_key] == self.ctrl][self.key]
            y = self.data[self.data[self.cond_key] == group][self.key]
            if self.test == 't-test':
                _, p = ttest_ind(x, y)
            elif self.test == 'anova':
                _, p = f_oneway(x, y)
            elif self.test == 'wilcoxon':
                _, p = mannwhitneyu(x, y, use_continuity=True)
            elif self.test == 'kruskal':
                _, p = kruskal(x, y)
            else:
                raise Exception(f'{self.test} not implemented')
            pvals.append(p)

        self.pvals = pvals
        return  None


    def run_test(self):
        """Method to run test.

        :return: None
        """
        if isinstance(self.data, ad.AnnData):
            self._test_adata()
        elif isinstance(self.data, pd.DataFrame):
            self._test_df()
        else:
            raise Exception('Input can only be an AnnData or DataFrame')
