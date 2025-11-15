import sys
from typing import Literal

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.patches import PathPatch
from scipy.stats import f_oneway, kruskal, mannwhitneyu, shapiro, ttest_ind

from dotools_py import logger
from dotools_py.utils import sanitize_anndata

DEFAULT_TXT_SIZE = 13
DEFAULT_TXT = "p ="
DEFAULT_LINES_OFFSET = 0.05
DEFAULT_TEST = "wilcoxon"
DEFAULT_MULTIPLE_TEST_CORRECTION = "benjamini-hochberg"


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
        list of p-values for the conditions in groups. Expected to be in the same order. If hue is provided, the order will
        depend on the groups and the x-ticks (e.g., x-ticks: healthy, disease; and groups: cell1, cell2, then pvals should be
        cell1_healthy Vs control_healthy, cell2_healthy Vs control_healthy, cell1_disease Vs control_disease, etc.). If hue
        is specified the number of expected values is len(x_axis groups) * len(hue groups).
    kind
        type of plot. Available: box, violin, bar.
    line_offset
        brackets are added in the highest y-value plus this offset. This offset is interpret as a percentage
        (i.e, a line offset of 0.05 means, we add an offset of 5 % to the height).
    hue:
        name of the variable used to split in subgroups.
    hue_order:
        order of the subgroups. Needs to be specified if hue is defined.

    See Also
    --------
        :func:`dotools_py.pl.TestData`: useful class to calculate statistics

    """

    def __init__(
        self,
        axis: plt.Axes,
        x_axis: str,
        y_axis: str,
        ctrl: str,
        groups: list,
        pvals: list,
        txt_size: int = None,
        txt: str = None,
        kind: str = None,
        line_offset: float = None,
        hue: str = None,
        hue_order: list = None,
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
        :param hue: name of the hue variable.
        :param hue_order: order for the hue variable.
        """
        if kind not in ["bar", "box", "violin"]:
            raise NotImplementedError(f"{kind} not implemented")

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
        self.line_offset = (
            DEFAULT_LINES_OFFSET if line_offset is None else line_offset
        )  # TODO should be adjusted depending on the y-values

        if pvals is not None:
            pvals = [float(p) for p in pvals]
            self.pvals = [
                str(np.round(p, 2))
                if p > 0.05
                else str(np.round(p, 4))
                if p > 0.009
                else f"{sys.float_info.min if p == 0 else p:0.2e}"
                for p in pvals
            ]
        else:
            self.pvals = pvals  # Will not do anything, might raise error?

        # Set-Up hue arguments
        self.hue = hue
        self.hue_order = hue_order

        if hue is not None:
            if hue_order is None:
                raise Exception('If hue is specified, hue_order need to be specified')
            if kind == "bar":
                hue_positions = [round(bar.get_x() + bar.get_width() / 2, 2) for bar in self.axis.patches]
            elif kind == "box" or kind == "violin":
                # hue_positions = []
                # for patch in self.axis.patches:
                #     x_pos = np.unique([x[0] for x in patch.get_path().vertices]).mean()   # [[x0, y0], [x1, y1]...]
                #     hue_positions.append(x_pos)
                hue_positions = []
                for line in self.axis.lines:
                    x_vals = line.get_xdata()
                    if len(x_vals) == 0:
                        continue
                    # Median x-value gives center of violin outline
                    x_pos = np.median(np.unique(x_vals))
                    hue_positions.append(round(x_pos, 2))
                hue_positions = list(np.unique(hue_positions))
            else:
                hue_positions = []
            hue_positions = np.sort(hue_positions)

            nhues = int(len(hue_positions) / len(self.x_ticks_labels))
            hue_labels = [x + '_' + s for x, s in
                          zip(np.repeat(self.x_ticks_labels, nhues), np.tile(hue_order, len(self.x_ticks_labels)))]

            self.hue_positions = hue_positions
            self.hue_labels = hue_labels
            self.hue_ctrl = [lab for lab in self.hue_labels if self.ctrl in lab]
            self.hue_groups = [
                [lab for lab in self.hue_labels if xtk in lab and any(g in lab for g in self.groups)]
                for xtk in self.x_ticks_labels
            ]

        # Check if we have the correct number of pvals
        # if hue is provided len(pvals) --> len(x_axis groups) * len(hue groups)
        if self.hue:
            assert len(self.pvals) == len(self.groups) * len(self.hue_groups), f"{len(self.pvals)} pvals provided but {len(self.groups) * len(self.hue_groups)} groups tested"
        else:
            assert len(self.pvals) == len(self.groups), f"{len(self.pvals)} pvals provided but {len(self.groups)} groups tested"
        return

    def _get_height(self):
        """Calculate the heigh of bars, violins and boxs.

        :return: Self
        """
        # For bars (with capsize) and boxplots
        if self.hue:
            heights = dict.fromkeys(self.hue_positions, 0)  # Adapt to consider when hue is provided
        else:
            heights = dict.fromkeys(self.x_tick_pos, 0)

        # ViolinPlots use Polycollection (Priority) and line2D(boxplot inside)
        if self.kind == "violin":
            for _, pc in enumerate(self.axis.collections):
                if isinstance(pc, PolyCollection):
                    y_vals = pc.get_paths()[0].vertices[:, 1]  # The second column is the y-values
                    x_vals = round(pc.get_paths()[0].vertices[:, 0].mean(), 2)
                    if x_vals in heights:
                        heights[x_vals] = max(
                            max(y_vals), heights[x_vals]
                        )  # We expect X to be Categorical and have always pos 0, 1, 2, ...
        if self.kind in ["bar", "box"]:
            #  Bars with errorbars and boxplots (with/without outliers)
            for line in self.axis.lines:
                x_data, y_data = line.get_xdata(), line.get_ydata()
                for x, y in zip(x_data, y_data, strict=False):
                    x = round(x, 2)
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
        if self.hue:
            pairs_xpos, pairs_ypos = [], []
            for jdx, group in enumerate(self.hue_groups):
                # Position in X Axis [[x0_start, x0_end], [x1_start, x1_end]]
                xpair = [[self.hue_positions[self.hue_labels.index(self.hue_ctrl[jdx])],
                          self.hue_positions[self.hue_labels.index(group[idx])]]
                         for idx in range(len(group))]
                pairs_xpos.extend(xpair)

                # Position in Y Axis  [y0, y1]
                ypair = [max([self.heights[self.hue_positions[self.hue_labels.index(self.hue_ctrl[jdx])]],
                              self.heights[self.hue_positions[self.hue_labels.index(group[idx])]]])
                         for idx in range(len(group))
                         ]

                pairs_ypos.extend([max(val, max(self.heights.values())) for val in ypair])  # Start in the highest spot
        else:
            pairs_xpos, pairs_ypos = [], []
            for group in self.groups:
                # Position in X Axis [[x0_start, x0_end], [x1_start, x1_end]]
                xpair = [
                    self.x_tick_pos[self.x_ticks_labels.index(self.ctrl)],
                    self.x_tick_pos[self.x_ticks_labels.index(group)],
                ]
                pairs_xpos.append(xpair)
                # Position in Y Axis  [y0, y1]
                ypair = [
                    self.heights[self.x_tick_pos[self.x_ticks_labels.index(self.ctrl)]],
                    self.heights[self.x_tick_pos[self.x_ticks_labels.index(group)]],
                ]
                pairs_ypos.append(max(max(ypair), max(self.heights.values())))  # Start in the highest spot
        self.pairs_xpos = pairs_xpos
        self.pairs_ypos = pairs_ypos
        return

    def _get_offsets(self):
        """Get the offset to add the stats.

        :return: Self
        """
        if self.hue:
            pairs_offset = {
                label: round(ypos, 2)
                for group, ypos in zip(self.hue_groups, self.pairs_ypos)
                for label in group}
            plotter_offset_hue = {xtck: {group: pairs_offset[group]
                                         for group in pairs_offset if xtck in group}
                                  for xtck in self.x_ticks_labels}

            for key, val in pairs_offset.items():
                offset_added = self.line_offset
                new_pos = val + val * offset_added
                current_hue = [g for g in self.x_ticks_labels if g in key][0]

                if new_pos in plotter_offset_hue[current_hue].values():
                    cont = 0
                    while new_pos in plotter_offset_hue[current_hue].values():
                        if cont == 100:
                            break
                        offset_added += 0.05
                        new_pos = val + val * offset_added
                        cont += 1
                pairs_offset[key] = new_pos
                plotter_offset_hue[current_hue][key] = new_pos
        else:
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
                        cont += 1
                pairs_offset[key] = new_pos
        self.heights_offset = list(pairs_offset.values())
        return

    def _draw_brackets(self):
        """Draw the brackets for the stats conecting ctrl and group.

        :return: Self
        """
        from matplotlib.path import Path

        if self.hue:
            rects, _extra = [], 0
            for group in self.hue_groups:
                for _stat in range(len(group)):
                    _stat += _extra
                    if len(self.groups) == 1:
                        stem_length = (self.heights_offset[_stat] - np.max(list(self.heights.values()))) / 3
                    else:
                        try:
                            stem_length = (self.heights_offset[_stat + 1] - self.heights_offset[_stat]) / 3
                        except IndexError:
                            stem_length = np.abs((self.heights_offset[_stat - 1] - self.heights_offset[_stat]) / 3)

                    verts = [
                        (self.pairs_xpos[_stat][0], self.heights_offset[_stat] - stem_length),
                        (self.pairs_xpos[_stat][0], self.heights_offset[_stat]),
                        (self.pairs_xpos[_stat][1], self.heights_offset[_stat]),
                        (self.pairs_xpos[_stat][1], self.heights_offset[_stat] - stem_length),
                    ]
                    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]
                    patch_path = Path(verts, codes)
                    patch = PathPatch(patch_path, linewidth=1, facecolor="none", edgecolor="k", clip_on=False)
                    rects.append(patch)
                _extra += len(group)
        else:
            rects = []
            for _stat in range(len(self.groups)):
                if len(self.groups) == 1:
                    stem_length = (self.heights_offset[_stat] - np.max(list(self.heights.values()))) / 3
                else:
                    try:
                        stem_length = (self.heights_offset[_stat + 1] - self.heights_offset[_stat]) / 3
                    except IndexError:
                        stem_length = np.abs((self.heights_offset[_stat - 1] - self.heights_offset[_stat]) / 3)

                verts = [
                    (self.pairs_xpos[_stat][0], self.heights_offset[_stat] - stem_length),
                    (self.pairs_xpos[_stat][0], self.heights_offset[_stat]),
                    (self.pairs_xpos[_stat][1], self.heights_offset[_stat]),
                    (self.pairs_xpos[_stat][1], self.heights_offset[_stat] - stem_length),
                ]
                codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]
                patch_path = Path(verts, codes)
                patch = PathPatch(patch_path, linewidth=1, facecolor="none", edgecolor="k", clip_on=False)
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
            self.axis.text(
                txt_x, txt_y, f"{self.txt}" + self.pvals[_stat], ha="center", va="bottom", fontsize=self.txt_size
            )

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
    category_key
        column with categorical metadata to split by (e.g., cell type annotation). The test will be done for each category across each condition.
    category_order
        order for the categories in category_key. If not specified will be inferred

    See Also
    --------
        :func:`dotools_py.pl.StatsPlotter`: class to plot the p-values in barplots, boxplots or violinplots

    """

    _pvals = []
    _hue_labels = []

    def __init__(
        self,
        data: pd.DataFrame | ad.AnnData,
        feature: str,
        cond_key: str,
        ctrl: str,
        groups: list,
        category_key: str = None,
        category_order: list = None,
        test: Literal["wilcoxon", "t-test", "kruskal", "anova", "logreg", "t-test_overestim_var"] = None,
        test_correction: Literal["benjamini-hochberg","bonferroni"] = None,
    ):
        """Initialise,

        :param data: annotated data matrix or pandas dataframe.
        :param feature: var_names or obs column in the AnnData or column in the DataFrame.
        :param cond_key: column with condition information.
        :param ctrl: name of the control condition.
        :param groups: list of the alternative conditions to test against.
        :param category_key: column with categorical metadata to split by (e.g., cell type annotation). The test will be done for each category across each condition.
        :param test: method to use for testing. Available: ['wilcoxon', 't-test', 'kruskal', 'anova', 'logreg', 't-test_overestim_var'].
        :param test_correction: correction method to use. Available: ['benjamini-hochberg', 'bonferroni'].
        """
        assert isinstance(data, ad.AnnData) or isinstance(data, pd.DataFrame), (
            "Provide a DataFrame in long format or AnnData"
        )
        self.data = data
        feature = [feature] if isinstance(feature, str) else feature
        assert len(feature) == 1, f"{len(feature)} features provided. Please provide only 1"
        self.key = feature[0]  # We only plot 1 feature
        if isinstance(data, pd.DataFrame):
            assert cond_key in list(data.columns), f"{cond_key} not in adata.obs or df.columns"
        if isinstance(data, ad.AnnData):
            assert cond_key in list(data.obs.columns), f"{cond_key} not in adata.obs or df.columns"
        self.cond_key = cond_key
        self.ctrl = ctrl
        self.groups = [groups] if isinstance(groups, str) else groups
        test = DEFAULT_TEST if test is None else test
        assert test in ["wilcoxon", "t-test", "kruskal", "anova", "logreg", "t-test_overestim_var"], (
            f'{test} not a valid test, use: "wilcoxon", "t-test", "kruskal", "anova", "logreg", "t-test_overestim_var"'
        )
        self.test = test  # ['wilcoxon', 't-test', 'kruskal', 'anova', 'logreg', 't-test_overestim_var']
        test_correction = DEFAULT_MULTIPLE_TEST_CORRECTION if test_correction is None else test_correction
        assert test_correction in ["benjamini-hochberg", "bonferroni"], (
            f'{test_correction} not a valid test correction method, use: "benjamini-hochberg", "bonferroni"'
        )
        self.test_corr = test_correction  # ['benjamini-hochberg', 'bonferroni']
        self.pvals = None
        self.correction = test_correction if test_correction is not None else DEFAULT_MULTIPLE_TEST_CORRECTION
        self.test = test if test is not None else DEFAULT_TEST
        self.hue = category_key

        if self.hue:
            if category_order is None:
                if isinstance(self.data, ad.AnnData):
                    sanitize_anndata(self.data)
                    self.hue_order = list(self.data.obs[self.hue].cat.categories)
                elif isinstance(self.data, pd.DataFrame):
                    self.hue_order = list(self.data[self.hue].unique())
            else:
                self.hue_order = category_order

    def _test_adata(self):
        """Run test on AnnData.

        :return: Self
        """
        import scanpy as sc

        pvals = []
        if self.key in self.data.var_names:
            if self.hue:
                # Note StatsPlotter and TestPlotter hue has reverse meaning
                # Note: StatsPlotter --> hue key specify the groups tested (e.g., hue = condition, we test condition for each x_ticks)
                # Note TestPlotter --> hue key specify the x_ticks and condition_key the groups tested

                sdf = pd.DataFrame([])
                for group in self.hue_order:
                    # Warning, we test for each hue
                    # but the adding to pvals is per condition
                    subset = self.data[self.data.obs[self.hue] == group]
                    try:
                        sc.tl.rank_genes_groups(
                            subset,
                            groupby=self.cond_key,
                            method=self.test,
                            tie_correct=True,
                            reference=self.ctrl,
                            groups=self.groups,
                            corr_method=self.test_corr
                        )
                        df = sc.get.rank_genes_groups_df(subset, group=None)
                        df = df[df["names"] == self.key]
                        df['hue'] = group
                    except ValueError:
                        g = self.groups.copy()
                        n = [self.key] * len(g)
                        scores = [0] * len(g)
                        lo2fcs = [0] * len(g)
                        p_values = [1] * len(g)
                        p_adjusted = [1] * len(g)
                        hue_col =  [group]
                        if len(self.groups) > 1:
                            df = pd.DataFrame([n,  scores, lo2fcs, p_values, p_adjusted, hue_col], index=['names', 'scores', 'logfoldchanges', 'pvals', 'pvals_adj', 'hue']).T
                        else:
                            df = pd.DataFrame([g, n,  scores, lo2fcs, p_values, p_adjusted, hue_col], index=['group', 'names', 'scores', 'logfoldchanges', 'pvals', 'pvals_adj', 'hue']).T
                    sdf = pd.concat([sdf, df])

                self.pvals_catgs_order = []
                if len(self.groups) == 1:
                    # Append keeping the order
                    for catg in self.hue_order:
                        pvals.append(sdf[sdf['hue'] == catg]['pvals_adj'].tolist()[0])
                        self.pvals_catgs_order.append(catg)
                else:
                    sdf['hue_group'] = sdf['hue'].astype(str) + "_" + sdf['group'].astype(str)
                    sdf.set_index("hue_group", inplace=True)
                    for catg in self.hue_order:  # Keep the order for the categories
                        for group in self.groups:  # Keep the order for the groups
                            pvals.append(sdf.loc[catg + "_" + group, "pvals_adj"])
                            self.pvals_catgs_order.append(group + "_" + catg)
            else:
                sc.tl.rank_genes_groups(
                    self.data,
                    groupby=self.cond_key,
                    method=self.test,
                    tie_correct=True,
                    reference=self.ctrl,
                    groups=self.groups,
                    corr_method=self.test_corr,
                )
                df = sc.get.rank_genes_groups_df(self.data, group=None)
                df = df[df["names"] == self.key]

                if len(self.groups) == 1:
                    pvals += df["pvals_adj"].tolist()
                else:
                    df.set_index("group", inplace=True)
                    for group in self.groups:
                        pvals.append(df.loc[group, "pvals_adj"])

        elif self.key in self.data.obs.columns:
            if self.hue:
                df_tmp = self.data.obs[[self.cond_key, self.hue, self.key]]
                self.pvals_catgs_order = []
                for catg in self.hue_order:
                    sdf = df_tmp[df_tmp[self.hue] == catg]
                    for group in self.groups:
                        _, p = mannwhitneyu(
                            sdf[sdf[self.cond_key] == self.ctrl][self.key],
                            sdf[sdf[self.cond_key] == group][self.key],
                            use_continuity=True,
                            nan_policy="omit",
                        )
                        pvals.append(p)
                        self.pvals_catgs_order.append(group + '_' + catg)
            else:
                df_tmp = self.data.obs[[self.cond_key, self.key]]
                for group in self.groups:
                    _, p = mannwhitneyu(
                        df_tmp[df_tmp[self.cond_key] == self.ctrl][self.key],
                        df_tmp[df_tmp[self.cond_key] == group][self.key],
                        use_continuity=True,
                        nan_policy="omit",
                    )
                    pvals.append(p)
        else:
            raise Exception(f"{self.key} is not in adata.obs or adata.var_names")
        self.pvals = pvals
        return None

    def _test_df(self):
        """Run test on DataFrame.

        :return:
        """
        pvals = []
        if self.hue:
            self.pvals_catgs_order = []
            for catg in self.hue_order:
                subset = self.data[self.data[self.hue] == catg]  # Subset for each hue category

                if self.test in ["t-test", "anova"]:
                    # Test for normality
                    for group in self.groups + [self.ctrl]:
                        _, p = shapiro(subset[subset[self.cond_key] == group][self.key])
                        if p > 0.05:
                            new_test = "wilcoxon" if self.test == "t-test" else "anova"
                            logger.warn(f"Data does not follow normality but {self.test} was set, changing to {new_test}")
                            self.test = new_test
                            break

                if len(self.groups) != 1 and self.test in ["t-test", "wilcoxon"]:
                    # For multiple conditions we use anova or kruskal
                    logger.warn(f"Running {self.test} but testing {len(self.groups)} conditions")


                for group in self.groups:
                    x = subset[subset[self.cond_key] == self.ctrl][self.key]
                    y = subset[subset[self.cond_key] == group][self.key]
                    if self.test == "t-test":
                        _, p = ttest_ind(x, y)
                    elif self.test == "anova":
                        _, p = f_oneway(x, y)
                    elif self.test == "wilcoxon":
                        _, p = mannwhitneyu(x, y, use_continuity=True)
                    elif self.test == "kruskal":
                        _, p = kruskal(x, y)
                    else:
                        raise Exception(f"{self.test} not implemented")
                    pvals.append(p)
                    self.pvals_catgs_order.append(group + "_" + catg)
        else:
            if self.test in ["t-test", "anova"]:
                # Test for normality
                for group in self.groups + [self.ctrl]:
                    _, p = shapiro(self.data[self.data[self.cond_key] == group][self.key])
                    if p > 0.05:
                        new_test = "wilcoxon" if self.test == "t-test" else "anova"
                        logger.warn(f"Data does not follow normality but {self.test} was set, changing to {new_test}")
                        self.test = new_test
                        break

            if len(self.groups) != 1 and self.test in ["t-test", "wilcoxon"]:
                logger.warn(f"Running {self.test} but testing {len(self.groups)} conditions")

            for group in self.groups:
                x = self.data[self.data[self.cond_key] == self.ctrl][self.key]
                y = self.data[self.data[self.cond_key] == group][self.key]
                if self.test == "t-test":
                    _, p = ttest_ind(x, y)
                elif self.test == "anova":
                    _, p = f_oneway(x, y)
                elif self.test == "wilcoxon":
                    _, p = mannwhitneyu(x, y, use_continuity=True)
                elif self.test == "kruskal":
                    _, p = kruskal(x, y)
                else:
                    raise Exception(f"{self.test} not implemented")
                pvals.append(p)

        self.pvals = pvals
        return None

    def run_test(self):
        """Method to run test.

        :return: None
        """
        if isinstance(self.data, ad.AnnData):
            self._test_adata()
        elif isinstance(self.data, pd.DataFrame):
            self._test_df()
        else:
            raise Exception("Input can only be an AnnData or DataFrame")
        TestData._pvals = self.pvals
        if self.hue:
            TestData._hue_labels = self.pvals_catgs_order

    @classmethod
    def pvalues(cls):
        """Get list with the p-vals from the test. If category_key is not set the order of the pvals match the order of the
        labels in groups.
        :return:
        """
        return TestData._pvals

    @classmethod
    def pvalues_labels(cls):
        """Get list with the labels of the group tested. The order matches the order of the `pvals` attribute. Only initialise
        if the hue category_key is set.
        :return:
        """
        return TestData._hue_labels

