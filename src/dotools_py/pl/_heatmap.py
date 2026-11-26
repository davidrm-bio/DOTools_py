from typing import Literal, Dict

import matplotlib.lines as mlines
from matplotlib.colors import ListedColormap

from matplotlib.cm import ScalarMappable

from prelude_py import ad, np, pd, sns, plt

from dotools_py.logger import logger
from dotools_py.tl import rank_genes_groups
from dotools_py.get import log2fc as get_log2fc
from dotools_py._utils import  convert_path, sanitize_anndata, iterase_input, check_missing
from dotools_py.pl._plot_utils import make_grid_spec, check_colornorm, square_color, small_squares
from dotools_py._custom_class import PathLike
from dotools_py.pl._Classes import MatrixPlot



def heatmap(
    adata: ad.AnnData,
    *,
    x_axis: str,
    features: str | list | dict,
    y_axis: str | None = None,
    xticks_order: list | None = None,
    yticks_order: list | None = None,
    layer: str | None = None,

    subset_adata: bool = False,
    logcounts: bool = True,
    expression_cutoff: float = 0.0,
    mean_only_expressed: bool = False,
    z_score: Literal["x_axis", "y_axis"] | None = None,
    standard_scale: Literal["x_axis", "y_axis"] | None = None,

    # Figure parameters
    figsize: tuple = (5, 6),
    ax: plt.Axes | None = None,
    swap_axes: bool = False,
    title: str = "",
    title_fontproperties: Dict[Literal["size", "weight"], str | int] | None = None,
    palette: str | None = None,
    cmap: str = None,
    vmin: float | None = None,
    vcenter: float | None = None,
    vmax: float | None = None,

    color_legend_title: str = "LogMean(nUMI)\nin group",
    feature_fontsize: float = 12,

    xticks_properties: dict | None = None,
    xticks_rotation: int = 45,
    cluster_x_axis: bool = False,
    yticks_properties: dict | None = None,
    yticks_rotation: int = 0,
    cluster_y_axis: bool = False,
    clustering_method: str = "complete",
    clustering_metric: str = "euclidean",
    var_group_offset: float = 0.75,
    var_group_rotation: float = 0,

    # IO
    path: PathLike | None = None,
    filename: str = "Heatmap.svg",
    show: bool = True,

    # Statistics
    add_stats: Literal["x_axis", "y_axis"] | None = None,
    test: Literal["wilcoxon", "t-test"] = "wilcoxon",
    correction_method: Literal["benjamini-hochberg", "bonferroni"] = "benjamini-hochberg",
    df_pvals: pd.DataFrame | None = None,
    stats_x_size: float | None = None,
    square_x_size: dict | None = None,
    pval_cutoff: float = 0.05,
    log2fc_cutoff: float = 0.0,

    # Fx specific
    rect_height: float = 0.35,
    linewidth: float = 0.1,
    **kwargs,
) -> dict | None:
    """Make a heatmap of the values of `adata.var_names` or `adata.obs`.

    There are two types of visualization:
        * Categorical heatmap: the ``features`` are grouped by ``x_axis``. The color represents the logarithmize mean
                               nUMI or the mean value for features in `adata.obs`.
        * 2-Categorical heatmap: the ``features`` are groupbed by ``x_axis`` and ``y_axis``. For each feature the
                                ``x_axis`` categories will be duplicates to show how is the expression across the
                                ``y_axis`` categories.

    .. note::
        A feature is considered expressed if the expression value in the `adata` is above the specified threshold
        which is zero by default.


    :param adata: Annotated data matrix.
    :param x_axis: A valid key from `adata.obs` to groupby.
    :param features: valid key from `adata.var_names` or a continuous value in `adata.obs` to show the mean values.
                    If `features` is a dictionary, then the key is used as label to group the values.
    :param y_axis: A valid additional key from `adata.obs` to groupby.
    :param xticks_order: Order of the categories in x_axis.
    :param yticks_order: Order of the categories in y_axis.
    :param layer: A valid key from `adata.layers` to be used for the mean expression calculations.
    :param subset_adata: Set to `True` to subset the AnnData if less `x_categories_order` and `y_categories_order`
                        are provided.
    :param logcounts: Set to `True` if the input is in log space.
    :param expression_cutoff: Cutoff used for binarizing the gene expression and determining the fraction of
                              cells expressing a given feature. A gene is expressed only if the expression value is greater
                              than this threshold.
    :param z_score: Apply Z-score transformation over the x-axis or y-axis.
    :param standard_scale: Apply MinMax normalisation over the x-axis or y-axis.
    :param figsize: Figure size, the format is (width, height).
    :param ax: Matplotlib axes to use for plotting. If not set, a new figure will be generated.
    :param swap_axes: Change the X and Y-axis.
    :param title:  Title for the figure.
    :param title_fontproperties: Dictionary which should contain 'size' and 'weight' to define the fontsize and fontweight of the title of the figure.
    :param palette: String denoting matplotlib colormap.
    :param cmap: String denoting matplotlib colormap.
    :param vmin: The value representing the lower limit of the color scale.
    :param vcenter: The value representing the center of the color scale.
    :param vmax: The value representing the upper limit of the color scale.
    :param color_legend_title: Title for the colorbar.
    :param feature_fontsize: Size of the feature names when y-ticks is specified.
    :param xticks_properties: Dictionary which should contain 'size' and 'weight' to define the fontsize and fontweight of font of the x-axis.
    :param xticks_rotation: Rotation of the x-ticks.
    :param cluster_x_axis: Hierarchically clustering the x-axis.
    :param yticks_properties: Dictionary which should contain 'size' and 'weight' to define the fontsize and fontweight of the font of the y-axis.
    :param yticks_rotation:  Rotation of the y-ticks.
    :param cluster_y_axis: Hierarchically clustering the y-axis.
    :param clustering_method: Linkage method to use for calculating clusters. See `scipy.cluster.hierarchy.linkage <https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage>`_.
    :param clustering_metric: Distance metric to use for the data. See `scipy.spatial.distance.pdist <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist>`_.
    :param var_group_offset: If `features` is a dictionary, set the rotation of the labels for each group.
    :param var_group_rotation: Offset for the labels text in the brackets.
    :param path: Path to save plot.
    :param filename: Filename of the plot.
    :param show: If set to `False`, returns a dictionary with the matplotlib axes.
    :param add_stats: Add an asterisk (*) in a square to indicate statistical significance.
    :param test: Statistical method to use.
    :param correction_method: Correction method for multiple testing.
    :param df_pvals: Dataframe with significant values with the same shape as the dotplot.
    :param stats_x_size: Size of the star.
    :param square_x_size: Dictionary specifying the `size` and thickness (`width`) of the squares.
    :param pval_cutoff: Cutoff for the p-value.
    :param log2fc_cutoff: Minimum value for the log2-foldchange to consider the feature to be differently regulated.
    :param rect_height: Height of the boxes of the feature names in the dotplot when `y_axis` is provided
    :param linewidth: Linewidth for the border of cells.
    :param kwargs: Additional arguments pass to `sns.heatmap <https://seaborn.pydata.org/generated/seaborn.heatmap.html>`_.
    :return: Depending on ``show``, returns the plot if set to `True` or a dictionary with the axes.


    Example
    -------
    Create a heatmap and add statistical information

    .. plot::
        :context: close-figs

        import dotools_py as do
        adata = do.dt.example_10x_processed()
        do.pl.heatmap(adata, x_axis='annotation', features=['CD4', 'CD79A'], add_stats="x_axis")

    Create a heatmap grouping by two categorical metadata

    .. plot::
        :context: close-figs

        do.pl.heatmap(adata, x_axis='condition',  features=['CD4', 'CD79A'], y_axis='annotation', figsize=(6, 4))

        # Add Statistical significance for groups with pvals < 0.05 and log2fc > 0.0
        # Note, the object is quite small, some groups cannot be tested for having one condition only.
        do.pl.heatmap(adata, x_axis='condition', features=['CD4', 'CD79A'], y_axis='annotation', figsize=(6, 4), add_stats='y_axis')

    """
    # todo implement mean_only_expressed

    if subset_adata:
        if xticks_order is not None:
            adata = adata[adata.obs[x_axis].isin(xticks_order)]
            logger.warn(f"Subsetting anndata for {xticks_order}")
        if yticks_order is not None:
            adata = adata[adata.obs[y_axis].isin(yticks_order)]
            logger.warn(f"Subsetting anndata for {yticks_order}")
    else:
        if xticks_order is not None:
            assert len(xticks_order) == adata.obs[x_axis].nunique(), (
                f"Not all {x_axis} categories provided. Specify all or use subset_adata = True"
            )
        if yticks_order is not None:
            assert len(yticks_order) == adata.obs[y_axis].nunique(), (
                f"Not all {y_axis} categories provided. Specify all or use subset_adata = True"
            )

    if cmap is not None and palette is not None:
        raise ValueError("Cannot provide both cmap and palette")
    elif cmap is None and palette is None:
        cmap = "Reds"

    colormap = cmap if cmap is not None else palette

    if mean_only_expressed:
        raise NotImplementedError("mean_only_expressed is currently not available")

    plotter = MatrixPlot(
        adata=adata, x_axis=x_axis, features=features, y_axis=y_axis,
        xticks_order=xticks_order, yticks_order=yticks_order, layer=layer, logcounts=logcounts,
        figsize=figsize, ax=ax, swap_axes=swap_axes, title=title, title_fontproperties=title_fontproperties,
        palette=None, cmap=colormap, xticks_properties=xticks_properties, yticks_properties=yticks_properties,
        xticks_rotation=xticks_rotation, yticks_rotation=yticks_rotation, cluster_x_axis=cluster_x_axis,
        cluster_y_axis=cluster_y_axis, var_group_offset=var_group_offset, var_group_rotation=var_group_rotation,
        legend_size_title="", legend_title=color_legend_title, path=path, filename=filename,
        show=show, add_stats=add_stats, test=test, correction_method=correction_method, df_pvals=df_pvals,
        stats_x_size=stats_x_size, square_x_size=square_x_size, pval_cutoff=pval_cutoff, log2fc_cutoff=log2fc_cutoff,
        z_score=z_score, standard_scale=standard_scale, clustering_method=clustering_method,
        clustering_metric=clustering_metric, linewidth=linewidth, vmin=vmin, vcenter=vcenter, vmax=vmax,
        annot_fontsize=feature_fontsize, annot_ratio=rect_height, expression_cutoff=expression_cutoff,
        dot_max=None, dot_min=None, size_exponent=None, largest_dot=None, smallest_dot=None,
        norm=None, edge_color=None, edge_lw=None, x_padding=None, y_padding=None, color_on="square",
        **kwargs
    )

    return  plotter.heatmap()




def heatmap_foldchange(
    # Data
    adata: ad.AnnData,
    *,
    group_by: str | list,
    features: str | list,
    condition_key: str,
    reference: str,
    groups_order: list = None,
    conditions_order: list = None,
    layer: str = None,

    # Figure parameters
    figsize: tuple = (5, 6),
    ax: plt.Axes | None = None,
    swap_axes: bool = True,
    title: str = None,
    title_fontproperties: Dict[Literal["size", "weight"], str | int] = None,
    palette: str = "RdBu_r",
    palette_conditions: str | dict = "tab10",
    ticks_fontproperties: Dict[Literal["size", "weight"], str | int] = None,
    xticks_rotation: int = None,
    yticks_rotation: int = None,
    vmin: float = None,
    vcenter: float = None,
    vmax: float = None,
    colorbar_legend_title: str = "Log2FC",
    groups_legend_title: str = "Comparison",
    group_legend_ncols: int = 1,

    # IO
    path: PathLike = None,
    filename: str = "Heatmap.svg",
    show: bool = True,

    # Statistics
    add_stats: bool = False,
    test: Literal["wilcoxon", "t-test"] = "wilcoxon",
    correction_method: Literal["benjamini-hochberg", "bonferroni"] = "benjamini-hochberg",
    df_pvals: pd.DataFrame = None,
    stats_x_size: float = None,
    square_x_size: dict = None,
    pval_cutoff: float = 0.05,
    log2fc_cutoff: float = 0.0,

    # Fx specific
    linewidth: float = 0.1,
    color_axis_ratio=0.15,
    **kargs,
) -> dict | None:
    """Heatmap of the log2-foldchange of genes across a groups between two conditions.

    Generate a heatmap of showing the log2-foldchange for a set of genes in different groups between different
    conditions. Differential gene expression analysis between the different conditions can be performed.

    :param adata: Annotated data matrix.
    :param group_by:  Name of a categorical column in `adata.obs` to groupby.
    :param features: A valid feature in `adata.var_names` or column in `adata.obs` with continuous values.
    :param condition_key: Name of a categorical column in `adata.obs` to compare to compute the log2foldchanges
    :param reference: Category in `adata.obs[condition_key]` to use as the reference to compute the log2foldchanges
    :param groups_order: Order for the categories in `adata.obs[group_by]`.
    :param conditions_order: Order for the categories in `adata.obs[condition_key]`
    :param path: Path to the folder to save the figure.
    :param filename: Name of file to use when saving the figure.
    :param layer: Name of the AnnData object layer that wants to be plotted. By default, `adata.X` is plotted.
                 If layer is set to a valid layer name, then the layer is plotted.
    :param swap_axes: Whether to swap the axes or not.
    :param palette: String denoting matplotlib colormap.
    :param palette_conditions: String denoting matplotlib colormap for the comparisons.
    :param title: Title for the figure.
    :param title_fontproperties: Dictionary which should contain 'size' and 'weight' to define the fontsize and
                                fontweight of the title of the figure.
    :param ax: Matplotlib axes to use for plotting. If not set, a new figure will be generated.
    :param figsize: Figure size, the format is (width, height).
    :param linewidth: Linewidth for the border of cells.
    :param ticks_fontproperties: Dictionary which should contain 'size' and 'weight' to define the fontsize and fontweight of the font of the x/y-axis.
    :param xticks_rotation: Rotation of the x-ticks.
    :param yticks_rotation: Rotations of the y-ticks.
    :param vmin: The value representing the lower limit of the color scale.
    :param vcenter: The value representing the center of the color scale.
    :param vmax: The value representing the upper limit of the color scale.
    :param colorbar_legend_title: Title for the colorbar.
    :param groups_legend_title: Title for the comparison legend.
    :param group_legend_ncols: Number of columns for the comparison legend.
    :param add_stats: Add statistical annotation. Will add a square with an '*' in the center if the expression is significantly different in a group with respect to the others.
    :param df_pvals: Dataframe with the pvals. Should be `gene x group` or `group x gene` in case of swap_axes is `False`.
    :param stats_x_size: Scaling factor to control the size of the asterisk.
    :param square_x_size: Size and thickness of the square.
    :param test: Name of the method to test for significance.
    :param correction_method: Correction method for multiple testing.
    :param pval_cutoff: Cutoff for the p-value.
    :param log2fc_cutoff: Minimum cutoff for the log2FC.
    :param show: If set to `False`, returns a dictionary with the matplotlib axes.
    :param color_axis_ratio: Ratio of the axis reserved for the colors denoting the comparisons.
    :param kargs: Additional arguments pass to `sns.heatmap <https://seaborn.pydata.org/generated/seaborn.heatmap.html>`_.
    :return: Depending on ``show``, returns the plot if set to `True` or a dictionary with the axes.

    Example
    -------

    .. plot::
        :context: close-figs

        import dotools_py as do
        adata = do.dt.example_10x_processed()
        do.pl.heatmap_foldchange(adata, group_by='annotation', features=['CD4', 'CD79A'], condition_key="condition", reference="healthy", add_stats=True, figsize=(5, 7))

    """
    import scanpy as sc  # Lazy load
    from matplotlib.colorbar import Colorbar
    from dotools_py.utility import get_hex_colormaps

    # Checks
    assert reference is not None, "Provide reference condition"
    assert condition_key is not None, "Provide a column to compute log-foldchanges and statistics"
    features = iterase_input(features)
    assert all(item in list(adata.var_names) for item in features), \
        "log-foldchanges can only be computed for features in adata.var_names"

    sanitize_anndata(adata)
    check_missing(adata, features=features)

    # Compute the log-foldchanges,
    # Format gene|class x group_by (Genes duplicated for each class)
    df = []
    for g in adata.obs[group_by].unique():
        adata_subset = adata[adata.obs[group_by] == g]
        df_tmp = get_log2fc(
            adata_subset, group_by=condition_key, features=features, layer=layer, reference=reference
        )
        df_tmp = df_tmp.melt(id_vars="genes", value_name="log2fc", var_name="class")
        df_tmp["class"] = df_tmp["class"].str.replace("log2fc_", "")
        df_tmp["group_by"] = g
        df.append(df_tmp)
    df = pd.concat(df)
    df = df.pivot(index=["genes", "class"], columns="group_by", values="log2fc")
    # Set colors for the groups
    if isinstance(palette_conditions, str):
        if len(adata.obs[condition_key].cat.categories) > len(get_hex_colormaps(palette_conditions)):
            logger.warn(f"There are {len(adata.obs[condition_key].cat.categories)} conditions but the colormap has"
                        f"{get_hex_colormaps(palette_conditions)} shades.")
        class_dictionaries = dict(zip(adata.obs[condition_key].cat.categories, get_hex_colormaps(palette_conditions)))
    elif isinstance(palette_conditions, dict):
        if all(item in list(adata.obs[condition_key].unique()) for item in palette_conditions):
            class_dictionaries = palette_conditions
        else:
            raise ValueError("Missing conditions in palette_conditions")
    else:
        raise NotImplementedError("palette_conditions is not a str or dict")

    # Set the order
    if conditions_order is not None:
        new_gene_class_order = [(g, c) for c in conditions_order for g in features]
    else:
        new_gene_class_order = [(g, c) for c in np.unique(df.index.get_level_values("class")) for g in features]
        conditions_order = np.unique(df.index.get_level_values("class"))
    groups_order = groups_order if groups_order is not None else list(adata.obs[group_by].unique())

    if isinstance(df.index, pd.MultiIndex):
        df = df.reindex(index=new_gene_class_order, columns=groups_order)
    else:
        df = df.reindex(columns=new_gene_class_order, index=groups_order)

    # Layout
    if swap_axes:
        df = df.T

    # Compute Statistics
    annot_pvals = None
    if add_stats:
        if df_pvals is None:
            if all(item in list(adata.var_names) for item in features):
                table_filt = []
                for g in adata.obs[group_by].unique():
                    adata_subset = adata[adata.obs[group_by] == g]
                    try:
                        rank_genes_groups(
                            adata_subset, groupby=condition_key, method=test, reference=reference, tie_correct=True,
                            corr_method=correction_method
                        )
                    except ValueError as e:
                        logger.warn(f"Failed to compute stats for {g}: {e}")
                        continue
                    table = sc.get.rank_genes_groups_df(
                        adata_subset, group=None, pval_cutoff=pval_cutoff, log2fc_min=log2fc_cutoff
                    )
                    table[group_by] = g
                    if "group" not in table.columns:
                        group = list(adata_subset.obs[condition_key].unique())
                        group.remove(reference)
                        table["group"] = group[0]

                    table_filt.append(table[table["names"].isin(features)])
                table_filt = pd.concat(table_filt)
            else:
                raise Exception('Not Implemented, all features needs to be in adata.var_names')

            # Dataframe with gene|class x groups with the pvals
            df_pvals = pd.DataFrame([], index=df.index, columns=df.columns)
            for idx, row in table_filt.iterrows():
                if isinstance(df.index, pd.MultiIndex):
                    if row["group"] in list(df.index.get_level_values("class")):
                        df_pvals.loc[(row["names"], row["group"]), row[group_by]] = row["pvals_adj"]
                else:
                    if row["group"] in list(df.columns.get_level_values("class")):
                        df_pvals.loc[row[group_by], (row["names"], row["group"])] = row["pvals_adj"]

            df_pvals[df_pvals.isna()] = 1
        else:
            df_pvals = (
                df_pvals
                .pivot(
                    index=group_by,
                    columns=["genes", condition_key],
                    values="value"
                )
                .reindex(index=list(df.index))
            )

            if list(df.index)[0] in list(df_pvals.index):
                pass
            else:
                df_pvals = df_pvals.T
        # Replace pvals < 0.05 with an X
        annot_pvals = df_pvals.map(lambda x: "*" if x < pval_cutoff else "")

    # -------------------------------------------- Arguments for the layout --------------------------------------------

    # # # # # # # #
    # Figure Layout
    # # # # # # # #
    width, height = figsize
    legends_width_spacer = 0.7 / width
    mainplot_width = width - (1.5 + 0)

    min_figure_height = max([0.35, height])
    cbar_legend_height = min_figure_height * 0.08
    sig_legend = min_figure_height * 0.15
    foldchange_legend = min_figure_height * 0.1
    spacer_height = min_figure_height * 0.27

    height_ratios = [
        height - sig_legend - cbar_legend_height - spacer_height - foldchange_legend,
        foldchange_legend,
        sig_legend,
        spacer_height,
        cbar_legend_height,
    ]

    # # # # # # # # #
    # Text Properties
    # # # # # # # # #  ticks_fontproperties = {} if ticks_fontproperties is None else ticks_fontproperties
    title_fontproperties = {} if title_fontproperties is None else title_fontproperties
    ticks_fontproperties = {} if ticks_fontproperties is None else ticks_fontproperties
    ticks_fontproperties = {
        "weight": ticks_fontproperties.get("weight", "bold"), "size": ticks_fontproperties.get("size", 13)
    }
    title_fontprop = {
        "weight": title_fontproperties.get("weight", "bold"), "size": title_fontproperties.get("size", 15)
    }

    # # # # # #
    # Colorbar
    # # # # # #
    vmin =  df.min().min()  if vmin is None else vmin
    vmax = df.max().max() if vmax is None else vmax
    vcenter = 0.0 if vcenter is None else vcenter


    colormap = plt.get_cmap(palette)
    normalize = check_colornorm(vmin=vmin, vmax=vmax, vcenter=vcenter)
    mappable = ScalarMappable(norm=normalize, cmap=colormap)

    #mean_flat = df.T.values.flatten()
    #color = colormap(normalize(mean_flat))
    #color = [square_color(c) for c in color]

    # # # # # # #
    # Statistics
    # # # # # # #
    square_x_size = {} if square_x_size is None else square_x_size
    square_x_size = {
        "width": square_x_size.get("weight", 1), "size": square_x_size.get("size", 0.8)
    }
    stats_x_size = min(width / df.shape[1], height / df.shape[1]) * 10 if stats_x_size is None else min(
        width / df.shape[1], height / df.shape[1]) * stats_x_size

    # # # # # # # #
    # Colors axis
    # # # # # # # #
    height_ratios_main, width_ratios_main, nrows_main, ncols_main = [height], [mainplot_width], 1, 1
    if "genes" in df.columns.names:
        nrows_main, ncols_main = 2, 1
        height_ratios_main = [
            color_axis_ratio,
            height - color_axis_ratio
        ]
        pos_groups, pos_main = 0, 1
    else:
        nrows_main, ncols_main = 1, 2
        width_ratios_main = [
            mainplot_width - color_axis_ratio,
            color_axis_ratio
        ]
        pos_groups, pos_main = 1, 0

    # # # #
    # Axis
    # # # #
    return_ax_dict = {}

    # -------------------------------------------- Generate the Figure -------------------------------------------------
    fig, gs = make_grid_spec(
        ax or (width, height), nrows=1, ncols=2, wspace=legends_width_spacer, width_ratios=[mainplot_width, 1.5]
    )

    # Create Main Axis
    main_ax = fig.add_subplot(gs[0])
    fig, main_ax_gs = make_grid_spec(
        main_ax, nrows=nrows_main, ncols=ncols_main, hspace=0.01, wspace=0.01, height_ratios=height_ratios_main,
        width_ratios=width_ratios_main
    )
    main_ax = fig.add_subplot(main_ax_gs[pos_main])

    # Create Legend Axis
    legend_ax = fig.add_subplot(gs[1])
    fig, legend_gs = make_grid_spec(legend_ax, nrows=len(height_ratios), ncols=1, height_ratios=height_ratios)
    color_legend_ax = fig.add_subplot(legend_gs[4])

    # Create Colors Axis
    groups_ax = fig.add_subplot(main_ax_gs[pos_groups])

    # Create Significance axis
    if add_stats:
        sig_ax = fig.add_subplot(legend_gs[3])

    # Main Plot
    # Correction to remove class from the dataframe (gene|class x groups)
    df_copy = df.copy()
    df_copy = df_copy.droplevel("class") if "genes" in df.index.names else df_copy.T.droplevel("class").T

    hm = sns.heatmap(
        data=df_copy, cmap=palette, ax=main_ax, linewidths=linewidth, cbar=False, annot=annot_pvals, fmt="s",
        square=False, annot_kws=
        {"color": "black", "size": stats_x_size, "ha": "center", "va": "center", "fontfamily": 'DejaVu Sans Mono'},
        vmax=vmax, vmin=vmin, center=vcenter, **kargs,
    )
    if isinstance(df.index, pd.MultiIndex):
        classes = df.reset_index()["class"].astype(
            pd.CategoricalDtype(categories=conditions_order, ordered=True)
        )
        tmp = pd.DataFrame(classes.cat.codes)
    else:
        classes = df.T.reset_index()["class"].astype(
            pd.CategoricalDtype(categories=conditions_order, ordered=True)
        )
        tmp = pd.DataFrame(classes.cat.codes).T

    unique_color_vector = [class_dictionaries[c] for c in classes.cat.categories]
    cmap = ListedColormap(unique_color_vector)
    sns.heatmap(
        tmp, cmap=cmap, ax=groups_ax, xticklabels=False, yticklabels=False, cbar=False
    )

    # Add colors group Legend
    groups_ax_legend = fig.add_subplot(legend_gs[1])
    handles = []

    for lab in conditions_order:
        txt = lab + " Vs " + reference
        c = class_dictionaries[lab]
        handles.append(
            mlines.Line2D(
                [0], [0], marker=".", color=c, lw=0, label=txt, markerfacecolor=c, markeredgecolor=None,
                markersize=18
            )
        )
    groups_ax_legend.legend(
        handles=handles, frameon=False, loc="center", ncols=group_legend_ncols, prop={"size": "small", "weight": "bold"}, title=groups_legend_title,
        title_fontproperties={"size": "small", "weight": "bold"}, borderaxespad=0.2, bbox_transform=groups_ax_legend.transAxes, bbox_to_anchor=(0.5, 0.5),

    )
    groups_ax_legend.axis("off")  # Hide axes for clean display
    return_ax_dict["color_group_ax"] = groups_ax
    return_ax_dict["legend_group_ax"] = groups_ax_legend


    # Add Colorbar Legend
    Colorbar(color_legend_ax, mappable=mappable, orientation="horizontal")
    color_legend_ax.xaxis.set_tick_params(labelsize="small")
    color_legend_ax.set_title(colorbar_legend_title, fontsize="small", fontweight="bold")
    return_ax_dict["legend_ax"] = color_legend_ax

    # Significance Legend
    if add_stats:
        x, y = 0, 0.5
        sig_ax.scatter(x, y, s=500, facecolors="none", edgecolors="black", marker="s")
        sig_ax.text(x, y, "*", fontsize=18, ha="center", va="center", color="black", fontfamily='DejaVu Sans Mono')
        sig_ax.text(x + 0.03, y, "FDR < 0.05", fontsize=12, va="center", fontweight="bold")
        sig_ax.set_xlim(x - 0.02, x + 0.1)
        sig_ax.set_title("Significance", fontsize="small", fontweight="bold")
        plt.gca().set_aspect("equal")
        sig_ax.axis("off")  # Hide axes for clean display
        return_ax_dict["signifiance_ax"] = sig_ax

    # Modify layout from main plot
    hm.spines[["top", "right", "bottom", "left"]].set_visible(True)
    hm.set_xlabel("")
    hm.set_ylabel("")

    rotation_props_x, rotation_props_y = {"rotation": None}, {"rotation": None}
    rotation_props_x = (
        {"rotation": xticks_rotation, "va": "top", "ha": "right"} if xticks_rotation is not None else rotation_props_x
    )
    rotation_props_y = (
        {"rotation": yticks_rotation, "va": "top", "ha": "right"} if yticks_rotation is not None else rotation_props_y
    )
    hm.set_xticklabels(hm.get_xticklabels(), fontdict={"weight": ticks_fontproperties["weight"], "size": ticks_fontproperties["size"]}, **rotation_props_x)
    hm.set_yticklabels(hm.get_yticklabels(), fontdict={"weight": ticks_fontproperties["weight"], "size": ticks_fontproperties["size"]}, **rotation_props_y)
    groups_ax.set_title(title, **title_fontprop)
    return_ax_dict["mainplot_ax"] = hm

    # Add Square around the Xs
    if add_stats:
        df_x = pd.DataFrame([], index=df.index, columns=df.columns)
        df_x[df_x.isna()] = "black"
        df_x = df.map(lambda x: square_color(colormap(normalize(x))))
        pos_rows, pos_cols = np.where(df_pvals < 0.05)
        pos = list(zip(pos_rows, pos_cols, strict=False))
        colors = [df_x.iloc[row, col] for row, col in pos]

        small_squares(
            hm,
            color=colors,
            pos=pos,
            size=square_x_size["size"],
            linewidth=square_x_size["width"],
        )

        # Now set colors manually on each annotation text base on the background
        for text, color in zip(hm.texts, df_x.values.flatten(), strict=False):
            text.set_color(color)

    if path is not None:
        plt.savefig(convert_path(path) / filename, bbox_inches="tight")
    if show:
        return plt.show()
    else:
        return return_ax_dict
