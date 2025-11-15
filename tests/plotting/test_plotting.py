import numpy as np
import pandas as pd

import dotools_py as do
import matplotlib.pyplot as plt



def test_dotplot():
    adata = do.dt.example_10x_processed()
    axs = do.pl.dotplot(adata, x_axis="condition", features="CD4", show=False, add_stats="x_axis")
    plt.close()
    assert isinstance(axs, dict)
    for key in ["mainplot_ax", "size_legend_ax", "color_legend_ax", "significance_legend_ax"]:
        assert key in axs
    axs = do.pl.dotplot(adata, x_axis="condition", features="CD4", y_axis="annotation", show=False)
    plt.close()
    assert isinstance(axs, dict)
    for key in ["mainplot_ax", "size_legend_ax", "color_legend_ax"]:
        assert key in axs

    axs = do.pl.dotplot(adata, x_axis="condition", features={"genes":["CD4"]}, show=False, add_stats="x_axis")
    assert isinstance(axs, dict)
    assert "var_group_ax" in axs
    axs = do.pl.dotplot(adata, x_axis="condition", features="CD4", show=False, standard_scale="group")
    plt.close()
    axs = do.pl.dotplot(adata, x_axis="condition", features="CD4", y_axis="annotation",  show=False, z_score="x_axis")
    plt.close()


    return


def test_downstream():
    # SplitBarGSEA
    adata = do.dt.example_10x_processed()
    do.tl.rank_genes_groups(adata, 'condition', method='wilcoxon', tie_correct=True, pts=True)
    table = do.get.dge_results(adata)
    table = table[table.group == 'disease']
    table_go = do.tl.go_analysis(table, 'GeneName', 'padj', 'log2fc', specie='Human',
                                 go_catgs=['GO_Molecular_Function_2023', 'GO_Cellular_Component_2023',
                                           'GO_Biological_Process_2023'])
    table_go = table_go[table_go['P-value'] < 0.25]
    axs = do.pl.split_bar_gsea(table_go, 'Term', 'Combined Score', 'state', 'enriched', show=False)
    plt.close()
    assert isinstance(axs, plt.Axes)

    # Volcano
    table = do.get.dge_results(adata)
    table = table[table.group == 'disease']
    axs = do.pl.volcano_plot(table, 'log2fc', 'padj', 'GeneName', show=False)
    plt.close()
    assert isinstance(axs, dict)
    for key in ["mainplot_ax", "legend_ax"]:
        assert key in axs

    # Expr Correlation
    axs = do.pl.correlation(adata, 'batch', show=False)
    plt.close()
    assert isinstance(axs, plt.Axes)

    axs = do.pl.correlation(adata, 'batch', show=False, mask="lower")
    plt.close()
    assert isinstance(axs, plt.Axes)
    axs = do.pl.cell_composition(adata, "annotation", "condition", "batch", condition_order=["healthy", "disease"], transform="arcsin", show=False)
    plt.close()
    assert isinstance(axs,dict)

    return


def test_embeddings():
    adata = do.dt.example_10x_processed()
    axs = do.pl.umap(adata, "annotation", show=False)
    plt.close()
    assert isinstance(axs, plt.Axes)

    axs = do.pl.umap(adata, "annotation", split_by="condition", show=False, labels="annotation")
    plt.close()
    assert all(isinstance(ax, plt.Axes) for ax in np.ravel(axs))

    axs = do.pl.umap(adata, ["annotation", "CD4"], show=False, labels="annotation", share_legend=True)
    plt.close()
    assert all(isinstance(ax, plt.Axes) for ax in np.ravel(axs))

    axs = do.pl.split_embeddding(adata, "annotation", show=False)
    plt.close()
    assert all(isinstance(ax, plt.Axes) for ax in np.ravel(axs))

    return


def test_experimental():
    adata = do.dt.example_10x_processed()
    axs = do.pl.lineplot(adata, "condition", "CD4", hue="annotation", show=False)
    plt.close()
    assert isinstance(axs, dict)
    for key in ["mainplot_ax", "legend_ax"]:
        assert key in axs



def test_expression():
    adata = do.dt.example_10x_processed()
    nk = adata[adata.obs.annotation == "NK"]

    ax = do.pl.violinplot(nk, feature="CD4", x_axis="condition", reference="healthy", groups="disease", figsize=(5, 6), show=False)
    plt.close()
    assert isinstance(ax, plt.Axes)

    ax = do.pl.barplot(nk, feature="CD4", x_axis="condition", reference="healthy", groups="disease", figsize=(5, 6), show=False)
    plt.close()
    assert isinstance(ax, plt.Axes)

    ax = do.pl.boxplot(nk, feature="CD4", x_axis="condition", reference="healthy", groups="disease", figsize=(5, 6), show=False)
    plt.close()
    assert isinstance(ax, plt.Axes)

    ax = do.pl.violinplot(adata, "condition", feature="CD4", hue="annotation", show=False)
    plt.close()
    assert isinstance(ax, dict)
    assert "mainplot_ax" in ax
    assert "legend_ax" in ax

    ax = do.pl.barplot(adata, "condition", feature="CD4", hue="annotation", show=False)
    plt.close()
    assert isinstance(ax, dict)
    assert "mainplot_ax" in ax
    assert "legend_ax" in ax

    ax = do.pl.boxplot(adata, "condition", feature="CD4", hue="annotation", show=False)
    plt.close()
    assert isinstance(ax, dict)
    assert "mainplot_ax" in ax
    assert "legend_ax" in ax

    axs = do.pl.boxplot(adata, 'annotation', 'RPL11', hue='condition', reference='healthy', groups=['disease'],
                  hue_order=['healthy', 'disease'], xticks_rotation=45, figsize=(6, 4), show=False)
    plt.close()
    assert isinstance(axs, dict)
    assert "mainplot_ax" in axs
    assert "legend_ax" in axs

    axs = do.pl.boxplot(adata, 'condition', 'total_counts', reference='healthy', groups=['disease'],
                        xticks_rotation=45, figsize=(6, 4), show=False)
    plt.close()
    assert isinstance(axs, plt.Axes)

    from dotools_py.pl import StatsPlotter, TestData
    import seaborn as sns
    try:
        df = do.get.expr(adata, "CD4", "condition")
        ax = sns.barplot(df, x="condition", y="expr")
        tester = TestData(df, "expr", "condition", "healthy", ["disease"])
        tester.run_test()
        plotter = StatsPlotter(ax, "condition", "expr", "healthy", ["disease"], tester.pvals, kind="bar")
        plotter.plot_stats()
        plt.close()
    except ValueError:
        pass







def test_heatmap():
    adata = do.dt.example_10x_processed()

    axs = do.pl.heatmap(adata, group_by="annotation", features="CD4", add_stats=True, show=False)
    plt.close()
    assert isinstance(axs, dict)
    assert "mainplot_ax" in axs
    assert "legend_ax" in axs
    assert "signifiance_ax" in axs



def test_plotter():
    adata = do.dt.example_10x_processed()

    from dotools_py.pl._Plotter import MatrixDataGenerator

    plotter = MatrixDataGenerator(adata=adata, x_axis="condition", features="CD4", y_axis="annotation")
    plotter.get_expr_df()
    assert isinstance(plotter.df_expr, pd.DataFrame)
    plotter.get_pct_df()
    assert isinstance(plotter.df_pct, pd.DataFrame)
    plotter = MatrixDataGenerator(adata=adata, x_axis="condition", features="CD4",  z_score="x_axis")
    plotter.get_expr_df()
    plotter.zscore_transform()
    assert isinstance(plotter.df_zscore, pd.DataFrame)
    plotter = MatrixDataGenerator(adata=adata, x_axis="condition", features="CD4", minmax="x_axis")
    plotter.get_expr_df()
    plotter.minmax_transform()
    assert isinstance(plotter.df_minmax, pd.DataFrame)
    plotter = MatrixDataGenerator(adata=adata, x_axis="condition", features="CD4", add_stats="x_axis")
    plotter.get_expr_df()
    plotter.test_significance()
    assert isinstance(plotter.df_pvals, pd.DataFrame)
    plotter = MatrixDataGenerator(adata=adata, x_axis="condition", features="CD4", y_axis="annotation", add_stats="y_axis")
    plotter.get_expr_df()
    plotter.test_significance()
    assert isinstance(plotter.df_pvals, pd.DataFrame)
    plotter = MatrixDataGenerator(adata=adata, x_axis="condition", features="log1p_n_genes_by_counts", y_axis="annotation")
    plotter.get_pct_df()
    assert isinstance(plotter.df_expr, pd.DataFrame)
    plotter = MatrixDataGenerator(adata=adata, x_axis="condition", features="CD4", z_score="y_axis")
    plotter.get_expr_df()
    plotter.zscore_transform()
    assert isinstance(plotter.df_zscore, pd.DataFrame)
    plotter = MatrixDataGenerator(adata=adata, x_axis="condition", features="CD4", y_axis="annotation", z_score="y_axis")
    plotter.get_expr_df()
    plotter.zscore_transform()
    assert isinstance(plotter.df_zscore, pd.DataFrame)




def test_spatial():
    # TODO - Update when a test dataset is added
    adata = do.dt.example_10x_processed()
    adata.obsm["X_spatial"] = adata.obsm["X_umap"].copy()
    do.pl.layers(adata, "CD4", layers=["counts", "logcounts"], show=False, library_id=None, spot_size=1)
    plt.close()
    try:
        do.pl.slides(adata, "CD4")
    except KeyError:
        pass
    plt.close()



