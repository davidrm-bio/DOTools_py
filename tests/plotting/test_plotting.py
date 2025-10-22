import dotools_py as do
import matplotlib.pyplot as plt



def test_dotplot():
    adata = do.dt.example_10x_processed()
    axs = do.pl.dotplot(adata, x_axis="condition", features="CD4", show=False)
    plt.close()
    assert isinstance(axs, dict)
    for key in ["mainplot_ax", "size_legend_ax", "color_legend_ax"]:
        assert key in axs
    axs = do.pl.dotplot(adata, x_axis="condition", features="CD4", y_axis="annotation", show=False)
    plt.close()
    assert isinstance(axs, dict)
    for key in ["mainplot_ax", "size_legend_ax", "color_legend_ax"]:
        assert key in axs
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
    axs = do.pl.expr_correlation(adata, 'batch', show=False)
    plt.close()
    assert isinstance(axs, plt.Axes)
    return








