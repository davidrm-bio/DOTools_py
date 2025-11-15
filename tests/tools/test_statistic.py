import dotools_py as do
import pandas as pd


def test_rank_genes_condition():
    adata = do.dt.example_10x_processed()

    df = do.tl.rank_genes_condition(adata, groupby="condition", subset_by="annotation", reference="healthy")

    assert isinstance(df, pd.DataFrame)
    assert "rank_genes_condition" in adata.uns.keys()
    cols = {'GeneName', 'statistic', 'log2fc', 'pvals', 'padj', 'pts_group', 'pts_ref', 'group', 'annotation'}
    assert cols.issubset(df.columns)

    df = do.tl.rank_genes_condition(adata, groupby="condition", subset_by="annotation", reference="healthy", method="t-test")
    assert isinstance(df, pd.DataFrame)

    from dotools_py.tl._rankGenes import filter_rank_genes_groups
    do.tl.rank_genes_groups(adata, groupby="condition")
    filter_rank_genes_groups(adata, "rank_genes_groups")
    assert "rank_genes_groups_filtered" in adata.uns.keys()

    return None


def test_ttest():
    import random
    adata = do.dt.example_10x_processed()

    # Generate pseudoreplicates
    batches = [f"batch{i}" for i in range(1, 7)]
    adata.obs["batch_technical"] = random.choices(batches, k=adata.n_obs)

    do.tl.grouped_ttest(adata, batch_key="batch_technical")
    assert "grouped_ttest" in adata.uns.keys()
    df = adata.uns["grouped_ttest"]
    cols = {"gene", "annotation", "condition",  "pval", "statistic"}
    assert cols.issubset(df.columns)
    assert df["pval"].max() <=1
    assert df["pval"].min() >=0
    return None


def test_enrichr():
    adata = do.dt.example_10x_processed()

    do.tl.rank_genes_groups(adata, "condition")
    table = do.get.dge_results(adata)
    df = do.tl.go_analysis(table, gene_key="GeneName", pval_key="padj", log2fc_key="log2fc")

    assert isinstance(df, pd.DataFrame)
    cols = {'Gene_set', 'Term', 'Overlap', 'P-value', 'Adjusted P-value', 'Old P-value', 'Old Adjusted P-value',
            'Odds Ratio', 'Combined Score', 'Genes', 'state'}
    assert cols.issubset(df.columns)
    return None


def test_rank_genes_groups():
    adata = do.dt.example_10x_processed()

    do.tl.rank_genes_groups(adata, "condition")
    assert "rank_genes_groups" in adata.uns.keys()

    return  None


def test_pseudobulk():
    adata = do.dt.example_10x_processed()
    df = do.tl.rank_genes_pseudobulk(adata, "healthy", "disease", "annotation", technical_replicates=3)
    assert isinstance(df, pd.DataFrame)
    cols = {'baseMean', 'log2FoldChange', 'lfcSE', 'stat', 'pvalue', 'padj','group'}
    assert cols.issubset(df.columns)
    return


def test_consensus():
    adata = do.dt.example_10x_processed()
    df = do.tl.rank_genes_consensus(adata, "healthy", "disease", "annotation", technical_replicates=3)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] == 18
    return


def test_run():
    adata = do.dt.example_10x_processed()
    try:
        do.tl.run_mast(adata, "condition", "healthy", "disease")
    except Exception:
        pass
