import dotools_py as do
import pandas as pd
import anndata as ad


def test_dummy():
    from dotools_py.get._generic import _expm1_anndata
    adata = do.dt.example_10x_processed()
    adata.X = adata.X.toarray()
    max_expr = adata.X.max()
    _expm1_anndata(adata)
    assert adata.X.max() > max_expr


def test_expr():
    adata = do.dt.example_10x_processed()
    # Check dummy
    df = do.get.expr(adata, features=None, out_format="wide")
    assert df.shape[1] == adata.n_vars

    # Check the long format output
    df = do.get.expr(adata, "CD4", "annotation", layer="logcounts")
    assert isinstance(df, pd.DataFrame)
    cols = {"annotation", "genes", "expr"}
    assert cols.issubset(df.columns)

    # Check the wide format output
    df = do.get.expr(adata, "CD4", "annotation", out_format="wide")
    assert isinstance(df, pd.DataFrame)
    cols = {"annotation", "CD4"}
    assert cols.issubset(df.columns)

    # Check that the expression matches
    adata_cd4 = adata[:, "CD4"]
    df_cd4 = adata_cd4.to_df()
    assert df_cd4["CD4"].equals(df_cd4["CD4"])
    return None


def test_mean_expr():
    adata = do.dt.example_10x_processed()

    # Check the long format
    df = do.get.mean_expr(adata, "annotation", layer="logcounts")
    assert isinstance(df, pd.DataFrame)
    cols = {"gene", "annotation", "expr"}
    assert cols.issubset(df.columns)

    # Check the wide format
    df = do.get.mean_expr(adata, ["annotation", "condition"], out_format="wide")
    assert isinstance(df, pd.DataFrame)
    cols = set([annot + "_" + cond for annot in set(adata.obs["annotation"].unique()) for cond in adata.obs.condition.unique()])
    assert cols.issubset(df.columns)

    # Check without logmean
    df = do.get.mean_expr(adata, "annotation", layer="logcounts", logmean=False, logcounts=True)
    assert  df.expr.max() >  do.get.mean_expr(adata, "annotation", layer="logcounts", logmean=True, logcounts=True).expr.max()
    return  None


def test_subset():
    adata = do.dt.example_10x_processed()

    # Subset by Obs_key and Generate a View - Include
    adata_subset = do.get.subset(adata, obs_key="annotation", obs_groups=["B_cells", "T_cells"], comparison="include")
    assert adata_subset.isview
    assert len(list(adata_subset.obs["annotation"].unique())) == 2

    # Subset by Obs_key and Generate a Copy - Exclude
    adata_subset = do.get.subset(adata, obs_key="annotation", obs_groups=["B_cells", "T_cells"], comparison="exclude")
    assert isinstance(adata, ad.AnnData)
    n_cts = len(adata.obs["annotation"].unique())
    assert len(list(adata_subset.obs["annotation"].unique())) == n_cts - 2

    # Subset by Obs_key - >=
    adata_subset = do.get.subset(adata, obs_key="total_counts", obs_groups=100_000, comparison=">=")
    assert adata_subset.n_obs == 0

    # Subset by Var_key
    adata_subset = do.get.subset(adata, var_key="mean", var_groups=100, comparison=">")
    assert adata_subset.n_vars == 0
    adata.var["tmp"] = ["groupA"]*1000 + ["groupB"]*851
    adata_subset = do.get.subset(adata, var_key="tmp", var_groups="groupA", comparison="exclude")
    assert adata_subset.var["tmp"].unique()[0] == "groupB"
    adata_subset = do.get.subset(adata, var_key="tmp", var_groups="groupA", comparison="include")
    assert adata_subset.var["tmp"].unique()[0] == "groupA"
    return  None


def test_log2fc():
    adata = do.dt.example_10x_processed()

    df = do.get.log2fc(adata, "condition", "healthy", "disease")
    assert isinstance(df, pd.DataFrame)
    cols = {"genes", "log2fc_disease"}
    assert cols.issubset(df.columns)
    return None


def test_pts():
    adata = do.dt.example_10x_processed()

    df = do.get.pcts_cells(adata, ["condition"])
    assert isinstance(df, pd.DataFrame)
    cols = {"genes", "disease", "healthy"}
    assert cols.issubset(df.columns)
    # Values should be between 0 and 1
    assert df[["healthy", "disease"]].max().max() <= 1
    assert df[["healthy", "disease"]].min().min() >= 0
    return  None


def test_get_dge_table():
    adata = do.dt.example_10x_processed()
    # Do DGE
    do.tl.rank_genes_groups(adata, groupby="annotation")
    table = do.get.dge_results(adata)
    assert isinstance(table, pd.DataFrame)
    cols = {'group', 'GeneName', 'statistic', 'log2fc', 'pvals', 'padj', 'pts_group', 'pts_ref'}
    assert cols.issubset(table.columns)
    table = do.get.dge_results(adata, min_log2fc=0, max_log2fc=0.5, padj_cutoff=0.01)
    assert table.log2fc.min() > 0
    assert table.log2fc.max() < 0.5
    assert table.padj.max() < 0.01

    return None


def test_pseudobulk():
    adata = do.dt.example_10x_processed()
    pdata = do.get.pseudobulk(adata, "condition", "annotation", min_cells=0, min_counts=0, technical_replicates=2, workers=1)
    assert isinstance(pdata, ad.AnnData)
    assert pdata.n_obs == 19
    return None

def test_subset_df():
    adata = do.dt.example_10x_processed()

    df = adata.obs.copy()
    # Subset by Obs_key and Generate a View - Include
    df_subset = do.get.subset_df(df, col="annotation", col_groups=["B_cells", "T_cells"], comparison="include")
    assert df_subset["annotation"].nunique() == 2

    # Subset by Obs_key and Generate a Copy - Exclude
    df_subset = do.get.subset_df(df, col="annotation", col_groups=["B_cells", "T_cells"], comparison="exclude")
    assert df_subset["annotation"].nunique() ==  df["annotation"].nunique() - 2

    # Subset by Obs_key - >=
    df_subset = do.get.subset_df(df, col="total_counts", col_groups=100_000, comparison=">=")
    assert df_subset.shape[0] == 0
    df_subset = do.get.subset_df(df, col=["total_counts", "annotation"], col_groups=[1000, ["B_cells", "T_cells"]],
                         comparison=["<", "include"])
    assert df_subset.shape[0] == 1
    df_subset = do.get.subset_df(df, col=["total_counts", "annotation"], col_groups=[1000, ["B_cells", "T_cells"]],
                                 comparison=[">", "exclude"])
    assert df_subset.shape[0] == 185

def test_metacells():
    adata = do.dt.example_10x_processed()

    pdata = do.get.metacells(
        adata, batch_key="batch", annotation_key="annotation", layer="counts"
    )

    assert pdata.n_obs < adata.n_obs





