
from typing import Literal
import operator

import anndata as ad
import pandas as pd
import numpy as np
import scipy as sp
from dotools_py import logger
from dotools_py.utility._general import free_memory


def _expm1_anndata(adata: ad.AnnData) -> None:
    """Apply expm1 transformation for the X dt.

    :param adata: annotated dt matrix
    :return: None, changes are inplace
    """
    if sp.sparse.issparse(adata.X):
        adata.X = adata.X.copy()
        adata.X.data = np.expm1(adata.X.data)
    else:
        adata.X = np.expm1(adata.X)


def expr(
    adata: ad.AnnData,
    features: str | list,
    groups: str | list | None = None,
    out_format: Literal["long", "wide"] = "long",
    layer: str | None = None,
) -> pd.DataFrame:
    """Extract the expression of features.

    This function extract the expression from an AnnData object and returns a dataframe. If layer
    is not specified the expression in `X` will be extracted. Additionally, metadata from `obs` can be added
    to the dataframe.

    :param adata: Annotated data matrix.
    :param groups: Metadata column in `obs` to include in the Dataframe.
    :param features: Gene names in `var_names` to include.
    :param out_format: Format of the dataframe (wide or long).
    :param layer: Layer in the anndata object to extract the expression from.
    :return: Returns a `DataFrame`.  If `out_format` is set to `wide`, the index will be cell barcodes and the column names
            will be set to the gene names. If `groups` are specified, extra columns will be added. If `out_format` is set to `long`, the following fields
            are included: `genes`, containing the gene names; `groups`, containing the groups, and `expr`, containing the mean expression values.

    Example
    -------
    >>> import dotools_py as do
    >>> adata = do.dt.example_10x_processed()
    >>> df = do.get.expr(adata, "CD4", "annotation")
    >>> df.head(5)
      annotation genes  expr
    0    B_cells   CD4   0.0
    1         NK   CD4   0.0
    2    T_cells   CD4   0.0
    3    T_cells   CD4   0.0
    4    T_cells   CD4   0.0
    >>> df = do.get.expr(adata, "CD4", "annotation", out_format="wide")
    >>> df.head(5)
                                   CD4 annotation
    CAAAGAATCAGATTGC-1-batch2  0.0    B_cells
    AGCTTCCCAGTCAACT-1-batch1  0.0         NK
    GAGAGGTTCCCTCTAG-1-batch1  0.0    T_cells
    CTAACTTCAGATCATC-1-batch1  0.0    T_cells
    CATGGTACAAACGGCA-1-batch1  0.0    T_cells

    """
    # Set-up configuration
    if features is not None:
        adata = adata[:, features]  # Retain only the specified features
    if layer is not None:
        adata.X = adata.layers[layer].copy()  # Select the specified layer

    # Check out_format specified
    assert out_format == "wide" or out_format == "long", f'{out_format} not recognize, try "long" or "wide"'
    features = [features] if isinstance(features, str) else features

    # Remove features not present and warn
    features_copy = []
    for g in features:
        if g not in list(adata.var_names):
            logger.warn(f"{g} not in adata.var_names, ignoring")
        else:
            features_copy.append(g)

    assert len(features_copy) != 0, "None of {features} in adata.var_names"
    features = features_copy

    # Extract expression
    table_expr = pd.DataFrame(
        adata[:, features].X.toarray(),  # densify the matrix (Replace .A)
        index=adata.obs_names,
        columns=features,
    )
    # Add Metadata
    if groups is not None:
        if isinstance(groups, str):
            if adata.obs[groups].dtype.name in ["category", "object"]:
                if any("-" in txt for txt in list(adata.obs[groups].cat.categories)):
                    logger.warn("Replacing '-' in groups categories by '_'")
                adata.obs[groups] = adata.obs[groups].str.replace("-", "_")
            table_expr[groups] = adata.obs[groups]  # One column
        else:
            for group in groups:  # Multiple columns
                if adata.obs[group].dtype.name in ["category", "object"]:
                    if any("-" in txt for txt in list(adata.obs[group].cat.categories)):
                        logger.warn("Replacing '-' in groups categories by '_'")
                    adata.obs[group] = adata.obs[group].str.replace("-", "_")
                table_expr[group] = adata.obs[group]
    if out_format == "long":
        table_expr = pd.melt(table_expr, id_vars=groups, var_name="genes", value_name="expr")
    free_memory()
    return table_expr


def mean_expr(
    adata: ad.AnnData,
    group_by: str | list,
    features: list | str | None = None,
    out_format: Literal["long", "wide"] = "long",
    layer: str | None = None,
    logcounts: bool = True,
) -> pd.DataFrame:
    """Calculate the average expression in an AnnData objects for features.

    This function calculates the average expression of a set of features grouping by one
    or several categories. Assume log-normalised counts. If logcounts is set to True, the
    log10 transformation is undone for the mean expression calculation. The reported mean
    expression is log-transformed.

    :param adata: Annotated data matrix.
    :param group_by: Metadata columns in `obs` to group by.
    :param features: List of features in `var_name` to use. If not set, it will be calculated over all the genes.
    :param out_format: Format of the Dataframe returned. This can be wide or long format.
    :param layer: Layer of the AnnData to use. If not set use `X`.
    :param logcounts: if set to True, the log1p transformation is undone to calculate the mean exoression.
    :return: Returns a `DataFrame`. If `out_format` is set to `wide`, the index will be set to the gene names and the
            column names will be set to the groups. If `out_format` is set to `long`, the following fields are included:
            `gene`, containing the gene names; `groupN` containing the groups (For each metadata column a new column will be added), and
            `expr`, containing the mean expression values.

    Example
    -------
    >>> import dotools_py as do
    >>> adata = do.dt.example_10x_processed()
    >>> df = do.get.mean_expr(adata, "annotation")
    >>> df.head(5)
             gene   group0      expr
    0  ATP2A1-AS1  B_cells  0.000000
    1      STK17A  B_cells  1.453713
    2    C19orf18  B_cells  0.000000
    3        TPP2  B_cells  0.126846
    4       MFSD1  B_cells  0.053630
    >>> df = do.get.mean_expr(adata, "annotation", out_format="wide")
    >>> df.head(5)
        group0   B_cells  Monocytes        NK   T_cells       pDC
    gene
    A4GALT  0.222505   0.000000  0.000000  0.000000  0.000000
    AAK1    0.000000   0.364976  1.126293  1.143016  0.128019
    ABAT    0.182251   0.146378  0.047404  0.045826  0.158761
    ABCB4   0.062785   0.000000  0.000000  0.000000  0.000000
    ABCB9   0.000000   0.000000  0.027683  0.057814  0.000000

    """
    features = [features] if isinstance(features, str) else features
    group_by = [group_by] if isinstance(group_by, str) else group_by
    assert out_format == "wide" or out_format == "long", f'{out_format} not recognize, try "long" or "wide"'

    # Set-up configuration
    if features is not None:
        adata = adata[:, features]
    if layer is not None:
        adata.X = adata.layers[layer].copy()

    data = adata.copy()

    if logcounts:
        _expm1_anndata(data)

    # Group dt by the specified values
    group_obs = adata.obs.groupby(group_by, as_index=False)

    # Compute AverageExpression
    main_df = pd.DataFrame([])
    for group_name, df in group_obs:
        if logcounts:
            df_tmp = np.log1p(
                pd.DataFrame(data[df.index].X.mean(axis=0).T, columns=["expr"])
            )  # Mean expr per gene in groupN
        else:
            df_tmp = pd.DataFrame(data[df.index].X.mean(axis=0).T, columns=["expr"])

        df_tmp["gene"] = adata[df.index].var_names  # Update with Gene names
        if type(group_name) is str:  # If only grouping by one category
            group_name = [group_name]
        for idx, name in enumerate(group_name):
            df_tmp["group" + str(idx)] = str(name).replace("-", "_")  # Update with metadata
        main_df = pd.concat([main_df, df_tmp], axis=0)
    main_df["expr"] = pd.to_numeric(main_df["expr"])  # Convert to numeric values

    # Move expr column to last position
    expr_col = main_df.pop("expr")
    main_df["expr"] = expr_col

    # Change to wide format
    if out_format == "wide":
        main_df = pd.pivot_table(
            main_df, index="gene", columns=list(main_df.columns[main_df.columns.str.startswith("group")]), values="expr"
        )
        if len(group_by) > 1:
            main_df.columns = main_df.columns.map("_".join)
    free_memory()
    return main_df


def dge_results(
    adata: ad.AnnData,
    key: str = "rank_genes_groups",
) -> pd.DataFrame:
    """Extract DEGs from AnnData object.

    This function extract the results of the DGE analysis results from the uns attribute of an AnnData object.

    :param adata: annotated data matrix.
    :param key: uns key with DGE results.
    :return: dataframe with DGE results.
    """
    import scanpy as sc

    update_columns = {
        "names": "GeneName",
        "scores": "wilcox_score",
        # U1 from formula, higher absolute indicate lower p-value; High score indicate high expression
        "pvals": "pvals",
        "group": "group",
        "logfoldchanges": "log2fc",
        "pvals_adj": "padj",
        "pct_nz_group": "pts_group",
        "pct_nz_reference": "pts_ref",
    }

    df_results = sc.get.rank_genes_groups_df(adata, group=None, key=key)
    df_results.columns = [update_columns[col] for col in df_results.columns]

    if "pts_ref" not in df_results.columns:
        result = adata.uns[key]
        ref = result["params"]["reference"]
        pts_ref = result["pts"][ref]
        if "group" in df_results and len(df_results.group.unique()) > 1:
            df_results["pts_ref"] = df_results["GeneName"].map(pts_ref)
        else:
            df_results["pts_ref"] = pts_ref.reindex(index=df_results.GeneName).tolist()
    return df_results


def subset(adata: ad.AnnData,
           obs_key: str = None,
           obs_groups: str | list | float | bool = None,
           var_key: str  = None,
           var_groups: str | list | float | bool = None,
           comparison: Literal[">=", ">", "==", "<", "<=", "include", "exclude"] = "include",
           copy: bool = False) -> ad.AnnData:
    """Subset AnnData object.

    Subset an AnnData object based on obs or var column. Currently it does not allow to subset
    by multiple obs/var columns at the same time.

    :param adata: AnnData Object.
    :param obs_key: obs column to subset for. If a list is provided, it will subset for each column.
    :param obs_groups: groups to include in the AnnData object
    :param var_key: var column to subset for
    :param var_groups: groups to include in the AnnData object
    :param comparison: comparison to used for.
    :param copy: if set to True, a copy is returned, otherwise a view of the AnnData is returned.
    :return: Returns a view or a new AnnData object.

    Example
    -------
    >>> import dotools_py as do
    >>> adata = do.dt.example_10x_processed()
    >>> tcells = subset(adata, obs_key="annotation", obs_groups="T_cells")
    >>> tcells
    View of AnnData object with n_obs × n_vars = 464 × 1851
        obs: 'batch', 'condition', 'n_genes_by_counts', 'log1p_n_genes_by_counts', 'total_counts', 'log1p_total_counts', 'total_counts_mt', 'log1p_total_counts_mt', 'pct_counts_mt', 'total_counts_ribo', 'log1p_total_counts_ribo', 'pct_counts_ribo', 'n_genes', 'n_counts', 'doublet_class', 'doublet_score', 'leiden', 'cell_type', 'autoAnnot', 'celltypist_conf_score', 'annotation', 'annotation_recluster'
        var: 'mean', 'std', 'highly_variable', 'means', 'dispersions', 'dispersions_norm', 'highly_variable_nbatches', 'highly_variable_intersection'
        uns: 'annotation_colors', 'annotation_recluster_colors', 'batch_colors', 'hvg', 'leiden', 'leiden_colors', 'log1p', 'neighbors', 'pca', 'umap'
        obsm: 'X_CCA', 'X_pca', 'X_umap'
        varm: 'PCs'
        layers: 'counts', 'logcounts'
        obsp: 'connectivities', 'distances'
    >>> adata_subset = subset(adata, obs_key="total_counts", obs_groups=1000, comparison=">=", copy=True)
    >>> adata_subset
    AnnData object with n_obs × n_vars = 699 × 1851
        obs: 'batch', 'condition', 'n_genes_by_counts', 'log1p_n_genes_by_counts', 'total_counts', 'log1p_total_counts', 'total_counts_mt', 'log1p_total_counts_mt', 'pct_counts_mt', 'total_counts_ribo', 'log1p_total_counts_ribo', 'pct_counts_ribo', 'n_genes', 'n_counts', 'doublet_class', 'doublet_score', 'leiden', 'cell_type', 'autoAnnot', 'celltypist_conf_score', 'annotation', 'annotation_recluster'
        var: 'mean', 'std', 'highly_variable', 'means', 'dispersions', 'dispersions_norm', 'highly_variable_nbatches', 'highly_variable_intersection'
        uns: 'annotation_colors', 'annotation_recluster_colors', 'batch_colors', 'hvg', 'leiden', 'leiden_colors', 'log1p', 'neighbors', 'pca', 'umap'
        obsm: 'X_CCA', 'X_pca', 'X_umap'
        varm: 'PCs'
        layers: 'counts', 'logcounts'
        obsp: 'connectivities', 'distances'

    """

    assert comparison in [">=", ">", "==", "<", "<=", "include", "exclude"], "Not a valid comparison key"
    if obs_key is not None:
        assert obs_key in adata.obs.columns, "Not a valid obs key"
    if var_key is not None:
        assert var_key in adata.var.columns, "Not a valid var key"

    if comparison in ["include", "exclude"]:
        obs_groups = [obs_groups] if isinstance(obs_groups, str) else obs_groups
        var_groups = [var_groups] if isinstance(var_groups, str) else var_groups

    operations = {
        "==": operator.eq,
        "!=": operator.ne,
        ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le
    }

    # Subset by obs
    if obs_key is not None:
        if comparison == "exclude":
            adata = adata[~adata.obs[obs_key].isin(obs_groups)]
        elif comparison == "include":
            adata = adata[adata.obs[obs_key].isin(obs_groups)]
        else:
            mask = operations[comparison](adata.obs[obs_key], obs_groups).values
            adata = adata[mask, :]

    # Subset by var
    if var_key is not None:
        if comparison == "exclude":
            adata = adata[~adata.var[var_key].isin(var_groups)]
        elif comparison == "include":
            adata = adata[adata.var[var_key].isin(var_groups)]
        else:
            mask = operations[comparison](adata.var[var_key], var_groups).values
            adata = adata[:, mask]
    if copy:
        return adata.copy()
    else:
        return adata



def log2fc(adata: ad.AnnData,
           group_by: str,
           reference: str,
           groups: str | list = None,
           features: str | list = None,
           layer: str = None,
           ) -> pd.DataFrame:
    """Calculate the log2foldchanges for a set of groups.

    :param adata: Annotated data matrix
    :param group_by: Column in `obs` to group by.
    :param reference: Reference condition to use for the calculation.
    :param groups: Alternative condiitons to use. If None, all the condiitons will be used.
    :param features: Features to use for calculating the log2foldchanges.
    :param layer: Layer in the AnnData to use for the calculation.
    :return: Returns a DataFrame with the log2foldchange. One column will be added for each condition.

    Example
    -------
    >>> import dotools_py as do
    >>> adata = do.dt.example_10x_processed()
    >>> df = do.get.log2fc(adata, group_by="condition", reference="healthy")
    >>> df.head(5)
                log2fc_disease
    gene
    A4GALT       26.073313
    AAK1         -0.429676
    ABAT          0.775196
    ABCB4       -22.599501
    ABCB9        -1.669137
    """

    features = list(adata.var_names) if features is None else features  # Calculate log2fc on all genes
    if groups is None:
        groups = list(adata.obs[group_by].unique())
        groups.remove(reference)
    elif isinstance(groups, str):
        groups = [groups]


    df_mean = mean_expr(adata, group_by=group_by, features=features, out_format="wide",  layer=layer)

    logfoldchanges = pd.DataFrame([])
    for group in groups:
        if group == reference:
            continue
        tmp = pd.DataFrame(np.log2((np.expm1(df_mean[groups[0]] + 1e-9)) /
                             (np.expm1(df_mean[reference]) + 1e-9)), columns=["log2fc_" + groups[0]])
        logfoldchanges = pd.concat([logfoldchanges, tmp], axis=1)
    logfoldchanges.index.name = None
    return logfoldchanges


def pcts_cells(adata,
               group_by: str | list,
               features: str | list = None,
               min_expr: float = 0.0,
               ) -> pd.DataFrame:
    """Calculate the percentage of cells that express a feature.

    :param adata: Annotated data matrix.
    :param group_by: Column in `obs` to group by. Several columns can be provided.
    :param features: Features to use for calculating the log2foldchanges.
    :param min_expr: Minimum value to use for the estimation of percentages.
    :return: Returns a DataFrame with the percentage of cells expressing a feature in each group.

    Example
    -------
    >>> import dotools_py as do
    >>> adata = do.dt.example_10x_processed()
    >>> df = do.get.pcts_cells(adata, group_by=["condition", "annotation"])
    >>> df.head(5)
            genes  disease_B_cells  ...  healthy_T_cells  healthy_pDC
    0  ATP2A1-AS1             0.00  ...             0.01         0.00
    1      STK17A             0.57  ...             0.49         0.17
    2    C19orf18             0.00  ...             0.00         0.00
    3        TPP2             0.03  ...             0.18         0.17
    4       MFSD1             0.03  ...             0.06         0.50
    [5 rows x 11 columns]

    """

    features = list(adata.var_names) if features is None else features  # Calculate log2fc on all genes

    df_expr = expr(
        adata, features=features, groups=group_by, out_format="wide"
    ).set_index(group_by)

    obs_bool = df_expr > min_expr
    df_pct = (
        obs_bool.groupby(level=group_by, observed=True).sum()
        / obs_bool.groupby(level=group_by, observed=True).count()
    ).T
    if isinstance(group_by, list):
        if len(group_by) > 1:
            df_pct.columns = ["_".join(col) for col in list(df_pct.columns)]
    df_pct = df_pct.round(2)
    df_pct.reset_index(inplace=True)
    df_pct.rename(columns={"index":"genes"}, inplace=True)

    return  df_pct




