
from typing import Literal

import anndata as ad
import pandas as pd
import numpy as np

from dotools_py import logger
from dotools_py.tl._get_stats import _expm1_anndata


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
                adata.obs[groups] = adata.obs[groups].str.replace("-", "_")
            table_expr[groups] = adata.obs[groups]  # One column
        else:
            for group in groups:  # Multiple columns
                if adata.obs[group].dtype.name in ["category", "object"]:
                    adata.obs[group] = adata.obs[group].str.replace("-", "_")
                table_expr[group] = adata.obs[group]
    if out_format == "long":
        table_expr = pd.melt(table_expr, id_vars=groups, var_name="genes", value_name="expr")

    return table_expr



def mean_expr(
    adata: ad.AnnData,
    group_by: str,
    features: list | str | None = None,
    out_format: Literal["long", "wide"] = "long",
    layer: str | None = None,
) -> pd.DataFrame:
    """Calculate the average expression in an AnnData objects for features.

    This function calculates the average expression of a set of features grouping by one
    or several categories. Assume log-normalised counts.

    :param adata: Annotated data matrix.
    :param group_by: Metadata columns in `obs` to group by.
    :param features: List of features in `var_name` to use. If not set, it will be calculated over all the genes.
    :param out_format: Format of the Dataframe returned. This can be wide or long format.
    :param layer: Layer of the AnnData to use. If not set use `X`.
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
    _expm1_anndata(data)

    # Group dt by the specified values
    group_obs = adata.obs.groupby(group_by, as_index=False)

    # Compute AverageExpression
    main_df = pd.DataFrame([])
    for group_name, df in group_obs:
        df_tmp = np.log1p(
            pd.DataFrame(data[df.index].X.mean(axis=0).T, columns=["expr"])
        )  # Mean expr per gene in groupN
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

