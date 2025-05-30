import anndata as ad
import numpy as np
import pandas as pd
import scipy as sp

from .. import logger


def _expm1_anndata(adata: ad.AnnData) -> None:
    """Apply expm1 transformation for the X data.

    :param adata: anndata object
    :return: None, changes are inplace
    """
    if sp.issparse(adata.X):
        adata.X = adata.X.copy()
        adata.X.data = np.expm1(adata.X.data)
    else:
        adata.X = np.expm1(adata.X)


def mean_expr(
    adata: ad.AnnData,
    group_by: str,
    features: list or str = None,
    out_format: str = "long",
    layer: str = None,
) -> pd.DataFrame:
    """Calculate Average Expression in AnnData Objects for features

    This function calculates the average expression of a set of features grouping by one
    or several categories.

    :param adata: anndata object
    :param group_by: .obs column name or list of names to group by
    :param features: list of features of .var to use. (Default is **all genes**)
    :param out_format: format of the dataframe returned. This can be wide or long format. (Default  is **long**)
    :param layer: layer of the anndata to use. (Default uses **.X**)
    :return: panda DataFrame in long (or wide) format with average expression
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

    # Group data by the specified values
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


def get_expr(
    adata: ad.AnnData, features: str, groups: str = None, out_format: str = "long", layer: str = None
) -> pd.DataFrame:
    """Extract the expression of features.

    This function extract the expression from an AnnData object and returns a dataframe. If layer
    is not specified the expression in X will be extracted. Additionally, metadata from obs can be added
    to the dataframe.

    :param adata: AnnData object
    :param groups: obs metadata column to include in the dataframe
    :param features: var_names to include
    :param out_format: format of the dataframe (wide or long)
    :param layer: layer in the anndata object to extract the expression from
    :return: dataframe with expression values
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
            table_expr[groups] = adata.obs[groups]  # One column
        else:
            for group in groups:  # Multiple columns
                table_expr[group] = adata.obs[group]
    if out_format == "long":
        table_expr = pd.melt(table_expr, id_vars=groups, var_name="genes", value_name="expr")

    return table_expr


def free_memory():
    """Garbage collector.

    :return: None
    """
    import ctypes
    import gc

    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    return
