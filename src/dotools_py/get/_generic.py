from typing import Literal

import anndata as ad
import pandas as pd
import numpy as np
from numba import njit

from dotools_py import logger
from dotools_py.utility._general import free_memory
from dotools_py.utils import sanitize_anndata, iterase_input, check_missing
from dotools_py.settings._global_parameters import FAST_ARRAY_UTILS

if FAST_ARRAY_UTILS:
    import fast_array_utils


def _expm1_anndata(adata: ad.AnnData) -> None:
    """Apply expm1 transformation for the X data.

    :param adata: annotated data matrix
    :return: None, changes are inplace
    """
    import scipy as sp

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

    This function extract the expression from an AnnData object and returns a DataFrame. If layer
    is not specified the expression in `X` will be extracted. Additionally, metadata from `obs` can be added
    to the dataframe.

    :param adata: Annotated data matrix.
    :param groups: Metadata column in `obs` to include in the DataFrame.
    :param features: Name of the features in `var_names` to extract the expression of.
    :param out_format: Format of the dataframe. The `wide` format will generate a DataFrame with shape n_obs x n_vars,
                      while the `long` format will generate an unpivot version.
    :param layer: Layer in the AnnData object to extract the expression from. If set to `None` the expression in
                  `X` will be used.

    Returns
    -------
    Returns a `DataFrame`. If `out_format` is set to `wide`, the index will be the cell barcodes and the column names
    will be set to the gene names. If `groups` are specified, extra columns will be present. If `out_format` is set to
    `long`, the following fields are included:

    `genes`
        Contains the gene names.
    `expr`
        Contains the expression values extracted.

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
    sanitize_anndata(adata)
    features = iterase_input(features)
    groups = iterase_input(groups)
    assert len(features) != 0, "No features provided"
    assert out_format == "wide" or out_format == "long", f'{out_format} not recognize, try "long" or "wide"'

    # Remove features not present and warn
    check_missing(adata, features=features, groups=groups)

    # Set-up configuration
    adata = adata[:, features]  # Retain only the specified features
    if layer is not None:
        adata.X = adata.layers[layer].copy()  # Select the specified layer

    # Extract expression
    table_expr = adata.to_df().copy()

    # Add Metadata
    if groups is not None:
        table_expr[groups] = adata.obs[groups]
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
    :param logcounts: Set to `True` if the input is in log space.

    Returns
    -------
    Returns a `DataFrame` with the mean expression in log1p transformation. If `out_format` is set to `wide`, the index
    will be set to the gene names and the column names will be set to the groups. If `out_format` is set to `long`,
    the following fields are included:

    `gene`
        Contains the gene names.
    `groupN`
        Contains the groups (For each metadata column a new column will be added).
    `expr`
        Contains the mean expression values after log1p transformation.

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

    sanitize_anndata(adata)
    features = list(adata.var_names) if len(iterase_input(features)) == 0 else iterase_input(features)
    group_by = iterase_input(group_by)
    check_missing(adata, features=features, groups=group_by)
    assert out_format == "wide" or out_format == "long", f'{out_format} not recognize, try "long" or "wide"'

    # Set-up configuration
    adata = adata[:, features]
    if layer is not None:
        adata.X = adata.layers[layer].copy()

    data = adata.copy()

    if logcounts:
        _expm1_anndata(data)

    # Group data by the specified values
    group_obs = adata.obs.groupby(group_by, as_index=False)

    # Compute AverageExpression
    main_df = pd.DataFrame([])
    for group_name, df in group_obs:
        if FAST_ARRAY_UTILS:
            current_mean = fast_array_utils.stats.mean(data[df.index].X, axis=0)
            current_mean = np.log1p(current_mean) if logcounts else current_mean
            df_tmp = pd.DataFrame(current_mean, columns=["expr"])
        else:
            if logcounts:
                df_tmp = np.log1p(
                    pd.DataFrame(data[df.index].X.mean(axis=0).T, columns=["expr"])
                )  # Mean expr per gene in groupN
            else:
                df_tmp = pd.DataFrame(data[df.index].X.mean(axis=0).T, columns=["expr"])

        df_tmp["gene"] = adata[df.index].var_names  # Update with Gene names
        group_name = iterase_input(group_name)
        for idx, name in enumerate(group_name):
            # df_tmp["group" + str(idx)] = str(name).replace("-", "_")  # Update with metadata
            df_tmp[group_by[idx]] = name
        main_df = pd.concat([main_df, df_tmp], axis=0)
    main_df["expr"] = pd.to_numeric(main_df["expr"])  # Convert to numeric values

    # Move expr column to last position
    expr_col = main_df.pop("expr")
    main_df["expr"] = expr_col

    # Change to wide format
    if out_format == "wide":
        main_df = pd.pivot_table(main_df, index="gene", columns=group_by, values="expr")
        if len(group_by) > 1:
            main_df.columns = main_df.columns.map("_".join)
    free_memory()
    return main_df


def dge_results(
    adata: ad.AnnData,
    key: str = "rank_genes_groups",
) -> pd.DataFrame:
    """Extract DEGs from AnnData object.

    This function extract the results of the differential gene expression analysis results from the `uns`
    attribute of an AnnData object.

    :param adata: Annotated data matrix.
    :param key: Key in `uns` with DGE results.

    Returns
    -------
    Returns a DataFrame with the results of the differential gene expression analysis generated from
    `rank_genes_groups`.

    Example
    -------
    >>> import dotools_py as do
    >>> adata = do.dt.example_10x_processed()
    >>> do.tl.rank_genes_groups(adata, groupby="condition")
    >>> df = do.get.dge_results(adata)
    >>> df.head(5)
             group GeneName  wilcox_score  ...          padj  pts_group   pts_ref
    0  disease   ZNF331     15.936105  ...  6.586861e-54   0.650000  0.096154
    1  disease      EZR     15.871257  ...  9.274798e-54   0.866667  0.367308
    2  disease     EIF1     14.823599  ...  6.361829e-47   0.994444  0.994231
    3  disease     SRGN     14.721976  ...  2.155706e-46   0.922222  0.636538
    4  disease     EGR1     12.330428  ...  1.916262e-32   0.316667  0.011538
    [5 rows x 8 columns]

    """
    import scanpy as sc

    update_columns = {
        "names": "GeneName",
        "scores": "statistic",
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
    try:
        if "pts_ref" not in df_results.columns:
            result = adata.uns[key]
            ref = result["params"]["reference"]
            pts_ref = result["pts"][ref]
            if "group" in df_results.columns and len(df_results.group.unique()) > 1:
                df_results["pts_ref"] = df_results["GeneName"].map(pts_ref)
            else:
                df_results["pts_ref"] = pts_ref.reindex(index=df_results.GeneName).tolist()
    except KeyError as e:
        logger.warn(f"Problem generating the DGE Table: {e}")
    return df_results


def subset(adata: ad.AnnData,
           obs_key: str | None = None,
           obs_groups: str | list | float | bool | None = None,
           var_key: str | None = None,
           var_groups: str | list | float | bool | None = None,
           comparison: Literal[">=", ">", "==", "<", "<=", "include", "exclude"] = "include",
           copy: bool = False) -> ad.AnnData:
    """Subset AnnData object.

    Subset an AnnData object based on `obs` or `var` column. Currently it does not allow to subset
    by multiple obs/var columns at the same time.

    :param adata: AnnData Object.
    :param obs_key: Column in `obs` to subset for.
    :param obs_groups: Groups or values to include or filter for the AnnData object.
    :param var_key: Column in `var` to subset for.
    :param var_groups: Groups or values to include or filter for in the AnnData object.
    :param comparison: Method to filter the AnnData object.
    :param copy: if set to `True`, a copy is returned, otherwise a view of the AnnData is returned.
    :return: Returns a view or a new AnnData object.

    Returns
    -------
    Returns an AnnData Object if copy is set to `True`, otherwise returns a View of an AnnData after subsetting.

    Example
    -------
    >>> import dotools_py as do
    >>> adata = do.dt.example_10x_processed()
    >>> tcells = do.get.subset(adata, obs_key="annotation", obs_groups="T_cells")
    >>> tcells
    View of AnnData object with n_obs × n_vars = 464 × 1851
        obs: 'batch', 'condition', 'n_genes_by_counts', 'log1p_n_genes_by_counts', 'total_counts', 'log1p_total_counts', 'total_counts_mt', 'log1p_total_counts_mt', 'pct_counts_mt', 'total_counts_ribo', 'log1p_total_counts_ribo', 'pct_counts_ribo', 'n_genes', 'n_counts', 'doublet_class', 'doublet_score', 'leiden', 'cell_type', 'autoAnnot', 'celltypist_conf_score', 'annotation', 'annotation_recluster'
        var: 'mean', 'std', 'highly_variable', 'means', 'dispersions', 'dispersions_norm', 'highly_variable_nbatches', 'highly_variable_intersection'
        uns: 'annotation_colors', 'annotation_recluster_colors', 'batch_colors', 'hvg', 'leiden', 'leiden_colors', 'log1p', 'neighbors', 'pca', 'umap'
        obsm: 'X_CCA', 'X_pca', 'X_umap'
        varm: 'PCs'
        layers: 'counts', 'logcounts'
        obsp: 'connectivities', 'distances'
    >>> adata_subset = do.get.subset(adata, obs_key="total_counts", obs_groups=1000, comparison=">=", copy=True)
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
    import operator

    sanitize_anndata(adata)
    check_missing(adata, groups=obs_key, variables=var_key)
    assert comparison in [">=", ">", "==", "<", "<=", "include", "exclude"], "Not a valid comparison key"

    if comparison in ["include", "exclude"]:
        obs_groups = iterase_input(obs_groups)
        var_groups = iterase_input(var_groups)

    operations = {"==": operator.eq, "!=": operator.ne,
                  ">": operator.gt, ">=": operator.ge,
                  "<": operator.lt, "<=": operator.le}

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
            adata = adata[:, ~adata.var[var_key].isin(var_groups)]
        elif comparison == "include":
            adata = adata[:, adata.var[var_key].isin(var_groups)]
        else:
            mask = operations[comparison](adata.var[var_key], var_groups).values
            adata = adata[:, mask]
    if copy:
        return adata.copy()
    else:
        return adata


@njit(parallel=True)
def _get_log2fc(group: np.ndarray, ref: np.ndarray, psc=1e-9):
    return np.log2((np.expm1(group) + psc) / (np.expm1(ref) + psc))


def log2fc(
    adata: ad.AnnData,
    group_by: str,
    reference: str,
    groups: str | list | None = None,
    features: str | list | None = None,
    layer: str | None = None,
) -> pd.DataFrame:
    """Calculate the log2foldchanges for a set of groups.

    :param adata: Annotated data matrix.
    :param group_by: Column in `obs` to group by.
    :param reference: Reference condition to use for the calculation.
    :param groups: Alternative condititons to use. If `None`, all the condititons will be used.
    :param features: Features to use for calculating the log2foldchanges. If set to `None` all features will be used.
    :param layer: Layer in the AnnData to use for the calculation.

    Returns
    -------
    Returns a DataFrame with the log2-foldchanges. One column will be added for each condition in `groups`

    Example
    -------
    >>> import dotools_py as do
    >>> adata = do.dt.example_10x_processed()
    >>> df = do.get.log2fc(adata, group_by="condition", reference="healthy")
    >>> df.head(5)
            genes  log2fc_disease
    0  ATP2A1-AS1       26.073313
    1      STK17A       -0.429677
    2    C19orf18        0.775196
    3        TPP2      -22.599501
    4       MFSD1       -1.669137

    """

    # Get the data
    features = iterase_input(features)
    features = list(adata.var_names) if len(iterase_input(features)) == 0 else iterase_input(features)
    groups = list(adata.obs[group_by].unique()) if len(iterase_input(groups)) == 0 else iterase_input(groups)
    if reference in groups:
        groups.remove(reference)

    df_mean = mean_expr(adata, group_by=group_by, features=features, out_format="wide", layer=layer)

    logfoldchanges = pd.DataFrame([], index=features)
    for group in groups:
        # Speed up with numba
        foldchanges = _get_log2fc(group=df_mean[group].to_numpy(), ref=df_mean[reference].to_numpy())
        logfoldchanges["log2fc_" + group] = foldchanges
    logfoldchanges.reset_index(inplace=True)
    logfoldchanges.rename(columns={"index": "genes"}, inplace=True)
    return logfoldchanges


def pcts_cells(
    adata,
    group_by: str | list,
    features: str | list = None,
    min_expr: float = 0.0,
) -> pd.DataFrame:
    """Calculate the percentage of cells that express a feature.

    :param adata: Annotated data matrix.
    :param group_by: Column in `obs` to group by. Several columns can be provided.
    :param features: Features to use for the calculation. If set to `None`, all features will be used.
    :param min_expr: Minimum value to use for the estimation of percentages.

    Returns
    -------
    Returns a DataFrame with the percentage of cells expressing a feature in each group.

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
    features = list(adata.var_names) if len(iterase_input(features)) == 0 else iterase_input(features)
    group_by = iterase_input(group_by)
    df_expr = expr(
        adata, features=features, groups=group_by, out_format="wide"
    ).set_index(group_by)

    obs_bool = df_expr > min_expr
    df_pct = (
        obs_bool.groupby(level=group_by, observed=True).sum()
        / obs_bool.groupby(level=group_by, observed=True).count()
    ).T
    if len(group_by) > 1:
        df_pct.columns = ["_".join(col) for col in list(df_pct.columns)]
    df_pct = df_pct.round(4)
    df_pct.reset_index(inplace=True)
    df_pct.rename(columns={"index": "genes"}, inplace=True)
    return df_pct


def pseudobulk(
    adata: ad.AnnData,
    batch_key: str,
    cluster_key: str,
    keep_metadata: list = None,
    min_cells: int = 10,
    pseudobulk_approach: Literal["sum", "mean"] = "sum",
    technical_replicates: int = 1,
    min_counts: int = 10,
    layer: str = None,
    workers: int = 5,
) -> ad.AnnData:
    """Generate pseudobulk AnnData of clusters.

    Generate a pseudobulk AnnData for each cluster, the input is expected to be raw counts. To generate
    the pseudobulk AnnData object two modes for aggregating the counts can be used: `sum` or `mean`. Additionally,
    pseudo-replicates can be generated if specified.

    :param adata: Annotated data matrix.
    :param batch_key: Metadata column in `obs` with batch groups.
    :param cluster_key: Metadata column in `obs` with cluster groups.
    :param keep_metadata: Metadata in `obs` to keep. If more than one value is available for a group the first one is taken.
    :param min_cells: Minimum number of cells in a cluster for each sample in order to generate a pseudobulk.
                      If the cluster has less it will be excluded.
    :param pseudobulk_approach: Mode of aggregations.
    :param technical_replicates: Number of technical replicates to generate.
    :param min_counts: Minimum number of counts for a gene to be included.
    :param layer: Layer to use.
    :param workers: Number of theads to use to parallelize the pseudo-bulking
    :return: AnnData with pseudobulk counts for each cluster.

    Example
    -------
    >>> import dotools_py as do
    >>> adata = do.dt.example_10x_processed()
    >>> pdata = do.get.pseudobulk(adata, batch_key="batch", cluster_key="annotation")
    Pseudo-bulked groups: 100%|██████████| 10/10 [00:08<00:00,  1.19it/s]
    OMP: Info #276: omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.
    2025-08-01 16:41:13,927 - Removed 796 genes for having less than 10 total counts
    >>> pdata
    AnnData object with n_obs × n_vars = 7 × 1055
        obs: 'annotation', 'batch', 'n_genes_by_counts', 'log1p_n_genes_by_counts', 'total_counts',
             'log1p_total_counts', 'pct_counts_in_top_50_genes', 'pct_counts_in_top_100_genes',
             'pct_counts_in_top_200_genes', 'pct_counts_in_top_500_genes'
        var: 'n_cells_by_counts', 'mean_counts', 'log1p_mean_counts', 'pct_dropout_by_counts', 'total_counts',
             'log1p_total_counts'

    """
    import polars as pl
    from joblib import Parallel, delayed
    import scipy.sparse as sp
    import gc
    import scanpy as sc
    from tqdm import tqdm
    import random

    keep_metadata = [] if keep_metadata is None else keep_metadata
    keep_metadata = [keep_metadata] if isinstance(keep_metadata, str) else keep_metadata
    keep_metadata = keep_metadata + [cluster_key, batch_key]

    # Define the groups to pseudo-bulk
    groups = adata.obs[[cluster_key, batch_key]]
    groups = groups.groupby([cluster_key, batch_key])
    groups = groups.groups

    # Create dictionary to specify how to pseudobulk
    aggregate_info = dict.fromkeys(adata.var_names, pseudobulk_approach)
    aggregate_info.update(dict.fromkeys(keep_metadata, "first"))
    del aggregate_info[batch_key]
    agg_exprs = [getattr(pl.col(col), func)().alias(col) for col, func in aggregate_info.items()]

    def _process_group(
        cluster: str,
        batch: str,
        bcs: list
    ) -> pl.DataFrame | None:
        sdata = adata[adata.obs_names.isin(bcs), :]

        if sdata.n_obs < min_cells:
            logger.info(f"Excluding {cluster} in {batch} for having less than {min_cells}")
            return None

        if technical_replicates == 1:
            mtx = sdata.to_df(layer=layer)
            mtx[keep_metadata] = sdata.obs[keep_metadata].values
            mtx_pl = pl.from_pandas(mtx).group_by(batch_key).agg(agg_exprs)
        else:
            # Generate technical replicates
            random.shuffle(bcs)
            idx = np.array_split(np.array(bcs), technical_replicates)
            mtx_pl = None
            for i, replicate in enumerate(idx):
                batch_replicate = sdata[replicate, :]
                mtx = batch_replicate.to_df(layer=layer)
                mtx[keep_metadata] = batch_replicate.obs[keep_metadata].values

                mtx_pl_tmp = pl.from_pandas(mtx)

                # mtx_pl_tmp[batch_key] = mtx_pl_tmp[batch_key] + "_" + str(i)

                mtx_pl_tmp = mtx_pl_tmp.with_columns(
                    (pl.col(batch_key) + "_" + pl.lit(str(i))).alias(batch_key)).group_by(batch_key).agg(agg_exprs)
                if mtx_pl is None:
                    mtx_pl = mtx_pl_tmp
                else:
                    mtx_pl = pl.concat([mtx_pl, mtx_pl_tmp], how="vertical")
                # mtx_pl = pl.concat([mtx_pl_tmp, mtx_pl], how="vertical")
        gc.collect()
        return mtx_pl

    # Process groups in parallel
    with Parallel(n_jobs=workers, backend="loky") as parallel:
        results = parallel(
            delayed(_process_group)(cluster, batch, list(bcs))
            for (cluster, batch), bcs in tqdm(groups.items(), desc="Pseudo-bulked groups")
        )

    # Generate the pseudo-bulked AnnData
    df_main = [res for res in results if res is not None]
    df_main = pl.concat(df_main, how="vertical")
    df_main = df_main.to_pandas()
    df_main = df_main.fillna(0)  # Make sure we do not have NaNs

    pdata = ad.AnnData(df_main[adata.var_names], obs=df_main[keep_metadata])
    pdata.obs_names = ['psBC-' + str(idx) for idx in range(pdata.n_obs)]
    pdata.var_names_make_unique()
    pdata.X = np.nan_to_num(pdata.X, nan=0)
    if not sp.isspmatrix_csr(pdata.X):
        pdata.X = sp.csr_matrix(pdata.X)

    # Remove genes that have low amount of counts
    sc.pp.calculate_qc_metrics(pdata, inplace=True)
    n_vars = pdata.n_vars
    pdata = pdata[:, pdata.var.total_counts > min_counts].copy()
    n_vars = n_vars - pdata.n_vars
    logger.info(f"Removed {n_vars} genes for having less than {min_counts} total counts")
    return pdata
