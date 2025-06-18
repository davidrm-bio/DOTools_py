from pathlib import Path
import os
import subprocess
import uuid
import itertools
from typing import Union, Literal

import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd
import scipy as sp
import gseapy

from dotools_py import logger
from dotools_py.utils import get_paths_utils, sanitize_anndata, convert_path
from dotools_py.tl import rank_genes_groups
from scipy.stats import ttest_ind


def _expm1_anndata(
    adata: ad.AnnData
) -> None:
    """Apply expm1 transformation for the X dt.

    :param adata: annotated dt matrix
    :return: None, changes are inplace
    """
    if sp.sparse.issparse(adata.X):
        adata.X = adata.X.copy()
        adata.X.data = np.expm1(adata.X.data)
    else:
        adata.X = np.expm1(adata.X)


def mean_expr(
    adata: ad.AnnData,
    group_by: str,
    features: Union[list, str, None] = None,
    out_format: Literal['long', 'wide'] = "long",
    layer: Union[str, None] = None,
) -> pd.DataFrame:
    """Calculate the average expression in an annData objects for features.

    This function calculates the average expression of a set of features grouping by one
    or several categories.

    :param adata: annotated data matrix
    :param group_by: `.obs` column or list of columns to group by.
    :param features: list of features in `.var` to use. If not set, it will be calculated over all the genes.
    :param out_format: format of the dataframe returned. This can be wide or long format.
    :param layer: layer of the anndata to use. If not set use `.X`.
    :return: DataFrame in long (or wide) format with average expression
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


def get_expr(
    adata: ad.AnnData,
    features: str,
    groups: Union[str, None] = None,
    out_format: Literal['long', 'wide'] = "long",
    layer: Union[str, None] = None
) -> pd.DataFrame:
    """Extract the expression of features.

    This function extract the expression from an AnnData object and returns a dataframe. If layer
    is not specified the expression in `.X` will be extracted. Additionally, metadata from `.obs` can be added
    to the dataframe.

    :param adata: annotated data matrix.
    :param groups: `.obs` metadata column to include in the dataframe.
    :param features: var_names to include.
    :param out_format: format of the dataframe (wide or long).
    :param layer: layer in the anndata object to extract the expression from.
    :return: dataframe with expression values.
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
            if adata.obs[groups].dtype.name in ['category', 'object']:
                adata.obs[groups] = adata.obs[groups].str.replace('-', '_')
            table_expr[groups] = adata.obs[groups]  # One column
        else:
            for group in groups:  # Multiple columns
                if adata.obs[group].dtype.name in ['category', 'object']:
                    adata.obs[group] = adata.obs[group].str.replace('-', '_')
                table_expr[group] = adata.obs[group]
    if out_format == "long":
        table_expr = pd.melt(table_expr, id_vars=groups, var_name="genes", value_name="expr")

    return table_expr


# DGE Analysis
def run_mast(
    adata: ad.AnnData,
    cond_key: str,
    reference: str,
    disease: Union[str, list],
    covariates: Union[str, list, None] = None
) -> pd.DataFrame:
    """Run MAST Test for sc/snRNAseq.

    :param adata: annotated data matrix.
    :param cond_key: obs column with condition information.
    :param reference: reference condition.
    :param disease: disease conditions.
    :param covariates: extra covariates to account for.
    :return: pandas dataframe with DGEs.
    """

    rscript = get_paths_utils("_Run_MAST.R")

    tmpdir_path = Path("/tmp") / f"MAST_Test_{uuid.uuid4().hex}"
    tmpdir_path.mkdir(parents=True, exist_ok=False)

    logger.info("Preprocessing to R")
    del adata.uns, adata.raw
    adata.write(tmpdir_path / "adata.h5ad")

    logger.info("Running MAST Integration")
    in_path = os.path.join(tmpdir_path, "adata.h5ad")

    disease = [disease] if isinstance(disease, str) else disease

    dge_main = pd.DataFrame()
    for alternative in disease:
        logger.info(f"Running test for {alternative}")

        cmd = ["Rscript",
               rscript,
               "--input=" + in_path,
               "--out=" + str(tmpdir_path) + "/dge_mast.csv",
               "--key=" + cond_key,
               "--ref=" + reference,
               "--disease=" + alternative
               ]
        cmd += ["--covariates=" + covariates] if covariates is not None else []
        subprocess.call(cmd)
        dge = pd.read_csv(os.path.join(tmpdir_path, "dge_mast.csv"))
        dge['groups'] = alternative
        dge_main = pd.concat([dge_main, dge])
    return dge_main


def generate_results(adata: ad.AnnData,
                     key: str = 'rank_genes_groups',
                     ) -> pd.DataFrame:
    """Extract DEGs from AnnData object.

    This function extract the results of the DGE analysis results from the uns attribute of an AnnData object.

    :param adata: annotated data matrix.
    :param key: uns key with DGE results.
    :return: dataframe with DGE results.
    """

    update_columns = {'names': 'GeneName',
                      'scores': 'wilcox_score',
                      # U1 from formula, higher absolute indicate lower p-value; High score indicate high expression
                      'pvals': 'pvals',
                      'group': 'group',
                      'logfoldchanges': 'log2fc',
                      'pvals_adj': 'padj',
                      'pct_nz_group': 'pts_group',
                      'pct_nz_reference': 'pts_ref'
                      }

    df_results = sc.get.rank_genes_groups_df(adata, group=None, key=key)
    df_results.columns = [update_columns[col] for col in df_results.columns]

    if 'pts_ref' not in df_results.columns:
        result = adata.uns[key]
        ref = result['params']['reference']
        pts_ref = result['pts'][ref]
        if 'group' in df_results and len(df_results.group.unique()) > 1:
            df_results['pts_ref'] = df_results['GeneName'].map(pts_ref)
        else:
            df_results['pts_ref'] = pts_ref.reindex(index=df_results.GeneName).tolist()
    return df_results


def _run_test(
    adata: ad.AnnData,
    method: str,
    groupby: str,
    reference: str,
    groups: list,
    covariates: list
) -> pd.DataFrame:
    """Run DGE test.

    :param adata: annotated data matrix.
    :param method: test to use.
    :param groupby: obs column to groupby.
    :param reference: reference condition.
    :param groups: alternative conditions.
    :param covariates: covariates to correct for in MAST test.
    :return: dataframe with DGE results.
    """
    if method.lower() == 'mast':
        logger.info('Running MAST test in R.')
        assert reference != 'rest', 'Specify a reference when using MAST test'
        dge = run_mast(adata, cond_key=groupby, reference=reference, disease=groups, covariates=covariates)
    elif method in ['wilcoxon', 'logreg', 't-test', 't-test_overestim_var']:
        logger.info(f'Running {method} test.')
        rank_genes_groups(adata, groupby=groupby, method=method, tie_correct=True,
                          pts=True, reference=reference, groups=groups)
        dge = generate_results(adata)
        if 'group' not in dge.columns:
            dge['group'] = groups[0]
    else:
        NameError(f'{method} not implemented. Use: mast, wilcoxon, logreg, t-test, t-test_overestim_var')
    return dge


def rank_genes_condition(
    adata: ad.AnnData,
    groupby: str,
    subset_by: str = None,
    reference: str = 'rest',
    groups: list = None,
    method: Literal['wilcoxon', 'mast', 't-test', 'logreg', 't-test_overestim_var'] = 'wilcoxon',
    pval_cutoff: float = 0.05,
    log2fc_cutoff: float = 0.25,
    path: str = None,
    filename: str = 'DGE.xlsx',
    layer: str = None,
    covariates: Union[list, None] = None,
) -> Union[pd.DataFrame, None]:
    """Run DGE Analysis.

    Run differential expression analysis. Besides the methods implemented in scanpy (wilcoxon, t-test, logreg and
    t-test_overestim_var), the MAST test can be used. If subset_by is provided the DGE analysis will be run over each
    category. Benjamini-hochberg correction method is used for multiple testing.

    After running DGE analysis and if path is provided an ExcelSheet will be generated with 3 sheets: 1) AllGenes
    containing all the genes, 2) UpregGenes containing upregulated genes and 3) DownregGenes containing downregulated
    genes. The up- and down-regulated genes are filtered depending on the pval_cutoff and log2fc_cutoff.


    :param adata: annotated data matrix.
    :param groupby: obs column with condition to test for.
    :param subset_by: obs column to subset by.  (e.g., column name with cell-type annotation)
    :param reference: reference condition.
    :param groups: alternative conditions to test for.
    :param method: method to test. Available: wilcoxon, mast, t-test, logreg and t-test_overestim_var.
    :param pval_cutoff: p-value cutoff.
    :param log2fc_cutoff: log2 foldchange cutoff.
    :param path: path to save ExcelSheet.
    :param filename: name of the ExcelSheet.
    :param layer: layer of the AnnData to use.
    :param covariates: extra covariates to correct for in the MAST test.
    :return: DGE dataframe. If a path is provided, the DataFrame with DGEs will be saved under the specified path.
    """

    sanitize_anndata(adata)
    adata_copy = adata.copy()
    if layer is not None:
        adata_copy.X = adata_copy.layers[layer].copy()

    if groups is not None:
        groups = [groups] if isinstance(groups, str) else groups
    else:
        groups = list(adata_copy.obs[groupby].unique())
        groups.remove(reference)

    if subset_by:
        categories = list(adata_copy.obs[subset_by].cat.categories())
        for catg in categories:
            logger.info(f'Running DGEs for {catg}.')
            sdata = adata_copy[adata_copy.obs[subset_by] == catg]
            dge = _run_test(sdata,
                            method=method,
                            groupby=groupby,
                            reference=reference,
                            groups=groups,
                            covariates=covariates)
            dge[subset_by] = catg
    else:
        logger.info('Running DGEs.')
        dge = _run_test(adata_copy,
                        method=method,
                        groupby=groupby,
                        reference=reference,
                        groups=groups,
                        covariates=covariates)

    if path is not None:
        out_path = convert_path(path) / filename
        logger.info(f'Saving DGE ExcelSheet in {str(out_path.name)}')
        with pd.ExcelWriter(out_path) as writer:
            dge.to_excel(writer, sheet_name='AllGenes', index=False)
            for case in groups:
                dge_up = dge[
                    (dge['padj'] < pval_cutoff) & (dge['log2fc'] > log2fc_cutoff) & (dge[dge['group'] == case])]
                dge_down = dge[
                    (dge['padj'] < pval_cutoff) & (dge['log2fc'] < -log2fc_cutoff) & (dge[dge['group'] == case])]

                dge_up.to_excel(writer, sheet_name=f'UpregGenes_{case}', index=False)
                dge_down.to_excel(writer, sheet_name=f'DownregGenes_{case}', index=False)
    else:
        return dge


def grouped_ttest(
    adata: ad.AnnData,
    annot_key: str = 'annotation',
    cond_key: str = 'condition',
    batch_key: str = 'batch',
    key_added: str = 'grouped_ttest',
    layer: str = None
) -> ad.AnnData:
    """Calculate grouped t-test.

    This function calculate a grouped t-test for all the genes in each group in annot_key. For each gene,
    the average expression per sample is employed for the test. If more than two conditions are available,
    the test will be applied to all possible combinations (for instance, for cond A, B and C; the grouped
    t-test will be computed for A-Vs-B; A-Vs-C and B-Vs-C). Results are saved as a dataframe in the
    uns attribute.

    :param adata: annotated data matrix.
    :param annot_key: obs column name with the cell type annotation.
    :param cond_key: obs column name with the conditions.
    :param batch_key: obs column name with the sample IDs.
    :param key_added: key to use in uns.
    :param layer: layer of the anndata object to use.
    :return: anndata object with results in uns attribute
    """
    if layer is not None:
        adata.X = adata.layers[layer].copy()  # Select the specified layer

    main_df = pd.DataFrame([])
    for cell in adata.obs[annot_key].unique():
        subset = adata[adata.obs[annot_key] == cell]  # Select a cell type
        df_expr = mean_expr(subset, [annot_key, cond_key, batch_key], layer=layer)  # Compute average expression

        cond_comb = [comb for comb in
                     itertools.combinations(adata.obs[cond_key].unique(), 2)]  # Get all conditions combinations

        # Compute t-test for all possible combinations
        for comb in cond_comb:
            df_a = df_expr[df_expr['group1'] == comb[0]]
            df_b = df_expr[df_expr['group1'] == comb[1]]

            df_a_wide = df_a.pivot(index='gene', values='expr', columns='group2')
            df_b_wide = df_b.pivot(index='gene', values='expr', columns='group2')

            p_values = pd.DataFrame(df_a_wide.index, columns=['gene'])
            p_values['annotation'] = cell
            p_values['condition'] = '-Vs-'.join(comb)
            p_values['pval'] = pd.DataFrame(ttest_ind(df_a_wide, df_b_wide, axis=1)[1])

            main_df = pd.concat([main_df, p_values], axis=0)

    adata.uns[key_added] = main_df
    return adata


def go_analysis(
    df: pd.DataFrame,
    gene_key: str,
    pval_key: str,
    log2fc_key: str,
    pval_cutoff: float = 0.05,
    log2fc_cutoff: float = 0.25,
    path: str = None,
    filename: str = '',
    specie: Literal['Mouse', 'Human'] = 'Mouse',
    go_catgs: Union[str, list] = ('GO_Molecular_Function_2023', 'GO_Cellular_Component_2023', 'GO_Biological_Process_2023')
) -> Union[pd.DataFrame, None]:
    """Run Gene Ontology using EnrichR API.

    Perform gene ontology analysis base on the enrichR interface.

    :param df: dataframe with results of differential gene expression analysis.
    :param gene_key: column with genes.
    :param pval_key: column with pvals.
    :param log2fc_key: column with log2 foldchanges.
    :param pval_cutoff: cutoff for pvals.
    :param log2fc_cutoff: cutoff for log2 foldchanges.
    :param path: folder where output Excel files will be saved. A SubFolder called GSA_Tables will be created
    :param filename: suffix for the filename. Format GSA_CellType_Suffix.xlsx
    :param specie: Available Human, Mouse, Yeast, Fly, Fish, Worm.
    :param go_catgs: terms to use
    :return: dataframe with gene ontology terms.
    """

    go_catgs = [go_catgs] if isinstance(go_catgs, str) else go_catgs

    logger.info('Running GSA on Up- and Down-regulated genes')
    df_up = df[(df[pval_key] < pval_cutoff) & (df[log2fc_key] > log2fc_cutoff)]
    df_down = df[(df[pval_key] < pval_cutoff) & (df[log2fc_key] < -log2fc_cutoff)]

    res_up = gseapy.enrichr(gene_list=list(df_up[gene_key]), organism=specie, gene_sets=go_catgs).results
    res_up['state'] = 'enriched'
    res_down = gseapy.enrichr(gene_list=list(df_down[gene_key]), organism=specie, gene_sets=go_catgs).results
    res_down['state'] = 'depleted'
    res = pd.concat([res_up, res_down])

    if path is not None:
        output_path = convert_path(path) / filename
        res.to_excel(output_path, index=False)
        return None
    else:
        return res
