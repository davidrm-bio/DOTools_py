import itertools
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd

from dotools_py import logger
from dotools_py.tl._rankGenes import rank_genes_groups
from dotools_py.utils import convert_path, get_paths_utils, sanitize_anndata, iterase_input
from dotools_py.get import mean_expr, dge_results
from dotools_py.get import log2fc as get_log2fc
from dotools_py.get import pcts_cells as get_pct_cells
from dotools_py.get import pseudobulk as pseudobulking

# DGE Analysis
def run_mast(
    adata: ad.AnnData,
    cond_key: str,
    reference: str,
    disease: str | list,
    covariates: str | list | None = None
) -> pd.DataFrame:
    """Run MAST Test for sc/snRNAseq.

    :param adata: Annotated Data matrix.
    :param cond_key: Metadata column in `obs`  with condition groups.
    :param reference: Reference condition.
    :param disease: Disease conditions.
    :param covariates: Extra covariates to account for.
    :return: Returns a `DataFrame`. The following fields are included:

             `GeneName`
                Name of the genes
             `pvals` and  `padj`
                The adjusted p-value uses Benjamini-Hochberg correction method.
             `log2fc`
                Log2FoldChamge
             `pts_ref` and `pts_group`
                Percentage of cells in the reference in the disease group expressing the gene
             `groups`
                Column containing the group tested

    See Also
    --------
        :func:`dotools_py.tl.rank_genes_groups`: run DEA at single-cell level
        :func:`dotools_py.tl.grouped_ttest`: run DEA at pseudobulk level


    Example
    -------
    >>> import dotools_py as do
    >>> adata = do.dt.example_10x_processed()
    >>> df = do.tl.run_mast(adata, "condition", "healthy", "disease")
    >>> df.head(5)
          GeneName     pvals    log2fc      padj   pts_ref  pts_group   groups
    0   A4GALT  0.001722 -1.018231  0.015546  0.003846   0.000000  disease
    1     AAK1  0.019197  0.517996  0.105754  0.457692   0.516667  disease
    2     ABAT  0.551787  1.530515  0.842536  0.000000   0.000000  disease
    3    ABCB4  0.581264 -1.968762  0.842536  0.176923   0.050000  disease
    4    ABCB9  0.458918 -1.468043  0.808238  0.121154   0.044444  disease

    """
    rscript = get_paths_utils("_Run_MAST.R")

    tmpdir_path = Path("/tmp") / f"MAST_Test_{uuid.uuid4().hex}"
    tmpdir_path.mkdir(parents=True, exist_ok=False)

    if "logcounts" not in adata.layers.keys():
        logger.warn("Layer 'logcounts' not available setting X to logcounts")
        adata.layers["logcounts"] = adata.X.copy()

    del adata.uns, adata.raw
    adata.write(tmpdir_path / "adata.h5ad")

    in_path = os.path.join(tmpdir_path, "adata.h5ad")

    disease = [disease] if isinstance(disease, str) else disease

    dge_main = pd.DataFrame()
    for alternative in disease:
        logger.info(f"Running test for {alternative}")

        cmd = [
            "Rscript",
            rscript,
            "--input=" + in_path,
            "--out=" + str(tmpdir_path) + "/dge_mast.csv",
            "--key=" + cond_key,
            "--ref=" + reference,
            "--disease=" + alternative,
        ]
        cmd += ["--covariates=" + covariates] if covariates is not None else []
        subprocess.call(cmd)
        dge = pd.read_csv(os.path.join(tmpdir_path, "dge_mast.csv"))

        # Pcts
        pct = get_pct_cells(adata, group_by=cond_key, features=dge["primerid"])
        pct.columns = ["pts_" + col if col != "genes" else "GeneName" for col in pct.columns]
        lfcs = get_log2fc(adata, group_by=cond_key, reference=reference, groups=alternative)
        lfcs.rename(columns={"genes": "GeneName"}, inplace=True)

        dge["groups"] = alternative

        if "Unnamed: 0" in dge.columns:
            del dge["Unnamed: 0"]

        dge.rename(columns={"primerid":"GeneName"}, inplace=True)
        dge = pd.merge(dge, lfcs, on="GeneName")
        dge = pd.merge(dge, pct, on="GeneName")

        dge.rename(
            columns={
                "pts_" + reference: "pts_ref",
                "pts_" + alternative: "pts_group",
                "log2fc_" + alternative: "log2fc"

            },
            inplace=True
        )

        dge_main = pd.concat([dge_main, dge])

    shutil.rmtree(tmpdir_path)

    return dge_main


def _run_test(
    adata: ad.AnnData,
    method: str,
    groupby: str,
    reference: str,
    groups: list,
    covariates: list = None,
) -> pd.DataFrame:
    """Run DGE test.

    :param adata: Annotated data matrix.
    :param method: Test to use.
    :param groupby: Column in `obs`  to group by.
    :param reference: Reference condition.
    :param groups: Alternative conditions.
    :param covariates: Covariates to correct for in MAST test.
    :return: dataframe with DGE results.
    """
    groups = iterase_input(groups)

    if method.lower() == "mast":
        logger.info("Running MAST test")
        assert reference != "rest", "Specify a reference when using MAST test"
        dge = run_mast(adata, cond_key=groupby, reference=reference, disease=groups, covariates=covariates)

    elif method in ["wilcoxon", "logreg", "t-test", "t-test_overestim_var"]:
        logger.info(f"Running {method} test.")
        try:
            rank_genes_groups(
                adata, groupby=groupby, method=method, tie_correct=True, pts=True, reference=reference, groups=groups
            )
            dge = dge_results(adata)
            if "group" not in dge.columns:
                dge["group"] = groups[0]
        except ValueError as e:
            logger.warn(f"Error running test:\n{e}")
            dge = None
    else:
        raise NameError(f"{method} not implemented. Use: mast, wilcoxon, logreg, t-test, t-test_overestim_var")
    return dge


def rank_genes_condition(
    adata: ad.AnnData,
    groupby: str,
    subset_by: str = None,
    reference: str = "rest",
    groups: list | str = None,
    method: Literal["wilcoxon", "mast", "t-test", "logreg", "t-test_overestim_var"] = "wilcoxon",
    pval_cutoff: float = 0.05,
    log2fc_cutoff: float = 0.25,
    path: str = None,
    filename: str = "DGE.xlsx",
    layer: str = None,
    covariates: list = None,
    get_results: bool = True,
    key_added: str = "rank_genes_condition",
) -> None | pd.DataFrame:
    """Run DGE Analysis.

    Run differential expression analysis. Besides the methods implemented in scanpy (wilcoxon, t-test, logreg and
    t-test_overestim_var), the `MAST <https://genomebiology.biomedcentral.com/articles/10.1186/s13059-015-0844-5>`_
    test can be used. If subset_by is provided the DGE analysis will be run over each group.
    Benjamini-Hochberg correction method is used for multiple testing.

    After running DGE analysis and if path is provided an ExcelSheet will be generated with 3 sheets: 1) AllGenes
    containing all the genes, 2) UpregGenes containing upregulated genes and 3) DownregGenes containing downregulated
    genes. The up- and down-regulated genes are filtered depending on the pval_cutoff and log2fc_cutoff. The results will
    be saved in the uns attribute under `rank_genes_condition`.

    :param adata: Annotated data matrix.
    :param groupby: Column in `obs` with conditions to test.
    :param subset_by: Column in `obs` to subset by.  (e.g., column with cell-type annotation)
    :param reference: Reference condition.
    :param groups: Alternative conditions.
    :param method: Method to test.
    :param pval_cutoff: P-value cutoff to filter when generating the ExcelSheet.
    :param log2fc_cutoff: log2 foldchange cutoff to filter when generating the ExcelSheet.
    :param path: Path to save ExcelSheet.
    :param filename: Name of the ExcelSheet.
    :param layer: Layer of the AnnData to use.
    :param covariates: list with extra covariates to correct for in the MAST test.
    :param get_results: Return a dataframe with results.
    :param key_added: Key to use in uns.
    :return: Returns a `DataFrame` if `get_results` is set to `True` with the results from the differential expression
             analysis. If a path is provided, the DataFrame  will be saved under the specified path.
             The following fields are included:

             `GeneName`
                Name of the genes
             `pvals` and  `padj`
                The adjusted p-value uses Benjamini-Hochberg correction method.
             `log2fc`
                Log2FoldChange
             `pts_ref` and `pts_group`
                Percentage of cells in the reference in the disease group expressing the gene
             `groups`
                Column containing the group tested
            `groupby`
                The column name is set to `groupby` and contains the cluster groups.
            `adata.uns['rank_genes_condition' | key_added]`
                Dataframe with results of the differential expression analysis.

    See Also
    --------
        :func:`dotools_py.tl.rank_genes_pseudobulk`: run DEA at pseudobulk level between condition for all clusters
        :func:`dotools_py.tl.rank_genes_consensus`: run DEA at pseudobulk and single-cell level between condition for all clusters

    """
    sanitize_anndata(adata)
    adata_copy = adata.copy()
    if layer is not None:
        adata_copy.X = adata_copy.layers[layer].copy()

    groups = list(adata_copy.obs[groupby].unique()) if groups is None else iterase_input(groups)
    if reference in groups:
        groups.remove(reference)

    if subset_by:
        categories = list(adata_copy.obs[subset_by].cat.categories)
        dge = pd.DataFrame([])
        for catg in categories:
            logger.info(f"Running DGEs for {catg}.")
            sdata = adata_copy[adata_copy.obs[subset_by] == catg]
            dge_s = _run_test(
                sdata, method=method, groupby=groupby, reference=reference, groups=groups, covariates=covariates,
            )
            if dge_s is None:
                continue
            dge_s[subset_by] = catg
            dge = pd.concat([dge, dge_s])
    else:
        logger.info("Running DGEs.")
        dge = _run_test(
            adata_copy, method=method, groupby=groupby, reference=reference, groups=groups, covariates=covariates,
        )

    adata.uns[key_added] = dge  # Save inplace
    if path is not None:
        out_path = convert_path(path) / filename
        logger.info(f"Saving DGE ExcelSheet in {str(out_path.name)}")
        with pd.ExcelWriter(out_path) as writer:
            dge.to_excel(writer, sheet_name="AllGenes", index=False)
            for case in groups:
                dge_up = dge[
                    (dge["padj"] < pval_cutoff) & (dge["log2fc"] > log2fc_cutoff) & (dge[dge["group"] == case])
                    ]
                dge_down = dge[
                    (dge["padj"] < pval_cutoff) & (dge["log2fc"] < -log2fc_cutoff) & (dge[dge["group"] == case])
                    ]

                dge_up.to_excel(writer, sheet_name=f"UpregGenes_{case}", index=False)
                dge_down.to_excel(writer, sheet_name=f"DownregGenes_{case}", index=False)
    if get_results:
        return dge
    else:
        return None


def grouped_ttest(
    adata: ad.AnnData,
    annot_key: str = "annotation",
    cond_key: str = "condition",
    batch_key: str = "batch",
    reference: str = "rest",
    groups: str | list = None,
    equal_var: bool = True,
    key_added: str = "grouped_ttest",
    layer: str = None,
    get_results: bool = False,
) -> None | pd.DataFrame :
    """Calculate grouped t-test.

    This function calculate a grouped t-test for all the genes in each group in annot_key. For each gene,
    the average expression per sample is employed for the test. If more than two conditions are available,
    the test will be applied to all possible combinations (for instance, for cond A, B and C; the grouped
    t-test will be computed for A-Vs-B; A-Vs-C and B-Vs-C). Results are saved as a dataframe in the
    uns attribute.

    :param adata: Annotated data matrix.
    :param annot_key: Column in `obs` with the cell type annotation.
    :param cond_key: Column in `obs` with the conditions.
    :param batch_key: Column in `obs`  with the sample IDs.
    :param reference: Reference condition.
    :param groups: Alternative conditions.
    :param equal_var: If set to `True`, assume equal variance for both populations tested.
    :param key_added: Key to use in uns.
    :param layer: Layer of the AnnData object to use.
    :param get_results: Return a DataFrame with the results.

    Returns
    -------
    Returns a `DataFrame` if `get_results` is set to `True` with the results from the differential expression
    analysis. The `DataFrame` with the results are also saved in the AnnData in:

    `adata.uns['grouped_ttest' | key_added]`.

    See Also
    --------
        :func:`dotools_py.tl.rank_genes_groups`: run DEA at single-cell level between condition for all genes

    """
    from scipy.stats import ttest_ind

    if layer is not None:
        adata.X = adata.layers[layer].copy()  # Select the specified layer
    groups = iterase_input(groups)

    main_df = pd.DataFrame([])
    for cell in adata.obs[annot_key].unique():
        subset = adata[adata.obs[annot_key] == cell]  # Select a cell type
        df_expr = mean_expr(subset, [annot_key, cond_key, batch_key], layer=layer)  # Mean vals in log space

        if reference == "rest":
            cond_comb = [
                comb for comb in itertools.combinations(adata.obs[cond_key].unique(), 2)
            ]  # Get all conditions combinations
        else:
            groups = groups if len(groups) != 0 else iterase_input(adata.obs[cond_key].unique())
            if reference in groups:
                groups.remove(reference)
            cond_comb = [(reference, g) for g in groups]

        # Compute t-test for all possible combinations
        for comb in cond_comb:
            df_a = df_expr[df_expr[cond_key] == comb[0]]
            df_b = df_expr[df_expr[cond_key] == comb[1]]

            df_a_wide = df_a.pivot(index="gene", values="expr", columns=batch_key)
            df_b_wide = df_b.pivot(index="gene", values="expr", columns=batch_key)

            p_values = pd.DataFrame(df_a_wide.index, columns=["gene"])
            p_values["statistic"] = pd.DataFrame(ttest_ind(df_a_wide, df_b_wide, axis=1, equal_var=equal_var)[0]).fillna(0)
            p_values["pval"] = pd.DataFrame(ttest_ind(df_a_wide, df_b_wide, axis=1, equal_var=equal_var)[1]).fillna(1)
            p_values["condition"] = "-Vs-".join(comb)
            p_values["annotation"] = cell

            main_df = pd.concat([main_df, p_values], axis=0)

    adata.uns[key_added] = main_df
    if get_results:
        return main_df
    else:
        return None


def go_analysis(
    df: pd.DataFrame,
    gene_key: str,
    pval_key: str,
    log2fc_key: str,
    pval_cutoff: float = 0.05,
    log2fc_cutoff: float = 0.25,
    path: str = None,
    filename: str = "",
    specie: Literal["Mouse", "Human"] = "Mouse",
    go_catgs: str | list = ("GO_Molecular_Function_2023", "GO_Cellular_Component_2023", "GO_Biological_Process_2023"),
) -> None | pd.DataFrame:
    """Run Gene Ontology using EnrichR API.

    Perform gene ontology analysis base on the `EnrichR <https://maayanlab.cloud/Enrichr/>`_ interface.

    :param df: Dataframe with results of differential gene expression analysis.
    :param gene_key: Column with genes.
    :param pval_key: Column with pvals.
    :param log2fc_key: Column with log2 foldchanges.
    :param pval_cutoff: Cutoff for pvals.
    :param log2fc_cutoff: Cutoff for log2 foldchanges.
    :param path: Folder where output Excel files will be saved. A SubFolder called GSA_Tables will be created
    :param filename: Suffix for the filename. Format GSA_CellType_Suffix.xlsx
    :param specie: Available Human, Mouse, Yeast, Fly, Fish, Worm.
    :param go_catgs: Gene Ontology Classes to use
    :return: Return a `DataFrame` with Gene Ontology Analysis results. If a path is provided the results will be saved.
    """
    import gseapy

    go_catgs = iterase_input(go_catgs)

    logger.info("Running GSA on Up- and Down-regulated genes")
    df_up = df[(df[pval_key] < pval_cutoff) & (df[log2fc_key] > log2fc_cutoff)]
    df_down = df[(df[pval_key] < pval_cutoff) & (df[log2fc_key] < -log2fc_cutoff)]

    res_up = gseapy.enrichr(gene_list=list(df_up[gene_key]), organism=specie, gene_sets=go_catgs).results
    res_up["state"] = "enriched"
    res_down = gseapy.enrichr(gene_list=list(df_down[gene_key]), organism=specie, gene_sets=go_catgs).results
    res_down["state"] = "depleted"
    res = pd.concat([res_up, res_down])

    if path is not None:
        output_path = convert_path(path) / filename
        res.to_excel(output_path, index=False)
        return None
    else:
        return res



def rank_genes_pseudobulk(
    adata: ad.AnnData,
    ctrl_cond: str,
    disease_cond: str,
    cluster_key: str,
    method: Literal["deseq2", "edger"] = "deseq2",
    batch_key: str = "batch",
    condition_key: str = "condition",
    design: str = "~condition",
    layer: str = "counts",
    min_cells: int = 50,
    pseudobulk_approach: Literal["sum", "mean"] = "sum",
    technical_replicates: int = 1,
    min_counts: int = 10,
    workers: int = 8,
    path: str = None,
    filename: str = "DEA_Pseudobulk.xlsx",
    get_results: bool = True,
    key_added: str = "rank_genes_pseudobulk",
) -> None | pd.DataFrame:
    """Running DEA using pseudobulk approach.

    Perform differential expression analysis (DEA) using DESeq2 or EdgeR. This functions has a similar behavior as
    :func:`dotools_py.tl.rank_genes_condition()`. For each cluster it will test for differential gene expression between two conditions.
    The input is expected to be raw counts.

    :param adata: Annotated data matrix.
    :param ctrl_cond: Control condition.
    :param disease_cond: Disease condition.
    :param cluster_key: Metadata column in `obs` with cluster groups.
    :param method: Differential expression method to use, DESeq2 or EdgeR.
    :param batch_key: Metadata column in `obs` with batch groups
    :param condition_key: Metadata column in `obs` with condition groups.
    :param design: Design factors for DESeq2.
    :param layer: Layer to use. Expected raw counts.
    :param min_cells: Minimum number of cells per batch/sample required when generating the pseudo-bulk. If there are
                      fewer cells, DESeq2 / EdgeR will not be run on the cluster.
    :param pseudobulk_approach: How to generate the pseudobulk counts.
    :param technical_replicates: How many technical replicates should be generated per sample.
    :param min_counts: Minimum number of total counts for a gene to be tested after pseudo-bulking.
    :param workers: Number of CPUs to use for DESEq2.
    :param path: Path to save the file.
    :param filename: Name of the file.
    :param get_results: Get dataframe with DEA results.
    :param key_added: Name of the uns attribute with the results.
    :return: Returns a `DataFrame` with DEA results if `get_results` is set to True. The following field will also be set:

             `adata.uns['rank_genes_pseudobulk' | key_added]`
                Dataframe with results of the differential expression analysis

    See Also
    --------
        :func:`dotools_py.tl.rank_genes_condition`: run DEA at single-cell level between condition for all clusters
        :func:`dotools_py.tl.rank_genes_consensus`: run DEA at pseudobulk and single-cell level between condition for all clusters

    """

    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.default_inference import DefaultInference
    from pydeseq2.ds import DeseqStats

    # Step 1 - Generate Pseudo-bulk data
    logger.info("Generating Pseudo-bulk data")
    pdata_cts = pseudobulking(
        adata,
        batch_key=batch_key,
        cluster_key=cluster_key,
        min_cells=min_cells,
        pseudobulk_approach=pseudobulk_approach,
        technical_replicates=technical_replicates,
        min_counts=min_counts,
        layer=layer,
        keep_metadata=[condition_key],
        workers=workers
    )

    sanitize_anndata(pdata_cts)

    # Step 2 - Run test
    if method == "deseq2":
        logger.info("Run DESeq2")
        inference = DefaultInference(n_cpus=workers)
        df_main = pd.DataFrame([])
        for ct in pdata_cts.obs[cluster_key].unique():
            try:
                pdata = pdata_cts[pdata_cts.obs[cluster_key] == ct].copy()
                pdata.X = pdata.X.toarray()
                if pdata.n_obs == 0:
                    logger.warn(f"Could not test for {ct}")
                    continue
                dds = DeseqDataSet(adata=pdata, design=design, refit_cooks=True, inference=inference)
                dds.deseq2()
                stat_res = DeseqStats(dds, contrast=[condition_key, disease_cond, ctrl_cond], inference=inference)
                stat_res.summary()
                results = stat_res.results_df.copy()
                results.loc[results.padj.isna(), "padj"] = 1  # Replace NaN with 1
                results["group"] = pdata.obs[cluster_key].unique()[0]
                df_main = pd.concat([df_main, results])
            except ValueError as e:
                logger.info(f"Test could not be computed for {ct} due to {e}")

    elif method == "edger":
        logger.info("Run edgeR")
        rscript = get_paths_utils("_run_edgeR.R")
        tmpdir_path = Path("/tmp") / f"EdgeR_Test_{uuid.uuid4().hex}"
        tmpdir_path.mkdir(parents=True, exist_ok=False)

        df_main = pd.DataFrame()
        for ct in pdata_cts.obs[cluster_key].unique():
            try:
                logger.info(f"Running DEA for {ct}")
                pdata = pdata_cts[pdata_cts.obs[cluster_key] == ct].copy()
                del pdata.uns, pdata.raw
                pdata.write(tmpdir_path / f"adata_{ct}.h5ad")
                in_path = os.path.join(tmpdir_path, f"adata_{ct}.h5ad")
                cmd = [
                    "Rscript",
                    rscript,
                    "--input=" + in_path,
                    "--out=" + str(tmpdir_path) + f"/dge_{ct}_edgeR.csv",
                    "--batch=" + batch_key,
                    "--condition=" + condition_key,
                    "--ref=" + ctrl_cond,
                    "--disease=" + disease_cond,
                ]
                subprocess.call(cmd)
                dge = pd.read_csv(os.path.join(tmpdir_path, f"dge_{ct}_edgeR.csv"))
                dge["group"] = pdata.obs[cluster_key].unique()[0]
                df_main = pd.concat([df_main, dge])
            except Exception as e:
                logger.info(f"Test could not be computed for {ct} due to {e}")

    else:
        raise Exception(f'{method} not implemented, use "deseq2" or "edger"')

    if path is not None:
        df_main.to_excel(convert_path(path) / filename)

    adata.uns[key_added] = df_main
    if get_results:
        return df_main
    else:
        return None


def rank_genes_consensus(
    adata: ad.AnnData,
    ctrl_cond: str,
    disease_cond: str,
    cluster_key: str,
    batch_key: str = "batch",
    condition_key: str = "condition",
    design: str = "~condition",
    count_layer: str = "counts",
    logcounts_layer: str = "logcounts",
    min_cells: int = 50,
    pseudobulk_approach: Literal["sum", "mean"] = "sum",
    technical_replicates: int = 1,
    min_counts: int = 10,
    workers: int = 8,
    path: str | Path = None,
    filename: str = "DEA.xlsx",
    test_pseudobulk: Literal["deseq2", "edger"] = "deseq2",
    test: Literal["wilcoxon", "mast", "t-test", "logreg", "t-test_overestim_var"] = "wilcoxon",
    mast_covariates: list = None,
    pval_cutoff: float = 0.05,
    get_results: bool = True,
    key_added: str = "rank_genes_consensus",
) -> None | pd.DataFrame :
    """Run single-cell and pseudo-bulk differential expression analysis.

    This function performs differential gene expression analysis between two conditions for
    an all the clusters in the AnnData object using a single-cell level and pseudo-bulk level approach.
    For the single-cell level, it will test for DEGs using wilcoxon, MAST, t-test, logistic regression or
    t-test overestimate. For the pseudobulk level it will test for DEGs using DESeq2 or edgeR.

    A dataframe will be produce with the results of both tests including the foldchanges, p-values, statistics,
    percentage of cells in each group expressing the gene and the mean expression per sample in each cluster for each gene.
    The dataframe will be saved in the `uns` attribute and can also be saved if a path a filename is provided.

    :param adata: Annotated data matrix.
    :param ctrl_cond: Control condition.
    :param disease_cond: Disease or alternative condition to test.
    :param cluster_key: Metadata column in obs with clustering groups.
    :param batch_key: Metadata column in obs with batch groups.
    :param condition_key: Metadata column in obs with condition groups.
    :param design: Design for the differential expression analysis in DESeq2.
    :param count_layer: Layer with counts. Required for DESeq2.
    :param logcounts_layer: Layer with logcounts.
    :param min_cells: Minimum number of cells per batch/sample required when generating the pseudo-bulk. If there are
                      fewer cells, DESeq2 / EdgeR will not be run on the cluster.
    :param pseudobulk_approach: How to generate the pseudobulk counts.
    :param technical_replicates: How many technical replicates should be generated per sample.
    :param min_counts: Minimum number of total counts for a gene to be tested in DESeq2 after pseudobulking.
    :param workers: Number of CPUs to use for DESEq2.
    :param path: Path to save results.
    :param filename: Name of the file.
    :param test_pseudobulk: Test to use for doing differential expression analysis on pseudobulk level.
    :param test: Test to use for doing differential expression analysis on single-cell level.
    :param mast_covariates: Covariates for MAST test.
    :param pval_cutoff: Cutoff for considering a gene significant.
    :param get_results: Get a dataframe with the consensus results
    :param key_added: Name of the uns attribute with the results
    :return: Returns a `DataFrame` with DEA results if `get_results` is set to True. The following field will also be set:

            `adata.uns['rank_genes_consensus' | key_added]`
                Dataframe with results of the differential expression analysis

    See Also
    --------
        :func:`dotools_py.tl.rank_genes_condition`: run DEA at single-cell level between condition for all clusters
        :func:`dotools_py.tl.rank_genes_pseudobulk`: run DEA at pseudobulk level between condition for all clusters

    """
    # Run single-cell dge
    logger.info(f"Running {test}")
    df_sc = rank_genes_condition(
        adata,
        groupby=condition_key,
        subset_by=cluster_key,
        reference=ctrl_cond,
        groups=disease_cond,
        method=test,
        layer=logcounts_layer,
        covariates=mast_covariates,
        get_results=True,
    )

    # Run pseudobulk
    logger.info(f"Running {test_pseudobulk}")
    df_pseudobulk = rank_genes_pseudobulk(
        adata,
        method=test_pseudobulk,
        ctrl_cond=ctrl_cond,
        disease_cond=disease_cond,
        cluster_key=cluster_key,
        batch_key=batch_key,
        condition_key=condition_key,
        design=design,
        layer=count_layer,
        min_cells=min_cells,
        min_counts=min_counts,
        pseudobulk_approach=pseudobulk_approach,
        technical_replicates=technical_replicates,
        workers=workers,
    )
    logger.info("Generating consensus DataFrame")

    if test_pseudobulk == "edger":
        df_pseudobulk.set_index("Unnamed: 0", inplace=True)
        df_pseudobulk.index.name = None


    # CleanUp
    df_pseudobulk["GeneName"] = df_pseudobulk.index
    df_pseudobulk.rename(
        columns={
            "log2FoldChange": f"log2fc_{test_pseudobulk}",
            "stat": f"stat_{test_pseudobulk}",
            "pvalue": f"pval_{test_pseudobulk}",
            "padj": f"padj_{test_pseudobulk}",
            "logFC": f"log2fc_{test_pseudobulk}",
            "F": f"stat_{test_pseudobulk}",
            "PValue": f"pval_{test_pseudobulk}",
            "FDR": f"padj_{test_pseudobulk}",
        },
        inplace=True,
    )

    df_pseudobulk = df_pseudobulk[
        [
            "GeneName",
            f"log2fc_{test_pseudobulk}",
            f"stat_{test_pseudobulk}",
            f"pval_{test_pseudobulk}",
            f"padj_{test_pseudobulk}",
            "group",
        ]
    ].reset_index(drop=True)

    # Add missing genes for the consensus
    missing = (
        df_pseudobulk.groupby("group")["GeneName"]
        .apply(lambda x: list(set(adata.var_names) - set(x)))
        .reset_index(name="missing_values")
    )
    tmp = pd.DataFrame([])
    cols_names = [
        f"log2fc_{test_pseudobulk}",
        f"stat_{test_pseudobulk}",
        f"pval_{test_pseudobulk}",
        f"padj_{test_pseudobulk}",
    ]
    for idx, row in missing.iterrows():
        new_rows = pd.DataFrame(row["missing_values"], columns=["GeneName"])
        template = pd.DataFrame(np.zeros((new_rows.shape[0], 4)), columns=cols_names)
        template["group"] = row["group"]
        template[f"pval_{test_pseudobulk}"] = 1
        template[f"padj_{test_pseudobulk}"] = 1
        new_rows = new_rows.join(template)
        tmp = pd.concat([tmp, new_rows])
    df_pseudobulk = pd.concat([df_pseudobulk, tmp])

    # If a group was not tested we also need to add it
    missing = [c for c in df_sc[cluster_key].unique() if c not in df_pseudobulk["group"].unique()]
    tmp = pd.DataFrame([])
    for m in missing:
        new_rows = pd.DataFrame(list(adata.var_names), columns=["GeneName"])
        template = pd.DataFrame(np.zeros((new_rows.shape[0], 4)), columns=cols_names)
        template["group"] = m
        template[f"pval_{test_pseudobulk}"] = 1
        template[f"padj_{test_pseudobulk}"] = 1
        new_rows = new_rows.join(template)
        tmp = pd.concat([tmp, new_rows])
    df_pseudobulk = pd.concat([df_pseudobulk, tmp])
    df_pseudobulk.rename(columns={"group": cluster_key}, inplace=True)
    df_sc = df_sc[df_sc.group == disease_cond]
    assert df_sc.shape[0] == df_pseudobulk.shape[0]
    df_consensus = df_sc.merge(df_pseudobulk, on=["GeneName", cluster_key])

    df_consensus["sc_signicant"] = ["Yes" if pval < pval_cutoff else "No" for pval in df_consensus["padj"]]
    df_consensus["psc_signicant"] = [
        "Yes" if pval < pval_cutoff else "No" for pval in df_consensus[f"padj_{test_pseudobulk}"]
    ]
    df_consensus["consensus_significant"] = df_consensus.apply(
        lambda row: "Yes" if row["sc_signicant"] == "Yes" and row["psc_signicant"] == "Yes" else "No", axis=1
    )

    # Mean per cluster for each sample correct
    df_mean = mean_expr(adata, group_by=[condition_key, cluster_key], features=list(adata.var_names))
    df_mean[condition_key] = "MeanExpr_" + df_mean[condition_key].astype(str)
    df_mean = df_mean.pivot(index=["gene", cluster_key], columns=condition_key, values="expr").reset_index()
    df_mean.rename(columns={"gene": "GeneName", cluster_key: cluster_key}, inplace=True)
    df_mean = df_mean.reset_index(drop=True)

    df_consensus = df_consensus.merge(df_mean, on=["GeneName", cluster_key])

    adata.uns[key_added] = df_consensus

    if path is not None:
        df_consensus.to_excel(convert_path(path) / filename, index=False)

    if get_results:
        return df_consensus
    else:
        return None
