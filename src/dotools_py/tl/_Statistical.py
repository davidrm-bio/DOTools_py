import os
import shutil
import uuid
import subprocess
from pathlib import Path
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd

from dotools_py.get import pseudobulk as get_pseudobulk
from dotools_py.get import mean_expr as get_mean_expr
from dotools_py.get import dge_results as get_dge_results
from dotools_py.get import pcts_cells as get_pcts_cells
from dotools_py.get import log2fc as get_log2fc

from dotools_py import logger
from dotools_py.utils import check_missing, iterase_input, get_paths_utils, sanitize_anndata
from dotools_py.tl._rankGenes import rank_genes_groups


def tag(tag_name):
    def tags_decorator(func):
        func._tag = tag_name
        return func

    return tags_decorator


class DGEAnalysis:
    """
    Class to perform differential gene expression (DGE) at the single-cell or sample-level for AnnData objects.
    To perform DGE analysis at the sample level, the available methods are: EdgeR, DESeq2 and T-test. For DGE analysis
    at the single cell level, the available methods are: wilcoxon, MAST, t-test and logreg.

    Parameters
    ----------
    adata (ad.AnnData)
        Annotated data matrix.
    groupby (str)
        Column in `obs` to use for testing.
    batch_key (str)
        Column in `obs` with batch information.
    pseudobulk_mode Literal["sum", "mean"]
        Method to generated pseudobulk counts.
    pseudobulk_groups (str)
        Column in `obs` to additionally group_by when generating pseudobulk profiles (e.g., cell type annotation).
        Differential gene expression will be performed for the groups (e.g., conditions) in `groupby` for each
        category (e.g., cell type) in `pseudobulk_groups`.
    technical_replicates (int)
        Number of technical replicates to generate for each sample (Experimental).

    Methods
    -------
    edger
        Run EdgeR. DGE over the sample level.
    deseq
        Run DESeq2. DGE over the sample level.
    cluster_ttest
        Run t-test. DGE over the sample level.
    wilcoxon
        Run wilcoxon. DGE over the single-cell level.
    ttest
        Run t-test. DGE over the single-cell level.
    ttest_overtim_var
        Run t-test overtimating variance. DGE over the single-cell level.
    logreg
        Run logistic regression. DGE over the single-cell level.
    mast
        Run MAST. DGE over the single-cell level.
    find_methods
        Returns a list with the name of methods to perform DGE at the single-cell or sample level.
    get_dge
        Returns a dictionary with DGEs for each method.



    """

    def __init__(self,
                 data: ad.AnnData | pd.DataFrame,
                 group_by: str,
                 batch_key: str = "batch",
                 pseudobulk_mode: Literal["sum", "mean"] = "sum",
                 pseudobulk_groups: str | None = None,
                 technical_replicates: int = None,
                 is_pseudobulk: bool = False,
                 ):
        """Initialize class.

        :param data: AnnData.
        :param group_by: Column in `obs` to use for testing.
        :param batch_key: Column in `obs` with sample information.
        :param pseudobulk_mode: Method to generated pseudobulk counts.
        :param pseudobulk_groups: Column in `obs` to additionally group_by when generating pseudobulk profiles
                                 (e.g., cell type annotation). Differential gene expression will be performed for
                                 the groups (e.g., conditions) in `groupby` for each category (e.g., cell type) in
                                 `pseudobulk_groups`.
        :param technical_replicates: Number of technical replicates to generate for each sample (Experimental).
        """

        # Checks
        self._is_numeric_counts(data, numeric=True, integers=False)
        if isinstance(data, ad.AnnData):
            sanitize_anndata(data)
            check_missing(data, groups=[group_by, batch_key] + iterase_input(pseudobulk_groups))
        if isinstance(data, pd.DataFrame):
            raise NotImplementedError("DataFrame Input not Implemented")

        self.adata = data
        self.groupby = group_by
        self.batch_key = batch_key
        self.pseudobulk_mode = pseudobulk_mode
        self.pseudobulk_groups = pseudobulk_groups
        self.technical_replicates = technical_replicates
        self.is_pseudobulk = is_pseudobulk

        self._dge = {}

    def _get_pseudobulk(
        self,
        sample_min_cells: int = 10,
        sample_min_counts: int = 100,
        gene_min_count: int = 0,
        gene_min_total_count: int = 0,
        layer: str = "counts",
    ):
        """Generate pseudobulk AnnData Object.

        :return: Returns None. The pseudo-bulked object is saved in the attribute `pdata`
        """
        import decoupler as dc
        import multiprocessing

        if self.is_pseudobulk:
            self.pdata = self.adata
            return
        else:
            self._is_numeric_counts(self.adata, numeric=True, integers=True, layer=layer)

            if self.technical_replicates is None:
                if int(dc.__version__[0]) >= 2:
                    pdata = dc.pp.pseudobulk(self.adata,
                                             sample_col=self.batch_key,
                                             groups_col=self.pseudobulk_groups,
                                             mode=self.pseudobulk_mode,
                                             layer=layer
                                             )
                    # Filter pseudobulk samples and genes
                    n_obs = pdata.n_obs
                    n_vars = pdata.n_vars
                    dc.pp.filter_samples(
                        pdata, min_cells=sample_min_cells, min_counts=sample_min_counts
                    )
                    dc.pp.filter_by_expr(
                        pdata, group=self.groupby, min_count=gene_min_count, min_total_count=gene_min_total_count
                    )
                    logger.info(f"Removed {n_obs - pdata.n_obs} samples and "
                                f"{n_vars - pdata.n_vars} genes that did not pass the filtering process")
                else:
                    raise NotImplemented("Not a valid decoupler version")
            else:
                n_cores = int(multiprocessing.cpu_count() / 2)
                pdata = get_pseudobulk(
                    self.adata,
                    batch_key=self.batch_key,
                    cluster_key=self.pseudobulk_groups,
                    keep_metadata=[self.groupby],
                    pseudobulk_approach=self.pseudobulk_mode,
                    technical_replicates=self.technical_replicates,
                    min_cells=sample_min_cells,
                    min_counts=sample_min_counts,
                    workers=n_cores,
                    layer=layer
                )
                pdata.X = pdata.X.toarray()

            self.pdata = pdata

    def _is_missing(self, vals: list) -> None:
        """Check for incorrect categorical groups.

        :param vals: Groups that should be present in adata.obs[self.groupby]
        :return: Returns None
        """
        vals = iterase_input(vals)
        tmp1 = list(self.adata.obs[self.groupby].unique())
        tmp3 = [val for val in vals if val not in tmp1]
        assert len(tmp3) == 0, f"{tmp3} not a valid group in adata.obs['{self.groupby}']"
        return None

    @staticmethod
    def _is_numeric_counts(
        data: ad.AnnData | pd.DataFrame,
        numeric: bool = True,
        integers: bool = False,
        layer: str = None
    ) -> None:
        """Test for valid inputs.

        This method test that the data contains non-negative integers.

        :param data: AnnData or Pandas DataFrame.
        :return: Returns a boolean value.
        """
        from scipy.sparse import issparse

        # Convert input to numpy array
        if isinstance(data, ad.AnnData):
            if layer is not None:
                data.X = data.layers[layer].copy()
            if issparse(data.X):
                matrix = data.X.data
            else:
                matrix = data.X.flatten()
        elif isinstance(data, pd.DataFrame):
            matrix = data.values.flatten()
        else:
            raise ValueError("Not a valid input, provide AnnData or DataFrame")

        if numeric:  # Check the input contain Numeric and non-NaN values
            if np.isnan(matrix).any():
                raise ValueError("NaNs are not allowed in the count matrix.")
            if not np.issubdtype(matrix.dtype, np.number):
                raise ValueError("The count matrix should only contain numbers.")
        if integers:  # check the input contain non-negative integer values
            if (matrix % 1 != 0).any():
                raise ValueError("The count matrix should only contain integers.")
            if (matrix < 0).any():
                raise ValueError("The count matrix should only contain non-negative values.")

        return None

    @staticmethod
    def _run_edger(
        adata: ad.AnnData,
        condition_key: str,
        design: str,
        reference: str,
        groups: str | list,
    ) -> pd.DataFrame:
        """Run EdgeR.

        :param adata: AnnData object.
        :param condition_key: Column in `obs` with conditions.
        :param design: Design for testing.
        :param reference: Reference condition.
        :param groups: Alternative groups to test against.
        :return: Returns a DataFrame.
        """
        import pertpy as pt
        pds2 = pt.tl.EdgeR(adata=adata, design=design)
        return pds2.compare_groups(adata=adata, column=condition_key, baseline=reference, groups_to_compare=groups)

    @staticmethod
    def _run_deseq(
        adata: ad.AnnData,
        condition_key: str,
        design: str,
        reference: str,
        groups: str | list,
    ) -> pd.DataFrame:
        """Run DESeq2.

        :param adata: AnnData object.
        :param condition_key: Column in `obs` with condition information.
        :param design: Design for testing.
        :param reference: Reference condition.
        :param groups: Alternative conditions to test against.
        :return: Returns a DataFrame.
        """
        import pertpy as pt
        pds2 = pt.tl.PyDESeq2(adata=adata, design=design)
        return pds2.compare_groups(adata=adata, column=condition_key, baseline=reference, groups_to_compare=groups)

    @staticmethod
    def _run_ttest(
        df: pd.DataFrame,
        comb: tuple,
        condition_key: str,
        batch_key: str,
        equal_var: bool = True
    ) -> pd.DataFrame:
        """Run t-test agregating over the samples.

        :param df: DataFrame.
        :param comb: Tuple with the two conditions to test.
        :param condition_key: Column in the DataFrame containing the condition information.
        :param batch_key: Column in the DataFrame containing the sample information.
        :param equal_var: Whether to assume equal variance or not.
        :return: Returns a DataFrame.
        """
        from scipy.stats import ttest_ind

        df_a = df[df[condition_key] == comb[0]]
        df_b = df[df[condition_key] == comb[1]]

        df_a_wide = df_a.pivot(index="gene", values="expr", columns=batch_key)
        df_b_wide = df_b.pivot(index="gene", values="expr", columns=batch_key)

        p_values = pd.DataFrame(df_a_wide.index, columns=["gene"])
        p_values["statistic"] = pd.DataFrame(
            ttest_ind(df_a_wide, df_b_wide, axis=1, equal_var=equal_var)[0]).fillna(0)
        p_values["pval"] = pd.DataFrame(ttest_ind(df_a_wide, df_b_wide, axis=1, equal_var=equal_var)[1]).fillna(
            1)
        p_values["condition"] = "-Vs-".join(comb)
        return p_values

    @staticmethod
    def _run_mast(
        adata: ad.AnnData,
        cond_key: str,
        reference: str,
        disease: str | list,
        covariates: str | list | None = None
    ) -> pd.DataFrame:
        """Run MAST test.

        :param adata: AnnData Object
        :param cond_key: Column in `obs` with the condition information.
        :param reference: Reference condition.
        :param disease: Alternative condition.
        :param covariates: Extra covariates to account for.
        :return: Returns a DataFrame.
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
            pct = get_pcts_cells(adata, group_by=cond_key, features=dge["primerid"])
            pct.columns = ["pts_" + col if col != "genes" else "GeneName" for col in pct.columns]
            lfcs = get_log2fc(adata, group_by=cond_key, reference=reference, groups=alternative)
            lfcs.rename(columns={"genes": "GeneName"}, inplace=True)

            dge["groups"] = alternative

            if "Unnamed: 0" in dge.columns:
                del dge["Unnamed: 0"]

            dge.rename(columns={"primerid": "GeneName"}, inplace=True)
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

    @staticmethod
    def _run_sc_test(
        adata: ad.AnnData,
        condition_key: str,
        reference: str = "rest",
        groups: str | list = None,
        method: Literal["logreg", "t-test", "wilcoxon", "t-test_overestim_var"] | None = "wilcoxon",
        logcounts: bool = True,
        layer: str = None,
    ) -> pd.DataFrame:
        """Wrapper around methods implemented in Scanpy.

        :param adata: AnnData object.
        :param condition_key:  Column in `obs` with the condition information.
        :param reference: Reference condition.
        :param groups: Alternative groups to test against.
        :param method: Method to use for testing.
        :param logcounts: Whether the data is log1p transform or not.
        :param layer: Layer in the AnnData object to use.
        :return: Returns a DataFrame.
        """
        groups = iterase_input(groups)
        logger.info(f"Running {method} test.")
        try:
            rank_genes_groups(
                adata, groupby=condition_key, method=method, tie_correct=True, pts=True, reference=reference,
                groups=groups, logcounts=logcounts, layer=layer,
            )
            dge = get_dge_results(adata)
            if "group" not in dge.columns:
                dge["group"] = groups[0]
        except ValueError as e:
            logger.warn(f"Error running test:\n{e}")
            dge = pd.DataFrame([])
        return dge

    def _run_sc(
        self,
        reference: str = "rest",
        groups: str | list = None,
        method: Literal["logreg", "t-test", "wilcoxon", "t-test_overestim_var", "mast"] | None = "wilcoxon",
        logcounts: bool = True,
        covariates: list = None,
        layer: str = None,
    ) -> pd.DataFrame:
        """Wrapper around _run_sc_test to run all methods.

        :param reference: Reference condition.
        :param groups: Alternative conditions to test against.
        :param method: Method for DGE analysis.
        :param logcounts: Whether the input is log1p transform or not.
        :param covariates: Additional covariates to correct for when running MAST test.
        :param layer: Layer in the AnnData to use for testing.
        :return: Returns a DataFrame.
        """

        if self.pseudobulk_groups is not None:  # We grouped by a second column (i.e, clusters)
            res_df = []
            for clust in self.adata.obs[self.pseudobulk_groups].unique():
                logger.info(f"Running {method} for {clust}")
                adata_clust = self.adata[self.adata.obs[self.pseudobulk_groups] == clust]
                if method == "mast":
                    df_current = self._run_mast(
                        adata=adata_clust, cond_key=self.groupby, reference=reference, disease=groups,
                        covariates=covariates
                    )
                else:
                    df_current = self._run_sc_test(
                        adata=adata_clust, condition_key=self.groupby, reference=reference,
                        groups=iterase_input(groups), method=method, logcounts=logcounts
                    )
                df_current[self.pseudobulk_groups] = clust
                res_df.append(df_current)
            res_df = pd.concat(res_df)

        else:
            if method == "mast":
                res_df = self._run_mast(
                    adata=self.adata, cond_key=self.groupby, reference=reference, disease=groups,
                    covariates=covariates
                )
            else:
                res_df = self._run_sc_test(
                    adata=self.adata, condition_key=self.groupby, reference=reference,
                    groups=iterase_input(groups), method=method, logcounts=logcounts, layer=layer
                )

        return res_df

    @tag("pseudobulk")
    def edger(
        self,
        design: str | pd.DataFrame,
        reference: str,
        groups: str | list,
        sample_min_cells: int = 10,
        sample_min_counts: int = 100,
        gene_min_count: int = 0,
        gene_min_total_count: int = 0,
        layer: str = "counts"
    ) -> None:
        """Differential Gene Expression Analysis with EdgeR.

        :param design: Design for the test.
        :param reference: Control condition.
        :param groups: Alternative conditions to test against.
        :param sample_min_cells: Minimum number of cells to retain a pseudobulk sample,
        :param sample_min_counts: Minimum number of counts to retain a pseudobulk sample.
        :param gene_min_count: Minimum number of counts to retain a gene.
        :param gene_min_total_count: Minimum number of total counts to retain a gene.
        :param layer: Layer with raw counts. Set to `None` if raw counts are in X
        :return: Returns None.
        """
        # Checks
        self._is_missing([reference] + iterase_input(groups))
        self._get_pseudobulk(sample_min_cells=sample_min_cells, sample_min_counts=sample_min_counts,
                             gene_min_count=gene_min_count, gene_min_total_count=gene_min_total_count, layer=layer)

        # Perform EdgeR test
        if self.pseudobulk_groups is not None:  # We grouped by a second column (i.e, clusters)
            res_df = []
            for clust in self.pdata.obs[self.pseudobulk_groups].unique():
                logger.info(f"Running EdgeR for {clust}")
                pdata_clust = self.pdata[self.pdata.obs[self.pseudobulk_groups] == clust]
                try:
                    df_current = self._run_edger(
                        adata=pdata_clust, condition_key=self.groupby, design=design, reference=reference,
                        groups=iterase_input(groups)
                    )
                    df_current[self.pseudobulk_groups] = clust
                    res_df.append(df_current)
                except ValueError as e:
                    logger.warn(f"Error while testing {clust}:\n{e}")
            res_df = pd.concat(res_df)

        else:
            res_df = self._run_edger(
                adata=self.pdata, condition_key=self.groupby, design=design, reference=reference,
                groups=iterase_input(groups)
            )

        del res_df["logCPM"]
        res_df.rename(columns={
            "variable": "GeneName", "log_fc": "log2fc", "F": "statistic",
            "p_value": "pval", "adj_p_value": "padj", "contrast": "group"}, inplace=True)
        res_df = res_df.reindex(columns=["GeneName", "statistic", "log2fc", "pval", "padj", "group"])
        self._dge["EdgeR"] = res_df
        return None

    @tag("pseudobulk")
    def deseq(
        self,
        design: str,
        reference: str,
        groups: str | list,
        sample_min_cells: int = 10,
        sample_min_counts: int = 100,
        gene_min_count: int = 0,
        gene_min_total_count: int = 0,
        layer: str = "counts",
    ):
        """Differential Gene Expression Analysis with DESeq2.

        :param design: Design for the test.
        :param reference: Control condition.
        :param groups: Alternative conditions to test against.
        :param sample_min_cells: Minimum number of cells to retain a pseudobulk sample,
        :param sample_min_counts: Minimum number of counts to retain a pseudobulk sample.
        :param gene_min_count: Minimum number of counts to retain a gene.
        :param gene_min_total_count: Minimum number of total counts to retain a gene.
        :param layer: Layer in AnnData to use.
        :return: Returns None.
        """
        # Checks
        self._is_missing([reference] + iterase_input(groups))
        self._get_pseudobulk(sample_min_cells=sample_min_cells, sample_min_counts=sample_min_counts,
                             gene_min_count=gene_min_count, gene_min_total_count=gene_min_total_count, layer=layer)

        # Perform DESeq2 test
        if self.pseudobulk_groups is not None:  # We grouped by a second column (i.e, clusters)
            res_df = []
            for clust in self.pdata.obs[self.pseudobulk_groups].unique():
                logger.info(f"Running DESeq2 for {clust}")
                pdata_clust = self.pdata[self.pdata.obs[self.pseudobulk_groups] == clust]
                try:
                    df_current = self._run_deseq(
                        adata=pdata_clust, condition_key=self.groupby, design=design, reference=reference,
                        groups=iterase_input(groups)
                    )
                    df_current[self.pseudobulk_groups] = clust
                    res_df.append(df_current)
                except ValueError as e:
                    logger.warn(f"Error while testing {clust}:\n{e}")

            res_df = pd.concat(res_df)

        else:
            res_df = self._run_deseq(
                adata=self.pdata, condition_key=self.groupby, design=design, reference=reference,
                groups=iterase_input(groups))

        del res_df["baseMean"]
        res_df.rename(columns={
            "variable": "GeneName", "log_fc": "log2fc", "lfcSE": "log2fcSE",
            "stat": "statistic", "p_value": "pval", "adj_p_value": "padj",
            "contrast": "group"}, inplace=True)

        res_df = res_df.reindex(columns=["GeneName", "statistic", "log2fc", "log2fcSE", "pval", "padj", "group"])
        res_df["pval"].fillna(1, inplace=True)
        res_df["padj"].fillna(1, inplace=True)
        self._dge["DESeq2"] = res_df

    @tag("pseudobulk")
    def cluster_ttest(
        self,
        reference: str,
        groups: str | list,
        equal_var: bool = True,
        layer: str = None
    ) -> None:
        """Differential Gene Expression Analysis with T-test.

        :param reference: Control condition.
        :param groups: Alternative conditions to test against.
        :param equal_var: Assume equal variance.
        :param layer: Layer in the AnnData to use.
        :return:
        """
        groups = iterase_input(groups)
        df_expr = get_mean_expr(
            self.adata, group_by=iterase_input(self.pseudobulk_groups) + [self.groupby, self.batch_key],
            layer=layer)

        df_main, cond_comb = [], [(reference, g) for g in groups]
        if self.pseudobulk_groups in df_expr.columns:
            for clust in df_expr[self.pseudobulk_groups].unique():
                logger.info(f"Running T-test for {clust}")
                expr_current = df_expr[df_expr[self.pseudobulk_groups] == clust]
                for comb in cond_comb:
                    df_tmp = self._run_ttest(expr_current,
                                             comb=comb,
                                             condition_key=self.groupby,
                                             batch_key=self.batch_key,
                                             equal_var=equal_var
                                             )
                    df_tmp[self.pseudobulk_groups] = clust
                    df_main.append(df_tmp)
        else:
            for comb in cond_comb:
                df_main.append(self._run_ttest(df_expr,
                                               comb=comb,
                                               condition_key=self.groupby,
                                               batch_key=self.batch_key,
                                               equal_var=equal_var
                                               )
                               )

        # TODO Compute Log2FC
        # df_tmp = df_expr.copy()
        # df_tmp["expr"] = np.expm1(df_tmp["expr"])
        # df_log2fc = df_expr.groupby(["gene", self.groupby]).agg({"expr": "mean"}).reset_index().pivot(index="gene",
        #                                                                                              columns=self.groupby,
        #                                                                                              values="expr")
        self._dge["cluster_ttest"] = pd.concat(df_main)
        return None

    @tag("single-cell")
    def wilcoxon(
        self,
        reference: str = "rest",
        groups: str | list = None,
        logcounts: bool = True,
        layer: str = None,
    ) -> None:
        self._dge["wilcoxon"] = self._run_sc(
            reference=reference, groups=groups, method="wilcoxon", logcounts=logcounts, layer=layer
        )
        return None

    @tag("single-cell")
    def ttest(
        self,
        reference: str = "rest",
        groups: str | list = None,
        logcounts: bool = True,
        layer: str = None
    ) -> None:
        self._dge["ttest"] = self._run_sc(
            reference=reference, groups=groups, method="t-test", logcounts=logcounts, layer=layer
        )
        return None

    @tag("single-cell")
    def ttest_overtim_var(
        self,
        reference: str = "rest",
        groups: str | list = None,
        logcounts: bool = True,
        layer: str = None
    ) -> None:
        self._dge["ttest_overtim_var"] = self._run_sc(
            reference=reference, groups=groups, method="t-test_overestim_var", logcounts=logcounts, layer=layer
        )
        return None

    @tag("single-cell")
    def logreg(
        self,
        reference: str = "rest",
        groups: str | list = None,
        logcounts: bool = True,
        layer: str = None
    ) -> None:
        self._dge["logreg"] = self._run_sc(
            reference=reference, groups=groups, method="logreg", logcounts=logcounts, layer=layer
        )
        return None

    @tag("single-cell")
    def mast(
        self,
        reference: str,
        groups: str | list = None,
        covariates: list = None,
        layer: str = None,
    ) -> None:
        self._dge["mast"] = self._run_sc(
            reference=reference, groups=groups, method="mast", logcounts=True, covariates=covariates, layer=layer
        )
        return None

    @classmethod
    def find_methods(
        cls,
        label: Literal["pseudobulk", "single-cell"]
    ) -> list:
        """Get list with pseudobulk or single-cell methods

        :param label: Tag of the method.
        :return: Returns a list with the names of the methods implemented to perform differential gene expression on
                pseudobulk or single-cell level.
        """
        return [
            name
            for name in dir(cls)
            if callable(getattr(cls, name))
               and getattr(getattr(cls, name), "_tag", None) == label
        ]

    @property
    def get_dge(self):
        return self._dge




import dotools_py as do


def test_DGEClass():
    adata = do.dt.example_10x_processed()
    tester = DGEAnalysis(data=adata, group_by="condition", batch_key="batch", pseudobulk_mode="sum",
                         pseudobulk_groups="annotation", technical_replicates=3)

    tester._get_pseudobulk()

    tester.cluster_ttest(
        reference="healthy",
        groups="disease"
    )

    tester.deseq(
        design="~condition",
        reference="healthy",
        groups="disease",
        layer="counts"
    )
    tester.edger(
        design="~condition",
        reference="healthy",
        groups="disease",
    )
    tester.logreg(
        reference="healthy",
        groups="disease",
    )
    tester.mast(
        reference="healthy",
        groups="disease"
    )
    tester.ttest(
        reference="healthy",
        groups="disease"
    )
    tester.ttest_overtim_var(
        reference="healthy",
        groups="disease"
    )
    tester.wilcoxon(
        reference="healthy",
        groups="disease"
    )

    checks = {
        "logreg": {"GeneName", "statistic", "group", "annotation"},
        "mast": {'GeneName', 'pvals', 'padj', 'groups', 'log2fc', 'pts_group', 'pts_ref', 'annotation'},
        "ttest": {'GeneName', 'statistic', 'log2fc', 'pvals', 'padj', 'pts_group', 'pts_ref', 'group', 'annotation'},
        "ttest_overtim_var":{'GeneName', 'statistic', 'log2fc', 'pvals', 'padj', 'pts_group', 'pts_ref', 'group', 'annotation'},
        "wilcoxon":{'GeneName', 'statistic', 'log2fc', 'pvals', 'padj', 'pts_group', 'pts_ref', 'group', 'annotation'},
        "Cluster_ttest":{'gene', 'statistic', 'pval', 'condition', 'annotation'},
        "DESeq2":{'GeneName', 'statistic', 'log2fc', 'log2fcSE', 'pval', 'padj', 'group'},
        "EdgeR":{'GeneName', 'statistic', 'log2fc', 'pval', 'padj', 'group'},
    }
    for method in tester.find_methods("single-cell"):
        assert checks[method].issubset(tester.dge[method])
    for method in tester.find_methods("pseudobulk"):
        assert checks[method].issubset(tester.dge[method])

    assert len( tester.find_methods("single-cell")) == 5
    assert len( tester.find_methods("pseudobulk")) == 3

# TODO edgeR problem generating pseudobulk if the condition is not maintain groupby AnatomicRegion check last request from Leonie

