import os
from typing import Literal

import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import issparse

from dotools_py import logger
from dotools_py.get._generic import mean_expr
from dotools_py.get._generic import expr as get_expr


class MastTest:
    """Class to perform MAST test.

    This class allows to run the MAST test on an AnnData object. Requires the R package MAST to be installed.
    The test assumes that the expression data is log-normalised.

    Parameters
    ----------
    adata
        Annotated data matrix.
    condition_key
        Column in `obs` with the conditions for each cell.
    reference
        Condition to be used as the reference.
    group
        Condition to be used as the alternative to test against. Only one condition can be accepted.
    layer
        Layer in the AnnData to use for testing.
    covariates
        Additional covariates to consider for the MAST test.
    n_cpus
        Number of cores to used for the inference.
    method
        Method to used for the inference. See `Mast::zlm <https://rglab.github.io/MAST/reference/zlm.html>`_.
    formula
        Formula for the test.
    ebayes
        If set to `True`, regularize variance using empirical bayes method.
    parallel
        If set to `True`, then multiple cores will be used in fitting.
    silent
        Silence common problems with fitting some genes.

    Example
    -------

    >>> import dotools_py as do
    >>> adata = do.dt.example_10x_processed()
    >>> tester = do.tl.MastTest(adata=adata, condition_key="condition", reference="healthy", group="disease", layer="logcounts", n_cpus=5, method="bayesglm", parallel=True, formula="~condition")
    >>> tester.fit()
    2025-09-24 13:27:56,435 - Reference level is set to healthy
    [2025-09-24 13:27:56] `fData` has no primerid.  I'll make something up.
    [2025-09-24 13:27:56] `cData` has no wellKey.  I'll make something up.
    [2025-09-24 13:27:56] Assuming data assay in position 1, with name et is log-transformed.
    [2025-09-24 13:27:59] Done!
    [2025-09-24 13:27:59] Combining coefficients and standard errors
    [2025-09-24 13:27:59] Calculating log-fold changes
    [2025-09-24 13:27:59] Calculating likelihood ratio tests
    [2025-09-24 13:27:59] Refitting on reduced model...
    [2025-09-24 13:28:03] Done!
    >>> tester.pvalues.head(3)
          GeneName     pvals      padj
    1   A4GALT  0.001722  0.015546
    2     AAK1  0.019197  0.105754
    3     ABAT  0.551787  0.842536
    >>> tester.dge_table.head(3)
          GeneName     log2fc     pvals      padj  pts_group   pts_ref
    0   A4GALT  26.073313  0.001722  0.015546   0.022222  0.000000
    1     AAK1  -0.429676  0.019197  0.105754   0.338889  0.457692
    2     ABAT   0.775196  0.551787  0.842536   0.033333  0.023077
    >>> tester.formula
    Out[17]: '~condition'
    >>> tester.mast_version()
    2025-09-24 13:30:45,484 - MAST v1.33.0

    """
    def __init__(self,
                 adata: ad.AnnData,
                 condition_key: str,
                 reference: str,
                 group: str,
                 layer: str = None,
                 covariates: list = None,
                 n_cpus: int = 20,
                 method: Literal["glm", "glmer", "bayesglm"] = "bayesglm",
                 ebayes: bool = True,
                 parallel: bool = True,
                 silent: bool = True,
                 formula: str = None,
                 ):
        """Initialise the class.

        :param adata: AnnData object to be used for testing
        :param condition_key: column in `obs` with conditions
        :param reference: reference condition
        :param group: alternative condition to test against
        :param layer: layer to use for testing
        :param covariates: covariates to correct for in the MAST test
        :param n_cpus: number of cores to used for the inference
        :param method: method to use for the inference
        :param ebayes: If set to `True`, regularize variance using empirical bayes method.
        :param parallel: allow parallelization
        :param silent: reduce verbosity
        :param formula: formula to use for the inference
        """

        os.environ["OMP_NUM_THREADS"] = "1"  # Avoid problems with running R code

        # TODO account when group is a list and several conditions are available

        adata_copy = adata.copy()
        self.adata = adata_copy
        self.condition_key = condition_key
        self.reference = reference
        self.group = group
        self.layer = layer
        self.covariates = covariates
        self.n_cpus = n_cpus
        self.method = method
        self.ebayes = ebayes
        self.parallel = parallel
        self.silent = silent
        self._formula = formula
        self.pvals = None

        self._set_rpy2_logger()


    def fit(self):
        """Fit the model.

        :return: The attribute p-values will be set containing the p-values and the adjusted p-values.
        """
        # Import rpy2
        try:
            import os
            os.environ["R_MAX_VSIZE"] = "128GB"
            from rpy2 import robjects as ro
            from rpy2.robjects import numpy2ri, pandas2ri, r, DataFrame, StrVector, FloatVector
            from rpy2.robjects.conversion import get_conversion, localconverter
            from rpy2.robjects.packages import importr
        except ImportError:
            raise ImportError("MastTest requires rpy2 to be installed.")

        # Import R packages
        try:
            MAST = importr("MAST")
        except ImportError as e:
            raise ImportError("MASTTest requires MAST to be installed") from e

        base = importr("base")
        stats = importr("stats")
        SummarizedExperiment = importr("SummarizedExperiment")

        self.adata.obs[self.condition_key] = pd.Categorical(self.adata.obs[self.condition_key],
                                                            categories=[self.reference, self.group])

        # Generate a dataframe with expression matrix
        with localconverter(get_conversion() + numpy2ri.converter):
            X = self.adata.X if self.layer is None else self.adata.layers[self.layer]  # Assume this is log-counts
            X = X.T.toarray() if issparse(X) else X.T

        # Transfer obs and vars to R
        if self.adata.var.shape[1] == 0:
            self.adata.var["genes"] = self.adata.var_names

        with localconverter(get_conversion() + pandas2ri.converter + numpy2ri.converter) as cv:
            # X_r = cv.py2rpy(pd.DataFrame(X, index=self.adata.var_names, columns=self.adata.obs_names))
            X_r = cv.py2rpy(X)
            obs_r = cv.py2rpy(self.adata.obs)
            var_r = cv.py2rpy(self.adata.var)

        levels = r('levels')(obs_r.rx2(self.condition_key))
        logger.warn(f"Reference level is set to {levels[0]}")

        # Generate SingleCellAssay
        logger.info("Generating MAST Object")
        sca = MAST.FromMatrix(base.as_matrix(X_r), obs_r, var_r)


        # Re-organise to set reference as first level
        col_data = SummarizedExperiment.colData(sca)
        condition_vector = r['slot'](col_data, 'listData').rx2(self.condition_key)
        condition_vector = stats.relevel(x=condition_vector, ref=self.reference)
        col_data.slots[self.condition_key] = condition_vector
        sca.slots['colData'] = col_data

        if self._formula is None:
            if self.covariates is None:
                latent_variables = base.c(self.condition_key)
            else:
                # Pass "condition" plus all elements in covariates as separate arguments
                latent_variables = base.c(self.condition_key, *self.covariates)
            formula = stats.as_formula(object=base.paste0(" ~ ", base.paste(latent_variables, collapse="+")))
            logger.info(f"formula is None, setting to {formula}")
        else:
            formula = stats.as_formula(str(self._formula))

        logger.info("Running Inference")
        #ro.r(f'options(mc.cores = {self.n_cpus})')  # Allow parallelisation
        zlmCond = MAST.zlm(formula=formula,
                           sca=sca,
                           method=self.method,
                           ebayes=self.ebayes,
                           ebayesControl=r('NULL'),
                           force=False,
                           hook=r('NULL'),
                           parallel=self.parallel,
                           onlyCoef=False,
                           silent=self.silent,
                           )
        logger.info("Done")
        #ro.r('options(mc.cores = 1)')  # Set to one thread again
        summary = r['summary']
        zlm_summary = summary(zlmCond, doLRT=base.paste0(self.condition_key, self.group))

        summaryDt = zlm_summary.rx2('datatable')  # equivalent to summaryCond$datatable

        with localconverter(ro.default_converter + pandas2ri.converter):
            summary_df = pandas2ri.rpy2py(summaryDt)

        # 1. Subset where component == 'H'
        filtered = summary_df[summary_df['component'] == 'H']
        # 2. Extract p-values
        p_val = filtered[['Pr(>Chisq)']].copy()
        p_val.columns = ['pvals']
        # 3. Extract gene IDs (assuming 1st column is 'primerid')
        genes = filtered[['primerid']].copy()
        # 4. Adjust p-values using R's p.adjust
        p_adjust = ro.r['p.adjust']
        from rpy2.robjects import FloatVector
        pvals_r = FloatVector(p_val['pvals'].values)
        padj_r = p_adjust(pvals_r, method='fdr')
        with localconverter(ro.default_converter + pandas2ri.converter):
            padj = pandas2ri.rpy2py(padj_r)
        p_val['padj'] = padj

        r_names = StrVector(genes["primerid"])
        r_pvals = FloatVector(p_val['pvals'])
        r_padj = FloatVector(p_val['padj'])

        # Create R data.frame
        df = DataFrame({
            'names': r_names,
            'pvals': r_pvals,
            'padj': r_padj,
        })
        with localconverter(ro.default_converter + pandas2ri.converter):
            df_py = pandas2ri.rpy2py(df)

        df_py.rename(columns = {"names":"GeneName"}, inplace=True)
        self.pvals = df_py
        self._formula = formula

    @property
    def dge_table(self) -> pd.DataFrame:
        """Generate table summarising the results from the DGE analysis.

        Returns
        --------
        Returns a dataframe with the following columns:

        `GeneName`
            Contains the genes that have been tested.
        `log2fc`
            Contains the log-foldchanges.
        `pvals`
            Contains the p-values.
        `padj`
            Contains the adjusted p-values. Correction performed with Benjamini-Hochberg.
        `pts_group`
            Percentage of cells in the group expressing the gene.
        `pts_ref`
            Percentage of cells in the reference condition expressing the gene.

        """
        df_mean = mean_expr(self.adata, group_by=self.condition_key, out_format="wide")
        logfoldchanges = np.log2((np.expm1(df_mean[self.group] + 1e-9)) /
                                 (np.expm1(df_mean[self.reference]) + 1e-9))
        # add small value to remove 0's
        df_expr = get_expr(
            self.adata, features=self.adata.var_names, groups=self.condition_key, layer=self.layer, out_format="wide"
        ).set_index(self.condition_key)

        obs_bool = df_expr > 0.0
        df_pct = (
            obs_bool.groupby(level=self.condition_key, observed=True).sum()
            / obs_bool.groupby(level=self.condition_key, observed=True).count()
        ).T
        df_pct.reset_index(inplace=True)
        df_pct.rename(columns={self.reference: "pts_ref",
                               self.group: "pts_group",
                               "index": "GeneName",}, inplace=True)
        logfoldchanges = pd.DataFrame(logfoldchanges, columns=["log2fc"]).reset_index()
        logfoldchanges.rename(columns={"gene": "GeneName"}, inplace=True)

        table = pd.merge(logfoldchanges, df_pct, on="GeneName")


        table = pd.merge(table, self.pvals, on="GeneName")
        table = table[["GeneName", "log2fc", "pvals", "padj", "pts_group", "pts_ref"]]
        return table


    @property
    def formula(self) -> str | None:
        """Formula used to fit the model.

        :return: Returns the formula used to fit the model.
        """
        if self._formula is None:
            logger.info("formula not initialise, run fit()")
            return None
        else:
            return str(self._formula).strip()


    @property
    def pvalues(self) -> pd.DataFrame | None:
        """Dataframe with the p-values from the test.

        :return: Returns a dataframe with the p-values and adjusted p-values from the test.
        """
        if self.pvals is None:
            logger.info("pvals not calculated, run fit()")
            return None
        else:
            return self.pvals


    @staticmethod
    def _set_rpy2_logger() -> None:
        """Manage the logger from rpy2.

        :return: None
        """
        import logging
        from datetime import datetime

        class RemovePrefixFilter(logging.Filter):
            def filter(self, record):
                prefix = "R callback write-console: "
                if record.msg.startswith(prefix):
                    record.msg = record.msg[len(prefix):]
                return True  # Keep the record

        # Get rpy2 logger
        rpy2_logger = logging.getLogger('rpy2')
        # Remove existing handlers
        for handler in rpy2_logger.handlers[:]:
            rpy2_logger.removeHandler(handler)

        # Create and add our handler
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)

        # Simple formatter: just message with timestamp prefix added here
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)

        # Add the filter to strip prefix
        handler.addFilter(RemovePrefixFilter())

        # Add handler and set level
        rpy2_logger.addHandler(handler)
        rpy2_logger.setLevel(logging.INFO)

        # Wrap the handler's emit to add timestamp
        old_emit = handler.emit

        def emit_with_timestamp(record):
            timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
            record.msg = f"{timestamp} {record.msg}"
            #old_emit(record)  #TODO test if hiving this is fine

        handler.emit = emit_with_timestamp
        return None

    @staticmethod
    def mast_version():
        """Get the version from MAST used.

        :return: Returns None.
        """
        from rpy2.robjects.packages import importr
        try:
            MAST = importr("MAST")
        except ImportError as e:
            raise ImportError("MASTTest requires MAST to be installed") from e
        logger.info("MAST v" + MAST.__version__)

    def __exit__(self, exc_type, exc_val, exc_tb):
        from rpy2 import robjects as ro
        import gc

        ro.r("rm(list = ls())")
        ro.r("invisible(gc())")
        gc.collect()

        return False
