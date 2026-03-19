import os
import subprocess
import uuid
from datetime import date
from pathlib import Path
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dotools_py import logger
from dotools_py.io import read_10x_h5, read_visium, read_10x_mtx

from dotools_py._utils import (
    convert_path,
    get_paths_utils,
    iterase_input,
    x_is_raw_counts,
    is_none
)

from dotools_py._custom_class import InputError, PathLike
from typing import TYPE_CHECKING, Any, Literal, Dict
from dotools_py.pp._utils import _qc_vln, _filter_quantiles, _normalise, _lower_strings, _run_sc_dbl_finder, _run_ovrlpy

if TYPE_CHECKING:
    try:
        from spatialdata import SpatialData
    except ModuleNotFoundError:
        SpatialData = Any


def find_doublets(
    adata: ad.AnnData | pd.DataFrame,
    batch_key: str | None = None,
    cluster_key: str | bool | None = None,
    doublet_rate: float = None,
    scdblfinder_metric: Literal['merror', 'logloss', 'auc', 'aucpr'] = "logloss",
    method: Literal["scDblFinder", "DoubletDetection", "Scrublet", "Ovrlpy"] = "scDblFinder",
    ovrlpy_keys: Dict = None,
    ovrlpy_report_path: PathLike = None,
    random_state: int = 0,
) -> None:
    """Detect doublets in scRNAseq and iST.

    Detect doublets in sc/snRNA-seq and imaged-based spatial transcriptomics (iST). For the iST, vertical doublets
    will be detected (i.e., regions where doublets are detected over the Z axis).

    .. note::
        For iST, a report will be generated but no vertical doublets will be removed.

    Parameters
    ----------
    adata:
        Annotated data matrix or a pandas DataFrame if method is set to `Ovrlpy`.
    batch_key
        Column in `adata.obs` with batch information. If omitted, doublets will be searched for with all cells together.
        If given, doublets will be searched for independently for each sample, which is preferable if they represent
        different captures.
    cluster_key
        Column in `adata.obs` with clustering information. This is used to make doublets more efficiently.
        Alternatively, if `cluster_key=True`, fast clustering will be performed. If `cluster_key` is None or False,
        purely random artificial doublets will be generated.
    doublet_rate
        The expected doublet rate, i.e. the proportion of the cells expected to be doublets.
        If omitted, will be calculated automatically for scDblFinder and will be set to 0.05 for Scrublet.
    scdblfinder_metric
        Error metric to optimize during training (e.g. 'merror', 'logloss', 'auc', 'aucpr').
    method
        Library to use for detecting doublets. For scRNA-seq data the available methods are:
        `scDblFinder <https://f1000research.com/articles/10-979/v2>`_,
        `DoubletDetection <https://zenodo.org/records/14827937>`_, and
        `Scrublet <https://www.sciencedirect.com/science/article/pii/S2405471218304745>`_.
        For Spatial Transcriptomics at single cell resolution, like Xenium the available methods are:
        `Ovrlpy <https://ovrlpy.readthedocs.io/latest/>`_ (Allow the detection of vertical doublets in image based ST).
    ovrlpy_keys
        Dictionary with the following keys: `gene_key`, `x_key`, `y_key` and `z_key` indicating the name of the column
        in the dataframe with the gene names and the x, y and z coordinate.
    ovrlpy_report_path
        Directory where the quality control plots and the ovrlpy object will be saved.
    random_state
        Seed for random number generator

    Returns
    -------
    None
        Returns `None`. Sets the following fields:

    `adata.obs['doublet_class']` : :class:`pandas.Series` (dtype `str`)
        Class indicating predicted doublet status
    `adata.obs['doublet_score']` : :class:`pandas.Series` (dtype `float`)
        Doublet scores for each observed transcriptome

    Examples
    --------
    >>> import dotools_py as do
    >>> adata = do.dt.example_10x_processed()
    >>> find_doublets(adata, batch_key="batch", method="scDblFinder")
    >>> adata.obs[["doublet_class", "doublet_score"]].head()
                                  doublet_class doublet_score
    CAAAGAATCAGATTGC-1-batch2       singlet      0.692706
    AGCTTCCCAGTCAACT-1-batch1       singlet      0.014858
    GAGAGGTTCCCTCTAG-1-batch1       singlet      0.172094
    CTAACTTCAGATCATC-1-batch1       singlet      0.092695
    CATGGTACAAACGGCA-1-batch1       singlet      0.237514

    """

    assert adata.n_obs != 0, "The AnnData is empty "

    if method == "scDblFinder":
        doublets = _run_sc_dbl_finder(
            adata=adata,
            batch_key=batch_key,
            cluster_key=cluster_key,
            doublet_rate=doublet_rate,
            scdblfinder_metric=scdblfinder_metric,
            random_state=random_state
        )
        adata.obs[["doublet_class", "doublet_score"]] = doublets.values

    elif method == "DoubletDetection":
        import doubletdetection
        clf = doubletdetection.BoostClassifier(
            n_iters=15, clustering_algorithm="leiden", standard_scaling=True, verbose=False, n_jobs=-1,
            random_state=random_state,
        )
        doublets = clf.fit(adata.X).predict()
        doublet_score = clf.doublet_score()
        mapped = np.full(doublets.shape, "singlet", dtype=object)
        mapped[doublets == 1.0] = "doublet"
        adata.obs["doublet_class"] = pd.Categorical(mapped, categories=["singlet", "doublet"])
        adata.obs["doublet_score"] = doublet_score

    elif method == "Scrublet":
        from scanpy.preprocessing import scrublet
        scrublet(adata, expected_doublet_rate=is_none(doublet_rate, 0.05), random_state=random_state)
        adata.obs["doublet_class"] = adata.obs["predicted_doublet"].map({False: "singlet", True: "doublet"})
        del adata.obs["predicted_doublet"]

    elif method == "Ovrlpy":
        _run_ovrlpy(df=adata, batch_key=batch_key, ovrlpy_report_path=ovrlpy_report_path, ovrlpy_keys=ovrlpy_keys)
    else:
        raise InputError("Doublet detection tool available: scDblFinder, Scrublet and DoubletDetection")
    adata.obs["doublet_class"] = pd.Categorical(adata.obs["doublet_class"].astype(str))
    adata.obs["doublet_score"] = adata.obs["doublet_score"].astype(float)
    return None


def log_normalize(
    adata: ad.AnnData,
    target_sum: int = 10_000,
    log_data: bool = True,
) -> None:
    """Apply LogNormalization.

    The data in X will be log-normalize to 10,000 reads per cell.  The shifted logarithm works beneficial for
    stabilizing variance for subsequent dimensionality reduction and identification of differentially expressed genes.
    The returned anndata object will contain 3 layers:
    * counts: contains the raw un-normalized counts
    * logcounts: contains the log-normalize counts
    Additionally, the log-normalize counts will also be saved under the X attribute. If `log_data` is set to `False`,
    the normalized counts without logarithm transformation are kept and a layer named `norm_counts` will be added.

    Parameters
    ----------
    adata
        Annotated data matrix.
    target_sum
         Target number of reads per cell to normalize to.
    log_data
        If set to `True` logarithm transformation is applied to the data.
    Returns
    -------
    Returns `None`. Changes will be performed inplace.

    """
    x_is_raw_counts(adata)  # LogNormalization should only be performed on raw counts
    adata.layers["counts"] = adata.X.copy()  # Save raw counts
    _normalise(adata, n_reads=target_sum, log_data=log_data)
    return None


def pearson_residuals_normalize(
    adata: ad.AnnData,
    batch_key: str = None,
    layer: str = None,
    backend: Literal["scanpy", "seurat"] = "scanpy",
    theta: int = 100,
) -> ad.AnnData:
    """Apply analytic Pearson residual normalization.

    The residuals are based on a negative binomial offset model with overdispersion theta shared across genes.
    By default, residuals are clipped to sqrt(n_obs) and overdispersion theta=100 is used. It expects raw counts as
    input.

    :param adata: Annotated data matrix.
    :param batch_key: Column in adata.obs with batch information.
    :param layer: Layer to use instead of `adata.X`
    :param backend: If set to `scanpy` it will use scanpy implementation. Otherwise set to `seutat` to use `SCTransform <https://github.com/satijalab/sctransform>`_.
    :param theta: he negative binomial overdispersion parameter for Pearson residuals.
    :return: Returns `AnnData`. Depending on the backend new layers will be added. The normalized values will also be set in `adata.X`

    Example
    -------
    >>> import dotools_py as do
    >>> adata = do.dt.example_10x_processed()
    >>> adata = pearson_residuals_normalisation(adata, batch_key="batch", layer="counts", backend="scanpy")
    normalizing counts per cell
    finished (0:00:00)
    computing analytic Pearson residuals on counts
        finished (0:00:00)
    computing analytic Pearson residuals on counts
        finished (0:00:00)
    >>> adata
    AnnData object with n_obs × n_vars = 700 × 1851
    obs: 'batch', 'condition', 'n_genes_by_counts', 'log1p_n_genes_by_counts', 'total_counts', 'log1p_total_counts',
         'total_counts_mt', 'log1p_total_counts_mt', 'pct_counts_mt', 'total_counts_ribo', 'log1p_total_counts_ribo',
         'pct_counts_ribo', 'n_genes', 'n_counts', 'doublet_class', 'doublet_score', 'leiden', 'cell_type', 'autoAnnot',
         'celltypist_conf_score', 'annotation', 'annotation_recluster'
    obsm: 'X_CCA', 'X_pca', 'X_umap'
    layers: 'counts', 'logcounts', 'sqrt_norm', 'pearson_norm'
    >>> adata = do.dt.example_10x_processed()
    >>> adata = pearson_residuals_normalisation(adata, batch_key="batch", layer="counts", backend="seurat")
    2026-03-05 15:45:26,911 - Preparing to transfer to R
    2026-03-05 15:45:26,928 - Running SCTransform in R
    >>> adata
    AnnData object with n_obs × n_vars = 700 × 1181
    obs: 'batch', 'condition', 'n_genes_by_counts', 'log1p_n_genes_by_counts', 'total_counts', 'log1p_total_counts',
         'total_counts_mt', 'log1p_total_counts_mt', 'pct_counts_mt', 'total_counts_ribo', 'log1p_total_counts_ribo',
         'pct_counts_ribo', 'n_genes', 'n_counts', 'doublet_class', 'doublet_score', 'leiden', 'cell_type', 'autoAnnot',
         'celltypist_conf_score', 'annotation', 'annotation_recluster'
    var: 'mean', 'std', 'highly_variable', 'means', 'dispersions', 'dispersions_norm', 'highly_variable_nbatches', 'highly_variable_intersection', 'SCT_rm'
    obsm: 'SCT_rm'
    varm: 'PCs'
    layers: 'counts', 'logcounts', 'SCT_norm', 'SCT_counts'
    obsp: 'connectivities', 'distances'

    """
    x_is_raw_counts(adata)

    if "counts" not in adata.layers.keys():
        adata.layers["counts"] = adata.X.copy()

    if backend == "scanpy":
        import scanpy as sc
        adata.layers["sqrt_norm"] = np.sqrt(sc.pp.normalize_total(adata, inplace=False)["X"])
        database = {}
        for batch in adata.obs[batch_key].unique():
            adata_batch = adata[adata.obs[batch_key] == batch].copy()
            sc.experimental.pp.normalize_pearson_residuals(adata_batch, theta=theta, layer=layer)
            adata_batch.layers["pearson_norm"] = adata_batch.X.copy()
            database[batch] = adata_batch
        adata = ad.concat(database.values(), join="outer")
    else:
        from scipy import sparse
        import polars
        rscript = get_paths_utils("_run_SCTransform.R")
        tmpdir_path = Path("/tmp") / f"SCTransform_{uuid.uuid4().hex}"
        tmpdir_path.mkdir(parents=True, exist_ok=False)

        logger.info("Preparing to transfer to R")
        adata_copy = adata.copy()
        if layer is not None:
            adata.X = adata.layers[layer].copy()
        del adata.uns
        del adata.obsm

        if batch_key is not None:
            adata_copy.obs["batch"] = adata_copy.obs[batch_key].copy()
        else:
            adata_copy.obs["batch"] = "batch1"
        adata_copy.write(tmpdir_path / "adata_tmp.h5ad")

        logger.info("Running SCTransform in R")
        subprocess.call(["Rscript", rscript, "--input=" + str(tmpdir_path) + "/", "--out=" + str(tmpdir_path) + "/"])

        raw_counts = polars.read_csv(os.path.join(tmpdir_path, "SCTransform_raw.csv"), infer_schema_length=0)
        raw_counts = raw_counts.to_pandas().astype(float)
        raw_counts = raw_counts.set_index(adata.obs_names)

        norm_counts = polars.read_csv(os.path.join(tmpdir_path, "SCTransform_norm.csv"), infer_schema_length=0)
        norm_counts = norm_counts.to_pandas().astype(float)
        norm_counts = norm_counts.set_index(adata.obs_names)

        # Transfer genes not kept during normalization to .obsm
        excluded_genes = [gene for gene in adata.var_names if gene not in norm_counts.columns]
        adata.var["SCT_rm"] = [True if gene in excluded_genes else False for gene in adata.var_names]
        adata.obsm["SCT_rm"] = adata[:, adata.var["SCT_rm"].values].X.toarray()
        adata = adata[:, ~adata.var["SCT_rm"].values]

        # Make sure we have the same order or barcodes and features
        norm_counts = norm_counts.reindex(index=adata.obs_names, columns=adata.var_names)
        raw_counts = raw_counts.reindex(index=adata.obs_names, columns=adata.var_names)

        adata.layers["SCT_norm"] = sparse.csr_matrix(norm_counts.values)
        adata.layers["SCT_counts"] = sparse.csr_matrix(raw_counts.values)
    return adata


class Importer:
    def __init__(
        self,
        adata: ad.AnnData = None,
        paths: list = None,
        ids: list = None,
        metadata: dict = None,
        batch_key: str = None,
        remove_doublets: bool = True,
        doublet_tool: Literal["scDblFinder", "Scrublet", "DoubletDetection"] = "scDblFinder",
        min_genes_in_cell: int = None,
        min_cells_with_genes: int = None,
        cut_mt: int = None,
        n_reads: int = None,
        min_counts: int = None,
        max_counts: int | None = None,
        min_genes: int | None = None,
        max_genes: int | None = None,
        low_quantile: int | None = None,
        high_quantile: int | None = None,
        random_state: int = 0,
        technology: Literal["scrna", "snrna", "visium", "xenium"] = "scrna",
        normalisation_method: Literal["LogNormalisation", "PearsonResiduals"] = "LogNormalisation",
        log_data: bool = True,
        report: bool = True,
        metrics_patterns: list | tuple = ("mt-", ("rbs", "rpl")),
        metrics_names: list = ("mt", "ribo")
    ):
        # Checks
        if (adata is None) == (paths is None):
            raise InputError(
                "Provide either, \n1) An unprocessed AnnData or \n2) A list of paths to an H5 file or "
                "10x-Genomics-formated mtx directory\n"
                ""
            )
        metadata = is_none(metadata, {})
        if len(metadata) != 0:
            assert all([len(val) == len(ids) for val in metadata.values()]), (
                "The number of ids and the entries for some metadata does not match"
            )
        if doublet_tool not in ["scDblFinder", "Scrublet", "DoubletDetection"]:
            raise InputError(
                f"{doublet_tool} is not a valid key for doublet_tool"
            )
        if technology not in ["scrna", "snrna", "visium", "xenium"]:
            raise InputError(
                f"{technology} is not a valid key for technology"
            )

        self.adata_raw = adata
        self.paths = paths
        self.batch_names = ids
        self.metadata = metadata
        self.batch_key = batch_key
        self.technology = technology
        self.report = report

        self.remove_doublets = remove_doublets
        self.doublet_tool = doublet_tool

        self.min_genes_in_cell = min_genes_in_cell
        self.min_cells_with_genes = min_cells_with_genes
        self.cut_mt = cut_mt
        self.n_reads = n_reads
        self.min_counts = min_counts
        self.max_counts = max_counts
        self.min_genes = min_genes
        self.max_genes = max_genes
        self.low_quantile = low_quantile
        self.high_quantile = high_quantile
        self.random_state = random_state
        self.norm_method = normalisation_method
        self.log_data = log_data
        self.patterns = _lower_strings(metrics_patterns)
        self.pattern_names = metrics_names

        self.qc_path = None
        self.history = []
        self.adata = None

    def _read_data(self, path: PathLike, batch_name: str) -> ad.AnnData:
        """Reads data into an AnnData object.

        Reads data in H5 format or a 10x-Genomics-formated mtx directory into an AnnData object. If
        metadata was provided in `__init__` this will be added. If `technology` is set to `visium`
        the `squidpy.read_visium` variant will be used.

        :param path: path to H5 file or 10x-Genomics-formated mtx directory.
        :param batch_name: Name of the batch.
        :return: Returns an AnnData Object.
        """

        if self.technology in ("snrna", "scrna"):
            if path.endswith(".h5"):
                adata = read_10x_h5(path)
            elif os.path.isdir(path):
                adata = read_10x_mtx(path)
            else:
                raise InputError(
                    f"The input path is neither an H5 file or a 10x-Genomics-formated mtx directory:\n{path}"
                )
        elif self.technology == "visium":
            try:
                adata = read_visium(path, library_id=batch_name, load_images=True)
            except Exception as e:
                raise InputError(
                    f"{path} is not recognise as a Visium folder"
                )
        else:
            raise NotImplementedError(f"{self.technology} is not a valid technology")

        adata.var_names_make_unique()
        adata.obs[self.batch_key] = batch_name  # Add batch name
        if self.metadata:  # Map metadata
            for key, value in self.metadata.items():
                adata.obs[key] = adata.obs[self.batch_key].map(
                    (dict(zip(self.batch_names, value, strict=True))))
        return adata

    @staticmethod
    def _check_data(adata: ad.AnnData) -> None:
        """Check if `adata.X` contains non-negative integers.

        :param adata: Annotated data matrix.
        :return: Returns None.
        """
        x_is_raw_counts(adata)
        adata.layers["counts"] = adata.X.copy()
        return None

    def generate_report(self, df: pd.DataFrame, filename: str | Path, batch_name: str) -> None:
        """Generate a report.

        Generate an ExcelSheet that contains the history of the quality control process (i.e.,
        how many genes and features were removed after each quality control step).

        :param df: Dataframe with the history.
        :param filename: Name of the file.
        :param batch_name: Name of the batch
        :return: Returns None.
        """
        from dotools_py.pl._plot_utils import make_grid_spec
        from dotools_py.utility import get_hex_colormaps
        import matplotlib.lines as mlines

        today = date.today().strftime("%y%m%d")
        df_plot = df.iloc[:, :-1].melt(id_vars="QC_Step")  # Exclude comments
        fig, gs = make_grid_spec(
            None or (8, 5), nrows=1, ncols=2, wspace=0.7 / 6, width_ratios=[6 - (0.9 + 0) + 0, 0.9]
        )
        ax = fig.add_subplot(gs[0])
        bp = sns.barplot(
            df_plot, hue="QC_Step", x="value", y="variable", order=["nCells", "nFeatures"],
            hue_order=list(df["QC_Step"]), palette="tab10", ax=ax, legend=False
        )
        for container in bp.containers:
            ax.bar_label(container, fmt='{:,.0f}')
        bp.set_title("Summary Quality Control", fontdict={"weight": "bold"})
        bp.set_ylabel("", fontsize=18)
        bp.set_xlabel("Counts", fontsize=18)
        bp.set_yticklabels(bp.get_yticklabels(), rotation=90, va="center", fontdict={"weight": "bold"})
        axs_legend = fig.add_subplot(gs[1])
        colors_dict = dict(zip(list(df["QC_Step"]), get_hex_colormaps("tab10"), strict=False))
        handles = []
        for lab, c in colors_dict.items():
            handles.append(mlines.Line2D([0], [0], marker=".", color=c, lw=0, label=lab,
                                         markerfacecolor=c, markeredgecolor=None, markersize=18))
        legend = axs_legend.legend(handles=handles, frameon=False, loc="center left", ncols=1, title="",
                                   prop={"size": 12, "weight": "bold"})
        legend.get_title().set_fontweight("bold")
        legend.get_title().set_fontsize(12 + 2)
        axs_legend.tick_params(axis="both", left=False, labelleft=False, labelright=False, bottom=False,
                               labelbottom=False)
        axs_legend.spines[["right", "left", "top", "bottom"]].set_visible(False)
        axs_legend.grid(visible=False)
        plt.savefig(os.path.join(self.qc_path, f"{today}_QC_Metrics{batch_name}.svg"), bbox_inches="tight")
        plt.close(fig)

        # Save Metric File
        df.to_excel(os.path.join(self.qc_path, filename), index=False)
        return None

    def compute_metrics(self, adata: ad.AnnData) -> None:
        """Calculate quality control metrics.

        :param adata: Annodated data matrix with raw counts in `adata.X`.
        :return: Returns None. Changes are made inplace.
        """
        import scanpy as sc

        self._check_data(adata)  # Input should be raw counts

        adata.var["gene_names"] = adata.var_names.str.lower()
        for idx, pttn in enumerate(self.patterns):
            adata.var[self.pattern_names[idx]] = adata.var["gene_names"].str.startswith(pttn)
        sc.pp.calculate_qc_metrics(
            adata, qc_vars=self.pattern_names, percent_top=None, log1p=True, inplace=True, parallel=True
        )
        return None

    def filter_cells_genes(self, adata: ad.AnnData) -> ad.AnnData:
        """Filters cells and features.

        :param adata: Unprocessed AnnData object.
        :return:
        """
        import scanpy as sc

        # Step 1 - Basic filtering
        logger.info("Remove cells with low number of genes")
        sc.pp.filter_cells(adata, min_genes=self.min_genes_in_cell, inplace=True)
        self.history.append([
            "Rm_Cells_lowGenes", adata.shape[0], adata.shape[1], f"Remove cells with <{self.min_genes_in_cell} genes"
        ])
        logger.info("Remove genes lowly expressed")
        sc.pp.filter_genes(adata, min_cells=self.min_cells_with_genes, inplace=True)
        self.history.append([
            "Rm_Genes_lowCells", adata.shape[0], adata.shape[1],
            f"Remove genes express in less than {self.min_cells_with_genes} cells",
        ])

        # Step 2 - Removed cells with high mitochondrial content
        if self.cut_mt is not None:
            logger.info("Remove cells with high Mt-content")
            if "pct_counts_mt" not in adata.obs.columns:
                logger.warn("Cannot remove cells based on mitochondrial content because 'pct_counts_mt' is not"
                            "in adata.obs")
            else:
                adata = adata[adata.obs["pct_counts_mt"] < self.cut_mt, :].copy()
                self.history.append([
                    "Rm_Cell_HighMT", adata.shape[0], adata.shape[1],
                    f"Remove cells with >{self.cut_mt}% of Mitochondrial genes",
                ])

        # Step 3 - Remove low quality cells
        logger.info("Remove cells based on nUMI counts")
        assert (self.min_counts is None) != (self.low_quantile is None), "Set min_count or low_quantile"
        assert (self.max_counts is None) != (self.high_quantile is None), "Set max_count or high_quantile"

        if self.min_counts is not None:
            sc.pp.filter_cells(adata, min_counts=self.min_counts)
        if self.max_counts is not None:
            sc.pp.filter_cells(adata, max_counts=self.max_counts)
        if self.min_genes is not None:
            sc.pp.filter_cells(adata, min_genes=self.min_genes)
        if self.max_genes is not None:
            sc.pp.filter_cells(adata, max_genes=self.max_genes)

        # Apply quantile-based filtering (conditionally)
        adata = _filter_quantiles(adata, self.low_quantile, self.high_quantile)
        self.history.append([
            "Rm_Cells_nUMI_nGenes", adata.shape[0], adata.shape[1],
            f"Remove cells based on nUMI counts[Absolute (Min/Max): {self.min_counts}/{self.max_counts}, "
            f"Quantile (low/high): {self.low_quantile}/{self.high_quantile}] and nFeatures [Absolute (Min/Max): "
            f"{self.min_genes}/{self.max_genes}]",
        ])

        # Step 4 - Remove doublets
        if self.remove_doublets:
            find_doublets(adata, batch_key=self.batch_key, method=self.doublet_tool, random_state=self.random_state)
            n_doublets = adata.obs["doublet_class"].value_counts()["doublet"]
            adata = adata[adata.obs["doublet_class"] == "singlet"].copy()
            logger.info(f"Removed {n_doublets} doublets")
            self.history.append([
                "Rm_Doublets", adata.shape[0], adata.shape[1], f"Remove neotypic doublets using {self.doublet_tool}"
            ])
        return adata

    def scrna_quality_control(self) -> None:
        """Quality Control pipeline for sc/snRNA-seq.

        :return: Returns None.
        """

        database = {}
        today = date.today().strftime("%y%m%d")

        if self.adata_raw is not None:  # Case 1 - Input is AnnData
            for batch_name in self.adata_raw.obs[self.batch_key].unique():
                adata_batch = self.adata_raw[self.adata_raw.obs[self.batch_key] == batch_name].copy()
                self.compute_metrics(adata_batch)
                adata_batch = self.filter_cells_genes(adata_batch)
                database[batch_name] = adata_batch

        else:  # Case 2 - Input is paths
            for idx, path in enumerate(self.paths):
                self.qc_path = convert_path("/".join(path.split("/")[:-1]))
                logger.info(f"QualityControl Plots will be saved in\n{self.qc_path}")

                batch_name = self.batch_names[idx]
                adata_batch = self._read_data(path, batch_name=batch_name)
                self.compute_metrics(adata_batch)

                # Metrics
                metrics_filename = f"{today}_Metrics_{batch_name}.xlsx"
                self.history = []
                self.history.append(["Input_Shape", adata_batch.shape[0], adata_batch.shape[1], ""])
                adata_batch = self.filter_cells_genes(adata_batch)

                _qc_vln(
                    adata_batch, title=f"PostQC for {batch_name}", path=self.qc_path,
                    filename=f"Vln_PostQC_{batch_name}.svg"
                )

                database[batch_name] = adata_batch
                if self.report:
                    df_history = pd.DataFrame(self.history, columns=["QC_Step", "nCells", "nFeatures", "Comments"])
                    self.generate_report(df=df_history, filename=metrics_filename, batch_name=batch_name)

        logger.info("Concatenating samples")
        adata = ad.concat(
            database.values(), label=self.batch_key, keys=database.keys(), join="outer", index_unique="-", fill_value=0
        )
        self.adata = adata
        return None

    def normalise(self) -> None:
        """Normalise AnnData Object.

        :return: Returns None.
        """
        if self.adata is None:
            raise ValueError("The data has not been processed, run quality_control")
        logger.info("Normalisation of the expression")
        if self.norm_method == "LogNormalisation":
            _normalise(self.adata, n_reads=self.n_reads, log_data=self.log_data)
        elif self.norm_method == "PearsonResiduals":
            raise NotImplementedError("Not implemented")
        else:
            raise ValueError("Not a valid method, use 'LogNormalisation' or 'PearsonResiduals'")
        return None

    def run_pca(self) -> None:
        """Compute HVGs and compute PCA using HVGs.

        :return: Returns None.
        """
        import scanpy as sc

        logger.info("Finding Highly Variable Genes shared across samples")
        sc.pp.highly_variable_genes(self.adata, batch_key=self.batch_key)

        logger.info("Run PCA")
        hvg = self.adata[:, self.adata.var.highly_variable].copy()
        sc.pp.scale(hvg, zero_center=True)  # Scale only on HVGs to replicate Seurat Approach
        sc.pp.pca(hvg, random_state=self.random_state)  # PCA on Scaled HVGs
        self.adata.obsm["X_pca"] = hvg.obsm["X_pca"].copy()  # Save in original object
        return None

    @property
    def get_adata(self) -> ad.AnnData:
        """Return the AnnData.

        :return: Returns AnnData.
        """
        return self.adata

    def visium_quality_control(self) -> None:
        """Quality Control pipeline for Visium.

        :return: Returns None.
        """
        database = {}
        today = date.today().strftime("%y%m%d")

        # Technology specific steps
        if self.cut_mt is not None:
            logger.info("For Visium, filtering based on Mitochondrial content is not recommended, ignoring this step")
            self.cut_mt = None
        if not self.remove_doublets:
            logger.info("For Visium, removing doublets is not recommended, ignoring this step")
            self.remove_doublets = False
        if "^Hb.*-" not in self.patterns:
            logger.info("For Visium, identifying hemoglobin genes is recommended, adding the pattern")
            self.patterns = iterase_input(self.patterns) + ["^hb.*-"]
            self.pattern_names = iterase_input(self.pattern_names) + ["hb"]

        if self.adata_raw is not None:  # Case 1 - Input is AnnData
            for batch_name in self.adata_raw.obs[self.batch_key].unique():
                adata_batch = self.adata_raw[self.adata_raw.obs[self.batch_key] == batch_name].copy()
                self.compute_metrics(adata_batch)
                adata_batch = self.filter_cells_genes(adata_batch)
                database[batch_name] = adata_batch

        else:  # Case 2 - Input is paths
            for idx, path in enumerate(self.paths):
                self.qc_path = convert_path("/".join(path.split("/")[:-1]))
                logger.info(f"QualityControl Plots will be saved in\n{self.qc_path}")

                batch_name = self.batch_names[idx]
                adata_batch = self._read_data(path, batch_name=batch_name)
                self.compute_metrics(adata_batch)

                # Metrics
                metrics_filename = f"{today}_Metrics_{batch_name}.xlsx"
                self.history = []
                self.history.append(["Input_Shape", adata_batch.shape[0], adata_batch.shape[1], ""])
                adata_batch = self.filter_cells_genes(adata_batch)

                _qc_vln(
                    adata_batch, title=f"PostQC for {batch_name}", path=self.qc_path,
                    filename=f"Vln_PostQC_{batch_name}.svg",
                    stats=["total_counts", "n_genes_by_counts", "pct_counts_mt", "pct_counts_hb"],
                )

                database[batch_name] = adata_batch
                if self.report:
                    df_history = pd.DataFrame(self.history, columns=["QC_Step", "nCells", "nFeatures", "Comments"])
                    self.generate_report(df=df_history, filename=metrics_filename, batch_name=batch_name)

        logger.info("Concatenating samples")
        adata = ad.concat(
            database.values(), label=self.batch_key, keys=database.keys(), join="outer", index_unique="-", fill_value=0
        )

        if "spatial" not in adata.uns.keys():
            uns_spatial = {}
            for key, val in database.items():
                uns_spatial[key] = val.uns["spatial"][key]
            adata.uns["spatial"] = uns_spatial
        self.adata = adata
        return None


def importer_py(
    # Step 1 - Basic input
    paths: list,
    ids: list,
    metadata: dict | None = None,
    batch_key: str = "batch",
    # Step 2 - Filter low quality cells and features
    min_genes_in_cell: int = 300,
    min_cells_with_genes: int = 5,
    cut_mt: int | None = 5,
    n_reads: int = 10_000,
    min_counts: int | None = None,
    max_counts: int | None = None,
    min_genes: int | None = None,
    max_genes: int | None = None,
    low_quantile: int | None = None,
    high_quantile: int | None = None,
    remove_doublets: bool = True,
    doublet_tool: Literal["scDblFinder", "Scrublet", "DoubletDetection"] = "scDblFinder",
    # Extra processing steps and configurations
    normalisation_method: Literal["LogNormalisation", "PearsonResiduals"] = "LogNormalisation",
    log_data: bool = True,
    metrics_patterns: tuple = ("mt-", ("rbs", "rpl")),
    metrics_names: list = ("mt", "ribo"),
    random_state: int = 0,
    technology: Literal["snrna", "scrna", "visium", "xenium"] = "snrna",
):
    """Quality control analysis for scRNA / Spatial Transcriptomics.

    The input is a list with paths to H5 files generated with
    `CellRanger <https://www.10xgenomics.com/support/software/cell-ranger/latest>`_,
    `Cellbender <https://cellbender.readthedocs.io/en/latest/>`_, or
    `STARsolo <https://github.com/alexdobin/STAR>`_ or `SpaceRanger <https://www.10xgenomics.com/support/software/space-ranger/latest>`_
    A list of batch names for each sample must also be provided.
    Optionally, a dictionary with additional metadata can be passed. The order of batch names and metadata must
    match the order of the file paths.

    For each sample, several quality and filtering steps are applied:

    - Filter genes expressed in a low number of cells.
    - Filter cells with a low number of genes.
    - Filter cells with high mitochondrial content (recommended: 5% for scRNA, 3% for snRNA).
    - Filter cells based on nUMI and features using two modes:
        1. **Absolute filtering**: Sets absolute values for min/max UMI and features.
        2. **Quantile filtering**: Filters top/lower quantiles.
    - Remove doublets using scDblFinder, Scrublet, or DoubletDetection.

    An Excel sheet summarizing how many cells/genes were removed at each step will be generated,
    along with violin plots showing the distribution of `total_counts`, `n_genes_by_counts`,
    and `pct_mt_content` before and after QC. These outputs will be saved in the folder containing the H5 files.

    After QC, the data will be log-normalized and scaled. Highly variable genes and PCA will also be computed.

    .. note::
        Depending on the type of technology some steps will be omitted or adapted.

    :param paths: list with the path to the H5 files.
    :param ids: list with the batch name for each sample.
    :param metadata: dictionary with metadata information.
    :param batch_key: key in `.obs` for the batch information.
    :param remove_doublets: if set to True, neotypic doublets will be removed.
    :param doublet_tool: doublet tool to use. Available scDblFinder, Scrublet and DoubletDetection.
    :param min_genes_in_cell: minimum number of genes per cell.
    :param min_cells_with_genes: minimum cells expressing a genes.
    :param n_reads: target sum after normalization per cell.
    :param cut_mt: maximum percentage of mitochondrial genes per cell.
    :param min_counts:  minimum number of counts per cell.
    :param max_counts: maximum number of counts per cell.
    :param min_genes: minimum number of genes per cell.
    :param max_genes: maximum number of genes per cell.
    :param low_quantile: low quantile to filter cells based on counts.
    :param high_quantile: upper quantile to filter cells based on counts.
    :param normalisation_method: Type of normalization method.
    :param log_data: Whether to log data after normalization or not.
    :param metrics_patterns: Patterns to use to annotate features. Use `mt-` for mitochondrial, `rps` and `rpl` for ribosomal, and `^hb*-` for hemoglobin. Should be written in lowercase.
    :param metrics_names: Name for the patterns use "mt" for mitochondrial, "ribo" for ribosomal and "hb" for hemoglobin.
    :param technology: Type of the input dataset.
    :param random_state: seed for random number generator.
    :return: Returns an Annotated data matrix of shape `n_obs` x `n_vars` with all the samples concatenated.

    Example
    -------
    >>> import dotools_py as do
    >>> paths = ["/path/sample1", "/path/sample2"]
    >>> batchname = ["sample1", "sample2"]
    >>> metadata = {
    ...     "condition": ["WT", "KO"],
    ...     "age": ["3m", "3m"],
    ... }
    >>> adata = do.pp.importer_py(
    ...     paths=paths,
    ...     ids=batchname,
    ...     metadata=metadata,
    ...     batch_key="batch",
    ...     remove_doublets=True,
    ...     min_genes_in_cell=300,
    ...     min_cells_with_genes=5,
    ...     n_reads=10_000,
    ...     cut_mt=5,
    ...     high_quantile=95,
    ...     min_counts=500,
    ... )
    """

    # Checks
    invalid = next((p for p in paths if not os.path.exists(p)), None)
    if invalid:
        raise InputError(f"{invalid} is not a valid path")

    assert len(ids) == len(paths), "The numbers of paths does not match the number of ids"

    if metadata:
        assert all([len(metadata[key]) == len(paths) for key in metadata]), (
            f"Some metadata does not have {len(paths)} values"
        )
    # Warnings
    if cut_mt == 5 and technology == "snrna":
        logger.warn(
            "For snRNA a lower 'cut_mt' is recommended since mitochondrial genes\n should not be highly "
            "expressed in the nuclei")

    processor = Importer(
        adata=None,
        paths=paths,
        ids=ids,
        metadata=metadata,
        batch_key=batch_key,
        remove_doublets=remove_doublets,
        doublet_tool=doublet_tool,
        min_genes_in_cell=min_genes_in_cell,
        min_cells_with_genes=min_cells_with_genes,
        cut_mt=cut_mt,
        n_reads=n_reads,
        min_counts=min_counts,
        max_counts=max_counts,
        min_genes=min_genes,
        max_genes=max_genes,
        low_quantile=low_quantile,
        high_quantile=high_quantile,
        random_state=random_state,
        technology=technology,
        normalisation_method=normalisation_method,
        log_data=log_data,
        report=True,
        metrics_patterns=metrics_patterns,
        metrics_names=metrics_names,
    )

    if technology in ("scrna", "snrna"):
        processor.scrna_quality_control()
    elif technology == "visium":
        processor.visium_quality_control()
    else:
        raise NotImplementedError(f"{technology} is currently not implemented")

    processor.normalise()
    processor.run_pca()
    adata = processor.get_adata
    return adata


def quality_control(
    # Step 1 - Basic input
    adata: ad.AnnData,
    batch_key: str,
    # Step 2 - Filter low quality cells and features
    min_genes_in_cell: int = 300,
    min_cells_with_genes: int = 5,
    cut_mt: int = 5,
    n_reads: int = 10_000,
    min_counts: int | None = None,
    max_counts: int | None = None,
    min_genes: int | None = None,
    max_genes: int | None = None,
    low_quantile: int | None = None,
    high_quantile: int | None = None,
    remove_doublets: bool = False,
    doublet_tool: Literal["scDblFinder", "DoubletDetection", "Scrublet"] = "scDblFinder",
    # Extra processing steps and configurations
    normalization_method: Literal["LogNormalisation", "PearsonResiduals"] = "LogNormalisation",
    log_data: bool = True,
    metrics_patterns: tuple = ("mt-", ("rbs", "rpl")),
    metrics_names: list = ("mt", "ribo"),
    random_state: int = 0,
    technology: Literal["snrna", "scrna", "visium", "xenium"] = "snrna",
) -> ad.AnnData:
    """Basic quality control for snRNA-seq / Spatial Transcriptomics.

    For each sample in an AnnData object, several quality and filtering steps are applied:

    - Filter genes expressed in a low number of cells.
    - Filter cells with a low number of genes.
    - Filter cells with high mitochondrial content (recommended: 5% for scRNA, 3% for snRNA).
    - Filter cells based on nUMI and features using two modes:
        1. **Absolute filtering**: Sets absolute values for min/max UMI and features.
        2. **Quantile filtering**: Filters top/lower quantiles.
    - Remove doublets using scDblFinder, Scrublet, or DoubletDetection.


    .. note::
        This function reproduces the quality control steps of :func:`dotools_py.pp.importer_py` but allows
        to provide an AnnData object as input.  This function assumes that `adata.X` contains raw counts.

    .. warning::
        Depending on the `technology` some steps will be omitted or adapted.

    Parameters
    ----------
    adata
        Annotated data matrix with raw counts in `adata.X`.
    batch_key
        Column in `adata.obs` with sample information.
    min_genes_in_cell
        Minimum number of genes per cell.
    min_cells_with_genes
        Minimum number of cells expressing a gene.
    cut_mt
         Maximum percentage of mitochondrial genes per cell.
    n_reads
        Target sum after normalization per cell.
    min_counts
        Minimum number of counts per cell.
    max_counts
        Maximum number of counts per cell.
    min_genes
        Minimum number of genes per cell.
    max_genes
        Maximum number of genes per cell.
    low_quantile
        Low quantile to filter cells based on counts.
    high_quantile
        Upper quantile to filter cells based on counts.
    remove_doublets
        Identify and remove doublets.
    doublet_tool
        Method to use for the removal of doublets.
    normalization_method
        Normalization method to use.
    log_data
        Whether to log the data after normalization or not.
    metrics_patterns
        Patterns to use to annotate features. Use `mt-` for mitochondrial, `rps` and `rpl` for ribosomal,
        and `^hb*-` for hemoglobin. Should be written in lowercase.
    metrics_names
        Name for the patterns use "mt" for mitochondrial, "ribo" for ribosomal and "hb" for hemoglobin.
    technology
        Type of the input dataset.
    random_state
        Seed for random number generator,

    Returns
    -------
    Returns a processed AnnData object.

    """
    # Warnings
    if cut_mt == 5 and technology == "snrna":
        logger.warn(
            "For snRNA a lower 'cut_mt' is recommended since mitochondrial genes\n should not be highly "
            "expressed in the nuclei")

    processor = Importer(
        adata=adata,
        paths=None,
        ids=None,
        metadata=None,
        batch_key=batch_key,
        remove_doublets=remove_doublets,
        doublet_tool=doublet_tool,
        min_genes_in_cell=min_genes_in_cell,
        min_cells_with_genes=min_cells_with_genes,
        cut_mt=cut_mt,
        n_reads=n_reads,
        min_counts=min_counts,
        max_counts=max_counts,
        min_genes=min_genes,
        max_genes=max_genes,
        low_quantile=low_quantile,
        high_quantile=high_quantile,
        random_state=random_state,
        technology=technology,
        normalisation_method=normalization_method,
        log_data=log_data,
        report=False,
        metrics_patterns=metrics_patterns,
        metrics_names=metrics_names,
    )

    if technology in ("scrna", "snrna"):
        processor.scrna_quality_control()
    elif technology == "visium":
        processor.visium_quality_control()
    else:
        raise NotImplementedError(f"{technology} is currently not implemented")

    processor.normalise()
    processor.run_pca()
    adata = processor.get_adata
    return adata
