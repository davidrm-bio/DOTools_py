import os
from typing import Literal
import subprocess
import uuid
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dotools_py import logger
from dotools_py.dt import standard_ct_labels_heart
from dotools_py.utils import convert_path, get_paths_utils, transfer_labels, check_missing
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scvi.model._scvi import SCVI
    from scvi.model._scanvi import SCANVI

DictUpdateCellLabels = standard_ct_labels_heart()


def _run_cca(
    adata: ad.AnnData,
    batch_key: str,
    version: str = "v4",
) -> np.ndarray:
    """Integrate AnnData using CCA from Seurat.

    :param adata: anndata object.
    :param batch_key: column in obs with batch IDs.
    :param version: version of Seurat to use.
    :return: integrated matrix.
    """
    import polars

    rscript = get_paths_utils("_run_CCA.R")

    tmpdir_path = Path("/tmp") / f"CCA_{uuid.uuid4().hex}"
    tmpdir_path.mkdir(parents=True, exist_ok=False)

    logger.info("Preprocessing to export to Seurat")
    del adata.uns, adata.raw
    adata.write(tmpdir_path / "adata_hvg.h5ad")

    logger.info("Running CCA Integration")
    in_path = os.path.join(tmpdir_path, "adata_hvg.h5ad")

    cmd = [
        "Rscript",
        rscript,
        "--input=" + str(in_path),
        "--out=" + str(tmpdir_path) + "/",
        "--name=" + batch_key,
        "--version=" + version,
    ]

    subprocess.call(cmd)

    logger.info("Loading corrected matrix")
    cca_matrix = polars.read_csv(
        os.path.join(tmpdir_path, "adata_hvg_seurat_AnchorIntegration.csv"), infer_schema_length=0
    )
    cca_matrix = cca_matrix.to_pandas().astype(float)
    return cca_matrix.values


def _run_scvi(
    adata: ad.AnnData,
    batch_key: str,
    layer_counts: str = "counts",
    categorical_covariates: list = None,
    continuous_covariates: list = None,
    n_hidden: int = 128,
    n_latent: int = 30,
    n_layers: int = 3,
    dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene-batch",
    gene_likelihood: Literal["zinb", "nb", "poisson", "normal"] = "zinb",
    get_model: bool = False,
    gene_key: str = "highly_variable",
    **kwargs,
) -> "SCVI | None":
    """Run scVI.

    Run scVI to integrate sc/snRNA more information on
    `scvi-tools <https://docs.scvi-tools.org/en/stable/api/reference/scvi.model.SCVI.html>`_.

    :param adata: annotated dt matrix.
    :param batch_key: `.obs` column with batch information.
    :param layer_counts: layer with counts. Raw counts are required.
    :param categorical_covariates: `.obs` column names with categorical covariates for scVI inference.
    :param continuous_covariates: `.obs` column names with continuous covariates for scVI inference.
    :param n_hidden: number of hidden layers.
    :param n_latent: dimensions of the latent space.
    :param n_layers: number of layers.
    :param dispersion: dispersion mode for scVI.
    :param gene_likelihood: gene likelihood.
    :param get_model: return the trained model.
    :param kwargs: additional arguments for `scvi.model.SCVI`.
    :return: None or the model, the latent space is saved in the anndata under X_scVI.
    """
    import scvi

    logger.info("Run scVI")
    assert layer_counts in adata.layers, "counts layer not in anndata"
    assert gene_key in list(adata.var.columns), f"{gene_key} not in adata.var"

    # Integration using only HVG
    hvg = adata[:, adata.var[gene_key]].copy()

    # Set-up anndata and model
    scvi.model.SCVI.setup_anndata(
        hvg,
        layer=layer_counts,
        batch_key=batch_key,
        continuous_covariate_keys=continuous_covariates,
        categorical_covariate_keys=categorical_covariates,
    )

    model_scvi = scvi.model.SCVI(
        hvg,
        n_hidden=n_hidden,
        n_latent=n_latent,
        n_layers=n_layers,
        dispersion=dispersion,
        gene_likelihood=gene_likelihood,
        **kwargs,
    )

    model_scvi.view_anndata_setup()
    model_scvi.train()  # Train
    adata.obsm["X_scVI"] = model_scvi.get_latent_representation()

    if get_model:
        return model_scvi
    else:
        del model_scvi
        return None


def run_scvi(
    adata: ad.AnnData,
    batch_key: str,
    gene_key: str | Literal["all"] = "highly_variable",
    layer_counts: str = "counts",
    categorical_covariates: list = None,
    continuous_covariates: list = None,
    n_hidden: int = 128,
    n_latent: int = 30,
    n_layers: int = 3,
    dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene-batch",
    gene_likelihood: Literal["zinb", "nb", "poisson", "normal"] = "zinb",
    get_model: bool = False,
    **kwargs,
) -> "SCVI | None":
    """Run scVI.

    Run scVI to integrate sc/snRNA more information on
    `scvi-tools <https://docs.scvi-tools.org/en/stable/api/reference/scvi.model.SCVI.html>`_.

    Parameters
    ----------
    adata
        Annotated data matrix.
    batch_key
        Column in `adata.obs` with batch information.
    gene_key
        Boolean column in `adata.var` used to select the genes that will be used for the inference.
    layer_counts
        Layer in `adata.layers` with raw counts.
    categorical_covariates
        Column in `adata.obs` with categorical covariates to correct for during scVI inference.
    continuous_covariates
        Column in `adara.obs` with continuous covariates to correct for during scVI inference.
    n_hidden
        Number of hidden layers.
    n_latent
        Dimensions of the latent space.
    n_layers
        Number of layers
    dispersion
        Gene dispersion mode for scVI.
    gene_likelihood
        Gene likelihood.
    get_model
        Return the trained model.
    kwargs
        Additional arguments for `scvi.model.SCVI <https://docs.scvi-tools.org/en/stable/api/reference/scvi.model.SCVI.html#scvi.model.SCVI>`_.

    Returns
    -------
    Returns `None` or the trained scVI model if `get_model` is set to `True`.
    The latent space is saved in the AnnData under X_scVI.

    """

    if gene_key == "all":
        adata.var["scvi_genes"] = True
        gene_key = "scvi_genes"

    model = _run_scvi(
        adata=adata,
        batch_key=batch_key,
        layer_counts=layer_counts,
        categorical_covariates=categorical_covariates,
        continuous_covariates=continuous_covariates,
        n_hidden=n_hidden,
        n_latent=n_latent,
        n_layers=n_layers,
        dispersion=dispersion,
        gene_likelihood=gene_likelihood,
        get_model=get_model,
        gene_key=gene_key,
        **kwargs,
    )

    return model


def run_scanvi(
    adata: ad.AnnData,
    batch_key: str,
    label_key: str,
    unlabel_group: str = "unknown",
    scvi_model: "SCVI" = None,
    gene_key: str | Literal["all"] = "highly_variable",
    layer_counts: str = "counts",
    categorical_covariates: list = None,
    continuous_covariates: list = None,
    n_hidden: int = 128,
    n_latent: int = 30,
    n_layers: int = 3,
    dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene-batch",
    gene_likelihood: Literal["zinb", "nb", "poisson", "normal"] = "zinb",
    get_model: bool = False,
    scvi_kwargs: dict = None,
    scanvi_kwargs: dict = None
) -> "None | SCANVI":
    """Run scANVI.

    Run scANVI to integrate sc/snRNA more information on
    `scvi-tools <https://docs.scvi-tools.org/en/stable/api/reference/scvi.model.SCANVI.html>`_.

    Parameters
    ----------
    adata
        Annotated data matrix.
    batch_key
        Column in `adata.obs` with batch information.
    label_key
        Column in `adata.obs` with label information.
    unlabel_group
        Value used for unlabeled cells in `labels_key`
    scvi_model
        Trained scVI model.
    gene_key
        Boolean column in `adata.var` used to select the genes that will be used for the inference.
    layer_counts
        Layer in `adata.layers` with raw counts.
    categorical_covariates
        Column in `adata.obs` with categorical covariates to correct for during scVI inference.
    continuous_covariates
        Column in `adara.obs` with continuous covariates to correct for during scVI inference.
    n_hidden
        Number of hidden layers.
    n_latent
        Dimensions of the latent space.
    n_layers
        Number of layers
    dispersion
        Gene dispersion mode for scVI.
    gene_likelihood
        Gene likelihood.
    get_model
        Return the trained scANVI model.
    scvi_kwargs
        Additional arguments for `scvi.model.SCVI <https://docs.scvi-tools.org/en/stable/api/reference/scvi.model.SCVI.html#scvi.model.SCVI>`_.
    scanvi_kwargs
        Additional arguments for `scvi.model.SCANVI <https://docs.scvi-tools.org/en/stable/api/reference/scvi.model.SCVI.html#scvi.model.SCANVI>`_.

    Returns
    -------
    Returns `None` or the trained scANVI model if `get_model` is set to `True`.
    The latent space is saved in the AnnData under X_scANVI.

    """
    import scvi
    logger.info("Run scVI")
    scvi_kwargs = {} if scvi_kwargs is None else scvi_kwargs
    scanvi_kwargs = {} if scanvi_kwargs is None else scanvi_kwargs

    if scvi_model is None:
        scvi_model = run_scvi(
            adata=adata,
            batch_key=batch_key,
            layer_counts=layer_counts,
            categorical_covariates=categorical_covariates,
            continuous_covariates=continuous_covariates,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            get_model=True,
            gene_key=gene_key,
            **scvi_kwargs
        )

    logger.info("Run scANVI")

    model_scanvi = scvi.model.SCANVI.from_scvi_model(
        scvi_model, labels_key=label_key, unlabeled_category=unlabel_group, **scanvi_kwargs
    )
    model_scanvi.view_anndata_setup()
    model_scanvi.train()
    adata.obsm["X_scANVI"] = model_scanvi.get_latent_representation()

    if get_model:
        return model_scanvi
    else:
        del model_scanvi
        return None


def run_harmony(
    adata: ad.AnnData,
    batch_key: str,
    use_rep: str = "X_pca",
    rep_added: str = "X_harmony",
    random_state: int = 0,
    use_gpu: bool = False,
    max_iter_harmony: int = 150,
    workers: int = 1,
    **kwargs,
) -> None:
    """Run Harmony integration.

    This functions runs the Pytorch implementation of
    Harmony.

    :param adata: Annotated data matrix.
    :param batch_key: Key in adata.obs with batch information.
    :param use_rep: Representation in adata.obsm that represents the input embedding with rows for cells and columns for embedding coordinates.
    :param rep_added: Name of the adjusted embedding.
    :param random_state: Seed for random number generator.
    :param use_gpu: If set to `True` will use GPU if available.
    :param max_iter_harmony: Maximum number of iterations for harmony.
    :param workers: Number of threads to use.
    :param kwargs: Additional arguments pass to
                   `harmony.harmonize <https://github.com/lilab-bcb/harmony-pytorch/blob/main/harmony/harmony.py>`_.
    :return: Returns `None`.
             A new slot will be set in ``adata.obsm[rep_added]``.

    Example
    -------
    >>> import dotools_py as do
    >>> adata = do.dt.example_10x_processed()
    >>> adata.obsm_keys()
    ['X_CCA', 'X_pca', 'X_umap']
    >>> do.tl.run_harmony(adata, batch_key="batch")
    >>> adata.obsm_keys()
    ['X_CCA', 'X_pca', 'X_umap', 'X_harmony']

    """

    try:
        from harmony import harmonize
    except ImportError as e:
        msg = "\nplease install harmony-pytorch:\n\n\tpip install harmony-pytorch"
        raise ImportError(msg) from e

    x = adata.obsm[use_rep].astype(np.float64)
    z = harmonize(x, adata.obs, batch_key, use_gpu=use_gpu, verbose=True, random_state=random_state, max_iter_harmony=max_iter_harmony, n_jobs=workers, **kwargs)
    try:
        adata.obsm[rep_added] = z
    except Exception as e:
        adata.obsm[rep_added] = z.T

def integrate_data(
    adata: ad.AnnData,
    batch_key: str,
    hvg_batch: bool = True,
    integration_method: Literal["scanorama", "scvi", "cca4", "cca5", "harmony", "pca"] = "scvi",
    bbknn: bool = False,
    resolution: float = 0.3,
    categorical_covariates: list = None,
    continuous_covariates: list = None,
    get_model: bool = False,
    random_state: int = 0,
    workers: int = 1,
    **kwargs,
) -> "SCVI | None":
    """Integrate a concatenated AnnData.

    Integrate and perform batch correction for an AnnData with several samples. Different batch correction methods are
    available: `Harmony <https://www.nature.com/articles/s41592-019-0619-0>`_,
    `Scanorama <https://www.nature.com/articles/s41587-019-0113-3>`_,
    `BBKNN <https://academic.oup.com/bioinformatics/article/36/3/964/5545955?login=true>`_,
    `scVI <https://www.nature.com/articles/s41587-021-01206-w>`_ and
    `CCA <https://www.cell.com/cell/fulltext/S0092-8674%2819%2930559-8>`_ (v4 or v5).

    .. note::
        The integration method CCA is based on Seurat. The v4 will generate a corrected expression matrix of all the
        highly variable genes (HVGs) that is then used to perform dimensionality reduction. In v5 the dimensionality
        reduction is performed before producing the CCA embeddings.

    :param adata: Annotated data matrix.
    :param batch_key: Metadata column in `obs` with batch information.
    :param hvg_batch: If set to `True`, the highly variable genes shared across samples will be used for the
                     integration.
    :param integration_method: Method to use for the integration.
    :param bbknn: Use BBKNN to compute neighbors instead of sc.pp.neighbors().
    :param resolution: Resolution for the leiden clustering.
    :param categorical_covariates: Categorical covariates for scVI.
    :param continuous_covariates: Continuous covariates for scVI.
    :param get_model: Set to True to Return the scVI model.
    :param random_state: seed for random number generator.
    :param workers: number of threads to use for harmony.
    :param kwargs: Additional arguments for
                  `scVI model <https://docs.scvi-tools.org/en/stable/api/reference/scvi.model.SCVI.html>`_.
    :return: Returns `None` or the scVI model if `get_model` is `True`. The following fields will be set:

            `adata.obsm['X_pca']`: :class:`numpy.ndarray` (dtype ``float``)
                PCA representation of data.
            `adata.varm['PCs']` : :class:`numpy.ndarray`
                The principal components containing the loadings.
            `adata.uns['pca']['variance_ratio']` : :class:`numpy.ndarray` (shape `(n_comps,)`)
                Ratio of explained variance.
            `adata.uns['pca']['variance']` : :class:`numpy.ndarray` (shape `(n_comps,)`)
                Explained variance, equivalent to the eigenvalues of the
                covariance matrix.
            `adata.obsm[representation]`: :class:`numpy.ndarray` (dtype ``float``)
                Representation will be set to `X_pca_harmony` for harmony; `X_scanorama` for scanorama;
                `X_CCA` for CCA4/CC5, and `X_scVI` for scVI.
            `adata.obsp['distances']` : :class:`scipy.sparse.csr_matrix` (dtype `float`)
                Distance matrix of the nearest neighbors search.
            `adata.obsp['connectivities']` : :class:`scipy.sparse._csr.csr_matrix` (dtype `float`)
                Weighted adjacency matrix of the neighborhood graph of data points.
                Weights should be interpreted as connectivities.
            `adata.uns['neighbors']` : :class:`dict`
                neighbors parameters.
            `adata.obsm['X_umap']`: :class:`numpy.ndarray` (dtype ``float``)
                UMAP coordinates of the data
            `adata.obs['leiden']`: :class:`pandas.Series` (dtype ``category``)
                Array that stores the cluster groups.
            `adata.uns['leiden']['params']` : :class:`dict`
                A dict with the values for the parameters `resolution`, `random_state`,
                and `n_iterations`.

    Example
    -------
    >>> import dotools_py as do
    >>> adata = do.dt.example_10x_processed()
    >>> do.tl.integrate_data(adata, batch_key="batch", harmony=True)
    >>> adata
    AnnData object with n_obs × n_vars = 700 × 1851
    obs: 'batch', 'condition', 'n_genes_by_counts', 'log1p_n_genes_by_counts', 'total_counts', 'log1p_total_counts',
         'total_counts_mt', 'log1p_total_counts_mt', 'pct_counts_mt', 'total_counts_ribo', 'log1p_total_counts_ribo',
          'pct_counts_ribo', 'n_genes', 'n_counts', 'doublet_class', 'doublet_score', 'leiden', 'cell_type',
          'autoAnnot', 'celltypist_conf_score', 'annotation', 'annotation_recluster'
    var: 'mean', 'std', 'highly_variable', 'means', 'dispersions', 'dispersions_norm', 'highly_variable_nbatches',
         'highly_variable_intersection'
    uns: 'annotation_colors', 'annotation_recluster_colors', 'batch_colors', 'hvg', 'leiden', 'leiden_colors',
         'log1p', 'neighbors', 'pca', 'umap'
    obsm: 'X_CCA', 'X_pca', 'X_umap', 'X_harmony'
    varm: 'PCs'
    layers: 'counts', 'logcounts'
    obsp: 'connectivities', 'distances'

    """
    import scanpy.external as sce
    import scanpy as sc
    logger.info("Computing HVGs")
    hvg_batch = batch_key if hvg_batch else None
    sc.pp.highly_variable_genes(adata, batch_key=hvg_batch)
    hvg = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(hvg)
    sc.pp.pca(hvg, random_state=random_state)

    dim_reduc = "X_pca"
    model = None
    neighbors_within_batch = 25 if adata.n_obs > 100_000 else 3  # Community recommendations

    if integration_method == "harmony":
        logger.info("Integration using Harmony")
        run_harmony(hvg, batch_key=batch_key, max_iter_harmony=150, random_state=random_state, workers=workers)
        adata.obsm["X_harmony"] = hvg.obsm["X_harmony"]
        dim_reduc = "X_harmony"
    elif integration_method == "scanorama":
        logger.info("Integration using Scanorama")
        sce.pp.scanorama_integrate(hvg, key=batch_key)
        adata.obsm["X_scanorama"] = hvg.obsm["X_scanorama"]
        dim_reduc = "X_scanorama"
    elif integration_method == "scvi":
        logger.info("Integration using scVI")
        model = _run_scvi(adata, batch_key=batch_key,
                          categorical_covariates=categorical_covariates,
                          continuous_covariates=continuous_covariates,
                          get_model=get_model,
                          **kwargs,
                          )
        dim_reduc = "X_scVI"
    elif integration_method == "cca4":
        logger.info("Integration using CCA (Seurat v4 approach)")
        adata.obsm["X_CCA"] = _run_cca(hvg, batch_key, version="v4")
        logger.info("Using CCA matrix for PCA")
        hvg.X = adata.obsm["X_CCA"].copy()
        sc.pp.pca(hvg)
        adata.obsm["X_pca"] = hvg.obsm["X_pca"]
    elif integration_method == "cca5":
        logger.info("Integration using CCA (Seurat v5 approach)")
        adata.obsm["X_CCA"] = _run_cca(hvg, batch_key, version="v5")
        dim_reduc = "X_CCA"
    elif integration_method == "pca":
        pass
    else:
        raise NotImplementedError("Not a valid method")

    logger.info("Finding neighbors")

    if bbknn:
        logger.info("Computing neighbors with BBKNN")
        sce.pp.bbknn(adata, use_rep=dim_reduc, neighbors_within_batch=neighbors_within_batch, batch_key=batch_key,
                     pynndescent_random_state=random_state)
    else:
        sc.pp.neighbors(adata, use_rep=dim_reduc, random_state=42)

    logger.info("Run UMAP")
    sc.tl.umap(adata, random_state=random_state)

    logger.info(f"Clustering cells using Leiden (resolution {resolution})")
    sc.tl.leiden(adata, resolution=resolution, flavor="igraph", n_iterations=2, directed=False,
                 random_state=random_state)
    return model


def update_cell_labels(
    adata: ad.AnnData, cell_col: str,
    key_added: str = "annotation",
    dict_data: str | dict = "default"
) -> None:
    """Rename cell-type labels generated by Celltypist.

    This function will rename the cell type labels returned by Celltypist when using the Heart Model.

    :param adata: Anndata object previously analyzed by Celltypist.
    :param cell_col: Column in `obs` with cell type labels.
    :param key_added: Column in `obs` where new labels will be saved.
    :param dict_data: Dictionary with the labels to use to update the labels.
    :return: Returns `None`.
    """
    if dict_data == "default":
        dict_data = DictUpdateCellLabels

    adata.obs[key_added] = [
        dict_data[cell] if cell in dict_data else list(adata.obs[cell_col])[idx]
        for idx, cell in enumerate(list(adata.obs[cell_col]))
    ]
    return None


def auto_annot(
    adata: ad.AnnData,
    cluster_key: str,
    model: str = "Healthy_Adult_Heart.pkl",
    key_added: str = "autoAnnot",
    majority: bool = True,
    convert: bool = True,
    update_label: bool = False,
    key_updated: str = "annotation",
    verbose: bool = False,
    update_models: bool = False,
    dict_labels: dict | str = "default",
    pl_cell_prob: bool = False,
    path: str | None = None,
    filename: str | None = "Dotplot_CellProbabilities.svg",
) -> None:
    """Semi-automatic annotation based on CellTypist.

    This function takes an AnnData object with log-counts in `X` and annotate the clusters employing a model available
    for `Celltypist <https://www.celltypist.org/>`_.

    :param adata: Annotated data matrix.
    :param cluster_key: Metadata column in `obs` with cluster groups.
    :param model: `Celltypist model <https://www.celltypist.org/models>`_ to use for the prediction.
    :param key_added: New metadata column in `obs` to save the predicted cell types.
    :param majority: Whether to refine the predicted labels by running the majority voting classifier after
                     over-clustering.
    :param convert: Convert the gene format of the model. If a Human model is provided, and is set to `True`, then gene
                    in mouse format will be use and viceverse.
    :param update_label: Add a new metadata column in  `obs` with cell type labels updated based on `dict_labels`.
    :param key_updated: Metadata column in `obs` to save the updated cell type labels. Ignored if `update_labels` is
                        set to `False`.
    :param verbose: Whether to show information of the analysis steps.
    :param update_models: Download the latest models.
    :param dict_labels: Dictionary with the updated labels for the names in celltypist model. Currently, only a
                        dictionary for the `Human_Adult_Heart.pkl` model.
                        See :func:`dotools_py.dt.standard_ct_labels_heart()`
    :param pl_cell_prob: Generate a Dotplot to visualize the cell probabilities for each cluster.
    :param path: Path to save the dotplot of cell probabilities.
    :param filename: Name of the file.

    Returns
    -------
    Return `None`. The following fields will be set:

    `adata.obs['autoAnnot' | key_added]`: :class:`pandas.Series` (dtype ``category``)
        Array that stores the predicted annotation for each cell.
    `adata.obs['celltypist_conf_score']`: :class:`pandas.Series` (dtype ``float``)
        Array that stores the confidence scores for the prediction.
    `adata.obs['annotation' | key_updated]`: :class:`pandas.Series` (dtype ``category``)
         If `update_label` is set to True, this  field will be set and contains an array that stores the
         predicted annotation for each cell updated based on the dictionary `dict_labels`.

    Example
    -------
    >>> import dotools_py as do
    >>> adata = do.dt.example_10x_processed()
    >>> do.tl.auto_annot(adata, "leiden", model="Healthy_COVID19_PBMC.pkl", pl_cell_prob=False, convert=False)
    🔬 Input data has 700 cells and 1851 genes
    🔗 Matching reference genes in the model
    🧬 358 features used for prediction
    ⚖️ Scaling input data
    🖋️ Predicting labels
    ✅ Prediction done!
    🗳️ Majority voting the predictions
    ✅ Majority voting done!

    """
    import celltypist
    from tqdm import tqdm

    check_missing(adata, groups=cluster_key)

    if update_models:
        celltypist.models.download_models(force_update=True)

    adata_copy = adata.copy()
    steps = ["Setting-up", "Predicting", "Saving predictions", "Updating labels"]
    total_steps = len(steps) if update_label else len(steps) - 1

    with tqdm(total=total_steps, desc="Progress", disable=not verbose, colour="tomato") as pbar:
        # Get model
        pbar.set_description(steps.pop(0))
        model_loaded = celltypist.models.Model.load(model=model)
        if convert:
            model_loaded.convert()
        adata_copy.X = adata_copy.X.toarray()  # Leads to high memory usage
        pbar.update(1)

        # Do the prediction
        pbar.set_description(steps.pop(0))
        predictions_cells = celltypist.annotate(
            adata_copy, model=model_loaded, majority_voting=majority, over_clustering=cluster_key
        )
        pbar.update(1)

        # Save predictions
        pbar.set_description(steps.pop(0))
        predictions_cells_adata = predictions_cells.to_adata()

        prediction_key = "majority_voting" if majority else "predicted_labels"
        if pl_cell_prob:
            try:
                axs = celltypist.dotplot(
                    predictions_cells, use_as_prediction="predicted_labels", use_as_reference=cluster_key,
                    title="", show=False)
                axs["mainplot_ax"].spines[["top", "right"]].set_visible(True)
                if path is not None:
                    plt.savefig(convert_path(path) / filename, bbox_inches="tight")
            except Exception as e:
                logger.warn(f'Error plotting {e}')

        adata_copy.obs["cell_type"] = predictions_cells_adata.obs.loc[adata_copy.obs.index, prediction_key]
        adata.obs[key_added] = adata_copy.obs["cell_type"]  # Transfer to original object
        adata.obs["celltypist_conf_score"] = predictions_cells_adata.obs["conf_score"]
        pbar.update(1)

        if update_label:
            # Update labels
            pbar.set_description(steps.pop(0))
            update_cell_labels(adata, key_added, key_updated, dict_data=dict_labels)
            pbar.update(1)

    return None


def reclustering(
    adata: ad.AnnData,
    cluster_key: str,
    batch_key: str,
    recluster_approach: Literal["cca4", "cca5", "harmony", "scanorama", "pca", "scvi"],
    use_clusters: str | list | None = None,
    bbknn: bool = False,
    hvg_batch: bool = False,
    use_rep: str = None,
    resolution: float = 0.3,
    neighbors_batch: int = 3,
    automatic_annot: bool = False,
    majority: bool = True,
    convert: bool = True,
    model: str = "Healthy_Adult_Heart.pkl",
    get_subset: bool = False,
    key_added: str = "annotation_recluster",
    key_added_autoannot: str = "autoAnnot_recluster",
    random_state: int = 0,
) -> ad.AnnData | None:
    """Re-clustering of dataset.

    Perform reclustering on an integrated AnnData object. Can recluster for the following integration methods:
        * CCA (v4/v5) integration from Seurat.
        * Harmony integration.
        * BBKNN integration.
        * SCVI integration.
        * PCA.

    Assume that `X` has logcounts.

    .. note::
        For CCA (v4/v5) and scVI the corrected expression matrix (CC4 v5), the CCA representation
        (CCA v5) and the latent space (scvi) to be in `.obsm`. When re-clustering with harmony and
        BBKNN the pipeline will be re-run over the clusters.

    :param adata: Annotated data matrix.
    :param cluster_key: Metadata column in `obs` with cluster groups.
    :param batch_key: Metadata column in `obs` with batch groups.
    :param use_clusters: Clusters in `cluster_key` to re-cluster. If several clusters are provided,
                         the re-clustering will be performed subsetting for all the clusters specified.
    :param hvg_batch: If set to `True`. The  highly variable genes that are shared across samples will be used.
    :param recluster_approach: Reclustering approach to use.
    :param bbknn: Use BBKNN to compute neighbors.
    :param use_rep: Name in `obsm` with the representation. Required for SCVI, CCA and Scanorama approach.
    :param resolution: Resolution for the leiden clustering.
    :param neighbors_batch: To compute the nearest neighbors distance matrix and a neighborhood graph of observations a
                            `BBKNN <https://academic.oup.com/bioinformatics/article/36/3/964/5545955?login=true>`_
                            is employed, which calculate a batch balanced KNN graph. It is recommended to use 3 with
                            when <100000 cells and 25 for >100000. If there are not enough cells per batch the default
                            approach will be used (`sc.pp.neighbors`).
    :param automatic_annot: Perform semi-automatic annotation with
                            `Celltypist <https://www.science.org/doi/10.1126/science.abl5197>`_.
    :param majority: Whether to refine the predicted labels by running the majority voting classifier after
                    over-clustering.
    :param convert: Convert the gene format of the model. If a Human model is provided, and is set to `True`, then gene
                    in mouse format will be use and viceverse.
    :param model: `Celltypist model <https://www.celltypist.org/models>`_ to use for the prediction.
    :param get_subset: if set to `True`, returns an AnnData of `use_clusters` after re-clustering.
    :param key_added: metadata column name in `obs` to save reclustering information.
    :param key_added_autoannot: metadata column name in `obs` to save reclustering information after automatic annotation.
    :param random_state: seed for random number generator.
    :return: Returns `None` if `get_subset` is set to False, otherwise a subsetted AnnData after the re-clustering is
             returned. Additionally, the following fields will be set:

             `adata.obs['annotation_recluster' | key_added]` : :class:`pandas.Series` (dtype ``category``)
                Array that stores the re-clusters groups consisting of the original group_id + the new cluster id (e.g., for
                a the monocyte cluster with 3 sub-clusters the new clusters are monocyte_0, monocyte_1, and monocyte_2).
             `adata.obs['autoAnnot_recluster' | key_added_autoannot]` : :class:`pandas.Series` (dtype ``category``)
                Array that stores the re-clusters groups after re-running the automatic annotation pipeline.

    See Also
    --------
        :func:`dotools_py.tl.full_recluster`: Recluster all clusters automatically

    Example
    -------
    >>> import dotools_py as do
    >>> adata = do.dt.example_10x_processed()
    >>> t_cells = do.tl.reclustering(adata, "annotation", "batch", "harmony", use_clusters="T_cells", get_subset=True)
    >>> t_cells
    AnnData object with n_obs × n_vars = 464 × 1851
    obs: 'batch', 'condition', 'n_genes_by_counts', 'log1p_n_genes_by_counts', 'total_counts', 'log1p_total_counts',
         'total_counts_mt', 'log1p_total_counts_mt', 'pct_counts_mt', 'total_counts_ribo', 'log1p_total_counts_ribo',
         'pct_counts_ribo', 'n_genes', 'n_counts', 'doublet_class', 'doublet_score', 'leiden', 'cell_type',
         'autoAnnot', 'celltypist_conf_score', 'annotation', 'annotation_recluster'
    var: 'mean', 'std', 'highly_variable', 'means', 'dispersions', 'dispersions_norm', 'highly_variable_nbatches',
         'highly_variable_intersection'
    uns: 'annotation_colors', 'annotation_recluster_colors', 'batch_colors', 'hvg', 'leiden', 'leiden_colors', 'log1p',
         'neighbors', 'pca', 'umap'
    obsm: 'X_CCA', 'X_pca', 'X_umap', 'X_pca_harmony'
    varm: 'PCs'
    layers: 'counts', 'logcounts'
    obsp: 'connectivities', 'distances'
    >>> adata
    AnnData object with n_obs × n_vars = 700 × 1851
    obs: 'batch', 'condition', 'n_genes_by_counts', 'log1p_n_genes_by_counts', 'total_counts', 'log1p_total_counts',
         'total_counts_mt', 'log1p_total_counts_mt', 'pct_counts_mt', 'total_counts_ribo', 'log1p_total_counts_ribo',
         'pct_counts_ribo', 'n_genes', 'n_counts', 'doublet_class', 'doublet_score', 'leiden', 'cell_type', 'autoAnnot',
         'celltypist_conf_score', 'annotation', 'annotation_recluster'
    var: 'mean', 'std', 'highly_variable', 'means', 'dispersions', 'dispersions_norm', 'highly_variable_nbatches',
         'highly_variable_intersection'
    uns: 'annotation_colors', 'annotation_recluster_colors', 'batch_colors', 'hvg', 'leiden', 'leiden_colors', 'log1p',
         'neighbors', 'pca', 'umap'
    obsm: 'X_CCA', 'X_pca', 'X_umap'
    varm: 'PCs'
    layers: 'counts', 'logcounts'
    obsp: 'connectivities', 'distances'
    """
    import scanpy as sc
    import scanpy.external as sce

    if key_added in adata.obs.columns:
        logger.warn(f"{key_added} will be overwritten")

    celltype = [use_clusters] if isinstance(use_clusters, str) else use_clusters
    hvg_key = batch_key if hvg_batch else None

    adata_subset = adata[adata.obs[cluster_key].isin(celltype)]

    # If CCA was used, redo PCA of the subsetted integrated matrix
    if recluster_approach.lower() == "cca4":
        logger.info("Reclustering using CCA4 approach")
        assert use_rep is not None, "Specify obsm key with integrated matrix"
        try:
            adata_tmp = ad.AnnData(adata_subset.obsm[use_rep].values, obs=pd.DataFrame(index=adata_subset.obs_names))
        except AttributeError:
            adata_tmp = ad.AnnData(adata_subset.obsm[use_rep], obs=pd.DataFrame(index=adata_subset.obs_names))
        sc.pp.scale(adata_tmp)
        sc.pp.pca(adata_tmp, random_state=random_state)
        representation = "X_pca"
        adata_subset.obsm[representation] = adata_tmp.obsm[representation]
    elif recluster_approach.lower() == "cca5":
        logger.info("Reclustering using CCA5 approach")
        assert use_rep is not None, "Specify obsm key with integrated matrix"
        representation = use_rep
    elif recluster_approach.lower() == "scanorama":
        logger.info("Reclustering using Scanorama approach")
        assert use_rep is not None, "Specify obsm key with integrated matrix"
        representation = use_rep
    # If harmony was used, redo harmony
    elif recluster_approach.lower() == "harmony":
        logger.info("Reclustering using Harmony approach")
        adata_tmp = adata_subset.copy()
        sc.pp.highly_variable_genes(adata_tmp, batch_key=hvg_key)
        sc.pp.scale(adata_tmp)
        sc.pp.pca(adata_tmp, random_state=random_state)
        run_harmony(adata_tmp, batch_key=batch_key, max_iter_harmony=150, random_state=random_state)
        representation = "X_harmony"
        adata_subset.obsm[representation] = adata_tmp.obsm[representation]
    # If bbknn was used, redo PCA
    elif recluster_approach.lower() == "pca":
        logger.info("Reclustering using PCA approach")
        adata_tmp = adata_subset.copy()
        sc.pp.highly_variable_genes(adata_tmp, batch_key=hvg_key)
        sc.pp.scale(adata_tmp)
        sc.pp.pca(adata_tmp, random_state=random_state)
        representation = "X_pca"
        adata_subset.obsm[representation] = adata_tmp.obsm[representation]
    # If scvi was used, take the scvi latent space
    elif recluster_approach.lower() == "scvi":
        logger.info("Reclustering using scVI approach")
        assert use_rep is not None, "Specify obsm key with integrated matrix"
        representation = use_rep
    else:
        raise NotImplementedError(f"{recluster_approach} not implemented, use: cca4, cca5, scanorama, harmony, pca or scvi")

    # Calculate neighbors, UMAP and leiden
    if bbknn:
        sce.pp.bbknn(adata_subset, use_rep=representation, batch_key=batch_key, neighbors_within_batch=neighbors_batch,
                     pynndescent_random_state=random_state)
    else:
        sc.pp.neighbors(adata_subset, use_rep=representation, random_state=random_state)

    sc.tl.umap(adata_subset, random_state=random_state)
    sc.tl.leiden(adata_subset, resolution=resolution, flavor="igraph", n_iterations=2, directed=False,
                 random_state=random_state)
    adata.obs[key_added] = adata.obs[cluster_key].copy()

    if automatic_annot:
        adata.obs[key_added_autoannot] = adata.obs[cluster_key].copy()
        auto_annot(
            adata_subset,
            "leiden",
            key_added=key_added_autoannot,
            update_label=False,
            convert=convert,
            majority=majority,
            model=model,
        )

        transfer_labels(
            adata,
            adata_subset,
            col_original=key_added_autoannot,
            col_subset=key_added_autoannot,
            labels_original=celltype,
        )

    preffix = "+".join(celltype) if isinstance(celltype, list) else celltype
    adata_subset.obs[key_added] = preffix + "_" + adata_subset.obs["leiden"].astype(str)

    transfer_labels(adata, adata_subset, col_original=key_added, col_subset=key_added, labels_original=celltype)

    # Remove colors in uns to avoid problems when plotting
    keys_colors = [k for k in adata.uns.keys() if '_colors' in k]
    for key in keys_colors:
        del adata.uns[key]
    keys_colors = [k for k in adata_subset.uns.keys() if '_colors' in k]
    for key in keys_colors:
        del adata_subset.uns[key]

    if get_subset:
        return adata_subset
    else:
        return None


def full_recluster(
    adata: ad.AnnData,
    cluster_key: str,
    batch_key: str,
    recluster_approach: Literal["cca4", "cca5", "harmony", "scanorama", "pca", "scvi"],
    hvg_batch: bool = False,
    use_rep: str = None,
    bbknn: bool = False,
    resolution: float = 0.3,
    neighbors_batch: int = 3,
    majority: bool = True,
    convert: bool = True,
    key_added: str = "annotation_fullrecluster",
) -> None:
    """Re-clustering of all clusters in dataset.

    Perform reclustering on an integrated AnnData object over all clusters. Can recluster for the following
    integration methods: CCA (v4/v5) integration from Seurat; Harmony integration; BBKNN integration; SCVI integration,
    Scanorama integration and PCA. Assumes that `X` has logcounts.

    .. note::
        For CCA (v4/v5) and scVI the corrected expression matrix (CC4 v5), the CCA representation
        (CCA v5) and the latent space (scvi) to be in `.obsm`. When re-clustering with harmony and
        BBKNN the pipeline will be re-run over the clusters.

    :param adata: Annotated data matrix.
    :param cluster_key: Metadata column  in `obs` with cluster groups.
    :param batch_key: Metadata column in `obs` with  batch groups.
    :param hvg_batch:  If set to `True`. The  highly variable genes that are shared across samples will be used.
    :param recluster_approach: Reclustering approach to use.
    :param use_rep: Name in `obsm` with the representation. Required for SCVI, CCA and Scanorama approach.
    :param bbknn: Use BBKNN to compute neighbors.
    :param resolution: Resolution for the leiden clustering.
    :param neighbors_batch: To compute the nearest neighbors distance matrix and a neighborhood graph of observations a
                            `BBKNN <https://academic.oup.com/bioinformatics/article/36/3/964/5545955?login=true>`_ is
                            employed, which calculate a batch balanced KNN graph. It is recommended to use 3 with
                            when <100000 cells and 25 for >100000. If there are not enough cells per batch the default
                            approach will be used (`sc.pp.neighbors`).
    :param majority: Whether to refine the predicted labels by running the majority voting classifier after
                     over-clustering.
    :param convert: Convert the gene format of the model. If a Human model is provided, and is set to `True`, then gene
                    in mouse format will be use and viceverse.
    :param key_added: Metadata column name in `obs` to save the  reclustering information.
    :return: Returns `None`. The following fields will be set:

             `adata.obs['annotation_fullrecluster' | key_added]`: :class:`pandas.Series` (dtype ``category``)
                Array that stores the re-clusters groups consisting of the original group_id + the new cluster id
                (e.g., for a the monocyte cluster with 3 sub-clusters the new clusters are monocyte_0,
                monocyte_1, and monocyte_2).

    See Also
    --------
        :func:`dotools_py.tl.reclustering`: re-cluster specific clusters.


    Example
    -------
    >>> import dotools_py as do
    >>> adata = do.dt.example_10x_processed()
    >>> do.tl.full_recluster(
    ...     adata, cluster_key="annotation", batch_key="batch", recluster_approach="cca5", use_rep="X_CCA"
    ... )
    >>> adata
    AnnData object with n_obs × n_vars = 700 × 1851
        obs: 'batch', 'condition', 'n_genes_by_counts', 'log1p_n_genes_by_counts', 'total_counts',
             'log1p_total_counts', 'total_counts_mt', 'log1p_total_counts_mt', 'pct_counts_mt', 'total_counts_ribo',
             'log1p_total_counts_ribo', 'pct_counts_ribo', 'n_genes', 'n_counts', 'doublet_class', 'doublet_score',
             'leiden', 'cell_type', 'autoAnnot', 'celltypist_conf_score', 'annotation', 'annotation_fullrecluster'
        var: 'mean', 'std', 'highly_variable', 'means', 'dispersions', 'dispersions_norm', 'highly_variable_nbatches',
             'highly_variable_intersection'
        uns: 'annotation_colors', 'annotation_recluster_colors', 'batch_colors', 'hvg', 'leiden', 'leiden_colors',
             'log1p', 'neighbors', 'pca', 'umap'
        obsm: 'X_CCA', 'X_pca', 'X_umap'
        varm: 'PCs'
        layers: 'counts', 'logcounts'
        obsp: 'connectivities', 'distances'

    """
    celltype = list(adata.obs[cluster_key].unique())
    adata.obs[key_added] = adata.obs[cluster_key].copy()
    for ct in celltype:
        try:
            adata_subset = reclustering(
                adata,
                cluster_key=cluster_key,
                batch_key=batch_key,
                recluster_approach=recluster_approach,
                use_clusters=[ct],
                hvg_batch=hvg_batch,
                use_rep=use_rep,
                resolution=resolution,
                neighbors_batch=neighbors_batch,
                automatic_annot=False,
                bbknn=bbknn,
                majority=majority,
                convert=convert,
                get_subset=True,
                key_added="annotation_recluster",
            )
        except TypeError:
            logger.warn(f"Error while reclustering {ct}, keeping original annotation")
            adata_subset = adata[adata.obs[cluster_key] == ct]
            adata_subset.obs["annotation_recluster"] = ct

        transfer_labels(
            adata, adata_subset, col_original=key_added, col_subset="annotation_recluster", labels_original=[ct]
        )
    del adata.obs["annotation_recluster"]
    return None
