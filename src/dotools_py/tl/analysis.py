import os
import subprocess
import uuid
from pathlib import Path
from typing import Union

import anndata as ad
import bbknn as bkn
import celltypist
import numpy as np
import pandas as pd
import polars
import scanpy as sc
import scanpy.external as sce
from tqdm import tqdm

from .. import logger
from ..utils import get_paths_utils, transfer_labels
from ._data import DictUpdateCellLabels


def _run_cca(
    adata: ad.AnnData,
    batch_key: str,
    version: str = "v4",
) -> np.array:
    """Integrate AnnData using CCA from Seurat

    :param adata: anndata object
    :param batch_key: column in obs with batch IDs
    :param version:  version of Seurat to use
    :return: integrated matrix
    """
    rscript = get_paths_utils("_run_CCA.R")

    tmpdir_path = Path("/tmp") / f"CCA_{uuid.uuid4().hex}"
    tmpdir_path.mkdir(parents=True, exist_ok=False)

    logger.info("Preprocessing to export to Seurat")
    del adata.uns, adata.raw
    adata.write(tmpdir_path / "adata_hvg.h5ad")

    logger.info("Running CCA Integration")
    in_path = os.path.join(tmpdir_path, "adata_hvg.h5ad")
    subprocess.call(
        [
            "Rscript",
            rscript,
            "--input=" + in_path,
            "--out=" + str(tmpdir_path) + "/",
            "--name=" + batch_key,
            "--version=" + version,
        ]
    )

    logger.info("Loading corrected matrix")
    # TODO Consider v4 or v5 reading
    cca_matrix = polars.read_csv(
        os.path.join(tmpdir_path, "adata_hvg_seurat_AnchorIntegration.csv"), infer_schema_length=0
    )
    cca_matrix = cca_matrix.to_pandas().astype(float)
    cca_matrix = cca_matrix.set_index(cca_matrix.obs_names)
    return cca_matrix.values


def _run_scvi(
    adata: ad.AnnData,
    batch_key: str,
    layer_counts: str = "counts",
    layer_logcounts: str = "logcounts",
    categorical_covariates: list = None,
    continuos_covariates: list = None,
    n_hidden: int = 128,
    n_latent: int = 30,
    n_layers: int = 3,
    dispersion: str = "gene-batch",
    gene_likelihood: str = "zinb",
    get_model: bool = False,
    **kwargs,
) -> None:
    """Run scVI

    Run scVI to integrate sc/snRNA more information on `scvi-tools <https://docs.scvi-tools.org/en/stable/api/reference/scvi.model.SCVI.html>`_

    :param adata: anndata
    :param batch_key: .obs column with batch information
    :param layer_counts: layer with counts. raw counts required for integration
    :param layer_logcounts: layer with log-counts. log-counts required for calculation of HVG
    :param categorical_covariates: .obs column names with categorical covariates for scVI inference
    :param continuos_covariates: .obs column names with continuous covariates for scVI inference
    :param n_hidden: number of hidden layers
    :param n_latent: dimensions of the latent space
    :param n_layers: number of layers
    :param dispersion: dispersion mode for scVI
    :param gene_likelihood: gene likelihood
    :param get_model: return the trained model
    :param kwargs: additional arguments for scvi.model.SCVI
    :return: None or the model, the latent space is saved in the anndata under X_scVI
    """
    import scvi

    logger.info("Run scVI")
    assert layer_logcounts in adata.layers, "logcounts layer not in anndata"
    assert layer_counts in adata.layers, "counts layer not in anndata"
    assert "highly_variable" in list(adata.var.columns), "highly_variable not in adata.var"

    # Integration using only HVG
    hvg = adata[:, adata.var.highly_variable].copy()

    # Set-up anndta and model
    scvi.model.SCVI.setup_anndata(
        hvg,
        layer=layer_counts,
        batch_key=batch_key,
        continuous_covariate_keys=continuos_covariates,
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


def integrate_data(
    adata,
    batch_key: str,
    hvg_batch: bool = True,
    harmony: bool = False,
    scanorama: bool = False,
    bbknn: bool = False,
    cca4: bool = False,
    cca5: bool = False,
    scvi: bool = False,
    resolution: float = 0.3,
    **kwargs,
):
    """Integrate a concatenated object.

    Integrate and perform batch correction for a AnnData with several samples.

    :param adata: anndata object
    :param batch_key: column in obs with batch IDs
    :param hvg_batch: if set to True, highly variable genes shared across samples will be used for the integration
    :param harmony: integrate using harmony
    :param scanorama: integrate using scanorama
    :param bbknn: integrate using bbknn
    :param cca4: integrate using cca version 4
    :param cca5: integrate using cca version 5
    :param scvi: integrate using scvi
    :param resolution: resolution for the leiden clustering
    :param kwargs: extra arguments for scVI integration
    :return:
    """
    logger.info("Computing HVGs")
    hvg_batch = batch_key if hvg_batch else None
    sc.pp.highly_variable_genes(adata, batch_key=hvg_batch)
    hvg = adata[:, adata.var_highly_variable].copy()
    sc.pp.scale(hvg)
    sc.pp.pca(hvg)

    dim_reduc = "X_pca"
    neighbors_within_batch = 25 if adata.n_obs > 100_000 else 3  # Community recommendations
    if harmony:
        logger.info("Integration using Harmony")
        sce.pp.harmony_integrate(hvg, key=batch_key, max_iter_harmony=100)
        adata.obsm["X_harmony"] = hvg.obsm["X_pca_harmony"]
        dim_reduc = "X_harmony"
    if scanorama:
        logger.info("Integration using Scanorama")
        sce.pp.scanorama_integrate(hvg, key=batch_key)
        adata.obsm["X_scanorama"] = hvg.obsm["X_scanorama"]
        dim_reduc = "X_scanorama"
    if bbknn:
        logger.info("Integration using BBKNN")
        bkn.bbknn(adata, batch_key=batch_key, neighbors_within_batch=neighbors_within_batch)
        sc.tl.leiden(adata, resolution=0.3, flavor="igraph", directed=False, n_iterations=2)
        bkn.ridge_regression(adata, batch_key=batch_key, confounder_key="leiden")
        sc.tl.pca(adata)
    if scvi:
        logger.info("Integration using scVI")
        _run_scvi(adata, batch_key, **kwargs)
        dim_reduc = "X_scVI"
    if cca4:
        logger.info("Integration using CCA (Seurat v4 approach)")
        adata.obsm["X_CCA"] = _run_cca(hvg, batch_key, version="v4")
        logger.info("Using CCA matrix for PCA")
        hvg.X = adata.obsm["X_CCA"].copy()
        sc.pp.pca(hvg)
        adata.obsm["X_pca"] = hvg.obsm["X_pca"]
    if cca5:
        logger.info("Integration using CCA (Seurat v5 approach)")
        adata.obsm["X_CCA"] = _run_cca(hvg, batch_key, version="v5")
        dim_reduc = "X_CCA"

    logger.info("Finding neighbors")
    bkn.bbknn(adata, batch_key=batch_key, neighbors_within_batch=neighbors_within_batch, use_rep=dim_reduc)

    logger.info("Run UMAP")
    sc.tl.umap(adata)

    logger.info(f"Clustering cells using Leiden (resolution {resolution})")
    sc.tl.leiden(adata, resolution=resolution, flavor="igraph", n_iterations=2, directed=False)
    return


def update_cell_labels(adata: ad.AnnData,
                       cell_col: str,
                       key_added: str = 'annotation',
                       dict_data: Union[str, dict] = 'default'
                       ) -> ad.AnnData:
    """Rename cell type labels generated by celltypist.

    This function will rename the cell type labels returned by celltypist
    when using the Heart Model.

    :param adata: anndata object previously analysed by Celltypist
    :param cell_col: .obs column with cell type labels
    :param key_added: .obs column where new labels are saved
    :return: anndata object
    """
    if dict_data == 'default':
        dict_data = DictUpdateCellLabels

    adata.obs[key_added] = [dict_data[cell] if cell in dict_data else
                            list(adata.obs[cell_col])[idx] for idx, cell in enumerate(list(adata.obs[cell_col]))]
    return adata


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
    dict_labels: Union[dict, str] = 'default',
) -> ad.AnnData:
    """Automatic Annotation base on CellTypist Package

    This function takes an anndata object and automatically annotate the clusters employing a model available for celltypist.

    :param adata: anndata object
    :param cluster_key: .obs column with leiden / louvain clusters
    :param model: model to use for the prediction. Default Healthy Adult Heart (Human)
    :param key_added: .obs column name where to save the predicted cell types
    :param majority: majority voting for predictions (See CellTypist documentation). Default True
    :param convert: convert the gene format of the model. If a Human  model is used, then gene in mouse format
                    will be use and viceverse.
    :param update_label: add a .obs column with cell type labels updated to standard names. Default False
    :param key_updated: .obs column name where updated cell type labels are saved. To be used when update_labels is set
                        to True. Default False
    :param verbose: show information of the analysis steps
    :param update_models: update the downloaded models
    :param dict_labels: dictionary with the updated labels for the names in celltypist model
    :return: AnnData
    """
    if update_models:
        celltypist.models.download_models(force_update=True)
    if model not in list(celltypist.models.models_description()["model"]):
        raise Exception(
            f"The model {model} is not available. Please specify a valid model \n\n{celltypist.models.models_description()}"
        )

    adata = adata.copy()
    steps = ["Setting-up", "Predicting", "Saving predictions", "Updating labels"]
    total_steps = len(steps) if update_label else len(steps) - 1

    with tqdm(total=total_steps, desc="Progress", disable=not verbose, colour="tomato") as pbar:
        # Get model
        pbar.set_description(steps.pop(0))
        model_loaded = celltypist.models.Model.load(model=model)
        if convert:
            model_loaded.convert()
        adata.X = adata.X.toarray()  # Leads to high memory usage
        pbar.update(1)

        # Do the prediction
        pbar.set_description(steps.pop(0))
        predictions_cells = celltypist.annotate(
            adata, model=model_loaded, majority_voting=majority, over_clustering=cluster_key
        )
        pbar.update(1)

        # Save predictions
        # TODO consider when majority voting is not avaialble
        pbar.set_description(steps.pop(0))
        predictions_cells_adata = predictions_cells.to_adata()
        adata.obs["cell_type"] = predictions_cells_adata.obs.loc[adata.obs.index, "majority_voting"]
        adata.obs[key_added] = adata.obs["cell_type"]  # Transfer to original object
        pbar.update(1)

        # TODo add the option to plot the probability dotplot
        if update_label:
            # Update labels
            pbar.set_description(steps.pop(0))
            update_cell_labels(adata, key_added, key_updated, dict_data=dict_labels)
            pbar.update(1)

    return adata



def reclustering(adata: ad.AnnData,
                 annotation_col: str,
                 batch_key: str,
                 celltype: Union[str, list],
                 recluster_apporach: str,
                 hvg_batch: bool = False,
                 use_rep: str = None,
                 resolution: float = 0.3,
                 neighbors_batch: int = 3,
                 automatic_annot: bool = True,
                 transfer_recluster: bool = False,
                 majority: bool = True,
                 convert: bool = True,
                 model: str = 'Healthy_Adult_Heart.pkl',
                 ) -> ad.AnnData:
    """Do re-clustering

    Perform reclustering on an integrated anndata object. Can recluster for the following integration methods:
    - CCA integration from Seurat. Specify the .obsm key with the integrated matrix
    - Harmony integration. Will perform a re-integration using harmony
    - BBKNN integration. A fast integration where neighbors are calculated being batched aware
    - SCVI integration. Specify the .obsm key with the latent representation
    !! We assume that X has logcounts

    :param adata: anndata object
    :param annotation_col: .obs column name with clusters
    :param batch_key: .obs column name with batch information
    :param celltype: values in batch_key to subset and recluster for
    :param recluster_apporach:
    :param use_rep: .obsm key with the representation. Required for SCVI and CCA approach
    :param resolution: resolution for the leiden clustering
    :param neighbors_batch: number of neighbors per batch
    :param automatic_annot: perform automatic annotation with celltypist
    :param transfer_recluster: transfer annotation to the input anndata and returns the anndata object with a new column
                               called annotation_recluster. If False, the subseted object will be returned.
    :param majority: use majority voting for automatic annotation
    :param convert: convert the gene format. Useful if using a human model and input is mouse or viceverse
    :param model: model name of celltypist to use
    :return: input anndata with reclustering or subsetted anndata with reclusters
    """

    celltype = [celltype] if isinstance(celltype, str) else celltype
    adata_subset = adata[adata.obs[annotation_col].isin(celltype)]
    hvg_key = batch_key if hvg_batch else None

    # If CCA was used, redo PCA of the subsetted integrated matrix
    if recluster_apporach.lower() == 'cca4':
        assert use_rep is not None, 'Specify obsm key with integrated matrix'
        try:
            adata_tmp = ad.AnnData(adata_subset.obsm[use_rep].values, obs=pd.DataFrame(index=adata_subset.obs_names))
        except AttributeError:
            adata_tmp = ad.AnnData(adata_subset.obsm[use_rep], obs=pd.DataFrame(index=adata_subset.obs_names))
        sc.pp.scale(adata_tmp)
        sc.pp.pca(adata_tmp)
        representation = 'X_pca'
        adata_subset.obsm[representation] = adata_tmp.obsm[representation]
    elif recluster_apporach.lower() == 'cca5':
        assert use_rep is not None, 'Specify obsm key with integrated matrix'
        representation = use_rep
    # If harmony was used, redo harmony
    elif recluster_apporach.lower() == 'harmony':
        adata_tmp = adata_subset.copy()
        sc.pp.highly_variable_genes(adata_tmp, batch_key=hvg_key)
        sc.pp.scale(adata_tmp)
        sc.pp.pca(adata_tmp)
        sce.pp.harmony_integrate(adata_tmp, key=batch_key, max_iter_harmony=20)
        representation = 'X_pca_harmony'
        adata_subset.obsm[representation] = adata_tmp.obsm[representation]
    # If bbknn was used, redo PCA
    elif recluster_apporach.lower() == 'bbknn':
        adata_tmp = adata_subset.copy()
        sc.pp.highly_variable_genes(adata_tmp, batch_key=hvg_key)
        sc.pp.scale(adata_tmp)
        sc.pp.pca(adata_tmp)
        representation = 'X_pca'
        adata_subset.obsm[representation] = adata_tmp.obsm[representation]
    # If scvi was used, take the scvi latent space
    elif recluster_apporach.lower() == 'scvi':
        assert use_rep is not None, 'Specify obsm key with integrated matrix'
        representation = use_rep
    else:
        raise NotImplemented(f'{recluster_apporach} not implemented, use: CCA4, CCA5, harmony, bbknn or scvi')

    # Calculate neighbors, UMAP and leiden
    try:
        sce.pp.bbknn(adata_subset, use_rep=representation, batch_key=batch_key, neighbors_within_batch=neighbors_batch)
    except ValueError:  # If cluster is too small, bbknn might fail
        sc.pp.neighbors(adata_subset, use_rep=representation)
    sc.tl.umap(adata_subset)
    sc.tl.leiden(adata_subset, resolution=resolution, flavor='igraph', n_iterations=2, directed=False)

    if automatic_annot:
        try:
            adata_subset = auto_annot(adata_subset,
                                      'leiden',
                                      key_added='autoAnnot',
                                      key_updated='annotation_recluster',
                                      update_label=True,
                                      convert=convert,
                                      majority=majority,
                                      model=model)
        except KeyError:  # Might fail if getting a cell not present in our update_labels dictionary
            adata_subset = auto_annot(adata_subset,
                                      'leiden',
                                      key_added='annotation_recluster',
                                      update_label=False,
                                      convert=convert,
                                      majority=majority,
                                      model=model)
    if transfer_recluster:
        adata.obs['annotation_recluster'] = adata.obs[annotation_col].copy()
        transfer_labels(adata, adata_subset,
                        col_original='annotation_recluster',
                        col_subset='annotation_recluster',
                        labels_original=celltype)
        return adata
    else:
        return adata_subset
