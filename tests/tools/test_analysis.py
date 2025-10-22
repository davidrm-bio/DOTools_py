import os
import shutil
import anndata as ad

import matplotlib.pyplot as plt
import dotools_py as do


def test_integrate():
    adata = do.dt.example_10x_processed()

    # Harmony Integration
    do.tl.integrate_data(adata, batch_key="batch", integration_method="harmony")
    assert "X_harmony" in adata.obsm.keys()
    subset = do.tl.reclustering(adata,"annotation", "batch",  use_clusters=["NK"],
                                recluster_apporach="harmony", use_rep="X_harmony", get_subset=True)
    assert isinstance(subset, ad.AnnData)
    assert subset.n_obs < adata.n_obs

    # BBKNN Integration
    keys = list(adata.obsm.keys())
    for key in keys:
        if key == "X_pca":
            continue
        del adata.obsm[key]
    do.tl.integrate_data(adata, batch_key="batch", integration_method="pca")
    assert "X_umap" in adata.obsm.keys()
    subset = do.tl.reclustering(adata, "annotation", "batch", use_clusters=["NK"],
                                recluster_apporach="pca", get_subset=True)
    assert isinstance(subset, ad.AnnData)
    assert subset.n_obs < adata.n_obs

    # scVI Integration
    do.tl.integrate_data(adata, batch_key="batch", integration_method="scvi")
    assert "X_scVI" in adata.obsm.keys()
    subset = do.tl.reclustering(adata, "annotation", "batch", use_clusters=["NK"],
                                recluster_apporach="scvi", use_rep="X_scVI", get_subset=True)
    assert isinstance(subset, ad.AnnData)
    assert subset.n_obs < adata.n_obs

    adata = adata[adata.obs["batch"].argsort()].copy()
    do.tl.integrate_data(adata, batch_key="batch", integration_method="scanorama")
    assert "X_scanorama" in adata.obsm.keys()
    subset = do.tl.reclustering(adata, "annotation", "batch", use_clusters=["NK"],
                                recluster_apporach="scanorama", use_rep="X_scanorama", get_subset=True)
    assert isinstance(subset, ad.AnnData)
    assert subset.n_obs < adata.n_obs

    return None


def test_autoannot():
    adata = do.dt.example_10x_processed()

    os.makedirs("./tmp", exist_ok=True)

    del adata.obs["autoAnnot"]
    do.tl.auto_annot(adata, "leiden", convert=False, pl_cell_prob=True,
                     path="./tmp", filename="test.svg")
    plt.close()
    assert "autoAnnot" in adata.obs.columns
    files = os.listdir("./tmp")
    assert "test.svg" in files
    shutil.rmtree('./tmp')
    return None


def test_reclustering():
    adata = do.dt.example_10x_processed()

    counts = adata.obs.value_counts("annotation")
    adata_subset  = do.tl.reclustering(adata, "annotation", "batch", "cca5",
                                       use_rep="X_CCA", use_clusters=["B_cells"], get_subset=True)
    assert isinstance(adata_subset, ad.AnnData)
    assert adata_subset.n_obs == counts["B_cells"]
    return None


def test_full_recluster():
    adata = do.dt.example_10x_processed()

    do.tl.full_recluster(adata, "leiden", batch_key="batch",
                         recluster_apporach="cca5", use_rep="X_CCA", resolution=1)

    assert "annotation_fullrecluster" in adata.obs.columns
    assert len(adata.obs["annotation_fullrecluster"].unique()) > len(adata.obs["leiden"].unique())

    return None




