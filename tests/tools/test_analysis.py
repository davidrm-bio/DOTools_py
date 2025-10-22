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

    # BBKNN Integration
    keys = list(adata.obsm.keys())
    for key in keys:
        if key == "X_pca":
            continue
        del adata.obsm[key]
    do.tl.integrate_data(adata, batch_key="batch", integration_method="bbknn")
    assert "X_umap" in adata.obsm.keys()

    # scVI Integration
    do.tl.integrate_data(adata, batch_key="batch", integration_method="scvi")
    assert "X_scVI" in adata.obsm.keys()

    adata = adata[adata.obs["batch"].argsort()].copy()
    do.tl.integrate_data(adata, batch_key="batch", integration_method="scanorama")
    assert "X_scanorama" in adata.obsm.keys()

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




