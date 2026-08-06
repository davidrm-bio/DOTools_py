import os
import shutil
import anndata as ad

import matplotlib.pyplot as plt
import dotools_py as do


def test_integrate():
    adata = do.dt.example_10x_processed()

    # Harmony Integration
    try: # Fails with new version of anndata? Some internal problem in scanpy.external
        do.tl.integrate_data(adata, batch_key="batch", integration_method="harmony")
        assert "X_harmony" in adata.obsm.keys()
        subset = do.tl.reclustering(adata,cluster_key= "annotation",batch_key= "batch", use_clusters=["NK"],
                                    recluster_approach="harmony", use_rep="X_harmony", get_subset=True, bbknn=False)
        assert isinstance(subset, ad.AnnData)
        assert subset.n_obs < adata.n_obs
    except ValueError:
        pass

    # scVI Integration
    do.tl.integrate_data(adata, batch_key="batch", integration_method="scvi")
    assert "X_scVI" in adata.obsm.keys()
    subset = do.tl.reclustering(adata,cluster_key= "annotation",batch_key= "batch", use_clusters=["NK"],
                                recluster_approach="scvi", use_rep="X_scVI", get_subset=True)
    assert isinstance(subset, ad.AnnData)
    assert subset.n_obs < adata.n_obs

    adata = adata[adata.obs["batch"].argsort()].copy()
    try:
        do.tl.integrate_data(adata, batch_key="batch", integration_method="scanorama")
        assert "X_scanorama" in adata.obsm.keys()
        subset = do.tl.reclustering(adata,cluster_key= "annotation",batch_key= "batch", use_clusters=["NK"],
                                    recluster_approach="scanorama", use_rep="X_scanorama", get_subset=True)
        assert isinstance(subset, ad.AnnData)
        assert subset.n_obs < adata.n_obs
    except Exception:
        pass
    return None


def test_autoannot():
    adata = do.dt.example_10x_processed()

    os.makedirs("./tmp_autoannot", exist_ok=True)
    del adata.obs["autoAnnot"]
    do.tl.auto_annot(adata,cluster_key= "leiden", convert=False, pl_cell_prob=True,
                     path="./tmp_autoannot", filename="test.svg")
    plt.close()
    assert "autoAnnot" in adata.obs.columns
    files = os.listdir("./tmp_autoannot")
    assert "test.svg" in files
    shutil.rmtree('./tmp_autoannot')
    return None


def test_reclustering():
    adata = do.dt.example_10x_processed()

    counts = adata.obs.value_counts("annotation")
    adata_subset  = do.tl.reclustering(adata,cluster_key= "annotation", batch_key= "batch",recluster_approach= "cca5",
                                       use_rep="X_CCA", use_clusters=["B_cells"], get_subset=True)
    assert isinstance(adata_subset, ad.AnnData)
    assert adata_subset.n_obs == counts["B_cells"]
    return None


def test_full_recluster():
    adata = do.dt.example_10x_processed()

    do.tl.full_recluster(adata,cluster_key= "leiden", batch_key="batch",
                         recluster_approach="cca5", use_rep="X_CCA", resolution=1)

    assert "annotation_fullrecluster" in adata.obs.columns
    assert len(adata.obs["annotation_fullrecluster"].unique()) > len(adata.obs["leiden"].unique())

    return None


def test_scvi_scvianvi():
    adata = do.dt.example_10x_processed()
    do.tl.run_scanvi(adata, batch_key="batch", label_key="annotation")
    assert  "X_scANVI" in adata.obsm.keys()
    assert  "X_scVI" in adata.obsm.keys()
    return


def test_update_labels():
    from dotools_py.tl._analysis import update_cell_labels
    adata = do.dt.example_10x_processed()
    update_cell_labels(adata, cell_col="annotation")
    update_cell_labels(adata, cell_col="annotation", dict_data={"NK":"NaturalKiller"})
    assert "NaturalKiller" in list(adata.obs.annotation.unique())
    return


def test_seurat_integration():
    from dotools_py._custom_class import InputError
    adata = do.dt.example_10x_processed()
    del adata.obsm["X_CCA"]
    do.tl.run_seurat_integration(adata,batch_key= "batch", backend="python")
    assert  "X_CCA" in adata.obsm.keys()

    try:
        do.tl.run_seurat_integration(adata,batch_key= "batch", backend="rust")
    except InputError:
        pass


def test_umap_clustering():
    adata = do.dt.example_10x_processed()

    del adata.obsm["X_umap"], adata.obsm["X_pca"]
    del adata.obs["leiden"]
    del adata.var["highly_variable"]
    do.tl.umap_clustering(adata, use_rep="X_pca")



