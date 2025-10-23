import os
import  dotools_py as do
import anndata as ad


def test_example10x():
    import shutil
    import scanpy as sc

    path = "./tmp"
    os.makedirs(path, exist_ok=True)  # Generate a tmp folder

    do.dt.example_10x(path)  # Download datasets

    # Check that two folders where created
    dirs = os.listdir(path)
    assert "disease" in dirs, "Disease dataset missing"
    assert  "healthy" in dirs, "Healthy dataset missing"

    # Load one test dataset
    adata = sc.read_10x_h5(os.path.join(path, "disease", "outs", "filtered_feature_bc_matrix.h5"))

    assert  isinstance(adata, ad.AnnData), "Loaded datasets is not an AnnData"  # Check we have an AnnData
    shutil.rmtree(path)  # remove the tmp folder
    return None


def test_processed10x():

    adata = do.dt.example_10x_processed()

    assert  isinstance(adata, ad.AnnData)

    # Expected 700 x 1851
    assert  adata.n_obs == 700, f"Expected 700 cells but object has {adata.n_obs}"
    assert  adata.n_vars == 1851, f"Expected 1851 genes but object has {adata.n_vars}"
    return


def test_heart_markers():
    for species in ["mouse", "human"]:
        markers = do.dt.heart_markers(species)
        cts = ['Art_EC', 'CapEC', 'VeinEC', 'LymphEC', 'EndoEC',
               'SMC', 'PC', 'FB', 'FBa', 'Neurons', 'CM', 'B_cells',
               'T_cells', 'Myeloid', 'MP_recruit', 'MP_resident',
               'ImmuneCells', 'Epicardial', 'Adip', 'Mast']

        assert  isinstance(markers, dict), "Markers are not a dictionary"
        for ct in cts:
            assert ct in markers.keys(), f"{ct} not in the marker list"

    try:
        do.dt.heart_markers("unknown")
    except Exception:
        pass
    return


def test_standard_labels():
    labels = do.dt.standard_ct_labels_heart()
    assert isinstance(labels, dict), "Labels to updated is not a dictionary"
    return
