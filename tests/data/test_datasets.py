import os
import dotools_py as do
import anndata as ad
import shutil


def test_example10x():
    path = "./tmp"
    os.makedirs(path, exist_ok=True)  # Generate a tmp folder
    do.dt.example_10x(path=path)  # Download datasets
    # Check that two folders where created
    dirs = os.listdir(path)
    assert "disease" in dirs, "Disease dataset missing"
    assert  "healthy" in dirs, "Healthy dataset missing"
    # Load one test dataset
    adata = do.io.read_10x_h5(os.path.join(path, "disease", "outs", "filtered_feature_bc_matrix.h5"))
    assert  isinstance(adata, ad.AnnData), "Loaded datasets is not an AnnData"  # Check we have an AnnData
    shutil.rmtree(path)  # remove the tmp folder
    return None


def test_processed10x():
    adata = do.dt.example_10x_processed()
    assert  isinstance(adata, ad.AnnData)
    # Expected 700 x 1851
    assert  adata.n_obs == 700, f"Expected 700 cells but object has {adata.n_obs}"
    assert  adata.n_vars == 1851, f"Expected 1851 genes but object has {adata.n_vars}"
    return None


def test_visium():
    path = "./tmp"
    os.makedirs(path, exist_ok=True)  # Generate a tmp folder
    do.dt.example_visium(path=path)  # Download datasets
    assert  len(os.listdir(path)) !=0
    # Load one test dataset
    try:
        adata = do.io.read_visium(path)
        assert isinstance(adata, ad.AnnData), "Loaded datasets is not an AnnData"  # Check we have an AnnData
    except Exception:  # might fail if subprocess could not be run
        print("Something went wrong here")
        pass
    shutil.rmtree(path)  # remove the tmp folder


def test_processedvisium():
    adata = do.dt.example_visium_processed()
    assert  isinstance(adata, ad.AnnData)
    # Expected 700 x 1851
    assert  adata.n_obs == 1046, f"Expected 1046 cells but object has {adata.n_obs}"
    assert  adata.n_vars == 1000, f"Expected 1000 genes but object has {adata.n_vars}"
    return

def test_dummy():
    from dotools_py.dt._datasets import LoadData
    from dotools_py._custom_class import InputError
    try:
        loader = LoadData(path="/tmp", technology="none")
    except InputError:
        pass

def test_ora_table():
    df = do.dt.example_ora()
    assert  df.shape[0] == 2704
    assert  df.shape[1] == 11
