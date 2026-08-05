import dotools_py as do
import os
import shutil
import anndata as ad


def test_dataframes():
    adata = do.dt.example_10x_processed()

    df = do.get.mean_expr(adata, group_by="condition", features="CD4")
    # Test Excel
    df.to_excel("./test.xlsx")
    df_new = do.io.read_excel(path="./", filename="test.xlsx")
    assert  all(df.columns == df_new.columns)
    os.remove("./test.xlsx")

    # Test CSV
    df.to_csv("./test.csv")
    df_new = do.io.read_csv(path="./", filename="test.csv")
    assert all(df.columns == df_new.columns)
    os.remove("./test.csv")

    # Test parquet
    df.to_parquet("./test.parquet")
    df_new = do.io.read_parquet(path="./", filename="test.parquet")
    assert all(df.columns == df_new.columns)
    os.remove("./test.parquet")


    # Use polars backend
    # Test Excel
    df.to_excel("./test.xlsx")
    df_new = do.io.read_excel(path="./", filename= "test.xlsx", backend="polars")
    assert all(df.columns == df_new.columns)
    os.remove("./test.xlsx")

    # Test CSV
    df.to_csv("./test.csv")
    df_new = do.io.read_csv(path="./", filename= "test.csv", backend="polars")
    assert all(df.columns == df_new.columns)
    os.remove("./test.csv")

    # Test parquet
    df.to_parquet("./test.parquet")
    df_new = do.io.read_parquet(path="./", filename= "test.parquet", backend="polars")
    assert all(df.columns == df_new.columns)
    os.remove("./test.parquet")


def test_objects():
    adata = do.dt.example_10x_processed()
    adata.write("./adata.h5ad")
    adata_new = do.io.read_h5ad("./", filename="adata.h5ad")
    os.remove("./adata.h5ad")

    adata.write_zarr("./adata.zarr")
    adata_new = do.io.read_zarr("./", filename="adata.zarr", backend="anndata")
    shutil.rmtree("./adata.zarr")

    os.makedirs("./tmp", exist_ok=True)
    do.dt.example_10x(path="./tmp")
    adata = do.io.read_10x_h5("./tmp/disease/outs/filtered_feature_bc_matrix.h5")
    assert isinstance(adata, ad.AnnData)
    shutil.rmtree("./tmp")


def test():
    #os.makedirs("./tmp", exist_ok=True)
    #do.dt.example_visium("./tmp")
    #adata = do.io.read_visium("./tmp")
    #assert isinstance(adata, ad.AnnData)
    #shutil.rmtree("./tmp")
    # TODO FAILS in GITHUB
    pass

