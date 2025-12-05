import dotools_py as do
import os
import shutil


def test_dataframes():
    adata = do.dt.example_10x_processed()

    df = do.get.mean_expr(adata, "condition", "CD4")
    # Test Excel
    df.to_excel("./test.xlsx")
    df_new = do.io.read_excel("./","test.xlsx")
    assert  all(df.columns == df_new.columns)
    os.remove("./test.xlsx")

    # Test CSV
    df.to_csv("./test.csv")
    df_new = do.io.read_csv("./", "test.csv")
    assert all(df.columns == df_new.columns)
    os.remove("./test.csv")

    # Test parquet
    df.to_parquet("./test.parquet")
    df_new = do.io.read_parquet("./", "test.parquet")
    assert all(df.columns == df_new.columns)
    os.remove("./test.parquet")


    # Use polars backend
    # Test Excel
    df.to_excel("./test.xlsx")
    df_new = do.io.read_excel("./", "test.xlsx", backend="polars")
    assert all(df.columns == df_new.columns)
    os.remove("./test.xlsx")

    # Test CSV
    df.to_csv("./test.csv")
    df_new = do.io.read_csv("./", "test.csv", backend="polars")
    assert all(df.columns == df_new.columns)
    os.remove("./test.csv")

    # Test parquet
    df.to_parquet("./test.parquet")
    df_new = do.io.read_parquet("./", "test.parquet", backend="polars")
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

