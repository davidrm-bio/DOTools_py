import dotools_py as do
import anndata as ad

def test_swap_layer():
    adata = do.dt.example_10x_processed()
    do.get.layer_swap(adata, "logcounts")
    assert "X" in adata.layers.keys()

    adata_new = do.get.layer_swap(adata, "logcounts", inplace=False)
    assert isinstance(adata_new, ad.AnnData)
    return
