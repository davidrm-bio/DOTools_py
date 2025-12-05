import dotools_py as do



def test_find_doublets():
    def remove_cols(adata, col):
        if col in list(adata.obs.columns):
            del adata.obs[col]

    adata = do.dt.example_10x_processed()

    cols = {"doublet_class", "doublet_score"}

    try:
        remove_cols(adata, "doublet_class")
        remove_cols(adata, "doublet_score")
        do.pp.find_doublets(adata, batch_key="batch", method="scDblFinder")  # Only works if R is installed
        assert cols.issubset(adata.obs.columns)
    except Exception as e:
        pass

    remove_cols(adata, "doublet_class")
    remove_cols(adata, "doublet_score")
    do.pp.find_doublets(adata, batch_key="batch", method="DoubletDetection")
    assert cols.issubset(adata.obs.columns)

    remove_cols(adata, "doublet_class")
    remove_cols(adata, "doublet_score")
    do.pp.find_doublets(adata, batch_key="batch", method="Scrublet")
    assert cols.issubset(adata.obs.columns)

    del adata
    return
