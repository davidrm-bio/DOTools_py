import dotools_py as do




def test_find_doublets():
    adata = do.dt.example_10x_processed()

    cols = {"doublet_class", "doublet_score"}
    try:
        del adata.obs["doublet_class"], adata.obs["doublet_score"]
        do.pp.find_doublets(adata, batch_key="batch", method="scDblFinder")  # Only works if R is installed
        assert cols.issubset(adata.obs.columns)
    except Exception as e:
        pass

    del adata.obs["doublet_class"], adata.obs["doublet_score"]
    do.pp.find_doublets(adata, batch_key="batch", method="DoubletDetection")
    assert cols.issubset(adata.obs.columns)

    del adata.obs["doublet_class"], adata.obs["doublet_score"]
    do.pp.find_doublets(adata, batch_key="batch", method="Scrublet")
    assert cols.issubset(adata.obs.columns)

    del adata
    return
