import anndata as ad



def swap_layer(
    adata: ad.AnnData,
    layer_key: str,
    x_key: str = "X",
    inplace: bool = True,
)-> ad.AnnData | None:
    """Swap adata.X with adata.layers.

    Parameters
    ----------
    adata
        Annotated data matrix.
    layer_key
        Valid key in adata.layers
    x_key
        Key to use to save adata.X in adata.layers
    inplace
        Whether to generate a new object or make changes inplace.

    Returns
    -------
    Returns None or an AnnData object if inplace is set to `False`.

    """
    assert layer_key in adata.layers.keys(), f"{layer_key} not a valid key in adata.layers"

    if inplace:
        adata.layers[x_key] = adata.X.copy()
        adata.X = adata.layers[layer_key].copy()
        return None
    else:
        adata_copy = adata.copy()
        adata_copy.layers[x_key] = adata_copy.X.copy()
        adata_copy.X = adata_copy.layers[layer_key].copy()
        return adata_copy







