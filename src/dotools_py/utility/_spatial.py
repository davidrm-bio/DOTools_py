import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm

from dotools_py.utils import require_dependencies


def select_slide(adata: ad.AnnData, s: str, s_col: str = "sample") -> ad.AnnData:
    """Subset a Spatial AnnData object.

    This function selects the data for one slide from the spatial AnnData object. Useful when working with
    Visium data. The keys in `adata.uns['spatial']` should be the same as in s_col.

    :param adata: Anndata object with multiple spatial experiments.
    :param s: name of selected experiment.
    :param s_col: column in obs listing experiment name for each location.
    :return: returns `AnnData` after subsetting.
    """
    slid = adata[adata.obs[s_col].isin([s]), :].copy()
    s_keys = list(slid.uns["spatial"].keys())
    s_keys.remove(s)
    for val in s_keys:
        del slid.uns["spatial"][val]
    return slid


@require_dependencies([{"name": "liana"}])
def add_smooth_kernel(
    adata: ad.AnnData, layer_name: str = "smooth_X", bandwidth: int = 100, multiple: bool = True
) -> None:
    """Compute a smooth kernel, i.e, expression matrix is smooth.

    :param adata: AnnData object.
    :param layer_name: name of the layer with smooth expression matrix.
    :param bandwidth: radius (the greater, the more neighbors are considered).
    :param multiple: AnnData Object Contains Multiple Sample.
    :return: Returns `None`. A new layer will be added `adata.layers['smooth_X' | layer_name]`
    """
    import liana

    adata = adata.copy()

    if multiple:
        smooth_x = pd.DataFrame([])
        for sample in tqdm(adata.obs["sample"].unique(), desc="Analysed samples :"):
            slid = select_slide(adata, sample, "sample")
            liana.ut.spatial_neighbors(
                slid, bandwidth=bandwidth, cutoff=0.1, kernel="gaussian", set_diag=True, standardize=True
            )
            slid.X = slid.obsp["spatial_connectivities"].toarray().dot(slid.X.toarray())
            current_x = ad.AnnData.to_df(slid)
            smooth_x = pd.concat([smooth_x, current_x])
    else:
        liana.ut.spatial_neighbors(
            adata, bandwidth=bandwidth, cutoff=0.1, kernel="gaussian", set_diag=True, standardize=True
        )
        adata.X = adata.obsp["spatial_connectivities"].A.dot(adata.X.toarray())
        smooth_x = ad.AnnData.to_df(adata)

    smooth_x = smooth_x.reindex(index=adata.obs_names, columns=adata.var_names)
    adata.layers[layer_name] = sp.csr_matrix(smooth_x.values, dtype=np.float32)
    return None
