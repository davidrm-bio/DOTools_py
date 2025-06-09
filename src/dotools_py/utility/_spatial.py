import anndata as ad
from tempfile import TemporaryDirectory
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
import numpy as np

try:
    import spatialdata as st
    SPATIALDATA_AVAILABLE = True
except ModuleNotFoundError:
    SPATIALDATA_AVAILABLE = False
    pass


from dotools_py.utils import convert_path, require_dependencies

def select_slide(adata: ad.AnnData,
                 s: str,
                 s_col: str = 'sample') -> ad.AnnData:
    """Subset a Spatial AnnData object.

    This function selects the data for one slide from the spatial anndata object. Useful when working with
    Visium data. The keys in `adata.uns['spatial']` should be the same as in s_col.

    :param adata: Anndata object with multiple spatial experiments.
    :param s: name of selected experiment.
    :param s_col: column in obs listing experiment name for each location.
    """
    slid = adata[adata.obs[s_col].isin([s]), :].copy()
    s_keys = list(slid.uns['spatial'].keys())
    s_keys.remove(s)
    for val in s_keys:
        del slid.uns['spatial'][val]
    return slid


def save_zarr(sdata: "st.SpatialData",
              path: str,
              filename: str) -> None:
    """Save changes from SpatialData object.

    :param sdata: SpatialData Object.
    :param path: path to the folder.
    :param filename: filename.
    :return:
    """
    if not SPATIALDATA_AVAILABLE:
        raise ImportError('spatialdata is not installed, this function is unavailable')

    path = convert_path(path)
    tmpdir = TemporaryDirectory()
    sdata.write(Path(tmpdir.name) / filename, overwrite=True)
    sdata = st.read_zarr(Path(tmpdir.name) / filename)
    sdata.write(path / filename, overwrite=True)
    return



@require_dependencies([{'name': 'liana'}])
def add_smooth_kernel(
    adata: ad.AnnData,
    layer_name: str = 'smooth_X',
    bandwidth: int = 100,
    multiple: bool = True
) -> ad.AnnData:
    """Compute a smooth kernel, i.e, expression matrix is smooth.

    :param adata: AnnData object.
    :param layer_name: name of the layer with smooth expression matrix.
    :param bandwidth: radius (the greater, the more neighbors are considered).
    :param multiple: AnnData Object Contains Multiple Sample.
    :return: anndata object with new layer.
    """
    import liana

    adata = adata.copy()

    if multiple:
        smooth_x = pd.DataFrame([])
        for sample in tqdm(adata.obs['sample'].unique(), desc='Analysed samples :'):
            slid = select_slide(adata, sample, 'sample')
            liana.ut.spatial_neighbors(slid,
                                       bandwidth=bandwidth, cutoff=0.1,
                                       kernel='gaussian', set_diag=True,
                                       standardize=True)
            slid.X = slid.obsp['spatial_connectivities'].toarray().dot(slid.X.toarray())
            current_x = ad.AnnData.to_df(slid)
            smooth_x = pd.concat([smooth_x, current_x])
    else:
        liana.ut.spatial_neighbors(adata,
                                   bandwidth=bandwidth, cutoff=0.1,
                                   kernel='gaussian', set_diag=True,
                                   standardize=True)
        adata.X = adata.obsp['spatial_connectivities'].A.dot(adata.X.toarray())
        smooth_x = ad.AnnData.to_df(adata)

    smooth_x = smooth_x.reindex(index=adata.obs_names, columns=adata.var_names)
    adata.layers[layer_name] = sp.csr_matrix(smooth_x.values, dtype=np.float32)
    return adata
