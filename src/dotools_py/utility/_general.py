import anndata as ad
from typing import Union


def free_memory():
    """Garbage collector.

    :return: None
    """
    import ctypes
    import gc

    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    return


def transfer_labels(
    adata_original: ad.AnnData,
    adata_subset: ad.AnnData,
    original_key: str,
    subset_key: str,
    original_labels: list,
    copy: bool = False
)-> Union[ad.AnnData, None]:
    """Transfer annotation from a subset of an AnnData.

    :param adata_original: original anndata.
    :param adata_subset: subsetted anndata.
    :param original_key: obs column name in the original anndata where new labels are added.
    :param subset_key: obs column name in the subsetted object with the new labels.
    :param original_labels: list of labels in the original anndata to replace.
    :param copy: if copy is True, returns the updated anndata, else changes are inplace
    :return: Nothing, changes are saved inplace
    """

    if copy:
        adata_original = adata_original.copy()
        adata_subset = adata_subset.copy()
    assert adata_subset.n_obs < adata_original.n_obs, 'adata_subset is not a subset of adata_original'

    labels_original = [original_labels] if isinstance(original_labels, str) else original_labels
    adata_original.obs[original_key] = adata_original.obs[original_key].astype(str)
    adata_original.obs[original_key] = adata_original.obs[original_key].where(
        ~adata_original.obs[original_key].isin(labels_original),
        adata_original.obs.index.map(adata_subset.obs[subset_key]))

    if copy:
        return adata_original
    else:
        return None
