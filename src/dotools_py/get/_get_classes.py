#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File:    _get_classes
Author:  david
Created: 28.07.26 11:44am

Module description here.
"""
import pandas as pd
import tqdm
from typing import Literal

from prelude_py import np, ad, sc
from scipy.sparse import csr_matrix

from dotools_py._utils import sanitize_anndata, iterase_input
from dotools_py._custom_class import InputError
from dotools_py.logger import logger
from dotools_py.pp import log_normalize
from dotools_py.settings._global_parameters import FAST_ARRAY_UTILS

if FAST_ARRAY_UTILS:
    import fast_array_utils


class GenerateMetaCells:
    def __init__(
        self,
        adata: ad.AnnData,
        batch_key: str,
        annotation_key: str,
        size: int = 10,
        min_cells: int = 50,
        layer: str = "counts",
        keep_obs: list | None = None,
        seed: int = 0,
        n_cpu: int = 8,
        agg_fx: Literal["count_nonzero", "mean", "sum", "var", "median"] = "sum",
    ):
        sanitize_anndata(adata)

        self.adata = adata
        self.batch_key = batch_key
        self.annotation_key = annotation_key
        self.size = size
        self.min_cells = min_cells
        self.layer = layer
        self.seed = seed
        self.workers = n_cpu
        self.agg_fx = agg_fx

        self.batches = adata.obs[batch_key].unique()
        self.cts = adata.obs[annotation_key].unique()
        self.keep_obs = (
            adata.obs.select_dtypes(include="category").columns.tolist() if keep_obs is None else iterase_input(keep_obs)
        )
        self.keep_obs.append("meta_cells")

        # Checks
        if size <= 0:
            raise InputError(f"Cannot create metacells by randomly aggregating {size} cells")
        if min_cells < 0:
            raise InputError(f"Cannot create metacells by using {min_cells} cells")

        self.dummy = None
        self.removed = None
        self._pdata = None


    @staticmethod
    def process_group(task):
        cells, min_cells, size, seed = task
        n_cells = len(cells)

        if n_cells < min_cells:
            return n_cells, None

        cells = cells.copy()

        if seed is None:
            np.random.shuffle(cells)
        else:
            rng = np.random.default_rng(seed + n_cells)
            rng.shuffle(cells)

        n_complete = (n_cells // size) * size
        if n_complete == 0:
            return 0, []

        cells = cells[:n_complete]
        metacells = cells.reshape(-1, size)
        return 0, metacells

    def sample(self):
        from concurrent.futures import ThreadPoolExecutor
        grouped = self.adata.obs.groupby([self.batch_key, self.annotation_key], observed=True, sort=False).groups

        tasks = []
        for batch in self.batches:
            for ct in self.cts:
                cells = grouped.get((batch, ct))
                if cells is None:
                    continue
                tasks.append((np.asarray(cells), self.min_cells, self.size, self.seed))

        dummy, removed, counter = {}, 0, 0
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            results = executor.map(self.process_group, tasks)
            for removed_count, metacells in tqdm.tqdm(results, total=len(tasks), desc="Creating metacells"):
                removed += removed_count
                if metacells is None:
                    continue
                for cells in metacells:
                    metacell_name = f"Metacell-{counter}"
                    dummy.update(dict.fromkeys(cells, metacell_name))
                    counter += 1

        self.dummy = dummy
        self.removed = removed
        return

    def aggregate(self):
        if self.dummy is None:
            self.sample()

        logger.warn(f"{self.removed} cells were not used to generate metacells")
        self.adata.obs["meta_cells"] = self.adata.obs_names.map(self.dummy)
        self.adata.obs["meta_cells"] = self.adata.obs["meta_cells"].fillna("excluded")

        pdata = sc.get.aggregate(self.adata, by="meta_cells", func=self.agg_fx, layer=self.layer)
        pdata = pdata[pdata.obs["meta_cells"] != "excluded"].copy()

        # Transfer metadata
        #meta = (
        #    self.adata.obs[self.keep_obs]
        #    .query("meta_cells != 'excluded'")
        #    .groupby("meta_cells", observed=True)
        #    .first()
        #    .reset_index()
        #)
        meta = (
            self.adata.obs[self.keep_obs]
            .drop_duplicates(subset="meta_cells")
        )

        #meta = self.adata.obs[self.keep_obs]
        #meta = meta.reset_index(drop=True).drop_duplicates()
        #meta = meta[meta["meta_cells"] != "excluded"]
        pdata.obs = pdata.obs.merge(meta, on="meta_cells")
        pdata.X = pdata.layers["sum"].copy()

        # Compute basic metrics
        sc.pp.calculate_qc_metrics(pdata, log1p=False, percent_top=None, inplace=True)
        log_normalize(pdata, target_sum=10_000)
        sc.pp.highly_variable_genes(pdata)

        pdata.obs_names = pdata.obs["meta_cells"].astype(str).values
        pdata.X = csr_matrix(pdata.X)
        pdata.layers["counts"] = csr_matrix(pdata.layers["counts"])
        pdata.layers["logcounts"] = csr_matrix(pdata.layers["logcounts"])
        del pdata.layers["sum"]

        del self.adata.obs["meta_cells"]
        self._pdata = pdata

    def get_pdata(self):
        if self._pdata is None:
            raise ValueError("Metadata has not been sample")
        else:
            return self._pdata




def _pseudobulk(
    adata: ad.AnnData,
    batch_key: str,
    cluster_key: str | None = None,
    seed: int = 0,
    layer: str | None = None,
    mode: Literal["sum", "mean", "median"] = "sum",
    technical_replicates: int | None = None,
):
    if technical_replicates is None or  technical_replicates < 1:
        technical_replicates = 1

    # Get the sparse matrix
    X = adata.X if layer is None else adata.layers[layer]
    obs = adata.obs.copy()

    group_cols = iterase_input(batch_key) + iterase_input(cluster_key)
    grouped = obs.groupby(group_cols, observed=True, sort=False)

    rng = np.random.default_rng(seed)

    group_names, metadata, profiles = [], [], []

    for group_key, indices in grouped.indices.items():
        if not isinstance(group_key, tuple):
            group_key = (group_key, )
        cell_idx = np.asarray(indices)

        if technical_replicates == 1:
            replicates_indices = [cell_idx]
        else:
            shuffled_idx = rng.permutation(cell_idx)
            replicates_indices = np.array_split(shuffled_idx, technical_replicates)
            if any(len(idx) == 0 for idx in replicates_indices):
                logger.warn(
                    f"Cannot generate {technical_replicates} technical replicates for group {group_key} containing only "
                    f"{len(cell_idx)} cells"
                )
                logger.warn(f"The groups {group_key} will be ignored")
                continue
        for technical_replicate, rep_cell_idx in enumerate(replicates_indices):
            technical_replicate += 1
            X_group = X[rep_cell_idx]

            if mode == "sum":
                profile = fast_array_utils.stats.sum(X_group, axis=0)
            elif mode == "mean":
                profile = fast_array_utils.stats.mean(X_group, axis=0)
            elif mode == "median":
                profile = np.median(X_group, axis=0)  # Not available in fast_array_utils
            else:
                raise ValueError(f"{mode} is not a valid pseudobulk mode. Use: `sum`, `mean` or `median`")

            profile = np.asarray(profile).ravel()
            profiles.append(profile)

            meta = {col:value for col, value in zip(group_cols, group_key)}
            if technical_replicates > 1:
                meta["technical_replicate"] = f"replicate_{technical_replicate}"

            cell_obs = obs.iloc[rep_cell_idx]
            for col in obs.columns:
                if col in group_cols:
                    continue
                values = cell_obs[col].dropna().unique()
                if len(values) == 0:
                    meta[col] = np.nan
                elif len(values) == 1:
                    meta[col] = values[0]
                else:
                    continue
            meta["psbulk_cells"] = len(rep_cell_idx)
            meta["psbulk_counts"] = fast_array_utils.stats.sum(profile, axis=None)
            metadata.append(meta)
            name = "_".join(map(str, group_key))
            if technical_replicates > 1:
                name += f"_rep{technical_replicate}"
            group_names.append(name)

    X_pb = np.vstack(profiles)
    obs_pb = pd.DataFrame(metadata, index=group_names).dropna(axis=1)
    pb = ad.AnnData(X=X_pb, obs=obs_pb, var=adata.var_names.to_frame(name="feature_name"))
    """
    for i, (group_key, indices) in enumerate(grouped.indices.items()):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        cell_idx = np.asarray(indices)
        X_group = X[cell_idx]

        if mode == "sum":
            profiles = fast_array_utils.stats.sum(X_group, axis=0)
        elif mode == "mean":
            profiles = fast_array_utils.stats.mean(X_group, axis=0)
        elif mode == "median":
            profiles = np.median(X_group, axis=0)  # Not available in fast_array_utils
        else:
            raise ValueError(f"{mode} is not a valid pseudobulk mode. Use: `sum`, `mean` or `median`")

        X_pb[i] = np.asarray(profiles).ravel()
        group_names.append("_".join(map(str, group_key)))
        cell_obs = obs.iloc[cell_idx]
        meta = {col: value for col, value in zip(group_cols, group_key)}

        # Add Metadata that is unique
        for col in obs.columns:
            if col in group_cols:
                continue
            values = cell_obs[col].dropna().unique()
            if len(values) == 0:
                meta[col] = np.nan
            elif len(values) == 1:
                meta[col] = values[0]
            else:
                continue

        # Add information on how many cells were agregated and total counts per metacell
        meta["psbulk_n_cells"] = len(cell_idx)
        meta["psbulk_counts"] = fast_array_utils.stats.sum(X_pb[i:i + 1], axis=None)
        metadata.append(meta)

    obs_pb = pd.DataFrame(metadata, index=group_names).dropna(axis=1)
    pb = ad.AnnData(X=X_pb, obs=obs_pb, var=adata.var_names.to_frame(name="feature_name"))
    """
    return pb
