#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File:    _get_classes
Author:  david
Created: 28.07.26 11:44am

Module description here.
"""
import tqdm
from typing import Literal

from prelude_py import np, ad, sc
from scipy.sparse import csr_matrix

from dotools_py._utils import sanitize_anndata, iterase_input
from dotools_py._custom_class import InputError
from dotools_py.logger import logger
from dotools_py.pp import log_normalize



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
