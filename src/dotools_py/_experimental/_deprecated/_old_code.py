# def _run_scdblfinder(
#     adata: ad.AnnData,
#     batch_key: str | None = None,
# ) -> None:
#     """Find doublets.
#
#     The inference is performed using `scDblFinder <https://github.com/plger/scDblFinder>`_ in R.
#
#     :param adata: annotated anndata matrix
#     :param batch_key: `.obs` column name with batch information. Required if the anndata contain more than 1 sample.
#     :return:
#     """
#     import polars
#
#     logger.info("Finding Neotypic doublets")
#     rscript = get_paths_utils("_run_scDblFinder.R")
#     tmpdir_path = Path("/tmp") / f"scDblFinder_{uuid.uuid4().hex}"
#     tmpdir_path.mkdir(parents=True, exist_ok=False)
#     adata.write(tmpdir_path / "adata_tmp.h5ad")
#
#     logger.info("Running scDblFinder")
#     cmd = ["Rscript", rscript, "--input=" + str(tmpdir_path) + "/adata_tmp.h5ad", "--out=" + str(tmpdir_path) + "/"]
#     if batch_key:
#         cmd += ["--name=" + batch_key]
#     subprocess.call(cmd)
#
#     doublets = polars.read_csv(tmpdir_path / "scDblFinder_inference.csv", infer_schema_length=0)
#     doublets = doublets.to_pandas()
#     doublets = doublets.set_index(adata.obs_names)  # Avoid ImplicitModificationWarning
#     adata.obs[["doublet_class", "doublet_score"]] = doublets.values
#     shutil.rmtree(tmpdir_path)
#     return


# def sctransform_normalize(
#     adata: ad.AnnData,
#     batch_key: str = None,
#     layer: str = None
# ) -> None:
#     """Normalization based on `SCTransform <https://github.com/satijalab/sctransform>`_.
#
#     This function performs an alternative normalization based on the SCTransform.
#
#     :param adata: AnnData object with counts in `X`.
#     :param batch_key: obs metadata with batch information.
#     :param layer: layer to use.
#     :return: Returns None. The input AnnData object will have two new layers containing the SCT counts and normalize data.
#
#     Example
#     ------
#     >>> import dotools_py as do
#     >>> adata = do.dt.example_10x_processed()
#     >>> adata
#     AnnData object with n_obs × n_vars = 700 × 1851
#     obs: 'batch', 'condition', 'n_genes_by_counts', 'log1p_n_genes_by_counts', 'total_counts', 'log1p_total_counts',
#          'total_counts_mt', 'log1p_total_counts_mt', 'pct_counts_mt', 'total_counts_ribo', 'log1p_total_counts_ribo',
#          'pct_counts_ribo', 'n_genes', 'n_counts', 'doublet_class', 'doublet_score', 'leiden', 'cell_type',
#          'autoAnnot', 'celltypist_conf_score', 'annotation', 'annotation_recluster'
#     var: 'mean', 'std', 'highly_variable', 'means', 'dispersions', 'dispersions_norm', 'highly_variable_nbatches',
#          'highly_variable_intersection'
#     uns: 'annotation_colors', 'annotation_recluster_colors', 'batch_colors', 'hvg', 'leiden', 'leiden_colors', 'log1p',
#          'neighbors', 'pca', 'umap'
#     obsm: 'X_CCA', 'X_pca', 'X_umap'
#     varm: 'PCs'
#     layers: 'counts', 'logcounts'
#     obsp: 'connectivities', 'distances'
#     >>>
#     >>> do.pp.sctransform_normalize(adata, batch_key="batch", layer="counts")
#     >>> adata
#     AnnData object with n_obs × n_vars = 700 × 1181
#     obs: 'batch', 'condition', 'n_genes_by_counts', 'log1p_n_genes_by_counts', 'total_counts', 'log1p_total_counts',
#          'total_counts_mt', 'log1p_total_counts_mt', 'pct_counts_mt', 'total_counts_ribo', 'log1p_total_counts_ribo',
#          'pct_counts_ribo', 'n_genes', 'n_counts', 'doublet_class', 'doublet_score', 'leiden', 'cell_type',
#          'autoAnnot', 'celltypist_conf_score', 'annotation', 'annotation_recluster'
#     var: 'mean', 'std', 'highly_variable', 'means', 'dispersions', 'dispersions_norm', 'highly_variable_nbatches',
#          'highly_variable_intersection', 'SCT_rm'
#     obsm: 'SCT_rm'
#     varm: 'PCs'
#     layers: 'counts', 'logcounts', 'SCT_norm', 'SCT_counts'
#     obsp: 'connectivities', 'distances'
#     """
#     from scipy import sparse
#     import polars
#
#     rscript = get_paths_utils("_run_SCTransform.R")
#     tmpdir_path = Path("/tmp") / f"SCTransform_{uuid.uuid4().hex}"
#     tmpdir_path.mkdir(parents=True, exist_ok=False)
#
#     logger.info("Preparing to transfer to R")
#     adata_copy = adata.copy()
#     if layer is not None:
#         adata.X = adata.layers[layer].copy()
#     del adata.uns
#     del adata.obsm
#
#     if batch_key is not None:
#         adata_copy.obs["batch"] = adata_copy.obs[batch_key].copy()
#     else:
#         adata_copy.obs["batch"] = "batch1"
#     adata_copy.write(tmpdir_path / "adata_tmp.h5ad")
#
#     logger.info("Running SCTransform in R")
#     subprocess.call(["Rscript", rscript, "--input=" + str(tmpdir_path) + "/", "--out=" + str(tmpdir_path) + "/"])
#
#     raw_counts = polars.read_csv(os.path.join(tmpdir_path, "SCTransform_raw.csv"), infer_schema_length=0)
#     raw_counts = raw_counts.to_pandas().astype(float)
#     raw_counts = raw_counts.set_index(adata.obs_names)
#
#     norm_counts = polars.read_csv(os.path.join(tmpdir_path, "SCTransform_norm.csv"), infer_schema_length=0)
#     norm_counts = norm_counts.to_pandas().astype(float)
#     norm_counts = norm_counts.set_index(adata.obs_names)
#
#     # Transfer genes not kept during normalization to .obsm
#     excluded_genes = [gene for gene in adata.var_names if gene not in norm_counts.columns]
#     adata.var["SCT_rm"] = [True if gene in excluded_genes else False for gene in adata.var_names]
#     adata.obsm["SCT_rm"] = adata[:, adata.var["SCT_rm"].values].X.toarray()
#     adata = adata[:, ~adata.var["SCT_rm"].values]
#
#     # Make sure we have the same order or barcodes and features
#     norm_counts = norm_counts.reindex(index=adata.obs_names, columns=adata.var_names)
#     raw_counts = raw_counts.reindex(index=adata.obs_names, columns=adata.var_names)
#
#     adata.layers["SCT_norm"] = sparse.csr_matrix(norm_counts.values)
#     adata.layers["SCT_counts"] = sparse.csr_matrix(raw_counts.values)
#     return None


def test():
    import dotools_py as do
    from dotools_py.pp._importer import importer_py_new

    adata = importer_py_new(
        paths=[
            "/Users/david/Downloads/tmp_test/healthy/outs/filtered_feature_bc_matrix.h5",
            "/Users/david/Downloads/tmp_test/disease/outs/filtered_feature_bc_matrix.h5",
        ],
        ids=["batch1", "batch2"],
        metadata={"condition": ["healthy", "disease"]},
        batch_key="batch",
        min_genes_in_cell=300,
        min_cells_with_genes=5,
        cut_mt=5,
        n_reads=10_000,
        min_counts=100,
        max_counts=None,
        min_genes=None,
        max_genes=None,
        low_quantile=None,
        high_quantile=95,
        remove_doublets=True,
        doublet_tool="scDblFinder",
        normalisation_method="LogNormalisation",
        log_data=True,
        metrics_patterns=("mt-", ("rbs", "rpl")),
        metrics_names=["mt", "ribo"],
        random_state=0,
        technology="snrna",
    )
