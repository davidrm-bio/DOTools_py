# from tqdm import tqdm
# from typing import Literal
#
# import anndata as ad
# import pandas as pd
# import numpy as np
# from scipy.stats import false_discovery_control
#
# from dotools_py.utility import free_memory
# from dotools_py import logger
# from joblib import Parallel, delayed
#
# def neigh_perm(adata: ad.AnnData,
#              ref: str,
#              connectivity_key: str = "spatial_connectivities",
#              annotation_key: str  = "annotation",
#              condition_key: str = "condition",
#              statistic: Literal["mean", "median", "sum"] = "sum",
#              n_perms: int = 100,
#              alternative: Literal["two-sided", "greater", "less"] = "two-sided"):
#     """Test for differential number of neighbors between cell-types in spatial transcriptomics.
#
#     Calculate differential number of connections between cell-types comparing across
#     two conditions in spatial transcriptomics. The significance is tested using permutation test.
#     When using connectivities calculated using  delaunay triangularity is recommended to use
#     "sum" as statistic.
#     .. note::
#         Connectivities should be calculated per section. Use `squidpy.gr.spatial_neighbors` setting
#         library_key to the batch_key.
#
#     :param adata: Annotated data matrix.
#     :param ref: Reference condition.
#     :param connectivity_key: Key in `obsp` with the connectivities.
#     :param annotation_key:  Column in `obs` with cell annotation.
#     :param condition_key: Column in `obs` with conditions.
#     :param statistic: statistic to use for summarise the connectivities.
#     :param n_perms: Number of permutations to do.
#     :param alternative: Method to use for permutation test.
#     :return: Returns a pandas dataframe with the results from the permutation test.
#     """
#
#     group_by = [condition_key, annotation_key]
#
#     labels = []
#     for idx, row in adata.obs[group_by].iterrows():
#         current_row = "-".join(row.values)
#         labels.append(current_row)
#
#     # Define the groups
#     connections = adata.obsp[connectivity_key]
#     connections_coo = connections.tocoo()
#     row_indices = connections_coo.row.tolist()
#     col_indices = connections_coo.col.tolist()
#     values = connections_coo.data.tolist()
#
#     row_labels = [labels[idx] for idx in row_indices]
#     col_labels = [labels[idx] for idx in col_indices]
#
#     connections = pd.DataFrame([row_labels, col_labels, values]).T
#     connections.columns = ["rows", "columns", "connections"]
#
#     # Extract conditions and clusters from labels like "condition-cluster"
#     connections["row_condition"] = connections["rows"].str.split("-").str[0]
#     connections["row_cluster"] = connections["rows"].str.split("-").str[1]
#     connections["col_condition"] = connections["columns"].str.split("-").str[0]
#     connections["col_cluster"] = connections["columns"].str.split("-").str[1]
#
#     del connections_coo, row_indices, row_labels, col_indices, col_labels, values
#     free_memory()
#
#     condition_groups = list(adata.obs[condition_key].unique())
#     condition_groups.remove(ref)
#     annotation_groups = list(adata.obs[annotation_key].unique())
#
#     def calculate_stat(g1, g2, statistic):
#         if statistic == 'mean':
#             observed_stat = np.mean(g1) - np.mean(g2)
#         elif statistic == 'median':
#             observed_stat = np.median(g1) - np.median(g2)
#         elif statistic == "sum":
#             observed_stat = np.sum(g1) - np.sum(g2)
#         else:
#             raise ValueError("Not a valid statistic method use 'mean' or 'median'")
#         return  observed_stat
#
#     def single_permutation_stat(connections, ref, group, ct1, ct2, statistic):
#         shuffled_conditions = np.random.permutation(connections["row_condition"].values)
#
#         # Rebuild shuffled labels using NumPy string ops
#         shuffled_rows = shuffled_conditions + "-" + connections["row_cluster"].values
#         shuffled_cols = shuffled_conditions + "-" + connections["col_cluster"].values
#
#         # Apply shuffled labels to the DataFrame
#         perm_group1 = connections[
#             (shuffled_rows == f"{ref}-{ct1}") &
#             (shuffled_cols == f"{ref}-{ct2}")
#             ]["connections"].values
#
#         perm_group2 = connections[
#             (shuffled_rows == f"{group}-{ct1}") &
#             (shuffled_cols == f"{group}-{ct2}")
#             ]["connections"].values
#
#         if len(perm_group1) == 0 or len(perm_group2) == 0:
#             return None
#
#         return calculate_stat(perm_group1, perm_group2, statistic)
#
#     pvalues = []
#     for group in condition_groups:
#         logger.info(f"Performing permutation test for {ref} vs {group}")
#         for ct1 in tqdm(annotation_groups):
#             for ct2 in annotation_groups:
#
#                 group1 = np.array(connections[(connections["rows"] == ref + "-" + ct1) &
#                                               (connections["columns"] == ref + "-" + ct2)
#                                               ]["connections"])
#                 group2 = np.array(connections[(connections["rows"] == group + "-" + ct1) &
#                                               (connections["columns"] == group + "-" + ct2)
#                                               ]["connections"])
#
#                 if statistic == "sum":
#                     log2fc = np.log2((group1.sum()) / (group2.sum()))
#                 else:
#                     log2fc = np.log2((group1.mean() + 1e-9) / (group2.mean() + 1e-9))
#
#                 observed_stat = calculate_stat(group1, group2, statistic)
#
#                 permuted_stats = Parallel(n_jobs=-1, backend="loky")(
#                     delayed(single_permutation_stat)(
#                         connections, ref, group, ct1, ct2, statistic
#                     )
#                     for _ in tqdm(range(n_perms))
#                 )
#                 permuted_stats = np.array(permuted_stats)
#
#                 # Compute p-value based on alternative hypothesis
#                 if alternative == 'two-sided':
#                     p_value = np.mean(np.abs(permuted_stats) >= np.abs(observed_stat))
#                 elif alternative == 'greater':
#                     p_value = np.mean(permuted_stats >= observed_stat)
#                 elif alternative == 'less':
#                     p_value = np.mean(permuted_stats <= observed_stat)
#                 else:
#                     raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
#
#                 pvalues.append((group, ct1, ct2, log2fc, group1.mean(), group2.mean(), p_value))
#
#     df_pvals = pd.DataFrame(pvalues, columns=["group", "ref_ct", "alt_ct", "log2fc", "mean_ref", "mean_group",  "pval"])
#     df_pvals["padj"] = false_discovery_control(df_pvals["pval"])
#     return  df_pvals
#
#
# import squidpy as sq
# import scanpy as sc
# adata = sc.read_h5ad("/Users/david/Desktop/AMId3_slide3.h5ad")
# sq.gr.spatial_neighbors(adata, library_key="batch", delaunay=True, coord_type="generic")
#
# df = neigh_perm(adata, "3m", statistic="sum")
#
#
#
#
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from joblib import Parallel, delayed
# from tqdm import tqdm
#
# def neigh_perm_fast(adata: ad.AnnData,
#              ref: str,
#              connectivity_key: str = "spatial_connectivities",
#              annotation_key: str  = "annotation",
#              condition_key: str = "condition",
#              statistic: str = "sum",
#              n_perms: int = 1000,
#              alternative: str = "two-sided"):
#
#     # Prepare labels for each cell (row in adata.obs)
#     conditions = adata.obs[condition_key].values
#     annotations = adata.obs[annotation_key].values
#
#     # Encode condition and annotation as integers for speed
#     le_condition = LabelEncoder()
#     cond_codes = le_condition.fit_transform(conditions)
#     le_annotation = LabelEncoder()
#     ann_codes = le_annotation.fit_transform(annotations)
#
#     ref_code = le_condition.transform([ref])[0]
#     condition_groups = [c for c in le_condition.classes_ if c != ref]
#     condition_group_codes = [le_condition.transform([c])[0] for c in condition_groups]
#
#     # Get connectivity matrix as COO
#     conn = adata.obsp[connectivity_key].tocoo()
#     row_idx = conn.row
#     col_idx = conn.col
#     values = conn.data
#
#     # Construct arrays of edge attributes: condition and annotation for rows and cols
#     row_cond = cond_codes[row_idx]
#     col_cond = cond_codes[col_idx]
#     row_ann = ann_codes[row_idx]
#     col_ann = ann_codes[col_idx]
#
#     # All unique annotation groups
#     ann_groups = np.arange(len(le_annotation.classes_))
#
#     # Precompute index masks for each (condition, annotation) pair on rows and columns
#     # We'll need these for fast filtering later on
#     # But better yet: create masks for edges connecting pairs of (cond, ann) to (cond, ann)
#
#     # Create a helper function for masks of edges connecting (cond1, ann1) to (cond2, ann2)
#     def edge_mask(cond1, ann1, cond2, ann2):
#         return (row_cond == cond1) & (row_ann == ann1) & (col_cond == cond2) & (col_ann == ann2)
#
#     # Precompute the baseline statistics for all pairs (ref vs group) and (ann1, ann2)
#     results = []
#
#     # Calculate observed stats first
#     def calc_stat(g1, g2):
#         if statistic == "mean":
#             return np.mean(g1) - np.mean(g2)
#         elif statistic == "median":
#             return np.median(g1) - np.median(g2)
#         elif statistic == "sum":
#             return np.sum(g1) - np.sum(g2)
#         else:
#             raise ValueError(f"Statistic {statistic} not supported")
#
#     # Encode all condition labels as array so we can shuffle permutations on condition only
#     original_conditions = cond_codes.copy()
#
#     # We will do all permutations in one batch later
#     # But first, for each group, each pair (ct1, ct2), compute observed stat and store masks
#
#     for group_code in condition_group_codes:
#         print(f"Processing {le_condition.inverse_transform([group_code])[0]} vs {ref}")
#         for ct1 in ann_groups:
#             for ct2 in ann_groups:
#                 mask_ref = edge_mask(ref_code, ct1, ref_code, ct2)
#                 mask_group = edge_mask(group_code, ct1, group_code, ct2)
#
#                 vals_ref = values[mask_ref]
#                 vals_group = values[mask_group]
#
#                 if len(vals_ref) == 0 or len(vals_group) == 0:
#                     # No edges for this pair - skip or set nan
#                     continue
#
#                 observed_stat = calc_stat(vals_ref, vals_group)
#
#                 # Calculate log2fc with pseudocount to avoid zero division
#                 sum_ref = vals_ref.sum()
#                 sum_group = vals_group.sum()
#                 if statistic == "sum":
#                     log2fc = np.log2((sum_group + 1e-9) / (sum_ref + 1e-9))
#                 else:
#                     mean_ref = vals_ref.mean()
#                     mean_group = vals_group.mean()
#                     log2fc = np.log2((mean_group + 1e-9) / (mean_ref + 1e-9))
#
#                 # Store info for permutation test
#                 results.append({
#                     "group_code": group_code,
#                     "ct1": ct1,
#                     "ct2": ct2,
#                     "observed_stat": observed_stat,
#                     "vals_ref": vals_ref,
#                     "vals_group": vals_group,
#                     "log2fc": log2fc,
#                     "sum_ref": sum_ref,
#                     "sum_group": sum_group,
#                 })
#
#     # Now, perform permutations
#     # We'll generate a matrix of shape (n_perms, n_cells) with permuted condition codes
#     n_cells = adata.n_obs
#     rng = np.random.default_rng()
#
#     # Generate all permutations at once
#     permutations = np.array([rng.permutation(original_conditions) for _ in range(n_perms)])
#
#     # For each permutation, we assign permuted cond codes to each edge's row and col
#
#     # Since edges connect cells, get the indices of the cells connected for rows and cols
#     # We'll do a quick lookup of permuted condition for each edge's row and col cell:
#     # permuted_row_cond = permutations[:, row_idx]
#     # permuted_col_cond = permutations[:, col_idx]
#     # Shapes: (n_perms, n_edges)
#
#     permuted_row_cond = permutations[:, row_idx]  # shape: (n_perms, n_edges)
#     permuted_col_cond = permutations[:, col_idx]
#
#     # The annotation does not change during permutation, so reuse row_ann and col_ann
#
#     # For each stored result, compute permutation stats vectorized
#
#     pvals = []
#
#     for res in tqdm(results, desc="Permutation testing"):
#         group_code = res["group_code"]
#         ct1 = res["ct1"]
#         ct2 = res["ct2"]
#         observed_stat = res["observed_stat"]
#
#         # Create mask arrays for permuted edges matching the pairs for ref and group
#
#         # Condition masks for permuted row and col for ref and group
#         # (n_perms, n_edges) boolean arrays:
#
#         # For ref condition edges:
#         mask_perm_ref = (permuted_row_cond == ref_code) & (row_ann == ct1) & \
#                         (permuted_col_cond == ref_code) & (col_ann == ct2)
#
#         # For group condition edges:
#         mask_perm_group = (permuted_row_cond == group_code) & (row_ann == ct1) & \
#                           (permuted_col_cond == group_code) & (col_ann == ct2)
#
#         # Sum or mean values for each permutation:
#         if statistic == "sum":
#             vals_perm_ref = (mask_perm_ref * values).sum(axis=1)
#             vals_perm_group = (mask_perm_group * values).sum(axis=1)
#         elif statistic == "mean":
#             # mean = sum / count; need to handle zero counts
#             counts_ref = mask_perm_ref.sum(axis=1)
#             counts_group = mask_perm_group.sum(axis=1)
#
#             sums_ref = (mask_perm_ref * values).sum(axis=1)
#             sums_group = (mask_perm_group * values).sum(axis=1)
#
#             vals_perm_ref = np.divide(sums_ref, counts_ref, out=np.zeros_like(sums_ref), where=counts_ref > 0)
#             vals_perm_group = np.divide(sums_group, counts_group, out=np.zeros_like(sums_group), where=counts_group > 0)
#
#         elif statistic == "median":
#             # median is tricky to vectorize for sparse masks, so skip or approximate with mean
#             # To do exact median, need loop - slows down
#             # For speed, we can skip median or do approximate median
#             # Let's skip median permutation for now
#             raise NotImplementedError("Median statistic is not optimized for permutation")
#
#         else:
#             raise ValueError(f"Statistic {statistic} not supported")
#
#         # Calculate permutation stats (difference ref - group)
#         perm_stats = vals_perm_ref - vals_perm_group
#
#         # Calculate p-values based on alternative hypothesis
#         if alternative == "two-sided":
#             p_value = np.mean(np.abs(perm_stats) >= np.abs(observed_stat))
#         elif alternative == "greater":
#             p_value = np.mean(perm_stats >= observed_stat)
#         elif alternative == "less":
#             p_value = np.mean(perm_stats <= observed_stat)
#         else:
#             raise ValueError(f"Alternative {alternative} not supported")
#
#         pvals.append({
#             "group": le_condition.inverse_transform([group_code])[0],
#             "ref_ct": le_annotation.inverse_transform([ct1])[0],
#             "alt_ct": le_annotation.inverse_transform([ct2])[0],
#             "log2fc": res["log2fc"],
#             "mean_ref": res["sum_ref"],
#             "mean_group": res["sum_group"],
#             "pval": p_value,
#         })
#
#     df_pvals = pd.DataFrame(pvals)
#     df_pvals["padj"] = false_discovery_control(df_pvals["pval"])
#     return df_pvals
#
# df = neigh_perm_fast(adata, "3m")
#
#
# df_wide = df.pivot(index="ref_ct", columns="alt_ct", values="log2fc")
# df_annot =  df.pivot(index="ref_ct", columns="alt_ct", values="padj")
# df_annot = df_annot.applymap(lambda x: "*" if x < 0.05 else " ").astype(str)
#
#
#
# import seaborn as sns
# sns.clustermap(df_wide, cmap="RdYlBu_r", annot=df_annot, fmt="", square=True, yticklabels=True,
#             xticklabels=True, method="complete", center=0)
