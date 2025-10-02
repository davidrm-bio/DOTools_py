from tqdm import tqdm
from typing import Literal

import anndata as ad
import pandas as pd
import numpy as np
from scipy.stats import false_discovery_control, rv_discrete

from dotools_py import logger



def neigh_perm(adata: ad.AnnData,
             ref: str,
             connectivity_key: str = "connectivities",
             annotation_key: str  = "annotation",
             condition_key: str = "condition",
             statistic: Literal["mean", "median", "sum"] = "mean",
             n_perms: int = 1000,
             alternative: Literal["two-sided", "greater", "less"] = "two-sided"):
    """Test for differential number of neighbors between cell-types in spatial transcriptomics.

    Calculate differential number of connections between cell-types comparing across
    two conditions in spatial transcriptomics. The significance is tested using permutation test.
    When using connectivities calculated using  delaunay triangularity is recommended to use
    "sum" as statistic.
    .. note::
        Connectivities should be calculated per section. Use `squidpy.gr.spatial_neighbors` setting
        library_key to the batch_key.

    :param adata: Annotated data matrix.
    :param ref: Reference condition.
    :param connectivity_key: Key in `obsp` with the connectivities.
    :param annotation_key:  Column in `obs` with cell annotation.
    :param condition_key: Column in `obs` with conditions.
    :param statistic: statistic to use for summarise the connectivities.
    :param n_perms: Number of permutations to do.
    :param alternative: Method to use for permutation test.
    :return: Returns a pandas dataframe with the results from the permutation test.
    """

    group_by = [condition_key, annotation_key]

    labels = []
    for idx, row in adata.obs[group_by].iterrows():
        current_row = "-".join(row.values)
        labels.append(current_row)

    # Define the groups
    connections = adata.obsp[connectivity_key].copy().toarray()
    if isinstance(connections.flatten(), rv_discrete) and statistic != "sum":
        logger.warn("Connectivities are discrete but statistic is not sum")


    connections = pd.DataFrame(connections, columns=labels, index=labels)
    connections = connections.reset_index().melt(id_vars="index")
    connections.columns = ["rows", "columns", "connections"]

    condition_groups = list(adata.obs[condition_key].unique())
    condition_groups.remove(ref)
    annotation_groups = list(adata.obs[annotation_key].unique())

    def calculate_stat(g1, g2, statistic):
        if statistic == 'mean':
            observed_stat = np.mean(g1) - np.mean(g2)
        elif statistic == 'median':
            observed_stat = np.median(g1) - np.median(g2)
        elif statistic == "sum":
            raise NotImplementedError("Currently not implemented")
        else:
            raise ValueError("Not a valid statistic method use 'mean' or 'median'")
        return  observed_stat

    pvalues = []
    for group in tqdm(condition_groups):
        for ct1 in annotation_groups:
            for ct2 in annotation_groups:
                group1 = np.array(connections[(connections["rows"] == ref + "-" + ct1) &
                                              (connections["columns"] == ref + "-" + ct2)
                                  ]["connections"])
                group2 = np.array(connections[(connections["rows"] == group + "-" + ct1) &
                                              (connections["columns"] == group + "-" + ct2)
                                              ]["connections"])
                combined = np.concatenate([group1, group2])

                log2fc = np.log2((group1.mean() + 1e-9) / (group2.mean() + 1e-9))

                observed_stat = calculate_stat(group1, group2, statistic)

                permuted_stats = []
                for _ in range(n_perms):
                    np.random.shuffle(combined)
                    perm_group1 = combined[:len(group1)]
                    perm_group2 = combined[len(group1):]

                    stat = calculate_stat(perm_group1, perm_group2, statistic)
                    permuted_stats.append(stat)
                permuted_stats = np.array(permuted_stats)

                # Compute p-value based on alternative hypothesis
                if alternative == 'two-sided':
                    p_value = np.mean(np.abs(permuted_stats) >= np.abs(observed_stat))
                elif alternative == 'greater':
                    p_value = np.mean(permuted_stats >= observed_stat)
                elif alternative == 'less':
                    p_value = np.mean(permuted_stats <= observed_stat)
                else:
                    raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

                pvalues.append((group, ct1, ct2, log2fc, group1.mean(), group2.mean(), p_value))

    df_pvals = pd.DataFrame(pvalues, columns=["group", "ref_ct", "alt_ct", "log2fc", "mean_ref", "mean_group",  "pval"])
    df_pvals["padj"] = false_discovery_control(df_pvals["pval"])
    return  df_pvals
