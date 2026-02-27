import tqdm
from typing import Any, Tuple, Literal
from numpy.typing import NDArray

import anndata as ad
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

from dotools_py.logger import logger
from dotools_py.utility import free_memory



def _compute_nndescent(
    data: Any,
    n_neighbors: int,
    metric: str = "euclidean",
    n_jobs: int = -1,
    random_state: int = 0
) -> Any:
    import pynndescent
    return pynndescent.NNDescent(
        data, metric=metric, n_neighbors=n_neighbors, random_state=random_state,
        parallel_batch_queries=True, n_jobs=n_jobs,
    )


def find_neighbor(
    cc1: Any,
    cc2: Any,
    k: int,
    random_state: int = 0,
    n_jobs: int = -1
) -> tuple:
    """Find k-nearest neighbors within and across two datasets.

    :param cc1: Reduced-dimensional representation of dataset 1 (cells x features)
    :param cc2: Reduced-dimensional representation of dataset 2 (cells x features)
    :param k: Number of neighbors to consider.
    :param random_state: Random seed.
    :param n_jobs: Number of parallel jobs for neighbor search.

    Returns
    -------
    Neighbor indices:

    - G11: neighbors within dataset 1
    - G12: neighbors from dataset 1 to 2
    - G21: neighbors from dataset 2 to 1
    - G22: neighbors within dataset 2

    """
    index = _compute_nndescent(cc1, n_neighbors=k + 1, random_state=random_state, n_jobs=n_jobs)
    g11 = index.neighbor_graph[0][:, 1:k + 1]
    g21 = index.query(cc2, k=k)[0]
    index = _compute_nndescent(cc2, n_neighbors=k + 1, random_state=random_state, n_jobs=n_jobs)
    g22 = index.neighbor_graph[0][:, 1:k + 1]
    g12 = index.query(cc1, k=k)[0]
    return g11, g12, g21, g22


def find_mnn(
    g12: Any,
    g21: Any,
    k_anchor: int
) -> NDArray:
    """Calculate the mutual nearest neighbor for two datasets.

    For every point i in dataset 1, check if it is also among the k_anchor neighbors
    of that point in dataset 2 (i.e, a cell X in dataset 1 is neighbor from a cell Y in dataset 2 and
    vice versa). We take bidirectional neighbors.

    :param g12: Neighbor matrix from dataset 1 to dataset 2
    :param g21: Neighbor matrix from dataset 2 to dataset 1
    :param k_anchor: Number of k-nearest neighbors.
    :return: Returns an array of anchor pairs (shape: n_anchor x 2)
    """
    anchor = [
        [i, g12[i, j]] for i in range(g12.shape[0])
        for j in range(k_anchor)
        if (i in g21[g12[i, j], :k_anchor])
    ]
    return np.array(anchor)


def min_max(
    tmp: Any,
    q_left: int = 1,
    q_right: int = 90
) -> NDArray:
    """
    Modified Min/Max normalization.

    Min/Max normalization between the q_left and q_right quantiles.
    The extreme values are clip.

    :param tmp: Numpy Array.
    :param q_left: Lower quantile that will correspond to 0.
    :param q_right: Upper quantile that will correspond to 1
    :return: Returns a normalized numpy array.
    """
    t_min, t_max = np.percentile(tmp, [q_left, q_right])
    tmp = (tmp - t_min) / (t_max - t_min)
    tmp[tmp > 1] = 1
    tmp[tmp < 0] = 0
    return tmp


def filter_anchor(
    anchor: Any,
    adata_ref: ad.AnnData = None,
    adata_qry: ad.AnnData = None,
    scale_ref: bool = False,
    scale_qry: bool = False,
    high_dim_feature=None,
    k_filter: int = 200,
    random_state: int = 0,
    n_jobs: int = -1,
):
    """
    Check if an anchor is still an anchor when only using the
    high_dim_features to construct KNN graph. Keep only anchors present
    in both cases.

    :param anchor: Numpy array with anchors.
    :param adata_ref: Reference annotated data matrix.
    :param adata_qry: Query annotated data matrix.
    :param scale_ref: Whether to Z-score transform the reference data.
    :param scale_qry: Whether to Z-score transform the query data.
    :param high_dim_feature: Features indices to use to compute the KNN graph.
    :param k_filter: The number of k-nearest neighbors to use.
    :param random_state: Random seed.
    :param n_jobs: Number of threads to use.
    :return: Returns a filtered array of anchors.
    """
    from scipy.stats import zscore
    from scipy.sparse import issparse
    ref_data = (
        adata_ref.X[:, high_dim_feature].toarray() if issparse(adata_ref.X) else adata_ref.X[:, high_dim_feature].copy()
    )
    if scale_ref:
        ref_data = zscore(ref_data, axis=0)
    ref_data = normalize(ref_data, axis=1)
    qry_data = (
        adata_qry.X[:, high_dim_feature].toarray() if issparse(adata_qry.X) else adata_qry.X[:, high_dim_feature].copy()
    )
    if scale_qry:
        qry_data = zscore(qry_data, axis=0)
    qry_data = normalize(qry_data, axis=1)

    index = _compute_nndescent(ref_data, n_neighbors=k_filter, random_state=random_state, n_jobs=n_jobs)
    graph = index.query(qry_data, k=k_filter)[0]
    anchor = np.array([xx for xx in anchor if (xx[0] in graph[xx[1]])])
    logger.debug(f"Anchor selected with high CC feature graph: {anchor.shape[0]} / {anchor.shape[0]}")
    return anchor


def score_anchor(
    anchor: Any,
    g11: NDArray,
    g12: NDArray,
    g21: NDArray,
    g22: NDArray,
    k_score: int = 30,
    gp1: NDArray = None,
    gp2: NDArray = None,
    k_local: int = 50
):
    """
    Score the anchor by the number of shared neighbors

    :param anchor: anchor pairs
    :param g11: Neighbor matrices for dataset pairs (1 to 1)
    :param g12: Neighbor matrices for dataset pairs (1 to 2)
    :param g21: Neighbor matrices for dataset pairs (2 to 1)
    :param g22: Neighbor matrices for dataset pairs (2 to 2)
    :param k_score: Number of neighbors for scoring
    :param gp1: Local intra-dataset KNN graphs
    :param gp2: Local intra-dataset KNN graphs
    :param k_local: Number of neighbors for local scoring
    :return: Returns a dataframe with anchors with scores
    """

    tmp = [
        len(set(g11[x, :k_score]).intersection(g21[y, :k_score]))
        + len(set(g12[x, :k_score]).intersection(g22[y, :k_score]))
        for x, y in anchor
    ]
    anchor_df = pd.DataFrame(anchor, columns=["x1", "x2"])
    anchor_df["score"] = min_max(tmp)

    if k_local:
        # if k_local is not None, then use local KNN to adjust the score
        share_nn = np.array([len(set(gp1[i]).intersection(g11[i, :k_local])) for i in range(len(gp1))])
        tmp = [share_nn[xx] for xx in anchor_df["x1"].values]
        anchor_df["score_local1"] = min_max(tmp)

        share_nn = np.array([len(set(gp2[i]).intersection(g22[i, :k_local])) for i in range(len(gp2))])
        tmp = [share_nn[xx] for xx in anchor_df["x2"].values]
        anchor_df["score_local2"] = min_max(tmp)

        anchor_df["score"] = anchor_df["score"] * anchor_df["score_local1"] * anchor_df["score_local2"]
    return anchor_df


def find_order(dist: NDArray, ncell: list) -> list:
    """Use dendrogram to find the order of the datasets pairs.

    :param dist:
    :param ncell:
    :return:
    """
    from scipy.cluster.hierarchy import linkage

    d = linkage(1 / dist, method="average")
    node_dict = {i: [i] for i in range(len(ncell))}
    alignment = []
    for xx in d[:, :2].astype(int):
        if ncell[xx[0]] < ncell[xx[1]]:
            xx = xx[::-1]
        alignment.append([node_dict[xx[0]], node_dict[xx[1]]])
        node_dict[len(ncell)] = node_dict[xx[0]] + node_dict[xx[1]]
        ncell.append(ncell[xx[0]] + ncell[xx[1]])
    return alignment


class SeuratIntegration:
    """
    Main class for Seurat integration. Adapted from AllCools
    `Hanqing L., et al. Nature (2021) <https://www.nature.com/articles/s41586-020-03182-8>`_

    Examples
    --------
    >>> import dotools_py as do
    >>> import numpy as np
    >>> import scanpy as sc
    >>> adata = do.dt.example_10x_processed()
    >>> integrator = SeuratIntegration(random_state=0, n_jobs=-1)
    >>> adata_list = [adata[adata.obs["batch"] == b] for b in adata.obs["batch"].unique()]
    >>> integrator.find_anchor(adata_list=adata_list, n_components=50)
    >>> X_CCA = integrator.integrate(key_correct="X")
    >>> X_CCA = np.concatenate(X_CCA)
    >>> adata.obsm["X_CCA"] = X_CCA
    >>> sc.pp.neighbors(adata, use_rep="X_CCA")
    >>> sc.tl.umap(adata)
    """

    def __init__(self, n_jobs: int = -1, random_state: int = 0) -> None:
        from collections import OrderedDict

        self.n_jobs = n_jobs

        # intra-dataset KNN graph
        self.k_local = None
        self.key_local = None
        self.local_knn = []

        self.adata_dict = OrderedDict()
        self.n_dataset = 0
        self.n_cells = []
        self.alignments = None
        self.all_pairs = np.array([])
        self._get_all_pairs()

        self.anchor = {}
        self.mutual_knn = {}
        self.raw_anchor = {}
        self.label_transfer_results = {}

        self.random_state = random_state

    def _calculate_local_knn(self) -> None:
        """Calculate local KNN graph for each dataset.

        If k_local is provided, we calculate the local knn graph to evaluate
        whether the anchor preserves local structure within the dataset.
        One can use a different obsm with key_local to compute knn for each dataset.

        k_local is a key in `adata.obsm`

        :return: Returns None. The local_knn attribute will contain the cell neighbors within each dataset.
        """
        if self.k_local is not None:
            logger.debug("Find neighbors within datasets")
            for adata in self.adata_dict.values():
                index = _compute_nndescent(
                    data=adata.obsm[self.key_local], n_neighbors=self.k_local + 1, random_state=self.random_state,
                    n_jobs=self.n_jobs
                )
                self.local_knn.append(index.neighbor_graph[0][:, 1:])  # Add connections except the self-one
        else:
            self.local_knn = [None for _ in self.adata_dict.values()]
        return None

    def _get_all_pairs(self) -> None:
        """Determine all possible pair combinations.

        `self.alignments` need to be defined, being this a list of pairs. For example, for 3 datasets (0, 1, and 2) we
        would have: [[[0], [1]], [[0,1], [2]]] leading to [0-1, 0-2, 1-2] indicated all the possible combinations
        we can do for the integration.

        :return: Returns None. All the possible pairs will be saved in `self.all_pairs`
        """
        if self.alignments is not None:
            self.all_pairs = np.unique([
                f"{min(x, y)}-{max(x, y)}" for a, b in self.alignments for x in a for y in b
            ])  # For every pair [[pair1], [pair2]] --> [pair1] = [[dataset1, ...], [dataset2, ...]]
        else:
            self.all_pairs = np.array([])
        return None

    def _prepare_matrix(self, i: int | str, j: int | str, key_anchor: str) -> Tuple[NDArray, NDArray]:
        """Extract matrix.

        If `key_anchor` is `X` it will extract the expression values in `adata.X`, otherwise it needs to be a valid
        key in `adata.obsm`. If `key_anchor` is set to `X` double checks if the order and number of genes is the same.

        :param i: Name of dataset1.
        :param j: Name of dataset2.
        :param key_anchor: It can be set to `X` to extract the `adata.X` or a key in `adata.obsm`.
        :return: Returns the matrix from dataset1 and dataset2.
        """
        adata1, adata2 = self.adata_dict[i], self.adata_dict[j]
        possible_keys = list(adata1.obsm.keys()) + ["X"]
        assert key_anchor in possible_keys, f"{key_anchor} not a valid key. Use {possible_keys}"
        if key_anchor == "X":
            # Check - Make sure adata.var is in the same order and it matches
            if (adata1.n_vars != adata2.n_vars) or ((adata1.var_names == adata2.var_names).sum() < adata1.n_vars):
                common_vars = list(adata1.var_names.intersect(adata2.var_names))
                u, v = adata1[:, common_vars].X.copy(), adata2[:, common_vars].X.copy()
            else:
                u, v = adata1.X.copy(), adata2.X.copy()
        else:
            u, v = adata1.obsm[key_anchor], adata2.obsm[key_anchor]
        return u, v

    def _calculate_mutual_knn_and_raw_anchors(
        self,
        i: int | str,
        j: int | str,
        u: NDArray,
        v: NDArray,
        k: int,
        k_anchor: int
    ) -> tuple:
        """Calculate the mutual KNN graph and raw anchors.

        :param i: name of dataset1.
        :param j: name of dataset2.
        :param u: Matrix from dataset1 extracted after running `self._prepare_matrix`. Data in `X` or `adata.obsm`.
        :param v: Matrix from dataset1 extracted after running `self._prepare_matrix`. Data in `X` or `adata.obsm`.
        :param k: Number of nearest neighbors.
        :param k_anchor: Number of mutual nearest neighbors.
        :return: Returns tuple of neighbors indices (1->1, 1->2, 2->1, 2->2) and anchors (mutual neighbors). This is
                also saved in `self.mutual_knn` and `self.raw_anchors`.

        """
        g11, g12, g21, g22 = find_neighbor(cc1=u, cc2=v, k=k, n_jobs=self.n_jobs, random_state=self.random_state)
        raw_anchors = find_mnn(g12=g12, g21=g21, k_anchor=k_anchor)
        self.mutual_knn[(i, j)] = (g11, g12, g21, g22)
        self.raw_anchor[(i, j)] = raw_anchors
        return g11, g12, g21, g22, raw_anchors

    def _pairwise_find_anchor(
        self,
        i: int | str,
        i_sel: Any,
        j: int | str,
        j_sel: Any,
        dim_red: Literal["cca", "pca", "lsi", "lsi-cca", "rpca", "rlsi"],
        key_anchor: str,
        svd_algorithm: str,
        scale1: bool,
        scale2: bool,
        k_anchor: int,
        k_local: int,
        k_score: int,
        ncc: int,
        max_cc_cell: int,
        k_filter: int,
        n_features: int,
        chunk_size: int,
        signorm: bool,
    ) -> pd.DataFrame:
        """Compute pairwise anchors between two datasets.

        :param i: Name of dataset1.
        :param i_sel: If set, a subset of cells from dataset1 is selected.
        :param j: Name of dataset2.
        :param j_sel: If set, a subset of cells from dataset2 is selected.
        :param dim_red: Dimensionality reduction method to use.
        :param key_anchor: What data should be used to find anchors. Can be `X` to use the expression or a key in `adata.obsm`.
        :param svd_algorithm: SVD solver to use. Either “arpack” for the ARPACK wrapper in SciPy, or 'randomized' for the randomized algorithm.
        :param scale1: Whether to scale dataset1 or not.
        :param scale2: Whether to scale dataset2 or not.
        :param k_anchor: Number of nearest neighbors for mutual KNN.
        :param k_local: Number of nearest neighbors for within data  KNN.
        :param k_score: Number of nearest neighbors for scoring.
        :param ncc: Number of components to compute during dimensionality reduction.
        :param max_cc_cell: Maximum number of cells to consider.
        :param k_filter:
        :param n_features: Number of top features to select.
        :param chunk_size: Chunk size to process the data.
        :param signorm: Whether to convert components from scaled SVD coordinates to unit-variance (whitened) coordinates.
        :return: Returns a DataFrame with the anchors.
        """
        from dotools_py.tl._CCA.cca import cca, lsi_cca, LSI, SVD, downsample

        adata1, adata2 = self.adata_dict[i], self.adata_dict[j]
        min_sample = min(adata1.n_obs, adata2.n_obs)

        # Subset in case `obs` values are selected
        if i_sel is not None:
            adata1 = adata1[i_sel, :]
        if j_sel is not None:
            adata2 = adata2[j_sel, :]

        if dim_red in ("cca", "pca", "lsi", "lsi-cca"):
            # 1. prepare input matrix for CCA
            u, v = self._prepare_matrix(i=i, j=j, key_anchor=key_anchor)

            # 2. run cca between datasets
            if dim_red in ("cca", "pca"):
                logger.debug("Run CCA")
                u, v, high_dim_feature = cca(
                    data1=u,
                    data2=v,
                    scale1=scale1,
                    scale2=scale2,
                    n_components=ncc,
                    max_cc_cell=max_cc_cell,
                    k_filter=k_filter,
                    n_features=n_features,
                    chunk_size=chunk_size,
                    svd_algorithm=svd_algorithm,
                    random_state=self.random_state,
                )
            elif dim_red in ("lsi", "lsi-cca"):
                logger.debug("Run LSI-CCA")
                u, v = lsi_cca(
                    data1=u,
                    data2=v,
                    scale_factor=100_000,
                    n_components=ncc,
                    max_cc_cell=max_cc_cell,
                    chunk_size=chunk_size,
                    svd_algorithm=svd_algorithm,
                    min_cov_filter=5,
                    random_state=self.random_state,
                )
                high_dim_feature = None
            else:
                raise ValueError(f"Dimension reduction method {dim_red} is not supported.")

            # 3. normalize CCV per sample/row
            u, v = normalize(u, axis=1), normalize(v, axis=1)

            # 4. find MNN of U and V to find anchors
            _k = max(_temp for _temp in [k_anchor, k_local, k_score] if _temp is not None)
            _k = min(min_sample - 2, _k)
            logger.debug(f"Find anchors using k={_k}")
            g11, g12, g21, g22, raw_anchors = self._calculate_mutual_knn_and_raw_anchors(
                i=i, j=j, u=u, v=v, k=_k, k_anchor=k_anchor
            )

            # 5. filter anchors by high dimensional neighbors
            if k_filter is not None and high_dim_feature is not None:
                # compute ccv feature loading
                if self.n_cells[i] >= self.n_cells[j]:
                    raw_anchors = filter_anchor(
                        anchor=raw_anchors,
                        adata_ref=adata1,
                        adata_qry=adata2,
                        scale_ref=scale1,
                        scale_qry=scale2,
                        high_dim_feature=high_dim_feature,
                        k_filter=k_filter,
                        random_state=self.random_state,
                        n_jobs=self.n_jobs,
                    )
                else:
                    raw_anchors = filter_anchor(
                        anchor=raw_anchors[:, ::-1],
                        adata_ref=adata2,
                        adata_qry=adata1,
                        scale_ref=scale2,
                        scale_qry=scale1,
                        high_dim_feature=high_dim_feature,
                        k_filter=k_filter,
                        random_state=self.random_state,
                        n_jobs=self.n_jobs,
                    )[:, ::-1]
        elif dim_red in ("rpca", "rlsi"):
            adata1, adata2 = adata1.X, adata2.X
            k = max(i for i in [k_anchor, k_local, k_score, 50] if i is not None)

            # Initialize model
            logger.debug(f"Run {dim_red}")
            model = (
                SVD(n_components=ncc, random_state=self.random_state) if dim_red == "rpca" else
                LSI(n_components=ncc, random_state=self.random_state)
            )

            tf1, tf2, scaler1, scaler2 = downsample(
                adata1, adata2, scale1=scale1, scale2=scale2, max_cc_cell=max_cc_cell,
                todense=True if dim_red == "rpca" else False,
            )

            # Project adata2 to adata1
            model.fit(tf1)
            u, v = (
                model.transform(adata1, chunk_size=chunk_size, scaler=scaler1),
                model.transform(adata2, chunk_size=chunk_size, scaler=scaler2),
            )
            if dim_red == "rpca" and signorm:
                u = u / model.model.singular_values_
                v = v / model.model.singular_values_
            index = _compute_nndescent(u, n_neighbors=k + 1, random_state=self.random_state, n_jobs=self.n_jobs)
            g11 = index.neighbor_graph[0][:, 1: k + 1]
            g21 = index.query(v, k=k)[0]

            # Project adata1 to adata2
            model.fit(tf2)
            u, v = (
                model.transform(adata1, chunk_size=chunk_size, scaler=scaler1),
                model.transform(adata2, chunk_size=chunk_size, scaler=scaler2),
            )
            if dim_red == "rpca" and signorm:
                u = u / model.model.singular_values_
                v = v / model.model.singular_values_
            index = _compute_nndescent(v, n_neighbors=k + 1, random_state=self.random_state, n_jobs=self.n_jobs)
            g22 = index.neighbor_graph[0][:, 1: k + 1]
            g12 = index.query(u, k=k)[0]
            raw_anchors = find_mnn(g12, g21, k_anchor)
        else:
            raise ValueError(f"Dimension reduction method {dim_red} is not supported.")

        # 6. score anchors with snn and local structure preservation
        logger.debug("Score Anchors")
        anchor_df = score_anchor(
            anchor=raw_anchors,
            g11=g11,
            g12=g12,
            g21=g21,
            g22=g22,
            k_score=k_score,
            k_local=k_local,
            gp1=self.local_knn[i],
            gp2=self.local_knn[j],
        )
        return anchor_df


    def find_anchor(
        self,
        adata_list: list,
        adata_names: list = None,
        k_local: int = None,
        key_local: str = "X_pca",
        key_anchor: str = "X",
        dim_red: Literal["cca", "pca", "lsi", "lsi-cca", "rpca", "rlsi"] = "pca",
        svd_algorithm: Literal["randomized", "arpack"] = "randomized",
        scale1: bool = True,
        scale2: bool = True,
        scale_list: list = None,
        k_filter: int = None,
        n_features: int = 200,
        n_components: int = 50,
        max_cc_cells: int = 50_000,
        chunk_size: int = 50_000,
        k_anchor: int = 5,
        k_score: int = 30,
        alignments: list = None,
        signorm: bool = True,
        key_match=None,
    ) -> None:
        """Find anchors for each dataset pair.

        :param adata_list: List of AnnData object to integrate.
        :param adata_names: Name assign to each AnnData object.
        :param k_local: k-nearest neighbors for within KNN graph.
        :param key_local: Key in `adata.obsm` with the reduction to use to compute local KNN graph.
        :param key_anchor: Matrix to use for finding anchors. If set to `X` use the expression otherwise use key in `adata.obsm`.
        :param dim_red: Dimensionality reduction method to use.
        :param svd_algorithm: SVD solver to use. Either 'arpack' for the ARPACK wrapper in SciPy, or 'randomized' for the randomized algorithm.
        :param scale1: Whether to scale dataset1 or not.
        :param scale2: Whether to scale dataset2 or not.
        :param scale_list: List of scale datasets.
        :param k_filter:
        :param n_features: Number of top features to select.
        :param n_components: Number of components to compute.
        :param max_cc_cells: Maximum number cells for components.
        :param chunk_size: Chunk size to process the data.
        :param k_anchor: K-nearest neighbors for finding mutual KNN.
        :param k_score: K-nearest neighbors for scoring CC.
        :param alignments: ALl possible pair combinations.
        :param signorm: Whether to convert components from scaled SVD coordinates to unit-variance (whitened) coordinates.
        :param key_match:
        :return: Returns None. The anchors will be saved in `self.anchors`.
        """
        if dim_red not in ["pca", "cca", "lsi", "lsi-cca", "rpca", "rlsi"]:
            raise ValueError(f"Dimension reduction method {dim_red} is not supported.")

        adata_names = list(range(len(adata_list))) if adata_names is None else adata_names
        assert len(adata_names) == len(adata_list), "Length of `adata_names` does not match length of `adata_list`"

        self.adata_dict = {k: v for k, v in zip(adata_names, adata_list)}
        self.n_dataset = len(adata_list)
        self.n_cells = [adata.n_obs for adata in adata_list]

        # Intra-dataset KNN for scoring the anchors
        self.k_local = k_local
        self.key_local = key_local
        self._calculate_local_knn()

        # alignments and all_pairs
        self.alignments = alignments
        self._get_all_pairs()

        logger.info("Finding anchors across datasets")
        for i in tqdm.tqdm(range(self.n_dataset - 1), total=self.n_dataset - 1, desc="Batches "):
            for j in range(i + 1, self.n_dataset):
                if scale_list is not None:
                    scale1, scale2 = scale_list[i], scale_list[j]
                    logger.debug("Get scale1 and scale2 from scale_list")
                    logger.debug(f"dataset {i} scale: {scale1}")
                    logger.debug(f"dataset {j} scale: {scale2}")

                if key_match is None:
                    anchor_df = self._pairwise_find_anchor(
                        i=i, i_sel=None,
                        j=j, j_sel=None,
                        dim_red=dim_red, key_anchor=key_anchor, svd_algorithm=svd_algorithm,
                        scale1=scale1, scale2=scale2,
                        k_anchor=k_anchor, k_local=k_local, k_score=k_score,
                        ncc=n_components, max_cc_cell=max_cc_cells,
                        k_filter=k_filter, n_features=n_features,
                        chunk_size=chunk_size, signorm=signorm,
                    )
                else:
                    tissue = [xx.obs[key_match].unique() for xx in adata_list]
                    sharet = list(set(tissue[i]).intersection(tissue[j]))
                    if len(sharet) > 0:
                        anchor_df_list = []
                        for t in sharet:
                            logger.debug(t)
                            adata1, adata2 = adata_list[i].copy(), adata_list[j].copy()

                            idx1 = np.where(adata1.obs[key_match] == t)[0]
                            idx2 = np.where(adata2.obs[key_match] == t)[0]
                            tmp = self._pairwise_find_anchor(
                                i=i, i_sel=idx1,
                                j=j, j_sel=idx2,
                                dim_red=dim_red, key_anchor=key_anchor, svd_algorithm=svd_algorithm,
                                scale1=scale1, scale2=scale2,
                                k_anchor=k_anchor, k_local=k_local, k_score=k_score,
                                ncc=n_components, max_cc_cell=max_cc_cells,
                                k_filter=k_filter, n_features=n_features,
                                chunk_size=chunk_size, signorm=signorm,
                            )
                            tmp["x1"] = idx1[tmp["x1"].values]
                            tmp["x2"] = idx2[tmp["x2"].values]
                            anchor_df_list.append(tmp)
                        anchor_df = pd.concat(anchor_df_list, axis=0)
                    else:
                        anchor_df = self._pairwise_find_anchor(
                            i=i, i_sel=None,
                            j=j, j_sel=None,
                            dim_red="rpca", key_anchor=key_anchor, svd_algorithm=svd_algorithm,
                            scale1=scale1, scale2=scale2,
                            k_anchor=k_anchor, k_local=k_local, k_score=k_score,
                            ncc=n_components, max_cc_cell=max_cc_cells,
                            k_filter=k_filter, n_features=n_features,
                            chunk_size=chunk_size, signorm=signorm,
                        )
                # save anchors
                self.anchor[(i, j)] = anchor_df.copy()
                logger.debug(f"Identified {len(self.anchor[i, j])} anchors between datasets {i} and {j}.")
        return None

    def find_nearest_anchor(
        self,
        data: NDArray,
        data_qry: NDArray,
        ref: list,
        qry: list,
        key_correct: str = "X_pca",
        npc: int = 30,
        k_weight: int = 100,
        sd: float = 1,
    ):
        """Find the nearest anchors for each cell in data."""
        from sklearn.decomposition import PCA

        logger.debug("Initialize")
        cum_ref, cum_qry = [0], [0]
        for xx in ref:
            cum_ref.append(cum_ref[-1] + data[xx].shape[0])
        for xx in qry:
            cum_qry.append(cum_qry[-1] + data[xx].shape[0])

        anchor = []
        for i, xx in enumerate(ref):
            for j, yy in enumerate(qry):
                if xx < yy:
                    tmp = self.anchor[(xx, yy)].copy()
                else:
                    tmp = self.anchor[(yy, xx)].copy()
                    tmp[["x1", "x2"]] = tmp[["x2", "x1"]]
                tmp["x1"] += cum_ref[i]
                tmp["x2"] += cum_qry[j]
                anchor.append(tmp)
        anchor = pd.concat(anchor)
        score = anchor["score"].values
        anchor = anchor[["x1", "x2"]].values

        if key_correct == "X":
            model = PCA(n_components=npc, svd_solver="arpack", random_state=self.random_state)
            reduce_qry = model.fit_transform(data_qry)
        else:
            reduce_qry = data_qry[:, :npc]

        logger.debug("Find nearest anchors")
        index = _compute_nndescent(
            data=reduce_qry[anchor[:, 1]], n_neighbors=k_weight, random_state=self.random_state, n_jobs=self.n_jobs
        )
        k_weight = min(k_weight, anchor.shape[0] - 5)
        k_weight = max(5, k_weight)
        logger.debug(f"k_weight: {k_weight}")
        g, d = index.query(reduce_qry, k=k_weight)

        logger.debug("Normalize graph")
        cell_filter = d[:, -1] == 0
        d = (1 - d / d[:, -1][:, None]) * score[g]
        d[cell_filter] = score[g[cell_filter]]
        d = 1 - np.exp(-d * (sd ** 2) / 4)
        d = d / (np.sum(d, axis=1) + 1e-6)[:, None]
        return anchor, g, d, cum_qry

    def transform(
        self,
        data: NDArray,
        ref: list,
        qry: list,
        key_correct: str,
        npc: int = 30,
        k_weight: int = 100,
        sd: float = 1,
        chunk_size: int = 50_000,
        row_normalize: bool = True,
    ) -> Any:
        """Transform query data to reference data.

        :param data: Matrix to transform.
        :param ref: List with indices of the reference data.
        :param qry:  List with indices of the query data.
        :param key_correct: Correct Matrix key.
        :param npc: Number of components.
        :param k_weight:
        :param sd: standard deviation
        :param chunk_size: Chunk size for processing.
        :param row_normalize: Whether to normalize to unit variance per row or not
        :return: Returns a numpy array transformed.
        """
        data_ref = np.concatenate(data[ref])
        data_qry = np.concatenate(data[qry])

        anchor, g, d, cum_qry = self.find_nearest_anchor(
            data=data, data_qry=data_qry, key_correct=key_correct, ref=ref, qry=qry,
            npc=npc, k_weight=k_weight, sd=sd,
        )

        logger.debug("Transform data")
        bias = data_ref[anchor[:, 0]] - data_qry[anchor[:, 1]]
        data_prj = np.zeros(data_qry.shape)

        for chunk_start in np.arange(0, data_prj.shape[0], chunk_size):
            data_prj[chunk_start: (chunk_start + chunk_size)] = (
                data_qry[chunk_start: (chunk_start + chunk_size)] + (
                d[chunk_start: (chunk_start + chunk_size), :, None] * bias[g[chunk_start: (chunk_start + chunk_size)]]
            ).sum(axis=1)
            )
        for i, xx in enumerate(qry):
            _data = data_prj[cum_qry[i]: cum_qry[i + 1]]
            if row_normalize:
                _data = normalize(_data, axis=1)
            data[xx] = _data
        return data

    def integrate(
        self,
        key_correct,
        row_normalize: bool = True,
        n_components: int = 30,
        k_weight: int = 100,
        sd: float = 1,
        alignments: list = None
    ) -> NDArray:
        """Integrate datasets by transforming data matrices from query to reference data using the
        MNN information.

        :param key_correct:
        :param row_normalize: Whether to scale row to unit-variance.
        :param n_components: Number of components to compute.
        :param k_weight:
        :param sd: standard deviation
        :param alignments: list of alignments pairs.
        :return: Returns numpy array with corrected matrix.
        """
        self.alignments = alignments if alignments is not None else None

        # Find order of pairwise datasets merging with hierarchical clustering
        if self.alignments is None:
            dist = []
            for i in range(self.n_dataset - 1):
                for j in range(i + 1, self.n_dataset):
                    dist.append(len(self.anchor[(i, j)]) / min([self.n_cells[i], self.n_cells[j]]))
            self.alignments = find_order(np.array(dist), self.n_cells)
            logger.debug(f"Alignments: {self.alignments}")

        logger.debug("Merge datasets")
        adata_list = list(self.adata_dict.values())

        # Initialize corrected with original data
        if key_correct == "X":
            # Correct the original feature matrix
            corrected = [adata_list[i].X.toarray().copy() for i in range(self.n_dataset)]
        else:
            # Correct dimensionality reduced matrix only
            corrected = [normalize(adata_list[i].obsm[key_correct], axis=1) for i in range(self.n_dataset)]

        for xx in tqdm.tqdm(self.alignments):
            logger.debug(xx)
            corrected = self.transform(
                data=np.array(corrected, dtype="object"),
                ref=xx[0], qry=xx[1],
                npc=n_components, k_weight=k_weight, sd=sd, row_normalize=row_normalize, key_correct=key_correct,
            )
            free_memory()
        return corrected

    # def label_transfer(
    #     self,
    #     ref,
    #     qry,
    #     categorical_key=None,
    #     continuous_key=None,
    #     key_dist="X_pca",
    #     k_weight=100,
    #     npc=30,
    #     sd=1,
    #     chunk_size=50000,
    #     random_state=0,
    # ):
    #     """Transfer labels from query to reference space."""
    #     adata_list = list(self.adata_dict.values())
    #
    #     data_qry = np.concatenate([normalize(adata_list[i].obsm[key_dist], axis=1) for i in qry])
    #     data_qry_index = np.concatenate([adata_list[i].obs_names for i in qry])
    #
    #     anchor, G, D, cum_qry = self.find_nearest_anchor(
    #         data=adata_list,
    #         data_qry=data_qry,
    #         ref=ref,
    #         qry=qry,
    #         npc=npc,
    #         k_weight=k_weight,
    #         key_correct=key_dist,
    #         sd=sd,
    #         random_state=random_state,
    #     )
    #     print("Label transfer")
    #     label_ref = []
    #     columns = []
    #     cat_counts = []
    #
    #     if categorical_key is None:
    #         categorical_key = []
    #     if continuous_key is None:
    #         continuous_key = []
    #     if len(categorical_key) == 0 and len(continuous_key) == 0:
    #         raise ValueError("No categorical or continuous key specified.")
    #
    #     if len(categorical_key) > 0:
    #         tmp = pd.concat([adata_list[i].obs[categorical_key] for i in ref], axis=0)
    #         enc = OneHotEncoder()
    #         label_ref.append(enc.fit_transform(tmp[categorical_key].values.astype(np.str_)).toarray())
    #         # add categorical key to make sure col is unique
    #         columns += enc.categories_
    #         # enc.categories_ are a list of arrays, each array are categories in that categorical_key
    #         cat_counts += [cats.size for cats in enc.categories_]
    #
    #     if len(continuous_key) > 0:
    #         tmp = pd.concat([adata_list[i].obs[continuous_key] for i in ref], axis=0)
    #         label_ref.append(tmp[continuous_key].values)
    #         columns += [[xx] for xx in continuous_key]
    #         cat_counts += [1 for _ in continuous_key]
    #
    #     label_ref = np.concatenate(label_ref, axis=1)
    #     label_qry = np.zeros((data_qry.shape[0], label_ref.shape[1]))
    #
    #     bias = label_ref[anchor[:, 0]]
    #     for chunk_start in np.arange(0, label_qry.shape[0], chunk_size):
    #         label_qry[chunk_start: (chunk_start + chunk_size)] = (
    #             D[chunk_start: (chunk_start + chunk_size), :, None] * bias[G[chunk_start: (chunk_start + chunk_size)]]
    #         ).sum(axis=1)
    #
    #     all_column_names = np.concatenate(columns)  # these column names might be duplicated
    #     all_column_variables = np.repeat(categorical_key + continuous_key, cat_counts)
    #     label_qry = pd.DataFrame(label_qry, index=data_qry_index, columns=all_column_names)
    #     result = {}
    #     for key in categorical_key + continuous_key:
    #         result[key] = label_qry.iloc[:, all_column_variables == key]
    #     return result
    #
    # def save(self, output_path, save_local_knn=False, save_raw_anchor=False, save_mutual_knn=False, save_adata=False):
    #     """Save the model and results to disk."""
    #     # save each adata in a separate dir
    #     output_path = pathlib.Path(output_path)
    #     output_path.mkdir(exist_ok=True, parents=True)
    #     if save_adata:
    #         # save adata and clear the self.adata_dict
    #         adata_dir = output_path / "adata"
    #         adata_dir.mkdir(exist_ok=True)
    #         with open(f"{adata_dir}/order.txt", "w") as f:
    #             for k, v in self.adata_dict.items():
    #                 for col, val in v.obs.items():
    #                     if val.dtype == "O":
    #                         v.obs[col] = val.fillna("nan").astype(str)
    #                     elif val.dtype == "category":
    #                         v.obs[col] = val.fillna("nan").astype(str)
    #                     else:
    #                         pass
    #                 v.write_h5ad(f"{adata_dir}/{k}.h5ad")
    #                 f.write(f"{k}\n")
    #
    #     # clear the adata in integrator
    #     self.adata_dict = {}
    #
    #     if not save_local_knn:
    #         self.local_knn = []
    #     if not save_raw_anchor:
    #         self.raw_anchor = {}
    #     if not save_mutual_knn:
    #         self.mutual_knn = {}
    #
    #     joblib.dump(self, f"{output_path}/model.lib")
    #     return
    #
    # @classmethod
    # def load(cls, input_path):
    #     """Load integrator from file."""
    #     adata_dir = f"{input_path}/adata"
    #     model_path = f"{input_path}/model.lib"
    #     obj = joblib.load(model_path)
    #     orders = pd.read_csv(f"{adata_dir}/order.txt", header=None, index_col=0).index
    #     adata_dict = OrderedDict()
    #     for k in orders:
    #         adata_path = f"{adata_dir}/{k}.h5ad"
    #         if pathlib.Path(adata_path).exists():
    #             adata_dict[k] = ad.read_h5ad(f"{adata_dir}/{k}.h5ad")
    #     obj.adata_dict = adata_dict
    #     return obj
    #
    # @classmethod
    # def save_transfer_results_to_adata(cls, adata, transfer_results, new_label_suffix="_transfer"):
    #     """Save transfer results to adata."""
    #     for key, df in transfer_results.items():
    #         adata.obs[key + new_label_suffix] = adata.obs[key].copy()
    #         adata.obs.loc[df.index, key + new_label_suffix] = df.idxmax(axis=1).values
    #    return



def run_seurat_integration(
    adata: ad.AnnData,
    batch_key: str,
    key_hvg: str = "highly_variable",
    use_rep: str = "X_pca",
    key_corrected: str = "X",
    method: Literal["cca", "pca", "lsi", "lsi-cca", "rpca", "rlsi"] = "pca",
    n_components: int = 50,
    random_state: int = 0,
    n_jobs: int = -1,
) -> None:
    """Run Seurat Integration methods.

     Code adapted from `Hanqing L., et al. Nature (2021) <https://www.nature.com/articles/s41586-020-03182-8>`_

    :param adata: Annotated data matrix.
    :param batch_key: Key in `adata.obs` with batch information.
    :param key_hvg: Key in `adata.var` with boolean indicating if a feature is highly variable or not.
    :param use_rep: Representation to use to compute within batch KNN to find the anchors.
    :param key_corrected: If set to `X` the expression values will be corrected, otherwise a key in `adata.obsm` needs to be set.
    :param method: Method available in Seurat Integration.
    :param n_components: Number of components to consider.
    :param random_state: Random seed.
    :param n_jobs: Number of threads to use.
    :return: Returns None. The corrected matrix will be saved in `adata.obsm`.

    """
    logger.info("This method is currently experimental")

    assert key_hvg in adata.var.columns, f"{key_hvg} not in adata.var"

    hvg = adata[:, adata.var[key_hvg]].copy()
    batches = hvg.obs[batch_key].unique()
    adata_list = [hvg[hvg.obs[batch_key] == batch].copy() for batch in batches]

    integrator = SeuratIntegration(random_state=random_state, n_jobs=n_jobs)
    integrator.find_anchor(adata_list, key_local=use_rep, dim_red=method, n_components=n_components)
    corrected = integrator.integrate(key_correct=key_corrected)
    corrected = np.concatenate(corrected)
    adata.obsm["X_cca"] = corrected
    return None


