import pandas as pd
import tqdm
import appdirs
from typing import Literal

import anndata as ad

from dotools_py import logger
from dotools_py._custom_class import PathLike, InputError
from dotools_py._utils import convert_path

HERE = convert_path(__file__).parent


class LoadData:
    DEFAULT_PATH = appdirs.user_cache_dir("dotools_datasets")

    def __init__(
        self,
        path: PathLike | None = None,
        technology: Literal["scrna", "visium"] = "scrna"
    ):
        """Initiate the class

        :param path: Absolute path to where the data is saved. The default directory is the Cache directory.
        :param technology: The type of dataset that will be downloaded
        """

        self.technology = technology
        self.paths = self._create_dir(path)
        self._get_links()

    def _get_links(self):
        """Get the links for the data.

        :return: The links and prefix attributes will be initialized
        """
        website = "https://cf.10xgenomics.com/samples/"
        if self.technology == "scrna":
            links = (
                ("healthy filtered",
                 f"{website}cell-exp/3.0.0/pbmc_10k_protein_v3/pbmc_10k_protein_v3_filtered_feature_bc_matrix.h5"),
                ("healthy raw",
                 f"{website}cell-exp/3.0.0/pbmc_10k_protein_v3/pbmc_10k_protein_v3_raw_feature_bc_matrix.h5"),
                ("disease filtered",
                 f"{website}cell-exp/3.0.0/malt_10k_protein_v3/malt_10k_protein_v3_filtered_feature_bc_matrix.h5"),
                ("disease raw",
                 f"{website}cell-exp/3.0.0/malt_10k_protein_v3/malt_10k_protein_v3_raw_feature_bc_matrix.h5")
            )
            prefix = "10k_protein_v3_"
        elif self.technology == "visium":
            links = (
            ("Molecule info", f"{website}spatial-exp/1.1.0/V1_Human_Heart/V1_Human_Heart_molecule_info.h5"),
            ("filtered matrix", f"{website}spatial-exp/1.1.0/V1_Human_Heart/V1_Human_Heart_filtered_feature_bc_matrix.h5"),
            ("raw matrix", f"{website}spatial-exp/1.1.0/V1_Human_Heart/V1_Human_Heart_raw_feature_bc_matrix.h5"),
            ("spatial", f"{website}spatial-exp/1.1.0/V1_Human_Heart/V1_Human_Heart_spatial.tar.gz"),
            ("metrics", f"{website}spatial-exp/1.1.0/V1_Human_Heart/V1_Human_Heart_metrics_summary.csv"),
            ("web summary", f"{website}spatial-exp/1.1.0/V1_Human_Heart/V1_Human_Heart_web_summary.html")
            )
            prefix = "V1_Human_Heart_"
        else:
            raise InputError(f"{self.technology} is not a valid value for technology")

        self.links = links
        self.prefix = prefix
        return  self


    def _create_dir(self, path: PathLike | None = None) -> tuple | PathLike:
        """Create a directory in path to save the data

        :param path: Absolute path where the data is going to be saved. If set to `None` save in the Cache folder.
        :return:  Returns a tuple or a pathlike object
        """
        path = path if path is not None else self.DEFAULT_PATH
        logger.info(f"Downloading data to {path}")
        path = convert_path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.technology == "scrna":
            healthy_path = path / "healthy" / "outs"
            healthy_path.mkdir(parents=True, exist_ok=True)
            disease_path = path / "disease" / "outs"
            disease_path.mkdir(parents=True, exist_ok=True)
            paths = (healthy_path, disease_path)
        elif self.technology == "visium":
            visium_path = path / "visium"
            visium_path.mkdir(parents=True, exist_ok=True)
            paths = visium_path
        else:
            raise InputError(f"{self.technology} is not a valid value for technology")
        return paths


    def download_data(self) -> None:
        """Downloads the data.

        :return: Returns `None`
        """
        import requests
        import subprocess

        for name, link in self.links:
            filename = link.split(self.prefix)[-1]
            response = requests.get(link, stream=True)  # Download in chunks
            total_size = int(response.headers.get("content-length", 0))

            if isinstance(self.paths, tuple):
                current_path = self.paths[0] if "healthy" in name else self.paths[1]
            else:
                current_path = self.paths

            with (
                open(current_path / filename, "wb") as file,
                tqdm.tqdm(
                    desc=f"Downloading {name}",
                    total=total_size,
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar,
            ):
                for data in response.iter_content(1024):
                    file.write(data)
                    bar.update(len(data))

        if self.technology == "visium":
            try:
                command = [
                    f"tar -xf {str(self.paths / 'spatial.tar.gz')}"
                ]
                _ = subprocess.run(command, check=True, cwd=str(self.paths))
            except Exception as e:
                logger.warn(
                    f"Could not uncompressed {self.paths / 'spatial.tar.gz'}, please do it manually\nError: {e}"
                )
        return None


def example_10x(path: PathLike | None = None) -> None:
    """Download scRNA 10x dataset.

    Downloads an example dataset of PBMC from healthy donors and malignant B cells. Two H5 files for each dataset
    will be downloaded (`raw_feature_bc_matrix.h5`) and (`filtered_feature_bc_matrix.h5`) and will be saved
    following the CellRanger output format (e.g., `dataset/outs/*.h5`).

    :param path: Absolute path where the data is saved. If set to `None`, it will be saved to the user cache folder.
    :return: Returns `None`.

    Example
    -------

    >>> import dotools_py as do
    >>> do.dt.example_10x()
    2026-04-17 14:37:49,503 - Downloading data to /Users/david/Library/Caches/dotools_datasets
    Downloading healthy filtered: 100%|██████████| 20.8M/20.8M [00:01<00:00, 14.0MiB/s]
    Downloading healthy raw: 100%|██████████| 147M/147M [00:01<00:00, 117MiB/s]
    Downloading disease filtered: 100%|██████████| 18.7M/18.7M [00:01<00:00, 12.7MiB/s]
    Downloading disease raw: 100%|██████████| 144M/144M [00:06<00:00, 22.9MiB/s]
    >>> adata = do.io.read_10x_h5("/Users/david/Library/Caches/dotools_datasets/healthy/outs/filtered_feature_bc_matrix.h5")
    >>> adata
    AnnData object with n_obs × n_vars = 7865 × 33538
    var: 'gene_ids', 'feature_types', 'genome', 'pattern', 'read', 'sequence'

    """
    loader = LoadData(path=path, technology="scrna")
    loader.download_data()
    return None


def example_10x_processed() -> ad.AnnData:
    """Load example scRNAseq from 10x processed.

    Loads a reduced version of the example datasets from healthy and malignant B cells from 10x used in the
    tutorial of the package.

    :return: Returns an `AnnData` object processed with 700 cells and 1851 genes.

    Example
    -------
    >>> import dotools_py as do
    >>> adata = do.dt.example_10x_processed()
    >>> adata
    AnnData object with n_obs × n_vars = 700 × 1851
    obs: 'batch', 'condition', 'n_genes_by_counts', 'log1p_n_genes_by_counts', 'total_counts', 'log1p_total_counts',
         'total_counts_mt', 'log1p_total_counts_mt', 'pct_counts_mt', 'total_counts_ribo', 'log1p_total_counts_ribo',
         'pct_counts_ribo', 'n_genes', 'n_counts', 'doublet_class', 'doublet_score', 'leiden', 'cell_type', 'autoAnnot',
         'celltypist_conf_score', 'annotation', 'annotation_recluster'
    var: 'mean', 'std', 'highly_variable', 'means', 'dispersions', 'dispersions_norm', 'highly_variable_nbatches',
         'highly_variable_intersection'
    uns: 'annotation_colors', 'annotation_recluster_colors', 'batch_colors', 'hvg', 'leiden', 'leiden_colors', 'log1p',
         'neighbors', 'pca', 'umap'
    obsm: 'X_CCA', 'X_pca', 'X_umap'
    varm: 'PCs'
    layers: 'counts', 'logcounts'
    obsp: 'connectivities', 'distances'

    """
    return ad.read_h5ad(HERE / "example_reduced.h5ad")


def example_visium(path: PathLike | None = None) -> None:
    """ Download a 10x Visium dataset from the heart.

    Downloads a dataset of the human heart. The sample comes from fresh frozen tissue and includes the H&E image.

    :param path: Absolute path where the data is saved. If set to `None`, it will be saved to the user cache folder.
    :return: Returns `None`.

    Examples
    --------
    >>> import dotools_py as do
    >>> do.dt.example_visium()
    2026-04-17 14:45:48,428 - Downloading data to /Users/david/Library/Caches/dotools_datasets
    Downloading Molecule info: 100%|██████████| 142M/142M [00:01<00:00, 118MiB/s]
    Downloading filtered matrix: 100%|██████████| 11.6M/11.6M [00:00<00:00, 112MiB/s]
    Downloading raw matrix: 100%|██████████| 13.4M/13.4M [00:00<00:00, 115MiB/s]
    Downloading spatial: 100%|██████████| 8.78M/8.78M [00:00<00:00, 111MiB/s]
    Downloading metrics: 100%|██████████| 945/945 [00:00<00:00, 19.3MiB/s]
    Downloading web summary: 7.29MiB [00:00, 13.7MiB/s]
    >>> adata = do.io.read_visium("/Users/david/Library/Caches/dotools_datasets/visium")
    >>> adata
    AnnData object with n_obs × n_vars = 4247 × 36601
    obs: 'in_tissue', 'array_row', 'array_col'
    var: 'gene_ids', 'feature_types', 'genome'
    uns: 'spatial'
    obsm: 'spatial'

    """
    loader = LoadData(path=path, technology="visium")
    loader.download_data()
    return None


def example_visium_processed()-> ad.AnnData:
    """Load example Visium datasets processed.

    Loads a reduced version of the example datasets from Visium used in the tutorial of the package.

    :return: Returns an `AnnData` object processed with 1046 cells and 1000 genes.

    Example
    -------
    >>> import dotools_py as do
    >>> adata = do.dt.example_visium_processed()
    >>> adata
    AnnData object with n_obs × n_vars = 1046 × 1000
    obs: 'in_tissue', 'array_row', 'array_col', 'batch', 'condition', 'tissue', 'n_genes_by_counts',
         'log1p_n_genes_by_counts', 'total_counts', 'log1p_total_counts', 'total_counts_mt', 'log1p_total_counts_mt',
         'pct_counts_mt', 'total_counts_ribo', 'log1p_total_counts_ribo', 'pct_counts_ribo', 'total_counts_hb',
         'log1p_total_counts_hb', 'pct_counts_hb', 'n_genes', 'n_counts', 'leiden'
    var: 'highly_variable', 'means', 'dispersions', 'dispersions_norm', 'highly_variable_nbatches',
         'highly_variable_intersection'
    uns: 'hvg', 'leiden_colors', 'log1p', 'neighbors', 'spatial', 'spatial_neighbors', 'umap'
    obsm: 'X_pca', 'X_umap', 'spatial'
    layers: 'counts', 'logcounts'
    obsp: 'connectivities', 'distances', 'spatial_connectivities', 'spatial_distances'

    """
    return ad.read_h5ad(HERE / "example_visium_reduced.h5ad")


def example_ora() -> pd.DataFrame:
    """Load example table from overrepresentation analysis.

    To generate the table the following code was used:

    .. code-block:: python

        import dotools_py as do
        adata = do.dt.example_10x_processed()
        do.tl.rank_genes_groups(adata, 'condition')
        table = do.get.dge_results(adata)
        table = table[table.group == 'disease']
        table_go = do.tl.go_analysis(table, 'GeneName', 'padj', 'log2fc', specie='Human', go_catgs = ['GO_Molecular_Function_2023', 'GO_Cellular_Component_2023', 'GO_Biological_Process_2023'])


    Returns
    -------
    Returns a pandas DataFrame

    Example
    -------
    >>> import dotools_py as do
    >>> df = do.dt.example_ora()
    >>> print(f'Shape: {df.shape}', df.head(), sep='\n')
    Shape: (2704, 11)
                         Gene_set  ...     state
    0  GO_Molecular_Function_2023  ...  enriched
    1  GO_Molecular_Function_2023  ...  enriched
    2  GO_Molecular_Function_2023  ...  enriched
    3  GO_Molecular_Function_2023  ...  enriched
    4  GO_Molecular_Function_2023  ...  enriched
    [5 rows x 11 columns]

    """
    return pd.read_parquet(HERE / "TableGO_Example.parquet")
