import tqdm
import subprocess

import anndata as ad

from dotools_py import logger
from dotools_py._custom_class import PathLike
from dotools_py._utils import convert_path

HERE = convert_path(__file__).parent


def example_10x(path: PathLike = "/tmp/dotools_datasets/") -> None:
    """Download 10X datasets.

    Downloads a dataset of PBMC from healty and malignant B cells. Two H5 files will be downloaded
    (`raw_feature_bc_matrix.h5`) and (`filtered_feature_bc_matrix.h5`) for each condition (healthy and disease). They
    will be saved following the structure output from CellRanger (e.g., `healhty/outs/*.h5` or `disease/outs/*.h5`).

    :param path: Path where the data is downloaded. Two subfolders will be created.
    :return: Returns `None`. H5 files are saved under the provided path. For each condition a subfolder will be created
            in the provided path.

    Example
    -------
    >>> import dotools_py as do
    >>> do.dt.example_10x("/tmp/dotools_datasets/")
    2025-10-22 13:29:52,730 - Downloading data to /tmp/dotools_datasets/
    Downloading healthy filtered: 100%|██████████| 20.8M/20.8M [00:00<00:00, 97.5MiB/s]
    Downloading healthy raw: 100%|██████████| 147M/147M [00:01<00:00, 88.1MiB/s]
    Downloading disease filtered: 100%|██████████| 18.7M/18.7M [00:00<00:00, 104MiB/s]
    Downloading disease raw: 100%|██████████| 144M/144M [00:01<00:00, 85.1MiB/s]
    >>> adata = do.io.read_10x_h5("/tmp/dotools_datasets/healthy/outs/filtered_feature_bc_matrix.h5")
    >>> adata
    AnnData object with n_obs × n_vars = 7865 × 33538
    var: 'gene_ids', 'feature_types', 'genome', 'pattern', 'read', 'sequence'

    """
    import requests

    logger.info(f"Downloading data to {path}")
    path = convert_path(path)
    path.mkdir(parents=True, exist_ok=True)
    healthy_path = path / "healthy" / "outs"
    healthy_path.mkdir(parents=True, exist_ok=True)
    disease_path = path / "disease" / "outs"
    disease_path.mkdir(parents=True, exist_ok=True)

    healthy_link1 = ("https://cf.10xgenomics.com/samples/cell-exp/"
                     "3.0.0/pbmc_10k_protein_v3/pbmc_10k_protein_v3_filtered_feature_bc_matrix.h5")
    healthy_link2 = ("https://cf.10xgenomics.com/samples/cell-exp/"
                     "3.0.0/pbmc_10k_protein_v3/pbmc_10k_protein_v3_raw_feature_bc_matrix.h5")
    disease_link1 = ("https://cf.10xgenomics.com/samples/cell-exp/"
                     "3.0.0/malt_10k_protein_v3/malt_10k_protein_v3_filtered_feature_bc_matrix.h5")
    disease_link2 = ("https://cf.10xgenomics.com/samples/cell-exp/"
                     "3.0.0/malt_10k_protein_v3/malt_10k_protein_v3_raw_feature_bc_matrix.h5")
    for name, link in [
        ("healthy filtered", healthy_link1),
        ("healthy raw", healthy_link2),
        ("disease filtered", disease_link1),
        ("disease raw", disease_link2),
    ]:
        filename = link.split("10k_protein_v3_")[-1]
        response = requests.get(link, stream=True)  # Download in chunks
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        current_path = healthy_path if "healthy" in name else disease_path
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
            for data in response.iter_content(block_size):
                file.write(data)
                bar.update(len(data))
    return None


def example_10x_processed() -> ad.AnnData:
    """Load example datasets from 10x processed.

    Loads a reduced version of the example datasets from healthy and malignant B cells from 10x used in the
    tutorial of the package.

    :return: Returns an AnnData object processed with 700 cells and 1851 genes.

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


def example_visium(path: PathLike = "tmp/dotools_datasets/") -> None:
    """Download 10X Visium datasets.

    Downloads a dataset of the human heart fresh frozen tissue including H&E image.

    :param path: Path where the data is downloaded.
    :return: Returns `None`.

    Example
    -------
    >>> import dotools_py as do
    >>> do.dt.example_visium("/tmp/dotools_datasets/")
    2026-03-03 11:53:51,993 - Downloading data to /tmp/dotools_datasets/
    Downloading Molecule info: 100%|██████████| 142M/142M [00:07<00:00, 20.8MiB/s]
    Downloading filtered matrix: 100%|██████████| 11.6M/11.6M [00:01<00:00, 7.77MiB/s]
    Downloading raw matrix: 100%|██████████| 13.4M/13.4M [00:01<00:00, 11.0MiB/s]
    Downloading spatial: 100%|██████████| 8.78M/8.78M [00:01<00:00, 8.02MiB/s]
    Downloading metrics: 100%|██████████| 945/945 [00:00<00:00, 4.36MiB/s]
    Downloading web summary: 7.29MiB [00:01, 6.73MiB/s]
    >>> adata = do.io.read_visium("/tmp/dotools_datasets")
    >>> adata
    reading /tmp/dotools_datasets/filtered_feature_bc_matrix.h5
    (0:00:00)
    AnnData object with n_obs × n_vars = 4247 × 36601
    obs: 'in_tissue', 'array_row', 'array_col'
    var: 'gene_ids', 'feature_types', 'genome'
    uns: 'spatial'
    obsm: 'spatial'

    """
    import requests

    logger.info(f"Downloading data to {path}")
    path = convert_path(path)
    path.mkdir(parents=True, exist_ok=True)

    visium_path = path
    visium_path.mkdir(parents=True, exist_ok=True)

    link1 = "https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/V1_Human_Heart/V1_Human_Heart_molecule_info.h5"
    link2 = (
        "https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/V1_Human_Heart/V1_Human_Heart_filtered_feature_bc_matrix.h5"
    )
    link3 = "https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/V1_Human_Heart/V1_Human_Heart_raw_feature_bc_matrix.h5"
    link4 = "https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/V1_Human_Heart/V1_Human_Heart_spatial.tar.gz"
    link5 = "https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/V1_Human_Heart/V1_Human_Heart_metrics_summary.csv"
    link6 = "https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/V1_Human_Heart/V1_Human_Heart_web_summary.html"

    for name, link in [
        ("Molecule info", link1),
        ("filtered matrix", link2),
        ("raw matrix", link3),
        ("spatial", link4),
        ("metrics", link5),
        ("web summary", link6),
    ]:
        filename = link.split("V1_Human_Heart_")[-1]
        response = requests.get(link, stream=True)  # Download in chunks
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        with (
            open(visium_path / filename, "wb") as file,
            tqdm.tqdm(
                desc=f"Downloading {name}",
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar,
        ):
            for data in response.iter_content(block_size):
                file.write(data)
                bar.update(len(data))
    try:
        command = [
            f"tar -xf {visium_path / 'spatial.tar.gz'}"
        ]
        _ = subprocess.run(command, shell=True, check=True, cwd=visium_path)
    except Exception as e:
        logger.warn(f"Could not uncompressed {visium_path / 'spatial.tar.gz'}, please do it manually")
    return None


def example_visium_processed()-> ad.AnnData:
    """Load example datasets from Visium processed.

    Loads a reduced version of the example datasets from Visium used in the tutorial of the package.

    :return: Returns an AnnData object processed with 1046 cells and 1000 genes.

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
