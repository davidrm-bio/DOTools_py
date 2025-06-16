import requests
from tqdm import tqdm
from pathlib import Path
from typing import Union
from dotools_py import logger
from dotools_py.utils import convert_path


def example_10x(
    path: Union[str, Path] = '/tmp/dootools_datasets/'
)->None:
    """Download 10X datasets.

    Download datasets of PBMC from healty and malignant condition. Two H5 files will be downloaded and saved
    following the structure ouput from CellRanger.

    :param path: path to save H5 files.
    :return:
    """
    logger.info(f'Downloading data to {path}')
    path = convert_path(path)
    path.mkdir(parents=True, exist_ok=True)
    healthy_path = path / 'healthy' / 'outs'
    healthy_path.mkdir(parents=True, exist_ok=True)
    disease_path = path / 'disease' / 'outs'
    disease_path.mkdir(parents=True, exist_ok=True)

    healthy_link1 = 'https://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_10k_protein_v3/pbmc_10k_protein_v3_filtered_feature_bc_matrix.h5'
    healthy_link2 = 'https://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_10k_protein_v3/pbmc_10k_protein_v3_raw_feature_bc_matrix.h5'
    disease_link1 = 'https://cf.10xgenomics.com/samples/cell-exp/3.0.0/malt_10k_protein_v3/malt_10k_protein_v3_filtered_feature_bc_matrix.h5'
    disease_link2 = 'https://cf.10xgenomics.com/samples/cell-exp/3.0.0/malt_10k_protein_v3/malt_10k_protein_v3_raw_feature_bc_matrix.h5'
    for name, link in [('healthy filtered', healthy_link1),
                       ('healthy raw', healthy_link2),
                       ('disease filtered', disease_link1),
                       ('disease raw', disease_link2)]:
        filename = link.split('10k_protein_v3_')[-1]
        response = requests.get(link, stream=True)  # Download in chunks
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        current_path = healthy_path if 'healthy' in name else disease_path
        with open(current_path / filename, 'wb') as file, tqdm(
            desc=f"Downloading {name}",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                file.write(data)
                bar.update(len(data))
    return None
