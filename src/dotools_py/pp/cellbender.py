
import os
from pathlib import Path
import subprocess
import scanpy as sc
import pandas as pd
import numpy as np

import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter

import logger
from utils import convert_path, get_paths_utils

def _run_barcoderanks(adata):
    """ Run BarcodeRanks from DropletUtils to estimate the lower and upper bound of cells to use to estimate cell
    probabilities.

    :param adata: anndata object with raw counts in X. Should be the raw matrix from cellranger
    :return: lower and upper bound
    """
    numpy2ri.activate()
    pandas2ri.activate()
    dropletutils = importr("DropletUtils")

    x_py = adata.X.T
    with localconverter(robjects.default_converter + numpy2ri.converter):
        # Extract CSC components
        x_r = robjects.FloatVector(x_py.data)
        i_r = robjects.IntVector(x_py.indices )
        p_r = robjects.IntVector(x_py.indptr)
        dim_r = robjects.IntVector(x_py.shape)

    r_dgcmatrix = robjects.r['new']("dgCMatrix", x = x_r, i = i_r, p = p_r, Dim = dim_r)
    result = dropletutils.barcodeRanks(r_dgcmatrix)
    metadata = result.do_slot("metadata")
    knee = metadata.rx2("knee")[0]
    inflection = metadata.rx2("inflection")[0]
    counts = np.array(adata.X.sum(axis=1)).ravel()
    total_cells = len(np.where(counts > inflection)[0])
    expected_cells = len(np.where(counts > knee)[0])
    return expected_cells, total_cells


def run_cellbender(cellranger_path: str,
                   output_path: str,
                   samplenames: list = None,
                   cuda: bool = True,
                   cpu_threads: int = 15,
                   epochs: int = 150,
                   lr: float = 0.00001,
                   estimator_multiple_cpu: bool = False,
                   log: bool = True,
                   conda_path: str = None,
                   ) -> None:
    """Run cellbender to remove ambient RNA.

    :param cellranger_path:
    :param output_path:
    :param samplenames:
    :param cuda:
    :param cpu_threads:
    :param epochs:
    :param lr:
    :param estimator_multiple_cpu:
    :param log:
    :param conda_path:
    :return:
    """

    # Check-Ups and Information
    samplenames = [samplenames] if isinstance(samplenames, str) else samplenames
    assert os.path.exists(cellranger_path), f'{cellranger_path} does not exist'
    assert os.path.exists(output_path), f'{cellranger_path} does not exist'

    bash_script = get_paths_utils('run_CellBender.sh')

    if estimator_multiple_cpu:
        logger.info('Estimator_multiple_cpu is set to True, this is not recommended for big datasets >20-30k cells')
    if epochs > 150:
        logger.info(f'Training {epochs} epochs. More than 150 epochs might lead to overfitting')
    if not cuda:
        logger.info('Training without GPU might lead to increase running time')

    # Set-Up - Check that CellBender is available
    conda_path = convert_path(
        os.path.expanduser('~') + '/.venv/cellbender') if conda_path is None else os.path.expanduser(conda_path)
    command = ['conda', 'create', '-y', "-p", conda_path, f'python=3.7'] + ['cellbender']
    if not os.path.exists(conda_path):
        logger.info('Path to conda env with cellbender not provided, installing cellbender...')
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)  # Quiet installation
            logger.info('Environment created')
        except subprocess.CalledProcessError as e:
            raise Exception('Error installing cellbender, provide a valid conda environment')
    else:
        logger.info(f'Conda environment with CellBender available using ({conda_path})')

    # Run CellBender Sequentially
    cellranger_path = convert_path(cellranger_path)
    if samplenames is None:
        samples = [d for d in os.listdir(cellranger_path) if os.path.isdir(cellranger_path / d)]
    else:
        samples = samplenames

    logger.info(f'Running cellbender for {len(samples)} samples')
    for batch in samples:
        # Run one by one but sequentially
        # Estimate the number of cells to be used as upper and lower bound
        tdata = sc.read_10x_h5(cellranger_path / batch / 'outs' / 'raw_feature_bc_matrix.h5')
        expected_cells, total_droplets = _run_barcoderanks(tdata)  # Run with rpy2; gives a good estimate

        command = ['conda', 'run', '-p', conda_path,
                   'bash', bash_script,
                   '-i', batch, '-o', output_path,
                   '--cellRanger-output', cellranger_path,
                   '--cpu-threads', cpu_threads,
                   '--epochs', epochs, '--lr', lr,
                   '--expected-cells', expected_cells,
                   '--total-droplets', total_droplets]

        command += ['--cuda'] if cuda else None
        command += ['--log'] if log else None
        command += ['--estimator_multiple_cpu'] if estimator_multiple_cpu else None

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            logger.info(f'Error running CellBender in conda environment: {e}')

    logger.info('Finished running cellbender')
    return None
