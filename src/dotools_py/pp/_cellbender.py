import os
import subprocess

import anndata as ad
import numpy as np

from dotools_py import logger
from dotools_py.utils import convert_path, get_paths_utils


def _run_barcoderanks(adata: ad.AnnData) -> tuple[int, int]:
    """Run BarcodeRanks from DropletUtils to estimate the lower and upper bound of true cells.

    :param adata: annotated dt matrix of shape with `.X` containing raw counts.
    :return: lower and upper bound
    """
    import rpy2.robjects as robjects
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr

    dropletutils = importr("DropletUtils")

    x_py = adata.X.T
    with localconverter(robjects.default_converter + numpy2ri.converter):
        # Extract CSC components
        x_r = robjects.FloatVector(x_py.data)
        i_r = robjects.IntVector(x_py.indices)
        p_r = robjects.IntVector(x_py.indptr)
        dim_r = robjects.IntVector(x_py.shape)

    r_dgcmatrix = robjects.r["new"]("dgCMatrix", x=x_r, i=i_r, p=p_r, Dim=dim_r)
    result = dropletutils.barcodeRanks(r_dgcmatrix)
    metadata = result.do_slot("metadata")
    knee = metadata.rx2("knee")[0]
    inflection = metadata.rx2("inflection")[0]
    counts = np.array(adata.X.sum(axis=1)).ravel()
    total_cells = len(np.where(counts > inflection)[0])
    expected_cells = len(np.where(counts > knee)[0])
    return expected_cells, total_cells


def run_cellbender(
    cellranger_path: str,
    output_path: str,
    samplenames: list | None = None,
    cuda: bool = True,
    cpu_threads: int = 15,
    epochs: int = 150,
    lr: float = 0.00001,
    estimator_multiple_cpu: bool = False,
    log: bool = True,
    conda_path: str | None = None,
    run_dropletutils: bool = False,
) -> None:
    """Run cellbender to remove ambient RNA.

    Remove ambient RNA using `Cellbender <https://cellbender.readthedocs.io/en/latest/>`_. Assumes that the FASTQ files
    have been mapped with `CellRanger <https://www.10xgenomics.com/support/software/cell-ranger/latest>`_.

    .. warning::
        It is recommended to have access to GPU when running cellbender. Running CellBender on
        CPU might lead to high running time.

    :param cellranger_path: path to folder containing subfolders for each sample.
    :param output_path: output folder to save the H5 files with the corrected expression matrix.
    :param samplenames: list with the name of the folders in `cellranger_path`. If not set, it will be infered.
    :param cuda: set to True to use GPU for the training.
    :param cpu_threads: number of CPUs to use for training.
    :param epochs: number of epochs to train for. The default number is 150, higher number might lead to overfitting.
    :param lr: learning rate.
    :param estimator_multiple_cpu: use multiple CPUs for the generation of results. It is not recommended for big
                                  datasets.
    :param log: generate a log file with the stdout from running CellBender.
    :param conda_path: path to the conda environment with cellbender installed. If not provided, a conda environment
                       will be created in `~/.venv/cellbender`.
    :param run_dropletutils: run DropletUtils to estimate the expected number of cells and total number of droplets to
                             use as a prior for cellbender.
    :return: H5 files with the corrected expression matrix will be saved in the output folder

    Example
    --------
    >>> import dotools_py as do
    >>> in_path = "/path/to/cellranger"
    >>> out_path = "/path/to/output"
    >>> do.pp.run_cellbender(in_path, out_path)
    """
    import scanpy as sc

    # Check-Ups and Information
    samplenames = [samplenames] if isinstance(samplenames, str) else samplenames
    assert os.path.exists(cellranger_path), f"{cellranger_path} does not exist"
    assert os.path.exists(output_path), f"{output_path} does not exist"

    bash_script = get_paths_utils("_run_CellBender.sh")

    if estimator_multiple_cpu:
        logger.info("Estimator_multiple_cpu is set to True, this is not recommended for big datasets >20-30k cells")
    if epochs > 150:
        logger.info(f"Training {epochs} epochs. More than 150 epochs might lead to overfitting")
    if not cuda:
        logger.info("Training without GPU might lead to increase running time")

    # Set-Up - Check that CellBender is available
    conda_path = (
        convert_path(os.path.expanduser("~") + "/.venv/cellbender")
        if conda_path is None
        else os.path.expanduser(conda_path)
    )
    command = ["conda", "create", "-y", "-p", conda_path, "python=3.7"]
    if not os.path.exists(conda_path):
        logger.info("Path to conda env with cellbender not provided, installing cellbender...")
        logger.info("This will take a few minutes")
        try:
            subprocess.run(
                command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )  # Quiet installation
            command = ["conda", "run", "-p", conda_path, "pip", "install", "cellbender", "lxml-html-clean"]
            subprocess.run(
                command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )  # Quiet installation
            logger.info("Environment created")
            logger.info(f"Conda environment create in {conda_path}")
        except subprocess.CalledProcessError as err:
            raise Exception("Error installing cellbender, provide a valid conda environment") from err
    else:
        logger.info(f"Conda environment with CellBender available using ({conda_path})")

    # Run CellBender Sequentially
    cellranger_path = convert_path(cellranger_path)
    if samplenames is None:
        samples = [d for d in os.listdir(cellranger_path) if os.path.isdir(cellranger_path / d)]
    else:
        samples = samplenames

    logger.info(f"Running cellbender for {len(samples)} samples")
    expected_cells, total_droplets = None, None
    for batch in samples:
        # Run one by one but sequentially
        # Estimate the number of cells to be used as upper and lower bound
        if run_dropletutils:
            file = next((cellranger_path / batch / "outs").glob("*raw_feature_bc_matrix.h5"))
            tdata = sc.read_10x_h5(file)
            expected_cells, total_droplets = _run_barcoderanks(tdata)  # Run with rpy2; gives a good estimate

        command = [
            "conda",
            "run",
            "-p",
            str(conda_path),
            "bash",
            str(bash_script),
            "-i",
            batch,
            "-o",
            str(output_path),
            "--cellRanger-output",
            str(cellranger_path),
            "--cpu-threads",
            str(cpu_threads),
            "--epochs",
            str(epochs),
            "--lr",
            str(lr),
        ]
        command += ["--expected-cells", expected_cells] if expected_cells is not None else []
        command += ["--total-droplets", total_droplets] if total_droplets is not None else []
        command += ["--cuda"] if cuda else []
        command += ["--log"] if log else []
        command += ["--estimator_multiple_cpu"] if estimator_multiple_cpu else []

        try:
            logger.info(f"Running Cellbender for {batch}, might take a while")
            subprocess.run(command, check=True, cwd=output_path)
        except subprocess.CalledProcessError as e:
            logger.info(f"Error running CellBender in conda environment: {e}")

    logger.info("Finished running cellbender")
    return None
