import os
import subprocess

from prelude_py import ad, np

from dotools_py import logger
from dotools_py._utils import convert_path, get_paths_utils
from dotools_py._custom_class import PathLike
from dotools_py.io._objects import read_10x_h5

# TODO implement scAR for ambient RNA detection

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
    *,
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
    >>> do.pp.run_cellbender(cellranger_path=in_path, output_path=out_path)
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




def run_ambidose(
    adata: ad.AnnData,
    *,
    raw_matrix: PathLike | ad.AnnData,
    cluster_key: str | None = None,
    layer: str | None = None,
    n_jobs: int | None = None,
    report_path: PathLike | None = None,
    report_filename: str = "Report.html",
    **kwargs
):
    """Remove ambient RNA using `Ambidose <https://ambidose.readthedocs.io/en/latest/index.html>`_.

    A filtered AnnData can be provided and should contain raw counts in `.X`, otherwise the argument `layer` can be use
    to provide raw counts. The feature_names are expected to be available in `adata.var_names`.
    It is preferable to provide specify the `cluster_key` to estimate cluster-level expression.

    :param adata: Filtered annotated data matrix.
    :param raw_matrix: Path to the directory with `raw_feature_bc_matrix.h5` or an AnnData object of the `raw_feature_bc_matrix.h5`
    :param cluster_key: Column in the filtered adata.obs with clusters. This is used to estimate cluster-level expression.
    :param layer: Layer in the filtered adata.obs to be used. It should contain raw counts.
    :param n_jobs: Number of cores to use for the estimation of ambient RNA.
    :param report_path: Path to save the report.
    :param report_filename: Name of the HTML report file.
    :param kwargs: Additional arguments pass to `ambidose.denoise <https://ambidose.readthedocs.io/en/latest/api/index.html#ambidose.denoise>`_
    :return: Returns None. A new layer will be set with the denoised expression matrix and additional metadata will be saved in adata.obs and adata.uns.

    Example
    -------
    >>> import dotools_py as do
    >>> adata = do.dt.example_10x_processed()
    >>> adata = do.get.subset(adata, obs_key='condition', obs_groups='healthy')
    >>> raw_path = '/tmp/PublicData10x/healthy/outs/'
    >>> do.pp.run_ambidose(adata, raw_matrix=raw_path, cluster_key='annotation', layer='counts')
    >>> adata
    AnnData object with n_obs × n_vars = 520 × 1851
        obs: 'batch', 'condition', 'n_genes_by_counts', 'log1p_n_genes_by_counts', 'total_counts', 'log1p_total_counts',
             'total_counts_mt', 'log1p_total_counts_mt', 'pct_counts_mt', 'total_counts_ribo', 'log1p_total_counts_ribo',
             'pct_counts_ribo', 'n_genes', 'n_counts', 'doublet_class', 'doublet_score', 'leiden', 'cell_type',
             'autoAnnot', 'celltypist_conf_score', 'annotation', 'annotation_recluster',
             'ambidose_droplet', 'ambidose_d', 'ambidose_rho', 'ambidose_removed_umi', 'ambidose_dose_execution_ratio',
             'ambidose_removed_fraction', 'ambidose_rho_trust'
        var: 'mean', 'std', 'highly_variable', 'means', 'dispersions', 'dispersions_norm', 'highly_variable_nbatches',
             'highly_variable_intersection'
        uns: 'annotation_colors', 'annotation_recluster_colors', 'batch_colors', 'hvg', 'leiden', 'leiden_colors',
             'log1p', 'neighbors', 'pca', 'umap', 'ambidose'
        obsm: 'X_CCA', 'X_pca', 'X_umap'
        varm: 'PCs'
        layers: 'counts', 'logcounts', 'ambidose_denoised'
        obsp: 'connectivities', 'distances'

    """

    try:
        import ambidose
    except ImportError as e:
        raise ImportError("Could not find `ambidose`. Install it using `pip install ambidose`") from e


    if not isinstance(raw_matrix, ad.AnnData):
        logger.info("Reading the raw_matrix")
        adata_raw = read_10x_h5(str(convert_path(raw_matrix) / "raw_feature_bc_matrix.h5"))
    else:
        adata_raw = raw_matrix

    adata_raw.var_names_make_unique()
    adata_raw.var["gene_ids"] = adata_raw.var_names

    if report_path is not None:
        report_path = convert_path(report_path)
        report_path.mkdir(exist_ok=True)
        report_path = report_path / report_filename


    # Check the barcodes matched
    raw_bcs = dict(zip(adata_raw.obs_names.to_numpy(), range(adata_raw.n_obs)))
    filt_bcs = dict(zip(adata.obs_names.to_numpy(), range(adata.n_obs)))

    counter = [True if bc in raw_bcs else False for bc in filt_bcs]
    if not all(counter):
        logger.warn(f"There {len(counter) - np.sum(counter)} barcodes in adata not found in adata.raw")
        logger.warn("Trying to remove suffix")
        # Barcodes look like letters-1-subset
        new_bcs = adata.obs_names.str.split("-1").str[0] + "-1"
        adata.obs["old_bcs"] = adata.obs_names
        adata.obs_names = new_bcs
        filt_bcs = dict(zip(adata.obs_names.to_numpy(), range(adata.n_obs)))
        counter = [True if bc in raw_bcs else False for bc in filt_bcs]
        if not all(counter):
            raise ValueError("Could not run match the barcodes in adata.obs_names with the barcodes in the raw matrix")

    if layer is not None:
        adata.X = adata.layers[layer].copy()

    adata_denoised = ambidose.denoise(
        adata,
        raw=adata_raw,
        type_key=cluster_key,
        sample_key=None,
        n_jobs=n_jobs,
        layer=None,
        report=report_path,
        **kwargs
    )

    adata.layers["ambidose_denoised"] = adata_denoised.layers["ambidose_denoised"].copy()
    ambidose_meta = [col for col in adata_denoised.obs.columns if col.startswith("ambidose")]

    adata.obs[ambidose_meta] = adata_denoised.obs[ambidose_meta]
    adata.uns["ambidose"] = adata_denoised.uns["ambidose"]
    if "old_bcs" in adata.obs.columns:
        adata.obs_names = adata.obs["old_bcs"].astype(str).values
        del adata.obs["old_bcs"]
    del adata_denoised
    return None












