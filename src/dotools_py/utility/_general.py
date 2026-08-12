import os
import sys
import tomlkit
import subprocess
from pathlib import Path
import platform
from typing import Literal
from rich.live import Live
import functools
from timeit import default_timer as timer
import datetime
from rich.console import Console


from prelude_py import ad, pd
from dotools_py._custom_class import PathLike
from  dotools_py._utils import convert_path

HERE = Path(__file__).parent


def free_memory(*, cuda: bool = False) -> None:
    """Garbage collector.
    :param cuda: If set to `True` clean the cache of cuda.
    :return:
    """
    import ctypes
    import gc

    gc.collect()

    system = platform.system()

    if system == "Linux":
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    else:
        pass
    if cuda:
        import torch
        torch.cuda.memory.empty_cache()
    return None


def transfer_labels(
    *,
    adata_original: ad.AnnData,
    adata_subset: ad.AnnData,
    original_key: str,
    subset_key: str,
    original_labels: list,
    copy: bool = False,
) -> ad.AnnData | None:
    """Transfer annotation from a subset AnnData to an AnnData.

    :param adata_original: original AnnData.
    :param adata_subset: subsetted AnnData.
    :param original_key: obs column name in the original AnnData where new labels are added.
    :param subset_key: obs column name in the subsetted AnnData with the new labels.
    :param original_labels: list of labels in `original_key` to replace.
    :param copy: if set to True, returns the updated anndata
    :return: If `copy` is set to `True`, returns the original AnnData with the updated labels, otherwise returns `None`.
             The  original_labels in original_key will be updated with the labels in subset_key.
    """
    if copy:
        adata_original = adata_original.copy()
        adata_subset = adata_subset.copy()
    assert adata_subset.n_obs < adata_original.n_obs, "adata_subset is not a subset of adata_original"

    labels_original = [original_labels] if isinstance(original_labels, str) else original_labels
    adata_original.obs[original_key] = adata_original.obs[original_key].astype(str)
    adata_original.obs[original_key] = adata_original.obs[original_key].where(
        ~adata_original.obs[original_key].isin(labels_original),
        adata_original.obs.index.map(adata_subset.obs[subset_key]),
    )

    if copy:
        return adata_original
    else:
        return None



def add_gene_metadata(
    *,
    data: ad.AnnData | pd.DataFrame,
    gene_key: str,
    species: Literal["mouse", "human"] = "mouse",
    add_gene_id: bool = False,
) -> ad.AnnData | pd.DataFrame:
    """Add gene metadata to AnnData or DataFrame.

    Add gene metadata obtained from the GTF or Uniprot-database. This information includes,
    the gene biotype (e.g., protein-coding, lncRNA, etc.); the ENSEMBL gene ID and the subcellular location.

    :param data:  Annotated data matrix or pandas dataframe with for example results from differential gene expression analysis.
    :param gene_key: name of the key with gene names. If an AnnData is provided the .var name column name with gene names. If the gene names are in
                     `var_names`, specify `var_names`.
    :param species: the input species.
    :param add_gene_id: Add gene id (ENSEMBL ID) information.
    :return:  Returns a dataframe or AnnData object. Three new columns will be set: `biotype`, `locations` and `gene_id`.

    Examples
    --------

    >>> import dotools_py as do
    >>> # AnnData Input
    >>> adata = do.dt.example_10x_processed()
    >>> adata = do.utility.add_gene_metadata(data=adata, gene_key="var_names", species="human")
    >>> adata.var[["biotype", "gene_id", "locations"]].head(5)
                           biotype          gene_id                locations
    ATP2A1-AS1          lncRNA  ENSG00000260442  Unreview status Uniprot
    STK17A      protein_coding  ENSG00000164543                  nucleus
    C19orf18    protein_coding  ENSG00000177025                 membrane
    TPP2        protein_coding  ENSG00000134900        nucleus,cytoplasm
    MFSD1       protein_coding  ENSG00000118855       membrane,cytoplasm
    >>>
    >>> # Dataframe Input
    >>> df = pd.DataFrame(["Acta2", "Tagln", "Ptprc", "Vcam1"], columns=["genes"])
    >>> df = add_gene_metadata(df, "genes")
    >>> df.head()
           genes         biotype          locations             gene_id
    0  Acta2  protein_coding          cytoplasm  ENSMUSG00000035783
    1  Tagln  protein_coding          cytoplasm  ENSMUSG00000032085
    2  Ptprc  protein_coding           membrane  ENSMUSG00000026395
    3  Vcam1  protein_coding  secreted,membrane  ENSMUSG00000027962


    """
    import gzip
    import pickle

    data_copy = data.copy()  # Changes will not be inplace

    assert species in ["mouse", "human"], "Not a valid species: use mouse or human"
    file = "MusMusculus_GeneMetadata.pickle.gz" if species == "mouse" else "HomoSapiens_GeneMetadata.pickle.gz"
    with gzip.open(os.path.join(HERE, file), "rb") as pickle_file:
        database = pickle.load(pickle_file)
    biotype = "gene_type" if species == "mouse" else "gene_biotype"

    if isinstance(data, pd.DataFrame):
        genes = data_copy[gene_key].tolist()
        data_copy["biotype"] = [database[g][biotype] if g in database else "NaN" for g in genes]
        data_copy["locations"] = [",".join(database[g]["locations"]) if g in database else "NaN" for g in genes]
        if add_gene_id:
            data_copy["gene_id"] = [database[g]["gene_id"] if g in database else "NaN" for g in genes]
    elif isinstance(data_copy, ad.AnnData):
        genes = list(data_copy.var_names) if gene_key == "var_names" else data_copy.var[gene_key].tolist()
        data_copy.var["biotype"] = [database[g][biotype] if g in database else "NaN" for g in genes]
        data_copy.var["locations"] = [",".join(database[g]["locations"]) if g in database else "NaN" for g in genes]
        if add_gene_id:
            data_copy.var["gene_id"] = [database[g]["gene_id"] if g in database else "NaN" for g in genes]
    else:
        raise Exception("Not a valid input, provide a DataFrame or AnnData")

    return data_copy



def create_report(
    log_file: str | Path,
) -> None:
    """Create a report file.

    This function takes a log_file that should have been set at the beginning of the session with
    `dotools_py.settings.set.set_kernel_logger` and add information regarding the session such as
    the machine characteristics and the version of the packages.

    :param log_file: Path to the log file
    :return: Returns None. The log file is updated with session information.

    Examples
    --------
    >>> import dotools_py as do
    >>> do.settings.set_kernel_logger('./History.log', overwrite=True)
    >>> adata = do.dt.example_10x_processed()
    >>> adata
    >>> do.utility.create_report("./History.log")
    >>> print(open("History.log").read())
    [CODE 2026-01-22 13:59:28.904757]
    >>> adata = do.dt.example_10x_processed()
    [CODE 2026-01-22 13:59:29.617186]
    >>> adata
    [OUTPUT 2026-01-22 13:59:29.619246]
    AnnData object with n_obs × n_vars = 700 × 1851
        obs: 'batch', 'condition', 'n_genes_by_counts', 'log1p_n_genes_by_counts', 'total_counts', 'log1p_total_counts', 'total_counts_mt', 'log1p_total_counts_mt', 'pct_counts_mt', 'total_counts_ribo', 'log1p_total_counts_ribo', 'pct_counts_ribo', 'n_genes', 'n_counts', 'doublet_class', 'doublet_score', 'leiden', 'cell_type', 'autoAnnot', 'celltypist_conf_score', 'annotation', 'annotation_recluster'
        var: 'mean', 'std', 'highly_variable', 'means', 'dispersions', 'dispersions_norm', 'highly_variable_nbatches', 'highly_variable_intersection'
        uns: 'annotation_colors', 'annotation_recluster_colors', 'batch_colors', 'hvg', 'leiden', 'leiden_colors', 'log1p', 'neighbors', 'pca', 'umap'
        obsm: 'X_CCA', 'X_pca', 'X_umap'
        varm: 'PCs'
        layers: 'counts', 'logcounts'
        obsp: 'connectivities', 'distances'
    ==================== Session Information ====================
    OS:macOS-26.2-arm64-arm-64bit
    Machine: arm64
    Processor: arm
    CPU cores (physical): 10
    CPU cores (logical): 10
    Total RAM (GB): 16.0
    Python version: 3.11.13
    -----
    anndata     0.11.4
    dotools_py  0.0.1
    pandas      2.3.2
    platform    1.0.8
    -----
    Cython                      3.1.4
    IPython                     9.5.0
    PIL                         11.3.0
    adjustText                  1.3.0
    altair                      6.0.0
    argparse                    1.1
    arrow                       1.3.0
    attr                        25.3.0
    attrs                       25.3.0
    beartype                    0.22.8
    charset_normalizer          3.4.3
    cloudpickle                 3.1.1
    comm                        0.2.3
    coverage                    7.11.0
    csv                         1.0
    ctypes                      1.1.0
    cycler                      0.12.1
    cython                      3.1.4
    dask                        2024.11.2
    dateutil                    2.9.0.post0
    decimal                     1.70
    decorator                   5.2.1
    defusedxml                  0.7.1
    deprecated                  1.2.18
    executing                   2.2.1
    h5py                        3.14.0
    idna                        3.10
    igraph                      0.11.9
    ipaddress                   1.0
    ipywidgets                  8.1.7
    jedi                        0.19.2
    jinja2                      3.1.6
    joblib                      1.5.2
    json                        2.0.9
    jsonpointer                 3.0.0
    jsonschema                  4.25.1
    kiwisolver                  1.4.9
    lark                        1.2.2
    leidenalg                   0.10.2
    llvmlite                    0.45.0
    logging                     0.5.1.2
    markupsafe                  3.0.2
    marshal                     4
    matplotlib                  3.10.6
    msgpack                     1.1.2
    narwhals                    2.5.0
    natsort                     8.4.0
    numba                       0.62.0
    numcodecs                   0.15.1
    numpy                       2.3.3
    packaging                   25.0
    parso                       0.8.5
    patsy                       1.0.1
    polars                      1.33.1
    prompt_toolkit              3.0.52
    psutil                      7.1.0
    pure_eval                   0.2.3
    pyarrow                     21.0.0
    pydot                       4.0.1
    pygments                    2.19.2
    pyparsing                   3.2.4
    pytz                        2025.2
    re                          2.2.1
    rfc3339_validator           0.1.4
    rfc3986_validator           0.1.1
    scanpy                      1.11.4
    scipy                       1.15.3
    seaborn                     0.13.2
    session_info                v1.0.1
    six                         1.17.0
    sklearn                     1.7.2
    socketserver                0.4
    sparse                      0.17.0
    sqlite3                     2.6.0
    stack_data                  0.6.3
    statsmodels                 0.14.5
    stdlib_list                 0.11.1
    sys                         3.11.13 (main, Jun  5 2025, 08:21:08) [Clang 14.0.6 ]
    tarfile                     0.9.0
    texttable                   1.7.0
    threadpoolctl               3.6.0
    tlz                         1.0.0
    toolz                       1.0.0
    torch                       2.8.0
    tqdm                        4.67.1
    traitlets                   5.14.3
    wcwidth                     0.2.13
    wrapt                       1.17.3
    yaml                        6.0.2
    zarr                        2.18.7
    zlib                        1.0
    -----
    Python 3.11.13 (main, Jun  5 2025, 08:21:08) [Clang 14.0.6 ]
    macOS-26.2-arm64-arm-64bit
    10 logical CPU cores, arm
    -----
    Session information updated at 2026-01-22 13:59
    =============================================================
    """
    import os
    import session_info
    import platform
    import psutil
    import io
    from contextlib import redirect_stdout

    if not os.path.exists(log_file):
        raise FileNotFoundError(f"{log_file} does not exist. Set kernel logger with do.settings.set_kernel_logger()")
    else:
        with open(log_file, "a") as f:
            f.write("\n\n")
            f.write("==================== Session Information ====================")
            f.write("\n")
            f.write("OS:" + platform.platform() + "\n")
            f.write("Machine: " + platform.machine()+ "\n")
            f.write("Processor: " + platform.processor()+ "\n")
            f.write("CPU cores (physical): " + str(psutil.cpu_count(logical=False))+ "\n")
            f.write("CPU cores (logical): " + str(psutil.cpu_count(logical=True))+ "\n")
            f.write("Total RAM (GB): " + str(round(psutil.virtual_memory().total / (1024 ** 3), 2))+ "\n")
            f.write("Python version: " + platform.python_version()+  "\n")
            f.write("\n\n")
            # Capture session_info.show() output
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                session_info.show(
                    na=False,
                    os=True,
                    cpu=True,
                    excludes=["backports"],
                    std_lib=True,
                    dependencies=True,
                    html=False,
                    jupyter=None,
                )
            f.write(buffer.getvalue())
            f.write("\n")
            f.write("=============================================================")
    return None



_default_console = Console()

def live_display(
    *,
    console:  None = None,
    current: int | None = None,
    total: int | None = None,
    msg: str | None = None
):
    """Decorator that displays a live status line and execution time for a function.

    .. note::
        This decorator is best suited for single-step or phase-based tasks.

    :param console: The Rich Console to use for output. If set to `None` a default console is used.
    :param current: The current step index.
    :param total: The total number of steps
    :param msg: Message to display. If not set, the name of the function will be displayed.
    :return: Returns a decorator that wraps the target function.

    Examples
    --------
    >>> import dotools_py as do
    >>> import  time
    >>> @do.utility.live_display(current=1, total=2)
    ... def step1():
    ...     time.sleep(2)
    >>> @do.utility.live_display(current=2, total=2)
    ... def step2():
    ...     time.sleep(3)
    >>> def testing():
    ...     step1()
    ...     step2()
    >>> testing()
    (1/2) step1 ...
    (1/2) step1 ✔ (0:00:02.001059)
    (2/2) step2 ...
    (2/2) step2 ✔ (0:00:03.000399)

    """
    console = console if console is not None else  _default_console
    current = 1 if current  is None else current
    total = 1 if total is None else total
    def decorator_live_display(func):
        display_msg = msg if msg is not None else func.__name__
        @functools.wraps(func)
        def wrapper_decorator(*args, **kwargs):
            console.print(f"({current}/{total}) {display_msg} ...")
            with Live(console=console, screen=False, auto_refresh=False) as live:
                start = timer()
                value = func(*args, **kwargs)
                end = timer()
                elapsed = datetime.timedelta(seconds=end - start)
                console.print(f"({current}/{total}) {display_msg} ✔ ({elapsed})")
                #live.update(
                #    f"({current}/{total}) {display_msg} [:heavy_check_mark:] ({elapsed})",
                #    refresh=True,
                #)
            return value

        return wrapper_decorator

    return decorator_live_display


def set_path(*, path: PathLike) -> PathLike:
    """Create the directory if it does not exist and return its path.

     Ensure that a directory exists and return its normalized path.
     The directory is then created if it does not already exist.

    :param path: A path-like object representing the directory to create, if it does not already exist.
    :return: Returns the normalized path to the directory.

    Example
    -------
    >>> import dotools_py as do
    >>> import os
    >>> test_path = do.utility.set_path(path="/tmp/testing_folder")
    >>> test_path
    PosixPath('/tmp/testing_folder')
    >>> os.path.exists("/tmp/testing_folder")
    True

    """
    if isinstance(path, str):
        path = convert_path(path)
    os.makedirs(path, exist_ok=True)
    return path





def dict_to_toml(dictionary: dict[str, str]):
    import tomlkit
    table = tomlkit.inline_table()

    for key, value in dictionary.items():
        table.add(key, value)
    array = tomlkit.array()
    array.append(table)
    return array


def get_installed_dependencies():
    from importlib.metadata import distributions
    dependencies = []
    for dist in distributions():
        name = dist.metadata["Name"]
        version = dist.version
        dependencies.append(f"{name}=={version}")
    return sorted(dependencies, key=str.lower)


def to_toml(
    path: PathLike,
    project_name: str,
    authors: dict,
    maintainers: dict | None = None,
    version: str | None = None,
    description: str | None = None,
    build_backend: str | None = None,
    build_requirements: list | None = None,
) -> None:
    """Export the current python environment to a TOML file.

    Creates a ``pyproject.toml``, ``uv.lock``, and ``.python-version`` in
    ``path``. The generated project pins the current Python interpreter
    version and records the currently installed packages. The lock file is
    generated by uv to resolve and pin the complete dependency tree.

    The resulting directory can be shared with another user, who can
    recreate the environment with::

        uv sync --locked

    :param path: Path to the folder were a TOML and lock file is created.
    :param project_name:  Name of the project
    :param authors: Author information as a dictionary. Keys should correspond to valid PEP 621 author fields, for example ``name`` and ``email``.
    :param maintainers:  Maintainer information as a dictionary. If ``None``, ``authors`` is used as the maintainer information.
    :param version: Version of the project. Defaults to ``"0.1.0"``.
    :param description: Description for the project.
    :param build_backend: Build backend.
    :param build_requirements: Requirements for building.
    :return: Returns None. This function creates the project files in ``path``

    Notes
    -----
    The generated ``pyproject.toml`` contains the currently installed packages, while ``uv.lock`` records the resolved dependency tree.
    together with ``.python-version``, these files are intended to reproduce the environment on another machine.

    Examples
    --------
    >>> from pathlib import Path
    >>> import os
    >>> import dotools_py as do
    >>> do.utility.to_toml(path='/tmp/CurrentEnv', project_name='DoToolsTest', authors={'name':'David Rodriguez Morales'})
    >>> os.listdir("/tmp/CurrentEnv")
    ['uv.lock', 'pyproject.toml', '.python-version']
    >>> print("".join(Path("/tmp/CurrentEnv/pyproject.toml").read_text(encoding="utf-8").splitlines(keepends=True)), end="")
    [build-system]
    build-backend = "hatchling.build"
    requires = ["hatchling"]

    [project]
    name = "DoToolsTest"
    version = "0.1.0"
    description = "Code for project DoToolsTest"
    authors = [{name = "David Rodriguez Morales"}]
    maintainers = [{name = "David Rodriguez Morales"}]

    requires-python = "==3.14.6"

    dependencies = [
        "absl-py==2.5.0",
        ...
        "zarr==3.2.1",
        "zipp==4.1.0",
    ]

    [tool.uv]
    python-downloads = "automatic"
    package = false

    """

    path = convert_path(path)
    path.mkdir(parents=True, exist_ok=True)
    toml_filename = path / "pyproject.toml"
    python_version_file = path / ".python-version"

    # -- Python Version
    major, minor, micro = sys.version_info[:3]
    python_version = f"{major}.{minor}.{micro}"

    # -- TOML Document
    doc = tomlkit.document()

    # -- Build System
    b_sys = tomlkit.table()
    b_sys.add("build-backend", build_backend or "hatchling.build")
    b_sys.add("requires", build_requirements or ["hatchling"])
    doc.add("build-system", b_sys)
    doc.add(tomlkit.nl())

    # -- Project Section
    project = tomlkit.table()
    project.add("name", project_name)  # Add project name
    project.add("version", version or "0.1.0")  # Add project version
    project.add("description", description or f"Code for project {project_name}")  # Add project description
    project.add("authors", dict_to_toml(authors))
    maintainers = maintainers or authors
    project.add("maintainers", dict_to_toml(maintainers))
    project.add(tomlkit.nl())
    project.add("requires-python", "==" + python_version)
    project.add(tomlkit.nl())

    # # -- Installed packages
    dependencies = tomlkit.array()
    dependencies.multiline(True)
    for dependency in get_installed_dependencies():
        dependencies.append(dependency)
    project.add("dependencies", dependencies)
    doc.add("project", project)

    #  UV section
    tool = tomlkit.table()
    uv = tomlkit.table()
    uv.add("python-downloads", "automatic")
    uv.add("package", False)
    tool.add("uv", uv)
    doc.add("tool", tool)

    with open(toml_filename, "w", encoding="utf-8") as f:
        f.write(tomlkit.dumps(doc))

    # -- Pin the exact python version for uv
    python_version_file.write_text(
        python_version + "\n",
        encoding="utf-8"
    )

    # -- Let uv resolve and lock the env
    subprocess.run(
        [
            "uv", "lock", "--directory", str(path)
        ], check=True,

    )
    return None



