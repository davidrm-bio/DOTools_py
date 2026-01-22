import os
import logging
import warnings
from pathlib import Path
import atexit
import datetime
import traceback
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from scanpy.plotting import palettes

from dotools_py import logger
from dotools_py.logger import set_verbosity

warnings.filterwarnings("ignore")


def interactive_session(enable: bool = True) -> None:
    """Make session interactive.

    :param enable: set to True to activate interactive plotting.
    :return:
    """
    from IPython import get_ipython

    if enable:
        try:
            shell = get_ipython().__class__.__name__
            if shell == "ZMQInteractiveShell":
                get_ipython().run_line_magic("matplotlib", "inline")
                logger.info('Jupyter enviroment detected. Using "inline" backend')
            else:
                if os.environ.get("DISPLAY", "") == "":
                    raise RuntimeError("No display found. Cannot use GUI backend")
                mpl.use("TkAgg", force=True)
                plt.ion()
                logger.info('Interactive plotting enabled. Using "TkAgg" backend')
        except Exception as e:
            logger.info(f"Interactive(True) Could not enable interactive plotting {e}.")
    else:
        try:
            plt.ioff()
            mpl.use("agg", force=True)
            logger.info('Interactive plotting disabled. Using "Agg" backend')
        except Exception as e:
            logger.info(f"Interactive(False) failed to disable interactive plotting {e}")

    return None



def matplotlib_backend(backend: str | Literal["pycharm", "jupyter"] = "tkagg") -> None:
    """Set matplotlib backend.

    Parameters
    ----------
    backend
         Use for displaying `Matplotlib figures <https://matplotlib.org/stable/users/explain/figure/backends.html#what-is-a-backend>`_.
         If backend is set to `pycharm` it will display plots in the SciView section.
         If backend is set to `jupyter` it will display the plots inline.


    Returns
    -------
    Returns `None`
    """
    from IPython import get_ipython

    if backend == "pycharm":
        mpl.use("module://backend_interagg")
    elif backend == "jupyter":
        get_ipython().run_line_magic("matplotlib", "inline")
    else:
        try:
            mpl.use(backend)
        except ValueError as e:
            logger.warn(f"{backend} not a valid matplotlib backend: {e}")
    return None


def set_random_state(random_state: int = 0, verbosity: bool = True)-> None:
    """Set random state.

    This function set the global seed for random number generator, specifically for
    NumPy, random, torch and tensorflow.

    :param random_state: seed.
    :param verbosity: show a message indicating the random_state that has been set.
    :return: Returns None
    """
    import random
    import numpy as np


    random.seed(random_state)
    np.random.seed(random_state)

    try:
        import torch
        torch.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)
    except ImportError:
        pass
    try:
        import tensorflow as tf
        tf.random.set_seed(random_state)
    except ImportError:
        pass

    if verbosity:
        logger.info(f"Setting random state to: {random_state}")
    return None


def session_settings(
    verbosity: int = 2,
    interactive: bool = True,
    dpi: int = 90,
    dpi_save: int = 300,
    facecolor: str = "white",
    colormap: str = "Reds",
    frameon: bool = True,
    transparent: bool = False,
    fontsize: int = 13,
    axes_fontsize: int = 16,
    axes_fontweight: str = "bold",
    title_fontsize: int = 18,
    title_fontweight: str = "bold",
    ticks_fontsize: int = 12,
    figsize: tuple = (4, 5),
    top_spine: bool = False,
    right_spine: bool = False,
    grid: bool = False,
    random_state: int = 0,
) -> None:
    """Set general settings.

    :param verbosity: set verbosity level. 0 for silent, 1 for Info/Warnings, 2 for Info/Warnings + Scanpy Info/Warnings
                      and 3 for debug mode.
    :param interactive: if set to true, activate interactive plotting.
    :param dpi: dpi for showing plots.
    :param dpi_save: dpi for saving plots.
    :param facecolor: Sets backgrounds via rcParams['figure.facecolor'] = facecolor and rcParams['axes.facecolor'] = facecolor.
    :param colormap: Convenience method for setting the default color map.
    :param frameon: Add frames and axes labels to scatter plots.
    :param transparent: Save figures with transparent background.
    :param fontsize: Set the fontsize.
    :param axes_fontsize: Set the fontsize for the x and y labels.
    :param axes_fontweight: Set the font-weight for the x and y labels.
    :param title_fontsize:  Set the fontsize for the title.
    :param title_fontweight: Set the font-weight for the title.
    :param ticks_fontsize: Set the fontsize for the x and y ticks.
    :param figsize: Set the figsize.
    :param top_spine: remove the top spine.
    :param right_spine: remove the right spine.
    :param grid: show the grid lines.
    :param random_state: seed for random number generator.
    :return:
    """
    import matplotlib.font_manager as fm

    available_fonts = sorted({f.name for f in fm.fontManager.ttflist})
    font_family = "Helvetica" if "Helvetica" in available_fonts else "sans-serif"

    # Scanpy Settings
    set_random_state(random_state, verbosity=False)
    interactive_session(interactive)
    logging.getLogger("fontTools.subset").setLevel(logging.ERROR)
    set_verbosity(verbosity)

    plt.rcParams.update(
        {
            # Font settings
            "font.family": font_family,
            "font.serif": ["Helvetica"],
            "font.size": fontsize,
            "font.weight": "normal",
            "axes.labelsize": axes_fontsize,
            "axes.labelweight": axes_fontweight,
            "axes.titlesize": title_fontsize,
            "axes.titleweight": title_fontweight,
            "xtick.labelsize": ticks_fontsize,
            "ytick.labelsize": ticks_fontsize,
            "legend.fontsize": fontsize * 0.92,
            # Same configuration as Scanpy
            "savefig.dpi": dpi_save,
            "savefig.transparent": transparent,
            "figure.subplot.left": 0.18,
            "figure.subplot.right": 0.96,
            "figure.subplot.bottom": 0.15,
            "figure.subplot.top": 0.91,
            "lines.markeredgewidth": 1,
            "legend.numpoints": 1,
            "legend.scatterpoints": 1,
            "legend.handlelength": 0.5,
            "legend.handletextpad": 0.4,
            "axes.prop_cycle": cycler(color=palettes.default_20),
            "axes.edgecolor": "black",
            "axes.facecolor": "white",
            "xtick.color": "k",
            "ytick.color": "k",
            "image.cmap": mpl.rcParams["image.cmap"] if colormap is None else colormap,
            # Figure and axes
            "figure.figsize": figsize,  # Single column width (inches)
            "figure.dpi": dpi,
            "figure.facecolor": facecolor,
            # Grid settings
            "axes.grid": grid,
            # Line settings
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
            # Spines
            "axes.spines.top": top_spine,
            "axes.spines.right": right_spine,
            "axes.linewidth": 1.2,
            # Ticks
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
            "xtick.major.width": 1,
            "ytick.major.width": 1,
            "xtick.minor.width": 0.8,
            "ytick.minor.width": 0.8,
            # Legend
            "legend.frameon": frameon,
            "legend.loc": "best",
            # Text and font rendering
            "text.usetex": False,  # Do not use LaTeX for text rendering
            "svg.fonttype": "none",  # Keep text as text in SVGs
            "figure.autolayout": True,  # Prevent overlapping elements
            "savefig.bbox": "tight",  # Remove unnecessary whitespace
        }
    )

    mpl.rcParams["pdf.fonttype"] = 42  # Use TrueType fonts in PDFs (editable text)

    return None



_kernel_logger_recording = False
_kernel_logger_file: Path | None = None
_kernel_logger_registered = False


def set_kernel_logger(
    filename: str | Path="./History.log",
    overwrite: bool =False,
    session: Literal["activate", "deactivate"] = "activate"
) -> None:
    """Save kernel history in a file.

    Parameters
    ----------
    filename
        Absolute path to the log file.
    overwrite
        Whether the log file should be overwritten or not.
    session
        Activate or deactivate the history logging.

    Returns
    -------
    Returns None.

    Examples
    --------

    >>> import dotools_py as do
    >>> do.settings.set_kernel_logger('./History.log')
    >>> adata = do.dt.example_10x_processed()
    >>> adata
    >>> do.settings.toogle_kernel_logger(False)
    >>> print(open("History.log").read())
    ========== KERNEL SESSION START 2025-12-12 13:48:22.118424 ==========
    [CODE 2025-12-12 13:48:22.120206]
    >>> do.settings.set_kernel_logger('./History.log')
    [CODE 2025-12-12 13:48:22.734407]
    >>> adata = do.dt.example_10x_processed()
    [CODE 2025-12-12 13:48:23.606794]
    >>> adata
    [OUTPUT 2025-12-12 13:48:23.609162]
    AnnData object with n_obs × n_vars = 700 × 1851
        obs: 'batch', 'condition', 'n_genes_by_counts', 'log1p_n_genes_by_counts', 'total_counts', 'log1p_total_counts', 'total_counts_mt', 'log1p_total_counts_mt', 'pct_counts_mt', 'total_counts_ribo', 'log1p_total_counts_ribo', 'pct_counts_ribo', 'n_genes', 'n_counts', 'doublet_class', 'doublet_score', 'leiden', 'cell_type', 'autoAnnot', 'celltypist_conf_score', 'annotation', 'annotation_recluster'
        var: 'mean', 'std', 'highly_variable', 'means', 'dispersions', 'dispersions_norm', 'highly_variable_nbatches', 'highly_variable_intersection'
        uns: 'annotation_colors', 'annotation_recluster_colors', 'batch_colors', 'hvg', 'leiden', 'leiden_colors', 'log1p', 'neighbors', 'pca', 'umap'
        obsm: 'X_CCA', 'X_pca', 'X_umap'
        varm: 'PCs'
        layers: 'counts', 'logcounts'
        obsp: 'connectivities', 'distances'
    ========== KERNEL SESSION PAUSED 2025-12-12 13:48:25.090038 ==========

    """
    from IPython import get_ipython
    from IPython.utils.capture import capture_output

    global _kernel_logger_recording, _kernel_logger_file, _kernel_logger_registered
    _session = True if session == "activate" else False
    _kernel_logger_recording = _session
    _kernel_logger_file = Path(filename)


    mode = "w" if overwrite else "a"
    if os.path.exists(filename):
        logger.warn(f"Log file '{filename}' already exists. 'overwrite' is set to {overwrite}, using '{mode}' mode")

    if _session:
        logger.info(f"Kernel history logging started. Log file '{filename}'")
        with open(filename, mode, encoding="utf-8") as f:
            f.write(f"========== KERNEL SESSION START {datetime.datetime.now()} ==========\n")

    ip = get_ipython()
    if not ip:
        logger.warn("No IPython kernel detected. This logger works in IPython/Jupyter.")
        return None

    if not _kernel_logger_registered:
        def log_post_run_cell(result):
            if not _kernel_logger_recording:  # skip logging if recording is False
                return

            code = result.info.raw_cell if hasattr(result, "info") else None
            with open(filename, "a", encoding="utf-8") as f:
                if code:
                    f.write(f"\n[CODE {datetime.datetime.now()}]\n{code}\n")

                # Log exceptions immediately after code
                if result.error_in_exec:
                    tb_str = "".join(traceback.format_exception(
                        type(result.error_in_exec), result.error_in_exec, result.error_in_exec.__traceback__)
                    )
                    f.write(f"[EXCEPTION {datetime.datetime.now()}]\n{tb_str}\n")

            # Capture stdout / stderr / return value
            try:
                with capture_output(display=True) as captured:
                    pass
            except Exception:
                captured = None

            with open(filename, "a", encoding="utf-8") as f:
                if captured:
                    if captured.stdout.strip():
                        f.write(f"[STDOUT {datetime.datetime.now()}]\n{captured.stdout}\n")
                    if captured.stderr.strip():
                        f.write(f"[STDERR {datetime.datetime.now()}]\n{captured.stderr}\n")

                if hasattr(result, "result") and result.result is not None:
                    try:
                        f.write(f"[OUTPUT {datetime.datetime.now()}]\n{repr(result.result)}\n")
                    except Exception:
                        f.write(f"[OUTPUT {datetime.datetime.now()}] <unrepresentable output>\n")
        ip.events.register("post_run_cell", log_post_run_cell)
        _kernel_logger_registered = True
    # End-of-session marker
    def session_end_marker():
        if _kernel_logger_file:
            with open(_kernel_logger_file, "a", encoding="utf-8") as f:
                f.write(f"========== KERNEL SESSION END {datetime.datetime.now()} ==========\n\n")
    atexit.register(session_end_marker)

    return None


def toogle_kernel_logger(state: bool):
    """Activate or deactivate kernel recording.

    Parameters
    ----------
    state
        Boolean indicating if the history logging should be activated or deactivated.

    Returns
    -------
    Returns None.

    See Also
    --------
        :func:`dotools_py.settings.set_kernel_logger`: initialize kernel history recording.


    """
    global _kernel_logger_recording, _kernel_logger_file
    if _kernel_logger_file is None:
        logger.warn("Kernel logger not initialized. Call set_kernel_logger first.")
        return None

    if state and not _kernel_logger_recording:
        # Start logging
        with open(_kernel_logger_file, "a", encoding="utf-8") as f:
            f.write(f"\n========== KERNEL SESSION START {datetime.datetime.now()} ==========\n")
        _kernel_logger_recording = True
        logger.info("Kernel logger resumed.")
    elif not state and _kernel_logger_recording:
        # Stop logging
        with open(_kernel_logger_file, "a", encoding="utf-8") as f:
            f.write(f"========== KERNEL SESSION PAUSED {datetime.datetime.now()} ==========\n")
        _kernel_logger_recording = False
        logger.info("Kernel logger paused.")
