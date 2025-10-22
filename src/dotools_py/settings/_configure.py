import logging
import os
import warnings

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


#def vector_friendly(enable: bool = True) -> None:
#    """Plot scatter plots using png backend even when exporting as pdf or svg.
#
#    :param enable: Set to true to enable.
#    :return: Returns None
#    """
#    import scanpy as sc
#    # Keep settings from session_settings default
#    sc.set_figure_params(scanpy=False, dpi=90, dpi_save=300,
#                         frameon=True, vector_friendly=True,
#                         fontsize=13, figsize=None, color_map='Reds',
#                         format='pdf', facecolor='white', transparent=False)
#    return


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
    :return:
    """
    import matplotlib.font_manager as fm
    available_fonts = sorted({f.name for f in fm.fontManager.ttflist})
    font_family = "Helvetica" if "Helvetica" in available_fonts else "sans-serif"

    # Scanpy Settings
    set_verbosity(verbosity)
    interactive_session(interactive)
    logging.getLogger("fontTools.subset").setLevel(logging.ERROR)

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
