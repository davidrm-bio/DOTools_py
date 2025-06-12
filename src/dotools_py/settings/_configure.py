import os
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import scanpy as sc

from dotools_py import logger
from dotools_py.logger import  set_verbosity

warnings.filterwarnings("ignore")


def interactive_session(enable: bool = True):
    """Make session interactive.

    :param enable: set to True to activate interactive plotting
    :return:
    """
    from IPython import get_ipython
    if enable:
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                get_ipython().run_line_magic('matplotlib', 'inline')
                logger.info('Jupyter enviroment detected. Using "inline" backend')
            else:
                if os.environ.get('DISPLAY', '') == '':
                    raise RuntimeError('No display found. Cannot use GUI backend')
                mpl.use('TkAgg', force=True)
                plt.ion()
                logger.info('Interactive plotting enabled. Using "TkAgg" backend')
        except Exception as e:
            logger.info(f'Interactive(True) Could not enable interactive plotting {e}.')
    else:
        try:
            plt.ioff()
            mpl.use('agg', force=True)
            logger.info('Interactive plotting disabled. Using "Agg" backend')
        except Exception as e:
            logger.info(f'Interactive(False) failed to disable interactive plotting {e}')

    return


def session_settings(
    verbosity: int = 2,
    interactive: bool = True,
    dpi: int = 100,
    dpi_save: int = 300,
    facecolor: str = 'white',
    colormap: str = 'Reds',
    frameon: bool = True,
    transparent: bool = False,
    fontsize: int = 10,
    axes_fontsize: int = 12,
    axes_fontweight: str = 'bold',
    title_fontsize: int = 12,
    title_fontweight: str = 'bold',
    legend_fontsize: int = 9,
    ticks_fontsize: int = 9,
    figsize: tuple =(4, 5),
    top_spine: bool = False,
    right_spine: bool = False,
    grid: bool = False,
) -> None:
    """Set general settings.

    :param verbosity: set verbosity level. 0 for silent, 1 for Info/Warnings, 2 for Info/Warnings + Scanpy Info/Warnings and 3 for debug mode.
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
    :param legend_fontsize: Set the fontsize for the legend.
    :param ticks_fontsize: Set the fontsize for the x and y ticks.
    :param figsize: Set the figsize.
    :param top_spine: remove the top spine.
    :param right_spine: remove the right spine.
    :param grid: show the grid lines.
    :param font_family: font family to use.
    :return:
    """

    # Scanpy Settings
    sc.settings.set_figure_params(dpi=dpi, dpi_save=dpi_save, facecolor=facecolor,
        color_map=colormap, frameon=frameon, transparent=transparent
    )
    set_verbosity(verbosity)
    interactive_session(interactive)

    plt.rcParams.update({
        # Font settings
        "font.family": "sans-serif",
        "font.serif": ["Helvetica"],
        "font.size": fontsize,
        "font.weight": 'normal',
        "axes.labelsize": axes_fontsize,
        "axes.labelweight": axes_fontweight,
        "axes.titlesize": title_fontsize,
        "axes.titleweight": title_fontweight,
        "xtick.labelsize": ticks_fontsize,
        "ytick.labelsize": ticks_fontsize,
        "legend.fontsize": legend_fontsize,

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
    })

    mpl.rcParams["pdf.fonttype"] = 42  # Use TrueType fonts in PDFs (editable text)

    return

