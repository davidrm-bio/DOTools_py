import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import scanpy as sc
from logger import set_verbosity


class DeprecatedFunctionError(Exception):
    pass


class CustomWarning(UserWarning):
    pass


warnings.filterwarnings("default", category=CustomWarning)


def iOn():
    """Activate interactive plotting if avaialble

    :return:
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import os

    try:
        # If in headless (no display), don't try to use TkAgg
        if os.environ.get("DISPLAY", "") == "":
            raise RuntimeError("No display found. Cannot use interactive backend.")

        mpl.use("TkAgg", force=True)
        plt.ion()
        print("Interactive plotting enabled (TkAgg).")
    except Exception as e:
        print(f"[iOn()] Interactive plotting not available: {e}")


def iOff():
    """Deactivate Interactive plotting.

    :return: None
    """
    plt.ioff()
    mpl.use("Agg")
    return


def settings(
    verbosity: int = 2,
    interactive: bool = True,
    dpi: int = 100,
    dpi_save: int = 300,
    facecolor: str = 'white',
    colormap: str = 'Reds',
    frameon: bool = True,
    transparent: bool = False,
    fontsize: int = 12,
    axes_fontsize: int = 18,
    axes_fontweight: str = 'bold',
    title_fontsize: int = 20,
    title_fontweight: str = 'bold',
    legend_fontsize: int = 14,
    ticks_fontsize: int = 12,
    top_spine: bool = False,
    right_spine: bool = False,
    grid: bool = False,
    font_family: str = 'sans-serif',
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
    :param top_spine: remove the top spine.
    :param right_spine: remove the right spine.
    :param grid: show the grid lines.
    :param font_family: font family to use.
    :return:
    """
    verbosity = set_verbosity(verbosity)
    if interactive:
        iOn()
    else:
        iOff()

    # Scanpy Settings
    sc.settings.set_figure_params(
        dpi=dpi, dpi_save=dpi_save, facecolor=facecolor,
        color_map=colormap, frameon=frameon, transparent=transparent
    )

    # Set global font sizes and styles
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["axes.labelsize"] = axes_fontsize
    plt.rcParams["axes.labelweight"] = axes_fontweight

    # Set title font size and style
    plt.rcParams["axes.titlesize"] = title_fontsize
    plt.rcParams["axes.titleweight"] = title_fontweight

    # Set legends and xticks
    plt.rcParams["legend.fontsize"] = legend_fontsize
    plt.rcParams["xtick.labelsize"] = ticks_fontsize
    plt.rcParams["ytick.labelsize"] = ticks_fontsize

    # Hide top and right spines
    plt.rcParams["axes.spines.top"] = top_spine
    plt.rcParams["axes.spines.right"] = right_spine

    # Remove grid
    plt.rcParams["axes.grid"] = grid
    plt.rcParams["axes.linewidth"] = 1.2  # Thicker axes lines
    plt.rcParams["lines.linewidth"] = 2.0  # Thicker lines

    # Set Font family
    plt.rcParams["font.family"] = font_family
    if font_family == 'sans-serif':
        plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["text.usetex"] = False
    plt.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["pdf.fonttype"] = 42

    plt.rcParams["figure.autolayout"] = True  # Prevent overlapping
    plt.rcParams["savefig.bbox"] = "tight"  # No unnecessary whitespace
    return

