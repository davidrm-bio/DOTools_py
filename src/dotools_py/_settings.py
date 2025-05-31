import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import scanpy as sc


class DeprecatedFunctionError(Exception):
    pass


class CustomWarning(UserWarning):
    pass


warnings.filterwarnings("default", category=CustomWarning)


def set_plt_theme():
    # Scanpy Settings
    sc.settings.set_figure_params(
        dpi=100, dpi_save=300, facecolor="white", color_map="Reds", frameon=True, transparent=False
    )

    # Set global font sizes and styles
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 18
    plt.rcParams["axes.labelweight"] = "bold"

    # Set title font size and style
    plt.rcParams["axes.titlesize"] = 20
    plt.rcParams["axes.titleweight"] = "bold"

    # Set legends and xticks
    plt.rcParams["legend.fontsize"] = 14
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12

    # Hide top and right spines
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False

    # Remove grid
    plt.rcParams["axes.grid"] = False
    plt.rcParams["axes.linewidth"] = 1.2  # Thicker axes lines
    plt.rcParams["lines.linewidth"] = 2.0  # Thicker lines

    # Set Font family
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["text.usetex"] = False
    plt.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["pdf.fonttype"] = 42

    plt.rcParams["figure.autolayout"] = True  # Prevent overlapping
    plt.rcParams["savefig.bbox"] = "tight"  # No unnecessary whitespace
    return


def iOn():
    """Activate Interactive plotting (tkagg backed)"""
    plt.ion()
    mpl.use("TkAgg")
    return


def iOff():
    """Deactivate Interactive plotting (agg backed)"""
    plt.ioff()
    mpl.use("Agg")
    return


set_plt_theme()
iOn()
