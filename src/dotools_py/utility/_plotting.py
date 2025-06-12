from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
import matplotlib.pyplot as plt


def generate_cmap(*args) -> LinearSegmentedColormap:
    """Generate a custom colormap.

    This functions returns a color map. Specify colors to set a gradient in the specified order. Use
    (1, 1, 1, 0) to set transparent

    :param args: colors, RGB or HexaCodes.
    :return: custom cmap.
    """
    colors = [col for col in args]
    return LinearSegmentedColormap.from_list('Custom', colors, N=256)


def get_hex_colormaps(
    colormap: str
):
    """Get a list with Hexa IDs for a colormap.

    :param colormap: colormap name
    :return: list with Hexa IDs
    """
    cmap = plt.get_cmap(colormap)
    return [mpl.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]


def extended_tab20(
    n_shades: int = 6
) -> list:
    """Extends the colormap tab20 to more shades for a color.

    :param n_shades: number of shades.
    :return: list of colors.
    """
    # Base colors from the 'tab20' colormap
    base_colors = plt.cm.tab20.colors
    extended_colors = []

    # Generate 6 shades per color
    for i in range(0, len(base_colors), 2):  # Go by pairs, as 'tab20' has pairs of each color
        main_color = base_colors[i]
        secondary_color = base_colors[i + 1]

        # Interpolate between main and secondary color
        for j in range(n_shades):
            # Linear interpolation between the main and secondary color
            interp = j / (n_shades - 1)
            color = [
                main_color[k] * (1 - interp) + secondary_color[k] * interp
                for k in range(3)
            ]
            extended_colors.append(color)
    return extended_colors


def spine_format(
    axis: plt.Axes,
    txt: str = "UMAP",
    fontsize: int = 12
) -> None:
    """Formatting the spines for Embeddings.

    :param axis: axis object.
    :param txt: type of embedding.
    :param fontsize: size of the text.
    :return:
    """
    axis.spines[["right", "top"]].set_visible(False)
    axis.set_xlabel(txt + "1", loc="left", fontsize=fontsize, fontweight="bold")
    axis.set_ylabel(txt + "2", loc="bottom", fontsize=fontsize, fontweight="bold")
    return

