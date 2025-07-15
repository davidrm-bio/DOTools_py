import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def generate_cmap(*args) -> LinearSegmentedColormap:
    """Generate a custom colormap.

    This functions returns a color map. Specify colors to set a gradient in the specified order. Use
    (1, 1, 1, 0) to set transparent

    :param args: colors, RGB or HexaCodes.
    :return: custom colormap.

    Example
    -------

    .. plot::
        :context: close-figs

        import dotools_py as do
        import matplotlib.pyplot as plt
        import numpy as np
        cbar = do.utility.generate_cmap('royalblue', 'lightsteelblue', 'white', 'tomato', 'firebrick')
        plt.figure(figsize=(6, 2))
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        gradient = np.vstack([gradient] * 10)  # Stack to make it thicker
        plt.imshow(gradient, aspect='auto', cmap=cbar)
        plt.axis('off')

    """
    colors = [col for col in args]
    return LinearSegmentedColormap.from_list("Custom", colors, N=256)


def get_hex_colormaps(colormap: str) -> list:
    """Get a list with Hexa IDs for a colormap.

    :param colormap: colormap name.
    :return: list with Hexa IDs.

    Example
    -------
    >>> import dotools_py as do
    >>> hex_list = do.utility.get_hex_colormaps("Reds")
    >>> hex_list[:5]
    ['#fff5f0', '#fff4ef', '#fff4ee', '#fff3ed', '#fff2ec']

    """
    cmap = plt.get_cmap(colormap)
    return [mpl.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]


def extended_tab20(n_shades: int = 6) -> list:
    """Extends the colormap tab20 to more shades for a color.

    :param n_shades: number of shades.
    :return: list of colors.

    Example
    -------
    >>> import dotools_py as do
    >>> shades_list = do.utility.extended_tab20()
    >>> shades_list[:5]
    [[0.12156862745098039, 0.4666666666666667, 0.7058823529411765],
     [0.23372549019607844, 0.5294117647058824, 0.7466666666666668],
     [0.3458823529411765, 0.592156862745098, 0.7874509803921569],
     [0.45803921568627454, 0.6549019607843137, 0.8282352941176472],
     [0.5701960784313725, 0.7176470588235294, 0.8690196078431373]]

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
            color = [main_color[k] * (1 - interp) + secondary_color[k] * interp for k in range(3)]
            extended_colors.append(color)
    return extended_colors


def spine_format(axis: plt.Axes, txt: str = "UMAP", fontsize: int = 12) -> None:
    """Formatting the spines for embeddings.

    :param axis: matplotlib axes object.
    :param txt: text of the type of embedding.
    :param fontsize: size of the text.
    :return:
    """
    axis.spines[["right", "top"]].set_visible(False)
    axis.set_xlabel(txt + "1", loc="left", fontsize=fontsize, fontweight="bold")
    axis.set_ylabel(txt + "2", loc="bottom", fontsize=fontsize, fontweight="bold")
    return None
