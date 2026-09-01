from prelude_py import mpl, plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from dotools_py.logger import  logger

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


def spine_format(*, axis: plt.Axes, txt: str = "UMAP", fontsize: int = 12) -> None:
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


def _get_ticks_defaults(properties: dict) -> tuple:
    properties = {} if properties is None else properties
    size, weight, rotation = (properties.get("size", 12),
                              properties.get("weight", "bold"),
                              properties.get("rotation", None))
    ha, va = ("center", "top") if rotation is None else ("right", "top")
    return size, weight, rotation, ha, va



def tab30() -> None:
    """Create a tab30 colormap.

    The colormap can be access using `tab30`

    :return: Returns `None`

    Example
    -------

    .. plot::
        :context: close-figs

        import dotools_py as do
        import matplotlib.pyplot as plt
        import numpy as np
        do.utility.tab30()
        plt.figure(figsize=(6, 2))
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        gradient = np.vstack([gradient] * 10)  # Stack to make it thicker
        plt.imshow(gradient, aspect='auto', cmap="tab30")
        plt.axis('off')

    """
    cmap = ListedColormap([
        "#1f77b4", "#ff7f0e", "#2ca02c", "#ee5c42", "#9467bd", "#cd661d", "#e377c2", "#ffbb78", "#bcbd22", "#17becf",
        "#eead0e", "#aec7e8", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d",
        "#9edae5", "#f4a460", "#ffe4b5", "#b0c4de", "#9932cc", "#ee8262", "#228b22", "#ffe4c4",
        "#b22222", "#cd853f", "#6a5acd"
    ], "tab30")
    try:
        mpl.colormaps.register(cmap, name="tab30")
    except ValueError as e:
        logger.debug("tab30 is already registered")
    return None

tab30()


def add_spines(
    ax: plt.Axes,
    *,
    left: bool = True,
    bottom: bool = True,
    top: bool = False,
    right: bool = False,
    all_spines: bool = False,
    invert: bool = False,
) -> None:
    """Add spines to a matplotlib axis.

    :param ax: Matplotlib axis.
    :param left: If set to `True`, the left spine will be visible.
    :param bottom:  If set to `True`, the lower spine will be visible.
    :param top: If set to `True`, the top spine will be visible.
    :param right: If set to `True`, the right spine will be visible.
    :param all_spines: If set to `True`, all spines will be visible.
    :param invert: If set to `True`, the inverse will be set, i.e., if ``top`` is set to `True`, the spine will be hidden.
    :return: Returns None.

    Example
    -------

    .. plot::
        :context: close-figs

        import dotools_py as do
        import matplotlib.pyplot as plt
        import seaborn as sns

        df = sns.load_dataset("iris")

        fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        sns.barplot(df, x="species", y="sepal_length", ax=axs[0])
        sns.barplot(df, x="species", y="sepal_length", ax=axs[1])
        sns.barplot(df, x="species", y="sepal_length", ax=axs[2])

        # lower and left by default are set to True, invert removes them and add the ones set to False
        do.utility.add_spines(axs[1], invert=True)
        do.utility.add_spines(axs[2], all_spines=True)

    """
    visibility = {"left": left, "bottom":bottom, "top": top, "right":right}
    if all_spines:
        visibility = {name: True for name in visibility}
    for name, spine in ax.spines.items():
        spine.set_visible(visibility[name] != invert)
    return None


