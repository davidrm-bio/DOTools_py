import matplotlib.pyplot as plt

import dotools_py as do


def test_generate_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    colors = do.utility.generate_cmap("white", "red")
    assert isinstance(colors, LinearSegmentedColormap)
    return


def test_get_hex_cmap():
    colors = do.utility.get_hex_colormaps("Reds")
    assert isinstance(colors, list)
    assert  '#67000d' in colors
    return


def test_extended_tab20():
    colors = do.utility.extended_tab20(6)
    assert len(colors) == 20/2*6
    return


def test_spine_format():
    fig, axs = plt.subplots(1, 1)
    axs.spines["top"].set_visible(True)
    do.utility.spine_format(axs)
    top_visible = axs.spines['top'].get_visible()
    assert top_visible == False
    plt.close()


    from dotools_py.utils import  spine_format

    fig, axs = plt.subplots(1, 1)
    axs.spines["top"].set_visible(True)
    spine_format(axs)
    top_visible = axs.spines['top'].get_visible()
    assert top_visible == False
    plt.close()

    return
