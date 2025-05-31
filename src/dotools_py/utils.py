from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec


def get_paths_utils(script: str):
    """Get path for a script within the project.

    :param script:
    :return:
    """
    module_dir = Path(__file__).parent
    return (module_dir / "util_scripts" / script).resolve()


def convert_path(path: Path | str) -> Path:
    """Convert to Path format if string is provided.

    :param path: string or Path variable
    :return: path
    """
    if not isinstance(path, Path):
        return Path(path)
    else:
        return path


def sanitize_anndata(adata: ad.AnnData) -> None:
    """Transform string annotations to categorical.

    :param adata: AnnData
    :return None
    """
    adata._sanitize()
    return None


def get_centroids(adata: ad.AnnData, cluster_key: str, basis: str = "X_umap"):
    """Get centroids for clusters in anndata object.

    :param adata: anndata
    :param cluster_key: .obs column with categorical information
    :param basis: embedding to use (Default X_umaP)
    :return: centroids as a panda dataframe
    """
    all_pos = pd.DataFrame(adata.obsm[basis], columns=["x", "y"])
    all_pos["group"] = adata.obs[cluster_key].values
    return all_pos.groupby("group", observed=True).median().sort_index()


def get_subplot_shape(n_samples: int, ncols: int) -> tuple:
    """Compute the number of rows and columns to use for defining the figure base on a desired number of samples and columns.

    :param n_samples: number of samples to plot
    :param ncols: number of columns to plot
    :return: nrows, ncols, extras (extra subplots that should be hidden)
    """
    if n_samples < ncols:  # Correction
        ncols = n_samples  # Adjust plot if more cols than samples are specified
    nrows = int(np.ceil(n_samples / ncols))
    extras = nrows * ncols - n_samples  # For hiding empty subplots
    return nrows, ncols, extras


def spine_format(axis: plt.axis, txt: str = "UMAP", fontsize: int = 10) -> None:
    """Formatting the spines for Embeddings.

    :param axis: axis object
    :param txt: type of embedding
    :param fontsize: size of the text
    :return:
    """
    axis.spines[["right", "top"]].set_visible(False)
    axis.set_xlabel(txt + "1", loc="left", fontsize=fontsize, fontweight="bold")
    axis.set_ylabel(txt + "2", loc="bottom", fontsize=fontsize, fontweight="bold")
    return


def remove_extra(extras: int, nrows: int, ncols: int, axs: plt.Axes) -> None:
    """Hide the last "extras" subplots.

    :param extras: number of subplots to remove
    :param nrows: number of rows of the plot
    :param ncols: number of columns of the plot
    :param axs: axis object
    :return:
    """
    if extras == 0:
        return
    else:
        for check in range(nrows * ncols - extras, nrows * ncols):
            axs[check].set_visible(False)
        return



def make_grid_spec(ax_or_figsize, *,
                   nrows: int,
                   ncols: int,
                   wspace: float = None,
                   hspace: float = None,
                   width_ratios: float = None,
                   height_ratios: float = None
                   ):
    # Taken from Scanpy
    kw = dict(wspace=wspace, hspace=hspace, width_ratios=width_ratios, height_ratios=height_ratios)

    if isinstance(ax_or_figsize, tuple):
        fig = plt.figure(figsize=ax_or_figsize)
        return fig, gridspec.GridSpec(nrows, ncols, **kw)
    else:
        ax = ax_or_figsize
        ax.axis("off")
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
        return ax.figure, ax.get_subplotspec().subgridspec(nrows, ncols, **kw)


def format_terms_gsea(df: pd.DataFrame, term_col: str, cutoff: int=35):
    """Format Terms from Gene Set Enrichment Analysis.

    :param df:
    :param term_col:
    :param cutoff:
    :return:
    """
    import re
    def remove_whitespace_around_newlines(text):
        # Replace whitespace before and after newlines with just the newline
        return re.sub(r'\s*\n\s*', '\n', text)

    newterms = []
    for text in df[term_col]:
        newterm, text_list_nchar, nchar, limit = [], [], 0, cutoff
        text_list = text.split(' ')
        for txt in text_list:  # From text_list get a list where we sum nchar from a word + previous word
            nchar += len(txt)
            text_list_nchar.append(nchar)
        for idx, word in enumerate(text_list_nchar):
            if word > limit:  # If we have more than cutoff characters in len add a break line
                newterm.append('\n')
                limit += cutoff
            newterm.append(text_list[idx])
        newterm = ' '.join(newterm)
        cleanterm = remove_whitespace_around_newlines(newterm)  # remove whitespace inserted
        newterms.append(cleanterm)
    df[term_col] = newterms

    return df
