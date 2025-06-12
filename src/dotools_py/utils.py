from pathlib import Path
from typing import Union

import anndata as ad
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import functools
import importlib
import subprocess
import sys


class DeprecatedFunctionError(Exception):
    pass


def get_paths_utils(
    script: str
):
    """Get path for a script within the project.

    :param script:
    :return:
    """
    module_dir = Path(__file__).parent
    return (module_dir / "util_scripts" / script).resolve()


def convert_path(
    path: Union[Path, str]
) -> Path:
    """Convert to Path format if string is provided.

    :param path: string or Path variable
    :return: path
    """
    if not isinstance(path, Path):
        return Path(path)
    else:
        return path


def sanitize_anndata(
    adata: ad.AnnData
) -> None:
    """Transform string annotations to categorical.

    :param adata: AnnData
    :return None
    """
    adata._sanitize()
    return None


def get_centroids(
    adata: ad.AnnData,
    cluster_key: str,
    basis: str = "X_umap"
)->pd.DataFrame:
    """Get centroids for clusters in anndata object.

    :param adata: anndata
    :param cluster_key: .obs column with categorical information
    :param basis: embedding to use (Default X_umaP)
    :return: centroids as a panda dataframe
    """
    all_pos = pd.DataFrame(adata.obsm[basis], columns=["x", "y"])
    all_pos["group"] = adata.obs[cluster_key].values
    return all_pos.groupby("group", observed=True).median().sort_index()


def get_subplot_shape(
    n_samples: int,
    ncols: int
) -> tuple:
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


def spine_format(
    axis: plt.Axes,
    txt: str = "UMAP",
    fontsize: int = 10
) -> None:
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


def remove_extra(
    extras: int,
    nrows: int,
    ncols: int,
    axs: plt.Axes
) -> None:
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


def make_grid_spec(
    ax_or_figsize,
    *,
    nrows: int,
    ncols: int,
    wspace: float = None,
    hspace: float = None,
    width_ratios: float = None,
    height_ratios: float = None,
):
    """Modified from Scanpy.

    :param ax_or_figsize:
    :param nrows:
    :param ncols:
    :param wspace:
    :param hspace:
    :param width_ratios:
    :param height_ratios:
    :return:
    """
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


def format_terms_gsea(
    df: pd.DataFrame,
    term_col: str,
    cutoff: int = 35
)->pd.DataFrame:
    """Format Terms from Gene Set Enrichment Analysis.

    :param df: dataframe with GSEA terms.
    :param term_col: column with terms.
    :param cutoff: maximum number of characters per line.
    :return: dataframe with modified terms
    """
    import re

    def remove_whitespace_around_newlines(text):
        # Replace whitespace before and after newlines with just the newline
        return re.sub(r"\s*\n\s*", "\n", text)

    newterms = []
    for text in df[term_col]:
        newterm, text_list_nchar, nchar, limit = [], [], 0, cutoff
        text_list = text.split(" ")
        for txt in text_list:  # From text_list get a list where we sum nchar from a word + previous word
            nchar += len(txt)
            text_list_nchar.append(nchar)
        for idx, word in enumerate(text_list_nchar):
            if word > limit:  # If we have more than cutoff characters in len add a break line
                newterm.append("\n")
                limit += cutoff
            newterm.append(text_list[idx])
        newterm = " ".join(newterm)
        cleanterm = remove_whitespace_around_newlines(newterm)  # remove whitespace inserted
        newterms.append(cleanterm)
    df[term_col] = newterms

    return df


def transfer_labels(
    adata_original: ad.AnnData,
    adata_subset: ad.AnnData,
    col_original: str,
    col_subset: str,
    labels_original: list,
    copy: bool = False
)-> Union[ad.AnnData, None]:
    """Transfer annotation from a subset of an anndata.

    :param adata_original: original anndata
    :param adata_subset: subsetted anndata
    :param col_original: .obs column name in the original anndata where new labels are added
    :param col_subset: .obs column name in the subsetted object with the new labels
    :param labels_original: list of labels in the original anndata to replace
    :param copy: if copy is True, returns the updated anndata, else changes are inplace
    :return: Nothing, changes are saved inplace
    """

    if copy:
        adata_original = adata_original.copy()
        adata_subset = adata_subset.copy()
    assert adata_subset.n_obs < adata_original.n_obs, 'adata_subset is not a subset of adata_original'

    labels_original = [labels_original] if isinstance(labels_original, str) else labels_original
    adata_original.obs[col_original] = adata_original.obs[col_original].astype(str)
    adata_original.obs[col_original] = adata_original.obs[col_original].where(
        ~adata_original.obs[col_original].isin(labels_original),
        adata_original.obs.index.map(adata_subset.obs[col_subset]))

    if copy:
        return adata_original
    return None




def require_dependencies(required_packages):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            missing = []
            for pkg in required_packages:
                import_name = pkg.get('import', pkg['name'])
                try:
                    importlib.import_module(import_name)
                except ImportError:
                    missing.append(pkg['name'])

            if missing:
                print("The following packages are missing:")
                for pkg in missing:
                    print(f" - {pkg}")
                choice = input("Do you want to install them now? [y/N]: ").strip().lower()
                if choice == 'y':
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])
                else:
                    raise ImportError("Missing required packages.")

            return func(*args, **kwargs)
        return wrapper
    return decorator



def deprecated_function(func):
    """Decorator to mark a function as deprecated."""
    import warnings

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"{func.__name__} is deprecated and cannot be called.",
            category=DeprecationWarning,
            stacklevel=2
        )
        raise DeprecatedFunctionError(f"{func.__name__} is no longer available.")
    return wrapper


def timer(func):
    """Decorator to measure how much time a function took to run.

    :param func: function
    :return:
    """
    import time
    
    def _timer(*args, **kwargs):
        start_time = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            time_taken = time.time() - start_time
            print(f"----Run {func.__name__} in {time_taken:0.2f} s ----\n")

    return _timer
