import functools
import importlib
from pathlib import Path
from collections.abc import Iterable

import anndata as ad
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class DeprecatedFunctionError(Exception):
    pass


def get_paths_utils(script: str) -> Path:
    """Get path for a script within the project.

    :param script: name of the script in util_scripts
    :return:
    """
    module_dir = Path(__file__).parent
    return (module_dir / "util_scripts" / script).resolve()


def convert_path(path: Path | str) -> Path:
    """Convert to Path format if string is provided.

    :param path: string or Path variable.
    :return: Path
    """
    if not isinstance(path, Path):
        return Path(path)
    else:
        return path


def sanitize_anndata(adata: ad.AnnData) -> None:
    """Transform string metadata to categorical.

    :param adata: AnnData
    :return None
    """
    adata._sanitize()
    return None


def get_centroids(adata: ad.AnnData, cluster_key: str, basis: str = "X_umap") -> pd.DataFrame:
    """Get centroids for clusters in anndata object.

    :param adata: AnnData.
    :param cluster_key: obs column with categorical information.
    :param basis: embedding to use.
    :return: centroids as a panda dataframe.
    """
    all_pos = pd.DataFrame(adata.obsm[basis], columns=["x", "y"])
    all_pos["group"] = adata.obs[cluster_key].values
    return all_pos.groupby("group", observed=True).median().sort_index()


def get_subplot_shape(n_samples: int, ncols: int) -> tuple:
    """Compute the number of rows and columns to use for defining the figure base on a desired number of samples and columns.

    :param n_samples: number of samples to plot.
    :param ncols: number of columns to plot.
    :return: nrows, ncols, extras (extra subplots that should be hidden).
    """
    if n_samples < ncols:  # Correction
        ncols = n_samples  # Adjust plot if more cols than samples are specified
    nrows = int(np.ceil(n_samples / ncols))
    extras = nrows * ncols - n_samples  # For hiding empty subplots
    return nrows, ncols, extras


def spine_format(axis: plt.Axes, txt: str = "UMAP", fontsize: int = 10) -> None:
    """Formatting the spines for Embeddings.

    Removes the top and right spines and set the x- and y-label for the left and bottom spine
    moving them to the corner.

    :param axis: matplozlib axes object.
    :param txt: text for the embedding.
    :param fontsize: size of the text.
    :return:
    """
    axis.spines[["right", "top"]].set_visible(False)
    axis.set_xlabel(txt + "1", loc="left", fontsize=fontsize, fontweight="bold")
    axis.set_ylabel(txt + "2", loc="bottom", fontsize=fontsize, fontweight="bold")
    return


def remove_extra(extras: int, nrows: int, ncols: int, axs: plt.Axes) -> None:
    """Hide the last subplots.

    :param extras: number of subplots to remove.
    :param nrows: number of rows of the plot.
    :param ncols: number of columns of the plot.
    :param axs: matplotlib axes object.
    :return:
    """
    if extras == 0:
        return None
    else:
        for check in range(nrows * ncols - extras, nrows * ncols):
            axs[check].set_visible(False)
        return None


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
    """Taken and adapted from Scanpy.

    :param ax_or_figsize: axes or figsize.
    :param nrows: number of rows.
    :param ncols: number of columns.
    :param wspace: width space.
    :param hspace: height space.
    :param width_ratios: width ratio.
    :param height_ratios: height ratio.
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


def format_terms_gsea(df: pd.DataFrame, term_col: str, cutoff: int = 35) -> pd.DataFrame:
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
    copy: bool = False,
) -> ad.AnnData | None:
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
    assert adata_subset.n_obs < adata_original.n_obs, "adata_subset is not a subset of adata_original"

    labels_original = [labels_original] if isinstance(labels_original, str) else labels_original
    adata_original.obs[col_original] = adata_original.obs[col_original].astype(str)
    adata_original.obs[col_original] = adata_original.obs[col_original].where(
        ~adata_original.obs[col_original].isin(labels_original),
        adata_original.obs.index.map(adata_subset.obs[col_subset]),
    )

    if copy:
        return adata_original
    return None


def require_dependencies(required_packages):
    """Display required dependencies and ask if the user wants to install it.

    :param required_packages: name of the package required
    :return:
    """

    def decorator(func):
        import subprocess
        import sys
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            missing = []
            for pkg in required_packages:
                import_name = pkg.get("import", pkg["name"])
                try:
                    importlib.import_module(import_name)
                except ImportError:
                    missing.append(pkg["name"])

            if missing:
                print("The following packages are missing:")
                for pkg in missing:
                    print(f" - {pkg}")
                #choice = input("Do you want to install them now? [y/N]: ").strip().lower()
                #if choice == "y":
                subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
                #else:
                raise ImportError("Missing required packages.")

            return func(*args, **kwargs)

        return wrapper

    return decorator


def deprecated_function(func):
    """Decorator to mark a function as deprecated."""
    import warnings

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(f"{func.__name__} will be deprecated and cannot be called.", category=DeprecationWarning, stacklevel=2)
        #raise DeprecatedFunctionError(f"{func.__name__} is no longer available.")

    return wrapper


def deprecated_fxn(message=None):
    """Decorator to mark a function as deprecated.

    Args:
        message (str, optional): Custom deprecation message. If not provided,
                                 a default message will be used.
    """
    import warnings
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warn_msg = (
                message
                if message
                else f"{func.__name__} is deprecated and will be removed in a future version."
            )
            warnings.warn(warn_msg, category=DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper
    return decorator



def draw_bracket(x_start, x_end, y_bottom=0, y_top=1, stem_length=0.2):
    import matplotlib.path

    verts = [
        (x_start, y_bottom),  # Start of the bracket (bottom-left)
        (x_start, y_top),  # Vertical stem up
        (x_end - stem_length, y_top),  # Horizontal part
        (x_end - stem_length, y_bottom)  # Down to bottom-right
    ]
    codes = [matplotlib.path.Path.MOVETO, matplotlib.path.Path.LINETO,
             matplotlib.path.Path.LINETO, matplotlib.path.Path.LINETO]
    return matplotlib.path.Path(verts, codes)


def iterase_input(data: str | float | int | Iterable | None) -> list:
    """Convert input to list.

    :param data: string or iterable (list, tuple, index, etc.)
    :return: Returns a list.
    """
    if data is None:
        return []
    elif isinstance(data, str):
        return [data]
    elif isinstance(data, float):
        return [data]
    elif isinstance(data, int):
        return [data]
    elif isinstance(data, list):
        return data
    elif isinstance(data, Iterable):
        return list(data)
    else:
        raise Exception("Input is not a string or iterable object")


def check_missing(adata: ad.AnnData, features: str | list = None, groups: str | list = None,
                  variables: str | list = None) -> None:
    """Check for missing features or columns in the observations from an AnnData Object.

    :param adata: AnnData Object.
    :param features: features to check for.
    :param groups: column names in the observations to check for.
    :param variables: column names in the variables to check for.
    :return: Returns None. Will raise an assertion if any feature or column name is missing.
    """

    if features:
        features = iterase_input(features)
        missing = [g for g in features if g not in adata.var_names]

        # features could be in .obs
        missing_x2 = []
        if len(missing) > 0:
            missing_x2 = [g for g in features if g not in adata.obs.columns]

        if len(missing_x2) > len(missing):
            assert len(missing) == 0, f"{missing} missing in the AnnData Object"
        else:
            assert len(missing_x2) == 0, f"{missing_x2} missing in the AnnData Object"

    if groups:
        groups = iterase_input(groups)
        missing = [g for g in groups if g not in adata.obs.columns]
        assert len(missing) == 0, f"{missing} missing in the AnnData Object"
    if variables:
        variables = iterase_input(variables)
        missing = [g for g in variables if g not in adata.var.columns]
        assert len(missing) == 0, f"{missing} missing in the AnnData Object"

    return None


def logmean(x):
    """Calculate mean expression of log data.

    :param x: Values in log space.
    :return: Returns the mean expression in log space.
    """
    return np.log1p(np.mean(np.expm1(x)))


def logsem(x):
    """Calculate standard error of the mean of log data.

    :param x: Values in log space
    :return: Returns the SEM in log space
    """
    from scipy.stats import sem
    return np.log1p(sem(np.expm1(x)))


def save_plot(path: str | Path | None, filename: str) -> None:
    """Save a plot.

    :param path: Path to the folder where to save the plot.
    :param filename: Name of the file.
    :return: Returns None. If path is None, no plot is saved.
    """
    if path is not None:
        plt.savefig(convert_path(path) / filename, bbox_inches="tight")
    return None


def return_axis(show: bool, axis: dict | plt.Axes, tight: bool = True) -> None | plt.Axes:
    """Whether to return axis or not.

    :param show: Boolean to indicate if the axis is returned or not.
    :param axis: Dictionary of axis or axis.
    :param tight: Tight layout.
    :return: Returns None if show is True, otherwise returns the axis.
    """
    if show:
        if tight:
            plt.tight_layout()
        return  plt.show()
    else:
        return axis

