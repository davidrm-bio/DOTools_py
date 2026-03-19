import functools
import importlib

from collections.abc import Iterable
from typing import Any
from pathlib import Path

import anndata as ad
import numpy as np

from dotools_py._custom_class import PathLike, InputError
import matplotlib.pyplot as plt


def convert_path(path: PathLike) -> Path:
    """Convert to Path format if string is provided.

    :param path: string or Path variable or PathLike variable.
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


def iterase_input(data: str | float | int | Iterable | None) -> list:
    """Transform input to a list.

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
    elif isinstance(data, plt.Axes):
        return [data]
    else:
        raise InputError("Input is not a string or iterable object")


def check_missing(
    adata: ad.AnnData,
    features: str | list = None,
    groups: str | list = None,
    variables: str | list = None
) -> None:
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


def get_paths_utils(script: str) -> PathLike:
    """Get path for a script within the project.

    :param script: name of the script in util_scripts
    :return:
    """
    module_dir = Path(__file__).parent
    return (module_dir / "util_scripts" / script).resolve()



def check_r_package(package: PathLike) -> None:
    from rpy2.robjects.packages import importr

    package = iterase_input(package)

    missing = []
    for p in package:
        try:
            _ = importr(p)
        except Exception:
            missing.append(p)
    if len(missing) != 0:
        raise ModuleNotFoundError(f"The R packages: {missing} are not installed")
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
                subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
                raise ImportError("Missing required packages.")

            return func(*args, **kwargs)

        return wrapper

    return decorator


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


def is_none(variable: Any, default: Any = None):
    return variable if variable is not None else default


def x_is_raw_counts(adata: ad.AnnData) -> None:
    from scipy.sparse import issparse
    matrix = adata.X.data if issparse(adata.X) else adata.X.flatten()
    if (matrix % 1 != 0).any():
        raise ValueError("The count matrix should only contain integers.")
    if (matrix < 0).any():
        raise ValueError("The count matrix should only contain non-negative values.")


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


