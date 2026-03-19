from collections.abc import Iterable

from pathlib import Path

import anndata as ad

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



def check_r_package(package: str | list) -> None:
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
