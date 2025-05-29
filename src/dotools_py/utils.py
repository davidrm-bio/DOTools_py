from pathlib import  Path

from typing import Union


def get_paths_utils(script: str):
    module_dir = Path(__file__).parent
    return (module_dir / 'util_scripts' / script).resolve()


def convert_path(path: Union[Path, str]) -> Path:
    """
    Convert to Path format if string is provided
    :param path: string or Path variable
    :return: path
    """
    if not isinstance(path, Path):
        return Path(path)
    else:
        return path
