from pathlib import Path


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
