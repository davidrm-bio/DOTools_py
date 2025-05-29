import logging
import scanpy as sc


logger = logging.getLogger("dotools")

def _setup_logger() -> None:
    """
    Logger settings
    :return: None
    """
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    verbosity(2)
    return None


def verbosity(level: int = 2) -> None:
    """
    Set verbosity
    :param level: 0 - Silent;
                  1 - Info/Warnings;
                  2 - Info/Warnings + Scanpy Info/Warnings;
                  3 - Debug mode
    :return: None
    """

    from scanpy._settings import settings as sc_settings

    if level == 0:
        # Completely silent
        logger.setLevel(logging.CRITICAL + 1)  # Higher than CRITICAL
        sc.settings.verbosity = 0
        sc.settings._root_logger.setLevel(logging.CRITICAL + 1)
    elif level == 1:
        logger.setLevel(logging.INFO)
        sc.settings.verbosity = 0
        sc.settings._root_logger.setLevel(logging.CRITICAL + 1)
    elif level == 2:
        logger.setLevel(logging.INFO)
        sc.settings.verbosity = 2
        sc.settings._root_logger.setLevel(logging.INFO)
    elif level == 3:
        logger.setLevel(logging.DEBUG)
        sc.settings.verbosity = 3
        sc.settings._root_logger.setLevel(logging.DEBUG)
    else:
        raise ValueError("Verbosity must be 0, 1, 2, or 3.")
    return None


# ---- Custom logging functions ----

def info(msg: str):
    logger.info(msg)

def warn(msg: str):
    logger.warning(msg)

def debug(msg: str):
    logger.debug(msg)

_setup_logger()
