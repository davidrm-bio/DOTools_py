import logging
import sys

logger = logging.getLogger("dotools")


def _setup_logger() -> None:
    """Logger settings.

    :return:
    """
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    set_verbosity(2)
    return None


def set_verbosity(level: int = 2) -> None:
    """Set verbosity.

    :param level: 0 - Silent;
                  1 - Info/Warnings;
                  2 - Info/Warnings + Scanpy Info/Warnings;
                  3 - Debug mode
    :return:
    """
    import scanpy as sc

    if level == 0:
        # Completely silent
        logger.setLevel(logging.CRITICAL + 1)  # Higher than CRITICAL
        sc.settings.verbosity = 0
        #sc.settings._root_logger.setLevel(logging.CRITICAL + 1)
    elif level == 1:
        logger.setLevel(logging.INFO)
        sc.settings.verbosity = 0
        #sc.settings._root_logger.setLevel(logging.CRITICAL + 1)
    elif level == 2:
        logger.setLevel(logging.INFO)
        #sc.settings._root_logger.setLevel(logging.INFO)
        sc.settings.verbosity = 3
    elif level == 3:
        logger.setLevel(logging.DEBUG)
        sc.settings.verbosity = 4
        #sc.settings._root_logger.setLevel(logging.DEBUG)
    else:
        raise ValueError("Verbosity must be 0, 1, 2, or 3.")
    return None


# ---- Custom logging functions ----


def info(msg: str):
    """Produce an info message.

    :param msg:
    :return:
    """
    logger.info(msg)


def warn(msg: str):
    """Produce a warn message.

    :param msg:
    :return:
    """
    logger.warning(msg)
    return None


def debug(msg: str):
    """Produce a debug message.

    :param msg:
    :return:
    """
    logger.debug(msg)
    return None


_setup_logger()
