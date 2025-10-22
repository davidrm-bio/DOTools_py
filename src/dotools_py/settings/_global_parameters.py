import platform

from dotools_py import logger

FAST_ARRAY_UTILS = False

py_version = platform.python_version()  # Check the Python Version
if int(py_version.split(".")[1]) > 10:
    try:
        import fast_array_utils
        FAST_ARRAY_UTILS = True
    except ImportError as e:
        logger.warn("Python > 3.10 but fast_array_utils not available, consider installing it to speed up"
                    "analysis")


