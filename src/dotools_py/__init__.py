from importlib.metadata import version
import os

from . import pl, pp, tl, dt, utility, settings

__all__ = ["pl", "pp", "tl", "dt", "utility", "settings"]

__version__ = version("DOTools_py")

if os.environ.get("READTHEDOCS") != "True":
    settings.session_settings()
