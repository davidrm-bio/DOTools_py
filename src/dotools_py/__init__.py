from importlib.metadata import version

from ._settings import settings
from . import pl, pp, tl, dt, utility

__all__ = ["pl", "pp", "tl", "dt", "utility"]

__version__ = version("DOTools_py")

