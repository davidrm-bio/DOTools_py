from importlib.metadata import version

from . import pl, pp, tl
from . import _settings

__all__ = ["pl", "pp", "tl"]

__version__ = version("DOTools_py")
