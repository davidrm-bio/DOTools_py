from importlib.metadata import version

from . import  pl, pp, tl
from ._settings import set_plt_theme, iOn

__all__ = ["pl", "pp", "tl"]

__version__ = version("DOTools_py")

set_plt_theme()
iOn()
