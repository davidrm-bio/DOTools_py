from importlib.metadata import version

from . import pl, pp, tl, dt, utility, settings

__all__ = ["pl", "pp", "tl", "dt", "utility", "settings"]

__version__ = version("DOTools_py")

if __name__ != "__sphinx__":
    settings.session_settings()
