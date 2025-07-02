from importlib.metadata import version
import os

from . import pl, pp, tl, dt, utility, settings

__all__ = ["pl", "pp", "tl", "dt", "utility", "settings"]

__version__ = version("DOTools_py")

def is_interactive():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and ip.__class__.__name__ in ("ZMQInteractiveShell", "TerminalInteractiveShell")
    except ImportError:
        return False

if is_interactive() and os.environ.get("READTHEDOCS", "").lower() != "true":
    settings.session_settings()
