import os
from importlib.metadata import version

from . import dt, pl, pp, settings, tl, utility, get

__all__ = ["pl", "pp", "tl", "dt", "get", "utility", "settings"]

__version__ = version("DOTools_py")


def _is_run_from_ipython():
    """Determine whether we're currently in IPython."""
    import builtins

    return getattr(builtins, "__IPYTHON__", False)


def is_interactive():
    try:
        if _is_run_from_ipython():
            return True
        from IPython import get_ipython

        ip = get_ipython()
        return ip is not None and ip.__class__.__name__ in ("ZMQInteractiveShell", "TerminalInteractiveShell")
    except ImportError:
        return False


if not is_interactive():
    settings.session_settings(verbosity=0, interactive=False)
if is_interactive() and os.environ.get("READTHEDOCS", "").lower() != "true":
    settings.session_settings()
