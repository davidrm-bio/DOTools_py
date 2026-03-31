import os
import pathlib
from typing import Literal


class DeprecatedFunctionError(Exception):
    pass


class InputError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(message)

    def __str__(self):
        return f"{self.message}"


class EmptyType:
    """A singleton sentinel representing an 'empty' value."""
    def __repr__(self) -> Literal["Empty"]:
        return "Empty"


PathLike = str | os.PathLike[str] | pathlib.Path
