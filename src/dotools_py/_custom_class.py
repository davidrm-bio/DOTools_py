from typing import Literal


class DeprecatedFunctionError(Exception):
    pass


class InputError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(message)


class EmptyType:
    """A singleton sentinel representing an 'empty' value."""
    def __repr__(self) -> Literal["Empty"]:
        return "Empty"
