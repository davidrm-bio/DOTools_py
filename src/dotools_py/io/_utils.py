from dotools_py._custom_class import  InputError


def _check_backend(backend: str, valid_keys: list):
    if backend not in valid_keys:
        raise  InputError(f"{backend} is not a valid key for backend. Use: {valid_keys}")
