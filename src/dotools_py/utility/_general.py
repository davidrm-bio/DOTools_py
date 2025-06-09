def free_memory():
    """Garbage collector.

    :return: None
    """
    import ctypes
    import gc

    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    return

