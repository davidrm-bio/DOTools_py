import dotools_py as do



def test_logger():
    do.settings.session_settings(verbosity=0)
    do.settings.session_settings(verbosity=1)
    do.settings.session_settings(verbosity=3)

    from dotools_py import logger

    logger.debug("Test")
    return

