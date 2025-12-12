import os.path

import dotools_py as do



def test_logger():
    do.settings.session_settings(verbosity=0)
    do.settings.session_settings(verbosity=1)
    do.settings.session_settings(verbosity=3)

    from dotools_py import logger

    logger.debug("Test")

    do.settings.set_kernel_logger("./history.log")
    assert os.path.exists("./history.log")
    do.settings.toogle_kernel_logger(False)
    do.settings.toogle_kernel_logger(True)
    do.settings.toogle_kernel_logger(False)
    os.remove("./history.log")


    return

