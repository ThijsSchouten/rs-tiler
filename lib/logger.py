import logging
import time


def configure_logger(name=__name__, filename="logfile.log", clear=False):
    if clear:
        with open(filename, "w"):
            pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] → %(lineno)03d   %(message)s",
        datefmt="%d.%m %H:%M",
        handlers=[logging.FileHandler(filename), logging.StreamHandler()],
    )

    logger = logging.getLogger(name)
    return logger
