import logging
import os
from logging.handlers import RotatingFileHandler


def get_logger(name, level=logging.INFO):
    if not os.path.isdir('./logs'):
        os.makedirs('./logs')

    log_file = os.path.join('./logs', 'project.log')

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)

    fmt = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    formatter = logging.Formatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=1,
        encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
