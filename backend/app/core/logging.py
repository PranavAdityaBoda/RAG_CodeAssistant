"""
One logging setup, shared by every module.
"""
import logging
import sys


def _configure_root_logger() -> None:
    root = logging.getLogger()
    if root.handlers:
        return  # already configured, avoid duplicate handlers on reload

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    _configure_root_logger()
    return logging.getLogger(name)
