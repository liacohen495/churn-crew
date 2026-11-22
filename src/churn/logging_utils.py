"""One logger for the whole project: prints to the console and writes logs/pipeline.log."""
import logging

from .config import LOG_DIR

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(f"churn.{name}")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_DIR / "pipeline.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter(_FMT))
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(_FMT))
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger
