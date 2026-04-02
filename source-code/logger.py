# logger.py
import logging
import time
from contextlib import contextmanager

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-22s | %(message)s"
DATE_FORMAT = "%H:%M:%S"

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)


def setup_logger(name: str, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


@contextmanager
def trace_step(logger: logging.Logger, step: str, **kwargs):
    """Context manager — logs entry, exit and elapsed time for any block."""
    extra = "  ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.info(f"┌─ START  {step}  {extra}")
    t0 = time.perf_counter()
    try:
        yield
        elapsed = round(time.perf_counter() - t0, 3)
        logger.info(f"└─ DONE   {step}  elapsed={elapsed}s")
    except Exception as e:
        elapsed = round(time.perf_counter() - t0, 3)
        logger.error(f"└─ ERROR  {step}  elapsed={elapsed}s  error={e}")
        raise
