"""
logging_config.py — Project-wide structured logging setup.

Call ``setup_logging()`` once at program entry (main.py / script entrypoints).
All other modules should obtain their logger via:

    import logging
    logger = logging.getLogger(__name__)
"""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with a consistent format.

    Args:
        level: Logging level (e.g. ``logging.DEBUG``, ``logging.INFO``).
    """
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Suppress noisy third-party loggers
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("absl").setLevel(logging.ERROR)
