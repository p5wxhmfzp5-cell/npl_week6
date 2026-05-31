"""
Centralized Logging Utilities

Features
--------
- Rich console logging
- File logging
- Timestamped benchmark logs
- Consistent formatting
- Experiment-aware logger creation
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)

LOG_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "logs"
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

console = Console()


############################################################
# FORMATTERS
############################################################

LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)s | "
    "%(name)s | "
    "%(message)s"
)

DATE_FORMAT = (
    "%Y-%m-%d %H:%M:%S"
)


############################################################
# LOGGER FACTORY
############################################################

def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Create configured logger.

    Parameters
    ----------
    name:
        Logger namespace.

    level:
        Logging level.

    log_file:
        Optional file output.

    Returns
    -------
    logging.Logger
    """

    logger = logging.getLogger(
        name
    )

    logger.setLevel(level)

    if logger.handlers:

        return logger

    ########################################################
    # Rich Console Handler
    ########################################################

    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        markup=True,
        show_path=False,
    )

    rich_handler.setLevel(
        level
    )

    rich_handler.setFormatter(
        logging.Formatter(
            fmt=LOG_FORMAT,
            datefmt=DATE_FORMAT,
        )
    )

    logger.addHandler(
        rich_handler
    )

    ########################################################
    # Optional File Handler
    ########################################################

    if log_file:

        file_path = (
            LOG_DIR / log_file
        )

        file_handler = (
            logging.FileHandler(
                file_path,
                encoding="utf-8",
            )
        )

        file_handler.setLevel(
            level
        )

        file_handler.setFormatter(
            logging.Formatter(
                fmt=LOG_FORMAT,
                datefmt=DATE_FORMAT,
            )
        )

        logger.addHandler(
            file_handler
        )

    logger.propagate = False

    return logger


############################################################
# EXPERIMENT LOGGER
############################################################

def get_experiment_logger(
    experiment_name: str,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create experiment-scoped logger.

    Example
    -------
    outputs/logs/msmarco_full.log
    """

    filename = (
        f"{experiment_name}.log"
    )

    return get_logger(
        name=f"experiment.{experiment_name}",
        level=level,
        log_file=filename,
    )


############################################################
# LOGGING HELPERS
############################################################

def banner(
    logger: logging.Logger,
    title: str,
) -> None:
    """
    Pretty section banner.
    """

    separator = "=" * 70

    logger.info(
        "\n%s\n%s\n%s",
        separator,
        title.upper(),
        separator,
    )


def log_metrics(
    logger: logging.Logger,
    metrics: dict,
) -> None:
    """
    Pretty-print metrics.
    """

    logger.info(
        "Benchmark Metrics"
    )

    for k, v in metrics.items():

        logger.info(
            "%s: %s",
            k,
            v,
        )


############################################################
# EXPORTS
############################################################

__all__ = [

    "get_logger",

    "get_experiment_logger",

    "banner",

    "log_metrics",
]