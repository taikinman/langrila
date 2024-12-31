import sys
from typing import TextIO

from loguru import logger
from loguru._logger import Logger

# from loguru._logger import Logger


def get_logger(
    sink: TextIO = sys.stderr,
    level: str = "DEBUG",
) -> Logger:
    logger.remove()
    logger.add(
        sink,
        level=level,
        format="<green>[{time:YYYY-MM-DD HH:mm:ss}]</green> <level>{level} | {message}</level>",
    )
    return logger


DEFAULT_LOGGER = get_logger()
