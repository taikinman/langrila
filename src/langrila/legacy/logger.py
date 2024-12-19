import datetime
import logging
import os
from pathlib import Path


class DefaultLogger:
    def __init__(
        self,
        path: str | Path | None = None,
        level: str = "DEBUG",
        include_timestamp: bool = False,
    ):
        if path is not None:
            path = Path(path)
            if path.exists():
                # recreate log file
                os.remove(path)

            path.parent.mkdir(parents=True, exist_ok=True)

            self.general_logger = logging.getLogger(path.as_posix())
        else:
            self.general_logger = logging.getLogger(__name__)

        # clear handlers
        if len(self.general_logger.handlers) > 0:
            self.general_logger.handlers.clear()

        stream_handler = logging.StreamHandler()
        self.general_logger.addHandler(stream_handler)

        if path is not None:
            file_general_handler = logging.FileHandler(path.as_posix())
            self.general_logger.addHandler(file_general_handler)

        self.general_logger.setLevel(getattr(logging, level))

        self.include_timestamp = include_timestamp

    def critical(self, message: str):
        if self.include_timestamp:
            message = self._add_timestamp(message)
            self.general_logger.critical(message)
        else:
            self.general_logger.critical(message)

    def fatal(self, message: str):
        if self.include_timestamp:
            message = self._add_timestamp(message)
            self.general_logger.fatal(message)
        else:
            self.general_logger.fatal(message)

    def error(self, message: str):
        if self.include_timestamp:
            message = self._add_timestamp(message)
            self.general_logger.error(message)
        else:
            self.general_logger.error(message)

    def warning(self, message: str):
        if self.include_timestamp:
            message = self._add_timestamp(message)
            self.general_logger.warning(message)
        else:
            self.general_logger.warning(message)

    def info(self, message: str):
        if self.include_timestamp:
            message = self._add_timestamp(message)
            self.general_logger.info(message)
        else:
            self.general_logger.info(message)

    def debug(self, message: str):
        if self.include_timestamp:
            message = self._add_timestamp(message)
            self.general_logger.debug(message)
        else:
            self.general_logger.debug(message)

    def _add_timestamp(self, message: str):
        return f"[{self.now_string}] {message}"

    @property
    def now_string(self):
        return str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
