import datetime
import logging
import os


class Logger:
    def __init__(self, path: str, name: str = "log", level: str = "DEBUG", include_timestamp: bool = False):
        path_log = os.path.join(path, f"{name}.log")
        if os.path.exists(path_log):
            os.remove(path_log)

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        self.general_logger = logging.getLogger(path_log)
        stream_handler = logging.StreamHandler()
        file_general_handler = logging.FileHandler(path_log)

        if len(self.general_logger.handlers) > 0:
            self.general_logger.handlers.clear()

        self.general_logger.addHandler(stream_handler)
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
        return f"[{self.now_string()}] {message}"

    @staticmethod
    def now_string():
        return str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
