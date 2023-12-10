import datetime
import logging
import os


class DefaultLogger:
    def __init__(self, path: str, name: str = "log", level: str = "INFO"):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        self.general_logger = logging.getLogger(path)
        stream_handler = logging.StreamHandler()
        file_general_handler = logging.FileHandler(os.path.join(path, f"{name}.txt"))
        if len(self.general_logger.handlers) == 0:
            self.general_logger.addHandler(stream_handler)
            self.general_logger.addHandler(file_general_handler)
            self.general_logger.setLevel(getattr(logging, level))

    def info(self, message):
        # display time
        self.general_logger.info("[{}] - {}".format(self.now_string(), message))

    def debug(self, message):
        # display time
        self.general_logger.debug("[{}] - {}".format(self.now_string(), message))

    @staticmethod
    def now_string():
        return str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
