import logging
import os

from enum import Enum
from logging.handlers import RotatingFileHandler

# Variables of possibles types of logs
log_type_debug: str = "DEBUG"
log_type_error: str = "ERROR"
log_type_info: str = "INFO"
log_type_warning: str = "WARNING"


class LogTypes(Enum):
    DEBUG: str = log_type_debug
    ERROR: str = log_type_error
    INFO: str = log_type_info
    WARNING: str = log_type_warning


class BasicLogger:
    def __init__(
            self,
            log_file: str,
            log_name: str = "Generic_Logger",
            max_log_size: int = (5 * 1024 * 1024),
            backup_count: int = 3,
    ):
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(logging.INFO)

        # Create a formatter to add the time, name, level and message of the log
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Create a file handler to store logs in a file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_log_size, backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Create a stream handler to print logs in the console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger
