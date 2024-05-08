import os

from src.logger.basic_logger import BasicLogger

app_logger = BasicLogger(
    log_file=os.path.join(os.getcwd(), "logs", "backend.log")
).get_logger()
