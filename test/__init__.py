from pathlib import Path
from src.logger import app_logger

app_logger.info("Setting up the test environment")
ROOT_PATH = Path(__file__).parent.parent

