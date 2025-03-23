import logging
import os
from pathlib import Path
from typing import Union

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PATH_LOGS = os.path.join(PROJECT_DIR, "logs")

def setup_logging_to_file(
    log_dir: Union[str, Path] = PATH_LOGS,
    format_str: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> logging.Logger:
    """Sets up a logger that writes to a file and the console."""

    # Ensure the logging directory exists
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Get a logger instance
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create a file handler
    log_file_path = Path(log_dir) / f"{__name__}.log"
    file_handler = logging.FileHandler(log_file_path, mode="a")
    file_handler.setFormatter(logging.Formatter(format_str))
    logger.addHandler(file_handler)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(format_str))
    logger.addHandler(console_handler)

    return logger

# Initialize the logger
logger = setup_logging_to_file()
logger.info("Logger is ready")
logger.info("This is a test log message.")