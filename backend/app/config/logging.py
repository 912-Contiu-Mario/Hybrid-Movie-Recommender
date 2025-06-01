import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

INFO_LOG = LOGS_DIR / "info.log"
ERROR_LOG = LOGS_DIR / "error.log"

MAX_BYTES = 5 * 1024 * 1024


BACKUP_COUNT = 1

def setup_logging():
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    info_handler = RotatingFileHandler(
        INFO_LOG,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT
    )
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(formatter)

    error_handler = RotatingFileHandler(
        ERROR_LOG,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    root_logger.addHandler(info_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(console_handler)

    logging.info("Logging system initialized") 