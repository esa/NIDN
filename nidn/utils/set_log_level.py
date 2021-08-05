from loguru import logger
import sys


def set_log_level(log_level: str):
    """Set the log level for the logger

    Args:
        log_level (str): The log level to set. Options are 'TRACE','DEBUG', 'INFO', 'SUCCESS', 'WARNING', 'ERROR', 'CRITICAL'
    """
    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        level=log_level,
        format="{time:HH:mm:ss}|NIDN-{level}| {message}",
        filter="nidn",
    )
    logger.debug(f"Setting LogLevel to {log_level}")
