import pickle as pk
from datetime import datetime
from loguru import logger


def load_run(filename):
    """Loads the saved DotMap run config.

    Args:
        filename (str): Path to file.
    Returns:
        DotMap: Run config
    """
    with open(filename, "rb") as f:
        run = pk.load(f)
    logger.info(f"Loaded run config from {filename}")
    return run
