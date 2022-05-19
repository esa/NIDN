import pickle as pk
from datetime import datetime
from loguru import logger

from pathlib import Path


def save_run(run_cfg, subfolder=""):
    """Saves results of a run to a file.

    Args:
        run_cfg (DotMap): Run config.
        subfolder (str): Subfolder to save to. Defaults to "".
    """

    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H-%M-%S")

    # Create subfolder if it does not exist
    Path("../results/" + subfolder + "/").mkdir(parents=True, exist_ok=True)

    filename = "../results/" + subfolder + "/" + run_cfg.name + "_" + dt_string + "_run.pk"

    logger.info("Saving run to file: {}".format(filename))
    with open(filename, "wb") as handle:
        pk.dump(run_cfg, handle, protocol=pk.HIGHEST_PROTOCOL)
