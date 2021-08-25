import pickle as pk
from datetime import datetime
from loguru import logger


def save_run(run_cfg):
    """Saves results of a run to a file.

    Args:
        run_cfg (DotMap): Run config.
    """

    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H-%M-%S")

    filename = "../results/" + dt_string + "_run.pk"

    logger.info("Saving run to file: {}".format(filename))
    with open(filename, "wb") as handle:
        pk.dump(run_cfg, handle, protocol=pk.HIGHEST_PROTOCOL)
