import os
import sys
from datetime import datetime

from dotmap import DotMap
import toml
from glob import glob
from loguru import logger

sys.path.append("../")
import nidn

logger.add(
    sys.stderr,
    colorize=True,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green>|NIDN-RUN-<blue>{level}</blue>| <level>{message}</level>",
    filter="__main__",
)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Usage: python run_cfgs.py <config_folder>")
        sys.exit(1)

    # Get target filepath from command line arguments
    target_filepath = sys.argv[1]

    # Find all cfg files in the target filepath
    cfg_filepaths = glob(target_filepath + "/*.toml")

    logger.info("Found {} cfg files in {}".format(len(cfg_filepaths), target_filepath))

    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H-%M-%S")
    result_folder_name = dt_string + "_results"
    logger.info(f"Results will be stored in results/{result_folder_name}")

    # Run training for all cfg files
    for cfg_filepath in cfg_filepaths:
        logger.info("Running {}".format(cfg_filepath))

        # Load cfg file
        with open(cfg_filepath) as cfg:
            cfg = DotMap(toml.load(cfg))

        logger.info("Loaded:")
        nidn.print_cfg(cfg)

        # Run training
        nidn.run_training(cfg)

        logger.info("Finished running {}".format(cfg_filepath))

        logger.info("Storing results...")
        subfolder_name = cfg.name
        nidn.save_run(cfg, result_folder_name + "/" + subfolder_name + "/")
        nidn.save_all_plots(
            cfg, "../results/" + result_folder_name + "/" + subfolder_name + "/"
        )

        logger.info("Finished storing results")

    logger.info("All runs finished!")
