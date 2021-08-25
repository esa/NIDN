from .plot_eps_per_point import plot_eps_per_point
from .plot_losses import plot_losses
from .plot_material_grid import plot_material_grid
from .plot_model_grid import plot_model_grid
from .plot_model_grid_per_freq import plot_model_grid_per_freq
from .plot_spectra import plot_spectra

from loguru import logger


def save_all_plots(run_cfg, save_path):
    """Creates all plots for the passed run_config and save them in the passed folder.

    Args:
        run_cfg (DotMap): Run configuration (incl. trained model).
        save_path (str): Folder to save the plots to.
    """
    logger.info("Saving all plots to {}".format(save_path))

    logger.debug("Saving spectra plot")
    plot_spectra(run_cfg, save_path=save_path)

    logger.debug("Saving losses plot")
    plot_losses(run_cfg, save_path=save_path)

    logger.debug("Saving model grid plot")
    plot_model_grid(run_cfg, save_path=save_path)

    logger.debug("Saving model grid per freq plot")
    plot_model_grid_per_freq(run_cfg, save_path=save_path)

    logger.debug("Saving material grid plot")
    plot_material_grid(run_cfg, save_path=save_path)

    logger.debug("Saving eps per point plot")
    plot_eps_per_point(run_cfg, save_path=save_path)
