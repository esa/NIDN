from dotmap import DotMap
import torch
from loguru import logger

from .init_trcwa import _init_trcwa


def compute_spectrum_trcwa(eps_grid, run_cfg: DotMap):
    """Evaluates TRCWA for the given epsilon values and run configuration.

    Args:
        eps_grid (torch.tensor): Epsilon values to run TRCWA with.
        run_cfg (DotMap): Run configuration.

    Raises:
        ValueError: Raises an error if epsilon contains NaN values

    Returns:
        tuple of lists: Reflection coefficients, transmission coefficients for the given epsilon values and frequencies
    """
    logger.trace("Computing spectrum")
    produced_R_spectrum = []
    produced_T_spectrum = []

    logger.debug("Testing input for NaNs")
    # Check for NaNs in eps_grid
    if torch.any(torch.isnan(eps_grid)):
        raise ValueError("Model output became NaN...", eps_grid)

    logger.debug("Checking frequencies were passed")
    assert "target_frequencies" in run_cfg, "No frequencies were passed."
    assert len(run_cfg.target_frequencies) == run_cfg.N_freq

    logger.debug("Checking passed eps_grid shape matches cfg shape")
    assert eps_grid.shape[0] == run_cfg.Nx
    assert eps_grid.shape[1] == run_cfg.Ny
    assert eps_grid.shape[2] == run_cfg.N_layers
    assert eps_grid.shape[3] == run_cfg.N_freq

    logger.debug("Iterating over passed frequencies")
    # For each frequency, evaluate TRCWA and with the grid of epsilon values
    for idx, freq in enumerate(run_cfg.target_frequencies):

        # Create TRCWA Object for this frequency
        trcwa = _init_trcwa(eps_grid[:, :, :, idx], freq, run_cfg)

        # Compute the spectrum
        reflectance, transmittance = trcwa.RT_Solve(normalize=1)

        produced_R_spectrum.append(reflectance)
        produced_T_spectrum.append(transmittance)

    return produced_R_spectrum, produced_T_spectrum
