from dotmap import DotMap
import torch

from .init_trcwa import _init_trcwa


def compute_spectrum(eps_grid, run_cfg: DotMap):
    """Evaluates TRCWA for the given epsilon values and run configuration.

    Args:
        eps_grid (torch.tensor): Epsilon values to evaluate TRCWA.
        run_cfg (DotMap): Run configuration.

    Raises:
        ValueError: Raises an error epsilon contains NaN values.

    Returns:
        tuple of lists: Reflection coefficients, transmission coefficients for the given epsilon values and frequencies.
    """
    produced_R_spectrum = []
    produced_T_spectrum = []

    # Check for NaNs in eps_grid
    if torch.any(torch.isnan(eps_grid)):
        raise ValueError("Model output became NaN...", eps_grid)

    # For each frequency, evaluate TRCWA and with the grid of epsilon values
    for idx, freq in enumerate(run_cfg.target_frequencies):

        # Create TRCWA Object for this frequency
        trcwa = _init_trcwa(eps_grid[:, :, :, idx], freq)

        # Compute the spectrum
        reflectance, transmittance = trcwa.RT_Solve(normalize=1)

        produced_R_spectrum.append(reflectance)
        produced_T_spectrum.append(transmittance)

    return produced_R_spectrum, produced_T_spectrum
