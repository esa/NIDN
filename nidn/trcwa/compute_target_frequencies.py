import numpy as np

from ..utils.convert_units import phys_wl_to_wl, freq_to_wl


def _compute_target_frequencies(
    min_physical_wl: float, max_physical_wl: float, N_freq: int
):
    """Computes the target frequencies for a given set of physical frequencies.

    Args:
        min_physical_wl (float): Minimum physical frequency in the target spectrum.
        max_physical_wl (float): Maximum physical frequency in the target spectrum.
        N_freq (int): The number of target frequencies.

    Returns:
        np.array: Corresponding target frequencies
    """
    max_physical_freq = freq_to_wl(phys_wl_to_wl(min_physical_wl))
    min_physical_freq = freq_to_wl(phys_wl_to_wl(max_physical_wl))

    logval_min = np.log10(min_physical_freq)
    logval_max = np.log10(max_physical_freq)

    target_frequencies = np.flip(
        np.logspace(logval_min, logval_max, num=N_freq, base=10)
    )
    return target_frequencies
