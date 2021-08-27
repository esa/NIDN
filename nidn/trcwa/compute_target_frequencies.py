import numpy as np

from ..utils.convert_units import phys_wl_to_wl, freq_to_wl


def compute_target_frequencies(
    min_physical_wl: float, max_physical_wl: float, N_freq: int, distribution: str
):
    """Computes the target frequencies for a given set of physical frequencies.

    Args:
        min_physical_wl (float): Minimum physical frequency in the target spectrum.
        max_physical_wl (float): Maximum physical frequency in the target spectrum.
        N_freq (int): The number of target frequencies.
        distribution (str): The distribution of the target frequencies. Either linear or log.

    Returns:
        np.array: Corresponding target frequencies
    """
    max_physical_freq = freq_to_wl(phys_wl_to_wl(min_physical_wl))
    min_physical_freq = freq_to_wl(phys_wl_to_wl(max_physical_wl))

    logval_min = np.log10(min_physical_freq)
    logval_max = np.log10(max_physical_freq)

    if distribution == "linear":
        target_frequencies = np.linspace(
            min_physical_freq, max_physical_freq, num=N_freq
        )
    elif distribution == "log":
        target_frequencies = np.logspace(logval_min, logval_max, num=N_freq, base=10)
    else:
        raise ValueError(
            "Unknown distribution type: {}. Needs to be 'log' or 'linear'.".format(
                distribution
            )
        )

    target_frequencies = np.sort(target_frequencies)
    return target_frequencies
