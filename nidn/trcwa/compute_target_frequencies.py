import numpy as np


def _compute_target_frequencies(physical_frequencies: np.array):
    """Computes the target frequencies for a given set of physical frequencies.

    Args:
        physical_frequencies (np.array): A set of physical frequencies.

    Returns:
        np.array: Corresponding target frequencies.
    """
    logvals = np.log10(physical_frequencies)
    target_frequencies = np.flip(
        np.logspace(logvals[0], logvals[-1], num=len(physical_frequencies), base=10)
    )
    return target_frequencies
