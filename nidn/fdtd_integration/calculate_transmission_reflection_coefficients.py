import torch
from loguru import logger
import warnings
from scipy.signal import find_peaks


def calculate_transmission_reflection_coefficients(
    transmission_signals,
    reflection_signals,
):
    """Calculates the transmission coefficient and reflection coefficient for the signals presented.

    Args:
        transmission_signals (tuple[array,array]): Transmission signal from a free-space fdtd simulaiton, and transmission signal from a fdtd simulation with an added object
        reflection_signals (_type_): Reflection signal from a free-space fdtd simulaiton, and reflection signal from a fdtd simulation with an added object

    Returns:
        tuple[float, float]: Transmission coefficient and reflection coefficient
    """
    # Substract the free_space reflection signal from the material reflection signal, to eliminate unreflected signal from detector
    # The detector detects signal passing through both ways, and is placed between the source and the material.
    # Thus, most of the signal present is the unreflected signal, which must be removed.
    true_reflection = reflection_signals[1] - reflection_signals[0]

    _check_for_all_zero_signal(transmission_signals)
    _check_for_all_zero_signal(reflection_signals)

    # find peaks for all signals
    peaks_transmission_freespace = find_peaks(transmission_signals[0])[0]
    peaks_transmission_material = find_peaks(transmission_signals[1])[0]
    peaks_reflection_freespace = find_peaks(reflection_signals[0])[0]
    peaks_reflection_material = find_peaks(true_reflection)[0]

    transmission_coefficient = _mean_square(
        transmission_signals[1][
            peaks_transmission_material[0] : peaks_transmission_material[-1]
        ]
    ) / _mean_square(
        transmission_signals[0][
            peaks_transmission_freespace[0] : peaks_transmission_freespace[-1]
        ]
    )
    reflection_coefficient = _mean_square(
        true_reflection[peaks_reflection_material[0] : peaks_reflection_material[-1]]
    ) / _mean_square(
        reflection_signals[0][
            peaks_reflection_freespace[0] : peaks_reflection_freespace[-1]
        ]
    )

    """if transmission_coefficient < 0 or transmission_coefficient > 1:
        raise ValueError(
            f"The transmission coefficient is outside of the physical range between 0 and 1"
        )

    if reflection_coefficient < 0 or reflection_coefficient > 1:
        raise ValueError(
            f"The reflection coefficient is outside of the physical range between 0 and 1"
        )"""
    if transmission_coefficient + reflection_coefficient > 1:
        logger.warning(
            f"The sum of the transmission and reflection coefficient is greater than 1, which is physically impossible"
        )
    return transmission_coefficient, reflection_coefficient


def _mean_square(tensor):
    """Calculates the mean of the squared signal
    Args:
        tensor (tensor): signal to perform the calculations on

    Returns:
        torch.float: The mean square value
    """
    return torch.sum(torch.square(tensor)) / len(tensor)


def _check_for_all_zero_signal(signals):

    if _mean_square(signals[0]) <= 0:
        raise ValueError(
            "The freen space signal is all zero. Increase the number of FDTD_niter to ensure that the signal reaches the detctor."
        )
    if _mean_square(signals[1]) <= 0:
        warnings.warn(
            "WARNING:The signal trough the material layer(s) never reaches the detector. Increase FDTD_niter to ensure that the signal reaches the detector. The signal usually travels slower in a material than in free space "
        )
