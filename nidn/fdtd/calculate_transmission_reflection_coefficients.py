from dotmap import DotMap
from torch.fft import rfft, rfftfreq
import numpy as np
import torch
import numpy

from nidn.utils.global_constants import SPEED_OF_LIGHT


def calculate_transmission_reflection_coefficients(
    transmission_signals,
    reflection_signals,
    time_to_frequency_domain_method,
    cfg: DotMap,
):
    """Calculates the transmission coefficient and reflection coefficient for the signals presented.

    Args:
        transmission_signals (tuple[array,array]): Transmission signal from a free-space fdtd simulaiton, and transmission signal from a fdtd simulation with an added object
        reflection_signals (_type_): Reflection signal from a free-space fdtd simulaiton, and reflection signal from a fdtd simulation with an added object
        time_to_frequency_domain_method (string): "MEAN SQUARE" if the mean square is to be used to calculate the coefficients. "FOURIER TRANSFORM" if the fft spectrums should be compared to calculate the coefficients
        cfg (DotMap): configuration used in the simulations which generated the signals.

    Returns:
        tuple[float, float]: Transmission coefficient and reflection coefficient
    """
    # If method == 'MEAN SQUARE, then use the mean square to get the coefficients
    # If method == 'FT', use fourier transform method

    if time_to_frequency_domain_method.upper() == "MEAN SQUARE":
        transmission_coefficient = _mean_square(transmission_signals[1]) / _mean_square(
            transmission_signals[0]
        )
        reflection_coefficient = _mean_square(reflection_signals[1]) / _mean_square(
            reflection_signals[0]
        )
    elif time_to_frequency_domain_method.upper() == "FOURIER TRANSFORM":
        transmission_coefficient = (
            _fft(transmission_signals[1], cfg)
            / _fft(transmission_signals[0], cfg)
            * np.exp()
        )  # Some exponential should be multiplied here
        reflection_coefficient = (
            _fft(reflection_signals[1], cfg)
            / _fft(reflection_signals[0], cfg)
            * np.exp()
        )
    return transmission_coefficient, reflection_coefficient


def _mean_square(arr):
    """Calculates the mean of the squared signal
    Args:
        arr (array): signal to perform the calculations on

    Returns:
        float: The mean square value
    """
    return sum([e**2 for e in arr]) / len(arr)


def _fft(signal, cfg: DotMap):
    """Calculates the fast fourier transform of the signal using torch

    Args:
        signal (array): signal to perform the fft on
        cfg (DotMap): configurations used in the simulation which produced the signal

    Returns:
        tuple[array,array]: fourier frequenices and their corresponding values
    """
    sampling_frequencies = (
        cfg.physical_wavelength_range[0] * 0.1 / (torch.sqrt(2) * SPEED_OF_LIGHT)
    )
    tensor_signal = torch.tensor(signal)

    yf = rfft(tensor_signal)
    xf = rfftfreq(cfg.FDTD_niter, sampling_frequencies)
    return xf, yf
