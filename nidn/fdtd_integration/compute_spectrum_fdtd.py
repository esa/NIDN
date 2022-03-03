from dotmap import DotMap
from tqdm import tqdm
import torch
from loguru import logger

from ..trcwa.get_frequency_points import get_frequency_points
from .calculate_transmission_reflection_coefficients import (
    calculate_transmission_reflection_coefficients,
)
from .init_fdtd import init_fdtd


def compute_spectrum_fdtd(permittivity, cfg: DotMap):
    """Generates a spectrum of transmission and reflection coefficients for the specified wavelengths in the cfg, by using FDTD simulations.

    Args:
        permitivity (torch.tensor): Array of permittivity for each layer
        cfg (DotMap): Configurations needed to perform the simulations

    Returns:
        tuple[array, array]: Transmission spectrum and reflection spectrum
    """
    transmission_spectrum = []
    reflection_spectrum = []
    physical_wavelengths, norm_freq = get_frequency_points(cfg)
    logger.debug("Wavelenghts in spectrum : ")
    logger.debug(physical_wavelengths)
    logger.debug("Number of layers: ")
    logger.debug(len(permittivity[0, 0, :, 0]))
    # For each wavelength, calculate transmission and reflection coefficents
    disable_progress_bar = logger._core.min_level >= 20
    for i in tqdm(range(len(physical_wavelengths)), disable=disable_progress_bar):
        transmission_signal = []
        reflection_signal = []

        # Create simulation in free space and run it
        grid, transmission_detector, reflection_detector = init_fdtd(
            cfg,
            include_object=False,
            wavelength=physical_wavelengths[i],
            permittivity=permittivity[:, :, :, i],
        )
        grid.run(cfg.FDTD_niter, progress_bar=False)
        transmission_free_space, reflection_free_space = _get_detector_values(
            transmission_detector, reflection_detector
        )
        transmission_signal.append(transmission_free_space)
        reflection_signal.append(reflection_free_space)

        # Create the same simulation, but add material in the form of one or many layers, and run again
        grid, transmission_detector, reflection_detector = init_fdtd(
            cfg,
            include_object=True,
            wavelength=physical_wavelengths[i],
            permittivity=permittivity[:, :, :, i],
        )
        grid.run(cfg.FDTD_niter, progress_bar=False)
        transmission_material, reflection_material = _get_detector_values(
            transmission_detector, reflection_detector
        )
        transmission_signal.append(transmission_material)
        reflection_signal.append(reflection_material)
        # Calculate transmission and reflection coefficients,
        # by using the signals from the free space simulation and the material simulation
        (
            transmission_coefficient,
            reflection_coefficient,
        ) = calculate_transmission_reflection_coefficients(
            transmission_signal, reflection_signal, "MEAN SQUARE", cfg
        )
        transmission_spectrum.append(transmission_coefficient)
        reflection_spectrum.append(reflection_coefficient)

    logger.debug("Trasmission spectrum")
    logger.debug(transmission_spectrum)
    logger.debug("Reflection spectrum")
    logger.debug(reflection_spectrum)

    return torch.tensor(transmission_spectrum), torch.tensor(reflection_spectrum)


def _get_detector_values(transmission_detector, reflection_detector):
    """Extract the signals detected by the transmission detector and the reflection detector

    Args:
        transmission_detector (fdtd.LineDetector): The transmission detector attached to the grid
        reflection_detector (fdtd.LineDetector): The reflection detector attached to the grid
        detector_position_y (int): The position along the linedetector which should be used

    Returns:
        tuple[array, array]: The electric field detected by the transmission detector, and the electric field detected by the reflection detector
    """
    e_transmission = _get_abs_value_from_3D_signal(
        transmission_detector.detector_values()["E"]
    )
    e_reflection = _get_abs_value_from_3D_signal(
        reflection_detector.detector_values()["E"]
    )
    h_transmission = _get_abs_value_from_3D_signal(
        transmission_detector.detector_values()["H"]
    )
    h_reflection = _get_abs_value_from_3D_signal(
        reflection_detector.detector_values()["H"]
    )
    return e_transmission, e_reflection


def _get_abs_value_from_3D_signal(signal):
    """Get the absolute value of a three-dimentional signal. For each time step, a =sqrt(x**2 + y**2 + z**2)

    Args:
        signal (array): Three dimentional time signal

    Returns:
        Array: One dimentional time-signal
    """
    signal = _average_along_detector(signal)

    abs_value = []
    for i in range(len(signal)):
        abs_value.append(
            torch.sqrt(signal[i][0] ** 2 + signal[i][1] ** 2 + signal[i][2] ** 2)
        )
    return abs_value


def _average_along_detector(signal):
    """Average the signal along each point of a line detector

    Args:
        signal (Array[timesteps, points_on_detector,3]): E or H -field signal from a line detector

    Returns:
        Array[timesteps, 3]: averaged signal along detector
    """
    avg = []
    for e in signal:
        s = torch.zeros(3)
        for p in e:
            s[0] += p[0] / len(e)
            s[1] += p[1] / len(e)
            s[2] += p[2] / len(e)
        avg.append(s)
    return avg
