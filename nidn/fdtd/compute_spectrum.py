from weakref import ref
from dotmap import DotMap
import sys

import torch
from nidn.fdtd.calculate_transmission_reflection_coefficients import (
    calculate_transmission_reflection_coefficients,
)

from nidn.trcwa.constants import TRCWA_EPS_0

sys.path.append("../")
import nidn
from init_fdtd import init_fdtd


def compute_spectrum(cfg: DotMap):
    """Generates a spectrum of transmission and reflection coefficients for the specified wavelengths in the cfg, by using FDTD simulations.

    Args:
        cfg (DotMap): Configurations needed to perform the simulations

    Returns:
        tuple[array, array]: Transmission spectrum and reflection spectrum
    """
    t_spectrum, r_spectrum = []
    physical_wavelengths, norm_freq = nidn.get_frequency_points(cfg)

    for w in physical_wavelengths:
        transmission, reflection = []
        grid, t_detector, r_detector = init_fdtd(cfg, False, w, permittivity=5.1984)
        grid.run(cfg.FDTD_niter, progress_bar=False)
        transmission[0], reflection[0] = _get_detector_values(
            t_detector, r_detector, cfg.FDTD_grid[2] / 2
        )

        grid, t_detector, r_detector = init_fdtd(cfg, True, w, permittivity=5.1984)
        grid.run(cfg.FDTD_niter, progress_bar=True)
        transmission[1], reflection[1] = _get_detector_values(
            t_detector, r_detector, cfg.FDTD_grid[2] / 2
        )

        (
            transmission_coefficient,
            reflection_coefficient,
        ) = calculate_transmission_reflection_coefficients(
            transmission, reflection, "MEAN SQUARE"
        )
        t_spectrum.append(transmission_coefficient)
        r_spectrum.append(reflection_coefficient)

    # For each wavelength:
    # Run simulation
    # Get detector values
    # Postprocess signal,  i.e. phase shift and more if applicable
    # Calculate transmission, reflection (and absorption coefficients)
    # Return transmission spectrum, reflection spectrum (and absorption spectrum)
    return t_spectrum, r_spectrum


def _get_detector_values(
    transmission_detector, reflection_detector, detector_position_y
):
    """Extract the signals detected by the transmission detector and the reflection detector

    Args:
        transmission_detector (fdtd.LineDetector): The transmission detector attached to the grid
        reflection_detector (fdtd.LineDetector): The reflection detector attached to the grid
        detector_position_y (int): The position along the linedetector which should be used

    Returns:
        tuple[array, array]: The electric field detected by the transmission detector, and the electric field detected by the reflection detector
    """
    e_transmission = _get_abs_value_from_3D_signal(
        transmission_detector.detector_values["E"][int(detector_position_y)]
    )
    e_reflection = _get_abs_value_from_3D_signal(
        reflection_detector.detector_values["E"][int(detector_position_y)]
    )
    h_transmission = _get_abs_value_from_3D_signal(
        transmission_detector.detector_values["H"][int(detector_position_y)]
    )
    h_reflection = _get_abs_value_from_3D_signal(
        reflection_detector.detector_values["H"][int(detector_position_y)]
    )

    return e_transmission, e_reflection


def _get_abs_value_from_3D_signal(signal):
    """Get the absolute value of a three-dimentional signal. For each time step, a =sqrt(x**2 + y**2 + z**2)

    Args:
        signal (array): Three dimentional time signal

    Returns:
        Array: One dimentional time-signal
    """
    abs_value = []
    for i in range(len(signal)):
        abs_value.append(
            torch.sqrt(signal[i][0] ** 2 + signal[i][1] ** 2 + signal[i][2] ** 2)
        )
    return abs_value
