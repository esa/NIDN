from dotmap import DotMap
from tqdm import tqdm
from loguru import logger
import torch

from nidn.fdtd_integration.constants import FDTD_GRID_SCALE
from nidn.fdtd_integration.compute_fdtd_grid_scaling import _compute_fdtd_grid_scaling
from nidn.utils.global_constants import UNIT_MAGNITUDE

from ..trcwa.get_frequency_points import get_frequency_points
from .calculate_transmission_reflection_coefficients import (
    calculate_transmission_reflection_coefficients,
)
from .init_fdtd import init_fdtd
from ..fdtd.backend import backend as bd


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
    physical_wavelengths, _ = get_frequency_points(cfg)
    logger.debug("Wavelenghts in spectrum : ")
    logger.debug(physical_wavelengths)
    logger.debug("Number of layers: " + str(len(permittivity[0, 0, :, 0])))

    if (len(cfg.PER_LAYER_THICKNESS) == 1) and (cfg.N_layers > 1):
        cfg.PER_LAYER_THICKNESS = cfg.PER_LAYER_THICKNESS * cfg.N_layers

    cfg.FDTD_grid_scaling = _compute_fdtd_grid_scaling(cfg)
    # For each wavelength, calculate transmission and reflection coefficents
    disable_progress_bar = logger._core.min_level >= 20
    for i in tqdm(range(len(physical_wavelengths)), disable=disable_progress_bar):

        _check_if_enough_timesteps(
            cfg, physical_wavelengths[i], permittivity[:, :, :, i]
        )
        logger.debug("Simulating for wavelenght: {}".format(physical_wavelengths[i]))
        transmission_signal = []
        reflection_signal = []

        # Create simulation in free space and run it
        grid, transmission_detector, reflection_detector = init_fdtd(
            cfg,
            include_object=False,
            wavelength=physical_wavelengths[i],
            permittivity=None,
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
            transmission_signal,
            reflection_signal,
            wavelength=physical_wavelengths[i],
            cfg=cfg,
        )
        transmission_spectrum.append(transmission_coefficient)
        reflection_spectrum.append(reflection_coefficient)
    logger.debug("Trasmission spectrum")
    logger.debug(transmission_spectrum)
    logger.debug("Reflection spectrum")
    logger.debug(reflection_spectrum)

    return reflection_spectrum, transmission_spectrum


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

    abs_value = bd.zeros(len(signal))
    for i in range(len(signal)):
        # Added 1e-16 to prevent gradient flow from breaking, without significantly changing the result
        squared_value = torch.square(signal[i] + 1e-16)
        summed_squared_value = torch.sum(squared_value)
        absolute_value = torch.sqrt(summed_squared_value)
        abs_value[i] = absolute_value
    # For now only returning z-component of electric field, because the signal is only in the z-direction.
    # The absoulte value of the three directions might be neccesary later if the electric field is present in other directions.
    return signal[:, 2]


def _average_along_detector(signal):
    """Average the signal along each point of a line detector

    Args:
        signal (Array[timesteps, points_on_detector,3]): E or H -field signal from a line detector

    Returns:
        Array[timesteps, 3]: averaged signal along detector
    """
    avg = bd.zeros([len(signal), 3])
    for i in range(len(signal)):
        s = bd.zeros(3)
        for p in signal[i]:
            s[0] += p[0] / len(signal[i])
            s[1] += p[1] / len(signal[i])
            s[2] += p[2] / len(signal[i])
        avg[i] = s
    return avg


def _summed_thickness_times_sqrt_permittivity(thicknesses, permittivity):
    """
    Helper function to calculate the sum of the product of the thickness and
    the square root of the permittivity for each layer in a material stack.

    Args:
        thicknesses (tensor): tensor with the thickness of each layer of the material stack
        permittivity (tensor): tensor with the relative permittivity for each layer of the material stack

    Returns:
        float: sum of thickness times sqrt(e_r) for each layer
    """
    summed_permittivity = 0
    for i in range(len(thicknesses)):
        summed_permittivity += thicknesses[i] * torch.sqrt(permittivity.real.max())
    return summed_permittivity


def _check_if_enough_timesteps(cfg: DotMap, wavelength, permittivity):
    """
    Function to find the recommended minimum number of timesteps.
    The signal should have passed trough the material, been reflected to the start of the material and reflected again to the
    rear detector before the signal is assumed to be steady state. The max permittivity is used for all layers, in case of patterned layers in the future.

    Args:
        cfg (DotMap): config
        wavelength (float): wavelength of the simulation
        permittivity (tensor): tensor of relative permittivities for each layer
    """
    wavelengths_of_steady_state_signal = 5
    number_of_internal_reflections = 3
    recommended_timesteps = int(
        (
            (
                cfg.FDTD_free_space_distance
                + number_of_internal_reflections
                * _summed_thickness_times_sqrt_permittivity(
                    cfg.PER_LAYER_THICKNESS, permittivity
                )
                + wavelengths_of_steady_state_signal * wavelength / UNIT_MAGNITUDE
            )
            * torch.sqrt(torch.tensor(2.0))
            * cfg.FDTD_grid_scaling
        ).item()
    )
    logger.debug("Minimum recomended timesteps: {}".format(recommended_timesteps))
    if cfg.FDTD_niter < recommended_timesteps:
        logger.warning(
            "The number of timesteps should be increased to minimum {} to ensure that the result from the simulation remains physically accurate.".format(
                recommended_timesteps
            )
        )
