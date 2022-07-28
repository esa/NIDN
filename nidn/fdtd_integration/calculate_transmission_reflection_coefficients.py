import torch
from loguru import logger

from ..utils.global_constants import UNIT_MAGNITUDE


def calculate_transmission_reflection_coefficients(
    transmission_signals, reflection_signals, wavelength, cfg
):
    """Calculates the transmission coefficient and reflection coefficient for the signals presented.

    Args:
        transmission_signals (tuple[array,array]): Transmission signal from a free-space fdtd simulaiton, and transmission signal from a fdtd simulation with an added object
        reflection_signals (_type_): Reflection signal from a free-space fdtd simulaiton, and reflection signal from a fdtd simulation with an added object
        wavelength (float): Physical wavelength of the light used in this simulation
        cfg (DotMap): Configuration dictionary

    Returns:
        tuple[float, float]: Transmission coefficient and reflection coefficient
    """
    # Substract the free_space reflection signal from the material reflection signal, to eliminate unreflected signal from detector
    # The detector detects signal passing through both ways, and is placed between the source and the material.
    # Thus, most of the signal present is the unreflected signal, which must be removed.
    true_reflection = reflection_signals[1] - reflection_signals[0]

    _check_for_all_zero_signal(transmission_signals)
    _check_for_all_zero_signal(reflection_signals)

    # Eliminate transient part of the signal
    transmission_signals[0] = _eliminate_transient_part(
        transmission_signals[0], wavelength, cfg, is_freespace=True
    )
    transmission_signals[1] = _eliminate_transient_part(
        transmission_signals[1], wavelength, cfg, is_freespace=False
    )
    reflection_signals[0] = _eliminate_transient_part(
        reflection_signals[0], wavelength, cfg, is_freespace=True
    )
    true_reflection = _eliminate_transient_part(
        true_reflection, wavelength, cfg, is_freespace=False
    )

    (
        reflection_coefficient,
        transmission_coefficient,
    ) = _peak_based_coefficient_computation(
        transmission_signals, reflection_signals[0], true_reflection
    )

    # Check if the coefficients are within the physical range
    (
        transmission_coefficient,
        reflection_coefficient,
    ) = _check_and_normalize_coefficients(
        transmission_coefficient, reflection_coefficient
    )

    return transmission_coefficient, reflection_coefficient


def _check_and_normalize_coefficients(transmission_coefficient, reflection_coefficient):
    if transmission_coefficient < 0 or transmission_coefficient > 1:
        logger.error(
            f"The transmission coefficient is outside of the physical range between 0 and 1. The transmission coefficient is {transmission_coefficient}"
        )

    if reflection_coefficient < 0 or reflection_coefficient > 1:
        logger.error(
            f"The reflection coefficient is outside of the physical range between 0 and 1. The reflection coefficient is {reflection_coefficient}"
        )
    coefficient_sum = transmission_coefficient + reflection_coefficient
    # Given the transient removal there can sometimes be a small error
    # in the calculation of the coefficients leading to coeffsum > 1.
    # Depending on the severity we either inform / warn and normalize
    # or raise an error.
    if coefficient_sum > 1 and coefficient_sum < 1.01:
        logger.info(
            f"The sum of the transmission and reflection coefficient is marginally greater than 1 ({coefficient_sum:.4f}), which is physically impossible. Normalizing to 1."
        )
        transmission_coefficient = transmission_coefficient / (coefficient_sum)
        reflection_coefficient = reflection_coefficient / (coefficient_sum)
    elif coefficient_sum > 1 and coefficient_sum < 1.05:
        logger.warning(
            f"The sum of the transmission and reflection coefficient is greater than 1 ({coefficient_sum:.4f}), which is physically impossible. Normalizing to 1."
        )
        transmission_coefficient = transmission_coefficient / (coefficient_sum)
        reflection_coefficient = reflection_coefficient / (coefficient_sum)
    elif coefficient_sum > 1.05:
        logger.error(
            f"The sum of the transmission and reflection coefficient is greater than 1 ({coefficient_sum:.4f}), which is physically impossible. The transmission coefficient is {transmission_coefficient} and the reflection coefficient is {reflection_coefficient}"
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
    if _mean_square(signals[0]) <= 1e-15:
        raise ValueError(
            "The free-space signal is all zero. Increase the number of FDTD_niter to ensure that the signal reaches the detector."
        )


def _FFT_based_coefficient_computation(
    transmission_signals, reflection_signal, true_reflection, plot=False
):
    """Calculates the transmission coefficient and reflection coefficient using the FFT method.

    Args:
        transmission_signals (tuple[array,array]): Transmission signal from a free-space fdtd simulaiton, and transmission signal from a fdtd simulation with an added object
        reflection_signal (array): Reflection signal from a free-space fdtd simulation, and reflection signal from a fdtd simulation with an added object
        true_reflection (array): The true reflection signal, which is the material reflection signal minus the detector reflection signal.
        plot (bool, optional): If True, plots the FFTs. Defaults to False.
    Returns:
        tuple[float, float]: Transmission coefficient and reflection coefficient
    """
    # TODO Improve the FFT method, e.g. as described here
    # https://eecs.wsu.edu/~schneidj/ufdtd/chap5.pdf P 22ff.
    raise (
        NotImplementedError(
            "FFT based coefficient computation still requires more work."
        )
    )
    # Calculate the FFT of the signals and amplitude as magnitude
    transmission_fft = torch.fft.rfft(transmission_signals[0], norm="forward").abs()
    transmission_free_space_fft = torch.fft.rfft(
        transmission_signals[1], norm="forward"
    ).abs()
    reflection_fft = torch.fft.rfft(reflection_signal, norm="forward").abs()
    true_reflection_fft = torch.fft.rfft(true_reflection, norm="forward").abs()

    if plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7.5, 5.0), dpi=150)
        plt.plot(transmission_fft, lw=3, linestyle="None", marker="o")
        plt.plot(transmission_free_space_fft, lw=3, linestyle="None", marker="d")
        plt.plot(reflection_fft, lw=3, linestyle="None", marker="s")
        plt.plot(true_reflection_fft, lw=3, linestyle="None", marker="x")
        plt.xscale("log")
        plt.ylabel("Magnitude", fontsize=8)
        plt.xlabel("Frequency", fontsize=8)
        plt.tick_params(axis="both", which="major", labelsize=8)
        plt.legend(
            ["Transmission", "FreeSpaceTransmission", "Reflection", "True Reflection"],
            fontsize=8,
        )
        plt.show()

    logger.debug(
        f"Transmission FFT min,max,mean,std: "
        + f"{transmission_fft.min():.2e}, {transmission_fft.max():.2e}, {transmission_fft.mean():.2e}, {transmission_fft.std():.2e}"
    )
    logger.debug(
        f"Transmission Free Space FFT min,max,mean,std: "
        + f"{transmission_free_space_fft.min():.2e}, {transmission_free_space_fft.max():.2e}, {transmission_free_space_fft.mean():.2e}, {transmission_free_space_fft.std():.2e}"
    )
    logger.debug(
        f"Reflection FFT min,max,mean,std: "
        + f"{reflection_fft.min():.2e}, {reflection_fft.max():.2e}, {reflection_fft.mean():.2e}, {reflection_fft.std():.2e}"
    )
    logger.debug(
        f"True Reflection FFT min,max,mean,std: "
        + f"{true_reflection_fft.min():.2e}, {true_reflection_fft.max():.2e}, {true_reflection_fft.mean():.2e}, {true_reflection_fft.std():.2e}"
    )

    # Calculate the transmission coefficient
    transmission_coefficient = (
        transmission_fft.max() / transmission_free_space_fft.max()
    )
    reflection_coefficient = reflection_fft.max() / true_reflection_fft.max()

    return reflection_coefficient, transmission_coefficient


def _peak_based_coefficient_computation(
    transmission_signals, reflection_signal, true_reflection
):
    """Calculates the transmission coefficient and reflection coefficient using the peak method.

    Args:
         transmission_signals (tuple[array,array]): Transmission signal from a free-space fdtd simulaiton, and transmission signal from a fdtd simulation with an added object
         reflection_signal (array): Reflection signal from a free-space fdtd simulation, and reflection signal from a fdtd simulation with an added object
         true_reflection (array): The true reflection signal, which is the material reflection signal minus the detector reflection signal.
     Returns:
         tuple[float, float]: Transmission coefficient and reflection coefficient
    """
    # find peaks for all signals
    peaks_transmission_freespace = _torch_find_peaks(transmission_signals[0])
    peaks_transmission_material = _torch_find_peaks(transmission_signals[1])
    peaks_reflection_freespace = _torch_find_peaks(reflection_signal)
    peaks_reflection_material = _torch_find_peaks(true_reflection)

    if len(peaks_transmission_material) > 1:
        mean_squared_transmission_material = _mean_square(
            transmission_signals[1][
                peaks_transmission_material[0]
                .item() : peaks_transmission_material[-1]
                .item()
            ]
        )
    else:
        mean_squared_transmission_material = (max(transmission_signals[1]) ** 2) / 2
        logger.warning(
            "There are not enough timesteps for the transmission signal to have the proper length / or no signal is transmited. The signal should at least contain 2 peaks, but {} is found.The FDTD_niter should be increased, to be sure that the resutls are valid.".format(
                len(peaks_transmission_material)
            )
        )

    if len(peaks_transmission_freespace) > 1:
        mean_squared_transmission_free_space = _mean_square(
            transmission_signals[0][
                peaks_transmission_freespace[0]
                .item() : peaks_transmission_freespace[-1]
                .item()
            ]
        )
    else:
        mean_squared_transmission_free_space = (max(transmission_signals[0]) ** 2) / 2

    if len(peaks_reflection_material) > 1:
        mean_squared_reflection_material = _mean_square(
            true_reflection[
                peaks_reflection_material[0]
                .item() : peaks_reflection_material[-1]
                .item()
            ]
        )
    else:
        mean_squared_reflection_material = (max(true_reflection) ** 2) / 2
        logger.warning(
            "There are not enough timesteps for the reflected signal to have the proper length. The signal should at least contain 2 peaks, but {} is found. The FDTD_niter should be increased, to be sure that the resutls are valid.".format(
                len(peaks_reflection_material)
            )
        )

    if len(peaks_reflection_freespace) > 1:
        mean_squared_reflection_free_space = _mean_square(
            reflection_signal[
                peaks_reflection_freespace[0]
                .item() : peaks_reflection_freespace[-1]
                .item()
            ]
        )
    else:
        mean_squared_reflection_free_space = (max(reflection_signal) ** 2) / 2

    transmission_coefficient = (
        mean_squared_transmission_material / mean_squared_transmission_free_space
    )

    reflection_coefficient = (
        mean_squared_reflection_material / mean_squared_reflection_free_space
    )

    return reflection_coefficient, transmission_coefficient


def _eliminate_transient_part(signal, wavelength, cfg, is_freespace=False, plot=False):
    """Eliminates the transient part of the signal
    Args:
        signal (tensor): signal to perform the calculations on
        cfg (DotMap): configuration dictionary
        wavelength (float): Physical wavelength of the light used in this simulation
        is_freespace (bool): Influences change thresholds, True if the signal is from a free-space simulation, False if it is from a simulation with an object.
        plot (bool, optional): If True, plots the signal before and after the elimination. Defaults to False.
    Returns:
        tensor: The signal without the transient part
    """
    # We threshold much more liberally for the free-space signal, since it is more likely to have a mild transient.
    relative_change_treshold = 0.5 if is_freespace else 0.1

    if plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7.5, 5.0), dpi=150)
        plt.plot(signal, lw=3, linestyle="--")
        plt.ylabel("Signal", fontsize=8)
        plt.xlabel("Timestep", fontsize=8)
        plt.tick_params(axis="both", which="major", labelsize=8)

    logger.debug(f"Eliminating transient part of the signal of length {len(signal)}")
    # Window size in which we investigate variance. We divide by physical wavelength as the oscillation frequency
    # depends on the wavelength. (the 1.33 was detemined heuristcally, there is probably an analytical term for it)
    window_size = int(
        3 * cfg.FDTD_min_gridpoints_per_unit_magnitude * (wavelength / UNIT_MAGNITUDE)
    )
    logger.trace(f"Window size is {window_size}")

    # Eliminate zero part of the signal
    first_non_zero_index = 0
    for i in range(len(signal)):
        if abs(signal[i]) > 1e-8:
            first_non_zero_index = i
            break
    logger.trace(
        f"First non-zero index: {first_non_zero_index} of length {len(signal)}"
    )
    signal = signal[first_non_zero_index:]

    # Check signal is still long enough
    signal_length = len(signal)
    if signal_length < 2 * window_size:
        logger.error(
            "FDTD Signal is too short to compute R,T,A. Please increase FDTD_niter in the config. Aborting transient removal."
        )
    else:
        # Split into chunks
        chunks = list(torch.split(signal, window_size))
        # Merge last two chunks to have rather one too big than too small
        chunks[-2] = torch.cat((chunks[-1], chunks[-2]))
        chunks = chunks[:-1]
        logger.trace(f"Number of chunks: {len(chunks)}")
        logger.trace(f"Chunksizes are {[len(chunk) for chunk in chunks]}")
        # Plot vertical lines for chunks
        if plot:
            xcoords = torch.cumsum(
                torch.tensor([first_non_zero_index] + [len(chunk) for chunk in chunks]),
                0,
            )
            for xc in xcoords:
                plt.axvline(x=xc, color="r", linestyle="--")
            plt.show()
        ranges_per_window = torch.tensor(
            [chunk.max() - chunk.min() for chunk in chunks]
        )
        logger.trace("Ranges per window: {}".format(ranges_per_window))
        change_to_previous_chunk = (
            ranges_per_window[0:-1] - ranges_per_window[1:]
        ).abs()
        logger.trace("Change to previous chunk: {}".format(change_to_previous_chunk))
        relative_change_to_maximum_change = (
            change_to_previous_chunk / change_to_previous_chunk.max()
        )
        logger.trace(
            "Relative change to maximum change: {}".format(
                relative_change_to_maximum_change
            )
        )

        # Now we find the last window where the relative change is above a certain threshold
        last_window_over_ten_percent_change = len(relative_change_to_maximum_change) - 1
        while (
            relative_change_to_maximum_change[last_window_over_ten_percent_change]
            < relative_change_treshold
        ):
            last_window_over_ten_percent_change -= 1
            if last_window_over_ten_percent_change < 0:
                raise (
                    logger.error(
                        "FDTD Signal did not reach a steady state. Please increase FDTD_niter in the config."
                    )
                )

        logger.debug(
            f"First window over 10% change: {last_window_over_ten_percent_change}"
        )

        signal = signal[(last_window_over_ten_percent_change + 1) * window_size :]

    if plot:
        plt.figure(figsize=(7.5, 5.0), dpi=150)
        plt.plot(signal, lw=3, linestyle="--")
        plt.ylabel("Signal", fontsize=8)
        plt.xlabel("Timestep", fontsize=8)
        plt.tick_params(axis="both", which="major", labelsize=8)
        plt.show()

    return signal


# From : https://stackoverflow.com/questions/54498775/pytorch-argrelmax-function-or-c
def _torch_find_peaks(signal):
    signaltemp1 = signal[1:-1] - signal[:-2]
    signaltemp2 = signal[1:-1] - signal[2:]

    # and checking where both shifts are positive;
    out1 = torch.where(signaltemp1 > 0, signaltemp1 * 0 + 1, signaltemp1 * 0)
    out2 = torch.where(signaltemp2 > 0, out1, signaltemp2 * 0)

    # argrelmax containing all peaks
    return torch.nonzero(out2, out=None) + 1
