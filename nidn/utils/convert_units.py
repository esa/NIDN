from ..trcwa.constants import _TRCWA_UNIT_MAGNITUDE, _TRCWA_SPEED_OF_LIGHT


def freq_to_wl(input):
    """Gives the wavelength in units of the lattice constants
    from the unitless frequency and vice versa. In TRCWA the magnitude is set to micrometres (um).

    Args:
        input (float): The unitless frequency, or the wavelength in units of _TRCWA_UNIT_MAGNITUDE.

    Returns:
        float: The wavelength with units of _TRCWA_UNIT_MAGNITUDE, or the unitless frequency.
    """
    return 1.0 / input


def wl_to_phys_wl(wl):
    """Gives the physical wavelength from the wavelength in units of _TRCWA_UNIT_MAGNITUDE.

    Args:
        input (float): The wavelength in units of _TRCWA_UNIT_MAGNITUDE.

    Returns:
        float: The physical wavelength.
    """
    return wl * _TRCWA_UNIT_MAGNITUDE


def phys_freq_to_wl(input):
    """Gives the physical frequency from the physical wavelength and vice versa.

    Args:
        input (float): The physical wavelength in metres or the physical frequency in Hz.

    Returns:
        float: The physical frequency or the physical wavelength.
    """
    c = _TRCWA_SPEED_OF_LIGHT
    return c / input
