from nidn.trcwa.constants import TRCWA_UNIT_MAGNITUDE, TRCWA_SPEED_OF_LIGHT


def freq_to_wl(freq):
    """Gives the wavelength in units of the lattice constants
    from the unitless frequency and vice versa. In TRCWA the magnitude is set to micrometres (um).

    Args:
        freq (float): The unitless frequency, or the wavelength in units of TRCWA_UNIT_MAGNITUDE.

    Returns:
        float: The wavelength with units of TRCWA_UNIT_MAGNITUDE, or the unitless frequency.
    """
    return 1.0 / freq


def phys_wl_to_wl(phys_wl):
    """Gives the wavelength from the physical wavelength and vice versa.

    Args:
        phys_wl (float): The physical wavelength in metres or the physical frequency in Hz.

    Returns:
        float: The physical frequency or the physical wavelength.
    """
    return phys_wl / TRCWA_UNIT_MAGNITUDE


def wl_to_phys_wl(wl):
    """Gives the physical wavelength from the wavelength in units of TRCWA_UNIT_MAGNITUDE.

    Args:
        wl (float): The wavelength in units of TRCWA_UNIT_MAGNITUDE.

    Returns:
        float: The physical wavelength.
    """
    return wl * TRCWA_UNIT_MAGNITUDE


def phys_freq_to_phys_wl(freq):
    """Gives the physical frequency from the physical wavelength and vice versa.

    Args:
        input (float): The physical wavelength in metres or the physical frequency in Hz.

    Returns:
        float: The physical frequency or the physical wavelength.
    """
    c = TRCWA_SPEED_OF_LIGHT
    return c / freq
