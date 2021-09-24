from dotmap import DotMap
from .compute_target_frequencies import compute_target_frequencies
from ..utils.convert_units import freq_to_wl, wl_to_phys_wl


def get_frequency_points(run_cfg: DotMap):
    """Computes both the physical and normalized frequency points which will be used.
    This can be helpful to determine the points at which you need to define the target
    spectrum.

    Args:
        run_cfg (DotMap): Config for the run.

    Returns:
        physical_target_frequencies (list): List of physical frequencies.
        target_frequencies (list): List of target frequencies.
    """

    # Determine target frequencies
    run_cfg.target_frequencies = compute_target_frequencies(
        run_cfg.physical_wavelength_range[0],
        run_cfg.physical_wavelength_range[1],
        run_cfg.N_freq,
        run_cfg.freq_distribution,
    )

    physical_target_frequencies = wl_to_phys_wl(freq_to_wl(run_cfg.target_frequencies))

    return physical_target_frequencies, run_cfg.target_frequencies
