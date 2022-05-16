from loguru import logger


from ..fdtd_integration.compute_spectrum_fdtd import compute_spectrum_fdtd
from ..trcwa.compute_spectrum_trcwa import compute_spectrum_trcwa
from .validate_config import _validate_config


def compute_spectrum(eps_grid, cfg, validate_cfg=True):
    """Computes spectrum, using either TRCWA or FDTD simulations

    Args:
        cfg (DotMap): Configurations for the simulation
        permittivity (Array[Nx][Ny][Nlayers][Nfrequencies]): Matrix of permittivity tensors
        validate_cfg (bool): Whether to validate the config

    Returns:
        Array, Array: Reflection spectrumand transmission spectrum
    """
    if validate_cfg:
        _validate_config(cfg)
    if cfg.solver == "FDTD":
        logger.debug("Using FDTD solver to find spectrum")
        return compute_spectrum_fdtd(eps_grid, cfg)
    elif cfg.solver == "TRCWA":
        logger.debug("Using TRCWA solver to find spectrum")
        return compute_spectrum_trcwa(eps_grid, cfg)
