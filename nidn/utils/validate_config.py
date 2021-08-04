from dotmap import DotMap


def validate_config(cfg: DotMap):
    """This function validates that all required entries are in the config.

    Args:
        cfg (DotMap): Run config you intend to use
    """
    required_keys = [
        "seed",
        "eps_oversampling",
        "real_min_eps",
        "real_max_eps",
        "imag_min_eps",
        "imag_max_eps",
        "Nx",
        "Ny",
        "N_layers",
        "physical_wavelength_range",
        "N_freq",
        "model_type",
    ]

    for key in required_keys:
        if key not in cfg:
            raise ValueError(f"Missing required key: {key}")
