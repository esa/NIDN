from dotmap import DotMap


def _validate_config(cfg: DotMap):
    """This function validates that all required entries are in the config.

    Args:
        cfg (DotMap): Run config you intend to use.
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
        "encoding_dim",
        "out_features",
        "siren_omega",
        "n_neurons",
        "hidden_layers",
        "learning_rate",
        "iterations",
        "L",
        "absorption_loss",
    ]

    for key in required_keys:
        if key not in cfg:
            raise ValueError(f"Missing required key: {key}")
