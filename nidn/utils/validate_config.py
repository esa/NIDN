from dotmap import DotMap
from loguru import logger

from nidn.fdtd_integration.constants import FDTD_GRID_SCALE
from ..utils.global_constants import UNIT_MAGNITUDE


def _validate_config(cfg: DotMap):
    """This function validates that all required entries are in the config.

    Args:
        cfg (DotMap): Run config you intend to use.
    """
    _check_for_keys(cfg)
    _check_entry_types(cfg)
    _check_value_ranges(cfg)
    if cfg.solver == "FDTD":
        _check_grid_scale(cfg)
    logger.debug("Config validated successfully.")


def _check_grid_scale(cfg):
    """Checks whether FDTD grid scale allows specified sizes.

    Args:
        cfg (DotMap): Configurations for the simulation
    """
    scale_in_wl = cfg.physical_wavelength_range[0] * FDTD_GRID_SCALE
    scaling = max(
        UNIT_MAGNITUDE / scale_in_wl,
        cfg.FDTD_min_gridpoints_per_unit_magnitude,
    )
    for i, thickness in enumerate(cfg.PER_LAYER_THICKNESS):
        x_start = cfg.FDTD_pml_thickness + cfg.FDTD_free_space_distance
        x_end = x_start + thickness
        size = int(scaling * x_end) - int(scaling * x_start)
        if thickness != size / scaling:
            downscaled = size / scaling
            logger.warning(
                f"Due to the grid resolution, the thickness of layer {i + 1} is set to {downscaled:.3f} µm"
                + f"instead of the specified {thickness} µm"
            )


def _check_for_keys(cfg: DotMap):
    """Checks that all required keys are present in the config"""
    # fmt: off
    required_keys = ["name","seed","eps_oversampling","real_min_eps","real_max_eps","imag_min_eps","imag_max_eps",
                     "Nx","Ny","N_layers","physical_wavelength_range","N_freq","model_type","encoding_dim","siren_omega",
                     "n_neurons","hidden_layers","learning_rate","iterations","L","absorption_loss","type","use_regularization_loss",
                     "reg_loss_weight","add_noise","noise_scale","solver","PER_LAYER_THICKNESS",
                     "TRCWA_L_grid","TRCWA_NG","TRCWA_TOP_LAYER_EPS","TRCWA_BOTTOM_LAYER_EPS",
                     "FDTD_min_gridpoints_per_unit_magnitude","FDTD_source_type","FDTD_pulse_type","FDTD_pml_thickness",
                     "FDTD_source_position","FDTD_free_space_distance","FDTD_niter","FDTD_gridpoints_from_material_to_detector",
                     "target_reflectance_spectrum","target_transmittance_spectrum","freq_distribution","use_gpu","avoid_zero_eps"]

    # Some keys that may be set in the cfg during runtime
    optional_keys = ["target_frequencies","FDTD_grid","model","out_features","results","best_model_state_dict","material_collection",
                     "N_materials", "thicknesses","FDTD_grid_scaling"]
    # fmt: on
    for key in required_keys:
        if key not in cfg:
            raise KeyError(f"CFG missing required key: {key}")

    for key in cfg.keys():
        if key not in required_keys and key not in optional_keys:
            raise KeyError(f"CFG Key {key} is not a valid key")


def _check_entry_types(cfg: DotMap):
    """Check that all entries in the config are of the correct type"""
    # fmt: off
    integer_keys = ["Nx","Ny","N_layers","N_freq","encoding_dim","n_neurons","eps_oversampling","seed",
                    "TRCWA_NG","FDTD_niter","FDTD_gridpoints_from_material_to_detector","FDTD_min_gridpoints_per_unit_magnitude"]
    float_keys = ["L","real_min_eps","real_max_eps","imag_min_eps","imag_max_eps","siren_omega","noise_scale","reg_loss_weight",
                  "TRCWA_TOP_LAYER_EPS","TRCWA_BOTTOM_LAYER_EPS","FDTD_free_space_distance","FDTD_pml_thickness",]
    boolean_keys = ["use_regularization_loss","add_noise","use_gpu","avoid_zero_eps",]
    string_keys = ["model_type","type","name","freq_distribution","solver","FDTD_source_type","FDTD_pulse_type",]
    list_keys = ["PER_LAYER_THICKNESS","TRCWA_L_grid","FDTD_source_position","target_reflectance_spectrum","target_transmittance_spectrum",
                 "physical_wavelength_range",]
    # fmt: on

    for key in integer_keys:
        if not isinstance(cfg[key], int):
            raise TypeError(f"{key} must be an integer")

    for key in float_keys:
        if not isinstance(cfg[key], float):
            raise TypeError(f"{key} must be a float")

    for key in boolean_keys:
        if not isinstance(cfg[key], bool):
            raise TypeError(f"{key} must be a boolean")

    for key in string_keys:
        if not isinstance(cfg[key], str):
            raise TypeError(f"{key} must be a string")

    for key in list_keys:
        if not isinstance(cfg[key], list):
            raise TypeError(f"{key} must be a list")


def _check_value_ranges(cfg: DotMap):
    """Check that all values in the config are within the correct range.
    This throws runtime errors as ValueErrors are caught in training to avoid NaNs crashing the training."""
    if not (cfg.real_min_eps <= cfg.real_max_eps):
        raise RuntimeError(f"real_min_eps must be less than or equal to real_max_eps")

    if not (cfg.imag_min_eps <= cfg.imag_max_eps):
        raise RuntimeError(f"imag_min_eps must be less than or equal to imag_max_eps")

    if not (cfg.physical_wavelength_range[0] < cfg.physical_wavelength_range[1]):
        raise RuntimeError(
            f"physical_wavelength_range must be ordered from low to high"
        )

    scaling = max(
        UNIT_MAGNITUDE / (cfg.physical_wavelength_range[0] * FDTD_GRID_SCALE),
        cfg.FDTD_min_gridpoints_per_unit_magnitude,
    )
    if (
        int(scaling * cfg.FDTD_free_space_distance)
        < cfg.FDTD_gridpoints_from_material_to_detector
    ):
        raise RuntimeError(
            "Reflection detector must be placed in the free space before an eventual object, and after the pml layer. Decrease FDTD_gridpoints_from_material_to_detector to a value smaller than {value} fix this.".format(
                value=int(scaling * cfg.FDTD_free_space_distance)
            )
        )

    # fmt: off
    positive_value_keys = ["L","n_neurons","Nx","Ny","N_layers","N_freq","learning_rate","iterations","eps_oversampling","noise_scale",
                           "TRCWA_NG","reg_loss_weight","FDTD_niter","FDTD_free_space_distance","FDTD_pml_thickness",
                           "FDTD_gridpoints_from_material_to_detector","FDTD_min_gridpoints_per_unit_magnitude",]
    # fmt: on

    for key in positive_value_keys:
        if not (cfg[key] > 0):
            raise RuntimeError(f"{key} must be a positive integer")

    if not cfg.Nx % cfg.eps_oversampling == 0:
        raise RuntimeError(f"Nx must be a multiple of eps_oversampling")
    if not cfg.Ny % cfg.eps_oversampling == 0:
        raise RuntimeError(f"Ny must be a multiple of eps_oversampling")

    if cfg.type != "classification" and cfg.type != "regression":
        raise RuntimeError(f"type must be either 'classification' or 'regression'")

    if not cfg.TRCWA_L_grid[1][0] < cfg.TRCWA_L_grid[1][1]:
        raise RuntimeError(f"TRCWA_L_grid dim1 must be ordered from low to high")
    if not cfg.TRCWA_L_grid[0][0] > cfg.TRCWA_L_grid[0][1]:
        raise RuntimeError(f"TRCWA_L_grid dim0 must be ordered from high to low")

    all_positive_list_keys = [
        "TRCWA_L_grid",
        "PER_LAYER_THICKNESS",
        "FDTD_grid",
        "FDTD_source_position",
    ]
    all_positive_or_zero_list_keys = [
        "target_transmittance_spectrum",
        "target_reflectance_spectrum",
    ]

    for key in all_positive_list_keys:
        if key in cfg.keys() and not (all(cfg[key]) > 0.0):
            raise RuntimeError(f"All elements in {key} must be a positive integer")
    for key in all_positive_or_zero_list_keys:
        if not (all(cfg[key]) >= 0.0):
            raise RuntimeError(
                f"All elements in {key} must be a positive integer or zero"
            )

    # Disabled this check temporarily as it is only relevant in training. For forward model this doesn't matter.
    # if not len(cfg.target_transmittance_spectrum) == cfg.N_freq:
    #     raise RuntimeError(
    #         f"target_transmittance_spectrum must have length N_freq. Got {len(cfg.target_transmittance_spectrum)}, expected {cfg.N_freq}"
    #     )

    # if not len(cfg.target_reflectance_spectrum) == cfg.N_freq:
    #     raise RuntimeError(
    #         f"target_reflectance_spectrum must have length N_freq. Got {len(cfg.target_reflectance_spectrum)}, expected {cfg.N_freq}"
    #     )

    if not (
        len(cfg.PER_LAYER_THICKNESS) == cfg.N_layers
        or len(cfg.PER_LAYER_THICKNESS) == 1
    ):
        raise RuntimeError(f"PER_LAYER_THICKNESS must have length 1 or N_layers")

    if not (cfg.freq_distribution == "linear" or cfg.freq_distribution == "log"):
        raise RuntimeError(f"freq_distribution must be either 'linear' or 'log'")

    if not (len(cfg.FDTD_source_position) == 2 or len(cfg.FDTD_source_position) == 3):
        raise RuntimeError(
            f"The FDTD source needs either 2- or 3-dimensional coordinates"
        )
    if not cfg.FDTD_source_type in ["point", "line"]:
        raise RuntimeError(f'The FDTD_source_type must either be "line" or "point"')

    if not cfg.FDTD_pulse_type in ["ricker", "hanning", "continuous"]:
        raise RuntimeError(
            f'The FDTD_pulse_type must either be "pulse" or "continuous"'
        )
