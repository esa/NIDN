from nidn.fdtd_integration.compute_fdtd_grid_scaling import _compute_fdtd_grid_scaling

from ..training.run_training import run_training
from ..utils.load_default_cfg import load_default_cfg


def _setup():
    # Load default cfg as starting point
    cfg = load_default_cfg()

    # Specify your desired range of wavelengths
    cfg.physical_wavelength_range[0] = 2e-6
    cfg.physical_wavelength_range[1] = 1e-5

    # Let's investigate 4 frequency points
    cfg.N_freq = 2

    # Currently, the target spectra is set manually as a list of numbers
    cfg.target_reflectance_spectrum = [0.75, 0.25]
    cfg.target_transmittance_spectrum = [0.25, 0.25]

    cfg.Nx = 1  # Set layer size  to 1x1 (interpreted as uniform)
    cfg.Ny = 1
    cfg.N_layers = 1  # Choose number of layers
    cfg.eps_oversampling = 1

    # Allowed range of epsilon values
    cfg.real_min_eps = 0.01
    cfg.real_max_eps = 20.0
    cfg.imag_min_eps = 0.0
    cfg.imag_max_eps = 1.0

    cfg.type = "regression"
    cfg.iterations = 3  # Set number of training iterations (that is forward model evaluations) to perform

    return cfg


def test_rcwa_training():
    """Tests the training with RCWA."""
    cfg = _setup()
    cfg.solver = "TRCWA"
    run_training(cfg)
    # Should have at least learned something
    assert cfg.results.loss_log[-1] < 0.3
    cfg.pop("model", None)
    # Forget the old model
    cfg.type = "classification"
    run_training(cfg)
    # Should have at least learned something
    assert cfg.results.loss_log[-1] < 0.3


def test_fdtd_training():
    cfg = _setup()
    cfg.solver = "FDTD"
    cfg.FDTD_grid_scaling = _compute_fdtd_grid_scaling(cfg)
    run_training(cfg)
    # Should have at least learned something
    assert cfg.results.loss_log[-1] < 0.3
    cfg.pop("model", None)
    # Forget the old model
    cfg.type = "classification"
    run_training(cfg)
    # Should have at least learned something
    assert cfg.results.loss_log[-1] < 0.3
