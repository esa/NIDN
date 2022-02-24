from dotmap import DotMap
from numpy import source

from ..fdtd_integration.init_fdtd import init_fdtd
import torch
from ..materials.layer_builder import LayerBuilder
from ..trcwa.compute_target_frequencies import compute_target_frequencies
from ..utils.load_default_cfg import load_default_cfg


def test_single_uniform_layer():
    # Create grid with uniform layer
    cfg = load_default_cfg()
    cfg.N_layers = 1
    cfg.N_freq = 1
    eps_grid = torch.zeros(1, 1, cfg.N_layers, cfg.N_freq, dtype=torch.cfloat)
    cfg.Nx = 1
    cfg.Ny = 1
    cfg.physical_wavelength_range[0] = 1e-5
    cfg.physical_wavelength_range[1] = 1e-5
    cfg.target_frequencies = compute_target_frequencies(
        cfg.physical_wavelength_range[0],
        cfg.physical_wavelength_range[1],
        cfg.N_freq,
        "linear",
    )
    layer_builder = LayerBuilder(cfg)
    eps_grid[:, :, 0, :] = layer_builder.build_uniform_layer("titanium_oxide")
    grid, transmission_detector, reflection_detetctor = init_fdtd(
        cfg, include_object=True, wavelength=1e-5, permittivity=eps_grid
    )
    # Check that it was made properly
    assert len(grid.objects) == 1
    assert grid.objects[0].permittivity == eps_grid[0, 0, 0, 0].real
    assert len(grid.detectors) == 2
    assert len(grid.sources) == 1
    assert len(grid.boundaries) >= 2


def test_multiple_uniform_layers():
    # Create grid with multiple uniform layer
    cfg = DotMap()
    cfg.N_layers = 4
    cfg.N_freq = 1
    eps_grid = torch.zeros(1, 1, cfg.N_layers, cfg.N_freq, dtype=torch.cfloat)
    cfg.Nx = 1
    cfg.Ny = 1
    cfg.physical_wavelength_range[0] = 1e-5
    cfg.physical_wavelength_range[1] = 1e-5
    cfg.target_frequencies = compute_target_frequencies(
        cfg.physical_wavelength_range[0],
        cfg.physical_wavelength_range[1],
        cfg.N_freq,
        "linear",
    )
    cfg.PER_LAYER_THICKNESS = [1.0, 1.0, 1.0, 1.0]
    cfg.FDTD_grid = [5.0, 2.0, 1.0]
    layer_builder = LayerBuilder(cfg)
    eps_grid[:, :, 0, :] = layer_builder.build_uniform_layer("titanium_oxide")
    eps_grid[:, :, 1, :] = layer_builder.build_uniform_layer("zirconium")
    eps_grid[:, :, 2, :] = layer_builder.build_uniform_layer("gallium_arsenide")
    eps_grid[:, :, 3, :] = layer_builder.build_uniform_layer("silicon_nitride")
    grid, transmission_detector, reflection_detetctor = init_fdtd(
        cfg, include_object=True, wavelength=1e-5, permittivity=eps_grid
    )
    # Check that it was made properly
    assert len(grid.objects) == 4
    for i in range(4):
        assert grid.objects[i].permittivity == eps_grid[0, 0, i, 0].real
    assert len(grid.detectors) == 2
    assert len(grid.sources) == 1
    # If periodic boundaries in both x and y, it is two, if pml in x and periodic in y there is 3 and 4 if pml in both directions (I think)
    assert len(grid.boundaries) >= 2


def test_single_patterned_layer():
    # TODO: Test patterned layer, must be implemented first
    pass


if __name__ == "__main__":
    test_single_uniform_layer()
    test_multiple_uniform_layers()
