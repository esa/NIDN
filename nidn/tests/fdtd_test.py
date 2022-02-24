import torch

from ..fdtd_integration.init_fdtd import init_fdtd
from ..materials.layer_builder import LayerBuilder
from ..trcwa.compute_target_frequencies import compute_target_frequencies
from ..utils.load_default_cfg import load_default_cfg
from ..utils.compute_spectrum import compute_spectrum


def test_fdtd_simulation():
    # Create grid with uniform layer
    cfg = load_default_cfg()
    cfg.N_layers = 1
    cfg.N_freq = 1
    eps_grid = torch.zeros(1, 1, cfg.N_layers, cfg.N_freq, dtype=torch.cfloat)
    cfg.Nx = 1
    cfg.Ny = 1
    # Note: something went wrong when smalles wavelength was 1e-5, guess its the grid scaling
    cfg.physical_wavelength_range[0] = 1e-6
    cfg.physical_wavelength_range[1] = 2e-6
    cfg.target_frequencies = compute_target_frequencies(
        cfg.physical_wavelength_range[0],
        cfg.physical_wavelength_range[1],
        cfg.N_freq,
        "linear",
    )
    cfg.solver = "FDTD"
    layer_builder = LayerBuilder(cfg)
    eps_grid[:, :, 0, :] = layer_builder.build_uniform_layer("titanium_oxide")
    transmission_spectrum, reflection_spectrum = compute_spectrum(eps_grid, cfg)
    assert sum(transmission_spectrum) > 0
    assert sum(reflection_spectrum) > 0


if __name__ == "__main__":
    test_fdtd_simulation()
