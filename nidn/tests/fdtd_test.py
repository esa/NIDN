from torch import zeros, tensor, cfloat
from numpy import subtract

from nidn.utils.global_constants import SPEED_OF_LIGHT

from ..fdtd_integration.init_fdtd import init_fdtd
from ..materials.layer_builder import LayerBuilder
from ..trcwa.compute_target_frequencies import compute_target_frequencies
from ..utils.load_default_cfg import load_default_cfg
from ..utils.compute_spectrum import compute_spectrum


def test_fdtd_grid_creation():
    """Test that the simulation is created in the right way, whith the correcto objects and correct realtive placement of them"""
    # Create grid with multiple uniform layer
    cfg = load_default_cfg()
    cfg.N_layers = 4
    cfg.N_freq = 1
    eps_grid = zeros(1, 1, cfg.N_layers, cfg.N_freq, dtype=cfloat)
    cfg.Nx = 1
    cfg.Ny = 1
    # Note: something went wrong when smalles wavelength was 1e-5, guess its the grid scaling
    cfg.target_frequencies = compute_target_frequencies(
        cfg.physical_wavelength_range[0],
        cfg.physical_wavelength_range[1],
        cfg.N_freq,
        "linear",
    )
    layer_builder = LayerBuilder(cfg)
    eps_grid[:, :, 0, :] = layer_builder.build_uniform_layer("titanium_oxide")
    eps_grid[:, :, 1, :] = layer_builder.build_uniform_layer("zirconium")
    eps_grid[:, :, 2, :] = layer_builder.build_uniform_layer("gallium_arsenide")
    eps_grid[:, :, 3, :] = layer_builder.build_uniform_layer("silicon_nitride")
    grid, transmission_detector, reflection_detetctor = init_fdtd(
        cfg,
        include_object=True,
        wavelength=cfg.physical_wavelength_range[0],
        permittivity=eps_grid,
    )
    # Check that it was made properly
    assert len(grid.objects) == 4
    for i in range(len(grid.objects)):
        assert grid.objects[i].permittivity == eps_grid[0, 0, i, 0].real
        assert (
            grid.objects[i].conductivity[0, 0, 0, 0]
            - (
                eps_grid[0, 0, i, 0].imag
                * SPEED_OF_LIGHT
                / cfg.physical_wavelength_range[0]
            )
            < 1e-8
        )

    assert len(grid.detectors) == 2
    # Check that the reflection detector is placed before the first layer, and the transmission detector is placed after the last layer
    assert transmission_detector.x[0] >= grid.objects[-1].x.stop
    assert reflection_detetctor.x[0] <= grid.objects[0].x.start

    assert len(grid.sources) == 1
    # If periodic boundaries in both x and y, it is two, if pml in x and periodic in y there is 3 and 4 if pml in both directions (I think)
    assert len(grid.boundaries) >= 2


def test_fdtd_simulation_single_layer():
    """Test that checks that the calculate_spectrum function returns the correct spectrum for a single layer"""
    # Create grid with uniform layer
    cfg = load_default_cfg()
    cfg.N_layers = 1
    eps_grid = zeros(1, 1, cfg.N_layers, cfg.N_freq, dtype=cfloat)
    # Note: something went wrong when smalles wavelength was 1e-5, guess its the grid scaling
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
    validated_transmission_spectrum = [
        tensor(0.0),
        tensor(0.0),
        tensor(2.0448e-308),
        tensor(2.5918e-277),
        tensor(3.8102e-132),
        tensor(0.6251),
        tensor(0.5362),
        tensor(0.6362),
        tensor(0.7259),
        tensor(0.6041),
        tensor(0.5613),
        tensor(0.6453),
        tensor(0.6915),
        tensor(0.5796),
        tensor(0.4658),
        tensor(0.5885),
        tensor(0.5805),
        tensor(0.4043),
        tensor(0.4957),
        tensor(0.4211),
    ]
    validated_reflection_spectrum = [
        tensor(0.9515),
        tensor(0.8697),
        tensor(0.8118),
        tensor(0.7570),
        tensor(0.6983),
        tensor(0.2333),
        tensor(0.3495),
        tensor(0.2044),
        tensor(0.0673),
        tensor(0.2649),
        tensor(0.3565),
        tensor(0.2003),
        tensor(0.0904),
        tensor(0.2549),
        tensor(0.3180),
        tensor(0.1525),
        tensor(0.1539),
        tensor(0.3921),
        tensor(0.2173),
        tensor(0.3437),
    ]
    assert all(
        abs(subtract(transmission_spectrum, validated_transmission_spectrum)) < 1e-4
    )
    assert all(abs(subtract(reflection_spectrum, validated_reflection_spectrum)) < 1e-4)


def test_fdtd_simulation_four_layers():
    """Test that checks that the calculate_spectrum function returns the correct spectrum for a simulation with four layers"""
    # Create grid with four uniform layer
    cfg = load_default_cfg()
    cfg.N_layers = 4
    cfg.FDTD_niter = 400
    eps_grid = zeros(1, 1, cfg.N_layers, cfg.N_freq, dtype=cfloat)
    cfg.target_frequencies = compute_target_frequencies(
        cfg.physical_wavelength_range[0],
        cfg.physical_wavelength_range[1],
        cfg.N_freq,
        "linear",
    )
    cfg.solver = "FDTD"
    layer_builder = LayerBuilder(cfg)
    eps_grid[:, :, 0, :] = layer_builder.build_uniform_layer("titanium_oxide")
    eps_grid[:, :, 1, :] = layer_builder.build_uniform_layer("zirconium")
    eps_grid[:, :, 2, :] = layer_builder.build_uniform_layer("gallium_arsenide")
    eps_grid[:, :, 3, :] = layer_builder.build_uniform_layer("silicon_nitride")
    transmission_spectrum, reflection_spectrum = compute_spectrum(eps_grid, cfg)
    validated_transmission_spectrum = [
        tensor(0.0),
        tensor(0.0),
        tensor(0.0),
        tensor(0.0),
        tensor(0.0),
        tensor(0.0),
        tensor(0.0),
        tensor(0.0),
        tensor(0.0),
        tensor(0.0),
        tensor(0.0),
        tensor(0.0),
        tensor(0.0),
        tensor(0.0),
        tensor(0.0),
        tensor(0.0),
        tensor(0.0),
        tensor(0.0),
        tensor(0.0),
        tensor(0.0),
    ]
    validated_reflection_spectrum = [
        tensor(0.9515),
        tensor(0.8781),
        tensor(0.8077),
        tensor(0.7559),
        tensor(0.7042),
        tensor(0.5871),
        tensor(0.5489),
        tensor(0.6336),
        tensor(0.4941),
        tensor(0.6071),
        tensor(0.4161),
        tensor(0.6353),
        tensor(0.5313),
        tensor(0.6544),
        tensor(0.6774),
        tensor(0.4997),
        tensor(0.6510),
        tensor(0.4196),
        tensor(0.5047),
        tensor(0.2888),
    ]
    assert all(
        abs(subtract(transmission_spectrum, validated_transmission_spectrum)) < 1e-4
    )
    assert all(abs(subtract(reflection_spectrum, validated_reflection_spectrum)) < 1e-4)


def test_single_patterned_layer():
    """Test that a pattern layer returns teh correct spectrum"""
    # TODO: Test patterned layer, must be implemented first
    pass


if __name__ == "__main__":
    test_fdtd_grid_creation()
    test_fdtd_simulation_single_layer()
    test_fdtd_simulation_four_layers()
    test_single_patterned_layer()
