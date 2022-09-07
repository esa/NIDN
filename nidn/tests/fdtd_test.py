import torch

from nidn.utils.global_constants import EPS_0, PI, SPEED_OF_LIGHT, UNIT_MAGNITUDE
from nidn.fdtd_integration.constants import FDTD_GRID_SCALE
from nidn.fdtd_integration.compute_fdtd_grid_scaling import _compute_fdtd_grid_scaling

from ..fdtd_integration.init_fdtd import init_fdtd
from ..fdtd_integration.compute_spectrum_fdtd import _get_detector_values
from ..materials.layer_builder import LayerBuilder
from ..trcwa.compute_target_frequencies import compute_target_frequencies
from ..utils.load_default_cfg import load_default_cfg
from ..utils.compute_spectrum import compute_spectrum


def test_fdtd_grid_creation():
    """Test that the simulation is created in the right way, with the correct objects and correct relative placement of them"""
    # Create grid with multiple uniform layer
    cfg = load_default_cfg()
    cfg.N_layers = 4
    cfg.PER_LAYER_THICKNESS = [1.0, 2.0, 1.5, 1.2]
    cfg.N_freq = 1
    eps_grid = torch.zeros(1, 1, cfg.N_layers, cfg.N_freq, dtype=torch.cfloat)
    cfg.Nx = 1
    cfg.Ny = 1
    cfg.target_frequencies = compute_target_frequencies(
        cfg.physical_wavelength_range[0],
        cfg.physical_wavelength_range[1],
        cfg.N_freq,
        "linear",
    )

    cfg.FDTD_grid_scaling = _compute_fdtd_grid_scaling(cfg)

    layer_builder = LayerBuilder(cfg)
    eps_grid[:, :, 0, :] = layer_builder.build_uniform_layer("titanium_oxide")
    eps_grid[:, :, 1, :] = layer_builder.build_uniform_layer("germanium")
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
            grid.objects[i].conductivity[0][0][0][0]
            - (
                eps_grid[0, 0, i, 0].imag
                * SPEED_OF_LIGHT
                / cfg.physical_wavelength_range[0]
                * 2
                * PI
                * EPS_0
            )
            < 1e-16
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
    cfg.N_freq = 5
    cfg.N_layers = 1
    cfg.physical_wavelength_range[0] = 0.4e-6
    cfg.physical_wavelength_range[1] = 0.5e-6
    cfg.FDTD_min_gridpoints_per_unit_magnitude = 100
    cfg.PER_LAYER_THICKNESS = [0.38]
    eps_grid = torch.zeros(1, 1, cfg.N_layers, cfg.N_freq, dtype=torch.cfloat)
    cfg.target_frequencies = compute_target_frequencies(
        cfg.physical_wavelength_range[0],
        cfg.physical_wavelength_range[1],
        cfg.N_freq,
        "linear",
    )
    cfg.FDTD_niter = 3000
    cfg.solver = "FDTD"
    layer_builder = LayerBuilder(cfg)
    eps_grid[:, :, 0, :] = layer_builder.build_uniform_layer("titanium_oxide")
    reflection_spectrum, transmission_spectrum = compute_spectrum(eps_grid, cfg)

    validated_reflection_spectrum = torch.tensor(
        [0.266668735247, 0.096685654550, 0.541316057208, 0.383675738997, 0.200519748705]
    )
    validated_transmission_spectrum = torch.tensor(
        [0.733331264753, 0.900710835488, 0.457884307088, 0.616324261003, 0.79215577611]
    )
    assert all(
        torch.abs(torch.tensor(transmission_spectrum) - validated_transmission_spectrum)
        < 1e-8
    )
    assert all(
        torch.abs(torch.tensor(reflection_spectrum) - validated_reflection_spectrum)
        < 1e-8
    )
    assert all(e <= 1 for e in transmission_spectrum)
    assert all(e <= 1 for e in reflection_spectrum)
    assert all(e >= 0 for e in transmission_spectrum)
    assert all(e >= 0 for e in reflection_spectrum)


def test_fdtd_simulation_four_layers():
    """Test that checks that the calculate_spectrum function returns the correct spectrum for a simulation with four layers"""
    # Create grid with four uniform layer
    cfg = load_default_cfg()
    cfg.N_layers = 4
    cfg.FDTD_niter = 1000
    cfg.N_freq = 5
    cfg.PER_LAYER_THICKNESS = [0.12, 0.12, 0.12, 0.12]
    cfg.physical_wavelength_range[0] = 4e-7
    cfg.physical_wavelength_range[1] = 5e-7
    eps_grid = torch.zeros(1, 1, cfg.N_layers, cfg.N_freq, dtype=torch.cfloat)
    cfg.target_frequencies = compute_target_frequencies(
        cfg.physical_wavelength_range[0],
        cfg.physical_wavelength_range[1],
        cfg.N_freq,
        "linear",
    )
    cfg.solver = "FDTD"
    layer_builder = LayerBuilder(cfg)
    eps_grid[:, :, 0, :] = layer_builder.build_uniform_layer("titanium_oxide")
    eps_grid[:, :, 1, :] = layer_builder.build_uniform_layer("zinc_oxide")
    eps_grid[:, :, 2, :] = layer_builder.build_uniform_layer("gallium_arsenide")
    eps_grid[:, :, 3, :] = layer_builder.build_uniform_layer("silicon_nitride")
    reflection_spectrum, transmission_spectrum = compute_spectrum(eps_grid, cfg)
    print(reflection_spectrum)
    print(transmission_spectrum)
    validated_reflection_spectrum = torch.tensor(
        [0.881822561593, 0.849144767680, 0.248671325302, 0.710441537464, 0.819290191321]
    )
    validated_transmission_spectrum = torch.tensor(
        [
            0.001828167336,
            0.000675301853,
            0.000104852612,
            3.007361599856e-06,
            1.295376141331e-06,
        ]
    )
    assert all(
        torch.abs(torch.tensor(transmission_spectrum) - validated_transmission_spectrum)
        < 1e-8
    )
    assert all(
        torch.abs(torch.tensor(reflection_spectrum) - validated_reflection_spectrum)
        < 1e-8
    )
    assert all(e <= 1 for e in transmission_spectrum)
    assert all(e <= 1 for e in reflection_spectrum)
    assert all(e >= 0 for e in transmission_spectrum)
    assert all(e >= 0 for e in reflection_spectrum)


def test_single_patterned_layer():
    """Test that a pattern layer returns teh correct spectrum"""
    # TODO: Test patterned layer, must be implemented first
    pass


def test_deviation_from_original_fdtd():
    """Test if the changed version of FDTD does not deviate much from the original fdtd package"""
    # Set settings
    cfg = load_default_cfg()
    cfg.N_freq = 1
    cfg.N_layers = 1
    cfg.PER_LAYER_THICKNESS = [0.3]
    cfg.physical_wavelength_range[0] = 10e-7
    cfg.physical_wavelength_range[1] = 10e-7
    cfg.solver = "FDTD"
    cfg.FDTD_niter = 400
    cfg.FDTD_pulse_type = "continuous"
    cfg.FDTD_source_type = "line"
    cfg.target_frequencies = compute_target_frequencies(
        cfg.physical_wavelength_range[0],
        cfg.physical_wavelength_range[1],
        cfg.N_freq,
        cfg.freq_distribution,
    )
    cfg.FDTD_grid_scaling = _compute_fdtd_grid_scaling(cfg)
    eps_grid = torch.zeros(cfg.Nx, cfg.Ny, cfg.N_layers, cfg.N_freq, dtype=torch.cfloat)
    layer_builder = LayerBuilder(cfg)
    eps_grid[:, :, 0, :] = layer_builder.build_uniform_layer("titanium_oxide")
    # NIDN FDTD
    grid, t_detector_material, _ = init_fdtd(
        cfg,
        include_object=True,
        wavelength=cfg.physical_wavelength_range[0],
        permittivity=eps_grid[:, :, 0, :],
    )
    grid.run(cfg.FDTD_niter)
    t_signal_material, r_ = _get_detector_values(t_detector_material, _)
    # Original fdtd
    # fmt: off
    original_fdtd = [0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000005,0.00000040,0.00000586,0.00003407,0.00016646,0.00063252,0.00197463,0.00525066,0.01205673,0.02430069,0.04351699,0.06982090,0.10100884,0.13219824,0.15633987,0.16563437,0.15355523,0.11692787,0.05753755,-0.01723113,-0.09515738,-0.16137626,-0.20155223,-0.20526758,-0.16884784,-0.09687829,-0.00189478,0.09784234,0.18205929,0.23285518,0.23877029,0.19756541,0.11701629,0.01346954,-0.09155471,-0.17600509,-0.22211654,-0.22031931,-0.17133863,-0.08603550,0.01697837,0.11544958,0.18834247,0.22038011,0.20524013,0.14678013,0.05804195,-0.04172718,-0.13131219,-0.19207620,-0.21181143,-0.18714530,-0.12401448,-0.03616888,0.05791365,0.13895934,0.19083733,0.20373021,0.17590359,0.11378927,0.03041218,-0.05739219,-0.13238847,-0.18030084,-0.19246623,-0.16732017,-0.11044858,-0.03332559,0.04900651,0.12084549,0.16876529,0.18406730,0.16427315,0.11348976,0.04156282,-0.03780004,-0.10958416,-0.16025953,-0.18025874,-0.16577304,-0.11948467,-0.05010768,0.02919859,0.10327787,0.15781279,0.18208027,0.17107093,0.12655524,0.05685002,-0.02468548,-0.10216793,-0.16026723,-0.18725373,-0.17739638,-0.13220335,-0.06024802,0.02442632,0.10502968,0.16537730,0.19315826,0.18247973,0.13513371,0.06033049,-0.02707944,-0.10957870,-0.17050730,-0.19746611,-0.18484645,-0.13502369,-0.05791166,0.03102573,0.11389246,0.17399627,0.19921420,0.18446755,0.13273388,0.05444253,-0.03465528,-0.11667350,-0.17519214,-0.19856574,-0.18223090,-0.12959832,-0.05130585,0.03693332,0.11750378,0.17440999,0.19645174,0.17942675,0.12689423,0.04943467,-0.03750293,-0.11667522,-0.17247258,-0.19399118,-0.17713441,-0.12538804,-0.04908230,0.03665883,0.11492264,0.17033936,0.19207863,0.17595758,0.12522969,0.04993973,-0.03506679,-0.11306945,-0.16874569,-0.19117164,-0.17595022,-0.12608342,-0.05138347,0.03345588,0.11174057,0.16805206,0.19127063,0.17677766,0.12738096,0.05277631,-0.03235356,-0.11121833,-0.16822415,-0.19205513,-0.17793189,-0.12857426,-0.05367996,0.03197392,0.11144335,0.16895956,0.19307682,0.17894736,0.12930722,0.05394194,-0.03223551,-0.11212765,-0.16986172,-0.19393731,-0.17953732,-0.12947463,-0.05365945,0.03286746,0.11292100,0.17058839,0.19440824,0.17963061,0.12918463,0.05308073,-0.03356030,-0.11353212,-0.17095899,-0.19444438,-0.17934186,-0.12865901,-0.05247975,0.03407173,0.11381789,0.17095198,0.19416189,0.17886836,0.12813994,0.05205447,-0.03428712,-0.11377891,-0.17067937,-0.19373869,-0.17842109,-0.12778919,-0.05189585,0.03422469,0.11351973,0.17030354,0.19335459,0.17813445,0.12767570,0.05197511,-0.03398198,-0.11318744,-0.16997567,-0.19312231,-0.17805708,-0.12776612,-0.05219974,0.03368936,0.11290958,0.16978895,0.19307401,0.17815370,0.12797243,0.05245613,-0.03345527,-0.11276105,-0.16976360,-0.19317297,-0.17834193,-0.12819592,-0.05265231,0.03333881,0.11275326,0.16986225,0.19334332,0.17853606,0.12835910,0.05274232,-0.03334439,-0.11284947,-0.17001528,-0.19351145,-0.17867040,-0.12842782,-0.05272627,0.03343603,0.11298665,0.17015990,0.19362117,0.17872185,0.12840384,0.05264085,-0.03355928,-0.11310953,-0.17024950,-0.19365783,-0.17869294,-0.12832501,-0.05253036,0.03366259,0.11318289,0.17027306,0.19362704,0.17861993,0.12822724,0.05244339,-0.03372203,-0.11319561,-0.17024229,-0.19355923,-0.17853503,-0.12815308,-0.05239672,0.03372696,0.11316500,0.17018012,0.19348628,0.17847125,0.12811782,0.05239682,-0.03369590,-0.11310955,-0.17011721,-0.19343204,-0.17844502,-0.12812155,-0.05242828,0.03364751,0.11305490,0.17007218,0.19341211,0.17845201,0.12815241,0.05247050,-0.03360039,-0.11301819,-0.17005671,-0.19342188,-0.17848131,-0.12818978,-0.05251030,0.03357054,0.11300644,0.17006815,0.19344940,0.17851456,0.12822289,0.05253439,-0.03356200,-0.11301846,-0.17009399,-0.19347879,-0.17854212,-0.12824170,-0.05254059,0.03357425,0.11304224,0.17012015,0.19350161,0.17855625,0.12824624,0.05252815,-0.03359529,-0.11306621,-0.17013833,-0.19351239,-0.17855893,-0.12823424,-0.05250980,0.03361756,0.11307997,0.17014713,0.19351250,0.17854861,0.12821755,0.05248993,-0.03362820,-0.11308707,-0.17014478,-0.19350445,-0.17853259,-0.12820113,-0.05248053,0.03363277,0.11308365,0.17013814,0.19348943,0.17851956,0.12819194,0.05247903,-0.03362982,-0.11307687,-0.17012544,-0.19347837,-0.17851120,-0.12819263,-0.05248127,0.03362229,0.11306729,0.17011494,0.19347203,0.17851231,0.12819525,0.05248869,-0.03361513,-0.11305726,-0.17011083,-0.19347251,-0.17851621,-0.12820140,-0.05249485,0.03360652,0.11305437,0.17011097,0.19347739,0.17852095,0.12820724,0.05250144,-0.03360386,]
    # fmt: on
    # Compare signals

    diff = t_signal_material - torch.tensor(original_fdtd)

    assert max(abs(diff)) < 5e-6


def test_gradient_flow():
    """Test if the gradients are available and not nan's"""
    # Set settings
    cfg = load_default_cfg()
    cfg.N_freq = 3
    cfg.N_layers = 1
    cfg.PER_LAYER_THICKNESS = [0.3]
    cfg.physical_wavelength_range[0] = 8e-7
    cfg.physical_wavelength_range[1] = 10e-7
    cfg.solver = "FDTD"
    cfg.FDTD_niter = 400
    cfg.FDTD_pulse_type = "continuous"
    cfg.FDTD_source_type = "line"
    cfg.target_frequencies = compute_target_frequencies(
        cfg.physical_wavelength_range[0],
        cfg.physical_wavelength_range[1],
        cfg.N_freq,
        cfg.freq_distribution,
    )
    eps_grid = torch.ones(
        cfg.Nx, cfg.Ny, cfg.N_layers, cfg.N_freq, dtype=torch.cfloat, requires_grad=True
    )
    # NIDN FDTD
    R, T = compute_spectrum(eps_grid, cfg)
    loss = sum(T)
    loss.retain_grad()
    loss.backward()
    assert type(loss.grad.item()) == float


if __name__ == "__main__":
    pass
