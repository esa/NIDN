from re import T
from numpy import NaN, require
import torch

from nidn.utils.global_constants import EPS_0, PI, SPEED_OF_LIGHT

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
    cfg.physical_wavelength_range[0] = 1e-6
    cfg.physical_wavelength_range[1] = 1e-5
    eps_grid = torch.zeros(1, 1, cfg.N_layers, cfg.N_freq, dtype=torch.cfloat)
    cfg.target_frequencies = compute_target_frequencies(
        cfg.physical_wavelength_range[0],
        cfg.physical_wavelength_range[1],
        cfg.N_freq,
        "linear",
    )
    cfg.FDTD_niter = 300
    cfg.solver = "FDTD"
    layer_builder = LayerBuilder(cfg)
    eps_grid[:, :, 0, :] = layer_builder.build_uniform_layer("titanium_oxide")
    reflection_spectrum, transmission_spectrum = compute_spectrum(eps_grid, cfg)
    validated_reflection_spectrum = torch.tensor(
        [
            0.01996015,
            0.39330551,
            0.44639092,
            0.18894569,
            0.43341999,
        ]
    )
    validated_transmission_spectrum = torch.tensor(
        [
            0.80365908,
            0.58930942,
            0.54395288,
            0.72771733,
            0.49183103,
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


def test_fdtd_simulation_four_layers():
    """Test that checks that the calculate_spectrum function returns the correct spectrum for a simulation with four layers"""
    # Create grid with four uniform layer
    cfg = load_default_cfg()
    cfg.N_layers = 4
    cfg.FDTD_niter = 600
    cfg.N_freq = 5
    cfg.PER_LAYER_THICKNESS = [1.0, 1.0, 1.0, 1.0]
    cfg.physical_wavelength_range[0] = 1e-6
    cfg.physical_wavelength_range[1] = 1e-5
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
    validated_reflection_spectrum = torch.tensor(
        [
            0.08163995,
            0.21881801,
            0.11293128,
            0.08326362,
            0.76207314,
        ]
    )
    validated_transmission_spectrum = torch.tensor(
        [
            0.14337483,
            0.63959249,
            0.48796375,
            0.21393022,
            0.00167027,
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
    original_fdtd = [0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000001,0.00000003,0.00000011,0.00000033,0.00000093,0.00000248,0.00000628,0.00001517,0.00003489,0.00007667,0.00016118,0.00032461,0.00062712,0.00116346,0.00207486,0.00355971,0.00587901,0.00935100,0.01432803,0.02114949,0.03006673,0.04114187,0.05413007,0.06836529,0.08267887,0.09538699,0.10438178,0.10735076,0.10212703,0.08714106,0.06190895,0.02745934,-0.01341745,-0.05620914,-0.09508659,-0.12368555,-0.13619096,-0.12855860,-0.09963637,-0.05191532,0.00833259,0.07168089,0.12694260,0.16315762,0.17183129,0.14896679,0.09640990,0.02211196,-0.06088770,-0.13668907,-0.18960948,-0.20751429,-0.18467895,-0.12349527,-0.03453577,0.06517337,0.15541939,0.21698238,0.23584427,0.20647889,0.13342907,0.03072841,-0.08080084,-0.17785128,-0.23962388,-0.25240947,-0.21271707,-0.12821362,-0.01622401,0.09990850,0.19581163,0.25132266,0.25481642,0.20566991,0.11433428,0.00002327,-0.11344412,-0.20271339,-0.24976605,-0.24555703,-0.19168311,-0.09979599,0.01103587,0.11839085,0.20109213,0.24338579,0.23788300,0.18674461,0.10096334,-0.00202469,-0.10186472,-0.17923177,-0.21945593,-0.21517836,-0.16759683,-0.08612358,0.01344405,0.11189207,0.19016463,0.23291499,0.23140399,0.18521995,0.10248543,-0.00152658,-0.10716006,-0.19405439,-0.24506563,-0.24968129,-0.20623905,-0.12246304,-0.01414659,0.09781811,0.19151327,0.24836956,0.25694728,0.21532816,0.13159394,0.02226253,-0.09101606,-0.18582868,-0.24352683,-0.25297204,-0.21272580,-0.13124474,-0.02506574,0.08461433,0.17623506,0.23215344,0.24204700,0.20478305,0.12843005,0.02847004,-0.07537697,-0.16305259,-0.21800955,-0.23028112,-0.19821977,-0.12862589,-0.03529466,0.06370433,0.14948289,0.20587894,0.22244431,0.19634187,0.13281924,0.04418197,-0.05254217,-0.13882046,-0.19812280,-0.21906777,-0.19761234,-0.13785901,-0.05130816,0.04532230,0.13330745,0.19554466,0.21989612,0.20158357,0.14415301,0.05880609,-0.03777976,-0.12671719,-0.19061535,-0.21699245,-0.20071898,-0.14501194,-0.06079022,0.03547308,0.12497194,0.19022780,0.21847674,0.20413299,0.14986238,0.06606966,-0.03109528,-0.12282956,-0.19128213,-0.22299129,-0.21150574,-0.15868847,-0.07444323,0.02510530,0.12065630,0.19344282,0.22890699,0.21962695,0.16690475,0.08066679,-0.02235094,-0.12181939,-0.19782822,-0.23490309,-0.22521022,-0.17026979,-0.08078792,0.02540233,0.12690330,0.20310464,0.23842780,0.22557154,0.16706483,0.07478486,-0.03247212,-0.13286394,-0.20600723,-0.23716710,-0.22025515,-0.15901732,-0.06618111,0.03923586,0.13587212,0.20438867,0.23135122,0.21184579,0.15032469,0.05955488,-0.04207242,-0.13429118,-0.19899746,-0.22377147,-0.20420986,-0.14464715,-0.05716896,0.04085112,0.13018619,0.19348908,0.21861153,0.20089037,0.14398464,0.05912187,-0.03709235,-0.12588286,-0.18992478,-0.21669083,-0.20088929,-0.14552142,-0.06134291,0.03518493,0.12507685,0.19053437,0.21846668,0.20312484,0.14731332,0.06191437,-0.03620700,-0.12752242,-0.19373345,-0.22145999,-0.20497302,-0.14740446,-0.06017090,0.03928891,0.13101477,0.19655669,0.22271566,0.20422623,0.14482889,0.05651038,-0.04293270,-0.13350654,-0.19706814,-0.22098106,-0.20063658,-0.14033479,-0.05235978,0.04554761,0.13379231,0.19486409,0.21681648,0.19559310,0.13575164,0.04946426,-0.04598052,-0.13165523,-0.19074672,-0.21184071,-0.19110039,-0.13293075,-0.04902440,0.04399940,0.12788218,0.18629116,0.20797064,0.18888042,0.13292409,0.05114354,-0.04045178,-0.12399642,-0.18322317,-0.20660108,-0.18956077,-0.13538332,-0.05458116,0.03711654,0.12179549,0.18283006,0.20813348,0.19255319,0.13893570,0.05763612,-0.03545342,-0.12198986,-0.18478706,-0.21123738,-0.19586288,-0.14147351,-0.05867988,0.03616355,0.12419022,0.18778861,0.21414767,0.19785949,0.14204314,0.05775192,-0.03822469,-0.12672365,-0.19005583,-0.21555856,-0.19813430,-0.14126438,-0.05629831,0.03983104,0.12799941,0.19070514,0.21554142,0.19763897,0.14060290,0.05576999,-0.04005470,-0.12792656,-0.19049641,-0.21541946,-0.19777450,-0.14102547,-0.05634146,0.03959121,0.12783998,0.19094506,0.21636251]
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
