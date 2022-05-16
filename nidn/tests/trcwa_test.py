import torch
from loguru import logger
from ..utils.load_default_cfg import load_default_cfg
from ..utils.compute_spectrum import compute_spectrum

_TRCWA_TEST_TOLERANCE = 1e-7
torch.set_printoptions(precision=12)
torch.set_default_tensor_type(torch.DoubleTensor)


def test_single_layer():
    """Tests a single patterned layer with a single frequency. TRCWA vs. GRCWA"""
    logger.info("Running single layer test...")
    run_cfg = load_default_cfg()
    run_cfg.Nx = 5
    run_cfg.Ny = 5
    run_cfg.N_layers = 1
    run_cfg.target_frequencies = [0.5]
    run_cfg.N_freq = 1
    run_cfg.TRCWA_L_grid = [[1.0, 0.0], [0.0, 1.0]]
    run_cfg.TRCWA_NG = 11
    run_cfg.TRCWA_TOP_LAYER_EPS = 1.0  # epsilon for top layer, 1.0 for vacuum
    run_cfg.TRCWA_BOTTOM_LAYER_EPS = 1.0  # epsilon for top layer, 1.0 for vacuum
    run_cfg.PER_LAYER_THICKNESS = [1.0]
    run_cfg.solver = "TRCWA"

    # Get eps shape
    shape = [
        run_cfg.Nx,
        run_cfg.Ny,
        run_cfg.N_layers,
        run_cfg.N_freq,
    ]

    # Get a grid of 1 + 1j
    eps_grid = (
        torch.ones(
            shape,
            dtype=torch.cfloat,
        )
        * (1.0 + 1j)
    )

    logger.debug("Computing spectrum...")

    R, T = compute_spectrum(eps_grid, run_cfg)

    # Computed with GRCWA
    R_error = torch.abs(R[0] - 0.04256584753869945)
    T_error = torch.abs(T[0] - 0.06064642157274551)

    logger.info(f"R_error = {R_error}")
    logger.info(f"T_error = {T_error}")

    assert R_error < _TRCWA_TEST_TOLERANCE
    assert T_error < _TRCWA_TEST_TOLERANCE


def test_uniform_layer():
    """Tests a single uniform layer at two frequencies. TRCWA vs. GRCWA"""
    logger.info("Running uniform layer test...")
    run_cfg = load_default_cfg()
    run_cfg.Nx = 1
    run_cfg.Ny = 1
    run_cfg.N_layers = 1
    run_cfg.target_frequencies = [0.05, 0.1]
    run_cfg.N_freq = 2
    run_cfg.TRCWA_L_grid = [[1.0, 0.0], [0.0, 1.0]]
    run_cfg.TRCWA_NG = 11
    run_cfg.TRCWA_TOP_LAYER_EPS = 1.0  # epsilon for top layer, 1.0 for vacuum
    run_cfg.TRCWA_BOTTOM_LAYER_EPS = 1.0  # epsilon for top layer, 1.0 for vacuum
    run_cfg.PER_LAYER_THICKNESS = [1.0]
    run_cfg.solver = "TRCWA"

    # Get eps shape
    shape = [
        run_cfg.Nx,
        run_cfg.Ny,
        run_cfg.N_layers,
        run_cfg.N_freq,
    ]

    # Get a grid of 3 + 1.5j
    eps_grid = torch.ones(
        shape,
        dtype=torch.cfloat,
    )

    eps_grid = eps_grid * (3.0 + 1.5j)

    logger.debug("Computing spectrum...")

    R, T = compute_spectrum(eps_grid, run_cfg)

    # Computed with GRCWA
    R_error, T_error = [], []
    R_error.append(torch.abs(R[0] - 0.08805902003703245))
    R_error.append(torch.abs(R[1] - 0.18524218068012266))
    T_error.append(torch.abs(T[0] - 0.6307193983893462))
    T_error.append(torch.abs(T[1] - 0.4510135948077737))

    logger.info(f"R_error = {R_error}")
    logger.info(f"T_error = {T_error}")

    assert all(torch.tensor(R_error) < _TRCWA_TEST_TOLERANCE)
    assert all(torch.tensor(T_error) < _TRCWA_TEST_TOLERANCE)


def test_three_layer():
    """Tests a three stacked patterned layer with a single frequency. TRCWA vs. GRCWA"""
    logger.info("Running three layer test...")
    run_cfg = load_default_cfg()
    run_cfg.Nx = 9
    run_cfg.Ny = 9
    run_cfg.N_layers = 3
    run_cfg.target_frequencies = [3.0]
    run_cfg.N_freq = 1
    run_cfg.TRCWA_L_grid = [[1.0, 0.0], [0.0, 1.0]]
    run_cfg.TRCWA_NG = 11
    run_cfg.TRCWA_TOP_LAYER_EPS = 1.0  # epsilon for top layer, 1.0 for vacuum
    run_cfg.TRCWA_BOTTOM_LAYER_EPS = 1.0  # epsilon for top layer, 1.0 for vacuum
    run_cfg.PER_LAYER_THICKNESS = [1.0]
    run_cfg.solver = "TRCWA"

    # Get eps shape
    shape = [
        run_cfg.Nx,
        run_cfg.Ny,
        run_cfg.N_layers,
        run_cfg.N_freq,
    ]

    # Get a somewhat interesting grid
    eps_grid = (
        torch.ones(
            shape,
            dtype=torch.cfloat,
        )
        * (-4.2 + 0.42j)
    )

    eps_grid[0:3, 0:3, 0, :] = eps_grid[0:3, 0:3, 0, :] * 1.0
    eps_grid[0:3, 0:3, 1, :] = eps_grid[0:3, 0:3, 1, :] * 2.0
    eps_grid[0:3, 0:3, 2, :] = eps_grid[0:3, 0:3, 2, :] * 3.0

    logger.debug("Computing spectrum...")

    R, T = compute_spectrum(eps_grid, run_cfg)

    # Computed with GRCWA
    R_error = torch.abs(R[0] - 0.9243750060320585)
    T_error = torch.abs(T[0] - 0.0)

    logger.info(f"R_error = {R_error}")
    logger.info(f"T_error = {T_error}")

    assert R_error < _TRCWA_TEST_TOLERANCE
    assert T_error < _TRCWA_TEST_TOLERANCE


if __name__ == "__main__":
    test_single_layer()
    test_uniform_layer()
    test_three_layer()
