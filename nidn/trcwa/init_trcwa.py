from loguru import logger

from .constants import *
from ..utils.global_constants import *
from .trcwa import TRCWA
from ..materials.material_collection import MaterialCollection


def _init_trcwa(eps_grid, target_frequency, run_cfg):
    """Creates a TRCWA object matching the given eps_grid and target_frequency.

    Args:
        eps_grid (torch.tensor): Grid of epsilon values. Should be [Nx, Ny, N_layers].
        target_frequency (float): Target frequency for this simulation run.
        run_cfg (DotMap): Run configuration.

    Returns:
        TRCWA obj: The created object which is ready to compute the spectrum
    """
    Nx, Ny, N_layers = eps_grid.shape[0:3]

    # Squeeze out Nx=1, Ny=1 dimension (for uniform layer)
    eps_grid = eps_grid.squeeze(0)  # squeeze X
    eps_grid = eps_grid.squeeze(0)  # squeeze Y (now at 0)

    # Adding a small imaginary part to the frequency to avoid singularities in RCWA
    # See page 2 of arxiv.org/pdf/2005.04840v1.pdf
    # or doi.org/10.1364/OE.21.030812 for a more thorough explanation
    freqcmp = target_frequency * (1 + (1j / (2.0 * TRCWA_Q_ABS)))

    if torch.tensor(run_cfg.TRCWA_L_grid).max() * target_frequency > 3.0:
        logger.warning(
            f"With a frequency of {target_frequency} and L_grid={run_cfg.TRCWA_L_grid} TRCWA may become unstable. Consider decreasing the target frequency or decreasing L_grid if R+T+A > 1."
        )

    # Initialize TRCWA object
    trcwa = TRCWA(
        run_cfg.TRCWA_NG,
        run_cfg.TRCWA_L_grid[0],
        run_cfg.TRCWA_L_grid[1],
        freqcmp,
        TRCWA_THETA,
        TRCWA_PHI,
        verbose=0,
    )

    # Add vacuum layer at the top
    trcwa.Add_LayerUniform(
        thickness=1.0, epsilon=torch.tensor(run_cfg.TRCWA_TOP_LAYER_EPS)
    )

    # Add material layers (homogeneous if Nx and Ny are 1)
    for layer in range(N_layers):

        # Set thickness based on config
        if len(run_cfg.PER_LAYER_THICKNESS) == 1:
            thickness = run_cfg.PER_LAYER_THICKNESS[0]
        else:
            thickness = run_cfg.PER_LAYER_THICKNESS[layer]

        if Nx > 1 or Ny > 1:
            trcwa.Add_LayerGrid(thickness, Nx, Ny)
        else:
            trcwa.Add_LayerUniform(thickness, eps_grid[layer])

    # Add layer at the bottom
    trcwa.Add_LayerUniform(
        thickness=1.0, epsilon=torch.tensor(run_cfg.TRCWA_BOTTOM_LAYER_EPS)
    )

    # Initialize the object properly
    trcwa.Init_Setup()

    trcwa.MakeExcitationPlanewave(
        TRCWA_PLANEWAVE["p_amp"],
        TRCWA_PLANEWAVE["p_phase"],
        TRCWA_PLANEWAVE["s_amp"],
        TRCWA_PLANEWAVE["s_phase"],
        order=0,
    )

    # Apply eps if heterogeneous layers
    if Nx > 1 or Ny > 1:
        trcwa.GridLayer_geteps(eps_grid)

    return trcwa
