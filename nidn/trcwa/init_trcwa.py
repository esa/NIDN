from .constants import *
from .trcwa import TRCWA


def _init_trcwa(eps_grid, target_frequency):
    """Creates a TRCWA object matching the given eps_grid and target_frequency.

    Args:
        eps_grid (torch.tensor): Grid of epsilon values. Should be [Nx, Ny, N_layers].
        target_frequency (float): Target frequency for this simulation run.

    Returns:
        TRCWA obj: The created object which is ready to compute the spectrum
    """
    Nx, Ny, N_layers = eps_grid.shape[0:3]

    # Squeeze out Nx=1, Ny=1 dimension (for uniform layer)
    eps_grid = eps_grid.squeeze(0)
    eps_grid = eps_grid.squeeze(0)

    # Adding a small imaginary part to the frequency to avoid singularities in RCWA
    # See page 2 of arxiv.org/pdf/2005.04840v1.pdf
    # or doi.org/10.1364/OE.21.030812 for a more thorough explanation
    freqcmp = target_frequency * (1 + (1j / (2.0 * TRCWA_Q_ABS)))

    # Initialize TRCWA object
    trcwa = TRCWA(
        TRCWA_NG,
        TRCWA_L1,
        TRCWA_L2,
        freqcmp,
        TRCWA_THETA,
        TRCWA_PHI,
        verbose=0,
    )

    # Add vacuum layer at the top
    trcwa.Add_LayerUniform(TRCWA_PER_LAYER_THICKNESS, TRCWA_VACUUM_EPS)

    # Add material layers (homogeneous if Nx and Ny are 1)
    for layer in range(N_layers):
        if Nx > 1 or Ny > 1:
            trcwa.Add_LayerGrid(TRCWA_PER_LAYER_THICKNESS, Nx, Ny)
        else:
            trcwa.Add_LayerUniform(TRCWA_PER_LAYER_THICKNESS, eps_grid[layer])

    # Add vacuum layer at the bottom
    trcwa.Add_LayerUniform(TRCWA_PER_LAYER_THICKNESS, TRCWA_VACUUM_EPS)

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
