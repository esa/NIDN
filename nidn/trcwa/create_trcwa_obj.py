from .constants import *
from .trcwa import obj


def _create_trcwa_obj(eps_grid, target_frequency):
    """Creates a TRCWA object matching the given eps_grid and target_frequency.

    Args:
        eps_grid (torch.tensor): Grid of epsilon values. Should be [Nx, Ny, N_layers].
        target_frequency (float): Target frequency for this simulation run.

    Returns:
        TRCWA obj: The created object which is ready to compute the spectrum.
    """

    Nx, Ny, N_layers = eps_grid.shape[0:2]

    # Adding a small imaginary part to the frequency to avoid singularities in RCWA
    # TODO Add a nice reference for this?
    freqcmp = target_frequency * (1 + (1j / (2.0 * _TRCWA_Q_ABS)))

    # Initialize TRCWA object
    trcwa_object = obj(
        _TRCWA_NG,
        _TRCWA_L1,
        _TRCWA_L2,
        freqcmp,
        _TRCWA_THETA,
        _TRCWA_PHI,
        verbose=0,
    )

    # Add vacuum layer at the top
    trcwa_object.Add_LayerUniform(_TRCWA_PER_LAYER_THICKNESS, _TRCWA_VACUUM_EPS)

    # Add material layers (homogeneous if Nx and Ny are 1)
    for layer in range(N_layers):
        if Nx > 1 or Ny > 1:
            trcwa_object.Add_LayerGrid(_TRCWA_PER_LAYER_THICKNESS, Nx, Ny)
        else:
            trcwa_object.Add_LayerUniform(
                _TRCWA_PER_LAYER_THICKNESS, eps_grid.squeeze()[layer]
            )

    # Add vacuum layer at the bottom
    trcwa_object.Add_LayerUniform(_TRCWA_PER_LAYER_THICKNESS, _TRCWA_VACUUM_EPS)

    # Initialize the object properly
    trcwa_object.Init_Setup()

    trcwa_object.MakeExcitationPlanewave(
        _TRCWA_PLANEWAVE["p_amp"],
        _TRCWA_PLANEWAVE["p_phase"],
        _TRCWA_PLANEWAVE["s_amp"],
        _TRCWA_PLANEWAVE["s_phase"],
        order=0,
    )

    # Apply eps if heterogeneous layers
    if Nx > 1 or Ny > 1:
        trcwa_object.GridLayer_geteps(eps_grid)

    return trcwa_object
