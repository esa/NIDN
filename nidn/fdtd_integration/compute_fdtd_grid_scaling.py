import torch
from .constants import FDTD_GRID_SCALE
from ..utils.global_constants import UNIT_MAGNITUDE


def _compute_fdtd_grid_scaling(cfg):
    """Compute the scaling of the FDTD grid

    Args:
        cfg (DotMap): Run config

    Returns:
        torch.float: The scaling is the number of grid points per unit magnitude. This is the maximum of the relation between the unit magnitude and 1/10th of the smallest wavelength,
                     and a constant which is defaulted to 10. If this scaling becomes too low, i.e. below 2, there might be some errors in creating the grid,
                     as there are too few grid points for certain elements to be placed correctly.
    """
    return torch.maximum(
        torch.tensor(
            UNIT_MAGNITUDE / (cfg.physical_wavelength_range[0] * FDTD_GRID_SCALE)
        ),
        torch.tensor(cfg.FDTD_min_gridpoints_per_unit_magnitude),
    )
