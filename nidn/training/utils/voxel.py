from torch import nn
import torch


class Voxel(nn.Module):
    """A simple model directly encoding a 3d grid"""

    def __init__(self, cfg):
        super().__init__()

        assert (
            cfg.type == "regression"
        ), "VoxelGrid is currently only implemented for regression."

        # init weights randomly
        init_w = torch.rand((cfg.Nx, cfg.Ny, cfg.N_layers, cfg.N_freq), dtype=torch.cfloat)

        # init to min,max as specified in cfg
        init_w.real *= cfg.real_max_eps - cfg.real_min_eps
        init_w.real += cfg.real_min_eps

        init_w.imag *= cfg.imag_max_eps - cfg.imag_min_eps
        init_w.imag += cfg.imag_min_eps

        # Init grid of weights
        self.weights = nn.Parameter(init_w   )

       

        self.weights = self.weights.requires_grad_(True)

    def forward(self):
        return self.weights
