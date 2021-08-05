from dotmap import DotMap
from loguru import logger
import torch.nn as nn

from ..utils.nerf import NERF
from ..utils.siren import Siren


def init_network(
    run_cfg: DotMap,
):
    """Network architecture. Note that the dimensionality of the first linear layer must match the output of the encoding chosen in the config file.
    Returns:
        torch model: Initialized model
    """
    logger.debug("Initializing model..." + run_cfg.model_type)
    if run_cfg.model_type == "nerf":
        return NERF(
            in_features=run_cfg.encoding_dim,
            out_features=run_cfg.out_features,
            n_neurons=run_cfg.n_neurons,
            activation=nn.Sigmoid(),
            skip=[4],
            hidden_layers=run_cfg.hidden_layers,
        )
    elif run_cfg.model_type == "siren":
        return Siren(
            in_features=run_cfg.encoding_dim,
            out_features=run_cfg.out_features,
            hidden_features=run_cfg.n_neurons,
            hidden_layers=run_cfg.hidden_layers,
            outermost_linear=True,
            outermost_activation=nn.Sigmoid(),
            first_omega_0=run_cfg.siren_omega,
            hidden_omega_0=run_cfg.siren_omega,
        )
