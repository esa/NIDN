from ..utils.siren import Siren
from ..utils.nerf import NERF
from dotmap import DotMap


def init_network(
    run_cfg: DotMap,
):
    """Network architecture. Note that the dimensionality of the first linear layer must match the output of the encoding chosen in the config file.
    Returns:
        torch model: Initialized model
    """
    if run_cfg.model_type == "nerf":
        return NERF(
            in_features=run_cfg.encoding_dim,
            n_neurons=run_cfg.n_neurons,
            activation=run_cfg.activation,
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
            outermost_activation=run_cfg.activation,
            first_omega_0=run_cfg.siren_omega,
            hidden_omega_0=run_cfg.siren_omega,
        )
