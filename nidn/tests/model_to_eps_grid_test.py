import torch

from ..materials.material_collection import MaterialCollection
from ..training.model.model_to_eps_grid import model_to_eps_grid
from ..training.model.init_network import init_network
from ..trcwa.compute_target_frequencies import compute_target_frequencies
from ..utils.load_default_cfg import load_default_cfg


def test_model_to_eps_grid():
    """Tests if these functions run. The correctness of the results is not checked."""

    # Test classification
    cfg = load_default_cfg()
    cfg.type = "classification"

    # Determine target frequencies
    cfg.target_frequencies = compute_target_frequencies(
        cfg.physical_wavelength_range[0],
        cfg.physical_wavelength_range[1],
        cfg.N_freq,
        cfg.freq_distribution,
    )

    cfg.material_collection = MaterialCollection(cfg.target_frequencies)
    cfg.N_materials = cfg.material_collection.N_materials
    # If classification, the model outputs a likelihood for the presence of each material.
    cfg.out_features = cfg.N_materials

    model = init_network(cfg)
    eps, ids = model_to_eps_grid(model, cfg)

    assert ids is not None
    assert eps.shape == torch.Size([cfg.Nx, cfg.Ny, cfg.N_layers, cfg.N_freq])

    # Test regression
    cfg.out_features = 2

    cfg.type = "regression"

    model = init_network(cfg)
    eps, ids = model_to_eps_grid(model, cfg)
    assert ids is None
    assert eps.shape == torch.Size([cfg.Nx, cfg.Ny, cfg.N_layers, cfg.N_freq])
