from .material_collection import MaterialCollection

import torch


def _find_closest_material(eps, run_cfg, material_collection):
    """Finds the closest matching material and distance to that (in epsilon)
    from a given epsilon grid.

    Args:
        eps (torch.tensor): A tensor of epsilon values (Nx,Ny,N_layers,N_freq).
        run_cfg (DotMap): Run configuration.
        material_collection (MaterialCollection): The material collection.

    Returns:
        tuple: The closest materials and distances to that.
    """
    Nx, Ny, N_layers, N_freq = run_cfg.Nx, run_cfg.Ny, run_cfg.N_layers, run_cfg.N_freq

    comparisons = torch.zeros(
        [
            Nx,
            Ny,
            N_layers,
            N_freq,
            material_collection.N_materials,
        ]
    )

    # Compute differences for all materials
    for idx, material_name in enumerate(material_collection.material_names):
        material_eps = material_collection[material_name]

        comparisons[..., idx] = torch.abs(eps - material_eps)

    # Find minimal entries
    comparisons = comparisons.mean(dim=3)  # Average over frequencies
    minimal_comparisons, indices = torch.min(comparisons, dim=-1)

    return minimal_comparisons, indices
