import torch
import numpy as np
from matplotlib import pyplot as plt

from ..materials.material_collection import MaterialCollection
from ..materials.find_closest_material import _find_closest_material
from ..training.model.model_to_eps_grid import model_to_eps_grid


def plot_material_grid(model, run_cfg, save_path=None):
    """Plots the materials closest to the used ones for each grid point. Optionally saves it.

    Args:
        model (torch.model): The model to be plotted.
        run_cfg (dict): The run configuration.
        save_path (str, optional): Path to save the plot at. Defaults to None, then the plot will not be saved.
    """
    Nx, Ny, N_layers = run_cfg.Nx, run_cfg.Ny, run_cfg.N_layers
    # Create epsilon grid from the model
    eps, _ = model_to_eps_grid(model, run_cfg)

    # Setup grid
    x = torch.linspace(-1, 1, Nx)
    y = torch.linspace(-1, 1, Ny)
    z = torch.linspace(-1, 1, N_layers)
    X, Y, Z = torch.meshgrid((x, y, z))

    # Load material data
    material_collection = MaterialCollection(run_cfg.target_frequencies)

    # Get closest materials
    errors, material_id = _find_closest_material(eps, run_cfg)

    cmap = plt.get_cmap("rainbow", material_collection.N_materials)

    # Here we plot it
    fig = plt.figure(figsize=(10, 5), dpi=150)
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(121, projection="3d")
    ax.view_init(elev=25, azim=100)
    p = ax.scatter(
        X.reshape(-1, 1),
        Y.reshape(-1, 1),
        Z.reshape(-1, 1),
        marker="s",
        s=120,
        linewidths=0,
        cmap=cmap,
        vmin=1 - 0.5,  # This is where we get the discrete colormap
        vmax=material_collection.N_materials
        + 0.5,  # This is where we get the discrete colormap
        alpha=1.0,
        c=material_id.detach().cpu().numpy() + 1,
    )
    cbar = fig.colorbar(
        p, ticks=np.arange(1, material_collection.N_materials + 1)
    )  # This is where we get the discrete colormap
    cbar.set_ticklabels(material_collection.material_names)
    cbar.ax.tick_params(labelsize=7)
    # cbar.set_label("Materials", labelpad=-1)

    ax.grid(False)  # Hide grid lines
    ax.set_xlabel("$N_x =$" + str(Nx))
    ax.set_ylabel("$N_y =$" + str(Ny))
    # ax.set_zlabel("# of layers", rotation=60)  # TODO Fix rotation
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks(np.linspace(-1, 1, N_layers))

    z_labels = [""] * N_layers
    for idx in range(N_layers):
        z_labels[idx] = "L" + str(idx + 1)
    ax.set_zticklabels(z_labels)  # Where L1 is (seemingly) the bottom one

    ax = fig.add_subplot(122, projection="3d")
    ax.view_init(elev=25, azim=100)
    p = ax.scatter(
        X.reshape(-1, 1),
        Y.reshape(-1, 1),
        Z.reshape(-1, 1),
        marker="s",
        s=120,
        linewidths=0,
        c=errors.detach().cpu().numpy(),
    )
    cbar = plt.colorbar(p, ax=ax)
    cbar.ax.tick_params(labelsize=7)
    ax.grid(False)  # Hide grid lines
    ax.set_xlabel("$N_x =$" + str(Nx))
    ax.set_ylabel("$N_y =$" + str(Ny))
    # ax.set_zlabel("# of layers", rotation=60)  # TODO Fix rotation
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks(np.linspace(-1, 1, N_layers))

    z_labels = [""] * N_layers
    for idx in range(N_layers):
        z_labels[idx] = "L" + str(idx + 1)
    ax.set_zticklabels(z_labels)  # Where L1 is (seemingly) the bottom one

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
