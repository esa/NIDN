import torch
import numpy as np
from matplotlib import pyplot as plt

from ..training.model.model_to_eps_grid import model_to_eps_grid


def plot_model_grid(run_cfg, save_path=None):
    """Plots the absolute value of the epsilon over all frequencies for each 3-D grid point. Optionally saves it.
    Args:
        run_cfg (dict): The run configuration.
        save_path (str, optional): Folder to save the plot in. Defaults to None, then the plot will not be saved.
    """
    Nx, Ny, N_layers = run_cfg.Nx, run_cfg.Ny, run_cfg.N_layers
    eps, _ = model_to_eps_grid(run_cfg.model, run_cfg)
    x = torch.linspace(-1, 1, Nx)
    y = torch.linspace(-1, 1, Ny)
    z = torch.linspace(-1, 1, N_layers)
    X, Y, Z = torch.meshgrid((x, y, z))

    # Here we calculate the absolute value of the permittivity over all frequencies for each grid point
    eps = torch.mean(eps, dim=3)
    eps = torch.sqrt(eps.real**2 + eps.imag**2)

    abs_values = eps.detach().cpu().numpy()

    X = X.cpu().numpy()
    Y = Y.cpu().numpy()
    Z = Z.cpu().numpy()

    # Here we plot it
    fig = plt.figure(figsize=(4, 4), dpi=150)
    fig.patch.set_facecolor("white")

    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=25, azim=100)
    p = ax.scatter(
        X.reshape(-1, 1),
        Y.reshape(-1, 1),
        Z.reshape(-1, 1),
        marker="s",
        s=120,
        linewidths=0,
        alpha=1.0,
        c=abs_values,
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
    ax.set_zticklabels(
        z_labels
    )  # Where L1 is (seemingly) the bottom one ( #TODO confirm this)
    cbar.set_label("Complex norm of the permittivity", labelpad=10)

    if save_path is not None:
        plt.savefig(save_path + "/model_grid.png", dpi=150)
    else:
        plt.show()
