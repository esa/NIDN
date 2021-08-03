import torch
import numpy as np
from matplotlib import pyplot as plt
from nidn.utils.convert_units import freq_to_wl


def plot_model_grid_per_freq(eps, grid_dim, target_frequencies, freq_idx=[0, 1, 2, 3]):
    """Plots the real and imaginary epsilon in two separate plots for each frequency in target_frequencies.
    
    Args: 
        eps (torch.tensor): Epsilon values.
        grid_dim (tuple): The dimensions of the grid, i.e. (Nx, Ny, N_layers)
        target_frequencies (list of float): The frequencies for which we calculate R, T, A. 
        freq_idx (list of int): Which of the frequency indices in target_frequencies we want to plot. Defaults to [0, 1, 2, 3].
    """
    Nx, Ny, N_layers = grid_dim
    x = torch.linspace(-1, 1, Nx)
    y = torch.linspace(-1, 1, Ny)
    z = torch.linspace(-1, 1, N_layers)
    X, Y, Z = torch.meshgrid((x, y, z))

    X = X.cpu().numpy()
    Y = Y.cpu().numpy()
    Z = Z.cpu().numpy()

    # Here we plot it
    fig = plt.figure(figsize=(8, 12), dpi=150)
    fig.patch.set_facecolor("white")

    min_eps_real, max_eps_real = eps.real.min(), eps.real.max()
    min_eps_imag, max_eps_imag = eps.imag.min(), eps.imag.max()

    for idx in range(len(freq_idx)):
        eps_real = eps[:, :, :, freq_idx[idx]].real
        eps_imag = eps[:, :, :, freq_idx[idx]].imag
        material_id_real = eps_real.detach().cpu().numpy()
        material_id_imag = eps_imag.detach().cpu().numpy()

        re_or_im_idx = 0
        for real_or_imag in [material_id_real, material_id_imag]:
            ax = fig.add_subplot(421 + re_or_im_idx + idx * 2, projection="3d")
            ax.view_init(elev=25, azim=100)
            vmin, vmax = min_eps_real, max_eps_real
            if re_or_im_idx > 0:
                vmin, vmax = min_eps_imag, max_eps_imag
            p = ax.scatter(
                X.reshape(-1, 1),
                Y.reshape(-1, 1),
                Z.reshape(-1, 1),
                marker="s",
                linewidths=0,
                s=140,
                alpha=1.0,
                c=real_or_imag,
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(
                "WL="
                + str(round(freq_to_wl(target_frequencies[freq_idx[idx]]), 2))
                + "um",
                fontsize=8,
            )
            # ax.set_xlabel("$N_x =$" + str(Nx))
            # ax.set_ylabel("$N_y =$" + str(Ny))
            # ax.set_zlabel("# of layers", rotation=60)  # TODO Fix rotation
            # ax.grid(False)  # Hide grid lines
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks(np.linspace(-1, 1, N_layers))
            z_labels = [""] * N_layers
            for layer in range(N_layers):
                z_labels[layer] = "L" + str(layer + 1)
            ax.set_zticklabels(z_labels)  # Where L1 is (seemingly) the bottom one
            cb = plt.colorbar(p, ax=ax)
            cb.set_label("Real permittivity, $\epsilon'$", rotation=270, labelpad=15)
            if re_or_im_idx == 1:
                cb.set_label(
                    "Imaginary permittivity, $\epsilon''$", rotation=270, labelpad=15
                )
            cb.ax.tick_params(labelsize=7)
            re_or_im_idx += 1
