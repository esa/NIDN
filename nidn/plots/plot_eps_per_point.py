import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoLocator, AutoMinorLocator

from ..utils.convert_units import freq_to_wl
from ..training.model.model_to_eps_grid import model_to_eps_grid
from ..trcwa.load_material_data import _load_material_data
from ..materials.material_collection import MaterialCollection
from ..materials.find_closest_material import _find_closest_material


def plot_eps_per_point(run_cfg, compare_to_material=None, save_path=None, legend=True):
    """This function plots the epsilon values of grid points against real materials. Optionally saves it.

    Args:
        run_cfg (dict): The run configuration.
        compare_to_material (str or list): Name(s) of the material to compare with. Available ones are in /materials/data.
        save_path (str, optional): Folder to save the plot in. Defaults to None, then the plot will not be saved.
    """
    # Create epsilon grid from the model
    eps, _ = model_to_eps_grid(run_cfg.model, run_cfg)
    eps_np = eps.detach().cpu().numpy()

    material_collection = MaterialCollection(run_cfg.target_frequencies)

    # Load material data for comparison
    if compare_to_material is not None:
        if type(compare_to_material) is str:
            material_data = material_collection[compare_to_material]
        else:
            material_data = []
            for mat in compare_to_material:
                material_data.append(material_collection[mat])

    # Create figure
    fig = plt.figure(figsize=(10, 6), dpi=150)
    fig.patch.set_facecolor("white")

    # Plot epsilon
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    wl = freq_to_wl(run_cfg.target_frequencies)

    # Add some horizontal space
    ax.set_xlim(wl.min(), wl.max() + (0.3 * (wl.max() - wl.min())))
    ax2.set_xlim(wl.min(), wl.max() + (0.3 * (wl.max() - wl.min())))

    ax.set_xlabel("Wavelength [µm]")
    ax.set_ylabel("Epsilon real part")
    ax.set_xscale("log")

    ax2.set_xlabel("Wavelength [µm]")
    ax2.set_ylabel("Epsilon imaginary part")
    ax2.set_xscale("log")

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: "{:.1f}".format(x)))
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: "{:.1f}".format(x)))

    ax.xaxis.set_major_locator(AutoLocator())
    ax2.xaxis.set_major_locator(AutoLocator())

    # Iterate over all grid points
    for x in range(eps.shape[0]):
        for y in range(eps.shape[1]):
            for N_layer in range(eps.shape[2]):
                eps_point_real = eps_np[x, y, N_layer].real
                eps_point_imag = eps_np[x, y, N_layer].imag
                ax.plot(wl, eps_point_real, linewidth=1)
                ax.minorticks_off()
                ax2.plot(wl, eps_point_imag, linewidth=1)
                ax2.minorticks_off()

                # if not legend:
                #     ax.text(
                #         wl.max(),
                #         eps_point_real[0],
                #         f" {N_layer},{x},{y}",
                #         va="center",
                #         fontsize=7,
                #     )
                #     ax2.text(
                #         wl.max(),
                #         eps_point_imag[0],
                #         f" {N_layer},{x},{y}",
                #         va="center",
                #         fontsize=7,
                #     )

    if legend:
        # Add legend
        names = [
            f"{N_layer},{x},{y}"
            for x in range(eps.shape[0])
            for y in range(eps.shape[1])
            for N_layer in range(eps.shape[2])
        ]
        ax.legend(names)
        ax2.legend(names)

    # Plot material data
    if compare_to_material is not None:
        if type(compare_to_material) is str:
            material_data = [material_data]
            compare_to_material = [compare_to_material]
        for mat_data, mat_name in zip(material_data, compare_to_material):

            ax.plot(wl, mat_data.real, "--", color="black", linewidth=1.5)
            ax.minorticks_off()
            ax2.plot(wl, mat_data.imag, "--", color="black", linewidth=1.5)
            ax2.minorticks_off()
            ax.text(
                wl.max(),
                mat_data.real[0],
                " " + mat_name,
                va="center",
                fontsize=7,
            )
            ax2.text(
                wl.max(),
                mat_data.imag[0],
                " " + mat_name,
                va="center",
                fontsize=7,
            )
    else:
        _, indices = _find_closest_material(eps, run_cfg, material_collection)
        unique_indices = torch.unique(indices)
        names = [material_collection.material_names[i] for i in unique_indices]
        for name in names:
            material_data = material_collection[name]
            ax.plot(wl, material_data.real, "--", color="black", linewidth=1.5)
            ax2.plot(wl, material_data.imag, "--", color="black", linewidth=1.5)
            ax.text(
                wl.max(), material_data.real[0], " " + name, va="center", fontsize=7
            )
            ax2.text(
                wl.max(), material_data.imag[0], " " + name, va="center", fontsize=7
            )

    fig.autofmt_xdate()

    if save_path is not None:
        plt.savefig(save_path + "/eps_per_points.png", dpi=150)
