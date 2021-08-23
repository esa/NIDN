import os
import inspect

from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from ..utils.convert_units import freq_to_wl
from ..training.model.model_to_eps_grid import model_to_eps_grid
from ..trcwa.load_material_data import _load_material_data
from ..materials.material_collection import MaterialCollection


def plot_eps_per_point(model, run_cfg, compare_to_material=None):
    """This function plots the epsilon values of grid points against real materials

    Args:
        model (torch.model): The model to be plotted.
        run_cfg (dict): The run configuration.
        compare_to_material (str): Name of the material to compare to. Available ones are in /materials/data.
    """
    # Create epsilon grid from the model
    eps, _ = model_to_eps_grid(model, run_cfg)
    eps = eps.detach().cpu().numpy()

    # Load material data for comparison
    if compare_to_material is not None:
        material_data = _load_material_data(
            os.path.dirname(inspect.getfile(MaterialCollection))
            + "/data/"
            + compare_to_material
            + ".csv",
            run_cfg.target_frequencies,
        )

    # Create figure
    fig = plt.figure(figsize=(10, 5), dpi=150)
    fig.patch.set_facecolor("white")

    # Plot epsilon
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    wl = freq_to_wl(run_cfg.target_frequencies)

    # Add some horizontal space
    ax.set_xlim(wl[0], 2.5 * wl[-1])
    ax2.set_xlim(wl[0], 2.5 * wl[-1])

    ax.set_xlabel("Wavelength [µm]")
    ax.set_ylabel("Epsilon real part")
    ax.set_xscale("log")

    ax2.set_xlabel("Wavelength [µm]")
    ax2.set_ylabel("Epsilon imaginary part")
    ax2.set_xscale("log")

    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax2.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    # Iterate over all grid points
    for x in range(eps.shape[0]):
        for y in range(eps.shape[1]):
            for N_layer in range(eps.shape[2]):
                eps_point_real = eps[x, y, N_layer].real
                eps_point_imag = eps[x, y, N_layer].imag
                ax.plot(wl, eps_point_real, linewidth=1)
                ax2.plot(wl, eps_point_imag, linewidth=1)

                ax.text(wl[-1], eps_point_real[-1], f" {N_layer},{x},{y}", va="center")
                ax2.text(wl[-1], eps_point_imag[-1], f" {N_layer},{x},{y}", va="center")

    # Plot material data
    if compare_to_material is not None:
        ax.plot(wl, material_data.real, "--", color="black", linewidth=1.5)
        ax2.plot(wl, material_data.imag, "--", color="black", linewidth=1.5)