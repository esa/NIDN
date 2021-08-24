import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from ..utils.convert_units import freq_to_wl
from ..trcwa.compute_spectrum import compute_spectrum
from ..training.model.model_to_eps_grid import model_to_eps_grid


def _add_plot(fig, target_frequencies, spectrum, ylimits, nr, type_name):
    ax = fig.add_subplot(nr)
    ax.plot(freq_to_wl(target_frequencies), spectrum, marker=7, c="limegreen")
    ax.legend([f"Produced {type_name}", f"Target {type_name}"])
    ax.set_xlabel("Wavelength [µm]")
    ax.set_ylabel(f"{type_name}")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    # ax.xaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
    ax.axhspan(-6, 0, facecolor="gray", alpha=0.3)
    ax.axhspan(1, 5, facecolor="gray", alpha=0.3)
    ax.set_ylim(ylimits)
    return fig


def plot_spectrum(run_cfg, R_spectrum, T_spectrum):
    """Plots the produced RTA spectra.

    Args:
        run_cfg (dict): The run configuration.
        R_spectrum (torch.tensor): The reflection spectrum.
        T_spectrum (torch.tensor): The transmission spectrum.
    """

    target_frequencies = run_cfg.target_frequencies

    # Convert the spectra to numpy arrays for matplotlib
    R_spectrum = torch.tensor(R_spectrum).detach().cpu().numpy()
    T_spectrum = torch.tensor(T_spectrum).detach().cpu().numpy()

    A_spectrum = np.ones_like(np.asarray(R_spectrum)) - (
        np.asarray(T_spectrum) + np.asarray(R_spectrum)
    )

    # To align all plots
    ylimits = [-0.05, 1]
    if (
        (max(A_spectrum) > 1 or min(A_spectrum) < 0)
        or (max(T_spectrum) > 1 or min(T_spectrum) < 0)
        or (max(R_spectrum) > 1 or min(R_spectrum) < 0)
    ):
        ylimits = [
            min(min(A_spectrum), min(T_spectrum), min(R_spectrum)) + ylimits[0],
            max(max(A_spectrum), max(T_spectrum), max(R_spectrum)) + 0.1,
        ]

    fig = plt.figure(figsize=(15, 5), dpi=150)
    fig.patch.set_facecolor("white")

    fig = _add_plot(fig, target_frequencies, R_spectrum, ylimits, 131, "Reflectance",)
    fig = _add_plot(fig, target_frequencies, T_spectrum, ylimits, 132, "Transmittance",)
    fig = _add_plot(fig, target_frequencies, A_spectrum, ylimits, 133, "Absorptance",)

    plt.tight_layout()
