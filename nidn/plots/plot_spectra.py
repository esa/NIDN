import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from ..utils.convert_units import freq_to_wl
from ..trcwa.compute_spectrum import compute_spectrum
from ..training.model.model_to_eps_grid import model_to_eps_grid


def _add_plot(
    fig, target_frequencies, produced_spectrum, target_spectrum, ylimits, nr, type_name
):
    ax = fig.add_subplot(nr)
    ax.plot(
        freq_to_wl(target_frequencies), produced_spectrum, marker=6, c="cornflowerblue"
    )
    ax.plot(freq_to_wl(target_frequencies), target_spectrum, marker=7, c="limegreen")
    ax.legend([f"Produced {type_name}", f"Target {type_name}"], loc="lower center")
    ax.set_xlabel("Wavelength [µm]")
    ax.set_ylabel(f"{type_name}")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    # ax.xaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
    ax.axhspan(-6, 0, facecolor="gray", alpha=0.3)
    ax.axhspan(1, 5, facecolor="gray", alpha=0.3)
    ax.set_ylim(ylimits)
    return fig


def plot_spectra(model, run_cfg):
    """Plots the produced RTA spectra together with the target spectra.

    Args:
        model (torch.model): The model to be plotted.
        run_cfg (dict): The run configuration.
    """
    target_R_spectrum = run_cfg.target_reflectance_spectrum
    target_T_spectrum = run_cfg.target_transmittance_spectrum

    # Create epsilon grid from the model
    eps, _ = model_to_eps_grid(model, run_cfg)

    # Compute the spectra for the given epsilon values
    prod_R_spectrum, prod_T_spectrum = compute_spectrum(eps, run_cfg)

    target_frequencies = run_cfg.target_frequencies

    # Convert the spectra to numpy arrays for matplotlib
    prod_R_spectrum = torch.tensor(prod_R_spectrum).detach().cpu().numpy()
    prod_T_spectrum = torch.tensor(prod_T_spectrum).detach().cpu().numpy()
    target_R_spectrum = torch.tensor(target_R_spectrum).detach().cpu().numpy()
    target_T_spectrum = torch.tensor(target_T_spectrum).detach().cpu().numpy()

    # Compute absorption spectra
    prod_A_spectrum = np.ones_like(np.asarray(prod_R_spectrum)) - (
        np.asarray(prod_T_spectrum) + np.asarray(prod_R_spectrum)
    )
    target_A_spectrum = np.ones_like(np.asarray(target_R_spectrum)) - (
        np.asarray(target_T_spectrum) + np.asarray(target_R_spectrum)
    )

    # To align all plots
    ylimits = [-0.225, 1]
    if (
        (max(prod_A_spectrum) > 1 or min(prod_A_spectrum) < 0)
        or (max(prod_T_spectrum) > 1 or min(prod_T_spectrum) < 0)
        or (max(prod_R_spectrum) > 1 or min(prod_R_spectrum) < 0)
    ):
        ylimits = [
            min(min(prod_A_spectrum), min(prod_T_spectrum), min(prod_R_spectrum))
            + ylimits[0],
            max(max(prod_A_spectrum), max(prod_T_spectrum), max(prod_R_spectrum)) + 0.1,
        ]

    fig = plt.figure(figsize=(15, 5), dpi=150)
    fig.patch.set_facecolor("white")

    fig = _add_plot(
        fig,
        target_frequencies,
        prod_R_spectrum,
        target_R_spectrum,
        ylimits,
        131,
        "Reflectance",
    )
    fig = _add_plot(
        fig,
        target_frequencies,
        prod_T_spectrum,
        target_T_spectrum,
        ylimits,
        132,
        "Transmittance",
    )
    fig = _add_plot(
        fig,
        target_frequencies,
        prod_A_spectrum,
        target_A_spectrum,
        ylimits,
        133,
        "Absorptance",
    )

    plt.tight_layout()
