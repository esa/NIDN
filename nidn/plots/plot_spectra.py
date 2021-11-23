import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from ..utils.convert_units import freq_to_wl, wl_to_phys_wl
from ..trcwa.compute_spectrum import compute_spectrum
from ..training.model.model_to_eps_grid import model_to_eps_grid


def _add_plot(
    fig, target_frequencies, produced_spectrum, target_spectrum, ylimits, nr, type_name
):
    fontsize = 16

    ax = fig.add_subplot(nr)
    freqs = wl_to_phys_wl(freq_to_wl(target_frequencies)) * 1e6  # in µm
    ax.plot(freqs, produced_spectrum, marker=6, c="cornflowerblue")
    ax.plot(freqs, target_spectrum, marker=7, c="limegreen")
    ax.legend(
        [f"Produced {type_name}", f"Target {type_name}"],
        loc="lower center",
        fontsize=fontsize - 4,
    )
    ax.set_xlabel("Wavelength [µm]", fontsize=fontsize)
    ax.set_ylabel(f"{type_name}", fontsize=fontsize)
    ax.set_xscale("log")
    ax.tick_params(axis="both", which="major", labelsize=fontsize - 2)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    # ax.xaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
    ax.axhspan(-6, 0, facecolor="gray", alpha=0.3)
    ax.axhspan(1, 5, facecolor="gray", alpha=0.3)
    ax.set_ylim(ylimits)

    L1_err = abs(target_spectrum - produced_spectrum).mean()
    ax.text(
        freqs[-1] - 0.02,
        1.06,
        f"L1 Error = {L1_err:.4f}",
        va="top",
        fontsize=fontsize - 4,
    )
    return fig


def plot_spectra(
    run_cfg, save_path=None, prod_R_spectrum=None, prod_T_spectrum=None, filename=None
):
    """Plots the produced RTA spectra together with the target spectra. Optionally saves it.

    Args:
        run_cfg (dict): The run configuration.
        save_path (str, optional): Folder to save the plot in. Defaults to None, then the plot will not be saved.
        prod_R_spectrum (torch.tensor, optional): The produced reflection spectrum. Defaults to None, then will compute from model.
        prod_T_spectrum (torch.tensor, optional): The produced transmission spectrum. Defaults to None, then will compute from model.
        filename (str, optional): Filename to save the plot in. Defaults to None, then the plot will be saved with the name "spectra.png".
    """
    target_R_spectrum = run_cfg.target_reflectance_spectrum
    target_T_spectrum = run_cfg.target_transmittance_spectrum

    if prod_R_spectrum is None or prod_T_spectrum is None:
        # Create epsilon grid from the model
        eps, _ = model_to_eps_grid(run_cfg.model, run_cfg)

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
    ylimits = [-0.2, 1.075]
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

    if save_path is not None:
        if filename is None:
            plt.savefig(save_path + "/spectra_comparison.png", dpi=150)
        else:
            plt.savefig(save_path + "/" + filename + ".png", dpi=300)
