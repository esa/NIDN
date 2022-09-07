import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoLocator
from ..utils.convert_units import freq_to_wl, wl_to_phys_wl
from ..utils.compute_spectrum import compute_spectrum
from ..training.model.model_to_eps_grid import model_to_eps_grid
from ..utils.global_constants import NIDN_PLOT_COLOR_1, NIDN_PLOT_COLOR_2, NIDN_FONTSIZE


def _add_plot(
    fig,
    target_frequencies,
    produced_spectrum,
    target_spectrum,
    ylimits,
    nr,
    type_name,
    logscale=False,
    markers=True,
):
    fontsize = NIDN_FONTSIZE
    markersize = 8 if markers else 0

    ax = fig.add_subplot(nr)
    freqs = wl_to_phys_wl(freq_to_wl(target_frequencies)) * 1e6  # in µm
    ax.plot(
        freqs,
        target_spectrum,
        marker="o",
        c=NIDN_PLOT_COLOR_2,
        lw=4,
        markersize=markersize,
    )
    ax.plot(
        freqs,
        produced_spectrum,
        linestyle=("--" if not markers else "-"),
        marker="o",
        c=NIDN_PLOT_COLOR_1,
        lw=4,
        markersize=markersize,
    )
    ax.legend(
        [f"Target {type_name}", f"Produced {type_name}"],
        # loc="lower center",
        fontsize=fontsize,
    )
    ax.set_xlabel("Wavelength [µm]", fontsize=fontsize + 2)
    ax.set_ylabel(f"{type_name}", fontsize=fontsize + 2)
    if logscale:
        ax.set_xscale("log")
    ax.tick_params(axis="both", which="major", labelsize=fontsize)

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: "{:.1f}".format(x)))
    ax.xaxis.set_major_locator(AutoLocator())
    plt.minorticks_off()
    # ax.xaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
    # ax.axhspan(-6, 0, facecolor="gray", alpha=0.3)
    # ax.axhspan(1, 5, facecolor="gray", alpha=0.3)
    ax.set_ylim(ylimits)

    L1_err = abs(target_spectrum - produced_spectrum).mean()
    ax.text(
        0.01,
        1.05,
        f"L1 Error = {L1_err:.4f}",
        va="center",
        fontsize=fontsize,
        transform=ax.transAxes,
    )
    return fig


def plot_spectra(
    run_cfg,
    save_path=None,
    prod_R_spectrum=None,
    prod_T_spectrum=None,
    markers=True,
    filename=None,
    ylim=[[0.0, 1.0], [0.0, 1.0]],
):
    """Plots the produced RTA spectra together with the target spectra. Optionally saves it.

    Args:
        run_cfg (dict): The run configuration.
        save_path (str, optional): Folder to save the plot in. Defaults to None, then the plot will not be saved.
        prod_R_spectrum (torch.tensor, optional): The produced reflection spectrum. Defaults to None, then will compute from model.
        prod_T_spectrum (torch.tensor, optional): The produced transmission spectrum. Defaults to None, then will compute from model.
        markers (bool): Whether to plot markers for the target and produced spectra.
        filename (str, optional): Filename to save the plot in. Defaults to None, then the plot will be saved with the name "spectra.png".
        ylim (list): The y-limits of the plot for the two spectra. Defaults to [[0.0, 1.0],[0.0, 1.0]].
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
    # Code below is for gray bars in spectra
    # ylimits = [-0.2, 1.075]
    # if (
    #     (max(prod_A_spectrum) > 1 or min(prod_A_spectrum) < 0)
    #     or (max(prod_T_spectrum) > 1 or min(prod_T_spectrum) < 0)
    #     or (max(prod_R_spectrum) > 1 or min(prod_R_spectrum) < 0)
    # ):
    #     ylimits = [
    #         min(min(prod_A_spectrum), min(prod_T_spectrum), min(prod_R_spectrum))
    #         + ylimits[0],
    #         max(max(prod_A_spectrum), max(prod_T_spectrum), max(prod_R_spectrum)) + 0.1,
    #     ]

    fig = plt.figure(figsize=(15, 5), dpi=300)
    fig.patch.set_facecolor("white")

    fig = _add_plot(
        fig,
        target_frequencies,
        prod_R_spectrum,
        target_R_spectrum,
        ylim[0],
        121,
        "Reflectance",
        markers=markers,
    )
    fig = _add_plot(
        fig,
        target_frequencies,
        prod_T_spectrum,
        target_T_spectrum,
        ylim[1],
        122,
        "Transmittance",
        markers=markers,
    )
    # fig = _add_plot(
    #     fig,
    #     target_frequencies,
    #     prod_A_spectrum,
    #     target_A_spectrum,
    #     ylimits,
    #     133,
    #     "Absorptance",
    # )

    fig.autofmt_xdate()
    plt.tight_layout()

    if save_path is not None:
        if filename is None:
            plt.savefig(save_path + "/spectra_comparison.png", dpi=150)
        else:
            plt.savefig(save_path + "/" + filename + ".png", dpi=300)
        # fig.clf()
        # plt.close(fig)
    else:
        plt.show()
