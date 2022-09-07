import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoLocator
from ..utils.convert_units import freq_to_wl
from ..trcwa.compute_target_frequencies import compute_target_frequencies
from ..utils.global_constants import NIDN_PLOT_COLOR_1, NIDN_FONTSIZE


def _add_plot(
    fig,
    target_frequencies,
    spectrum,
    ylimits,
    nr,
    type_name,
    logscale=False,
    markers=True,
):
    fontsize = NIDN_FONTSIZE
    markersize = 8 if markers else 0

    ax = fig.add_subplot(nr)
    ax.plot(
        freq_to_wl(target_frequencies),
        spectrum,
        marker="o",
        c=NIDN_PLOT_COLOR_1,
        lw=2,
        markersize=markersize,
    )
    ax.legend(
        [f"Produced {type_name}", f"Target {type_name}"],
        fontsize=fontsize,
    )
    ax.set_xlabel("Wavelength [µm]", fontsize=fontsize + 2)
    ax.set_ylabel(f"{type_name}", fontsize=fontsize + 2)
    if logscale:
        ax.set_xscale("log")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: "{:.1f}".format(x)))
    # ax.xaxis.set_minor_formatter(plt.FuncFormatter(lambda x, pos: "{:.1f}".format(x)))
    ax.tick_params(axis="both", which="major", labelsize=fontsize)

    ax.xaxis.set_major_locator(AutoLocator())
    plt.minorticks_off()
    # ax.axhspan(-6, 0, facecolor="gray", alpha=0.3)
    # ax.axhspan(1, 5, facecolor="gray", alpha=0.3)
    ax.set_ylim(ylimits)
    return fig


def plot_spectrum(
    run_cfg,
    R_spectrum,
    T_spectrum,
    markers=True,
    save_path=None,
    filename=None,
    show_absorption=False,
):
    """Plots the produced RTA spectra. Optionally saves it.

    Args:
        run_cfg (dict): The run configuration.
        R_spectrum (torch.tensor): The reflection spectrum.
        T_spectrum (torch.tensor): The transmission spectrum.
        markers (bool): Whether to plot markers for the target and produced spectra.
        save_path (str, optional): Folder to save the plot in. Defaults to None, then the plot will not be saved.
        filename (str, optional): Filename to save the plot in. Defaults to None, then the plot will be saved with the name "spectrum.png".
        show_absorption (bool, optional): Whether to show the absorption spectrum. Defaults to False.
    """

    if not "target_frequencies" in run_cfg.keys():
        run_cfg.target_frequencies = compute_target_frequencies(
            run_cfg.physical_wavelength_range[0],
            run_cfg.physical_wavelength_range[1],
            run_cfg.N_freq,
            run_cfg.freq_distribution,
        )
    target_frequencies = run_cfg.target_frequencies

    # Convert the spectra to numpy arrays for matplotlib
    R_spectrum = torch.tensor(R_spectrum).detach().cpu().numpy()
    T_spectrum = torch.tensor(T_spectrum).detach().cpu().numpy()

    A_spectrum = np.ones_like(np.asarray(R_spectrum)) - (
        np.asarray(T_spectrum) + np.asarray(R_spectrum)
    )

    # To align all plots
    ylimits = [0.0, 1.0]
    # if (
    #     (max(A_spectrum) > 1 or min(A_spectrum) < 0)
    #     or (max(T_spectrum) > 1 or min(T_spectrum) < 0)
    #     or (max(R_spectrum) > 1 or min(R_spectrum) < 0)
    # ):
    #     ylimits = [
    #         min(min(A_spectrum), min(T_spectrum), min(R_spectrum)) + ylimits[0],
    #         max(max(A_spectrum), max(T_spectrum), max(R_spectrum)) + 0.1,
    #     ]

    fig = plt.figure(figsize=(12, 4), dpi=150)
    fig.patch.set_facecolor("white")

    fig = _add_plot(
        fig,
        target_frequencies,
        R_spectrum,
        ylimits,
        (121 if not show_absorption else 131),
        "Reflectance",
        markers=markers,
    )
    fig = _add_plot(
        fig,
        target_frequencies,
        T_spectrum,
        ylimits,
        (122 if not show_absorption else 132),
        "Transmittance",
        markers=markers,
    )
    if show_absorption:
        fig = _add_plot(
            fig,
            target_frequencies,
            A_spectrum,
            ylimits,
            133,
            "Absorptance",
        )

    fig.autofmt_xdate()
    plt.tight_layout()

    if save_path is not None:
        if filename is None:
            plt.savefig(save_path + "/spectrum.png", dpi=150)
        else:
            plt.savefig(save_path + "/" + filename + ".png", dpi=300)
    else:
        plt.show()
