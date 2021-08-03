import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from nidn.utils.convert_units import freq_to_wl


def plot_spectra(
    prod_R_spectrum,
    prod_T_spectrum,
    target_R_spectrum,
    target_T_spectrum,
    target_frequencies,
):
    """Plots the produced RTA spectra together with the target spectra.
    
    Args: 
        prod_R_spectrum (torch.tensor): The produced reflection spectrum.
        prod_T_spectrum (torch.tensor): The produced transmission spectrum.
        target_R_spectrum (torch.tensor): The target reflection spectrum.
        target_T_spectrum (torch.tensor): The target transmission spectrum.
        target_frequencies (list of float): The frequencies for which we calculate R, T, A. 

    """

    prod_R_spectrum = torch.tensor(prod_R_spectrum).detach().cpu().numpy()
    prod_T_spectrum = torch.tensor(prod_T_spectrum).detach().cpu().numpy()
    target_R_spectrum = torch.tensor(target_R_spectrum).detach().cpu().numpy()
    target_T_spectrum = torch.tensor(target_T_spectrum).detach().cpu().numpy()

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
            - 0.225,
            max(max(prod_A_spectrum), max(prod_T_spectrum), max(prod_R_spectrum)) + 0.1,
        ]

    fig = plt.figure(figsize=(15, 5), dpi=150)
    fig.patch.set_facecolor("white")

    ax = fig.add_subplot(131)
    ax.plot(
        freq_to_wl(target_frequencies), prod_R_spectrum, marker=6, c="cornflowerblue"
    )
    ax.plot(freq_to_wl(target_frequencies), target_R_spectrum, marker=7, c="limegreen")
    ax.legend(["Produced Reflectance", "Target Reflectance"], loc="lower center")
    ax.set_xlabel("Wavelength [µm]")
    ax.set_ylabel("Reflectance")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    # ax.xaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
    ax.axhspan(-6, 0, facecolor="gray", alpha=0.3)
    ax.axhspan(1, 5, facecolor="gray", alpha=0.3)
    ax.set_ylim(ylimits)

    ax = fig.add_subplot(132)
    ax.plot(
        freq_to_wl(target_frequencies), prod_T_spectrum, marker=6, c="cornflowerblue"
    )
    ax.plot(freq_to_wl(target_frequencies), target_T_spectrum, marker=7, c="limegreen")
    ax.legend(["Produced Transmittance", "Target Transmittance"], loc="lower center")
    ax.set_xlabel("Wavelength [µm]")
    ax.set_ylabel("Transmittance")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    # ax.xaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
    ax.axhspan(-6, 0, facecolor="gray", alpha=0.3)
    ax.axhspan(1, 5, facecolor="gray", alpha=0.3)
    ax.set_ylim(ylimits)

    ax = fig.add_subplot(133)
    ax.plot(
        freq_to_wl(target_frequencies), prod_A_spectrum, marker=6, c="cornflowerblue"
    )
    ax.plot(freq_to_wl(target_frequencies), target_A_spectrum, marker=7, c="limegreen")

    ax.legend(["Produced Absorptance", "Target Absorptance"], loc="lower center")
    ax.set_xlabel("Wavelength [µm]")
    ax.set_ylabel("Absorptance")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    # ax.xaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
    ax.axhspan(-6, 0, facecolor="gray", alpha=0.3)
    ax.axhspan(1, 5, facecolor="gray", alpha=0.3)
    ax.set_ylim(ylimits)

    plt.tight_layout()
