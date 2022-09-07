from matplotlib import pyplot as plt
import numpy as np


def plot_losses(run_cfg, save_path=None):
    """Plots the loss over training, optionally saving it.

    Args:
        run_cfg (DotMap): Run configuration.
        save_path (str, optional): Folder to save the plot in. Defaults to None, then the plot will not be saved.
    """
    fig = plt.figure(figsize=(8, 4), dpi=150)
    fig.patch.set_facecolor("white")
    plt.semilogy(run_cfg.results.L1_errs)
    plt.semilogy(run_cfg.results.loss_log)
    plt.semilogy(run_cfg.results.weighted_average_log)
    plt.xlabel("Model evaluations")
    plt.ylabel("Loss")
    plt.legend(["L1", "Loss", "Weighted Average Loss"])
    if save_path is not None:
        plt.savefig(save_path + "/losses.png", dpi=150)
    else:
        plt.show()

