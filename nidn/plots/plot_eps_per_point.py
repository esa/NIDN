from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from ..utils.convert_units import freq_to_wl
from ..training.model.model_to_eps_grid import model_to_eps_grid


def plot_eps_per_point(model, run_cfg):
    """This function plots the epsilon values of grid points against real materials

    Args:
        model (torch.model): The model to be plotted.
        run_cfg (dict): The run configuration.
    """
    # Create epsilon grid from the model
    eps, _ = model_to_eps_grid(model, run_cfg)
    eps = eps.detach().cpu().numpy()

    # Create figure
    fig = plt.figure(figsize=(15, 5), dpi=150)
    fig.patch.set_facecolor("white")

    # Plot epsilon
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    # Iterate over all grid points
    for x in range(eps.shape[0]):
        for y in range(eps.shape[1]):
            for N_layer in range(eps.shape[2]):
                eps_point_real = eps[x, y, N_layer].real
                eps_point_imag = eps[x, y, N_layer].imag
                ax.plot(freq_to_wl(run_cfg.target_frequencies), eps_point_real)
                ax2.plot(freq_to_wl(run_cfg.target_frequencies), eps_point_imag)
