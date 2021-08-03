import os

# Set main device by default to cpu if no other choice was made before
if "TORCH_DEVICE" not in os.environ:
    os.environ["TORCH_DEVICE"] = "cpu"

# Add exposed features here
from .utils.convert_units import freq_to_wl, wl_to_phys_wl, phys_freq_to_phys_wl
from .utils.fix_random_seeds import fix_random_seeds
from .plots.plot_model_grid import plot_model_grid
from .plots.plot_model_grid_per_freq import plot_model_grid_per_freq
from .plots.plot_spectra import plot_spectra

__all__ = [
    "freq_to_wl",
    "wl_to_phys_wl",
    "phys_freq_to_phys_wl",
    "fix_random_seeds",
    "plot_model_grid",
    "plot_model_grid_per_freq",
    "plot_spectra",
]
