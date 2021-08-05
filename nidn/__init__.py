import os

# Set main device by default to cpu if no other choice was made before
if "TORCH_DEVICE" not in os.environ:
    os.environ["TORCH_DEVICE"] = "cpu"

# Add exposed features here
from .plots.plot_model_grid import plot_model_grid
from .plots.plot_model_grid_per_freq import plot_model_grid_per_freq
from .plots.plot_spectra import plot_spectra
from .training.run_training import run_training
from .trcwa.compute_target_frequencies import compute_target_frequencies
from .trcwa.get_frequency_points import get_frequency_points
from .utils.convert_units import freq_to_wl, wl_to_phys_wl, phys_freq_to_phys_wl
from .utils.fix_random_seeds import fix_random_seeds
from .utils.load_default_cfg import load_default_cfg
from .utils.print_cfg import print_cfg

__all__ = [
    "compute_target_frequencies",
    "get_frequency_points",
    "fix_random_seeds",
    "freq_to_wl",
    "load_default_cfg",
    "run_training",
    "phys_freq_to_phys_wl",
    "plot_model_grid",
    "plot_model_grid_per_freq",
    "plot_spectra",
    "print_cfg",
    "wl_to_phys_wl",
]
