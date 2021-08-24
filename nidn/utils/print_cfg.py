from dotmap import DotMap
import numpy as np

from ..materials.material_collection import MaterialCollection


def print_cfg(cfg: DotMap):
    """Prints the config in a more readable way.

    Args:
        cfg (DotMap): Config to print.
    """
    # Print the config three values per line
    idx = 0
    for key, value in cfg.items():
        if isinstance(value, list) or isinstance(value, np.ndarray):
            print()
            print(f"{key}: {value}")
            idx = 0
        elif isinstance(value, MaterialCollection):
            print()
        else:
            if idx % 3 == 2:
                print(f"{key:<23}: {value:<15}|")
            else:
                print(f"{key:<23}: {value:<15}|", end="")
        idx += 1
    print()
