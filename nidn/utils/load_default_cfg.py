import os
import toml
from dotmap import DotMap


def load_default_cfg():
    """Loads the default toml config file from the cfg folder."""
    with open("../../cfg/default_config.toml") as cfg:
        return DotMap(toml.load(cfg))
