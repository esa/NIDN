import os
import toml
from dotmap import DotMap


def load_default_cfg():
    """Loads the default toml config file from the cfg folder."""
    path = os.path.join(
        os.path.dirname(__file__) + "/resources/", "default_config.toml"
    )
    with open(path) as cfg:
        # dynamic=False inhibits automatic generation of non-existing keys
        return DotMap(toml.load(cfg), _dynamic=False)
