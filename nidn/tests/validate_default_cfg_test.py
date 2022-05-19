from ..utils.load_default_cfg import load_default_cfg
from ..utils.validate_config import _validate_config


def test_default_config():
    """Tests whether the default config is valid"""
    cfg = load_default_cfg()
    _validate_config(cfg)


if __name__ == "__main__":
    test_default_config()
