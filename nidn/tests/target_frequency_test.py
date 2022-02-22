from loguru import logger

from ..trcwa.compute_target_frequencies import compute_target_frequencies
from ..utils.global_constants import UNIT_MAGNITUDE


def test_conversion():
    """Test whether conversion from real wavelengths into target frequencies
    is correctly done.
    """
    logger.trace("Running frequency conversion test...")
    min_wl = 300e-9  # 300 nm
    max_wl = 3e-6  # 3 µm

    target_frequencies = compute_target_frequencies(min_wl, max_wl, 5, "linear")
    logger.debug("Proposed frequencies are")
    logger.debug(target_frequencies)

    assert len(target_frequencies) == 5

    prod_min_wl = 1.0 / target_frequencies[-1] * UNIT_MAGNITUDE
    prod_max_wl = 1.0 / target_frequencies[0] * UNIT_MAGNITUDE

    assert abs(prod_min_wl - min_wl) < 1e-8
    assert abs(prod_max_wl - max_wl) < 1e-8


if __name__ == "__main__":
    test_conversion()
