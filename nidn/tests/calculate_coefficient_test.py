import numpy as np
from ..fdtd_integration.calculate_transmission_reflection_coefficients import (
    calculate_transmission_reflection_coefficients,
)
from ..utils.load_default_cfg import load_default_cfg


def test_calculate_coefficient():
    cfg = load_default_cfg()
    time_points = [0.002 * np.pi * i for i in range(1000)]
    signal_a = 2 * np.sin(time_points)
    signal_b = np.sin(time_points)
    signal_array = [signal_a, signal_b]
    (
        transmission_coefficient_ms,
        reflection_coefficient_ms,
    ) = calculate_transmission_reflection_coefficients(
        signal_array, signal_array, "mean square", cfg
    )
    # TODO: Add test for fft, when the method is complete
    assert transmission_coefficient_ms - 0.25 == 0
    assert reflection_coefficient_ms - 0.25 == 0


if __name__ == "__main__":
    test_calculate_coefficient()
