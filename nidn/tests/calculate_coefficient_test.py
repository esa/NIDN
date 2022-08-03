import torch

from nidn.utils.global_constants import PI
from ..fdtd_integration.calculate_transmission_reflection_coefficients import (
    calculate_transmission_reflection_coefficients,
)
from ..utils.load_default_cfg import load_default_cfg


def test_calculate_coefficient():
    cfg = load_default_cfg()
    time_points = torch.tensor([0.002 * PI * i for i in range(4000)])
    signal_a = 2 * torch.sin(time_points)
    signal_b = torch.sin(time_points)
    signal_array = [signal_a, signal_b]
    (
        transmission_coefficient_ms,
        reflection_coefficient_ms,
    ) = calculate_transmission_reflection_coefficients(
        signal_array, signal_array, 3e-5, cfg
    )
    assert transmission_coefficient_ms.item() - 0.25 < 1e-7
    assert reflection_coefficient_ms.item() - 0.25 < 1e-7


if __name__ == "__main__":
    test_calculate_coefficient()
