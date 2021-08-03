import torch

# We can probably steal the code in ConvertUnits from S4 for this purpose:
# S4.ConvertUnits(value, from_units, to_units)
# NB: Here, we assume that the standard is micrometre (um).
_TRCWA_UNIT_MAGNITUDE = 10 ** (-6)

# m /s
_TRCWA_SPEED_OF_LIGHT = 299792458.0

_TRCWA_Q_ABS = 1e5

# Truncation order (actual number might be smaller)
_TRCWA_NG = 11

_TRCWA_L1 = [1, 0]  # everything is set relative to this
_TRCWA_L2 = [0, 1]  # it is easiest to say 1 = 1 micron

# Dielectric for top and bottom layer
_TRCWA_VACUUM_EPS = torch.tensor(1.0)

_TRCWA_PER_LAYER_THICKNESS = 1.0

# Planewave excitation
_TRCWA_PLANEWAVE = {"p_amp": 0, "s_amp": 1, "p_phase": 0, "s_phase": 0}

# Angles
_TRCWA_THETA = 0.0
_TRCWA_PHI = 0.0
