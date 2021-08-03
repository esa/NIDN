import torch

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

# elementary charge, C
_TRCWA_E_CHARGE = 1.602176634 * 10 ** (-19)
# weight of electron, kg
_TRCWA_M_E = 9.1093837015 * 10 ** (-31)
# vacuum permittivity, F/m
_TRCWA_EPS_0 = 8.8541878128 * 10 ** (-12)

# There is no torch.pi so we define it here
_TRCWA_PI = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732
