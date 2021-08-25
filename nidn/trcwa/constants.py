import torch

# NB: Here, we assume that the standard is micrometre (um).
TRCWA_UNIT_MAGNITUDE = 10 ** (-7)

# m /s
TRCWA_SPEED_OF_LIGHT = 299792458.0

# Q_ABS is a parameter for relaxation to better approach global optimal, at Q_ABS= inf, it will describe the real physics.
# It can also be used to resolve singular matrix issues by setting a large but finite Q_ABS, e.g. Q_ABS= 1e5.
# See page 2 of arxiv.org/pdf/2005.04840v1.pdf
# or doi.org/10.1364/OE.21.030812 for a more thorough explanation
TRCWA_Q_ABS = 1e5

# Dielectric constant for top and bottom layer (vacuum layer)
TRCWA_VACUUM_EPS = torch.tensor(1.0)

TRCWA_PER_LAYER_THICKNESS = 1.0

# Planewave excitation
TRCWA_PLANEWAVE = {"p_amp": 0, "s_amp": 1, "p_phase": 0, "s_phase": 0}

# Angles
TRCWA_THETA = 0.0
TRCWA_PHI = 0.0

# elementary charge, C
TRCWA_E_CHARGE = 1.602176634 * 10 ** (-19)
# weight of electron, kg
TRCWA_M_E = 9.1093837015 * 10 ** (-31)
# vacuum permittivity, F/m
TRCWA_EPS_0 = 8.8541878128 * 10 ** (-12)

# There is no torch.pi so we define it here
TRCWA_PI = torch.tensor(3.14159265358979323846264338327950288)
