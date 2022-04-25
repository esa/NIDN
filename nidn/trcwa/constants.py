import torch


# Q_ABS is a parameter for relaxation to better approach global optimal, at Q_ABS= inf, it will describe the real physics.
# It can also be used to resolve singular matrix issues by setting a large but finite Q_ABS, e.g. Q_ABS= 1e5.
# See page 2 of arxiv.org/pdf/2005.04840v1.pdf
# or doi.org/10.1364/OE.21.030812 for a more thorough explanation
TRCWA_Q_ABS = 1e5

# Planewave excitation
TRCWA_PLANEWAVE = {"p_amp": 0, "s_amp": 1, "p_phase": 0, "s_phase": 0}

# Angles
TRCWA_THETA = 0.0
TRCWA_PHI = 0.0
