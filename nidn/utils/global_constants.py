import torch

# Elemental charge
E_CHARGE = 1.602176634 * 10 ** (-19)
# Mass of electron
M_E = 9.1093837015 * 10 ** (-31)
# Vacuum permittivity
EPS_0 = torch.tensor(8.8541878128 * 10 ** (-12))
# There is no torch.pi so we define it here
PI = torch.tensor(3.14159265358979323846264338327950288)
# Speed of light in vacuum
SPEED_OF_LIGHT = 299792458.0
# Unit size in simulations
UNIT_MAGNITUDE = 10 ** (-6)
# Vacuum permeability
MU_0: float = 4e-7 * PI
# Vacuum impedance
ETA_0: float = MU_0 * SPEED_OF_LIGHT
