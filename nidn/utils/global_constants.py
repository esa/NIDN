import torch

# Elemental charge
E_CHARGE = 1.602176634 * 10 ** (-19)
# Mass of electron
M_E = 9.1093837015 * 10 ** (-31)
# Vacuum permittivity
EPS_0 = 8.8541878128 * 10 ** (-12)
# There is no torch.pi so we define it here
PI = torch.tensor(3.14159265358979323846264338327950288)
# Speed of light in vacuum
SPEED_OF_LIGHT = 299792458.0
# Unit size in simulations
UNIT_MAGNITUDE = 10 ** (-6)
# Vacuum permeability
MU_0 = 4e-7 * PI
# Vacuum impedance
ETA_0 = MU_0 * SPEED_OF_LIGHT


##### PLOTTING
NIDN_FONTSIZE = 16
NIDN_PLOT_COLOR_1 = "#0173b2"
NIDN_PLOT_COLOR_2 = "#de8f05"
