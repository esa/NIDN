# We can probably steal the code in ConvertUnits from S4 for this purpose:
# S4.ConvertUnits(value, from_units, to_units)
# NB: Here, we assume that the standard is micrometre (um).
_TRCWA_UNIT_MAGNITUDE = 10 ** (-6)

# m /s
_TRCWA_SPEED_OF_LIGHT = 299792458.0
