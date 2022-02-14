from scipy.fft import fft


def CalculateCoefficient(transmission, reflection, method):
        # Get transmission coeffiient and reflection coefficient for the specified wavelength
        # Run two simulation, one with object and one in free space, and compare the two detector values to get coefficients
        # If method == 'MS*, then use the mean square to get the coefficients
        # If method == 'FT', use fourier transform method

        if method.upper() == 'MS':
            transmission_coefficient = _mean_square(transmission[1])/_mean_square(transmission[0])
            reflection_coefficient = _mean_square(reflection[1])/_mean_square(reflection[0])
        elif method.upper() == 'FFT':
            transmission_coefficient = _fft(transmission[1])/fft(transmission[0])*  # Some exponentian should be multiplied here
            reflection_coefficient = _fft(reflection[1])/fft(reflection[0])*
def _mean_square(arr):
    return sum([e**2 for e in arr]) / len(arr)

def _fft():

    return fft