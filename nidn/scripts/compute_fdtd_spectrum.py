import fdtd
from dotmap import DotMap
import sys
import torch
sys.path.append("../")

import nidn

# This function computes the transmission and reflection spectrum of the material composition specified in the cfg file,
# by using FDTD simulations. Example and documentations can be found here: https://fdtd.readthedocs.io/en/latest/examples.html
def compute_fdtd_spectrum(cfg: DotMap):
    transmission_spectrum = []
    reflection_spectrum = []
    fdtd.set_backend("torch")
    grid_spacing = cfg.physical_wavelength_range[0]*0.1
    grid_size_x = 6e-6
    grid_size_y = grid_spacing * 50 # optimal value to be investigated 
    transmission_detector_x = int(3.6*10**(-6)/(grid_spacing))
    reflection_detector_x = int(2.4*10**(-6)/(grid_spacing))
    source_x = int(1.6*10**(-6)/(grid_spacing))
    source_y = grid_size_y/2
    pml_thickness = int(1.5*10**(-6)/(grid_spacing)) # PML is a Perfectly Matched Layer, used as boundaries to absorb almost all radiation.
    object_start_x = int(2.5*10**(-6)/(grid_spacing))
    object_end_x = int(3.5*10**(-6)/(grid_spacing))
    pulse_source = False
    niter = 200
    speed_of_light: float = 299_792_458.0  # [m/s] speed of light
    use_point_source = True # If false, line source is used
    # Run two simulations for each frequency, one in free space and one with an object placed. Two detectors are placed in the grid, one just before the object and one just after. 
    # The transmission and reflectance coefficiioent respectively are calculating by dividing the rms value for the case with an object by the case wiithout the object
    
    wavelengths, normalized_freq = nidn.get_frequency_points(cfg)
    for wavelength in range(wavelengths):
        #init grid
        grid = fdtd.Grid(
            (grid_size_x, grid_size_y, 1),
            grid_spacing=grid_spacing,
            permittivity=1.0,
            permeability=1.0
        )
        grid = _add_boundaries_to_grid(grid,pml_thickness)
        #Source
        grid = _add_source_to_grid(grid,use_point_source,source_x,source_y,wavelength/speed_of_light,pulse_source)

        #Detectors
        grid, transmission_detector, reflection_detector = _add_detectors_to_grid(grid, transmission_detector_x, reflection_detector_x)        

        # run simulation
        grid.run(niter, progress_bar=False)
        transmission_output = transmission_detector.detector_values()
        reflection_output = reflection_detector.detector_values()
        # Add object
        grid2 = fdtd.Grid(
            (grid_size_x, grid_size_y, 1),
            grid_spacing=grid_spacing,
            permittivity=1.0,
            permeability=1.0
        )
        grid2 = _add_boundaries_to_grid(grid2,pml_thickness)
        grid2 = _add_source_to_grid(grid2,use_point_source,source_x,source_y,wavelength/speed_of_light,pulse_source)
        grid2 = _add_object_to_grid(grid2,object_start_x,object_end_x)
        
        # Run simulation again
        grid2.run(niter, progress_bar = False)
        transmission_output2 = transmission_detector2.detector_values()
        reflection_output2 = reflection_detector2.detector_values()

        transmission1_e_field_z_direction = [e[int(source_x)][2] for e in transmission_output['E']]
        reflection_e_field_z_direction = [e[int(source_x)][2] for e in reflection_output['E']]
        transmission1_e_field_z_direction2 = [e[int(source_x)][2] for e in transmission_output2['E']]
        reflection_e_field_z_direction2 = [e[int(source_x)][2] for e in reflection_output2['E']]
        # Substract the free space signal from the reflection detector in the case where the object is present
        # to ensure oly the reflected signal is measured 
        compensated_reflection = [reflection_e_field_z_direction2[i]-reflection_e_field_z_direction[i] for i in range(len(reflection_e_field_z_direction2))]


        # As the energy of a wave is proportional to E^2, and rms is theoretically equal E/2, only the mean squre is used, ie
        transmission_ms_1 = _calculate_mean_square(transmission1_e_field_z_direction)
        transmission_ms_2 = _calculate_mean_square(transmission1_e_field_z_direction2)
        reflection_ms_1 = _calculate_mean_square(reflection_e_field_z_direction)
        reflection_ms_2 = _calculate_mean_square(compensated_reflection)

        transmission_spectrum.append(transmission_ms_2/transmission_ms_1)
        reflection_spectrum.append(reflection_ms_2/reflection_ms_1)
        grid.reset()
        grid2.reset()
    return transmission_spectrum, reflection_spectrum

# Calculate the spectrum from a cfg file, and plot this.
# params: 
# cfg: the cfg file containing parameters to use in the simulation
# returns : None
def plot_fdtd_spectrum(cfg: DotMap):
    transmission_spectrum, reflection_spectrum = compute_fdtd_spectrum(cfg)
    nidn.plot_spectrum(cfg,reflection_spectrum,transmission_spectrum)

# Helper function to add boundaries to a created grid. It creates PML boundaries in the x-direction, and periodic boundaries in the y-direction
# TODO: Implement further customability of boundary types, i.e. choose where pml or periodic boundaries should be
#
# params: 
# grid: the grid which to add the boundaries
# pml_thickness: the thickness of the perfectly matched layer.

# returns:
# grid: the grid with the attached boundaries

def _add_boundaries_to_grid(grid,pml_thickness):
    grid[0:pml_thickness, :, :] = fdtd.PML(name="pml_xlow")
    grid[-pml_thickness:, :, :] = fdtd.PML(name="pml_xhigh")
    grid[:, 0, :] = fdtd.PeriodicBoundary(name="ybounds")

    return grid

# Helper function to add a source to a created grid. Uses either a point source or a line source at the specified location
#
# params: 
# grid: the grid which to add the boundaries
# use_point_source: bool - Point source if true, Line source if false
# source_x, source_y: x and y location of the source, y position insignificant if line-source
# period: period of the wave the source is emitting. Pass wavelength/speed_of_light
# pulse_source: Whether the source should emitt a single pulse (True), or use a continuouss wave(False) 

# returns:
# grid: the grid with the attached source

def _add_source_to_grid(grid, use_point_source, source_x, source_y, period, pulse_source):
    if(use_point_source):
        grid[source_x,source_y, 0] = fdtd.PointSource(
        period=period, name="linesource",
        pulse =pulse_source,
        cycle=1,
        hanning_dt=1e-15)
    else:
        grid[source_x,:, 0] = fdtd.LineSource(
        period=period, name="linesource")

    return grid


# Helper function to add detectors to the grid
#
# params: 
# grid: the grid which to add the boundaries
# transmission_detector_x, reflection_detector_x: The x location of the transmission and reflection detector.

# returns:
# grid: the grid with the attached detectors
# transmission_detector, reflection_detector: The two detectors for further use after the simulation is run
def _add_detectors_to_grid(grid, transmission_detector_x, reflection_detector_x):

    transmisssion_detector = fdtd.LineDetector(name="t_detector")
    reflection_detector = fdtd.LineDetector(name="r_detector")
    grid[transmission_detector_x, :, 0] =transmisssion_detector
    grid[reflection_detector_x,:,0] = reflection_detector

    return grid, transmisssion_detector, reflection_detector

# Helper function to add an object to the grid
#
# params: 
# grid: the grid which to add the object
# object_start_x, object_end_x: The start and end position of the object

# returns:
# grid: the grid with the attached detectors
def _add_object_to_grid(grid, object_start_x, object_end_x):
    grid[object_start_x:object_end_x,:, :] = fdtd.AnisotropicObject(permittivity=5.1984, name="object")
    return grid


# Helper function which calculates the mean square of an array
# params: 
# arr: signal array on which t do the calculations on

# returns:
# The mean square value of the signal 
def _calculate_mean_square(arr):
    return torch.mean(arr**2)