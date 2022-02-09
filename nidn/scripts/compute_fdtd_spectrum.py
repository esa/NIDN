from cProfile import run
import fdtd
from dotmap import DotMap
import sys
import math
sys.path.append(r"C:\Users\to-bo\OneDrive\Documents\ESA\NIDN\Developer\nidn")

import nidn


def compute_fdtd_spectrum(cfg: DotMap):
    transmission_spectrum = []
    reflection_spectrum = []
    fdtd.set_backend("torch")
    GRID_SPACING = cfg.physical_wavelength_range[0]*0.1
    GRID_SIZE_X = 6e-6
    GRID_SIZE_Y = GRID_SPACING * 50 # optimal value to be investigated 
    TRANSMISSION_DETECTOR_X = int(3.6*10**(-6)/(GRID_SPACING))
    REFLECTION_DETECTOR_X = int(2.4*10**(-6)/(GRID_SPACING))
    SOURCE_X = int(1.6*10**(-6)/(GRID_SPACING))
    SOURCE_Y = GRID_SIZE_Y/2
    PML_THICKNESS = int(1.5*10**(-6)/(GRID_SPACING))
    OBJECT_START_X = int(2.5*10**(-6)/(GRID_SPACING))
    OBJECT_END_X = int(3.5*10**(-6)/(GRID_SPACING))
    PULSE_SOURCE = False
    NITER = 200
    SPEED_LIGHT: float = 299_792_458.0  # [m/s] speed of light
    # Run two simulations for each frequency, one in free space and one with an object placed. Two detectors are placed in the grid, one just before the object and one just after. 
    # The transmission and reflectance coefficiioent respectively are calculating by dividing the rms value for the case with an object by the case wiithout the object
    for i in range(cfg.N_freq):
        WAVELENGTH: float = cfg.physical_wavelength_range[0] + i*(cfg.physical_wavelength_range[1]-cfg.physical_wavelength_range[0])/20
        grid = fdtd.Grid(
            (GRID_SIZE_X, GRID_SIZE_Y, 1),
            grid_spacing=GRID_SPACING,
            permittivity=1.0,
            permeability=1.0
        )
        # PML boundaries in x direction and periodic boundary in y direction.
        grid[0:PML_THICKNESS, :, :] = fdtd.PML(name="pml_xlow")
        grid[-PML_THICKNESS:, :, :] = fdtd.PML(name="pml_xhigh")
        grid[:, 0, :] = fdtd.PeriodicBoundary(name="ybounds")

        #Source
        point = True

        if(point):
            grid[SOURCE_X,SOURCE_Y, 0] = fdtd.PointSource(
            period=WAVELENGTH / SPEED_LIGHT, name="linesource",
            pulse =PULSE_SOURCE,
            cycle=1,
            hanning_dt=1e-15)
        else:
                grid[SOURCE_X,:, 0] = fdtd.LineSource(
            period=WAVELENGTH / SPEED_LIGHT, name="linesource")

        #Detectors

        transmisssion_detector = fdtd.LineDetector(name="t_detector")
        reflection_detector = fdtd.LineDetector(name="r_detector")
        grid[TRANSMISSION_DETECTOR_X, :, 0] =transmisssion_detector
        grid[REFLECTION_DETECTOR_X,:,0] = reflection_detector


        # run simulation
        grid.run(NITER, progress_bar=False)
        transmission_output = transmisssion_detector.detector_values()
        reflection_output = reflection_detector.detector_values()
        # Add object
        grid2 = fdtd.Grid(
            (GRID_SIZE_X, GRID_SIZE_Y, 1),
            grid_spacing=GRID_SPACING,
            permittivity=1.0,
            permeability=1.0
        )
        # PML boundaries in x direction and periodic boundary in y direction.
        grid2[0:PML_THICKNESS, :, :] = fdtd.PML(name="pml_xlow")
        grid2[-PML_THICKNESS:, :, :] = fdtd.PML(name="pml_xhigh")
        grid2[:, 0, :] = fdtd.PeriodicBoundary(name="ybounds")

        if(point):
            grid2[SOURCE_X,SOURCE_Y, 0] = fdtd.PointSource(
            period=WAVELENGTH / SPEED_LIGHT, name="pointsource",
            pulse =PULSE_SOURCE,
            cycle=1,
            hanning_dt=1e-15)
        else:
            grid2[SOURCE_X,:, 0] = fdtd.LineSource(
            period=WAVELENGTH / SPEED_LIGHT, name="linesource")

        #Detectors

        transmisssion_detector2 = fdtd.LineDetector(name="t_detector")
        reflection_detector2 = fdtd.LineDetector(name="r_detector")
        grid2[TRANSMISSION_DETECTOR_X, :, 0] =transmisssion_detector2
        grid2[REFLECTION_DETECTOR_X,:,0] = reflection_detector2
        grid2[OBJECT_START_X:OBJECT_END_X,:, :] = fdtd.AnisotropicObject(permittivity=5.1984, name="object")

        # run new simulation
        
        grid2.run(NITER, progress_bar = False)
        transmission_output2 = transmisssion_detector2.detector_values()
        reflection_output2 = reflection_detector2.detector_values()

        z_vals_t = [e[int(SOURCE_X)][2] for e in transmission_output['E']]
        z_vals_r = [e[int(SOURCE_X)][2] for e in reflection_output['E']]
        z_vals_t2 = [e[int(SOURCE_X)][2] for e in transmission_output2['E']]
        z_vals_r2 = [e[int(SOURCE_X)][2] for e in reflection_output2['E']]
        compensated_reflection = [z_vals_r2[i]-z_vals_r[i] for i in range(len(z_vals_r2))]
        #Phase-shift signal?
     
        #calculate rms
        t = [i for i in range(len(z_vals_t))]

        t_rms_1 = 0
        for i in z_vals_t:
            t_rms_1 += math.pow(i,2)
        t_rms_1 = t_rms_1/len(z_vals_t)

        t_rms_2 = 0
        for i in z_vals_t2:
            t_rms_2 += math.pow(i,2)
        t_rms_2 = t_rms_2/len(z_vals_t2)

        r_rms_1 = 0
        for i in z_vals_r:
            r_rms_1 += math.pow(i,2)
        r_rms_1 = r_rms_1/len(z_vals_r)

        r_rms_2 = 0
        for i in compensated_reflection:
            r_rms_2 += math.pow(i,2)
        r_rms_2 = r_rms_2/len(compensated_reflection)


        transmission_spectrum.append(t_rms_2/t_rms_1)
        reflection_spectrum.append(r_rms_2/r_rms_1)
        grid.reset()
        grid2.reset()
    return transmission_spectrum, reflection_spectrum

def plot_fdtd_spectrum(cfg: DotMap):
    transmission_spectrum, reflection_spectrum = compute_fdtd_spectrum(cfg)
    nidn.plot_spectrum(cfg,reflection_spectrum,transmission_spectrum)
