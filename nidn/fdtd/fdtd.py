import fdtd


class FDTD:
    def __init__(self, grid_size_x, grid_size_y, grid_spacing):
        self.grid = fdtd.Grid(
            (grid_size_x, grid_size_y, 1),
            grid_spacing=grid_spacing,
            permittivity=1.0,
            permeability=1.0,
        )

    def AddBoundaries(self, pml_thickness):
        # Add theboundaries to the FDTD grid

        self.grid[0:pml_thickness, :, :] = fdtd.PML(name="pml_xlow")
        self.grid[-pml_thickness:, :, :] = fdtd.PML(name="pml_xhigh")
        self.grid[:, 0, :] = fdtd.PeriodicBoundary(name="ybounds")

    def AddSource(self, source_x, source_y, period, use_pulse_source, use_point_source):
        # Add the source to the FDTD grid

        if use_point_source:
            self.grid[source_x, source_y, 0] = fdtd.PointSource(
                period=period,
                name="linesource",
                pulse=use_pulse_source,
                cycle=1,
                hanning_dt=1e-15,
            )
        else:
            self.grid[source_x, :, 0] = fdtd.LineSource(
                period=period, name="linesource"
            )

    def AddDetectors(self, transmission_detector_x, reflection_detector_x):
        # Add detectors to the FDTD grid

        self.transmisssion_detector = fdtd.LineDetector(name="t_detector")
        self.reflection_detector = fdtd.LineDetector(name="r_detector")
        self.grid[transmission_detector_x, :, 0] = self.transmisssion_detector
        self.grid[reflection_detector_x, :, 0] = self.reflection_detector

    def AddObject(self):
        # Add an object to the FDTD grid
        pass

    def Run(self, niter):
        # Run the FDTD simulation with the current grid
        pass

    def GetDetectorValues(self):
        # Get values from the transmission detector and the reflection detector.
        # This will be six lists, each list having either magnetic or electric field in either x,y or z direction

        pass

    def GetCoefficients(self, wavelength):
        # Get transmission coeffiient and reflection coefficient for the specified wavelength
        # Run two simulation, one with object and one in free space, and compare the two detector values to get coefficients
        pass

    def GetSpectrum(self):
        # Run GetCoeiccients for each wavelength to generate spectrum
        pass

    def PlotSpectrum(self):
        # Plot the spectrum
        pass
