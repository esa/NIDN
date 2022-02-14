from dotmap import DotMap
import fdtd

from nidn.fdtd.constants import FDTD_UNIT_MAGNITUDE
from nidn.utils.global_constants import SPEED_OF_LIGHT


def init_fdtd(cfg: DotMap, include_object, wavelength, permittivity):
    """Initialize the FDTD grid, with detectors, source, boundaries and optional objects

    Args:
        cfg (DotMap): configuration
        include_object (bool): Whether an object should be added to the grid or not
        wavelength (float): Wavelength to be used by the source
        permittivity (array): array of permittivities of the object for each layer

    Returns:
        fdtd:Grid: Grid with all the added object, ready to be run
    """
    scaling = FDTD_UNIT_MAGNITUDE / (cfg.physical_wavelength_range[0] * 0.1)
    grid = fdtd.Grid(
        (
            int(cfg.FDTD_grid[0] * scaling),
            int(
                cfg.FDTD_grid[1] * scaling
                + cfg.N_layers * scaling * cfg.FDTD_per_layer_thickness
            ),
            int(cfg.FDTD_grid[2] * scaling),
        ),
        grid_spacing=cfg.physical_wavelength_range[0] * 0.1,
        permittivity=1.0,
        permeability=1.0,
    )
    grid = _add_boundaries(grid, int(cfg.FDTD_pml_thickness * scaling))
    grid, t_detector, r_detector = _add_detectors(
        grid,
        int(
            cfg.FDTD_reflection_detector_x * scaling
            + cfg.N_layers * scaling * cfg.FDTD_per_layer_thickness
        ),
        int(cfg.FDTD_reflection_detector_x * scaling),
    )
    grid = _add_source(
        grid,
        int(cfg.FDTD_source[0] * scaling),
        int(cfg.FDTD_source[1] * scaling),
        wavelength / SPEED_OF_LIGHT,
        cfg.FDTD_use_pulsesource,
        cfg.FDTD_use_pointsource,
    )
    if include_object:
        for i in range(cfg.N_layers):
            grid = _add_object(
                grid,
                int(
                    cfg.FDTD_free_space_distance * scaling
                    + i * scaling * cfg.FDTD_per_layer_thickness
                ),
                int(
                    cfg.FDTD_object[1] * scaling
                    + (i + 1) * scaling * cfg.FDTD_per_layer_thickness
                ),
                permittivity[i],
            )
    return grid, t_detector, r_detector


def _add_boundaries(grid, pml_thickness):
    """Add the desired boundaries to teh grid

    Args:
        grid (fdtd.Grid): The grid object which to add the boundaries
        pml_thickness (float): Thickness of the PML boundaries,

    Returns:
        fdtd.Grid: The grid with the added boundaries
    """

    grid[0:pml_thickness, :, :] = fdtd.PML(name="pml_xlow")
    grid[-pml_thickness:, :, :] = fdtd.PML(name="pml_xhigh")
    grid[:, 0, :] = fdtd.PeriodicBoundary(name="ybounds")
    return grid


def _add_source(grid, source_x, source_y, period, use_pulse_source, use_point_source):
    """Add a specified source to the fdtd grid

    Args:
        grid (fdtd.Grid): The grid object which to add the source
        source_x (float): The x coordinates of the placement of the source
        source_y (float): The y coordinates of the placement of the source
        period (float): The period which the source should use, given by wavelength / speed_of_light
        use_pulse_source (bool): True if pulse source is desired, Fasle to use continuous wave
        use_point_source (bool): True if a point source is desired, False if a LineSource is desired

    Returns:
        fdtd.Grid: The grid with the added source
    """

    if use_point_source:
        grid[source_x, source_y, 0] = fdtd.PointSource(
            period=period,
            name="linesource",
            pulse=use_pulse_source,
            cycle=1,
            hanning_dt=1e-15,
        )
    else:
        grid[source_x, :, 0] = fdtd.LineSource(period=period, name="linesource")

    return grid


def _add_detectors(grid, transmission_detector_x, reflection_detector_x):
    """Add a transmission detector and reflection detector to the fdtd grid

    Args:
        grid (fdtd.Grid): The grid object which to add the detectors
        transmission_detector_x (float): x coordinates of the transmission detector
        reflection_detector_x (float): x coordinates of teh reflection detector

    Returns:
        fdtd.Grid: The grid with the added detectors
    """

    transmission_detector = fdtd.LineDetector(name="t_detector")
    reflection_detector = fdtd.LineDetector(name="r_detector")
    grid[transmission_detector_x, :, 0] = transmission_detector
    grid[reflection_detector_x, :, 0] = reflection_detector
    return grid, transmission_detector, reflection_detector


def _add_object(grid, object_start_x, object_end_x, permittivity):
    """Add a object to the fdtd grid, with a specified permittivity. The object covers the entire grid in the y-direction.

    Args:
        grid (fdtd:Grid): The grid object which to add the object
        object_start_x (float): Coordinate where the object should start in the x-direction
        object_end_x (float): Coordinate where the object should end in the x-direction
        permittivity (float): The dielectric permittivity of the object

    Returns:
        fdtd.Grid: The grid with the added object
    """

    grid[object_start_x:object_end_x, :, :] = fdtd.AnisotropicObject(
        permittivity=permittivity, name="object"
    )
    return grid
