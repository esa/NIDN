from dotmap import DotMap

from ..fdtd import *
from ..utils.global_constants import EPS_0, SPEED_OF_LIGHT, UNIT_MAGNITUDE
from .constants import FDTD_GRID_SCALE


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
    set_backend("torch")
    scaling = UNIT_MAGNITUDE / (cfg.physical_wavelength_range[0] * FDTD_GRID_SCALE)
    x_grid_size = int(
        scaling * (cfg.FDTD_grid[0] + cfg.N_layers * cfg.PER_LAYER_THICKNESS[0])
    )
    y_grid_size = int(cfg.FDTD_grid[1] * scaling)
    grid = Grid(
        (x_grid_size, y_grid_size, 1),
        grid_spacing=cfg.physical_wavelength_range[0] * FDTD_GRID_SCALE,
        permittivity=1.0,
        permeability=1.0,
    )
    grid = _add_boundaries(grid, int(cfg.FDTD_pml_thickness * scaling))
    grid, t_detector, r_detector = _add_detectors(
        grid,
        int(
            scaling
            * (
                cfg.FDTD_reflection_detector_x
                + cfg.N_layers * cfg.PER_LAYER_THICKNESS[0]
            )
            + cfg.FDTD_min_gridpoints_between_detectors
        ),
        int(cfg.FDTD_reflection_detector_x * scaling),
    )
    use_pulse = True
    if cfg.FDTD_pulse_type == "pulse":
        pass
    elif cfg.FDTD_pulse_type == "continuous":
        use_pulse = False
    else:
        raise ValueError(f' FDTD_pulse_type must either be "pulse" or "continuous"')
    grid = _add_source(
        grid,
        int(cfg.FDTD_source_position[0] * scaling),
        int(cfg.FDTD_source_position[1] * scaling),
        wavelength / SPEED_OF_LIGHT,
        use_pulse,
        cfg.FDTD_source_type,
    )
    if include_object:
        for i in range(cfg.N_layers):
            grid = _add_object(
                grid,
                int(
                    scaling
                    * (
                        cfg.FDTD_pml_thickness
                        + cfg.FDTD_free_space_distance
                        + i * cfg.PER_LAYER_THICKNESS[0]
                    )
                ),
                int(
                    scaling
                    * (
                        cfg.FDTD_pml_thickness
                        + cfg.FDTD_free_space_distance
                        + (i + 1) * cfg.PER_LAYER_THICKNESS[0]
                    )
                ),
                permittivity[0][0][
                    i
                ],  # TODO: Implement possibility for patterned grid, currently uniform layer is used
                frequency=SPEED_OF_LIGHT / wavelength,
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
    # Add PML boundary to the left side of the grid
    grid[0:pml_thickness, :, :] = PML(name="pml_xlow")
    # Add PML boundary to the right side of the grid
    grid[-pml_thickness:, :, :] = PML(name="pml_xhigh")
    # Add periodic boundaries at both sides in y-direction
    grid[:, 0, :] = PeriodicBoundary(name="ybounds")
    # Add periodic boundaries on both sides in z-direction. Only applicable for 3D grids
    # grid[:, :, 0] = fdtd.PeriodicBoundary(name="zbounds")
    return grid


def _add_source(grid, source_x, source_y, period, use_pulse_source, source_type):
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

    if source_type == "point":
        grid[source_x, source_y, 0] = PointSource(
            period=period,
            name="pointsource",
            pulse=use_pulse_source,
            cycle=1,
            hanning_dt=1e-15,
        )
    elif source_type == "line":
        grid[source_x, :, 0] = LineSource(period=period, name="linesource")
    else:
        raise ValueError(f'FDTD_source_type must be either "line" or "point"')
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

    transmission_detector = LineDetector(name="t_detector")
    reflection_detector = LineDetector(name="r_detector")
    grid[transmission_detector_x, :, 0] = transmission_detector
    grid[reflection_detector_x, :, 0] = reflection_detector
    return grid, transmission_detector, reflection_detector


def _add_object(grid, object_start_x, object_end_x, permittivity, frequency):
    """Add a object to the fdtd grid, with a specified permittivity. The object covers the entire grid in the y-direction.

    Args:
        grid (fdtd:Grid): The grid object which to add the object
        object_start_x (float): Coordinate where the object should start in the x-direction
        object_end_x (float): Coordinate where the object should end in the x-direction
        permittivity (float): The dielectric permittivity of the object

    Returns:
        fdtd.Grid: The grid with the added object
    """
    # Not sure whether the conductivity should be relative or absolute, i.e. if it should be multiplied with EPS_0. Multiplied with 2pi to get w(angular frequency)?
    # Since the permittivity is set to 1 for the free space grid, I'll leave it at an relative value for now. Also, permittivity for object is relative.
    grid[object_start_x:object_end_x, :, :] = AbsorbingObject(
        permittivity=permittivity.real,
        conductivity=permittivity.imag * frequency,
    )
    return grid