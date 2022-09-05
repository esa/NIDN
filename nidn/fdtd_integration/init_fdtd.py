from dotmap import DotMap
from loguru import logger
import torch

from ..fdtd import (
    AbsorbingObject,
    set_backend,
    Grid,
    PML,
    PeriodicBoundary,
    LineSource,
    LineDetector,
    PointSource,
)
from ..utils.global_constants import EPS_0, PI, SPEED_OF_LIGHT, UNIT_MAGNITUDE


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

    x_grid_size = (
        cfg.FDTD_grid_scaling
        * (
            cfg.FDTD_pml_thickness * 2
            + cfg.FDTD_free_space_distance * 2
            + torch.sum(torch.tensor(cfg.PER_LAYER_THICKNESS))
        )
    ).int()

    y_grid_size = 3
    logger.debug(
        "Initializing FDTD grid with size {} by {} grid points, with a scaling factor of {} grid points per um".format(
            x_grid_size, y_grid_size, cfg.FDTD_grid_scaling
        )
    )
    grid = Grid(
        (x_grid_size.item(), y_grid_size, 1),
        grid_spacing=UNIT_MAGNITUDE / cfg.FDTD_grid_scaling,
        permittivity=1.0,
        permeability=1.0,
    )
    grid = _add_boundaries(grid, int(cfg.FDTD_pml_thickness * cfg.FDTD_grid_scaling))

    # Determine detector positions
    detector_x = (
        cfg.FDTD_pml_thickness
        + cfg.FDTD_free_space_distance
        + torch.sum(torch.tensor(cfg.PER_LAYER_THICKNESS))
    )

    detector_y = torch.tensor(cfg.FDTD_pml_thickness + cfg.FDTD_free_space_distance)
    detector_x *= cfg.FDTD_grid_scaling
    detector_y *= cfg.FDTD_grid_scaling
    detector_x += cfg.FDTD_gridpoints_from_material_to_detector
    detector_y -= cfg.FDTD_gridpoints_from_material_to_detector

    logger.trace(
        "Placing detectors at "
        + str(detector_x.int().item())
        + ","
        + str(detector_y.int().item())
    )

    grid, t_detector, r_detector = _add_detectors(
        grid,
        detector_x.int().item(),
        detector_y.int().item(),
    )

    source_x = (cfg.FDTD_source_position[0] * cfg.FDTD_grid_scaling).int()
    source_y = (cfg.FDTD_source_position[1] * cfg.FDTD_grid_scaling).int()

    logger.trace(
        "Placing source at "
        + str(source_x.int().item())
        + ","
        + str(source_y.int().item())
    )

    grid = _add_source(
        grid,
        source_x.item(),
        source_y.item(),
        wavelength / SPEED_OF_LIGHT,
        cfg.FDTD_pulse_type,
        cfg.FDTD_source_type,
    )

    if include_object:
        x_start = torch.tensor(cfg.FDTD_pml_thickness + cfg.FDTD_free_space_distance)
        x_end = x_start + cfg.PER_LAYER_THICKNESS[0]
        for i in range(permittivity.shape[2]):
            # TODO: Implement possibility for patterned grid, currently uniform layer is used
            grid = _add_object(
                grid,
                (cfg.FDTD_grid_scaling * x_start).int(),
                (cfg.FDTD_grid_scaling * x_end).int(),
                permittivity[0][0][i],
                frequency=SPEED_OF_LIGHT / wavelength,
            )

            if i < permittivity.shape[2] - 1:
                x_start += cfg.PER_LAYER_THICKNESS[i]
                x_end += cfg.PER_LAYER_THICKNESS[i + 1]
            else:
                break

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


def _add_source(grid, source_x, source_y, period, signal_type, source_type):
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
    assert signal_type in ["continuous", "hanning", "ricker"]
    if source_type == "point":
        grid[source_x, source_y, 0] = PointSource(
            period=period,
            name="pointsource",
            signal_type=signal_type,
            cycle=5,
            hanning_dt=1e-15,
        )
    elif source_type == "line":
        grid[source_x, :, 0] = LineSource(
            period=period,
            name="linesource",
            signal_type=signal_type,
            cycle=5,
            hanning_dt=1e-15,
        )
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
        conductivity=permittivity.imag * frequency * 2 * PI * EPS_0,
    )
    return grid
