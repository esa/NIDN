from dotmap import DotMap
import fdtd
from dotmap import DotMap

def init_fdtd(cfg: DotMap, include_object, period):
    grid = fdtd.Grid(
        (cfg.grid_size_x, cfg.grid_size_y, 1),
        grid_spacing=cfg.grid_spacing,
        permittivity=1.0,
        permeability=1.0,
    )
    grid = _add_boundaries(grid, cfg.pml_thickness)
    grid, t_detector, r_detector = _add_detectors(
        grid, cfg.transmission_detector_x, cfg.reflection_detector_x
    )
    grid = _add_source(
        grid, cfg.source_x, cfg.source_y, period, cfg.use_pulse_source, cfg.use_point_source
    )
    # TODO: Support multiple layers
    if include_object:
        grid = _add_object(grid, cfg.object_start_x, cfg.object_end_x, cfg.permittivity)
    return grid, t_detector, r_detector


def _add_boundaries(grid, pml_thickness):
    # Add theboundaries to the FDTD grid

    grid[0:pml_thickness, :, :] = fdtd.PML(name="pml_xlow")
    grid[-pml_thickness:, :, :] = fdtd.PML(name="pml_xhigh")
    grid[:, 0, :] = fdtd.PeriodicBoundary(name="ybounds")
    return grid


def _add_source(grid, source_x, source_y, period, use_pulse_source, use_point_source):
    # Add the source to the FDTD grid

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
    # Add detectors to the FDTD grid

    transmission_detector = fdtd.LineDetector(name="t_detector")
    reflection_detector = fdtd.LineDetector(name="r_detector")
    grid[transmission_detector_x, :, 0] = transmission_detector
    grid[reflection_detector_x, :, 0] = reflection_detector
    return grid, transmission_detector, reflection_detector


def _add_object(grid, object_start_x, object_end_x, permittivity):
    # Add an object to the FDTD grid

    grid[object_start_x:object_end_x, :, :] = fdtd.AnisotropicObject(
        permittivity=permittivity, name="object"
    )
    return grid
