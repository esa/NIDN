from dotmap import DotMap


def compute_spectrum(grid, cfg: DotMap):
    # For each wavelength:
    # Run simulation
    # Get detector values
    # Postprocess signal,  i.e. phase shift and more if applicable
    # Calculate transmission, reflection (and absorption coefficients)
    # Return transmission spectrum, reflection spectrum (and absorption spectrum)
    return t_spectrum, r_spectrum


def _get_detector_values(
    grid, transmission_detector, reflection_detector, detector_value_y
):
    # Get values from the transmission detector and the reflection detector.
    # This will be six lists for each detector, each list having either magnetic or electric field in either x,y or z direction
    ex_transmission = transmission_detector.detector_values["E"][int(detector_value_y)][
        0
    ]
    ey_transmission = transmission_detector.detector_values["E"][int(detector_value_y)][
        1
    ]
    ez_transmission = transmission_detector.detector_values["E"][int(detector_value_y)][
        2
    ]
    hx_transmission = transmission_detector.detector_values["H"][int(detector_value_y)][
        0
    ]
    hy_transmission = transmission_detector.detector_values["H"][int(detector_value_y)][
        1
    ]
    hz_transmission = transmission_detector.detector_values["H"][int(detector_value_y)][
        2
    ]

    ex_reflection = reflection_detector.detector_values["E"][int(detector_value_y)][0]
    ey_reflection = reflection_detector.detector_values["E"][int(detector_value_y)][1]
    ez_reflection = reflection_detector.detector_values["E"][int(detector_value_y)][2]
    hx_reflection = reflection_detector.detector_values["H"][int(detector_value_y)][0]
    hy_reflection = reflection_detector.detector_values["H"][int(detector_value_y)][1]
    hz_reflection = reflection_detector.detector_values["H"][int(detector_value_y)][2]
