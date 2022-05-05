import torch
from dotmap import DotMap


def _get_transformed_grid(Nx_undersampled, Ny_undersampled, N_layers, thicknesses):
    """Creates x,y,z in the space [-1,1]**3 . Z weighted by thicknesses

    Args:
          model (torch.model): Trained neural network model. Should map one 4D input.
        Nx_undersampled (int): Number of grid points in x direction. Potentially unesampled if eps_oversampling > 1.
        Ny_undersampled (int): Number of grid points in y direction. Potentially unesampled if eps_oversampling > 1.
        N_layers (int): Number of layers in the model.
        target_frequencies (list): Target frequencies.
        freq_distribution (list): Distribution of points in frequency space. Should be "linear" or "log".
        thicknesses (list): List of thicknesses for each layer or scalar for all.
    Returns:
       [torch.tensor,torch.tensor,torch.tensor]: x,y,z

    """
    # Get the grid ticks
    x = torch.linspace(-1, 1, Nx_undersampled)
    y = torch.linspace(-1, 1, Ny_undersampled)

    # Space z uniform for equally thick layers, else weight by thickness
    if len(thicknesses) == 1:
        z = torch.linspace(-1, 1, N_layers)
    else:
        total_thickness = sum(thicknesses)

        # create weighted z
        z = torch.cumsum(torch.tensor(thicknesses) / total_thickness, dim=0)
        z -= thicknesses[0] / total_thickness

        z = z / z.max() - 0.5
        z *= 2

    return x, y, z


def _avoid_zero_eps(eps, cutoff=1e-2):
    """Clips epsilon values close to zero to either cutoff or -cutoff.

    Args:
        eps (torch.tensor): epsilon to be clipped.
        cutoff (float, optional): Cutoff border. Defaults to 1e-2.

    Returns:
        torch.tensor: modified eps tensor
    """
    # catch zeroes
    indices = eps.real == 0.0
    eps.real[indices] = cutoff
    # catch close to zeroes
    indices = torch.logical_and(eps.real < cutoff, eps.real > -cutoff)
    eps.real[indices] = cutoff * torch.sign(eps.real[indices])
    return eps


def _eval_model(
    model,
    Nx_undersampled,
    Ny_undersampled,
    N_layers,
    target_frequencies,
    freq_distribution,
    thicknesses,
):
    """Evaluates the model on the grid.

    Args:
        model (torch.model): Trained neural network model. Should map one 4D input.
        Nx_undersampled (int): Number of grid points in x direction. Potentially unesampled if eps_oversampling > 1.
        Ny_undersampled (int): Number of grid points in y direction. Potentially unesampled if eps_oversampling > 1.
        N_layers (int): Number of layers in the model.
        target_frequencies (list): Target frequencies.
        freq_distribution (list): Distribution of points in frequency space. Should be "linear" or "log".
        thicknesses (list): List of thicknesses for each layer or scalar for all.
    Returns:
       [torch.tensor]: Resulting 4D [real,imag] epsilon grid
    """
    x, y, z = _get_transformed_grid(
        Nx_undersampled, Ny_undersampled, N_layers, thicknesses
    )

    # Scales sampling domain of frequencies
    freq_scaling = 32.0

    # Layers should be independent, so we scale the sampling in that dimension
    z_scaling = 1000.0
    z *= z_scaling

    if freq_distribution == "linear":
        # Linearly spaced frequency points
        freq = torch.linspace(-freq_scaling, freq_scaling, len(target_frequencies))
    elif freq_distribution == "log":
        # Logspaced frequency points as in cfg
        # Normalize to max val = 1
        freq = torch.tensor(target_frequencies / max(target_frequencies))
        # Transform to -scaling to scaling
        freq = (freq - 0.5) * 2 * freq_scaling
    else:
        raise ValueError("Unknown frequency distribution. Should be 'log' or 'linear'")

    # Create a meshgrid from the grid ticks
    X, Y, Z, FREQ = torch.meshgrid((x, y, z, freq))

    # Reshape the grid to have a Nx4 tensor
    nn_inputs = torch.cat(
        (X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1), FREQ.reshape(-1, 1)),
        dim=1,
    )

    # Compute model output at each grid point
    return model(nn_inputs)


def _regression_model_to_eps_grid(model, run_cfg: DotMap):
    """Computes a 4D grid of epsilons for a regression model.

    Args:
        model (torch.model): Trained neural network model. Should map one 4D input to a [real,imag] epsilon.
        run_cfg (DotMap): Configuration for the run.

    Returns:
        [torch.tensor]: Resulting 4D [real,imag] epsilon grid
    """

    Nx_undersampled = run_cfg.Nx // run_cfg.eps_oversampling
    Ny_undersampled = run_cfg.Ny // run_cfg.eps_oversampling

    # Evaluate the model at the grid points
    out = _eval_model(
        model,
        Nx_undersampled,
        Ny_undersampled,
        run_cfg.N_layers,
        run_cfg.target_frequencies,
        run_cfg.freq_distribution,
        run_cfg.PER_LAYER_THICKNESS,
    )

    # Reshape the output to have a 4D tensor again
    # Note we ouput real and imaginary parts separatly, hence 2*N_freq
    out = out.reshape(
        Nx_undersampled, Ny_undersampled, run_cfg.N_layers, 2 * run_cfg.N_freq
    )

    # Oversample the created eps grid
    out = torch.repeat_interleave(out, run_cfg.eps_oversampling, dim=0)
    out = torch.repeat_interleave(out, run_cfg.eps_oversampling, dim=1)

    # Initialize the epsilon grid
    eps = torch.zeros(
        [run_cfg.Nx, run_cfg.Ny, run_cfg.N_layers, run_cfg.N_freq],
        dtype=torch.cfloat,
    )

    # Net out is [0,1] thus we transform to desired real and imaginary ranges
    # first half contains real entries
    eps.real = out[:, :, :, 0 : run_cfg.N_freq]

    # scale up to [0 , (max-min_eps)]
    eps.real = eps.real * (run_cfg.real_max_eps - run_cfg.real_min_eps)

    # translate to [min_eps, max_eps] and clip outliers
    eps.real = (eps.real + run_cfg.real_min_eps).clip(
        run_cfg.real_min_eps, run_cfg.real_max_eps
    )

    if run_cfg.avoid_zero_eps:
        _avoid_zero_eps(eps)

    # second half imaginary
    eps.imag = out[:, :, :, run_cfg.N_freq :]
    eps.imag = (eps.imag * (run_cfg.imag_max_eps - run_cfg.imag_min_eps)).clip(
        run_cfg.imag_min_eps, run_cfg.imag_max_eps
    )

    # Adds noise scaled to desired eps range times noise_scale
    if run_cfg.add_noise:
        eps.real += (
            (torch.randn_like(eps.real) - 0.5)
            * 2.0
            * run_cfg.noise_scale
            * (run_cfg.real_max_eps - run_cfg.real_min_eps)
        )
        eps.real = eps.real.clip(run_cfg.real_min_eps, run_cfg.real_max_eps)
        eps.imag += (
            (torch.randn_like(eps.imag) - 0.5)
            * 2.0
            * run_cfg.noise_scale
            * (run_cfg.imag_max_eps - run_cfg.imag_min_eps)
        )
        eps.imag = eps.imag.clip(run_cfg.imag_min_eps, run_cfg.imag_max_eps)

    return eps


def _classification_model_to_eps_grid(model, run_cfg: DotMap):
    """Computes a 4D grid of epsilons for a classification model.

    Args:
        model (torch.model): Trained neural network model. Should map one 4D input to a material index.
        run_cfg (DotMap): Configuration for the run.

    Returns:
        [torch.tensor]: Resulting 4D [real,imag] epsilon grid
    """

    Nx_undersampled = run_cfg.Nx // run_cfg.eps_oversampling
    Ny_undersampled = run_cfg.Ny // run_cfg.eps_oversampling

    # Evaluate the model at the grid points
    out = _eval_model(
        model,
        Nx_undersampled,
        Ny_undersampled,
        run_cfg.N_layers,
        run_cfg.target_frequencies,
        run_cfg.freq_distribution,
        run_cfg.PER_LAYER_THICKNESS,
    )

    # Reshape the output to have a 4D tensor again
    # Note we output a likelihood for each material
    out = out.reshape(
        Nx_undersampled,
        Ny_undersampled,
        run_cfg.N_layers,
        run_cfg.N_freq,
        run_cfg.N_materials,
    )

    # Oversample the created grid
    out = torch.repeat_interleave(out, run_cfg.eps_oversampling, dim=0)
    out = torch.repeat_interleave(out, run_cfg.eps_oversampling, dim=1)

    # We take mean over the frequency dimension in determining the fitting
    out = torch.mean(out, dim=3)

    # Adds noise according to specified noise_scale
    if run_cfg.add_noise:
        out += torch.randn_like(out) * run_cfg.noise_scale

    beta = 16
    # Softmax with a high beta to push towards 1
    exponential = torch.exp(beta * out)
    material_id = exponential / exponential.sum(dim=-1).unsqueeze(-1)

    material_id = torch.divide(material_id, material_id.sum(-1).unsqueeze(-1))

    # Convert material probabilities to epsilons
    eps = (material_id.unsqueeze(-1) * run_cfg.material_collection.epsilon_matrix).sum(
        -2
    )

    if run_cfg.avoid_zero_eps:
        _avoid_zero_eps(eps)

    return eps, material_id


def model_to_eps_grid(model, run_cfg: DotMap):
    """Computes a 4D grid of epsilons for the given model.

    Args:
        model (torch.model): Trained neural network model. Should map one 4D input to a [real,imag] epsilon.
        run_cfg (DotMap): Configuration for the run.

    Returns:
        torch.tensor, torch.tensor: Resulting 4D [real,imag] epsilon grid and tensor of the material_ids (None for regression)
    """
    if run_cfg.type == "classification":
        return _classification_model_to_eps_grid(model, run_cfg)
    elif run_cfg.type == "regression":
        # For consistent returns we return a tuple with None
        return _regression_model_to_eps_grid(model, run_cfg), None
