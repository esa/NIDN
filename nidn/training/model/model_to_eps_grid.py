import torch
from dotmap import DotMap


def model_to_eps_grid(model, run_cfg: DotMap):
    """Computes a 4D grid of epsilons for the given model.

    Args:
        model (torch.model): Trained neural network model. Should map one 4D input to a [real,imag] epsilon.
        run_cfg (DotMap): Configuration for the run.

    Returns:
        [torch.tensor]: Resulting 4D [real,imag] epsilon grid
    """

    # Get the grid ticks
    x = torch.linspace(-1, 1, run_cfg.Nx)
    y = torch.linspace(-1, 1, run_cfg.Ny)
    z = torch.linspace(-1, 1, run_cfg.N_layers)

    # Linearly spaced frequency points
    #     freq = torch.linspace(-1,1,len(run_cfg.target_frequencies))

    # Logspaced frequency points
    # Normalize to max val = 1
    freq = torch.tensor(run_cfg.target_frequencies / max(run_cfg.target_frequencies))
    # Transform to -1 to 1
    freq = (freq * 2) - 1

    # Create a meshgrid from the grid ticks
    X, Y, Z, FREQ = torch.meshgrid((x, y, z, freq))

    # Reshape the grid to have a Nx4 tensor
    nn_inputs = torch.cat(
        (X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1), FREQ.reshape(-1, 1)),
        dim=1,
    )

    # Compute model output at each grid point
    out = model(nn_inputs)

    # Reshape the output to have a 4D tensor again
    # Note we ouput real and imaginary parts separatly, hence 2*N_freq
    out = out.reshape(run_cfg.Nx, run_cfg.Ny, run_cfg.N_layers, 2 * run_cfg.N_freq)

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
    eps.real = (eps.real * (run_cfg.real_max_eps - run_cfg.real_min_eps)).clip(
        run_cfg.real_min_eps, run_cfg.real_max_eps
    )
    # second half imaginary
    eps.imag = out[:, :, :, run_cfg.N_freq :]
    eps.imag = (eps.imag * (run_cfg.imag_max_eps - run_cfg.imag_min_eps)).clip(
        run_cfg.imag_min_eps, run_cfg.imag_max_eps
    )

    return eps
