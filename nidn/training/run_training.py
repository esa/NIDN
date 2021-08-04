from dotmap import DotMap

import torch
import numpy as np
import numpy.typing as npt
from collections import deque
from copy import deepcopy
from loguru import logger

from ..utils.fix_random_seeds import fix_random_seeds
from .model.init_network import init_network
from ..trcwa.compute_spectrum import compute_spectrum
from .model.model_to_eps_grid import model_to_eps_grid
from .losses.spectrum_loss import _spectrum_loss_fn


def run_training(
    run_cfg: DotMap,
    target_reflectance_spectrum: npt.NDArray,
    target_transmittance_spectrum: npt.NDArray,
    model=None,
):
    """Runs a training run with the passed config, target reflectance and transmittance spectra. Optionally a model can be passed to continue training

    Args:
        run_cfg (DotMap): Run configuration.
        target_reflectance_spectrum (np.array): Target reflectance spectrum.
        target_transmittance_spectrum (np.array): Target transmittance spectrum.
        model (torch.model, optional): Model to continue training. If None, a new model will be created according to the run configuration. Defaults to None.

    Returns:
        torch.model,DotMap: The best model achieved in the training run, and the loss results of the training run.
    """
    # Fix random seed for reproducibility
    fix_random_seeds(run_cfg.seed)

    # Init model
    if model is None:
        model = init_network(run_cfg)

    # Initialize some utility
    optimizer = torch.optim.Adam(model.parameters(), lr=run_cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.75, patience=200, min_lr=1e-6, verbose=True
    )

    results = DotMap()

    # When a new network is created we init empty training logs
    results.loss_log = []
    results.weighted_average_log = []
    weighted_average = deque([], maxlen=20)

    # And store the best results
    best_loss = np.inf
    best_model_state_dict = model.state_dict()

    # Training Loop
    for it in range(run_cfg.iterations):
        torch.cuda.empty_cache()

        # Compute the epsilon values from the model
        eps_grid = model_to_eps_grid(model, run_cfg)

        # Compute the spectrum using TRCWA for this grid
        produced_R_spectrum, produced_T_spectrum = compute_spectrum(eps_grid, run_cfg)

        # Compute loss between target spectrum and
        # the one from the current network structure
        spectrum_loss = _spectrum_loss_fn(
            produced_R_spectrum,
            produced_T_spectrum,
            target_reflectance_spectrum,
            target_transmittance_spectrum,
            run_cfg.L,
            run_cfg.absorption_loss,
        )

        loss = spectrum_loss

        # We store the model if it has the lowest loss yet
        # (this is to avoid losing good results during a run that goes wild)
        if loss < best_loss:
            best_model_state_dict = deepcopy(model.state_dict())
            best_loss = loss
            logger.info(
                f"New Best={loss.item():.4f} SpectrumLoss={spectrum_loss.detach().item():4f}"
            )

        # Update the loss trend indicators
        weighted_average.append(loss.item())

        # Update the logs
        results.weighted_average_log.append(np.mean(weighted_average))
        results.loss_log.append(loss.item())

        # Print every i iterations
        if it % 5 == 0:
            wa_out = np.mean(weighted_average)
            logger.info(
                f"It={it}\t loss={loss.item():.3e}\t  weighted_average={wa_out:.3e}\t SpectrumLoss={spectrum_loss.detach().item():4f}"
            )

        # Zeroes the gradient (otherwise would accumulate)
        optimizer.zero_grad()

        # Compute gradient of the loss with respect to model parameters
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Perform a step in LR scheduler to update LR if necessary
        scheduler.step(loss.item())

        total_it = total_it + 1

    # Return best model in the end
    model.load_state_dict(best_model_state_dict)

    return model, results
