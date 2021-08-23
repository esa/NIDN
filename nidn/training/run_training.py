from dotmap import DotMap

import torch
import numpy as np
import numpy.typing as npt
from collections import deque
from copy import deepcopy
from loguru import logger

from .losses.spectrum_loss import _spectrum_loss_fn
from ..materials.material_collection import MaterialCollection
from .model.init_network import init_network
from .model.model_to_eps_grid import model_to_eps_grid
from ..trcwa.compute_spectrum import compute_spectrum
from ..trcwa.compute_target_frequencies import compute_target_frequencies
from ..utils.fix_random_seeds import fix_random_seeds
from .utils.validate_config import _validate_config


def _init_training(run_cfg: DotMap, model):
    """Initializes additional parameters required for training.
    Args:
        run_cfg (DotMap): Run configuration.
        model (torch.model, optional): Model to continue training. If None, a new model will be created according to the run configuration.

    Returns:
        DotMap, torch.model, torch.opt, torch.scheduler: Run config with additional entries, model, optimizer, scheduler
    """

    # Validate config
    _validate_config(run_cfg)

    if run_cfg.type == "classification":
        run_cfg.material_collection = MaterialCollection(run_cfg.target_frequencies)
        run_cfg.N_materials = run_cfg.material_collection.N_materials
        # If classification, the model outputs a likelihood for the presence of each material.
        run_cfg.out_features = run_cfg.N_materials
    else:
        run_cfg.out_features = 2  # For regression, the model outputs epsilon directly.

    # Fix random seed for reproducibility
    fix_random_seeds(run_cfg.seed)

    # Determine target frequencies
    run_cfg.target_frequencies = compute_target_frequencies(
        run_cfg.physical_wavelength_range[0],
        run_cfg.physical_wavelength_range[1],
        run_cfg.N_freq,
    )

    logger.debug("Computed target frequencies:")
    logger.debug(run_cfg.target_frequencies)

    # Init model
    if model is None:
        model = init_network(run_cfg)

    # Initialize some utility
    optimizer = torch.optim.Adam(model.parameters(), lr=run_cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.75, patience=200, min_lr=1e-6, verbose=True
    )

    return run_cfg, model, optimizer, scheduler


def run_training(
    run_cfg: DotMap,
    target_reflectance_spectrum: npt.NDArray,
    target_transmittance_spectrum: npt.NDArray,
    model=None,
):
    """Runs a training run with the passed config, target reflectance and transmittance spectra. Optionally a model can be passed to continue training.

    Args:
        run_cfg (DotMap): Run configuration.
        target_reflectance_spectrum (np.array): Target reflectance spectrum.
        target_transmittance_spectrum (np.array): Target transmittance spectrum.
        model (torch.model, optional): Model to continue training. If None, a new model will be created according to the run configuration. Defaults to None.

    Returns:
        torch.model, DotMap: The best model achieved in the training run, and the loss results of the training run.
    """
    logger.trace("Initializing training...")

    # Initialize training parameters, model and optimizer etc.
    run_cfg, model, optimizer, scheduler = _init_training(run_cfg, model)

    results = DotMap()

    # When a new network is created we init empty training logs
    results.loss_log = []
    results.weighted_average_log = []
    weighted_average = deque([], maxlen=20)

    # And store the best results
    best_loss = np.inf
    best_model_state_dict = model.state_dict()

    logger.trace("Starting training...")
    # Training Loop
    for it in range(run_cfg.iterations):
        torch.cuda.empty_cache()

        # Compute the epsilon values from the model
        eps_grid, material_ids = model_to_eps_grid(model, run_cfg)

        # Compute the spectrum using TRCWA for this grid
        try:
            produced_R_spectrum, produced_T_spectrum = compute_spectrum(
                eps_grid, run_cfg
            )
        except ValueError:
            logger.warning(
                "ValueError encountered in compute_spectrum. This likely means the LR was too high. Reloading best model, and reducing LR."
            )
            model.load_state_dict(best_model_state_dict)
            logger.info(
                "Setting LR to {}".format(optimizer.param_groups[0]["lr"] * 0.5)
            )
            optimizer.param_groups[0]["lr"] *= 0.5
            continue

        # Compute loss between target spectrum and
        # the one from the current network structure
        spectrum_loss, renormalized = _spectrum_loss_fn(
            produced_R_spectrum,
            produced_T_spectrum,
            target_reflectance_spectrum,
            target_transmittance_spectrum,
            run_cfg.target_frequencies,
            run_cfg.L,
            run_cfg.absorption_loss,
        )

        loss = spectrum_loss

        # We store the model if it has the lowest loss yet
        # (this is to avoid losing good results during a run that goes wild)
        if loss < best_loss:
            best_loss = loss
            logger.info(
                f"###  New Best={loss.item():<6.4f} with SpectrumLoss={spectrum_loss.detach().item():<6.4f} ###"
            )
            if not renormalized:
                logger.debug("Saving model state...")
                best_model_state_dict = deepcopy(model.state_dict())

        # Update the loss trend indicators
        weighted_average.append(loss.item())

        # Update the logs
        results.weighted_average_log.append(np.mean(weighted_average))
        results.loss_log.append(loss.item())

        # Print every i iterations
        if it % 5 == 0:
            wa_out = np.mean(weighted_average)
            logger.info(
                f"It={it:<5} Loss={loss.item():<6.4f}   |  weighted_avg={wa_out:<6.4f}  |  SpectrumLoss={spectrum_loss.detach().item():<6.4f}"
            )

        # Zeroes the gradient (otherwise would accumulate)
        optimizer.zero_grad()

        # Compute gradient of the loss with respect to model parameters
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Perform a step in LR scheduler to update LR if necessary
        scheduler.step(loss.item())

    logger.trace("Reloading best model state...")
    # Return best model in the end
    model.load_state_dict(best_model_state_dict)

    return model, results
