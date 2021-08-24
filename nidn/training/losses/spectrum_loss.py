from loguru import logger
import torch


def _spectrum_loss_fn(
    produced_R_spectrum,
    produced_T_spectrum,
    target_R_spectrum,
    target_T_spectrum,
    target_frequencies,
    L=2.0,
    include_absorption=False,
):
    """Computes the loss for the difference in reflectance,
    transmittance and optionally absorption spectra.

    Args:
        produced_R_spectrum (torch.tensor): Produced reflectance spectrum.
        produced_T_spectrum (torch.tensor): Produced transmittance spectrum.
        target_R_spectrum (torch.tensor): Target reflectance spectrum.
        target_T_spectrum (torch.tensor): Target transmittance spectrum.
        target_frequencies (list of float): Target frequencies.
        L (float, optional): Exponent of the loss, e.g. L=2 for squared loss values. Defaults to 0.5.
        include_absorption (bool, optional): If absorption should be included in loss term. Defaults to False.

    Returns:
        [torch.tensor], renormalized: Loss values for reflectance, transmittance and optionally absorption and if renormalization happened.
    """
    logger.trace("Computing spectrum loss..")

    R_loss, T_loss, A_loss = 0, 0, 0
    # Tracks if renormalization happened
    renormalized = False

    # Iterate over all frequencies
    for prod_R, prod_T, target_R, target_T, freq in zip(
        produced_R_spectrum,
        produced_T_spectrum,
        target_R_spectrum,
        target_T_spectrum,
        target_frequencies,
    ):

        # If the unphysical scenario of negative absorption, i.e.
        # reflection + transmission > 1, then renormalize them to 1 and warn
        if prod_R + prod_T > 1.0:
            logger.warning(
                f"R+T>1 for freq={freq:.4f}. Renormalizing this point in the loss..."
            )
            prod_R = prod_R / (prod_R + prod_T)
            prod_T = prod_T / (prod_R + prod_T)
            renormalized = True

        # Accumulate losses
        R_loss += torch.abs(prod_R - target_R) ** L
        T_loss += torch.abs(prod_T - target_T) ** L

        if include_absorption:
            prod_A = 1 - prod_T - prod_R
            target_A = 1 - target_T - target_R
            A_loss += torch.abs(prod_A - target_A) ** L

    if include_absorption:
        return (A_loss + R_loss + T_loss) / (3 * len(produced_R_spectrum)), renormalized
    else:
        return (R_loss + T_loss) / (2 * len(produced_R_spectrum)), renormalized
