import torch


def _likelihood_regularization_loss_fn(material_predictions, L=0.5, use_max=False):
    """Computes a likelihood regularization loss for the distance between predictions
    and one / zero respectively.

    Args:
        material_predictions (torch.tensor): Predictions of the material for each grid point.
        L (float, optional): L (float, optional): Exponent of the loss, e.g. L=2 for squared loss values. Defaults to 0.5.
        use_max (bool, optional): If True, the max of the material predictions is used. Defaults to False.

    Returns:
        torch.tensor: Likelihood regularization loss
    """
    if use_max:
        # Take maximum value per grid point as distance
        max_per_point, _ = torch.max(material_predictions, dim=-1)
        distances = 1.0 - max_per_point
    else:
        # Average of minimum of distance from one and zero
        distance_to_one = torch.pow(1 - material_predictions, L)
        distance_to_zero = torch.pow(material_predictions, L)
        distances = torch.minimum(distance_to_one, distance_to_zero)

    return torch.mean(distances)
