from torch import nn
import torch


class AbsLayer(nn.Module):
    """Very simple activation layer to allow different abs layer activations of the siren
    """

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.abs(input)
