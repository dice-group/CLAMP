"""Coefficient-wise activations selected by ``--gnn_activation``."""

import torch
from torch import nn

#: Activations applied to each coefficient independently.
SPLIT_ACTIVATIONS = {
    "relu": torch.relu,
    "leaky_relu": nn.functional.leaky_relu,
    "elu": nn.functional.elu,
    "none": None,
}
