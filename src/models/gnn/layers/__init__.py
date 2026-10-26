"""Layers of the dense GNN encoders; RGCN and GATv2 use PyG's RGCNConv and GATv2Conv directly."""
from .activations import SPLIT_ACTIVATIONS
from .compgcn import CompGCNLayer
from .graphsage import RelationalSAGELayer

__all__ = [
    "SPLIT_ACTIVATIONS",
    "CompGCNLayer",
    "RelationalSAGELayer",
]
