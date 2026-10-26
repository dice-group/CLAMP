"""Relational GNN encoders with a dense (matrix) relation transform."""
from .base import RelationalGNN
from .compgcn import CompGCN
from .gatv2 import GATv2
from .graphsage import GraphSAGE
from .rgcn import RGCN

__all__ = [
    "RelationalGNN",
    "RGCN",
    "GATv2",
    "GraphSAGE",
    "CompGCN",
]
