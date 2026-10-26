"""GNN encoders whose relation transforms are Clifford products (one multivector per relation)."""
from .gatv2 import CliffordGATv2
from .graphsage import CliffordGraphSAGE
from .rgcn import CliffordRGCN

__all__ = [
    "CliffordRGCN",
    "CliffordGATv2",
    "CliffordGraphSAGE",
]
