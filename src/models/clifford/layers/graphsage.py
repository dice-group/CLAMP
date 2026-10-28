"""Relational GraphSAGE (Hamilton et al., 2017, https://arxiv.org/abs/1706.02216) with one
learned multivector per relation in place of SAGEConv's neighbour matrix.

Each embedding is a full Cl(n, 0) multivector with 2**n coefficients; * is the Clifford product.
"""

import torch
from torch import nn

from src.models.clifford.layers.relational import CliffordLinear, relational_aggregate


class CliffordRelationalSAGELayer(nn.Module):
    """Relational SAGE layer with one learned multivector per relation.

    With mean or sum, the linear product commutes with aggregation. With max, this layer
    pools transformed messages; PyG's SAGEConv applies its neighbour matrix after pooling,
    so the ordering differs.
    """
    def __init__(self, algebra, num_relations: int, aggregator: str = "mean"):
        super().__init__()
        if aggregator not in ("mean", "sum", "max"):
            raise ValueError(f"Unknown aggregator '{aggregator}'")
        self.algebra = algebra
        self.aggregator = aggregator
        self.rel_weight = nn.Parameter(algebra.init_weight(num_relations))
        self.root_lin = CliffordLinear(algebra)

    def forward(
        self, x: torch.Tensor, edge_index: torch.LongTensor, edge_type: torch.LongTensor
    ) -> torch.Tensor:
        """Compute x'_i = x_i * w_root + b + sum_r AGG_{j in N_r(i)}(x_j * w_r).

        The encoder applies activation and coefficient L2 normalization afterward.
        """
        matrices = self.algebra.clifford_multiplication_matrix(self.rel_weight)
        messages = relational_aggregate(x, edge_index, edge_type, matrices, self.aggregator)
        return self.root_lin(x) + messages
