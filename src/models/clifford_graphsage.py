"""Relational GraphSAGE (Hamilton et al., 2017, https://arxiv.org/abs/1706.02216) with one
learned multivector per relation in place of SAGEConv's neighbour matrix.

Each embedding is a full Cl(n, 0) multivector with 2**n coefficients; * is the Clifford product.
"""

import torch
import torch.nn.functional as F
from torch import nn

from src.clifford_algebra import CliffordAlgebra
from src.models.clifford_layers import CliffordLinear, relational_aggregate
from src.models.gnn import RelationalGNN


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
        messages = relational_aggregate(
            self.algebra, x, edge_index, edge_type, self.rel_weight, self.aggregator
        )
        return self.root_lin(x) + messages


class CliffordGraphSAGE(RelationalGNN):
    """Stacked relation-wise Clifford aggregators; every layer output is L2-normalized.

    Hidden layers: ReLU (``--gnn_activation``), x_i <- x_i / max(||x_i||_2, eps), dropout.
    The norm is over coefficients, not the Clifford scalar product.
    """
    name = "CliffordGraphSAGE"

    def __init__(self, args: dict):
        super().__init__(args)
        self.activation = args.get("gnn_activation", "relu")
        self.clifford_algebra = CliffordAlgebra.for_embedding_dim(
            self.embedding_dim, args.get("clifford_signature")
        )
        self.gnn_layers = nn.ModuleList(
            [
                CliffordRelationalSAGELayer(
                    self.clifford_algebra, self.num_relations, "mean"
                )
                for _ in range(self.num_gnn_layers)
            ]
        )

    def encode(self, edge_index, edge_type) -> torch.FloatTensor:
        x = self.node_emb.weight
        for i, layer in enumerate(self.gnn_layers):
            x = layer(x, edge_index, edge_type)
            hidden = i < self.num_gnn_layers - 1
            if hidden and self.activation == "relu":
                x = F.relu(x)
            x = F.normalize(x, p=2, dim=1)
            if hidden:
                x = self.gnn_dropout(x)
        return x
