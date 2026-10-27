"""Relational GraphSAGE (Hamilton et al., 2017, https://arxiv.org/abs/1706.02216) with one
learned multivector per relation in place of SAGEConv's neighbour matrix.

Each embedding is a full Cl(n, 0) multivector with 2**n coefficients; * is the Clifford product.
"""

import torch
import torch.nn.functional as F
from torch import nn

from src.models.clifford.clifford_algebra import CliffordAlgebra
from src.models.clifford.layers.graphsage import CliffordRelationalSAGELayer
from src.models.gnn.base import RelationalGNN


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
