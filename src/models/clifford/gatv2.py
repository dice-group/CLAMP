"""GATv2-inspired attention (Brody et al., 2022, https://arxiv.org/abs/2105.14491) with
multivector query, key, value and relation maps.

Each embedding is a full Cl(n, 0) multivector with 2**n coefficients; * is the Clifford product
and <z>_0 its scalar part.
"""

import torch
import torch.nn.functional as F
from torch import nn

from src.models.clifford.clifford_algebra import CliffordAlgebra
from src.models.clifford.layers.gatv2 import CliffordGATv2Conv
from src.models.gnn.base import RelationalGNN


class CliffordGATv2(RelationalGNN):
    """Stacked full-width Clifford attention layers with ELU and dropout between layers.

    ``--gnn_activation none`` removes the ELU.
    """
    name = "CliffordGATv2"

    def __init__(self, args: dict):
        super().__init__(args)
        self.activation = args.get("gnn_activation", "relu")
        self.clifford_algebra = CliffordAlgebra.for_embedding_dim(
            self.embedding_dim, args.get("clifford_signature")
        )
        heads = int(args.get("gnn_heads", 4))
        self.gnn_layers = nn.ModuleList(
            [
                CliffordGATv2Conv(
                    self.clifford_algebra,
                    self.num_relations,
                    heads=heads,
                    dropout=self.gnn_dropout_rate,
                )
                for _ in range(self.num_gnn_layers)
            ]
        )

    def encode(self, edge_index, edge_type) -> torch.FloatTensor:
        x = self.node_emb.weight
        for i, conv in enumerate(self.gnn_layers):
            x = conv(x, edge_index, edge_type)
            if i < self.num_gnn_layers - 1:
                if self.activation == "relu":
                    x = F.elu(x)
                x = self.gnn_dropout(x)
        return x
