"""R-GCN (Schlichtkrull et al., 2018, https://arxiv.org/abs/1703.06103) with learned
multivectors in place of RGCNConv's per-relation matrices.

Each embedding holds C = dim / 2**(p+q) full Cl(p, q) multivectors; * is the Clifford product.
"""

import torch
from torch import nn

from src.models.clifford.clifford_algebra import CliffordAlgebra
from src.models.clifford.layers.activations import clifford_activation
from src.models.clifford.layers.rgcn import CliffordRGCNConv
from src.models.gnn.base import RelationalGNN, residual_mode


class CliffordRGCN(RelationalGNN):
    """Stacked CliffordRGCNConv layers with ``--gnn_activation`` and dropout between layers.

    ``--gnn_activation`` takes the coefficient-wise activations (relu, leaky_relu, elu, none)
    and the multivector gates mv_gate and mv_grade_gate, which scale a whole multivector or
    a whole grade by a sigmoid of an invariant. ``--gnn_residual`` adds each layer's input to
    its output, inside (``post``) or around (``pre``) the activation. ``--clifford_num_bases`` selects the basis decomposition; unset, each relation
    has its own multivector. The dense ``--num_bases`` flag is not read, so runs made before
    the basis option keep their meaning. ``--cl_channel_mix`` lets the channels interact,
    per relation or through one shared grid per layer (see CliffordRGCNConv).
    """
    name = "CliffordRGCN"

    def __init__(self, args: dict):
        super().__init__(args)
        self.clifford_algebra = CliffordAlgebra.for_embedding_dim(
            self.embedding_dim, args.get("clifford_signature")
        )
        self.gnn_residual = residual_mode(args)
        num_bases = args.get("clifford_num_bases")
        channel_mix = args.get("cl_channel_mix", "none")
        self.gnn_layers = nn.ModuleList(
            [
                CliffordRGCNConv(self.clifford_algebra, self.num_relations, num_bases, channel_mix)
                for _ in range(self.num_gnn_layers)
            ]
        )
        # One activation module per hidden layer: the gates carry their own parameters.
        activation = args.get("gnn_activation", "relu")
        self.activations = nn.ModuleList(
            [
                clifford_activation(self.clifford_algebra, activation) or nn.Identity()
                for _ in range(max(self.num_gnn_layers - 1, 0))
            ]
        )

    def encode(self, edge_index, edge_type) -> torch.FloatTensor:
        x = self.node_emb.weight
        for i, conv in enumerate(self.gnn_layers):
            out = conv(x, edge_index, edge_type)
            if self.gnn_residual == "post":
                out = out + x
            if i < self.num_gnn_layers - 1:
                out = self.gnn_dropout(self.activations[i](out))
            if self.gnn_residual == "pre":
                out = out + x
            x = out
        return x
