"""R-GCN (Schlichtkrull et al., 2018, https://arxiv.org/abs/1703.06103) with learned
multivectors in place of RGCNConv's per-relation matrices.

Each embedding holds C = dim / 2**(p+q) full Cl(p, q) multivectors; * is the Clifford product.
"""

import torch
from torch import nn
from torch_geometric.nn.inits import glorot

from src.clifford_algebra import CliffordAlgebra
from src.models.clifford_layers import (
    CliffordLinear,
    clifford_activation,
    relational_aggregate,
    residual_mode,
)
from src.models.gnn import RelationalGNN


class CliffordRGCNConv(nn.Module):
    """R-GCN adaptation with multivector relation weights.

    Geometric products replace RGCNConv's matrices; incoming messages are averaged
    separately for each relation, as in RGCNConv(aggr='mean').

    Without num_bases every relation has its own multivector w_r. With num_bases B,
    w_r = sum_b comp[r, b] * basis_b over B shared multivectors, RGCNConv's basis
    decomposition with multivectors in place of matrices. As in RGCNConv, comp is
    glorot-initialized; the bases use the algebra's init.
    """
    def __init__(self, algebra, num_relations: int, num_bases: int = None):
        super().__init__()
        if num_bases is not None and num_bases < 1:
            raise ValueError(f"num_bases must be positive, got {num_bases}")
        self.algebra = algebra
        self.num_relations = num_relations
        self.num_bases = num_bases
        if num_bases is None:
            self.rel_weight = nn.Parameter(algebra.init_weight(num_relations))
        else:
            self.bases = nn.Parameter(algebra.init_weight(num_bases))
            self.comp = nn.Parameter(torch.empty(num_relations, num_bases))
            glorot(self.comp)
        self.root = CliffordLinear(algebra)

    def relation_weights(self) -> torch.Tensor:
        """Return the (num_relations, dim) multivectors w_r."""
        if self.num_bases is None:
            return self.rel_weight
        return self.comp @ self.bases

    def forward(
        self, x: torch.Tensor, edge_index: torch.LongTensor, edge_type: torch.LongTensor
    ) -> torch.Tensor:
        """Compute x'_i = x_i * w_root + b + sum_r mean_{j in N_r(i)} x_j * w_r."""
        messages = relational_aggregate(
            self.algebra, x, edge_index, edge_type, self.relation_weights(), "mean"
        )
        return messages + self.root(x)

    def extra_repr(self) -> str:
        return f"num_relations={self.num_relations}, num_bases={self.num_bases}"


class CliffordRGCN(RelationalGNN):
    """Stacked CliffordRGCNConv layers with ``--gnn_activation`` and dropout between layers.

    ``--gnn_activation`` takes the coefficient-wise activations (relu, leaky_relu, elu, none)
    and the multivector gates mv_gate and mv_grade_gate, which scale a whole multivector or
    a whole grade by a sigmoid of an invariant. ``--gnn_residual`` adds each layer's input to
    its output, inside (``post``) or around (``pre``) the activation. ``--clifford_num_bases`` selects the basis decomposition; unset, each relation
    has its own multivector. The dense ``--num_bases`` flag is not read, so runs made before
    the basis option keep their meaning.
    """
    name = "CliffordRGCN"

    def __init__(self, args: dict):
        super().__init__(args)
        self.clifford_algebra = CliffordAlgebra.for_embedding_dim(
            self.embedding_dim, args.get("clifford_signature")
        )
        self.gnn_residual = residual_mode(args)
        num_bases = args.get("clifford_num_bases")
        self.gnn_layers = nn.ModuleList(
            [
                CliffordRGCNConv(self.clifford_algebra, self.num_relations, num_bases)
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
