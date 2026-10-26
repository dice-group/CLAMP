"""R-GCN (Schlichtkrull et al., 2018, https://arxiv.org/abs/1703.06103) with learned
multivectors in place of RGCNConv's per-relation matrices.

Each embedding holds C = dim / 2**(p+q) full Cl(p, q) multivectors; * is the Clifford product.
"""

import torch
from torch import nn
from torch_geometric.nn.inits import glorot

from src.models.clifford.layers.relational import CliffordLinear, relational_aggregate


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
