"""R-GCN (Schlichtkrull et al., 2018, https://arxiv.org/abs/1703.06103) with learned
multivectors in place of RGCNConv's per-relation matrices.

Each embedding holds C = dim / 2**(p+q) full Cl(p, q) multivectors; * is the Clifford product.
"""

import torch
from torch import nn
from torch_geometric.nn.inits import glorot

from src.models.clifford.layers.relational import CliffordLinear, relational_aggregate

CHANNEL_MIX = ("none", "relation", "shared")


class CliffordRGCNConv(nn.Module):
    """R-GCN adaptation with multivector relation weights.

    Geometric products replace RGCNConv's matrices; incoming messages are averaged
    separately for each relation, as in RGCNConv(aggr='mean').

    Without num_bases every relation has its own multivector w_r. With num_bases B,
    w_r = sum_b comp[r, b] * basis_b over B shared multivectors, RGCNConv's basis
    decomposition with multivectors in place of matrices. As in RGCNConv, comp is
    glorot-initialized; the bases use the algebra's init.

    channel_mix lets the C channels interact. 'none': channel c of x_j is multiplied only by
    channel c of w_r. 'relation': each relation has a C x C grid of multivectors W_r[o, c] and
    output channel o is sum_c x_{j,c} * W_r[o, c], the channel mixing of Clifford Neural Layers
    (Brandstetter et al., 2023); no basis decomposition. 'shared': relations keep their
    multivector per channel and one C x C grid M per layer mixes the summed messages,
    sum_c m_c * M[o, c]. Mixing grids are sampled from N(0, 1 / (blades * C)), so each output
    coefficient sums blades * C products at the scale of the algebra's init.
    """
    def __init__(
        self, algebra, num_relations: int, num_bases: int = None, channel_mix: str = "none"
    ):
        super().__init__()
        if num_bases is not None and num_bases < 1:
            raise ValueError(f"num_bases must be positive, got {num_bases}")
        if channel_mix not in CHANNEL_MIX:
            raise ValueError(f"channel_mix must be one of {CHANNEL_MIX}, got '{channel_mix}'")
        if channel_mix == "relation" and num_bases is not None:
            raise ValueError("channel_mix 'relation' has no basis decomposition")
        self.algebra = algebra
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.channel_mix = channel_mix
        if channel_mix == "relation":
            self.rel_weight = nn.Parameter(self._mixing_init(num_relations))
        elif num_bases is None:
            self.rel_weight = nn.Parameter(algebra.init_weight(num_relations))
        else:
            self.bases = nn.Parameter(algebra.init_weight(num_bases))
            self.comp = nn.Parameter(torch.empty(num_relations, num_bases))
            glorot(self.comp)
        self.root = CliffordLinear(algebra)
        if channel_mix == "shared":
            self.mix_weight = nn.Parameter(self._mixing_init())

    def _mixing_init(self, *shape: int) -> torch.Tensor:
        a = self.algebra
        return torch.randn(*shape, a.channels, a.channels, a.num_blades) * (a.num_blades * a.channels) ** -0.5

    def relation_weights(self) -> torch.Tensor:
        """Return the multivectors w_r: (num_relations, dim), or (num_relations, C, C, blades)
        with channel_mix 'relation'."""
        if self.num_bases is None:
            return self.rel_weight
        return self.comp @ self.bases

    def relation_matrices(self) -> torch.Tensor:
        """Return the (num_relations, dim, dim) matrices M_r with x_j @ M_r the message."""
        if self.channel_mix == "relation":
            return self.algebra.mixing_matrix(self.rel_weight)
        return self.algebra.clifford_multiplication_matrix(self.relation_weights())

    def forward(
        self, x: torch.Tensor, edge_index: torch.LongTensor, edge_type: torch.LongTensor
    ) -> torch.Tensor:
        """Compute x'_i = x_i * w_root + b + sum_r mean_{j in N_r(i)} x_j * w_r."""
        messages = relational_aggregate(x, edge_index, edge_type, self.relation_matrices(), "mean")
        if self.channel_mix == "shared":
            messages = messages @ self.algebra.mixing_matrix(self.mix_weight)
        return messages + self.root(x)

    def extra_repr(self) -> str:
        return (f"num_relations={self.num_relations}, num_bases={self.num_bases}, "
                f"channel_mix={self.channel_mix}")
