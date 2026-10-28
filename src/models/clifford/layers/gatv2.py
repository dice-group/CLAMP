"""GATv2-inspired attention (Brody et al., 2022, https://arxiv.org/abs/2105.14491) with
multivector query, key, value and relation maps.

Each embedding is a full Cl(n, 0) multivector with 2**n coefficients; * is the Clifford product
and <z>_0 its scalar part.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import softmax

from src.models.clifford.layers.relational import CliffordLinear, relational_transform


class CliffordGATv2Conv(nn.Module):
    """GATv2-inspired attention with multivector query, key, and value maps.

    Relation products affect both attention and values. PyG's GATv2Conv
    uses additive edge features in attention only. This layer uses a separate
    root map and averages full-width heads at every depth.
    """
    def __init__(
        self,
        algebra,
        num_relations: int,
        heads: int = 4,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.algebra = algebra
        self.heads = heads
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.rel_weight = nn.Parameter(algebra.init_weight(num_relations))
        self.query = nn.Parameter(algebra.init_weight(heads))
        self.key = nn.Parameter(algebra.init_weight(heads))
        self.value = nn.Parameter(algebra.init_weight(heads))
        self.att = nn.Parameter(algebra.init_weight(heads))
        self.root = CliffordLinear(algebra)

    def forward(
        self, x: torch.Tensor, edge_index: torch.LongTensor, edge_type: torch.LongTensor
    ) -> torch.Tensor:
        """Attention-weighted messages over edges j -> i of relation r, plus x_i * w_root + b.

        For each head h, with z_j = x_j * w_r:
            u_ijh = LeakyReLU(x_i * q_h + z_j * k_h)
            e_ijh = <u_ijh * a_h>_0
            alpha_ijh = dropout(softmax_{j -> i}(e_ijh))
            x'_i = sum_j mean_h(alpha_ijh (z_j * v_h)) + x_i * w_root + b
        Queries are computed once per node; all heads share one batched product.
        """
        src, dst = edge_index
        mul = self.algebra.clifford_multiplication_matrix
        z = relational_transform(x[src], edge_type, mul(self.rel_weight))
        query = torch.einsum("nd,hdk->nhk", x, mul(self.query))  # (nodes, heads, dim)
        key = torch.einsum("ed,hdk->ehk", z, mul(self.key))       # (edges, heads, dim)
        value = torch.einsum("ed,hdk->ehk", z, mul(self.value))
        u = F.leaky_relu(query[dst] + key, self.negative_slope)
        logits = self.algebra.scalar_product(u, self.att)          # (edges, heads)
        alpha = softmax(logits, dst, num_nodes=x.size(0))
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        messages = (value * alpha.unsqueeze(-1)).mean(dim=1)
        return x.new_zeros(x.shape).index_add_(0, dst, messages) + self.root(x)
