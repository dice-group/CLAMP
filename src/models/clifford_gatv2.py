"""GATv2-inspired attention (Brody et al., 2022, https://arxiv.org/abs/2105.14491) with
multivector query, key, value and relation maps.

Each embedding is a full Cl(n, 0) multivector with 2**n coefficients; * is the Clifford product
and <z>_0 its scalar part.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import softmax

from src.clifford_algebra import CliffordAlgebra
from src.models.clifford_layers import CliffordLinear, relational_transform
from src.models.gnn import RelationalGNN


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
        z = relational_transform(self.algebra, x[src], edge_type, self.rel_weight)
        left = self.algebra.left_matrix
        query = torch.einsum("nd,hdk->nhk", x, left(self.query))  # (nodes, heads, dim)
        key = torch.einsum("ed,hdk->ehk", z, left(self.key))       # (edges, heads, dim)
        value = torch.einsum("ed,hdk->ehk", z, left(self.value))
        u = F.leaky_relu(query[dst] + key, self.negative_slope)
        logits = self.algebra.scalar_product(u, self.att)          # (edges, heads)
        alpha = softmax(logits, dst, num_nodes=x.size(0))
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        messages = (value * alpha.unsqueeze(-1)).mean(dim=1)
        return x.new_zeros(x.shape).index_add_(0, dst, messages) + self.root(x)


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
