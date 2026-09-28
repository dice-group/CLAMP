"""Local Clifford layers built on PyG's MessagePassing and grouped softmax.

The architecture references are R-GCN (Schlichtkrull et al., 2018),
GraphSAGE (Hamilton et al., 2017), and GATv2 (Brody et al., 2022):
https://arxiv.org/abs/1703.06103
https://arxiv.org/abs/1706.02216
https://arxiv.org/abs/2105.14491
The geometric-product operators below are local adaptations of these designs.
In the equations, * denotes the Clifford product and <z>_0 its scalar part.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


def relational_transform(
    algebra, x: torch.Tensor, edge_type: torch.LongTensor, weight: torch.Tensor
) -> torch.Tensor:
    # Group by relation to avoid a (num_edges, dim, dim) intermediate.
    """Return out[e] = x[e] * weight[edge_type[e]] for each edge."""
    out = x.new_zeros(x.shape)
    for relation_id in range(weight.shape[0]):
        mask = edge_type == relation_id
        if not torch.any(mask):
            continue
        out[mask] = algebra.transform(x[mask], weight[relation_id])
    return out


class CliffordLinear(nn.Module):
    """Affine multivector map x * w + b, with an optional bias."""
    def __init__(self, algebra, bias: bool = True):
        super().__init__()
        self.algebra = algebra
        self.weight = nn.Parameter(algebra.init_weight())
        self.bias = nn.Parameter(torch.zeros(algebra.dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.algebra.transform(x, self.weight)
        return out if self.bias is None else out + self.bias


class CliffordRGCNConv(MessagePassing):
    """R-GCN adaptation with one learned multivector per relation.

    PyG supplies propagation; geometric products replace RGCNConv's matrices.
    Incoming messages are averaged separately for each relation.
    """
    def __init__(self, algebra, num_relations: int):
        super().__init__(aggr="add", node_dim=0)
        self.algebra = algebra
        self.num_relations = num_relations
        self.rel_weight = nn.Parameter(algebra.init_weight(num_relations))
        self.root = CliffordLinear(algebra)

    def forward(
        self, x: torch.Tensor, edge_index: torch.LongTensor, edge_type: torch.LongTensor
    ) -> torch.Tensor:
        """Compute x'_i = x_i * w_root + b + sum_r mean_{j in N_r(i)} x_j * w_r.

        N_r(i) contains incoming neighbors of relation r; empty groups add zero.
        """
        return self.propagate(edge_index, x=x, edge_type=edge_type) + self.root(x)

    def message(
        self, x_j: torch.Tensor, edge_type: torch.LongTensor, index: torch.LongTensor
    ) -> torch.Tensor:
        # Normalize by the incoming count for each (target, relation).
        key = index * self.num_relations + edge_type
        count = torch.bincount(key)[key].to(x_j.dtype).unsqueeze(-1)
        return (
            relational_transform(self.algebra, x_j, edge_type, self.rel_weight) / count
        )


class CliffordConv(MessagePassing):
    """Single-relation aggregation of messages x_j * w.

    With mean or sum, the linear product commutes with aggregation.
    With max, this implementation pools transformed messages; PyG's SAGEConv
    applies its neighbor matrix after pooling, so the ordering differs.
    """
    def __init__(self, algebra, aggr: str = "mean"):
        super().__init__(aggr=aggr, node_dim=0)
        self.algebra = algebra
        self.weight = nn.Parameter(algebra.init_weight())

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        return self.propagate(edge_index, x=x)

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        return self.algebra.transform(x_j, self.weight)


class CliffordRelationalSAGELayer(nn.Module):
    def __init__(self, algebra, num_relations: int, aggregator: str = "mean"):
        super().__init__()
        self.num_relations = num_relations
        self.convs = nn.ModuleList(
            [CliffordConv(algebra, aggr=aggregator) for _ in range(num_relations)]
        )
        self.root_lin = CliffordLinear(algebra)

    def forward(
        self, x: torch.Tensor, edge_index: torch.LongTensor, edge_type: torch.LongTensor
    ) -> torch.Tensor:
        """Compute x'_i = x_i * w_root + b + sum_r AGG_{j in N_r(i)}(x_j * w_r).

        The encoder applies activation and coefficient L2 normalization afterward.
        """
        out = self.root_lin(x)
        for relation_id, conv in enumerate(self.convs):
            relation_mask = edge_type == relation_id
            if not torch.any(relation_mask):
                continue
            out = out + conv(x, edge_index[:, relation_mask])
        return out


class CliffordGATv2Conv(MessagePassing):
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
        super().__init__(aggr="add", node_dim=0)
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
        return self.propagate(edge_index, x=x, edge_type=edge_type) + self.root(x)

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_type: torch.LongTensor,
        index: torch.LongTensor,
    ) -> torch.Tensor:
        """Compute attention-weighted messages for edges j -> i of relation r.

        For each head h, with z_j = x_j * w_r:
            u_ijh = LeakyReLU(x_i * q_h + z_j * k_h)
            e_ijh = <u_ijh * a_h>_0
            alpha_ijh = dropout(softmax_{j -> i}(e_ijh))
            m_ij = mean_h(alpha_ijh (z_j * v_h))
        PyG sums m_ij at each target; forward adds x_i * w_root + b.
        """
        x_rel = relational_transform(self.algebra, x_j, edge_type, self.rel_weight)
        logits, values = [], []
        for head in range(self.heads):
            query = self.algebra.transform(x_i, self.query[head])
            key = self.algebra.transform(x_rel, self.key[head])
            y = query + key
            y = F.leaky_relu(y, self.negative_slope)
            logits.append(self.algebra.scalar_product(y, self.att[head]))
            values.append(self.algebra.transform(x_rel, self.value[head]))
        alpha = softmax(torch.stack(logits, dim=1), index)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        values = torch.stack(values, dim=1)
        # Averaging heads preserves the full multivector dimension.
        return (values * alpha.unsqueeze(-1)).mean(dim=1)
