"""Clifford-algebra counterparts of the linear and relational-convolution layers
used by :mod:`src.models.rgcn`, :mod:`src.models.graphsage` and :mod:`src.models.rgat`.

Each layer keeps the *structure* of the layer it copies and replaces only its
learned linear maps. Where the originals learn a weight matrix ``W`` (or one
``W_r`` per relation) and send ``x_j @ W_r``, these learn a multivector ``w``
(or one ``w_r`` per relation) and send the geometric product ``x_j * w_r`` -
the multivector-valued generalisation of the per-relation multiplicative
transform that RotatE (complex), QuatE (quaternion) and Keci (Clifford) use
as their relation operator.

Every layer reaches the algebra through
:class:`~src.clifford_algebra.CliffordAlgebra`'s ``transform`` /
``scalar_product`` / ``init_weight``, and one algebra instance is shared by
every layer of a model.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


def relational_transform(algebra, x: torch.Tensor, edge_type: torch.LongTensor,
                          weight: torch.Tensor) -> torch.Tensor:
    """Per-edge geometric product ``x[e] * weight[edge_type[e]]``.

    Loops over relations and transforms each relation's rows in one call, the
    same way ``RGCNConv`` applies its per-relation weight matrices: the
    alternative - gathering one multivector per edge - would make the full
    algebra materialise an ``(num_edges, dim, dim)`` operator.
    """
    out = x.new_zeros(x.shape)
    for relation_id in range(weight.shape[0]):
        mask = edge_type == relation_id
        if not torch.any(mask):
            continue
        out[mask] = algebra.transform(x[mask], weight[relation_id])
    return out


class CliffordLinear(nn.Module):
    """``x * w + b``: the Clifford analogue of ``nn.Linear`` as a root/self transform."""

    def __init__(self, algebra, bias: bool = True):
        super().__init__()
        self.algebra = algebra
        self.weight = nn.Parameter(algebra.init_weight())
        self.bias = nn.Parameter(torch.zeros(algebra.dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.algebra.transform(x, self.weight)
        return out if self.bias is None else out + self.bias


class CliffordRGCNConv(MessagePassing):
    """Clifford analogue of ``torch_geometric.nn.RGCNConv``.

    One learned multivector ``w_r`` per relation in place of ``RGCNConv``'s
    per-relation weight matrix; messages ``x_j * w_r`` are mean-aggregated over
    each node's incoming edges (``RGCNConv``'s default ``aggr='mean'``,
    reproducing the paper's ``1/c_{i,r}`` normalisation) and a shared
    :class:`CliffordLinear` supplies the self-loop (``RGCNConv``'s ``root``).
    """

    def __init__(self, algebra, num_relations: int):
        super().__init__(aggr="mean", node_dim=0)
        self.algebra = algebra
        self.rel_weight = nn.Parameter(algebra.init_weight(num_relations))
        self.root = CliffordLinear(algebra)

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor,
                edge_type: torch.LongTensor) -> torch.Tensor:
        return self.propagate(edge_index, x=x, edge_type=edge_type) + self.root(x)

    def message(self, x_j: torch.Tensor, edge_type: torch.LongTensor) -> torch.Tensor:
        return relational_transform(self.algebra, x_j, edge_type, self.rel_weight)


class CliffordConv(MessagePassing):
    """Clifford analogue of a single-relation ``SAGEConv`` neighbour transform:
    messages ``x_j * w`` for one learned multivector, aggregated with ``aggr``.

    Building block for :class:`CliffordRelationalSAGELayer`, which owns one per relation.
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
    """Clifford analogue of :class:`src.models.graphsage.RelationalSAGELayer`:
    one :class:`CliffordConv` per relation, summed, plus a shared
    :class:`CliffordLinear` root transform applied once (not once per relation)."""

    def __init__(self, algebra, num_relations: int, aggregator: str = "mean"):
        super().__init__()
        self.num_relations = num_relations
        self.convs = nn.ModuleList([CliffordConv(algebra, aggr=aggregator)
                                    for _ in range(num_relations)])
        self.root_lin = CliffordLinear(algebra)

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor,
                edge_type: torch.LongTensor) -> torch.Tensor:
        out = self.root_lin(x)
        for relation_id, conv in enumerate(self.convs):
            relation_mask = edge_type == relation_id
            if not torch.any(relation_mask):
                continue
            out = out + conv(x, edge_index[:, relation_mask])
        return out


class CliffordGATv2Conv(MessagePassing):
    """Clifford, relation-aware analogue of ``GATv2Conv``.

    Query/key/value projections are geometric products with learned
    multivectors, one set per head. Relation identity enters the way the
    non-Clifford model injects it as ``edge_attr`` (:mod:`src.models.rgat`):
    the source state is first multiplied by the edge's relation multivector,
    and that relation-transformed state feeds both the attention logit and the
    message. GATv2's logit ``a^T LeakyReLU(W_l x_i + W_r x_j)`` becomes
    ``<a * LeakyReLU(x_i * w_l + (x_j * w_rel) * w_r)>_0``, the scalar part of a
    geometric product standing in for the dot product with ``a``.

    Heads are averaged at every layer rather than concatenated on hidden layers
    as in the GAT/GATv2 papers: concatenation would change the width, but a
    multivector's width is fixed by the algebra, so there is no
    ``embedding_dim // heads`` per-head representation to concatenate.
    """

    def __init__(self, algebra, num_relations: int, heads: int = 4,
                 dropout: float = 0.0, negative_slope: float = 0.2):
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

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor,
                edge_type: torch.LongTensor) -> torch.Tensor:
        return self.propagate(edge_index, x=x, edge_type=edge_type) + self.root(x)

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_type: torch.LongTensor,
                index: torch.LongTensor) -> torch.Tensor:
        x_rel = relational_transform(self.algebra, x_j, edge_type, self.rel_weight)
        logits, values = [], []
        for head in range(self.heads):
            y = self.algebra.transform(x_i, self.query[head]) \
                + self.algebra.transform(x_rel, self.key[head])
            y = F.leaky_relu(y, self.negative_slope)
            logits.append(self.algebra.scalar_product(y, self.att[head]))
            values.append(self.algebra.transform(x_rel, self.value[head]))
        alpha = softmax(torch.stack(logits, dim=1), index)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return (torch.stack(values, dim=1) * alpha.unsqueeze(-1)).mean(dim=1)
