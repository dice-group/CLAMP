"""Per-relation Clifford products and the affine multivector map.

In the equations, * denotes the Clifford product.
"""

import torch
from torch import nn
from torch_geometric.utils import scatter


def relational_transform(
    x: torch.Tensor, edge_type: torch.LongTensor, matrices: torch.Tensor
) -> torch.Tensor:
    """Return out[e] = x[e] @ matrices[edge_type[e]] for each row.

    ``matrices`` (num_relations, dim, dim) come from algebra.clifford_multiplication_matrix (x * w_r) or
    algebra.mixing_matrix. Rows are sorted by relation and each relation's rows go through
    one matrix product. This avoids a (rows, dim, dim) intermediate and needs a single
    device sync.
    """
    order = torch.argsort(edge_type, stable=True)
    counts = torch.bincount(edge_type, minlength=matrices.shape[0]).tolist()
    chunks = torch.split(x[order], counts)
    out = torch.empty_like(x)
    out[order] = torch.cat([chunk @ matrices[r] for r, chunk in enumerate(chunks)])
    return out


def relational_aggregate(
    x: torch.Tensor,
    edge_index: torch.LongTensor,
    edge_type: torch.LongTensor,
    matrices: torch.Tensor,
    reduce: str = "mean",
) -> torch.Tensor:
    """Return out_i = sum_r REDUCE_{j in N_r(i)} (x_j @ M_r); empty groups add zero.

    For mean and sum the map is linear, so neighbours are reduced per (target, relation)
    first and each group is multiplied once. Max pools the transformed messages.
    """
    src, dst = edge_index
    num_nodes = x.size(0)
    groups, group_of_edge = torch.unique(edge_type * num_nodes + dst, return_inverse=True)
    if reduce == "max":
        messages = relational_transform(x[src], edge_type, matrices)
        pooled = scatter(messages, group_of_edge, dim=0, dim_size=len(groups), reduce="max")
    else:
        pooled = scatter(x[src], group_of_edge, dim=0, dim_size=len(groups), reduce=reduce)
        pooled = relational_transform(pooled, groups // num_nodes, matrices)
    return x.new_zeros(x.shape).index_add_(0, groups % num_nodes, pooled)


class CliffordLinear(nn.Module):
    """Affine multivector map x * w + b, with an optional bias."""
    def __init__(self, algebra, bias: bool = True):
        super().__init__()
        self.algebra = algebra
        self.weight = nn.Parameter(algebra.init_weight())
        self.bias = nn.Parameter(torch.zeros(algebra.dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.algebra.clifford_multiplication(x, self.weight)
        return out if self.bias is None else out + self.bias
