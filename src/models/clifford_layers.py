"""Shared Clifford building blocks: the affine multivector map and per-relation products.

In the equations, * denotes the Clifford product and <z>_0 its scalar part.
"""

import torch
from torch import nn
from torch_geometric.utils import scatter


def relational_transform(
    algebra, x: torch.Tensor, edge_type: torch.LongTensor, weight: torch.Tensor
) -> torch.Tensor:
    """Return out[e] = x[e] * weight[edge_type[e]] for each row.

    Rows are sorted by relation and each relation's rows go through one matrix product.
    This avoids a (rows, dim, dim) intermediate and needs a single device sync.
    """
    order = torch.argsort(edge_type, stable=True)
    counts = torch.bincount(edge_type, minlength=weight.shape[0]).tolist()
    matrices = algebra.left_matrix(weight)
    chunks = torch.split(x[order], counts)
    out = torch.empty_like(x)
    out[order] = torch.cat([chunk @ matrices[r] for r, chunk in enumerate(chunks)])
    return out


def relational_aggregate(
    algebra,
    x: torch.Tensor,
    edge_index: torch.LongTensor,
    edge_type: torch.LongTensor,
    weight: torch.Tensor,
    reduce: str = "mean",
) -> torch.Tensor:
    """Return out_i = sum_r REDUCE_{j in N_r(i)} (x_j * w_r); empty groups add zero.

    For mean and sum the product is linear, so neighbours are reduced per (target, relation)
    first and each group is multiplied once. Max pools the transformed messages.
    """
    src, dst = edge_index
    num_nodes = x.size(0)
    groups, group_of_edge = torch.unique(edge_type * num_nodes + dst, return_inverse=True)
    if reduce == "max":
        messages = relational_transform(algebra, x[src], edge_type, weight)
        pooled = scatter(messages, group_of_edge, dim=0, dim_size=len(groups), reduce="max")
    else:
        pooled = scatter(x[src], group_of_edge, dim=0, dim_size=len(groups), reduce=reduce)
        pooled = relational_transform(algebra, pooled, groups // num_nodes, weight)
    return x.new_zeros(x.shape).index_add_(0, groups % num_nodes, pooled)


#: Activations that act on the multivector as a whole rather than coefficient by coefficient.
MULTIVECTOR_ACTIVATIONS = ("mv_gate", "mv_grade_gate")

#: Activations applied to each coefficient independently.
SPLIT_ACTIVATIONS = {
    "relu": torch.relu,
    "leaky_relu": nn.functional.leaky_relu,
    "elu": nn.functional.elu,
    "none": None,
}


class MultivectorGate(nn.Module):
    """Scale each multivector by sigmoid(a * q(x) + b), with q(x) = <x~ x>_0.

    One gate per channel. Because q is invariant up to sign under a common left versor,
    and messages x * w_r commute with left multiplication, this scaling preserves that
    symmetry where a coefficient-wise activation does not.
    """

    def __init__(self, algebra):
        super().__init__()
        self.algebra = algebra
        self.scale = nn.Parameter(torch.ones(algebra.channels))
        self.shift = nn.Parameter(torch.zeros(algebra.channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.scale * self.algebra.quadratic_form(x) + self.shift)
        return x * self.algebra.broadcast_over_blades(gate)


class MultivectorGradeGate(nn.Module):
    """Scale each grade by its own sigmoid gate, as in CGENN's MVSiLU.

    The grade-0 invariant is the scalar coefficient itself; higher grades use
    <x~ x>_0 restricted to that grade. Gates are per grade and channel.
    """

    def __init__(self, algebra):
        super().__init__()
        self.algebra = algebra
        self.scale = nn.Parameter(torch.ones(algebra.grades, algebra.channels))
        self.shift = nn.Parameter(torch.zeros(algebra.grades, algebra.channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        invariants = self.algebra.grade_quadratic_form(x)
        invariants = torch.cat(
            [self.algebra.scalar_part(x).unsqueeze(-2), invariants[..., 1:, :]], dim=-2
        )
        gate = torch.sigmoid(self.scale * invariants + self.shift)
        # Gate per grade, then spread back over that grade's blades.
        per_blade = gate.transpose(-2, -1) @ self.algebra.grade_onehot
        return x * per_blade.transpose(-2, -1).reshape(*x.shape[:-1], self.algebra.dim)


def clifford_activation(algebra, name: str):
    """Return the activation module (or None) that ``--gnn_activation`` selects."""
    if name == "mv_gate":
        return MultivectorGate(algebra)
    if name == "mv_grade_gate":
        return MultivectorGradeGate(algebra)
    if name in SPLIT_ACTIVATIONS:
        fn = SPLIT_ACTIVATIONS[name]
        return None if fn is None else _Split(fn)
    raise ValueError(f"Unknown gnn_activation '{name}'")


class _Split(nn.Module):
    """Coefficient-wise activation, the split nonlinearity of Clifford Neural Layers."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)


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


#: Where a residual connection sits relative to the hidden activation.
RESIDUAL_MODES = ("none", "post", "pre")


def residual_mode(args: dict) -> str:
    """Read ``--gnn_residual``, accepting the boolean recorded by runs before 2026-10-16.

    Those runs stored True for what is now ``post``: the activation saw the summed signal.
    """
    value = args.get("gnn_residual", "none")
    if isinstance(value, bool):
        return "post" if value else "none"
    value = value or "none"
    if value not in RESIDUAL_MODES:
        raise ValueError(f"Unknown gnn_residual '{value}'. Available: {list(RESIDUAL_MODES)}")
    return value
