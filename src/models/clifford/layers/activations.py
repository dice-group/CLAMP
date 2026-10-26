"""Activations for multivector features: coefficient-wise (split) or gated on an invariant.

In the equations, <z>_0 denotes the scalar part of a multivector z.
"""

import torch
from torch import nn

from src.models.gnn.layers.activations import SPLIT_ACTIVATIONS

#: Activations that act on the multivector as a whole rather than coefficient by coefficient.
MULTIVECTOR_ACTIVATIONS = ("mv_gate", "mv_grade_gate")


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
