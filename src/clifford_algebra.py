import math
from typing import Optional, Sequence

import torch
from torch import nn


def canonical_reordering_sign(a: int, b: int) -> int:
    """Return s(a, b) in e_a * e_b = s(a, b) e_(a XOR b) when every generator squares to +1.

    Each swap of distinct generators changes the sign.
    The shifted bit counts give the number of swaps modulo two.
    """
    shifted = a >> 1
    swaps = 0
    while shifted:
        swaps += bin(shifted & b).count("1")
        shifted >>= 1
    return -1 if swaps % 2 else 1


def blade_product_sign(a: int, b: int, squares: Sequence[int]) -> int:
    """Return s(a, b) in e_a * e_b = s(a, b) e_(a XOR b), where generator i squares to squares[i].

    The reordering sign times the square of every generator that the two blades share.
    """
    sign = canonical_reordering_sign(a, b)
    for i, square in enumerate(squares):
        if (a >> i) & (b >> i) & 1:
            sign *= square
    return sign


class CliffordAlgebra(nn.Module):
    """Clifford algebra Cl(p, q): p generators square to +1 and q to -1.

    A multivector has 2**(p + q) coefficients ("blades"); blade indices are generator
    bitmasks, generators ordered as the p positive, then the q negative ones.
    An embedding of width ``dim`` holds C = dim / blades independent multivectors ("channels"),
    and every operation acts on each channel separately. Coefficients are stored blade-major,
    as in dicee's Keci: position b * C + c holds blade b of channel c, so Cl(0, 1) is the
    [real | imaginary] layout of ComplEx. With one channel the embedding is a single
    multivector, as in Cl(log2 dim, 0).

    Product tables cover the whole embedding as a (dim, dim) map that only couples
    coefficients of the same channel, so multiplying by a multivector is one matrix
    product for any signature.
    """

    def __init__(self, p: int, q: int = 0, dim: Optional[int] = None):
        super().__init__()
        if min(p, q) < 0:
            raise ValueError(f"Signature must be nonnegative, got ({p}, {q})")
        self.p, self.q = p, q
        n = p + q
        self.blades = 1 << n
        self.dim = self.blades if dim is None else dim
        if self.dim % self.blades:
            raise ValueError(
                f"Cl({p}, {q}) multivectors have {self.blades} coefficients, "
                f"which must divide the embedding dim {self.dim}"
            )
        self.channels = self.dim // self.blades

        # One multivector: blade i times blade partner[i, k] contributes to blade k.
        blade = torch.arange(self.blades)
        partner = blade.unsqueeze(1) ^ blade.unsqueeze(0)
        popcount = torch.tensor([bin(v).count("1") for v in range(self.blades)])
        swaps = torch.zeros_like(partner)
        shifted = blade.unsqueeze(1).expand_as(partner) >> 1
        for _ in range(n):
            swaps = swaps + popcount[shifted & partner]
            shifted = shifted >> 1
        sign = torch.where(swaps % 2 == 1, -1.0, 1.0)
        # Generators shared by blade i and its partner meet and contribute their squares.
        squares = [1.0] * p + [-1.0] * q
        shared = blade.unsqueeze(1) & partner
        for i, square in enumerate(squares):
            sign = torch.where((shared >> i) & 1 == 1, sign * square, sign)

        # Whole embedding: the same table within each channel, zeros between channels.
        # Blade b of channel c sits at position b * channels + c.
        channel = torch.arange(self.channels)[:, None, None]
        coeff = (partner * self.channels + channel).flatten()
        rows = (blade[None, :, None] * self.channels + channel).expand(-1, -1, self.blades).flatten()
        cols = (blade[None, None, :] * self.channels + channel).expand(-1, self.blades, -1).flatten()
        full_partner = torch.zeros(self.dim, self.dim, dtype=torch.long)
        full_sign = torch.zeros(self.dim, self.dim)
        full_partner[rows, cols] = coeff
        full_sign[rows, cols] = sign.repeat(self.channels, 1, 1).flatten()

        self.register_buffer("partner", full_partner, persistent=False)
        self.register_buffer("sign", full_sign, persistent=False)
        # Only equal blades multiply to a scalar: the column of the channel's scalar blade.
        position = torch.arange(self.dim)
        self_sign = full_sign[position, position % self.channels]
        self.register_buffer("self_sign", self_sign.clone(), persistent=False)

        # Grade of each blade, and the reversion sign (-1)^(k(k-1)/2) of a grade-k blade.
        self.grades = n + 1
        grade = popcount.clone()
        reversion = torch.where((grade * (grade - 1) // 2) % 2 == 1, -1.0, 1.0)
        self.register_buffer("blade_grade", grade, persistent=False)
        # <x~ x>_0 per channel sums self_sign * reversion * x^2 over blades.
        quad = (self_sign.view(self.blades, self.channels) * reversion[:, None])
        self.register_buffer("quad_weight", quad, persistent=False)
        onehot = torch.zeros(self.grades, self.blades)
        onehot[grade, torch.arange(self.blades)] = 1.0
        self.register_buffer("grade_onehot", onehot, persistent=False)

    def split_blades(self, x: torch.Tensor) -> torch.Tensor:
        """View (..., dim) coefficients as (..., blades, channels); storage is blade-major."""
        return x.reshape(*x.shape[:-1], self.blades, self.channels)

    def quadratic_form(self, x: torch.Tensor) -> torch.Tensor:
        """Return q(x) = <x~ x>_0 per channel, shape (..., channels).

        Invariant up to sign under a common left versor g, since (gx)~(gx) = x~ (g~g) x
        and g~g = +-1. Indefinite when q > 0, so callers must not take its square root.
        """
        return (self.split_blades(x) ** 2 * self.quad_weight).sum(dim=-2)

    def grade_quadratic_form(self, x: torch.Tensor) -> torch.Tensor:
        """Return <x~ x>_0 restricted to each grade, shape (..., grades, channels)."""
        weighted = self.split_blades(x) ** 2 * self.quad_weight
        return self.grade_onehot @ weighted

    def scalar_part(self, x: torch.Tensor) -> torch.Tensor:
        """Return the grade-0 coefficient of each channel, shape (..., channels)."""
        return self.split_blades(x)[..., 0, :]

    def broadcast_over_blades(self, gate: torch.Tensor) -> torch.Tensor:
        """Expand a per-channel (..., channels) factor back to (..., dim)."""
        return gate.unsqueeze(-2).expand(*gate.shape[:-1], self.blades, self.channels).reshape(
            *gate.shape[:-1], self.dim
        )

    @staticmethod
    def for_embedding_dim(
        embedding_dim: int, signature: Optional[Sequence[int]] = None
    ) -> "CliffordAlgebra":
        """Cl(signature) channels filling embedding_dim; by default one Cl(log2 dim, 0) multivector."""
        if signature is None:
            n = round(math.log2(embedding_dim)) if embedding_dim > 0 else -1
            if n < 0 or 2**n != embedding_dim:
                raise ValueError(
                    f"embedding_dim must be a power of two without --clifford_signature; "
                    f"got {embedding_dim}"
                )
            return CliffordAlgebra(n, dim=embedding_dim)
        if len(signature) != 2:
            raise ValueError(f"--clifford_signature takes p q, got {list(signature)}")
        return CliffordAlgebra(*signature, dim=embedding_dim)

    def left_matrix(self, w: torch.Tensor) -> torch.Tensor:
        """Return M[i, k] = s(i, i XOR k) w[i XOR k] per channel, for w of shape (..., dim).

        For row-vector coefficients, x @ M is x * w channel by channel: multiplication by w
        on the right, despite the historical method name.
        """
        return self.sign * w[..., self.partner]

    def transform(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Compute (x * w)_k = sum_i x_i s(i, i XOR k) w_(i XOR k) in every channel.

        One matrix multiplication handles every row of x (..., dim).
        """
        return x @ self.left_matrix(w)

    def scalar_product(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return sum over channels of <x * y>_0 = sum_i s(i, i) x_i y_i.

        No reversion is applied to y; this bilinear form is not a Euclidean
        inner product and need not be positive for x == y.
        """
        return (x * y * self.self_sign).sum(dim=-1)

    def init_weight(self, *shape: int) -> torch.Tensor:
        """Sample coefficients from N(0, 1/blades); each output coefficient sums blades products."""
        return torch.randn(*shape, self.dim) * self.blades**-0.5

    def extra_repr(self) -> str:
        return f"Cl({self.p}, {self.q}), blades={self.blades}, channels={self.channels}"
