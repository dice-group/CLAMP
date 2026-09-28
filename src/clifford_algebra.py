import math

import torch
from torch import nn


def canonical_reordering_sign(a: int, b: int) -> int:
    """Return s(a, b) in e_a * e_b = s(a, b) e_(a XOR b).

    Each swap of distinct generators changes the sign; e_i * e_i = +1.
    The shifted bit counts give the number of swaps modulo two.
    """
    shifted = a >> 1
    swaps = 0
    while shifted:
        swaps += bin(shifted & b).count("1")
        shifted >>= 1
    return -1 if swaps % 2 else 1


class CliffordAlgebra(nn.Module):
    """Euclidean Cl(n, 0); blade indices are generator bitmasks.

    A multivector has 2**n coefficients, including its scalar coefficient.
    Product tables are fixed buffers and move with the module's device.
    """

    def __init__(self, n: int):
        super().__init__()
        if n < 0:
            raise ValueError(f"n must be nonnegative, got {n}")
        self.n = n
        self.dim = 1 << n

        blade = torch.arange(self.dim)
        # Blade i times blade partner[i, k] contributes to blade k.
        partner = blade.unsqueeze(1) ^ blade.unsqueeze(0)
        popcount = torch.tensor([bin(v).count("1") for v in range(self.dim)])
        swaps = torch.zeros_like(partner)
        shifted = blade.unsqueeze(1).expand_as(partner) >> 1
        for _ in range(n):
            swaps = swaps + popcount[shifted & partner]
            shifted = shifted >> 1
        sign = torch.where(swaps % 2 == 1, -1.0, 1.0)

        self.register_buffer("partner", partner, persistent=False)
        self.register_buffer("sign", sign, persistent=False)
        # Only equal blades multiply to a scalar.
        self.register_buffer("self_sign", sign[:, 0].clone(), persistent=False)

    @staticmethod
    def for_embedding_dim(embedding_dim: int) -> "CliffordAlgebra":
        n = round(math.log2(embedding_dim)) if embedding_dim > 0 else -1
        if n < 0 or 2**n != embedding_dim:
            raise ValueError(
                f"embedding_dim must be a power of two; got {embedding_dim}"
            )
        return CliffordAlgebra(n)

    def left_matrix(self, w: torch.Tensor) -> torch.Tensor:
        """Return M[i, k] = s(i, i XOR k) w[i XOR k].

        For row-vector coefficients, x @ M is x * w: multiplication by w
        on the right, despite the historical method name.
        """
        return self.sign * w[self.partner]

    def transform(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Compute (x * w)_k = sum_i x_i s(i, i XOR k) w_(i XOR k).

        One matrix multiplication handles every row of x (..., dim).
        """
        return x @ self.left_matrix(w)

    def scalar_product(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return <x * y>_0 = sum_i s(i, i) x_i y_i.

        No reversion is applied to y; this bilinear form is not a Euclidean
        inner product and need not be positive for x == y.
        """
        return (x * y * self.self_sign).sum(dim=-1)

    def init_weight(self, *shape: int) -> torch.Tensor:
        """Sample coefficients from N(0, 1/dim); each output sums dim products."""
        return torch.randn(*shape, self.dim) * self.dim**-0.5

    def extra_repr(self) -> str:
        return f"Cl({self.n}, 0), dim={self.dim}"
