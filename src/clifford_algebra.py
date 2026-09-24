"""The Euclidean Clifford (geometric) algebra Cl(n, 0) the Clifford GNN layers are built over.

An embedding is one complete multivector: all ``2**n`` basis blades, so
``embedding_dim`` must be a power of two. Blades are indexed ``0 .. 2**n - 1``
by bitmask over the ``n`` orthonormal generators ``e_1 .. e_n`` (bit ``i`` set
<=> ``e_{i+1}`` participates), the standard encoding for implementing the
geometric product with bit tricks instead of symbolic algebra: the product of
two basis blades ``a``, ``b`` is always ``+/- e_{a^b}``, since XOR clears any
repeated generator and each generator squares to ``+1`` under this signature.

:class:`CliffordAlgebra` exposes the three operations the layers in
:mod:`src.models.clifford_layers` need:

* ``transform(x, w)`` - multiply a batch of multivectors ``x`` by a single
  learned multivector ``w``, the Clifford stand-in for a weight matrix;
* ``scalar_product(x, y)`` - the scalar (grade-0) part of ``x * y``, the
  Clifford stand-in for a dot product;
* ``init_weight(...)`` - sample a learnable multivector at a sensible scale.
"""
import math

import torch
from torch import nn


def canonical_reordering_sign(a: int, b: int) -> int:
    """Sign of the product of basis blades ``a`` and ``b`` (bitmasks) in Cl(n, 0).

    The sign is the parity of the number of transpositions needed to sort the
    concatenated generator lists of ``a`` and ``b``. Reference implementation,
    kept scalar and obvious; the vectorised table built in
    :class:`CliffordAlgebra` is checked against it in the tests.
    """
    shifted = a >> 1
    swaps = 0
    while shifted:
        swaps += bin(shifted & b).count("1")
        shifted >>= 1
    return -1 if swaps % 2 else 1


class CliffordAlgebra(nn.Module):
    """Full multivector Euclidean Clifford algebra Cl(n, 0), ``dim == 2**n``.

    An ``nn.Module`` so its lookup tables travel with ``.to(device)``; they are
    non-persistent buffers because they are constants derived from ``n``, not
    state worth writing into a checkpoint. One instance is shared by every
    layer of a model.
    """

    def __init__(self, n: int):
        super().__init__()
        if n < 0:
            raise ValueError(f"n must be nonnegative, got {n}")
        self.n = n
        self.dim = 1 << n

        blade = torch.arange(self.dim)
        # partner[i, k] is the blade j with i ^ j == k, i.e. the blade of y that
        # lands on output blade k when multiplied by blade i of x.
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
        # Blade i times blade i is the only product landing on the scalar blade.
        self.register_buffer("self_sign", sign[:, 0].clone(), persistent=False)

    @staticmethod
    def for_embedding_dim(embedding_dim: int) -> "CliffordAlgebra":
        """Build the algebra whose full multivector dimension equals ``embedding_dim``."""
        n = round(math.log2(embedding_dim)) if embedding_dim > 0 else -1
        if n < 0 or 2 ** n != embedding_dim:
            raise ValueError(
                f"CliffordAlgebra represents an embedding as one complete multivector of "
                f"Cl(n, 0), which has 2**n basis blades, so embedding_dim must be a power "
                f"of two; got {embedding_dim}")
        return CliffordAlgebra(n)

    def left_matrix(self, w: torch.Tensor) -> torch.Tensor:
        """Matrix of the linear map ``x -> x * w`` for a single multivector ``w``."""
        return self.sign * w[self.partner]

    def transform(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Geometric product of a batch ``x`` ``(..., dim)`` with one multivector ``w``.

        Realised as a matmul against :meth:`left_matrix` rather than a per-row
        ``(dim, dim)`` product, so a whole relation's edges go through one GEMM
        and no ``(num_edges, dim, dim)`` intermediate is ever materialised.
        """
        return x @ self.left_matrix(w)

    def scalar_product(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Scalar (grade-0) part of ``x * y``: ``(..., dim), (..., dim) -> (...)``.

        Only blade ``i`` of ``x`` against blade ``i`` of ``y`` can reach the
        scalar blade, so this is a signed dot product rather than a full product.
        """
        return (x * y * self.self_sign).sum(dim=-1)

    def init_weight(self, *shape: int) -> torch.Tensor:
        """Sample learnable multivectors of shape ``(*shape, dim)``.

        ``std = dim ** -0.5`` because every output blade sums ``dim`` products.
        """
        return torch.randn(*shape, self.dim) * self.dim ** -0.5

    def extra_repr(self) -> str:
        return f"Cl({self.n}, 0), dim={self.dim}"
