"""Clifford algebra Cl(p, q) on embedding vectors.

The math:

    generators   e_1 .. e_n (n = p + q),  e_i e_i = +1 for i <= p, -1 otherwise,
                 e_i e_j = -e_j e_i for i != j
    blades       products of distinct generators, written in increasing order:
                 1, e1, e2, e1e2, e3, e1e3, ...  (2^n of them; grade = number of generators)
    multivector  x = sum_A x_A e_A, one coefficient per blade
    product      x * y = sum_{A,B} x_A y_B (e_A e_B), and e_A e_B = s e_C for one blade C, s = +-1

    Cl(0,1):  e1 e1 = -1, so (a + b e1)(c + d e1) = (ac - bd) + (ad + bc) e1, the complex product.
    Cl(0,2):  e1, e2 and e1e2 all square to -1: the quaternions.

Storage: blade A sits at index sum_{i in A} 2^(i-1), so the bits of the index list its generators
(index 3 = 0b11 = e1e2). An embedding of width dim holds C = dim / 2^n multivectors ("channels"),
stored blade-major: position b * C + c holds blade b of channel c, so Cl(0,1) is the
[real | imaginary] layout of ComplEx and of dicee's Keci with one generator. (Keci keeps only the
scalar and the n generator coefficients, [scalars | p block | q block], and inside a block each
multivector's coefficients sit together, so for n > 1 its layout differs from this one.)

    dim = 8, Cl(0,1), C = 4:   [ 1  1  1  1 | e1 e1 e1 e1 ]
                                 c0 c1 c2 c3   c0 c1 c2 c3

The product rule is applied once, in Python, to build tables; training multiplies by them.
"""

import math
from typing import Optional, Sequence, Tuple

import torch
from torch import nn


def blade_generators(index: int) -> Tuple[int, ...]:
    """Return the generators (0-based, increasing) of the blade stored at ``index``."""
    return tuple(i for i in range(index.bit_length()) if (index >> i) & 1)


def blade_index(generators: Sequence[int]) -> int:
    """Return the storage index of the blade made of distinct ``generators``."""
    return sum(1 << i for i in generators)


def blade_name(index: int) -> str:
    """Return a readable name, e.g. 0 -> '1', 5 -> 'e1e3'."""
    return "".join(f"e{i + 1}" for i in blade_generators(index)) or "1"


def multiply_blades(a: Sequence[int], b: Sequence[int], squares: Sequence[float]):
    """Return (s, C) with e_A e_B = s e_C, for blades given as increasing generator lists.

    Write the generators of A then B side by side, sort them by swapping neighbours
    (e_i e_j = -e_j e_i, so each swap flips the sign), then replace every equal pair
    e_i e_i by its square squares[i].
        e1e2 * e1 = e1 e2 e1 = -e1 e1 e2 = -squares[0] e2
    """
    generators, sign = list(a) + list(b), 1.0
    for end in range(len(generators) - 1, 0, -1):  # bubble sort, counting swaps
        for j in range(end):
            if generators[j] > generators[j + 1]:
                generators[j], generators[j + 1] = generators[j + 1], generators[j]
                sign = -sign
    result = []
    for g in generators:  # contract e_i e_i = squares[i]
        if result and result[-1] == g:
            result.pop()
            sign *= squares[g]
        else:
            result.append(g)
    return sign, tuple(result)


def cayley_table(p: int, q: int):
    """Return (sign, product), both (2^n, 2^n): e_i e_j = sign[i, j] e_product[i, j]."""
    squares = [1.0] * p + [-1.0] * q
    blades = 1 << (p + q)
    sign = torch.zeros(blades, blades)
    product = torch.zeros(blades, blades, dtype=torch.long)
    for i in range(blades):
        for j in range(blades):
            s, c = multiply_blades(blade_generators(i), blade_generators(j), squares)
            sign[i, j], product[i, j] = s, blade_index(c)
    return sign, product


def reference_product(x: torch.Tensor, y: torch.Tensor, p: int, q: int) -> torch.Tensor:
    """x * y for single multivectors (2^n coefficients each), term by term:
    out[C] += s x_A y_B for every pair with e_A e_B = s e_C. Slow; the definition the
    matrix path is tested against."""
    sign, product = cayley_table(p, q)
    out = torch.zeros(len(x), dtype=torch.result_type(x, y))
    for a in range(len(x)):
        for b in range(len(y)):
            out[product[a, b]] += sign[a, b] * x[a] * y[b]
    return out


class CliffordAlgebra(nn.Module):
    """Clifford algebra Cl(p, q) acting on embeddings of width ``dim`` (see the module docstring).

    Every operation acts on each of the C channels separately. Right multiplication by a
    multivector w is linear in x, so it is a matrix: (x * w) = x @ M(w), with
        M[i, k] = s w_j   for the blade j with e_i e_j = s e_k.
    The tables below hold, for each (i, k), that j (``partner``) and that s (``sign``), over the
    whole embedding, so one matrix product multiplies every channel for any signature.
    """

    def __init__(self, p: int, q: int = 0, dim: Optional[int] = None):
        super().__init__()
        if min(p, q) < 0:
            raise ValueError(f"Signature must be nonnegative, got ({p}, {q})")
        self.p, self.q = p, q
        n = p + q
        self.num_blades = 1 << n
        self.dim = self.num_blades if dim is None else dim
        if self.dim % self.num_blades:
            raise ValueError(
                f"Cl({p}, {q}) multivectors have {self.num_blades} coefficients, "
                f"which must divide the embedding dim {self.dim}"
            )
        self.channels = self.dim // self.num_blades
        B, C = self.num_blades, self.channels

        # One multivector: e_i e_j = s e_k, stored as partner[i, k] = j and sign[i, k] = s.
        cayley_sign, cayley_product = cayley_table(p, q)
        partner = torch.zeros(B, B, dtype=torch.long)
        sign = torch.zeros(B, B)
        for i in range(B):
            for j in range(B):
                k = cayley_product[i, j]
                partner[i, k], sign[i, k] = j, cayley_sign[i, j]

        # Whole embedding: blade b of channel c is at b * C + c; channel c only meets channel c.
        full_partner = torch.zeros(self.dim, self.dim, dtype=torch.long)
        full_sign = torch.zeros(self.dim, self.dim)
        for c in range(C):
            rows = torch.arange(B)[:, None] * C + c
            cols = torch.arange(B)[None, :] * C + c
            full_partner[rows, cols] = partner * C + c
            full_sign[rows, cols] = sign

        self.register_buffer("partner", full_partner, persistent=False)
        self.register_buffer("sign", full_sign, persistent=False)
        # One multivector's tables, for maps that mix channels.
        self.register_buffer("blade_partner", partner, persistent=False)
        self.register_buffer("blade_sign", sign, persistent=False)
        # e_b e_b = self_sign[b] (a scalar), repeated for every channel.
        self_sign = torch.diagonal(cayley_sign).repeat_interleave(C)
        self.register_buffer("self_sign", self_sign, persistent=False)

        # Grade k = number of generators. Reversing e_1..e_k takes k(k-1)/2 swaps: sign (-1)^that.
        self.num_grades = n + 1
        grade = torch.tensor([len(blade_generators(b)) for b in range(B)])
        reversion = torch.where((grade * (grade - 1) // 2) % 2 == 1, -1.0, 1.0)
        self.register_buffer("blade_grade", grade, persistent=False)
        # <x~ x>_0 per channel = sum_b reversion[b] * self_sign[b] * x_b^2.
        quad = self_sign.view(B, C) * reversion[:, None]
        self.register_buffer("quad_weight", quad, persistent=False)
        onehot = torch.zeros(self.num_grades, B)
        onehot[grade, torch.arange(B)] = 1.0
        self.register_buffer("grade_onehot", onehot, persistent=False)

    def split_blades(self, x: torch.Tensor) -> torch.Tensor:
        """View (..., dim) coefficients as (..., blades, channels); storage is blade-major."""
        return x.reshape(*x.shape[:-1], self.num_blades, self.channels)

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
        return gate.unsqueeze(-2).expand(*gate.shape[:-1], self.num_blades, self.channels).reshape(
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

    def clifford_multiplication_matrix(self, w: torch.Tensor) -> torch.Tensor:
        """Return M with x @ M = x * w per channel, for w of shape (..., dim):
        M[i, k] = s w_j where e_i e_j = s e_k. w multiplies from the right.
        """
        return self.sign * w[..., self.partner]

    def mixing_matrix(self, w: torch.Tensor) -> torch.Tensor:
        """Return M with x @ M = (sum_c x_c * w[o, c])_o, for w of shape (..., C, C, blades).

        Output channel o sums the products of every input channel c with its own multivector
        w[o, c]; clifford_multiplication_matrix is the case w[o, c] = 0 for o != c. Rows and columns are blade-major.
        """
        t = self.blade_sign * w[..., self.blade_partner]  # [..., o, c, i, k]: blade i of x to blade k
        n = t.dim()
        t = t.permute(*range(n - 4), n - 2, n - 3, n - 1, n - 4)  # [..., i, c, k, o]
        return t.reshape(*w.shape[:-3], self.dim, self.dim)

    def clifford_multiplication(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Compute x * w in every channel: (x * w)_k = sum_i x_i s w_j over e_i e_j = s e_k.

        One matrix multiplication handles every row of x (..., dim).
        """
        return x @ self.clifford_multiplication_matrix(w)

    def scalar_product(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return sum over channels of <x * y>_0 = sum_i (e_i e_i) x_i y_i.

        No reversion is applied to y; this bilinear form is not a Euclidean
        inner product and need not be positive for x == y.
        """
        return (x * y * self.self_sign).sum(dim=-1)

    def init_weight(self, *shape: int) -> torch.Tensor:
        """Sample coefficients from N(0, 1/blades); each output coefficient sums blades products."""
        return torch.randn(*shape, self.dim) * self.num_blades**-0.5

    def extra_repr(self) -> str:
        return f"Cl({self.p}, {self.q}), num_blades={self.num_blades}, channels={self.channels}"
