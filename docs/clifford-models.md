# Clifford-algebra encoders

Three encoders — `CliffordRGCN`, `CliffordGATv2`, `CliffordGraphSAGE`.
`CliffordRGCN` and `CliffordGraphSAGE` keep the message-passing structure of
`RGCN` / `GraphSAGE` and replace only their learned linear maps with geometric
products in a Clifford algebra. `CliffordGATv2` is a redesign rather than a
substitution: the relation multiplies the neighbour state, so it enters the
values as well as the attention; the self term sits outside the attention
instead of a self-loop inside it; and heads are averaged on every layer. A
gain of `CliffordGATv2` over `GATv2` therefore cannot be attributed to the
Clifford product alone without a matched dense control.

The layers are not equivariant under the Clifford group: coefficient-wise
activations and biases depend on the basis. The geometric product serves as a
structured linear map with `dim` parameters per relation.

Results are in [`experiments.md`](experiments.md#clifford-encoders).

| File | Contents |
|---|---|
| `src/clifford_algebra.py` | `CliffordAlgebra`: the algebra Cl(n, 0) and its three operations |
| `src/models/clifford_layers.py` | The layers: `CliffordLinear`, `CliffordRGCNConv`, `CliffordConv`, `CliffordRelationalSAGELayer`, `CliffordGATv2Conv` |
| `src/models/clifford.py` | The three encoders |
| `tests/test_clifford_algebra.py` | Correctness gates for the algebra and the relation loop |

## The idea

R-GCN learns a weight **matrix** `W_r` per relation and sends `x_j @ W_r` along
each edge of relation `r`. These models learn a **multivector** `w_r` per
relation and send the geometric product `x_j * w_r`.

This is the same move RotatE and QuatE make — a relation acting as a
multiplicative operator in a normed algebra (complex, quaternion) rather than a
free matrix — generalised to an arbitrary Clifford algebra, and applied inside
the message-passing layer instead of in the decoder.

The consequence that shows up in the results is parametric: `W_r` costs
`dim^2` parameters per relation, `w_r` costs `dim`. With 84 propagation
relation types on CoDEx-S, that is where the 85-97% encoder-parameter
reductions come from.

## The algebra

`CliffordAlgebra(n)` is the Euclidean Clifford algebra Cl(n, 0): `n`
orthonormal generators `e_1 .. e_n`, each squaring to `+1`, generating `2**n`
basis blades. An embedding is one complete multivector, so

> **`embedding_dim` must be a power of two**, with `n = log2(embedding_dim)`.

`embedding_dim=64` means Cl(6, 0) and all 64 blades. A non-power-of-two
dimension raises `ValueError` at construction.

**Blade encoding.** Blade `i` is the bitmask of which generators it contains,
so the product of blades `a` and `b` is always `± e_{a XOR b}`: XOR clears any
repeated generator, which squares to `+1`. The sign is the parity of the
transpositions needed to sort the concatenated generator lists, the standard
canonical-reordering rule. This makes the whole multiplication table
constructible from bit operations; `canonical_reordering_sign` is the scalar
reference version, and the class builds the same table vectorised.

Tables are registered as **non-persistent** buffers: they travel with
`.to(device)` but stay out of checkpoints, since they are constants derived
from `n`. One algebra instance is shared by every layer of a model.

**Three operations**, which is the entire surface the layers use:

- `transform(x, w)` — geometric product of a batch of multivectors with one
  multivector. Computed as `x @ left_matrix(w)`, materialising the `(dim, dim)`
  matrix of the linear map `x -> x * w` once and pushing a whole relation's
  edges through a single GEMM. The naive alternative — one multivector per
  edge — would build an `(num_edges, dim, dim)` intermediate.
- `scalar_product(x, y)` — the grade-0 part of `x * y`. Only blade `i` of `x`
  against blade `i` of `y` can reach the scalar blade, so this is a signed dot
  product costing `O(dim)`, not a full `O(dim^2)` product.
- `init_weight(*shape)` — samples multivectors at `std = dim ** -0.5`, since
  every output blade sums `dim` products.

Cost: `transform` is `O(dim^2)` per row, against `O(dim^2)` for the matrix
multiply it replaces — the saving is in parameters, not in arithmetic.

## Layer by layer

`CliffordLinear`, `CliffordRGCNConv` and the SAGE layers keep their original's
aggregation and structure exactly; `CliffordGATv2Conv` does not (see above).

| Clifford layer | Replaces | What changes |
|---|---|---|
| `CliffordLinear` | `nn.Linear` as a root/self transform | `x * w + b` for a learned multivector `w` |
| `CliffordRGCNConv` | `RGCNConv` | Per-relation `w_r` instead of `W_r`; messages `x_j * w_r`, averaged per (target, relation) and summed over relations (`RGCNConv`'s `aggr='mean'`, the paper's `1/c_{i,r}`); shared `CliffordLinear` self-loop |
| `CliffordConv` | single-relation `SAGEConv` | Messages `x_j * w`, aggregator configurable via `--sage_aggregator` |
| `CliffordRelationalSAGELayer` | `RelationalSAGELayer` | One `CliffordConv` per relation, summed, plus one shared root transform |
| `CliffordGATv2Conv` | `GATv2Conv` | Query/key/value are geometric products; the attention logit uses `scalar_product` in place of the dot product with `a` |

**`relational_transform`** is the shared helper: it loops over relations and
transforms each relation's rows in one call, mirroring how `RGCNConv` applies
its per-relation matrices. This is what keeps the `(num_edges, dim, dim)`
intermediate from ever existing.

**Attention.** GATv2's logit `a^T LeakyReLU(W_l x_i + W_r x_j)` becomes
`<a * LeakyReLU(x_i * w_q + (x_j * w_rel) * w_k)>_0`. Relation identity enters
by multiplying the source state by the edge's relation multivector first, and
that relation-transformed state feeds both the logit and the message — the
Clifford counterpart of how the non-Clifford model injects relations as
`edge_attr`.

## Constraints and deliberate deviations

These are consequences of the algebra, not oversights, and each is noted in the
relevant docstring.

1. **`embedding_dim` must be a power of two.** An embedding is one complete
   multivector and Cl(n, 0) has `2**n` blades.
2. **`--num_bases` is ignored by `CliffordRGCN`.** A multivector basis,
   `w_r = sum_b c_rb v_b`, would be possible, but each `w_r` already has only
   `dim` parameters, so it is not implemented.
3. **Attention heads are averaged at every layer, not concatenated on hidden
   layers** as in the GAT/GATv2 papers. Concatenation changes the width, but a
   multivector's width is fixed by the algebra, so there is no
   `embedding_dim // heads` per-head representation to concatenate. This is a
   real deviation from the paper and applies to `CliffordGATv2` only.
4. **`--gnn_heads` need not divide `embedding_dim`**, unlike the non-Clifford
   `GATv2`, because heads are not slices of the representation.

## Verification

`tests/test_clifford_algebra.py` checks the geometric product against
independent references rather than against itself:

- the vectorised sign table equals the scalar `canonical_reordering_sign` rule
  for every blade pair;
- hand-computed Cl(3, 0) products hold: `e1 e2 = e12`, `e2 e1 = -e12`,
  `e1 e1 = 1`, `e12 e12 = -1`;
- the product is associative and the scalar blade is its unit, for n = 1..4;
- `scalar_product(x, y)` equals blade 0 of `transform(x, y)`;
- `relational_transform` equals a naive per-edge product, including when a
  relation has no edges.

`tests/test_models.py` covers the encoders alongside the non-Clifford ones:
interface and output shapes, gradient flow, and `test_every_parameter_is_trained`,
which asserts that **every** registered parameter receives a gradient. That last
one exists because an earlier draft stored a per-relation weight as
`nn.Parameter(...).squeeze(0)` — which returns a plain `Tensor`, so the weight
was never registered and never trained, while the weaker "some parameter got a
gradient" check still passed.

## Open questions

- Hyperparameters are borrowed from the non-Clifford models; the Clifford
  encoders have never been tuned in their own right.
- Only CoDEx-S has been run. UMLS and WN18RR have not.
- Only the Euclidean signature Cl(n, 0) is implemented. A chunked Cl(p, q)
  variant in the style of Keci (Demir et al., ECML 2023) — splitting the
  embedding into `p + q + 1` chunks, which lifts the power-of-two constraint
  and makes the product `O(dim)` — was prototyped and removed; it is worth
  revisiting if the power-of-two restriction or the `O(dim^2)` product becomes
  a practical limit.
- No like-for-like runtime comparison against the non-Clifford encoders.
