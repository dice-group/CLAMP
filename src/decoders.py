import torch


def _distmult_triple(h, r, t):
    return (h * r * t).sum(dim=-1)


def _distmult_query(h, r):
    return h * r



def _transe_triple(h, r, t, p: float = 1.0):
    return -((h + r) - t).norm(p=p, dim=-1)


def _transe_k_vs_all(h, r, entity_repr, p: float = 1.0):
    diff = (h + r).unsqueeze(1) - entity_repr.unsqueeze(0)
    return -diff.norm(p=p, dim=-1)



def _complex_split(x):
    d = x.shape[-1] // 2
    return x[..., :d], x[..., d:]


def _complex_ex_triple(h, r, t):
    h_re, h_im = _complex_split(h)
    r_re, r_im = _complex_split(r)
    t_re, t_im = _complex_split(t)
    a = h_re * r_re - h_im * r_im
    b = h_re * r_im + h_im * r_re
    return (a * t_re).sum(dim=-1) + (b * t_im).sum(dim=-1)


def _complex_ex_query(h, r):
    """The complex product h * r as [real | imaginary]; scores are query @ [e_re | e_im]^T."""
    h_re, h_im = _complex_split(h)
    r_re, r_im = _complex_split(r)
    return torch.cat([h_re * r_re - h_im * r_im, h_re * r_im + h_im * r_re], dim=-1)



def _rotate_phase(r):
    d = r.shape[-1] // 2
    phase = r[..., :d]
    return torch.cos(phase), torch.sin(phase)


def _rotate_triple(h, r, t):
    h_re, h_im = _complex_split(h)
    t_re, t_im = _complex_split(t)
    r_re, r_im = _rotate_phase(r)
    rot_re = h_re * r_re - h_im * r_im
    rot_im = h_re * r_im + h_im * r_re
    diff_re = rot_re - t_re
    diff_im = rot_im - t_im
    return -torch.sqrt(diff_re.pow(2).sum(-1) + diff_im.pow(2).sum(-1))


def _rotate_k_vs_all(h, r, entity_repr):
    h_re, h_im = _complex_split(h)
    r_re, r_im = _rotate_phase(r)
    e_re, e_im = _complex_split(entity_repr)
    rot_re = h_re * r_re - h_im * r_im
    rot_im = h_re * r_im + h_im * r_re
    diff_re = rot_re.unsqueeze(1) - e_re.unsqueeze(0)
    diff_im = rot_im.unsqueeze(1) - e_im.unsqueeze(0)
    return -torch.sqrt(diff_re.pow(2).sum(-1) + diff_im.pow(2).sum(-1))



def clifford_decoder(algebra):
    """Triple score and query for <h * r, t>: the Clifford product of h and r, dotted with t.

    Full multivector product in every channel of ``algebra`` (see src/models/clifford/clifford_algebra.py),
    then the Euclidean inner product over all coefficients. Cl(0, 0) is DistMult and, in the
    blade-major layout, Cl(0, 1) is ComplEx.
    """
    def query(h, r):
        return torch.einsum("bd,bdk->bk", h, algebra.clifford_multiplication_matrix(r))

    def triple(h, r, t):
        return (query(h, r) * t).sum(dim=-1)

    return triple, query


# Each decoder provides triple and tail-ranking scores; head queries use inverse relations.
DECODERS = {
    "TransE": (_transe_triple, _transe_k_vs_all),
    "RotatE": (_rotate_triple, _rotate_k_vs_all),
}

# Bilinear decoders score every tail as query(h, r) @ e: they provide triple scores and the query.
BILINEAR_DECODERS = {
    "DistMult": (_distmult_triple, _distmult_query),
    "ComplEx": (_complex_ex_triple, _complex_ex_query),
}

COMPLEX_DECODERS = frozenset({"ComplEx", "RotatE"})
