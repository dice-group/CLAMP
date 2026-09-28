import torch


def _distmult_triple(h, r, t):
    return (h * r * t).sum(dim=-1)


def _distmult_k_vs_all(h, r, entity_repr):
    return (h * r) @ entity_repr.transpose(0, 1)


def _distmult_k_vs_all_head(r, t, entity_repr):
    return _distmult_k_vs_all(t, r, entity_repr)


def _transe_triple(h, r, t, p: float = 1.0):
    return -((h + r) - t).norm(p=p, dim=-1)


def _transe_k_vs_all(h, r, entity_repr, p: float = 1.0):
    diff = (h + r).unsqueeze(1) - entity_repr.unsqueeze(0)
    return -diff.norm(p=p, dim=-1)


def _transe_k_vs_all_head(r, t, entity_repr, p: float = 1.0):
    target = t - r
    diff = entity_repr.unsqueeze(0) - target.unsqueeze(1)
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


def _complex_ex_k_vs_all(h, r, entity_repr):
    h_re, h_im = _complex_split(h)
    r_re, r_im = _complex_split(r)
    e_re, e_im = _complex_split(entity_repr)
    a = h_re * r_re - h_im * r_im
    b = h_re * r_im + h_im * r_re
    return a @ e_re.transpose(0, 1) + b @ e_im.transpose(0, 1)


def _complex_ex_k_vs_all_head(r, t, entity_repr):
    r_re, r_im = _complex_split(r)
    t_re, t_im = _complex_split(t)
    e_re, e_im = _complex_split(entity_repr)
    c = r_re * t_re + r_im * t_im
    d = r_re * t_im - r_im * t_re
    return c @ e_re.transpose(0, 1) + d @ e_im.transpose(0, 1)


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


def _rotate_k_vs_all_head(r, t, entity_repr):
    t_re, t_im = _complex_split(t)
    r_re, r_im = _rotate_phase(r)
    e_re, e_im = _complex_split(entity_repr)
    rot_re = t_re * r_re + t_im * r_im
    rot_im = t_im * r_re - t_re * r_im
    diff_re = e_re.unsqueeze(0) - rot_re.unsqueeze(1)
    diff_im = e_im.unsqueeze(0) - rot_im.unsqueeze(1)
    return -torch.sqrt(diff_re.pow(2).sum(-1) + diff_im.pow(2).sum(-1))


# Each decoder provides triple, tail-ranking, and head-ranking scores.
DECODERS = {
    "DistMult": (_distmult_triple, _distmult_k_vs_all, _distmult_k_vs_all_head),
    "TransE": (_transe_triple, _transe_k_vs_all, _transe_k_vs_all_head),
    "ComplEx": (_complex_ex_triple, _complex_ex_k_vs_all, _complex_ex_k_vs_all_head),
    "RotatE": (_rotate_triple, _rotate_k_vs_all, _rotate_k_vs_all_head),
}

COMPLEX_DECODERS = frozenset({"ComplEx", "RotatE"})
