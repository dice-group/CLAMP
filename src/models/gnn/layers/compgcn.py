"""Local CompGCN implementation with multiplication, subtraction, complex
multiplication (ComplEx) or rotation (RotatE) composition.

Vashishth et al., Composition-based Multi-Relational Graph Convolutional
Networks, ICLR 2020. https://arxiv.org/abs/1911.03082
Degree normalization here uses the whole graph rather than each direction.
"""

import torch

from src.decoders import _complex_split, _rotate_phase


def _complex_mult(x: torch.Tensor, r_re: torch.Tensor, r_im: torch.Tensor) -> torch.Tensor:
    x_re, x_im = _complex_split(x)
    return torch.cat([x_re * r_re - x_im * r_im, x_re * r_im + x_im * r_re], dim=-1)


def _compose(x: torch.Tensor, r: torch.Tensor, opn: str) -> torch.Tensor:
    if opn == "mult":
        return x * r
    if opn == "sub":
        return x - r
    if opn == "cmult":
        return _complex_mult(x, *_complex_split(r))
    if opn == "rotate":
        return _complex_mult(x, *_rotate_phase(r))
    raise ValueError(f"Unknown compgcn_opn '{opn}'")


class CompGCNLayer(torch.nn.Module):
    def __init__(self, dim: int, num_relations: int, opn: str, dropout: float):
        super().__init__()
        self.num_relations = num_relations
        self.opn = opn
        self.w_in = torch.nn.Linear(dim, dim, bias=False)
        self.w_out = torch.nn.Linear(dim, dim, bias=False)
        self.w_loop = torch.nn.Linear(dim, dim, bias=False)
        self.w_rel = torch.nn.Linear(dim, dim, bias=False)
        self.loop_rel = torch.nn.Parameter(torch.empty(1, dim))
        torch.nn.init.xavier_normal_(self.loop_rel)
        self.bias = torch.nn.Parameter(torch.zeros(dim))
        self.bn = torch.nn.BatchNorm1d(dim)
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_type, rel):
        src, dst = edge_index
        msg = _compose(x[src], rel[edge_type], self.opn)
        is_inverse = (edge_type >= self.num_relations).unsqueeze(1)
        msg = torch.where(is_inverse, self.w_out(msg), self.w_in(msg))
        deg = torch.zeros(x.size(0), device=x.device)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        deg = deg.clamp(min=1)
        msg = msg * (deg[src] * deg[dst]).rsqrt().unsqueeze(1)
        agg = torch.zeros_like(x).index_add_(0, dst, msg)
        loop = self.w_loop(_compose(x, self.loop_rel, self.opn))
        out = (self.drop(agg) + loop) / 3 + self.bias
        return torch.tanh(self.bn(out))
