"""CompGCN encoder.

Reference: Vashishth, Sanyal, Nitin, Talukdar, *Composition-based
Multi-Relational Graph Convolutional Networks*, ICLR 2020.
https://arxiv.org/abs/1911.03082

Follows the official implementation's layer (``CompGCNCov`` in
``malllabiisc/CompGCN``, re-used by Zhang et al. 2022): each edge's message is
a composition ``phi(x_src, r)`` of the source entity with the edge's relation
embedding, transformed by a direction-specific weight (``W_in`` for original
edges, ``W_out`` for inverse ones); a self-loop message ``W_loop phi(x, r_loop)``
is added, the three are averaged, followed by bias, batch normalisation and
tanh. Relation embeddings are updated alongside, ``r' = W_rel r``, and the
decoder scores with the final layer's relations.

Its relation table is indexed by the propagation graph's edge types
(``r`` and ``r + num_relations`` for inverse edges), which is the reciprocal
decoder's relation table, so ``--reciprocal`` is required. Deviations from the
official code: edge normalisation is ``1/sqrt(deg_src deg_dst)`` over the whole
graph instead of per direction, and only the ``mult``/``sub`` compositions
are implemented (not ``corr``).
"""
import torch

from src.models.base_gnn import RelationalGNNEncoder


def _compose(x: torch.Tensor, r: torch.Tensor, opn: str) -> torch.Tensor:
    return x * r if opn == "mult" else x - r


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
        deg = torch.zeros(x.size(0), device=x.device).index_add_(
            0, dst, torch.ones_like(dst, dtype=x.dtype)).clamp(min=1)
        msg = msg * (deg[src] * deg[dst]).rsqrt().unsqueeze(1)
        agg = torch.zeros_like(x).index_add_(0, dst, msg)
        out = (self.drop(agg) + self.w_loop(_compose(x, self.loop_rel, self.opn))) / 3 + self.bias
        return torch.tanh(self.bn(out))


class CompGCN(RelationalGNNEncoder):
    """Composition-based multi-relational GCN knowledge graph encoder."""

    def __init__(self, args: dict):
        if not args.get("reciprocal", False):
            raise ValueError("CompGCN scores with its per-edge-type relation table and needs --reciprocal")
        super().__init__(args)
        self.name = "CompGCN"

    def _build_layers(self) -> torch.nn.ModuleList:
        opn = self.args.get("compgcn_opn", "mult")
        return torch.nn.ModuleList([
            CompGCNLayer(self.embedding_dim, self.num_relations, opn, self.gnn_dropout_rate)
            for _ in range(self.num_gnn_layers)
        ])

    def _relation_states(self, num_layers: int) -> torch.FloatTensor:
        rel = self.rel_emb.weight
        for layer in self.gnn_layers[:num_layers]:
            rel = layer.w_rel(rel)
        return rel

    def _apply_layer(self, layer_idx, layer, x, edge_index, edge_type):
        return layer(x, edge_index, edge_type, self._relation_states(layer_idx))

    def relation_repr(self) -> torch.FloatTensor:
        return self._relation_states(len(self.gnn_layers))
