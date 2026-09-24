"""Shallow KGE baseline: the embedding tables and a decoder, with no message passing.

This is the control the three GNN encoders are measured against. It reuses the
whole pipeline unchanged - the same embedding tables, the same KvsAll training
loop, the same pooled head+tail evaluator, and the same scoring functions in
:mod:`src.decoders` - and removes exactly one thing: the encoder. Entity
representations are the raw ``node_emb`` table rather than a refined version of
it, so whatever separates this from ``RGCN``/``GATv2``/``GraphSAGE`` under an
otherwise identical configuration is what message passing contributed.

``--decoder`` picks which classical model this *is*: ``DistMult`` gives a
DistMult baseline, ``ComplEx`` a ComplEx baseline, and likewise for ``TransE``
and ``RotatE``. These are the project's own implementations of those scoring
functions (see :mod:`src.decoders` for where they simplify the originals - in
particular ComplEx/RotatE split one real table in half rather than keeping a
separate complex one), not the published reference implementations, so they are
an internal control rather than a reproduction of those papers' numbers.

``--num_gnn_layers``, ``--num_bases``, ``--gnn_heads`` and the other
encoder-only flags are accepted and ignored.
"""
from typing import Optional, Tuple

import torch

from src.models.base_gnn import RelationalGNNEncoder


class KGE(RelationalGNNEncoder):
    """Embedding-only knowledge graph model; scoring function chosen by ``--decoder``."""

    name = "KGE"

    def _build_layers(self) -> torch.nn.ModuleList:
        # No layers: RelationalGNNEncoder.encode() then returns node_emb.weight
        # untouched, which is exactly the shallow-embedding baseline.
        return torch.nn.ModuleList()

    @property
    def graph(self) -> Tuple[Optional[torch.LongTensor], Optional[torch.LongTensor]]:
        """No propagation graph: nothing here consumes one."""
        return None, None

    def _graph_excluding_batch(self, idx_h: torch.LongTensor,
                               idx_r: torch.LongTensor) -> Tuple[None, None]:
        """No-op: with no message passing there is no edge to leak an answer along."""
        return None, None


class LTE(KGE):
    """KGE with linearly transformed entity embeddings (LTE-KGE).

    Reference: Zhang, Wang, Ye, Wu, *Rethinking Graph Convolutional Networks in
    Knowledge Graph Completion*, WWW 2022. https://arxiv.org/abs/2202.05679

    Every entity embedding passes through one shared bias-free linear map,
    batch normalisation and dropout (``--gnn_dropout_rate``) before scoring;
    relations are scored raw. This is the paper's DistMult/ConvE setting
    (``x_ops="p.b.d"``), which it shows matches CompGCN with no graph at all.
    The paper's per-entity score bias is not included.
    """

    name = "LTE"

    def __init__(self, args: dict):
        super().__init__(args)
        self.lte_bn = torch.nn.BatchNorm1d(self.embedding_dim)

    def _build_layers(self) -> torch.nn.ModuleList:
        return torch.nn.ModuleList([torch.nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)])

    def _apply_layer(self, layer_idx, layer, x, edge_index, edge_type):
        return self.gnn_dropout(self.lte_bn(layer(x)))
