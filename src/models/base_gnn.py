"""Shared base class for GNN-encoder / DistMult-decoder KGE models.

Every concrete model in this package (RGCN, GATv2, GraphSAGE) follows the
same encoder-decoder recipe popularised by R-GCN for link prediction
(Schlichtkrull et al. 2018, Section 4): a relational graph encoder refines
the entity embedding table by message passing over the *whole* training
graph, and a fixed DistMult decoder scores triples from the refined
representations. Keeping the decoder identical across the three encoders
means differences in link-prediction quality can be attributed to the
encoder, not to incidental decoder choices.

Subclasses extend :class:`RelationalGNNEncoder`, which itself extends
``dicee.models.base_model.BaseKGE`` so every model instantiated here is a
first-class dicee KGE model (same ``forward``/``forward_triples``/
``forward_k_vs_all`` contract) and can be dropped into dice-embeddings by
copying the file into ``dicee/models/`` and registering it, exactly as
described in dice-embeddings' ``/add-model`` skill.
"""
from typing import List

import torch
from dicee.models.base_model import BaseKGE

from src.graph_utils import build_relational_edge_index


class RelationalGNNEncoder(BaseKGE):
    """``BaseKGE`` subclass that refines entity embeddings via message passing.

    Required ``args`` keys (beyond the usual dicee ones):

    * ``train_set_idx``: ``(N, 3)`` array of indexed training triples, used
      once at construction time to build the message-passing graph.

    Optional ``args`` keys:

    * ``num_gnn_layers`` (default 2)
    * ``gnn_dropout_rate`` (default 0.2) — dropout between GNN layers
    * ``gnn_add_inverse_edges`` (default True) — see
      :func:`src.graph_utils.build_relational_edge_index`
    """

    def __init__(self, args: dict):
        super().__init__(args)
        train_set_idx = args.get("train_set_idx")
        if train_set_idx is None:
            raise ValueError(
                f"{self.__class__.__name__} requires args['train_set_idx'] (the indexed "
                "training triples) to build its message-passing graph."
            )
        self.add_inverse_edges = bool(args.get("gnn_add_inverse_edges", True))
        edge_index, edge_type, num_relations_gnn = build_relational_edge_index(
            train_set_idx, self.num_relations, self.add_inverse_edges
        )
        self.num_relations_gnn = num_relations_gnn
        # persistent=False: the graph is rebuilt from train_set_idx whenever the
        # model is re-instantiated, so it need not be serialised in the checkpoint.
        self.register_buffer("edge_index", edge_index, persistent=False)
        self.register_buffer("edge_type", edge_type, persistent=False)

        self.num_gnn_layers = int(args.get("num_gnn_layers", 2))
        self.gnn_dropout_rate = float(args.get("gnn_dropout_rate", 0.2))
        self.gnn_dropout = torch.nn.Dropout(self.gnn_dropout_rate)
        self.gnn_layers = self._build_layers()

    def _build_layers(self) -> torch.nn.ModuleList:
        """Return the per-layer GNN modules. Must be implemented by subclasses."""
        raise NotImplementedError

    def _apply_layer(self, layer_idx: int, layer: torch.nn.Module, x: torch.FloatTensor) -> torch.FloatTensor:
        """Apply one GNN layer (incl. any layer-specific activation). Subclasses implement this."""
        raise NotImplementedError

    def encode(self) -> torch.FloatTensor:
        """Run message passing over the full training graph once.

        Returns:
            ``(num_entities, embedding_dim)`` refined entity representations.
        """
        x = self.entity_embeddings.weight
        num_layers = len(self.gnn_layers)
        for i, layer in enumerate(self.gnn_layers):
            x = self._apply_layer(i, layer, x)
            if i < num_layers - 1:
                x = self.gnn_dropout(x)
        return x

    def k_vs_all_score(self, emb_h: torch.FloatTensor, emb_r: torch.FloatTensor,
                        emb_E: torch.FloatTensor) -> torch.FloatTensor:
        """DistMult scoring: ``(h * r) @ E^T``, as in R-GCN's link-prediction decoder."""
        return torch.mm(self.hidden_dropout(self.hidden_normalizer(emb_h * emb_r)), emb_E.transpose(1, 0))

    def forward_triples(self, x: torch.LongTensor) -> torch.FloatTensor:
        entity_repr = self.encode()
        idx_h, idx_r, idx_t = x[:, 0], x[:, 1], x[:, 2]
        emb_h = self.normalize_head_entity_embeddings(entity_repr.index_select(0, idx_h))
        emb_t = self.normalize_tail_entity_embeddings(entity_repr.index_select(0, idx_t))
        emb_r = self.normalize_relation_embeddings(self.relation_embeddings(idx_r))
        return (self.hidden_dropout(self.hidden_normalizer(emb_h * emb_r)) * emb_t).sum(dim=1)

    def forward_k_vs_all(self, x: torch.LongTensor) -> torch.FloatTensor:
        entity_repr = self.encode()
        idx_h, idx_r = x[:, 0], x[:, 1]
        emb_h = self.normalize_head_entity_embeddings(entity_repr.index_select(0, idx_h))
        emb_r = self.normalize_relation_embeddings(self.relation_embeddings(idx_r))
        return self.k_vs_all_score(emb_h, emb_r, entity_repr)
