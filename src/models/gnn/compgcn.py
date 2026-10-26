"""CompGCN encoder (Vashishth et al., ICLR 2020, https://arxiv.org/abs/1911.03082)."""

import torch

from src.models.gnn.base import RelationalGNN
from src.models.gnn.layers.compgcn import CompGCNLayer


class CompGCN(RelationalGNN):
    """Stacked CompGCN layers; relation states are updated by each layer's w_rel."""
    name = "CompGCN"

    def __init__(self, args: dict):
        super().__init__(args)
        opn = args.get("compgcn_opn", "mult")
        if opn in ("cmult", "rotate") and self.embedding_dim % 2:
            raise ValueError(f"compgcn_opn '{opn}' requires an even embedding_dim")
        self.gnn_layers = torch.nn.ModuleList(
            [
                CompGCNLayer(
                    self.embedding_dim, self.num_direct_relations, opn,
                    self.gnn_dropout_rate,
                )
                for _ in range(self.num_gnn_layers)
            ]
        )

    def relation_states(self, num_layers: int) -> torch.FloatTensor:
        """Relation embeddings after the w_rel of the first num_layers layers."""
        rel = self.rel_emb.weight
        for layer in self.gnn_layers[:num_layers]:
            rel = layer.w_rel(rel)
        return rel

    def encode(self, edge_index, edge_type) -> torch.FloatTensor:
        x = self.node_emb.weight
        for i, layer in enumerate(self.gnn_layers):
            x = layer(x, edge_index, edge_type, self.relation_states(i))
            if i < self.num_gnn_layers - 1:
                x = self.gnn_dropout(x)
        return x

    def relation_repr(self) -> torch.FloatTensor:
        return self.relation_states(self.num_gnn_layers)
