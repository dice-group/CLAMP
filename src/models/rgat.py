"""Relation-aware GATv2 encoder with a DistMult decoder.

Reference: Brody, Alon, Yahav, *How Attentive are Graph Attention
Networks?*, ICLR 2022 (GATv2). https://arxiv.org/abs/2105.14491

GATv2 itself is defined over homogeneous graphs and has no notion of edge
labels, so it cannot be dropped onto a knowledge graph unmodified. We keep
`torch_geometric`'s reference ``GATv2Conv`` layer exactly as published and
make it relation-aware the way the layer itself was designed to consume
edge information: relation embeddings are passed in as ``edge_attr``
(``GATv2Conv``'s ``edge_dim`` argument), so relation identity enters both the
attention coefficients and the messages, without inventing a new attention
formula. This differs from Relational GAT (Busbridge et al. 2019), which
learns per-relation attention parameters instead of per-relation edge
features; that is a legitimate alternative we do not implement here.

As in the original GAT/GATv2 papers, attention heads are concatenated on
hidden layers and averaged on the final layer.
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from src.models.base_gnn import RelationalGNNEncoder


class GATv2(RelationalGNNEncoder):
    """GATv2 knowledge graph encoder, relation-aware via edge features."""

    def __init__(self, args: dict):
        super().__init__(args)
        self.name = "GATv2"
        self.gnn_relation_embeddings = torch.nn.Embedding(self.num_relations_gnn, self.embedding_dim)
        self.param_init(self.gnn_relation_embeddings.weight.data)

    def _build_layers(self) -> torch.nn.ModuleList:
        heads = int(self.args.get("gnn_heads", 4))
        assert self.embedding_dim % heads == 0, (
            f"embedding_dim ({self.embedding_dim}) must be divisible by gnn_heads ({heads}) "
            "so concatenated per-head outputs stay at embedding_dim."
        )
        per_head_dim = self.embedding_dim // heads
        layers = torch.nn.ModuleList()
        for i in range(self.num_gnn_layers):
            is_last_layer = i == self.num_gnn_layers - 1
            if is_last_layer:
                # Final layer: average heads instead of concatenating (original GAT, Section 3.3).
                layers.append(GATv2Conv(
                    in_channels=self.embedding_dim,
                    out_channels=self.embedding_dim,
                    heads=heads,
                    concat=False,
                    edge_dim=self.embedding_dim,
                    dropout=self.gnn_dropout_rate,
                ))
            else:
                layers.append(GATv2Conv(
                    in_channels=self.embedding_dim,
                    out_channels=per_head_dim,
                    heads=heads,
                    concat=True,
                    edge_dim=self.embedding_dim,
                    dropout=self.gnn_dropout_rate,
                ))
        return layers

    def _apply_layer(self, layer_idx: int, layer: torch.nn.Module, x: torch.FloatTensor,
                      edge_index: torch.LongTensor, edge_type: torch.LongTensor) -> torch.FloatTensor:
        edge_attr = self.gnn_relation_embeddings(edge_type)
        x = layer(x, edge_index, edge_attr=edge_attr)
        if layer_idx < self.num_gnn_layers - 1:
            x = F.elu(x)
        return x
