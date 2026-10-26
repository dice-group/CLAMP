"""Relational GraphSAGE encoder (Hamilton et al., 2017, https://arxiv.org/abs/1706.02216)."""

import torch
import torch.nn.functional as F

from src.models.gnn.base import RelationalGNN
from src.models.gnn.layers.graphsage import RelationalSAGELayer


class GraphSAGE(RelationalGNN):
    """Stacked relational SAGE layers.

    Hidden layers: ReLU, unit L2 norm, dropout. Last layer: ``--sage_final_norm``
    (unit L2 norm, none, or unit L2 norm times a learned scalar).
    """
    name = "GraphSAGE"

    def __init__(self, args: dict):
        super().__init__(args)
        self.gnn_layers = torch.nn.ModuleList(
            [
                RelationalSAGELayer(
                    self.embedding_dim, self.embedding_dim, self.num_relations, "mean"
                )
                for _ in range(self.num_gnn_layers)
            ]
        )
        self.final_norm = args.get("sage_final_norm", "l2")
        if self.final_norm not in ("l2", "none", "l2_scale"):
            raise ValueError(f"Unknown sage_final_norm '{self.final_norm}'")
        if self.final_norm == "l2_scale":
            self.output_scale = torch.nn.Parameter(torch.ones(()))

    def encode(self, edge_index, edge_type) -> torch.FloatTensor:
        x = self.node_emb.weight
        for i, conv in enumerate(self.gnn_layers):
            x = conv(x, edge_index, edge_type)
            if i < self.num_gnn_layers - 1:
                x = self.gnn_dropout(F.normalize(F.relu(x), p=2, dim=1))
        if self.final_norm == "none" or not self.num_gnn_layers:
            return x
        x = F.normalize(x, p=2, dim=1)
        return x * self.output_scale if self.final_norm == "l2_scale" else x
