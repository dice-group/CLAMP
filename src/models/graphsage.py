r"""SAGEConv layer documentation (verbatim excerpt).

The GraphSAGE operator from the `"Inductive Representation Learning on
Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper.

.. math::
    \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot
    \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j
"""
# PyG source: torch_geometric/nn/conv/sage_conv.py
# Copyright (c) 2023 PyG Team. MIT license: PYG_LICENSE.

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from src.models.base_gnn import RelationalGNNEncoder


class RelationalSAGELayer(torch.nn.Module):
    """Sum per-relation SAGEConv outputs and add one shared root transform.

    root_weight=False prevents adding the root once per relation.
    """
    def __init__(
        self, in_channels: int, out_channels: int,
        num_relations: int, aggr: str = "mean",
    ):
        super().__init__()
        self.num_relations = num_relations
        self.convs = torch.nn.ModuleList(
            [
                SAGEConv(
                    in_channels, out_channels, aggr=aggr, root_weight=False, bias=False
                )
                for _ in range(num_relations)
            ]
        )
        self.root_lin = torch.nn.Linear(in_channels, out_channels)

    def forward(
        self,
        x: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_type: torch.LongTensor,
    ) -> torch.FloatTensor:
        out = self.root_lin(x)
        for relation_id, conv in enumerate(self.convs):
            relation_mask = edge_type == relation_id
            if not torch.any(relation_mask):
                continue
            out = out + conv(x, edge_index[:, relation_mask])
        return out


class GraphSAGE(RelationalGNNEncoder):
    def __init__(self, args: dict):
        super().__init__(args)
        self.name = "GraphSAGE"

    def _build_layers(self) -> torch.nn.ModuleList:
        aggr = self.args.get("sage_aggregator", "mean")
        return torch.nn.ModuleList(
            [
                RelationalSAGELayer(
                    self.embedding_dim,
                    self.embedding_dim,
                    self.num_relations_gnn,
                    aggr,
                )
                for _ in range(self.num_gnn_layers)
            ]
        )

    def _apply_layer(
        self,
        i: int,
        conv: torch.nn.Module,
        x: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_type: torch.LongTensor,
    ) -> torch.FloatTensor:
        x = conv(x, edge_index, edge_type)
        if i < self.num_gnn_layers - 1:
            x = F.relu(x)
        x = F.normalize(x, p=2, dim=1)
        return x
