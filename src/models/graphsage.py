"""Relational GraphSAGE encoder with a DistMult decoder.

Reference: Hamilton, Ying, Leskovec, *Inductive Representation Learning on
Large Graphs*, NeurIPS 2017. https://arxiv.org/abs/1706.02216

GraphSAGE (Algorithm 2 in the paper) is also defined over homogeneous
graphs: it (a) samples a fixed-size neighbourhood, (b) aggregates neighbour
representations (we use the paper's mean aggregator, `torch_geometric`'s
``SAGEConv`` default), (c) concatenates/combines with the node's own
representation, and (d) L2-normalises the result. We keep (b)-(d) exactly
as published and extend (a)/(b) to relations the same way R-GCN extends
plain graph convolution: one ``SAGEConv`` per relation type, whose outputs
are summed, plus a single shared root/self transform (rather than one root
transform per relation, which would double-count the node's own state once
per relation). Neighbourhood sampling itself is not implemented — on graphs
the size of UMLS/Kinship/FB15k-237 full-neighbourhood aggregation is exactly
what GraphSAGE reduces to when the sample size is not smaller than the
actual degree, so this is a size restriction, not a different algorithm.
"""
from typing import List

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from src.models.base_gnn import RelationalGNNEncoder


class RelationalSAGELayer(torch.nn.Module):
    """One relational GraphSAGE layer: per-relation mean aggregation + shared root transform."""

    def __init__(self, in_dim: int, out_dim: int, num_relations: int, aggregator: str = "mean"):
        super().__init__()
        self.num_relations = num_relations
        # root_weight=False: the node's own contribution is added once below via root_lin,
        # not once per relation (SAGEConv would otherwise add it inside every relation's conv).
        self.convs = torch.nn.ModuleList([
            SAGEConv(in_dim, out_dim, aggr=aggregator, root_weight=False)
            for _ in range(num_relations)
        ])
        self.root_lin = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.FloatTensor, edge_index: torch.LongTensor,
                edge_type: torch.LongTensor) -> torch.FloatTensor:
        out = self.root_lin(x)
        for relation_id, conv in enumerate(self.convs):
            relation_mask = edge_type == relation_id
            if not torch.any(relation_mask):
                continue
            out = out + conv(x, edge_index[:, relation_mask])
        return out


class GraphSAGE(RelationalGNNEncoder):
    """GraphSAGE knowledge graph encoder, extended to relations à la R-GCN."""

    def __init__(self, args: dict):
        super().__init__(args)
        self.name = "GraphSAGE"

    def _build_layers(self) -> torch.nn.ModuleList:
        aggregator = self.args.get("sage_aggregator", "mean")
        return torch.nn.ModuleList([
            RelationalSAGELayer(self.embedding_dim, self.embedding_dim, self.num_relations_gnn, aggregator)
            for _ in range(self.num_gnn_layers)
        ])

    def _apply_layer(self, layer_idx: int, layer: torch.nn.Module, x: torch.FloatTensor) -> torch.FloatTensor:
        x = layer(x, self.edge_index, self.edge_type)
        if layer_idx < self.num_gnn_layers - 1:
            x = F.relu(x)
        # GraphSAGE Algorithm 2, line 7: unit-normalise node representations at every layer.
        x = F.normalize(x, p=2, dim=1)
        return x
