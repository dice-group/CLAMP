"""Relational GraphSAGE layer (Hamilton et al., 2017, https://arxiv.org/abs/1706.02216).

PyG's SAGEConv computes x'_i = W_1 x_i + W_2 mean_{j in N(i)} x_j; this layer keeps one
shared root map W_1 and gives every relation its own neighbour matrix W_r.
"""

import math

import torch
from torch_geometric.utils import scatter


class RelationalSAGELayer(torch.nn.Module):
    """x'_i = W_root x_i + b + sum_r W_r AGG_{j in N_r(i)} x_j, one matrix per relation.

    Each W_r is initialized like a bias-free SAGEConv neighbour map, U(-1/sqrt(in), 1/sqrt(in)).
    """
    def __init__(
        self, in_channels: int, out_channels: int,
        num_relations: int, aggr: str = "mean",
    ):
        super().__init__()
        self.num_relations = num_relations
        self.aggr = aggr
        bound = 1 / math.sqrt(in_channels)
        self.rel_weight = torch.nn.Parameter(
            torch.empty(num_relations, in_channels, out_channels).uniform_(-bound, bound)
        )
        self.root_lin = torch.nn.Linear(in_channels, out_channels)

    def forward(
        self,
        x: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_type: torch.LongTensor,
    ) -> torch.FloatTensor:
        """Neighbours are pooled per (target, relation) group, then each relation's groups,
        contiguous because unique sorts the keys, go through one matmul with W_r.
        """
        src, dst = edge_index
        num_nodes = x.size(0)
        groups, group_of_edge = torch.unique(edge_type * num_nodes + dst, return_inverse=True)
        pooled = scatter(x[src], group_of_edge, dim=0, dim_size=len(groups), reduce=self.aggr)
        counts = torch.bincount(groups // num_nodes, minlength=self.num_relations).tolist()
        chunks = torch.split(pooled, counts)
        messages = torch.cat([chunk @ w for chunk, w in zip(chunks, self.rel_weight)])
        return self.root_lin(x).index_add(0, groups % num_nodes, messages)
