r"""RGCNConv layer documentation (verbatim excerpt).

The relational graph convolutional operator from the `"Modeling
Relational Data with Graph Convolutional Networks"
<https://arxiv.org/abs/1703.06103>`_ paper.

.. math::
    \mathbf{x}^{\prime}_i = \mathbf{\Theta}_{\textrm{root}} \cdot
    \mathbf{x}_i + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
    \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j,

where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
stores a relation identifier
:math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}` for each edge.
"""
# PyG source: torch_geometric/nn/conv/rgcn_conv.py
# Copyright (c) 2023 PyG Team. MIT license: PYG_LICENSE.

import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

from src.models.base_gnn import RelationalGNNEncoder


class RGCN(RelationalGNNEncoder):
    def __init__(self, args: dict):
        super().__init__(args)
        self.name = "RGCN"

    def _build_layers(self) -> torch.nn.ModuleList:
        num_bases = self.args.get("num_bases", None)
        return torch.nn.ModuleList(
            [
                RGCNConv(
                    in_channels=self.embedding_dim,
                    out_channels=self.embedding_dim,
                    num_relations=self.num_relations_gnn,
                    num_bases=num_bases,
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
        return x
