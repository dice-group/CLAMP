r"""GATv2Conv layer documentation (verbatim excerpt).

The GATv2 operator from the `"How Attentive are Graph Attention
Networks?" <https://arxiv.org/abs/2105.14491>`_ paper, which fixes the
static attention problem of the standard
:class:`~torch_geometric.conv.GATConv` layer.
Since the linear layers in the standard GAT are applied right after each
other, the ranking of attended nodes is unconditioned on the query node.
In contrast, in :class:`GATv2`, every node can attend to any other node.

.. math::
    \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i) \cup \{ i \}}
    \alpha_{i,j}\mathbf{\Theta}_{t}\mathbf{x}_{j},

where the attention coefficients :math:`\alpha_{i,j}` are computed as

.. math::
    \alpha_{i,j} =
    \frac{
    \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(
    \mathbf{\Theta}_{s} \mathbf{x}_i + \mathbf{\Theta}_{t} \mathbf{x}_j
    \right)\right)}
    {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
    \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(
    \mathbf{\Theta}_{s} \mathbf{x}_i + \mathbf{\Theta}_{t} \mathbf{x}_k
    \right)\right)}.

If the graph has multi-dimensional edge features :math:`\mathbf{e}_{i,j}`,
the attention coefficients :math:`\alpha_{i,j}` are computed as

.. math::
    \alpha_{i,j} =
    \frac{
    \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(
    \mathbf{\Theta}_{s} \mathbf{x}_i
    + \mathbf{\Theta}_{t} \mathbf{x}_j
    + \mathbf{\Theta}_{e} \mathbf{e}_{i,j}
    \right)\right)}
    {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
    \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(
    \mathbf{\Theta}_{s} \mathbf{x}_i
    + \mathbf{\Theta}_{t} \mathbf{x}_k
    + \mathbf{\Theta}_{e} \mathbf{e}_{i,k}]
    \right)\right)}.
"""
# PyG source: torch_geometric/nn/conv/gatv2_conv.py
# Copyright (c) 2023 PyG Team. MIT license: PYG_LICENSE.

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from src.models.base_gnn import RelationalGNNEncoder


class GATv2(RelationalGNNEncoder):
    def __init__(self, args: dict):
        super().__init__(args)
        self.name = "GATv2"
        self.gnn_relation_embeddings = torch.nn.Embedding(
            self.num_relations_gnn, self.embedding_dim
        )
        self.param_init(self.gnn_relation_embeddings.weight.data)

    def _build_layers(self) -> torch.nn.ModuleList:
        heads = int(self.args.get("gnn_heads", 4))
        assert self.embedding_dim % heads == 0, (
            f"embedding_dim ({self.embedding_dim}) must be divisible by "
            f"gnn_heads ({heads}) "
            "so concatenated per-head outputs stay at embedding_dim."
        )
        per_head_dim = self.embedding_dim // heads
        layers = torch.nn.ModuleList()
        for i in range(self.num_gnn_layers):
            is_last_layer = i == self.num_gnn_layers - 1
            layers.append(
                GATv2Conv(
                    in_channels=self.embedding_dim,
                    out_channels=self.embedding_dim if is_last_layer else per_head_dim,
                    heads=heads,
                    concat=not is_last_layer,
                    edge_dim=self.embedding_dim,
                    dropout=self.gnn_dropout_rate,
                )
            )
        return layers

    def _apply_layer(
        self,
        i: int,
        conv: torch.nn.Module,
        x: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_type: torch.LongTensor,
    ) -> torch.FloatTensor:
        edge_attr = self.gnn_relation_embeddings(edge_type)
        x = conv(x, edge_index, edge_attr=edge_attr)
        if i < self.num_gnn_layers - 1:
            x = F.elu(x)
        return x
