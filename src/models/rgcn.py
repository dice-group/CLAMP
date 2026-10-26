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
from torch_geometric.nn import RGCNConv

from src.models.clifford_layers import SPLIT_ACTIVATIONS, residual_mode
from src.models.gnn import RelationalGNN


class RGCN(RelationalGNN):
    """Stacked RGCNConv layers with ``--gnn_activation`` and dropout between layers.

    ``--gnn_residual`` adds each layer's input to its output, either inside the activation
    (``post``) or around it (``pre``).
    """
    name = "RGCN"

    def __init__(self, args: dict):
        super().__init__(args)
        self.activation = args.get("gnn_activation", "relu")
        self.gnn_residual = residual_mode(args)
        self.gnn_layers = torch.nn.ModuleList(
            [
                RGCNConv(
                    in_channels=self.embedding_dim,
                    out_channels=self.embedding_dim,
                    num_relations=self.num_relations,
                    num_bases=args.get("num_bases", None),
                )
                for _ in range(self.num_gnn_layers)
            ]
        )

    def layer(self, i, x, edge_index, edge_type) -> torch.FloatTensor:
        """Layer i with its residual path and the hidden activation.

        ``post`` computes f(F(x) + x), so the activation also clips the identity term.
        ``pre`` computes x + f(F(x)), leaving the identity path unclipped.
        """
        out = self.gnn_layers[i](x, edge_index, edge_type)
        activation = SPLIT_ACTIVATIONS[self.activation]
        hidden = i < self.num_gnn_layers - 1
        if self.gnn_residual == "post":
            out = out + x
        if hidden and activation is not None:
            out = activation(out)
        if self.gnn_residual == "pre":
            out = out + x
        return out

    def encode(self, edge_index, edge_type) -> torch.FloatTensor:
        x = self.node_emb.weight
        for i in range(self.num_gnn_layers):
            x = self.layer(i, x, edge_index, edge_type)
            if i < self.num_gnn_layers - 1:
                x = self.gnn_dropout(x)
        return x
