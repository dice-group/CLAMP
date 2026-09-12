"""R-GCN encoder with a DistMult decoder.

Reference: Schlichtkrull, Kipf, Bloem, van den Berg, Titov, Welling,
*Modeling Relational Data with Graph Convolutional Networks*, ESWC 2018.
https://arxiv.org/abs/1703.06103

This is deliberately close to the paper's link-prediction setup (Section
4.1-4.2): relation-specific linear transforms aggregated with mean
normalisation over each relation's neighbourhood (``RGCNConv``'s default
``aggr='mean'`` reproduces the paper's :math:`1/c_{i,r}` term), optional
basis-function decomposition (``--num_bases``) to regularise the
relation-specific weights, and a DistMult decoder on the encoder's output
entity representations.
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

from src.models.base_gnn import RelationalGNNEncoder


class RGCN(RelationalGNNEncoder):
    """Relational Graph Convolutional Network (R-GCN) knowledge graph encoder."""

    def __init__(self, args: dict):
        super().__init__(args)
        self.name = "RGCN"

    def _build_layers(self) -> torch.nn.ModuleList:
        num_bases = self.args.get("num_bases", None)
        return torch.nn.ModuleList([
            RGCNConv(
                in_channels=self.embedding_dim,
                out_channels=self.embedding_dim,
                num_relations=self.num_relations_gnn,
                num_bases=num_bases,
            )
            for _ in range(self.num_gnn_layers)
        ])

    def _apply_layer(self, layer_idx: int, layer: torch.nn.Module, x: torch.FloatTensor) -> torch.FloatTensor:
        x = layer(x, self.edge_index, self.edge_type)
        if layer_idx < self.num_gnn_layers - 1:
            x = F.relu(x)
        return x
