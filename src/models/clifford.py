"""Encoders using the local Clifford layers in clifford_layers.py.

Each embedding is a full Cl(n, 0) multivector with 2**n coefficients.
One algebra instance supplies the fixed product tables for all model layers.
Architecture references and layer equations are attached to the layer classes.
"""

import torch
import torch.nn.functional as F

from src.clifford_algebra import CliffordAlgebra
from src.models.base_gnn import RelationalGNNEncoder
from src.models.clifford_layers import (
    CliffordGATv2Conv,
    CliffordRelationalSAGELayer,
    CliffordRGCNConv,
)


class CliffordRGCN(RelationalGNNEncoder):
    """Stack CliffordRGCNConv layers with ReLU between layers.

    Each relation uses a full multivector; num_bases is unused.
    """
    name = "CliffordRGCN"

    def _build_layers(self) -> torch.nn.ModuleList:
        self.clifford_algebra = CliffordAlgebra.for_embedding_dim(self.embedding_dim)
        return torch.nn.ModuleList(
            [
                CliffordRGCNConv(self.clifford_algebra, self.num_relations_gnn)
                for _ in range(self.num_gnn_layers)
            ]
        )

    def _apply_layer(
        self,
        layer_idx: int,
        layer: torch.nn.Module,
        x: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_type: torch.LongTensor,
    ) -> torch.FloatTensor:
        x = layer(x, edge_index, edge_type)
        if layer_idx < self.num_gnn_layers - 1:
            x = F.relu(x)
        return x


class CliffordGraphSAGE(RelationalGNNEncoder):
    """Stack relation-wise Clifford aggregators and normalize each output.

    After hidden-layer ReLU, x_i <- x_i / max(||x_i||_2, eps).
    The norm is over coefficients, not the Clifford scalar product.
    """
    name = "CliffordGraphSAGE"

    def _build_layers(self) -> torch.nn.ModuleList:
        self.clifford_algebra = CliffordAlgebra.for_embedding_dim(self.embedding_dim)
        aggregator = self.args.get("sage_aggregator", "mean")
        return torch.nn.ModuleList(
            [
                CliffordRelationalSAGELayer(
                    self.clifford_algebra, self.num_relations_gnn, aggregator
                )
                for _ in range(self.num_gnn_layers)
            ]
        )

    def _apply_layer(
        self,
        layer_idx: int,
        layer: torch.nn.Module,
        x: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_type: torch.LongTensor,
    ) -> torch.FloatTensor:
        x = layer(x, edge_index, edge_type)
        if layer_idx < self.num_gnn_layers - 1:
            x = F.relu(x)
        return F.normalize(x, p=2, dim=1)


class CliffordGATv2(RelationalGNNEncoder):
    """Stack full-width Clifford attention heads with ELU between layers."""
    name = "CliffordGATv2"

    def _build_layers(self) -> torch.nn.ModuleList:
        self.clifford_algebra = CliffordAlgebra.for_embedding_dim(self.embedding_dim)
        heads = int(self.args.get("gnn_heads", 4))
        return torch.nn.ModuleList(
            [
                CliffordGATv2Conv(
                    self.clifford_algebra,
                    self.num_relations_gnn,
                    heads=heads,
                    dropout=self.gnn_dropout_rate,
                )
                for _ in range(self.num_gnn_layers)
            ]
        )

    def _apply_layer(
        self,
        layer_idx: int,
        layer: torch.nn.Module,
        x: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_type: torch.LongTensor,
    ) -> torch.FloatTensor:
        x = layer(x, edge_index, edge_type)
        if layer_idx < self.num_gnn_layers - 1:
            x = F.elu(x)
        return x
