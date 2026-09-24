"""Clifford-algebra encoders: one per architecture in this package.

Each of the three architectures here has a Clifford counterpart, keeping the
original's message-passing structure and swapping only its learned linear maps
for geometric products (see :mod:`src.models.clifford_layers`). An embedding is
one complete multivector of the Euclidean algebra Cl(n, 0), so
``embedding_dim`` must be a power of two - see
:class:`src.clifford_algebra.CliffordAlgebra`.
"""
import torch
import torch.nn.functional as F

from src.clifford_algebra import CliffordAlgebra
from src.models.base_gnn import RelationalGNNEncoder
from src.models.clifford_layers import (CliffordGATv2Conv, CliffordRelationalSAGELayer,
                                         CliffordRGCNConv)


class CliffordRGCN(RelationalGNNEncoder):
    """R-GCN over multivectors: per-relation geometric product in place of ``W_r``.

    Clifford counterpart of :class:`src.models.rgcn.RGCN`. Basis decomposition
    (``--num_bases``) has no analogue here - the relation operator is already a
    single multivector rather than a full matrix, so it is ignored.
    """

    name = "CliffordRGCN"

    def _build_layers(self) -> torch.nn.ModuleList:
        self.clifford_algebra = CliffordAlgebra.for_embedding_dim(self.embedding_dim)
        return torch.nn.ModuleList([
            CliffordRGCNConv(self.clifford_algebra, self.num_relations_gnn)
            for _ in range(self.num_gnn_layers)
        ])

    def _apply_layer(self, layer_idx: int, layer: torch.nn.Module, x: torch.FloatTensor,
                      edge_index: torch.LongTensor, edge_type: torch.LongTensor) -> torch.FloatTensor:
        x = layer(x, edge_index, edge_type)
        if layer_idx < self.num_gnn_layers - 1:
            x = F.relu(x)
        return x


class CliffordGraphSAGE(RelationalGNNEncoder):
    """Relational GraphSAGE over multivectors.

    Clifford counterpart of :class:`src.models.graphsage.GraphSAGE`: one
    neighbour transform per relation, summed, plus a shared root transform,
    followed by the unit L2-normalisation of GraphSAGE Algorithm 2 line 7.
    """

    name = "CliffordGraphSAGE"

    def _build_layers(self) -> torch.nn.ModuleList:
        self.clifford_algebra = CliffordAlgebra.for_embedding_dim(self.embedding_dim)
        aggregator = self.args.get("sage_aggregator", "mean")
        return torch.nn.ModuleList([
            CliffordRelationalSAGELayer(self.clifford_algebra, self.num_relations_gnn, aggregator)
            for _ in range(self.num_gnn_layers)
        ])

    def _apply_layer(self, layer_idx: int, layer: torch.nn.Module, x: torch.FloatTensor,
                      edge_index: torch.LongTensor, edge_type: torch.LongTensor) -> torch.FloatTensor:
        x = layer(x, edge_index, edge_type)
        if layer_idx < self.num_gnn_layers - 1:
            x = F.relu(x)
        return F.normalize(x, p=2, dim=1)


class CliffordGATv2(RelationalGNNEncoder):
    """Relation-aware GATv2 over multivectors.

    Clifford counterpart of :class:`src.models.rgat.GATv2`; see
    :class:`src.models.clifford_layers.CliffordGATv2Conv` for the attention
    construction and why heads are averaged rather than concatenated.
    """

    name = "CliffordGATv2"

    def _build_layers(self) -> torch.nn.ModuleList:
        self.clifford_algebra = CliffordAlgebra.for_embedding_dim(self.embedding_dim)
        heads = int(self.args.get("gnn_heads", 4))
        return torch.nn.ModuleList([
            CliffordGATv2Conv(self.clifford_algebra, self.num_relations_gnn, heads=heads,
                               dropout=self.gnn_dropout_rate)
            for _ in range(self.num_gnn_layers)
        ])

    def _apply_layer(self, layer_idx: int, layer: torch.nn.Module, x: torch.FloatTensor,
                      edge_index: torch.LongTensor, edge_type: torch.LongTensor) -> torch.FloatTensor:
        x = layer(x, edge_index, edge_type)
        if layer_idx < self.num_gnn_layers - 1:
            x = F.elu(x)
        return x
