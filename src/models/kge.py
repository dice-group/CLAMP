from typing import Optional, Tuple

import torch

from src.models.base_gnn import RelationalGNNEncoder


class KGE(RelationalGNNEncoder):
    name = "KGE"

    def _build_layers(self) -> torch.nn.ModuleList:
        return torch.nn.ModuleList()

    @property
    def graph(self) -> Tuple[Optional[torch.LongTensor], Optional[torch.LongTensor]]:
        return None, None

    def _graph_excluding_batch(
        self,
        idx_h: torch.LongTensor,
        idx_r: torch.LongTensor,
        idx_t: Optional[torch.LongTensor] = None,
    ) -> Tuple[None, None]:
        return None, None


class LTE(KGE):
    name = "LTE"

    def __init__(self, args: dict):
        super().__init__(args)
        self.lte_bn = torch.nn.BatchNorm1d(self.embedding_dim)

    def _build_layers(self) -> torch.nn.ModuleList:
        return torch.nn.ModuleList(
            [torch.nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)]
        )

    def _apply_layer(self, layer_idx, layer, x, edge_index, edge_type):
        return self.gnn_dropout(self.lte_bn(layer(x)))
