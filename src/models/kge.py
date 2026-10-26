import torch

from src.models.base import KGModel


class KGE(KGModel):
    """Decoder-only baseline: entity representations are the embedding table."""
    name = "KGE"


class LTE(KGModel):
    """LTE (Zhang et al., 2022): one graph-free linear transform with BatchNorm and dropout."""
    name = "LTE"

    def __init__(self, args: dict):
        super().__init__(args)
        self.gnn_dropout = torch.nn.Dropout(float(args.get("gnn_dropout_rate", 0.2)))
        self.gnn_layers = torch.nn.ModuleList(
            [torch.nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)]
        )
        self.lte_bn = torch.nn.BatchNorm1d(self.embedding_dim)

    def encode(self, edge_index=None, edge_type=None) -> torch.FloatTensor:
        return self.gnn_dropout(self.lte_bn(self.gnn_layers[0](self.node_emb.weight)))
