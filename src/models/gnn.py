from typing import Optional, Tuple

import numpy as np
import torch

from src.graph_utils import build_relational_edge_index
from src.models.base import KGModel


class RelationalGNN(KGModel):
    """KGModel whose entity representations come from message passing over the training graph.

    This class only owns the graph: storing it, the graph controls (``gnn_graph``: real, random
    tails or no edges) and which edges a training batch may see (``query_edge_exclusion``,
    ``gnn_edge_dropout``). Each subclass builds ``self.gnn_layers`` and writes its own ``encode``.
    """

    def __init__(self, args: dict):
        super().__init__(args)
        self.gnn_graph = args.get("gnn_graph", "train")
        if self.gnn_graph not in ("train", "random", "none"):
            raise ValueError(f"Unknown gnn_graph '{self.gnn_graph}'")
        self.query_edge_exclusion = args.get("query_edge_exclusion", "pair")
        if self.query_edge_exclusion not in ("pair", "triple", "none"):
            raise ValueError(
                f"Unknown query_edge_exclusion '{self.query_edge_exclusion}'"
            )
        self.gnn_edge_dropout = float(args.get("gnn_edge_dropout", 0.0) or 0.0)

        self.register_buffer("gnn_train_triples", torch.empty(0, 3, dtype=torch.long))
        self._graph_cache: Optional[Tuple[torch.LongTensor, torch.LongTensor]] = None
        if args.get("train_set_idx") is not None:
            self.set_graph(args["train_set_idx"])

        self.num_gnn_layers = int(args.get("num_gnn_layers", 2))
        if self.num_gnn_layers < 0:
            raise ValueError(
                "num_gnn_layers must be nonnegative; zero disables message passing"
            )
        self.gnn_dropout_rate = float(args.get("gnn_dropout_rate", 0.2))
        self.gnn_dropout = torch.nn.Dropout(self.gnn_dropout_rate)

    def encode(
        self, edge_index: torch.LongTensor, edge_type: torch.LongTensor
    ) -> torch.FloatTensor:
        raise NotImplementedError

    def set_graph(self, triples) -> "RelationalGNN":
        triples = torch.as_tensor(np.asarray(triples), dtype=torch.long)
        if triples.ndim != 2 or triples.shape[1] != 3 or not len(triples):
            raise ValueError(
                f"{self.__class__.__name__} requires a nonempty (N, 3) training graph"
            )
        if int(triples[:, [0, 2]].max()) >= self.num_entities:
            raise ValueError("Entity index is outside the declared vocabulary")
        if int(triples[:, 1].max()) >= self.num_relations:
            raise ValueError("Relation index is outside the declared vocabulary")
        head, relation, tail = triples.unbind(dim=1)
        keys = (head * self.num_relations + relation) * self.num_entities + tail
        inverse_relation = (relation + self.num_direct_relations) % self.num_relations
        mirrored = (tail * self.num_relations + inverse_relation) * self.num_entities + head
        if not torch.isin(mirrored, keys).all():
            raise ValueError(
                "The training graph must hold each triple's reciprocal (t, r_inverse, h); "
                "build it with load_kg or create_reciprocal_triples"
            )
        self.gnn_train_triples = triples.to(self.gnn_train_triples.device)
        self._graph_cache = None
        return self

    @property
    def graph(self) -> Tuple[torch.LongTensor, torch.LongTensor]:
        if not len(self.gnn_train_triples):
            raise RuntimeError(
                f"Attach a training graph with set_graph() (or args['train_set_idx']) "
                f"before scoring {self.__class__.__name__}"
            )
        if (
            self._graph_cache is None
            or self._graph_cache[0].device != self.gnn_train_triples.device
        ):
            triples = self.gnn_train_triples.cpu()
            if self.gnn_graph == "random":
                # Resample the tails of the direct triples, then mirror them so the
                # graph keeps both directions.
                direct = triples[triples[:, 1] < self.num_direct_relations].clone()
                generator = torch.Generator().manual_seed(0)
                direct[:, 2] = torch.randint(
                    self.num_entities, (len(direct),), generator=generator
                )
                inverse = torch.stack(
                    [direct[:, 2], direct[:, 1] + self.num_direct_relations, direct[:, 0]],
                    dim=1,
                )
                triples = torch.cat([direct, inverse], dim=0)
            edge_index, edge_type = build_relational_edge_index(
                triples, self.num_relations
            )
            if self.gnn_graph == "none":
                edge_index, edge_type = edge_index[:, :0], edge_type[:0]
            self._graph_cache = (
                edge_index.to(self.gnn_train_triples.device),
                edge_type.to(self.gnn_train_triples.device),
            )
        return self._graph_cache

    def _query_edge_mask(
        self,
        edge_index: torch.LongTensor,
        edge_type: torch.LongTensor,
        idx_h: torch.LongTensor,
        idx_r: torch.LongTensor,
        idx_t: Optional[torch.LongTensor] = None,
    ) -> torch.BoolTensor:
        """Edges answering the queries (h, r) or (h, r, t), plus their mirrored copies."""
        num_relations = self.num_relations
        head, tail = edge_index[0], edge_index[1]
        # The mirror of edge (h, r, t) is (t, r^-1, h), with r^-1 = (r + R) mod 2R.
        inverse_type = (edge_type + self.num_direct_relations) % num_relations
        query_keys = idx_h * num_relations + idx_r
        forward_keys = head * num_relations + edge_type
        mirrored_keys = tail * num_relations + inverse_type
        if idx_t is not None:
            query_keys = query_keys * self.num_entities + idx_t
            forward_keys = forward_keys * self.num_entities + tail
            mirrored_keys = mirrored_keys * self.num_entities + head
        return torch.isin(forward_keys, query_keys) | torch.isin(
            mirrored_keys, query_keys
        )

    def _graph_excluding_batch(
        self,
        idx_h: torch.LongTensor,
        idx_r: torch.LongTensor,
        idx_t: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        edge_index, edge_type = self.graph
        if not self.training:
            return edge_index, edge_type
        if self.gnn_edge_dropout:
            keep = (
                torch.rand(edge_type.shape, device=edge_type.device)
                >= self.gnn_edge_dropout
            )
            edge_index, edge_type = edge_index[:, keep], edge_type[keep]
        if self.query_edge_exclusion == "none":
            return edge_index, edge_type
        triple = self.query_edge_exclusion == "triple" and idx_t is not None
        remove = self._query_edge_mask(
            edge_index, edge_type, idx_h, idx_r, idx_t if triple else None
        )
        if not torch.any(remove):
            return edge_index, edge_type
        keep = ~remove
        return edge_index[:, keep], edge_type[keep]
