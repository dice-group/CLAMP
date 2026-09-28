from typing import Optional, Tuple

import numpy as np
import torch
from torch_geometric.nn.kge.base import KGEModel

from src.decoders import COMPLEX_DECODERS, DECODERS
from src.graph_utils import build_relational_edge_index


class RelationalGNNEncoder(KGEModel):
    """Graph encoder and decoder built on PyG's KGEModel embedding tables.

    Subclasses supply message-passing layers; scoring functions are local.
    """
    def __init__(self, args: dict):
        self.args = args
        self.embedding_dim = int(args["embedding_dim"])
        super().__init__(
            num_nodes=int(args["num_entities"]),
            num_relations=int(args["num_relations"]),
            hidden_channels=self.embedding_dim,
        )
        self.num_entities = self.num_nodes

        # Relation r + num_relations answers the inverse query.
        self.reciprocal = bool(args.get("reciprocal", False))
        if self.reciprocal:
            self.rel_emb = torch.nn.Embedding(
                2 * self.num_relations, self.embedding_dim
            )

        add_inverse_edges = args.get("gnn_add_inverse_edges", True)
        self.add_inverse_edges = bool(add_inverse_edges)
        self.num_relations_gnn = (
            2 * self.num_relations if self.add_inverse_edges else self.num_relations
        )

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
        self.gnn_residual = bool(args.get("gnn_residual", False))

        self._init_decoder(args)
        self._init_normalization_and_dropout(args)
        self._init_param(args)

        self.gnn_layers = self._build_layers()

    def _init_decoder(self, args: dict) -> None:
        decoder = args.get("decoder", "DistMult")
        if decoder not in DECODERS:
            raise ValueError(
                f"Unknown decoder '{decoder}'. Available: {sorted(DECODERS)}"
            )
        if decoder in COMPLEX_DECODERS and self.embedding_dim % 2 != 0:
            raise ValueError(
                f"Decoder '{decoder}' splits embedding_dim into equal real/imaginary "
                f"halves and requires an even embedding_dim, got {self.embedding_dim}"
            )
        self.decoder = decoder
        self.triple_score_fn, self.k_vs_all_score_fn, self.head_k_vs_all_score_fn = (
            DECODERS[decoder]
        )

    def _init_normalization_and_dropout(self, args: dict) -> None:
        self.input_dropout_rate = float(args.get("input_dropout_rate", 0.0) or 0.0)
        self.hidden_dropout_rate = float(args.get("hidden_dropout_rate", 0.0) or 0.0)
        self.input_dp_ent_real = torch.nn.Dropout(self.input_dropout_rate)
        self.input_dp_rel_real = torch.nn.Dropout(self.input_dropout_rate)
        self.hidden_dropout = torch.nn.Dropout(self.hidden_dropout_rate)
        # DistMult query dropout uses the input rate for dicee parity.
        self.query_dropout = torch.nn.Dropout(self.input_dropout_rate)

        normalization = args.get("normalization")
        if normalization is None:
            self.normalize_head_entity_embeddings = torch.nn.Identity()
            self.normalize_tail_entity_embeddings = torch.nn.Identity()
            self.normalize_relation_embeddings = torch.nn.Identity()
            self.hidden_normalizer = torch.nn.Identity()
        elif normalization in ("LayerNorm", "BatchNorm1d"):
            cls = (
                torch.nn.LayerNorm
                if normalization == "LayerNorm"
                else torch.nn.BatchNorm1d
            )
            kwargs = {} if normalization == "LayerNorm" else {"affine": False}
            self.normalize_head_entity_embeddings = cls(self.embedding_dim, **kwargs)
            self.normalize_tail_entity_embeddings = torch.nn.Identity()
            self.normalize_relation_embeddings = cls(self.embedding_dim, **kwargs)
            self.hidden_normalizer = torch.nn.Identity()
        else:
            raise ValueError(f"Unknown normalization '{normalization}'")

    def _init_param(self, args: dict) -> None:
        init_param = args.get("init_param")
        if init_param is None:
            self.param_init = torch.nn.Identity()
        elif init_param == "xavier_normal":
            self.param_init = torch.nn.init.xavier_normal_
            self.param_init(self.node_emb.weight.data)
            self.param_init(self.rel_emb.weight.data)
        else:
            raise ValueError(f"Unknown init_param '{init_param}'")

    def set_graph(self, triples) -> "RelationalGNNEncoder":
        triples = torch.as_tensor(np.asarray(triples), dtype=torch.long)
        if triples.ndim != 2 or triples.shape[1] != 3 or not len(triples):
            raise ValueError(
                f"{self.__class__.__name__} requires a nonempty (N, 3) training graph"
            )
        if int(triples[:, [0, 2]].max()) >= self.num_entities:
            raise ValueError("Entity index is outside the declared vocabulary")
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
                triples = triples.clone()
                generator = torch.Generator().manual_seed(0)
                triples[:, 2] = torch.randint(
                    self.num_entities, (len(triples),), generator=generator
                )
            edge_index, edge_type = build_relational_edge_index(
                triples, self.num_relations, self.add_inverse_edges
            )
            if self.gnn_graph == "none":
                edge_index, edge_type = edge_index[:, :0], edge_type[:0]
            self._graph_cache = (
                edge_index.to(self.gnn_train_triples.device),
                edge_type.to(self.gnn_train_triples.device),
            )
        return self._graph_cache

    def _build_layers(self) -> torch.nn.ModuleList:
        raise NotImplementedError

    def _apply_layer(
        self,
        layer_idx: int,
        layer: torch.nn.Module,
        x: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_type: torch.LongTensor,
    ) -> torch.FloatTensor:
        raise NotImplementedError

    def _query_edge_mask(
        self,
        edge_index: torch.LongTensor,
        edge_type: torch.LongTensor,
        idx_h: torch.LongTensor,
        idx_r: torch.LongTensor,
        idx_t: Optional[torch.LongTensor] = None,
    ) -> torch.BoolTensor:
        num_relations = self.num_relations
        num_query_relations = 2 * num_relations
        is_synthetic = edge_type >= num_relations
        relation = torch.where(is_synthetic, edge_type - num_relations, edge_type)
        head = torch.where(is_synthetic, edge_index[1], edge_index[0])
        tail = torch.where(is_synthetic, edge_index[0], edge_index[1])
        # Match each edge to both forward and reciprocal queries.
        query_keys = idx_h * num_query_relations + idx_r
        forward_keys = head * num_query_relations + relation
        reciprocal_keys = tail * num_query_relations + relation + num_relations
        if idx_t is not None:
            query_keys = query_keys * self.num_entities + idx_t
            forward_keys = forward_keys * self.num_entities + tail
            reciprocal_keys = reciprocal_keys * self.num_entities + head
        return torch.isin(forward_keys, query_keys) | torch.isin(
            reciprocal_keys, query_keys
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

    def encode(
        self, edge_index: torch.LongTensor, edge_type: torch.LongTensor
    ) -> torch.FloatTensor:
        x = self.node_emb.weight
        num_layers = len(self.gnn_layers)
        for i, layer in enumerate(self.gnn_layers):
            x_in = x
            x = self._apply_layer(i, layer, x, edge_index, edge_type)
            if self.gnn_residual:
                x = x + x_in
            if i < num_layers - 1:
                x = self.gnn_dropout(x)
        return x

    def relation_repr(self) -> torch.FloatTensor:
        return self.rel_emb.weight

    def k_vs_all_score(
        self,
        emb_h: torch.FloatTensor,
        emb_r: torch.FloatTensor,
        emb_E: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if self.decoder == "DistMult":
            return self.query_dropout(emb_h * emb_r) @ emb_E.transpose(0, 1)
        scores = self.k_vs_all_score_fn(emb_h, emb_r, emb_E)
        if self.decoder == "ComplEx":
            return scores
        return self.hidden_dropout(self.hidden_normalizer(scores))

    def head_k_vs_all_score(
        self,
        emb_r: torch.FloatTensor,
        emb_t: torch.FloatTensor,
        emb_E: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if self.decoder == "DistMult":
            return self.query_dropout(emb_t * emb_r) @ emb_E.transpose(0, 1)
        scores = self.head_k_vs_all_score_fn(emb_r, emb_t, emb_E)
        if self.decoder == "ComplEx":
            return scores
        return self.hidden_dropout(self.hidden_normalizer(scores))

    def forward_triples(self, x: torch.LongTensor) -> torch.FloatTensor:
        idx_h, idx_r, idx_t = x[:, 0], x[:, 1], x[:, 2]
        edge_index, edge_type = self._graph_excluding_batch(idx_h, idx_r)
        entity_repr = self.encode(edge_index, edge_type)
        emb_h = entity_repr.index_select(0, idx_h)
        emb_h = self.input_dp_ent_real(emb_h)
        emb_h = self.normalize_head_entity_embeddings(emb_h)
        emb_t = self.normalize_tail_entity_embeddings(
            entity_repr.index_select(0, idx_t)
        )
        emb_r = self.relation_repr().index_select(0, idx_r)
        emb_r = self.input_dp_rel_real(emb_r)
        emb_r = self.normalize_relation_embeddings(emb_r)
        return self.triple_score_fn(emb_h, emb_r, emb_t)

    def forward_k_vs_all(
        self, x: torch.LongTensor, query_tails: Optional[torch.LongTensor] = None
    ) -> torch.FloatTensor:
        idx_h, idx_r = x[:, 0], x[:, 1]
        edge_index, edge_type = self._graph_excluding_batch(idx_h, idx_r, query_tails)
        entity_repr = self.encode(edge_index, edge_type)
        emb_h = entity_repr.index_select(0, idx_h)
        emb_h = self.input_dp_ent_real(emb_h)
        emb_h = self.normalize_head_entity_embeddings(emb_h)
        emb_r = self.relation_repr().index_select(0, idx_r)
        emb_r = self.input_dp_rel_real(emb_r)
        emb_r = self.normalize_relation_embeddings(emb_r)
        return self.k_vs_all_score(emb_h, emb_r, entity_repr)

    def forward_k_vs_all_head(self, x: torch.LongTensor) -> torch.FloatTensor:
        idx_r, idx_t = x[:, 0], x[:, 1]
        if self.reciprocal:
            queries = torch.stack((idx_t, idx_r + self.num_relations), dim=1)
            return self.forward_k_vs_all(queries)
        edge_index, edge_type = self.graph
        entity_repr = self.encode(edge_index, edge_type)
        emb_r = self.relation_repr().index_select(0, idx_r)
        emb_r = self.input_dp_rel_real(emb_r)
        emb_r = self.normalize_relation_embeddings(emb_r)
        emb_t = self.normalize_tail_entity_embeddings(
            entity_repr.index_select(0, idx_t)
        )
        return self.head_k_vs_all_score(emb_r, emb_t, entity_repr)

    def forward(
        self, x: torch.LongTensor, query_tails: Optional[torch.LongTensor] = None
    ) -> torch.FloatTensor:
        if x.ndim != 2 or x.shape[1] not in (2, 3):
            raise ValueError(
                f"Expected a (B, 2) or (B, 3) LongTensor, got shape {tuple(x.shape)}"
            )
        if x.shape[1] == 3:
            return self.forward_triples(x)
        return self.forward_k_vs_all(x, query_tails)
