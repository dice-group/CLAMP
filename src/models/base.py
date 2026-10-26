from typing import Optional, Tuple

import torch
from torch_geometric.nn.kge.base import KGEModel

from src.clifford_algebra import CliffordAlgebra
from src.decoders import BILINEAR_DECODERS, COMPLEX_DECODERS, DECODERS, clifford_decoder


class KGModel(KGEModel):
    """Entity and relation tables plus a decoder; no graph, no encoder.

    Scoring follows dicee (see tests/test_dicee_parity.py). Encoder models override
    ``encode`` and the graph accessors; the plain KGE baseline is this class as is.
    """
    name = "KGE"

    def __init__(self, args: dict):
        self.args = args
        self.embedding_dim = int(args["embedding_dim"])
        super().__init__(
            num_nodes=int(args["num_entities"]),
            num_relations=int(args["num_relations"]),
            hidden_channels=self.embedding_dim,
        )
        self.num_entities = self.num_nodes

        # Relations come in (r, r + R) pairs: the KG carries reciprocal triples, so
        # head prediction is a tail query on the inverse relation.
        if self.num_relations % 2:
            raise ValueError(
                "num_relations must be even: the KG holds each relation and its inverse"
            )
        self.num_direct_relations = self.num_relations // 2

        self._init_decoder(args)
        self._init_normalization_and_dropout(args)
        self._init_param(args)

    def _init_decoder(self, args: dict) -> None:
        """Set triple scoring and, for bilinear decoders, query_fn: scores are query_fn(h, r) @ E^T."""
        decoder = args.get("decoder", "DistMult")
        known = sorted([*DECODERS, *BILINEAR_DECODERS, "Clifford"])
        if decoder not in known:
            raise ValueError(f"Unknown decoder '{decoder}'. Available: {known}")
        if decoder in COMPLEX_DECODERS and self.embedding_dim % 2 != 0:
            raise ValueError(
                f"Decoder '{decoder}' splits embedding_dim into equal real/imaginary "
                f"halves and requires an even embedding_dim, got {self.embedding_dim}"
            )
        self.decoder = decoder
        self.query_fn = None
        if decoder == "Clifford":
            # Algebra from --clifford_signature, shared with Clifford encoders.
            self.decoder_algebra = CliffordAlgebra.for_embedding_dim(
                self.embedding_dim, args.get("clifford_signature")
            )
            self.triple_score_fn, self.query_fn = clifford_decoder(self.decoder_algebra)
        elif decoder in BILINEAR_DECODERS:
            self.triple_score_fn, self.query_fn = BILINEAR_DECODERS[decoder]
        else:
            self.triple_score_fn, self.k_vs_all_score_fn = DECODERS[decoder]

    def _init_normalization_and_dropout(self, args: dict) -> None:
        self.input_dropout_rate = float(args.get("input_dropout_rate", 0.0) or 0.0)
        self.hidden_dropout_rate = float(args.get("hidden_dropout_rate", 0.0) or 0.0)
        self.input_dp_ent_real = torch.nn.Dropout(self.input_dropout_rate)
        self.input_dp_rel_real = torch.nn.Dropout(self.input_dropout_rate)
        self.hidden_dropout = torch.nn.Dropout(self.hidden_dropout_rate)
        # Dropout on the bilinear query q(h, r) before scoring all tails. dicee's DistMult applies it
        # at input_dropout_rate; set --query_dropout_rate to that rate to reproduce it.
        self.query_dropout_rate = float(args.get("query_dropout_rate", 0.0) or 0.0)
        if self.query_dropout_rate and self.query_fn is None:
            raise ValueError(f"query_dropout_rate needs a bilinear decoder, not '{self.decoder}'")
        self.query_dropout = torch.nn.Dropout(self.query_dropout_rate)

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

    # Graph accessors: a model without an encoder has no graph.
    @property
    def graph(self) -> Tuple[Optional[torch.LongTensor], Optional[torch.LongTensor]]:
        return None, None

    def _graph_excluding_batch(
        self,
        idx_h: torch.LongTensor,
        idx_r: torch.LongTensor,
        idx_t: Optional[torch.LongTensor] = None,
    ) -> Tuple[Optional[torch.LongTensor], Optional[torch.LongTensor]]:
        return self.graph

    def encode(self, edge_index=None, edge_type=None) -> torch.FloatTensor:
        return self.node_emb.weight

    def relation_repr(self) -> torch.FloatTensor:
        return self.rel_emb.weight

    def k_vs_all_score(
        self,
        emb_h: torch.FloatTensor,
        emb_r: torch.FloatTensor,
        emb_E: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if self.query_fn is not None:
            return self.query_dropout(self.query_fn(emb_h, emb_r)) @ emb_E.transpose(0, 1)
        scores = self.k_vs_all_score_fn(emb_h, emb_r, emb_E)
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
        self, x: torch.LongTensor, query_tails: Optional[torch.LongTensor] = None,
        *, entity_repr: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        idx_h, idx_r = x[:, 0], x[:, 1]
        if entity_repr is not None and self.training:
            raise ValueError("Precomputed representations are for evaluation only")
        if entity_repr is None:
            edge_index, edge_type = self._graph_excluding_batch(idx_h, idx_r, query_tails)
            entity_repr = self.encode(edge_index, edge_type)
        emb_h = entity_repr.index_select(0, idx_h)
        emb_h = self.input_dp_ent_real(emb_h)
        emb_h = self.normalize_head_entity_embeddings(emb_h)
        emb_r = self.relation_repr().index_select(0, idx_r)
        emb_r = self.input_dp_rel_real(emb_r)
        emb_r = self.normalize_relation_embeddings(emb_r)
        return self.k_vs_all_score(emb_h, emb_r, entity_repr)

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
