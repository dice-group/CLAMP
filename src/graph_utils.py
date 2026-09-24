"""Build message-passing graphs (edge_index / edge_type) from indexed triples.

All three GNN encoders in ``src/models`` need the training graph in
``torch_geometric``'s ``edge_index`` / ``edge_type`` format rather than flat
``(h, r, t)`` triples. This module builds that graph (see
``RelationalGNNEncoder`` in ``src/models/base_gnn.py``) and is the only place
inverse-relation bookkeeping happens, so every encoder sees the same
convention.

Convention (matches Schlichtkrull et al. 2018, R-GCN, Section 4.1): for every
training triple ``(h, r, t)`` the propagation graph also contains the inverse
edge ``(t, r + num_relations, h)``. Message passing therefore has
``2 * num_relations`` edge types; the decoder still scores triples with the
original ``num_relations`` relation embeddings.
"""
from typing import Tuple

import numpy as np
import torch


def build_relational_edge_index(train_set_idx, num_relations: int,
                                add_inverse_edges: bool = True) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """Convert indexed training triples into a message-passing graph.

    Args:
        train_set_idx: ``(N, 3)`` integer array/tensor of ``[head, relation, tail]``.
        num_relations: Number of distinct relations used by the decoder.
        add_inverse_edges: If True, add a reverse edge ``(t, r + num_relations, h)``
            for every training triple ``(h, r, t)`` so information can flow against
            the original edge direction, as in the R-GCN reference implementation.

    Returns:
        edge_index: ``(2, E)`` LongTensor, row 0 source, row 1 target.
        edge_type: ``(E,)`` LongTensor of relation ids in ``[0, num_relations_gnn)``,
            where ``num_relations_gnn`` is ``2 * num_relations`` if inverse edges are
            added and ``num_relations`` otherwise.
    """
    if isinstance(train_set_idx, torch.Tensor):
        triples = train_set_idx.long()
    else:
        triples = np.asarray(train_set_idx)
        if triples.ndim != 2 or triples.shape[1] != 3:
            raise ValueError("Expected an (N, 3) array of integer triples")
        if not np.issubdtype(triples.dtype, np.integer):
            raise ValueError("Triple indices must be integers")
        triples = torch.as_tensor(triples, dtype=torch.long)
    if triples.ndim != 2 or triples.shape[1] != 3:
        raise ValueError("Expected an (N, 3) array of integer triples")
    if num_relations <= 0 or bool(torch.any(triples < 0)):
        raise ValueError("Relation count must be positive and indices nonnegative")
    if triples.numel() and bool(torch.any(triples[:, 1] >= num_relations)):
        raise ValueError("Relation index is outside the declared vocabulary")

    h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]
    if add_inverse_edges:
        src = torch.cat([h, t], dim=0)
        dst = torch.cat([t, h], dim=0)
        rel = torch.cat([r, r + num_relations], dim=0)
    else:
        src, dst, rel = h, t, r
    return torch.stack([src, dst], dim=0), rel
