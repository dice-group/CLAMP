"""Build message-passing graphs (edge_index / edge_type) from indexed triples.

All three GNN encoders in ``src/models`` need the training graph in
``torch_geometric``'s ``edge_index`` / ``edge_type`` format rather than the
flat ``(h, r, t)`` triples dicee's dataset classes hand to ordinary KGE
models. This module builds that graph once per model (see
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


def build_relational_edge_index(
    train_set_idx: np.ndarray,
    num_relations: int,
    add_inverse_edges: bool = True,
) -> Tuple[torch.LongTensor, torch.LongTensor, int]:
    """Convert indexed training triples into a message-passing graph.

    Args:
        train_set_idx: ``(N, 3)`` integer array of ``[head, relation, tail]``.
        num_relations: Number of distinct relations used by the decoder.
        add_inverse_edges: If True, add a reverse edge
            ``(t, r + num_relations, h)`` for every training triple ``(h, r,
            t)`` so information can flow against the original edge
            direction, as in the R-GCN reference implementation.

    Returns:
        edge_index: ``(2, E)`` LongTensor, row 0 source, row 1 target.
        edge_type: ``(E,)`` LongTensor of relation ids in
            ``[0, num_relations_gnn)``.
        num_relations_gnn: Number of relation types used for message
            passing (``2 * num_relations`` if inverse edges are added,
            otherwise ``num_relations``).
    """
    triples = np.asarray(train_set_idx)[:, :3]
    h = torch.as_tensor(triples[:, 0], dtype=torch.long)
    r = torch.as_tensor(triples[:, 1], dtype=torch.long)
    t = torch.as_tensor(triples[:, 2], dtype=torch.long)

    if add_inverse_edges:
        src = torch.cat([h, t], dim=0)
        dst = torch.cat([t, h], dim=0)
        rel = torch.cat([r, r + num_relations], dim=0)
        num_relations_gnn = 2 * num_relations
    else:
        src, dst, rel = h, t, r
        num_relations_gnn = num_relations

    edge_index = torch.stack([src, dst], dim=0)
    return edge_index, rel, num_relations_gnn
