from typing import Tuple

import numpy as np
import torch


def build_relational_edge_index(
    train_set_idx, num_relations: int
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """Edges of the reciprocal-augmented training set: each row (h, r, t) is h -> t of type r."""
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

    return triples[:, [0, 2]].T.contiguous(), triples[:, 1].contiguous()
