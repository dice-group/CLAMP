import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class KG:
    train_set: np.ndarray
    valid_set: Optional[np.ndarray]
    test_set: Optional[np.ndarray]
    entity_to_idx: Dict[str, int]
    relation_to_idx: Dict[str, int]

    @property
    def num_entities(self) -> int:
        return len(self.entity_to_idx)

    @property
    def num_relations(self) -> int:
        return len(self.relation_to_idx)


def _read_triples(path: str):
    if not os.path.exists(path):
        return []
    triples = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 3:
                raise ValueError(
                    f"Expected 'head relation tail' per line in {path}, got: {line!r}"
                )
            triples.append(tuple(parts))
    return triples


def create_reciprocal_triples(triples: np.ndarray, num_relations: int) -> np.ndarray:
    """(h, r, t) union (t, r + num_relations, h), as dicee's create_recipriocal_triples."""
    triples = np.asarray(triples).reshape(-1, 3)
    inverse = np.stack(
        [triples[:, 2], triples[:, 1] + num_relations, triples[:, 0]], axis=1
    )
    return np.concatenate([triples, inverse], axis=0)


def load_kg(dataset_dir: str, vocab_from_all_splits: bool = False) -> KG:
    """Read a KG; train/valid/test all carry reciprocal triples, as in dicee.

    Relation ``r`` keeps its index and its inverse ``r_inverse`` is ``r + R``, so
    ``num_relations`` is ``2R``. Head prediction is then a tail query on ``r_inverse``.
    """
    train = _read_triples(os.path.join(dataset_dir, "train.txt"))
    valid = _read_triples(os.path.join(dataset_dir, "valid.txt"))
    test = _read_triples(os.path.join(dataset_dir, "test.txt"))
    if not train:
        raise ValueError(f"No training triples found under {dataset_dir}")

    vocab_source = train + valid + test if vocab_from_all_splits else train
    entities = sorted({h for h, _, _ in vocab_source} | {t for _, _, t in vocab_source})
    relations = sorted({r for _, r, _ in vocab_source})
    entity_to_idx = {e: i for i, e in enumerate(entities)}
    if any(f"{r}_inverse" in relations for r in relations):
        raise ValueError("A relation named '<r>_inverse' clashes with the generated inverse relations")
    num_direct = len(relations)
    relation_to_idx = {r: i for i, r in enumerate(relations)}
    relation_to_idx.update({f"{r}_inverse": i + num_direct for r, i in relation_to_idx.items()})

    def index(triples):
        rows = [
            (entity_to_idx[h], relation_to_idx[r], entity_to_idx[t])
            for h, r, t in triples
            if h in entity_to_idx and r in relation_to_idx and t in entity_to_idx
        ]
        return create_reciprocal_triples(np.asarray(rows, dtype=np.int64), num_direct)

    return KG(
        train_set=index(train),
        valid_set=index(valid) if valid else None,
        test_set=index(test) if test else None,
        entity_to_idx=entity_to_idx,
        relation_to_idx=relation_to_idx,
    )
