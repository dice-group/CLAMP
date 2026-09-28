from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset


def mapping_from_first_two_cols_to_third(triples: np.ndarray) -> dict:
    store = defaultdict(list)
    for h, r, t in triples:
        store[(int(h), int(r))].append(int(t))
    return dict(store)


def add_reciprocal_triples(triples: np.ndarray, num_relations: int) -> np.ndarray:
    triples = np.asarray(triples)
    inverse = np.stack(
        [triples[:, 2], triples[:, 1] + num_relations, triples[:, 0]], axis=1
    )
    return np.concatenate([triples, inverse], axis=0)


class KvsAll(Dataset):
    def __init__(self, train_set_idx: np.ndarray, num_entities: int):
        super().__init__()
        assert len(train_set_idx) > 0
        store = mapping_from_first_two_cols_to_third(np.asarray(train_set_idx))
        self.train_data = torch.LongTensor(list(store.keys()))
        self.train_target = list(store.values())
        self.target_dim = num_entities

    def __len__(self) -> int:
        return len(self.train_data)

    def __getitem__(self, idx: int):
        y_vec = torch.zeros(self.target_dim)
        y_vec[self.train_target[idx]] = 1.0
        return self.train_data[idx], y_vec


class OneVsAll(Dataset):
    def __init__(self, train_set_idx: np.ndarray):
        super().__init__()
        triples = torch.as_tensor(np.asarray(train_set_idx), dtype=torch.long)
        assert len(triples) > 0
        self.train_data = triples[:, :2]
        self.train_target = triples[:, 2]

    def __len__(self) -> int:
        return len(self.train_data)

    def __getitem__(self, idx: int):
        return self.train_data[idx], self.train_target[idx]


def collate_fn(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)
