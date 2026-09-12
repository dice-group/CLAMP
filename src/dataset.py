"""KvsAll training dataset, adapted from dice-embeddings' dataset class.

Identical in spirit to ``literal-embeddings/src/dataset.py::KvsAll`` and to
``dicee.dataset_classes._label_based.KvsAll``: build one ``(head, relation)``
row per unique pair seen in training, with a multi-hot target vector over
all entities. Reimplemented locally (rather than imported) only so this
project has no hidden dependency on dicee's internal, non-public dataset
module layout.
"""
import numpy as np
import torch
from dicee.static_preprocess_funcs import mapping_from_first_two_cols_to_third
from torch.utils.data import Dataset


class KvsAll(Dataset):
    """KvsAll dataset: ``x = (h, r)``, ``y`` is a multi-hot vector over entities.

    Args:
        train_set_idx: ``(N, 3)`` integer array of indexed ``[h, r, t]`` triples.
        num_entities: Size of the multi-hot target vector.
    """

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


def collate_fn(batch):
    xs, y_vecs = zip(*batch)
    return torch.stack(xs), torch.stack(y_vecs)
