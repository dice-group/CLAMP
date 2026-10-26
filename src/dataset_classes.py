"""KvsAll and 1vsAll training datasets, copied from dicee.

Source: dice-group/dice-embeddings @ 07e372ddf
(``dicee/dataset_classes/_label_based.py`` and ``_storage.py``). The classes are kept
identical to dicee's; ``tests/test_dicee_parity.py`` compares them with the installed
dicee. Unused parts (``AllvsAll``, ``KvsSampleDataset``, relation-vocabulary helpers)
are left out.
"""
import operator
from dataclasses import dataclass

import numpy as np
import torch

@dataclass
class _TensorArray:
    tensor: torch.Tensor


@dataclass
class _MemmapArray:
    filename: str
    dtype: str
    shape: tuple
    offset: int


class WorkerDataset(torch.utils.data.Dataset):
    """Keep read-only NumPy data shared when a loader pickles its dataset.

    Fork already shares numeric array buffers. For spawn/forkserver, reopen
    contiguous file mappings and let PyTorch's multiprocessing reducers share
    ordinary arrays as tensor storage. Cache the tensor wrapper so subsequent
    workers reuse the same shared storage instead of making another copy.
    Dataset buffers must not be modified while workers are using them.
    """

    def __getstate__(self):
        state = self.__dict__.copy()
        cache = state.pop('_worker_array_cache', {})
        for name, array in state.items():
            if not isinstance(array, np.ndarray):
                continue
            if array.dtype.hasobject:
                raise TypeError('Worker datasets require numeric arrays, not dtype=object')
            cached = cache.get(name)
            if cached is None or cached[0] is not array:
                storage: _MemmapArray | _TensorArray
                root = array
                while isinstance(root.base, np.ndarray):
                    root = root.base
                if isinstance(root, np.memmap) and root.filename and array.flags.c_contiguous:
                    offset = root.offset + array.ctypes.data - root.ctypes.data
                    storage = _MemmapArray(str(root.filename), array.dtype.str, array.shape, offset)
                else:
                    storage = _TensorArray(torch.from_numpy(array))
                cached = cache[name] = (array, storage)
            state[name] = cached[1]
        self._worker_array_cache = cache
        return state

    def __setstate__(self, state):
        for name, value in state.items():
            if isinstance(value, _MemmapArray):
                state[name] = np.memmap(value.filename, dtype=value.dtype, mode='r',
                                        shape=value.shape, offset=value.offset)
            elif isinstance(value, _TensorArray):
                state[name] = value.tensor.numpy()
        self.__dict__.update(state)


class RaggedIndices:
    """Variable-length integer rows stored as two contiguous CPU tensors."""

    def __init__(self, values, offsets):
        self.values = torch.as_tensor(values, dtype=torch.long).contiguous()
        self.offsets = torch.as_tensor(offsets, dtype=torch.long).contiguous()


    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, index):
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        # numpy() is a view of the tensor buffer, including in spawned workers.
        start, end = self.offsets.numpy()[index:index + 2]
        return self.values[int(start):int(end)]

    @property
    def max_length(self):
        return int(np.diff(self.offsets.numpy()).max(initial=0))


class PairIndex:
    """Sorted integer pairs and their ragged targets, without a Python dict."""

    def __init__(self, keys, targets: RaggedIndices, max_input_count=None):
        self.keys = torch.as_tensor(keys, dtype=torch.long).contiguous()
        self.targets = targets
        self.max_input_count = targets.max_length if max_input_count is None else max_input_count

    @classmethod
    def from_triples(cls, triples, columns=(0, 1, 2), *, sort_targets=False, unique=False):
        # lexsort is stable: by default targets retain their original ordering
        # within each pair, matching the old dictionary-of-lists implementation.
        first, second, target = (np.asarray(triples[:, col]) for col in columns)
        sort_keys = (target, second, first) if sort_targets or unique else (second, first)
        order = np.lexsort(sort_keys)
        keys = np.column_stack((first[order], second[order])).astype(np.int64, copy=False)
        values = np.asarray(target[order], dtype=np.int64)
        starts = np.flatnonzero(np.r_[True, (keys[1:] != keys[:-1]).any(axis=1)]) if len(keys) else np.empty(0, dtype=np.int64)
        max_input_count = int(np.diff(np.r_[starts, len(values)]).max(initial=0))
        if unique and len(values):
            keep = np.r_[True, (keys[1:] != keys[:-1]).any(axis=1) | (values[1:] != values[:-1])]
            keys, values = keys[keep], values[keep]
            starts = np.flatnonzero(np.r_[True, (keys[1:] != keys[:-1]).any(axis=1)])
        offsets = np.r_[starts, len(values)]
        return cls(keys[starts], RaggedIndices(values, offsets), max_input_count)


class OnevsAllDataset(WorkerDataset):
    """Dataset for the 1-vs-All training strategy (multi-class).

    Each sample is a ``(head, relation)`` pair with a one-hot target vector
    whose single active position corresponds to the true tail entity.

    Parameters
    ----------
    train_set_idx : numpy.ndarray
        ``(N, 3)`` integer-indexed triples.
    entity_idxs : dict
        Entity-name → index mapping (used to determine the target dimension).
    """

    def __init__(self, train_set_idx: np.ndarray, entity_idxs):
        super().__init__()
        assert isinstance(train_set_idx, (np.memmap, np.ndarray))
        assert len(train_set_idx) > 0
        # Sort by (head, relation, tail) to ensure order-independent training
        # This prevents different input orderings from affecting optimization
        sorted_indices = np.lexsort(
            (train_set_idx[:, 2], train_set_idx[:, 1], train_set_idx[:, 0])
        )
        self.train_data = train_set_idx[sorted_indices]
        self.target_dim = len(entity_idxs)
        self.collate_fn = None

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        y_vec = torch.zeros(self.target_dim)
        triple = torch.from_numpy(self.train_data[idx].copy()).long()
        y_vec[triple[2]] = 1
        return triple[:2], y_vec


class KvsAll(WorkerDataset):
    """Dataset for KvsAll training (multi-label).

    D := {(x, y)_i}_{i=1}^{N} where
        * x = (h, r) is a unique (entity, relation) pair observed in the KG,
        * y ∈ [0, 1]^{|E|} is a multi-label vector with y_j = 1 iff
          (h, r, e_j) ∈ KG.

    Parameters
    ----------
    train_set_idx : numpy.ndarray
        ``(N, 3)`` integer-indexed triples.
    entity_idxs : dict
        Entity-name → index mapping.
    relation_idxs : dict
        Relation-name → index mapping.
    form : str
        ``'EntityPrediction'`` or ``'RelationPrediction'``.
    label_smoothing_rate : float, optional
        Label smoothing coefficient (default ``0.0``).
    """

    def __init__(
        self,
        train_set_idx: np.ndarray,
        entity_idxs,
        relation_idxs,
        form,
        store=None,
        label_smoothing_rate: float = 0.0,
    ):
        super().__init__()
        assert len(train_set_idx) > 0
        assert isinstance(train_set_idx, (np.memmap, np.ndarray))
        self.train_data: torch.Tensor
        self.train_target: RaggedIndices
        self.label_smoothing_rate = torch.tensor(label_smoothing_rate)
        self.collate_fn = None

        if store is not None:
            raise ValueError("A prebuilt store is not supported")
        if form == "RelationPrediction":
            self.target_dim = len(relation_idxs)
            index = PairIndex.from_triples(train_set_idx, columns=(0, 2, 1))
        elif form == "EntityPrediction":
            self.target_dim = len(entity_idxs)
            index = PairIndex.from_triples(train_set_idx)
        else:
            raise NotImplementedError(form)
        self.train_data = index.keys
        self.train_target = index.targets

    def __len__(self):
        assert len(self.train_data) == len(self.train_target)
        return len(self.train_data)

    def __getitem__(self, idx):
        y_vec = torch.zeros(self.target_dim)
        y_vec[self.train_target[idx]] = 1.0

        if self.label_smoothing_rate:
            y_vec = y_vec * (1 - self.label_smoothing_rate) + self.label_smoothing_rate / self.target_dim
        return self.train_data[idx], y_vec
