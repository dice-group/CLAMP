"""Filtered ranking and tie policies, copied from dicee.

Source: dice-group/dice-embeddings @ 07e372ddf (``dicee/evaluation/_filtering.py``).
``FilteredRanker`` is identical to dicee's except that ``bounds_batch`` always takes
the torch path (dicee dispatches to an optional Triton kernel on CUDA, which computes
the same two integers), and ``-np.inf`` replaces ``-np.Inf``, which NumPy 2 removed. ``tests/test_dicee_parity.py`` compares it with the installed
dicee.
"""
from collections.abc import Iterable, Mapping
from typing import Dict, List, Optional, SupportsInt, Tuple

import numpy as np
import torch

TIE_POLICIES = ("sort", "optimistic", "random", "pessimistic")


def evaluation_tie_options(args) -> Dict:
    """Read evaluation options, including configurations saved before tie policies."""
    settings = args if isinstance(args, Mapping) else vars(args)
    return {
        "tie_policy": settings.get("eval_tie_policy", "sort"),
        "tie_seed": settings.get("random_seed", 0),
    }


class FilteredRanker:
    """One evaluation's tie policy and independent CPU random stream.

    Ties use exact score equality. Random ranks are uniform integers between
    the optimistic and pessimistic ranks, inclusive. Recreate this object for
    each evaluation to replay its random stream without changing model RNGs.
    A supplied generator allows callers to continue the stream across chunks.
    """

    def __init__(self, tie_policy: str = "sort", tie_seed: int = 0,
                 generator: Optional[torch.Generator] = None):
        if tie_policy not in TIE_POLICIES:
            raise ValueError(f"Unknown tie policy {tie_policy!r}; choose from {TIE_POLICIES}")
        self.tie_policy = tie_policy
        self.generator = generator
        if tie_policy == "random":
            if generator is None:
                self.generator = torch.Generator(device="cpu").manual_seed(tie_seed)
            elif generator.device.type != "cpu":
                raise ValueError("Tie-breaking requires a CPU torch.Generator")

    def rank(self, predictions: torch.Tensor, target_idx: SupportsInt,
             filter_indices: Iterable[int], exclude_target: bool = True) -> int:
        target_idx = int(target_idx)
        filtered = predictions.clone()
        filters = set(filter_indices)
        if exclude_target:
            filters.discard(target_idx)
        if filters:
            filtered[list(filters)] = -np.inf

        if self.tie_policy == "sort":
            # Preserve DICE's original per-vector torch.sort ordering.
            order = torch.sort(filtered, descending=True).indices
            return int(torch.where(order == target_idx)[0].item()) + 1

        # An explicit mask also excludes filtered candidates when the target
        # itself has score -Inf. Masking scores alone would count false ties.
        eligible = torch.ones_like(predictions, dtype=torch.bool)
        if filters:
            eligible[list(filters)] = False
        eligible[target_idx] = False
        other_scores = filtered[eligible]
        target_score = filtered[target_idx]
        if torch.isnan(target_score) or torch.isnan(other_scores).any():
            raise ValueError("Cannot rank NaN prediction scores")
        optimistic = 1 + int((other_scores > target_score).sum().item())
        if self.tie_policy == "optimistic":
            return optimistic
        tied = int((other_scores == target_score).sum().item())
        if self.tie_policy == "pessimistic":
            return optimistic + tied
        if tied == 0:
            return optimistic
        return optimistic + int(torch.randint(tied + 1, (), generator=self.generator).item())

    def rank_batch(self, predictions: torch.Tensor, target_indices: Iterable[SupportsInt],
                   filter_indices_list: List[List[int]]) -> List[int]:
        if self.tie_policy != "sort":
            return self.ranks_from_bounds(self.bounds_batch(predictions, target_indices, filter_indices_list))

        # Existing KvsAll/ensemble evaluators sorted whole batches. Keep that
        # operation intact, including its device-dependent ordering of ties.
        filtered = predictions.clone()
        for i, (target, filters) in enumerate(zip(target_indices, filter_indices_list)):
            target = int(target)
            if len(filters):
                filtered[i, list(filters)] = -np.inf
            filtered[i, target] = predictions[i, target]
        order = torch.sort(filtered, dim=1, descending=True).indices
        return [int(torch.where(row == int(target))[0].item()) + 1
                for row, target in zip(order, target_indices)]

    def bounds_batch(self, predictions, target_indices, filter_indices_list, row_indices=None) -> List[Tuple[int, int]]:
        """Compute optimistic ranks and tie counts together on the score device.

        Only two integers per query leave the GPU. Sampling is deliberately
        separate so grouped graph inference can restore original query order
        before consuming the independent random-tie stream.
        """
        if row_indices is not None:
            predictions = predictions.index_select(0, torch.as_tensor(row_indices, device=predictions.device, dtype=torch.long))
        targets = torch.as_tensor(target_indices, device=predictions.device, dtype=torch.long)
        if not len(targets):
            return []
        eligible = torch.ones_like(predictions, dtype=torch.bool)
        rows, columns = [], []
        for row, filters in enumerate(filter_indices_list):
            columns.extend(filters)
            rows.extend([row] * len(filters))
        if columns:
            eligible[torch.tensor(rows, device=predictions.device, dtype=torch.long),
                     torch.tensor(columns, device=predictions.device, dtype=torch.long)] = False
        eligible[torch.arange(len(targets), device=predictions.device), targets] = False
        target_scores = predictions.gather(1, targets[:, None])
        if target_scores.isnan().any() or (predictions.isnan() & eligible).any():
            raise ValueError('Cannot rank NaN prediction scores')
        better = ((predictions > target_scores) & eligible).sum(1) + 1
        tied = ((predictions == target_scores) & eligible).sum(1)
        return [(int(b), int(t)) for b, t in torch.stack((better, tied), 1).cpu().tolist()]

    def ranks_from_bounds(self, bounds: Iterable[Tuple[int, int]]) -> List[int]:
        ranks = []
        for optimistic, tied in bounds:
            if self.tie_policy == 'pessimistic':
                ranks.append(optimistic + tied)
            elif self.tie_policy == 'random' and tied:
                ranks.append(optimistic + int(torch.randint(tied + 1, (), generator=self.generator)))
            else:
                ranks.append(optimistic)
        return ranks
