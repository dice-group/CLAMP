"""Filtered link-prediction evaluator (MRR, MR, Hits@1/3/10).

Adapted from ``literal-embeddings/src/evaluator.py``: standard filtered
ranking protocol (Bordes et al. 2013) — for each test triple ``(h, r, t)``,
rank ``t`` against all entities under ``(h, r, ?)`` after masking out every
other known-true tail for that ``(h, r)`` pair, *and* rank ``h`` against all
entities under ``(?, r, t)`` after masking out every other known-true head
for that ``(r, t)`` pair. Ranks from both directions are pooled into the
same MRR/MR/Hits@k averages, matching the standard combined head+tail
evaluation protocol reported in the KGE literature (a tail-only evaluator
is not comparable to published numbers).
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class Evaluator:
    """Runs filtered link-prediction evaluation for ``forward_k_vs_all`` models."""

    def __init__(self, args):
        self.args = args
        self.report: Dict[str, Dict[str, float]] = {}

    def eval(self, dataset, trained_model, during_training: bool = False) -> Optional[Dict]:
        eval_mode = self._resolve_eval_mode()
        if eval_mode is None:
            return None

        er_vocab = self._build_er_vocab(dataset, include_test=not during_training)
        re_vocab = self._build_re_vocab(dataset, include_test=not during_training)
        batch_size = int(getattr(self.args, "batch_size", 1024))

        train_loader = self._build_loader(dataset.train_set, batch_size) if "train" in eval_mode else None
        val_loader = self._build_loader(dataset.valid_set, batch_size) if "val" in eval_mode else None
        test_loader = self._build_loader(dataset.test_set, batch_size) if "test" in eval_mode and not during_training else None

        raw_report = {}
        split_times = {}
        for split_name, loader in (("Train", train_loader), ("Val", val_loader), ("Test", test_loader)):
            if loader is None:
                continue
            t0 = time.perf_counter()
            raw_report[split_name] = self._evaluate_split(trained_model, loader, er_vocab, re_vocab)
            split_times[split_name] = time.perf_counter() - t0

        if not during_training:
            for split_name, scores in raw_report.items():
                print(f"{split_name} Scores")
                print(scores)
            print("Evaluation Runtime (s)", {k: round(v, 4) for k, v in split_times.items()})

        self.report = raw_report
        if not during_training and getattr(self.args, "save_experiment", True):
            full_storage_path = getattr(self.args, "full_storage_path", None)
            if full_storage_path:
                os.makedirs(full_storage_path, exist_ok=True)
                with open(os.path.join(full_storage_path, "eval_report.json"), "w") as f:
                    json.dump(self.report, f, indent=4)
        return dict(self.report)

    @torch.no_grad()
    def _evaluate_split(self, model, data_loader, er_vocab: Dict[Tuple[int, int], List[int]],
                        re_vocab: Dict[Tuple[int, int], List[int]]) -> Dict[str, float]:
        model.eval()
        device = next(model.parameters()).device
        rank_sum = 0.0
        reciprocal_rank_sum = 0.0
        hits1 = hits3 = hits10 = 0
        num_rankings = 0

        for e1_idx, r_idx, e2_idx in data_loader:
            e1_idx, r_idx, e2_idx = e1_idx.to(device), r_idx.to(device), e2_idx.to(device)

            tail_predictions = model.forward_k_vs_all(torch.stack((e1_idx, r_idx), dim=1))
            tail_ranks = self._rank_batch(tail_predictions, e2_idx,
                                          keys=list(zip(e1_idx.tolist(), r_idx.tolist())), vocab=er_vocab)

            head_predictions = model.forward_k_vs_all_head(torch.stack((r_idx, e2_idx), dim=1))
            head_ranks = self._rank_batch(head_predictions, e1_idx,
                                          keys=list(zip(r_idx.tolist(), e2_idx.tolist())), vocab=re_vocab)

            for ranks in (tail_ranks, head_ranks):
                num_rankings += ranks.numel()
                rank_sum += ranks.sum().item()
                reciprocal_rank_sum += torch.reciprocal(ranks).sum().item()
                hits1 += (ranks <= 1).sum().item()
                hits3 += (ranks <= 3).sum().item()
                hits10 += (ranks <= 10).sum().item()

        if num_rankings == 0:
            return {"MRR": 0.0, "MR": 0.0, "H@1": 0.0, "H@3": 0.0, "H@10": 0.0}
        return {
            "MRR": reciprocal_rank_sum / num_rankings,
            "MR": rank_sum / num_rankings,
            "H@1": hits1 / num_rankings,
            "H@3": hits3 / num_rankings,
            "H@10": hits10 / num_rankings,
        }

    @staticmethod
    def _rank_batch(predictions: torch.Tensor, target_idx: torch.Tensor,
                    keys: List[Tuple[int, int]], vocab: Dict[Tuple[int, int], List[int]]) -> torch.Tensor:
        """Filtered, average-rank-tie-broken rank of ``target_idx`` in each row of ``predictions``.

        ``keys[j]`` indexes ``vocab`` to find every other entity known-true for
        that row (besides ``target_idx[j]`` itself), which is masked out before
        ranking (standard filtered-ranking protocol, Bordes et al. 2013).
        """
        if not torch.isfinite(predictions).all():
            raise ValueError("Non-finite predictions cannot be ranked")

        batch_size = predictions.size(0)
        target_scores = predictions.gather(1, target_idx.view(-1, 1)).squeeze(1)

        row_ids, col_ids = [], []
        for j in range(batch_size):
            target = int(target_idx[j].item())
            for other in vocab.get(keys[j], []):
                if other != target:
                    row_ids.append(j)
                    col_ids.append(other)
        if row_ids:
            rows = torch.as_tensor(row_ids, dtype=torch.long, device=predictions.device)
            cols = torch.as_tensor(col_ids, dtype=torch.long, device=predictions.device)
            predictions[rows, cols] = float("-inf")
        predictions.scatter_(1, target_idx.view(-1, 1), target_scores.view(-1, 1))

        # Average-rank tie-breaking: a strict ">" comparison alone would rank a
        # degenerate model (identical score for every entity) at 1 on every query
        # instead of near-chance. Split credit for ties, excluding the target's own
        # filtered-back score (Sun et al. 2019, RotatE, evaluation-protocol appendix).
        strictly_greater = (predictions > target_scores.unsqueeze(1)).sum(dim=1)
        tied_with_target = (predictions == target_scores.unsqueeze(1)).sum(dim=1) - 1  # exclude target itself
        return strictly_greater.to(torch.float32) + 1.0 + 0.5 * tied_with_target.to(torch.float32)

    def _resolve_eval_mode(self) -> Optional[str]:
        eval_model = getattr(self.args, "eval_model", "train_val_test")
        if eval_model in (None, "None"):
            return None
        return eval_model

    @staticmethod
    def _build_loader(triples, batch_size: int) -> Optional[DataLoader]:
        if triples is None or len(triples) == 0:
            return None
        arr = np.asarray(triples)
        heads = torch.as_tensor(arr[:, 0], dtype=torch.long)
        rels = torch.as_tensor(arr[:, 1], dtype=torch.long)
        tails = torch.as_tensor(arr[:, 2], dtype=torch.long)
        return DataLoader(TensorDataset(heads, rels, tails), batch_size=batch_size, shuffle=False)

    @staticmethod
    def _build_er_vocab(dataset, include_test: bool = True) -> Dict[Tuple[int, int], List[int]]:
        vocab: Dict[Tuple[int, int], set] = {}
        for attr in (("train_set", "valid_set", "test_set") if include_test else ("train_set", "valid_set")):
            triples = getattr(dataset, attr, None)
            if triples is None or len(triples) == 0:
                continue
            for h, r, t in np.asarray(triples)[:, :3]:
                vocab.setdefault((int(h), int(r)), set()).add(int(t))
        return {k: sorted(v) for k, v in vocab.items()}

    @staticmethod
    def _build_re_vocab(dataset, include_test: bool = True) -> Dict[Tuple[int, int], List[int]]:
        """``(r, t) -> sorted[h, ...]`` — the head-prediction mirror of ``_build_er_vocab``."""
        vocab: Dict[Tuple[int, int], set] = {}
        for attr in (("train_set", "valid_set", "test_set") if include_test else ("train_set", "valid_set")):
            triples = getattr(dataset, attr, None)
            if triples is None or len(triples) == 0:
                continue
            for h, r, t in np.asarray(triples)[:, :3]:
                vocab.setdefault((int(r), int(t)), set()).add(int(h))
        return {k: sorted(v) for k, v in vocab.items()}
