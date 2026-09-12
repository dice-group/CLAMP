"""Filtered link-prediction evaluator (MRR, MR, Hits@1/3/10).

Adapted from ``literal-embeddings/src/evaluator.py``: standard filtered
ranking protocol (Bordes et al. 2013) — for each test triple ``(h, r, t)``,
rank ``t`` against all entities under ``(h, r, ?)`` after masking out every
other known-true tail for that ``(h, r)`` pair.
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

        er_vocab = self._build_er_vocab(dataset)
        batch_size = int(getattr(self.args, "batch_size", 1024))

        train_loader = self._build_loader(dataset.train_set, batch_size) if "train" in eval_mode else None
        val_loader = self._build_loader(dataset.valid_set, batch_size) if "val" in eval_mode else None
        test_loader = self._build_loader(dataset.test_set, batch_size) if "test" in eval_mode else None

        raw_report = {}
        split_times = {}
        for split_name, loader in (("Train", train_loader), ("Val", val_loader), ("Test", test_loader)):
            if loader is None:
                continue
            t0 = time.perf_counter()
            raw_report[split_name] = self._evaluate_split(trained_model, loader, er_vocab)
            split_times[split_name] = time.perf_counter() - t0

        if not during_training:
            for split_name, scores in raw_report.items():
                print(f"{split_name} Scores")
                print(scores)
            print("Evaluation Runtime (s)", {k: round(v, 4) for k, v in split_times.items()})

        self.report = raw_report
        if not during_training:
            full_storage_path = getattr(self.args, "full_storage_path", None)
            if full_storage_path:
                os.makedirs(full_storage_path, exist_ok=True)
                with open(os.path.join(full_storage_path, "eval_report.json"), "w") as f:
                    json.dump(self.report, f, indent=4)
        return dict(self.report)

    @torch.no_grad()
    def _evaluate_split(self, model, data_loader, er_vocab: Dict[Tuple[int, int], List[int]]) -> Dict[str, float]:
        model.eval()
        device = next(model.parameters()).device
        rank_sum = 0.0
        reciprocal_rank_sum = 0.0
        hits1 = hits3 = hits10 = 0
        num_triples = 0

        for e1_idx, r_idx, e2_idx in data_loader:
            e1_idx, r_idx, e2_idx = e1_idx.to(device), r_idx.to(device), e2_idx.to(device)
            batch_size = e1_idx.size(0)
            num_triples += batch_size

            hr = torch.stack((e1_idx, r_idx), dim=1)
            predictions = model.forward_k_vs_all(hr)

            target_scores = predictions.gather(1, e2_idx.view(-1, 1)).squeeze(1)

            row_ids, col_ids = [], []
            for j in range(batch_size):
                key = (int(e1_idx[j].item()), int(r_idx[j].item()))
                for other_tail in er_vocab.get(key, []):
                    if other_tail != int(e2_idx[j].item()):
                        row_ids.append(j)
                        col_ids.append(other_tail)
            if row_ids:
                rows = torch.as_tensor(row_ids, dtype=torch.long, device=predictions.device)
                cols = torch.as_tensor(col_ids, dtype=torch.long, device=predictions.device)
                predictions[rows, cols] = float("-inf")
            predictions.scatter_(1, e2_idx.view(-1, 1), target_scores.view(-1, 1))

            # Average-rank tie-breaking: a strict ">" comparison alone would count every
            # tied (non-target) entity as ranked *below* the target, so a degenerate model
            # that outputs an identical score for every entity (a real failure mode we hit
            # in practice — a GNN encoder can collapse to one representation for all nodes
            # under an unstable learning rate) would score a fraudulent rank of 1 on every
            # query instead of the near-chance rank such a model actually deserves. Splitting
            # the credit for ties (excluding the target's own filtered-back score) is the
            # standard fix (see Sun et al. 2019, RotatE, Appendix on evaluation protocol).
            strictly_greater = (predictions > target_scores.unsqueeze(1)).sum(dim=1)
            tied_with_target = (predictions == target_scores.unsqueeze(1)).sum(dim=1) - 1  # exclude target itself
            ranks = strictly_greater.to(torch.float32) + 1.0 + 0.5 * tied_with_target.to(torch.float32)
            rank_sum += ranks.to(torch.float32).sum().item()
            reciprocal_rank_sum += torch.reciprocal(ranks.to(torch.float32)).sum().item()
            hits1 += (ranks <= 1).sum().item()
            hits3 += (ranks <= 3).sum().item()
            hits10 += (ranks <= 10).sum().item()

        if num_triples == 0:
            return {"MRR": 0.0, "MR": 0.0, "H@1": 0.0, "H@3": 0.0, "H@10": 0.0}
        return {
            "MRR": reciprocal_rank_sum / num_triples,
            "MR": rank_sum / num_triples,
            "H@1": hits1 / num_triples,
            "H@3": hits3 / num_triples,
            "H@10": hits10 / num_triples,
        }

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
    def _build_er_vocab(dataset) -> Dict[Tuple[int, int], List[int]]:
        vocab: Dict[Tuple[int, int], set] = {}
        for attr in ("train_set", "valid_set", "test_set"):
            triples = getattr(dataset, attr, None)
            if triples is None or len(triples) == 0:
                continue
            for h, r, t in np.asarray(triples)[:, :3]:
                vocab.setdefault((int(h), int(r)), set()).add(int(t))
        return {k: sorted(v) for k, v in vocab.items()}
