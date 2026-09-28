from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class Evaluator:
    def __init__(self, args):
        self.args = args
        self.report: Dict[str, Dict[str, float]] = {}

    def eval(
        self, dataset, trained_model, during_training: bool = False
    ) -> Optional[Dict]:
        eval_mode = self._resolve_eval_mode()
        if eval_mode is None:
            return None

        er_vocab = self._build_er_vocab(dataset, include_test=not during_training)
        re_vocab = self._build_re_vocab(dataset, include_test=not during_training)
        batch_size = int(getattr(self.args, "batch_size", 1024))

        raw_report = {}
        split_times = {}
        for mode, split_name, attr in (
            ("train", "Train", "train_set"),
            ("val", "Val", "valid_set"),
            ("test", "Test", "test_set"),
        ):
            if mode not in eval_mode or (mode == "test" and during_training):
                continue
            loader = self._build_loader(getattr(dataset, attr), batch_size)
            if loader is None:
                continue
            t0 = time.perf_counter()
            raw_report[split_name] = self._evaluate_split(
                trained_model, loader, er_vocab, re_vocab
            )
            split_times[split_name] = time.perf_counter() - t0

        if not during_training:
            for split_name, scores in raw_report.items():
                print(f"{split_name} Scores")
                print(scores)
            print(
                "Evaluation Runtime (s)",
                {k: round(v, 4) for k, v in split_times.items()},
            )

        self.report = raw_report
        if not during_training and getattr(self.args, "save_experiment", True):
            full_storage_path = getattr(self.args, "full_storage_path", None)
            if full_storage_path:
                os.makedirs(full_storage_path, exist_ok=True)
                with open(
                    os.path.join(full_storage_path, "eval_report.json"), "w"
                ) as f:
                    json.dump(self.report, f, indent=4)
        return dict(self.report)

    @torch.no_grad()
    def _evaluate_split(
        self,
        model,
        data_loader,
        er_vocab: Dict[Tuple[int, int], List[int]],
        re_vocab: Dict[Tuple[int, int], List[int]],
    ) -> Dict[str, float]:
        model.eval()
        device = next(model.parameters()).device
        rank_sum = 0.0
        reciprocal_rank_sum = 0.0
        hits1 = hits3 = hits10 = 0
        num_rankings = 0
        tie_policy = getattr(self.args, "eval_tie_policy", "average")

        for e1_idx, r_idx, e2_idx in data_loader:
            e1_idx, r_idx, e2_idx = (
                e1_idx.to(device),
                r_idx.to(device),
                e2_idx.to(device),
            )

            tail_predictions = model.forward_k_vs_all(
                torch.stack((e1_idx, r_idx), dim=1)
            )
            tail_ranks = self._rank_batch(
                tail_predictions,
                e2_idx,
                keys=list(zip(e1_idx.tolist(), r_idx.tolist())),
                vocab=er_vocab,
                tie_policy=tie_policy,
            )

            head_predictions = model.forward_k_vs_all_head(
                torch.stack((r_idx, e2_idx), dim=1)
            )
            head_ranks = self._rank_batch(
                head_predictions,
                e1_idx,
                keys=list(zip(r_idx.tolist(), e2_idx.tolist())),
                vocab=re_vocab,
                tie_policy=tie_policy,
            )

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
    def _rank_batch(
        predictions: torch.Tensor,
        target_idx: torch.Tensor,
        keys: List[Tuple[int, int]],
        vocab: Dict[Tuple[int, int], List[int]],
        tie_policy: str = "average",
    ) -> torch.Tensor:
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

        if tie_policy == "sort":
            order = torch.sort(predictions, dim=1, descending=True).indices
            positions = (order == target_idx.view(-1, 1)).nonzero()[:, 1]
            return positions.to(torch.float32) + 1.0

        strictly_greater = (predictions > target_scores.unsqueeze(1)).sum(dim=1)
        # Exclude the target itself from the tie count.
        tied_with_target = (predictions == target_scores.unsqueeze(1)).sum(dim=1) - 1
        return (
            strictly_greater.to(torch.float32)
            + 1.0
            + 0.5 * tied_with_target.to(torch.float32)
        )

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
        return DataLoader(
            TensorDataset(heads, rels, tails), batch_size=batch_size, shuffle=False
        )

    @staticmethod
    def _build_er_vocab(
        dataset, include_test: bool = True
    ) -> Dict[Tuple[int, int], List[int]]:
        vocab: Dict[Tuple[int, int], set] = {}
        splits = ["train_set", "valid_set"]
        if include_test:
            splits.append("test_set")
        for attr in splits:
            triples = getattr(dataset, attr, None)
            if triples is None or len(triples) == 0:
                continue
            for h, r, t in np.asarray(triples)[:, :3]:
                vocab.setdefault((int(h), int(r)), set()).add(int(t))
        return {k: sorted(v) for k, v in vocab.items()}

    @staticmethod
    def _build_re_vocab(
        dataset, include_test: bool = True
    ) -> Dict[Tuple[int, int], List[int]]:
        vocab: Dict[Tuple[int, int], set] = {}
        splits = ["train_set", "valid_set"]
        if include_test:
            splits.append("test_set")
        for attr in splits:
            triples = getattr(dataset, attr, None)
            if triples is None or len(triples) == 0:
                continue
            for h, r, t in np.asarray(triples)[:, :3]:
                vocab.setdefault((int(r), int(t)), set()).add(int(h))
        return {k: sorted(v) for k, v in vocab.items()}
