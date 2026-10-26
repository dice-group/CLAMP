from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.filtering import FilteredRanker, evaluation_tie_options
from src.models.base import KGModel


class Evaluator:
    def __init__(self, args):
        self.args = args
        FilteredRanker(**evaluation_tie_options(args))  # rejects an unknown tie policy now
        self.report: Dict[str, Dict[str, float]] = {}

    def eval(
        self, dataset, trained_model, during_training: bool = False
    ) -> Optional[Dict]:
        eval_mode = self._resolve_eval_mode()
        if eval_mode is None:
            return None

        er_vocab = self._build_er_vocab(dataset, include_test=not during_training)
        batch_size = int(
            getattr(self.args, "eval_batch_size", None)
            or getattr(self.args, "batch_size", 1024)
        )

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
                trained_model, loader, er_vocab
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
        if not during_training:
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
    ) -> Dict[str, float]:
        """Filtered tail ranking; the split holds reciprocal triples, so it covers heads too."""
        model.eval()
        device = next(model.parameters()).device
        score_kwargs = {}
        if type(model).forward_k_vs_all is KGModel.forward_k_vs_all:
            # Weights and graph stay fixed for the whole evaluation split.
            score_kwargs["entity_repr"] = model.encode(*model.graph)
        ranker = FilteredRanker(**evaluation_tie_options(self.args))
        ranks: List[int] = []
        for e1_idx, r_idx, e2_idx in data_loader:
            predictions = model.forward_k_vs_all(
                torch.stack((e1_idx, r_idx), dim=1).to(device), **score_kwargs
            )
            if not torch.isfinite(predictions).all():
                raise ValueError("Non-finite predictions cannot be ranked")
            filter_indices = [
                er_vocab.get(key, []) for key in zip(e1_idx.tolist(), r_idx.tolist())
            ]
            ranks.extend(ranker.rank_batch(predictions, e2_idx.to(device), filter_indices))

        if not ranks:
            return {"MRR": 0.0, "MR": 0.0, "H@1": 0.0, "H@3": 0.0, "H@10": 0.0}
        ranks = torch.tensor(ranks, dtype=torch.float64)
        return {
            "MRR": torch.reciprocal(ranks).mean().item(),
            "MR": ranks.mean().item(),
            "H@1": (ranks <= 1).double().mean().item(),
            "H@3": (ranks <= 3).double().mean().item(),
            "H@10": (ranks <= 10).double().mean().item(),
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
