"""Small glue functions: model/dataloader construction, experiment storage.

Structured after ``literal-embeddings/src/static_train_utils.py`` and
``src/static_funcs.py``.
"""
import json
import os
from datetime import datetime
from typing import Optional

import numpy as np
from torch.utils.data import DataLoader

from src.dataset import KvsAll, collate_fn
from src.models import MODEL_MAP


def get_model(args, entity_dataset):
    """Instantiate one of RGCN / GATv2 / GraphSAGE as a plain ``BaseKGE`` model.

    Unlike dicee's ``intialize_model``, this does not go through dicee's
    ``MODEL_REGISTRY`` (these models are not registered there — see
    README.md for why) — it looks the class up in this project's own
    ``src.models.MODEL_MAP`` instead. Everything else about the resulting
    object (constructor signature, ``forward``/``forward_triples``/
    ``forward_k_vs_all`` contract) is identical to a dicee-registered model.
    """
    try:
        model_cls = MODEL_MAP[args.model]
    except KeyError as exc:
        raise ValueError(f"Unknown model '{args.model}'. Available: {sorted(MODEL_MAP)}") from exc

    model_args = vars(args).copy()
    model_args["num_entities"] = entity_dataset.num_entities
    model_args["num_relations"] = entity_dataset.num_relations
    model_args["train_set_idx"] = entity_dataset.train_set
    return model_cls(args=model_args)


def get_dataloader(args, entity_dataset) -> DataLoader:
    train_dataset = KvsAll(train_set_idx=entity_dataset.train_set, num_entities=entity_dataset.num_entities)
    return DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_core,
        collate_fn=collate_fn,
    )


def get_full_storage_path(args) -> str:
    if args.full_storage_path:
        return args.full_storage_path
    dataset_name = os.path.basename(args.dataset_dir.rstrip("/")) if args.dataset_dir else "unknown_dataset"
    if getattr(args, "test_runs", False):
        exp_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
        return f"{args.storage_path}/Test_runs/{exp_time}"
    return f"{args.storage_path}/{dataset_name}/{args.model}_{args.embedding_dim}"


def prepare_experiment_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_report(args, model, entity_dataset, eval_report: Optional[dict], runtime_seconds: float) -> None:
    total_params = sum(p.numel() for p in model.parameters())
    report = {
        "model": args.model,
        "embedding_dim": args.embedding_dim,
        "num_entities": entity_dataset.num_entities,
        "num_relations": entity_dataset.num_relations,
        "num_train_triples": len(entity_dataset.train_set),
        "num_params": total_params,
        "estimated_size_mb": total_params * 4 / (1024 ** 2),
        "runtime_seconds": runtime_seconds,
        "config": {k: v for k, v in vars(args).items() if k != "train_set_idx"},
    }
    path = os.path.join(args.full_storage_path, "report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=4, default=str)
    if eval_report:
        with open(os.path.join(args.full_storage_path, "eval_report.json"), "w") as f:
            json.dump(eval_report, f, indent=4)
