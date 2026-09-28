import csv
import json
import os
from datetime import datetime
from typing import Optional

import torch
from torch.utils.data import DataLoader

from src.dataset import KvsAll, OneVsAll, add_reciprocal_triples, collate_fn
from src.models import MODEL_MAP


def get_model(args, entity_dataset):
    try:
        model_cls = MODEL_MAP[args.model]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model '{args.model}'. Available: {sorted(MODEL_MAP)}"
        ) from exc

    model_args = vars(args).copy()
    model_args["num_entities"] = entity_dataset.num_entities
    model_args["num_relations"] = entity_dataset.num_relations
    model_args["train_set_idx"] = entity_dataset.train_set
    return model_cls(args=model_args)


def get_dataloader(args, entity_dataset) -> DataLoader:
    triples = entity_dataset.train_set
    if getattr(args, "reciprocal", False):
        triples = add_reciprocal_triples(triples, entity_dataset.num_relations)
    if args.scoring_technique == "1vsAll":
        train_dataset = OneVsAll(train_set_idx=triples)
    else:
        train_dataset = KvsAll(
            train_set_idx=triples, num_entities=entity_dataset.num_entities
        )
    return DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_core,
        collate_fn=collate_fn,
    )


_OPTIMIZERS = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "SGD": torch.optim.SGD,
    "NAdam": torch.optim.NAdam,
    "Adagrad": torch.optim.Adagrad,
    "ASGD": torch.optim.ASGD,
}


def smooth_labels(
    y_batch: torch.Tensor, label_smoothing: float, num_entities: int
) -> torch.Tensor:
    if not label_smoothing:
        return y_batch
    return y_batch * (1.0 - label_smoothing) + label_smoothing / num_entities


def get_optimizer(args, params) -> torch.optim.Optimizer:
    if args.optim == "Adopt":
        from dicee.models.adopt import ADOPT

        return ADOPT(params, lr=args.lr)
    try:
        optim_cls = _OPTIMIZERS[args.optim]
    except KeyError as exc:
        raise ValueError(
            f"Unknown optimizer '{args.optim}'. Available: {sorted(_OPTIMIZERS)}"
        ) from exc
    return optim_cls(params, lr=args.lr, weight_decay=args.weight_decay)


def get_full_storage_path(args) -> str:
    if args.full_storage_path:
        return args.full_storage_path
    dataset_name = (
        os.path.basename(args.dataset_dir.rstrip("/"))
        if args.dataset_dir
        else "unknown_dataset"
    )
    if getattr(args, "test_runs", False):
        exp_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
        return f"{args.storage_path}/Test_runs/{exp_time}"
    return f"{args.storage_path}/{dataset_name}/{args.model}_{args.embedding_dim}"


def prepare_experiment_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_report(
    args, model, entity_dataset, eval_report: Optional[dict], runtime_seconds: float
) -> None:
    total_params = sum(p.numel() for p in model.parameters())
    report = {
        "model": args.model,
        "embedding_dim": args.embedding_dim,
        "num_entities": entity_dataset.num_entities,
        "num_relations": entity_dataset.num_relations,
        "num_train_triples": len(entity_dataset.train_set),
        "num_params": total_params,
        "estimated_size_mb": total_params * 4 / (1024**2),
        "runtime_seconds": runtime_seconds,
        "config": {k: v for k, v in vars(args).items() if k != "train_set_idx"},
    }
    path = os.path.join(args.full_storage_path, "report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=4, default=str)
    if eval_report:
        with open(os.path.join(args.full_storage_path, "eval_report.json"), "w") as f:
            json.dump(eval_report, f, indent=4)


def _write_idx_csv(path: str, name_to_idx: dict, column: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["", column])
        for name, idx in sorted(name_to_idx.items(), key=lambda kv: kv[1]):
            writer.writerow([idx, name])


def save_model_artifacts(args, model, entity_dataset) -> None:
    torch.save(model.state_dict(), os.path.join(args.full_storage_path, "model.pt"))
    _write_idx_csv(
        os.path.join(args.full_storage_path, "entity_to_idx.csv"),
        entity_dataset.entity_to_idx,
        "entity",
    )
    _write_idx_csv(
        os.path.join(args.full_storage_path, "relation_to_idx.csv"),
        entity_dataset.relation_to_idx,
        "relation",
    )
