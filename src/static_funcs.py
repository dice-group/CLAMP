import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Optional

import torch
from torch.utils.data import DataLoader

from src.dataset_classes import KvsAll, OnevsAllDataset
from src.models import model_class


def get_model(args, entity_dataset):
    model_args = vars(args).copy()
    model_args["num_entities"] = entity_dataset.num_entities
    model_args["num_relations"] = entity_dataset.num_relations
    model_args["train_set_idx"] = entity_dataset.train_set
    return model_class(model_args)(args=model_args)


def get_dataloader(args, entity_dataset) -> DataLoader:
    # The train set already holds the reciprocal triples (see load_kg).
    if args.scoring_technique == "1vsAll":
        train_dataset = OnevsAllDataset(
            train_set_idx=entity_dataset.train_set,
            entity_idxs=entity_dataset.entity_to_idx,
        )
    else:
        train_dataset = KvsAll(
            train_set_idx=entity_dataset.train_set,
            entity_idxs=entity_dataset.entity_to_idx,
            relation_idxs=entity_dataset.relation_to_idx,
            form="EntityPrediction",
            label_smoothing_rate=float(getattr(args, "label_smoothing", 0.0) or 0.0),
        )
    # Own generator: batch order depends on the seed only, not on RNG use during model construction.
    return DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(args.random_seed),
    )


_OPTIMIZERS = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "SGD": torch.optim.SGD,
    "NAdam": torch.optim.NAdam,
    "Adagrad": torch.optim.Adagrad,
    "ASGD": torch.optim.ASGD,
}


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


STORAGE_ROOT = "Experiments"


def get_full_storage_path(args) -> str:
    if args.full_storage_path:
        return args.full_storage_path
    dataset_name = (
        os.path.basename(args.dataset_dir.rstrip("/"))
        if args.dataset_dir
        else "unknown_dataset"
    )
    return f"{STORAGE_ROOT}/{dataset_name}/{args.model}_{args.embedding_dim}"


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


def _git(*cmd) -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        return subprocess.run(["git", *cmd], cwd=root, capture_output=True, text=True).stdout
    except OSError:
        return ""


def save_run_metadata(args, model, entity_dataset) -> None:
    """Code revision, environment and initial-state checksums, written before training."""
    import torch_geometric

    def digest(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()[:16]

    params = b"".join(p.detach().cpu().numpy().tobytes() for p in model.state_dict().values())
    vocab = json.dumps([sorted(entity_dataset.entity_to_idx.items()),
                        sorted(entity_dataset.relation_to_idx.items())]).encode()
    meta = {
        "git_revision": _git("rev-parse", "HEAD").strip(),
        "git_diff_sha256": digest(_git("diff", "HEAD").encode()),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "cuda_device": torch.cuda.get_device_name() if args.accelerator == "gpu" else None,
        "initial_state_sha256": digest(params),
        "vocab_sha256": digest(vocab),
        "data_order_seed": args.random_seed,
    }
    with open(os.path.join(args.full_storage_path, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=4)


def _read_idx_csv(path: str) -> dict:
    with open(path, newline="") as f:
        rows = list(csv.reader(f))[1:]
    return {name: int(idx) for idx, name in rows}


def load_initial_state(path: str, model, entity_dataset) -> None:
    """Copy entity and relation tables (and any other matching tensors) from a saved model.pt.

    The checkpoint's vocabulary files must match this dataset's indices. Parameters absent from the
    checkpoint, such as encoder layers and gates, keep their fresh initialization.
    """
    folder = os.path.dirname(os.path.abspath(path))
    for name, vocab in (("entity_to_idx.csv", entity_dataset.entity_to_idx),
                        ("relation_to_idx.csv", entity_dataset.relation_to_idx)):
        saved = os.path.join(folder, name)
        if not os.path.exists(saved):
            raise FileNotFoundError(f"--init_from needs {saved} to check the vocabulary")
        if _read_idx_csv(saved) != dict(vocab):
            raise ValueError(f"{saved} does not match this dataset's vocabulary")
    state = torch.load(path, map_location="cpu")
    # The propagation graph comes from the dataset, not the checkpoint.
    state.pop("gnn_train_triples", None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or {"node_emb.weight", "rel_emb.weight"} & set(missing):
        raise ValueError(f"Incompatible checkpoint: missing {missing}, unexpected {unexpected}")
    print(f"[init] loaded {path}; kept fresh: {missing}")
