import copy
import json
from pathlib import Path
import time

import torch

from src.diagnostics import FitProbe
from src.evaluator import Evaluator
from src.kg_loader import load_kg
from src.static_funcs import (
    get_dataloader,
    get_full_storage_path,
    get_model,
    get_optimizer,
    load_initial_state,
    prepare_experiment_dir,
    save_model_artifacts,
    save_report,
    save_run_metadata,
)


def train_gnn_kge_model(args) -> dict:
    args.full_storage_path = get_full_storage_path(args)
    prepare_experiment_dir(args.full_storage_path)
    print(f"[{args.model}] storing experiment at {args.full_storage_path}")

    torch.manual_seed(args.random_seed)
    device = torch.device("cuda" if args.accelerator == "gpu" else "cpu")

    start_time = time.time()

    entity_dataset = load_kg(
        args.dataset_dir,
        vocab_from_all_splits=getattr(args, "vocab_from_all_splits", False),
    )

    model = get_model(args, entity_dataset)
    if getattr(args, "init_from", None):
        load_initial_state(args.init_from, model, entity_dataset)
    if getattr(args, "freeze_embeddings", False):
        if not getattr(args, "init_from", None):
            raise ValueError("--freeze_embeddings needs --init_from")
        model.node_emb.weight.requires_grad_(False)
        model.rel_emb.weight.requires_grad_(False)
    model = model.to(device)
    save_run_metadata(args, model, entity_dataset)
    train_dataloader = get_dataloader(args, entity_dataset)
    optimizer = get_optimizer(args, [p for p in model.parameters() if p.requires_grad])
    label_smoothing = float(getattr(args, "label_smoothing", 0.0) or 0.0)
    if args.eval_every and (
        entity_dataset.valid_set is None or not len(entity_dataset.valid_set)
    ):
        raise ValueError("--eval_every requires a nonempty validation split")
    one_vs_all = args.scoring_technique == "1vsAll"
    if one_vs_all:
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()

    eval_every = int(getattr(args, "eval_every", 0) or 0)
    patience = int(getattr(args, "early_stopping_patience", 0) or 0)
    val_args = copy.copy(args)
    val_args.eval_model = "val"
    val_evaluator = Evaluator(val_args)
    best_val_mrr, best_epoch, best_state, stale = -1.0, None, None, 0
    fit_probe = None
    if getattr(args, "fit_diagnostics", False):
        fit_probe = FitProbe(entity_dataset, eval_batch_size=val_args.eval_batch_size or args.batch_size)
        fit_probe.snapshot(model)  # reference for the first update norm

    validation_history = []

    def validate(epoch: int, train_objective) -> bool:
        """Record validation at this epoch, keep the best state; True when patience runs out."""
        nonlocal best_val_mrr, best_epoch, best_state, stale
        val_report = val_evaluator.eval(entity_dataset, model, during_training=True)
        val_mrr = val_report["Val"]["MRR"]
        record = {"epoch": epoch, **val_report["Val"]}
        if fit_probe is not None:
            record["fit"] = fit_probe(model, train_objective)
        validation_history.append(record)
        path = Path(args.full_storage_path) / "validation_history.json"
        path.write_text(json.dumps(validation_history, indent=2))
        model.train()
        if val_mrr > best_val_mrr:
            best_val_mrr, best_epoch, stale = val_mrr, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        print(
            f"[{args.model}] epoch {epoch} val MRR {val_mrr:.4f} "
            f"(best {best_val_mrr:.4f} @ {best_epoch})"
        )
        return bool(patience and stale >= patience)

    # A loaded starting point is itself a candidate: best epoch 0 means no validation gain.
    if eval_every and getattr(args, "init_from", None):
        validate(0, None)
    model.train()
    for epoch in range(args.num_epochs):
        epoch_losses = []
        for x_batch, y_batch in train_dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            # 1vsAll targets are one-hot rows; the answer tail feeds triple exclusion.
            query_tails = y_batch.argmax(dim=1) if one_vs_all else None
            scores = model(x_batch, query_tails)
            loss = loss_fn(scores, y_batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        if (epoch + 1) % max(1, args.num_epochs // 10) == 0 or epoch == 0:
            print(
                f"[{args.model}] epoch {epoch + 1}/{args.num_epochs} "
                f"loss {sum(epoch_losses) / len(epoch_losses):.4f}"
            )
        if eval_every and (epoch + 1) % eval_every == 0:
            if validate(epoch + 1, sum(epoch_losses) / len(epoch_losses)):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
        args.best_epoch = best_epoch

    model.eval()
    evaluator = Evaluator(args=args)
    eval_report = evaluator.eval(
        dataset=entity_dataset, trained_model=model, during_training=False
    )

    runtime_seconds = time.time() - start_time

    model.to("cpu")
    save_model_artifacts(args, model, entity_dataset)
    save_report(args, model, entity_dataset, eval_report, runtime_seconds)

    print(
        f"[{args.model}] finished in {runtime_seconds:.1f}s -> {args.full_storage_path}"
    )
    return eval_report
