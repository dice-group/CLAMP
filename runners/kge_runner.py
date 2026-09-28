import copy
import time

import torch

from src.evaluator import Evaluator
from src.kg_loader import load_kg
from src.static_funcs import (
    get_dataloader,
    get_full_storage_path,
    get_model,
    get_optimizer,
    prepare_experiment_dir,
    save_model_artifacts,
    save_report,
    smooth_labels,
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

    model = get_model(args, entity_dataset).to(device)
    train_dataloader = get_dataloader(args, entity_dataset)
    optimizer = get_optimizer(args, model.parameters())
    label_smoothing = float(getattr(args, "label_smoothing", 0.0) or 0.0)
    num_entities = entity_dataset.num_entities
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

    model.train()
    for epoch in range(args.num_epochs):
        epoch_losses = []
        for x_batch, y_batch in train_dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            query_tails = y_batch if one_vs_all else None
            scores = model(x_batch, query_tails)
            if not one_vs_all:
                y_batch = smooth_labels(y_batch, label_smoothing, num_entities)
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
            val_report = val_evaluator.eval(entity_dataset, model, during_training=True)
            val_mrr = val_report["Val"]["MRR"]
            model.train()
            if val_mrr > best_val_mrr:
                best_val_mrr, best_epoch, stale = val_mrr, epoch + 1, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            print(
                f"[{args.model}] epoch {epoch + 1} val MRR {val_mrr:.4f} "
                f"(best {best_val_mrr:.4f} @ {best_epoch})"
            )
            if patience and stale >= patience:
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

    if args.save_experiment:
        model.to("cpu")
        save_model_artifacts(args, model, entity_dataset)
        save_report(args, model, entity_dataset, eval_report, runtime_seconds)

    print(
        f"[{args.model}] finished in {runtime_seconds:.1f}s -> {args.full_storage_path}"
    )
    return eval_report
