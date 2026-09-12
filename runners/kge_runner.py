"""Training entry point: load a KG via dicee, train one GNN-KGE model, evaluate.

Structured after ``literal-embeddings/runners/kge_runner.py``: load the
knowledge graph with dicee's own ``KG``/``read_or_load_kg``, then hand off to
a plain ``lightning.pytorch.Trainer``. Unlike literal-embeddings this project
does not need a custom ``LightningModule`` wrapper: every model here extends
dicee's ``BaseKGE``, which already *is* a ``lightning.pytorch.LightningModule``
(``BaseKGELightning`` implements ``training_step``/``configure_optimizers``),
so the model can be handed to ``Trainer.fit`` directly.
"""
import time

import lightning.pytorch as pl
import torch
from dicee.knowledge_graph import KG
from dicee.static_funcs import read_or_load_kg, store

from src.evaluator import Evaluator
from src.static_funcs import get_dataloader, get_full_storage_path, get_model, prepare_experiment_dir, save_report


def train_gnn_kge_model(args) -> dict:
    """Train one of RGCN / GATv2 / GraphSAGE and return its evaluation report."""
    args.learning_rate = args.lr
    args.full_storage_path = get_full_storage_path(args)
    prepare_experiment_dir(args.full_storage_path)
    print(f"[{args.model}] storing experiment at {args.full_storage_path}")

    torch.manual_seed(args.random_seed)
    pl.seed_everything(args.random_seed, verbose=False)

    start_time = time.time()

    # (1) Load & index the KG. path_for_serialization writes entity_to_idx.csv /
    # relation_to_idx.csv into full_storage_path, which dicee's own `store()` below expects.
    args.path_for_serialization = args.full_storage_path
    entity_dataset = read_or_load_kg(args, KG)

    # (2) Build the model (RGCN / GATv2 / GraphSAGE), all instances of dicee's BaseKGE.
    model = get_model(args, entity_dataset)
    train_dataloader = get_dataloader(args, entity_dataset)

    # (3) Train. `model` is itself a lightning.pytorch.LightningModule (via BaseKGE ->
    # BaseKGELightning), so no custom training loop / LightningModule wrapper is needed.
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        deterministic=True,
    )
    trainer.fit(model, train_dataloader)

    # (4) Evaluate with the filtered ranking protocol.
    model.eval()
    evaluator = Evaluator(args=args)
    eval_report = evaluator.eval(dataset=entity_dataset, trained_model=model, during_training=False)

    runtime_seconds = time.time() - start_time

    # (5) Persist model + report in dicee's usual experiment-folder layout.
    if args.save_experiment:
        store(trained_model=model, model_name="model", full_storage_path=args.full_storage_path)
        save_report(args, model, entity_dataset, eval_report, runtime_seconds)

    print(f"[{args.model}] finished in {runtime_seconds:.1f}s -> {args.full_storage_path}")
    return eval_report
