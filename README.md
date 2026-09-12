# gnn_kge

RGCN, GATv2, and GraphSAGE knowledge graph embedding models, built as
`BaseKGE` subclasses so they slot into the [dice-embeddings](https://github.com/dice-group/dice-embeddings)
(`dicee`) framework the same way any of its built-in models
(`DistMult`, `Keci`, ...) do. The project structure and training/evaluation
pipeline follow [literal-embeddings](../literal-embeddings), a sister repo
that builds on `dicee` the same way without modifying it.

## Why a separate repo instead of editing dice-embeddings

`dicee`'s models are per-triple/per-batch scorers: `forward_triples` and
`forward_k_vs_all` take an index batch and look embeddings up in a table.
A GNN encoder needs the *whole* training graph to run message passing
before it can score anything. This repo keeps that graph-construction and
message-passing logic self-contained here (`src/graph_utils.py`,
`src/models/base_gnn.py`) while every model still **is** a `dicee.models.base_model.BaseKGE`
instance and satisfies its exact interface — so moving a finished model
into `dicee/models/` later (per `dice-embeddings/.claude/skills/add-model`)
is a file copy plus a `MODEL_REGISTRY` entry, not a rewrite.

## Models

All three follow the same encoder-decoder recipe R-GCN introduced for link
prediction: a relational GNN encoder refines the entity embedding table by
message passing over the full training graph, and a fixed **DistMult**
decoder scores triples from the refined representations. The decoder is
kept identical across all three so that differences in results come from
the encoder, not from an incidental decoder choice.

| File | Model | Reference | Encoder detail kept faithful to the paper |
|---|---|---|---|
| `src/models/rgcn.py` | `RGCN` | [Schlichtkrull et al. 2018](https://arxiv.org/abs/1703.06103) | Relation-specific linear transforms, mean-normalised aggregation per relation (`RGCNConv`'s `aggr='mean'` = the paper's `1/c_{i,r}`), optional basis decomposition (`--num_bases`) |
| `src/models/rgat.py` | `GATv2` | [Brody et al. 2021](https://arxiv.org/abs/2105.14491) | Unmodified `GATv2Conv` attention formula; made relation-aware via `edge_dim` edge features (relation embeddings), which is how the layer itself is designed to consume edge information — not a new attention mechanism |
| `src/models/graphsage.py` | `GraphSAGE` | [Hamilton et al. 2017](https://arxiv.org/abs/1706.02216) | Mean neighbourhood aggregator + per-layer L2 normalisation (Algorithm 2, steps 4-7), unchanged; extended to relations the way R-GCN extends plain GCN (one `SAGEConv` per relation, summed, plus **one** shared root/self transform) |

**What is not original-paper-faithful, and why:** GATv2 and GraphSAGE were
proposed for homogeneous graphs / node classification, not multi-relational
link prediction — extending them to relations at all is already an
addition on top of the papers, done here the same (R-GCN-style)
way for both so the three encoders are comparable. GraphSAGE's neighbourhood
*sampling* (its main scalability contribution) is not implemented: on graphs
this size (UMLS, Kinship, FB15k-237) full-neighbourhood aggregation is what
GraphSAGE reduces to once the sample size is not smaller than the actual
degree — a size restriction, not a different algorithm. See each model
file's docstring for the exact design decision and the citation behind it.

## Project layout

```
main.py                    CLI entry point
src/config.py               argparse Namespace (dicee-standard + GNN-specific args)
src/graph_utils.py           builds edge_index/edge_type (+ inverse edges) from indexed triples
src/dataset.py               KvsAll torch Dataset (adapted from dicee/literal-embeddings)
src/evaluator.py              filtered MRR/MR/Hits@k evaluator
src/static_funcs.py           model/dataloader construction, experiment storage
src/models/base_gnn.py        RelationalGNNEncoder(BaseKGE): shared encode() + DistMult decode
src/models/rgcn.py             RGCN
src/models/rgat.py             GATv2 (relation-aware)
src/models/graphsage.py        GraphSAGE (relation-aware)
runners/kge_runner.py         load KG -> build model -> lightning.pytorch.Trainer.fit -> evaluate -> store
tests/test_models.py          shape/gradient/BaseKGE-interface smoke tests
scripts/run_umls.sh           train all three models on UMLS
KGs/UMLS                      copied from dice-embeddings/KGs/UMLS
```

Dataset loading (`dicee.knowledge_graph.KG`, `dicee.static_funcs.read_or_load_kg`)
and per-triple embedding storage (`BaseKGE.entity_embeddings`/`relation_embeddings`,
`dicee.static_funcs.store`) are imported directly from `dicee`, not
reimplemented.

Because these models are not registered in `dicee`'s `MODEL_REGISTRY`,
`src/static_funcs.py::get_model` looks them up in this project's own
`src/models.MODEL_MAP` instead of calling `dicee.static_funcs.intialize_model`
— everything downstream (training loop, evaluator, checkpoint format) is
otherwise plain `dicee`.

## Usage

```bash
pip install -r requirements.txt
python main.py --model RGCN --dataset_dir KGs/UMLS --embedding_dim 32 --num_epochs 200
python main.py --model GATv2 --dataset_dir KGs/UMLS --embedding_dim 32 --gnn_heads 4 --num_epochs 200
python main.py --model GraphSAGE --dataset_dir KGs/UMLS --embedding_dim 32 --num_epochs 200

bash scripts/run_umls.sh     # all three, one after another
python -m pytest tests/ -q   # unit tests (no training)
```

Results land under `Experiments/<dataset>/<model>_<dim>/`: `model.pt`,
`entity_to_idx.csv`, `relation_to_idx.csv`, `report.json`, `eval_report.json`
— the same layout dice-embeddings itself produces.

## First-experiment gate: UMLS only

Per project scope, UMLS is the only dataset used until these three
encoders are validated. A 60-epoch, untuned smoke run (`embedding_dim=32`,
`lr=0.01`, `batch_size=512`, no basis decomposition, 4 GATv2 heads) gives
non-degenerate, clearly-learning filtered test MRR for all three
(chance level on UMLS's 135 entities is ≈0.02 MRR):

| Model | Test MRR | Test H@10 |
|---|---|---|
| RGCN | 0.233 | 0.433 |
| GATv2 | 0.628 | 0.862 |
| GraphSAGE | 0.237 | 0.348 |

These numbers are a pipeline-correctness check, not a tuned benchmark —
no hyperparameter search, early stopping, or seed averaging was done. Do
not read them as a claim about which encoder is "better"; extend to other
datasets or draw comparative conclusions only after tuning each model
separately and averaging over seeds.

## Tests

`tests/test_models.py` checks, for each model: it is an instance of
`dicee.models.base_model.BaseKGE`; `forward_triples` returns `(B,)`,
`forward_k_vs_all` returns `(B, num_entities)`; `forward()` routes `(B,3)`/`(B,2)`
batches correctly (the same dispatch `BaseKGE.forward` uses); and gradients
reach the model's parameters from a `forward_k_vs_all` loss.
