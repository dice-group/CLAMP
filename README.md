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
encoders are validated.

**A first 60-epoch smoke run badly undertrained all three models**
(`lr=0.01`, `embedding_dim=32`, default settings): test MRR of 0.23 / 0.63 /
0.24 for RGCN / GATv2 / GraphSAGE. Since `encode()` reprocesses the *whole*
UMLS graph on every mini-batch (135 entities, 5216 training triples — a
few milliseconds per forward pass), training much longer than a typical
per-triple KGE model costs almost nothing here, so a first tuning pass
just trained longer and swept `--lr` — which surfaced two real bugs before
it produced a trustworthy number:

1. **Encoder answer-leakage** (`src/models/base_gnn.py::_graph_excluding_batch`).
   Every training triple `(h, r, t)` is both a training *label* and a live
   edge in the message-passing graph; because inverse edges are added for
   propagation, `t`'s embedding sits one hop from `h`, so the encoder could
   trivially hand the decoder the answer instead of learning to generalise.
   R-GCN's own paper (Schlichtkrull et al. 2018, Section 4.2) reports and
   fixes exactly this for link prediction; this project now excludes each
   mini-batch's own query edges (both directions) from the propagation
   graph on every training step, only at training time (validation/test
   triples were never in the graph to begin with, so nothing changes for
   them). Fixing this alone *dropped* several "good-looking" scores (e.g.
   RGCN's apparent 0.806 test MRR fell to 0.648) — those earlier numbers
   were inflated by the leak, not achieved by real learning.
2. **Evaluator tie-breaking** (`src/evaluator.py`). The naive filtered-rank
   formula `rank = 1 + count(scores > target)` gives rank 1 to *every* tied
   score, so a degenerate model that collapses to outputting the exact
   same score for every entity (a real GNN failure mode — representation
   over-smoothing/collapse — that this project hit at high learning rates)
   was reported as a perfect MRR of 1.0 instead of the near-chance result
   it actually deserved. Fixed by splitting credit across ties (`rank = 1 +
   count(strictly greater) + 0.5 * count(tied)`), the standard convention
   (see the RotatE paper's evaluation-protocol discussion). This is why
   `--lr 0.05` for RGCN and `--lr 0.15` for GATv2 both now correctly show
   near-chance MRR (0.017, 0.128) instead of 1.0 — both configurations
   destabilise training into a collapsed encoder, they don't "solve" UMLS.

With both fixes in place, a small `--lr` sweep at `--num_epochs 500` (exact
commands in `scripts/run_umls.sh`) gives:

| Model | Config (beyond the shared `embedding_dim=32`, 2 GNN layers) | Train MRR | Val MRR | **Test MRR** | Test H@10 |
|---|---|---|---|---|---|
| RGCN | `lr=0.01`, 500 epochs | 0.83 | 0.78 | **0.648** | 0.882 |
| GATv2 | `lr=0.02`, 500 epochs, 4 heads | — | — | **0.864** | 0.980 |
| GraphSAGE | `lr=0.15`, 500 epochs | — | — | **0.834** | 0.959 |

(Chance level on UMLS's 135 entities is ≈0.02 MRR.) GATv2 and GraphSAGE
land in the range reported for strong published KGE models on UMLS (e.g.
ConvE, ComplEx, TuckER: test MRR ≈0.85-0.95); RGCN's honest number is
lower — plausible given basis decomposition is off and it has no
attention/larger-aggregator capacity to compensate for the harder,
leakage-free training signal, not evidence of a remaining bug (its
train→val→test progression is smooth, and it's still far above chance).
Higher learning rates destabilise all three past this point (RGCN
collapses by `lr=0.05`, GATv2 by `lr=0.15`) — see the note above.

This is still a manual sweep over `lr` only (`scripts/run_umls.sh` has the
exact commands used per model), not a full grid search, not averaged over
seeds, and with no early stopping or basis decomposition/weight-decay
tuning for RGCN specifically. Treat these as "each encoder demonstrably
works, generalises past its own training graph, and is in a credible
range for this dataset" — not a tuned benchmark suitable for ranking the
three encoders against each other. RGCN in particular likely has room
left (basis decomposition, weight decay) that wasn't explored here.

## WN18RR: a much sparser, harder graph

WN18RR (`KGs/WN18RR`: ~40,943 entities, 11 relations, 86,835 train
triples) has ~4 propagation edges per entity on average versus UMLS's
~150 — exactly the regime where the encoder answer-leakage bug above is
most damaging (a sparse node's *only* neighbour of a relation can be the
very training edge being predicted), so this dataset is also what exposed
that bug in the first place: before the fix, WN18RR training was
essentially unusable (test MRR 0.01-0.11 across all three models at 300
epochs). Training here also uses `--accelerator gpu` (added in
`src/config.py`, off by default per this project's CPU-first default) —
full-graph message passing over ~174k propagation edges is the bottleneck
on CPU, not the DistMult decode step.

Results on WN18RR are still being tuned; expect this section to be
updated once a trustworthy configuration is found. For reference, strong
published KGE models (not GNN-encoder based) report test MRR ≈0.43-0.48
on WN18RR (e.g. RotatE ≈0.476, ConvE ≈0.46); R-GCN-style encoders
specifically are reported lower than that in follow-up literature (e.g.
CompGCN reports R-GCN at MRR ≈0.40 on WN18RR), which is the credible
target range for these models here, not the UMLS numbers above.

## Tests

`tests/test_models.py` checks, for each model: it is an instance of
`dicee.models.base_model.BaseKGE`; `forward_triples` returns `(B,)`,
`forward_k_vs_all` returns `(B, num_entities)`; `forward()` routes `(B,3)`/`(B,2)`
batches correctly (the same dispatch `BaseKGE.forward` uses); and gradients
reach the model's parameters from a `forward_k_vs_all` loss.
