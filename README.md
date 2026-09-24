# gnn_kge

RGCN, GATv2, and GraphSAGE knowledge graph embedding models, built directly
on [PyTorch Geometric](https://pyg.org)'s own KGE base class
(`torch_geometric.nn.kge.base.KGEModel`) and its `RGCNConv`/`GATv2Conv`/
`SAGEConv` message-passing layers. The conv layers are kept local (not
imported as opaque model classes) because experiments here modify them
directly; everything else this project needs from PyG (the embedding
tables, and optionally the DistMult/TransE scoring formulas) is used as-is.
This project has no dependency on `dicee` or `lightning` — KG loading,
the training loop, and experiment storage are all plain PyTorch, kept
deliberately small.

## Models

All three follow the same encoder-decoder recipe R-GCN introduced for link
prediction: a relational GNN encoder refines the entity embedding table by
message passing over the full training graph, and a decoder (`--decoder
{DistMult,TransE,ComplEx,RotatE}`, DistMult by default, see
`src/decoders.py`) scores triples from the refined representations. The
decoder is kept identical across all three encoders by default so that
differences in results come from the encoder, not from an incidental
decoder choice.

ComplEx and RotatE reuse the single flat real-valued entity/relation
embedding tables the encoders already produce, by splitting `embedding_dim`
in half as (real, imaginary) — no separate complex embedding table, and no
encoder changes. This is a project-specific simplification (RotatE's phase
here is unscaled, unlike the original paper's `embedding_range` rescaling)
and requires an even `embedding_dim`. They were added because DistMult
cannot represent antisymmetric relations at all, while ComplEx/RotatE top
the published leaderboards on every dataset this project targets (WN18RR,
CoDEx-S/M) among embedding-based methods.

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
src/config.py               argparse Namespace (KG/model/optimisation/storage args)
src/graph_utils.py           builds edge_index/edge_type (+ inverse edges) from indexed triples
src/kg_loader.py              reads {train,valid,test}.txt -> indexed KG (entity_to_idx, ...)
src/dataset.py                KvsAll torch Dataset
src/decoders.py                pluggable DistMult/TransE/ComplEx/RotatE scoring
src/evaluator.py               filtered MRR/MR/Hits@k evaluator, head+tail averaged
src/static_funcs.py            model/dataloader/optimizer construction, experiment storage
src/models/base_gnn.py         RelationalGNNEncoder(torch_geometric.nn.kge.base.KGEModel):
                                shared encode() + pluggable decode (tail and head direction)
src/models/rgcn.py              RGCN (torch_geometric.nn.RGCNConv)
src/models/rgat.py              GATv2 (torch_geometric.nn.GATv2Conv, relation-aware via edge_attr)
src/models/graphsage.py         GraphSAGE (torch_geometric.nn.SAGEConv, one per relation)
runners/kge_runner.py          load KG -> build model -> plain PyTorch training loop -> evaluate -> store
tests/test_models.py           shape/gradient/KGEModel-interface smoke tests
tests/test_kg_loader.py         KG-loading smoke test
tests/test_evaluator.py         pooled head+tail ranking/tie-breaking smoke tests
scripts/run_umls.sh            train all three models on UMLS
scripts/run_codexs.sh           train all three models x decoders on CoDEx-S
scripts/run_wn18rr.sh           train all three models x decoders on WN18RR (GPU)
KGs/UMLS                       train/valid/test triple files
KGs/CoDEx-S                     train/valid/test triple files (Safavi & Koutra 2020)
KGs/WN18RR                      train/valid/test triple files
```

The GNN encoder's entity/relation embedding tables (`node_emb`/`rel_emb`)
are PyG's own (`KGEModel.__init__` creates them); everything specific to
this project — message-passing graph construction, the per-batch
answer-leakage exclusion, normalisation/dropout/init options, and the
decoder — lives in `src/models/base_gnn.py`.

## Usage

```bash
conda env create -f environment.yml
conda activate gnn_kge

python main.py --model RGCN --dataset_dir KGs/UMLS --embedding_dim 32 --num_epochs 200
python main.py --model GATv2 --dataset_dir KGs/UMLS --embedding_dim 32 --gnn_heads 4 --num_epochs 200
python main.py --model GraphSAGE --dataset_dir KGs/UMLS --embedding_dim 32 --num_epochs 200
python main.py --model RGCN --decoder TransE --dataset_dir KGs/UMLS --embedding_dim 32 --num_epochs 200
python main.py --model RGCN --decoder ComplEx --dataset_dir KGs/CoDEx-S --embedding_dim 32 --num_epochs 500

bash scripts/run_umls.sh     # all three, one after another
bash scripts/run_codexs.sh   # all three x {DistMult,ComplEx,RotatE}, + RGCN num_bases sweep
bash scripts/run_wn18rr.sh   # all three x {DistMult,ComplEx,RotatE}, --accelerator gpu
python -m pytest tests/ -q   # unit tests (no training)
```

Results land under `Experiments/<dataset>/<model>_<dim>/`: `model.pt`,
`entity_to_idx.csv`, `relation_to_idx.csv`, `report.json`, `eval_report.json`.

## First-experiment gate: UMLS only

Per project scope, UMLS is the only dataset used until these three
encoders are validated. The results below predate the move off dicee/lightning
onto PyG's `KGEModel` + a plain training loop (2026-09-18); the DistMult
scoring formula, training loop semantics, and encoder math are unchanged by
that move, so these numbers are still the reference values — rerun
`scripts/run_umls.sh` to reconfirm them after any further change.

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

## Evaluator: head+tail averaged ranking

The evaluator originally only ranked the tail direction (`(h, r, ?)`).
Every MRR/Hits@k number reported in the KGE literature averages both
ranking directions — tail prediction `(h, r, ?)` *and* head prediction
`(?, r, t)` — over the same test triples, so a tail-only evaluator produces
numbers that aren't comparable to published baselines. `src/evaluator.py`
now also builds a `re_vocab` (`(r, t) -> [h, ...]`, the head-prediction
mirror of the existing `(h, r) -> [t, ...]` `er_vocab`) and calls a new
`model.forward_k_vs_all_head` (`src/models/base_gnn.py`) for the head
direction; ranks from both directions are pooled into the same MRR/MR/
Hits@k averages rather than reported separately. Every decoder
(`src/decoders.py`) implements both a tail and a head k-vs-all scoring
function — for DistMult, the tail formula is symmetric in `h`/`t` so the
head function is the same formula with `t` standing in for `h`; the other
three decoders have distinct, exactly-derived head-direction formulas (see
the module docstring). This changes the absolute numbers reported for every
model (a UMLS/WN18RR rerun after this change is not directly comparable to
older tail-only numbers in earlier version of this file), not the training.

## CoDEx-S results

CoDEx-S (`KGs/CoDEx-S`: 2,034 entities, 42 relations, 32,888 train triples,
[Safavi & Koutra 2020](https://arxiv.org/abs/2009.07810)) was picked as the
next dataset after UMLS specifically because it avoids the memory blow-ups
that block FB15k-237/NELL-995 at this project's default settings: RGCN's
`RGCNConv` with `num_bases=None` allocates one full `embedding_dim x
embedding_dim` weight matrix *per relation* (`~474` relations after inverse
edges on FB15k-237 vs. `84` here), and GraphSAGE's per-relation `SAGEConv`
list scales the same way. Neither is a real constraint at CoDEx-S's scale.
No RGCN/CompGCN baseline exists for CoDEx in the published literature, so
there is no specific number being chased here — see `scripts/run_codexs.sh`
for the exact sweep (all three encoders x `{DistMult, ComplEx, RotatE}`,
plus an RGCN `--num_bases` sweep, following the same informal-manual-sweep
convention as the UMLS section above, not a full grid search or seed
average).

A first pass with `--decoder ComplEx` (the primary pick here — see the
evaluator section above) at the UMLS-tuned per-model LRs, `embedding_dim=32`,
2 GNN layers, 500 epochs, single seed, head+tail-averaged evaluation:

| Model | Config | Train MRR | Val MRR | **Test MRR** | Test H@10 |
|---|---|---|---|---|---|
| RGCN | `lr=0.01` | 0.288 | 0.261 | **0.257** | 0.402 |
| GATv2 | `lr=0.02`, 4 heads | 0.251 | 0.249 | **0.247** | 0.420 |
| GraphSAGE | `lr=0.15` | 0.324 | 0.256 | **0.259** | 0.391 |

Train/val/test MRR are close together for all three (no leakage-style gap),
and loss decreases smoothly to a low plateau (GATv2 has a mild mid-training
bump around epoch 300 that recovers, not a collapse) — this is a real,
non-degenerate result, just an untuned first pass (single seed, LRs
inherited from UMLS rather than swept for CoDEx-S). For reference, the
CoDEx paper's own LibKGE baselines on CoDEx-S report test MRR ≈0.46
(ComplEx) / ≈0.44 (TuckER) / ≈0.32 (RESCAL) — those are per-triple
embedding models, not GNN-encoder based, and tuned specifically for this
dataset, so this project's numbers landing below them is expected for a
first pass, not necessarily a ceiling. The remaining sweep axes in
`scripts/run_codexs.sh` (DistMult/RotatE decoders, RGCN `--num_bases`,
per-model LR retuning, seed averaging) haven't been run yet.

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

WN18RR's 11 relations are dominated by symmetric/antisymmetric/inverse
patterns (`_hypernym`/`_hyponym`, etc.) — exactly what DistMult cannot
represent (it treats every relation as symmetric) and exactly why RotatE is
the strongest published embedding-based model here. `scripts/run_wn18rr.sh`
now sweeps `--decoder {DistMult,ComplEx,RotatE}` per encoder on
`--accelerator gpu` (full-graph message passing over ~174k propagation
edges costs roughly 1.5-7s/epoch depending on the encoder even on GPU, so a
full 500-epoch run per config is the dominant cost here, not the decode
step).

A first pass — RGCN, `--decoder ComplEx`, `lr=0.01` (the UMLS-tuned value,
unchanged), `embedding_dim=32`, 500 epochs, `--accelerator gpu` — landed
**Test MRR 0.048, H@10 0.074** (Train 0.053, Val 0.046: no leakage-style
train/test gap, so this isn't the answer-leakage bug recurring). Training
loss collapses to ~0.0002 by epoch 50 and stays there while MRR stays flat
and low — the diagnosis below rules out both "not enough capacity" and
"not enough regularization" as the cause.

**Diagnosis: KvsAll/BCE class imbalance, not embedding_dim or decoder
choice.** Two follow-up runs isolated the cause:

1. **RGCN + DistMult** (the decoder this encoder was already tuned with,
   not ComplEx) at the same `lr=0.01`: **Test MRR 0.058** — just as poor.
   Since the *known-good* decoder fails almost identically to ComplEx, the
   problem isn't ComplEx-specific (ruling out "wrong LR/regularization for
   a distance-based decoder" as the primary cause).
2. **RGCN + ComplEx with `--weight_decay 0.0001`** (L2 regularization,
   which the ComplEx paper itself, Trouillon et al. 2016, treats as
   essential): **Test MRR 0.0007** — catastrophically *worse*, near-total
   collapse. This rules out "needs more regularization" too.

The actual cause: `src/dataset.py`'s KvsAll target is a hard multi-hot
vector, and the mean number of true tails per `(h, r)` training row vs.
`num_entities` varies enormously across these three datasets —

| Dataset | Entities | Mean positives/row | Positive rate |
|---|---|---|---|
| UMLS | 135 | 6.44 | 4.77% |
| CoDEx-S | 2,034 | 3.14 | 0.15% |
| WN18RR | 40,559 | 1.39 | 0.0034% |

— so on WN18RR, `BCEWithLogitsLoss` can drive loss to near-zero just by
predicting every entity negative (correct 99.9966% of the time by
construction), starving the gradient on the one rare positive dimension per
row. This is exactly the failure mode ConvE's own paper (Dettmers et al.
2018 — the paper that introduced 1-N/KvsAll scoring in the first place)
built label smoothing to fix. Added `--label_smoothing` (`src/config.py`,
`src/static_funcs.py::smooth_labels`, applied in the training loop,
`runners/kge_runner.py`): `y = y*(1-eps) + eps/num_entities`, off by default
(`0.0`, so UMLS/CoDEx-S numbers above are unaffected).

**Result: real but small, not a fix.** RGCN + DistMult + `--label_smoothing
0.1`, otherwise identical to the diagnostic runs above: **Test MRR 0.061**
(vs. 0.058 without smoothing) — a ~6% relative improvement, still far below
the ~0.40 credible target, and training loss still collapses to ~0.0002 by
epoch 50. At WN18RR's scale (`num_entities=40,559`), `eps/num_entities ≈
2.5e-6` is negligible, so `--label_smoothing 0.1` mostly just softens the
*positive* target from `1.0` to `~0.9` rather than meaningfully lifting the
40,558 negative targets — the correction is real but too weak at this
entity count to be the dominant fix by itself.

This matters because ConvE's own paper reports MRR ≈0.43-0.46 on WN18RR
*with the same KvsAll class-imbalance and the same label-smoothing formula*
(they just don't use a GNN encoder — pure embeddings + a 2D conv decoder).
CompGCN separately reports an R-GCN-style encoder at MRR ≈0.40 on WN18RR.
Both numbers being achievable in the literature with this same training
signal means the *remaining* gap here isn't the loss imbalance (that's now
a small, understood, partially-corrected effect) — it's most likely the
`lr=0.01` inherited from UMLS's 135-entity graph being poorly matched to a
40,559-entity, ~174k-edge graph. The next experiment should be an
independent WN18RR-specific `--lr` sweep (not reused from UMLS), rather
than further tuning `--label_smoothing`. Not yet run. The rest of
`scripts/run_wn18rr.sh`'s sweep (GATv2/GraphSAGE, RotatE) hasn't been run
either.

## Tests

`tests/test_models.py` checks, for each model: it is an instance of
`torch_geometric.nn.kge.base.KGEModel`; `forward_triples` returns `(B,)`,
`forward_k_vs_all` returns `(B, num_entities)`; `forward()` routes `(B,3)`/`(B,2)`
batches correctly; gradients reach the model's parameters from a
`forward_k_vs_all` loss; and each of the four decoders (`--decoder`) scores
correctly in both `forward_k_vs_all` and `forward_k_vs_all_head`, rejects an
unknown name, and (ComplEx/RotatE only) rejects an odd `embedding_dim`.
`tests/test_kg_loader.py` checks `src/kg_loader.py` against `KGs/UMLS`.
`tests/test_evaluator.py` checks the pooled head+tail filtered-ranking
evaluator in isolation from training: a degenerate constant-output model
gets near-chance (not 1.0) MRR, a model that always ranks the correct answer
first in both directions gets MRR 1.0, and `_build_re_vocab` mirrors
`_build_er_vocab` correctly.

## Controlled KG reasoning lab (archived)

A separate `kg_reasoning_lab` package (CPU-only synthetic-graph diagnostics for
relational-GNN reasoning: a smoke harness, a matched-route shortcut-removal
pilot, a reconvergence/compression-isolation diagnostic, and a run of the
actual RGCN/GATv2/GraphSAGE models on the matched graphs) was built and run to
completion, then removed from this repo on 2026-09-18 — it was never
git-tracked and is unrelated to the KGE encoder work above. What was done, the
findings, exact configs, and how to reproduce it if needed are recorded in
`kg_reasoning_lab_SUMMARY.md`, kept outside this repo at
`/data/cs/backup/backup_kge_reasoning/` (it was never git-tracked); the code
itself is not kept and would need to be rewritten from that description.
