# Experiment record

Every experiment run in this repository, with the configuration that produced
it and what it does and does not establish. Numbers are read from
`Experiments/<run>/report.json` (configuration, parameter count, runtime) and
`Experiments/<run>/eval_report.json` (metrics).

Every number is a single run unless its table gives a mean ± standard deviation
over seeds (the CoDEx-S d = 32 comparison in
[CoDEx-S with the literature recipe](#codex-s-with-the-literature-recipe)).

## Datasets

Counts are what `src/kg_loader.py` actually loads, not what the datasets'
publications report:

| Dataset | Entities | Relations | Train | Valid | Test | Mean positives per KvsAll row |
|---|---|---|---|---|---|---|
| UMLS | 135 | 46 | 5,216 | 652 | 661 | 6.44 (4.77% positive rate) |
| CoDEx-S | 2,034 | 42 | 32,888 | 1,827 | 1,828 | 3.14 (0.15%) |
| WN18RR | 40,559 | 11 | 86,835 | 2,824 | 2,924 | 1.39 (0.0034%) |

The entity vocabulary is built from the training split only, and evaluation
triples mentioning an entity absent from training are dropped. On WN18RR this
removes 210 validation and 210 test triples relative to the official splits
(3,034 and 3,134), so WN18RR numbers here are **not** directly comparable to
published WN18RR results. UMLS and CoDEx-S are unaffected in practice.

The positive-rate column is the reason WN18RR behaves differently under the
KvsAll objective; see [WN18RR](#wn18rr-unresolved).

## Protocol

Shared by every run unless stated otherwise.

**Training.** KvsAll (ConvE-style 1-N) scoring: each `(h, r)` row carries a
multi-hot target over all entities, optimised with `BCEWithLogitsLoss`. Adam,
2 message-passing layers, full-batch message passing over the entire training
graph on every mini-batch. Inverse edges are added to the propagation graph, so
a dataset with `R` relations propagates over `2R` relation types.

**Answer-leakage exclusion.** Each training triple is both a label and a live
edge in the propagation graph, so the encoder could read the answer off the
graph instead of generalising. Every training step removes the mini-batch's own
query edges, both directions, from the propagation graph
(`src/models/gnn.py::_graph_excluding_batch`), following R-GCN
(Schlichtkrull et al. 2018, §4.2). Inactive at evaluation, where the query
edges were never in the graph.

**Evaluation.** Filtered MRR / MR / Hits@{1,3,10}, pooled over both ranking
directions — tail `(h, r, ?)` and head `(?, r, t)` — into single averages, the
convention used in the KGE literature. Ties split credit
(`rank = 1 + #{strictly greater} + 0.5 * #{tied}`) so a collapsed model that
emits one score for every entity scores near chance rather than a perfect 1.0.

**Decoders.** `DistMult`, `TransE`, `ComplEx`, `RotatE`, selected with
`--decoder` and applied identically across encoders so differences are
attributable to the encoder. ComplEx and RotatE split the existing real-valued
embedding in half as (real, imaginary) rather than keeping a separate complex
table, and require an even `embedding_dim`.

## CoDEx-S

The main comparison in this repo. CoDEx-S was chosen after UMLS because it is
large enough to be non-trivial but avoids the per-relation weight-matrix blowup
that makes FB15k-237 impractical at these defaults (84 propagation relation
types here versus ~474).

### Untuned first pass

Learning rates inherited from UMLS, `--decoder ComplEx`, `embedding_dim=32`,
500 epochs. Reproduce with `bash scripts/run_codexs.sh`.

| Model | lr | Val MRR | Test MRR | Test H@10 | Params |
|---|---|---|---|---|---|
| RGCN | 0.01 | 0.2613 | 0.2568 | 0.4024 | 240,576 |
| GATv2 | 0.02 | 0.2485 | 0.2470 | 0.4204 | 85,024 |
| GraphSAGE | 0.15 | 0.2556 | 0.2592 | 0.3911 | 240,576 |

Train, validation and test MRR sit close together for all three, so there is no
leakage-style generalisation gap.

### Random search

`python scripts/sweep_codexs.py`. Phase 1 samples 12 configurations per model
(36 total) and trains each for 150 epochs, ranking by validation MRR; phase 2
retrains the best configuration per model for the full 500 epochs and saves it
to `Experiments/CoDEx-S-tuned/`. Phase-1 results for all 36 configurations are
in `scripts/codexs_sweep_results.csv`.

Search space: decoder `{ComplEx, RotatE}`, `embedding_dim {32, 64, 128}`,
lr `{0.005, 0.01, 0.02, 0.05, 0.1}`, weight decay `{0, 1e-5, 1e-4}`,
dropout `{0.1, 0.2, 0.3}`, and for RGCN `num_bases {None, 8, 16}`.

Phase-2 results, the tuned baselines:

| Model | Decoder | dim | lr | wd | dropout | bases | Val MRR | Test MRR | H@1 | H@3 | H@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RGCN | ComplEx | 64 | 0.02 | 1e-5 | 0.3 | 8 | 0.2514 | 0.2520 | 0.1671 | 0.2722 | 0.4360 |
| GATv2 | RotatE | 32 | 0.02 | 0 | 0.1 | — | 0.2097 | 0.2071 | 0.1253 | 0.2188 | 0.3848 |
| GraphSAGE | ComplEx | 32 | 0.02 | 0 | 0.3 | — | 0.2480 | 0.2585 | 0.1846 | 0.2793 | 0.4018 |

Tuning helped RGCN marginally and moved GraphSAGE and GATv2 slightly the other
way on test MRR compared to the untuned pass — within the range a single seed
can produce, which is the main reason these should not be read as a ranking.

### Clifford encoders

Each Clifford encoder was run at **its non-Clifford counterpart's tuned
configuration** — the same decoder, dimension, learning rate, weight decay,
dropout, and 500-epoch budget — so the encoder is the only thing that changes.
`--num_bases` has no Clifford analogue (the relation operator is a multivector,
not a matrix) and was dropped for `CliffordRGCN`. See
[`clifford-models.md`](clifford/clifford-models.md) for the models themselves.

```bash
python main.py --model CliffordRGCN --decoder ComplEx --dataset_dir KGs/CoDEx-S \
    --embedding_dim 64 --lr 0.02 --weight_decay 1e-05 --gnn_dropout_rate 0.3 \
    --num_gnn_layers 2 --num_epochs 500 --batch_size 512 --eval_model train_val_test \
    --full_storage_path Experiments/CoDEx-S-clifford/CliffordRGCN_ComplEx_64
```

(and likewise `CliffordGATv2 --decoder RotatE --embedding_dim 32 --lr 0.02
--weight_decay 0.0 --gnn_dropout_rate 0.1 --gnn_heads 4`, `CliffordGraphSAGE
--decoder ComplEx --embedding_dim 32 --lr 0.02 --weight_decay 0.0
--gnn_dropout_rate 0.3`.)

| Model | Val MRR | Test MRR | H@1 | H@3 | H@10 | Params | Encoder params |
|---|---|---|---|---|---|---|---|
| RGCN | 0.2514 | 0.2520 | 0.1671 | 0.2722 | 0.4360 | 208,064 | 75,200 |
| **CliffordRGCN** | **0.2643** | **0.2614** | **0.1772** | **0.2853** | **0.4409** | 143,872 | **11,008** |
| GATv2 | 0.2097 | 0.2071 | 0.1253 | 0.2188 | 0.3848 | 85,024 | 18,592 |
| **CliffordGATv2** | **0.2274** | **0.2212** | 0.1253 | **0.2434** | **0.4324** | 72,960 | **6,528** |
| GraphSAGE | 0.2480 | **0.2585** | **0.1846** | **0.2793** | 0.4018 | 240,576 | 174,144 |
| **CliffordGraphSAGE** | 0.2400 | 0.2441 | 0.1652 | 0.2694 | **0.4155** | 71,936 | **5,504** |

"Encoder params" excludes the entity and relation embedding tables, which are
identical between each pair, and so isolates what the layer swap actually
changes.

Reading these:

- **CliffordRGCN** improves on every metric while using 85% fewer encoder
  parameters.
- **CliffordGATv2** gives the largest gain (+0.048 H@10) on 65% fewer encoder
  parameters. Its H@1 is identical to GATv2's at 0.1253 while every other
  metric differs — most plausibly coincidence on 1,828 test triples, but it has
  not been checked.
- **CliffordGraphSAGE** is the one mixed case: down 0.014 test MRR and 0.019
  H@1, up 0.014 H@10, on 97% fewer encoder parameters.

Two things these runs do **not** establish. The hyperparameters were tuned for
the non-Clifford encoders, so the Clifford numbers are a floor rather than a
best. And with one seed per configuration, a ~0.01 MRR difference is not
separable from noise — the CliffordGraphSAGE MRR/H@10 split especially needs
repeats before it is called either way.

**Runtimes are not comparable and are deliberately omitted above.** The
`runtime_seconds` field in these runs' `report.json` (3,913 / 3,928 / 1,043 s
for the Clifford runs versus 541 / 123 / 605 s for the baselines) reflects that
the Clifford runs were executed three-at-once with `OMP_NUM_THREADS=4` while
the baselines had the machine to themselves. A like-for-like cost comparison
has not been measured. What *is* structural: the full-multivector geometric
product costs `O(dim^2)` per transformed row, so `CliffordRGCN` at `dim=64`
does about four times the per-edge work of the same model at `dim=32`.

## CoDEx-S with the literature recipe

The runs above use the KvsAll/BCE objective, train only tail queries while the
evaluator also ranks heads, and stop at the last epoch. Those choices left even
the encoder-free model far below the published ComplEx (0.266 vs 0.465). The
runs in this section use the training recipe from the CoDEx paper's own LibKGE
configs instead, available as opt-in flags:

- `--reciprocal`: an inverse-relation table for the decoder; head queries
  `(?, r, t)` are trained and answered as `(t, r^-1, ?)`.
- `--scoring_technique 1vsAll`: softmax cross-entropy over all entities, one
  row per triple.
- `--init_param xavier_normal`, `--input_dropout_rate 0.07`, batch 1024.
- `--eval_every 5 --early_stopping_patience 10`: the weights with the best
  validation MRR are restored before the final evaluation, 400 epochs at most.

Always with `--decoder ComplEx`. During training, validation ranks are filtered
without the test split, so the validation MRR printed in the logs is lower than
the final one; the final numbers below use the standard filter over all splits.

### Reproducing the published baseline (d = 512)

| Model | lr | Val MRR | Test MRR | Test H@1 | Test H@10 |
|---|---|---|---|---|---|
| KGE (ComplEx) | 3.4e-4 | 0.474 | **0.462** | 0.370 | 0.645 |
| RGCN | 3.4e-4 | 0.403 | 0.392 | 0.282 | 0.610 |
| GraphSAGE | 1e-3 | 0.356 | 0.346 | 0.225 | 0.580 |
| GATv2 | 1e-3 | 0.338 | 0.325 | 0.214 | 0.545 |

The encoder-free model matches the CoDEx paper's ComplEx (0.465), which
validates the training and evaluation pipeline. Results in
`Experiments/CoDEx-S-litrecipe/`; one seed; the query-edge exclusion was
`pair` (see below). A second learning rate (1e-3 for RGCN, 3.4e-4 for the
others) scored lower on test MRR for every GNN.

### Embedding dimension (encoder-free ComplEx)

| d | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|
| Test MRR | 0.363 | 0.371 | 0.418 | 0.454 | 0.462 |

lr 3.4e-4, one seed, `Experiments/CoDEx-S-litrecipe/dims/`. The d = 64 run
early-stopped at epoch 110 while the others kept improving to epoch 240–360,
so its value is probably low; d = 256 stopped at the 400-epoch cap still
improving slightly.

### d = 32: which literature idea helps a GNN encoder?

Phase 1 (results in `Experiments/CoDEx-S-dim32-literature/`) ran 13 configurations at lr 3.4e-4 and
1e-3, one seed. It added two models and three graph controls:

- `LTE`: shared bias-free linear map, BatchNorm and dropout on entity
  embeddings, no graph (Zhang et al., WWW 2022).
- `CompGCN`: composition-based encoder following the official layer
  (Vashishth et al., ICLR 2020); needs `--reciprocal`.
- `--gnn_graph random|none`: the training graph with random tails, or no
  edges (Zhang et al.'s RAT and WNI controls).
- `--query_edge_exclusion pair|triple|none`: which training edges each batch
  hides from the propagation graph. `pair` (the default, and the only
  behaviour before this change) hides every `(h, r, *)` edge of a query;
  `triple` hides only the batch's own triples; `none` keeps everything, as the
  official R-GCN and CompGCN code does. Validation and test edges are never in
  the graph, so `none` does not leak evaluation answers.
- `--gnn_edge_dropout`, `--gnn_residual`.

Phase-1 test MRR at the learning rate with the better validation MRR:

| Configuration | Test MRR |
|---|---|
| KGE | 0.360 |
| LTE | 0.375 |
| RGCN (2 layers, `pair`) / 1 layer / residual | 0.342 / 0.343 / 0.336 |
| RGCN, `none` / `none` + edge dropout 0.4 | 0.371 / 0.372 |
| RGCN, random graph / no graph | 0.357 / 0.354 |
| CompGCN (1 layer, `pair`) / `none` / random graph / no graph | 0.361 / 0.358 / 0.355 / 0.354 |

With `pair` exclusion, RGCN on the real graph scores below RGCN on a random
graph or no graph; with `none`, the real graph helps. CompGCN is unaffected by
the graph, which reproduces Zhang et al.'s finding.

Phase 2 (`bash scripts/run_codexs_dim32_exclusion_seeds.sh`, results in
`Experiments/CoDEx-S-dim32-exclusion/`) ran every encoder under each exclusion
mode at lr 1e-3 (the rate phase 1 selected for all of them) over seeds 1–3.
Test MRR, mean ± standard deviation:

| Model | `pair` | `triple` | `none` |
|---|---|---|---|
| KGE (no graph) | | | 0.366 ± 0.005 |
| LTE (no graph) | | | **0.375 ± 0.004** |
| RGCN | 0.331 ± 0.009 | 0.342 ± 0.009 | **0.372 ± 0.004** |
| GATv2 | 0.252 ± 0.016 | 0.337 ± 0.003 | 0.365 ± 0.008 |
| GraphSAGE | 0.334 ± 0.006 | 0.235 ± 0.013 | **0.372 ± 0.002** |
| CompGCN (1 layer) | 0.360 ± 0.004 | 0.323 ± 0.012 | 0.363 ± 0.005 |

What this establishes:

- `none` is the best exclusion mode for every encoder, and `pair`, the previous
  default, costs RGCN and GraphSAGE about 0.04 and GATv2 0.11.
- Under `none`, the four GNN encoders (0.363–0.372) tie the encoder-free model
  (0.366 ± 0.005); none is ahead by more than about one standard deviation.
- LTE, which uses no graph, is the best or tied-best model.
- `triple` helps RGCN and GATv2 relative to `pair` but hurts GraphSAGE and
  CompGCN badly. This is not explained yet.

What it does not establish: only the learning rate was tuned (two values), and
the other settings were taken from the CoDEx paper's d = 512 ComplEx config, so
none of these models has been tuned at d = 32. Three seeds cannot separate
differences below about 0.01.

## UMLS

The first-gate dataset, used to establish that each encoder learns at all.
Reproduce with `bash scripts/run_umls.sh`. The top-level README documents this
phase in full, including the two bugs it exposed (encoder answer-leakage and
evaluator tie-breaking), both since fixed.

| Model | lr | Test MRR | Test H@10 |
|---|---|---|---|
| RGCN | 0.01 | 0.648 | 0.882 |
| GATv2 | 0.02 | 0.864 | 0.980 |
| GraphSAGE | 0.15 | 0.834 | 0.959 |

Chance is ≈0.02 MRR on 135 entities. **These predate the switch to pooled
head+tail evaluation**, which changes absolute values, so they are not
comparable to the CoDEx-S tables above or to current runs. They are kept as a
sanity threshold — each encoder learns something far above chance — and would
need a rerun to be used for anything else. No Clifford encoder has been run on
UMLS.

## WN18RR (unresolved)

WN18RR does not work in this repo yet, and the cause is only partly understood.
All configurations tried land near 0.05 test MRR against a credible target of
≈0.40 for R-GCN-style encoders (CompGCN's reported figure).

| Run | Decoder | dim | lr | Epochs | Val MRR | Test MRR | H@10 |
|---|---|---|---|---|---|---|---|
| `WN18RR/RGCN_32` | DistMult | 32 | 0.01 | 500 | 0.0590 | 0.0614 | 0.0964 |
| `WN18RR-tuned/RGCN_RotatE_32` | RotatE | 32 | 0.05 | 200 | 0.0258 | 0.0268 | 0.0513 |
| `WN18RR-tuned/RGCN_ComplEx_32` | ComplEx | 32 | 0.05 | 200 | ~0.0000 | ~0.0000 | 0.0000 |
| `WN18RR-tuned/RGCN_ComplEx_64` | ComplEx | 64 | 0.05 | 200 | ~0.0000 | ~0.0000 | 0.0000 |

The two ComplEx runs at `lr=0.05` collapsed completely (MRR ≈ 4.9e-5, H@10
exactly 0), which is a degenerate encoder rather than a weak one.

What has been ruled out, per the top-level README: the decoder choice (DistMult
fails alike), insufficient regularisation (weight decay made it far worse), and
the answer-leakage bug (no train/test gap). Label smoothing was added to address
the KvsAll class imbalance and produced a real but small improvement
(0.058 → 0.061), too small to be the fix at 40,559 entities. The leading
hypothesis is that the learning rate inherited from UMLS's 135-entity graph is
simply wrong for this one, and the next step is a WN18RR-specific learning-rate
sweep.

`scripts/sweep_wn18rr.py` implements that sweep (12 configurations). It was
started and **stopped after 3 of 12 configurations**; its partial results are in
`scripts/wn18rr_sweep_results.csv`. The table above is those partial results
plus the earlier single run. No GATv2, GraphSAGE, or Clifford encoder has been
run on WN18RR.

## Gaps

Ordered roughly by how much they limit the conclusions above.

1. **Seed averaging only for the CoDEx-S d = 32 comparison.** Every other
   number is one run, and the margins between encoders are small enough that
   this matters.
2. **No model is tuned at d = 32.** The literature-recipe runs tune only the
   learning rate; dropout, batch size, regularisation, label smoothing and
   layer count come from the CoDEx paper's d = 512 ComplEx config.
3. **Clifford encoders use borrowed hyperparameters.** They have never been
   tuned in their own right; `scripts/sweep_codexs.py` would need its `MODELS`
   list extended to cover them.
4. **WN18RR unresolved**, and its sweep is 3/12 done.
5. **No like-for-like runtime measurement** of Clifford versus non-Clifford
   encoders.
6. **Clifford encoders have only been run on CoDEx-S** — not UMLS, not WN18RR.
7. **The evaluation set is reduced on WN18RR** by the train-only entity
   vocabulary, so those numbers cannot be compared to published results even
   once training works.
