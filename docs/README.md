# Documentation

Supplementary documentation for `gnn_kge`. The top-level [`README.md`](../README.md)
is the entry point and carries the project's design rationale and its
narrative history (which bugs were found, what they changed). These files go
deeper on two things that outgrew it.

| File | Contents |
|---|---|
| [`experiments.md`](experiments.md) | Every experiment run in this repo: datasets, training and evaluation protocol, hyperparameters, results, and what each run does and does not establish. |
| [`clifford-models.md`](clifford-models.md) | The Clifford-algebra encoders: the algebra, how each layer maps onto the layer it replaces, the constraints that follow, and how the implementation is verified. |
| [`literature-baselines.md`](literature-baselines.md) | Published state-of-the-art per benchmark dataset (FB15k-237, WN18RR, UMLS, CoDEx-S/M, NELL-995), pulled from papers/leaderboards, with paper-vs-implementation caveats (missing code, incompatible eval protocols, split-versioning traps) checked rather than assumed. |
| [`literature-sweep.md`](literature-sweep.md) | September 2026 sweep of ~120 papers: results at low dimension (d ≤ 64), whether GNN-encoder papers compare against the same decoder under matched conditions, every CoDEx/UMLS/Kinship result found, numbers not to rely on, and open gaps. |
| [`literature-results.csv`](literature-results.csv) | The sweep's results as one table (dataset, model, family, dimension, MRR/H@1/H@10, protocol, source, verification, caution flag) for filtering or plotting against this project's runs. |

Where the two overlap with the top-level README, the README is the summary and
these are the record. Numbers here were read directly out of
`Experiments/*/report.json` and `Experiments/*/eval_report.json`; the commands
that produced them are given alongside each table so they can be rerun.
