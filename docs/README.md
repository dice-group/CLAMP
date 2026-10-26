# Documentation

Supplementary documentation for `gnn_kge`. The top-level [`README.md`](../README.md)
is the entry point and carries the project's design rationale and its
narrative history (which bugs were found, what they changed). These files go
deeper on two things that outgrew it.

| File | Contents |
|---|---|
| [`experiments.md`](experiments.md) | Every experiment run in this repo: datasets, training and evaluation protocol, hyperparameters, results, and what each run does and does not establish. |
| [`rgcn-gap/`](rgcn-gap/README.md) | Track: why RGCN lags the decoder-only baseline. Status, working plan, capacity analysis. |
| [`clifford/`](clifford/README.md) | Track: Clifford message passing. Status, model docs, attention derivation, GA survey notes. |
| [`analysis_todos.md`](analysis_todos.md) | Why no encoder beats the decoder-only KGE: three proposed explanations checked against the controls, and the analyses that would test them. |
| [`related-work.md`](related-work.md) | The literature in one place: relational GNN encoders, matched re-evaluations, decoder-expressivity and GNN-expressiveness theory, path-based and inductive GNNs, degree effects, hypercomplex and Clifford models; then the September 2026 sweep of ~120 papers (results at d <= 64, every CoDEx/UMLS/Kinship number found, numbers not to rely on) and published state of the art per benchmark. |
| [`literature-results.csv`](literature-results.csv) | The sweep's results as one table (dataset, model, family, dimension, MRR/H@1/H@10, protocol, source, verification, caution flag) for filtering or plotting against this project's runs. |

Where the two overlap with the top-level README, the README is the summary and
these are the record. Numbers here were read directly out of
`Experiments/*/report.json` and `Experiments/*/eval_report.json`; the commands
that produced them are given alongside each table so they can be rerun.
