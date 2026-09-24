# Literature baselines: leaderboard SOTA per benchmark

What the published state of the art actually is, per KGE benchmark dataset,
pulled from papers and paperswithcode (paperswithcode.com's site itself now
redirects elsewhere, so this was reconstructed via web search and Wayback),
with paper/implementation details checked rather than taken from memory.
Purpose: know what this project's own numbers (`experiments.md`) are and are
not competitive with, and which literature numbers are safe to cite as a
comparison point.

This is a snapshot of an external, moving target (leaderboards change), not a
repo result — treat the numbers here as "as researched," and re-verify before
citing in anything that outlives this doc by much.

## FB15k-237 (MRR)

| Model | MRR | Notes |
|---|---|---|
| NBFNet | 0.416 | Path-based GNN, current top "pure" (non-LM) method |
| CompGCN | 0.355 | Relation-composition GNN |
| RotatE | 0.338 | Complex-plane rotation |
| ConvE | 0.325 | Convolutional scoring |
| R-GCN | 0.248 | Original paper's own number (Schlichtkrull et al. 2018) |
| R-GCN (this project's repro) | 0.233 | At reduced `hidden_dim`; not directly comparable to the paper's 500-dim config |

## WN18RR (MRR)

| Model | MRR | Notes |
|---|---|---|
| MoCoSA | 0.696 | LM-augmented hybrid, not a pure structural method |
| SimKGC | 0.671 | LM-augmented hybrid, not a pure structural method |
| **NBFNet** | **0.551** | Best *pure structural* method |
| CompGCN | 0.479 | |
| RotatE | 0.476 | The RotatE paper itself explains why it wins here: WN18RR's relations are composition/symmetry-heavy, which RotatE's complex-rotation operator represents directly |

LM-augmented hybrids (MoCoSA, SimKGC) top the leaderboard but use pretrained
language models over entity/relation text, not graph structure alone — not a
fair comparison point for a pure GNN-encoder project. NBFNet is the right
"best pure method" reference.

## UMLS (Hits@10 — MRR is not consistently tracked for this dataset)

| Model | Hits@10 |
|---|---|
| LP-BERT / KGLM / LASS / StAR | ~0.99–1.0 |
| Classic TransE / ComplEx / ConvE / DistMult | ~0.85–0.99 |

UMLS is small and near-saturated across the literature; the classic
embedding models already reach the same range this project's GATv2/GraphSAGE
results land in (`experiments.md`'s UMLS section).

## CoDEx-S / CoDEx-M (MRR)

| Model | CoDEx-S | CoDEx-M | Notes |
|---|---|---|---|
| MuCo-KGC (2025) | 0.550 | 0.470 | Hits@10: 0.764 / 0.600. **Confirmed no public code exists** — checked both the arXiv abstract/HTML (arxiv.org/abs/2503.03091) and the paper text directly for a GitHub link or code-availability statement; there is none. This is a real literature gap, not something missed by search. |
| ComplEx / TuckER / RESCAL | ~0.46 / ~0.44 / ~0.32 (CoDEx-S) | — | The CoDEx paper's own LibKGE baselines (Safavi & Koutra, EMNLP 2020) |

**No RGCN or CompGCN numbers exist for CoDEx at all** in the literature —
this project's own RGCN/GATv2/GraphSAGE runs on CoDEx-S have nothing
published to be compared against; they're a new data point, not a
reproduction.

## NELL-995 (MRR) — split-versioning trap

| Model | Metric | Value | Protocol |
|---|---|---|---|
| NBFNet | Transductive MRR | 0.525 | Standard full-corpus ranking |
| A\*Net | Transductive MRR | 0.521 | Standard full-corpus ranking |
| KRST | Transductive MRR / Inductive MRR | 0.800 / 0.769 | GraIL-style inductive-subgraph split, ranked against **50 candidates per query** (1 positive + 49 negatives) |
| APST (Su et al., AAAI 2024, arXiv:2312.13596, code: `github.com/ZhixiangSu/APST`) | Transductive MRR / Inductive MRR | 0.801 / 0.769 | Same 50-candidate protocol as KRST |

**These two rows are not comparable to each other.** "NELL-995" in the
literature refers to two different benchmarks under the same name:

1. The standard transductive completion benchmark (NBFNet/A*Net's numbers
   above) — but its test set only covers **12 of NELL-995's 200 relations**,
   a narrow evaluation even before any model-specific caveats.
2. The Teru et al. (GraIL) inductive-subgraph version used by
   KRST/APST-era papers, evaluated by ranking each test triple against only
   50 candidates (1 positive + 49 sampled negatives) rather than the full
   entity set. This is a much easier ranking task than full-corpus MRR, so
   KRST/APST's ~0.80 MRR is **not** evidence that they outperform NBFNet's
   0.525 — the numbers are not on the same scale.

Any future NELL-995 result in this project must state which of these two
benchmarks (and which eval protocol) it is being compared against.

## Cross-cutting findings

- **Graph density predicts where message passing helps.** Mean degree:
  FB15k-237 (~42.6) > CoDEx-S (~32) > CoDEx-M (~22) > WN18RR (~4.5) ≈
  NELL-995 (~4). Sparse graphs starve 1-hop message passing — consistent
  with this project's own WN18RR finding that RGCN (test MRR 0.3995) and
  plain DistMult (0.398) are statistically indistinguishable there (see the
  `dice-embeddings` repro session referenced in `experiments.md`).
- **Relation count predicts how much a relation-composition GNN gains.**
  CompGCN's improvement over non-relational baselines is much larger on
  FB15k-237 (237 relations) than WN18RR (11 relations).
- **The single sharpest finding, worth treating as a standing caveat on any
  "GNN encoder helps" claim this project makes:** two independent papers.
  Zhang et al., WWW 2022 (arXiv:2202.05679, "Rethinking Graph Convolutional
  Networks in Knowledge Graph Completion", code: `github.com/MIRALab-USTC/GCN4KGC`)
  reproduce RGCN/WGCN/CompGCN in one codebase and show that replacing the
  graph with a random one (RAT) or removing neighbours entirely (WNI) leaves
  MRR essentially unchanged (e.g. CompGCN+DistMult on FB15k-237: 0.342 real
  graph, 0.336 random, 0.335 no neighbours); the gain over a plain KGE model
  comes from the encoder's linear transform + batch norm + dropout on entity
  embeddings, and their LTE-KGE (that transform alone, no graph) matches
  CompGCN. Li et al., ACL 2023 (arXiv:2205.10652, "Are Message Passing Neural
  Networks Really Helpful for Knowledge Graph Completion?") show that
  MLP-only versions match the MPNNs and that the scoring and loss functions
  matter more. Both suggest the scoring function, loss and entity transform,
  not message passing itself, account for most of these encoders' reported
  gains. Cross-reference: this project's own CoDEx-S sweep found a plain
  encoder-free `KGE` (embedding tables + DistMult decoder, no message
  passing) baseline scoring *higher* (test MRR 0.2663) than every GNN
  encoder tried, Clifford variants included — an independent data point
  pointing the same direction.
- **Harder, rule-derived splits change the ranking entirely.** Liu et al.,
  KR 2023 (arXiv:2306.04814, "Revisiting Inferential Benchmarks for
  Knowledge Graph Completion," code: `github.com/shuwen-liu-ox/LogInfer`)
  built "LogInfer" benchmarks from FB15k-237/WN18RR/synthetic LUBM using
  rule-derived train/test splits per inference pattern (symmetry, inversion,
  hierarchy, composition, intersection, triangle, diamond) and three harder
  negative-sampling strategies (relevance-based, position-aware corruption,
  query-guided) instead of random splits + random-corruption negatives.
  Findings: **BoxE and RotatE are the best embedding-based models** across
  patterns; every system performs far worse than on the standard benchmarks
  (confirming these are more diagnostic); rule miners (AnyBURL/RuleN) beat
  embeddings on simple patterns (symmetry/inversion/hierarchy) but not
  complex ones (composition/triangle/diamond); classification AUC drops
  sharply as negative sampling gets harder (random corruption ~90-96% AUC to
  query-guided negatives ~85.6%), confirming standard random-corruption
  negatives are too easy a test. R-GCN specifically does well on
  relation-ranking (R-MRR) but worse on entity-ranking (C-MRR).

## What this means for dataset choice in this project

CoDEx-S is the most tractable next dataset for this project's full-graph,
`KvsAll` training loop: it avoids the memory wall `RGCNConv`/relation-specific
`SAGEConv` hit on FB15k-237 (`num_bases=None` allocates one full
`embedding_dim x embedding_dim` matrix per relation — 474 relations after
inverse edges on FB15k-237 vs. 84 on CoDEx-S) and gives a denser graph than
WN18RR/NELL-995 for message passing to have a chance of mattering.

NELL-995 was considered and set aside: it would hit the same memory wall as
FB15k-237 (200 relations) while, per the density argument above, giving less
signal than CoDEx-S — the least attractive combination of cost and expected
benefit of the datasets considered here.

## Sources referenced

- Schlichtkrull et al., *Modeling Relational Data with Graph Convolutional
  Networks*, ESWC 2018 (R-GCN).
- Vashishth et al., *Composition-based Multi-Relational Graph Convolutional
  Networks*, ICLR 2020 (CompGCN).
- Sun et al., *RotatE: Knowledge Graph Embedding by Relational Rotation in
  Complex Space*, ICLR 2019.
- Dettmers et al., *Convolutional 2D Knowledge Graph Embeddings*, AAAI 2018
  (ConvE; also introduced WN18RR).
- Safavi & Koutra, *CoDEx: A Comprehensive Knowledge Graph Completion
  Benchmark*, EMNLP 2020, arXiv:2009.07810.
- Su et al., *APST*, AAAI 2024, arXiv:2312.13596, code:
  `github.com/ZhixiangSu/APST`.
- Teru et al., *Inductive Relation Prediction by Subgraph Reasoning* (GraIL;
  origin of the inductive NELL-995 split/protocol used by KRST/APST).
- Zhang, Wang, Ye, Wu, *Rethinking Graph Convolutional Networks in Knowledge
  Graph Completion*, WWW 2022, arXiv:2202.05679.
- Li et al., *Are Message Passing Neural Networks Really Helpful for Knowledge
  Graph Completion?*, ACL 2023, arXiv:2205.10652.
- Liu, Cuenca Grau, Horrocks, Kostylev, *Revisiting Inferential Benchmarks
  for Knowledge Graph Completion*, KR 2023, arXiv:2306.04814, code:
  `github.com/shuwen-liu-ox/LogInfer`.
- MuCo-KGC, arXiv:2503.03091 (no public code found).
