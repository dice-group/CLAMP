# CLAMP

Clifford Algebra Message Passing.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Usage

```bash
python main.py --model RGCN --dataset_dir KGs/UMLS --num_epochs 500

python main.py --model CliffordRGCN --decoder ComplEx \
  --dataset_dir KGs/CoDEx-S --embedding_dim 64 --num_epochs 500

python main.py --model CompGCN --decoder ComplEx \
  --dataset_dir KGs/CoDEx-S --scoring_technique 1vsAll --lr 0.001
```

## Models

Each model is one class in its own file that builds its layers and writes its own `encode` loop.
Layers live in a `layers/` package next to the models that use them.

```
src/models/
├── base.py              KGModel: entity/relation tables, decoder, dicee-matched scoring; no graph
├── kge.py               KGE (decoder only), LTE
├── gnn/                 dense encoders
│   ├── base.py          RelationalGNN(KGModel): training graph, --gnn_graph controls, query-edge
│   │                    exclusion; --gnn_residual
│   ├── rgcn.py  gatv2.py  graphsage.py  compgcn.py
│   └── layers/          RelationalSAGELayer, CompGCNLayer, coefficient-wise activations
│                        (RGCN and GATv2 use PyG's RGCNConv and GATv2Conv)
└── clifford/            Clifford encoders: one multivector per relation in place of a matrix
    ├── clifford_algebra.py  CliffordAlgebra Cl(p, q): product tables, quadratic form, init
    ├── rgcn.py  gatv2.py  graphsage.py
    └── layers/          relational.py (CliffordLinear, relational_transform, relational_aggregate),
                         activations.py (split activations, multivector gates),
                         rgcn.py  gatv2.py  graphsage.py (one layer per encoder)
```

## Results

Each dense encoder next to its Clifford counterpart, in which every per-relation matrix is replaced
by one multivector per relation.

**Recipe, identical for every row:**
- **Scoring and loss:** DistMult decoder unless the row names another, 1vsAll with reciprocal relations, filtered evaluation.
- **Optimisation:** Adagrad at lr 0.1, input dropout 0.2, xavier initialisation, batch 1024.
- **Stopping:** at most 300 epochs, validation every 25 epochs, early stopping with patience 4.
- **Encoder settings:** `--query_edge_exclusion none`; dense RGCN with `--num_bases 2`; GNN
  dropout 0.2 between ReLU layers, 0 for 1L and linear models.

"2L linear" removes the activation between the two layers. "No encoder" rows score the embedding
tables directly. Params is the total parameter count, embedding tables included.

### CoDEx-S, d = 32 (seed 1)

Clifford models use Cl(0,1), i.e. 16 complex channels.

| Model | Params | Train MRR | Test MRR | Test H@1 | Test H@10 |
|---|---:|---:|---:|---:|---:|
| RGCN 1L | 71,048 | 0.521 | 0.372 | 0.258 | 0.604 |
| CliffordRGCN 1L | 70,528 | 0.541 | 0.379 | 0.266 | 0.608 |
| RGCN 2L | 74,320 | 0.399 | 0.346 | 0.238 | 0.557 |
| CliffordRGCN 2L | 73,280 | 0.428 | 0.346 | 0.239 | 0.561 |
| RGCN 2L linear | 74,320 | 0.510 | 0.370 | 0.257 | 0.598 |
| CliffordRGCN 2L linear | 73,280 | 0.532 | 0.370 | 0.258 | 0.595 |
| GATv2 2L | 86,368 | 0.398 | 0.343 | 0.233 | 0.562 |
| CliffordGATv2 2L | 74,304 | 0.501 | 0.376 | 0.267 | 0.593 |
| GraphSAGE 2L | 241,920 | 0.435 | 0.363 | 0.248 | 0.590 |
| CliffordGraphSAGE 2L | 73,280 | 0.423 | 0.343 | 0.232 | 0.568 |

### CoDEx-S, d = 128 (seed 1)

Clifford models in Cl(0,1) (64 complex channels) and in Cl(7,0) (one 128-coefficient multivector).

| Model | Params | Train MRR | Test MRR | Test H@1 | Test H@10 |
|---|---:|---:|---:|---:|---:|
| RGCN 1L | 320,552 | 0.962 | 0.424 | 0.313 | 0.639 |
| CliffordRGCN 1L, Cl(0,1) | 282,112 | 0.955 | 0.413 | 0.298 | 0.641 |
| CliffordRGCN 1L, Cl(7,0) | 282,112 | 0.924 | 0.404 | 0.286 | 0.634 |
| RGCN 2L | 370,000 | 0.630 | 0.360 | 0.248 | 0.575 |
| CliffordRGCN 2L, Cl(0,1) | 293,120 | 0.763 | 0.381 | 0.272 | 0.597 |
| CliffordRGCN 2L, Cl(7,0) | 293,120 | 0.706 | 0.379 | 0.266 | 0.605 |
| RGCN 2L linear | 370,000 | 0.935 | 0.419 | 0.303 | 0.649 |
| CliffordRGCN 2L linear, Cl(0,1) | 293,120 | 0.955 | 0.413 | 0.303 | 0.639 |
| CliffordRGCN 2L linear, Cl(7,0) | 293,120 | 0.906 | 0.395 | 0.275 | 0.629 |
| GATv2 2L | 529,792 | 0.539 | 0.381 | 0.265 | 0.606 |
| CliffordGATv2 2L, Cl(0,1) | 297,216 | 0.910 | 0.404 | 0.291 | 0.634 |
| CliffordGATv2 2L, Cl(7,0) | 297,216 | 0.818 | 0.389 | 0.271 | 0.624 |
| GraphSAGE 2L | 3,056,640 | 0.490 | 0.370 | 0.257 | 0.591 |
| CliffordGraphSAGE 2L, Cl(0,1) | 293,120 | 0.514 | 0.362 | 0.249 | 0.591 |
| CliffordGraphSAGE 2L, Cl(7,0) | 293,120 | 0.612 | 0.360 | 0.243 | 0.585 |

### CoDEx-M, d = 32 (mean of seeds 1–2)

| Model | Params | Train MRR | Test MRR (seed 1 / 2) | Test H@1 | Test H@10 |
|---|---:|---:|---:|---:|---:|
| DistMult, no encoder | 548,864 | 0.454 | 0.302 (0.303 / 0.301) | 0.225 | 0.447 |
| ComplEx, no encoder | 548,864 | 0.519 | 0.306 (0.305 / 0.306) | 0.228 | 0.451 |
| Clifford decoder, Cl(0,2), no encoder | 548,864 | 0.518 | 0.304 (0.304 / 0.303) | 0.225 | 0.451 |
| RGCN 1L | 552,172 | 0.440 | 0.304 (0.304 / 0.303) | 0.225 | 0.452 |
| CliffordRGCN 1L, Cl(0,1) | 552,192 | 0.458 | 0.306 (0.305 / 0.308) | 0.230 | 0.453 |

### CoDEx-M, d = 32, two layers (seed 1)

| Model | Params | Train MRR | Test MRR | Test H@1 | Test H@10 |
|---|---:|---:|---:|---:|---:|
| RGCN 2L | 555,480 | 0.323 | 0.268 | 0.197 | 0.402 |
| CliffordRGCN 2L, Cl(0,1) | 555,520 | 0.347 | 0.271 | 0.198 | 0.409 |
| RGCN 2L linear | 555,480 | 0.436 | 0.302 | 0.224 | 0.449 |
| CliffordRGCN 2L linear, Cl(0,1) | 555,520 | 0.466 | 0.308 | 0.231 | 0.456 |
| GATv2 2L | 568,032 | 0.334 | 0.274 | 0.202 | 0.415 |
| CliffordGATv2 2L, Cl(0,1) | 556,544 | 0.399 | 0.292 | 0.216 | 0.438 |
| CliffordGraphSAGE 2L, Cl(0,1) | 555,520 | 0.360 | 0.279 | 0.204 | 0.426 |

### CoDEx-M, d = 128, no encoder (mean of seeds 1–2)

| Model | Params | Train MRR | Test MRR (seed 1 / 2) | Test H@1 | Test H@10 |
|---|---:|---:|---:|---:|---:|
| DistMult | 2,195,456 | 0.919 | 0.325 (0.326 / 0.325) | 0.251 | 0.467 |
| ComplEx | 2,195,456 | 0.979 | 0.327 (0.327 / 0.327) | 0.253 | 0.466 |
| Clifford decoder, Cl(0,2) | 2,195,456 | 0.980 | 0.328 (0.328 / 0.328) | 0.255 | 0.468 |

### WN18RR, d = 32 (seed 1)

Clifford models use Cl(0,1), i.e. 16 complex channels.

| Model | Params | Train MRR | Test MRR | Test H@1 | Test H@10 |
|---|---:|---:|---:|---:|---:|
| DistMult, no encoder | 1,310,880 | 0.946 | 0.439 | 0.407 | 0.498 |
| RGCN 1L | 1,314,028 | 0.961 | 0.424 | 0.391 | 0.485 |
| CliffordRGCN 1L | 1,311,648 | 0.966 | 0.435 | 0.401 | 0.496 |
| RGCN 2L | 1,317,176 | 0.825 | 0.372 | 0.320 | 0.464 |
| CliffordRGCN 2L | 1,312,416 | 0.828 | 0.351 | 0.298 | 0.447 |
| RGCN 2L linear | 1,317,176 | 0.962 | 0.415 | 0.383 | 0.477 |
| CliffordRGCN 2L linear | 1,312,416 | 0.972 | 0.429 | 0.396 | 0.487 |
| GATv2 2L\* | 1,327,488 | 0.464 | 0.188 | 0.070 | 0.439 |
| CliffordGATv2 2L | 1,313,440 | 0.905 | 0.413 | 0.371 | 0.489 |
| GraphSAGE 2L | 1,358,048 | 0.871 | 0.410 | 0.350 | 0.519 |
| CliffordGraphSAGE 2L | 1,312,416 | 0.850 | 0.388 | 0.332 | 0.487 |

\* Outlier, not yet investigated: H@10 is in line with the other models but H@1 is far below,
and train MRR is about half of theirs, so this run likely failed to fit rather than reflecting
GATv2 itself. Read the GATv2 pair with care.
