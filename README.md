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
    ├── rgcn.py  gatv2.py  graphsage.py
    └── layers/          relational.py (CliffordLinear, relational_transform, relational_aggregate),
                         activations.py (split activations, multivector gates),
                         rgcn.py  gatv2.py  graphsage.py (one layer per encoder)
```

## Results

Each dense encoder next to its Clifford counterpart, in which every per-relation matrix is replaced
by one multivector per relation. Earlier results are recorded in
[docs/experiments.md](docs/experiments.md).

**Recipe, identical for every row:**
- **Scoring and loss:** DistMult decoder, 1vsAll with reciprocal relations, filtered evaluation of
  tail queries in both directions.
- **Optimisation:** Adagrad at lr 0.1, input dropout 0.2, xavier initialisation, batch 1024.
- **Stopping:** at most 300 epochs, validation every 25 epochs, early stopping with patience 4.
- **Encoder settings:** query-edge exclusion none; dense RGCN with 2 bases; 4 heads for GATv2 and
  CliffordGATv2.

"2L linear" removes the activation between the two layers. Params is the total parameter count,
embedding tables included.

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
| RGCN 1L | 552,172 | 0.440 | 0.304 (0.304 / 0.303) | 0.225 | 0.452 |
| CliffordRGCN 1L, Cl(0,1) | 552,192 | 0.458 | 0.306 (0.305 / 0.308) | 0.230 | 0.453 |
