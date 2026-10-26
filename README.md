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

[Full configurations and runtimes](docs/experiments.md).

### UMLS

Config: DistMult, dim=64, Adam, lr=0.001, 1vsAll, reciprocal=true, batch=512, epochs=300, eval_every=50, seed=1.

| Model | Train MRR | Train Hits@1 | Train Hits@3 | Train Hits@10 | Test MRR | Test Hits@1 | Test Hits@3 | Test Hits@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [RGCN](Experiments/distmult-gnn-kge-300ep-20260928/UMLS/RGCN_DistMult/eval_report.json) | 0.9472 | 0.8993 | 0.9948 | 0.9991 | 0.8922 | 0.8048 | 0.9758 | 0.9947 |
| [GATv2](Experiments/distmult-gnn-kge-300ep-20260928/UMLS/GATv2_DistMult/eval_report.json) | 0.9327 | 0.8735 | 0.9905 | 0.9993 | 0.8909 | 0.8011 | 0.9781 | 0.9962 |
| [GraphSAGE](Experiments/distmult-gnn-kge-300ep-20260928/UMLS/GraphSAGE_DistMult/eval_report.json) | 0.6506 | 0.5214 | 0.7308 | 0.8721 | 0.6480 | 0.5219 | 0.7277 | 0.8669 |
| [KGE (DistMult)](Experiments/distmult-gnn-kge-300ep-20260928/UMLS/KGE_DistMult/eval_report.json) | 0.9881 | 0.9765 | 0.9996 | 1.0000 | 0.9073 | 0.8374 | 0.9750 | 0.9970 |

Config: ComplEx, dim=64, Adam, lr=0.001, 1vsAll, reciprocal=true, batch=512, epochs=300, eval_every=50, seed=1.

| Model | Train MRR | Train Hits@1 | Train Hits@3 | Train Hits@10 | Test MRR | Test Hits@1 | Test Hits@3 | Test Hits@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [KGE (ComplEx)](Experiments/distmult-gnn-kge-300ep-20260928/UMLS/KGE_ComplEx/eval_report.json) | 0.9805 | 0.9612 | 0.9999 | 1.0000 | 0.8869 | 0.7897 | 0.9834 | 0.9962 |

Config: RotatE, dim=64, Adam, lr=0.001, 1vsAll, reciprocal=true, batch=512, epochs=300, eval_every=50, seed=1.

| Model | Train MRR | Train Hits@1 | Train Hits@3 | Train Hits@10 | Test MRR | Test Hits@1 | Test Hits@3 | Test Hits@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [KGE (RotatE)](Experiments/distmult-gnn-kge-300ep-20260928/UMLS/KGE_RotatE/eval_report.json) | 0.8708 | 0.7618 | 0.9795 | 0.9927 | 0.8062 | 0.6513 | 0.9576 | 0.9864 |


### Kinship

Config: dim=64, lr=0.001, batch=512, epochs=300, eval_every=50.

Results pending.

### CoDEx-S

Config: DistMult, dim=64, Adam, lr=0.001, 1vsAll, reciprocal=true, batch=512, epochs=300, eval_every=50, seed=1.

| Model | Train MRR | Train Hits@1 | Train Hits@3 | Train Hits@10 | Test MRR | Test Hits@1 | Test Hits@3 | Test Hits@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [RGCN](Experiments/distmult-gnn-kge-300ep-20260928/CoDEx-S/RGCN_DistMult/eval_report.json) | 0.3910 | 0.2725 | 0.4458 | 0.6278 | 0.2938 | 0.1901 | 0.3299 | 0.5030 |
| [GATv2](Experiments/distmult-gnn-kge-300ep-20260928/CoDEx-S/GATv2_DistMult/eval_report.json) | 0.3491 | 0.2358 | 0.3964 | 0.5763 | 0.3345 | 0.2311 | 0.3734 | 0.5367 |
| [GraphSAGE](Experiments/distmult-gnn-kge-300ep-20260928/CoDEx-S/GraphSAGE_DistMult/eval_report.json) | 0.2075 | 0.1099 | 0.2362 | 0.4126 | 0.1845 | 0.0960 | 0.2032 | 0.3734 |
| [KGE (DistMult)](Experiments/distmult-gnn-kge-300ep-20260928/CoDEx-S/KGE_DistMult/eval_report.json) | 0.9646 | 0.9401 | 0.9878 | 0.9978 | 0.4026 | 0.2888 | 0.4562 | 0.6209 |

Config: ComplEx, dim=64, Adam, lr=0.001, 1vsAll, reciprocal=true, batch=512, epochs=300, eval_every=50, seed=1.

| Model | Train MRR | Train Hits@1 | Train Hits@3 | Train Hits@10 | Test MRR | Test Hits@1 | Test Hits@3 | Test Hits@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [KGE (ComplEx)](Experiments/distmult-gnn-kge-300ep-20260928/CoDEx-S/KGE_ComplEx/eval_report.json) | 0.9721 | 0.9499 | 0.9943 | 0.9993 | 0.3953 | 0.2823 | 0.4516 | 0.6078 |

Config: RotatE, dim=64, Adam, lr=0.001, 1vsAll, reciprocal=true, batch=512, epochs=300, eval_every=50, seed=1.

| Model | Train MRR | Train Hits@1 | Train Hits@3 | Train Hits@10 | Test MRR | Test Hits@1 | Test Hits@3 | Test Hits@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [KGE (RotatE)](Experiments/distmult-gnn-kge-300ep-20260928/CoDEx-S/KGE_RotatE/eval_report.json) | 0.6242 | 0.4944 | 0.7136 | 0.8501 | 0.3684 | 0.2429 | 0.4281 | 0.6083 |


### WN18RR

Config: dim=64, lr=0.001, batch=128, epochs=300, eval_every=50.

Results pending.

### CoDEx-M

Config: dim=64, lr=0.001, batch=128, epochs=300, eval_every=50.

Results pending.
