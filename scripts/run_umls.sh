#!/usr/bin/env bash
# First-experiment gate: train each GNN encoder on UMLS before trying any other dataset.
# Hyperparameters below were found by a small manual lr/epoch sweep (see README's
# "UMLS results" section) — they are not a claim of a globally optimal configuration,
# just enough tuning to get each encoder past undertraining.
set -euo pipefail
cd "$(dirname "$0")/.."

python main.py --model RGCN --dataset_dir KGs/UMLS \
    --embedding_dim 32 --num_gnn_layers 2 \
    --lr 0.01 --num_epochs 500 --batch_size 512 \
    --eval_model train_val_test

python main.py --model GATv2 --dataset_dir KGs/UMLS \
    --embedding_dim 32 --num_gnn_layers 2 --gnn_heads 4 \
    --lr 0.02 --num_epochs 500 --batch_size 512 \
    --eval_model train_val_test

python main.py --model GraphSAGE --dataset_dir KGs/UMLS \
    --embedding_dim 32 --num_gnn_layers 2 \
    --lr 0.15 --num_epochs 500 --batch_size 512 \
    --eval_model train_val_test
