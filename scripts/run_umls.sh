#!/usr/bin/env bash
# First-experiment gate: train each GNN encoder on UMLS before trying any other dataset.
set -euo pipefail
cd "$(dirname "$0")/.."

for MODEL in RGCN GATv2 GraphSAGE; do
    echo "=== ${MODEL} on UMLS ==="
    python main.py \
        --model "${MODEL}" \
        --dataset_dir KGs/UMLS \
        --embedding_dim 32 \
        --num_gnn_layers 2 \
        --num_epochs 200 \
        --lr 0.01 \
        --batch_size 512 \
        --eval_model train_val_test
done
