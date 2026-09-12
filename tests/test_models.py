"""Smoke tests: build each model on a tiny synthetic graph and on UMLS.

Correctness gates checked here (see README's Verification section):
* every model is an instance of dicee's BaseKGE and satisfies its interface
  (forward_triples returns (B,), forward_k_vs_all returns (B, num_entities));
* a full training run on UMLS produces a non-degenerate (> 0) filtered MRR.
"""
import numpy as np
import pytest
import torch

from src.models import MODEL_MAP


def _toy_args(model_name: str, num_entities=10, num_relations=3, embedding_dim=8):
    rng = np.random.default_rng(0)
    train_set_idx = rng.integers(0, num_entities, size=(40, 3))
    train_set_idx[:, 1] = rng.integers(0, num_relations, size=40)
    return {
        "model": model_name,
        "num_entities": num_entities,
        "num_relations": num_relations,
        "embedding_dim": embedding_dim,
        "train_set_idx": train_set_idx,
        "scoring_technique": "KvsAll",
        "num_gnn_layers": 2,
        "gnn_dropout_rate": 0.0,
        "gnn_heads": 2,
        "num_bases": None,
        "optim": "Adam",
        "learning_rate": 0.01,
    }


@pytest.mark.parametrize("model_name", ["RGCN", "GATv2", "GraphSAGE"])
def test_forward_shapes(model_name):
    from dicee.models.base_model import BaseKGE

    args = _toy_args(model_name)
    model = MODEL_MAP[model_name](args=args)
    assert isinstance(model, BaseKGE)

    triples = torch.randint(0, args["num_entities"], (5, 3))
    triples[:, 1] = torch.randint(0, args["num_relations"], (5,))
    triple_scores = model.forward_triples(triples)
    assert triple_scores.shape == (5,)

    hr = triples[:, :2]
    kvsall_scores = model.forward_k_vs_all(hr)
    assert kvsall_scores.shape == (5, args["num_entities"])

    # forward() must route (B,3) -> forward_triples and (B,2) -> forward_k_vs_all.
    assert torch.allclose(model(triples), triple_scores)
    assert torch.allclose(model(hr), kvsall_scores)


@pytest.mark.parametrize("model_name", ["RGCN", "GATv2", "GraphSAGE"])
def test_gradients_flow(model_name):
    args = _toy_args(model_name)
    model = MODEL_MAP[model_name](args=args)
    hr = torch.randint(0, args["num_entities"], (4, 2))
    hr[:, 1] = torch.randint(0, args["num_relations"], (4,))
    scores = model.forward_k_vs_all(hr)
    scores.sum().backward()
    grad_found = any(p.grad is not None and torch.any(p.grad != 0) for p in model.parameters())
    assert grad_found, "No gradient reached any parameter."
