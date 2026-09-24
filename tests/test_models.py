"""Smoke tests: build each model on a tiny synthetic graph.

Correctness gates checked here (see README's Verification section):
* every model is an instance of PyG's KGEModel and satisfies this project's
  interface (forward_triples returns (B,), forward_k_vs_all returns
  (B, num_entities));
* gradients reach every model's parameters.
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
        "reciprocal": model_name == "CompGCN",  # CompGCN requires it
    }


ALL_MODELS = ["KGE", "LTE", "RGCN", "GATv2", "GraphSAGE", "CompGCN",
              "CliffordRGCN", "CliffordGATv2", "CliffordGraphSAGE"]


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_forward_shapes(model_name):
    from torch_geometric.nn.kge.base import KGEModel

    args = _toy_args(model_name)
    model = MODEL_MAP[model_name](args=args)
    assert isinstance(model, KGEModel)

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


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_gradients_flow(model_name):
    args = _toy_args(model_name)
    model = MODEL_MAP[model_name](args=args)
    hr = torch.randint(0, args["num_entities"], (4, 2))
    hr[:, 1] = torch.randint(0, args["num_relations"], (4,))
    scores = model.forward_k_vs_all(hr)
    scores.sum().backward()
    grad_found = any(p.grad is not None and torch.any(p.grad != 0) for p in model.parameters())
    assert grad_found, "No gradient reached any parameter."


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_every_parameter_is_trained(model_name):
    """Every registered parameter must be reached by the loss.

    The toy graph is seeded and dense enough to exercise all relations, so a
    parameter without a gradient means it is unreachable by construction - e.g.
    a weight that was accidentally stored as a plain tensor rather than a
    ``Parameter``, which would silently never be optimized.
    """
    args = _toy_args(model_name)
    model = MODEL_MAP[model_name](args=args)
    hr = torch.randint(0, args["num_entities"], (4, 2))
    hr[:, 1] = torch.randint(0, args["num_relations"], (4,))
    model.forward_k_vs_all(hr).sum().backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing, f"No gradient reached: {missing}"


def test_kge_baseline_does_no_message_passing():
    """The KGE baseline's entity representations must be the raw embedding table.

    That is the whole definition of the control: identical pipeline, encoder
    removed. It must also score without a propagation graph attached at all.
    """
    args = _toy_args("KGE")
    del args["train_set_idx"]
    model = MODEL_MAP["KGE"](args=args)

    assert len(model.gnn_layers) == 0
    assert torch.equal(model.encode(*model.graph), model.node_emb.weight)

    hr = torch.randint(0, args["num_entities"], (4, 2))
    hr[:, 1] = torch.randint(0, args["num_relations"], (4,))
    assert model.forward_k_vs_all(hr).shape == (4, args["num_entities"])


@pytest.mark.parametrize("model_name", ["RGCN", "KGE"])
@pytest.mark.parametrize("decoder", ["DistMult", "TransE", "ComplEx", "RotatE"])
def test_decoder_shapes(decoder, model_name):
    args = _toy_args(model_name)
    args["decoder"] = decoder
    model = MODEL_MAP[model_name](args=args)

    triples = torch.randint(0, args["num_entities"], (5, 3))
    triples[:, 1] = torch.randint(0, args["num_relations"], (5,))
    assert model.forward_triples(triples).shape == (5,)
    assert model.forward_k_vs_all(triples[:, :2]).shape == (5, args["num_entities"])

    rt = triples[:, [1, 2]]
    assert model.forward_k_vs_all_head(rt).shape == (5, args["num_entities"])


def test_unknown_decoder_rejected():
    args = _toy_args("RGCN")
    args["decoder"] = "Nope"
    with pytest.raises(ValueError):
        MODEL_MAP["RGCN"](args=args)


@pytest.mark.parametrize("decoder", ["ComplEx", "RotatE"])
def test_complex_decoder_rejects_odd_embedding_dim(decoder):
    args = _toy_args("RGCN", embedding_dim=7)
    args["decoder"] = decoder
    with pytest.raises(ValueError):
        MODEL_MAP["RGCN"](args=args)


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_reciprocal_head_scores_are_the_inverse_tail_query(model_name):
    args = {**_toy_args(model_name), "reciprocal": True}
    model = MODEL_MAP[model_name](args=args).eval()
    assert model.rel_emb.num_embeddings == 2 * args["num_relations"]
    rt = torch.tensor([[0, 1], [2, 3]])
    inverse_query = torch.tensor([[1, 0 + args["num_relations"]], [3, 2 + args["num_relations"]]])
    assert torch.allclose(model.forward_k_vs_all_head(rt), model.forward_k_vs_all(inverse_query))


def test_reciprocal_query_excludes_its_own_edges():
    args = {**_toy_args("RGCN"), "train_set_idx": np.array([[0, 0, 1], [2, 0, 1], [0, 1, 3]])}
    model = MODEL_MAP["RGCN"](args=args).train()
    num_relations = args["num_relations"]
    # (1, 0^-1, ?) asks for the heads of (*, 0, 1): both such triples leave the graph, both directions.
    edge_index, edge_type = model._graph_excluding_batch(torch.tensor([1]), torch.tensor([num_relations]))
    remaining = {(int(s), int(r), int(d)) for (s, d), r in zip(edge_index.t(), edge_type)}
    assert remaining == {(0, 1, 3), (3, 1 + num_relations, 0)}


def test_compgcn_requires_reciprocal():
    with pytest.raises(ValueError, match="reciprocal"):
        MODEL_MAP["CompGCN"](args={**_toy_args("CompGCN"), "reciprocal": False})


@pytest.mark.parametrize("gnn_graph, expected_edges", [("train", 6), ("random", 6), ("none", 0)])
def test_gnn_graph_controls(gnn_graph, expected_edges):
    triples = np.array([[0, 0, 1], [2, 0, 1], [0, 1, 3]])
    args = {**_toy_args("RGCN"), "train_set_idx": triples, "gnn_graph": gnn_graph}
    edge_index, edge_type = MODEL_MAP["RGCN"](args=args).graph
    assert edge_index.shape[1] == expected_edges
    if gnn_graph == "random":
        # Heads and relations are kept; only tails are resampled.
        assert sorted(edge_index[0, :3].tolist()) == [0, 0, 2]
        assert edge_type[:3].tolist() == [0, 0, 1]


def test_query_edges_kept_when_exclusion_off():
    args = {**_toy_args("RGCN"), "train_set_idx": np.array([[0, 0, 1], [2, 0, 1]]),
            "query_edge_exclusion": "none"}
    model = MODEL_MAP["RGCN"](args=args).train()
    edge_index, _ = model._graph_excluding_batch(torch.tensor([0]), torch.tensor([0]))
    assert edge_index.shape[1] == 4


def test_triple_exclusion_keeps_the_query_pairs_other_answers():
    # (0, 0, ?) has two known answers; predicting 1 must hide only (0, 0, 1).
    triples = np.array([[0, 0, 1], [0, 0, 2], [3, 1, 0]])
    args = {**_toy_args("RGCN"), "train_set_idx": triples, "query_edge_exclusion": "triple"}
    model = MODEL_MAP["RGCN"](args=args).train()
    num_relations = args["num_relations"]
    edge_index, edge_type = model._graph_excluding_batch(torch.tensor([0]), torch.tensor([0]), torch.tensor([1]))
    remaining = {(int(s), int(r), int(d)) for (s, d), r in zip(edge_index.t(), edge_type)}
    assert remaining == {(0, 0, 2), (2, num_relations, 0), (3, 1, 0), (0, 1 + num_relations, 3)}
    # The reciprocal query (1, 0^-1, ?) with answer 0 hides the same triple.
    edge_index, _ = model._graph_excluding_batch(
        torch.tensor([1]), torch.tensor([num_relations]), torch.tensor([0]))
    assert edge_index.shape[1] == 4
    # Without answers it falls back to hiding every (0, 0, *) edge.
    edge_index, _ = model._graph_excluding_batch(torch.tensor([0]), torch.tensor([0]))
    assert edge_index.shape[1] == 2
