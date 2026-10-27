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

from src.kg_loader import create_reciprocal_triples
from src.models import MODEL_MAP


def _doubled(triples, num_direct=3):
    """Reciprocal-augmented training set, as load_kg builds it."""
    return create_reciprocal_triples(np.array(triples), num_direct)


def _toy_args(model_name: str, num_entities=10, num_relations=6, embedding_dim=8):
    rng = np.random.default_rng(0)
    train_set_idx = rng.integers(0, num_entities, size=(40, 3))
    train_set_idx[:, 1] = rng.integers(0, num_relations // 2, size=40)
    train_set_idx = create_reciprocal_triples(train_set_idx, num_relations // 2)
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

    assert not hasattr(model, "gnn_layers")
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


def test_odd_relation_count_rejected():
    with pytest.raises(ValueError, match="even"):
        MODEL_MAP["RGCN"](args={**_toy_args("RGCN"), "num_relations": 5})


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


def test_reciprocal_query_excludes_its_own_edges():
    args = {**_toy_args("RGCN"), "train_set_idx": _doubled([[0, 0, 1], [2, 0, 1], [0, 1, 3]])}
    model = MODEL_MAP["RGCN"](args=args).train()
    # (1, 0^-1, ?) asks for the heads of (*, 0, 1): both such triples leave the graph, both directions.
    edge_index, edge_type = model._graph_excluding_batch(torch.tensor([1]), torch.tensor([3]))
    remaining = {(int(s), int(r), int(d)) for (s, d), r in zip(edge_index.t(), edge_type)}
    assert remaining == {(0, 1, 3), (3, 4, 0)}


@pytest.mark.parametrize("gnn_graph, expected_edges", [("train", 6), ("random", 6), ("none", 0)])
def test_gnn_graph_controls(gnn_graph, expected_edges):
    triples = _doubled([[0, 0, 1], [2, 0, 1], [0, 1, 3]])
    args = {**_toy_args("RGCN"), "train_set_idx": triples, "gnn_graph": gnn_graph}
    edge_index, edge_type = MODEL_MAP["RGCN"](args=args).graph
    assert edge_index.shape[1] == expected_edges
    if gnn_graph == "random":
        # Heads and relations are kept; only tails are resampled.
        assert sorted(edge_index[0, :3].tolist()) == [0, 0, 2]
        assert edge_type[:3].tolist() == [0, 0, 1]
        # The inverse edges mirror the resampled ones.
        assert edge_index[:, 3:].flip(0).tolist() == edge_index[:, :3].tolist()
        assert edge_type[3:].tolist() == [3, 3, 4]


def test_query_edges_kept_when_exclusion_off():
    args = {**_toy_args("RGCN"), "train_set_idx": _doubled([[0, 0, 1], [2, 0, 1]]),
            "query_edge_exclusion": "none"}
    model = MODEL_MAP["RGCN"](args=args).train()
    edge_index, _ = model._graph_excluding_batch(torch.tensor([0]), torch.tensor([0]))
    assert edge_index.shape[1] == 4


def test_triple_exclusion_keeps_the_query_pairs_other_answers():
    # (0, 0, ?) has two known answers; predicting 1 must hide only (0, 0, 1).
    triples = _doubled([[0, 0, 1], [0, 0, 2], [3, 1, 0]])
    args = {**_toy_args("RGCN"), "train_set_idx": triples, "query_edge_exclusion": "triple"}
    model = MODEL_MAP["RGCN"](args=args).train()
    edge_index, edge_type = model._graph_excluding_batch(torch.tensor([0]), torch.tensor([0]), torch.tensor([1]))
    remaining = {(int(s), int(r), int(d)) for (s, d), r in zip(edge_index.t(), edge_type)}
    assert remaining == {(0, 0, 2), (2, 3, 0), (3, 1, 0), (0, 4, 3)}
    # The reciprocal query (1, 0^-1, ?) with answer 0 hides the same triple.
    edge_index, _ = model._graph_excluding_batch(
        torch.tensor([1]), torch.tensor([3]), torch.tensor([0]))
    assert edge_index.shape[1] == 4
    # Without answers it falls back to hiding every (0, 0, *) edge.
    edge_index, _ = model._graph_excluding_batch(torch.tensor([0]), torch.tensor([0]))
    assert edge_index.shape[1] == 2


def test_graph_without_reciprocal_triples_rejected():
    args = {**_toy_args("RGCN"), "train_set_idx": np.array([[0, 0, 1], [2, 1, 3]])}
    with pytest.raises(ValueError, match="reciprocal"):
        MODEL_MAP["RGCN"](args=args)


@pytest.mark.parametrize("model_name", ["RGCN", "CliffordRGCN", "CliffordGATv2", "CliffordGraphSAGE"])
@pytest.mark.parametrize("activation", ["relu", "none"])
def test_gnn_activation_control(model_name, activation):
    """--gnn_activation none must skip the hidden activation, so it encodes differently."""
    encodings = {}
    for setting in (activation, "none"):
        torch.manual_seed(0)
        args = {**_toy_args(model_name), "gnn_activation": setting}
        model = MODEL_MAP[model_name](args=args).eval()
        with torch.no_grad():
            encodings[setting] = model.encode(*model.graph)
    differs = not torch.allclose(encodings[activation], encodings["none"], atol=1e-6)
    assert differs == (activation != "none")


@pytest.mark.parametrize("final_norm", ["l2", "none", "l2_scale"])
def test_graphsage_final_norm(final_norm):
    args = _toy_args("GraphSAGE")
    args["sage_final_norm"] = final_norm
    model = MODEL_MAP["GraphSAGE"](args=args).eval()
    if final_norm == "l2_scale":
        with torch.no_grad():
            model.output_scale.fill_(3.0)
    norms = model.encode(*model.graph).norm(dim=1)
    expected = {"l2": 1.0, "l2_scale": 3.0}.get(final_norm)
    if expected is None:
        assert not torch.allclose(norms, torch.ones_like(norms))
    else:
        assert torch.allclose(norms, torch.full_like(norms, expected), atol=1e-5)
    hr = torch.tensor([[0, 1], [2, 4]])
    model.train().forward_k_vs_all(hr).sum().backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing, f"No gradient reached: {missing}"


@pytest.mark.parametrize("aggr", ["mean", "sum", "max"])
def test_relational_sage_layer_matches_per_relation_convs(aggr):
    """The grouped pass equals summing PyG's SAGEConv, holding W_r, over each relation's edges."""
    from torch_geometric.nn import SAGEConv

    from src.models.gnn.layers.graphsage import RelationalSAGELayer

    torch.manual_seed(0)
    num_nodes, num_relations = 12, 5
    layer = RelationalSAGELayer(8, 8, num_relations, aggr)
    x = torch.randn(num_nodes, 8)
    edge_index = torch.randint(0, num_nodes, (2, 60))
    edge_type = torch.randint(0, num_relations - 1, (60,))  # last relation has no edges
    expected = layer.root_lin(x)
    conv = SAGEConv(8, 8, aggr=aggr, root_weight=False, bias=False)
    for r in range(num_relations):
        mask = edge_type == r
        if mask.any():
            with torch.no_grad():
                conv.lin_l.weight.copy_(layer.rel_weight[r].T)
            expected = expected + conv(x, edge_index[:, mask])
    assert torch.allclose(layer(x, edge_index, edge_type), expected, atol=1e-6)


@pytest.mark.parametrize("signature", [None, [0, 1]])
def test_clifford_rgcn_bases(signature):
    """With bases, w_r = comp[r] @ bases: the layer equals the free layer holding those w_r."""
    from src.models.clifford.layers.rgcn import CliffordRGCNConv

    args = {**_toy_args("CliffordRGCN"), "clifford_num_bases": 2, "clifford_signature": signature}
    model = MODEL_MAP["CliffordRGCN"](args=args)
    conv = model.gnn_layers[0]
    assert conv.bases.shape == (2, args["embedding_dim"])
    assert conv.comp.shape == (args["num_relations"], 2)
    assert not hasattr(conv, "rel_weight")

    free = CliffordRGCNConv(model.clifford_algebra, args["num_relations"])
    with torch.no_grad():
        free.rel_weight.copy_(conv.comp @ conv.bases)
        free.root.load_state_dict(conv.root.state_dict())
    x = torch.randn(args["num_entities"], args["embedding_dim"])
    edge_index, edge_type = model.graph
    assert torch.allclose(conv(x, edge_index, edge_type), free(x, edge_index, edge_type), atol=1e-6)

    hr = torch.tensor([[0, 1], [2, 4], [3, 5]])
    model.forward_k_vs_all(hr).sum().backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing, f"No gradient reached: {missing}"

    with pytest.raises(ValueError, match="num_bases"):
        CliffordRGCNConv(model.clifford_algebra, args["num_relations"], 0)


def test_clifford_rgcn_ignores_dense_num_bases():
    """--num_bases is the dense RGCN flag; CliffordRGCN reads only --clifford_num_bases."""
    model = MODEL_MAP["CliffordRGCN"](args={**_toy_args("CliffordRGCN"), "num_bases": 2})
    assert model.gnn_layers[0].num_bases is None


@pytest.mark.parametrize("signature, reference", [([0, 0], "DistMult"), ([0, 1], "ComplEx")])
def test_clifford_decoder_reduces_to_distmult_and_complex(signature, reference):
    """<h * r, t> is DistMult in Cl(0,0) and ComplEx in Cl(0,1) (blade-major = [re | im])."""
    args = {**_toy_args("KGE"), "decoder": "Clifford", "clifford_signature": signature}
    model = MODEL_MAP["KGE"](args=args).eval()
    ref = MODEL_MAP["KGE"](args={**_toy_args("KGE"), "decoder": reference}).eval()
    ref.load_state_dict(model.state_dict())
    triples = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 5, 7]])
    assert torch.allclose(model.forward_triples(triples), ref.forward_triples(triples), atol=1e-6)
    hr = triples[:, :2]
    scores = model.forward_k_vs_all(hr)
    assert torch.allclose(scores, ref.forward_k_vs_all(hr), atol=1e-6)
    assert torch.allclose(scores[torch.arange(3), triples[:, 2]], model.forward_triples(triples), atol=1e-6)


@pytest.mark.parametrize("query_rate", [0.0, 0.5])
@pytest.mark.parametrize("decoder, signature, reference",
                         [("DistMult", None, [0, 0]), ("ComplEx", None, [0, 1])])
def test_query_dropout_is_shared_by_bilinear_decoders(decoder, signature, reference, query_rate):
    """The same query dropout gives the same training scores for DistMult/ComplEx and Clifford."""
    base = {**_toy_args("KGE"), "input_dropout_rate": 0.3, "query_dropout_rate": query_rate}
    model = MODEL_MAP["KGE"](args={**base, "decoder": decoder}).train()
    clifford = MODEL_MAP["KGE"](
        args={**base, "decoder": "Clifford", "clifford_signature": reference}).train()
    clifford.load_state_dict(model.state_dict())
    hr = torch.tensor([[0, 1], [2, 4], [3, 5]])
    torch.manual_seed(0)
    scores = model.forward_k_vs_all(hr)
    torch.manual_seed(0)
    assert torch.allclose(scores, clifford.forward_k_vs_all(hr), atol=1e-6)
    no_query = MODEL_MAP["KGE"](args={**base, "decoder": decoder, "query_dropout_rate": 0.0}).train()
    no_query.load_state_dict(model.state_dict())
    torch.manual_seed(0)
    assert torch.allclose(no_query.forward_k_vs_all(hr), scores, atol=1e-6) == (query_rate == 0.0)


def test_query_dropout_rejects_distance_decoders():
    with pytest.raises(ValueError, match="bilinear"):
        MODEL_MAP["KGE"](args={**_toy_args("KGE"), "decoder": "TransE", "query_dropout_rate": 0.2})


@pytest.mark.parametrize("model_name", ["KGE", "CliffordRGCN"])
@pytest.mark.parametrize("signature", [None, [0, 2], [1, 1]])
def test_clifford_decoder_trains(model_name, signature):
    args = {**_toy_args(model_name), "decoder": "Clifford", "clifford_signature": signature}
    model = MODEL_MAP[model_name](args=args)
    hr = torch.tensor([[0, 1], [2, 4], [3, 5]])
    model.forward_k_vs_all(hr).sum().backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing, f"No gradient reached: {missing}"
    assert not any(k.startswith("decoder_algebra") for k in model.state_dict())


def test_compgcn_complex_compositions_match_decoders():
    from src.models.gnn.layers.compgcn import _compose

    x, r = torch.randn(5, 8), torch.randn(5, 8)
    xc = torch.complex(x[:, :4], x[:, 4:])
    rc = torch.complex(r[:, :4], r[:, 4:])
    out = _compose(x, r, "cmult")
    assert torch.allclose(torch.complex(out[:, :4], out[:, 4:]), xc * rc, atol=1e-6)
    out = _compose(x, r, "rotate")
    rot = torch.polar(torch.ones(5, 4), r[:, :4])
    assert torch.allclose(torch.complex(out[:, :4], out[:, 4:]), xc * rot, atol=1e-6)
    assert torch.allclose(out[:, :4] ** 2 + out[:, 4:] ** 2, xc.abs() ** 2, atol=1e-5)


@pytest.mark.parametrize("opn", ["cmult", "rotate"])
def test_compgcn_complex_compositions_train(opn):
    args = _toy_args("CompGCN")
    args["compgcn_opn"] = opn
    model = MODEL_MAP["CompGCN"](args=args)
    hr = torch.tensor([[0, 1], [2, 4], [3, 5]])
    assert model.forward_k_vs_all(hr).shape == (3, args["num_entities"])
    args["embedding_dim"] = 7
    with pytest.raises(ValueError, match="even embedding_dim"):
        MODEL_MAP["CompGCN"](args=args)
