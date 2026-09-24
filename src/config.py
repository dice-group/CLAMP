"""CLI configuration for the GNN-KGE playground.

A flat ``argparse`` namespace carrying both standard KGE fields
(``dataset_dir``, ``embedding_dim``, ``num_epochs``, ...) and the extra
fields these GNN encoders need (``num_gnn_layers``, ``gnn_heads``,
``num_bases``, ...). Passed to models as ``vars(args)``: ``args`` is a plain
dict inside a model, never a ``Namespace``.
"""
import argparse


def get_default_arguments(args_list=None) -> argparse.Namespace:
    """Parse CLI arguments, or ``args_list`` if given (for tests)."""
    parser = argparse.ArgumentParser(add_help=True)

    # --- Knowledge graph input -------------------------------------------------
    parser.add_argument("--dataset_dir", type=str, default="KGs/UMLS",
                         help="Folder containing train.txt, valid.txt, test.txt")

    # --- Model -------------------------------------------------------------
    parser.add_argument("--model", type=str, default="RGCN",
                         choices=["KGE", "LTE", "RGCN", "GATv2", "GraphSAGE", "CompGCN",
                                  "CliffordRGCN", "CliffordGATv2", "CliffordGraphSAGE"],
                         help="Which model to train. KGE is the encoder-free baseline (no "
                              "message passing; --decoder alone decides whether it is "
                              "DistMult, ComplEx, TransE or RotatE). The Clifford* variants "
                              "encode an embedding as one full multivector of Cl(n, 0) and "
                              "therefore require a power-of-two embedding_dim.")
    parser.add_argument("--decoder", type=str, default="DistMult",
                         choices=["DistMult", "TransE", "ComplEx", "RotatE"],
                         help="Scoring function on top of the encoder's entity representations "
                              "(src/decoders.py). ComplEx/RotatE split embedding_dim in half as "
                              "(real, imaginary) and require an even embedding_dim.")
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--num_gnn_layers", type=int, default=2, help="Number of message-passing layers.")
    parser.add_argument("--gnn_dropout_rate", type=float, default=0.2, help="Dropout between GNN layers.")
    parser.add_argument("--gnn_add_inverse_edges", action=argparse.BooleanOptionalAction, default=True,
                         help="Add inverse-relation edges to the message-passing graph (R-GCN convention).")
    parser.add_argument("--gnn_residual", action=argparse.BooleanOptionalAction, default=False,
                         help="Residual connection around every GNN layer (x + layer(x)), so the "
                              "encoder can fall back to the shallow embedding model.")
    parser.add_argument("--num_bases",type=int, default=None,
                         help="RGCN basis-decomposition count. None = one weight matrix per relation.")
    parser.add_argument("--gnn_graph", type=str, default="train", choices=["train", "random", "none"],
                         help="Propagation graph: the training graph, a random-tail copy of it "
                              "(Zhang et al. 2022's RAT control), or no edges at all (their WNI).")
    parser.add_argument("--exclude_query_edges", action=argparse.BooleanOptionalAction, default=True,
                         help="Drop each training batch's own query edges from the propagation "
                              "graph. The official R-GCN/CompGCN code keeps them (--no-...).")
    parser.add_argument("--gnn_edge_dropout", type=float, default=0.0,
                         help="Fraction of propagation edges dropped per training step (R-GCN, "
                              "Schlichtkrull et al. 2018, used 0.4).")
    parser.add_argument("--compgcn_opn", type=str, default="mult", choices=["mult", "sub"],
                         help="CompGCN composition operator phi(entity, relation).")
    parser.add_argument("--gnn_heads",type=int, default=4, help="GATv2 attention heads.")
    parser.add_argument("--sage_aggregator", type=str, default="mean", choices=["mean", "max", "sum"],
                         help="GraphSAGE per-relation neighbourhood aggregator.")
    parser.add_argument("--input_dropout_rate", type=float, default=0.0)
    parser.add_argument("--hidden_dropout_rate", type=float, default=0.0)
    parser.add_argument("--normalization", type=str, default=None, choices=["LayerNorm", "BatchNorm1d", None])
    parser.add_argument("--init_param", type=str, default=None, choices=["xavier_normal", None])

    # --- Optimisation -------------------------------------------------------
    parser.add_argument("--optim", type=str, default="Adam",
                         choices=["Adam", "AdamW", "SGD", "NAdam", "Adagrad", "ASGD"])
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_epochs", type=int, default=500,
                         help="500 was needed on UMLS for RGCN/GATv2/GraphSAGE to leave the "
                              "undertrained regime (see README's 'UMLS results' section) — the "
                              "encode()->decode() full-graph forward pass is cheap, so training "
                              "longer than a typical per-triple KGE model costs little here.")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                         help="KvsAll target smoothing (ConvE/Dettmers et al. 2018: "
                              "y = y*(1-eps) + eps/num_entities), applied in the training loop. "
                              "0.0 = off (hard 0/1 targets). ConvE's own paper reports this as "
                              "essential on sparse, large-vocabulary graphs (e.g. WN18RR), where "
                              "the KvsAll positive rate is tiny (~0.003%% there vs ~5%% on UMLS) "
                              "and BCE can trivially near-zero its loss by just predicting "
                              "every entity negative, starving the gradient on the rare positive.")
    parser.add_argument("--scoring_technique", type=str, default="KvsAll", choices=["KvsAll", "1vsAll"],
                         help="KvsAll: multi-hot BCE per (h, r). 1vsAll: softmax cross-entropy per "
                              "triple (Lacroix et al. 2018; the CoDEx paper's LibKGE baselines).")
    parser.add_argument("--reciprocal", action=argparse.BooleanOptionalAction, default=False,
                         help="Reciprocal-relations decoder: also train (t, r^-1, ?) queries and "
                              "answer head queries with them. Off = head prediction is never trained.")
    parser.add_argument("--eval_every", type=int, default=0,
                         help="Validate every N epochs, keeping the best-val-MRR weights for the "
                              "final evaluation. 0 = off (last epoch is evaluated).")
    parser.add_argument("--early_stopping_patience", type=int, default=0,
                         help="Stop after this many validations without val-MRR improvement. "
                              "0 = never stop early. Needs --eval_every.")
    parser.add_argument("--num_core", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="cpu", choices=["cpu", "gpu"],
                         help="Device to train on. CPU by default per this project's "
                              "develop-on-CPU-first convention; pass --accelerator gpu explicitly "
                              "for larger graphs (e.g. WN18RR) where full-graph message passing on "
                              "CPU is the bottleneck.")

    # --- Evaluation -------------------------------------------------------
    parser.add_argument("--eval_model", type=str, default="train_val_test",
                         choices=["None", "train", "train_val", "train_val_test", "test", "val_test"])

    # --- Storage -------------------------------------------------------
    parser.add_argument("--storage_path", type=str, default="Experiments")
    parser.add_argument("--full_storage_path", type=str, default=None)
    parser.add_argument("--save_experiment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--test_runs", action="store_true", default=False)

    parsed_args = parser.parse_args(args_list)
    parsed_args.full_storage_path = parsed_args.full_storage_path or None
    return parsed_args
