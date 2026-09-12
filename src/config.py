"""CLI configuration for the GNN-KGE playground.

Mirrors ``literal-embeddings/src/config.py``: a flat ``argparse`` namespace
that carries both dicee-standard fields (``dataset_dir``, ``embedding_dim``,
``num_epochs``, ...) and the extra fields these GNN encoders need
(``num_gnn_layers``, ``gnn_heads``, ``num_bases``, ...). Passed to models as
``vars(args)`` per dicee convention (``args`` is a plain dict inside a
``BaseKGE`` subclass, never a ``Namespace``).
"""
import argparse


def get_default_arguments(args_list=None) -> argparse.Namespace:
    """Parse CLI arguments, or ``args_list`` if given (for tests)."""
    parser = argparse.ArgumentParser(add_help=True)

    # --- Knowledge graph input -------------------------------------------------
    parser.add_argument("--dataset_dir", type=str, default="KGs/UMLS",
                         help="Folder containing train.txt, valid.txt, test.txt")
    parser.add_argument("--backend", type=str, default="pandas", choices=["pandas", "polars", "rdflib"])
    parser.add_argument("--separator", type=str, default=r"\s+")

    # --- Model -------------------------------------------------------------
    parser.add_argument("--model", type=str, default="RGCN", choices=["RGCN", "GATv2", "GraphSAGE"],
                         help="Which GNN-encoder / DistMult-decoder KGE model to train.")
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--num_gnn_layers", type=int, default=2, help="Number of message-passing layers.")
    parser.add_argument("--gnn_dropout_rate", type=float, default=0.2, help="Dropout between GNN layers.")
    parser.add_argument("--gnn_add_inverse_edges", action="store_true", default=True,
                         help="Add inverse-relation edges to the message-passing graph (R-GCN convention).")
    parser.add_argument("--num_bases", type=int, default=None,
                         help="RGCN basis-decomposition count. None = one weight matrix per relation.")
    parser.add_argument("--gnn_heads", type=int, default=4, help="GATv2 attention heads.")
    parser.add_argument("--sage_aggregator", type=str, default="mean", choices=["mean", "max", "sum"],
                         help="GraphSAGE per-relation neighbourhood aggregator.")
    parser.add_argument("--input_dropout_rate", type=float, default=0.0)
    parser.add_argument("--hidden_dropout_rate", type=float, default=0.0)
    parser.add_argument("--normalization", type=str, default=None, choices=["LayerNorm", "BatchNorm1d", None])
    parser.add_argument("--init_param", type=str, default=None, choices=["xavier_normal", None])

    # --- Optimisation -------------------------------------------------------
    parser.add_argument("--optim", type=str, default="Adam",
                         choices=["Adam", "AdamW", "SGD", "NAdam", "Adagrad", "ASGD", "Adopt"])
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_epochs", type=int, default=500,
                         help="500 was needed on UMLS for RGCN/GATv2/GraphSAGE to leave the "
                              "undertrained regime (see README's 'UMLS results' section) — the "
                              "encode()->decode() full-graph forward pass is cheap, so training "
                              "longer than a typical per-triple KGE model costs little here.")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--scoring_technique", type=str, default="KvsAll", choices=["KvsAll"])
    parser.add_argument("--num_core", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--random_seed", type=int, default=1)

    # --- Evaluation -------------------------------------------------------
    parser.add_argument("--eval_model", type=str, default="train_val_test",
                         choices=["None", "train", "train_val", "train_val_test", "test", "val_test"])

    # --- dicee KG-loading passthroughs (required by dicee.static_funcs.read_or_load_kg) ---
    parser.add_argument("--path_single_kg", type=str, default=None)
    parser.add_argument("--sparql_endpoint", type=str, default=None)
    parser.add_argument("--byte_pair_encoding", action="store_true", default=False)
    parser.add_argument("--add_noise_rate", type=float, default=0.0)
    parser.add_argument("--read_only_few", type=int, default=None)
    parser.add_argument("--sample_triples_ratio", type=float, default=None)
    parser.add_argument("--apply_reciprical_or_noise", action="store_true", default=False,
                         help="Off by default: this project builds its own inverse edges for message "
                              "passing (see src/graph_utils.py) and does not want dicee to also inject "
                              "reciprocal relations into the decoder's relation set.")

    # --- Storage -------------------------------------------------------
    parser.add_argument("--storage_path", type=str, default="Experiments")
    parser.add_argument("--full_storage_path", type=str, default=None)
    parser.add_argument("--save_experiment", action="store_true", default=True)
    parser.add_argument("--test_runs", action="store_true", default=False)

    parsed_args = parser.parse_args(args_list)
    parsed_args.learning_rate = parsed_args.lr
    parsed_args.full_storage_path = parsed_args.full_storage_path or None
    return parsed_args
