import argparse


def get_default_arguments(args_list=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_dir", type=str, default="KGs/UMLS",
        help="Folder containing train.txt, valid.txt, test.txt",
    )
    parser.add_argument(
        "--vocab_from_all_splits", action=argparse.BooleanOptionalAction, default=False,
        help="Include validation/test entities and relations in the vocabulary.",
    )

    parser.add_argument(
        "--model", type=str, default="RGCN",
        choices=[
            "KGE", "LTE", "RGCN", "GATv2", "GraphSAGE", "CompGCN", "CliffordRGCN",
            "CliffordGATv2", "CliffordGraphSAGE",
        ],
        help="Encoder; Clifford models need a power-of-two embedding dimension.",
    )
    parser.add_argument(
        "--decoder", type=str, default="DistMult",
        choices=["DistMult", "TransE", "ComplEx", "RotatE"],
        help="ComplEx and RotatE require an even embedding dimension.",
    )
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument(
        "--num_gnn_layers", type=int, default=2,
        help="Number of message-passing layers.",
    )
    parser.add_argument(
        "--gnn_dropout_rate", type=float, default=0.2,
        help="Dropout between GNN layers.",
    )
    parser.add_argument(
        "--gnn_add_inverse_edges", action=argparse.BooleanOptionalAction, default=True,
        help="Add reverse edges with separate relation IDs.",
    )
    parser.add_argument(
        "--gnn_residual", action=argparse.BooleanOptionalAction, default=False,
        help="Add each layer input to its output.",
    )
    parser.add_argument(
        "--num_bases", type=int, default=None,
        help="RGCN basis count; unset uses one matrix per relation.",
    )
    parser.add_argument(
        "--gnn_graph", type=str, default="train", choices=["train", "random", "none"],
        help="Training edges, random-tail edges, or no propagation edges.",
    )
    parser.add_argument(
        "--query_edge_exclusion", type=str, default="pair",
        choices=["pair", "triple", "none"],
        help="Hide pairs, exact triples (1vsAll only), or no edges during training.",
    )
    parser.add_argument(
        "--gnn_edge_dropout", type=float, default=0.0,
        help="Fraction of propagation edges dropped per training step.",
    )
    parser.add_argument(
        "--compgcn_opn", type=str, default="mult", choices=["mult", "sub"],
        help="CompGCN composition operator phi(entity, relation).",
    )
    parser.add_argument(
        "--gnn_heads", type=int, default=4, help="GATv2 attention heads.",
    )
    parser.add_argument(
        "--sage_aggregator", type=str, default="mean", choices=["mean", "max", "sum"],
        help="GraphSAGE per-relation neighbourhood aggregator.",
    )
    parser.add_argument("--input_dropout_rate", type=float, default=0.0)
    parser.add_argument("--hidden_dropout_rate", type=float, default=0.0)
    parser.add_argument(
        "--normalization", type=str, default=None,
        choices=["LayerNorm", "BatchNorm1d", None],
    )
    parser.add_argument(
        "--init_param", type=str, default=None, choices=["xavier_normal", None],
    )

    parser.add_argument(
        "--optim", type=str, default="Adam",
        choices=["Adam", "Adopt", "AdamW", "SGD", "NAdam", "Adagrad", "ASGD"],
        help="Adopt is dicee's ADOPT optimizer (needs dicee installed).",
    )
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument(
        "--label_smoothing", type=float, default=0.0,
        help="Target smoothing: y*(1-eps) + eps/num_entities. Zero disables it.",
    )
    parser.add_argument(
        "--scoring_technique", type=str, default="KvsAll", choices=["KvsAll", "1vsAll"],
        help="KvsAll: multi-hot BCE per (h, r). 1vsAll: cross-entropy per triple.",
    )
    parser.add_argument(
        "--reciprocal", action=argparse.BooleanOptionalAction, default=False,
        help="Train inverse queries and use them for head prediction.",
    )
    parser.add_argument(
        "--eval_every", type=int, default=0,
        help="Validate every N epochs; keep best MRR weights. Zero disables it.",
    )
    parser.add_argument(
        "--early_stopping_patience", type=int, default=0,
        help="Stop after N stale validations; zero disables it. Needs --eval_every.",
    )
    parser.add_argument(
        "--num_core", type=int, default=0, help="DataLoader worker count.",
    )
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument(
        "--accelerator", type=str, default="cpu", choices=["cpu", "gpu"],
        help="Training device.",
    )

    parser.add_argument(
        "--eval_tie_policy", type=str, default="average", choices=["average", "sort"],
        help="Ties: average rank or position in descending torch.sort order.",
    )
    parser.add_argument(
        "--eval_model", type=str, default="train_val_test",
        choices=["None", "train", "train_val", "train_val_test", "test", "val_test"],
    )

    parser.add_argument("--storage_path", type=str, default="Experiments")
    parser.add_argument("--full_storage_path", type=str, default=None)
    parser.add_argument(
        "--save_experiment", action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument("--test_runs", action="store_true", default=False)

    parsed_args = parser.parse_args(args_list)
    parsed_args.full_storage_path = parsed_args.full_storage_path or None
    return parsed_args
