import argparse

from src.filtering import TIE_POLICIES
from src.models.gnn.base import RESIDUAL_MODES


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
        help="Encoder; see --clifford_signature for the Clifford models' algebra.",
    )
    parser.add_argument(
        "--clifford_signature", type=int, nargs=2, default=None, metavar=("P", "Q"),
        help="Clifford encoders: algebra Cl(p, q) with p generators squaring to +1 and q to -1. "
        "2**(p+q) must divide embedding_dim; the embedding holds embedding_dim / 2**(p+q) "
        "independent multivectors. Unset: one Cl(log2 dim, 0) multivector (power-of-two dim).",
    )
    parser.add_argument(
        "--decoder", type=str, default="DistMult",
        choices=["DistMult", "TransE", "ComplEx", "RotatE", "Clifford"],
        help="ComplEx and RotatE require an even embedding dimension. Clifford scores "
        "<h * r, t> with the full product in the --clifford_signature algebra.",
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
        "--gnn_activation",
        choices=["relu", "leaky_relu", "elu", "none", "mv_gate", "mv_grade_gate"],
        default="relu",
        help="Activation between layers (RGCN and Clifford encoders; CliffordGATv2 uses ELU). "
        "'none' gives a linear encoder control. mv_gate and mv_grade_gate are Clifford-only "
        "multivector gates: sigmoid of an invariant scales a whole multivector or a whole grade.",
    )
    parser.add_argument(
        "--gnn_residual", choices=list(RESIDUAL_MODES), default="none",
        help="Residual connection in RGCN and CliffordRGCN. 'post' computes f(F(x) + x), so "
        "the activation clips the identity term too; 'pre' computes x + f(F(x)), leaving the "
        "identity path unclipped. Runs before 2026-10-16 recorded a boolean, where true = post.",
    )
    parser.add_argument(
        "--init_from", type=str, default=None,
        help="model.pt whose entity and relation tables initialize this model (optimizer starts fresh).",
    )
    parser.add_argument(
        "--freeze_embeddings", action=argparse.BooleanOptionalAction, default=False,
        help="Keep the --init_from entity and relation tables fixed; train only the remaining parameters.",
    )
    parser.add_argument(
        "--num_bases", type=int, default=None,
        help="RGCN basis count; unset uses one matrix per relation.",
    )
    parser.add_argument(
        "--clifford_num_bases", type=int, default=None,
        help="CliffordRGCN: relation multivectors as combinations of this many shared "
        "multivectors (RGCN basis decomposition); unset uses one multivector per relation.",
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
        "--compgcn_opn", type=str, default="mult",
        choices=["mult", "sub", "cmult", "rotate"],
        help="CompGCN composition operator phi(entity, relation); cmult and rotate "
        "use the ComplEx and RotatE relation operators.",
    )
    parser.add_argument(
        "--gnn_heads", type=int, default=4, help="GATv2 attention heads.",
    )
    parser.add_argument(
        "--sage_final_norm", type=str, default="l2", choices=["l2", "none", "l2_scale"],
        help="GraphSAGE last-layer output: unit L2 norm, unnormalised, or unit L2 "
        "norm times a learnable scalar.",
    )
    parser.add_argument("--input_dropout_rate", type=float, default=0.0)
    parser.add_argument(
        "--query_dropout_rate", type=float, default=0.0,
        help="Dropout on the bilinear query q(h, r) (DistMult h * r, ComplEx and Clifford products) "
        "before scoring all tails. dicee's DistMult applies it at the input dropout rate.",
    )
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
        "--eval_every", type=int, default=0,
        help="Validate every N epochs; keep best MRR weights. Zero disables it.",
    )
    parser.add_argument(
        "--fit_diagnostics", action="store_true", default=False,
        help="At each validation, log fit diagnostics (src/diagnostics.py) in validation_history.json.",
    )
    parser.add_argument(
        "--early_stopping_patience", type=int, default=0,
        help="Stop after N stale validations; zero disables it. Needs --eval_every.",
    )
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument(
        "--accelerator", type=str, default="cpu", choices=["cpu", "gpu"],
        help="Training device.",
    )

    parser.add_argument(
        "--eval_batch_size", type=int, default=None,
        help="Evaluation batch size; defaults to the training batch size.",
    )
    parser.add_argument(
        "--eval_tie_policy", type=str, default="sort", choices=list(TIE_POLICIES),
        help="How exact score ties are ranked (dicee FilteredRanker policies).",
    )
    parser.add_argument(
        "--eval_model", type=str, default="train_val_test",
        choices=["None", "train", "train_val", "train_val_test", "test", "val_test"],
    )

    parser.add_argument(
        "--full_storage_path", type=str, default=None,
        help="Run directory; unset derives Experiments/<dataset>/<model>_<dim>.",
    )

    parsed_args = parser.parse_args(args_list)
    parsed_args.full_storage_path = parsed_args.full_storage_path or None
    return parsed_args
