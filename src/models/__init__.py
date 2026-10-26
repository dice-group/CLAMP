from .kge import KGE, LTE
from .gnn import RGCN, GATv2, GraphSAGE, CompGCN
from .clifford import CliffordRGCN, CliffordGATv2, CliffordGraphSAGE

MODEL_MAP = {
    "KGE": KGE,
    "LTE": LTE,
    "RGCN": RGCN,
    "GATv2": GATv2,
    "GraphSAGE": GraphSAGE,
    "CompGCN": CompGCN,
    "CliffordRGCN": CliffordRGCN,
    "CliffordGATv2": CliffordGATv2,
    "CliffordGraphSAGE": CliffordGraphSAGE,
}


def model_class(args: dict):
    """The class for args['model']."""
    name = args["model"]
    if name not in MODEL_MAP:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODEL_MAP)}")
    return MODEL_MAP[name]


def build_model(args: dict):
    return model_class(args)(args=args)
