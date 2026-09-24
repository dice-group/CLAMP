from .kge import KGE, LTE  # noqa
from .compgcn import CompGCN  # noqa
from .rgcn import RGCN  # noqa
from .rgat import GATv2  # noqa
from .graphsage import GraphSAGE  # noqa
from .clifford import CliffordGATv2, CliffordGraphSAGE, CliffordRGCN  # noqa

MODEL_MAP = {
    # Encoder-free baseline; --decoder selects the scoring function.
    "KGE": KGE,
    # KGE + shared linear map/BatchNorm/dropout on entities (Zhang et al. 2022).
    "LTE": LTE,
    "RGCN": RGCN,
    "GATv2": GATv2,
    "GraphSAGE": GraphSAGE,
    "CompGCN": CompGCN,
    "CliffordRGCN": CliffordRGCN,
    "CliffordGATv2": CliffordGATv2,
    "CliffordGraphSAGE": CliffordGraphSAGE,
}
