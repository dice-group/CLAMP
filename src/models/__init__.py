from .kge import KGE, LTE
from .compgcn import CompGCN
from .rgcn import RGCN
from .rgat import GATv2
from .graphsage import GraphSAGE
from .clifford import CliffordGATv2, CliffordGraphSAGE, CliffordRGCN

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
