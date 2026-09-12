from .rgcn import RGCN  # noqa
from .rgat import GATv2  # noqa
from .graphsage import GraphSAGE  # noqa

MODEL_MAP = {
    "RGCN": RGCN,
    "GATv2": GATv2,
    "GraphSAGE": GraphSAGE,
}
