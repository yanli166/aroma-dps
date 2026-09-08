"""GNN backbone convolution layers.

Vendored verbatim from unified_models/{gnn,gin,gat,mpnn,graphsage}/model.py.
These are the convolution layers shared by every ring-conditioned variant.
"""
from aroma_dps.models.backbones.gnn import ConvolutionLayer
from aroma_dps.models.backbones.gin import GINLayer
from aroma_dps.models.backbones.gat import MultiHeadGAT
from aroma_dps.models.backbones.mpnn import MPNNLayer
from aroma_dps.models.backbones.graphsage import GraphSAGELayer

CONV_LAYERS = {
    'GNN': ConvolutionLayer,
    'GIN': GINLayer,
    'GAT': MultiHeadGAT,
    'MPNN': MPNNLayer,
    'GraphSAGE': GraphSAGELayer,
}

__all__ = ['CONV_LAYERS', 'ConvolutionLayer', 'GINLayer', 'MultiHeadGAT',
           'MPNNLayer', 'GraphSAGELayer']
