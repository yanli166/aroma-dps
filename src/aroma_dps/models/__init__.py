"""
Models: ring-conditioned architectures and shared backbones.

Module structure:
- backbones/          : vendored convolution layers (GNN/GIN/GAT/MPNN/GraphSAGE)
- readout.py          : FixedRingAvg / AttentionRing / GlobalMean ring readouts
- ring_conditioned.py : RingConditionedGNN + build_model (Fig.3 factorial variants)
- pyg_backbones.py    : DMPNN with ring-level readout (requires torch_geometric)

Ring conditioning and readout stay separate to preserve the Fig.3 factorial
design (Base / Membership / LearnableReadout / Joint).
"""
from aroma_dps.models.ring_conditioned import RingConditionedGNN, build_model
from aroma_dps.models.readout import (
    AttentionRingReadout,
    FixedRingAvgReadout,
    GlobalMeanReadout,
    build_readout,
)
