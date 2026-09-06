"""
Ring-conditioned GNN: RC-GNN / RC-MPNN.

Uses unified:
    binary target-ring membership (0/1)
    + learnable projection / embedding (nn.Embedding(2, hidden_dim))

NOT ring_flag=10. The final model uses binary membership.

However, Fig.3 2×2 ablation MUST preserve the original factorial definition:
    Base: no ring info
    Membership: ring_flag=10 (fixed encoding)
    LearnableReadout: attention readout
    Joint: ring_flag=10 + attention readout

Do NOT unify the ablation variants with the final model.
"""
# Re-export from 0831-end-code (canonical implementation)
try:
    from models.ring_conditioned_gnn import RingConditionedGNN, build_model
except ImportError:
    pass

# Re-export readout (kept separate for factorial design)
try:
    from models.ring_readout import (
        FixedRingAvgReadout,
        AttentionRingReadout,
        GlobalMeanReadout,
        build_readout,
    )
except ImportError:
    pass
