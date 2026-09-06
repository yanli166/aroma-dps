"""
Base GNN: shared forward logic for all backbone architectures.

The 5 GNN architectures (GNN/GIN/GAT/MPNN/GraphSAGE) share identical forward
structure (init_transform → mask → conv layers → pool/mean → hidden → output).
This base class avoids duplicating that logic. The individual model files
only need to implement the convolution layer.

NOTE: We do NOT rewrite the verified GNN forward implementations.
This base class is for new models only; existing models in unified_models/
remain as-is and are re-exported.
"""
