"""
Models: GNN architectures and ring-conditioned models.

Module structure:
- base.py: BaseGNN with shared forward logic
- ring_conditioned.py: RC-GNN (ring-conditioned GNN)
- mpnn.py: DMPNN (PyG-based)
- gin.py, gat.py: Additional backbones (re-export from unified_models)

Ring conditioning and readout are kept as separate modules to preserve
the Fig.3 factorial design (Base / Membership / LearnableReadout / Joint).
"""
