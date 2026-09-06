"""
Chemistry: SMILES parsing, target-ring mapping, ring family classification.

API:
    smiles_model(smiles) -> canonical SMILES
    target_atom_indices_model(smiles, atom_on_ring) -> List[int]

All training, OOD, interpretability, Fig.5, Fig.6 inference must go through
the same ring_mapping module.
"""
