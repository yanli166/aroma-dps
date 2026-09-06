"""
Target-ring mapping: single source of truth.

Provides the canonical API:
    smiles_model(smiles) -> str
    target_atom_indices_model(smiles, atom_on_ring) -> List[int]

atom_on_ring in CSV is 0-based (verified by P0-2 audit, 20,605 rows).
This module replaces the scattered implementations in:
- unified_models/common/graphs.py (correct, 0-based)
- 0831-end-code/common/features.py (was 1-based, fixed in Commit 2)
- unified_models/ml_models.py (was 1-based, fixed in Commit 2)
- 0901-end-code/fig4_lunci10/data/build_manifest.py (v1 find_target_ring, deprecated)
"""
from typing import List, Union
from rdkit import Chem


def smiles_model(smiles: str) -> str:
    """Return canonical SMILES for model input.

    All downstream code must use this instead of raw `smiles` to ensure
    canonicalization consistency.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol)


def target_atom_indices_model(
    smiles: str,
    atom_on_ring: Union[str, List[int]],
) -> List[int]:
    """Return 0-based target ring atom indices, validated against RDKit mol.

    Args:
        smiles: SMILES string (will be canonicalized).
        atom_on_ring: Either a list of 0-based indices or a string like "[5, 6, 7, 8, 9]".

    Returns:
        List of validated 0-based atom indices within the molecule.

    Raises:
        ValueError: If SMILES is invalid or indices are out of range.
    """
    import ast

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    n_atoms = mol.GetNumAtoms()

    # Parse atom_on_ring
    if isinstance(atom_on_ring, str):
        try:
            atom_on_ring = ast.literal_eval(atom_on_ring)
        except (ValueError, SyntaxError):
            raise ValueError(f"Cannot parse atom_on_ring: {atom_on_ring}")

    if not isinstance(atom_on_ring, (list, tuple)):
        raise ValueError(f"atom_on_ring must be list or str, got {type(atom_on_ring)}")

    # Validate 0-based indices
    indices = []
    for idx in atom_on_ring:
        if not isinstance(idx, (int, float)):
            continue
        i = int(idx)
        if 0 <= i < n_atoms:
            indices.append(i)
        else:
            raise ValueError(
                f"Atom index {i} out of range [0, {n_atoms}) for SMILES: {smiles}"
            )

    return indices


def assert_ring_validity(smiles: str, atom_on_ring: Union[str, List[int]], ring_size: int = None):
    """Runtime assertion that target ring atoms form a valid RDKit ring.

    Args:
        smiles: SMILES string.
        atom_on_ring: 0-based atom indices.
        ring_size: Expected ring size (optional).

    Raises:
        AssertionError: If the indices do not match a real RDKit ring.
    """
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles}"

    indices = set(target_atom_indices_model(smiles, atom_on_ring))

    ring_info = mol.GetRingInfo()
    atom_rings = [set(r) for r in ring_info.AtomRings()]

    matched = False
    for ring in atom_rings:
        if indices == ring:
            matched = True
            if ring_size is not None:
                assert len(ring) == ring_size, (
                    f"Ring size mismatch: expected {ring_size}, got {len(ring)}"
                )
            break

    assert matched, (
        f"Target ring atoms {indices} do not match any RDKit ring in {smiles}. "
        f"Available rings: {atom_rings}"
    )
