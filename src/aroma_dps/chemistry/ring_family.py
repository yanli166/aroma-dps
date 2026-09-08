"""
Ring family classification for target-ring level analysis.

Vendored from fig4_ring_family_audit.py (aroma-dps-code working directory),
which produced 00_audit/ring_family_mapping.csv and RING_FAMILY_DEFINITION.md.

Two deliberate deviations from that source:
1. Target-ring indices go through target_atom_indices_model() (strict 0-based,
   raises on out-of-range) instead of the script's
   "if max_idx >= n_atoms: subtract 1" auto-conversion heuristic.
2. The source label table matched `comp == ""` for all-carbon rings while
   defining comp as "C" when no heteroatom exists, so benzene-like /
   azepine-like / indane-like / naphthalene-like were unreachable and every
   all-carbon ring collapsed into other_<size>(C). Reinstating those branches is
   not sufficient on its own: a label keyed only on (size, composition) would
   call cyclohexane "benzene-like". Family labels are therefore restricted to
   aromatic rings, and non-aromatic rings are labelled nonaromatic_<size>(<comp>).
   The historical audit CSV keeps the old labels.

Not ported: code_end/generalization_test/code/ring_utils.py::identify_ring_type
(a whole-molecule SMARTS-priority labeler, different concept). Its SMARTS table
contains suspect entries (pyridazine, phthalazine) and must be reviewed before
it is reused.
"""
from typing import List, Tuple, Union

from rdkit import Chem

from aroma_dps.chemistry.ring_mapping import target_atom_indices_model


def safe_canonical(smiles: str) -> str:
    """Canonical SMILES, or empty string when RDKit cannot parse the input."""
    if not smiles or not isinstance(smiles, str):
        return ""
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        return Chem.MolToSmiles(mol) if mol else ""
    except Exception:
        return ""


def extract_target_ring_smiles(
    smiles: str,
    atom_on_ring: Union[str, List[int]],
) -> str:
    """Subgraph SMILES of the target ring atoms only (internal bonds preserved)."""
    canonical = safe_canonical(smiles)
    if not canonical:
        return ""
    mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        return ""

    try:
        indices = target_atom_indices_model(canonical, atom_on_ring)
    except ValueError:
        return ""
    if len(indices) < 2:
        return ""

    submol = Chem.RWMol()
    atom_map = {}
    for idx in indices:
        atom_map[idx] = submol.AddAtom(mol.GetAtomWithIdx(idx))

    bond_added = set()
    for i, idx1 in enumerate(indices):
        for j, idx2 in enumerate(indices):
            if i >= j:
                continue
            bond = mol.GetBondBetweenAtoms(idx1, idx2)
            if bond is None:
                continue
            key = (min(atom_map[idx1], atom_map[idx2]), max(atom_map[idx1], atom_map[idx2]))
            if key not in bond_added:
                submol.AddBond(atom_map[idx1], atom_map[idx2], bond.GetBondType())
                bond_added.add(key)

    try:
        return Chem.MolToSmiles(submol.GetMol())
    except Exception:
        return ""


# (ring size, heteroatom composition) -> family label; "C" means all-carbon
_ALL_CARBON = "C"

_KNOWN_FAMILIES = {
    (6, _ALL_CARBON): "benzene-like",
    (6, "N"): "pyridine-like",
    (6, "NN"): "pyrazine-like",
    (6, "NNO"): "pyridazinone-like",
    (5, "N"): "pyrrole-like",
    (5, "NN"): "imidazole-like",
    (5, "NO"): "isoxazole-like",
    (5, "O"): "furan-like",
    (5, "OS"): "thiazole-like",
    (7, _ALL_CARBON): "azepine-like",
    (9, _ALL_CARBON): "indane-like",
    (10, "N"): "indole-like",
    (11, "N"): "quinoline-like",
    (12, _ALL_CARBON): "naphthalene-like",
}


def define_ring_family(target_ring_smi: str) -> Tuple[int, str, str, str]:
    """Classify a target ring as (size, composition, fusion_state, label).

    Composition lists heteroatoms only; all-carbon rings report "C".
    Aromatic family labels are assigned only to aromatic rings, so a saturated
    ring of the same size/composition cannot inherit an aromatic family name.

    Note: fusion_state is perceived on the extracted subgraph, which by
    construction holds a single ring, so it is always "isolated" and carries no
    information — inherited from the source script (its audit CSV has the same
    limitation). True fusion state must come from the parent molecule.
    """
    if not target_ring_smi:
        return (0, "", "unknown", "unknown")
    mol = Chem.MolFromSmiles(target_ring_smi)
    if mol is None:
        return (0, "", "unknown", "unknown")

    size = mol.GetNumAtoms()
    hetero_atoms = sorted(a.GetSymbol() for a in mol.GetAtoms() if a.GetAtomicNum() != 6)
    comp = "".join(hetero_atoms) if hetero_atoms else _ALL_CARBON

    fusion_state = "fused" if mol.GetRingInfo().NumRings() > 1 else "isolated"

    if not any(a.GetIsAromatic() for a in mol.GetAtoms()):
        return (size, comp, fusion_state, f"nonaromatic_{size}({comp})")

    label = _KNOWN_FAMILIES.get((size, comp))
    if label is None:
        label = "three-membered" if size == 3 else f"other_{size}({comp})"

    return (size, comp, fusion_state, label)


def ring_family_of(smiles: str, atom_on_ring: Union[str, List[int]]) -> Tuple[int, str, str, str]:
    """Target ring SMILES extraction + family classification in one call."""
    return define_ring_family(extract_target_ring_smiles(smiles, atom_on_ring))
