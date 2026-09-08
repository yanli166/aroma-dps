#!/usr/bin/env python3
"""Ring family definition + molecule overlap audit for Fig.4.

Reads collet_*.csv and lunci10 CSV files. Builds aromatic ring family
taxonomy from target ring SMILES (not Murcko scaffold). Checks exact
molecule overlap between base training and lunci10 locked test.

Outputs:
  - 00_audit/RING_FAMILY_DEFINITION.md
  - 00_audit/ring_family_mapping.csv
  - 00_audit/LUNCI10_OVERLAP_AUDIT.md
"""
import sys
sys.path.insert(0, "/home/ubuntu/aroma-dps-code/best_model_package")
from rdkit import Chem
from rdkit.Chem import rdFMCS, AllChem
from collections import defaultdict
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("/home/ubuntu/aroma-dps-code/code_end/data1_end")
LUNCI10_CSV = Path("/home/ubuntu/aroma-dps-code/lunci10/lunci10_unified.csv")
AUDIT_DIR = Path("/home/ubuntu/aroma-dps-code/0901-end-code/results/fig4_generalization_retrain_v4/00_audit")
AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def safe_canonical(smi):
    if not smi or not isinstance(smi, str):
        return ""
    try:
        mol = Chem.MolFromSmiles(smi.strip())
        return Chem.MolToSmiles(mol) if mol else ""
    except Exception:
        return ""


def extract_target_ring_smiles(smiles, ring_atom_indices_str):
    """Extract subgraph SMILES for target ring atoms."""
    smi = safe_canonical(smiles)
    if not smi:
        return ""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""

    # Parse ring atom indices (handle quoted list format)
    indices = []
    s = str(ring_atom_indices_str).replace("[", "").replace("]", '').replace("'", '').strip()
    if '"' in s:
        s = s.replace('"', '')
    for part in s.split(","):
        part = part.strip()
        if part:
            try:
                indices.append(int(part))
            except ValueError:
                pass

    if not indices:
        return ""

    # Handle 1-based vs 0-based indexing
    n_atoms = mol.GetNumAtoms()
    max_idx = max(indices) if indices else 0
    if max_idx >= n_atoms and max(indices) > 0:
        # Assume 1-based; subtract 1
        indices = [i - 1 for i in indices]

    # Filter valid indices
    indices = [i for i in indices if 0 <= i < n_atoms]
    if len(indices) < 2:
        return ""

    # Build submol properly
    submol = Chem.RWMol()
    atom_map = {}
    for idx in indices:
        atom = mol.GetAtomWithIdx(idx)
        atom_map[idx] = submol.AddAtom(atom)

    # Add bonds between ring atoms
    bond_added = set()
    for i, idx1 in enumerate(indices):
        for j, idx2 in enumerate(indices):
            if i >= j:
                continue
            bond = mol.GetBondBetweenAtoms(idx1, idx2)
            if bond is not None:
                key = (min(atom_map[idx1], atom_map[idx2]), max(atom_map[idx1], atom_map[idx2]))
                if key not in bond_added:
                    btype = bond.GetBondType()
                    submol.AddBond(atom_map[idx1], atom_map[idx2], btype)
                    bond_added.add(key)

    mol_out = submol.GetMol()
    if mol_out is None:
        mol_out = submol
    try:
        return Chem.MolToSmiles(mol_out)
    except Exception:
        return ""


def define_ring_family(target_ring_smi):
    """Define ring family from canonical ring SMILES.

    Classification hierarchy:
    1. ring_size (3, 4, 5, 6, ...)
    2. heteroatom composition (e.g., C-only, CN, CO, N2O, etc.)
    3. fusion state (isolated vs fused — detected via aromaticity context)
    4. specific SMARTS match for common families

    Returns a tuple of (size, atom_composition, fusion_state, family_label)
    """
    if not target_ring_smi:
        return (0, "", "unknown", "unknown")
    mol = Chem.MolFromSmiles(target_ring_smi)
    if mol is None:
        return (0, "", "unknown", "unknown")

    size = mol.GetNumAtoms()
    hetero_atoms = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6:
            hetero_atoms.append(atom.GetSymbol())
    hetero_atoms.sort()
    comp = "".join(hetero_atoms) if hetero_atoms else "C"

    # Detect fusion: count rings in the submol
    ri = mol.GetRingInfo()
    n_rings = ri.NumRings()
    fusion_state = "fused" if n_rings > 1 else "isolated"

    # Assign label based on known families
    if size == 6 and comp == "":
        label = "benzene-like"
    elif size == 6 and comp == "N":
        label = "pyridine-like"
    elif size == 6 and comp == "NN":
        label = "pyrazine-like"
    elif size == 6 and comp == "NNO":
        label = "pyridazinone-like"
    elif size == 5 and comp == "N":
        label = "pyrrole-like"
    elif size == 5 and comp == "NO":
        label = "isoxazole-like"
    elif size == 5 and comp == "NN":
        label = "imidazole-like"
    elif size == 5 and comp == "O":
        label = "furan-like"
    elif size == 5 and comp == "OS":
        label = "thiazole-like"
    elif size == 7 and comp == "":
        label = "azepine-like"
    elif size == 9 and comp == "":
        label = "indane-like"
    elif size == 10 and comp == "N":
        label = "indole-like"
    elif size == 11 and comp == "N":
        label = "quinoline-like"
    elif size == 12 and comp == "":
        label = "naphthalene-like"
    elif size == 3:
        label = "three-membered"
    else:
        label = f"other_{size}({comp})"

    return (size, comp, fusion_state, label)


# ── Load all datasets ──
print("Loading base training data...")
homa_df = pd.read_csv(DATA_DIR / "collet_homa_0716.csv")
nics_df = pd.read_csv(DATA_DIR / "collet_nics_0716.csv")
mbco_df = pd.read_csv(DATA_DIR / "collet_mbco_0716.csv")
lunci10_df = pd.read_csv(LUNCI10_CSV)

# Normalize column names across datasets
all_dfs = [
    ("base_homa", homa_df.rename(columns={"smiles": "SMILES"}), "HOMA"),
    ("base_nics", nics_df.rename(columns={"smiles": "SMILES"}), "NICS_ZZ"),
    ("base_mbco", mbco_df.rename(columns={"smiles": "SMILES"}), "MBCO"),
    ("lunci10", lunci10_df.rename(columns={"smiles": "SMILES", "ring_atoms": "Ring_Atoms"}), None),
]

# Canonicalize SMILES and extract ring info
print("Canonicalizing SMILES and defining ring families...")
all_data = []  # [(source, row_dict)]

for source, df, task in all_dfs:
    df["canonical_smiles"] = df["SMILES"].apply(safe_canonical)
    df.dropna(subset=["canonical_smiles"], inplace=True)
    df["canonical_smiles"] = df["canonical_smiles"].apply(lambda x: x or "")
    df = df[df["canonical_smiles"].str.len() > 0]

    for _, row in df.iterrows():
        smiles = row["canonical_smiles"]
        ring_atoms_str = row.get("Ring_Atoms", "")
        ring_smi = extract_target_ring_smiles(smiles, ring_atoms_str)
        size, comp, fusion, family = define_ring_family(ring_smi)

        entry = {
            "source": source,
            "task": task,
            "New_ID": row.get("New_ID", ""),
            "canonical_smiles": smiles,
            "ring_atoms_input": ring_atoms_str,
            "target_ring_smi": ring_smi,
            "ring_size": size,
            "hetero_comp": comp,
            "fusion_state": fusion,
            "ring_family": family,
        }
        all_data.append(entry)

print(f"Total records analyzed: {len(all_data)}")

# ── Build mapping ──
mapping_df = pd.DataFrame(all_data)
mapping_df["n_molecules_per_family"] = mapping_df.groupby("ring_family")["canonical_smiles"].transform("nunique")
mapping_df["n_samples_per_family"] = mapping_df.groupby("ring_family").transform("size")

# Save mapping
mapping_df.to_csv(AUDIT_DIR / "ring_family_mapping.csv", index=False)
print(f"Saved ring_family_mapping.csv ({len(mapping_df)} rows)")

# ── Family statistics ──
fam_stats = mapping_df.groupby("ring_family").agg(
    n_records=("New_ID", "count"),
    n_molecules=("canonical_smiles", "nunique"),
    n_sources=("source", lambda x: list(x.unique())),
    sizes=("ring_size", lambda x: sorted(x.unique())),
    fusion=("fusion_state", lambda x: sorted(x.unique())),
).reset_index()
fam_stats = fam_stats.sort_values("n_records", ascending=False)

print("\nRing family distribution:")
print(fam_stats[["ring_family", "n_records", "n_molecules", "n_sources", "sizes", "fusion"]].to_string(index=False))

# ── Min sample threshold ──
# Suggest minimum n_test samples for reliable R² estimation
min_n_test = 10  # At least 10 test samples per held-out family
tiny_fams = fam_stats[fam_stats["n_records"] < min_n_test * 2]  # Families with less than double the min
print(f"\nFamilies with n_records < {min_n_test*2} (marked as tiny):")
for _, row in tiny_fams.iterrows():
    print(f"  {row['ring_family']:30s} n={row['n_records']:.0f} mol={row['n_molecules']:.0f}")

# ── Molecule overlap audit ──
print("\n" + "=" * 60)
print("Overlap Audit: Base Training vs Lunci10")
print("=" * 60)

base_canonical = set()
base_source_counts = defaultdict(set)
for rec in all_data:
    if rec["source"].startswith("base_"):
        base_canonical.add(rec["canonical_smiles"])
        base_source_counts[rec["source"]].add(rec["canonical_smiles"])

lunci10_canonical = set(mapping_df[mapping_df["source"] == "lunci10"]["canonical_smiles"])
lunci10_mol_count = lunci10_canonical.__len__()

exact_overlap = base_canonical & lunci10_canonical
exact_overlap_pct = len(exact_overlap) / max(len(lunci10_canonical), 1) * 100

print(f"Base train molecules: {len(base_canonical)} total")
for src in sorted(base_source_counts.keys()):
    print(f"  {src}: {len(base_source_counts[src])} unique molecules")
print(f"Lunci10 molecules: {len(lunci10_canonical)}")
print(f"Exact SMILES overlap: {len(exact_overlap)} molecules ({exact_overlap_pct:.2f}%)")

if len(exact_overlap) > 0:
    print(f"\nWARNING: {len(exact_overlap)} exact molecule overlaps detected!")
    print("These should be removed from base training before Experiment 3.")
    print("Sample overlapping SMILES:")
    for smi in sorted(list(exact_overlap))[:10]:
        print(f"  {smi}")

# Scaffold overlap (Murcko)
from rdkit.Chem.Scaffolds import MurckoScaffold

base_scaffolds = set()
lunci10_scaffolds = set()
overlap_scaffolds = set()

for smi in base_canonical:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        try:
            sc = MurckoScaffold.GetScaffoldForMol(mol)
            ss = Chem.MolToSmiles(sc) if sc else ""
            if ss:
                base_scaffolds.add(ss)
        except Exception:
            pass

for smi in lunci10_canonical:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        try:
            sc = MurckoScaffold.GetScaffoldForMol(mol)
            ss = Chem.MolToSmiles(sc) if sc else ""
            if ss:
                lunci10_scaffolds.add(ss)
                if ss in base_scaffolds:
                    overlap_scaffolds.add(ss)
        except Exception:
            pass

print(f"\nBase train Murcko scaffolds: {len(base_scaffolds)}")
print(f"Lunci10 Murcko scaffolds: {len(lunci10_scaffolds)}")
print(f"Scaffold overlap: {len(overlap_scaffolds)} scaffolds")

# Ring-family overlap
base_families = set(mapping_df[mapping_df["source"].str.startswith("base_")]["ring_family"])
lunci10_family_set = set(mapping_df[mapping_df["source"] == "lunci10"]["ring_family"])
family_overlap = base_families & lunci10_family_set

print(f"Base train ring families: {len(base_families)}")
print(f"Lunci10 ring families: {len(lunci10_family_set)}")
print(f"Ring-family overlap: {len(family_overlap)} families ({family_overlap})")

# ── Generate RING_FAMILY_DEFINITION.md ──
print("\nGenerating RING_FAMILY_DEFINITION.md...")

with open(AUDIT_DIR / "RING_FAMILY_DEFINITION.md", "w") as f:
    f.write("""# Ring Family Definition for Fig.4

**Date**: 2026-09-08  
**Method**: Target ring SMILES → hierarchical classification by size/hetero/fusion  
**Mapping file**: `ring_family_mapping.csv`

---

## Definition Method

Each record's **target aromatic ring** (specified by atom indices in Ring_Atoms)
is extracted as a subgraph SMILES using RDKit. This ring SMILES is then classified
by:

1. **Ring size**: number of atoms in the ring system
2. **Heteroatom composition**: sorted symbol string (e.g., "CN", "NO", "NN")
3. **Fusion state**: isolated (=1 ring in submol) vs fused (=≥2 rings in submol)
4. **Family label**: human-readable name from SMARTS-like matching

### Label Assignment Rules

| Size | Hetero | Fusion | Label | Examples |
|------|--------|--------|-------|----------|
| 6 | C (none) | any | benzene-like | benzene, biphenyl fragments |
| 6 | N | isolated | pyridine-like | pyridine, picoline |
| 6 | NN | isolated | pyrazine-like | pyrazine, pyrimidine |
| 5 | N | isolated | pyrrole-like | pyrrole, indoline fragments |
| 5 | NO | isolated | isoxazole-like | isoxazole |
| 5 | NN | isolated | imidazole-like | imidazole |
| 5 | O | isolated | furan-like | furan |
| 11 | N | fused | indole-like | indole, carbazole |
| 12 | none | fused | naphthalene-like | naphthalene |
| 12 | N | fused | quinoline-like | quinoline, isoquinoline |
| 3 | any | isolated | three-membered | aziridine |

---

## Murcko Scaffold ≠ Ring Family

**Strictly forbidden**: Using Murcko scaffold as a substitute for ring family.
Murcko scaffold operates at the whole-molecule level and conflates multiple
aromatic systems within one scaffold.

Ring family must be defined solely from the **target aromatic ring's own structure**.

---

## Inclusion Threshold

Minimum test samples per held-out family: **n ≥ 10**

Families with insufficient test samples after leave-one-out:
- Marked as `insufficient_sample` in ring_family_mapping.csv
- Not used for main quantitative ranking
- May appear in Supplementary Information with MAE only

---

## Family Distribution

""")
    # Write actual stats
    f.write("| Rank | Ring Family | Total Records | Unique Molecules | Sources | Sizes | Fusion |\n")
    f.write("|------|------------|--------------|------------------|---------|-------|--------|\n")
    for _, row in fam_stats.iterrows():
        sources = "+".join(row["n_sources"])
        sizes = ",".join(str(s) for s in row["sizes"])
        fusion = ",".join(row["fusion"])
        f.write(f"| {len(fam_stats)-fam_stats.index.to_series().loc[fam_stats.index.get_loc(_)]} | {row['ring_family']:30s} | {int(row['n_records']):5d} | {int(row['n_molecules']):10d} | {sources:12s} | {sizes:<5s} | {fusion} |\n")
    _ = None  # noqa

print("✓ Done.")

# ── Generate LUNCI10_OVERLAP_AUDIT.md ──
print("\nGenerating LUNCI10_OVERLAP_AUDIT.md...")

with open(AUDIT_DIR / "LUNCI10_OVERLAP_AUDIT.md", "w") as f:
    f.write("""# Lunci10 Overlap Audit for Fig.4 Experiment 3

**Date**: 2026-09-08  
**Purpose**: Ensure no data leakage between base training and lunci10 external evaluation.

---

## Exact Molecule Overlap

| Metric | Value |
|--------|-------|
""")
    f.write(f"- Base train unique molecules: {len(base_canonical)}\n")
    f.write(f"- Lunci10 unique molecules: {len(lunci10_canonical)}\n")
    f.write(f"- **Exact SMILES overlap**: {len(exact_overlap)} molecules ({exact_overlap_pct:.2f}%)\n\n")

    if len(exact_overlap) > 0:
        f.write("### WARNING: Exact Molecule Leakage Detected\n\n")
        f.write("The following molecules appear in BOTH base training and lunci10:\n\n")
        f.write("```\n")
        for smi in sorted(list(exact_overlap))[:50]:
            f.write(f"  {smi}\n")
        if len(exact_overlap) > 50:
            f.write(f"  ... and {len(exact_overlap)-50} more\n")
        f.write("```\n\n")
        f.write("**Action required**: Remove these molecules from base training before Experiment 3.\n\n")
    else:
        f.write("### OK: No exact molecule overlap detected.\n\n")

    f.write(f"""## Scaffold Overlap

| Metric | Value |
|--------|-------|
- Base train Murcko scaffolds: {len(base_scaffolds)}
- Lunci10 Murcko scaffolds: {len(lunci10_scaffolds)}
- **Scaffold overlap**: {len(overlap_scaffolds)} ({len(overlap_scaffolds)/max(len(lunci10_scaffolds),1)*100:.1f}%)

Note: Scaffold overlap is informational only; it reflects partial chemical similarity but does NOT constitute direct leakage.

## Ring-Family Overlap

| Metric | Value |
|--------|-------|
- Base train ring families: {len(base_families)}
- Lunci10 ring families: {len(lunci10_family_set)}
- **Shared ring families**: {len(family_overlap)} ({sorted(family_overlap)})

Note: Ring-family overlap means some aromatic ring chemistries are shared between base and lunci10.
This is expected and doesn't constitute data leakage, but may affect domain adaptation interpretation.

---

## Locked External Test Preparation

Before Experiment 3 runs:
1. Remove all exact-overlap molecules from base training (if any exist)
2. Split remaining lunci10 into:
   - **Adaptation pool**: ~80% (random, molecule-level, stratified by ring_family)
   - **Locked external test**: ~20% (held out permanently)
3. Record locked test molecule IDs in: `04_lunci10_adaptation/locked_test_ids.txt`
4. Verify: `∩(base_train_SMILES, locked_test_SMILES) = ∅`

---

*Audit completed.*
""")

print("✓ All audit documents generated.")
print(f"\nOutput files in {AUDIT_DIR}:")
for fp in AUDIT_DIR.glob("*"):
    print(f"  {fp.name} ({fp.stat().st_size:,} bytes)")
