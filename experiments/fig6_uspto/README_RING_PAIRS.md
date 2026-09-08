# Stage-2 A → ML target-ring pairs

## Purpose

Convert Stage-2 exact dearomatization results into explicit `(molecule, target ring)` samples.
This is the recommended representation for ring-property learning:

```text
(molecular graph, target-ring mask) -> HOMA / NICS(1)zz / nMCBO
```

## Inputs

```text
dearom_stage2_full/stage2_ring_evidence.csv
dearom_stage2_full/stage2_reactions.csv
```

## Run

```bash
cd /home/ubuntu/aroma-dps-code/uspto-5k

python build_ml_ring_pairs.py \
  --ring-evidence dearom_stage2_full/stage2_ring_evidence.csv \
  --reactions dearom_stage2_full/stage2_reactions.csv \
  --outdir dearom_ring_pairs_A \
  --tier A \
  --local-radius 2
```

## Outputs

```text
dearom_ring_pairs_A/
├── ring_pairs_ml.csv
├── ring_pairs_ml.jsonl
├── ring_property_ml_samples.csv
├── ring_pair_errors.csv
└── ring_pair_report.json
```

### `ring_pairs_ml.csv`
One row per reaction target ring, containing both reactant and product:

- exact target atom-map numbers
- reactant/product target atom indices
- reactant/product binary ring masks
- target-ring bonds
- ring cyclic order
- full molecular component containing the ring
- ring-only fragment for inspection
- radius-2 local environment
- Stage-2 validation evidence

### `ring_property_ml_samples.csv`
One reaction-ring pair becomes two independent ML samples:

```text
pair_xxx__R    reactant
pair_xxx__P    product
```

with:

```text
sample_id
pair_id
reaction_id
side
smiles
smiles_mapped
target_atom_indices
target_mask
target_ring_map_numbers
```

Use this file directly when building PyG/DGL molecular graphs.

## Important modeling choice

Do **not** use only the ring fragment as the primary ML input. Aromaticity depends on
substituents, fusion, heteroatoms and the surrounding conjugated environment. Use the
whole molecular component plus a target-ring mask.

Recommended model call:

```python
A = model(molecular_graph, target_ring_mask)
```

Then for one reaction:

```python
A_R = model(reactant_graph, reactant_ring_mask)
A_P = model(product_graph, product_ring_mask)
Delta_A = A_R - A_P
```

## Recommended future labels

Append after DFT/model prediction:

```text
HOMA_R, HOMA_P, delta_HOMA
NICSzz_R, NICSzz_P, delta_NICSzz
nMCBO_R, nMCBO_P, delta_nMCBO
```

This gives the direct dataset for reaction-level Fig.5/Fig.6.

## Aromaticity-feature leakage

For the main quantitative aromaticity model, strongly consider an ablation that removes:

```text
atom.GetIsAromatic()
bond.GetIsAromatic()
```

from the input features. The target-ring mask should stay: it says *which ring* to
predict, not *how aromatic* that ring is.
