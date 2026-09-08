# Fix canonical-SMILES target-ring indices before aromaticity prediction

## The bug

The original `ring_property_ml_samples.csv` contains:

```text
smiles
smiles_mapped
target_atom_indices
target_ring_map_numbers
```

`target_atom_indices` was generated from the RDKit molecule parsed from
`smiles_mapped`.

However, `smiles` is a separately generated **canonical SMILES**.

Canonicalization can reorder atoms.

Therefore this is unsafe:

```python
predictor.predict(
    row["smiles"],
    parse(row["target_atom_indices"])
)
```

The indices may point at entirely different atoms.

Your first inspected example demonstrates exactly this.

---

# Correct mapping

Suppose the mapped molecule has original atom indices:

```text
old idx:
0 1 2 3 4 5 6 7 8 9
```

and target map numbers identify old indices:

```text
4 5 6 7 8
```

RDKit canonical SMILES generation may emit atoms in the order:

```text
[7,6,5,4,3,2,1,0,9,8]
```

This means:

```text
new canonical index 0 <- old index 7
new canonical index 1 <- old index 6
new canonical index 2 <- old index 5
...
```

So the target ring becomes:

```text
old: [4,5,6,7,8]
new: [0,1,2,3,9]
```

The script obtains this permutation from RDKit's:

```python
mol.GetProp("_smilesAtomOutputOrder")
```

after `Chem.MolToSmiles(..., canonical=True)`.

This is deterministic and does NOT require ambiguous substructure matching.

---

# Step 1 — Fix all 8,886 sample indices

Run:

```bash
cd /home/ubuntu/aroma-dps-code/uspto-5k

python fix_ring_indices.py \
  --input dearom_ring_pairs_A_tierA/ring_property_ml_samples.csv \
  --output dearom_ring_pairs_A_tierA/ring_property_ml_samples_fixed.csv
```

Outputs:

```text
ring_property_ml_samples_fixed.csv
ring_property_ml_samples_fixed_index_errors.csv
ring_property_ml_samples_fixed_index_report.json
```

Important new columns:

```text
smiles_model
target_atom_indices_original
target_atom_indices_model
target_ring_mask_model

index_changed
existing_smiles_matches_model
target_cycle_valid_model
index_status
```

Only rows with:

```text
index_status = PASS_REINDEXED
```

or:

```text
index_status = PASS_UNCHANGED
```

should be sent to the model.

---

# Step 2 — Inspect the report before prediction

```bash
cat dearom_ring_pairs_A_tierA/ring_property_ml_samples_fixed_index_report.json
```

For the strict Tier-A corpus we expect approximately:

```text
n_samples = 8886
n_pass ~= 8886
target_cycle_valid ~= 8886
```

`n_index_changed` may be large. That is not an error; it quantifies how dangerous
the original index assumption was.

If any sample fails:

```text
FAIL_EXISTING_SMILES_STRUCTURE_MISMATCH
FAIL_TARGET_NOT_SIMPLE_CYCLE
ERROR
```

do not predict it until inspected.

---

# Step 3 — Batch prediction with corrected indices

```bash
source /home/ubuntu/apps/anaconda3/etc/profile.d/conda.sh
conda activate torch_env

cd /home/ubuntu/aroma-dps-code/uspto-5k

python predict_ring_properties_fixed.py \
  --input dearom_ring_pairs_A_tierA/ring_property_ml_samples_fixed.csv \
  --model-package /home/ubuntu/aroma-dps-code/best_model_package \
  --output dearom_ring_pairs_A_tierA/ring_property_predictions.csv \
  --pairs-output dearom_ring_pairs_A_tierA/ring_pair_aromaticity_predictions.csv
```

The second output gives one row per `pair_id` and automatically computes:

```text
Delta_HOMA
Delta_nMCBO
Delta_NICS_star
```

with the convention:

```text
Delta > 0  ==> aromaticity loss
```

---

# Do NOT use substructure matching as the primary fix

The proposed approach:

```python
plain.GetSubstructMatches(ring_mol)
```

can work for many samples, but it creates ambiguity in:

- symmetric benzene rings
- equivalent fused rings
- repeated heteroarene motifs
- molecules containing multiple identical rings

Atom-map numbers already tell us exactly which atoms are the target.

The only missing information is the canonicalization permutation.

Therefore the clean solution is:

```text
atom maps
+
RDKit canonical output permutation
```

not substructure guessing.

---

# Even better long-term design

For all future datasets, generate the model input and model indices **together**.

Starting from mapped mol:

```python
mapped_mol
```

1. identify target atom indices by atom maps;
2. remove atom-map properties in a copy;
3. canonicalize;
4. retrieve `_smilesAtomOutputOrder`;
5. remap target indices;
6. save:

```text
smiles_model
target_atom_indices_model
```

at data-generation time.

Never independently generate canonical SMILES and ring indices again.

---

# Alternative architecture-level fix

If you are willing to modify `best_model_package/predict.py`, an even stronger API is:

```python
predict_mol(mol, atom_on_ring)
```

where `mol` is an RDKit Mol object.

Then no SMILES serialization/reparse step occurs and atom indices cannot drift.

The current toolkit avoids modifying the validated model package and is therefore
safer for immediate use.
