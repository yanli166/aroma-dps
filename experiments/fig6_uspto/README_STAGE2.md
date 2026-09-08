# Dearomatization Stage-2 exact validation

## Why Stage 2 is necessary

Stage 1 intentionally favors recall. On USPTO_STEREO it returned:

- P1: 33,341 reactions / 42,200 candidate rings
- P2: 3,114 reactions / 5,668 candidate rings
- P3: 93,056 reactions
- P0: 873,404 reactions

A 3.3% P1 rate is too large to assume that all P1 reactions are true dearomatizations.
Stage 2 therefore switches from global/SMARTS evidence to **exact atom-mapped ring tracking**.

The key criterion is not "the number of aromatic rings decreased". It is:

    the same target ring atoms survive in the product
    + the same ring topology is retained
    + aromatic atoms/bonds on that ring decrease
    + a local reaction-center event occurs on/around that ring

This explicitly rejects many false positives such as protecting-group cleavage,
aryl-fragment disappearance, simple aromatic substitution/coupling, and ring opening.

---

## Files

- `dearom_stage2_exact.py`
  Main Stage-2 validator.
- `summarize_manual_review.py`
  Calculates manual precision after you label a review sample.
- `requirements_stage2.txt`
  Minimal dependencies.

---

## Environment

Recommended: use a separate environment because RXNMapper pins/uses deep-learning
dependencies that may conflict with an existing PyTorch/Transformers stack.

Example:

```bash
conda create -n dearom-stage2 python=3.10 -y
conda activate dearom-stage2

pip install pandas rdkit tqdm
pip install rxnmapper
```

If RXNMapper has dependency conflicts in your existing environment, keep Stage 1 and
Stage 2 in separate environments.

---

## Recommended input

Use **P1 + P2 from Stage 1**, not all one million USPTO_STEREO reactions.

Example files:

```text
/home/ubuntu/aroma-dps-code/uspto-5k/dearom_stereo_fast_full/priority_1.csv
/home/ubuntu/aroma-dps-code/uspto-5k/dearom_stereo_fast_full/priority_2.csv
```

The script automatically detects common reaction columns:
`reactions`, `reaction`, `mapped_reactions`, `rxn_smiles`, etc.

---

## First: 1,000-reaction pilot

```bash
cd /home/ubuntu/aroma-dps-code/uspto-5k

python dearom_stage2_exact.py \
  --input \
    dearom_stereo_fast_full/priority_1.csv \
    dearom_stereo_fast_full/priority_2.csv \
  --outdir dearom_stage2_pilot \
  --batch-size 16 \
  --limit 1000 \
  --sample-per-tier 50
```

Check:

```bash
cat dearom_stage2_pilot/stage2_report.json
column -s, -t < dearom_stage2_pilot/decision_breakdown.csv | head -30
```

If RXNMapper runs out of GPU/CPU memory, reduce:

```bash
--batch-size 4
```

or

```bash
--batch-size 8
```

---

## Full P1 + P2 run

```bash
python dearom_stage2_exact.py \
  --input \
    dearom_stereo_fast_full/priority_1.csv \
    dearom_stereo_fast_full/priority_2.csv \
  --outdir dearom_stage2_full \
  --batch-size 16 \
  --sample-per-tier 100
```

Mapping results are cached incrementally to:

```text
dearom_stage2_full/mapped_reactions_cache.csv
```

If the job stops after mapping some reactions, rerunning the same command reuses this
cache instead of remapping completed reaction strings.

---

## Stage-2 output tiers

### Tier A — exact high confidence

The same mapped ring is retained and all of the following are required by default:

- 100% target-ring atoms present in product
- all target-ring atoms stay in one product component
- 100% original ring edges retained
- >= 33% aromatic-atom loss
- >= 33% aromatic-bond loss
- at least 2 aromatic atoms lost
- at least 2 aromatic ring bonds lost
- >= 50% of pre-existing ring substituent context retained
- local event score >= 3
- RXNMapper confidence >= 0.60 (unless reaction was already mapped)

This is the pool that should approach the eventual DearomCorpus-Silver-High.

### Tier B — plausible / partial

Used for cases such as:

- partial dearomatization
- fused-ring systems
- indole/naphthalene one-ring loss
- weaker but still mapped local aromaticity loss

Tier B should be manually sampled because it may contain chemically valuable edge cases.

### Tier C — manual review

Typical reasons:

- ring opening / major rearrangement
- mapped local aromaticity loss but topology is not cleanly retained
- ambiguous fused-ring event

### R — rejected

Typical reasons include:

- `TARGET_RING_ATOM_LOSS_OR_MAPPING_INCOMPLETE`
  Very common for protecting groups or aromatic fragments that simply disappear.
- `TARGET_RING_SPLIT_ACROSS_PRODUCTS`
- `NO_RING_SPECIFIC_AROMATICITY_LOSS`
  Simple aromatic substitution/coupling often ends here.
- `WEAK_LABEL_ONLY_OR_TAUTOMER_LIKE_CHANGE`
- `MAPPING_FAILED`

---

## Important chemistry logic

### 1. Why atom mapping is central

Suppose:

```text
benzyl-protected substrate -> deprotected substrate
```

The global aromatic-ring count decreases by one, but the phenyl ring does **not**
become a nonaromatic ring: the entire benzyl fragment simply leaves.

Stage 2 requires every atom of the candidate aromatic ring to be mapped into the
product. Such a reaction is therefore rejected.

### 2. Fused rings are handled ring-locally

The script enumerates fully aromatic 5–7-membered rings using `Chem.GetSymmSSSR`.
For each reactant ring it follows the mapped atom cycle itself; it does not depend on
the product having the same SSSR decomposition.

This is important for:

- indole -> indoline
- naphthalene partial dearomatization
- quinoline partial reduction
- benzofuran/benzothiophene dearomatization

### 3. Aromaticity loss alone is not enough

The script also measures:

- new sp3 atoms on target ring
- in-ring bond-order changes
- hybridization changes
- formal-charge changes
- hydrogen-count changes
- degree changes
- new/deleted exocyclic bonds
- new stereocenters on target ring

A "local event score" prevents minor aromaticity-perception changes from automatically
becoming Tier A.

---

## Output files

```text
dearom_stage2_full/
├── mapped_reactions_cache.csv
├── stage2_ring_evidence.csv
├── stage2_reactions.csv
├── tier_A_exact.csv
├── tier_B_plausible.csv
├── tier_C_review.csv
├── rejected.csv
├── decision_breakdown.csv
├── manual_review_sample.csv
└── stage2_report.json
```

### stage2_ring_evidence.csv

One row per *exact reactant aromatic ring*.

Important columns:

```text
reaction_id
ring_size
ring_formula
ring_map_numbers
shared_ring_atoms
fused_ring_edges

mapped_fraction
same_product_component
ring_edge_retention
context_retention

lost_aromatic_atoms
aromatic_atom_loss_fraction
lost_aromatic_ring_edges
aromatic_bond_loss_fraction

new_sp3_ring_atoms
ring_bond_order_changes
hybridization_changes
new_external_bonds
new_stereocenters_on_ring

local_event_score
exact_score
tier
decision_reason
```

This file is the best starting point for the future ring-level DearomCorpus.

---

## Manual precision calibration

`manual_review_sample.csv` contains balanced samples from A/B/C/R and two empty columns:

```text
human_label
human_note
```

Fill `human_label` with:

```text
TRUE
FALSE
UNCERTAIN
```

Then:

```bash
python summarize_manual_review.py \
  --input dearom_stage2_full/manual_review_sample.csv \
  --output dearom_stage2_full/manual_precision_summary.csv
```

Do **not** freeze Stage-2 thresholds before this manual audit.

Suggested target:

- Tier A precision: ideally >= 90%
- Tier B: high recall is more important than very high precision
- Tier C: used to measure what chemistry the exact retained-ring definition misses

---

## Recommended next refinement after this run

After manually checking ~100 Tier A and ~100 Tier B reactions, classify false positives
by mechanism, e.g.:

- deprotection / fragment loss
- aromatic substitution
- cross-coupling
- tautomerization
- ring opening
- rearrangement
- mapping error
- salt/charge normalization
- true dearomatization

Then add chemistry-specific exclusion rules only for failure modes that are actually
observed. Avoid over-engineering the rule set before measuring false positives.

---

## Scientific interpretation

Tier A is a *structural dearomatization label*:

    an aromatic reactant ring is retained as the same atom-mapped cycle in the product
    while local aromaticity is substantially reduced.

It does not prove:

- the reaction is catalytic,
- it is asymmetric,
- the isolated product is the mapped major stereoisomer,
- the mechanism is "dearomative" in the authors' terminology.

Those require the later corpus-curation/metadata stage.
