Dearomatization screening quick start
=====================================

1) Pure RDKit first pass (works on your current unmapped USPTO-50K):

python dearom_screen.py \
  --input USPTO_50K.tsv \
  --outdir dearom_50k_firstpass \
  --mcs-timeout 1 \
  --draw 200

2) Recommended publication-grade second pass: atom map first.

pip install rxnmapper
python map_uspto_rxnmapper.py \
  --input USPTO_50K.tsv \
  --output USPTO_50K_mapped.csv \
  --batch-size 64

python dearom_screen.py \
  --input USPTO_50K_mapped.csv \
  --reaction-col mapped_reactions \
  --outdir dearom_50k_mapped \
  --mcs-timeout 1 \
  --draw 300

Outputs:
- screened_reactions.csv: one row/reaction; reaction_priority 0/1/2/3
- candidate_rings.csv: one row/target-ring candidate with all ring-level evidence
- best_candidate_per_reaction.csv: best candidate ring per reaction
- priority_1.csv: high-confidence likely dearomatization
- priority_2.csv: probable but weaker/ambiguous
- priority_3.csv: manual-review pool
- review_images/: highlighted reactant target ring / mapped product atoms
- screening_report.json: counts and configuration

Suggested use:
- Manually inspect all P1 from an initial 50K run.
- Estimate precision on a random sample of P1/P2/P3.
- Tune thresholds only after blind/manual validation, not by looking only at desired examples.
