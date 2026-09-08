"""
Reaction prediction: USPTO reaction-scale application for Fig.6.

NOT IMPLEMENTED — no library code here yet. The authoritative implementation
is still the script pipeline under experiments/fig6_uspto/ (vendored from
aroma-dps-code/uspto-5k):

    dearom_screen_fast.py / dearom_screen.py  Stage-1 screening (P1/P2/P3)
    dearom_stage2_exact.py                    RXNMapper atom-mapped Tier A/B/C
    build_ml_ring_pairs.py                    (molecule, target ring) ML samples
    fix_ring_indices.py                       canonical-SMILES index repair
    fig6_v2_*.py                              analysis and figures

Refactoring targets for this module (pure functions to lift out of the scripts):
reaction parsing, ring inventory per side, atom-mapped target-ring tracking,
Tier-A filtering, target vs spectator ring assignment.

Fig.6 is prediction-only: the run must also emit ensemble uncertainty and a
failure manifest (see experiments/fig6/config.yaml).
"""
