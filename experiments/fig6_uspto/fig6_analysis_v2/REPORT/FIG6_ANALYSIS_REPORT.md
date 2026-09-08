# Fig.6 v2 — Analysis Report

**USPTO-scale ML-enabled chemical discovery** (Fig.6 in the manuscript series Fig.1–Fig.6).

Scope: 4,377 complete target-ring pairs × 4,306 Tier-A reactions (corpus frozen at
`dearom_ring_pairs_A_tierA/`), 5,592 re-predicted spectator rings (internal control).
All panels use only the frozen RC-GNN model; no retraining.

---

## 0. Data audit summary

See `DATA_AUDIT/Fig6_DATA_AUDIT.md` for the full 10-point integrity check.

| Item | Value |
|---|---|
| Tier-A target ring pairs (raw) | 4,443 |
| Complete R/P target pairs (used) | **4,377** |
| Spectator rings input | 5,722 |
| Spectator rings with both R/P PASS | **5,592** (97.7%) |
| Tier-A reactions | 4,306 (unique) |
| Reactions with both target + spectator complete | 3,392 |
| NaN/Inf in any descriptor | 0 |
| pair_id duplicates | 0 |
| Reactions with >1 target ring | rare (cluster bootstrap required) |

Excluded records: see `DATA_AUDIT/excluded_records.csv` (66 Tier-A pairs lacking
both R/P PASS).

---

## 1. Panel-by-panel summary

### Panel 6a — Pipeline
- **Scientific question:** what pipeline goes from 1 M USPTO reactions to 4,377
  reaction-scale aromaticity vectors?
- **Data used:** all-up; no quantitative analysis.
- **N:** 1,002,915 → 4,306 → 4,443 → 4,377 (main) + 5,722 → 5,592 (control).
- **Method:** schematic with anchored counts.
- **Main numerical result:** see `PANEL_A_PIPELINE/anchors.json`.
- **Recommended:** **MAIN** (Fig.6a).

### Panel 6b — Target vs Spectator internal control
- **Scientific question:** Is aromaticity loss localized to the chemically
  transformed target ring?
- **Data used:** target pairs (4,377) + re-predicted spectator pairs (5,592
  complete spectator R/P pairs, derived from 5,722 candidate spectator rings) at
  the reaction level (3,392 reactions with both present). A structural
  aromaticity-retention audit (4,000-row sample) confirmed that the spectator
  ring retains its RDKit aromatic flag on both reactant and product sides in
  100% of rows → the control set is composed of structurally unaffected,
  valid internal-control rings (RDKit aromatic-flag retention only; not a
  quantum-chemical validation; see `spectator_control_audit.json`).
- **N_reactions = 3,392**; per-reaction gap
  `d_i = median(L_target | reaction_i) − median(L_spectator | reaction_i)`,
  aggregated as `d̂ = median_i(d_i)`.
- **Method:** paired-dots/slope plot, reaction-clustered bootstrap (5,000 iters;
  resamples the 3,392 reaction units and recomputes the median of d_i each
  iteration, CI around d̂), Cliff's delta, Wilcoxon signed-rank.
- **Main numerical result:**

| Descriptor | Median target | Median spectator | d̂ = median d_i | 95% boot CI (reaction-clustered) | Cliff's δ | reactions with d_i > 0 |
|---|---|---|---|---|---|---|
| L_HOMA | 2.197 | −0.020 | **+2.241** | [+2.192, +2.270] | 0.991 | 3,368/3,392 (99.3%) |
| L_nMCBO | 0.216 | −0.006 | **+0.225** | [+0.224, +0.230] | 0.966 | 3,332/3,392 (98.2%) |
| L_NICS | 24.69 | +0.519 | **+23.20** | [+22.87, +23.44] | 0.968 | 3,355/3,392 (99.0%) |

The point estimate d̂ lies inside the bootstrap CI for every descriptor. Cliff's δ
values (0.966–0.991, within [−1, +1]) indicate an essentially total separation of
the per-reaction target and spectator gap distributions, and the target-ring loss
exceeds the spectator-ring median in 98–99% of all 3,392 reactions containing a
valid internal spectator control (not 100%: a small number of reactions carry
spectator rings with tiny nonzero predicted losses).

- **Possible interpretation:** the model and the chemistry agree that aromaticity
  loss is overwhelmingly localized to the chemically transformed target ring;
  spectator rings show negligible aromaticity change (their predicted
  aromaticity is nearly unchanged from reactant to product).
- **Possible confounder:** reactions with only one target ring still have a
  single-value "median" so the paired comparison is meaningful; reactions with
  multiple spectator rings use median aggregation (robust to imbalance).
- **Recommended:** **MAIN** (Fig.6b).

### Panel 6c — Global aromaticity-loss landscape
- **Scientific question:** Are these losses directionally robust across geometric,
  electronic, and magnetic descriptors?
- **Data used:** 4,377 target pairs.
- **Method:** raincloud (half-violin + jittered dots + box) + L=0 reference line;
  concordance stacked bar.
- **Main numerical result:**

| Descriptor | median | IQR | fraction Δ > 0 |
|---|---|---|---|
| L_HOMA | 2.284 | 1.654 | **99.6%** |
| L_nMCBO | 0.251 | 0.135 | **98.3%** |
| L_NICS | 25.53 | 9.01 | **99.4%** |

Concordance (N=4,377): ALL_LOSS 97.4%, DISCORDANT 2.6%, ALL_GAIN 0.0%.

- **Interpretation:** the three diagnostics agree on **direction** across the corpus,
  but **disagree on magnitude** (PCA PC1=70.5%, PC2=19.4%, PC3=10.1%).
- **Recommended:** **MAIN** (Fig.6c).

### Panel 6d — Ring-family fingerprint
- **Scientific question:** Is the dearomatization signal homogeneous across ring
  families, or does the chemistry decompose by ring identity?
- **Data used:** 4,377 target pairs × 13 ring families (all families with n≥30,
  including the large "other N-heteroarene" bucket) + z_robust heatmap
  (median-centred, MAD-normalised, zero = global median).
- **Method:** D1 raw violin/box/jitter per family (raw units preserved). D2
  z_robust heatmap (rows=families, cols=L_HOMA / L_nMCBO / L_NICS).
- **Main numerical result:** see `PANEL_D_RING_FAMILY/ring_family_z_robust.csv`.
- **Headline:** ring families are **NOT** all the same. nMCBO response is much
  stronger for indole / benzofuran / benzothiophene than for benzene / phenol;
  HOMA response is more uniform across the mainstream families; NICS is
  consistently large (always >0 by ~20+ ppm) but shows strong family dependence,
  concentrated in the small low-n families (other S-heteroarene, other
  carbocycle, other O-heteroarene). Note that "other N-heteroarene" (the
  second-largest family, ~675 pairs) is included in the main heatmap.
- **Caveat:** family sample sizes are uneven (n = 31–706 across the 13 families).
  Family sizes are uneven; the heatmap therefore reports robust standardized
  medians (z_robust) for comparability, while uncertainty and raw distributions
  are provided separately (bootstrap CI in Panel 6f; raw distributions in SI).
- **Recommended:** **MAIN** (Fig.6d = z_robust heatmap; raw distributions go SI).

### Panel 6e — Structural decoupling
- **Scientific question:** Does global molecular change predict target-ring
  aromaticity loss?
- **Data used:** 4,377 pairs. StructuralChange = 1 − Tanimoto(R, P) using Morgan
  r=2 / 2048-bit on whole molecules.
- **Method:** hexbin + LOWESS (statsmodels), Pearson + Spearman + bootstrap 95% CI.
- **Main numerical result:**

| pair | Pearson r | Spearman ρ | 95% CI |
|---|---|---|---|
| StructuralChange × L_HOMA | −0.022 | −0.043 | [−0.053, +0.009] |
| StructuralChange × L_nMCBO | −0.005 | +0.034 | [−0.035, +0.024] |
| StructuralChange × L_NICS | +0.012 | −0.012 | [−0.015, +0.038] |

- **Interpretation:** essentially zero correlation — whole-molecule structural
  perturbation is a poor proxy for local aromaticity loss.
- **Fingerprint preprocessing (Morgan r=2 / 2048-bit, RDKit):** the R/P pair used
  for Tanimoto is built from `reactant_component_smiles` / `product_component_smiles`
  in `ring_pairs_ml.csv`, produced by `strip_maps()` in `build_ml_ring_pairs.py`:
  - **Atom mapping:** removed before fingerprinting (`SetAtomMapNum(0)`, then
    canonical SMILES) — mapping numbers cannot inflate Tanimoto.
  - **Reagents / counterions:** each side uses only the single component that
    contains the target ring (selected as the component with maximum coverage of
    the target-ring atom-map numbers among that side's components); separate
    reagent components are excluded from the comparison.
  - **Multi-fragment components:** if the selected component itself contains
    several fragments (e.g., co-formulated salt), all fragments are retained in
    the Morgan fingerprint (whole component).
  - **Charge:** preserved (no neutralisation).
  - **Stereochemistry:** preserved (`isomericSmiles=True`), so R/S/E/Z are
    retained in the fingerprint.
  - **R/P consistency:** identical preprocessing on both sides.
- **Recommended:** **MAIN** (Fig.6e).

### Panel 6f — Ring-family explanatory strength across descriptors
- **Scientific question:** How strongly is aromaticity loss organised by ring
  identity, and does that differ across geometric, electronic, and magnetic
  descriptors? (No reaction-mechanism classification is involved.)
- **Data used:** 4,377 target pairs; for each descriptor
  `L_descriptor ~ RingFamily` (one-hot encoding, linear regression).
- **Method:** 5-fold GroupKFold cross-validation grouped by reaction_id;
  cross-validated R² with reaction-clustered bootstrap 95% CI; nonparametric
  effect size Kruskal–Wallis ε².
- **Main numerical result:** see
  `PANEL_F_DISCORDANT/fig6f_ring_family_cv_r2_stats.csv`:

| Descriptor | CV R² (ring family) | 95% boot CI | Kruskal–Wallis ε² |
|---|---|---|---|
| L_HOMA (geometric) | 0.364 | [+0.341, +0.381] | 0.345 |
| L_nMCBO (delocalization) | **0.603** | [+0.574, +0.635] | 0.646 |
| L_NICS (magnetic) | 0.543 | [+0.521, +0.563] | 0.430 |

Ring identity alone explains ~1/3 of HOMA variance, over half of NICS variance,
and nearly 2/3 of nMCBO variance. The chemical determinants of aromaticity loss
are **descriptor-dependent**: electron-delocalization loss is strongly organised
by ring identity, whereas the geometric response (HOMA) retains substantially
larger within-family heterogeneity; the magnetic response (NICS) is intermediate.
(No claim of causality — this quantifies family-level association strength only.)
- **Discordant cases:** the 2.6% descriptor-discordant pairs are the rare
  exceptions to this family pattern and are presented in the **SI**
  (`Fig6fA_discordant_cases.*`, `discordant_enrichment.csv`). They are dominated
  by nMCBO_NEG (73/114, 64%); no category is enriched after BH-FDR at α=0.05, so
  we note only that **nMCBO accounts for most directional discordance** without
  assigning a mechanistic reason.
- **Recommended:** **MAIN** (Fig.6f = three point estimates + bootstrap CI).

---

## 2. Enhanced panels (only included if user wants Fig.6 ENHANCED)

### Reaction-family classification (conservative topology)
- **N=4,377 target pairs.**
- **Reduction-like (n_added_heavy_bonds=0, BO↓):** 3,251 (74.3%)
- **Single-addition-like (n_added_heavy_bonds=1):** 912 (20.8%)
- **Multi-addition-like (n_added_heavy_bonds≥2):** 141 (3.2%)
- **Complex/other:** 73 (1.7%)
- **Audited:** `REACTION_FAMILY_OPTIONAL/reaction_family_contact_sheets/`
  (50 reactions per family, manual review recommended).
- **Caveat:** this is **not** a mechanism claim — it only classifies target-ring
  reaction-center topology. Naming reductions as "hydrogenation" or additions as
  "alkylation" would require additional evidence not present in USPTO SMILES.

### Ring × Reaction heatmaps
- 3 descriptors × (Ring × Reaction) cell medians, only cells with n≥15 shown.
- See `VARIANCE_DECOMPOSITION_OPTIONAL/Fig6_RingReaction_heatmaps.{png,svg,pdf}`
  and the z_robust multidimensional heatmap
  `Fig6_RingReaction_zrobust_heatmap.{png,svg,pdf}`.

### Controlled slices
- Slice A (fixed reaction family = reduction_like, vary ring family): see
  `Fig6_slice_A_fixed_reaction.{png,svg,pdf}` — confirms ring-family effect
  survives within the dominant reaction family.
- Slice B (fixed ring family = benzene-type, vary reaction family): see
  `Fig6_slice_B_fixed_ring.{png,svg,pdf}` — confirms reaction-family effect
  within a single ring family is much smaller.

### Variance decomposition (OLS, descriptive)
| Descriptor | Ring unique | Reaction unique | Shared | Residual |
|---|---|---|---|---|
| L_HOMA | 24.5% | 12.7% | 0.0% | **63.5%** |
| L_nMCBO | **57.3%** | 0.1% | 0.3% | 42.4% |
| L_NICS | 20.0% | 0.5% | 0.0% | **79.7%** |

- **Interpretation:** L_nMCBO is mostly a *ring-family* property (57%); L_HOMA
  and L_NICS have large residuals — driven by finer-grained local geometry
  and magnetic response that the ring-family category does not capture.
- **Caveat:** this is **descriptive** (variance partitioning in an OLS model),
  not causal. Reaction-family categories are a coarse proxy.

---

## 3. Exploratory (SI only)

- **Initial aromaticity vs loss:** see `SI/S11_initial_aromaticity_vs_loss.*`
  + `SI/initial_aromaticity_vs_loss.csv`.
- **PCA of L vector:** see `SI/S12_pca_loss_vector.*` + loadings/scores in SI.
  PC1=70.5%, PC2=19.4%, PC3=10.1%. **Single scalar DPS is not supported.**
- **Descriptor pairwise relationships:** see `PANEL_E`/`SI` outputs from earlier
  Fig.5 work (Pearson r 0.42–0.65).
- **Descriptor-discordant chemistry (SI):** 114 discordant pairs (2.6% of the
  corpus) as the rare exceptions to the Fig.6f ring-family pattern. Discordance
  is dominated by nMCBO_NEG (73/114, 64%); NICS_NEG 23/114; HOMA_NEG 16/114;
  nMCBO+NICS_NEG 2. Fisher-exact enrichment against background shows **no
  category significant after BH-FDR at α=0.05**; therefore the report states
  only that **nMCBO accounts for most directional discordance**, without the
  causal claim that it flips because it is "more family-sensitive". Top-6
  discordant reaction images: `Fig6fA_discordant_cases.*`.

---

## 4. Recommended version for the manuscript main text

Both `Fig6_CORE_candidate.{png,svg,pdf}` and `Fig6_ENHANCED_candidate.{png,svg,pdf}`
have been generated. Each panel's individual recommendation:

| Panel | Recommended position |
|---|---|
| 6a pipeline | **MAIN** |
| 6b target vs spectator | **MAIN** |
| 6c global loss | **MAIN** |
| 6d ring-family z_robust heatmap | **MAIN** (D2); D1 raw goes SI |
| 6e structural decoupling | **MAIN** |
| 6f ring-family explanatory strength | **MAIN** |
| Discordant cases (top-6) | **SI** |
| Ring × Reaction heatmap | **SI** (or MAIN if reviewer asks for determinant analysis) |
| Controlled slices | **SI** |
| Variance decomposition | **SI** (or MAIN if reviewer pushes for variance partition) |
| Reaction-family contact sheets | **SI** (audit only) |
| Initial aromaticity / PCA | **SI** |

**Final recommendation: Fig.6 = CORE version.**
Rationale:
1. **Scientific novelty:** CORE answers the new Fig.6 question directly
   ("what becomes visible at scale?"). ENHANCED adds variance-partition
   storytelling that is *descriptive*, not causal, and easier to attack.
2. **Chemical interpretability:** CORE panels each have a single
   defensible claim (localized, robust, descriptor-dependently family-organised,
   structurally decoupled).
3. **Statistical robustness:** CORE stats all use reaction-clustered
   bootstrap; Cliff's δ for Panel B (0.966–0.991, within [−1, +1]) indicates a
   large, near-total separation and the direction is unambiguous.
4. **Reviewer defensibility:** CORE avoids the trap of naming reductions
   or additions as mechanisms (which ENHANCED could invite).

ENHANCED should be **submitted as SI** — it provides extra depth for reviewers
who want determinant analysis, but its variance partition is descriptive and
the reaction-family labels are conservative topology calls, not mechanisms.

---

## 5. Cautions and not-claimed

- Not claimed: "DFT cannot calculate these systems."
- Claimed: "ML enables systematic, reaction-scale aromaticity analysis that
  would be computationally burdensome by routine quantum-chemical workflows."
- Not claimed: a single scalar Dearomatization Score (PC1 < 85%).
- Not claimed: mechanism names for any reaction family.
- Not claimed: causality from ring-family to loss magnitude (variance partition is
  descriptive).
- Acknowledged: spectator rings were re-predicted in this analysis using the
  frozen model (no retraining), see
  `PANEL_B_TARGET_SPECTATOR/spectator_prediction_summary.json`.
- Acknowledged: 66 Tier-A pairs excluded for incomplete R/P prediction (one
  side failed UFF/charge handling); see `DATA_AUDIT/excluded_records.csv`.
- Acknowledged: in Panel 6b the target-ring loss exceeds the spectator-ring
  median in 98–99% of the 3,392 reactions (not 100%); the claim is therefore
  "nearly universal localisation", not "in every reaction".

---

## 6. Output inventory

```
DATA_AUDIT/                     Fig6_DATA_AUDIT.md, data_audit_summary.json, excluded_records.csv
PANEL_A_PIPELINE/               Fig6a_pipeline.{svg,pdf,png}, anchors.json
PANEL_B_TARGET_SPECTATOR/       Fig6b_*.{svg,pdf,png}, spectator_*.csv, fig6b_target_spectator_stats.csv
PANEL_C_GLOBAL_LOSS/            Fig6c_global_loss.*, Fig6c_concordance_bar.*, fig6c_global_loss_stats.csv
PANEL_D_RING_FAMILY/            ring_family_assignment.csv, ring_family_z_robust.csv,
                                Fig6d1_raw_family_distributions.*, Fig6d2_family_fingerprint_heatmap.*,
                                contact_sheets/family_*.png
PANEL_E_STRUCTURAL_CHANGE/      Fig6e_structural_decoupling.*, structural_change_vs_loss.csv,
                                fig6e_stats.csv
PANEL_F_DISCORDANT/             Fig6f_ring_family_cv_r2.*, fig6f_ring_family_cv_r2_stats.csv|json,
                                Fig6fA_discordant_cases.* (SI), discordant_cases_annotated.csv,
                                discordant_enrichment.csv, discordant_top_cases.csv,
                                discordance_type_counts.csv
REACTION_FAMILY_OPTIONAL/       reaction_family_assignment.csv, reaction_family_counts.csv,
                                reaction_family_audit_report.md,
                                reaction_family_contact_sheets/audit_*.png
VARIANCE_DECOMPOSITION_OPTIONAL/ring_x_reaction_master.csv, variance_decomposition.csv,
                                Fig6_RingReaction_heatmaps.*, Fig6_RingReaction_zrobust_heatmap.*,
                                Fig6_slice_A_fixed_reaction.*, Fig6_slice_B_fixed_ring.*,
                                Fig6fB_variance_decomposition.*
SI/                             S11_initial_aromaticity_vs_loss.*, S12_pca_loss_vector.*,
                                initial_aromaticity_vs_loss.csv, descriptor_loss_pca_*.csv,
                                pca_loss_explained_variance.json
Fig6_CORE_candidate.{png,svg,pdf}
Fig6_ENHANCED_candidate.{png,svg,pdf}
FIG6_ANALYSIS_REPORT.md          (this file)
```
