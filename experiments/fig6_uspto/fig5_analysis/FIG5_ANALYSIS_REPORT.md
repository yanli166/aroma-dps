# FIG5_ANALYSIS_REPORT

Reaction-level aromaticity change across 4,377 complete reactant/product target-ring pairs
from the USPTO dearomatization corpus. All figures and tables in `fig5_analysis/`.

**Data fixed at:** `dearom_ring_pairs_A_tierA/ring_pair_aromaticity_predictions_complete.csv`
(4,377 complete R/P pairs; 46 incomplete pairs excluded because one side lacked a valid model
prediction — see SI S7).

---

## 1. Per-descriptor aromaticity-loss distribution (Fig.5b/5c, Table 1)

| descriptor | N | median R | median P | median Δ (loss) | frac Δ>0 |
|---|---|---|---|---|---|
| HOMA | 4377 | 0.779 | −1.570 | **2.284** | **99.6%** |
| nMCBO | 4377 | 0.583 | 0.318 | **0.251** | **98.3%** |
| NICS(1)zz | 4377 | −23.63 | 2.40 | **ΔNICS\*=25.53** | **99.4%** |

All three paired distributions are strongly displaced (rank-biserial r ≈ 0.999, P ≈ 0 by
Wilcoxon signed-rank; effect magnitude is the key message, not the p-value).

Within the present corpus, predicted aromaticity loss occurred in 99.6%, 98.3% and 99.4% of
reactions for HOMA, nMCBO and NICS-derived measures, respectively.

## 2. Multidimensional directional concordance (Fig.5d)

- **ALL_LOSS** (all three Δ > 0): **97.4%**
- DISCORDANT (≥1 descriptor ≤ 0): **2.6%** (114 pairs)
- ALL_GAIN_OR_NONLOSS (all three ≤ 0): **0%**

The quantitative predictions are highly consistent with the structurally defined
dearomatization labels.

## 3. Descriptor coupling (Fig.5d, Pearson / Spearman / bootstrap 95% CI)

| pair | Pearson r | Spearman ρ | Pearson 95% CI |
|---|---|---|---|
| ΔHOMA × ΔnMCBO | 0.595 | 0.657 | [0.570, 0.618] |
| ΔHOMA × ΔNICS\* | 0.421 | 0.477 | [0.403, 0.439] |
| ΔnMCBO × ΔNICS\* | 0.649 | 0.617 | [0.625, 0.671] |

ΔHOMA and ΔnMCBO are moderately correlated; ΔNICS\* is the most weakly coupled of the three
(especially vs ΔHOMA). The three dimensions share direction but their amplitudes are **partially
decoupled** — geometric (HOMA) and electron-delocalization (nMCBO) responses are more coupled to
each other than to the magnetic (NICS) response. We therefore do not compress the response into a
single scalar without the PCA check below.

## 4. PCA of standardized Δ descriptors (Fig.5e)

- **PC1 = 70.5%**, PC2 = **19.4%**, PC3 = 10.1%
- Loadings: PC1 ≈ (0.54, 0.62, 0.57) — a genuine consensus "overall loss" axis
- PC2 still carries ~19% variance and separates ΔHOMA (+0.76) from ΔNICS\* (−0.65)

**Verdict:** PC1 > 85–90% is **not** reached. A single scalar dearomatization score is **not**
justified by the data. The reaction-scale response should be represented as a
**multidimensional aromaticity-change vector** (ΔHOMA, ΔnMCBO, ΔNICS\*), with PC2 capturing a
HOMA-vs-NICS decoupling component.

## 5. Discordant cases (114 pairs; Fig.5d/5f, SI S5/S6)

- Mostly **nMCBO_NEGATIVE_ONLY (73)**, then NICS_star_NEGATIVE_ONLY (23),
  HOMA_NEGATIVE_ONLY (16), nMCBO+NICS negative (2).
- Discordance is therefore dominated by nMCBO (the weakest-signal descriptor for already-low
  aromaticity substrates), not by a systematic HOMA/NICS conflict.
- **Chemical-space clustering check:** UMAP 30-NN mean distance for discordant (0.0475) ≈
  all pairs (0.0517), ratio 0.92. Discordant cases are **not strongly clustered** in
  target-ring chemical space — they are distributed, not a single OOD pocket.
- `top_50_descriptor_discordant_cases.csv` (ranked by max−min of standardized Δ) is the
  prioritized list for manual / DFT / OOD inspection.

## 6. Target-ring chemical space (Fig.5f, SI S9/S10)

- Target-centered fingerprint: full reactant molecule, Morgan r=2 / 2048 bits, with atom
  invariants anchoring the target ring and its radius-2 environment (all 4,377 pairs covered).
- PCA→50D (46% variance), UMAP n_neighbors=30, min_dist=0.15, metric=jaccard, seed fixed.
- Colorings by ΔHOMA / ΔnMCBO / ΔNICS\* (robust 2.5–97.5 percentile ranges) show a broadly
  continuous spread; no obvious isolated low-Δ cluster, consistent with the distributed
  discordant findings.
- UMAP is visualization only; no quantitative distance claims are made from it.

## 7. Ring-family breakdown (SI S8, `ring_family_delta_summary.csv`)

ALL_LOSS fraction is **≥99%** for all common families (benzene, pyridine, indole,
quinoline/isoquinoline, phenol-type, furan, benzofuran, etc.). The only families below 95% are
the small/miscellaneous "other" buckets:
- other N-heteroarene: 91.9%
- other O-heteroarene: 92.7%
- other S-heteroarene: 85.9%
- other carbocycle: 90.8%

All named single-ring/fused families with n≥20 are effectively 100% ALL_LOSS. The 97.4% global
concordance therefore **holds within every major ring family**, and the residual discordance
concentrates in miscellaneous heteroaromatic/carbocycle classes (likely low-baseline-aromaticity
or model-OOD substrates).

## 8. Initial aromaticity vs loss (`initial_aromaticity_analysis.csv`)

- HOMA: R vs ΔHOMA r = 0.311 (weak positive)
- nMCBO: R vs ΔnMCBO r = 0.854 (strong positive)
- NICS: −R vs ΔNICS\* r = −0.962 (strong, near-mechanical)

Because Δ = A_R − A_P contains A_R, these raw correlations are partly mathematical. Residual
analysis (A_P ~ A_R OLS) shows the OLS residual correlates weakly with structural change
(|r| ≤ 0.11). Conclusion: initial aromaticity carries genuine but descriptor-dependent predictive
content; for HOMA the independent effect is modest, while for NICS the loss is largely
proportional to the initial (anti)aromatic character.

## 9. Structural change vs target-ring loss (`reactant_product_structural_change.csv`)

Whole-molecule Morgan r=2 Tanimoto-based StructuralChange = 1 − Tanimoto(R,P):

| target | Pearson | Spearman |
|---|---|---|
| StructuralChange × ΔHOMA | −0.022 | −0.043 |
| StructuralChange × ΔnMCBO | −0.005 | 0.034 |
| StructuralChange × ΔNICS\* | 0.012 | −0.012 |

**No meaningful linear relationship.** Overall molecular structural change is **not equivalent to**
target-ring aromaticity loss. This directly supports a target-ring-aware representation (the
model takes the whole molecule + target-ring mask).

## 10. Answers to the 12 required questions

1. **Loss fractions:** HOMA 99.6%, nMCBO 98.3%, NICS 99.4% (Δ>0).
2. **97.4% ALL_LOSS across ring families?** Yes — all named families ≥99%; only miscellaneous
   heteroaromatic/carbocycle buckets fall to 86–93%.
3. **Pearson/Spearman:** ΔHOMA×ΔnMCBO 0.60/0.66; ΔHOMA×ΔNICS\* 0.42/0.48; ΔnMCBO×ΔNICS\* 0.65/0.62.
4. **PCA:** PC1 70.5%, PC2 19.4%, PC3 10.1%.
5. **Single scalar DPS?** **No.** PC1 < 85–90%; use a multidimensional vector.
6. **Discordant cause:** dominated by nMCBO_NEGATIVE_ONLY (73/114).
7. **Discordant clustering:** no strong clustering (UMAP 30-NN ratio ≈ 0.92); distributed.
8. **Initial aromaticity vs loss:** descriptor-dependent; strong mechanical coupling for NICS,
   modest for HOMA, strong for nMCBO.
9. **Structural change vs loss:** no correlation; supports target-ring-aware modeling.
10. **Ready for JACS main text:** Fig.5a workflow, Fig.5b R-vs-P distributions, Fig.5c Δ
    distributions, Fig.5d concordance + coupling, Fig.5e PCA, Fig.5f chemical space; the
    "multidimensional response, no single scalar" conclusion is data-supported.
11. **Hypothesis-level (not yet proven):** mechanism-specific Δ patterns; DFT ground-truth of
    the 50 top discordant cases; OOD attribution of "other" family discordance — all require
    further validation.
12. **Highest-priority DFT targets:** the 50 rows in
    `top_50_descriptor_discordant_cases.csv` (esp. HOMA_NEGATIVE_ONLY and nMCBO_NEGATIVE_ONLY in
    "other S-heteroarene" / "other N-heteroarene"), plus low-baseline-aromaticity substrates
    where nMCBO reads loss even when HOMA/NICS do not.

## 11. Output inventory

```
01_master/            reaction_aromaticity_fig5_master.csv
02_global_change/     global_delta_summary.*, paired_aromaticity_statistics.csv, Fig5b/c
03_descriptor_coupling/ descriptor_pairwise_correlations.csv, pearson/spearman_matrix.csv, Fig5d
04_pca/               descriptor_delta_pca_scores.csv, _loadings.csv, pca_explained_variance.json, Fig5e
05_discordant/        discordance_type_counts.csv, discordant_cases_annotated.csv,
                      top_50_descriptor_discordant_cases.csv
06_chemical_space/    target_ring_chemical_space.csv, discordant_clustering_check.json, Fig5f
07_ring_family/       ring_family_annotations.csv, _counts.csv, _delta_summary.csv
08_statistics/        reactant_product_structural_change.csv, initial_aromaticity_analysis.csv,
                      incomplete_pairs_excluded.csv
09_fig5_main/         Fig5a–5f (pdf/svg/png 600dpi)
10_fig5_si/           S1–S10
logs/
```

## 12. Cautions (explicitly not done)

- No single Dearomatization Score constructed (PC2 too large).
- No mechanism claims from USPTO SMILES.
- Discordant cases not deleted; UMAP distances not used as chemical distances.
- Small p-values not used as evidence of large effects (rank-biserial / medians reported instead).
- Model predictions not asserted to equal DFT ground truth; only consistency with structural
  labels is claimed.
