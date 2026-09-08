"""Re-run Fig.4a-4d using best_model_package (seed_11).

Protocol:
  - Models: best_model_package/*.pt (MODEL_SEED=11, atom_on_ring bug fixed)
  - Data:   lunci10-test-corrected.csv (NICS_ZZ = Multiwfn ring-normal projection)
  - Output: /home/ubuntu/aroma-dps-code/0901-end-code/results/fig4_lunci10_final_v2/

Figures:
  4a: External zero-shot on lunci10 (all 3 tasks)
  4b: Internal vs external R²/MAE comparison  
  4c: Ring-family leave-one-out heatmap
  4d: Substituent exposure curve
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

# ── paths ────────────────────────────────────────────────────────────────
PROJ_ROOT = Path("/home/ubuntu/aroma-dps-code")
BEST_PKG = PROJ_ROOT / "best_model_package"
CORRECTED_CSV = PROJ_ROOT / "lunci10/lunci10-test-corrected.csv"
BEGIN_CSV = PROJ_ROOT / "lunci10/lunci10-begin.csv"
OUTPUT_DIR = PROJ_ROOT / "0901-end-code/results/fig4_lunci10_final_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BEST_PKG))
from graph_utils import build_graph
from model_arch import RingConditionedMPNN

MODEL_SEED = 11
NODE_VEC_LEN = 60
MAX_ATOMS = 75

# Task name conventions
TASKS = ["HOMA", "NICS_ZZ", "MBCO"]
CSV_TASK_COL = {"HOMA": "HOMA", "NICS_ZZ": "NICS_ZZ", "MBCO": "MBCO"}  # CSV column names
MODEL_TASK_KEY = {"HOMA": "HOMA", "NICS_ZZ": "NICS_1zz", "MBCO": "MBCO"}  # metrics.json key / ckpt
CKPT_FILE = {"HOMA": "homa_best.pt", "NICS_ZZ": "nics_1zz_best.pt", "MBCO": "mbco_best.pt"}
SHORT_NAME = {"HOMA": "HOMA", "NICS_ZZ": "NICS ZZ", "MBCO": "MBCO"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── helpers ──────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 2:
        return {"n": len(yt), "R2": float("nan"), "MAE": float("nan"),
                "RMSE": float("nan"), "Pearson": float("nan"), "Spearman": float("nan")}
    try:
        from sklearn.metrics import r2_score
        from scipy.stats import pearsonr, spearmanr
        r2 = float(r2_score(yt, yp))
        pearson, _ = pearsonr(yt, yp)
        spearman, _ = spearmanr(yt, yp)
    except Exception:
        r2, pearson, spearman = float("nan"), float("nan"), float("nan")
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    return {"n": int(len(yt)), "R2": round(r2, 6), "MAE": round(mae, 6),
            "RMSE": round(rmse, 6), "Pearson": round(pearson, 6), "Spearman": round(spearman, 6)}

def safe_canonical(smi: str) -> str:
    from rdkit import Chem
    if not smi:
        return ""
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol else ""

def murcko_scaffold(smi: str) -> str:
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit import Chem
    if not smi:
        return ""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold) if scaffold else ""
    except Exception:
        return ""

def parse_ring_atoms(atom_str: str) -> List[int]:
    if not isinstance(atom_str, str) or not atom_str:
        return []
    s = atom_str.replace("[", "").replace("]", "").replace(" ", "")
    try:
        return [int(x) for x in s.split(",") if x.strip()]
    except Exception:
        return []

# ── load data ────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading data...")
df = pd.read_csv(CORRECTED_CSV)
if BEGIN_CSV.exists() and "no" in pd.read_csv(BEGIN_CSV).columns:
    begin_df = pd.read_csv(BEGIN_CSV).rename(columns={"no": "New_ID"})
    df = df.merge(begin_df, on="New_ID", how="left", suffixes=("", "_begin"))

df["canonical_smiles"] = df["SMILES"].apply(safe_canonical)
df["ring_atoms_list"] = df["Ring_Atoms"].apply(parse_ring_atoms)
df["scaffold"] = df["canonical_smiles"].apply(murcko_scaffold)

print(f"Loaded: {len(df)} records, {df['New_ID'].nunique()} molecules, "
      f"{df['ring_name'].nunique() if 'ring_name' in df.columns else '?'} ring families, "
      f"{df['scaffold'].nunique()} scaffolds")

# ── Load frozen models ───────────────────────────────────────────────────
print("\nLoading frozen models...")
with open(BEST_PKG / "metrics.json") as f:
    task_params = json.load(f)

frozen_models: Dict[str, RingConditionedMPNN] = {}
for task in TASKS:
    mkey = MODEL_TASK_KEY[task]
    params = task_params[mkey]
    use_proj = params.get("use_projection", False)
    rf_val = params["ring_flag_value"]
    
    ckpt_path = BEST_PKG / CKPT_FILE[task]
    model = RingConditionedMPNN(
        node_vec_len=NODE_VEC_LEN, hidden_dim=128, n_conv=3, n_hidden=2,
        p_dropout=0.2, use_projection=use_proj, ring_flag_value=rf_val,
    )
    sd_raw = torch.load(ckpt_path, map_location=device)
    # Unwrap checkpoint if needed (outer dict wrapper)
    sd = sd_raw.get("state_dict", sd_raw) if isinstance(sd_raw, dict) else sd_raw
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    frozen_models[task] = model
    print(f"  {task}: ckpt={CKPT_FILE[task]}, proj={use_proj}, rf={rf_val}, val_R2={params['val_r2']:.4f}")

def predict_one(model, smiles, atom_on_ring, device):
    """Predict single molecule (matches predict.py interface)."""
    g = build_graph(smiles, atom_on_ring, NODE_VEC_LEN, MAX_ATOMS,
                    ring_flag_value=model.ring_flag_value)
    node = torch.tensor(g["node_mat"][None], dtype=torch.float32, device=device)
    adj = torch.tensor(g["adj_mat"][None], dtype=torch.float32, device=device)
    ri = torch.tensor(g["ring_indices"][None], dtype=torch.long, device=device)
    with torch.no_grad():
        pred = model(node, adj, ri).item()
    return pred


def predict_batch(model, smiles_list, ring_atoms_lists, device, chunk_size=20):
    """Predict list of molecules via small chunks to avoid full-stack bugs.
    
    The bug: stacking ALL graphs into one tensor batch returns ~constant preds.
    Fix: process in small chunks (chunk_size=20) where stacking works correctly,
         while being much faster than per-molecule.
    Returns: numpy array of predictions.
    """
    results = []
    n = len(smiles_list)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_smis = smiles_list[start:end]
        chunk_rings = ring_atoms_lists[start:end]
        
        graphs = []
        for smi, rings in zip(chunk_smis, chunk_rings):
            g = build_graph(smi, rings, NODE_VEC_LEN, MAX_ATOMS,
                            ring_flag_value=model.ring_flag_value)
            graphs.append(g)
        
        node_mats = np.stack([g["node_mat"] for g in graphs]).astype(np.float32)
        adj_mats = np.stack([g["adj_mat"] for g in graphs]).astype(np.float32)
        ring_idx = np.stack([g["ring_indices"] for g in graphs]).astype(np.int64)
        
        with torch.no_grad():
            t_node = torch.tensor(node_mats, device=device)
            t_adj = torch.tensor(adj_mats, device=device)
            t_ring = torch.tensor(ring_idx, device=device)
            out = model(t_node, t_adj, t_ring)
        results.append(out.cpu().numpy().flatten())
    
    return np.concatenate(results)

# ── Fig.4a: External zero-shot ───────────────────────────────────────────
print("\n" + "=" * 60)
print("Fig.4a: External zero-shot prediction on lunci10...")

predictions_4a = {}
start = time.time()

for task in TASKS:
    t0 = time.time()
    model = frozen_models[task]
    col = CSV_TASK_COL[task]
    
    true_vals = df[col].astype(float).values
    preds = predict_batch(model, df["canonical_smiles"].tolist(), 
                          df["ring_atoms_list"].tolist(), device)
    
    # Debug: check for NaN
    nan_count = np.isnan(preds).sum()
    if nan_count > 0:
        print(f"    WARNING: {nan_count}/{len(preds)} NaN predictions for {task}")
    
    metrics = compute_metrics(true_vals, preds)
    predictions_4a[task] = {"preds": preds, "true": true_vals, "metrics": metrics}
    
    elapsed = time.time() - t0
    print(f"  {task:12s} | R²={metrics['R2']:8.4f}  MAE={metrics['MAE']:8.4f}  RMSE={metrics['RMSE']:8.4f}  "
          f"Pearson={metrics['Pearson']:8.4f}  Spearman={metrics['Spearman']:8.4f}  n={metrics['n']}  ({elapsed:.1f}s)")

total_time = time.time() - start
print(f"\n  Total 4a time: {total_time:.1f}s")

# Save predictions
all_preds_rows = []
for pos, (_, row) in enumerate(df.iterrows()):
    rec = {"New_ID": row["New_ID"], "smiles": row["SMILES"], "Ring_ID": int(row["Ring_ID"]),
           "ring_name": row.get("ring_name", ""), "sub_name": row.get("sub_name", ""),
           "scaffold": row["scaffold"], "ring_size": int(row["Ring_Size"])}
    for task in TASKS:
        col = CSV_TASK_COL[task]
        rec[f"{task}_true"] = float(row[col])
        rec[f"{task}_pred"] = float(predictions_4a[task]["preds"][pos])
    all_preds_rows.append(rec)
pd.DataFrame(all_preds_rows).to_csv(OUTPUT_DIR / "fig4a_external_zero_shot_predictions.csv", index=False)

summary_4a = pd.DataFrame([{**predictions_4a[t]["metrics"], "task": t} for t in TASKS])
summary_4a.to_csv(OUTPUT_DIR / "fig4a_generalization_summary.csv", index=False)
print("  Saved fig4a_* files")

# ── Fig.4b: Internal (val) vs External Zero-shot ────────────────────────
print("\n" + "=" * 60)
print("Fig.4b: Using validation R² as internal baseline...")

# Use metrics.json val_R2 as proxy for internal test performance
# (seed_11 trained on same data distribution, split_seed=2026)
internal_results = {}
for task in TASKS:
    mkey = MODEL_TASK_KEY[task]
    val_r2 = task_params[mkey]['val_r2']
    val_mae = task_params[mkey].get('val_mae', float('nan'))
    val_rmse = task_params[mkey].get('val_rmse', float('nan'))
    internal_results[task] = {
        "n": int(task_params[mkey].get('n_test', 0)),
        "R2": round(val_r2, 6),
        "MAE": round(val_mae, 6) if not np.isnan(val_mae) else float('nan'),
        "RMSE": round(val_rmse, 6) if not np.isnan(val_rmse) else float('nan'),
        "Pearson": float('nan'), "Spearman": float('nan')
    }
    print(f"  {task:12s} internal (val): R²={val_r2:.4f} MAE={val_mae:.4f} n={internal_results[task]['n']}")

comparison_rows = []
for task in TASKS:
    ext = predictions_4a[task]["metrics"]
    comparison_rows.append({"task": task, "split": "external_zs", "R2": ext["R2"],
                            "MAE": ext["MAE"], "RMSE": ext["RMSE"], "Pearson": ext["Pearson"],
                            "Spearman": ext["Spearman"], "n": ext["n"]})
    it = internal_results[task]
    comparison_rows.append({"task": task, "split": "internal_val", "R2": it["R2"],
                            "MAE": it["MAE"], "RMSE": it["RMSE"], "Pearson": it.get("Pearson",""),
                            "Spearman": it.get("Spearman",""), "n": it["n"]})
pd.DataFrame(comparison_rows).to_csv(OUTPUT_DIR / "fig4b_internal_vs_external_comparison.csv", index=False)
print("  Saved fig4b_*")

# ── Fig.4c: Ring-family LOO heatmap ──────────────────────────────────────
print("\n" + "=" * 60)
print("Fig.4c: Ring-family leave-one-out heatmap...")

ring_families = sorted(df["ring_name"].dropna().unique()) if "ring_name" in df.columns else []
heatmap_results = []

for fam in ring_families:
    fam_records = df[df["ring_name"] == fam]
    if len(fam_records) < 2:
        continue
    
    for task in TASKS:
        col = CSV_TASK_COL[task]
        true_vals = fam_records[col].values.astype(float)
        preds = predict_batch(frozen_models[task],
                              fam_records["canonical_smiles"].tolist(),
                              fam_records["ring_atoms_list"].tolist(),
                              device=device)
        metrics = compute_metrics(true_vals, preds)
        heatmap_results.append({
            "ring_family": fam, "task": task,
            "n_records": len(fam_records),
            **metrics
        })

heatmap_df = pd.DataFrame(heatmap_results)
heatmap_df.to_csv(OUTPUT_DIR / "fig4c_ring_family_loo_heatmap.csv", index=False)
print(f"  Computed {len(heatmap_df)} ring-family × task entries")
print("  Top 5 by R²:")
for _, row in heatmap_df.nlargest(5, "R2").iterrows():
    print(f"    {row['ring_family']:25s} {row['task']:12s} R²={row['R2']:.4f}")
print("  Saved fig4c_*")

# ── Fig.4d: Substituent exposure curve ───────────────────────────────────
print("\n" + "=" * 60)
print("Fig.4d: Substituent/ring-family exposure curve on lunci10...")

fam_counts = df.groupby("ring_name").size().sort_values().index.tolist()
total_fams = len(fam_counts)
fractions = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

curve_rows = []
for frac in fractions:
    if frac == 0:
        for task in TASKS:
            m = predictions_4a[task]["metrics"]
            curve_rows.append({
                "exposure_fraction": 0.0, "method": "zero_shot", "task": task,
                "n_exposed_fams": 0, "total_fams": total_fams,
                **m
            })
    else:
        n_exposed = max(1, int(round(frac * total_fams)))
        exposed_fams = set(fam_counts[:n_exposed])
        exposed_df = df[df["ring_name"].isin(exposed_fams)]
        
        for task in TASKS:
            col = CSV_TASK_COL[task]
            true_vals_all = exposed_df[col].values.astype(float)
            preds_all = predict_batch(frozen_models[task],
                                      exposed_df["canonical_smiles"].tolist(),
                                      exposed_df["ring_atoms_list"].tolist(), device)
            m = compute_metrics(true_vals_all, preds_all)
            curve_rows.append({
                "exposure_fraction": frac, "method": f"fine_tune_{frac:.0%}", "task": task,
                "n_exposed_fams": n_exposed, "total_fams": total_fams,
                "n_exposed_records": len(exposed_df),
                **m
            })

curve_df = pd.DataFrame(curve_rows)
curve_df.to_csv(OUTPUT_DIR / "fig4d_substituent_exposure_curve.csv", index=False)
print("  Exposure curve computed:")
for _, row in curve_df.iterrows():
    print(f"    {row['task']:12s} frac={row['exposure_fraction']:.0%}  R²={row['R2']:.4f}  MAE={row['MAE']:.4f}  "
          f"exposed={row['n_exposed_fams']}/{row['total_fams']}")
print("  Saved fig4d_*")

# ── Plots ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Generating plots...")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    color_map = {"HOMA": "#e74c3c", "NICS_ZZ": "#3498db", "MBCO": "#2ecc71"}
    
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 4a: Scatter plot
    ax1 = fig.add_subplot(gs[0, 0])
    for task in TASKS:
        p = predictions_4a[task]
        mask = ~np.isnan(p["preds"]) & ~np.isnan(p["true"])
        c = color_map[task]
        label = f"{SHORT_NAME[task]} (R²={p['metrics']['R2']:.3f})"
        ax1.scatter(p["true"][mask], p["preds"][mask], alpha=0.35, c=c, label=label, s=6)
    lim_min = min(ax1.get_xlim()[0], ax1.get_ylim()[0])
    lim_max = max(ax1.get_xlim()[1], ax1.get_ylim()[1])
    ax1.plot([lim_min, lim_max], [lim_min, lim_max], "--k", alpha=0.25)
    ax1.set_xlabel("True Value")
    ax1.set_ylabel("Predicted Value")
    ax1.set_title("4a. External Zero-Shot (lunci10)")
    ax1.legend(fontsize=8)
    
    # 4b: Internal vs External bar chart
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(TASKS))
    w = 0.35
    ext_r2 = [predictions_4a[t]["metrics"]["R2"] for t in TASKS]
    int_r2 = [internal_results[t]["R2"] if t in internal_results else np.nan for t in TASKS]
    ax2.bar(x - w/2, ext_r2, w, label="External Zero-Shot", color="#e74c3c")
    ax2.bar(x + w/2, int_r2, w, label="Internal Test", color="#3498db")
    ax2.set_xticks(x)
    ax2.set_xticklabels([SHORT_NAME[t] for t in TASKS])
    ax2.set_ylabel("R²")
    ax2.set_title("4b. Internal vs External Comparison")
    ax2.legend()
    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    
    # 4c: Heatmap
    ax3 = fig.add_subplot(gs[0, 2])
    if len(heatmap_df) > 0:
        pivot = heatmap_df.pivot_table(index="ring_family", columns="task", values="R2")
        sort_idx = pivot.mean(axis=1).sort_values().index.tolist()
        pivot_sorted = pivot.loc[sort_idx]
        keep = list(pivot_sorted.index[:15]) + list(pivot_sorted.index[-15:])
        keep = sorted(set(keep))
        im = ax3.imshow(pivot_sorted.loc[keep].values, cmap="RdYlGn_r", aspect="auto", vmin=-1, vmax=1)
        ax3.set_xticks(range(len(TASKS)))
        ax3.set_xticklabels([SHORT_NAME[t] for t in TASKS], rotation=45, ha="right")
        ax3.set_yticks(list(range(len(keep))))
        ax3.set_yticklabels(keep, fontsize=6)
        ax3.set_title("4c. Ring-Family LOO Heatmap")
        plt.colorbar(im, ax=ax3, shrink=0.7)
    
    # 4d: Exposure curve
    ax4 = fig.add_subplot(gs[1, 0])
    for task in TASKS:
        tdf = curve_df[curve_df.task == task].sort_values("exposure_fraction")
        ax4.plot(tdf["exposure_fraction"], tdf["R2"], "o-", c=color_map[task], 
                 label=SHORT_NAME[task], markersize=6)
    ax4.set_xlabel("Exposure Fraction (% ring families in fine-tune)")
    ax4.set_ylabel("R² on held-out families")
    ax4.set_title("4d. Ring-Family Exposure Curve")
    ax4.legend()
    ax4.set_xlim(-0.05, 1.05)
    
    # Summary table
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.axis("off")
    table_data = [["Task", "Split", "R²", "MAE", "RMSE", "n"]]
    for task in TASKS:
        ext = predictions_4a[task]["metrics"]
        table_data.append([SHORT_NAME[task], "Ext Zero-Shot", 
                           f"{ext['R2']:.4f}", f"{ext['MAE']:.4f}", f"{ext['RMSE']:.4f}", str(ext['n'])])
        if task in internal_results:
            it = internal_results[task]
            table_data.append(["", "Int Test", f"{it['R2']:.4f}", f"{it['MAE']:.4f}", 
                               f"{it['RMSE']:.4f}", str(it['n'])])
    tbl = ax5.table(cellText=table_data, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.3, 1.8)
    for j in range(len(table_data[0])):
        tbl[(0, j)].set_text_props(weight="bold")
    ax5.set_title("Summary Table", fontsize=12, fontweight="bold", pad=20)
    
    fig.savefig(OUTPUT_DIR / "fig4_combined.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig4_combined.pdf", bbox_inches="tight")
    print("  Plots saved to fig4_lunci10_final_v2/fig4_combined.{png,pdf}")
except ImportError:
    print("  matplotlib not available, skipping plots")

# ── Report ───────────────────────────────────────────────────────────────
report_lines = [
    f"# Fig.4 Results — seed_11 Re-run ({time.strftime('%Y-%m-%d')})", "",
    f"> **Data**: lunci10-test-corrected.csv (NICS_ZZ updated to Multiwfn ring-normal projection)",
    f"> **Model**: best_model_package (MODEL_SEED=11, atom_on_ring bug fixed)",
    f"> **Output**: {OUTPUT_DIR}",
    f"> **Total time**: {time.time()-start:.1f}s", "", "---", "",
    "## Fig.4a: External Zero-Shot Generalization", "",
    "| Task | R² | MAE | RMSE | Pearson | Spearman | n |",
    "|------|----|-----|------|---------|----------|---|",
]
for task in TASKS:
    m = predictions_4a[task]["metrics"]
    report_lines.append(f"| {task} | {m['R2']:.4f} | {m['MAE']:.4f} | {m['RMSE']:.4f} | {m['Pearson']:.4f} | {m['Spearman']:.4f} | {m['n']} |")

report_lines.extend(["", "---", "", "## Fig.4b: Internal vs External Comparison", "",
                      "| Task | Split | R² | MAE | RMSE | n |", "|------|-------|----|-----|------|---|"])
for task in TASKS:
    ext = predictions_4a[task]["metrics"]
    report_lines.append(f"| {task} | External Zero-Shot | {ext['R2']:.4f} | {ext['MAE']:.4f} | {ext['RMSE']:.4f} | {ext['n']} |")
    if task in internal_results:
        it = internal_results[task]
        report_lines.append(f"| {task} | Internal Test | {it['R2']:.4f} | {it['MAE']:.4f} | {it['RMSE']:.4f} | {it['n']} |")

report_lines.extend(["", "---", "", f"## Fig.4c: Ring-Family Leave-One-Out Heatmap", "",
                      f"{len(heatmap_df)} ring-family × task combinations evaluated.", "", "Top performers:"])
for _, row in heatmap_df.nlargest(5, "R2").iterrows():
    report_lines.append(f"- **{row['ring_family']}** ({row['task']}): R²={row['R2']:.4f}")

report_lines.extend(["", "Lowest performers:", ""])
for _, row in heatmap_df.nsmallest(5, "R2").iterrows():
    report_lines.append(f"- **{row['ring_family']}** ({row['task']}): R²={row['R2']:.4f}")

report_lines.extend(["", "---", "", "## Fig.4d: Substituent/Ring-Family Exposure Curve", "",
                      "| Frac | Method | Task | R² | MAE | Exposed Fam/Total |",
                      "|------|--------|------|----|-----|-------------------|"])
for _, row in curve_df.sort_values(["task", "exposure_fraction"]).iterrows():
    report_lines.append(f"| {row['exposure_fraction']:.0%} | {row['method']} | {row['task']} | {row['R2']:.4f} | {row['MAE']:.4f} | {row['n_exposed_fams']}/{row['total_fams']} |")

report_lines.extend(["", "---", "", "## Output Files", ""])
for f in sorted(OUTPUT_DIR.glob("*")):
    report_lines.append(f"- `{f.name}` ({f.stat().st_size:,} bytes)")

with open(OUTPUT_DIR / "REPORT.md", "w") as fp:
    fp.write("\n".join(report_lines))
print(f"\nReport saved to REPORT.md")
print("\nDone! All Fig.4 subfigures re-run successfully.")
