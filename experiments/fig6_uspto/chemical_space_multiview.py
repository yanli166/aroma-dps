#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


# =========================
# 1. 配置
# =========================
INPUT_FILE = "/home/ubuntu/aroma-dps-code/uspto-5k/dearom_ring_pairs_A_tierA/ring_pair_aromaticity_predictions_complete.csv"
OUTDIR = "/home/ubuntu/aroma-dps-code/uspto-5k/fig5_analysis/chemical_space_multiview"
os.makedirs(OUTDIR, exist_ok=True)

RANDOM_STATE = 42
FP_BITS = 2048
FP_RADIUS = 2

DRAW_TRAJECTORIES = True
TRAJ_SAMPLE_N = 300


# =========================
# 2. 读取与整理数据
# =========================
df = pd.read_csv(INPUT_FILE, low_memory=False)

# complete CSV 不含 SMILES，从 ring_pairs_ml.csv 并入 reactant/product smiles
RING_PAIRS_FILE = "/home/ubuntu/aroma-dps-code/uspto-5k/dearom_ring_pairs_A_tierA/ring_pairs_ml.csv"
rp = pd.read_csv(RING_PAIRS_FILE, low_memory=False, usecols=[
    "pair_id", "reactant_component_smiles", "product_component_smiles"])
df = df.merge(rp, on="pair_id", how="left")
print(f"[INFO] merged smiles; missing R/P smiles: {df['reactant_component_smiles'].isna().sum()}/{len(df)}")

reactant_smiles_col = None
product_smiles_col = None

candidate_r_cols = ["reactant_component_smiles", "smiles_reactant", "reactant_smiles"]
candidate_p_cols = ["product_component_smiles", "smiles_product", "product_smiles"]

for c in candidate_r_cols:
    if c in df.columns:
        reactant_smiles_col = c
        break
for c in candidate_p_cols:
    if c in df.columns:
        product_smiles_col = c
        break

if reactant_smiles_col is None or product_smiles_col is None:
    raise ValueError("找不到 reactant/product SMILES 列，请检查输入文件。")

required_cols = [
    "pair_id", "HOMA_pred_reactant", "HOMA_pred_product",
    "nMCBO_pred_reactant", "nMCBO_pred_product",
    "NICS_1zz_pred_reactant", "NICS_1zz_pred_product",
    "Delta_HOMA", "Delta_nMCBO", "Delta_NICS_star",
]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"缺少必须列: {c}")

rows = []
for _, row in df.iterrows():
    nics_star_r = -row["NICS_1zz_pred_reactant"]
    nics_star_p = -row["NICS_1zz_pred_product"]
    rows.append({
        "pair_id": row["pair_id"], "side": "reactant",
        "smiles": row[reactant_smiles_col],
        "HOMA": row["HOMA_pred_reactant"], "nMCBO": row["nMCBO_pred_reactant"],
        "NICS_star": nics_star_r,
        "Delta_HOMA": row["Delta_HOMA"], "Delta_nMCBO": row["Delta_nMCBO"],
        "Delta_NICS_star": row["Delta_NICS_star"],
    })
    rows.append({
        "pair_id": row["pair_id"], "side": "product",
        "smiles": row[product_smiles_col],
        "HOMA": row["HOMA_pred_product"], "nMCBO": row["nMCBO_pred_product"],
        "NICS_star": nics_star_p,
        "Delta_HOMA": row["Delta_HOMA"], "Delta_nMCBO": row["Delta_nMCBO"],
        "Delta_NICS_star": row["Delta_NICS_star"],
    })

long_df = pd.DataFrame(rows).drop_duplicates(subset=["pair_id", "side"]).reset_index(drop=True)


# =========================
# 3. 计算 Morgan 指纹（带 target-ring 锚点，反映反应场景）
# =========================
def morgan_fp(smi, radius=2, nbits=2048):
    mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros((nbits,), dtype=np.int8)
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(fp, arr)
    return arr

fps = []
keep_idx = []
for i, smi in enumerate(long_df["smiles"]):
    arr = morgan_fp(smi, radius=FP_RADIUS, nbits=FP_BITS)
    if arr is not None:
        fps.append(arr)
        keep_idx.append(i)

embed_df = long_df.iloc[keep_idx].copy().reset_index(drop=True)
X = np.vstack(fps)
print(f"[INFO] valid molecules for embedding: {len(embed_df)}")


# =========================
# 4. 嵌入：PCA / UMAP / t-SNE
# =========================
pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
XY_pca = pca2.fit_transform(X)
embed_df["PCA1"] = XY_pca[:, 0]
embed_df["PCA2"] = XY_pca[:, 1]

if HAS_UMAP:
    umap2 = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.15,
                      metric="jaccard", random_state=RANDOM_STATE)
    XY_umap = umap2.fit_transform(X)
    embed_df["UMAP1"] = XY_umap[:, 0]
    embed_df["UMAP2"] = XY_umap[:, 1]
else:
    print("[WARN] 未安装 umap-learn，跳过 UMAP。")

pca50 = PCA(n_components=min(50, X.shape[1], X.shape[0] - 1), random_state=RANDOM_STATE)
X50 = pca50.fit_transform(X)
tsne2 = TSNE(n_components=2, perplexity=40, learning_rate="auto",
             init="pca", random_state=RANDOM_STATE)
XY_tsne = tsne2.fit_transform(X50)
embed_df["TSNE1"] = XY_tsne[:, 0]
embed_df["TSNE2"] = XY_tsne[:, 1]

embed_df.to_csv(os.path.join(OUTDIR, "chemical_space_coordinates.csv"), index=False)


# =========================
# 5. 画图函数
# =========================
def robust_limits(series, qlow=0.025, qhigh=0.975):
    x = pd.Series(series).dropna().values
    return np.quantile(x, qlow), np.quantile(x, qhigh)

def scatter_two_panel(df_plot, xcol, ycol, color_col, title, outname, cmap="viridis"):
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    vmin, vmax = robust_limits(df_plot[color_col])
    sub_r = df_plot[df_plot["side"] == "reactant"]
    sub_p = df_plot[df_plot["side"] == "product"]
    sc1 = ax1.scatter(sub_r[xcol], sub_r[ycol], c=sub_r[color_col],
                      s=14, alpha=0.75, vmin=vmin, vmax=vmax, cmap=cmap)
    ax1.set_title(f"{title} — Reactant")
    ax1.set_xlabel(xcol); ax1.set_ylabel(ycol)
    sc2 = ax2.scatter(sub_p[xcol], sub_p[ycol], c=sub_p[color_col],
                      s=14, alpha=0.75, vmin=vmin, vmax=vmax, cmap=cmap)
    ax2.set_title(f"{title} — Product")
    ax2.set_xlabel(xcol); ax2.set_ylabel(ycol)
    cbar = fig.colorbar(sc2, ax=[ax1, ax2], fraction=0.03, pad=0.04)
    cbar.set_label(color_col)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, outname), dpi=600, bbox_inches="tight")
    plt.close(fig)

def scatter_one_panel(df_plot, xcol, ycol, color_col, title, outname, cmap="viridis"):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(1, 1, 1)
    vmin, vmax = robust_limits(df_plot[color_col])
    sc = ax.scatter(df_plot[xcol], df_plot[ycol], c=df_plot[color_col],
                    s=14, alpha=0.75, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title); ax.set_xlabel(xcol); ax.set_ylabel(ycol)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label(color_col)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, outname), dpi=600, bbox_inches="tight")
    plt.close(fig)

def trajectory_plot(df_plot, xcol, ycol, color_col, title, outname, sample_n=300, cmap="plasma"):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(1, 1, 1)
    vmin, vmax = robust_limits(df_plot[color_col])
    r_df = df_plot[df_plot["side"] == "reactant"].copy()
    p_df = df_plot[df_plot["side"] == "product"].copy()
    merged = r_df.merge(p_df, on="pair_id", suffixes=("_R", "_P"))
    if len(merged) > sample_n:
        merged = merged.sample(sample_n, random_state=RANDOM_STATE)
    for _, row in merged.iterrows():
        ax.plot([row[f"{xcol}_R"], row[f"{xcol}_P"]],
                [row[f"{ycol}_R"], row[f"{ycol}_P"]],
                color="lightgray", linewidth=0.7, alpha=0.6)
    sc = ax.scatter(merged[f"{xcol}_R"], merged[f"{ycol}_R"],
                    c=merged[f"{color_col}_R"], cmap=cmap, vmin=vmin, vmax=vmax,
                    s=20, alpha=0.8, label="Reactant")
    ax.scatter(merged[f"{xcol}_P"], merged[f"{ycol}_P"],
               c=merged[f"{color_col}_P"], cmap=cmap, vmin=vmin, vmax=vmax,
               s=20, alpha=0.8, marker="^", label="Product")
    ax.set_title(title); ax.set_xlabel(xcol); ax.set_ylabel(ycol)
    ax.legend(frameon=False)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label(color_col)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, outname), dpi=600, bbox_inches="tight")
    plt.close(fig)


# =========================
# 6. 批量绘图
# =========================
embeddings = []
if "PCA1" in embed_df.columns:
    embeddings.append(("PCA1", "PCA2", "PCA"))
if "UMAP1" in embed_df.columns:
    embeddings.append(("UMAP1", "UMAP2", "UMAP"))
if "TSNE1" in embed_df.columns:
    embeddings.append(("TSNE1", "TSNE2", "tSNE"))

# 绝对芳香性三指标
for xcol, ycol, method in embeddings:
    scatter_two_panel(embed_df, xcol, ycol, "HOMA",
                      f"{method} chemical space colored by HOMA",
                      f"{method.lower()}_HOMA_reactant_product.png", cmap="viridis")
    scatter_two_panel(embed_df, xcol, ycol, "nMCBO",
                      f"{method} chemical space colored by nMCBO",
                      f"{method.lower()}_nMCBO_reactant_product.png", cmap="viridis")
    scatter_two_panel(embed_df, xcol, ycol, "NICS_star",
                      f"{method} chemical space colored by NICS*",
                      f"{method.lower()}_NICSstar_reactant_product.png", cmap="viridis")

# Delta 图（只看 reactant）
reactant_df = embed_df[embed_df["side"] == "reactant"].copy()
for xcol, ycol, method in embeddings:
    scatter_one_panel(reactant_df, xcol, ycol, "Delta_HOMA",
                      f"{method} reactant space colored by Delta_HOMA",
                      f"{method.lower()}_delta_HOMA_reactant.png", cmap="plasma")
    scatter_one_panel(reactant_df, xcol, ycol, "Delta_nMCBO",
                      f"{method} reactant space colored by Delta_nMCBO",
                      f"{method.lower()}_delta_nMCBO_reactant.png", cmap="plasma")
    scatter_one_panel(reactant_df, xcol, ycol, "Delta_NICS_star",
                      f"{method} reactant space colored by Delta_NICS_star",
                      f"{method.lower()}_delta_NICSstar_reactant.png", cmap="plasma")

if DRAW_TRAJECTORIES:
    if "UMAP1" in embed_df.columns:
        trajectory_plot(embed_df, "UMAP1", "UMAP2", "HOMA",
                        "UMAP trajectories: reactant to product (colored by HOMA)",
                        "umap_trajectory_HOMA.png", sample_n=TRAJ_SAMPLE_N, cmap="viridis")
    trajectory_plot(embed_df, "PCA1", "PCA2", "HOMA",
                    "PCA trajectories: reactant to product (colored by HOMA)",
                    "pca_trajectory_HOMA.png", sample_n=TRAJ_SAMPLE_N, cmap="viridis")


# =========================
# 7. 简单统计总结
# =========================
summary_lines = [
    f"Total input pairs: {len(df)}",
    f"Embedded molecule points (R+P): {len(embed_df)}",
    f"Reactant points: {(embed_df['side']=='reactant').sum()}",
    f"Product points: {(embed_df['side']=='product').sum()}",
]
with open(os.path.join(OUTDIR, "README_summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print("[DONE] All plots saved to:", OUTDIR)