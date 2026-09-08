#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig.5f + chemical space: target-ring-centered radius-2 local subgraph fingerprint,
PCA -> 50D, UMAP -> 2D.  Outputs target_ring_chemical_space.csv and UMAP figures.
Also computes target-centered UMAP stability across params for SI.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
import umap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["font.size"] = 8

ROOT = Path("/home/ubuntu/aroma-dps-code/uspto-5k")
DATA = ROOT / "dearom_ring_pairs_A_tierA"
OUT = ROOT / "fig5_analysis"
CHEM = OUT / "06_chemical_space"
MAIN = OUT / "09_fig5_main"
SI = OUT / "10_fig5_si"
for d in (CHEM, MAIN, SI):
    d.mkdir(parents=True, exist_ok=True)

DESC = ["Delta_HOMA", "Delta_nMCBO", "Delta_NICS_star"]
RANDOM_STATE = 42


def parse_bits(s):
    if not isinstance(s, str) or not s:
        return None
    return np.array([int(c) for c in s], dtype=np.uint8)


def robust_range(x, lo=2.5, hi=97.5):
    return (np.percentile(x, lo), np.percentile(x, hi))


def save(fig, name, main=True):
    d = MAIN if main else SI
    for ext in ("pdf", "svg", "png"):
        fig.savefig(d / f"{name}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    rfam = pd.read_csv(OUT / "07_ring_family/ring_family_annotations.csv", low_memory=False)
    master = pd.read_csv(OUT / "01_master/reaction_aromaticity_fig5_master.csv", low_memory=False)
    from rdkit import Chem
    from rdkit.Chem import AllChem

    def mfp(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

    # Build target-ring-centered fingerprint directly on the mapped reactant molecule:
    # use RDKit atom invariants to mark target ring atoms (and their neighbors stay real)
    # so the fingerprint is dominated by the target ring + its local environment.
    def target_centered_fp(smi, target_maps):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        target = set(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum() in target_maps)
        if not target:
            return None
        inv = [0] * mol.GetNumAtoms()
        # give target-ring atoms a distinct invariant value so they anchor the fingerprint
        for i in target:
            inv[i] = 1
        # neighbors within radius-2 hops of the target ring
        env = set(target)
        frontier = set(target)
        for _ in range(2):
            nxt = set()
            for i in frontier:
                nxt.update(nb.GetIdx() for nb in mol.GetAtomWithIdx(i).GetNeighbors())
            nxt -= env
            env |= nxt
            frontier = nxt
        for i in env:
            inv[i] = 2
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=2, nBits=2048, atomInvariants=inv)
            return fp
        except Exception:
            return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

    master2 = master.merge(
        rfam[["pair_id", "local_fragment_smiles"]], on="pair_id", how="left")
    maps_by_pair = {}
    for _, r in master.iterrows():
        maps_by_pair[r["pair_id"]] = set(
            int(x) for x in str(r["target_ring_map_numbers"]).replace(";", " ").split()
            if x.strip().isdigit())

    fps = []
    valid = []
    for smi, pid in zip(master2["reactant_component_smiles_mapped"], master2["pair_id"]):
        f = target_centered_fp(smi, maps_by_pair.get(pid, set()))
        if f is None:
            fps.append(None); valid.append(False)
        else:
            fps.append(np.array(list(map(int, f.ToBitString())), dtype=np.uint8)); valid.append(True)
    rf = master2.loc[valid, ["pair_id"]].copy()
    rf["local_fp_arr"] = fps
    print(f"[chemspace] samples with target-centered fp: {len(rf)} / {len(master2)}")

    m = master.merge(rf, on="pair_id", how="inner")
    print("merged:", len(m))
    fp_mat = np.stack(m["local_fp_arr"].values)
    print("fingerprint matrix:", fp_mat.shape)

    pca = PCA(n_components=50, random_state=RANDOM_STATE)
    Z50 = pca.fit_transform(fp_mat.astype(float))
    print(f"PCA50 explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    # UMAP fixed params
    u = umap.UMAP(n_neighbors=30, min_dist=0.15, metric="jaccard",
                  n_components=2, random_state=RANDOM_STATE)
    emb = u.fit_transform(Z50)
    print("UMAP done:", emb.shape)

    m = m.reset_index(drop=True)
    m["UMAP1"] = emb[:, 0]
    m["UMAP2"] = emb[:, 1]
    out_cols = ["pair_id", "UMAP1", "UMAP2",
                "Delta_HOMA", "Delta_nMCBO", "Delta_NICS_star", "aromaticity_pattern"]
    m[out_cols].to_csv(CHEM / "target_ring_chemical_space.csv", index=False)
    print("wrote target_ring_chemical_space.csv")

    is_disc = (m["aromaticity_pattern"] == "DISCORDANT").values

    # A: uniform grey
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.scatter(emb[:, 0], emb[:, 1], s=4, alpha=0.35, color="#8a939b", rasterized=True)
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.set_title(f"Target-ring chemical space (N={len(m)})", fontsize=9)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    save(fig, "chemspace_A_uniform")

    # B/C/D: color by delta, robust range
    for dcol in DESC:
        x = m[dcol].astype(float).values
        lo, hi = robust_range(x)
        fig, ax = plt.subplots(figsize=(4.8, 4.4))
        sc = ax.scatter(emb[:, 0], emb[:, 1], s=4, c=x, cmap="viridis",
                        vmin=lo, vmax=hi, alpha=0.7, rasterized=True)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046)
        cb.set_label(dcol, fontsize=8)
        ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
        ax.set_title(f"{dcol} (robust 2.5-97.5 pct)", fontsize=9)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        save(fig, f"chemspace_color_{dcol.replace('Delta_','')}")

    # E: ALL_LOSS grey, DISCORDANT highlight
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.scatter(emb[~is_disc, 0], emb[~is_disc, 1], s=4, alpha=0.35, color="#9aa5ad",
               label="ALL_LOSS", rasterized=True)
    ax.scatter(emb[is_disc, 0], emb[is_disc, 1], s=14, alpha=0.9, color="#C44E52",
               marker="x", label=f"DISCORDANT (n={is_disc.sum()})")
    ax.legend(fontsize=7, frameon=False)
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.set_title("Discordant cases in target-ring chemical space", fontsize=9)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    save(fig, "Fig5f_chemical_space_discordant")

    # Quantitative check: are discordant cases clustered? (Euclidean kNN density proxy)
    # compare mean pairwise distance of discordant vs random baseline in UMAP coords
    rng = np.random.default_rng(1)
    disc_idx = np.where(is_disc)[0]
    n_disc = len(disc_idx)
    # mean distance to k nearest ALL_LOSS neighbors
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=30).fit(emb)
    dists, _ = nn.kneighbors(emb)
    disc_knn = dists[disc_idx, 1:].mean()
    base_knn = dists[:, 1:].mean()
    print(f"UMAP 30-NN mean dist: discordant={disc_knn:.4f} all={base_knn:.4f} "
          f"ratio={disc_knn/base_knn:.3f}")
    (CHEM / "discordant_clustering_check.json").write_text(json.dumps({
        "n_discordant": int(n_disc),
        "umap_30nn_mean_dist_discordant": float(disc_knn),
        "umap_30nn_mean_dist_all": float(base_knn),
        "note": "UMAP is visualization only; for quantitative similarity use Tanimoto.",
    }, indent=2))

    # SI: target-centered UMAP stability across n_neighbors
    for nn_, md_ in [(20, 0.15), (40, 0.15), (30, 0.05)]:
        uu = umap.UMAP(n_neighbors=nn_, min_dist=md_, metric="jaccard",
                       n_components=2, random_state=RANDOM_STATE)
        e = uu.fit_transform(Z50)
        fig, ax = plt.subplots(figsize=(4.4, 4.0))
        ax.scatter(e[:, 0], e[:, 1], s=4, alpha=0.35, color="#8a939b", rasterized=True)
        ax.scatter(e[is_disc, 0], e[is_disc, 1], s=14, alpha=0.9, color="#C44E52", marker="x")
        ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
        ax.set_title(f"Stability: n_neighbors={nn_}, min_dist={md_}", fontsize=9)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        save(fig, f"S10_umap_stability_{nn_}_{str(md_).replace('.','p')}", main=False)

    # SI: whole-molecule UMAP (radius-2 2048 on reactant component)
    print("computing whole-molecule UMAP for SI...")
    fps = []
    valid = []
    for smi in m["reactant_component_smiles"]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(None); valid.append(False)
        else:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)); valid.append(True)
    ok = np.array(valid)
    fp_arr = np.array([list(map(int, list(f.ToBitString()))) for f in
                       [x for x in fps if x is not None]], dtype=np.uint8)
    pca2 = PCA(n_components=50, random_state=RANDOM_STATE)
    Z50w = pca2.fit_transform(fp_arr.astype(float))
    uw = umap.UMAP(n_neighbors=30, min_dist=0.15, metric="jaccard",
                   n_components=2, random_state=RANDOM_STATE).fit_transform(Z50w)
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.scatter(uw[:, 0], uw[:, 1], s=4, alpha=0.35, color="#8a939b", rasterized=True)
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.set_title("Whole-molecule reactant UMAP (SI)", fontsize=9)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    save(fig, "S9_whole_molecule_umap", main=False)
    print("DONE chemical space")


if __name__ == "__main__":
    main()
