# -*- coding: utf-8 -*-
"""
Step 5: 用 merged 重训 HOMA 模型绘制环注意力 saliency 图像 (单模型版)

用户决定: 只训练一个模型 (HOMA) 已足够, 重点是注意力图。
输出到 0908-end-code/attention_viz/
  HOMA/{phenol_polyhalogen,...}.png       单分子 saliency 图
  HOMA_category_comparison.png            5 类分子同排对比
  HOMA_summary_panel.png                  环内 vs 取代基 统计面板 (单任务)
  attention_stats_HOMA.csv

用法: CUDA_VISIBLE_DEVICES=2 python3 saliency_homa_driver.py
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import saliency_merged as sm  # noqa: E402

from rdkit import Chem  # noqa: E402

TASK = "HOMA"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sm.DEVICE = DEVICE
print(f"Device: {DEVICE}")
print(f"输出目录: {sm.OUTPUT_ROOT}")


def load_homa_model():
    fname = sm.MODEL_FILES[TASK]
    ckpt = torch.load(os.path.join(sm.MODEL_PKG, fname), map_location="cpu")
    cfg = ckpt["config"]
    model = sm.build_model(use_projection=cfg["use_projection"],
                           node_vec_len=cfg["node_vec_len"],
                           hidden_dim=cfg["hidden_dim"], n_conv=cfg["n_conv"],
                           n_hidden=cfg["n_hidden"], p_dropout=cfg["p_dropout"],
                           ring_flag_value=cfg["ring_flag_value"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE)
    model.eval()
    print(f"  [load] {fname}: use_projection={cfg['use_projection']} "
          f"ring_flag={cfg['ring_flag_value']}")
    return model


def summary_panel_single(df, save_path):
    """单任务统计面板: 环内 vs 取代基 (原子级) + 边级 rr/rs"""
    df = df.set_index("molecule")
    mols = list(df.index)
    x = np.arange(len(mols))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=150)
    ax = axes[0, 0]
    ax.bar(x - 0.15, df["ring_mean"], 0.3, label="环内", color="#E76F51",
           alpha=0.85, edgecolor="black", linewidth=0.4)
    ax.bar(x + 0.15, df["sub_mean"], 0.3, label="取代基", color="#E76F51",
           alpha=0.45, hatch="//", edgecolor="black", linewidth=0.4)
    for xi, m, s in zip(x, df["ring_mean"], df["sub_mean"]):
        ax.text(xi - 0.15, m, f"{m:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + 0.15, s, f"{s:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(mols, fontsize=8, rotation=12)
    ax.set_ylabel("saliency 均值"); ax.set_title("原子级: 环内 vs 取代基")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

    ax = axes[0, 1]
    ax.bar(x, df["ratio_ring_over_sub"], 0.5, color="#E76F51", alpha=0.85,
           edgecolor="black", linewidth=0.4)
    for xi, v in zip(x, df["ratio_ring_over_sub"]):
        ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(1, color="red", ls="--", alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(mols, fontsize=8, rotation=12)
    ax.set_ylabel("环内/取代基"); ax.set_title("原子级聚焦系数 (>1 关注环内)")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 0]
    ax.bar(x - 0.15, df["rr_edge_mean"], 0.3, label="环-环", color="#3568C0",
           alpha=0.85, edgecolor="black", linewidth=0.4)
    ax.bar(x + 0.15, df["rs_edge_mean"], 0.3, label="环-取代基", color="#3568C0",
           alpha=0.45, hatch="//", edgecolor="black", linewidth=0.4)
    for xi, a, b in zip(x, df["rr_edge_mean"], df["rs_edge_mean"]):
        ax.text(xi - 0.15, a, f"{a:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + 0.15, b, f"{b:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(mols, fontsize=8, rotation=12)
    ax.set_ylabel("边权重均值"); ax.set_title("边级: 环-环 vs 环-取代基")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 1]
    ax.bar(x, df["rr_over_rs"], 0.5, color="#3568C0", alpha=0.85,
           edgecolor="black", linewidth=0.4)
    for xi, v in zip(x, df["rr_over_rs"]):
        ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(1, color="red", ls="--", alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(mols, fontsize=8, rotation=12)
    ax.set_ylabel("环-环/环-取代"); ax.set_title("边级聚焦系数 (>1 关注环内键)")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"HOMA Merged 重训模型 · 环注意力聚焦统计 · {sm.MODEL_TAG}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close()
    print("[saved]", save_path)


def main():
    model = load_homa_model()
    task_dir = os.path.join(sm.OUTPUT_ROOT, TASK)
    os.makedirs(task_dir, exist_ok=True)

    mol_cache = {}
    all_stats = []
    print(f"\n--- {sm.TASK_LABELS[TASK]} (merged 重训) ---")
    for mol_info in sm.TEST_MOLECULES:
        try:
            smiles = mol_info["smiles"]
            aor = mol_info["atom_on_ring"]
            if smiles not in mol_cache:
                mol_cache[smiles] = Chem.AddHs(Chem.MolFromSmiles(smiles))
            mol = mol_cache[smiles]

            g = sm.build_graph(smiles, aor, sm.NODE_VEC_LEN, sm.MAX_ATOMS, ring_flag_value=1)
            node = torch.tensor(g["node_mat"][None], dtype=torch.float32, device=DEVICE)
            adj = torch.tensor(g["adj_mat"][None], dtype=torch.float32, device=DEVICE)
            ri = torch.tensor(g["ring_indices"][None], dtype=torch.long, device=DEVICE)
            sal, pred = sm.extract_saliency(model, node, adj, ri)
            true_val = sm.load_true_value(TASK, smiles, aor)
            stats = sm.compute_stats(mol, sal, aor)
            stats.update({"task": TASK, "molecule": mol_info["name"],
                          "category": mol_info["category"],
                          "description": mol_info["description"], "smiles": smiles,
                          "atom_on_ring": aor, "pred": pred, "true": true_val,
                          "method": "saliency (|x·∇x|)"})
            all_stats.append(stats)

            save_path = os.path.join(task_dir, f"{mol_info['name']}.png")
            sm.draw_molecule(TASK, mol_info, sal, pred, true_val, save_path)
            print(f"  {mol_info['name']:22s} ring={stats['ring_mean']:.4f} "
                  f"sub={stats['sub_mean']:.4f} ratio={stats['ratio_ring_over_sub']:.2f} "
                  f"rr/rs={stats['rr_over_rs']:.2f} pred={pred:.3f} true={true_val:.3f}")
        except Exception as e:
            print(f"  ERROR {mol_info['name']}: {e}")
            import traceback; traceback.print_exc()

    df = pd.DataFrame(all_stats)
    df.to_csv(os.path.join(sm.OUTPUT_ROOT, "attention_stats_HOMA.csv"), index=False)
    print(f"\n统计表 -> {os.path.join(sm.OUTPUT_ROOT, 'attention_stats_HOMA.csv')}")

    # 5 分子同排对比 + 统计面板
    comp_path = os.path.join(sm.OUTPUT_ROOT, f"{TASK}_category_comparison.png")
    sm.make_single_category_comparison(TASK, model, comp_path)
    summary_panel_single(df, os.path.join(sm.OUTPUT_ROOT, f"{TASK}_summary_panel.png"))
    print("\n完成!")


if __name__ == "__main__":
    main()
