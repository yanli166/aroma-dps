# -*- coding: utf-8 -*-
"""
Step C: 在 l10 验证集分子上重画环注意力 (双归因方法, 视觉优化版)
  - Integrated Gradients (signed): 红=推高芳香性 / 蓝=拉低
  - 输入×梯度 |x·dy/dx| (magma)
  颜色用对称幂次 PowerNorm 放大弱贡献, 使取代基贡献可见且 colorbar 自洽;
  取代基重原子上标注原始归因数值。

用法:
  CUDA_VISIBLE_DEVICES=2 python3 saliency_l10val.py --task HOMA --gpu 0
  CUDA_VISIBLE_DEVICES=2 python3 saliency_l10val.py --task NICS_1zz --gpu 0
  CUDA_VISIBLE_DEVICES=2 python3 saliency_l10val.py --task MBCO --gpu 0
输出: attention_l10val[_task]/*.png
"""
import os
import sys
import ast
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, "/home/ubuntu/aroma-dps-code/best_model_package")
from graph_utils import build_graph, NODE_VEC_LEN, MAX_ATOMS  # noqa: E402
from model_arch import build_model                             # noqa: E402

from matplotlib import font_manager
for fpath in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
              '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc']:
    if os.path.exists(fpath):
        try:
            font_manager.fontManager.addfont(fpath)
        except Exception:
            pass
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# task -> (pred_csv, ckpt, 输出子目录, 显示名, 颜色语义方向 flip)
#   NICS(1)zz 越负越芳香: 显示归因乘 -1, 使 红=增强芳香性 与 HOMA/MBCO 对齐
TASKS = {
    "HOMA": ("predictions/HOMA_l10val_predictions.csv", "models/HOMA_l10val_best.pt",
             "attention_l10val", "HOMA", 1.0),
    "NICS_1zz": ("predictions/NICS_1zz_l10val_predictions.csv", "models/NICS_1zz_l10val_best.pt",
                 "attention_l10val_nics", "NICS(1)zz", -1.0),
    "MBCO": ("predictions/MBCO_l10val_predictions.csv", "models/MBCO_l10val_best.pt",
             "attention_l10val_mbco", "MBCO", 1.0),
}
CPK_COLORS = {"C": "#303030", "H": "#FFFFFF", "O": "#FF0D0D", "N": "#3050F8",
              "S": "#DAA520", "F": "#90E050", "Cl": "#1FF01F", "Br": "#A62929",
              "I": "#940094", "P": "#FF8000", "B": "#FFB5B5"}
N_SELECT = 6
GAMMA = 0.45  # <1 放大弱贡献


class PowNorm(Normalize):
    """正值幂次拉伸: (v-vmin)/(vmax-vmin) 再开幂"""
    def __init__(self, vmin=None, vmax=None, gamma=GAMMA, clip=False):
        super().__init__(vmin, vmax, clip)
        self.gamma = gamma

    def __call__(self, value, clip=None):
        value, is_scalar = self.process_value(value)
        self.autoscale_None(value)
        vmin, vmax = self.vmin, self.vmax
        if vmin >= vmax:
            out = np.full_like(value, 0.5)
        else:
            out = np.clip((value - vmin) / (vmax - vmin), 0, 1) ** self.gamma
        out = np.ma.masked_array(out, np.ma.getmask(value))
        return out[()] if is_scalar else out

    def inverse(self, value):
        value = np.asarray(value)
        return self.vmin + (np.clip(value, 0, 1) ** (1.0 / self.gamma)) * (self.vmax - self.vmin)


class SignedPowNorm(Normalize):
    """对称带符号幂次拉伸, 适合正负归因: 中间 0 处最淡"""
    def __init__(self, vmin=None, vmax=None, gamma=GAMMA, clip=False):
        super().__init__(vmin, vmax, clip)
        self.gamma = gamma

    def __call__(self, value, clip=None):
        value, is_scalar = self.process_value(value)
        self.autoscale_None(value)
        m = max(abs(self.vmin), abs(self.vmax), 1e-12)
        out = np.sign(value) * (np.abs(np.clip(value, -m, m)) / m) ** self.gamma
        out = 0.5 + 0.5 * out
        out = np.ma.masked_array(out, np.ma.getmask(value))
        return out[()] if is_scalar else out

    def inverse(self, value):
        value = np.asarray(value)
        z = 2.0 * np.clip(value, 0, 1) - 1.0
        m = max(abs(self.vmin), abs(self.vmax), 1e-12)
        return np.sign(z) * (np.abs(z) ** (1.0 / self.gamma)) * m


def parse_aor(x):
    return ast.literal_eval(x) if isinstance(x, str) else list(x)


def load_model(device, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    model = build_model(use_projection=cfg["use_projection"],
                        node_vec_len=cfg["node_vec_len"], hidden_dim=cfg["hidden_dim"],
                        n_conv=cfg["n_conv"], n_hidden=cfg["n_hidden"],
                        p_dropout=cfg["p_dropout"],
                        ring_flag_value=cfg["ring_flag_value"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def integrated_gradients(model, node_mat, adj_mat, ring_indices, steps=40):
    """零特征基线路径积分 -> 逐原子带符号归因"""
    x1 = node_mat.clone().detach()
    x0 = torch.zeros_like(x1)
    total = torch.zeros_like(x1)
    for a in torch.linspace(0, 1, steps, device=x1.device):
        xm = (x0 + a * (x1 - x0)).requires_grad_(True)
        model.zero_grad()
        model(xm, adj_mat, ring_indices).sum().backward()
        if xm.grad is not None:
            total += xm.grad
    attrib = (x1 - x0) * (total / steps)
    return attrib.sum(dim=-1).squeeze(0).detach().cpu().numpy()


def grad_input_saliency(model, node_mat, adj_mat, ring_indices):
    x = node_mat.clone().detach().requires_grad_(True)
    model.zero_grad()
    model(x, adj_mat, ring_indices).sum().backward()
    return (x * x.grad).abs().sum(dim=-1).squeeze(0).detach().cpu().numpy()


def get_2d_coords(mol):
    try:
        AllChem.Compute2DCoords(mol)
    except Exception:
        pass
    conf = mol.GetConformer()
    return np.array([(conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y)
                     for i in range(mol.GetNumAtoms())])


def target_ring_polygon(coords, ring_atoms):
    pts = coords[list(ring_atoms)]
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return pts[np.argsort(ang)]


def ring_highlight_rings(mol, target_atoms):
    ri = mol.GetRingInfo()
    tset = set(target_atoms)
    return [r for r in ri.AtomRings() if len(set(r) & tset) >= 3]


def draw_panel(ax, mol, coords, atom_vals, target_atoms, cmap, norm,
               show_values=False):
    colors = cmap(norm(atom_vals))
    for ring in ring_highlight_rings(mol, target_atoms):
        poly = target_ring_polygon(coords, ring)
        ax.add_patch(Polygon(poly, closed=True, facecolor="#FFD700", alpha=0.15,
                             edgecolor="#FF8C00", linewidth=2.0, linestyle="--", zorder=1))
    # 键 (宽度正比于 |归因|)
    tset = set(target_atoms)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        si, sj = mol.GetAtomWithIdx(i).GetSymbol(), mol.GetAtomWithIdx(j).GetSymbol()
        if si == "H" or sj == "H":
            continue
        w = 0.5 * (abs(atom_vals[i]) + abs(atom_vals[j]))
        wrel = float(w / max(abs(atom_vals).max(), 1e-12))
        ax.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]],
                color=(0.35, 0.35, 0.35), linewidth=0.8 + 3.5 * wrel, zorder=2)
    # 原子
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        sym = atom.GetSymbol()
        if sym == "H":
            continue
        x, y = coords[i]
        r = 0.24 if sym == "C" else 0.30
        ax.add_patch(Circle((x, y), r, facecolor=colors[i], edgecolor="black",
                            linewidth=1.1, zorder=3))
        in_ring = i in tset
        if sym != "C":
            cpk = CPK_COLORS.get(sym, "#888")
            ax.add_patch(Circle((x, y), r + 0.05, facecolor="none",
                                edgecolor=cpk, linewidth=2.2, zorder=2.5))
        if in_ring:
            ax.text(x, y, sym if sym != "C" else "", ha="center", va="center",
                    fontsize=9, fontweight="bold", zorder=4)
        else:
            # 取代基: 标注元素 + 归因数值
            lab = sym if sym != "C" else "C"
            ax.text(x, y - r - 0.34, lab, ha="center", va="center", fontsize=9,
                    fontweight="bold", zorder=4)
            if show_values:
                ax.text(x, y + r + 0.30, f"{atom_vals[i]:+.3f}", ha="center",
                        va="center", fontsize=7.5, color="#111111", zorder=4)
    ax.set_aspect("equal")
    ax.axis("off")
    p = 1.3
    ax.set_xlim(coords[:, 0].min() - p, coords[:, 0].max() + p)
    ax.set_ylim(coords[:, 1].min() - p, coords[:, 1].max() + p)


def compute_ring_sub_stats(mol, atom_vals, target_atoms):
    ring, sub = [], []
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        if atom.GetSymbol() == "H":
            continue
        v = abs(float(atom_vals[i]))
        (ring if i in set(target_atoms) else sub).append(v)
    return {"ring_mean": float(np.mean(ring)) if ring else 0,
            "sub_mean": float(np.mean(sub)) if sub else 0,
            "ring_over_sub": (float(np.mean(ring) / (np.mean(sub) + 1e-10))
                              if ring and sub else np.inf)}


PRIORITY_SUBS = ["Br", "Cl", "F", "I", "vinyl", "Et", "Me",
                 "OMe", "OH", "NH2", "NMe2",
                 "SO2CF3", "CF3", "NCO", "CONH2", "COOMe", "COOH", "COMe", "CN",
                 "SMe", "SH", "SiMe3"]


def select_examples(pred_df, n=N_SELECT):
    pred_df = pred_df.copy()
    pred_df["atom_on_ring"] = pred_df["atom_on_ring"].apply(parse_aor)
    ex = []
    for _, r in pred_df.iterrows():
        mol = Chem.AddHs(Chem.MolFromSmiles(r["smiles"]))
        tset = set(r["atom_on_ring"])
        n_out = sum(1 for a in mol.GetAtoms()
                    if a.GetSymbol() != "H" and a.GetIdx() not in tset)
        ex.append({"ring_name": r["ring_name"], "sub_name": str(r["sub_name"]),
                   "sub_type": r["sub_type"], "y_true": r["y_true"],
                   "y_pred": r["y_pred"], "atom_on_ring": r["atom_on_ring"],
                   "smiles": r["smiles"], "mol": mol, "n_out": n_out,
                   "n_ring": len(tset)})
    ex = pd.DataFrame(ex)
    ex = ex[ex["n_out"] > 0].reset_index(drop=True)
    available = sorted(ex["sub_name"].unique())
    chosen_subs, chosen = [], []
    for pat in PRIORITY_SUBS:
        if len(chosen_subs) >= n:
            break
        hit = [s for s in available if pat.lower() in s.lower() and s not in chosen_subs]
        if hit:
            s = hit[0]
            chosen_subs.append(s)
            sub = ex[ex["sub_name"] == s].sort_values(["n_ring", "n_out"])
            chosen.append(sub.iloc[0])
    if len(chosen) < n:
        used = set(chosen_subs)
        rest = ex[~ex["sub_name"].isin(used)].sort_values(["n_ring", "n_out"])
        for _, r in rest.iterrows():
            if len(chosen) >= n:
                break
            chosen.append(r)
    return pd.DataFrame(chosen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="HOMA", choices=list(TASKS))
    ap.add_argument("--gpu", type=int, default=2)
    args = ap.parse_args()
    task = args.task
    pred_csv, ckpt, out_sub, disp_name, sign_flip = TASKS[task]
    pred_csv = os.path.join(ROOT, pred_csv)
    ckpt = os.path.join(ROOT, ckpt)
    out_root = os.path.join(ROOT, out_sub)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_root, exist_ok=True)
    print(f"[task={task}] Device: {device}  Out: {out_root}  (IG display sign flip={sign_flip})")

    pred = pd.read_csv(pred_csv)
    examples = select_examples(pred)
    print("选中示例:")
    for _, e in examples.iterrows():
        print(f"  {e['ring_name']:14s} sub={str(e['sub_name']):10s} "
              f"true={e['y_true']:.3f} pred={e['y_pred']:.3f}")

    model = load_model(device, ckpt)
    rows = []
    for idx, e in examples.iterrows():
        mol = e["mol"]
        aor = [int(x) for x in e["atom_on_ring"]]
        g = build_graph(e["smiles"], aor, NODE_VEC_LEN, MAX_ATOMS, ring_flag_value=1)
        node = torch.tensor(g["node_mat"][None], dtype=torch.float32, device=device)
        adj = torch.tensor(g["adj_mat"][None], dtype=torch.float32, device=device)
        ri = torch.tensor(g["ring_indices"][None], dtype=torch.long, device=device)

        ig = integrated_gradients(model, node, adj, ri) * sign_flip  # 显示方向对齐芳香性
        gi = grad_input_saliency(model, node, adj, ri)
        n_atoms = mol.GetNumAtoms()
        ig_a, gi_a = ig[:n_atoms], gi[:n_atoms]
        coords = get_2d_coords(mol)

        m = max(abs(ig_a).max(), 1e-9)
        ig_norm = SignedPowNorm(vmin=-m, vmax=m)
        gi_norm = PowNorm(vmin=gi_a.min(), vmax=gi_a.max())

        fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), dpi=150)
        draw_panel(axes[0], mol, coords, ig_a, aor, plt.get_cmap("RdBu_r"),
                   ig_norm, show_values=True)
        draw_panel(axes[1], mol, coords, gi_a, aor, plt.get_cmap("magma"), gi_norm)
        for ax_, cmap_, norm_, label in [
                (axes[0], "RdBu_r", ig_norm, f"IG (红=增强芳香性 · 蓝=减弱 · {disp_name})"),
                (axes[1], "magma", gi_norm, "Saliency |x·∇x|")]:
            sm = ScalarMappable(cmap=cmap_, norm=norm_)
            sm.set_array([])
            cb = plt.colorbar(sm, ax=ax_, fraction=0.045, pad=0.02)
            cb.set_label(label, fontsize=9)
        axes[0].set_title(f"Integrated Gradients (signed)\n{disp_name}  "
                          f"true={e['y_true']:.3f}  pred={e['y_pred']:.3f}", fontsize=11)
        axes[1].set_title(f"Input×Gradient |x·∇x| (对照)\n{disp_name}  "
                          f"true={e['y_true']:.3f}  pred={e['y_pred']:.3f}", fontsize=11)
        fig.suptitle(f"{disp_name} · {e['ring_name']} · 取代基 {e['sub_name']} ({e['sub_type']})\n"
                     "目标环=金框虚线 · 取代基原子下标注元素/归因值 · 颜色经幂次拉伸",
                     fontsize=12, fontweight="bold")
        save = os.path.join(out_root, f"{idx:02d}_{e['ring_name']}_{e['sub_name']}.png")
        plt.tight_layout()
        plt.savefig(save, bbox_inches="tight", dpi=150, facecolor="white")
        plt.close()

        st_ig = compute_ring_sub_stats(mol, ig_a, aor)
        st_gi = compute_ring_sub_stats(mol, gi_a, aor)
        rows.append({"task": task, "ring_name": e["ring_name"], "sub_name": e["sub_name"],
                     "sub_type": e["sub_type"], "y_true": e["y_true"],
                     "y_pred": e["y_pred"],
                     "ig_ring_mean": st_ig["ring_mean"], "ig_sub_mean": st_ig["sub_mean"],
                     "ig_ring_over_sub": st_ig["ring_over_sub"],
                     "gi_ring_mean": st_gi["ring_mean"], "gi_sub_mean": st_gi["sub_mean"],
                     "gi_ring_over_sub": st_gi["ring_over_sub"]})
        print(f"[saved] {save}")

    stats_csv = os.path.join(out_root, f"{task}_l10val_attribution_stats.csv")
    pd.DataFrame(rows).to_csv(stats_csv, index=False)
    print(f"[saved] {stats_csv}")


if __name__ == "__main__":
    main()
