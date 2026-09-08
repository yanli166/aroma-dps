"""
当前最优模型 (Stage 6 最终方案) 注意力/显著性可视化
====================================================
复刻 gnn_label_complete 可解释性研究的方法, 应用于 831-end-code 六阶段流水线的最终模型:

  Stage 6 最终 membership 方案 (best_model_package, seed=11):
    - HOMA:    MPNN + fixed_avg, ring_flag=1 + learnable projection (nn.Embedding)
    - NICS_1zz: MPNN + fixed_avg, ring_flag=1
    - MBCO:    MPNN + fixed_avg, ring_flag=1

模型为 MPNN (GRU-style message passing, 无原生注意力), 故使用与参考工作中 GNN/GIN
一致的基于梯度的 saliency map:  saliency_i = |x_i * dy/dx_i|.sum(features).

回答核心化学可解释性问题:
  模型预测目标环芳香性 (HOMA/NICS/MBCO) 时, 关注的是"环内共轭原子"还是"取代基侧链"?

测试分子: 与参考研究一致的 5 类代表性分子
  - 多卤代苯酚 (取代苯, 给/吸电子基混合)
  - 烷基氨基苯 (苯环 + 脂肪链侧链)
  - 吡啶衍生物 (6元含N杂环)
  - 噻吩衍生物 (5元含S杂环)
  - 取代萘 (双环融合芳香体系)
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')

from matplotlib import font_manager
for fpath in [
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
]:
    if os.path.exists(fpath):
        try:
            font_manager.fontManager.addfont(fpath)
        except Exception:
            pass
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Sans CJK JP',
                                          'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from rdkit import Chem
from rdkit.Chem import AllChem

# ============ 路径 (merged 重训模型) ============
PKG_DIR = '/home/ubuntu/aroma-dps-code/best_model_package'
MERGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PKG = os.path.join(MERGE_ROOT, 'models')
OUTPUT_ROOT = os.path.join(MERGE_ROOT, 'attention_viz')
sys.path.insert(0, PKG_DIR)

from graph_utils import build_graph, NODE_VEC_LEN, MAX_ATOMS   # noqa: E402
from model_arch import build_model                              # noqa: E402

# merged 重训模型 (base a/b + lunci10, Stage 6 配置)
MODEL_FILES = {
    'HOMA':     'HOMA_merged_best.pt',
    'NICS_1zz': 'NICS_1zz_merged_best.pt',
    'MBCO':     'MBCO_merged_best.pt',
}
MODEL_TAG = 'Merged 重训 (collet a/b + lunci10)'
TASK_LABELS = {
    'HOMA': 'HOMA', 'NICS_1zz': 'NICS(1)zz', 'MBCO': 'MBCO',
}
TASK_COLORS = {'HOMA': '#E76F51', 'NICS_1zz': '#3568C0', 'MBCO': '#2A9D8F'}

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 参考真实数据 CSV (用于显示 true 值)
DATA_DIR = '/home/ubuntu/aroma-dps-code/code_end/data1_end'
TASK_CSV = {
    'HOMA': os.path.join(DATA_DIR, 'collet_homa_0716.csv'),
    'NICS_1zz': os.path.join(DATA_DIR, 'collet_nics_0716.csv'),
    'MBCO': os.path.join(DATA_DIR, 'collet_mbco_0716.csv'),
}
TASK_COL = {'HOMA': 'homa_value', 'NICS_1zz': 'NICS_value', 'MBCO': 'mbco_value'}

CPK_COLORS = {
    'C': '#303030', 'H': '#FFFFFF', 'O': '#FF0D0D', 'N': '#3050F8',
    'S': '#DAA520', 'F': '#90E050', 'Cl': '#1FF01F', 'Br': '#A62929',
    'I': '#940094', 'P': '#FF8000', 'B': '#FFB5B5',
}

# 5 类代表性分子 (atom_on_ring 取自当前 0716 数据, 0-based 与 AddHs 后原子索引对齐)
TEST_MOLECULES = [
    {
        'name': 'phenol_polyhalogen',
        'smiles': 'Oc1cc(F)c(Br)cc1Cl',
        'atom_on_ring': [1, 8, 7, 5, 3, 2],
        'category': 'substituted_benzene',
        'description': '多卤代苯酚 (OH/F/Br/Cl)',
        'chem_note': '苯环为芳香共轭体系, OH为给电子基, F/Br/Cl为吸电子基',
    },
    {
        'name': 'alkyl_aminobenzene',
        'smiles': 'Cc1ccc(F)cc1[C@@H](C)N',
        'atom_on_ring': [1, 7, 6, 4, 3, 2],
        'category': 'substituted_benzene',
        'description': '烷基胺取代苯 (CH3/F/CH(CH3)NH2)',
        'chem_note': '苯环+多个脂肪链取代基, 验证模型不被侧链分散注意',
    },
    {
        'name': 'pyridine_deriv',
        'smiles': 'CC(=O)c1nccc(C)c1Cl',
        'atom_on_ring': [3, 9, 7, 6, 5, 4],
        'category': 'pyridine',
        'description': '吡啶衍生物 (6元环含1个N)',
        'chem_note': 'N电负性强, 拉低环上π电子密度',
    },
    {
        'name': 'thiophene_deriv',
        'smiles': 'CC[C@H](NCc1ccsc1C)C(=O)O',
        'atom_on_ring': [5, 6, 7, 8, 9],
        'category': 'thiophene',
        'description': '噻吩衍生物 (5元环含S)',
        'chem_note': 'S孤对电子参与芳香6π体系, 富电子杂环',
    },
    {
        'name': 'naphthalene_deriv',
        'smiles': 'Cc1cc(Br)c2ccccc2c1',
        'atom_on_ring': [1, 11, 10, 5, 3, 2],
        'category': 'naphthalene',
        'description': '取代萘 (双环6+6融合芳香体系)',
        'chem_note': '两个芳环共享一条边',
    },
]


def load_models():
    """加载 Stage 6 三任务最终模型"""
    models = {}
    configs = {}
    for task, fname in MODEL_FILES.items():
        ckpt = torch.load(os.path.join(MODEL_PKG, fname), map_location='cpu')
        cfg = ckpt['config']
        model = build_model(
            use_projection=cfg['use_projection'],
            node_vec_len=cfg['node_vec_len'],
            hidden_dim=cfg['hidden_dim'],
            n_conv=cfg['n_conv'],
            n_hidden=cfg['n_hidden'],
            p_dropout=cfg['p_dropout'],
            ring_flag_value=cfg['ring_flag_value'],
        )
        model.load_state_dict(ckpt['state_dict'])
        model.to(DEVICE)
        model.eval()
        models[task] = model
        configs[task] = cfg
        print(f"  [load] {task}: use_projection={cfg['use_projection']}, "
              f"ring_flag={cfg['ring_flag_value']}, hidden={cfg['hidden_dim']}, "
              f"n_conv={cfg['n_conv']}, n_hidden={cfg['n_hidden']}")
    return models, configs


def load_true_value(task, smiles, atom_on_ring):
    """从对应任务 CSV 读取该分子目标环的真实芳香性值; collet 找不到时回退到 merged 数据"""
    df = pd.read_csv(TASK_CSV[task])
    col = TASK_COL[task]
    aor = atom_on_ring
    for _, r in df.iterrows():
        csv_smi = str(r['smiles']).strip()
        csv_aor = eval(r['atom_on_ring']) if isinstance(r['atom_on_ring'], str) else r['atom_on_ring']
        if csv_smi == smiles and list(csv_aor) == aor:
            return float(r[col])
    # 回退: merged 数据 (collet a/b + lunci10)
    merged_csv = os.path.join(MERGE_ROOT, 'data', f'merged_{task}.csv')
    if os.path.exists(merged_csv):
        mdf = pd.read_csv(merged_csv)
        for _, r in mdf.iterrows():
            csv_smi = str(r['smiles']).strip()
            csv_aor = eval(r['atom_on_ring']) if isinstance(r['atom_on_ring'], str) else r['atom_on_ring']
            if csv_smi == smiles and list(csv_aor) == aor:
                return float(r['y'])
    return np.nan


def extract_saliency(model, node_mat, adj_mat, ring_indices):
    """saliency_i = |x_i * dy/dx_i|.sum(features)"""
    x = node_mat.clone().detach().requires_grad_(True)
    model.zero_grad()
    out = model(x, adj_mat, ring_indices).squeeze()
    out.backward()
    saliency = (x * x.grad).abs().sum(dim=-1).squeeze(0).detach().cpu().numpy()
    x.requires_grad_(False)
    return saliency, out.item()


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
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return pts[np.argsort(angles)]


def find_rdkit_target_ring(mol, target_atoms):
    """在 AddHs 分子上找到与 target_atoms 完全匹配的 RDKit 环 (0-based)"""
    ri = mol.GetRingInfo()
    tset = set(target_atoms)
    for ring in ri.AtomRings():
        if set(ring) == tset:
            return ring
    # 退而求其次: 重叠度最高的环
    best, best_overlap = None, 0
    for ring in ri.AtomRings():
        ov = len(set(ring) & tset)
        if ov > best_overlap:
            best, best_overlap = ring, ov
    return best


def draw_molecule(task, mol_info, atom_weights, pred, true_val, save_path):
    """绘制单个分子注意力图: 原子按 saliency 着色 + 目标环高亮"""
    mol = Chem.AddHs(Chem.MolFromSmiles(mol_info['smiles']))
    n_atoms_all = mol.GetNumAtoms()
    coords = get_2d_coords(mol)

    # 只取真实原子 (含 H), H 不画但保留索引用于 colormap 归一化对齐
    heavy_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() != 'H']
    n_heavy = len(heavy_idx)
    w_all = atom_weights[:n_atoms_all]

    # 在重原子上归一化
    w_min, w_max = w_all[heavy_idx].min(), w_all[heavy_idx].max()
    if w_max - w_min < 1e-10:
        w_norm = np.zeros_like(w_all)
    else:
        w_norm = (w_all - w_min) / (w_max - w_min)

    cmap = plt.get_cmap('magma')

    fig, ax = plt.subplots(figsize=(9, 8), dpi=150)
    pad = 1.3
    ax.set_xlim(coords[:, 0].min() - pad, coords[:, 0].max() + pad)
    ax.set_ylim(coords[:, 1].min() - pad, coords[:, 1].max() + pad)
    ax.set_aspect('equal')
    ax.axis('off')

    # 目标环高亮 (金色半透明, 高优先级)
    tset = set(mol_info['atom_on_ring'])
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        rset = set(ring)
        if len(rset & tset) >= 3:  # 目标环
            poly_pts = target_ring_polygon(coords, ring)
            ax.add_patch(Polygon(poly_pts, closed=True, facecolor='#FFD700',
                                 alpha=0.18, edgecolor='#FF8C00',
                                 linewidth=2.5, linestyle='--', zorder=1))

    # 键 (粗细 = 端点 saliency 平均)
    bond_lines, bond_colors, bond_lws = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        si, sj = mol.GetAtomWithIdx(i).GetSymbol(), mol.GetAtomWithIdx(j).GetSymbol()
        if si == 'H' or sj == 'H':
            continue
        bw = 0.5 * (w_norm[i] + w_norm[j])
        lw = 1.0 + 5.0 * bw
        bond_lines.append([(coords[i, 0], coords[i, 1]), (coords[j, 0], coords[j, 1])])
        bond_colors.append(cmap(bw))
        bond_lws.append(lw)
    if bond_lines:
        ax.add_collection(LineCollection(bond_lines, colors=bond_colors,
                                         linewidths=bond_lws, alpha=0.85,
                                         zorder=2, capstyle='round'))

    # 原子
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        sym = atom.GetSymbol()
        if sym == 'H':
            continue
        x, y = coords[i]
        w = w_norm[i]
        radius = 0.22 if sym == 'C' else 0.28
        ax.add_patch(Circle((x, y), radius, facecolor=cmap(w),
                            edgecolor='black', linewidth=1.2, zorder=3))
        if sym != 'C':
            cpk = CPK_COLORS.get(sym, '#888888')
            ax.add_patch(Circle((x, y), radius + 0.06, facecolor='none',
                                edgecolor=cpk, linewidth=2.5, zorder=2.5))
            ax.text(x, y, sym, ha='center', va='center', fontsize=10,
                    fontweight='bold', color='white' if w > 0.5 else 'black', zorder=4)
        else:
            ax.text(x, y, f'{w:.2f}', ha='center', va='center', fontsize=7,
                    color='white' if w > 0.5 else 'black', zorder=4)

    sm = ScalarMappable(cmap='magma', norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label('Saliency (normalized)', fontsize=10)

    true_str = f'{true_val:.3f}' if not np.isnan(true_val) else 'N/A'
    cfg_note = ' (ring_flag=1+proj)' if task == 'HOMA' else ' (ring_flag=1)'
    ax.set_title(f'{TASK_LABELS[task]} · MPNN{f" + projection" if task=="HOMA" else ""}'
                 f'{cfg_note}\n'
                 f'{mol_info["description"]}\n'
                 f'Pred={pred:.3f}   True={true_str}',
                 fontsize=11, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='white')
    plt.close()


def compute_stats(mol, atom_weights, target_atoms):
    """原子级统计: 环内 vs 取代基 (跳过 H)"""
    ring_w, sub_w = [], []
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        if atom.GetSymbol() == 'H':
            continue
        w = float(atom_weights[i])
        if i in set(target_atoms):
            ring_w.append(w)
        else:
            sub_w.append(w)
    stats = {
        'n_ring_atoms': len(ring_w),
        'n_sub_atoms': len(sub_w),
        'ring_mean': float(np.mean(ring_w)) if ring_w else 0,
        'ring_std': float(np.std(ring_w)) if ring_w else 0,
        'sub_mean': float(np.mean(sub_w)) if sub_w else 0,
        'sub_std': float(np.std(sub_w)) if sub_w else 0,
        'ratio_ring_over_sub': (float(np.mean(ring_w) / (np.mean(sub_w) + 1e-10))
                                if ring_w and sub_w else 0),
    }
    # 边级 (proxy = 端点 saliency 平均)
    rr, rs, ss = [], [], []
    tset = set(target_atoms)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        si, sj = mol.GetAtomWithIdx(i).GetSymbol(), mol.GetAtomWithIdx(j).GetSymbol()
        if si == 'H' or sj == 'H':
            continue
        w = 0.5 * (float(atom_weights[i]) + float(atom_weights[j]))
        i_in, j_in = i in tset, j in tset
        if i_in and j_in:
            rr.append(w)
        elif i_in != j_in:
            rs.append(w)
        else:
            ss.append(w)
    stats.update({
        'rr_edge_mean': float(np.mean(rr)) if rr else 0,
        'rr_edge_n': len(rr),
        'rs_edge_mean': float(np.mean(rs)) if rs else 0,
        'rs_edge_n': len(rs),
        'ss_edge_mean': float(np.mean(ss)) if ss else 0,
        'ss_edge_n': len(ss),
        'rr_over_rs': (float(np.mean(rr) / (np.mean(rs) + 1e-10))
                       if rr and rs else 0),
    })
    return stats


def make_single_category_comparison(task, model, save_path):
    """同模型 5 类分子注意力对比 panel (简化版, 无单个分子那么详细)"""
    fig, axes = plt.subplots(1, len(TEST_MOLECULES),
                             figsize=(5.2 * len(TEST_MOLECULES), 5.2), dpi=120)
    sm = ScalarMappable(cmap='magma', norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])

    for ax, mol_info in zip(axes, TEST_MOLECULES):
        g = build_graph(mol_info['smiles'], mol_info['atom_on_ring'],
                        NODE_VEC_LEN, MAX_ATOMS, ring_flag_value=1)
        node = torch.tensor(g['node_mat'][None], dtype=torch.float32, device=DEVICE)
        adj = torch.tensor(g['adj_mat'][None], dtype=torch.float32, device=DEVICE)
        ri_ = torch.tensor(g['ring_indices'][None], dtype=torch.long, device=DEVICE)
        sal, pred = extract_saliency(model, node, adj, ri_)

        mol = Chem.AddHs(Chem.MolFromSmiles(mol_info['smiles']))
        coords = get_2d_coords(mol)
        heavy_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() != 'H']
        w = sal[:mol.GetNumAtoms()]
        wmin, wmax = w[heavy_idx].min(), w[heavy_idx].max()
        w_norm = (w - wmin) / (wmax - wmin + 1e-10)

        tset = set(mol_info['atom_on_ring'])
        ri2 = mol.GetRingInfo()
        for ring in ri2.AtomRings():
            if len(set(ring) & tset) >= 3:
                poly_pts = target_ring_polygon(coords, ring)
                ax.add_patch(Polygon(poly_pts, closed=True, facecolor='#FFD700',
                                     alpha=0.15, edgecolor='#FF8C00',
                                     linewidth=1.8, linestyle='--', zorder=1))
        bond_lines, colors_, lws = [], [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if mol.GetAtomWithIdx(i).GetSymbol() == 'H' or mol.GetAtomWithIdx(j).GetSymbol() == 'H':
                continue
            bw = 0.5 * (w_norm[i] + w_norm[j])
            bond_lines.append([(coords[i, 0], coords[i, 1]), (coords[j, 0], coords[j, 1])])
            colors_.append(plt.get_cmap('magma')(bw))
            lws.append(1.0 + 4.5 * bw)
        if bond_lines:
            ax.add_collection(LineCollection(bond_lines, colors=colors_,
                                             linewidths=lws, alpha=0.85,
                                             zorder=2, capstyle='round'))
        for atom in mol.GetAtoms():
            i = atom.GetIdx()
            sym = atom.GetSymbol()
            if sym == 'H':
                continue
            x, y = coords[i]
            ww = w_norm[i]
            r = 0.22 if sym == 'C' else 0.28
            ax.add_patch(Circle((x, y), r, facecolor=plt.get_cmap('magma')(ww),
                                edgecolor='black', linewidth=1, zorder=3))
            if sym != 'C':
                cpk = CPK_COLORS.get(sym, '#888')
                ax.add_patch(Circle((x, y), r + 0.06, facecolor='none',
                                    edgecolor=cpk, linewidth=2, zorder=2.5))
                ax.text(x, y, sym, ha='center', va='center', fontsize=9,
                        fontweight='bold',
                        color='white' if ww > 0.5 else 'black', zorder=4)
        stats = compute_stats(mol, sal, mol_info['atom_on_ring'])
        ax.set_title(f"{mol_info['description']}\n"
                     f"环/取代基 = {stats['ratio_ring_over_sub']:.2f}\n"
                     f"Pred={pred:.3f}", fontsize=9)
        ax.set_aspect('equal'); ax.axis('off')
        p = 1.1
        ax.set_xlim(coords[:, 0].min() - p, coords[:, 0].max() + p)
        ax.set_ylim(coords[:, 1].min() - p, coords[:, 1].max() + p)

    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.01)
    cbar.set_label('Saliency (normalized)', fontsize=10)
    fig.suptitle(f'{TASK_LABELS[task]} 模型 · MPNN · 5 类分子注意力对比 · {MODEL_TAG}\n'
                 f'(原子级 saliency: |x·∇x|, 金框=目标环)', fontsize=13, fontweight='bold')
    plt.savefig(save_path, bbox_inches='tight', dpi=120, facecolor='white')
    plt.close()


def make_3task_comparison(mol_info, models, save_path):
    """同一分子在 HOMA/NICS/MBCO 三个最终模型上的注意力对比"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), dpi=140)
    sm = ScalarMappable(cmap='magma', norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])

    for ax, (task, model) in zip(axes, models.items()):
        g = build_graph(mol_info['smiles'], mol_info['atom_on_ring'],
                        NODE_VEC_LEN, MAX_ATOMS, ring_flag_value=1)
        node = torch.tensor(g['node_mat'][None], dtype=torch.float32, device=DEVICE)
        adj = torch.tensor(g['adj_mat'][None], dtype=torch.float32, device=DEVICE)
        ri_ = torch.tensor(g['ring_indices'][None], dtype=torch.long, device=DEVICE)
        sal, pred = extract_saliency(model, node, adj, ri_)

        mol = Chem.AddHs(Chem.MolFromSmiles(mol_info['smiles']))
        coords = get_2d_coords(mol)
        heavy_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() != 'H']
        w = sal[:mol.GetNumAtoms()]
        wmin, wmax = w[heavy_idx].min(), w[heavy_idx].max()
        w_norm = (w - wmin) / (wmax - wmin + 1e-10)

        tset = set(mol_info['atom_on_ring'])
        ri2 = mol.GetRingInfo()
        for ring in ri2.AtomRings():
            if len(set(ring) & tset) >= 3:
                poly_pts = target_ring_polygon(coords, ring)
                ax.add_patch(Polygon(poly_pts, closed=True, facecolor='#FFD700',
                                     alpha=0.15, edgecolor='#FF8C00',
                                     linewidth=1.8, linestyle='--', zorder=1))
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if mol.GetAtomWithIdx(i).GetSymbol() == 'H' or mol.GetAtomWithIdx(j).GetSymbol() == 'H':
                continue
            bw = 0.5 * (w_norm[i] + w_norm[j])
            ax.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]],
                    color=plt.get_cmap('magma')(bw), linewidth=1.0 + 4.5 * bw,
                    alpha=0.85, zorder=2)
        for atom in mol.GetAtoms():
            i = atom.GetIdx()
            sym = atom.GetSymbol()
            if sym == 'H':
                continue
            x, y = coords[i]
            ww = w_norm[i]
            r = 0.22 if sym == 'C' else 0.28
            ax.add_patch(Circle((x, y), r, facecolor=plt.get_cmap('magma')(ww),
                                edgecolor='black', linewidth=1, zorder=3))
            if sym != 'C':
                cpk = CPK_COLORS.get(sym, '#888')
                ax.add_patch(Circle((x, y), r + 0.06, facecolor='none',
                                    edgecolor=cpk, linewidth=2, zorder=2.5))
                ax.text(x, y, sym, ha='center', va='center', fontsize=9,
                        fontweight='bold',
                        color='white' if ww > 0.5 else 'black', zorder=4)
        stats = compute_stats(mol, sal, mol_info['atom_on_ring'])
        proj_note = ' +proj' if task == 'HOMA' else ''
        ax.set_title(f'{TASK_LABELS[task]} (MPNN{proj_note})\n'
                     f'环/取代基={stats["ratio_ring_over_sub"]:.2f}   '
                     f'Pred={pred:.3f}', fontsize=10)
        ax.set_aspect('equal'); ax.axis('off')
        p = 1.1
        ax.set_xlim(coords[:, 0].min() - p, coords[:, 0].max() + p)
        ax.set_ylim(coords[:, 1].min() - p, coords[:, 1].max() + p)

    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.01)
    cbar.set_label('Saliency (normalized)', fontsize=10)
    fig.suptitle(f'三任务 Merged 模型注意力对比 — {mol_info["description"]} · {MODEL_TAG}\n'
                 f'SMILES: {mol_info["smiles"]}', fontsize=12, fontweight='bold')
    plt.savefig(save_path, bbox_inches='tight', dpi=140, facecolor='white')
    plt.close()


def make_summary_panel(all_stats, save_path):
    """统计对比面板: 原子级环内 vs 取代基 + 聚焦系数"""
    df = pd.DataFrame(all_stats)
    tasks = ['HOMA', 'NICS_1zz', 'MBCO']
    mols = list(df['molecule'].unique())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)

    ax = axes[0, 0]
    x = np.arange(len(mols)); width = 0.12
    for i, t in enumerate(tasks):
        sub = df[df['task'] == t].set_index('molecule').reindex(mols)
        ax.bar(x + (i - 0.5) * width * 3, sub['ring_mean'], width * 3,
               label=f'{TASK_LABELS[t]} 环内', color=TASK_COLORS[t], alpha=0.85,
               edgecolor='black', linewidth=0.4)
        ax.bar(x + (i - 0.5) * width * 3 + width * 1.5, sub['sub_mean'],
               width * 3, color=TASK_COLORS[t], alpha=0.45, hatch='//',
               edgecolor='black', linewidth=0.4,
               label=f'{TASK_LABELS[t]} 取代基')
    ax.set_xticks(x); ax.set_xticklabels(mols, fontsize=8, rotation=12)
    ax.set_ylabel('saliency 均值'); ax.set_title('原子级: 环内 vs 取代基')
    ax.legend(fontsize=7, ncol=3, loc='upper left'); ax.grid(axis='y', alpha=0.3)

    ax = axes[0, 1]
    for i, t in enumerate(tasks):
        sub = df[df['task'] == t].set_index('molecule').reindex(mols)
        ax.bar(x + (i - 0.5) * width * 3, sub['ratio_ring_over_sub'],
               width * 3, label=TASK_LABELS[t], color=TASK_COLORS[t], alpha=0.85,
               edgecolor='black', linewidth=0.4)
    ax.axhline(1, color='red', ls='--', alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(mols, fontsize=8, rotation=12)
    ax.set_ylabel('环内/取代基'); ax.set_title('原子级聚焦系数 (>1 关注环内)')
    ax.legend(fontsize=8, loc='upper right'); ax.grid(axis='y', alpha=0.3)

    ax = axes[1, 0]
    for i, t in enumerate(tasks):
        sub = df[df['task'] == t].set_index('molecule').reindex(mols)
        ax.bar(x + (i - 0.5) * width * 3, sub['rr_edge_mean'], width * 3,
               color=TASK_COLORS[t], alpha=0.85, edgecolor='black', linewidth=0.4,
               label=f'{TASK_LABELS[t]} 环-环')
        ax.bar(x + (i - 0.5) * width * 3 + width * 1.5, sub['rs_edge_mean'],
               width * 3, color=TASK_COLORS[t], alpha=0.45, hatch='//',
               edgecolor='black', linewidth=0.4, label=f'{TASK_LABELS[t]} 环-取代')
    ax.set_xticks(x); ax.set_xticklabels(mols, fontsize=8, rotation=12)
    ax.set_ylabel('边权重均值'); ax.set_title('边级: 环-环 vs 环-取代基')
    ax.legend(fontsize=7, ncol=2, loc='upper left'); ax.grid(axis='y', alpha=0.3)

    ax = axes[1, 1]
    for i, t in enumerate(tasks):
        sub = df[df['task'] == t].set_index('molecule').reindex(mols)
        ax.bar(x + (i - 0.5) * width * 3, sub['rr_over_rs'], width * 3,
               label=TASK_LABELS[t], color=TASK_COLORS[t], alpha=0.85,
               edgecolor='black', linewidth=0.4)
    ax.axhline(1, color='red', ls='--', alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(mols, fontsize=8, rotation=12)
    ax.set_ylabel('环-环/环-取代'); ax.set_title('边级聚焦系数 (>1 关注环内键)')
    ax.legend(fontsize=8, loc='upper right'); ax.grid(axis='y', alpha=0.3)

    fig.suptitle(f'Merged 重训模型可解释性 — 环内共轭 vs 取代基 (原子级 + 边级) · {MODEL_TAG}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='white')
    plt.close()


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    print(f"输出目录: {OUTPUT_ROOT}")
    print(f"Device: {DEVICE}")

    print("\n[1/4] 加载 Stage 6 最终模型...")
    models, _ = load_models()

    print("\n[2/4] 逐模型 × 逐分子 saliency 可视化 + 统计...")
    all_stats = []
    mol_cache = {}  # 缓存加氢分子

    for task, model in models.items():
        task_dir = os.path.join(OUTPUT_ROOT, task)
        os.makedirs(task_dir, exist_ok=True)
        print(f"\n--- {TASK_LABELS[task]} ---")
        for mol_info in TEST_MOLECULES:
            try:
                smiles = mol_info['smiles']
                aor = mol_info['atom_on_ring']
                if smiles not in mol_cache:
                    mol_cache[smiles] = Chem.AddHs(Chem.MolFromSmiles(smiles))

                g = build_graph(smiles, aor, NODE_VEC_LEN, MAX_ATOMS, ring_flag_value=1)
                node = torch.tensor(g['node_mat'][None], dtype=torch.float32, device=DEVICE)
                adj = torch.tensor(g['adj_mat'][None], dtype=torch.float32, device=DEVICE)
                ri_ = torch.tensor(g['ring_indices'][None], dtype=torch.long, device=DEVICE)

                sal, pred = extract_saliency(model, node, adj, ri_)
                mol = mol_cache[smiles]
                stats = compute_stats(mol, sal, aor)
                true_val = load_true_value(task, smiles, aor)

                stats.update({
                    'task': task, 'molecule': mol_info['name'],
                    'category': mol_info['category'],
                    'description': mol_info['description'],
                    'smiles': smiles, 'atom_on_ring': aor,
                    'pred': pred, 'true': true_val,
                    'method': 'saliency (|x·∇x|)',
                })
                all_stats.append(stats)

                save_path = os.path.join(task_dir, f"{mol_info['name']}.png")
                draw_molecule(task, mol_info, sal, pred, true_val, save_path)
                print(f"  {mol_info['name']:22s} ring={stats['ring_mean']:.4f} "
                      f"sub={stats['sub_mean']:.4f} "
                      f"ratio={stats['ratio_ring_over_sub']:.2f} pred={pred:.3f} "
                      f"true={true_val:.3f}")
            except Exception as e:
                print(f"  ERROR {mol_info['name']}: {e}")
                import traceback; traceback.print_exc()

    # 保存统计
    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv(os.path.join(OUTPUT_ROOT, 'attention_stats.csv'), index=False)
    print(f"\n统计表 -> {os.path.join(OUTPUT_ROOT, 'attention_stats.csv')}")

    print("\n[3/4] 生成对比图...")
    for task, model in models.items():
        path = os.path.join(OUTPUT_ROOT, f'{task}_category_comparison.png')
        make_single_category_comparison(task, model, path)
        print(f"  -> {path}")

    three_dir = os.path.join(OUTPUT_ROOT, 'three_task_comparison')
    os.makedirs(three_dir, exist_ok=True)
    for mol_info in TEST_MOLECULES:
        path = os.path.join(three_dir, f"{mol_info['name']}_3tasks.png")
        make_3task_comparison(mol_info, models, path)
        print(f"  -> {path}")

    summary_path = os.path.join(OUTPUT_ROOT, 'summary_panel.png')
    make_summary_panel(all_stats, summary_path)
    print(f"  -> {summary_path}")

    print("\n完成!")


if __name__ == '__main__':
    main()
