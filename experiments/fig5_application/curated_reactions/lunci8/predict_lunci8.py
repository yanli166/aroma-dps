"""
lunci8 三任务芳香性预测脚本 (使用最优模型)

基于 baseline_gnn/results/all_gnn_summary.csv 中 test_r2 最高的模型:
  HOMA     -> MPNN      (test_r2 = 0.9888)
  NICS_1zz -> GNN       (test_r2 = 0.9758)
  MBCO     -> GraphSAGE (test_r2 = 0.9905)

模型权重:
  /home/ubuntu/aroma-dps-code/baseline_gnn/results/HOMA/MPNN/best_model.pth
  /home/ubuntu/aroma-dps-code/baseline_gnn/results/NICS_1zz/GNN/best_model.pth
  /home/ubuntu/aroma-dps-code/baseline_gnn/results/MBCO/GraphSAGE/best_model.pth

预测流程:
  1. 读取输入 CSV (含 SMILES 列)
  2. 对每个分子, 用 RDKit GetRingInfo().AtomRings() 枚举所有环
  3. 对每个环, 将该环原子作为 atom_on_ring (1-indexed, 与训练数据一致),
     使用原始 Graph 类生成节点矩阵 + 邻接矩阵 (label 编码, ring_flag_value=10)
  4. 送入 3 个最优模型预测 HOMA / NICS_1zz / MBCO
  5. 输出 CSV: 原始列 + ring_id, ring_atoms, ring_size, is_aromatic,
     HOMA_pred, NICS_1zz_pred, MBCO_pred (每个分子的每个环一行)

使用方法:
  # 默认预测 lunci8-begin.csv
  python predict_lunci8.py --gpu 0

  # 预测任意 CSV (必须含 SMILES 或 smiles 列)
  python predict_lunci8.py --input /path/to/input.csv \\
                           --output /path/to/output.csv --gpu 0

  # 仅预测芳香环 (与训练分布一致)
  python predict_lunci8.py --gpu 0 --aromatic_only
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

# 项目路径
PROJ_ROOT = '/home/ubuntu/aroma-dps-code'
ORIG_ROOT = '/home/ubuntu/data_90/alldata_in_3090/model1'
sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, ORIG_ROOT)

from unified_models.gat.model import GATModel
from unified_models.gin.model import GINModel
from unified_models.gnn.model import GNNModel
from unified_models.mpnn.model import MPNNModel
from unified_models.graphsage.model import GraphSAGEModel
from unified_models.common.graphs import Graph

# ============== 最优模型配置 (基于 all_gnn_summary.csv test_r2 最高) ==============
BEST_MODELS = {
    'HOMA': {
        'model_name': 'MPNN',
        'model_cls': MPNNModel,
        'weights': os.path.join(PROJ_ROOT, 'baseline_gnn/results/HOMA/MPNN/best_model.pth'),
        'pred_col': 'HOMA_pred',
    },
    'NICS_1zz': {
        'model_name': 'GNN',
        'model_cls': GNNModel,
        'weights': os.path.join(PROJ_ROOT, 'baseline_gnn/results/NICS_1zz/GNN/best_model.pth'),
        'pred_col': 'NICS_1zz_pred',
    },
    'MBCO': {
        'model_name': 'GraphSAGE',
        'model_cls': GraphSAGEModel,
        'weights': os.path.join(PROJ_ROOT, 'baseline_gnn/results/MBCO/GraphSAGE/best_model.pth'),
        'pred_col': 'MBCO_pred',
    },
}

# 与训练时完全一致的模型超参 (见 baseline_gnn/code/gnn_train_eval.py DEFAULT_PARAMS)
DEFAULT_PARAMS = {
    'hidden_dim': 128,
    'n_conv_layers': 3,
    'n_hidden_layers': 2,
    'p_dropout': 0.2,
}
NVL, MAX_ATOMS = 60, 75
RING_FLAG_VALUE = 10  # label 编码 (与第二层 GNN 基线训练一致)


# ============== 模型构建与加载 ==============
def build_model(task_name, device):
    """构建指定任务的最优模型并加载训练权重

    Args:
        task_name: 'HOMA' / 'NICS_1zz' / 'MBCO'
        device: torch.device

    Returns:
        加载好权重的模型 (eval 模式)
    """
    cfg = BEST_MODELS[task_name]
    kwargs = dict(
        node_vec_len=NVL,
        hidden_dim=DEFAULT_PARAMS['hidden_dim'],
        n_conv=DEFAULT_PARAMS['n_conv_layers'],
        n_hidden=DEFAULT_PARAMS['n_hidden_layers'],
        n_outputs=1,
        p_dropout=DEFAULT_PARAMS['p_dropout'],
        mode='label',  # label 编码 (与训练一致)
    )
    if cfg['model_name'] == 'GAT':
        kwargs['n_heads'] = 4
    model = cfg['model_cls'](**kwargs).to(device)
    state = torch.load(cfg['weights'], map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def load_all_models(device):
    """加载全部 3 个最优模型, 返回 {task_name: model} 字典"""
    models = {}
    for task in BEST_MODELS:
        cfg = BEST_MODELS[task]
        print(f"  {task:10s} -> {cfg['model_name']:10s} ({cfg['weights']})")
        models[task] = build_model(task, device)
    return models


# ============== 环枚举与图构建 ==============
def enumerate_rings(smiles, aromatic_only=False,
                    nvl=NVL, max_atoms=MAX_ATOMS,
                    ring_flag_value=RING_FLAG_VALUE):
    """枚举分子中的所有环, 为每个环生成图数据

    使用 RDKit GetRingInfo().AtomRings() 获取 SSSR (最小环集合),
    对每个环生成 Graph 对象 (label 编码, ring_flag_value=10).

    Args:
        smiles: SMILES 字符串
        aromatic_only: True 则仅保留芳香环 (所有环原子 GetIsAromatic()=True)
        nvl, max_atoms: 节点特征维度与最大原子数 (与训练一致)
        ring_flag_value: 目标环标记值 (10 = label 编码)

    Returns:
        list of dict, 每项含:
          - ring_atoms_1idx: list[int]  (1-indexed, 与训练数据 atom_on_ring 一致)
          - ring_size: int
          - is_aromatic: bool
          - node_mat: np.ndarray  (max_atoms, nvl)
          - adj_mat: np.ndarray   (max_atoms, max_atoms)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    mol_h = Chem.AddHs(mol)

    ring_info = mol_h.GetRingInfo()
    atom_rings = ring_info.AtomRings()  # 0-indexed (含 H)

    results = []
    for ring in atom_rings:
        # 跳过含氢原子的环 (训练数据 atom_on_ring 仅含重原子)
        if any(mol_h.GetAtomWithIdx(idx).GetAtomicNum() == 1 for idx in ring):
            continue

        is_arom = all(mol_h.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring)
        if aromatic_only and not is_arom:
            continue

        # 1-indexed (与训练数据 atom_on_ring 字段一致)
        atom_on_ring_1idx = [idx + 1 for idx in ring]
        try:
            g = Graph(smiles, atom_on_ring_1idx, nvl, max_atoms,
                      ring_flag_value=ring_flag_value)
            results.append({
                'ring_atoms_1idx': atom_on_ring_1idx,
                'ring_size': len(ring),
                'is_aromatic': is_arom,
                'node_mat': g.node_mat,
                'adj_mat': g.adj_mat,
            })
        except Exception as e:
            print(f"  警告: 环 {atom_on_ring_1idx} 图构建失败 ({smiles}): {e}")
            continue
    return results


# ============== 批量预测 ==============
@torch.no_grad()
def predict_batch(model, node_mats, adj_mats, device, batch_size=64):
    """批量预测单个任务

    Args:
        model: 已加载权重的 GNN 模型 (eval 模式)
        node_mats: list of np.ndarray
        adj_mats:  list of np.ndarray
        device: torch.device
        batch_size: 批大小

    Returns:
        np.ndarray, shape=(n_samples,) 的预测值
    """
    n = len(node_mats)
    if n == 0:
        return np.array([])
    preds = []
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        nm = torch.tensor(np.array(node_mats[i:end]), dtype=torch.float32, device=device)
        am = torch.tensor(np.array(adj_mats[i:end]), dtype=torch.float32, device=device)
        p = model(nm, am).squeeze(-1).cpu().numpy()
        preds.append(p)
    return np.concatenate(preds)


# ============== 主预测函数 ==============
def predict_csv(input_csv, output_csv, device, batch_size=64, aromatic_only=False):
    """对输入 CSV 中每个分子的每个环进行三任务预测

    输入 CSV 必须含 SMILES 或 smiles 列, 其他列原样保留。
    输出 CSV: 原始列 + ring_id, ring_atoms, ring_size, is_aromatic,
              HOMA_pred, NICS_1zz_pred, MBCO_pred (每个分子的每个环一行)

    Args:
        input_csv: 输入 CSV 路径
        output_csv: 输出 CSV 路径
        device: torch.device
        batch_size: 批大小
        aromatic_only: True 则仅预测芳香环

    Returns:
        pd.DataFrame 预测结果
    """
    df = pd.read_csv(input_csv)
    # 列名归一化 (允许 SMILES 或 smiles)
    if 'SMILES' in df.columns:
        smiles_col = 'SMILES'
    elif 'smiles' in df.columns:
        smiles_col = 'smiles'
    else:
        raise ValueError(f"输入 CSV 必须包含 SMILES 或 smiles 列, 现有列: {df.columns.tolist()}")

    print(f"\n加载 3 个最优模型 (label 编码, ring_flag_value={RING_FLAG_VALUE})...")
    models = load_all_models(device)

    print(f"\n开始预测: {len(df)} 个分子, aromatic_only={aromatic_only}")
    rows = []
    n_failed = 0
    for i, row in df.iterrows():
        smi = row[smiles_col]
        if pd.isna(smi) or not str(smi).strip():
            continue
        smi = str(smi).strip()

        ring_data = enumerate_rings(smi, aromatic_only=aromatic_only)
        if not ring_data:
            n_failed += 1
            label = row.get('no', row.get('New_ID', f'row{i}'))
            print(f"  [{i+1}/{len(df)}] {label}: 无可用环 (跳过)")
            continue

        node_mats = [r['node_mat'] for r in ring_data]
        adj_mats = [r['adj_mat'] for r in ring_data]

        # 三任务预测
        preds = {}
        for task in BEST_MODELS:
            preds[task] = predict_batch(models[task], node_mats, adj_mats,
                                        device, batch_size)

        label = row.get('no', row.get('New_ID', f'row{i}'))
        for ridx, r in enumerate(ring_data):
            new_row = row.to_dict()
            new_row['ring_id'] = ridx + 1
            new_row['ring_atoms'] = str(r['ring_atoms_1idx'])
            new_row['ring_size'] = r['ring_size']
            new_row['is_aromatic'] = int(r['is_aromatic'])
            new_row['HOMA_pred'] = float(preds['HOMA'][ridx])
            new_row['NICS_1zz_pred'] = float(preds['NICS_1zz'][ridx])
            new_row['MBCO_pred'] = float(preds['MBCO'][ridx])
            rows.append(new_row)

        if (i + 1) % 10 == 0 or i == len(df) - 1:
            print(f"  [{i+1}/{len(df)}] {label}: {len(ring_data)} 环 (累计 {len(rows)} 预测)")

    if not rows:
        print("\n警告: 无有效预测结果")
        return pd.DataFrame()

    out_df = pd.DataFrame(rows)
    # 列顺序: 原始列 + 预测列
    pred_cols = ['ring_id', 'ring_atoms', 'ring_size', 'is_aromatic',
                 'HOMA_pred', 'NICS_1zz_pred', 'MBCO_pred']
    base_cols = [c for c in df.columns if c in out_df.columns]
    out_df = out_df[base_cols + pred_cols]
    out_df.to_csv(output_csv, index=False)
    print(f"\n预测完成: {len(out_df)} 行 (跳过 {n_failed} 分子) → {output_csv}")
    return out_df


# ============== CLI 入口 ==============
def main():
    parser = argparse.ArgumentParser(
        description='lunci8 三任务芳香性预测 (使用最优模型 MPNN/GNN/GraphSAGE)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument('--input', type=str,
                        default='/home/ubuntu/aroma-dps-code/lunci8/lunci8-begin.csv',
                        help='输入 CSV 路径 (含 SMILES 列)')
    parser.add_argument('--output', type=str,
                        default='/home/ubuntu/aroma-dps-code/lunci8/lunci8-predicted.csv',
                        help='输出 CSV 路径')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--aromatic_only', action='store_true',
                        help='仅预测芳香环 (与训练分布更一致, 推荐)')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"输入: {args.input}")
    print(f"输出: {args.output}")

    predict_csv(args.input, args.output, device,
                batch_size=args.batch_size,
                aromatic_only=args.aromatic_only)


if __name__ == '__main__':
    main()
