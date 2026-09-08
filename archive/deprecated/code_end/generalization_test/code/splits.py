"""
四种泛化验证拆分策略 (适配 0716 新数据):

1. scaffold_split: DeepChem 风格骨架拆分
2. ring_type_split: 按芳香环类型拆分
3. external_test: lunci6 集外测试
4. external_test: lunci78 集外测试
"""
import os
import sys
import numpy as np
import pandas as pd

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJ_ROOT, 'code_end'))

from common.tasks import EXTERNAL_TEST_FILES, LUNCI_COL_MAP
from generalization_test.code.ring_utils import (
    get_murcko_scaffold, identify_ring_type,
    get_scaffold_distribution, get_ring_type_distribution)


def scaffold_split(df, smiles_col='smiles', holdout_scaffold=None,
                   val_ratio=0.125, seed=42, min_test_size=30,
                   frac_train=0.8, frac_valid=0.1, frac_test=0.1,
                   random_scaffold=False):
    """DeepChem 风格骨架拆分"""
    from collections import defaultdict
    from rdkit.Chem.Scaffolds import MurckoScaffold

    smiles_list = df[smiles_col].tolist()
    N = len(smiles_list)

    scaffolds = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                smiles=smi, includeChirality=True)
        except Exception:
            scaffold = ''
        scaffolds[scaffold].append(i)

    scaffold_sets_list = list(scaffolds.values())
    if random_scaffold:
        rng = np.random.RandomState(seed)
        scaffold_sets = list(rng.permutation(np.array(scaffold_sets_list, dtype=object)))
    else:
        scaffold_sets = sorted(scaffold_sets_list, key=lambda x: (-len(x), x[0]))

    train_cutoff = frac_train * N
    valid_cutoff = (frac_train + frac_valid) * N
    train_idx, val_idx, test_idx = [], [], []
    for scaffold_set in scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(val_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                val_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(test_idx) & set(val_idx)) == 0

    if len(test_idx) < min_test_size:
        print(f"  警告: 测试集仅 {len(test_idx)} 样本 (< {min_test_size})")
        return None, None, None, None

    from collections import Counter
    test_scaffolds = [sk for sk, idx_list in scaffolds.items()
                      if any(i in set(test_idx) for i in idx_list)]
    n_test_scaffolds = len(test_scaffolds)
    holdout_desc = f"deepchem_{'random' if random_scaffold else 'sorted'}_{n_test_scaffolds}scaffolds"

    print(f"  DeepChem 骨架拆分: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    print(f"  test 集含 {n_test_scaffolds} 种不同骨架 (总计 {len(scaffolds)} 种)")

    return train_idx, val_idx, test_idx, holdout_desc


def ring_type_split(df, smiles_col='smiles', holdout_ring_type=None,
                    val_ratio=0.125, seed=42, min_test_size=30):
    """按芳香环类型拆分: 指定环类型仅作测试集

    修复: 测试 furan 时, benzofuran 等含 furan 子结构的分子也排除出训练集
    (基于 SMARTS 子结构匹配, 而非仅按分类名称排除)
    """
    from rdkit import Chem
    from generalization_test.code.ring_utils import RING_SMARTS

    smiles_list = df[smiles_col].tolist()
    ring_types = [identify_ring_type(s) for s in smiles_list]

    if holdout_ring_type is None:
        from collections import Counter
        dist = Counter(ring_types)
        valid = {k: v for k, v in dist.items()
                 if k not in ('invalid', 'non_aromatic', 'unknown', '')}
        if not valid:
            return None, None, None, None
        holdout_ring_type = max(valid, key=valid.get)
        print(f"  自动选择最大环类型: {holdout_ring_type} ({valid[holdout_ring_type]} 样本)")

    # 获取 holdout 环类型的 SMARTS 模式
    holdout_smarts = None
    for name, smarts in RING_SMARTS:
        if name == holdout_ring_type:
            holdout_smarts = smarts
            break

    # 精确匹配 (分类名称 == holdout) 的作为测试集
    exact_mask = np.array([rt == holdout_ring_type for rt in ring_types])

    # 子结构匹配: 含有 holdout 环类型的分子也排除出训练集
    if holdout_smarts is not None:
        patt = Chem.MolFromSmarts(holdout_smarts)
        if patt is not None:
            substruct_mask = np.array([
                (lambda m: m is not None and m.HasSubstructMatch(patt))(Chem.MolFromSmiles(s))
                for s in smiles_list
            ])
            # 精确匹配 = 测试集; 子结构匹配但非精确匹配 = 也排除出训练集 (放入测试集)
            test_mask = exact_mask.copy()
            extra_mask = substruct_mask & ~exact_mask  # 含子结构但分类不同
            test_mask = test_mask | extra_mask
            n_extra = int(extra_mask.sum())
            if n_extra > 0:
                print(f"  子结构扩展: 额外将 {n_extra} 个含 {holdout_ring_type} 子结构的分子归入测试集")
        else:
            test_mask = exact_mask
    else:
        test_mask = exact_mask

    test_idx = np.where(test_mask)[0]
    trainval_idx = np.where(~test_mask)[0]

    if len(test_idx) < min_test_size:
        print(f"  警告: 环类型 '{holdout_ring_type}' 仅 {len(test_idx)} 样本 (< {min_test_size})")
        return None, None, None, None

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(trainval_idx))
    n_val = int(val_ratio * len(trainval_idx))
    val_idx = trainval_idx[perm[:n_val]]
    train_idx = trainval_idx[perm[n_val:]]

    print(f"  环类型拆分 [{holdout_ring_type}]: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist(), holdout_ring_type


def prepare_external_test_csv(task_name, test_source='lunci6', tmp_dir='/tmp'):
    """将 lunci6/lunci78 CSV 列名归一化, 生成与主数据集兼容的临时 CSV

    Args:
        task_name: 'HOMA' / 'NICS_1zz' / 'MBCO'
        test_source: 'lunci6' 或 'lunci78'

    Returns:
        (临时 CSV 路径, 目标列名)
    """
    csv_path = EXTERNAL_TEST_FILES[test_source]
    # 处理可能的 Windows 换行符
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    # 清理列名 (去除尾部空格和 \r)
    df.columns = df.columns.str.strip()
    df = df.rename(columns=LUNCI_COL_MAP)

    if task_name == 'HOMA':
        target_col = 'homa_value'
    elif task_name == 'NICS_1zz':
        target_col = 'NICS_value'
    elif task_name == 'MBCO':
        target_col = 'mbco_value'
    else:
        raise ValueError(f"未知任务: {task_name}")

    required = ['smiles', 'atom_on_ring', target_col, 'Ring_ID', 'Ring_Size']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"{test_source} 缺少列: {col} (现有: {df.columns.tolist()})")

    df = df.dropna(subset=['smiles', target_col]).reset_index(drop=True)

    tmp_path = os.path.join(tmp_dir, f'{test_source}_{task_name}_normalized.csv')
    df.to_csv(tmp_path, index=False)
    print(f"  {test_source} 归一化: {len(df)} 行 → {tmp_path} (目标列: {target_col})")
    return tmp_path, target_col


def report_distributions(df, smiles_col='smiles'):
    """报告数据集的骨架和环类型分布"""
    smiles_list = df[smiles_col].tolist()
    print("\n  === 骨架分布 (Top 10) ===")
    sc_dist = get_scaffold_distribution(smiles_list)
    for sc, cnt in sc_dist.most_common(10):
        print(f"    {sc}: {cnt}")
    print(f"  ... 共 {len(sc_dist)} 种骨架")

    print("\n  === 芳香环类型分布 (Top 15) ===")
    rt_dist = get_ring_type_distribution(smiles_list)
    for rt, cnt in rt_dist.most_common(15):
        print(f"    {rt}: {cnt}")
    print(f"  ... 共 {len(rt_dist)} 种环类型")

    return sc_dist, rt_dist
