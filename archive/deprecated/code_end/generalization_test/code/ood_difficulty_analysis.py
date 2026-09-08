
# --- Auto path bootstrap (do not remove) ---
import os as _os
_THIS_FILE = _os.path.abspath(__file__)
_d = _os.path.dirname(_THIS_FILE)
while not _os.path.exists(_os.path.join(_d, 'unified_models')) and _d != '/':
    _d = _os.path.dirname(_d)
_PROJ_ROOT = _d
# --- End auto path bootstrap ---

#!/usr/bin/env python3
"""
P1b-2: Residual OOD Difficulty Analysis

对每个 fraction 的测试集计算:
  1. mean / median nearest-neighbor Tanimoto similarity (to training set)
  2. <0.3 similarity 比例
  3. ring-type frequency in full dataset
  4. 每种 ring type 的样本数
  5. heteroatom composition
  6. ring size
  7. fused/non-fused
  8. HOMA target range / variance

然后画:
  - fraction vs mean test NN similarity
  - R²_HOMA vs test-set difficulty
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from collections import Counter
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

PROJ_ROOT = '_PROJ_ROOT + "/code_end"'
sys.path.insert(0, PROJ_ROOT)
from generalization_test.code.ring_utils import identify_ring_type

L10_CSV = '_PROJ_ROOT/lunci10/lunci10-expanded-test.csv'
MAIN_HOMA = '_PROJ_ROOT + "/code_end"/data1_end/collet_homa_0716.csv'
RESULTS_DIR = '_PROJ_ROOT + "/code_end"/results/learning_curve'
OUTPUT_DIR = '_PROJ_ROOT + "/code_end"/results/ood_difficulty_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
FRACTIONS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]


def smiles_to_fp(smi, radius=2, nbits=2048):
    """SMILES -> Morgan fingerprint"""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def compute_nn_similarity(test_smiles_list, train_smiles_list, sample_size=500):
    """计算测试集到训练集的最近邻 Tanimoto similarity"""
    # 训练集 fingerprints
    train_fps = []
    for smi in train_smiles_list:
        fp = smiles_to_fp(smi)
        if fp is not None:
            train_fps.append(fp)

    if not train_fps:
        return None, None, None

    # 测试集 fingerprints
    test_fps = []
    for smi in test_smiles_list:
        fp = smiles_to_fp(smi)
        if fp is not None:
            test_fps.append(fp)

    if not test_fps:
        return None, None, None

    # 如果测试集太大, 采样
    if len(test_fps) > sample_size:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(len(test_fps), sample_size, replace=False)
        test_fps_sample = [test_fps[i] for i in idx]
    else:
        test_fps_sample = test_fps

    # 计算每个测试分子到训练集的最近邻 similarity
    nn_sims = []
    for tfp in test_fps_sample:
        sims = DataStructs.BulkTanimotoSimilarity(tfp, train_fps)
        nn_sims.append(max(sims))

    nn_sims = np.array(nn_sims)
    return nn_sims.mean(), np.median(nn_sims), (nn_sims < 0.3).mean()


def get_heteroatom_composition(smiles):
    """获取杂原子组成"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    atoms = [a.GetSymbol() for a in mol.GetAtoms() if not a.GetSymbol() == 'C']
    return dict(Counter(atoms))


def is_fused_ring(smiles):
    """判断是否为稠合环"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    ri = mol.GetRingInfo()
    if len(ri.AtomRings()) < 2:
        return False
    # 检查是否有共享原子
    rings = ri.AtomRings()
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            if set(rings[i]) & set(rings[j]):
                return True
    return False


def get_ring_sizes(smiles):
    """获取所有环的大小"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    ri = mol.GetRingInfo()
    return [len(r) for r in ri.AtomRings()]


def analyze_fraction(task_name, fraction, df_l10_train, df_l10_test, df_main_smiles):
    """分析单个 fraction 的测试集难度"""
    test_smiles = df_l10_test['SMILES'].unique().tolist()
    # 安全处理空 df_l10_train
    if len(df_l10_train) > 0 and 'SMILES' in df_l10_train.columns:
        l10_train_smiles = df_l10_train['SMILES'].unique().tolist()
    else:
        l10_train_smiles = []
    train_smiles = list(set(df_main_smiles + l10_train_smiles))

    # 1. Tanimoto NN similarity
    mean_nn, median_nn, frac_low = compute_nn_similarity(test_smiles, train_smiles)

    # 2. Ring-type 统计
    test_ring_types = [identify_ring_type(s) for s in test_smiles]
    rt_counter = Counter(test_ring_types)

    # 3. Ring-type frequency in full dataset (lunci10 全集)
    all_ring_types = Counter()
    # 使用 lunci10 全数据的 SMILES (而非仅 train/test 子集)
    df_l10_full = pd.read_csv(L10_CSV)
    for smi in df_l10_full['SMILES'].unique():
        rt = identify_ring_type(smi)
        all_ring_types[rt] += 1

    # 测试集 ring types 在全数据集中的频率
    test_rt_freq_in_full = []
    for rt in rt_counter:
        test_rt_freq_in_full.append(all_ring_types[rt])

    # 4. Heteroatom composition
    hetero_compositions = []
    for smi in test_smiles[:100]:  # 采样
        hetero_compositions.append(get_heteroatom_composition(smi))

    # 5. Fused/non-fused
    fused_count = sum(1 for s in test_smiles if is_fused_ring(s))

    # 6. Ring sizes
    all_ring_sizes = []
    for smi in test_smiles:
        all_ring_sizes.extend(get_ring_sizes(smi))

    # 7. HOMA target range
    target_col = {'HOMA': 'HOMA', 'NICS_1zz': 'NICS_ZZ', 'MBCO': 'MBCO'}[task_name]
    targets = df_l10_test[target_col].dropna()

    result = {
        'task': task_name,
        'fraction': fraction,
        'n_test_smiles': len(test_smiles),
        'n_test_samples': len(df_l10_test),
        'n_train_smiles': len(train_smiles),
        'mean_nn_tanimoto': mean_nn,
        'median_nn_tanimoto': median_nn,
        'frac_below_0.3': frac_low,
        'n_test_ring_types': len(rt_counter),
        'mean_rt_freq_in_full': np.mean(test_rt_freq_in_full) if test_rt_freq_in_full else 0,
        'min_rt_freq_in_full': np.min(test_rt_freq_in_full) if test_rt_freq_in_full else 0,
        'fused_fraction': fused_count / len(test_smiles) if test_smiles else 0,
        'mean_ring_size': np.mean(all_ring_sizes) if all_ring_sizes else 0,
        'target_min': targets.min() if len(targets) > 0 else 0,
        'target_max': targets.max() if len(targets) > 0 else 0,
        'target_mean': targets.mean() if len(targets) > 0 else 0,
        'target_std': targets.std() if len(targets) > 0 else 0,
        'ring_types': dict(rt_counter),
    }

    return result


def prepare_l10_data(task_name, fraction):
    """复制 run_learning_curve_rerun.py 的分组逻辑"""
    df_l10 = pd.read_csv(L10_CSV)
    target_col = {'HOMA': 'HOMA', 'NICS_1zz': 'NICS_ZZ', 'MBCO': 'MBCO'}[task_name]
    df_l10 = df_l10.dropna(subset=[target_col]).reset_index(drop=True)

    smiles_list = sorted(df_l10['SMILES'].unique())
    smi_to_rt = {smi: identify_ring_type(smi) for smi in smiles_list}

    rt_to_smiles = {}
    for smi, rt in smi_to_rt.items():
        rt_to_smiles.setdefault(rt, []).append(smi)

    rt_list = sorted(rt_to_smiles.keys())
    rng = np.random.RandomState(SEED)
    rng.shuffle(rt_list)

    n_train_rt = int(len(rt_list) * fraction)
    train_rts = set(rt_list[:n_train_rt])
    test_rts = set(rt_list[n_train_rt:])

    train_smiles = set()
    test_smiles = set()
    for rt in train_rts:
        train_smiles.update(rt_to_smiles[rt])
    for rt in test_rts:
        test_smiles.update(rt_to_smiles[rt])

    df_train = df_l10[df_l10['SMILES'].isin(train_smiles)].reset_index(drop=True)
    df_test = df_l10[df_l10['SMILES'].isin(test_smiles)].reset_index(drop=True)

    return df_train, df_test, train_rts, test_rts


def main():
    print("=" * 60)
    print("P1b-2: Residual OOD Difficulty Analysis")
    print("=" * 60)

    # 加载主训练数据 SMILES
    df_main = pd.read_csv(MAIN_HOMA)
    df_main_smiles = df_main['smiles'].unique().tolist()
    print(f"主训练数据 SMILES 数: {len(df_main_smiles)}")

    # 0% 基线 (全部 lunci10 作为测试)
    df_l10 = pd.read_csv(L10_CSV).dropna(subset=['HOMA'])
    all_results = []

    # 分析 0% (无 lunci10 训练)
    print(f"\n--- Fraction=0% (all lunci10 as test) ---")
    result = analyze_fraction('HOMA', 0.0, pd.DataFrame(), df_l10, df_main_smiles)
    all_results.append(result)
    print(f"  mean_nn={result['mean_nn_tanimoto']:.4f}, median_nn={result['median_nn_tanimoto']:.4f}")

    # 分析各 fraction
    for frac in FRACTIONS:
        print(f"\n--- Fraction={frac*100:.0f}% ---")
        df_train, df_test, train_rts, test_rts = prepare_l10_data('HOMA', frac)
        result = analyze_fraction('HOMA', frac, df_train, df_test, df_main_smiles)
        all_results.append(result)
        print(f"  test_rts={len(test_rts)}, n_test_smiles={result['n_test_smiles']}")
        print(f"  mean_nn={result['mean_nn_tanimoto']:.4f}, median_nn={result['median_nn_tanimoto']:.4f}, "
              f"frac<0.3={result['frac_below_0.3']:.4f}")
        print(f"  target: mean={result['target_mean']:.4f}, std={result['target_std']:.4f}")

    # 保存结果
    df_results = pd.DataFrame([{k: v for k, v in r.items() if k != 'ring_types'} for r in all_results])
    csv_path = os.path.join(OUTPUT_DIR, 'ood_difficulty_analysis.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"\n保存: {csv_path}")

    # 保存 ring_types 详情
    rt_details = []
    for r in all_results:
        for rt, cnt in r['ring_types'].items():
            rt_details.append({
                'fraction': r['fraction'],
                'ring_type': rt,
                'count': cnt,
            })
    df_rt = pd.DataFrame(rt_details)
    rt_path = os.path.join(OUTPUT_DIR, 'test_ring_types_by_fraction.csv')
    df_rt.to_csv(rt_path, index=False)
    print(f"保存: {rt_path}")

    # 合并 R² 结果
    lc_csv = os.path.join(RESULTS_DIR, 'learning_curve_full.csv')
    if os.path.exists(lc_csv):
        df_lc = pd.read_csv(lc_csv)
        df_homa = df_lc[df_lc['task'] == 'HOMA'][['fraction', 'test_r2', 'test_mae']]
        df_results = df_results.merge(df_homa, on='fraction', how='left')

    # 画图
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=120)

    # 1. fraction vs mean NN similarity
    ax = axes[0, 0]
    ax.plot(df_results['fraction'] * 100, df_results['mean_nn_tanimoto'], 'o-',
            linewidth=2, markersize=8, color='steelblue', label='Mean NN Sim')
    ax.plot(df_results['fraction'] * 100, df_results['median_nn_tanimoto'], 's--',
            linewidth=2, markersize=8, color='coral', label='Median NN Sim')
    ax.set_xlabel('Fraction of Ring Types in Training (%)')
    ax.set_ylabel('Tanimoto Similarity to Training Set')
    ax.set_title('Test Set Chemical Distance vs Training Coverage')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. fraction vs frac<0.3
    ax = axes[0, 1]
    ax.plot(df_results['fraction'] * 100, df_results['frac_below_0.3'] * 100, 'o-',
            linewidth=2, markersize=8, color='red')
    ax.set_xlabel('Fraction of Ring Types in Training (%)')
    ax.set_ylabel('% Test Molecules with NN Sim < 0.3')
    ax.set_title('Fraction of OOD Test Molecules')
    ax.grid(True, alpha=0.3)

    # 3. R²_HOMA vs mean NN similarity
    ax = axes[1, 0]
    if 'test_r2' in df_results.columns:
        ax.scatter(df_results['mean_nn_tanimoto'], df_results['test_r2'],
                   s=100, c=df_results['fraction'] * 100, cmap='viridis', edgecolors='black')
        for _, row in df_results.iterrows():
            ax.annotate(f"{row['fraction']*100:.0f}%",
                        (row['mean_nn_tanimoto'], row['test_r2']),
                        textcoords="offset points", xytext=(5, 5), fontsize=9)
        ax.set_xlabel('Mean NN Tanimoto Similarity')
        ax.set_ylabel('HOMA Test R²')
        ax.set_title('R² vs Chemical Distance (color = % coverage)')
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)

    # 4. fraction vs n_test_ring_types + R²
    ax = axes[1, 1]
    ax2 = ax.twinx()
    ax.plot(df_results['fraction'] * 100, df_results['n_test_ring_types'], 'o-',
            linewidth=2, markersize=8, color='green', label='# Test Ring Types')
    if 'test_r2' in df_results.columns:
        ax2.plot(df_results['fraction'] * 100, df_results['test_r2'], 's--',
                 linewidth=2, markersize=8, color='purple', label='HOMA R²')
        ax2.set_ylabel('HOMA Test R²', color='purple')
    ax.set_xlabel('Fraction of Ring Types in Training (%)')
    ax.set_ylabel('# Test Ring Types', color='green')
    ax.set_title('Test Set Composition vs R²')
    ax.legend(loc='upper left')
    if 'test_r2' in df_results.columns:
        ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'ood_difficulty_analysis.png')
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f"保存: {fig_path}")

    # 打印汇总表
    print("\n" + "=" * 80)
    print("汇总: Residual OOD Difficulty Analysis (HOMA)")
    print("=" * 80)
    print(df_results[['fraction', 'n_test_ring_types', 'mean_nn_tanimoto',
                       'median_nn_tanimoto', 'frac_below_0.3', 'test_r2']].to_string(index=False))


if __name__ == '__main__':
    main()
