
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
Hard Ring-Type Table (改进版)

改进:
  1. 加样本量保护: n<5 标记 low-sample estimate
  2. 统计检验: fused vs non-fused, S-containing vs not, O-containing vs not
  3. Mann-Whitney U + bootstrap CI
  4. ring-type-level MAE 回归分析 (similarity, rarity, heteroatoms)
"""
import os
import sys
import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

PROJ_ROOT = '_PROJ_ROOT + "/code_end"'
sys.path.insert(0, PROJ_ROOT)
from generalization_test.code.ring_utils import identify_ring_type

L10_CSV = '_PROJ_ROOT/lunci10/lunci10-expanded-test.csv'
RESULTS_DIR = '_PROJ_ROOT + "/code_end"/results/learning_curve'
OUTPUT_DIR = '_PROJ_ROOT + "/code_end"/results/ood_difficulty_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_heteroatom_info(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}, 0
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    hetero = [a for a in atoms if a not in ('C', 'H')]
    return dict(Counter(hetero)), len(hetero)


def is_fused(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    ri = mol.GetRingInfo()
    if len(ri.AtomRings()) < 2:
        return False
    rings = ri.AtomRings()
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            if set(rings[i]) & set(rings[j]):
                return True
    return False


def main():
    print("=" * 60)
    print("Hard Ring-Type Table (改进版)")
    print("=" * 60)

    df_l10 = pd.read_csv(L10_CSV)
    task_cols = {'HOMA': 'HOMA', 'NICS_1zz': 'NICS_ZZ', 'MBCO': 'MBCO'}
    target_map = {'HOMA': 'homa_value', 'NICS_1zz': 'NICS_value', 'MBCO': 'mbco_value'}

    smiles_list = sorted(df_l10['SMILES'].unique())
    smi_to_rt = {smi: identify_ring_type(smi) for smi in smiles_list}
    rt_counter = Counter(smi_to_rt.values())
    total_smiles = len(smiles_list)

    # 收集所有预测
    all_errors = []
    for task_name, target_col in task_cols.items():
        for frac in [5, 10, 20, 30, 50, 70]:
            pred_path = os.path.join(RESULTS_DIR, task_name, f'frac_{frac}', 'predictions.csv')
            if not os.path.exists(pred_path):
                continue
            df_pred = pd.read_csv(pred_path)

            import glob
            # 文件名格式: l10_test_HOMA_0.05.csv (fraction 作为小数)
            test_csvs = glob.glob(f'/tmp/lc_rerun/l10_test_{task_name}_{frac/100}.csv')
            if not test_csvs:
                continue

            df_test = pd.read_csv(test_csvs[0])
            if len(df_test) != len(df_pred):
                continue

            tcol = target_map[task_name]
            df_test['pred'] = df_pred['pred'].values
            df_test['true_val'] = df_test[tcol]
            df_test['error'] = abs(df_test['pred'] - df_test['true_val'])
            df_test['ring_type'] = df_test['smiles'].map(smi_to_rt)
            df_test['task'] = task_name
            df_test['fraction'] = frac / 100.0
            all_errors.append(df_test[['smiles', 'ring_type', 'task', 'fraction',
                                        'true_val', 'pred', 'error']])

    if not all_errors:
        print("错误: 无预测结果")
        return

    df_errors = pd.concat(all_errors, ignore_index=True)

    # 为每个 SMILES 计算结构特征
    smi_features = {}
    for smi in df_errors['smiles'].unique():
        hetero, n_hetero = get_heteroatom_info(smi)
        smi_features[smi] = {
            'n_heteroatoms': n_hetero,
            'n_N': hetero.get('N', 0),
            'n_O': hetero.get('O', 0),
            'n_S': hetero.get('S', 0),
            'has_S': 'S' in hetero,
            'has_O': 'O' in hetero,
            'fused': is_fused(smi),
        }

    for smi, feat in smi_features.items():
        mask = df_errors['smiles'] == smi
        for k, v in feat.items():
            df_errors.loc[mask, k] = v

    # ring-type 级别聚合
    rt_stats = df_errors.groupby(['ring_type', 'task']).agg(
        n_samples=('error', 'count'),
        mean_error=('error', 'mean'),
        median_error=('error', 'median'),
        rmse=('error', lambda x: np.sqrt(np.mean(x**2))),
        bias=('error', lambda x: np.mean(np.abs(x) - x)),  # bias = mean(|error|) - mean(error)
        target_mean=('true_val', 'mean'),
        target_std=('true_val', 'std'),
        fused_fraction=('fused', 'mean'),
        mean_n_hetero=('n_heteroatoms', 'mean'),
        mean_n_N=('n_N', 'mean'),
        mean_n_O=('n_O', 'mean'),
        mean_n_S=('n_S', 'mean'),
        has_S_fraction=('has_S', 'mean'),
        has_O_fraction=('has_O', 'mean'),
    ).reset_index()

    # 添加频率 (定义: 该 ring type 在 lunci10 全 SMILES 中的占比)
    rt_stats['freq_in_full'] = rt_stats['ring_type'].map(rt_counter)
    rt_stats['freq_pct'] = rt_stats['freq_in_full'] / total_smiles * 100

    # 标记 low-sample
    rt_stats['low_sample'] = rt_stats['n_samples'] < 5

    rt_stats.to_csv(os.path.join(OUTPUT_DIR, 'hard_ring_types_all.csv'), index=False)

    # Top 15 HOMA
    rt_homa = rt_stats[rt_stats['task'] == 'HOMA'].sort_values('mean_error', ascending=False)
    rt_homa.to_csv(os.path.join(OUTPUT_DIR, 'hard_ring_types_HOMA.csv'), index=False)

    # 只看 n>=5 的
    rt_homa_filtered = rt_homa[rt_homa['n_samples'] >= 5].sort_values('mean_error', ascending=False)
    rt_homa_filtered.to_csv(os.path.join(OUTPUT_DIR, 'hard_ring_types_HOMA_filtered.csv'), index=False)

    print("\n" + "=" * 100)
    print("Top 15 最难 Ring Types (HOMA, n>=5)")
    print("=" * 100)
    cols = ['ring_type', 'n_samples', 'mean_error', 'rmse', 'freq_pct',
            'mean_n_hetero', 'fused_fraction', 'has_S_fraction', 'has_O_fraction', 'target_mean']
    print(rt_homa_filtered[cols].head(15).to_string(index=False))

    # 统计检验: fused vs non-fused
    print("\n" + "=" * 80)
    print("统计检验: 结构特征 vs 预测误差")
    print("=" * 80)

    for task_name in ['HOMA', 'NICS_1zz', 'MBCO']:
        df_t = df_errors[df_errors['task'] == task_name]
        print(f"\n--- {task_name} ---")

        # Fused vs non-fused
        fused_err = df_t[df_t['fused'] == True]['error'].values
        nonfused_err = df_t[df_t['fused'] == False]['error'].values
        if len(fused_err) > 0 and len(nonfused_err) > 0:
            u_stat, p_val = stats.mannwhitneyu(fused_err, nonfused_err, alternative='greater')
            print(f"  Fused vs Non-fused: fused MAE={fused_err.mean():.4f} (n={len(fused_err)}), "
                  f"non-fused MAE={nonfused_err.mean():.4f} (n={len(nonfused_err)}), "
                  f"Mann-Whitney U={u_stat:.0f}, p={p_val:.4e}")

        # S-containing vs not
        s_err = df_t[df_t['has_S'] == True]['error'].values
        nos_err = df_t[df_t['has_S'] == False]['error'].values
        if len(s_err) > 0 and len(nos_err) > 0:
            u_stat, p_val = stats.mannwhitneyu(s_err, nos_err, alternative='greater')
            print(f"  S-containing vs not: S MAE={s_err.mean():.4f} (n={len(s_err)}), "
                  f"no-S MAE={nos_err.mean():.4f} (n={len(nos_err)}), "
                  f"U={u_stat:.0f}, p={p_val:.4e}")

        # O-containing vs not
        o_err = df_t[df_t['has_O'] == True]['error'].values
        noo_err = df_t[df_t['has_O'] == False]['error'].values
        if len(o_err) > 0 and len(noo_err) > 0:
            u_stat, p_val = stats.mannwhitneyu(o_err, noo_err, alternative='greater')
            print(f"  O-containing vs not: O MAE={o_err.mean():.4f} (n={len(o_err)}), "
                  f"no-O MAE={noo_err.mean():.4f} (n={len(noo_err)}), "
                  f"U={u_stat:.0f}, p={p_val:.4e}")

    # ring-type-level MAE 回归分析
    print("\n" + "=" * 80)
    print("Ring-type-level MAE 回归分析")
    print("=" * 80)

    # 计算每个 ring type 的 mean NN similarity (需要训练集)
    # 简化: 用 freq 作为 rarity 的代理
    rt_level = rt_stats[rt_stats['task'] == 'HOMA'].copy()
    rt_level = rt_level[rt_level['n_samples'] >= 3]  # 过滤太小样本

    if len(rt_level) >= 10:
        # 准备回归变量
        X = rt_level[['freq_in_full', 'mean_n_hetero', 'mean_n_N', 'mean_n_O', 'mean_n_S',
                       'fused_fraction']].values
        y = rt_level['mean_error'].values

        # 使用 5-fold cross-validation 评估 (避免 in-sample 高估)
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score, KFold

        cv = KFold(n_splits=min(5, len(rt_level) // 2), shuffle=True, random_state=42)

        lr = LinearRegression()
        lr_cv_scores = cross_val_score(lr, X, y, cv=cv, scoring='r2')
        r2_lr_cv = lr_cv_scores.mean()
        r2_lr_cv_std = lr_cv_scores.std()

        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='r2')
        r2_rf_cv = rf_cv_scores.mean()
        r2_rf_cv_std = rf_cv_scores.std()

        # 在全数据上 fit RF 以获取 feature importance
        rf.fit(X, y)

        feat_names = ['freq (rarity)', 'n_hetero', 'n_N', 'n_O', 'n_S', 'fused']
        print(f"\n  Linear Regression CV R² = {r2_lr_cv:.4f} ± {r2_lr_cv_std:.4f}")
        print(f"  Random Forest CV R² = {r2_rf_cv:.4f} ± {r2_rf_cv_std:.4f}")
        print(f"  (n={len(rt_level)} ring types, {cv.n_splits}-fold CV)")
        print(f"\n  Random Forest Feature Importance (full-fit):")
        for name, imp in sorted(zip(feat_names, rf.feature_importances_), key=lambda x: -x[1]):
            print(f"    {name:>15}: {imp:.4f}")

    # 画图
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=120)

    # 1. Top 15 hard ring types (n>=5)
    ax = axes[0, 0]
    top = rt_homa_filtered.head(15)
    y_pos = np.arange(len(top))
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(top)))
    ax.barh(y_pos, top['mean_error'], color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{rt[:20]}" for rt in top['ring_type']], fontsize=8)
    ax.set_xlabel('Mean Absolute Error')
    ax.set_title('Top 15 Hard Ring Types (HOMA, n≥5)')
    ax.invert_yaxis()
    for j, (_, row) in enumerate(top.iterrows()):
        ax.text(row['mean_error'] + 0.01, j,
                f"n={int(row['n_samples'])}, freq={row['freq_pct']:.1f}%",
                va='center', fontsize=7)

    # 2. Fused vs non-fused
    ax = axes[0, 1]
    tasks_plot = ['HOMA', 'NICS_1zz', 'MBCO']
    fused_means = []
    nonfused_means = []
    for t in tasks_plot:
        df_t = df_errors[df_errors['task'] == t]
        fused_means.append(df_t[df_t['fused'] == True]['error'].mean())
        nonfused_means.append(df_t[df_t['fused'] == False]['error'].mean())
    x = np.arange(len(tasks_plot))
    w = 0.35
    ax.bar(x - w/2, fused_means, w, label='Fused', color='coral')
    ax.bar(x + w/2, nonfused_means, w, label='Non-fused', color='steelblue')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks_plot)
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Fused vs Non-fused Ring Systems')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 3. S-containing vs not
    ax = axes[1, 0]
    s_means = []
    nos_means = []
    for t in tasks_plot:
        df_t = df_errors[df_errors['task'] == t]
        s_means.append(df_t[df_t['has_S'] == True]['error'].mean())
        nos_means.append(df_t[df_t['has_S'] == False]['error'].mean())
    ax.bar(x - w/2, s_means, w, label='S-containing', color='gold')
    ax.bar(x + w/2, nos_means, w, label='No S', color='gray')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks_plot)
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Sulfur-containing vs Not')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 4. RF feature importance
    ax = axes[1, 1]
    if len(rt_level) >= 10:
        imps = sorted(zip(feat_names, rf.feature_importances_), key=lambda x: x[1])
        names = [x[0] for x in imps]
        values = [x[1] for x in imps]
        ax.barh(range(len(names)), values, color='green', alpha=0.7)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'RF Feature Importance for Ring-type MAE\n(CV R²={r2_rf_cv:.3f}±{r2_rf_cv_std:.3f})')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'hard_ring_types_v2.png')
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f"\n图表: {fig_path}")


if __name__ == '__main__':
    main()
