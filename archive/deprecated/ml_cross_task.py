"""
ML 模型对比 + 跨任务特征增强实验

实验设计：
1. 三任务独立 ML 基线：用 RDKit 描述符 + Morgan 指纹训练 RF/XGBoost/LightGBM/GBDT/SVR/KNN/Ridge
2. 跨任务特征增强：用 Collet HOMA 和/或 MBCO 值作为额外描述符预测 NICS

评估协议：8/2 划分 + 5 折 CV（与 GNN-label 完全一致）
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, rdmolops
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb


# --- Auto path bootstrap (do not remove) ---
import os as _os
_THIS_FILE = _os.path.abspath(__file__)
_d = _os.path.dirname(_THIS_FILE)
while not _os.path.exists(_os.path.join(_d, 'unified_models')) and _d != '/':
    _d = _os.path.dirname(_d)
_PROJ_ROOT = _d
# --- End auto path bootstrap ---

# Morgan 指纹 PCA 降维后的维数（加速 SVR/KNN）
FP_PCA_COMPONENTS = 100
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde

warnings.filterwarnings('ignore')

PROJ_ROOT = '_PROJ_ROOT'
OUTPUT_DIR = os.path.join(PROJ_ROOT, 'ml_cross_task_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ========== 特征工程 ==========

def compute_molecular_descriptors(smiles_list):
    """计算 RDKit 分子描述符 + 环特征"""
    print(f"计算分子描述符 ({len(smiles_list)} 分子)...")
    features = []
    smiles_cache = {}

    for idx, smiles in enumerate(smiles_list):
        if idx % 1000 == 0:
            print(f"  进度: {idx}/{len(smiles_list)}")

        if smiles in smiles_cache:
            features.append(smiles_cache[smiles].copy())
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            features.append({})
            continue

        # 基本描述符
        feat = {
            'mw': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol),
            'mr': Descriptors.MolMR(mol),
            'n_heavy': mol.GetNumHeavyAtoms(),
            'n_bonds': mol.GetNumBonds(),
            'n_rings': rdMolDescriptors.CalcNumRings(mol),
            'n_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
            'n_aliphatic_rings': rdMolDescriptors.CalcNumAliphaticRings(mol),
            'n_heteroatoms': Descriptors.NumHeteroatoms(mol),
            'n_rotatable': Descriptors.NumRotatableBonds(mol),
            'n_hbd': Descriptors.NumHDonors(mol),
            'n_hba': Descriptors.NumHAcceptors(mol),
            'fsp3': rdMolDescriptors.CalcFractionCSP3(mol),
            'qed': Descriptors.qed(mol),
            'bertz': Descriptors.BertzCT(mol),
            'n_ring_atoms': 0,
            'ring_n_C': 0, 'ring_n_N': 0, 'ring_n_O': 0, 'ring_n_S': 0,
            'ring_n_aromatic': 0, 'ring_n_aliphatic': 0,
        }
        smiles_cache[smiles] = feat.copy()
        features.append(feat)

    return pd.DataFrame(features)


def compute_ring_descriptors(df):
    """计算环特定的描述符（每个环行）"""
    print("计算环特定描述符...")
    ring_feats = []

    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        feat = {
            'ring_size': row['Ring_Size'],
            'ring_id': row['Ring_ID'],
        }

        if mol is not None:
            ring_info = mol.GetRingInfo()
            atom_rings = ring_info.AtomRings()

            # 找到与 Ring_Size 匹配的环
            matching_rings = [r for r in atom_rings if len(r) == row['Ring_Size']]

            if matching_rings:
                # 取第 Ring_ID 个匹配的环（或第一个）
                idx = min(int(row['Ring_ID']) - 1, len(matching_rings) - 1) if len(matching_rings) > 0 else 0
                ring = matching_rings[max(0, idx)]

                ring_atoms = [mol.GetAtomWithIdx(i) for i in ring]
                feat['ring_n_C'] = sum(1 for a in ring_atoms if a.GetAtomicNum() == 6)
                feat['ring_n_N'] = sum(1 for a in ring_atoms if a.GetAtomicNum() == 7)
                feat['ring_n_O'] = sum(1 for a in ring_atoms if a.GetAtomicNum() == 8)
                feat['ring_n_S'] = sum(1 for a in ring_atoms if a.GetAtomicNum() == 16)
                feat['ring_n_aromatic'] = sum(1 for a in ring_atoms if a.GetIsAromatic())
                feat['ring_n_aliphatic'] = len(ring_atoms) - feat['ring_n_aromatic']
                feat['n_ring_atoms'] = len(ring_atoms)

                # 环上原子度数统计
                degrees = [a.GetDegree() for a in ring_atoms]
                feat['ring_avg_degree'] = np.mean(degrees) if degrees else 0
                feat['ring_max_degree'] = max(degrees) if degrees else 0

                # 环上取代基数量（度 > 2 的原子）
                feat['ring_n_substituents'] = sum(1 for d in degrees if d > 2)
            else:
                feat.update({k: 0 for k in ['ring_n_C', 'ring_n_N', 'ring_n_O', 'ring_n_S',
                                             'ring_n_aromatic', 'ring_n_aliphatic', 'n_ring_atoms',
                                             'ring_avg_degree', 'ring_max_degree', 'ring_n_substituents']})
        else:
            feat.update({k: 0 for k in ['ring_n_C', 'ring_n_N', 'ring_n_O', 'ring_n_S',
                                         'ring_n_aromatic', 'ring_n_aliphatic', 'n_ring_atoms',
                                         'ring_avg_degree', 'ring_max_degree', 'ring_n_substituents']})

        ring_feats.append(feat)

    return pd.DataFrame(ring_feats)


def compute_morgan_fingerprints(smiles_list, n_bits=512):
    """计算 Morgan 指纹（PCA 降维到 n_components）"""
    print(f"计算 Morgan 指纹 ({len(smiles_list)} 分子, {n_bits} bits)...")
    fps = []
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)

    for idx, smiles in enumerate(smiles_list):
        if idx % 1000 == 0:
            print(f"  进度: {idx}/{len(smiles_list)}")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            fps.append(np.zeros(n_bits))
            continue
        fp = fp_gen.GetFingerprint(mol)
        arr = np.zeros(n_bits)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)

    return np.array(fps)


# ========== 模型定义 ==========

def get_models():
    """返回所有模型字典"""
    return {
        'RandomForest': RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_leaf=2,
                                              n_jobs=16, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8, n_jobs=16, random_state=42),
        'LightGBM': lgb.LGBMRegressor(n_estimators=300, max_depth=-1, num_leaves=63,
                                      learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                      n_jobs=8, random_state=42, verbose=-1),
        'GBDT': GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                                          subsample=0.8, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0, gamma='scale', cache_size=2000, max_iter=5000),
        'KNN': KNeighborsRegressor(n_neighbors=15, n_jobs=8, algorithm='kd_tree'),
        'Ridge': Ridge(alpha=1.0),
    }


# ========== 评估 ==========

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """训练和评估单个模型"""
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    return {
        'test_r2': r2_score(y_test, y_pred_test),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_r2': r2_score(y_train, y_pred_train),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'y_pred_test': y_pred_test,
        'y_pred_train': y_pred_train,
    }


def run_5fold_cv(models, X, y):
    """5 折交叉验证"""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {name: {'r2': [], 'mae': [], 'rmse': []} for name in models}

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"  Fold {fold+1}/5...")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        for name, model in models.items():
            import time
            t0 = time.time()
            m = type(model)(**model.get_params())

            # SVR 子采样加速（最多 2000 样本）
            if name == 'SVR' and len(X_train_s) > 2000:
                np.random.seed(42)
                sub_idx = np.random.choice(len(X_train_s), 2000, replace=False)
                m.fit(X_train_s[sub_idx], y_train[sub_idx])
            else:
                m.fit(X_train_s, y_train)

            y_pred = m.predict(X_val_s)
            cv_results[name]['r2'].append(r2_score(y_val, y_pred))
            cv_results[name]['mae'].append(mean_absolute_error(y_val, y_pred))
            cv_results[name]['rmse'].append(np.sqrt(mean_squared_error(y_val, y_pred)))
            print(f"    {name}: R²={cv_results[name]['r2'][-1]:.4f}, MAE={cv_results[name]['mae'][-1]:.4f} ({time.time()-t0:.1f}s)", flush=True)

    return cv_results


# ========== 可视化 ==========

def plot_density_scatter(true, pred, title, save_path):
    """密度散点图"""
    true = np.array(true)
    pred = np.array(pred)
    r2 = r2_score(true, pred)
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))

    xy = np.vstack([true, pred])
    kde = gaussian_kde(xy)
    density = kde(xy)
    idx = np.argsort(density)

    fig, ax = plt.subplots(figsize=(9, 9), dpi=200)
    sc = ax.scatter(true[idx], pred[idx], c=density[idx], cmap='magma',
                    s=12, alpha=0.85, edgecolors='none')

    lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
    pad = 0.02 * (lims[1] - lims[0])
    lims = [lims[0] - pad, lims[1] + pad]
    ax.plot(lims, lims, 'r-', linewidth=1.8, alpha=0.85, label='y=x')

    z = np.polyfit(true, pred, 1)
    x_fit = np.linspace(lims[0], lims[1], 200)
    ax.plot(x_fit, np.poly1d(z)(x_fit), '--', color='#3498db', linewidth=1.5,
            alpha=0.8, label=f'Fit: y={z[0]:.3f}x+{z[1]:.3f}')

    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('True Values', fontsize=14, fontweight='bold')
    ax.set_ylabel('Predicted Values', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.02)
    cbar.set_label('Local Density', fontsize=12)

    stats_text = f'R² = {r2:.4f}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}\nN = {len(true):,}'
    ax.text(0.97, 0.03, stats_text, transform=ax.transAxes,
            fontsize=12, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                      edgecolor='black', alpha=0.92, linewidth=1.2),
            family='monospace', fontweight='bold')

    ax.legend(loc='upper left', fontsize=11, framealpha=0.92)
    ax.grid(True, linestyle='--', alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()


# ========== 主实验 ==========

def prepare_features(df, task_name, extra_cols=None):
    """准备特征矩阵"""
    print(f"\n{'='*60}")
    print(f"准备特征: {task_name}")
    print(f"{'='*60}")

    # 分子描述符
    mol_desc = compute_molecular_descriptors(df['smiles'].tolist())

    # 环描述符
    ring_desc = compute_ring_descriptors(df)

    # Morgan 指纹
    fps = compute_morgan_fingerprints(df['smiles'].tolist(), n_bits=512)

    # 合并
    feature_cols = ['mw', 'logp', 'tpsa', 'mr', 'n_heavy', 'n_bonds', 'n_rings',
                    'n_aromatic_rings', 'n_aliphatic_rings', 'n_heteroatoms',
                    'n_rotatable', 'n_hbd', 'n_hba', 'fsp3', 'qed', 'bertz']

    X_mol = mol_desc[feature_cols].fillna(0).values
    X_ring = ring_desc.fillna(0).values
    X_fp = fps

    # Morgan 指纹 PCA 降维（加速 SVR/KNN）
    n_components = min(FP_PCA_COMPONENTS, X_fp.shape[0], X_fp.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_fp_pca = pca.fit_transform(X_fp)
    print(f"  Morgan 指纹 PCA: {X_fp.shape[1]} → {n_components} (解释方差: {pca.explained_variance_ratio_.sum()*100:.1f}%)")

    # 合并所有特征
    X = np.hstack([X_mol, X_ring, X_fp_pca])

    feature_names = list(feature_cols) + list(ring_desc.columns) + [f'fp_pca_{i}' for i in range(n_components)]
    print(f"特征矩阵: {X.shape}")

    # 添加额外描述符
    if extra_cols is not None:
        for col_name, col_values in extra_cols.items():
            X = np.column_stack([X, col_values])
            feature_names.append(col_name)
            print(f"  添加额外描述符: {col_name} (相关性将计算)")

    return X, feature_names


def run_task_experiment(df, task_name, target_col, output_subdir, extra_cols=None):
    """运行单个任务的完整实验"""
    task_dir = os.path.join(OUTPUT_DIR, output_subdir)
    os.makedirs(task_dir, exist_ok=True)

    y = df[target_col].values
    X, feature_names = prepare_features(df, task_name, extra_cols)

    # 额外描述符相关性
    if extra_cols:
        for col_name, col_values in extra_cols.items():
            corr = np.corrcoef(col_values, y)[0, 1]
            print(f"  {col_name} vs {target_col}: r = {corr:.4f}")

    # 8/2 划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # 标准化
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 5 折 CV
    print("\n--- 5 折交叉验证 ---")
    models = get_models()
    cv_results = run_5fold_cv(models, X_train_s, y_train)

    cv_summary = []
    for name in models:
        cv_summary.append({
            'Model': name,
            'CV_R2_mean': np.mean(cv_results[name]['r2']),
            'CV_R2_std': np.std(cv_results[name]['r2']),
            'CV_MAE_mean': np.mean(cv_results[name]['mae']),
            'CV_MAE_std': np.std(cv_results[name]['mae']),
            'CV_RMSE_mean': np.mean(cv_results[name]['rmse']),
            'CV_RMSE_std': np.std(cv_results[name]['rmse']),
        })
    cv_df = pd.DataFrame(cv_summary).sort_values('CV_R2_mean', ascending=False)
    cv_df.to_csv(os.path.join(task_dir, 'cv_results.csv'), index=False)
    print(f"\n5 折 CV 结果:")
    print(cv_df.to_string(index=False))

    # 最终模型：在 80% 上训练，20% 测试
    print("\n--- 最终模型测试 ---")
    final_results = []
    best_model_name = None
    best_r2 = -999
    best_result = None

    for name, model in models.items():
        import time
        t0 = time.time()
        m = type(model)(**model.get_params())

        # SVR 子采样
        if name == 'SVR' and len(X_train_s) > 2000:
            np.random.seed(42)
            sub_idx = np.random.choice(len(X_train_s), 2000, replace=False)
            m.fit(X_train_s[sub_idx], y_train[sub_idx])
        else:
            m.fit(X_train_s, y_train)

        y_pred_test = m.predict(X_test_s)
        y_pred_train = m.predict(X_train_s)
        result = {
            'test_r2': r2_score(y_test, y_pred_test),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_r2': r2_score(y_train, y_pred_train),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'y_pred_test': y_pred_test,
            'y_pred_train': y_pred_train,
        }
        print(f"  {name}: Test R²={result['test_r2']:.4f}, MAE={result['test_mae']:.4f} ({time.time()-t0:.1f}s)", flush=True)
        final_results.append({
            'Model': name,
            'Test_R2': result['test_r2'],
            'Test_MAE': result['test_mae'],
            'Test_RMSE': result['test_rmse'],
            'Train_R2': result['train_r2'],
            'Train_MAE': result['train_mae'],
            'Train_RMSE': result['train_rmse'],
        })
        if result['test_r2'] > best_r2:
            best_r2 = result['test_r2']
            best_model_name = name
            best_result = result

    final_df = pd.DataFrame(final_results).sort_values('Test_R2', ascending=False)
    final_df.to_csv(os.path.join(task_dir, 'final_results.csv'), index=False)
    print(f"\n最终测试结果:")
    print(final_df.to_string(index=False))

    # 最优模型散点图（训练+测试）
    plot_density_scatter(
        y_test, best_result['y_pred_test'],
        f'{task_name} - Best: {best_model_name} (Test)\nR²={best_result["test_r2"]:.4f}, '
        f'MAE={best_result["test_mae"]:.4f}, RMSE={best_result["test_rmse"]:.4f}',
        os.path.join(task_dir, 'best_test_parity.png'))
    plot_density_scatter(
        y_train, best_result['y_pred_train'],
        f'{task_name} - Best: {best_model_name} (Train)\nR²={best_result["train_r2"]:.4f}',
        os.path.join(task_dir, 'best_train_parity.png'))

    # 保存预测数据
    pd.DataFrame({'true': y_test, 'pred': best_result['y_pred_test']}).to_csv(
        os.path.join(task_dir, 'test_set_predictions.csv'), index=False)

    return cv_df, final_df, best_model_name


def main():
    print("=" * 70)
    print("ML 模型对比 + 跨任务特征增强实验")
    print("=" * 70)

    # 加载数据
    nics = pd.read_csv(os.path.join(PROJ_ROOT, 'nics-nics1zz-out-no3.csv'))
    collet = pd.read_csv(os.path.join(PROJ_ROOT, 'collet_homa_0702.csv'))
    mbco = pd.read_csv(os.path.join(PROJ_ROOT, 'outcsv', 'lunci2-mbcout.csv'))

    # ================================================================
    # Part 1: 三任务独立 ML 基线
    # ================================================================
    print("\n" + "=" * 70)
    print("Part 1: 三任务独立 ML 基线")
    print("=" * 70)

    all_results = {}

    # 1a. NICS 独立预测
    cv1, final1, best1 = run_task_experiment(
        nics, 'NICS (Original)', 'homa_value', 'nics_baseline')

    # 1b. Collet HOMA 独立预测
    cv2, final2, best2 = run_task_experiment(
        collet, 'HOMA (Collet 0702)', 'homa_value', 'collet_baseline')

    # 1c. MBCO 独立预测
    cv3, final3, best3 = run_task_experiment(
        mbco, 'MBCO (Lunci2)', 'homa_value', 'mbco_baseline')

    # ================================================================
    # Part 2: 跨任务特征增强预测 NICS
    # ================================================================
    print("\n" + "=" * 70)
    print("Part 2: 跨任务特征增强预测 NICS")
    print("=" * 70)

    # 合并数据：用 (smiles, Ring_ID) 作为 key
    nics['key'] = nics['smiles'] + '_' + nics['Ring_ID'].astype(str)
    collet['key'] = collet['smiles'] + '_' + collet['Ring_ID'].astype(str)
    mbco['key'] = mbco['smiles'] + '_' + mbco['Ring_ID'].astype(str)

    merged = nics[['key', 'smiles', 'Ring_ID', 'Ring_Size', 'homa_value']].rename(
        columns={'homa_value': 'nics_value'})
    merged = merged.merge(
        collet[['key', 'homa_value']].rename(columns={'homa_value': 'collet_homa'}),
        on='key', how='inner')
    merged = merged.merge(
        mbco[['key', 'homa_value']].rename(columns={'homa_value': 'mbco_value'}),
        on='key', how='inner')

    print(f"合并后行数: {len(merged)}")
    print(f"  nics_value: [{merged['nics_value'].min():.2f}, {merged['nics_value'].max():.2f}]")
    print(f"  collet_homa: [{merged['collet_homa'].min():.2f}, {merged['collet_homa'].max():.2f}]")
    print(f"  mbco_value: [{merged['mbco_value'].min():.2f}, {merged['mbco_value'].max():.2f}]")
    print(f"  Corr(nics, collet): {merged['nics_value'].corr(merged['collet_homa']):.4f}")
    print(f"  Corr(nics, mbco): {merged['nics_value'].corr(merged['mbco_value']):.4f}")

    # 2a. NICS + Collet HOMA 作为额外描述符
    cv4, final4, best4 = run_task_experiment(
        merged, 'NICS + Collet HOMA Feature', 'nics_value', 'nics_plus_collet',
        extra_cols={'collet_homa': merged['collet_homa'].values})

    # 2b. NICS + MBCO 作为额外描述符
    cv5, final5, best5 = run_task_experiment(
        merged, 'NICS + MBCO Feature', 'nics_value', 'nics_plus_mbco',
        extra_cols={'mbco_value': merged['mbco_value'].values})

    # 2c. NICS + Collet HOMA + MBCO 双重增强
    cv6, final6, best6 = run_task_experiment(
        merged, 'NICS + Collet + MBCO', 'nics_value', 'nics_plus_both',
        extra_cols={'collet_homa': merged['collet_homa'].values,
                    'mbco_value': merged['mbco_value'].values})

    # ================================================================
    # Part 3: 汇总对比
    # ================================================================
    print("\n" + "=" * 70)
    print("Part 3: 汇总对比")
    print("=" * 70)

    # 各实验最优 ML 模型的测试集结果
    summary_rows = [
        {'Experiment': 'NICS Baseline', 'Best_Model': best1,
         'Test_R2': final1.iloc[0]['Test_R2'], 'Test_MAE': final1.iloc[0]['Test_MAE'],
         'Test_RMSE': final1.iloc[0]['Test_RMSE'],
         'CV_R2': cv1.iloc[0]['CV_R2_mean'], 'CV_R2_std': cv1.iloc[0]['CV_R2_std']},
        {'Experiment': 'HOMA(Collet) Baseline', 'Best_Model': best2,
         'Test_R2': final2.iloc[0]['Test_R2'], 'Test_MAE': final2.iloc[0]['Test_MAE'],
         'Test_RMSE': final2.iloc[0]['Test_RMSE'],
         'CV_R2': cv2.iloc[0]['CV_R2_mean'], 'CV_R2_std': cv2.iloc[0]['CV_R2_std']},
        {'Experiment': 'MBCO Baseline', 'Best_Model': best3,
         'Test_R2': final3.iloc[0]['Test_R2'], 'Test_MAE': final3.iloc[0]['Test_MAE'],
         'Test_RMSE': final3.iloc[0]['Test_RMSE'],
         'CV_R2': cv3.iloc[0]['CV_R2_mean'], 'CV_R2_std': cv3.iloc[0]['CV_R2_std']},
        {'Experiment': 'NICS + Collet HOMA', 'Best_Model': best4,
         'Test_R2': final4.iloc[0]['Test_R2'], 'Test_MAE': final4.iloc[0]['Test_MAE'],
         'Test_RMSE': final4.iloc[0]['Test_RMSE'],
         'CV_R2': cv4.iloc[0]['CV_R2_mean'], 'CV_R2_std': cv4.iloc[0]['CV_R2_std']},
        {'Experiment': 'NICS + MBCO', 'Best_Model': best5,
         'Test_R2': final5.iloc[0]['Test_R2'], 'Test_MAE': final5.iloc[0]['Test_MAE'],
         'Test_RMSE': final5.iloc[0]['Test_RMSE'],
         'CV_R2': cv5.iloc[0]['CV_R2_mean'], 'CV_R2_std': cv5.iloc[0]['CV_R2_std']},
        {'Experiment': 'NICS + Collet + MBCO', 'Best_Model': best6,
         'Test_R2': final6.iloc[0]['Test_R2'], 'Test_MAE': final6.iloc[0]['Test_MAE'],
         'Test_RMSE': final6.iloc[0]['Test_RMSE'],
         'CV_R2': cv6.iloc[0]['CV_R2_mean'], 'CV_R2_std': cv6.iloc[0]['CV_R2_std']},
    ]

    # GNN-label 基准（来自之前实验）
    summary_rows.append({
        'Experiment': 'NICS GNN-label (参考)', 'Best_Model': 'GNN-label',
        'Test_R2': 0.9771, 'Test_MAE': 1.1289, 'Test_RMSE': 1.8406,
        'CV_R2': 0.9764, 'CV_R2_std': 0.0015
    })
    summary_rows.append({
        'Experiment': 'HOMA(Collet) GNN-label (参考)', 'Best_Model': 'GNN-label',
        'Test_R2': 0.9846, 'Test_MAE': 0.1652, 'Test_RMSE': 0.2869,
        'CV_R2': 0.9840, 'CV_R2_std': 0.0011
    })
    summary_rows.append({
        'Experiment': 'MBCO GNN-label (参考)', 'Best_Model': 'GNN-label',
        'Test_R2': 0.9863, 'Test_MAE': 0.0230, 'Test_RMSE': 0.0376,
        'CV_R2': 0.9770, 'CV_R2_std': 0.0039
    })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'all_experiments_summary.csv'), index=False)
    print("\n汇总对比:")
    print(summary_df.to_string(index=False))

    # NICS 预测对比图（基线 vs 各种增强）
    fig, ax = plt.subplots(figsize=(12, 7), dpi=200)
    nics_exps = summary_df[summary_df['Experiment'].str.contains('NICS')].copy()
    nics_exps['label'] = nics_exps['Experiment'] + '\n(' + nics_exps['Best_Model'] + ')'
    colors = ['#4C72B0', '#55A868', '#DD8452', '#C44E52', '#8172B3', '#937860']
    bars = ax.bar(range(len(nics_exps)), nics_exps['Test_R2'], color=colors, edgecolor='white', width=0.6)
    ax.set_xticks(range(len(nics_exps)))
    ax.set_xticklabels(nics_exps['label'], fontsize=9, rotation=15, ha='right')
    ax.set_ylabel('Test R²', fontsize=13, fontweight='bold')
    ax.set_title('NICS Prediction: ML Baseline vs Cross-Task Feature Enhancement', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    for bar, val in zip(bars, nics_exps['Test_R2']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(nics_exps['Test_R2']) * 1.08)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'nics_comparison_bar.png'), bbox_inches='tight', facecolor='white')
    plt.close()

    # 所有实验 R² 对比图
    fig, ax = plt.subplots(figsize=(14, 7), dpi=200)
    all_exps = summary_df.copy()
    all_exps['label'] = all_exps['Experiment'] + '\n(' + all_exps['Best_Model'] + ')'
    colors_all = plt.cm.Set2(np.linspace(0, 1, len(all_exps)))
    bars = ax.barh(range(len(all_exps)), all_exps['Test_R2'], color=colors_all, edgecolor='white')
    ax.set_yticks(range(len(all_exps)))
    ax.set_yticklabels(all_exps['label'], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Test R²', fontsize=13, fontweight='bold')
    ax.set_title('All Experiments: Test R² Comparison (ML + GNN)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='x')
    for bar, val in zip(bars, all_exps['Test_R2']):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'all_experiments_r2_comparison.png'), bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\n所有结果保存到: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
