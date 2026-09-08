"""
ML 模型对比 + 跨任务特征增强实验 (v2 修复版)

修复问题:
1. 目标列名澄清: NICS 数据集的 homa_value 列实际存储 NICS-ZZ 值，统一重命名
2. PCA 数据泄露: 先划分再在训练集上拟合 PCA/Scaler
3. 环描述符精确对应: 用 atom_on_ring 列精确定位目标环原子
4. 基线与增强版样本集一致: Part 2 全部在 merged 6523 行上对比
5. 移除 Ring_ID 作为数值特征
6. 清理死代码

实验设计:
  Part 1: 三任务独立 ML 基线 (全量数据, 与 GNN-label 对比)
  Part 2: 跨任务特征增强预测 NICS (merged 6523 行, 公平对比)
"""
import os
import sys
import time
import ast
import warnings
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
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
import matplotlib

# --- Auto path bootstrap (do not remove) ---
import os as _os
_THIS_FILE = _os.path.abspath(__file__)
_d = _os.path.dirname(_THIS_FILE)
while not _os.path.exists(_os.path.join(_d, 'unified_models')) and _d != '/':
    _d = _os.path.dirname(_d)
_PROJ_ROOT = _d
# --- End auto path bootstrap ---

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

warnings.filterwarnings('ignore')

PROJ_ROOT = _PROJ_ROOT
OUTPUT_DIR = os.path.join(PROJ_ROOT, 'ml_cross_task_results_v2')
os.makedirs(OUTPUT_DIR, exist_ok=True)

FP_PCA_COMPONENTS = 100


# ========== 特征工程 ==========

def compute_molecular_descriptors(smiles_list):
    """计算 RDKit 分子描述符 (16 个)"""
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
        }
        smiles_cache[smiles] = feat.copy()
        features.append(feat)

    return pd.DataFrame(features)


def compute_ring_descriptors(df):
    """
    基于 atom_on_ring 列精确计算目标环的描述符 (10 个)
    atom_on_ring 格式: '[9, 14, 13, 12, 11, 10]' -> 环上原子的 RDKit 索引
    """
    print("计算目标环描述符 (基于 atom_on_ring)...")
    ring_feats = []

    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        feat = {'ring_size': row['Ring_Size']}

        if mol is None:
            feat.update({k: 0 for k in [
                'ring_n_C', 'ring_n_N', 'ring_n_O', 'ring_n_S',
                'ring_n_aromatic', 'ring_n_aliphatic', 'n_ring_atoms',
                'ring_avg_degree', 'ring_max_degree', 'ring_n_substituents']})
            ring_feats.append(feat)
            continue

        # 解析 atom_on_ring 获取目标环的精确原子索引
        atom_indices = []
        try:
            atom_indices = ast.literal_eval(str(row['atom_on_ring']))
            if not isinstance(atom_indices, list):
                atom_indices = []
        except Exception:
            atom_indices = []

        # 过滤越界索引
        atom_indices = [i for i in atom_indices if isinstance(i, int) and i < mol.GetNumAtoms()]

        if atom_indices:
            ring_atoms = [mol.GetAtomWithIdx(i) for i in atom_indices]
            feat['ring_n_C'] = sum(1 for a in ring_atoms if a.GetAtomicNum() == 6)
            feat['ring_n_N'] = sum(1 for a in ring_atoms if a.GetAtomicNum() == 7)
            feat['ring_n_O'] = sum(1 for a in ring_atoms if a.GetAtomicNum() == 8)
            feat['ring_n_S'] = sum(1 for a in ring_atoms if a.GetAtomicNum() == 16)
            feat['ring_n_aromatic'] = sum(1 for a in ring_atoms if a.GetIsAromatic())
            feat['ring_n_aliphatic'] = len(ring_atoms) - feat['ring_n_aromatic']
            feat['n_ring_atoms'] = len(ring_atoms)
            degrees = [a.GetDegree() for a in ring_atoms]
            feat['ring_avg_degree'] = float(np.mean(degrees)) if degrees else 0.0
            feat['ring_max_degree'] = max(degrees) if degrees else 0
            feat['ring_n_substituents'] = sum(1 for d in degrees if d > 2)
        else:
            feat.update({k: 0 for k in [
                'ring_n_C', 'ring_n_N', 'ring_n_O', 'ring_n_S',
                'ring_n_aromatic', 'ring_n_aliphatic', 'n_ring_atoms',
                'ring_avg_degree', 'ring_max_degree', 'ring_n_substituents']})

        ring_feats.append(feat)

    return pd.DataFrame(ring_feats)


def compute_morgan_fingerprints(smiles_list, n_bits=512):
    """计算 Morgan 指纹 (原始 512 维, 不做 PCA)"""
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


# ========== 预处理 (无数据泄露) ==========

def fit_transform_features(X_mol_ring_train, X_fp_train, X_mol_ring_val, X_fp_val, extra_train=None, extra_val=None):
    """
    在训练集上拟合 PCA 和 StandardScaler, 然后 transform 验证/测试集
    返回处理后的训练集和验证/测试集
    """
    # PCA 只在训练集上拟合 (修复数据泄露)
    n_components = min(FP_PCA_COMPONENTS, X_fp_train.shape[0], X_fp_train.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_fp_train_pca = pca.fit_transform(X_fp_train)
    X_fp_val_pca = pca.transform(X_fp_val)

    # 合并特征
    X_train = np.hstack([X_mol_ring_train, X_fp_train_pca])
    X_val = np.hstack([X_mol_ring_val, X_fp_val_pca])

    # 添加额外描述符 (1D 或 2D 均可)
    if extra_train is not None:
        if extra_train.ndim == 1:
            extra_train = extra_train.reshape(-1, 1)
            extra_val = extra_val.reshape(-1, 1)
        X_train = np.hstack([X_train, extra_train])
        X_val = np.hstack([X_val, extra_val])

    # StandardScaler 只在训练集上拟合
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    return X_train_s, X_val_s


# ========== 评估 ==========

def run_5fold_cv(models, X_mol_ring, X_fp, y, extra=None):
    """5 折交叉验证 (每折内重新拟合 PCA/Scaler, 无数据泄露)"""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {name: {'r2': [], 'mae': [], 'rmse': []} for name in models}

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_mol_ring)):
        print(f"  Fold {fold+1}/5...", flush=True)

        X_mr_train, X_mr_val = X_mol_ring[train_idx], X_mol_ring[val_idx]
        X_fp_train, X_fp_val = X_fp[train_idx], X_fp[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        e_train = extra[train_idx] if extra is not None else None
        e_val = extra[val_idx] if extra is not None else None

        X_train_s, X_val_s = fit_transform_features(
            X_mr_train, X_fp_train, X_mr_val, X_fp_val, e_train, e_val)

        for name, model in models.items():
            t0 = time.time()
            m = type(model)(**model.get_params())

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


def run_final_test(models, X_mol_ring, X_fp, y, extra=None):
    """最终模型测试 (8/2 划分后, 在训练集上拟合 PCA/Scaler)"""
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    X_mr_train = X_mol_ring[train_idx]
    X_mr_test = X_mol_ring[test_idx]
    X_fp_train = X_fp[train_idx]
    X_fp_test = X_fp[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    e_train = extra[train_idx] if extra is not None else None
    e_test = extra[test_idx] if extra is not None else None

    X_train_s, X_test_s = fit_transform_features(
        X_mr_train, X_fp_train, X_mr_test, X_fp_test, e_train, e_test)

    print(f"Train: {len(X_train_s)}, Test: {len(X_test_s)}", flush=True)

    final_results = []
    best_model_name = None
    best_r2 = -999
    best_result = None

    for name, model in models.items():
        t0 = time.time()
        m = type(model)(**model.get_params())

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
            'y_test': y_test,
            'y_train': y_train,
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

    return final_results, best_model_name, best_result


# ========== 可视化 ==========

def plot_density_scatter(true, pred, title, save_path):
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

def prepare_raw_features(df, task_name):
    """
    准备原始特征 (不做 PCA/标准化, 留给划分后处理)
    返回: X_mol_ring (分子+环描述符), X_fp (Morgan指纹原始)
    """
    print(f"\n{'='*60}")
    print(f"准备特征: {task_name}")
    print(f"{'='*60}")

    mol_desc = compute_molecular_descriptors(df['smiles'].tolist())
    ring_desc = compute_ring_descriptors(df)
    fps = compute_morgan_fingerprints(df['smiles'].tolist(), n_bits=512)

    # 分子描述符列 (16个, 不含 Ring_ID)
    feature_cols = ['mw', 'logp', 'tpsa', 'mr', 'n_heavy', 'n_bonds', 'n_rings',
                    'n_aromatic_rings', 'n_aliphatic_rings', 'n_heteroatoms',
                    'n_rotatable', 'n_hbd', 'n_hba', 'fsp3', 'qed', 'bertz']

    X_mol = mol_desc[feature_cols].fillna(0).values
    X_ring = ring_desc.fillna(0).values
    X_mol_ring = np.hstack([X_mol, X_ring])
    X_fp = fps

    print(f"分子+环描述符: {X_mol_ring.shape}")
    print(f"Morgan 指纹 (原始): {X_fp.shape}")
    return X_mol_ring, X_fp


def run_task_experiment(df, task_name, target_col, output_subdir, extra_col=None, extra_name=None):
    """
    运行单个任务的完整实验
    extra_col: 额外描述符的 numpy 数组 (n_samples,) 或 None
    extra_name: 额外描述符名称
    """
    task_dir = os.path.join(OUTPUT_DIR, output_subdir)
    os.makedirs(task_dir, exist_ok=True)

    y = df[target_col].values
    X_mol_ring, X_fp = prepare_raw_features(df, task_name)

    # 额外描述符
    extra = None
    if extra_col is not None:
        extra = np.array(extra_col)
        corr = np.corrcoef(extra, y)[0, 1]
        print(f"  额外描述符: {extra_name}, 与目标相关性 r = {corr:.4f}")

    # 5 折 CV (无数据泄露)
    print("\n--- 5 折交叉验证 ---", flush=True)
    models = get_models()
    cv_results = run_5fold_cv(models, X_mol_ring, X_fp, y, extra)

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

    # 最终模型测试
    print("\n--- 最终模型测试 ---", flush=True)
    final_results, best_model_name, best_result = run_final_test(
        models, X_mol_ring, X_fp, y, extra)

    final_df = pd.DataFrame(final_results).sort_values('Test_R2', ascending=False)
    final_df.to_csv(os.path.join(task_dir, 'final_results.csv'), index=False)
    print(f"\n最终测试结果:")
    print(final_df.to_string(index=False))

    # 最优模型散点图
    plot_density_scatter(
        best_result['y_test'], best_result['y_pred_test'],
        f'{task_name} - Best: {best_model_name} (Test)\nR²={best_result["test_r2"]:.4f}, '
        f'MAE={best_result["test_mae"]:.4f}, RMSE={best_result["test_rmse"]:.4f}',
        os.path.join(task_dir, 'best_test_parity.png'))
    plot_density_scatter(
        best_result['y_train'], best_result['y_pred_train'],
        f'{task_name} - Best: {best_model_name} (Train)\nR²={best_result["train_r2"]:.4f}',
        os.path.join(task_dir, 'best_train_parity.png'))

    # 保存预测数据
    pd.DataFrame({'true': best_result['y_test'],
                  'pred': best_result['y_pred_test']}).to_csv(
        os.path.join(task_dir, 'test_set_predictions.csv'), index=False)

    return cv_df, final_df, best_model_name


def main():
    print("=" * 70)
    print("ML 模型对比 + 跨任务特征增强实验 (v2 修复版)")
    print("=" * 70)

    # 加载数据
    # 注意: NICS 数据集的 homa_value 列实际存储的是 NICS-ZZ 值 (范围 -39~7.5)
    # Collet 数据集的 homa_value 列存储 HOMA 值 (范围 -7.5~1.0)
    # MBCO 数据集的 homa_value 列存储 MBCO 值 (范围 -0.5~0.7)
    nics = pd.read_csv(os.path.join(PROJ_ROOT, 'nics-nics1zz-out-no3.csv'))
    collet = pd.read_csv(os.path.join(PROJ_ROOT, 'collet_homa_0702.csv'))
    mbco = pd.read_csv(os.path.join(PROJ_ROOT, 'outcsv', 'lunci2-mbcout.csv'))

    # ================================================================
    # Part 1: 三任务独立 ML 基线 (全量数据, 与 GNN-label 对比)
    # ================================================================
    print("\n" + "=" * 70)
    print("Part 1: 三任务独立 ML 基线 (全量数据)")
    print("=" * 70)

    # 1a. NICS-ZZ 独立预测 (全量 6592 行)
    # NICS 数据集 homa_value 列实际是 NICS-ZZ, 重命名为 nics_value 以避免混淆
    nics_full = nics.copy()
    nics_full['nics_value'] = nics_full['homa_value']
    cv1, final1, best1 = run_task_experiment(
        nics_full, 'NICS-ZZ (Original)', 'nics_value', 'nics_baseline')

    # 1b. Collet HOMA 独立预测 (全量 7249 行)
    cv2, final2, best2 = run_task_experiment(
        collet, 'HOMA (Collet 0702)', 'homa_value', 'collet_baseline')

    # 1c. MBCO 独立预测 (全量 6699 行)
    cv3, final3, best3 = run_task_experiment(
        mbco, 'MBCO (Lunci2)', 'homa_value', 'mbco_baseline')

    # ================================================================
    # Part 2: 跨任务特征增强预测 NICS (merged 6523 行, 公平对比)
    # ================================================================
    print("\n" + "=" * 70)
    print("Part 2: 跨任务特征增强预测 NICS (统一 6523 行)")
    print("=" * 70)

    # 用 (smiles, Ring_ID) 作为 key 合并三个数据集
    nics['key'] = nics['smiles'] + '_' + nics['Ring_ID'].astype(str)
    collet['key'] = collet['smiles'] + '_' + collet['Ring_ID'].astype(str)
    mbco['key'] = mbco['smiles'] + '_' + mbco['Ring_ID'].astype(str)

    merged = nics[['key', 'smiles', 'Ring_ID', 'Ring_Size', 'atom_on_ring', 'homa_value']].copy()
    merged = merged.rename(columns={'homa_value': 'nics_value'})  # 实际是 NICS-ZZ
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

    # 2a. NICS Baseline (merged 6523 行, 无额外特征 - 公平对比基准)
    cv4, final4, best4 = run_task_experiment(
        merged, 'NICS Baseline (merged)', 'nics_value', 'nics_merged_baseline')

    # 2b. NICS + Collet HOMA
    cv5, final5, best5 = run_task_experiment(
        merged, 'NICS + Collet HOMA', 'nics_value', 'nics_plus_collet',
        extra_col=merged['collet_homa'].values, extra_name='collet_homa')

    # 2c. NICS + MBCO
    cv6, final6, best6 = run_task_experiment(
        merged, 'NICS + MBCO', 'nics_value', 'nics_plus_mbco',
        extra_col=merged['mbco_value'].values, extra_name='mbco_value')

    # 2d. NICS + Collet + MBCO
    # 两个额外描述符需要分别传入, 这里用合并的 2D 数组
    cv7, final7, best7 = run_task_experiment_both(
        merged, 'NICS + Collet + MBCO', 'nics_value', 'nics_plus_both',
        extra_cols={
            'collet_homa': merged['collet_homa'].values,
            'mbco_value': merged['mbco_value'].values,
        })

    # ================================================================
    # Part 3: 汇总对比
    # ================================================================
    print("\n" + "=" * 70)
    print("Part 3: 汇总对比")
    print("=" * 70)

    summary_rows = [
        # Part 1: 全量数据基线
        {'Part': '1-独立', 'Experiment': 'NICS-ZZ Baseline (全量)', 'Best_Model': best1,
         'N_Samples': 6592, 'Test_R2': final1.iloc[0]['Test_R2'],
         'Test_MAE': final1.iloc[0]['Test_MAE'], 'Test_RMSE': final1.iloc[0]['Test_RMSE'],
         'CV_R2': cv1.iloc[0]['CV_R2_mean'], 'CV_R2_std': cv1.iloc[0]['CV_R2_std']},
        {'Part': '1-独立', 'Experiment': 'HOMA(Collet) Baseline (全量)', 'Best_Model': best2,
         'N_Samples': 7249, 'Test_R2': final2.iloc[0]['Test_R2'],
         'Test_MAE': final2.iloc[0]['Test_MAE'], 'Test_RMSE': final2.iloc[0]['Test_RMSE'],
         'CV_R2': cv2.iloc[0]['CV_R2_mean'], 'CV_R2_std': cv2.iloc[0]['CV_R2_std']},
        {'Part': '1-独立', 'Experiment': 'MBCO Baseline (全量)', 'Best_Model': best3,
         'N_Samples': 6699, 'Test_R2': final3.iloc[0]['Test_R2'],
         'Test_MAE': final3.iloc[0]['Test_MAE'], 'Test_RMSE': final3.iloc[0]['Test_RMSE'],
         'CV_R2': cv3.iloc[0]['CV_R2_mean'], 'CV_R2_std': cv3.iloc[0]['CV_R2_std']},

        # Part 2: merged 6523 行 (公平对比)
        {'Part': '2-增强', 'Experiment': 'NICS Baseline (merged)', 'Best_Model': best4,
         'N_Samples': 6523, 'Test_R2': final4.iloc[0]['Test_R2'],
         'Test_MAE': final4.iloc[0]['Test_MAE'], 'Test_RMSE': final4.iloc[0]['Test_RMSE'],
         'CV_R2': cv4.iloc[0]['CV_R2_mean'], 'CV_R2_std': cv4.iloc[0]['CV_R2_std']},
        {'Part': '2-增强', 'Experiment': 'NICS + Collet HOMA', 'Best_Model': best5,
         'N_Samples': 6523, 'Test_R2': final5.iloc[0]['Test_R2'],
         'Test_MAE': final5.iloc[0]['Test_MAE'], 'Test_RMSE': final5.iloc[0]['Test_RMSE'],
         'CV_R2': cv5.iloc[0]['CV_R2_mean'], 'CV_R2_std': cv5.iloc[0]['CV_R2_std']},
        {'Part': '2-增强', 'Experiment': 'NICS + MBCO', 'Best_Model': best6,
         'N_Samples': 6523, 'Test_R2': final6.iloc[0]['Test_R2'],
         'Test_MAE': final6.iloc[0]['Test_MAE'], 'Test_RMSE': final6.iloc[0]['Test_RMSE'],
         'CV_R2': cv6.iloc[0]['CV_R2_mean'], 'CV_R2_std': cv6.iloc[0]['CV_R2_std']},
        {'Part': '2-增强', 'Experiment': 'NICS + Collet + MBCO', 'Best_Model': best7,
         'N_Samples': 6523, 'Test_R2': final7.iloc[0]['Test_R2'],
         'Test_MAE': final7.iloc[0]['Test_MAE'], 'Test_RMSE': final7.iloc[0]['Test_RMSE'],
         'CV_R2': cv7.iloc[0]['CV_R2_mean'], 'CV_R2_std': cv7.iloc[0]['CV_R2_std']},

        # GNN-label 参考
        {'Part': '参考', 'Experiment': 'NICS GNN-label (参考)', 'Best_Model': 'GNN-label',
         'N_Samples': 6592, 'Test_R2': 0.9771, 'Test_MAE': 1.1289, 'Test_RMSE': 1.8406,
         'CV_R2': 0.9764, 'CV_R2_std': 0.0015},
        {'Part': '参考', 'Experiment': 'HOMA(Collet) GNN-label (参考)', 'Best_Model': 'GNN-label',
         'N_Samples': 7249, 'Test_R2': 0.9846, 'Test_MAE': 0.1652, 'Test_RMSE': 0.2869,
         'CV_R2': 0.9840, 'CV_R2_std': 0.0011},
        {'Part': '参考', 'Experiment': 'MBCO GNN-label (参考)', 'Best_Model': 'GNN-label',
         'N_Samples': 6699, 'Test_R2': 0.9863, 'Test_MAE': 0.0230, 'Test_RMSE': 0.0376,
         'CV_R2': 0.9770, 'CV_R2_std': 0.0039},
    ]

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'all_experiments_summary.csv'), index=False)
    print("\n汇总对比:")
    print(summary_df.to_string(index=False))

    # 对比图: Part 2 NICS 预测 (公平对比)
    fig, ax = plt.subplots(figsize=(12, 7), dpi=200)
    part2 = summary_df[summary_df['Part'] == '2-增强'].copy()
    part2['label'] = part2['Experiment'] + '\n(' + part2['Best_Model'] + ')'
    colors = ['#4C72B0', '#55A868', '#DD8452', '#C44E52']
    bars = ax.bar(range(len(part2)), part2['Test_R2'], color=colors, edgecolor='white', width=0.6)
    ax.set_xticks(range(len(part2)))
    ax.set_xticklabels(part2['label'], fontsize=9, rotation=15, ha='right')
    ax.set_ylabel('Test R²', fontsize=13, fontweight='bold')
    ax.set_title('NICS Prediction: Baseline vs Cross-Task Enhancement\n(unified 6523 samples, no data leakage)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    for bar, val in zip(bars, part2['Test_R2']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(part2['Test_R2']) * 1.08)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'nics_comparison_bar.png'), bbox_inches='tight', facecolor='white')
    plt.close()

    # 全部实验 R² 对比图
    fig, ax = plt.subplots(figsize=(14, 8), dpi=200)
    all_exps = summary_df.copy()
    all_exps['label'] = all_exps['Experiment'] + '\n(' + all_exps['Best_Model'] + ')'
    colors_all = plt.cm.Set2(np.linspace(0, 1, len(all_exps)))
    bars = ax.barh(range(len(all_exps)), all_exps['Test_R2'], color=colors_all, edgecolor='white')
    ax.set_yticks(range(len(all_exps)))
    ax.set_yticklabels(all_exps['label'], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Test R²', fontsize=13, fontweight='bold')
    ax.set_title('All Experiments: Test R² Comparison (ML v2 + GNN)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='x')
    for bar, val in zip(bars, all_exps['Test_R2']):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'all_experiments_r2_comparison.png'), bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\n所有结果保存到: {OUTPUT_DIR}")


def run_task_experiment_both(df, task_name, target_col, output_subdir, extra_cols):
    """处理两个额外描述符的情况"""
    task_dir = os.path.join(OUTPUT_DIR, output_subdir)
    os.makedirs(task_dir, exist_ok=True)

    y = df[target_col].values
    X_mol_ring, X_fp = prepare_raw_features(df, task_name)

    # 两个额外描述符拼成 2D 数组
    extra = np.column_stack([extra_cols['collet_homa'], extra_cols['mbco_value']])
    for name, values in extra_cols.items():
        corr = np.corrcoef(values, y)[0, 1]
        print(f"  额外描述符: {name}, 与目标相关性 r = {corr:.4f}")

    # 5 折 CV
    print("\n--- 5 折交叉验证 ---", flush=True)
    models = get_models()
    cv_results = run_5fold_cv(models, X_mol_ring, X_fp, y, extra)

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

    # 最终模型测试
    print("\n--- 最终模型测试 ---", flush=True)
    final_results, best_model_name, best_result = run_final_test(
        models, X_mol_ring, X_fp, y, extra)

    final_df = pd.DataFrame(final_results).sort_values('Test_R2', ascending=False)
    final_df.to_csv(os.path.join(task_dir, 'final_results.csv'), index=False)
    print(f"\n最终测试结果:")
    print(final_df.to_string(index=False))

    plot_density_scatter(
        best_result['y_test'], best_result['y_pred_test'],
        f'{task_name} - Best: {best_model_name} (Test)\nR²={best_result["test_r2"]:.4f}, '
        f'MAE={best_result["test_mae"]:.4f}, RMSE={best_result["test_rmse"]:.4f}',
        os.path.join(task_dir, 'best_test_parity.png'))
    plot_density_scatter(
        best_result['y_train'], best_result['y_pred_train'],
        f'{task_name} - Best: {best_model_name} (Train)\nR²={best_result["train_r2"]:.4f}',
        os.path.join(task_dir, 'best_train_parity.png'))

    pd.DataFrame({'true': best_result['y_test'],
                  'pred': best_result['y_pred_test']}).to_csv(
        os.path.join(task_dir, 'test_set_predictions.csv'), index=False)

    return cv_df, final_df, best_model_name


if __name__ == '__main__':
    main()
