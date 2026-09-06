"""
XGBoost / Random Forest 环级 HOMA 预测
从 SMILES + atom_on_ring 提取环专属特征，预测单个环的 HOMA 值

两个版本：
  - no_nics: 不使用 NICS 特征（与 GNN 公平对比）
  - with_nics: 使用 NICS 特征（看性能上限）
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, rdmolops
from rdkit.Chem import rdDistGeom as molDG
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
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
from scipy import stats

warnings.filterwarnings('ignore')

PROJ_ROOT = '_PROJ_ROOT'
DATASET_PATH = os.path.join(PROJ_ROOT, 'nics-nics1zz-out-no3.csv')
OUTPUT_DIR = os.path.join(PROJ_ROOT, '0427_unified_results')
ML_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'ml_models')
os.makedirs(ML_OUTPUT_DIR, exist_ok=True)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    print("安装 xgboost...")
    os.system(f"{sys.executable} -m pip install xgboost -q")
    import xgboost as xgb
    HAS_XGB = True


def extract_ring_features(smiles, atom_on_ring, ring_size, nics_zz1=None, nics_zz2=None):
    """
    从分子 SMILES 和环原子索引中提取环级特征向量

    参数:
        smiles: 分子 SMILES
        atom_on_ring: 环上原子索引列表 (1-indexed)
        ring_size: 环大小
        nics_zz1, nics_zz2: NICS 值 (可选)

    返回:
        特征字典
    """
    features = {}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol_h = Chem.AddHs(mol)
    try:
        AllChem.ComputeGasteigerCharges(mol_h)
    except Exception:
        pass

    # atom_on_ring 是 1-indexed，转换为 0-indexed
    ring_atom_indices = [idx - 1 for idx in atom_on_ring if idx - 1 >= 0 and idx - 1 < mol.GetNumAtoms()]
    n_ring_atoms = len(ring_atom_indices)

    if n_ring_atoms == 0:
        return None

    ring_atoms = [mol.GetAtomWithIdx(i) for i in ring_atom_indices]

    # ========== 1. 环大小特征 ==========
    features['ring_size'] = float(ring_size)
    features['n_ring_atoms'] = float(n_ring_atoms)
    features['is_5_ring'] = 1.0 if ring_size == 5 else 0.0
    features['is_6_ring'] = 1.0 if ring_size == 6 else 0.0
    features['is_3_ring'] = 1.0 if ring_size == 3 else 0.0

    # ========== 2. 环上原子类型 ==========
    atom_nums = [a.GetAtomicNum() for a in ring_atoms]
    features['ring_n_carbon'] = float(sum(1 for z in atom_nums if z == 6))
    features['ring_n_nitrogen'] = float(sum(1 for z in atom_nums if z == 7))
    features['ring_n_oxygen'] = float(sum(1 for z in atom_nums if z == 8))
    features['ring_n_sulfur'] = float(sum(1 for z in atom_nums if z == 16))
    features['ring_n_other'] = float(sum(1 for z in atom_nums if z not in [6, 7, 8, 16]))
    features['ring_carbon_fraction'] = features['ring_n_carbon'] / n_ring_atoms

    # ========== 3. 环上原子属性 ==========
    aromatic_flags = [int(a.GetIsAromatic()) for a in ring_atoms]
    features['ring_n_aromatic'] = float(sum(aromatic_flags))
    features['ring_aromatic_fraction'] = sum(aromatic_flags) / n_ring_atoms
    features['ring_all_aromatic'] = 1.0 if all(aromatic_flags) else 0.0
    features['ring_none_aromatic'] = 1.0 if not any(aromatic_flags) else 0.0

    # 杂化状态
    hybridizations = [a.GetHybridization() for a in ring_atoms]
    features['ring_n_sp'] = float(sum(1 for h in hybridizations if h == Chem.HybridizationType.SP))
    features['ring_n_sp2'] = float(sum(1 for h in hybridizations if h == Chem.HybridizationType.SP2))
    features['ring_n_sp3'] = float(sum(1 for h in hybridizations if h == Chem.HybridizationType.SP3))

    # 氢原子数
    total_hs = [a.GetTotalNumHs() for a in ring_atoms]
    features['ring_avg_h'] = float(np.mean(total_hs))
    features['ring_total_h'] = float(sum(total_hs))

    # 形式电荷
    charges = [a.GetFormalCharge() for a in ring_atoms]
    features['ring_total_charge'] = float(sum(charges))
    features['ring_max_charge'] = float(max(charges))
    features['ring_min_charge'] = float(min(charges))

    # Gasteiger 电荷
    gasteiger_charges = []
    for a in mol_h.GetAtoms():
        if a.GetIdx() in ring_atom_indices:
            try:
                gasteiger_charges.append(float(a.GetProp('_GasteigerCharge')))
            except KeyError:
                gasteiger_charges.append(0.0)
    if gasteiger_charges:
        features['ring_avg_gasteiger'] = float(np.mean(gasteiger_charges))
        features['ring_std_gasteiger'] = float(np.std(gasteiger_charges))
        features['ring_max_gasteiger'] = float(max(gasteiger_charges))
        features['ring_min_gasteiger'] = float(min(gasteiger_charges))
    else:
        features['ring_avg_gasteiger'] = 0.0
        features['ring_std_gasteiger'] = 0.0
        features['ring_max_gasteiger'] = 0.0
        features['ring_min_gasteiger'] = 0.0

    # ========== 4. 环上键类型 ==========
    ring_bonds = []
    for i in range(n_ring_atoms):
        for j in range(i + 1, n_ring_atoms):
            bond = mol.GetBondBetweenAtoms(ring_atom_indices[i], ring_atom_indices[j])
            if bond is not None:
                ring_bonds.append(bond)

    bond_types = [b.GetBondType() for b in ring_bonds]
    features['ring_n_bonds'] = float(len(ring_bonds))
    features['ring_n_single'] = float(sum(1 for bt in bond_types if bt == Chem.BondType.SINGLE))
    features['ring_n_double'] = float(sum(1 for bt in bond_types if bt == Chem.BondType.DOUBLE))
    features['ring_n_aromatic_bonds'] = float(sum(1 for bt in bond_types if bt == Chem.BondType.AROMATIC))
    features['ring_double_bond_fraction'] = features['ring_n_double'] / max(len(ring_bonds), 1)

    # 环键级总和 (芳香键=1.5, 双键=2, 单键=1)
    bond_order_map = {Chem.BondType.SINGLE: 1.0, Chem.BondType.DOUBLE: 2.0,
                      Chem.BondType.AROMATIC: 1.5, Chem.BondType.TRIPLE: 3.0}
    total_bond_order = sum(bond_order_map.get(bt, 1.0) for bt in bond_types)
    features['ring_avg_bond_order'] = total_bond_order / max(len(ring_bonds), 1)
    features['ring_total_bond_order'] = float(total_bond_order)

    # ========== 5. 环上取代基 ==========
    substituents = []
    for atom in ring_atoms:
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIdx() not in ring_atom_indices:
                substituents.append(neighbor)

    features['ring_n_substituents'] = float(len(substituents))
    sub_atom_nums = [a.GetAtomicNum() for a in substituents]
    features['ring_sub_n_carbon'] = float(sum(1 for z in sub_atom_nums if z == 6))
    features['ring_sub_n_nitrogen'] = float(sum(1 for z in sub_atom_nums if z == 7))
    features['ring_sub_n_oxygen'] = float(sum(1 for z in sub_atom_nums if z == 8))
    features['ring_sub_n_halogens'] = float(sum(1 for z in sub_atom_nums if z in [9, 17, 35, 53]))

    # 取代基中含双键氧 (C=O)
    has_carbonyl = False
    for atom in ring_atoms:
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIdx() not in ring_atom_indices:
                bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
                if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                    if neighbor.GetSymbol() == 'O':
                        has_carbonyl = True
    features['ring_has_carbonyl_sub'] = 1.0 if has_carbonyl else 0.0

    # ========== 6. 分子级特征 ==========
    features['mol_n_atoms'] = float(mol.GetNumAtoms())
    features['mol_n_bonds'] = float(mol.GetNumBonds())
    features['mol_n_heavy_atoms'] = float(mol.GetNumHeavyAtoms())
    features['mol_mw'] = float(Descriptors.MolWt(mol))

    ring_info = mol.GetRingInfo()
    features['mol_n_rings'] = float(ring_info.NumRings())
    features['mol_n_aromatic_rings'] = float(rdMolDescriptors.CalcNumAromaticRings(mol))
    features['mol_n_aliphatic_rings'] = float(rdMolDescriptors.CalcNumAliphaticRings(mol))
    features['mol_n_heteroatoms'] = float(Descriptors.NumHeteroatoms(mol))
    features['mol_n_rotatable'] = float(Descriptors.NumRotatableBonds(mol))

    # 环的融合程度
    atom_rings = ring_info.AtomRings()
    ring_atom_set = set(ring_atom_indices)
    fused_rings = 0
    for ar in atom_rings:
        ar_set = set(ar)
        if len(ar_set & ring_atom_set) >= 2 and ar_set != ring_atom_set:
            fused_rings += 1
    features['ring_n_fused'] = float(fused_rings)

    # ========== 7. NICS 特征 (可选) ==========
    if nics_zz1 is not None:
        features['nics_zz1'] = float(nics_zz1)
    if nics_zz2 is not None:
        features['nics_zz2'] = float(nics_zz2)
        features['nics_diff'] = float(nics_zz1 - nics_zz2) if nics_zz1 is not None else 0.0

    return features


def build_feature_dataset(df, use_nics=False):
    """从数据集构建特征矩阵"""
    features_list = []
    targets = []
    valid_indices = []

    for idx, row in df.iterrows():
        smiles = row['smiles']
        atom_on_ring = eval(row['atom_on_ring']) if isinstance(row['atom_on_ring'], str) else row['atom_on_ring']
        ring_size = row['Ring_Size']
        nics_zz1 = row.get('Ring_NICS_ZZ_1') if use_nics else None
        nics_zz2 = row.get('Ring_NICS_ZZ_2') if use_nics else None

        feats = extract_ring_features(smiles, atom_on_ring, ring_size, nics_zz1, nics_zz2)
        if feats is not None:
            features_list.append(feats)
            targets.append(row['homa_value'])
            valid_indices.append(idx)

    X = pd.DataFrame(features_list)
    y = np.array(targets)
    print(f"特征矩阵: {X.shape}, 目标: {y.shape}")
    print(f"特征列: {list(X.columns)}")
    return X, y, valid_indices


def train_and_evaluate(X, y, model_name, model, use_nics=False):
    """训练和评估模型"""
    # 与 GNN 相同的划分: 80/20, seed=42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练
    print(f"\n{'='*60}")
    print(f"训练 {model_name}...")
    print(f"  训练集: {X_train.shape}, 测试集: {X_test.shape}")
    model.fit(X_train_scaled, y_train)

    # 预测
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)

    # 评估
    train_r2 = r2_score(y_train, train_pred)
    train_mae = mean_absolute_error(y_train, train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))

    test_r2 = r2_score(y_test, test_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

    print(f"  训练集: R²={train_r2:.4f}, MAE={train_mae:.4f}, RMSE={train_rmse:.4f}")
    print(f"  测试集: R²={test_r2:.4f}, MAE={test_mae:.4f}, RMSE={test_rmse:.4f}")

    # 保存结果
    suffix = 'with_nics' if use_nics else 'no_nics'
    model_dir = os.path.join(ML_OUTPUT_DIR, f'{model_name}_{suffix}')
    os.makedirs(model_dir, exist_ok=True)

    # 保存 summary
    summary = pd.DataFrame([
        {'metric': 'train_r2', 'value': train_r2},
        {'metric': 'train_mae', 'value': train_mae},
        {'metric': 'train_rmse', 'value': train_rmse},
        {'metric': 'test_r2', 'value': test_r2},
        {'metric': 'test_mae', 'value': test_mae},
        {'metric': 'test_rmse', 'value': test_rmse},
    ])
    summary.to_csv(os.path.join(model_dir, 'summary.csv'), index=False)

    # 保存预测数据
    pred_df = pd.DataFrame({'true': y_test, 'pred': test_pred})
    pred_df.to_csv(os.path.join(model_dir, 'test_set_data.csv'), index=False)

    # 散点图
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    errors = np.abs(y_test - test_pred)
    sc = ax.scatter(y_test, test_pred, c=errors, cmap='RdYlBu_r', alpha=0.5, s=12, edgecolors='none')
    plt.colorbar(sc, ax=ax, label='|Error|', shrink=0.8)
    lims = [min(y_test.min(), test_pred.min()) - 2, max(y_test.max(), test_pred.max()) + 2]
    ax.plot(lims, lims, 'r-', linewidth=2, alpha=0.8, label='Ideal (y=x)')
    slope, intercept, r_val, _, _ = stats.linregress(y_test, test_pred)
    x_fit = np.linspace(lims[0], lims[1], 100)
    ax.plot(x_fit, slope * x_fit + intercept, 'g--', linewidth=1.5, alpha=0.7,
            label=f'Fit (y={slope:.3f}x+{intercept:.3f})')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('True HOMA Value', fontsize=13)
    ax.set_ylabel('Predicted HOMA Value', fontsize=13)
    ax.set_title(f'{model_name} ({suffix})\nR²={test_r2:.4f}, MAE={test_mae:.4f}, RMSE={test_rmse:.4f}',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 特征重要性
    if hasattr(model, 'feature_importances_'):
        imp_df = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 8))
        top20 = imp_df.head(20)
        ax.barh(range(len(top20)), top20['importance'].values, color='steelblue')
        ax.set_yticks(range(len(top20)))
        ax.set_yticklabels(top20['feature'].values, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'{model_name} ({suffix}) - Top 20 Feature Importance', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()

        imp_df.to_csv(os.path.join(model_dir, 'feature_importance.csv'), index=False)
        print(f"\n  Top 10 重要特征:")
        for _, r in imp_df.head(10).iterrows():
            print(f"    {r['feature']}: {r['importance']:.4f}")

    return {
        'model': model_name,
        'version': suffix,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
    }


def main():
    print("="*60)
    print("XGBoost / Random Forest 环级 HOMA 预测")
    print("="*60)

    df = pd.read_csv(DATASET_PATH)
    print(f"数据集: {len(df)} 行")

    all_results = []

    # ========== 版本1: 不含 NICS (与 GNN 公平对比) ==========
    print("\n\n" + "#"*60)
    print("# 版本1: 不含 NICS 特征 (与 GNN 公平对比)")
    print("#"*60)

    X, y, _ = build_feature_dataset(df, use_nics=False)

    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=8,
    )
    r = train_and_evaluate(X, y, 'xgboost', xgb_model, use_nics=False)
    all_results.append(r)

    # Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=8,
    )
    r = train_and_evaluate(X, y, 'random_forest', rf_model, use_nics=False)
    all_results.append(r)

    # ========== 版本2: 含 NICS (看性能上限) ==========
    print("\n\n" + "#"*60)
    print("# 版本2: 含 NICS 特征 (性能上限)")
    print("#"*60)

    X2, y2, _ = build_feature_dataset(df, use_nics=True)

    xgb_model2 = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=8,
    )
    r = train_and_evaluate(X2, y2, 'xgboost', xgb_model2, use_nics=True)
    all_results.append(r)

    rf_model2 = RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=8,
    )
    r = train_and_evaluate(X2, y2, 'random_forest', rf_model2, use_nics=True)
    all_results.append(r)

    # ========== 汇总 ==========
    print("\n\n" + "="*60)
    print("全部结果汇总")
    print("="*60)
    results_df = pd.DataFrame(all_results)
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join(ML_OUTPUT_DIR, 'ml_summary.csv'), index=False)

    # 对比柱状图
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(results_df))
    colors = ['#4C72B0', '#4C72B0', '#DD8452', '#DD8452']
    bars = ax.bar(x, results_df['test_r2'], color=colors, edgecolor='white')
    for bar, val in zip(bars, results_df['test_r2']):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.003, f'{val:.4f}',
                ha='center', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r['model']}\n({r['version']})" for _, r in results_df.iterrows()],
        fontsize=10
    )
    ax.set_ylabel('Test R²', fontsize=13)
    ax.set_title('XGBoost / RF vs GNN - Test R² Comparison', fontsize=14)
    ax.axhline(y=0.9786, color='red', linestyle='--', linewidth=1.5, label='GNN-label (best): 0.9786')
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ML_OUTPUT_DIR, 'ml_vs_gnn_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n结果保存到: {ML_OUTPUT_DIR}")


if __name__ == '__main__':
    main()
