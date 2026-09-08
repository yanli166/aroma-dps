"""
为汇总.xlsx中的每个SMILES计算立体程度相关描述符
并添加到对应行后面

计算的描述符:
  1. Ring_RPD              - Target ring 偏离平面的RMSD (Ring Plane Deviation)
  2. Ring_Cremer_Pople_Q   - Target ring Cremer-Pople puckering amplitude
  3. Ring_Max_Dev          - Target ring 单个原子偏离平面的最大值
  4. Mol_PBF               - Whole molecule Plane Best Fit (整体偏离最佳平面)
  5. NPR1                  - Normalized Principal Moment 1 (I1/I3)
  6. NPR2                  - Normalized Principal Moment 2 (I2/I3)
  7. Fsp3                  - Fraction of sp3 carbons
  8. Asphericity           - 偏离球形的程度 (RDKit)
  9. Sphericity            - 接近球形的程度 (1 - Asphericity 衍生)
  10. Eccentricity         - 偏心率
  11. InertialShapeFactor  - 惯性形状因子
"""
import os
import ast
import warnings
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit import RDLogger

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

XLSX_PATH = '/home/ubuntu/aroma-dps-code/汇总.xlsx'
OUTPUT_PATH = '/home/ubuntu/aroma-dps-code/汇总_stereo.xlsx'
N_CONFS = 20  # 每个分子生成的构象数
SEED = 42


# ── 3D构象生成 ─────────────────────────────────────────────────────
def generate_best_conformer(mol, n_confs=N_CONFS, seed=SEED):
    """生成多个构象，MMFF优化，返回能量最低的构象ID"""
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed

    # 生成多个构象
    cids = AllChem.EmbedMultipleConfs(mol_h, numConfs=n_confs, params=params)
    if len(cids) == 0:
        # 回退：单个构象
        params.randomSeed = seed + 1
        if AllChem.EmbedMolecule(mol_h, params) == 0:
            cids = [0]
        else:
            return None, mol_h

    # MMFF优化所有构象并找最低能量
    try:
        results = AllChem.MMFFOptimizeMoleculeConfs(mol_h, numThreads=0)
        min_energy = float('inf')
        best_cid = cids[0]
        for cid in cids:
            if cid < len(results) and results[cid][0] == 0:
                energy = results[cid][1]
                if energy < min_energy:
                    min_energy = energy
                    best_cid = cid
        return best_cid, mol_h
    except Exception:
        return cids[0], mol_h


# ── 平面拟合与偏离计算 ─────────────────────────────────────────────
def compute_plane_deviation(coords):
    """
    给定N个原子的3D坐标，计算最佳拟合平面
    返回: (displacements, rmsd, q, max_dev)
      - displacements: 各原子偏离平面的距离 (有符号)
      - rmsd: RMSD = sqrt(mean(d^2))
      - q: Cremer-Pople Q = sqrt(sum(d^2))
      - max_dev: 最大绝对偏离
    """
    coords = np.array(coords, dtype=float)
    n = len(coords)
    if n < 3:
        return np.zeros(n), 0.0, 0.0, 0.0

    # 中心化
    centroid = coords.mean(axis=0)
    centered = coords - centroid

    # SVD: 最小奇异值对应的右奇异向量是平面法向量
    U, S, Vt = np.linalg.svd(centered)
    normal = Vt[-1]  # 最后一行对应最小奇异值

    # 各原子到平面的有符号距离
    displacements = centered @ normal

    # RMSD
    rmsd = np.sqrt(np.mean(displacements**2))
    # Cremer-Pople Q
    q = np.sqrt(np.sum(displacements**2))
    # 最大偏离
    max_dev = np.max(np.abs(displacements))

    return displacements, rmsd, q, max_dev


# ── PBF (Plane Best Fit) ──────────────────────────────────────────
def compute_pbf(mol_h, conf, heavy_atoms_only=True):
    """
    计算整个分子的PBF:
    所有重原子偏离最佳拟合平面的平均距离
    """
    if heavy_atoms_only:
        atom_indices = [a.GetIdx() for a in mol_h.GetAtoms() if a.GetAtomicNum() > 1]
    else:
        atom_indices = list(range(mol_h.GetNumAtoms()))

    if len(atom_indices) < 3:
        return 0.0

    pos = conf.GetPositions()
    coords = pos[atom_indices]

    _, _, _, _ = compute_plane_deviation(coords)
    # PBF 定义为平均绝对偏离
    centered = coords - coords.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered)
    normal = Vt[-1]
    displacements = centered @ normal
    pbf = np.mean(np.abs(displacements))

    return pbf


# ── 主计算 ─────────────────────────────────────────────────────────
def compute_all_descriptors(smiles, ring_atoms):
    """
    计算单个分子的所有立体描述符
    ring_atoms: 目标环的原子索引列表 (0-based)
    """
    result = {
        'Ring_RPD': np.nan,
        'Ring_Cremer_Pople_Q': np.nan,
        'Ring_Max_Dev': np.nan,
        'Mol_PBF': np.nan,
        'NPR1': np.nan,
        'NPR2': np.nan,
        'Fsp3': np.nan,
        'Asphericity': np.nan,
        'Sphericity': np.nan,
        'Eccentricity': np.nan,
        'InertialShapeFactor': np.nan,
    }

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return result

    # 2D描述符 (不需要3D构象)
    result['Fsp3'] = rdMolDescriptors.CalcFractionCSP3(mol)

    # 生成3D构象
    best_cid, mol_h = generate_best_conformer(mol)
    if best_cid is None:
        return result

    conf = mol_h.GetConformer(best_cid)

    # ── Target ring 描述符 ──
    if ring_atoms and len(ring_atoms) >= 3:
        pos = conf.GetPositions()
        ring_coords = pos[ring_atoms]
        _, rmsd, q, max_dev = compute_plane_deviation(ring_coords)
        result['Ring_RPD'] = rmsd
        result['Ring_Cremer_Pople_Q'] = q
        result['Ring_Max_Dev'] = max_dev

    # ── Whole molecule PBF ──
    result['Mol_PBF'] = compute_pbf(mol_h, conf, heavy_atoms_only=True)

    # ── PMI-based 描述符 (RDKit, 基于含氢构象) ──
    try:
        pmi1 = rdMolDescriptors.CalcPMI1(mol_h, confId=best_cid)
        pmi2 = rdMolDescriptors.CalcPMI2(mol_h, confId=best_cid)
        pmi3 = rdMolDescriptors.CalcPMI3(mol_h, confId=best_cid)
        result['NPR1'] = pmi1 / pmi3 if pmi3 > 0 else np.nan
        result['NPR2'] = pmi2 / pmi3 if pmi3 > 0 else np.nan
    except Exception:
        pass

    # ── RDKit 内置形状描述符 ──
    try:
        result['Asphericity'] = rdMolDescriptors.CalcAsphericity(mol_h, confId=best_cid)
    except Exception:
        pass
    try:
        result['Eccentricity'] = rdMolDescriptors.CalcEccentricity(mol_h, confId=best_cid)
    except Exception:
        pass
    try:
        result['InertialShapeFactor'] = rdMolDescriptors.CalcInertialShapeFactor(mol_h, confId=best_cid)
    except Exception:
        pass

    # Sphericity: 衍生指标 (1 - Asphericity)
    if not np.isnan(result['Asphericity']):
        result['Sphericity'] = 1.0 - result['Asphericity']

    return result


def main():
    print("=" * 60)
    print("计算立体程度描述符")
    print("=" * 60)

    df = pd.read_excel(XLSX_PATH)
    print(f"读取: {len(df)} 行, {df['SMILES'].nunique()} 个唯一分子")

    # 缓存: SMILES -> 描述符 (同一SMILES不同环共享3D构象相关描述符)
    smiles_cache = {}

    # 新列初始化
    new_cols = [
        'Ring_RPD', 'Ring_Cremer_Pople_Q', 'Ring_Max_Dev',
        'Mol_PBF', 'NPR1', 'NPR2', 'Fsp3',
        'Asphericity', 'Sphericity', 'Eccentricity', 'InertialShapeFactor',
    ]
    for col in new_cols:
        df[col] = np.nan

    for idx, row in df.iterrows():
        smiles = row['SMILES']

        # 解析 Ring_Atoms
        ring_atoms_str = row['Ring_Atoms']
        try:
            if isinstance(ring_atoms_str, str):
                ring_atoms = ast.literal_eval(ring_atoms_str)
            else:
                ring_atoms = list(ring_atoms_str)
        except Exception:
            ring_atoms = []

        # 检查是否缓存了该SMILES的基础描述符
        cache_key = (smiles, tuple(sorted(ring_atoms)))
        if cache_key in smiles_cache:
            desc = smiles_cache[cache_key]
        else:
            if idx % 10 == 0:
                print(f"  进度: {idx}/{len(df)}  (SMILES: {smiles[:40]}...)")
            desc = compute_all_descriptors(smiles, ring_atoms)
            smiles_cache[cache_key] = desc

        # 写入
        for col in new_cols:
            df.at[idx, col] = desc[col]

    # 保存
    df.to_excel(OUTPUT_PATH, index=False)
    print(f"\n保存到: {OUTPUT_PATH}")
    print(f"新增 {len(new_cols)} 列: {new_cols}")

    # 打印统计摘要
    print("\n" + "=" * 60)
    print("描述符统计摘要")
    print("=" * 60)
    for col in new_cols:
        vals = df[col].dropna()
        if len(vals) > 0:
            print(f"  {col:25s}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
                  f"min={vals.min():.4f}, max={vals.max():.4f}, n={len(vals)}")
        else:
            print(f"  {col:25s}: NO DATA")

    print(f"\n完成! 共处理 {len(df)} 行, 缓存 {len(smiles_cache)} 个唯一分子-环组合")


if __name__ == '__main__':
    main()
