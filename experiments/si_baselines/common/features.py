"""
分子指纹 + 描述符特征提取 (第一层传统ML基线使用) — 修复版

修复:
  - C1: build_fingerprint_matrix 增加 raise_on_invalid 参数,
    默认 True 时遇到无效 SMILES 直接 raise (与 Layer 2/3 一致),
    保证三层在相同索引空间做划分, 消除划分不一致隐患。

特征组合 (label编码模式):
  1. MACCS (167 bit)
  2. Morgan (2048 bit, radius=2)
  3. 分子描述符 (16维)
  4. 环描述符 (13维)
"""
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, rdMolDescriptors
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

MACCS_DIM = 167
MORGAN_BITS = 2048
MORGAN_RADIUS = 2
MOL_DESC_DIM = 16
RING_DESC_DIM = 13
FEAT_DIM = MACCS_DIM + MORGAN_BITS + MOL_DESC_DIM + RING_DESC_DIM  # 2244

MOL_DESC_COLS = [
    'mw', 'logp', 'tpsa', 'mr', 'n_heavy', 'n_bonds', 'n_rings',
    'n_aromatic_rings', 'n_aliphatic_rings', 'n_heteroatoms',
    'n_rotatable', 'n_hbd', 'n_hba', 'fsp3', 'qed', 'bertz',
]
RING_DESC_COLS = [
    'ring_size', 'ring_n_C', 'ring_n_N', 'ring_n_O', 'ring_n_S',
    'ring_n_aromatic', 'ring_n_aliphatic', 'n_ring_atoms',
    'ring_avg_degree', 'ring_max_degree', 'ring_n_substituents',
    'target_ring_flag', 'target_ring_ratio',
]


def smiles_to_fingerprint(smiles):
    """SMILES -> MACCS + Morgan 拼接指纹 (1D numpy 数组)"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(MACCS_DIM + MORGAN_BITS, dtype=np.float32)

    maccs = np.zeros(MACCS_DIM, dtype=np.float32)
    maccs_keys = MACCSkeys.GenMACCSKeys(mol)
    for i in range(MACCS_DIM):
        maccs[i] = maccs_keys.GetBit(i + 1) if (i + 1) < maccs_keys.GetNumBits() else 0

    morgan = np.zeros(MORGAN_BITS, dtype=np.float32)
    try:
        bit_vec = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_BITS)
        ConvertToNumpyArray(bit_vec, morgan)
    except Exception:
        pass

    return np.concatenate([maccs, morgan])


def compute_molecular_descriptors(smiles_list):
    """计算 RDKit 分子描述符 (16维)"""
    features = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            features.append(np.zeros(MOL_DESC_DIM, dtype=np.float32))
            continue
        feat = np.array([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.MolMR(mol),
            mol.GetNumHeavyAtoms(),
            mol.GetNumBonds(),
            rdMolDescriptors.CalcNumRings(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcNumAliphaticRings(mol),
            Descriptors.NumHeteroatoms(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
            Descriptors.qed(mol),
            Descriptors.BertzCT(mol),
        ], dtype=np.float32)
        features.append(feat)
    return np.array(features)


def compute_ring_descriptors(df):
    """计算目标环描述符 (13维) - label编码在ML中的体现"""
    n = len(df)
    features = np.zeros((n, RING_DESC_DIM), dtype=np.float32)

    for i, (_, row) in enumerate(df.iterrows()):
        smiles = row['smiles']
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        atom_on_ring = row.get('atom_on_ring', [])
        if isinstance(atom_on_ring, str):
            try:
                atom_on_ring = eval(atom_on_ring)
            except Exception:
                atom_on_ring = []
        if not isinstance(atom_on_ring, (list, tuple)):
            atom_on_ring = []

        ring_size = row.get('Ring_Size', 0)
        ring_id = row.get('Ring_ID', 1)

        # atom_on_ring is 0-based (verified by P0-2 audit across 20,605 rows)
        target_ring_atoms = [int(idx) for idx in atom_on_ring if isinstance(idx, (int, float)) and 0 <= idx < mol.GetNumAtoms()]

        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()

        matching_rings = [r for r in atom_rings if len(r) == ring_size] if atom_rings else []

        if matching_rings:
            idx = min(int(ring_id) - 1, len(matching_rings) - 1) if len(matching_rings) > 0 else 0
            ring = matching_rings[max(0, idx)]
            ring_atoms = [mol.GetAtomWithIdx(i) for i in ring]

            n_C = sum(1 for a in ring_atoms if a.GetAtomicNum() == 6)
            n_N = sum(1 for a in ring_atoms if a.GetAtomicNum() == 7)
            n_O = sum(1 for a in ring_atoms if a.GetAtomicNum() == 8)
            n_S = sum(1 for a in ring_atoms if a.GetAtomicNum() == 16)
            n_arom = sum(1 for a in ring_atoms if a.GetIsAromatic())

            degrees = [a.GetDegree() for a in ring_atoms]
            avg_deg = float(np.mean(degrees)) if degrees else 0
            max_deg = float(max(degrees)) if degrees else 0
            n_sub = sum(1 for d in degrees if d > 2)

            features[i] = np.array([
                ring_size, n_C, n_N, n_O, n_S,
                n_arom, len(ring_atoms) - n_arom, len(ring_atoms),
                avg_deg, max_deg, n_sub,
                len(target_ring_atoms) * 10.0,
                len(target_ring_atoms) / max(mol.GetNumAtoms(), 1),
            ], dtype=np.float32)
        else:
            features[i] = np.array([
                ring_size, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                len(target_ring_atoms) * 10.0,
                len(target_ring_atoms) / max(mol.GetNumAtoms(), 1),
            ], dtype=np.float32)

    return features


def build_fingerprint_matrix(smiles_list, df=None, raise_on_invalid=True):
    """批量构建特征矩阵 (label编码模式)

    C1 修复: 默认 raise_on_invalid=True, 遇到无效 SMILES 直接 raise,
    保证与 Layer 2/3 (process_and_save_data 也会 raise) 行为一致,
    从而三层在完全相同的索引空间做 canonical_splits。

    Args:
        smiles_list: SMILES列表
        df: 包含 atom_on_ring/Ring_Size/Ring_ID 列的DataFrame
        raise_on_invalid: True=遇到无效SMILES报错(默认, 保证三层一致);
                          False=静默返回 valid mask (向后兼容)

    Returns: (X, valid_mask)  X shape=(n, 2244)
    """
    n = len(smiles_list)
    fp_dim = MACCS_DIM + MORGAN_BITS
    fp_mat = np.zeros((n, fp_dim), dtype=np.float32)
    valid = np.ones(n, dtype=bool)
    for i, smi in enumerate(smiles_list):
        fp = smiles_to_fingerprint(smi)
        if fp.sum() == 0:
            valid[i] = False
        fp_mat[i] = fp

    # C1: 检查无效样本, 与 Layer 2/3 一致地 raise
    if raise_on_invalid and not valid.all():
        invalid_idx = np.where(~valid)[0]
        raise ValueError(
            f"发现 {len(invalid_idx)} 个无效 SMILES (索引: {invalid_idx.tolist()}). "
            f"Layer 2/3 的 process_and_save_data 也会对此报错, "
            f"为保证三层划分一致, 此处同样 raise 而非静默过滤。"
        )

    mol_desc = compute_molecular_descriptors(smiles_list)

    if df is not None:
        ring_desc = compute_ring_descriptors(df)
    else:
        ring_desc = np.zeros((n, RING_DESC_DIM), dtype=np.float32)

    X = np.concatenate([fp_mat, mol_desc, ring_desc], axis=1)
    return X, valid
