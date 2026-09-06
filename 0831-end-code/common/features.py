"""
分子指纹 + 描述符特征提取 (Stage1 传统ML基线使用)

特征组合:
  1. MACCS (167 bit)
  2. Morgan (2048 bit, radius=2)
  3. 分子描述符 (16维)
  4. 环描述符 (13维) — 目标环局部表示

[0831 重构] 增加 feature_mode 双轨:
  - 'standard'                     : 保留所有特征 (含显式芳香性)
  - 'explicit_aromaticity_ablated': 移除以下显式芳香性 flag/count:
                                     - 分子描述符中的 n_aromatic_rings
                                     - 环描述符中的 ring_n_aromatic
                                     注: Morgan / MACCS 指纹本身依赖 RDKit 原子的芳香性判据,
                                        仅通过移除 handcrafted 显式芳香性特征来减弱依赖;
                                        不声称完全 aromaticity-free (指纹仍有隐式依赖)。
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
MOL_DESC_DIM_AROM_ABLATED = 15   # 移除 n_aromatic_rings
RING_DESC_DIM = 13
RING_DESC_DIM_AROM_ABLATED = 12  # 移除 ring_n_aromatic
FEAT_DIM = MACCS_DIM + MORGAN_BITS + MOL_DESC_DIM + RING_DESC_DIM          # 2244
FEAT_DIM_AROM_ABLATED = MACCS_DIM + MORGAN_BITS + MOL_DESC_DIM_AROM_ABLATED + RING_DESC_DIM_AROM_ABLATED  # 2242

# feature_mode -> 指纹生成方式 (写进 metadata, 不声称指纹完全 aromaticity-free)
FINGERPRINT_GENERATION = {
    'standard':                     'RDKit MACCSkeys + Morgan(r=2, nBits=2048) on aromaticity-aware atoms',
    'explicit_aromaticity_ablated': 'RDKit MACCSkeys + Morgan(r=2, nBits=2048) on aromaticity-aware atoms; '
                                     'handcrafted explicit aromaticity flags/counts removed '
                                     '(n_aromatic_rings, ring_n_aromatic). '
                                     'NOTE: standard RDKit fingerprints retain aromaticity perception '
                                     '(MACCS keys and Morgan bits reflect RDKit aromatic atom flags). '
                                     'This mode ablates only the handcrafted explicit descriptors, not '
                                     'the fingerprint bits; no Kekulization beyond RDKit default.',
}

MOL_DESC_COLS = [
    'mw', 'logp', 'tpsa', 'mr', 'n_heavy', 'n_bonds', 'n_rings',
    'n_aromatic_rings', 'n_aliphatic_rings', 'n_heteroatoms',
    'n_rotatable', 'n_hbd', 'n_hba', 'fsp3', 'qed', 'bertz',
]
MOL_DESC_COLS_AROM_ABLATED = [c for c in MOL_DESC_COLS if c != 'n_aromatic_rings']

RING_DESC_COLS = [
    'ring_size', 'ring_n_C', 'ring_n_N', 'ring_n_O', 'ring_n_S',
    'ring_n_aromatic', 'ring_n_aliphatic', 'n_ring_atoms',
    'ring_avg_degree', 'ring_max_degree', 'ring_n_substituents',
    'target_ring_flag', 'target_ring_ratio',
]
RING_DESC_COLS_AROM_ABLATED = [c for c in RING_DESC_COLS if c != 'ring_n_aromatic']

FEATURE_MODE_DIM = {
    'standard': FEAT_DIM,
    'explicit_aromaticity_ablated': FEAT_DIM_AROM_ABLATED,
}


def get_feature_meta(feature_mode):
    """返回 (feature_mode, fingerprint_generation_note, n_features)。"""
    if feature_mode not in FEATURE_MODE_DIM:
        raise ValueError(f"未知 feature_mode: {feature_mode}, 可选: {list(FEATURE_MODE_DIM)}")
    return {
        'feature_mode': feature_mode,
        'fingerprint_generation': FINGERPRINT_GENERATION[feature_mode],
        'n_features': FEATURE_MODE_DIM[feature_mode],
        'explicit_aromaticity_columns_removed': (
            ['n_aromatic_rings', 'ring_n_aromatic']
            if feature_mode == 'explicit_aromaticity_ablated' else []
        ),
    }


def smiles_to_fingerprint(smiles):
    """SMILES -> MACCS + Morgan 拼接指纹 (1D numpy 数组)

    注: 此函数与 feature_mode 无关, 因为 fingerprint 本身依赖 RDKit 默认芳香性判据。
    """
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
        DataStructs.ConvertToNumpyArray(bit_vec, morgan)
    except Exception:
        pass

    return np.concatenate([maccs, morgan])


def compute_molecular_descriptors(smiles_list, feature_mode='standard'):
    """计算 RDKit 分子描述符。
    当 feature_mode='explicit_aromaticity_ablated' 时移除 n_aromatic_rings 列。
    """
    n_dim = FEATURE_MODE_DIM[feature_mode] - MACCS_DIM - MORGAN_BITS - (
        RING_DESC_DIM if feature_mode == 'standard' else RING_DESC_DIM_AROM_ABLATED)
    include_arom = (feature_mode == 'standard')

    features = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            features.append(np.zeros(n_dim, dtype=np.float32))
            continue
        base = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.MolMR(mol),
            mol.GetNumHeavyAtoms(),
            mol.GetNumBonds(),
            rdMolDescriptors.CalcNumRings(mol),
        ]
        if include_arom:
            base.append(rdMolDescriptors.CalcNumAromaticRings(mol))
        base += [
            rdMolDescriptors.CalcNumAliphaticRings(mol),
            Descriptors.NumHeteroatoms(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
            Descriptors.qed(mol),
            Descriptors.BertzCT(mol),
        ]
        features.append(np.array(base, dtype=np.float32))
    return np.array(features)


def compute_ring_descriptors(df, feature_mode='standard'):
    """计算目标环描述符。
    当 feature_mode='explicit_aromaticity_ablated' 时移除 ring_n_aromatic 列。
    """
    include_arom = (feature_mode == 'standard')
    n_dim = RING_DESC_DIM if include_arom else RING_DESC_DIM_AROM_ABLATED
    n = len(df)
    features = np.zeros((n, n_dim), dtype=np.float32)

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

        target_ring_atoms = [idx - 1 for idx in atom_on_ring if isinstance(idx, (int, float)) and idx > 0]

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

            base = [ring_size, n_C, n_N, n_O, n_S]
            if include_arom:
                base.append(n_arom)
            base += [
                len(ring_atoms) - n_arom,
                len(ring_atoms),
                avg_deg, max_deg, n_sub,
                len(target_ring_atoms) * 10.0,
                len(target_ring_atoms) / max(mol.GetNumAtoms(), 1),
            ]
        else:
            base = [ring_size, 0, 0, 0, 0]
            if include_arom:
                base.append(0)
            base += [0, 0, 0, 0, 0,
                     len(target_ring_atoms) * 10.0,
                     len(target_ring_atoms) / max(mol.GetNumAtoms(), 1)]

        features[i] = np.array(base, dtype=np.float32)

    return features


def build_fingerprint_matrix(smiles_list, df=None, raise_on_invalid=True,
                            feature_mode='standard'):
    """批量构建特征矩阵 (含 feature_mode 双轨)

    Returns:
        X:     (n, n_features)
        valid: (n,) 有效 mask
    """
    if feature_mode not in FEATURE_MODE_DIM:
        raise ValueError(f"未知 feature_mode: {feature_mode}, 可选: {list(FEATURE_MODE_DIM)}")

    n = len(smiles_list)
    fp_dim = MACCS_DIM + MORGAN_BITS
    fp_mat = np.zeros((n, fp_dim), dtype=np.float32)
    valid = np.ones(n, dtype=bool)
    for i, smi in enumerate(smiles_list):
        fp = smiles_to_fingerprint(smi)
        if fp.sum() == 0:
            valid[i] = False
        fp_mat[i] = fp

    if raise_on_invalid and not valid.all():
        invalid_idx = np.where(~valid)[0]
        raise ValueError(
            f"发现 {len(invalid_idx)} 个无效 SMILES (索引: {invalid_idx.tolist()}). "
            f"为保证四阶段划分一致, 此处 raise 而非静默过滤。"
        )

    mol_desc = compute_molecular_descriptors(smiles_list, feature_mode=feature_mode)
    if df is not None:
        ring_desc = compute_ring_descriptors(df, feature_mode=feature_mode)
    else:
        ring_desc = np.zeros((n, FEATURE_MODE_DIM[feature_mode] - fp_dim - mol_desc.shape[1]),
                             dtype=np.float32)

    X = np.concatenate([fp_mat, mol_desc, ring_desc], axis=1)
    return X, valid
