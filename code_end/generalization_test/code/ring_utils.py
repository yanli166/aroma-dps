"""
环信息工具: 骨架(Scaffold)提取 + 芳香环类型识别

- Murcko骨架: 用 RDKit 提取分子的 Bemis-Murcko 骨架, 用于骨架拆分
- 芳香环类型: 提取分子的芳香核心 (融合芳香环系统), 映射为可读名称
  (如 benzene, pyrrole, thiophene, indole, benzothiophene, ...)
"""
import re
from collections import Counter
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


# ---- 常见芳香环 SMARTS 模式 (按融合度从大到小排列, 优先匹配) ----
RING_SMARTS = [
    # 双环融合
    ('indole',        'c1ccc2[nH]ccc2c1'),
    ('benzofuran',    'c1ccc2ccoc2c1'),
    ('benzothiophene','c1ccc2ccsc2c1'),
    ('benzimidazole', 'c1ccc2[nH]cnc2c1'),
    ('benzoxazole',   'c1ccc2ocnc2c1'),
    ('benzothiazole', 'c1ccc2scnc2c1'),
    ('naphthalene',   'c1ccc2ccccc2c1'),
    ('quinoline',     'c1ccc2ncccc2c1'),
    ('isoquinoline',  'c1ccc2ccncc2c1'),
    ('quinazoline',   'c1ccc2ncncn2c1'),
    ('cinnoline',     'c1ccc2nnccc2c1'),
    ('phthalazine',   'c1ccc2nzncc2c1'.replace('z', 'n')),  # naphthyridine-like
    ('purine',        'c1nc2c(c1)[nH]cnc2'),
    # 单环 6 元
    ('benzene',       'c1ccccc1'),
    ('pyridine',      'c1ccncc1'),
    ('pyrimidine',    'c1cncnc1'),
    ('pyrazine',      'c1cnccn1'),
    ('pyridazine',    'c1ccccn1'.replace('c1ccccn1', 'c1ccnnn1')),  # fix
    # 单环 5 元
    ('pyrrole',       'c1cc[nH]c1'),
    ('furan',         'c1ccoc1'),
    ('thiophene',     'c1ccsc1'),
    ('imidazole',     'c1c[nH]cn1'),
    ('oxazole',       'c1cncn1'.replace('c1cncn1', 'c1cncoc1')),
    ('thiazole',      'c1cncsc1'),
    ('pyrazole',      'c1c[nH]c[nH]1'),
    ('isoxazole',     'c1cncoc1'.replace('c1cncoc1', 'c1ccnoc1')),
    ('isothiazole',   'c1ccnsc1'),
    ('triazole',      'c1nncnn1'),
]
# 预编译为 mol pattern
_RING_PATTERNS = []
for name, smarts in RING_SMARTS:
    patt = Chem.MolFromSmarts(smarts)
    if patt is not None:
        _RING_PATTERNS.append((name, patt))


def get_murcko_scaffold(smiles):
    """提取 Bemis-Murcko 骨架 SMILES (去除取代基, 保留核心环框架)"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 'INVALID'
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return 'INVALID'


def get_aromatic_core(smiles):
    """提取芳香核心: 所有芳香原子 + 它们之间的键构成的子图"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ''
        aromatic_atoms = set(a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic())
        if not aromatic_atoms:
            return 'non_aromatic'
        # 提取芳香子图
        submol = Chem.PathToSubmol(mol, list(aromatic_atoms))
        return Chem.MolToSmiles(submol, canonical=True)
    except Exception:
        return 'unknown'


def identify_ring_type(smiles):
    """识别芳香环类型 (返回可读名称)

    优先匹配 SMARTS 模式; 若无匹配, 回退到芳香核心 SMILES。
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 'invalid'
        # 检查是否有芳香原子
        has_aromatic = any(a.GetIsAromatic() for a in mol.GetAtoms())
        if not has_aromatic:
            return 'non_aromatic'
        # 按 SMARTS 优先级匹配
        for name, patt in _RING_PATTERNS:
            if mol.HasSubstructMatch(patt):
                return name
        # 回退: 用芳香核心 SMILES 作为类型
        return get_aromatic_core(smiles)
    except Exception:
        return 'unknown'


def get_scaffold_distribution(smiles_list):
    """统计骨架分布"""
    scaffolds = [get_murcko_scaffold(s) for s in smiles_list]
    return Counter(scaffolds)


def get_ring_type_distribution(smiles_list):
    """统计芳香环类型分布"""
    types = [identify_ring_type(s) for s in smiles_list]
    return Counter(types)
