"""
[P0 重构] 正式协议 split 管理器 (docs: 参考思路2.txt)

1. 固定 20% final holdout (split seed === 2026, 与 model seed 解耦)
2. 母分子分组 -> canonical SMILES (isomericSmiles=True)
3. 80% dev 内部再做 Group 5-fold CV
4. 自动泄漏检查: train/test 与 train/val 的分子分组交集必须为空
5. split 持久化到 splits/
"""
import os
import json
import numpy as np
from rdkit import Chem
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from common.constants import LAST_END_ROOT

SPLIT_SEED      = 2026
MODEL_SEEDS     = [11, 22, 33, 44, 55]
TEST_SIZE_RATIO = 0.20
N_FOLDS         = 5
SPLITS_DIR      = os.path.join(LAST_END_ROOT, 'splits')

_CACHE = {}


def canonicalize_smiles(smi):
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        raise ValueError(f"[protocol] 非法 SMILES: {smi}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def make_group_ids(smiles_list, use_inchikey=False):
    gs = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            raise ValueError(f"[protocol] 非法 SMILES: {smi}")
        gs.append(Chem.MolToInchiKey(Chem.AddHs(mol)) if use_inchikey
                  else Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    return np.asarray(gs)


def assert_no_leak(idxa, idxb, groups, tag=""):
    a = set(groups[idxa]); b = set(groups[idxb])
    inter = a & b
    if inter:
        raise RuntimeError(f"[protocol] {tag} 分子泄漏: {len(inter)} 个分子同时出现在两侧")


def make_final_holdout(n_samples, groups, seed=SPLIT_SEED, persist=None, tag='default'):
    """固定 group-aware 80/20。返回 dict{train_idx, test_idx}。

    Args:
        tag: 标识本次划分的语义 (e.g. 'explicit_aromaticity_ablated'), 用作持久化文件名后缀,
             避免不同 feature_mode 互相覆盖。
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE_RATIO, random_state=seed)
    dev, test = next(gss.split(np.arange(n_samples), groups=np.asarray(groups)))
    out = {"train_idx": np.asarray(dev), "test_idx": np.asarray(test)}
    if persist:
        os.makedirs(persist, exist_ok=True)
        meta = {"split_seed": seed, "n_total": int(n_samples),
                "n_dev": int(len(dev)), "n_test": int(len(test)),
                "tag": tag}
        meta_path = os.path.join(persist, f"final_holdout_meta_{tag}.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
    return out


def make_inner_group_cv(dev_idx, groups, n_splits=N_FOLDS):
    gkf = GroupKFold(n_splits=n_splits)
    folds = []
    for tr, va in gkf.split(dev_idx, groups=groups[dev_idx]):
        folds.append((dev_idx[tr], dev_idx[va]))
    return folds


def get_final_splits(n_samples, groups, split_seed=SPLIT_SEED, persist=SPLITS_DIR, tag='default'):
    """总入口 (固定, 与 model seed 无关): dev/test + dev 内 5 折。

    Args:
        tag: 语义标识 (e.g. 'explicit_aromaticity_ablated'), 持久化到 split_manifest_<tag>.json。
    """
    h = make_final_holdout(n_samples, groups=groups, seed=split_seed, persist=persist, tag=tag)
    dev, test = h["train_idx"], h["test_idx"]
    assert_no_leak(dev, test, groups, "final 80/20")
    folds = make_inner_group_cv(dev, groups)
    for k, (tr, va) in enumerate(folds):
        assert_no_leak(tr, va, groups, f"fold{k+1}")
    if persist:
        os.makedirs(persist, exist_ok=True)
        rec = {"split_seed": int(split_seed), "tag": tag,
               "train_idx": dev.tolist(), "test_idx": test.tolist(),
               "folds": [[tr.tolist(), va.tolist()] for tr, va in folds]}
        with open(os.path.join(persist, f"split_manifest_{tag}.json"), "w") as f:
            json.dump(rec, f, indent=2)
    return {"train_idx": dev, "test_idx": test, "folds": folds,
            "meta": {"n_dev": int(len(dev)), "n_test": int(len(test)), "tag": tag,
                     "split_seed": int(split_seed)}}


def completeness_check(index_set, expected_index, label):
    """expected_index 是 pd.MultiIndex(或 set of tuples)；缺则 raise。"""
    actual = set(map(tuple, index_set))
    expected = set(map(tuple, expected_index))
    missing = expected.difference(actual)
    if missing:
        raise RuntimeError(f"[protocol] {label} 缺失 {len(missing)} 个 run:\n"
                           + "\n".join(sorted(map(str, missing))))
    return True