"""Task 5 (按协议修正): Substituent annotation feasibility.

严格要求 (来自用户协议):
  - 不要把 substituent_overlap=0 当成 "所有 substituents unseen". 标记为 unknown.
  - 尝试利用 SMARTS 自动识别 substituent (基于 Fig.3 已存在的 layer4_substituent)
  - 必须先在具有真实 sub_name 的 lunci10 上验证算法
  - 输出 coverage / accuracy / ambiguity
  - 如果无法可靠恢复, 取消 substituent seen/unseen 四分类; Fig.4b 改用:
      scaffold seen/unseen + ring-family seen/unseen + nearest-Tanimoto
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from rdkit import Chem, RDLogger


# --- Auto path bootstrap (do not remove) ---
import os as _os
_THIS_FILE = _os.path.abspath(__file__)
_d = _os.path.dirname(_THIS_FILE)
while not _os.path.exists(_os.path.join(_d, 'unified_models')) and _d != '/':
    _d = _os.path.dirname(_d)
_PROJ_ROOT = _d
# --- End auto path bootstrap ---

RDLogger.DisableLog("rdApp.*")

PROJ_ROOT = Path("_PROJ_ROOT")
CODE_END = PROJ_ROOT / "code_end"
FIG4_ROOT = PROJ_ROOT / "0901-end-code/fig4_lunci10"
AUDIT_OUT = PROJ_ROOT / "0901-end-code/results/fig4_lunci10_final/00_audit"

CLEAN_MANIFEST = AUDIT_OUT / "lunci10_clean_manifest.csv"
SEEN_MANIFEST = AUDIT_OUT / "lunci10_exact_seen_manifest.csv"

OUT_JSON = AUDIT_OUT / "substituent_annotation_feasibility.json"
OUT_CSV = AUDIT_OUT / "substituent_annotation_l10_validation.csv"

# 复用 Fig.3 的 substituent SMARTS lookup (存在则在原项目 layer4_substituent 中)
sys.path.insert(0, str(CODE_END))
try:
    from layer4_substituent.code.hammett_constants import HAMMETT_TABLE  # noqa: F401
    HAVE_LAYER4 = True
except Exception:
    HAVE_LAYER4 = False


# 常见 substituent SMARTS (足够覆盖 30 个 lunci10 sub_name)
# 优先放在这里是因为 layer4_substituent.code.substituent_smarts 不一定存在
SUBSTITUENT_SMARTS: Dict[str, str] = {
    "F": "[F]",
    "Cl": "[Cl]",
    "Br": "[Br]",
    "I": "[I]",
    "OH": "[OX2H]",
    "OMe": "[OX2][CH3]",
    "NH2": "[NX3;H2]",
    "NHMe": "[NX3;H1][CH3]",
    "NMe2": "[NX3]([CH3])[CH3]",
    "Me": "[CH3]",
    "Et": "[CH2][CH3]",
    "iPr": "[CH]([CH3])[CH3]",
    "tBu": "[C]([CH3])([CH3])[CH3]",
    "CF3": "[CX4](F)(F)F",
    "OCF3": "[OX2][CX4](F)(F)F",
    "CN": "[CX2]#[NX1]",
    "CCH": "[CX2]#[CH]",
    "CHO": "[CX3H1](=O)",
    "Ac": "[CX3](=O)[CH3]",
    "COOH": "[CX3](=O)[OX2H]",
    "COOMe": "[CX3](=O)[OX2][CH3]",
    "CONH2": "[CX3](=O)[NX3;H2]",
    "NO2": "[NX3](=O)=O",
    "SO2Me": "[SX4](=O)(=O)[CH3]",
    "SO3H": "[SX4](=O)(=O)[OX2H]",
    "SMe": "[SX2][CH3]",
    "Ph": "c1ccccc1",
    "CO2H": "[CX3](=O)[OX2H]",  # alias of COOH
    "PhNH2": "[NX3;H2]c1ccccc1",
    "BocNH": "[NX3][CX3](=O)[OX2][C]([CH3])([CH3])[CH3]",
}


def parse_target_ring(atom_str: str) -> List[int]:
    """Parse "1,2,3,4,5" -> [0,1,2,3,4] (convert from 1-indexed to 0-indexed RDKit).

    The source manifest uses 1-indexed atom positions. RDKit uses 0-indexed.
    """
    if not isinstance(atom_str, str) or not atom_str:
        return []
    try:
        return [int(x) - 1 for x in atom_str.replace("[", "").replace("]", "").split(",") if x.strip()]
    except Exception:
        return []


def match_substituent_on_ring(mol, target_ring: List[int]) -> Tuple[str, int]:
    """For each candidate SMARTS, try matching it on a single atom that is bonded
    to (but NOT inside) the target ring. Return (sub_name, atom_index) of best
    (most specific) match.

    Returns ('', -1) if no clear winner.
    """
    if not target_ring:
        return ("", -1)
    ring_set = set(target_ring)

    candidates = []
    for name, smarts in SUBSTITUENT_SMARTS.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        for match in mol.GetSubstructMatches(patt):
            # need exactly one atom of the match outside the target ring,
            # and that atom must be bonded to at least one atom in the ring.
            outside = [i for i in match if i not in ring_set]
            inside = [i for i in match if i in ring_set]
            if len(outside) == 1 and len(inside) >= 1:
                # check bonding: the outside atom must be bonded to some inside atom
                atom = mol.GetAtomWithIdx(outside[0])
                ring_neighbors = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() in ring_set]
                if ring_neighbors:
                    candidates.append((name, outside[0], len(smarts)))
    if not candidates:
        return ("", -1)
    # if multiple candidates, pick the most specific (longest SMARTS pattern)
    candidates.sort(key=lambda x: x[2], reverse=True)
    # return top1 name
    return (candidates[0][0], candidates[0][1])


def validate_on_l10(df_l10: pd.DataFrame) -> Dict[str, Any]:
    """验证算法: 在 lunci10 上 (有真实 sub_name) 检验匹配准确率."""
    n_total = 0
    n_matched = 0
    n_exact_match = 0
    n_alias_match = 0  # 如 COOH vs CO2H
    n_ambiguous = 0
    per_sub: Dict[str, Dict[str, int]] = {}

    for _, row in df_l10.iterrows():
        smi = row.get("raw_smiles") or row.get("canonical_smiles")
        target = parse_target_ring(row.get("target_ring_atoms", ""))
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        true_sub = row.get("sub_name", "")
        if mol is None or not target or not true_sub:
            continue
        n_total += 1
        pred_sub, _ = match_substituent_on_ring(mol, target)
        per_sub.setdefault(true_sub, {"n": 0, "matched": 0, "alias": 0})
        per_sub[true_sub]["n"] += 1
        if not pred_sub:
            n_ambiguous += 1
            continue
        n_matched += 1
        if pred_sub == true_sub:
            n_exact_match += 1
            per_sub[true_sub]["matched"] += 1
        elif pred_sub in {"CO2H", "COOH"} and true_sub in {"CO2H", "COOH"}:
            n_alias_match += 1
            per_sub[true_sub]["alias"] += 1

    coverage = n_matched / n_total if n_total else 0
    exact_acc = n_exact_match / n_total if n_total else 0
    alias_acc = (n_exact_match + n_alias_match) / n_total if n_total else 0

    return {
        "n_total": n_total,
        "n_matched": n_matched,
        "n_exact_match": n_exact_match,
        "n_alias_match": n_alias_match,
        "n_ambiguous": n_ambiguous,
        "coverage": coverage,
        "exact_accuracy": exact_acc,
        "effective_accuracy_with_alias": alias_acc,
        "per_sub_name": per_sub,
    }


def main() -> None:
    print("[load] clean + seen manifests for full coverage...")
    df_clean = pd.read_csv(CLEAN_MANIFEST)
    df_seen = pd.read_csv(SEEN_MANIFEST)
    df_all = pd.concat([df_clean, df_seen], ignore_index=True)
    print(f"  rows: {len(df_all)}")

    print("[validate] SMARTS-based substituent recognition on lunci10 (with true sub_name)...")
    val = validate_on_l10(df_all)
    print(f"  total: {val['n_total']}")
    print(f"  matched: {val['n_matched']}")
    print(f"  exact accuracy: {val['exact_accuracy']:.3f}")
    print(f"  effective accuracy (with COOH/CO2H alias): {val['effective_accuracy_with_alias']:.3f}")

    # decision
    reliable = val["effective_accuracy_with_alias"] >= 0.85 and val["coverage"] >= 0.85
    decision = "RELIABLE: can use SMARTS-recognised substituents for A/B/C/D categories" if reliable \
        else "UNRELIABLE: cancel A/B/C/D seen/unseen substituent categories; use scaffold/ring-family/nearest-Tanimoto instead"

    out = {
        "protocol": "SMARTS_recognition_validated_on_l10",
        "internal_data_substituent_annotation": "NOT_PRESENT_in_internal_training_files",
        "previous_status_substituent_seen": "UNKNOWN (not false)",
        "l10_validation": val,
        "decision": decision,
        "reliable": reliable,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {OUT_JSON}")

    # 输出每个分子的 per-row substituent recognition
    rows = []
    for _, row in df_all.iterrows():
        smi = row.get("raw_smiles") or row.get("canonical_smiles")
        target = parse_target_ring(row.get("target_ring_atoms", ""))
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        pred, idx = match_substituent_on_ring(mol, target) if mol and target else ("", -1)
        rows.append({
            "sample_id": row.get("sample_id"),
            "true_sub_name": row.get("sub_name"),
            "pred_sub_name": pred,
            "match": pred == row.get("sub_name"),
            "in_exact_seen_set": row.get("sample_id") in set(df_seen["sample_id"].astype(str).tolist()),
        })
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"  wrote {OUT_CSV}")
    print()
    print("DECISION:", decision)


if __name__ == "__main__":
    main()