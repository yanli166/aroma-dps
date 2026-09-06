"""Phase 8 (internal pair feasibility check): 检查内部训练集
(collet_homa_0716.csv / collet_mbco_0716.csv / collet_nics_0716.csv) 是否具备
scaffold/Ring_ID, ring_pos, substituent identity, matched substituent series
四个关键信息。

判定逻辑:
  - 若同时具备 (scaffold/Ring_ID) + (ring_pos) + (substituent identity) + (matched
    substituent series across contexts), 则标记 Protocol I (strict external
    zero-shot) 可行, 并生成 internal_pair_manifest.csv。
  - 否则标记 Protocol II (within-lunci10 adaptation, NOT strict external), 即
    只能在 internal CV 选择模型然后一次性测试 lunci10 pairs.

产物:
  - internal_pair_manifest.csv   (Protocol I 可行时存在)
  - internal_pair_feasibility.json  (协议判定与覆盖率统计)
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

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
from rdkit.Chem import MurckoScaffold  # noqa: E402

PROJ_ROOT = "_PROJ_ROOT"
CODE_END = f"{PROJ_ROOT}/code_end"
for p in (PROJ_ROOT, CODE_END):
    if Path(p).exists() and p not in sys.path:
        sys.path.insert(0, p)

FIG4_ROOT = Path("_PROJ_ROOT/0901-end-code/fig4_lunci10")
OUT_DIR = Path("_PROJ_ROOT/0901-end-code/results/fig4_lunci10_final/03_pairwise")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FEASIBILITY_JSON = OUT_DIR / "internal_pair_feasibility.json"
INTERNAL_PAIR_CSV = OUT_DIR / "internal_pair_manifest.csv"

INTERNAL_FILES = {
    "HOMA": Path("_PROJ_ROOT + "/code_end"/data1_end/collet_homa_0716.csv"),
    "MBCO": Path("_PROJ_ROOT + "/code_end"/data1_end/collet_mbco_0716.csv"),
    "NICS_1zz": Path("_PROJ_ROOT + "/code_end"/data1_end/collet_nics_0716.csv"),
}

REQUIRED_COLS = {
    "scaffold": ["scaffold", "Scaffold", "murcko", "MurckoScaffold", "Ring_ID"],
    "ring_pos": ["ring_pos", "Ring_Position", "ring_position", "position"],
    "substituent": ["substituent", "Substituent", "sub_name", "sub_type"],
}


def _smiles_canonical(s: str) -> Optional[str]:
    if not isinstance(s, str) or not s:
        return None
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m is not None else None


def _murcko(smi: str) -> Optional[str]:
    if not smi:
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    sc = MurckoScaffold.GetScaffoldForMol(m)
    return Chem.MolToSmiles(sc) if sc and sc.GetNumAtoms() else ""


def _first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _coverage_report(df: pd.DataFrame, file_path: Path) -> Dict[str, Any]:
    """对单个内部 csv 返回列覆盖率与对齐情况."""
    out: Dict[str, Any] = {
        "file": str(file_path),
        "n_rows": int(len(df)),
        "columns": list(df.columns),
    }
    for key, cands in REQUIRED_COLS.items():
        col = _first_present(df, cands)
        if col is None:
            out[f"has_{key}"] = False
            out[f"{key}_col"] = None
            out[f"{key}_non_null"] = 0
        else:
            non_null = int(df[col].notna().sum())
            out[f"has_{key}"] = non_null > 0
            out[f"{key}_col"] = col
            out[f"{key}_non_null"] = non_null
    # smiles 列
    smi_col = _first_present(df, ["canonical_smiles", "smiles", "SMILES", "mol"])
    out["smiles_col"] = smi_col
    out["smiles_non_null"] = int(df[smi_col].notna().sum()) if smi_col else 0
    return out


def _series_match_check(dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """检查是否存在 matched substituent series — 即同一 (scaffold, ring_pos)
    下, 多于 1 个不同 substituent 出现, 且 ≥2 个 substituent 共享出现在多个
    上下文里 (允许 cross-context Hammett 分析)."""
    smi_col_map: Dict[str, str] = {}
    for task, df in dfs.items():
        col = _first_present(df, ["canonical_smiles", "smiles", "SMILES", "mol"])
        if col:
            smi_col_map[task] = col
    if not smi_col_map:
        return {"has_matched_series": False, "reason": "no SMILES column in any internal file"}

    # 收集所有 (scaffold, ring_pos, sub_name, task) → count
    contexts: Dict[tuple, Set[str]] = defaultdict(set)
    sub_global: Set[str] = set()
    for task, df in dfs.items():
        sc_col = _first_present(df, ["scaffold", "Scaffold", "murcko", "MurckoScaffold"])
        rp_col = _first_present(df, ["ring_pos", "Ring_Position", "ring_position", "position"])
        sb_col = _first_present(df, ["substituent", "Substituent", "sub_name", "sub_type"])
        smi_col = smi_col_map[task]
        for _, row in df.iterrows():
            smi = row.get(smi_col)
            if pd.isna(smi):
                continue
            sc = row.get(sc_col) if sc_col else _murcko(str(smi))
            rp = row.get(rp_col) if rp_col else "unknown"
            sb = row.get(sb_col) if sb_col else ""
            if pd.isna(sc) or pd.isna(rp) or pd.isna(sb):
                continue
            contexts[(str(sc), str(rp), str(sb), task)].add(str(smi))
            sub_global.add(str(sb))

    # 对每个 (scaffold, ring_pos) 检查: 至少 2 个不同 sub_name
    ctx_subs: Dict[tuple, Set[str]] = defaultdict(set)
    for (sc, rp, sb, task), _ in contexts.items():
        ctx_subs[(sc, rp)].add(sb)
    n_contexts_multi_sub = sum(1 for _, subs in ctx_subs.items() if len(subs) >= 2)
    n_total_contexts = len(ctx_subs)

    has_series = n_contexts_multi_sub >= 3  # ≥3 contexts with ≥2 subs ⇒ 可做配对
    return {
        "has_matched_series": bool(has_series),
        "n_total_contexts": int(n_total_contexts),
        "n_contexts_with_>=2_substituents": int(n_contexts_multi_sub),
        "n_unique_substituents_global": int(len(sub_global)),
        "reason": (
            "" if has_series
            else f"only {n_contexts_multi_sub}/{n_total_contexts} contexts have >=2 substituents"
        ),
    }


def _build_internal_pair_manifest(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """若 Protocol I 可行, 构造 internal_pair_manifest.csv.

    Schema: task, ring_name (= scaffold), ring_pos, target_ring_id, sub_type,
            sub_i, sub_j, sample_id_i, sample_id_j, canonical_smiles_i,
            canonical_smiles_j, delta_HOMA_true (or MBCO / NICS_ZZ)
    """
    rows: List[Dict[str, Any]] = []
    for task, df in dfs.items():
        sc_col = _first_present(df, ["scaffold", "Scaffold", "murcko", "MurckoScaffold", "Ring_ID"])
        rp_col = _first_present(df, ["ring_pos", "Ring_Position", "ring_position", "position"])
        sb_col = _first_present(df, ["substituent", "Substituent", "sub_name", "sub_type"])
        smi_col = smi_col_map = _first_present(df, ["canonical_smiles", "smiles", "SMILES", "mol"])
        sid_col = _first_present(df, ["New_ID", "id", "sample_id", "name"])
        # descriptor column
        truth_col_map = {"HOMA": "homa_value", "MBCO": "mbco_value",
                         "NICS_1zz": "nics_value"}
        truth_col = None
        for c in (truth_col_map[task], task, task.upper(), task.lower()):
            if c in df.columns:
                truth_col = c
                break
        if any(v is None for v in (sc_col, rp_col, sb_col, smi_col, truth_col)):
            continue
        # per-context grouping
        for keys, sub in df.groupby([sc_col, rp_col, sb_col], dropna=True):
            sc, rp, sb = keys
            # canonical orientation by sb (stable order)
            sub_sorted = sub.sort_values(smi_col, kind="stable").reset_index(drop=True)
            n = len(sub_sorted)
            for i in range(n):
                for j in range(i + 1, n):
                    ri = sub_sorted.iloc[i]
                    rj = sub_sorted.iloc[j]
                    vi = pd.to_numeric(pd.Series([ri.get(truth_col)]), errors="coerce").iloc[0]
                    vj = pd.to_numeric(pd.Series([rj.get(truth_col)]), errors="coerce").iloc[0]
                    if pd.isna(vi) or pd.isna(vj):
                        continue
                    rows.append({
                        "task": task,
                        "ring_name": str(sc),
                        "ring_pos": str(rp),
                        "sub_type": "internal",
                        "sub_i": str(ri.get(sb_col, "")),
                        "sub_j": str(rj.get(sb_col, "")),
                        "sample_id_i": str(ri.get(sid_col, "")) if sid_col else "",
                        "sample_id_j": str(rj.get(sid_col, "")) if sid_col else "",
                        "canonical_smiles_i": str(ri.get(smi_col, "")),
                        "canonical_smiles_j": str(rj.get(smi_col, "")),
                        f"delta_{task}_true": float(vi) - float(vj),
                    })
    return pd.DataFrame(rows)


def main(verbose: bool = True) -> Dict[str, Any]:
    coverage: Dict[str, Dict[str, Any]] = {}
    dfs: Dict[str, pd.DataFrame] = {}
    for task, p in INTERNAL_FILES.items():
        if not p.is_file():
            coverage[task] = {"file": str(p), "exists": False, "n_rows": 0,
                              "has_scaffold": False, "has_ring_pos": False,
                              "has_substituent": False, "has_matched_series": False}
            continue
        df = pd.read_csv(p)
        dfs[task] = df
        report = _coverage_report(df, p)
        coverage[task] = report

    series_report = _series_match_check(dfs)
    coverage["__series__"] = series_report

    # Protocol 判定
    per_task_ok = []
    for task in INTERNAL_FILES:
        r = coverage.get(task, {})
        ok = all(bool(r.get(f"has_{k}", False)) for k in ("scaffold", "ring_pos", "substituent"))
        per_task_ok.append(ok)

    protocol_i_feasible = (
        all(per_task_ok)
        and bool(series_report.get("has_matched_series", False))
        and len(dfs) >= 2
    )

    decision: Dict[str, Any] = {
        "protocol_I_feasible": bool(protocol_i_feasible),
        "protocol_I_description": (
            "Strict external zero-shot: train on internal pair manifest, "
            "evaluate once on lunci10 pair manifest, no lunci10 fit."
        ),
        "protocol_II_description": (
            "Within-lunci10 adaptation (NOT strict external): CV on lunci10 "
            "to select Siamese hyperparameters; final test on lunci10 holdout."
        ),
        "per_task_required_columns_present": {
            t: bool(ok) for t, ok in zip(INTERNAL_FILES.keys(), per_task_ok)
        },
        "matched_substituent_series": series_report,
        "rationale": (
            "Protocol I feasible" if protocol_i_feasible
            else (
                f"Protocol I not feasible — "
                f"missing columns: {per_task_ok}, "
                f"series check: {series_report.get('reason', '')}"
            )
        ),
    }

    if protocol_i_feasible:
        pair_df = _build_internal_pair_manifest(dfs)
        pair_df.to_csv(INTERNAL_PAIR_CSV, index=False)
        decision["internal_pair_manifest_csv"] = str(INTERNAL_PAIR_CSV)
        decision["n_internal_pairs"] = int(len(pair_df))
    else:
        decision["internal_pair_manifest_csv"] = None
        decision["n_internal_pairs"] = 0

    decision["per_task_coverage"] = {
        task: {
            "file": cov.get("file"),
            "n_rows": cov.get("n_rows", 0),
            "has_scaffold": cov.get("has_scaffold", False),
            "has_ring_pos": cov.get("has_ring_pos", False),
            "has_substituent": cov.get("has_substituent", False),
            "smiles_col": cov.get("smiles_col"),
            "smiles_non_null": cov.get("smiles_non_null", 0),
        } for task, cov in coverage.items() if task != "__series__"
    }

    with FEASIBILITY_JSON.open("w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2, ensure_ascii=False)
    if verbose:
        print(f"[feasibility] protocol_I_feasible={protocol_i_feasible}")
        print(f"[feasibility] rationale={decision['rationale']}")
        print(f"[feasibility] saved {FEASIBILITY_JSON}")
    return decision


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2, ensure_ascii=False))