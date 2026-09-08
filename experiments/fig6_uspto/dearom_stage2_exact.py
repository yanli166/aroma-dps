#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-2 exact dearomatization validator for USPTO reaction corpora.

Purpose
-------
Stage 1 is a fast, high-recall filter. This script performs a precision-oriented
second pass on P1/P2 reactions by:
  1) atom-mapping the reaction when needed (RXNMapper);
  2) enumerating each aromatic ring in the reactant;
  3) tracing the exact same ring atoms into the product via atom-map numbers;
  4) verifying retention of the ring topology;
  5) measuring local loss of aromatic atoms/bonds and local reaction-center changes;
  6) explicitly rejecting common false positives:
       - aromatic protecting/aryl fragment disappears,
       - simple aromatic substitution/coupling,
       - ring opening/fragmentation,
       - mapping mismatch,
       - aromaticity label-only / tautomer-like changes with weak local chemistry evidence.

Designed for the output of dearom_screen_fast.py, but accepts any CSV/TSV
containing a reaction-SMILES column.

Output tiers
------------
A : exact high-confidence dearomatization
B : plausible / partial dearomatization, manual review recommended
C : ambiguous structural event requiring manual review
R : rejected as non-dearomatization / mapping failure

This is a structural filter, not a mechanistic classifier. "A" means that the
same mapped ring is retained while aromaticity is lost locally; it does not by
itself prove catalytic/asymmetric dearomatization.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set, Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdchem


RXN_COL_CANDIDATES = (
    "mapped_reactions", "mapped_reaction", "reactions", "reaction",
    "rxn_smiles", "reaction_smiles", "canonical_rxn", "rxn"
)
ID_COL_CANDIDATES = ("id", "reaction_id", "rxn_id", "patent_id", "ID")
CLASS_COL_CANDIDATES = ("class", "reaction_class", "rxn_class", "priority")

MAP_RE = re.compile(r":(\d+)\]")


# ----------------------------- IO helpers --------------------------------- #

def detect_sep(path: str) -> str:
    p = path.lower()
    if p.endswith(".tsv") or p.endswith(".txt"):
        return "\t"
    return ","


def choose_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower = {str(c).lower(): c for c in columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def normalize_reaction_smiles(rxn: str) -> Tuple[str, str, str]:
    """
    Return (reactants, agents, products).
    Supports A.B>>C and A.B>agents>C.
    """
    rxn = str(rxn).strip()
    if ">>" in rxn:
        left, right = rxn.split(">>", 1)
        return left.strip(), "", right.strip()
    parts = rxn.split(">")
    if len(parts) == 3:
        return parts[0].strip(), parts[1].strip(), parts[2].strip()
    raise ValueError("Reaction SMILES must contain >> or exactly two > separators")


def to_mapping_input(rxn: str) -> str:
    left, _, right = normalize_reaction_smiles(rxn)
    return f"{left}>>{right}"


def has_atom_mapping(rxn: str) -> bool:
    return MAP_RE.search(rxn) is not None


def read_inputs(paths: Sequence[str], reaction_col: Optional[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        sep = detect_sep(p)
        df = pd.read_csv(p, sep=sep, low_memory=False)
        if reaction_col and reaction_col not in df.columns:
            raise ValueError(f"{p}: requested --reaction-col {reaction_col!r} not found")
        rc = reaction_col or choose_column(df.columns, RXN_COL_CANDIDATES)
        if rc is None:
            raise ValueError(
                f"{p}: could not detect reaction column. Columns={list(df.columns)}. "
                "Use --reaction-col."
            )
        idc = choose_column(df.columns, ID_COL_CANDIDATES)
        cc = choose_column(df.columns, CLASS_COL_CANDIDATES)

        out = pd.DataFrame()
        out["source_file"] = os.path.basename(p)
        out["source_row"] = range(len(df))
        out["reaction_id"] = (
            df[idc].astype(str) if idc else
            [f"{Path(p).stem}_{i:08d}" for i in range(len(df))]
        )
        out["source_class"] = df[cc].astype(str) if cc else ""
        out["reaction"] = df[rc].astype(str)
        # carry first-stage priority/score if available
        for c in ("reaction_priority", "priority", "best_score", "score"):
            if c in df.columns:
                out[f"stage1_{c}"] = df[c]
        frames.append(out)

    all_df = pd.concat(frames, ignore_index=True)
    # Deduplicate identical reaction strings, preferring first occurrence.
    all_df = all_df.drop_duplicates(subset=["reaction"], keep="first").reset_index(drop=True)
    return all_df


# ----------------------------- mapping ------------------------------------ #

class ReactionMapper:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._mapper = None

    def _load(self):
        if self._mapper is None:
            try:
                from rxnmapper import RXNMapper
            except ImportError as e:
                raise RuntimeError(
                    "RXNMapper is required for unmapped reactions. Install with:\n"
                    "  pip install rxnmapper\n"
                    "or run only already-mapped reactions with --no-rxnmapper."
                ) from e
            self._mapper = RXNMapper()

    def map_batch(self, reactions: List[str]) -> List[Tuple[Optional[str], float, str]]:
        """
        Returns tuples: mapped_rxn, confidence, status
        """
        if not reactions:
            return []
        if not self.enabled:
            return [(None, float("nan"), "UNMAPPED_RXNMAPPER_DISABLED") for _ in reactions]

        self._load()
        mapping_inputs = [to_mapping_input(r) for r in reactions]
        try:
            results = self._mapper.get_attention_guided_atom_maps(mapping_inputs)
        except Exception as exc:
            return [(None, float("nan"), f"RXNMAPPER_BATCH_ERROR:{type(exc).__name__}") for _ in reactions]

        out = []
        for res in results:
            mapped = res.get("mapped_rxn") or res.get("mapped_reaction")
            conf = res.get("confidence", float("nan"))
            try:
                conf = float(conf)
            except Exception:
                conf = float("nan")
            if mapped:
                out.append((mapped, conf, "MAPPED_RXNMAPPER"))
            else:
                out.append((None, conf, "RXNMAPPER_EMPTY_RESULT"))
        return out


def map_reactions_incrementally(
    df: pd.DataFrame,
    mapper: ReactionMapper,
    batch_size: int,
    mapped_cache_path: Path,
) -> pd.DataFrame:
    """
    Reuses cache if present. New rows are appended as mapping proceeds.
    """
    cache: Dict[str, Tuple[str, float, str]] = {}
    if mapped_cache_path.exists():
        try:
            old = pd.read_csv(mapped_cache_path, low_memory=False)
            for _, row in old.iterrows():
                cache[str(row["reaction"])] = (
                    str(row.get("mapped_reaction", "")),
                    float(row.get("mapping_confidence", float("nan"))),
                    str(row.get("mapping_status", "")),
                )
        except Exception:
            pass

    rows_to_write = []
    mapped_rxns, confs, statuses = [], [], []

    pending_idx = []
    pending_rxn = []

    def flush_pending():
        nonlocal pending_idx, pending_rxn, rows_to_write
        if not pending_rxn:
            return {}
        results = mapper.map_batch(pending_rxn)
        result_map = {}
        for idx, original, (mapped, conf, status) in zip(pending_idx, pending_rxn, results):
            result_map[idx] = (mapped or "", conf, status)
            rows_to_write.append({
                "reaction": original,
                "mapped_reaction": mapped or "",
                "mapping_confidence": conf,
                "mapping_status": status,
            })
        pending_idx, pending_rxn = [], []
        return result_map

    results_by_idx: Dict[int, Tuple[str, float, str]] = {}

    for i, rxn in enumerate(df["reaction"].astype(str)):
        if rxn in cache:
            results_by_idx[i] = cache[rxn]
            continue

        if has_atom_mapping(rxn):
            mapped = to_mapping_input(rxn)
            results_by_idx[i] = (mapped, 1.0, "ALREADY_MAPPED")
            rows_to_write.append({
                "reaction": rxn,
                "mapped_reaction": mapped,
                "mapping_confidence": 1.0,
                "mapping_status": "ALREADY_MAPPED",
            })
            continue

        pending_idx.append(i)
        pending_rxn.append(rxn)
        if len(pending_rxn) >= batch_size:
            results_by_idx.update(flush_pending())

    results_by_idx.update(flush_pending())

    # Append cache rows once per run.
    if rows_to_write:
        pd.DataFrame(rows_to_write).to_csv(
            mapped_cache_path,
            mode="a",
            header=not mapped_cache_path.exists(),
            index=False,
        )

    for i in range(len(df)):
        mapped, conf, status = results_by_idx.get(i, ("", float("nan"), "MAPPING_MISSING"))
        mapped_rxns.append(mapped)
        confs.append(conf)
        statuses.append(status)

    out = df.copy()
    out["mapped_reaction"] = mapped_rxns
    out["mapping_confidence"] = confs
    out["mapping_status"] = statuses
    return out


# ----------------------------- chemistry ---------------------------------- #

def mols_from_side(side: str) -> List[Chem.Mol]:
    mols = []
    for s in side.split("."):
        s = s.strip()
        if not s:
            continue
        m = Chem.MolFromSmiles(s)
        if m is not None:
            mols.append(m)
    return mols


def map_index(mols: List[Chem.Mol]) -> Dict[int, Tuple[int, int]]:
    """
    map number -> (component index, atom index)
    """
    idx = {}
    for ci, mol in enumerate(mols):
        for atom in mol.GetAtoms():
            n = atom.GetAtomMapNum()
            if n > 0:
                idx[n] = (ci, atom.GetIdx())
    return idx


def unique_aromatic_rings(mol: Chem.Mol, min_size: int = 5, max_size: int = 7) -> List[Tuple[int, ...]]:
    rings = []
    seen = set()
    for ring in Chem.GetSymmSSSR(mol):
        atoms = tuple(int(x) for x in ring)
        if not (min_size <= len(atoms) <= max_size):
            continue
        key = frozenset(atoms)
        if key in seen:
            continue
        # Keep rings with all atoms aromatic. This makes the target definition explicit.
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in atoms):
            rings.append(atoms)
            seen.add(key)
    return rings


def cycle_edges(mol: Chem.Mol, ring_atoms: Sequence[int]) -> List[Tuple[int, int]]:
    aset = set(ring_atoms)
    edges = []
    for i in ring_atoms:
        atom = mol.GetAtomWithIdx(i)
        for b in atom.GetBonds():
            j = b.GetOtherAtomIdx(i)
            if j in aset:
                pair = tuple(sorted((i, j)))
                if pair not in edges:
                    edges.append(pair)
    return edges


def bond_signature(b: Optional[Chem.Bond]) -> Tuple[bool, float]:
    if b is None:
        return (False, 0.0)
    return (b.GetIsAromatic(), float(b.GetBondTypeAsDouble()))


def atom_signature(a: Chem.Atom) -> Dict[str, Any]:
    return {
        "aromatic": bool(a.GetIsAromatic()),
        "hyb": str(a.GetHybridization()),
        "charge": int(a.GetFormalCharge()),
        "degree": int(a.GetDegree()),
        "total_h": int(a.GetTotalNumHs(includeNeighbors=True)),
        "chiral": str(a.GetChiralTag()),
        "atomic_num": int(a.GetAtomicNum()),
    }


def external_mapped_neighbors(mol: Chem.Mol, atom_idx: int, ring_set: Set[int]) -> Set[int]:
    out = set()
    atom = mol.GetAtomWithIdx(atom_idx)
    for nb in atom.GetNeighbors():
        if nb.GetIdx() in ring_set:
            continue
        mp = nb.GetAtomMapNum()
        if mp > 0:
            out.add(mp)
    return out


def ring_formula(mol: Chem.Mol, ring_atoms: Sequence[int]) -> str:
    c = Counter(mol.GetAtomWithIdx(i).GetSymbol() for i in ring_atoms)
    return "".join(f"{el}{c[el]}" for el in sorted(c))


def ring_fusion_metrics(mol: Chem.Mol, ring_atoms: Sequence[int]) -> Tuple[int, int]:
    """Return (# ring atoms shared with another ring, # fused ring edges)."""
    ring_sets = [set(int(x) for x in r) for r in Chem.GetSymmSSSR(mol)]
    target = set(ring_atoms)
    shared_atoms = set()
    shared_edges = 0
    target_edges = {tuple(sorted(e)) for e in cycle_edges(mol, ring_atoms)}
    for rs in ring_sets:
        if rs == target:
            continue
        inter = target & rs
        shared_atoms |= inter
        if len(inter) >= 2:
            other_edges = set()
            for i in rs:
                a = mol.GetAtomWithIdx(i)
                for b in a.GetBonds():
                    j = b.GetOtherAtomIdx(i)
                    if j in rs:
                        other_edges.add(tuple(sorted((i, j))))
            shared_edges += len(target_edges & other_edges)
    return len(shared_atoms), shared_edges


@dataclass
class RingEvidence:
    reaction_id: str
    source_file: str
    source_row: int
    source_class: str
    mapping_status: str
    mapping_confidence: float

    reactant_component: int
    ring_index: int
    ring_size: int
    ring_formula: str
    ring_map_numbers: str
    shared_ring_atoms: int
    fused_ring_edges: int

    mapped_fraction: float
    same_product_component: bool
    ring_edge_retention: float
    context_retention: float

    reactant_aromatic_atoms: int
    product_aromatic_atoms: int
    lost_aromatic_atoms: int
    aromatic_atom_loss_fraction: float

    reactant_aromatic_ring_edges: int
    product_aromatic_ring_edges: int
    lost_aromatic_ring_edges: int
    aromatic_bond_loss_fraction: float

    new_sp3_ring_atoms: int
    hybridization_changes: int
    ring_bond_order_changes: int
    formal_charge_changes: int
    total_h_changes: int
    degree_changes: int
    new_external_bonds: int
    deleted_external_bonds: int
    new_stereocenters_on_ring: int

    local_event_score: int
    exact_score: float
    tier: str
    decision_reason: str


def evaluate_ring(
    reaction_meta: Dict[str, Any],
    rmol: Chem.Mol,
    rcomp_idx: int,
    ring_idx: int,
    ring_atoms: Sequence[int],
    pmols: List[Chem.Mol],
    pmap: Dict[int, Tuple[int, int]],
    map_conf_a: float,
    map_conf_b: float,
) -> RingEvidence:

    rset = set(ring_atoms)
    redges = cycle_edges(rmol, ring_atoms)
    ring_maps = [rmol.GetAtomWithIdx(i).GetAtomMapNum() for i in ring_atoms]
    valid_maps = [m for m in ring_maps if m > 0 and m in pmap]
    mapped_fraction = len(valid_maps) / max(1, len(ring_atoms))

    # map reactant atom index -> product (component, atom index)
    r_to_p = {}
    for ri in ring_atoms:
        mp = rmol.GetAtomWithIdx(ri).GetAtomMapNum()
        if mp > 0 and mp in pmap:
            r_to_p[ri] = pmap[mp]

    product_components = {ci for ci, _ in r_to_p.values()}
    same_component = len(product_components) == 1 and len(r_to_p) == len(ring_atoms)

    # Ring-edge retention
    retained_edges = 0
    comparable_edges = 0
    prod_arom_edges = 0
    react_arom_edges = 0
    ring_bond_order_changes = 0

    for i, j in redges:
        rb = rmol.GetBondBetweenAtoms(i, j)
        if rb and rb.GetIsAromatic():
            react_arom_edges += 1
        if i not in r_to_p or j not in r_to_p:
            continue
        ci, pi = r_to_p[i]
        cj, pj = r_to_p[j]
        comparable_edges += 1
        pb = pmols[ci].GetBondBetweenAtoms(pi, pj) if ci == cj else None
        if pb is not None:
            retained_edges += 1
            if pb.GetIsAromatic():
                prod_arom_edges += 1
            if bond_signature(rb) != bond_signature(pb):
                ring_bond_order_changes += 1

    ring_edge_retention = retained_edges / max(1, len(redges))
    lost_arom_edges = max(0, react_arom_edges - prod_arom_edges)
    arom_bond_loss_fraction = lost_arom_edges / max(1, react_arom_edges)

    # Atom-level changes
    react_arom_atoms = sum(rmol.GetAtomWithIdx(i).GetIsAromatic() for i in ring_atoms)
    prod_arom_atoms = 0
    new_sp3 = 0
    hyb_changes = 0
    charge_changes = 0
    total_h_changes = 0
    degree_changes = 0
    new_stereo = 0

    # Context comparison
    context_total = 0
    context_retained = 0
    new_external_bonds = 0
    deleted_external_bonds = 0

    for ri in ring_atoms:
        ra = rmol.GetAtomWithIdx(ri)
        if ri not in r_to_p:
            continue
        pci, pi = r_to_p[ri]
        pa = pmols[pci].GetAtomWithIdx(pi)
        rs = atom_signature(ra)
        ps = atom_signature(pa)

        prod_arom_atoms += int(ps["aromatic"])
        if rs["hyb"] != "SP3" and ps["hyb"] == "SP3":
            new_sp3 += 1
        hyb_changes += int(rs["hyb"] != ps["hyb"])
        charge_changes += int(rs["charge"] != ps["charge"])
        total_h_changes += int(rs["total_h"] != ps["total_h"])
        degree_changes += int(rs["degree"] != ps["degree"])
        if rs["chiral"] == "CHI_UNSPECIFIED" and ps["chiral"] != "CHI_UNSPECIFIED":
            new_stereo += 1

        r_ext = external_mapped_neighbors(rmol, ri, rset)
        # Product ring atom indices corresponding to the same mapped ring
        p_ring_idx_set = {pidx for rc, pidx in r_to_p.values() if rc == pci}
        p_ext = external_mapped_neighbors(pmols[pci], pi, p_ring_idx_set)

        context_total += len(r_ext)
        context_retained += len(r_ext & p_ext)
        new_external_bonds += len(p_ext - r_ext)
        deleted_external_bonds += len(r_ext - p_ext)

    context_retention = context_retained / context_total if context_total else 1.0

    lost_arom_atoms = max(0, react_arom_atoms - prod_arom_atoms)
    arom_atom_loss_fraction = lost_arom_atoms / max(1, react_arom_atoms)

    shared_atoms, fused_edges = ring_fusion_metrics(rmol, ring_atoms)

    # Local event score: deliberately chemistry-oriented, not ML-derived.
    local_event_score = 0
    if new_sp3 > 0:
        local_event_score += 3
    if ring_bond_order_changes >= 2:
        local_event_score += 2
    elif ring_bond_order_changes == 1:
        local_event_score += 1
    if new_external_bonds > 0:
        local_event_score += 2
    if hyb_changes > 0:
        local_event_score += 1
    if total_h_changes > 0:
        local_event_score += 1
    if charge_changes > 0:
        local_event_score += 1
    if new_stereo > 0:
        local_event_score += 1

    map_conf = reaction_meta["mapping_confidence"]
    eff_map_conf = 1.0 if math.isnan(map_conf) else map_conf

    # Precision-oriented deterministic tiering.
    tier = "R"
    reason = ""

    if mapped_fraction < 0.999:
        # Ring atom(s) disappeared or were not mapped -> usually deprotection / aryl fragment loss.
        tier = "R"
        reason = "TARGET_RING_ATOM_LOSS_OR_MAPPING_INCOMPLETE"
    elif not same_component:
        tier = "R"
        reason = "TARGET_RING_SPLIT_ACROSS_PRODUCTS"
    elif ring_edge_retention < 0.80:
        tier = "C"
        reason = "RING_OPENING_OR_MAJOR_REARRANGEMENT"
    elif lost_arom_atoms <= 0 and lost_arom_edges <= 0:
        tier = "R"
        reason = "NO_RING_SPECIFIC_AROMATICITY_LOSS"
    elif lost_arom_atoms <= 1 and lost_arom_edges <= 1 and local_event_score <= 1:
        tier = "R"
        reason = "WEAK_LABEL_ONLY_OR_TAUTOMER_LIKE_CHANGE"
    else:
        strong_loss = (
            arom_atom_loss_fraction >= 0.33
            and arom_bond_loss_fraction >= 0.33
            and lost_arom_atoms >= 2
            and lost_arom_edges >= 2
        )
        topology_exact = ring_edge_retention >= 0.999
        context_ok = context_retention >= 0.50
        strong_local = local_event_score >= 3

        if (
            strong_loss
            and topology_exact
            and context_ok
            and strong_local
            and eff_map_conf >= map_conf_a
        ):
            tier = "A"
            reason = "EXACT_RING_RETAINED_WITH_STRONG_LOCAL_AROMATICITY_LOSS"
        elif (
            ring_edge_retention >= 0.83
            and lost_arom_atoms >= 1
            and lost_arom_edges >= 1
            and local_event_score >= 1
            and eff_map_conf >= map_conf_b
        ):
            tier = "B"
            reason = "PLAUSIBLE_PARTIAL_OR_FUSED_RING_DEAROMATIZATION"
        else:
            tier = "C"
            reason = "AMBIGUOUS_RING_SPECIFIC_AROMATICITY_LOSS"

    # A transparent score only for ranking inside a tier.
    exact_score = 0.0
    exact_score += 20.0 * mapped_fraction
    exact_score += 15.0 * ring_edge_retention
    exact_score += 10.0 * context_retention
    exact_score += 20.0 * arom_atom_loss_fraction
    exact_score += 15.0 * arom_bond_loss_fraction
    exact_score += min(10.0, 2.0 * local_event_score)
    exact_score += min(5.0, 2.0 * new_sp3)
    exact_score += min(5.0, 2.0 * new_external_bonds)

    return RingEvidence(
        reaction_id=str(reaction_meta["reaction_id"]),
        source_file=str(reaction_meta["source_file"]),
        source_row=int(reaction_meta["source_row"]),
        source_class=str(reaction_meta["source_class"]),
        mapping_status=str(reaction_meta["mapping_status"]),
        mapping_confidence=float(reaction_meta["mapping_confidence"]),

        reactant_component=rcomp_idx,
        ring_index=ring_idx,
        ring_size=len(ring_atoms),
        ring_formula=ring_formula(rmol, ring_atoms),
        ring_map_numbers=";".join(str(x) for x in sorted(m for m in ring_maps if m > 0)),
        shared_ring_atoms=shared_atoms,
        fused_ring_edges=fused_edges,

        mapped_fraction=mapped_fraction,
        same_product_component=same_component,
        ring_edge_retention=ring_edge_retention,
        context_retention=context_retention,

        reactant_aromatic_atoms=react_arom_atoms,
        product_aromatic_atoms=prod_arom_atoms,
        lost_aromatic_atoms=lost_arom_atoms,
        aromatic_atom_loss_fraction=arom_atom_loss_fraction,

        reactant_aromatic_ring_edges=react_arom_edges,
        product_aromatic_ring_edges=prod_arom_edges,
        lost_aromatic_ring_edges=lost_arom_edges,
        aromatic_bond_loss_fraction=arom_bond_loss_fraction,

        new_sp3_ring_atoms=new_sp3,
        hybridization_changes=hyb_changes,
        ring_bond_order_changes=ring_bond_order_changes,
        formal_charge_changes=charge_changes,
        total_h_changes=total_h_changes,
        degree_changes=degree_changes,
        new_external_bonds=new_external_bonds,
        deleted_external_bonds=deleted_external_bonds,
        new_stereocenters_on_ring=new_stereo,

        local_event_score=local_event_score,
        exact_score=round(exact_score, 3),
        tier=tier,
        decision_reason=reason,
    )


def evaluate_reaction(row: pd.Series, map_conf_a: float, map_conf_b: float) -> Tuple[List[Dict], Dict]:
    mapped_rxn = str(row.get("mapped_reaction", "") or "")
    meta = {
        "reaction_id": row["reaction_id"],
        "source_file": row["source_file"],
        "source_row": row["source_row"],
        "source_class": row.get("source_class", ""),
        "mapping_status": row.get("mapping_status", ""),
        "mapping_confidence": float(row.get("mapping_confidence", float("nan"))),
    }

    summary = {
        **meta,
        "original_reaction": row["reaction"],
        "mapped_reaction": mapped_rxn,
        "n_aromatic_reactant_rings": 0,
        "n_A_rings": 0,
        "n_B_rings": 0,
        "n_C_rings": 0,
        "n_R_rings": 0,
        "stage2_tier": "R",
        "best_exact_score": 0.0,
        "decision_reason": "",
    }

    if not mapped_rxn:
        summary["decision_reason"] = "MAPPING_FAILED"
        return [], summary

    try:
        left, _, right = normalize_reaction_smiles(mapped_rxn)
        rmols = mols_from_side(left)
        pmols = mols_from_side(right)
        if not rmols or not pmols:
            summary["decision_reason"] = "SMILES_PARSE_FAILED"
            return [], summary
        pmap = map_index(pmols)
    except Exception as exc:
        summary["decision_reason"] = f"REACTION_PARSE_ERROR:{type(exc).__name__}"
        return [], summary

    evidences: List[RingEvidence] = []
    for rci, rmol in enumerate(rmols):
        rings = unique_aromatic_rings(rmol)
        summary["n_aromatic_reactant_rings"] += len(rings)
        for ridx, ring in enumerate(rings):
            ev = evaluate_ring(meta, rmol, rci, ridx, ring, pmols, pmap, map_conf_a, map_conf_b)
            evidences.append(ev)

    for ev in evidences:
        summary[f"n_{ev.tier}_rings"] += 1

    order = {"A": 3, "B": 2, "C": 1, "R": 0}
    if evidences:
        best = max(evidences, key=lambda e: (order[e.tier], e.exact_score))
        summary["stage2_tier"] = best.tier
        summary["best_exact_score"] = best.exact_score
        summary["decision_reason"] = best.decision_reason
    else:
        summary["decision_reason"] = "NO_FULLY_AROMATIC_5_TO_7_MEMBERED_REACTANT_RING"

    return [asdict(x) for x in evidences], summary


# ----------------------------- reporting ---------------------------------- #

def make_review_sample(summary_df: pd.DataFrame, out_path: Path, n_per_tier: int, seed: int):
    samples = []
    for tier in ("A", "B", "C", "R"):
        part = summary_df[summary_df["stage2_tier"] == tier]
        if len(part):
            samples.append(part.sample(min(n_per_tier, len(part)), random_state=seed))
    if samples:
        s = pd.concat(samples, ignore_index=True)
        s["human_label"] = ""
        s["human_note"] = ""
        s.to_csv(out_path, index=False)


def main():
    ap = argparse.ArgumentParser(
        description="Stage-2 exact atom-mapped dearomatization validator"
    )
    ap.add_argument("--input", nargs="+", required=True,
                    help="One or more P1/P2 CSV/TSV files from Stage 1")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--reaction-col", default=None)
    ap.add_argument("--batch-size", type=int, default=32,
                    help="RXNMapper batch size; reduce if GPU/CPU memory is limited")
    ap.add_argument("--no-rxnmapper", action="store_true",
                    help="Do not map unmapped reactions; unmapped rows will be rejected")
    ap.add_argument("--map-confidence-a", type=float, default=0.60)
    ap.add_argument("--map-confidence-b", type=float, default=0.30)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample-per-tier", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = read_inputs(args.input, args.reaction_col)
    if args.limit:
        df = df.head(args.limit).copy()

    print(f"[Stage2] unique reactions loaded: {len(df):,}", flush=True)

    mapper = ReactionMapper(enabled=not args.no_rxnmapper)
    cache_path = outdir / "mapped_reactions_cache.csv"

    t0 = time.time()
    df = map_reactions_incrementally(df, mapper, args.batch_size, cache_path)
    print(f"[Stage2] mapping finished in {(time.time()-t0)/60:.1f} min", flush=True)

    ring_rows = []
    reaction_rows = []

    for i, (_, row) in enumerate(df.iterrows(), 1):
        rings, summary = evaluate_reaction(row, args.map_confidence_a, args.map_confidence_b)
        ring_rows.extend(rings)
        reaction_rows.append(summary)
        if i % 1000 == 0 or i == len(df):
            print(f"[Stage2] validated {i:,}/{len(df):,}", flush=True)

    ring_df = pd.DataFrame(ring_rows)
    summary_df = pd.DataFrame(reaction_rows)

    ring_df.to_csv(outdir / "stage2_ring_evidence.csv", index=False)
    summary_df.to_csv(outdir / "stage2_reactions.csv", index=False)

    for tier, filename in (
        ("A", "tier_A_exact.csv"),
        ("B", "tier_B_plausible.csv"),
        ("C", "tier_C_review.csv"),
        ("R", "rejected.csv"),
    ):
        summary_df[summary_df["stage2_tier"] == tier].to_csv(outdir / filename, index=False)

    # False-positive breakdown.
    breakdown = (
        summary_df.groupby(["stage2_tier", "decision_reason"], dropna=False)
        .size().reset_index(name="n_reactions")
        .sort_values(["stage2_tier", "n_reactions"], ascending=[True, False])
    )
    breakdown.to_csv(outdir / "decision_breakdown.csv", index=False)

    make_review_sample(
        summary_df,
        outdir / "manual_review_sample.csv",
        args.sample_per_tier,
        args.seed,
    )

    report = {
        "n_unique_input_reactions": int(len(df)),
        "mapping_status_counts": {
            str(k): int(v) for k, v in df["mapping_status"].value_counts(dropna=False).items()
        },
        "stage2_tier_counts": {
            str(k): int(v) for k, v in summary_df["stage2_tier"].value_counts(dropna=False).items()
        },
        "decision_reason_counts": {
            str(k): int(v) for k, v in summary_df["decision_reason"].value_counts(dropna=False).items()
        },
        "n_ring_evidence_rows": int(len(ring_df)),
        "thresholds": {
            "map_confidence_A": args.map_confidence_a,
            "map_confidence_B": args.map_confidence_b,
            "A_requires": {
                "mapped_fraction": 1.0,
                "same_product_component": True,
                "ring_edge_retention": 1.0,
                "aromatic_atom_loss_fraction_min": 0.33,
                "aromatic_bond_loss_fraction_min": 0.33,
                "lost_aromatic_atoms_min": 2,
                "lost_aromatic_bonds_min": 2,
                "context_retention_min": 0.50,
                "local_event_score_min": 3,
            },
        },
    }
    with open(outdir / "stage2_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n[Stage2] DONE")
    print(json.dumps(report["stage2_tier_counts"], ensure_ascii=False, indent=2))
    print(f"Outputs: {outdir}")


if __name__ == "__main__":
    main()
