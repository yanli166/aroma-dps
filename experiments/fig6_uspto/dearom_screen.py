#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USPTO dearomatization candidate screener
=========================================

Input columns (TSV/CSV):
    id, class, reactions
where reactions is "reactants>>products" (or "reactants>reagents>products").

Outputs:
    screened_reactions.csv   one row per reaction
    candidate_rings.csv      one row per candidate aromatic ring
    priority_1.csv           high-confidence reaction candidates
    priority_2.csv           medium-confidence reaction candidates
    priority_3.csv           manual-review reaction candidates
    review_images/           optional highlighted ring images

Design principle:
    1) detect aromatic rings in reactants;
    2) map reactant components to product components (existing atom-map numbers if
       available, otherwise bond-order-agnostic RDKit MCS);
    3) track each reactant aromatic ring into the product;
    4) quantify loss of aromatic atoms/bonds, ring-edge retention, hybridization
       changes and reaction-center evidence;
    5) assign P1/P2/P3 confidence without treating global aromatic-ring counts as
       proof by themselves.

This is a SCREENING tool, not the final chemical annotation. P1 should still be
spot-checked before entering a publication-quality Gold corpus.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, rdFMCS
from tqdm import tqdm


# ----------------------------- configuration ----------------------------- #

@dataclass
class Config:
    allowed_ring_sizes: Tuple[int, ...] = (5, 6, 7)
    mcs_timeout_s: int = 1
    min_element_overlap: float = 0.60
    min_mcs_reactant_coverage: float = 0.30
    p1_min_ring_mapping: float = 0.83
    p1_min_edge_retention: float = 0.83
    p1_min_pair_conf: float = 0.40
    p2_min_ring_mapping: float = 0.67
    p2_min_edge_retention: float = 0.60
    p2_min_pair_conf: float = 0.25
    draw_limit: int = 0


# ------------------------------ data models ------------------------------- #

@dataclass
class RingRecord:
    ring_idx: int
    atom_indices: Tuple[int, ...]
    bond_indices: Tuple[int, ...]
    size: int
    elements: str
    is_fully_aromatic: bool
    aromatic_atom_count: int
    aromatic_bond_count: int


@dataclass
class PairMapping:
    reactant_component: int
    product_component: int
    method: str
    r_to_p: Dict[int, int]
    mapped_atoms: int
    reactant_heavy_atoms: int
    product_heavy_atoms: int
    reactant_coverage: float
    product_coverage: float
    pair_confidence: float
    ambiguous_product_atomset: bool
    mcs_smarts: str = ""


# ----------------------------- basic utilities ---------------------------- #

def mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        return mol
    except Exception:
        return None


def split_reaction_smiles(rxn: str) -> Tuple[str, str]:
    """Support A>>B and A>agents>B."""
    if ">>" in rxn:
        left, right = rxn.split(">>", 1)
        return left.strip(), right.strip()
    parts = rxn.split(">")
    if len(parts) == 3:
        return parts[0].strip(), parts[2].strip()
    raise ValueError(f"Cannot parse reaction SMILES: {rxn[:120]}")


def parse_components(side: str) -> Tuple[List[str], List[Chem.Mol]]:
    smiles = [x.strip() for x in side.split(".") if x.strip()]
    valid_smiles, mols = [], []
    for s in smiles:
        m = mol_from_smiles(s)
        if m is not None:
            valid_smiles.append(s)
            mols.append(m)
    return valid_smiles, mols


def heavy_atom_count(mol: Chem.Mol) -> int:
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)


def element_counter(mol: Chem.Mol) -> Counter:
    return Counter(a.GetAtomicNum() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)


def element_overlap_fraction(rmol: Chem.Mol, pmol: Chem.Mol) -> float:
    r = element_counter(rmol)
    p = element_counter(pmol)
    denom = max(1, sum(r.values()))
    shared = sum(min(n, p.get(z, 0)) for z, n in r.items())
    return shared / denom


def bond_numeric_type(bond: Chem.Bond) -> float:
    if bond.GetIsAromatic():
        return 1.5
    return float(bond.GetBondTypeAsDouble())


def has_atom_maps(mol: Chem.Mol) -> bool:
    maps = [a.GetAtomMapNum() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    return bool(maps) and sum(x > 0 for x in maps) / max(1, len(maps)) >= 0.8


# ------------------------------- ring logic ------------------------------- #

def get_rings(mol: Chem.Mol, allowed_sizes: Optional[Sequence[int]] = None) -> List[RingRecord]:
    """Use RDKit SymmSSSR ring basis; report each ring separately."""
    ring_info = mol.GetRingInfo()
    atom_rings = list(ring_info.AtomRings())
    bond_rings = list(ring_info.BondRings())
    out: List[RingRecord] = []

    for i, atoms in enumerate(atom_rings):
        if allowed_sizes and len(atoms) not in allowed_sizes:
            continue
        bonds = bond_rings[i] if i < len(bond_rings) else tuple()
        arom_atoms = sum(mol.GetAtomWithIdx(a).GetIsAromatic() for a in atoms)
        arom_bonds = sum(mol.GetBondWithIdx(b).GetIsAromatic() for b in bonds)
        elements = "-".join(sorted(mol.GetAtomWithIdx(a).GetSymbol() for a in atoms))
        out.append(
            RingRecord(
                ring_idx=i,
                atom_indices=tuple(atoms),
                bond_indices=tuple(bonds),
                size=len(atoms),
                elements=elements,
                is_fully_aromatic=(arom_atoms == len(atoms) and arom_bonds == len(bonds)),
                aromatic_atom_count=int(arom_atoms),
                aromatic_bond_count=int(arom_bonds),
            )
        )
    return out


def aromatic_rings(mol: Chem.Mol, cfg: Config) -> List[RingRecord]:
    return [r for r in get_rings(mol, cfg.allowed_ring_sizes) if r.is_fully_aromatic]


def mol_summary(mol: Chem.Mol, cfg: Config) -> Dict[str, int]:
    rings_all = get_rings(mol, None)
    rings_allowed = get_rings(mol, cfg.allowed_ring_sizes)
    return {
        "n_heavy_atoms": heavy_atom_count(mol),
        "n_rings_all": len(rings_all),
        "n_rings_5_7": len(rings_allowed),
        "n_aromatic_rings_5_7": sum(r.is_fully_aromatic for r in rings_allowed),
        "n_aromatic_atoms": sum(a.GetIsAromatic() for a in mol.GetAtoms()),
        "n_aromatic_bonds": sum(b.GetIsAromatic() for b in mol.GetBonds()),
    }


# ----------------------------- atom correspondence ------------------------ #

def mapping_from_atom_maps(
    rmol: Chem.Mol,
    pmols: Sequence[Chem.Mol],
    r_idx: int,
) -> Optional[PairMapping]:
    r_map = {a.GetAtomMapNum(): a.GetIdx() for a in rmol.GetAtoms() if a.GetAtomMapNum() > 0}
    if not r_map:
        return None

    best = None
    for p_idx, pmol in enumerate(pmols):
        p_map = {a.GetAtomMapNum(): a.GetIdx() for a in pmol.GetAtoms() if a.GetAtomMapNum() > 0}
        shared = sorted(set(r_map) & set(p_map))
        if not shared:
            continue
        r_to_p = {r_map[m]: p_map[m] for m in shared}
        cov_r = len(r_to_p) / max(1, heavy_atom_count(rmol))
        cov_p = len(r_to_p) / max(1, heavy_atom_count(pmol))
        conf = min(1.0, cov_r)
        obj = PairMapping(
            reactant_component=r_idx,
            product_component=p_idx,
            method="atom_map",
            r_to_p=r_to_p,
            mapped_atoms=len(r_to_p),
            reactant_heavy_atoms=heavy_atom_count(rmol),
            product_heavy_atoms=heavy_atom_count(pmol),
            reactant_coverage=cov_r,
            product_coverage=cov_p,
            pair_confidence=conf,
            ambiguous_product_atomset=False,
        )
        if best is None or obj.mapped_atoms > best.mapped_atoms:
            best = obj
    return best


def mapping_from_mcs(
    rmol: Chem.Mol,
    pmols: Sequence[Chem.Mol],
    r_idx: int,
    cfg: Config,
) -> Optional[PairMapping]:
    """
    Bond-order-agnostic MCS so aromatic C-C can map to single/double/sp3 ring bonds.
    Ring atoms are constrained to map to ring atoms.
    """
    best: Optional[PairMapping] = None

    for p_idx, pmol in enumerate(pmols):
        elem_overlap = element_overlap_fraction(rmol, pmol)
        if elem_overlap < cfg.min_element_overlap:
            continue
        try:
            mcs = rdFMCS.FindMCS(
                [rmol, pmol],
                maximizeBonds=True,
                threshold=1.0,
                timeout=cfg.mcs_timeout_s,
                verbose=False,
                matchValences=False,
                ringMatchesRingOnly=True,
                completeRingsOnly=False,
                matchChiralTag=False,
                atomCompare=rdFMCS.AtomCompare.CompareElements,
                bondCompare=rdFMCS.BondCompare.CompareAny,
                ringCompare=rdFMCS.RingCompare.IgnoreRingFusion,
            )
        except Exception:
            continue

        if mcs.numAtoms < 3 or not mcs.smartsString:
            continue
        query = Chem.MolFromSmarts(mcs.smartsString)
        if query is None:
            continue
        r_matches = rmol.GetSubstructMatches(query, uniquify=True, maxMatches=64)
        p_matches = pmol.GetSubstructMatches(query, uniquify=True, maxMatches=64)
        if not r_matches or not p_matches:
            continue

        # Distinguish true ambiguity (different product atom sets) from mere
        # automorphic permutations of the same ring/fragment.
        unique_p_sets = {tuple(sorted(x)) for x in p_matches}
        ambiguous = len(unique_p_sets) > 1

        # Use first canonical match; for high-confidence P1 we later require
        # non-ambiguous product atom set.
        r_match = r_matches[0]
        p_match = p_matches[0]
        r_to_p = {ra: pa for ra, pa in zip(r_match, p_match)}

        cov_r = len(r_to_p) / max(1, heavy_atom_count(rmol))
        cov_p = len(r_to_p) / max(1, heavy_atom_count(pmol))
        if cov_r < cfg.min_mcs_reactant_coverage:
            continue

        # confidence is intentionally conservative: ring-specific mapping
        # fraction is evaluated later for each target ring.
        conf = min(1.0, cov_r * (0.90 if ambiguous else 1.0))
        obj = PairMapping(
            reactant_component=r_idx,
            product_component=p_idx,
            method="mcs",
            r_to_p=r_to_p,
            mapped_atoms=len(r_to_p),
            reactant_heavy_atoms=heavy_atom_count(rmol),
            product_heavy_atoms=heavy_atom_count(pmol),
            reactant_coverage=cov_r,
            product_coverage=cov_p,
            pair_confidence=conf,
            ambiguous_product_atomset=ambiguous,
            mcs_smarts=mcs.smartsString,
        )
        if best is None:
            best = obj
        else:
            key = (obj.mapped_atoms, obj.pair_confidence, -int(obj.ambiguous_product_atomset))
            best_key = (best.mapped_atoms, best.pair_confidence, -int(best.ambiguous_product_atomset))
            if key > best_key:
                best = obj
    return best


def find_pair_mapping(
    rmol: Chem.Mol,
    pmols: Sequence[Chem.Mol],
    r_idx: int,
    cfg: Config,
) -> Optional[PairMapping]:
    if has_atom_maps(rmol) and any(has_atom_maps(p) for p in pmols):
        direct = mapping_from_atom_maps(rmol, pmols, r_idx)
        if direct is not None:
            return direct
    return mapping_from_mcs(rmol, pmols, r_idx, cfg)


# --------------------------- ring change analysis ------------------------- #

def mapped_ring_metrics(
    rmol: Chem.Mol,
    pmol: Chem.Mol,
    ring: RingRecord,
    mapping: PairMapping,
) -> Dict[str, object]:
    r_to_p = mapping.r_to_p
    r_atoms = list(ring.atom_indices)
    mapped_r_atoms = [a for a in r_atoms if a in r_to_p]
    mapped_fraction = len(mapped_r_atoms) / max(1, ring.size)
    p_atoms = [r_to_p[a] for a in mapped_r_atoms]

    product_arom_atoms = sum(pmol.GetAtomWithIdx(a).GetIsAromatic() for a in p_atoms)
    reactant_arom_atoms_mapped = sum(rmol.GetAtomWithIdx(a).GetIsAromatic() for a in mapped_r_atoms)
    lost_arom_atoms = max(0, reactant_arom_atoms_mapped - product_arom_atoms)

    new_sp3 = 0
    hybridization_changes = 0
    aromaticity_flag_changes = 0
    degree_changes = 0
    charge_changes = 0
    for ra in mapped_r_atoms:
        pa = r_to_p[ra]
        ar = rmol.GetAtomWithIdx(ra)
        ap = pmol.GetAtomWithIdx(pa)
        if ap.GetHybridization() == Chem.rdchem.HybridizationType.SP3 and ar.GetHybridization() != Chem.rdchem.HybridizationType.SP3:
            new_sp3 += 1
        if ar.GetHybridization() != ap.GetHybridization():
            hybridization_changes += 1
        if ar.GetIsAromatic() != ap.GetIsAromatic():
            aromaticity_flag_changes += 1
        if ar.GetDegree() != ap.GetDegree():
            degree_changes += 1
        if ar.GetFormalCharge() != ap.GetFormalCharge():
            charge_changes += 1

    # Track original ring edges into product. This is more robust than requiring
    # the same SSSR basis in fused systems.
    mapped_edges = 0
    retained_edges = 0
    react_arom_bonds_mapped = 0
    product_arom_bonds = 0
    bond_order_changes = 0
    missing_edges = 0

    for bidx in ring.bond_indices:
        rb = rmol.GetBondWithIdx(bidx)
        ra1, ra2 = rb.GetBeginAtomIdx(), rb.GetEndAtomIdx()
        if ra1 not in r_to_p or ra2 not in r_to_p:
            continue
        mapped_edges += 1
        if rb.GetIsAromatic():
            react_arom_bonds_mapped += 1
        pa1, pa2 = r_to_p[ra1], r_to_p[ra2]
        pb = pmol.GetBondBetweenAtoms(pa1, pa2)
        if pb is None:
            missing_edges += 1
            continue
        retained_edges += 1
        if pb.GetIsAromatic():
            product_arom_bonds += 1
        if abs(bond_numeric_type(rb) - bond_numeric_type(pb)) > 0.1 or rb.GetIsAromatic() != pb.GetIsAromatic():
            bond_order_changes += 1

    edge_retention = retained_edges / max(1, mapped_edges)
    lost_arom_bonds = max(0, react_arom_bonds_mapped - product_arom_bonds)
    atom_arom_loss_fraction = lost_arom_atoms / max(1, reactant_arom_atoms_mapped)
    bond_arom_loss_fraction = lost_arom_bonds / max(1, react_arom_bonds_mapped)

    # Reaction center evidence in or immediately at the target ring.
    center_evidence = (
        bond_order_changes
        + missing_edges
        + hybridization_changes
        + aromaticity_flag_changes
        + degree_changes
        + charge_changes
    )

    return {
        "ring_mapping_fraction": round(mapped_fraction, 4),
        "mapped_ring_atoms": len(mapped_r_atoms),
        "mapped_product_atom_indices": ",".join(map(str, p_atoms)),
        "product_aromatic_atoms_on_ring": int(product_arom_atoms),
        "lost_aromatic_atoms": int(lost_arom_atoms),
        "atom_aromaticity_loss_fraction": round(atom_arom_loss_fraction, 4),
        "mapped_ring_edges": int(mapped_edges),
        "retained_ring_edges": int(retained_edges),
        "ring_edge_retention_fraction": round(edge_retention, 4),
        "product_aromatic_bonds_on_ring": int(product_arom_bonds),
        "lost_aromatic_bonds": int(lost_arom_bonds),
        "bond_aromaticity_loss_fraction": round(bond_arom_loss_fraction, 4),
        "new_sp3_ring_atoms": int(new_sp3),
        "hybridization_changes": int(hybridization_changes),
        "aromaticity_flag_changes": int(aromaticity_flag_changes),
        "ring_atom_degree_changes": int(degree_changes),
        "ring_atom_charge_changes": int(charge_changes),
        "ring_bond_order_changes": int(bond_order_changes),
        "missing_original_ring_edges": int(missing_edges),
        "reaction_center_evidence": int(center_evidence),
    }


def score_ring_candidate(metrics: Dict[str, object], mapping: PairMapping, cfg: Config) -> Tuple[int, int, List[str]]:
    """Return (priority, score, reasons), where priority 1 is highest."""
    mf = float(metrics["ring_mapping_fraction"])
    er = float(metrics["ring_edge_retention_fraction"])
    laa = int(metrics["lost_aromatic_atoms"])
    lab = int(metrics["lost_aromatic_bonds"])
    a_loss = float(metrics["atom_aromaticity_loss_fraction"])
    b_loss = float(metrics["bond_aromaticity_loss_fraction"])
    sp3 = int(metrics["new_sp3_ring_atoms"])
    center = int(metrics["reaction_center_evidence"])
    bochg = int(metrics["ring_bond_order_changes"])

    score = 0
    reasons: List[str] = []

    if mf >= 0.95:
        score += 3; reasons.append("target ring almost fully mapped")
    elif mf >= 0.83:
        score += 2; reasons.append("target ring well mapped")
    elif mf >= 0.67:
        score += 1; reasons.append("target ring partially mapped")

    if er >= 0.95:
        score += 2; reasons.append("ring topology retained")
    elif er >= 0.75:
        score += 1; reasons.append("most ring edges retained")
    else:
        score -= 1; reasons.append("ring topology uncertain/opened")

    if laa >= 3 or a_loss >= 0.50:
        score += 2; reasons.append("large aromatic-atom loss")
    elif laa >= 1:
        score += 1; reasons.append("partial aromatic-atom loss")

    if lab >= 3 or b_loss >= 0.50:
        score += 2; reasons.append("large aromatic-bond loss")
    elif lab >= 1:
        score += 1; reasons.append("partial aromatic-bond loss")

    if sp3 >= 1:
        score += 2; reasons.append("new sp3 atom(s) formed on target ring")
    elif bochg >= 2:
        score += 1; reasons.append("multiple ring bond-order changes")

    if center >= 2:
        score += 1; reasons.append("reaction-center evidence on target ring")

    if mapping.method == "atom_map":
        score += 1; reasons.append("explicit atom mapping")
    if mapping.ambiguous_product_atomset:
        score -= 2; reasons.append("multiple distinct product MCS atom sets")
    if mapping.pair_confidence < cfg.p2_min_pair_conf:
        score -= 2; reasons.append("weak reactant-product correspondence")

    strong_loss = (laa >= 2 or lab >= 2 or a_loss >= 0.34 or b_loss >= 0.34)
    structural_support = (sp3 >= 1 or bochg >= 2 or int(metrics["hybridization_changes"]) >= 1)

    if (
        mf >= cfg.p1_min_ring_mapping
        and er >= cfg.p1_min_edge_retention
        and mapping.pair_confidence >= cfg.p1_min_pair_conf
        and strong_loss
        and structural_support
        and center >= 1
        and not mapping.ambiguous_product_atomset
        and score >= 8
    ):
        return 1, score, reasons

    if (
        mf >= cfg.p2_min_ring_mapping
        and er >= cfg.p2_min_edge_retention
        and mapping.pair_confidence >= cfg.p2_min_pair_conf
        and (laa >= 1 or lab >= 1)
        and center >= 1
        and score >= 4
    ):
        return 2, score, reasons

    if laa >= 1 or lab >= 1 or a_loss > 0 or b_loss > 0:
        return 3, score, reasons

    return 0, score, reasons


# --------------------------- reaction-level logic ------------------------- #

def reaction_global_counts(rmols: Sequence[Chem.Mol], pmols: Sequence[Chem.Mol], cfg: Config) -> Dict[str, int]:
    r_summ = [mol_summary(m, cfg) for m in rmols]
    p_summ = [mol_summary(m, cfg) for m in pmols]
    keys = ["n_rings_all", "n_rings_5_7", "n_aromatic_rings_5_7", "n_aromatic_atoms", "n_aromatic_bonds"]
    out = {}
    for k in keys:
        rv = sum(x[k] for x in r_summ)
        pv = sum(x[k] for x in p_summ)
        out[f"reactant_{k}"] = rv
        out[f"product_{k}"] = pv
        out[f"delta_{k}_RminusP"] = rv - pv
    return out


def analyze_reaction(rxn_id: str, rxn_class: object, reaction: str, cfg: Config) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    summary: Dict[str, object] = {
        "id": rxn_id,
        "class": rxn_class,
        "reactions": reaction,
        "status": "ok",
        "reaction_priority": 0,
        "best_score": 0,
        "n_candidate_rings": 0,
        "reason": "",
    }
    candidates: List[Dict[str, object]] = []

    try:
        left, right = split_reaction_smiles(reaction)
    except Exception as e:
        summary.update(status="parse_failed", reason=str(e))
        return summary, candidates

    r_smiles, rmols = parse_components(left)
    p_smiles, pmols = parse_components(right)
    summary["n_reactant_components"] = len(rmols)
    summary["n_product_components"] = len(pmols)
    if not rmols or not pmols:
        summary.update(status="mol_parse_failed", reason="No valid reactant or product molecules")
        return summary, candidates

    global_counts = reaction_global_counts(rmols, pmols, cfg)
    summary.update(global_counts)

    aromatic_component_count = 0
    for r_idx, rmol in enumerate(rmols):
        arings = aromatic_rings(rmol, cfg)
        if not arings:
            continue
        aromatic_component_count += 1
        mapping = find_pair_mapping(rmol, pmols, r_idx, cfg)
        if mapping is None:
            continue
        pmol = pmols[mapping.product_component]

        r_comp_summary = mol_summary(rmol, cfg)
        p_comp_summary = mol_summary(pmol, cfg)

        for ring in arings:
            metrics = mapped_ring_metrics(rmol, pmol, ring, mapping)
            priority, score, reasons = score_ring_candidate(metrics, mapping, cfg)
            if priority == 0:
                continue

            row: Dict[str, object] = {
                "id": rxn_id,
                "class": rxn_class,
                "reaction": reaction,
                "priority": priority,
                "score": score,
                "reactant_component": r_idx,
                "product_component": mapping.product_component,
                "reactant_component_smiles": r_smiles[r_idx],
                "product_component_smiles": p_smiles[mapping.product_component],
                "mapping_method": mapping.method,
                "mapping_pair_confidence": round(mapping.pair_confidence, 4),
                "mapping_reactant_coverage": round(mapping.reactant_coverage, 4),
                "mapping_product_coverage": round(mapping.product_coverage, 4),
                "mapping_ambiguous_product_atomset": mapping.ambiguous_product_atomset,
                "mcs_smarts": mapping.mcs_smarts,
                "ring_idx": ring.ring_idx,
                "ring_size": ring.size,
                "ring_elements": ring.elements,
                "reactant_ring_atom_indices": ",".join(map(str, ring.atom_indices)),
                "reactant_aromatic_rings_component": r_comp_summary["n_aromatic_rings_5_7"],
                "product_aromatic_rings_component": p_comp_summary["n_aromatic_rings_5_7"],
                "component_delta_aromatic_rings_RminusP": r_comp_summary["n_aromatic_rings_5_7"] - p_comp_summary["n_aromatic_rings_5_7"],
                "evidence": "; ".join(reasons),
            }
            row.update(metrics)
            row.update(global_counts)
            candidates.append(row)

    summary["n_aromatic_reactant_components"] = aromatic_component_count

    if candidates:
        best_priority = min(int(c["priority"]) for c in candidates)
        best_score = max(int(c["score"]) for c in candidates if int(c["priority"]) == best_priority)
        summary["reaction_priority"] = best_priority
        summary["best_score"] = best_score
        summary["n_candidate_rings"] = len(candidates)
        summary["reason"] = f"best matched target-ring evidence: P{best_priority}"
    else:
        # Global count loss alone is intentionally only P3 because it may be due
        # to an aromatic reactant/reagent not incorporated into the product.
        if (
            global_counts["delta_n_aromatic_rings_5_7_RminusP"] > 0
            or global_counts["delta_n_aromatic_atoms_RminusP"] >= 2
            or global_counts["delta_n_aromatic_bonds_RminusP"] >= 2
        ):
            summary["reaction_priority"] = 3
            summary["reason"] = "global aromaticity count decreased, but no reliable matched target ring; manual review"
        else:
            summary["reaction_priority"] = 0
            summary["reason"] = "no structural evidence of product-side dearomatization"

    return summary, candidates


# ------------------------------- review PNGs ------------------------------ #

def draw_candidate_row(row: pd.Series, out_path: Path) -> None:
    rmol = mol_from_smiles(str(row["reactant_component_smiles"]))
    pmol = mol_from_smiles(str(row["product_component_smiles"]))
    if rmol is None or pmol is None:
        return
    try:
        r_atoms = [int(x) for x in str(row["reactant_ring_atom_indices"]).split(",") if str(x).strip()]
        p_atoms = [int(x) for x in str(row["mapped_product_atom_indices"]).split(",") if str(x).strip()]
        img = Draw.MolsToGridImage(
            [rmol, pmol],
            molsPerRow=2,
            subImgSize=(500, 400),
            legends=["Reactant target ring", "Mapped product atoms"],
            highlightAtomLists=[r_atoms, p_atoms],
            useSVG=False,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(out_path))
    except Exception:
        pass


# ---------------------------------- main ---------------------------------- #

def load_table(path: str, reaction_col: str = "reactions") -> pd.DataFrame:
    # sep=None lets pandas infer tab/comma/semicolon in most USPTO exports.
    df = pd.read_csv(path, sep=None, engine="python")
    required = {"id", "class", reaction_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}. Found: {list(df.columns)}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Screen USPTO reactions for likely dearomatization")
    ap.add_argument("--input", required=True, help="USPTO-50K TSV/CSV with id,class,reactions")
    ap.add_argument("--outdir", default="dearom_screen_output", help="output directory")
    ap.add_argument("--reaction-col", default="reactions", help="reaction SMILES column; e.g. reactions or mapped_reactions")
    ap.add_argument("--limit", type=int, default=0, help="debug: only process first N rows")
    ap.add_argument("--mcs-timeout", type=int, default=1, help="MCS timeout per pair, seconds")
    ap.add_argument("--draw", type=int, default=0, help="draw up to N P1/P2 candidates")
    args = ap.parse_args()

    cfg = Config(mcs_timeout_s=args.mcs_timeout, draw_limit=args.draw)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_table(args.input, args.reaction_col)
    if args.limit > 0:
        df = df.head(args.limit).copy()

    summaries: List[Dict[str, object]] = []
    candidates: List[Dict[str, object]] = []

    records = df[["id", "class", args.reaction_col]].rename(columns={args.reaction_col: "reactions"}).to_dict(orient="records")
    for row in tqdm(records, total=len(records), desc="Screening"):
        rxn_id = str(row["id"])
        rxn_class = row["class"]
        reaction = str(row["reactions"])
        s, c = analyze_reaction(rxn_id, rxn_class, reaction, cfg)
        summaries.append(s)
        candidates.extend(c)

    sdf = pd.DataFrame(summaries)
    cdf = pd.DataFrame(candidates)

    sdf.to_csv(outdir / "screened_reactions.csv", index=False)
    cdf.to_csv(outdir / "candidate_rings.csv", index=False)

    for p in (1, 2, 3):
        sub = sdf[sdf["reaction_priority"] == p].copy()
        sub.to_csv(outdir / f"priority_{p}.csv", index=False)

    if not cdf.empty:
        # A compact best-ring table: one best candidate ring per reaction.
        best = (
            cdf.sort_values(["priority", "score"], ascending=[True, False])
               .drop_duplicates("id", keep="first")
        )
        best.to_csv(outdir / "best_candidate_per_reaction.csv", index=False)

        if args.draw > 0:
            review = best[best["priority"].isin([1, 2])].head(args.draw)
            for _, r in tqdm(review.iterrows(), total=len(review), desc="Drawing review images"):
                safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(r["id"]))
                draw_candidate_row(r, outdir / "review_images" / f"P{int(r['priority'])}_{safe_id}_ring{int(r['ring_idx'])}.png")

    counts = sdf["reaction_priority"].value_counts().sort_index().to_dict()
    report = {
        "input_rows": int(len(df)),
        "priority_counts": {str(k): int(v) for k, v in counts.items()},
        "candidate_ring_rows": int(len(cdf)),
        "config": asdict(cfg),
        "interpretation": {
            "P1": "high-confidence: mapped/retained target ring with strong aromaticity loss and local structural change",
            "P2": "probable: mapped ring with partial aromaticity loss, but weaker correspondence or magnitude",
            "P3": "manual review: global/partial evidence or uncertain ring correspondence",
            "P0": "no detected dearomatization signal",
        },
    }
    with open(outdir / "screening_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
