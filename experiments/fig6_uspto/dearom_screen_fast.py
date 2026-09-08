#!/usr/bin/env python3
"""Fast two-stage prescreener for dearomatization reactions.

Designed for large USPTO-style corpora (including one-column USPTO_STEREO.csv).
It avoids FMCS on every reaction. The fast path tracks an aromatic reactant ring
into products using atom maps when available, otherwise an element-specific ring
SMARTS with any bond order. Only retained-ring aromaticity loss is promoted to
P1/P2; global aromaticity loss without a retained target ring is P3.

Input formats supported:
  reactions
  A.B>>C
or
  id,class,reactions
  ...
TSV is also supported.

Reaction SMILES may be reactants>>products or reactants>agents>products.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from dataclasses import dataclass, asdict
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")


# ------------------------------ input ---------------------------------

def _strip_outer_quotes(s: str) -> str:
    s = s.strip().lstrip("\ufeff")
    if len(s) >= 2 and s[0] == s[-1] and s[0] in {'"', "'"}:
        return s[1:-1]
    return s


def detect_delimiter(header_line: str) -> str:
    # Do not use csv.Sniffer here. On a one-column header named "reactions"
    # it can incorrectly infer the letter "t" as a delimiter.
    return "\t" if "\t" in header_line else ","


def iter_reactions(path: str, start_row: int = 0, limit: Optional[int] = None):
    """Yield (row_index, reaction_id, reaction_class, reaction_smiles).

    Robust to a one-column file and to commas inside an unquoted reaction string.
    """
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        first = fh.readline()
        if not first:
            return
        delim = detect_delimiter(first)
        header = next(csv.reader([first], delimiter=delim))
        header = [h.strip().lstrip("\ufeff") for h in header]

        # common case: a literal one-column header "reactions"
        has_header = "reactions" in header
        if has_header:
            rxn_idx = header.index("reactions")
            id_idx = header.index("id") if "id" in header else None
            class_idx = header.index("class") if "class" in header else None
        else:
            # no recognized header: treat the first physical line as data
            rxn_idx, id_idx, class_idx = 0, None, None
            fh.seek(0)

        emitted = 0
        data_idx = -1
        for physical_line in fh:
            physical_line = physical_line.rstrip("\r\n")
            if not physical_line:
                continue
            data_idx += 1
            if data_idx < start_row:
                continue
            if limit is not None and emitted >= limit:
                break

            if delim == "\t":
                row = next(csv.reader([physical_line], delimiter="\t"))
            else:
                row = next(csv.reader([physical_line], delimiter=","))

            # For a one-column corpus, reconstruct the raw reaction if commas were
            # unquoted. For id,class,reactions, reaction occupies the tail.
            if has_header and len(header) == 1:
                rxn = ",".join(row)
                rid = f"ROW_{data_idx:07d}"
                rclass = -1
            elif has_header:
                rid = row[id_idx] if id_idx is not None and id_idx < len(row) else f"ROW_{data_idx:07d}"
                rclass = row[class_idx] if class_idx is not None and class_idx < len(row) else -1
                if rxn_idx == len(header) - 1 and len(row) > len(header):
                    rxn = delim.join(row[rxn_idx:])
                else:
                    rxn = row[rxn_idx] if rxn_idx < len(row) else ""
            else:
                rxn = ",".join(row)
                rid = f"ROW_{data_idx:07d}"
                rclass = -1

            rxn = _strip_outer_quotes(rxn)
            yield data_idx, str(rid), rclass, rxn
            emitted += 1


def split_reaction(rxn: str) -> Optional[Tuple[str, str, str]]:
    # Both A>>B and A>agents>B become exactly three parts with split('>')
    parts = rxn.strip().split(">")
    if len(parts) != 3:
        return None
    return parts[0], parts[1], parts[2]


def mols_from_side(side: str) -> Optional[List[Chem.Mol]]:
    if side is None:
        return None
    mols: List[Chem.Mol] = []
    for s in side.split("."):
        s = s.strip()
        if not s:
            continue
        m = Chem.MolFromSmiles(s)
        if m is None:
            return None
        mols.append(m)
    return mols


# ------------------------------ ring chemistry -------------------------

@dataclass(frozen=True)
class RingRecord:
    comp_idx: int
    ring_idx: int
    atoms: Tuple[int, ...]
    bonds: Tuple[int, ...]
    size: int
    elements: Tuple[str, ...]
    aromatic_atoms: int
    aromatic_bonds: int
    fully_aromatic: bool


def ring_records(mol: Chem.Mol, comp_idx: int, min_ring: int, max_ring: int) -> List[RingRecord]:
    ri = mol.GetRingInfo()
    atom_rings = list(ri.AtomRings())
    bond_rings = list(ri.BondRings())
    out: List[RingRecord] = []
    for i, atoms in enumerate(atom_rings):
        n = len(atoms)
        if n < min_ring or n > max_ring:
            continue
        bonds = bond_rings[i] if i < len(bond_rings) else tuple()
        aa = sum(int(mol.GetAtomWithIdx(a).GetIsAromatic()) for a in atoms)
        ab = sum(int(mol.GetBondWithIdx(b).GetIsAromatic()) for b in bonds)
        elems = tuple(mol.GetAtomWithIdx(a).GetSymbol() for a in atoms)
        out.append(RingRecord(comp_idx, i, tuple(atoms), tuple(bonds), n, elems, aa, ab, aa == n and ab == n))
    return out


def ring_signature(rr: RingRecord) -> Tuple[int, Tuple[Tuple[str, int], ...]]:
    return rr.size, tuple(sorted(Counter(rr.elements).items()))


def side_inventory(mols: Sequence[Chem.Mol], min_ring: int, max_ring: int) -> Dict[str, int]:
    rings = []
    for ci, m in enumerate(mols):
        rings.extend(ring_records(m, ci, min_ring, max_ring))
    return {
        "heavy_atoms": sum(m.GetNumHeavyAtoms() for m in mols),
        "rings": len(rings),
        "fully_aromatic_rings": sum(r.fully_aromatic for r in rings),
        "aromatic_atoms": sum(sum(int(a.GetIsAromatic()) for a in m.GetAtoms()) for m in mols),
        "aromatic_bonds": sum(sum(int(b.GetIsAromatic()) for b in m.GetBonds()) for m in mols),
    }


def _ordered_cycle_atoms(mol: Chem.Mol, rr: RingRecord) -> Optional[List[int]]:
    """Order ring atoms around the cycle using only bonds belonging to this SSSR ring."""
    aset = set(rr.atoms)
    adj = {a: [] for a in rr.atoms}
    for bidx in rr.bonds:
        b = mol.GetBondWithIdx(bidx)
        a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if a1 in aset and a2 in aset:
            adj[a1].append(a2)
            adj[a2].append(a1)
    if not adj or any(len(v) != 2 for v in adj.values()):
        # Usually AtomRings() is already ordered; fallback to it.
        return list(rr.atoms)
    start = rr.atoms[0]
    order = [start]
    prev = None
    cur = start
    for _ in range(1, rr.size):
        nbrs = adj[cur]
        nxt = nbrs[0] if nbrs[0] != prev else nbrs[1]
        if nxt == start:
            break
        order.append(nxt)
        prev, cur = cur, nxt
    if len(order) != rr.size:
        return list(rr.atoms)
    return order


def ring_query_smarts(mol: Chem.Mol, rr: RingRecord) -> str:
    order = _ordered_cycle_atoms(mol, rr) or list(rr.atoms)
    atoms = [f"[#{mol.GetAtomWithIdx(a).GetAtomicNum()};R]" for a in order]
    # ring closure label 1; any bond order (~) so aromatic -> single/double is matchable
    if len(atoms) < 3:
        return ""
    return atoms[0] + "1~" + "~".join(atoms[1:]) + "~1"


def context_retention(rmol: Chem.Mol, ring_order: Sequence[int], pmol: Chem.Mol, match: Sequence[int]) -> float:
    """Fraction of inherited non-ring heavy-neighbor element counts retained.

    Additions are not penalized; only reactant-side external neighbors are required
    to be retained. With no external neighbors, return 1.0.
    """
    rset = set(ring_order)
    numerator = 0
    denominator = 0
    for ra, pa in zip(ring_order, match):
        r_ext = Counter(
            n.GetAtomicNum() for n in rmol.GetAtomWithIdx(ra).GetNeighbors()
            if n.GetIdx() not in rset and n.GetAtomicNum() > 1
        )
        p_match_set = set(match)
        p_ext = Counter(
            n.GetAtomicNum() for n in pmol.GetAtomWithIdx(pa).GetNeighbors()
            if n.GetIdx() not in p_match_set and n.GetAtomicNum() > 1
        )
        denominator += sum(r_ext.values())
        numerator += sum((r_ext & p_ext).values())
    return 1.0 if denominator == 0 else numerator / denominator


def _mapped_product_atoms_by_map(product_mols: Sequence[Chem.Mol]):
    d = {}
    for pi, pm in enumerate(product_mols):
        for a in pm.GetAtoms():
            amap = a.GetAtomMapNum()
            if amap > 0:
                d[amap] = (pi, a.GetIdx())
    return d


def direct_atom_map_match(rmol: Chem.Mol, ring_order: Sequence[int], product_mols: Sequence[Chem.Mol]):
    amap_index = _mapped_product_atoms_by_map(product_mols)
    maps = [rmol.GetAtomWithIdx(i).GetAtomMapNum() for i in ring_order]
    if not maps or any(x <= 0 for x in maps):
        return None
    hits = [amap_index.get(x) for x in maps]
    if any(h is None for h in hits):
        return None
    product_ids = {h[0] for h in hits}
    if len(product_ids) != 1:
        return None
    pi = next(iter(product_ids))
    return pi, tuple(h[1] for h in hits)


def ring_match_metrics(rmol: Chem.Mol, rr: RingRecord, ring_order: Sequence[int], pmol: Chem.Mol,
                       match: Sequence[int], method: str) -> Dict:
    n = len(ring_order)
    p_arom_atoms = sum(int(pmol.GetAtomWithIdx(i).GetIsAromatic()) for i in match)
    p_arom_bonds = 0
    retained_edges = 0
    bond_order_changes = 0
    for i in range(n):
        ra1, ra2 = ring_order[i], ring_order[(i + 1) % n]
        pa1, pa2 = match[i], match[(i + 1) % n]
        rb = rmol.GetBondBetweenAtoms(ra1, ra2)
        pb = pmol.GetBondBetweenAtoms(pa1, pa2)
        if pb is not None:
            retained_edges += 1
            p_arom_bonds += int(pb.GetIsAromatic())
            if rb is not None and (rb.GetBondType() != pb.GetBondType() or rb.GetIsAromatic() != pb.GetIsAromatic()):
                bond_order_changes += 1

    lost_atoms = n - p_arom_atoms  # reactant target is fully aromatic
    lost_bonds = n - p_arom_bonds
    new_sp3 = 0
    hyb_changes = 0
    degree_changes = 0
    charge_changes = 0
    new_stereocenters = 0
    for ra, pa in zip(ring_order, match):
        rat = rmol.GetAtomWithIdx(ra)
        pat = pmol.GetAtomWithIdx(pa)
        if pat.GetHybridization() == Chem.rdchem.HybridizationType.SP3:
            new_sp3 += 1
        if rat.GetHybridization() != pat.GetHybridization():
            hyb_changes += 1
        if rat.GetDegree() != pat.GetDegree():
            degree_changes += 1
        if rat.GetFormalCharge() != pat.GetFormalCharge():
            charge_changes += 1
        if (rat.GetChiralTag() == Chem.rdchem.ChiralType.CHI_UNSPECIFIED and
                pat.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED):
            new_stereocenters += 1

    ctx = context_retention(rmol, ring_order, pmol, match)
    edge_ret = retained_edges / n if n else 0.0
    loss_atom_frac = lost_atoms / n if n else 0.0
    loss_bond_frac = lost_bonds / n if n else 0.0

    # Evidence-weighted score; it is a ranking score, not a probability.
    score = 0.0
    score += 25.0 * min(1.0, edge_ret)
    score += 20.0 * max(0.0, min(1.0, loss_atom_frac))
    score += 20.0 * max(0.0, min(1.0, loss_bond_frac))
    score += 15.0 * max(0.0, min(1.0, ctx))
    score += 8.0 if new_sp3 > 0 else 0.0
    score += 5.0 if hyb_changes > 0 else 0.0
    score += 4.0 if bond_order_changes >= 2 else (2.0 if bond_order_changes == 1 else 0.0)
    score += 3.0 if new_stereocenters > 0 else 0.0
    if method == "atom_map":
        score += 5.0

    reaction_center_evidence = int(
        new_sp3 > 0 or hyb_changes > 0 or bond_order_changes > 0 or
        degree_changes > 0 or charge_changes > 0
    )

    return {
        "mapping_method": method,
        "ring_edge_retention": round(edge_ret, 4),
        "context_retention": round(ctx, 4),
        "product_aromatic_atoms_on_ring": p_arom_atoms,
        "product_aromatic_bonds_on_ring": p_arom_bonds,
        "lost_aromatic_atoms": lost_atoms,
        "lost_aromatic_bonds": lost_bonds,
        "aromatic_atom_loss_fraction": round(loss_atom_frac, 4),
        "aromatic_bond_loss_fraction": round(loss_bond_frac, 4),
        "new_sp3_ring_atoms": new_sp3,
        "hybridization_changes": hyb_changes,
        "ring_bond_order_changes": bond_order_changes,
        "ring_atom_degree_changes": degree_changes,
        "formal_charge_changes": charge_changes,
        "new_stereocenters_on_ring": new_stereocenters,
        "reaction_center_evidence": reaction_center_evidence,
        "score": round(score, 2),
    }


def classify_candidate(m: Dict, ring_size: int) -> str:
    # High-confidence structural dearomatization: same cyclic topology retained,
    # substantial aromaticity loss, and at least one local structural-change signal.
    substantial = (
        m["lost_aromatic_atoms"] >= 2 or m["lost_aromatic_bonds"] >= 2
    ) and (
        m["aromatic_atom_loss_fraction"] >= 0.33 or m["aromatic_bond_loss_fraction"] >= 0.33
    )
    if (m["ring_edge_retention"] >= 0.999 and substantial and
            m["reaction_center_evidence"] and m["context_retention"] >= 0.34):
        return "P1"
    # Partial dearomatization or weaker correspondence.
    if (m["ring_edge_retention"] >= 0.80 and
            (m["lost_aromatic_atoms"] >= 1 or m["lost_aromatic_bonds"] >= 1)):
        return "P2"
    return "P0"


def choose_best_ring_match(rmol: Chem.Mol, rr: RingRecord, product_mols: Sequence[Chem.Mol]) -> Optional[Dict]:
    order = _ordered_cycle_atoms(rmol, rr) or list(rr.atoms)

    # Exact mapped-atom path if available.
    dmatch = direct_atom_map_match(rmol, order, product_mols)
    matches: List[Tuple[int, Tuple[int, ...], str]] = []
    if dmatch is not None:
        matches.append((dmatch[0], dmatch[1], "atom_map"))
    else:
        smarts = ring_query_smarts(rmol, rr)
        q = Chem.MolFromSmarts(smarts) if smarts else None
        if q is None:
            return None
        for pi, pm in enumerate(product_mols):
            try:
                for mt in pm.GetSubstructMatches(q, uniquify=False, maxMatches=128):
                    matches.append((pi, tuple(mt), "ring_smarts"))
            except Exception:
                continue

    best = None
    for pi, mt, method in matches:
        pm = product_mols[pi]
        met = ring_match_metrics(rmol, rr, order, pm, mt, method)
        # Prefer retained context, then evidence score.
        key = (met["context_retention"], met["score"], met["lost_aromatic_atoms"], met["lost_aromatic_bonds"])
        if best is None or key > best[0]:
            best = (key, pi, mt, met)
    if best is None:
        return None
    _, pi, mt, met = best
    met = dict(met)
    met["product_comp_idx"] = pi
    met["product_match_atom_indices"] = ";".join(map(str, mt))
    met["ring_query_smarts"] = ring_query_smarts(rmol, rr)
    return met


# ------------------------------ worker ---------------------------------

WORKER_CFG = {"min_ring": 5, "max_ring": 7}


def init_worker(min_ring: int, max_ring: int):
    WORKER_CFG["min_ring"] = min_ring
    WORKER_CFG["max_ring"] = max_ring
    RDLogger.DisableLog("rdApp.*")


def screen_one(item):
    row_idx, rid, rclass, rxn = item
    parts = split_reaction(rxn)
    if parts is None:
        return {"status": "invalid_reaction", "row_idx": row_idx, "id": rid, "class": rclass, "reaction": rxn, "priority": "INVALID", "candidates": []}
    rside, agents, pside = parts
    rmols = mols_from_side(rside)
    pmols = mols_from_side(pside)
    if rmols is None or pmols is None or not rmols or not pmols:
        return {"status": "parse_failed", "row_idx": row_idx, "id": rid, "class": rclass, "reaction": rxn, "priority": "INVALID", "candidates": []}

    min_ring, max_ring = WORKER_CFG["min_ring"], WORKER_CFG["max_ring"]
    rinv = side_inventory(rmols, min_ring, max_ring)
    pinv = side_inventory(pmols, min_ring, max_ring)
    react_arom_rings = []
    for ci, m in enumerate(rmols):
        react_arom_rings.extend([r for r in ring_records(m, ci, min_ring, max_ring) if r.fully_aromatic])

    # Very cheap rejection: no fully aromatic 5-7 membered reactant ring.
    if not react_arom_rings:
        return {"status": "ok", "row_idx": row_idx, "id": rid, "class": rclass, "reaction": rxn, "priority": "P0", "candidates": [],
                "react_aromatic_rings": rinv["fully_aromatic_rings"], "prod_aromatic_rings": pinv["fully_aromatic_rings"],
                "delta_aromatic_rings": rinv["fully_aromatic_rings"] - pinv["fully_aromatic_rings"],
                "delta_aromatic_atoms": rinv["aromatic_atoms"] - pinv["aromatic_atoms"],
                "delta_aromatic_bonds": rinv["aromatic_bonds"] - pinv["aromatic_bonds"]}

    candidates = []
    best_priority = "P0"
    priority_rank = {"P0": 0, "P3": 1, "P2": 2, "P1": 3}

    for rr in react_arom_rings:
        rm = rmols[rr.comp_idx]
        met = choose_best_ring_match(rm, rr, pmols)
        if met is None:
            continue
        pr = classify_candidate(met, rr.size)
        if pr == "P0":
            continue
        rec = {
            "row_idx": row_idx,
            "id": rid,
            "class": rclass,
            "reaction": rxn,
            "reactant_comp_idx": rr.comp_idx,
            "reactant_ring_idx": rr.ring_idx,
            "ring_size": rr.size,
            "ring_elements": "-".join(rr.elements),
            "ring_signature": str(ring_signature(rr)),
            "reactant_ring_atom_indices": ";".join(map(str, rr.atoms)),
            **met,
            "priority": pr,
        }
        candidates.append(rec)
        if priority_rank[pr] > priority_rank[best_priority]:
            best_priority = pr

    global_loss = (
        rinv["fully_aromatic_rings"] > pinv["fully_aromatic_rings"] or
        rinv["aromatic_atoms"] > pinv["aromatic_atoms"] or
        rinv["aromatic_bonds"] > pinv["aromatic_bonds"]
    )
    # P3 means global aromaticity loss but no retained-ring dearomatization was
    # established. This intentionally captures fragment loss/ring opening/etc.
    if not candidates and global_loss:
        best_priority = "P3"

    return {
        "status": "ok",
        "row_idx": row_idx,
        "id": rid,
        "class": rclass,
        "reaction": rxn,
        "priority": best_priority,
        "candidates": candidates,
        "react_aromatic_rings": rinv["fully_aromatic_rings"],
        "prod_aromatic_rings": pinv["fully_aromatic_rings"],
        "delta_aromatic_rings": rinv["fully_aromatic_rings"] - pinv["fully_aromatic_rings"],
        "delta_aromatic_atoms": rinv["aromatic_atoms"] - pinv["aromatic_atoms"],
        "delta_aromatic_bonds": rinv["aromatic_bonds"] - pinv["aromatic_bonds"],
    }


SUMMARY_FIELDS = [
    "row_idx", "id", "class", "priority", "react_aromatic_rings", "prod_aromatic_rings",
    "delta_aromatic_rings", "delta_aromatic_atoms", "delta_aromatic_bonds", "reaction"
]

CANDIDATE_FIELDS = [
    "row_idx", "id", "class", "priority", "score", "mapping_method",
    "reactant_comp_idx", "product_comp_idx", "reactant_ring_idx", "ring_size", "ring_elements", "ring_signature",
    "reactant_ring_atom_indices", "product_match_atom_indices", "ring_query_smarts",
    "ring_edge_retention", "context_retention",
    "product_aromatic_atoms_on_ring", "product_aromatic_bonds_on_ring",
    "lost_aromatic_atoms", "lost_aromatic_bonds", "aromatic_atom_loss_fraction", "aromatic_bond_loss_fraction",
    "new_sp3_ring_atoms", "hybridization_changes", "ring_bond_order_changes", "ring_atom_degree_changes",
    "formal_charge_changes", "new_stereocenters_on_ring", "reaction_center_evidence", "reaction"
]


def safe_write(writer, row, fields):
    writer.writerow({k: row.get(k, "") for k in fields})


def main():
    ap = argparse.ArgumentParser(description="Fast retained-ring dearomatization prescreener for USPTO corpora")
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--workers", type=int, default=max(1, min(24, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--chunksize", type=int, default=256)
    ap.add_argument("--start-row", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-ring", type=int, default=5)
    ap.add_argument("--max-ring", type=int, default=7)
    ap.add_argument("--write-all", action="store_true", help="Also write P0 rows to screened_reactions_all.csv")
    ap.add_argument("--checkpoint-every", type=int, default=10000)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cand_path = outdir / "candidate_rings.csv"
    p_paths = {p: outdir / f"priority_{p[-1]}.csv" for p in ("P1", "P2", "P3")}
    screened_path = outdir / "screened_candidates.csv"
    all_path = outdir / "screened_reactions_all.csv"
    report_path = outdir / "screening_report.json"

    counts = Counter()
    processed = 0

    with open(cand_path, "w", newline="", encoding="utf-8") as fcand, \
         open(screened_path, "w", newline="", encoding="utf-8") as fscreen, \
         open(p_paths["P1"], "w", newline="", encoding="utf-8") as fp1, \
         open(p_paths["P2"], "w", newline="", encoding="utf-8") as fp2, \
         open(p_paths["P3"], "w", newline="", encoding="utf-8") as fp3:

        wcand = csv.DictWriter(fcand, fieldnames=CANDIDATE_FIELDS); wcand.writeheader()
        wscreen = csv.DictWriter(fscreen, fieldnames=SUMMARY_FIELDS); wscreen.writeheader()
        pfiles = {"P1": fp1, "P2": fp2, "P3": fp3}
        pwriters = {p: csv.DictWriter(f, fieldnames=SUMMARY_FIELDS) for p, f in pfiles.items()}
        for w in pwriters.values(): w.writeheader()

        fall = None; wall = None
        if args.write_all:
            fall = open(all_path, "w", newline="", encoding="utf-8")
            wall = csv.DictWriter(fall, fieldnames=SUMMARY_FIELDS); wall.writeheader()

        iterable = iter_reactions(args.input, start_row=args.start_row, limit=args.limit)
        with Pool(processes=args.workers, initializer=init_worker, initargs=(args.min_ring, args.max_ring)) as pool:
            for res in tqdm(pool.imap_unordered(screen_one, iterable, chunksize=args.chunksize), desc="Screening", unit="rxn"):
                processed += 1
                status = res.get("status", "unknown")
                counts[status] += 1
                pr = res.get("priority", "INVALID")
                counts[pr] += 1

                summary = {k: res.get(k, "") for k in SUMMARY_FIELDS}
                if pr in ("P1", "P2", "P3"):
                    safe_write(wscreen, summary, SUMMARY_FIELDS)
                    safe_write(pwriters[pr], summary, SUMMARY_FIELDS)
                if wall is not None:
                    safe_write(wall, summary, SUMMARY_FIELDS)

                for c in res.get("candidates", []):
                    safe_write(wcand, c, CANDIDATE_FIELDS)
                    counts[f"candidate_{c.get('priority','?')}"] += 1

                if processed % args.checkpoint_every == 0:
                    fcand.flush(); fscreen.flush(); fp1.flush(); fp2.flush(); fp3.flush()
                    if fall is not None: fall.flush()
                    tmp = {
                        "processed": processed,
                        "start_row": args.start_row,
                        "counts": dict(counts),
                        "input": str(args.input),
                        "workers": args.workers,
                        "min_ring": args.min_ring,
                        "max_ring": args.max_ring,
                    }
                    with open(report_path, "w", encoding="utf-8") as fr:
                        json.dump(tmp, fr, indent=2, ensure_ascii=False)

        if fall is not None:
            fall.close()

    report = {
        "processed": processed,
        "start_row": args.start_row,
        "counts": dict(counts),
        "input": str(args.input),
        "workers": args.workers,
        "min_ring": args.min_ring,
        "max_ring": args.max_ring,
        "notes": [
            "P1/P2 require a retained target-ring topology with local aromaticity loss.",
            "P3 is global aromaticity loss without a securely retained dearomatized ring; it is intentionally a manual-review pool.",
            "This is a structural prescreener, not a mechanistic dearomatization classifier.",
        ],
    }
    with open(report_path, "w", encoding="utf-8") as fr:
        json.dump(report, fr, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
