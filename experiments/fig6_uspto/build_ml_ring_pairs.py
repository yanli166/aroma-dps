#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Optional
import pandas as pd
from rdkit import Chem


def normalize_reaction_smiles(rxn: str) -> Tuple[str,str,str]:
    rxn=str(rxn).strip()
    if '>>' in rxn:
        l,r=rxn.split('>>',1); return l.strip(),'',r.strip()
    p=rxn.split('>')
    if len(p)==3: return p[0].strip(),p[1].strip(),p[2].strip()
    raise ValueError('bad reaction SMILES')


def mols_from_side(side: str):
    smis,mols=[],[]
    for s in side.split('.'):
        s=s.strip()
        if not s: continue
        m=Chem.MolFromSmiles(s)
        if m is not None:
            smis.append(s); mols.append(m)
    return smis,mols


def parse_map_numbers(v: Any)->List[int]:
    if v is None or (isinstance(v,float) and math.isnan(v)): return []
    s=str(v).strip().strip('[](){}').replace(',',';').replace(' ',';')
    return sorted({int(x) for x in s.split(';') if x.strip().isdigit() and int(x)>0})


def atom_map_to_idx(mol: Chem.Mol)->Dict[int,int]:
    return {a.GetAtomMapNum():a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum()>0}


def find_component(mols: List[Chem.Mol], maps: Set[int]):
    best=(None,None,0.0)
    for i,m in enumerate(mols):
        have=set(atom_map_to_idx(m)); cov=len(have & maps)/max(1,len(maps))
        if cov>best[2]: best=(i,m,cov)
    return best


def strip_maps(mol: Chem.Mol)->str:
    m=Chem.Mol(mol)
    for a in m.GetAtoms(): a.SetAtomMapNum(0)
    return Chem.MolToSmiles(m,canonical=True,isomericSmiles=True)


def mapped_smiles(mol: Chem.Mol)->str:
    return Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True)


def ring_indices(mol: Chem.Mol,maps:Set[int])->List[int]:
    d=atom_map_to_idx(mol); return sorted(d[x] for x in maps if x in d)


def mask(n:int,idx:List[int])->List[int]:
    s=set(idx); return [1 if i in s else 0 for i in range(n)]


def ring_bonds_idx(mol: Chem.Mol,idx:List[int])->List[Tuple[int,int]]:
    s=set(idx); out=set()
    for i in idx:
        for b in mol.GetAtomWithIdx(i).GetBonds():
            j=b.GetOtherAtomIdx(i)
            if j in s: out.add(tuple(sorted((i,j))))
    return sorted(out)


def ring_bonds_map(mol: Chem.Mol,maps:Set[int])->List[Tuple[int,int]]:
    d=atom_map_to_idx(mol); rev={v:k for k,v in d.items()}
    idx=[d[x] for x in maps if x in d]
    return sorted({tuple(sorted((rev[i],rev[j]))) for i,j in ring_bonds_idx(mol,idx)})


def local_env(mol: Chem.Mol,idx:List[int],radius:int)->List[int]:
    sel=set(idx); frontier=set(idx)
    for _ in range(radius):
        new=set()
        for i in frontier: new.update(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors())
        new-=sel; sel|=new; frontier=new
    return sorted(sel)


def frag_smiles(mol: Chem.Mol,idx:List[int])->str:
    try: return Chem.MolFragmentToSmiles(mol,atomsToUse=idx,canonical=True,isomericSmiles=True)
    except Exception: return ''


def ordered_cycle_maps(mol: Chem.Mol,maps:Set[int])->List[int]:
    d=atom_map_to_idx(mol)
    if any(x not in d for x in maps): return []
    idxs={d[x] for x in maps}; adj={i:[] for i in idxs}
    for i in idxs:
        for nb in mol.GetAtomWithIdx(i).GetNeighbors():
            if nb.GetIdx() in idxs: adj[i].append(nb.GetIdx())
    start=min(idxs,key=lambda i: mol.GetAtomWithIdx(i).GetAtomMapNum())
    n=len(idxs)
    def dfs(path):
        cur=path[-1]
        if len(path)==n: return path if start in adj[cur] else None
        for nxt in sorted(adj[cur],key=lambda x: mol.GetAtomWithIdx(x).GetAtomMapNum()):
            if nxt==start or nxt in path: continue
            z=dfs(path+[nxt])
            if z is not None: return z
        return None
    p=dfs([start])
    if p is None: return sorted(maps)
    vals=[mol.GetAtomWithIdx(i).GetAtomMapNum() for i in p]
    rev=[vals[0]]+list(reversed(vals[1:]))
    return min(vals,rev)


def main():
    ap=argparse.ArgumentParser(description='Build ML-ready reactant/product target-ring pairs from Stage-2 outputs')
    ap.add_argument('--ring-evidence',required=True)
    ap.add_argument('--reactions',required=True)
    ap.add_argument('--outdir',required=True)
    ap.add_argument('--tier',default='ALL')
    ap.add_argument('--reactions-filter',default=None,
                    help='CSV whitelist (e.g. tier_A_exact.csv); only ring-evidence rows whose reaction_id appears here are kept')
    ap.add_argument('--local-radius',type=int,default=2)
    args=ap.parse_args()
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    rdf=pd.read_csv(args.ring_evidence,low_memory=False)
    xdf=pd.read_csv(args.reactions,low_memory=False)
    if args.tier.upper()!='ALL' and 'tier' in rdf.columns:
        rdf=rdf[rdf['tier'].astype(str).str.upper()==args.tier.upper()].copy()
    if args.reactions_filter:
        fdf=pd.read_csv(args.reactions_filter,low_memory=False,usecols=['reaction_id'])
        keep=set(fdf['reaction_id'].astype(str))
        before=len(rdf)
        rdf=rdf[rdf['reaction_id'].astype(str).isin(keep)].copy()
        print(f'[filter] ring_evidence {before} -> {len(rdf)} rows kept by reactions-filter whitelist')
    use_row='source_row' in rdf.columns and 'source_row' in xdf.columns
    lookup={}
    for _,r in xdf.iterrows():
        k=(str(r['reaction_id']),int(r['source_row'])) if use_row else str(r['reaction_id'])
        lookup[k]=r
    flat=[]; nested=[]; errs=[]
    for ri,ev in rdf.iterrows():
        rid=str(ev['reaction_id']); srow=int(ev['source_row']) if use_row else None
        k=(rid,srow) if use_row else rid; rxn=lookup.get(k)
        if rxn is None:
            errs.append({'reaction_id':rid,'source_row':srow,'error':'REACTION_NOT_FOUND'}); continue
        mapped=str(rxn.get('mapped_reaction','') or '')
        maps=set(parse_map_numbers(ev.get('ring_map_numbers')))
        if not mapped or mapped.lower()=='nan' or len(maps)<3:
            errs.append({'reaction_id':rid,'source_row':srow,'error':'MISSING_MAPPING_OR_RING_MAPS'}); continue
        try:
            l,_,p=normalize_reaction_smiles(mapped); _,rmols=mols_from_side(l); _,pmols=mols_from_side(p)
        except Exception:
            errs.append({'reaction_id':rid,'source_row':srow,'error':'REACTION_PARSE_FAILED'}); continue
        rci,rmol,rcov=find_component(rmols,maps); pci,pmol,pcov=find_component(pmols,maps)
        if rmol is None or pmol is None or rcov<1 or pcov<1:
            errs.append({'reaction_id':rid,'source_row':srow,'error':'TARGET_RING_NOT_FULLY_PRESENT','reactant_cov':rcov,'product_cov':pcov}); continue
        ridx=ring_indices(rmol,maps); pidx=ring_indices(pmol,maps)
        redges=set(ring_bonds_map(rmol,maps)); pedges=set(ring_bonds_map(pmol,maps)); retained=len(redges & pedges)
        pair_id=f'{rid}__ring_{ri}'
        rmask=mask(rmol.GetNumAtoms(),ridx); pmask=mask(pmol.GetNumAtoms(),pidx)
        ro=ordered_cycle_maps(rmol,maps); po=ordered_cycle_maps(pmol,maps)
        row={
            'pair_id':pair_id,'reaction_id':rid,'source_row':srow,
            'stage2_tier':str(ev.get('tier','')),'decision_reason':str(ev.get('decision_reason','')),
            'mapped_reaction':mapped,'mapping_confidence':rxn.get('mapping_confidence',''),
            'reactant_component_index':rci,'product_component_index':pci,
            'reactant_component_smiles':strip_maps(rmol),'product_component_smiles':strip_maps(pmol),
            'reactant_component_smiles_mapped':mapped_smiles(rmol),'product_component_smiles_mapped':mapped_smiles(pmol),
            'target_ring_size':len(maps),'target_ring_map_numbers':';'.join(map(str,sorted(maps))),
            'target_ring_order_reactant':';'.join(map(str,ro)),'target_ring_order_product':';'.join(map(str,po)),
            'reactant_ring_atom_indices':';'.join(map(str,ridx)),'product_ring_atom_indices':';'.join(map(str,pidx)),
            'reactant_ring_mask':''.join(map(str,rmask)),'product_ring_mask':''.join(map(str,pmask)),
            'reactant_ring_bonds_map':';'.join(f'{a}-{b}' for a,b in sorted(redges)),
            'product_ring_bonds_map':';'.join(f'{a}-{b}' for a,b in sorted(pedges)),
            'ring_edge_retention':retained/max(1,len(redges)),'same_cycle_edges':redges==pedges,
            'reactant_ring_fragment_smiles':frag_smiles(rmol,ridx),'product_ring_fragment_smiles':frag_smiles(pmol,pidx),
            f'reactant_radius{args.local_radius}_atom_indices':';'.join(map(str,local_env(rmol,ridx,args.local_radius))),
            f'product_radius{args.local_radius}_atom_indices':';'.join(map(str,local_env(pmol,pidx,args.local_radius))),
            'lost_aromatic_atoms':ev.get('lost_aromatic_atoms',''),'lost_aromatic_ring_edges':ev.get('lost_aromatic_ring_edges',''),
            'new_sp3_ring_atoms':ev.get('new_sp3_ring_atoms',''),'new_stereocenters_on_ring':ev.get('new_stereocenters_on_ring',''),
            'exact_score':ev.get('exact_score','')
        }
        flat.append(row)
        nested.append({
            'pair_id':pair_id,'reaction_id':rid,'target_ring':{'map_numbers':sorted(maps),'reactant_order':ro,'product_order':po},
            'reactant':{'smiles':strip_maps(rmol),'smiles_mapped':mapped_smiles(rmol),'ring_atom_indices':ridx,'ring_mask':rmask,'ring_bonds_map':sorted(redges)},
            'product':{'smiles':strip_maps(pmol),'smiles_mapped':mapped_smiles(pmol),'ring_atom_indices':pidx,'ring_mask':pmask,'ring_bonds_map':sorted(pedges)},
            'validation':{'ring_edge_retention':retained/max(1,len(redges)),'same_cycle_edges':redges==pedges}
        })
    fdf=pd.DataFrame(flat); edf=pd.DataFrame(errs)
    fdf.to_csv(out/'ring_pairs_ml.csv',index=False); edf.to_csv(out/'ring_pair_errors.csv',index=False)
    with open(out/'ring_pairs_ml.jsonl','w',encoding='utf-8') as f:
        for x in nested: f.write(json.dumps(x,ensure_ascii=False)+'\n')
    ml=[]
    for _,r in fdf.iterrows():
        for side in ('reactant','product'):
            ml.append({
                'sample_id':f"{r['pair_id']}__{'R' if side=='reactant' else 'P'}",'pair_id':r['pair_id'],'reaction_id':r['reaction_id'],'side':side,
                'smiles':r[f'{side}_component_smiles'],'smiles_mapped':r[f'{side}_component_smiles_mapped'],
                'target_atom_indices':r[f'{side}_ring_atom_indices'],'target_mask':r[f'{side}_ring_mask'],
                'target_ring_map_numbers':r['target_ring_map_numbers']
            })
    mdf=pd.DataFrame(ml); mdf.to_csv(out/'ring_property_ml_samples.csv',index=False)
    rep={'tier_requested':args.tier,'n_ring_pairs':len(fdf),'n_ml_samples':len(mdf),'n_errors':len(edf),
         'same_cycle_edge_fraction':float(fdf['same_cycle_edges'].mean()) if len(fdf) else None,
         'mean_ring_edge_retention':float(fdf['ring_edge_retention'].mean()) if len(fdf) else None,'local_radius':args.local_radius}
    (out/'ring_pair_report.json').write_text(json.dumps(rep,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(rep,ensure_ascii=False,indent=2)); print('Outputs:',out)

if __name__=='__main__': main()
