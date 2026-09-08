#!/usr/bin/env python3
"""
Step 1: 准备输入 (config-driven)
- 从SMILES提取环信息 -> ring_info.csv
- 基于chk生成HOMA/MBCO输入txt
- 基于优化log生成NICS GJF (含Bq点, 单重态)

用法:
    python3 01_prepare_inputs.py <dataset>
    python3 01_prepare_inputs.py lunci9
"""
import os
import sys
import re
import ast
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from numpy.linalg import svd, norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def extract_coords_from_log(log_path):
    """从Gaussian优化log提取最后一个 'Standard orientation:' 后的坐标"""
    with open(log_path) as f:
        lines = f.readlines()
    indices = [i + 5 for i, l in enumerate(lines) if 'Standard orientation:' in l]
    if not indices:
        return []
    coords = []
    for i in range(indices[-1], len(lines)):
        line = lines[i].strip()
        if not line or '----' in line:
            break
        parts = line.split()
        if len(parts) == 6 and parts[0].isdigit() and parts[1].isdigit():
            try:
                el = Chem.GetPeriodicTable().GetElementSymbol(int(parts[1]))
                coords.append((el, float(parts[3]), float(parts[4]), float(parts[5])))
            except ValueError:
                continue
    return coords


def calc_bq_points(ring_coords, height=1.0):
    """SVD拟合环平面, 返回环中心 ± height*法线 的两个Bq点"""
    if len(ring_coords) < 3:
        return []
    center = np.mean(ring_coords, axis=0)
    U, S, Vt = svd(ring_coords - center)
    normal = Vt[2] / norm(Vt[2])
    return [tuple(center + height * normal), tuple(center - height * normal)]


def main():
    if len(sys.argv) < 2:
        print("用法: python3 01_prepare_inputs.py <dataset>")
        sys.exit(1)
    dataset = sys.argv[1]

    INPUT_DIR = config.get(dataset, "input_dir")
    INPUT_CSV = config.get(dataset, "csv_path")
    OUT_DIR = config.get(dataset, "out_dir")
    RING_INFO_CSV = config.get(dataset, "ring_info_csv")
    HOMA_TXT_DIR = config.get(dataset, "homa_txt_dir")
    MBCO_TXT_DIR = config.get(dataset, "mbco_txt_dir")
    NICS_DIR = config.get(dataset, "nics_gjf_dir")
    OPT_LOG_PATTERN = config.get(dataset, "opt_log_pattern")
    OPT_LOG_PREFIX = config.get(dataset, "opt_log_prefix")
    MULTIPLICITY = config.get(dataset, "multiplicity")
    NICS_TITLE = config.get(dataset, "nics_title")

    for d in (OUT_DIR, HOMA_TXT_DIR, MBCO_TXT_DIR, NICS_DIR):
        os.makedirs(d, exist_ok=True)

    print("=" * 60)
    print(f"Step 1: 准备输入 ({dataset}, 多重度={MULTIPLICITY})")
    print(f"  输入目录: {INPUT_DIR}")
    print(f"  SMILES CSV: {INPUT_CSV}")
    print(f"  输出目录: {OUT_DIR}")
    print("=" * 60)

    # --- 1.1 提取环信息 ---
    df = pd.read_csv(INPUT_CSV)
    if 'SMILES' not in df.columns or 'no' not in df.columns:
        raise ValueError(f"CSV需要'no'和'SMILES'列, 实际列: {list(df.columns)}")

    results, errors = [], []
    for _, row in df.iterrows():
        no, smiles = row['no'], row['SMILES']
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            errors.append((no, smiles, 'Invalid SMILES'))
            continue
        AllChem.Compute2DCoords(mol)
        ring_info = mol.GetRingInfo()
        for ring_id, ring in enumerate(ring_info.AtomRings(), 1):
            results.append({
                'New_ID': no, 'SMILES': smiles,
                'Ring_ID': ring_id, 'Ring_Size': len(ring),
                'Ring_Atoms': str(list(ring))
            })

    ring_df = pd.DataFrame(results)
    ring_df.to_csv(RING_INFO_CSV, index=False)
    print(f"环信息: {len(ring_df)}条 -> {RING_INFO_CSV}")
    if errors:
        print(f"错误: {len(errors)}个: {errors[:5]}")

    # --- 1.2 生成HOMA/MBCO输入txt ---
    id_to_rings = {}
    for _, row in ring_df.iterrows():
        new_id = row['New_ID']
        ring_id = row['Ring_ID']
        atoms = ast.literal_eval(row['Ring_Atoms'])
        atoms_str = ','.join(str(a + 1) for a in atoms)  # RDKit 0-based → Gaussian 1-based
        id_to_rings.setdefault(new_id, {})[ring_id] = atoms_str

    homa_count = mbco_count = 0
    for mol_file in os.listdir(INPUT_DIR):
        if not (mol_file.endswith('.chk') or mol_file.endswith('.fchk')):
            continue
        mol_id = mol_file.split('.')[0]
        if mol_id not in id_to_rings:
            continue
        for ring_id, atoms_str in id_to_rings[mol_id].items():
            # HOMA: Multiwfn 25→6→0→atoms→q
            with open(os.path.join(HOMA_TXT_DIR, f'homa-{mol_id}-ring{ring_id}.txt'), 'w') as f:
                f.write(f'25\n6\n0\n{atoms_str}\nq\n')
            homa_count += 1
            # MBCO: Multiwfn 9→2→atoms→q
            with open(os.path.join(MBCO_TXT_DIR, f'mbco-{mol_id}-ring{ring_id}.txt'), 'w') as f:
                f.write(f'9\n2\n{atoms_str}\nq\n')
            mbco_count += 1

    print(f"HOMA输入txt: {homa_count}个 -> {HOMA_TXT_DIR}")
    print(f"MBCO输入txt: {mbco_count}个 -> {MBCO_TXT_DIR}")

    # --- 1.3 生成NICS GJF ---
    smiles_to_rings = {}
    for _, row in ring_df.iterrows():
        new_id = row['New_ID']
        smiles_to_rings.setdefault(new_id, {'SMILES': row['SMILES'], 'rings': []})
        smiles_to_rings[new_id]['rings'].append((row['Ring_ID'], ast.literal_eval(row['Ring_Atoms'])))

    nics_count = nics_errors = 0
    for log_file in sorted(os.listdir(INPUT_DIR)):
        if not log_file.endswith('.log'):
            continue
        match = re.match(OPT_LOG_PATTERN, log_file)
        if not match:
            continue
        mol_id = OPT_LOG_PREFIX + match.group(1)
        if mol_id not in smiles_to_rings:
            continue
        coords = extract_coords_from_log(os.path.join(INPUT_DIR, log_file))
        if not coords:
            nics_errors += 1
            continue

        # 构建3D分子用于环法线计算
        mol3d = Chem.RWMol()
        conf = Chem.Conformer()
        for i, (sym, x, y, z) in enumerate(coords):
            mol3d.AddAtom(Chem.Atom(sym))
            conf.SetAtomPosition(i, (x, y, z))
        mol3d.AddConformer(conf)

        all_bq = []
        for _, ring_atoms in smiles_to_rings[mol_id]['rings']:
            ring_coords = np.array([list(conf.GetAtomPosition(i)) for i in ring_atoms])
            all_bq.extend(calc_bq_points(ring_coords, 1.0))

        if all_bq:
            chk_name = f"{mol_id}-nics.chk"
            out_path = os.path.join(NICS_DIR, f"{mol_id}-nics.gjf")
            with open(out_path, 'w') as f:
                f.write("%mem=200GB\n%nprocshared=32\n")
                f.write(f"%chk={chk_name}\n")
                f.write("#P B3LYP/def2svp NMR nosymm\n\n")
                f.write(f"{NICS_TITLE}\n\n{MULTIPLICITY}\n")
                c = mol3d.GetConformer()
                for i in range(mol3d.GetNumAtoms()):
                    p = c.GetAtomPosition(i)
                    f.write(f"{mol3d.GetAtomWithIdx(i).GetSymbol()}    {p.x:.6f}    {p.y:.6f}    {p.z:.6f}\n")
                for bq in all_bq:
                    f.write(f"Bq    {bq[0]:.6f}    {bq[1]:.6f}    {bq[2]:.6f}\n")
                f.write("\n\n")
            nics_count += 1

    print(f"NICS GJF: {nics_count}个 -> {NICS_DIR} (错误: {nics_errors})")
    print(f"\nStep 1 完成。下一步: bash 02_run_formchk.sh {dataset}")


if __name__ == "__main__":
    main()
