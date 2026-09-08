#!/usr/bin/env python3
"""
Step 5: 生成Multiwfn NICS_ZZ输入txt (严格遵循CODE-IN的Multiwfn方式) — config-driven
- 从NMR log提取所有Bq屏蔽张量(9分量)
- 结合ring_info.csv的环原子信息
- 生成Multiwfn function 25 option 4的输入文件

Multiwfn输入格式:
  25                          <- 进入功能25 (芳香性分析)
  4                           <- 选择 NICS_ZZ for non-planar or tilted system
  (空行)                      <- 使用环原子几何中心
  ring_atoms                  <- 环原子索引(1-based, 逗号分隔)
  XX,YX,ZX                    <- 张量第1行
  XY,YY,ZY                    <- 张量第2行
  XZ,YZ,ZZ                    <- 张量第3行
  4                           <- 再次选择option 4 (下一个Bq点)
  (空行)
  ring_atoms
  ...
  0                           <- 返回主菜单
  q                           <- 退出

用法:
    python3 05_gen_nics_multiwfn_input.py <dataset>
    python3 05_gen_nics_multiwfn_input.py lunci9
"""
import os
import sys
import re
import ast
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def extract_all_bq_tensors(log_path):
    """从NMR log提取所有Bq原子的屏蔽张量(9分量)
    Gaussian格式:
         26  Bq   Isotropic =    10.3923   Anisotropy =    23.6952
       XX=     5.1903   YX=     1.3294   ZX=     4.9421
       XY=     3.3050   YY=     4.3791   ZY=     9.3249
       XZ=     7.6334   YZ=     4.5411   ZZ=    21.6077
    返回: [(XX, YX, ZX, XY, YY, ZY, XZ, YZ, ZZ), ...]
    """
    tensors = []
    with open(log_path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        if re.search(r'^\s*\d+\s+Bq\s+Isotropic\s*=', line):
            tensor = []
            for j in range(i + 1, min(i + 4, len(lines))):
                vals = re.findall(r'[XYZ][XYZZ]=\s*([-\d\.]+)', lines[j])
                if len(vals) == 3:
                    tensor.extend(float(v) for v in vals)
            if len(tensor) == 9:
                tensors.append(tuple(tensor))
        i += 1
    return tensors


def main():
    if len(sys.argv) < 2:
        print("用法: python3 05_gen_nics_multiwfn_input.py <dataset>")
        sys.exit(1)
    dataset = sys.argv[1]

    NICS_LOG_DIR = config.get(dataset, "nics_log_dir")
    RING_INFO_CSV = config.get(dataset, "ring_info_csv")
    LOG_PATTERN = config.get(dataset, "nics_log_pattern")
    INPUT_DIR = config.get(dataset, "multiwfn_input_dir")

    os.makedirs(INPUT_DIR, exist_ok=True)

    print("=" * 60)
    print(f"Step 5: 生成Multiwfn NICS_ZZ输入 ({dataset})")
    print(f"  NMR log目录: {NICS_LOG_DIR}")
    print(f"  环信息CSV: {RING_INFO_CSV}")
    print(f"  输出目录: {INPUT_DIR}")
    print("=" * 60)

    # 读取环信息
    ring_df = pd.read_csv(RING_INFO_CSV)
    mol_rings = {}
    for _, row in ring_df.iterrows():
        mol_id = row['New_ID']
        ring_id = row['Ring_ID']
        atoms_str = row['Ring_Atoms']
        if pd.isna(atoms_str):
            continue
        try:
            atoms = ast.literal_eval(str(atoms_str))
        except (ValueError, SyntaxError):
            continue
        atoms_str_1based = ','.join(str(a + 1) for a in atoms)  # RDKit 0-based → Gaussian 1-based
        mol_rings.setdefault(mol_id, []).append((ring_id, atoms_str_1based))

    print(f"环信息: {len(ring_df)}条记录, {len(mol_rings)}个分子")

    # 遍历NMR log文件
    input_count = 0
    skip_count = 0
    for log_file in sorted(os.listdir(NICS_LOG_DIR)):
        if not log_file.endswith('.log'):
            continue
        match = re.match(LOG_PATTERN, log_file)
        if not match:
            continue
        mol_id = match.group(1)

        if mol_id not in mol_rings:
            skip_count += 1
            continue

        rings = mol_rings[mol_id]
        num_rings = len(rings)
        expected_bq = num_rings * 2

        log_path = os.path.join(NICS_LOG_DIR, log_file)
        tensors = extract_all_bq_tensors(log_path)

        if len(tensors) < expected_bq:
            print(f"警告: {log_file} Bq数={len(tensors)}, 预期{expected_bq} (分子{mol_id}, {num_rings}环)")
            skip_count += 1
            continue

        # 生成Multiwfn输入
        lines = ["25"]  # 进入功能25
        for ring_idx, (ring_id, atoms_str) in enumerate(rings):
            bq_idx_base = ring_idx * 2  # 每环2个Bq点
            for bq_offset in range(2):
                bq = tensors[bq_idx_base + bq_offset]
                lines.append("4")           # 选择option 4
                lines.append("")            # 空行: 使用几何中心
                lines.append(atoms_str)     # 环原子
                lines.append(f"{bq[0]},{bq[1]},{bq[2]}")   # XX,YX,ZX
                lines.append(f"{bq[3]},{bq[4]},{bq[5]}")   # XY,YY,ZY
                lines.append(f"{bq[6]},{bq[7]},{bq[8]}")   # XZ,YZ,ZZ
        lines.append("0")  # 返回主菜单
        lines.append("q")  # 退出

        out_path = os.path.join(INPUT_DIR, f"{mol_id}-nics-input.txt")
        with open(out_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        input_count += 1

    print(f"\n生成Multiwfn输入: {input_count}个 -> {INPUT_DIR}")
    print(f"跳过: {skip_count}个")
    print(f"\n下一步: bash 06_run_multiwfn_nics.sh {dataset}")


if __name__ == "__main__":
    main()
