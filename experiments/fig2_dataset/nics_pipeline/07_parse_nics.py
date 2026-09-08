#!/usr/bin/env python3
"""
Step 7: 解析Multiwfn NICS_ZZ输出, 生成最终汇总CSV — config-driven
- 从Multiwfn输出提取 "The NICS_ZZ value is thus"
- 每环2个Bq点(±1.0Å), 取平均
- 同时保留NICS_iso (各向同性, 与方向无关, 直接从Gaussian log解析)
- 合并 HOMA/MBCO/NICS_iso/NICS_ZZ 到最终CSV

用法:
    python3 07_parse_nics.py <dataset>
    python3 07_parse_nics.py lunci9
"""
import os
import sys
import re
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def extract_nics_zz_from_multiwfn(output_path):
    """从Multiwfn输出提取所有NICS_ZZ值
    返回: [nics_zz1, nics_zz2, ...] 按Bq顺序
    """
    values = []
    with open(output_path) as f:
        for line in f:
            m = re.search(r'The NICS_ZZ value is thus\s+([-\d\.]+)', line)
            if m:
                values.append(float(m.group(1)))
    return values


def extract_nics_iso_from_log(log_path):
    """从Gaussian NMR log直接提取NICS_iso (各向同性, 与方向无关)
    返回: [iso1, iso2, ...] 按Bq顺序
    """
    values = []
    with open(log_path) as f:
        for line in f:
            m = re.search(r'^\s*\d+\s+Bq\s+Isotropic\s*=\s*([-\d\.]+)', line)
            if m:
                values.append(float(m.group(1)))
    return values


def main():
    if len(sys.argv) < 2:
        print("用法: python3 07_parse_nics.py <dataset>")
        sys.exit(1)
    dataset = sys.argv[1]

    NICS_LOG_DIR = config.get(dataset, "nics_log_dir")
    RING_INFO_CSV = config.get(dataset, "ring_info_csv")
    HOMA_MBCO_CSV = config.get(dataset, "homa_mbco_csv")
    LOG_PATTERN = config.get(dataset, "nics_log_pattern")
    OUTPUT_DIR = config.get(dataset, "multiwfn_output_dir")
    FINAL_CSV = config.get(dataset, "final_csv")

    print("=" * 60)
    print(f"Step 7: 解析Multiwfn NICS_ZZ ({dataset})")
    print("=" * 60)

    # 读取环信息
    ring_df = pd.read_csv(RING_INFO_CSV)
    mol_rings = {}
    for _, row in ring_df.iterrows():
        mol_id = row['New_ID']
        ring_id = row['Ring_ID']
        mol_rings.setdefault(mol_id, []).append(ring_id)

    print(f"环信息: {len(ring_df)}条记录, {len(mol_rings)}个分子")

    # 读取HOMA/MBCO汇总
    summary_df = pd.read_csv(HOMA_MBCO_CSV)
    print(f"HOMA/MBCO汇总: {len(summary_df)}条记录")

    summary_df['NICS_iso'] = None
    summary_df['NICS_ZZ'] = None

    # 解析Multiwfn输出
    multiwfn_count = 0
    for output_file in sorted(os.listdir(OUTPUT_DIR)):
        if not output_file.endswith('-nics-output.txt'):
            continue
        mol_id = output_file.replace('-nics-output.txt', '')

        if mol_id not in mol_rings:
            continue

        ring_ids = mol_rings[mol_id]
        num_rings = len(ring_ids)
        expected_bq = num_rings * 2

        # 从Multiwfn输出提取NICS_ZZ
        output_path = os.path.join(OUTPUT_DIR, output_file)
        nics_zz_values = extract_nics_zz_from_multiwfn(output_path)

        # 从Gaussian log提取NICS_iso (各向同性)
        log_name = f"{mol_id}-nics.log"
        log_path = os.path.join(NICS_LOG_DIR, log_name)
        nics_iso_raw = []
        if os.path.exists(log_path):
            iso_values = extract_nics_iso_from_log(log_path)
            nics_iso_raw = [-v for v in iso_values]  # NICS_iso = -shielding_iso

        if len(nics_zz_values) < expected_bq:
            print(f"警告: {output_file} NICS_ZZ数={len(nics_zz_values)}, 预期{expected_bq}")
            continue

        # 分配到每个环
        for ring_idx, ring_id in enumerate(ring_ids):
            bq_base = ring_idx * 2
            zz1 = nics_zz_values[bq_base]
            zz2 = nics_zz_values[bq_base + 1]
            nics_zz_avg = (zz1 + zz2) / 2.0

            nics_iso_avg = None
            if len(nics_iso_raw) >= bq_base + 2:
                nics_iso_avg = (nics_iso_raw[bq_base] + nics_iso_raw[bq_base + 1]) / 2.0

            mask = (summary_df['New_ID'] == mol_id) & (summary_df['Ring_ID'] == ring_id)
            if mask.any():
                summary_df.loc[mask, 'NICS_iso'] = nics_iso_avg
                summary_df.loc[mask, 'NICS_ZZ'] = nics_zz_avg

        multiwfn_count += 1

    print(f"解析Multiwfn输出: {multiwfn_count}个分子")

    # 保存最终CSV
    summary_df = summary_df[['New_ID', 'SMILES', 'Ring_ID', 'Ring_Size', 'Ring_Atoms',
                              'HOMA', 'MBCO', 'NICS_iso', 'NICS_ZZ']]
    summary_df.to_csv(FINAL_CSV, index=False)

    total = len(summary_df)
    print(f"\n最终汇总CSV: {FINAL_CSV}")
    print(f"总记录: {total}")
    print(f"HOMA有效: {summary_df['HOMA'].notna().sum()}/{total}")
    print(f"MBCO有效: {summary_df['MBCO'].notna().sum()}/{total}")
    print(f"NICS_iso有效: {summary_df['NICS_iso'].notna().sum()}/{total}")
    print(f"NICS_ZZ有效: {summary_df['NICS_ZZ'].notna().sum()}/{total}")
    print(f"\nNICS_ZZ说明: Multiwfn function 25 option 4, 沿环法线方向 (正确NICS(1)ZZ)")


if __name__ == "__main__":
    main()
