#!/usr/bin/env python3
"""
Step 4: 解析HOMA和MBCO的Multiwfn输出 (S态闭壳层) — config-driven
输出: {out_dir}/{dataset}-homa-mbco-summary.csv

用法:
    python3 04_parse_homa_mbco.py <dataset>
    python3 04_parse_homa_mbco.py lunci9
"""
import os
import sys
import re
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def extract_homa(txt_file):
    values = []
    with open(txt_file) as f:
        for line in f:
            m = re.search(r'HOMA value is\s+([-\d\.]+)', line)
            if m:
                values.append(float(m.group(1)))
    return values


def extract_mbco_closed_shell(txt_file):
    """S态(闭壳层): 直接取 'The normalized multicenter bond order:'"""
    values = []
    with open(txt_file) as f:
        for line in f:
            m = re.search(r'The normalized multicenter bond order:\s+([-\d\.]+)', line)
            if m:
                values.append(float(m.group(1)))
    return values


def parse_filename(filename, prefix):
    m = re.search(rf'{prefix}-(\w+)-ring(\d+)\.txt-out\.txt', filename)
    if m:
        return m.group(1), int(m.group(2))
    return None, None


def main():
    if len(sys.argv) < 2:
        print("用法: python3 04_parse_homa_mbco.py <dataset>")
        sys.exit(1)
    dataset = sys.argv[1]

    RING_INFO_CSV = config.get(dataset, "ring_info_csv")
    HOMA_OUT_DIR = config.get(dataset, "outhoma_dir")
    MBCO_OUT_DIR = config.get(dataset, "outmbco_dir")
    OUTPUT_CSV = config.get(dataset, "homa_mbco_csv")

    ring_df = pd.read_csv(RING_INFO_CSV)
    print(f"环信息: {len(ring_df)}条记录 ({dataset})")

    results = {}
    for _, row in ring_df.iterrows():
        key = (row['New_ID'], row['Ring_ID'])
        results[key] = {
            'New_ID': row['New_ID'], 'SMILES': row['SMILES'],
            'Ring_ID': row['Ring_ID'], 'Ring_Size': row['Ring_Size'],
            'Ring_Atoms': row['Ring_Atoms'],
            'HOMA': None, 'MBCO': None
        }

    # 解析HOMA
    homa_count = 0
    if os.path.isdir(HOMA_OUT_DIR):
        for filename in os.listdir(HOMA_OUT_DIR):
            if not (filename.startswith('homa-') and filename.endswith('out.txt')):
                continue
            new_id, ring_id = parse_filename(filename, 'homa')
            if new_id is None:
                continue
            key = (new_id, ring_id)
            if key in results:
                values = extract_homa(os.path.join(HOMA_OUT_DIR, filename))
                if values:
                    results[key]['HOMA'] = values[0]
                    homa_count += 1
    print(f"HOMA解析: {homa_count}个值")

    # 解析MBCO (S态闭壳层)
    mbco_count = 0
    if os.path.isdir(MBCO_OUT_DIR):
        for filename in os.listdir(MBCO_OUT_DIR):
            if not (filename.startswith('mbco-') and filename.endswith('out.txt')):
                continue
            new_id, ring_id = parse_filename(filename, 'mbco')
            if new_id is None:
                continue
            key = (new_id, ring_id)
            if key in results:
                values = extract_mbco_closed_shell(os.path.join(MBCO_OUT_DIR, filename))
                if values:
                    results[key]['MBCO'] = values[0]
                    mbco_count += 1
    print(f"MBCO解析: {mbco_count}个值")

    result_df = pd.DataFrame(list(results.values()))
    result_df = result_df[['New_ID', 'SMILES', 'Ring_ID', 'Ring_Size', 'Ring_Atoms', 'HOMA', 'MBCO']]
    result_df.to_csv(OUTPUT_CSV, index=False)

    total = len(result_df)
    print(f"\n汇总CSV: {OUTPUT_CSV}")
    print(f"总记录: {total}, HOMA有效: {result_df['HOMA'].notna().sum()}, MBCO有效: {result_df['MBCO'].notna().sum()}")


if __name__ == "__main__":
    main()
