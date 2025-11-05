import os
import re
import pandas as pd
import numpy as np

def extract_nics_zz(txt_file):
    """提取 txt 文件中的所有 NICS_ZZ 值"""
    values = []
    with open(txt_file, 'r') as f:
        for line in f:
            match = re.search(r'The NICS_ZZ value is thus\s+([\d\.-]+)', line)
            if match:
                values.append(float(match.group(1)))
    return values

def parse_new_id(filename):
    """从文件名解析 New_ID（支持 c/b 等前缀）"""
    match = re.search(r'nics-([a-zA-Z]\d+)-out.txt', filename)
    return match.group(1) if match else None

def merge_nics_data(df, nics_values, new_id, output_file, mode='w'):
    """将 NICS 值合并到 DataFrame 并保存"""
    # 确保目标列存在
    for col in ['Ring_NICS_ZZ_1', 'Ring_NICS_ZZ_2']:
        if col not in df.columns:
            df[col] = np.nan
    
    ring_id = 1
    while nics_values and len(nics_values) >= 2:
        # 定位当前环的行
        mask = (df['New_ID'] == new_id) & (df['Ring_ID'] == ring_id)
        
        if not df[mask].empty:
            # 更新现有行
            df.loc[mask, 'Ring_NICS_ZZ_1'] = nics_values[0]
            df.loc[mask, 'Ring_NICS_ZZ_2'] = nics_values[1]
        else:
            # 新增行
            new_row = {
                'New_ID': new_id,
                'Ring_ID': ring_id,
                'Ring_NICS_ZZ_1': nics_values[0],
                'Ring_NICS_ZZ_2': nics_values[1]
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        nics_values = nics_values[2:]
        ring_id += 1
    
    # 保存到 CSV
    df.to_csv(output_file, index=False, mode=mode)
    return df

def process_folder(txt_folder, csv_path, output_path):
    """处理所有 txt 文件并更新 CSV"""
    df = pd.read_csv(csv_path)
    for filename in os.listdir(txt_folder):
        if not filename.endswith('.txt'):
            continue
            
        new_id = parse_new_id(filename)
        if not new_id:
            print(f"警告: 无法解析文件名 {filename}，跳过")
            continue
            
        txt_file = os.path.join(txt_folder, filename)
        nics_values = extract_nics_zz(txt_file)
        
        if not nics_values:
            print(f"警告: {filename} 中未找到 NICS_ZZ 值")
            continue
            
        df = merge_nics_data(df, nics_values, new_id, output_path, mode='a')
    
    # 最终去重保存
    df.drop_duplicates(subset=['New_ID', 'Ring_ID'], inplace=True)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    txt_folder = "/home/ubuntu/cal/DPSCAL/lunci3/NICSOUT1ZZ/"  # 替换为你的 txt 文件夹路径
    csv_path = "/home/ubuntu/cal/DPSCAL/lunci3/merged-lunci3-out.csv"
    output_path = "/home/ubuntu/cal/DPSCAL/lunci3/lunci3-nics1ZZ-out.csv"
    
    process_folder(txt_folder, csv_path, output_path)