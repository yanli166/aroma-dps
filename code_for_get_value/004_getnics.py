import os
import re
import pandas as pd
import numpy as np
import ast
import subprocess

# 设置路径（替换为您的实际路径）
csv_path = "/home/ubuntu/cal/DPSCAL/lunci4/merged-lunci4-out.csv"
fchk_directory = "/home/ubuntu/cal/DPSCAL/lunci4/NICS1"
log_directory = "/home/ubuntu/cal/DPSCAL/lunci4/NICS1"
txt_output_directory = "/home/ubuntu/cal/DPSCAL/lunci4/nicstxt"
multiwfn_output_directory = "/home/ubuntu/cal/DPSCAL/lunci4/NICSOUT1ZZ"
final_output_csv = "/home/ubuntu/cal/DPSCAL/lunci4/lunci4-nics1ZZ-out.csv"

# 确保目录存在
os.makedirs(txt_output_directory, exist_ok=True)
os.makedirs(multiwfn_output_directory, exist_ok=True)

# 从文档2复用的函数
def get_fchk_id(filename):
    base_name = filename.split('.')[0]
    return base_name.lstrip('B')

def parse_log_file(log_path):
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    data_blocks = []
    current_block = []
    capture = False
    
    for line in lines:
        if "Bq   Isotropic" in line:
            capture = True
            current_block = []
            continue
        if capture:
            if line.strip().startswith("XX="):
                current_block.append(line.strip())
            elif len(current_block) == 3:
                data_blocks.append(current_block)
                capture = False
    return data_blocks[:2]

# 生成TXT文件（文档2的逻辑）
def generate_txt_files():
    fchk_id_to_ring_data = {}
    df = pd.read_csv(csv_path)
    
    for index, row in df.iterrows():
        fchk_id = get_fchk_id(row['New_ID'])
        ring_id = row['Ring_ID']
        
        atoms = ast.literal_eval(row['Ring_Atoms'])
        atoms_plus1 = [str(atom + 1) for atom in atoms]
        atoms_str = ','.join(atoms_plus1)
        
        log_file = f"B{fchk_id}.log"
        log_path = os.path.join(log_directory, log_file)
        tensor_data = parse_log_file(log_path) if os.path.exists(log_path) else []
        
        if fchk_id not in fchk_id_to_ring_data:
            fchk_id_to_ring_data[fchk_id] = {}
        fchk_id_to_ring_data[fchk_id][ring_id] = {
            'atoms': atoms_str,
            'tensors': tensor_data
        }
    
    for fchk_file in os.listdir(fchk_directory):
        if fchk_file.endswith('.fchk'):
            fchk_id = get_fchk_id(fchk_file)
            if fchk_id in fchk_id_to_ring_data:
                merged_content = []
                for ring_id in sorted(fchk_id_to_ring_data[fchk_id].keys()):
                    data = fchk_id_to_ring_data[fchk_id][ring_id]
                    atoms = data['atoms']
                    tensors = data['tensors']
                    
                    block = [
                        "y",
                        "25",
                        "4",
                        atoms,
                        atoms
                    ]
                    for tensor in tensors:
                        block.extend(tensor)
                        block.extend(["4", atoms, atoms])
                    merged_content.extend(block)
                
                txt_filename = f"nics-{fchk_id}.txt"
                with open(os.path.join(txt_output_directory, txt_filename), 'w') as f:
                    f.write('\n'.join(merged_content))

# 运行Multiwfn（替代文档3的Bash脚本）
def run_multiwfn():
    for fchk_file in os.listdir(fchk_directory):
        if fchk_file.endswith('.fchk'):
            fchk_id = get_fchk_id(fchk_file)
            txt_filename = f"nics-{fchk_id}.txt"
            txt_file = os.path.join(txt_output_directory, txt_filename)
            output_txt = os.path.join(multiwfn_output_directory, f"nics-{fchk_id}-out.txt")
            
            if not os.path.exists(txt_file):
                print(f"Warning: No TXT file found for {fchk_file}.")
                continue
            
            # 使用subprocess运行Multiwfn
            fchk_path = os.path.join(fchk_directory, fchk_file)
            with open(txt_file, 'r') as input_file:
                with open(output_txt, 'w') as output_file:
                    subprocess.run(['multiwfn', fchk_path], stdin=input_file, stdout=output_file)
            print(f"Processed {fchk_file}")

# 从文档1复用的函数
def extract_nics_zz(txt_file):
    values = []
    with open(txt_file, 'r') as f:
        for line in f:
            match = re.search(r'The NICS_ZZ value is thus\s+([\d\.-]+)', line)
            if match:
                values.append(float(match.group(1)))
    return values

def parse_new_id(filename):
    match = re.search(r'nics-([a-zA-Z]\d+)-out.txt', filename)
    return match.group(1) if match else None

def merge_nics_data(df, nics_values, new_id, output_file, mode='w'):
    for col in ['Ring_NICS_ZZ_1', 'Ring_NICS_ZZ_2']:
        if col not in df.columns:
            df[col] = np.nan
    
    ring_id = 1
    while nics_values and len(nics_values) >= 2:
        mask = (df['New_ID'] == new_id) & (df['Ring_ID'] == ring_id)
        
        if not df[mask].empty:
            df.loc[mask, 'Ring_NICS_ZZ_1'] = nics_values[0]
            df.loc[mask, 'Ring_NICS_ZZ_2'] = nics_values[1]
        else:
            new_row = {
                'New_ID': new_id,
                'Ring_ID': ring_id,
                'Ring_NICS_ZZ_1': nics_values[0],
                'Ring_NICS_ZZ_2': nics_values[1]
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        nics_values = nics_values[2:]
        ring_id += 1
    
    df.to_csv(output_file, index=False, mode=mode)
    return df

def process_nics_output():
    df = pd.read_csv(csv_path)
    for filename in os.listdir(multiwfn_output_directory):
        if not filename.endswith('.txt'):
            continue
            
        new_id = parse_new_id(filename)
        if not new_id:
            print(f"警告: 无法解析文件名 {filename}，跳过")
            continue
            
        txt_file = os.path.join(multiwfn_output_directory, filename)
        nics_values = extract_nics_zz(txt_file)
        
        if not nics_values:
            print(f"警告: {filename} 中未找到 NICS_ZZ 值")
            continue
            
        df = merge_nics_data(df, nics_values, new_id, final_output_csv, mode='a')
    
    df.drop_duplicates(subset=['New_ID', 'Ring_ID'], inplace=True)
    df.to_csv(final_output_csv, index=False)

# 主流程
if __name__ == "__main__":
    generate_txt_files()  # 步骤1: 生成输入TXT文件
    run_multiwfn()        # 步骤2: 运行Multiwfn
    process_nics_output() # 步骤3: 处理输出并更新CSV