import csv
import os
import ast
import re

csv_filename = r"/home/ubuntu/cal/DPSCAL/lunci3/merged-lunci3-out.csv"
fchk_directory = r'/home/ubuntu/cal/DPSCAL/lunci3/NICS1' 
output_directory = r'/home/ubuntu/cal/DPSCAL/lunci3/nicstxt2'
txt_prefix = 'nics-'
log_directory = r'/home/ubuntu/cal/DPSCAL/lunci3/NICS1'  # 新增log文件目录

# 修改1：处理fchk文件编号
def get_fchk_id(filename):
    base_name = filename.split('.')[0]
    return base_name.lstrip('B')  # 去除开头的B字符

# 新增函数：解析log文件
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
            elif len(current_block) == 3:  # 完整捕获3行数据
                data_blocks.append(current_block)
                capture = False
    return data_blocks[:2]  # 只取前两个符合条件的数据块

# 修改后的主逻辑
fchk_id_to_ring_data = {}
with open(csv_filename, mode='r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        fchk_id = get_fchk_id(row['New_ID'])
        ring_id = row['Ring_ID']
        
        # 处理Ring_Atoms
        atoms = ast.literal_eval(row['Ring_Atoms'])
        atoms_plus1 = [str(atom+1) for atom in atoms]
        atoms_str = ','.join(atoms_plus1)
        
        # 读取log文件数据
        log_file = f"B{fchk_id}.log"
        log_path = os.path.join(log_directory, log_file)
        tensor_data = parse_log_file(log_path) if os.path.exists(log_path) else []
        
        if fchk_id not in fchk_id_to_ring_data:
            fchk_id_to_ring_data[fchk_id] = {}
        fchk_id_to_ring_data[fchk_id][ring_id] = {
            'atoms': atoms_str,
            'tensors': tensor_data
        }

# 生成TXT文件
for fchk_file in os.listdir(fchk_directory):
    if fchk_file.endswith('.fchk'):
        fchk_id = get_fchk_id(fchk_file)
        if fchk_id in fchk_id_to_ring_data:
            merged_content = []
            for ring_id in sorted(fchk_id_to_ring_data[fchk_id].keys()):
                data = fchk_id_to_ring_data[fchk_id][ring_id]
                atoms = data['atoms']
                tensors = data['tensors']
                
                # 生成每个ring的内容
                block = [
                    "25",
                    "4",
                    atoms,
                    atoms
                ]
                for tensor in tensors:
                    block.extend(tensor)
                    block.extend(["4", atoms, atoms])
                merged_content.extend(block)
            
            # 写入合并后的文件
            txt_filename = f"{txt_prefix}{fchk_id}.txt"
            with open(os.path.join(output_directory, txt_filename), 'w') as f:
                f.write('\n'.join(merged_content))