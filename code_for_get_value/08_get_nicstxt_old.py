import csv
import os
import ast
import re


csv_filename = r"/home/ubuntu/cal/DPSCAL/lunci3/merged-lunci3-out.csv"
fchk_directory = r'/home/ubuntu/cal/DPSCAL/lunci3/NICS1/'  # fchk文件所在的目录
log_directory = r'/home/ubuntu/cal/DPSCAL/lunci3/NICS1/'  # log文件所在的目录
output_directory = r'/home/ubuntu/cal/DPSCAL/lunci3/NICSTXT'  # 指定输出目录
txt_prefix = 'nics-'  # TXT文件名前缀

# 确保输出目录存在
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 读取CSV文件并创建一个映射表，将fchk文件的编号映射到对应的Ring_Atoms序号
fchk_id_to_ring_atoms = {}
with open(csv_filename, mode='r', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    headers = next(reader)  # 读取标题行
    # 找到所有Ring_Atoms列的索引
    ring_atoms_indices = [index for index, header in enumerate(headers) if header == 'Ring_Atoms']
    for row in reader:
        fchk_id = row[0]  # 假设第一列是fchk文件的编号，即New_ID
        ring_id = row[2]  # Ring_ID
        ring_atoms_list = []  # 存储每个环的原子序号列表
        for index in ring_atoms_indices:
            # 清理Ring_Atoms列的字符串并尝试转换为整数列表
            ring_atoms_str = row[index].strip()
            if ring_atoms_str:  # 检查字符串是否不为空
                try:
                    # 尝试将清理后的字符串转换为整数列表
                    atoms = ast.literal_eval(ring_atoms_str)
                    if isinstance(atoms, list):
                        # 将当前环的原子序号加1后加入列表
                        ring_atoms_list.append(','.join(str(atom + 1) for atom in atoms))
                except (ValueError, SyntaxError, TypeError):
                    print(f"Error parsing Ring_Atoms for fchk_id {fchk_id}: {ring_atoms_str}")
                    continue  # 如果解析失败，跳过当前条目
        if fchk_id not in fchk_id_to_ring_atoms:
            fchk_id_to_ring_atoms[fchk_id] = {}
        fchk_id_to_ring_atoms[fchk_id][ring_id] = ring_atoms_list  # 存储映射关系

# 遍历fchk文件并生成TXT文件
for fchk_file in os.listdir(fchk_directory):
    if fchk_file.endswith('.fchk'):
        # 提取fchk_id，去除B的结果
        fchk_id = fchk_file.split('.')[0].lstrip('B')  # 去除开头的B
        if fchk_id in fchk_id_to_ring_atoms:
            # 对应的log文件名
            log_file = os.path.join(log_directory, f'B{fchk_id}.log')
            if os.path.exists(log_file):
                # 提取log文件中的Bq Isotropic数据
                with open(log_file, 'r', encoding='utf-8') as logfile:
                    log_content = logfile.read()
                # 查找所有Bq Isotropic的段落
                bq_pattern = re.compile(r'Bq\s+Isotropic\s*=\s*\d+\.\d+\s+Anisotropy\s*=\s*\d+\.\d+(.*?)Eigenvalues', re.DOTALL)
                bq_matches = bq_pattern.findall(log_content)
                # 提取每个Bq段落中的XX, YX, ZX, XY, YY, ZY, XZ, YZ, ZZ值
                data_pattern = re.compile(r'XX\s*=\s*(\S+)\s+YX\s*=\s*(\S+)\s+ZX\s*=\s*(\S+)\s+XY\s*=\s*(\S+)\s+YY\s*=\s*(\S+)\s+ZY\s*=\s*(\S+)\s+XZ\s*=\s*(\S+)\s+YZ\s*=\s*(\S+)\s+ZZ\s*=\s*(\S+)')
                bq_data = []
                for match in bq_matches:
                    data_match = data_pattern.search(match)
                    if data_match:
                        bq_data.append(data_match.group(0).strip())
                # 合并所有ring_id的TXT内容
                merged_txt_content = 'y\n25\n'
                ring_ids = list(fchk_id_to_ring_atoms[fchk_id].keys())
                # 每个环对应两个Bq数据
                for i in range(len(bq_data) // 2):
                    ring_id = ring_ids[i]
                    ring_atoms_lines = fchk_id_to_ring_atoms[fchk_id][ring_id]
                    # 写入第一个Bq数据
                    merged_txt_content += f'4\n{ring_atoms_lines[0]}\n{ring_atoms_lines[0]}\n {bq_data[2*i]}\n'
                    # 写入第二个Bq数据
                    merged_txt_content += f'4\n{ring_atoms_lines[0]}\n{ring_atoms_lines[0]}\n {bq_data[2*i+1]}\n'
                # 保存合并后的TXT文件
                merged_txt_filename = f'{txt_prefix}{fchk_id}.txt'
                with open(os.path.join(output_directory, merged_txt_filename), mode='w', encoding='utf-8') as merged_txtfile:
                    merged_txtfile.write(merged_txt_content)
            else:
                print(f'Warning: No log file found for {fchk_file}.')
        else:
            print(f'Warning: No Ring_Atoms data found for {fchk_file} in CSV file.')

print('TXT文件生成完成。')