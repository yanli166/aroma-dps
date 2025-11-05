import csv
import os
import ast


csv_filename = r"/home/ubuntu/cal/DPSCAL/lunci3/merged-lunci3-out.csv"
fchk_directory = r'/home/ubuntu/cal/DPSCAL/lunci3/gjf-1'  # fchk文件所在的目录
output_directory = r'/home/ubuntu/cal/DPSCAL/lunci3/homatxt'  # 指定输出目录
txt_prefix = 'homa-'  # TXT文件名前缀

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
        fchk_id = fchk_file.split('.')[0]  # 假设fchk文件的名称就是其编号
        if fchk_id in fchk_id_to_ring_atoms:
            for ring_id, ring_atoms_lines in fchk_id_to_ring_atoms[fchk_id].items():
                # 生成TXT文件内容
                txt_content = '25\n6\n0\n' + '\n'.join(ring_atoms_lines) + '\nq\n'

                # 保存TXT文件
                txt_filename = f'{txt_prefix}{fchk_id}-ring{ring_id}.txt'  # txt文件名是homa-加上fchk文件的编号和ring ID
                with open(os.path.join(output_directory, txt_filename), mode='w', encoding='utf-8') as txtfile:
                    txtfile.write(txt_content)
        else:
            print(f'Warning: No Ring_Atoms data found for {fchk_file} in CSV file.')

print('TXT文件生成完成。')