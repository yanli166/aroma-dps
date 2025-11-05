import os
import re
import pandas as pd

def extract_mbc_values(txt_file):
    mbc_values = []
    with open(txt_file, 'r') as file:
        for line in file:
            # 匹配归一化的多中心键级
            match = re.search(r'The normalized multicenter bond order:\s+([-\d\.]+)', line)
            if match:
                mbc_values.append(float(match.group(1)))
    return mbc_values

def parse_new_id(file_name):
    # 使用正则表达式从文件名中提取New_ID和Ring_ID
    match = re.search(r'mbco-(\w+)-ring(\d+).txt-out.txt', file_name)
    if match:
        new_id = match.group(1)
        ring_id = int(match.group(2))
        return new_id, ring_id
    else:
        raise ValueError(f"文件名格式不正确: {file_name}")

def add_mbc_to_csv(df, mbc_values, new_id, ring_id, output_file, mode='a'):
    # 只保留New_ID和Ring_ID匹配的行
    filtered_df = df[(df['New_ID'] == new_id) & (df['Ring_ID'] == ring_id)].copy()  # 使用.copy()确保是一个副本
    
    # 检查是否有足够的列来存储MBC值
    max_rings = len(mbc_values)
    for i in range(1, max_rings + 1):
        if f'Ring_{i}_MBCO' not in filtered_df.columns:
            filtered_df[f'Ring_{i}_MBCO'] = None
    
    # 添加MBC值
    for i, mbc in enumerate(mbc_values):
        filtered_df.loc[filtered_df.index[0], f'Ring_{i+1}_MBCO'] = mbc  # 只更新匹配的行
    
    # 将结果追加到现有的CSV文件
    filtered_df.to_csv(output_file, mode=mode, header=mode=='w', index=False)

def process_folder(folder_path, df, output_file):
    for filename in os.listdir(folder_path):
        if filename.startswith('mbco-') and filename.endswith('out.txt'):
            txt_file = os.path.join(folder_path, filename)
            new_id, ring_id = parse_new_id(filename)
            mbc_values = extract_mbc_values(txt_file)
            if (new_id in df['New_ID'].values) and (ring_id in df['Ring_ID'].values):
                add_mbc_to_csv(df, mbc_values, new_id, ring_id, output_file)

def main(folder_path, csv_file_path, output_file_path):
    csv_file = pd.read_csv(csv_file_path)
    
    # 确保输出文件不存在或者在开始时清空
    if os.path.exists(output_file_path):
        os.remove(output_file_path)
    
    process_folder(folder_path, csv_file, output_file_path)

if __name__ == "__main__":
    # 设置文件夹路径和文件路径
    folder_path = r"/home/ubuntu/cal/DPSCAL/lunci3/1-OUTMBCO"  # 文件夹路径
    csv_file_path = r"/home/ubuntu/cal/DPSCAL/lunci3/merged-lunci3-out.csv"  # 输入CSV文件路径
    output_file_path = r"/home/ubuntu/cal/DPSCAL/lunci3/lunci3-mbcout.csv"  # 输出CSV文件路径

    main(folder_path, csv_file_path, output_file_path)