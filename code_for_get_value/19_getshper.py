import os
import re
import pandas as pd

def extract_sphericity_values(txt_file):
    """从文件中提取球形度值"""
    sphericity = None
    with open(txt_file, 'r') as file:
        for line in file:
            # 匹配球形度值
            match = re.search(r'Sphericity:\s+([\d\.]+)', line)
            if match:
                sphericity = float(match.group(1))
                break
    return sphericity

def parse_file_name(file_name):
    """从文件名中提取编号"""
    match = re.search(r'spher-c(\d+).txt-out.txt', file_name)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"文件名格式不正确: {file_name}")

def process_folder(folder_path, output_file):
    """处理文件夹中的所有文件，并将结果保存到输出文件"""
    results = []
    for filename in os.listdir(folder_path):
        if filename.startswith('spher-c') and filename.endswith('out.txt'):
            txt_file = os.path.join(folder_path, filename)
            file_id = parse_file_name(filename)
            sphericity = extract_sphericity_values(txt_file)
            if sphericity is not None:
                results.append({'ID': file_id, 'Sphericity': sphericity})
            else:
                print(f"未找到球形度值: {filename}")
    
    # 将结果保存到CSV文件
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False)
    print(f"结果已保存到: {output_file}")

def main(folder_path, output_file_path):
    """主函数"""
    process_folder(folder_path, output_file_path)

if __name__ == "__main__":
    # 设置文件夹路径和输出文件路径
    folder_path = r"/home/ubuntu/cal/DPSCAL/lunci3/1-OUTMBCO"  # 文件夹路径
    output_file_path = r"/home/ubuntu/cal/DPSCAL/lunci3/sphericity_results.csv"  # 输出CSV文件路径

    main(folder_path, output_file_path)



    