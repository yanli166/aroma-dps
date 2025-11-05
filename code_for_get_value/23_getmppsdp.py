import os
import re
import pandas as pd

def extract_mpp_sdp_values(txt_file):
    """从文件中提取MPP和SDP值"""
    mpp_value = None
    sdp_value = None
    
    with open(txt_file, 'r') as file:
        for line in file:
            # 匹配MPP值
            mpp_match = re.search(r'Molecular planarity parameter $MPP$ is\s+([\d\.]+)', line)
            if mpp_match:
                mpp_value = float(mpp_match.group(1))
                
            # 匹配SDP值
            sdp_match = re.search(r'Span of deviation from plane $SDP$ is\s+([\d\.]+)', line)
            if sdp_match:
                sdp_value = float(sdp_match.group(1))
                
            # 如果两个值都已找到，提前退出循环
            if mpp_value is not None and sdp_value is not None:
                break
                
    return mpp_value, sdp_value

def parse_file_name(file_name):
    """从文件名中提取分子编号"""
    match = re.search(r'mpp-c(\d+)\.txt-out\.txt', file_name)
    if match:
        return f"c{match.group(1)}"
    else:
        raise ValueError(f"文件名格式不正确: {file_name}")

def process_folder(folder_path, output_file):
    """处理文件夹中的所有文件，并将结果保存到CSV"""
    results = []
    
    for filename in os.listdir(folder_path):
        if filename.startswith('mpp-c') and filename.endswith('.txt-out.txt'):
            try:
                txt_file = os.path.join(folder_path, filename)
                molecule_id = parse_file_name(filename)
                mpp, sdp = extract_mpp_sdp_values(txt_file)
                
                if mpp is not None and sdp is not None:
                    results.append({
                        'New_ID': molecule_id,
                        'MPP': mpp,
                        'SDP': sdp
                    })
                else:
                    print(f"警告: 未找到MPP或SDP值: {filename}")
                    
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")
    
    # 保存结果到CSV
    if results:
        result_df = pd.DataFrame(results)
        result_df.to_csv(output_file, index=False)
        print(f"成功处理 {len(results)} 个文件，结果已保存至: {os.path.abspath(output_file)}")
        return result_df
    else:
        print("未找到有效文件")
        return None

def main(folder_path, output_file_path):
    """主函数"""
    return process_folder(folder_path, output_file_path)

if __name__ == "__main__":
    # 配置路径
    folder_path = r"/home/ubuntu/cal/DPSCAL/lunci3/OUT_mppsdp"  # 包含mpp-*.txt-out.txt的文件夹
    output_file_path = r"/home/ubuntu/cal/DPSCAL/lunci3/mpp_sdp_results.csv"  # 输出CSV路径
    
    # 执行处理
    result_df = main(folder_path, output_file_path)
    
    # 结果预览
    if result_df is not None:
        print("\n结果预览:")
        print(result_df.head())