import pandas as pd

# 读取第一个表格
sphericity_df = pd.read_csv("/home/ubuntu/cal/DPSCAL/lunci3/lunci3-sphericity_results.csv")

# 读取第二个表格
main_df = pd.read_csv("/home/ubuntu/cal/DPSCAL/lunci3/lunci3-alldata2.csv")

# 将第一个表格的 Sphericity 值根据 New_ID 合并到第二个表格中
merged_df = pd.merge(main_df, sphericity_df[['New_ID', 'Sphericity']], on='New_ID', how='left')

# 保存合并后的表格
output_file = "/home/ubuntu/cal/DPSCAL/lunci3/lunci3-alldata3.csv"  # 输出文件路径
merged_df.to_csv(output_file, index=False)

print(f"合并后的表格已保存到: {output_file}")