import csv
import os
import ast
import re
import pandas as pd
import subprocess
from pathlib import Path

class MultiwfnProcessor:
    """统一的Multiwfn处理类，支持HOMA和MBCO两种计算类型"""
    
    def __init__(self, base_path, calculation_type="homa"):
        """
        初始化处理器
        
        Args:
            base_path: 基础路径
            calculation_type: 计算类型，'homa' 或 'mbco'
        """
        self.calculation_type = calculation_type
        self.base_path = Path(base_path)
        
        # 根据计算类型设置参数
        if calculation_type == "homa":
            self.txt_prefix = "homa-"
            self.multiwfn_commands = "25\n6\n0\n"
            self.output_folder_name = "OUTHOMA"
            self.result_column_prefix = "Ring_{}_HOMA"
            self.extract_pattern = r'HOMA value is\s+([-\d\.]+)'
        else:  # mbco
            self.txt_prefix = "mbco-"
            self.multiwfn_commands = "9\n2\n"
            self.output_folder_name = "OUTMBCO"
            self.result_column_prefix = "Ring_{}_MBCO"
            self.extract_pattern = r'The normalized multicenter bond order:\s+([-\d\.]+)'
    
    def generate_input_files(self, csv_filename, fchk_directory, output_directory):
        """生成Multiwfn输入文件"""
        print(f"开始生成{self.calculation_type.upper()}输入文件...")
        
        # 确保输出目录存在
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        # 读取CSV文件并创建映射表
        fchk_id_to_ring_atoms = {}
        with open(csv_filename, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)
            
            # 找到所有Ring_Atoms列的索引
            ring_atoms_indices = [index for index, header in enumerate(headers) 
                                if header == 'Ring_Atoms']
            
            for row in reader:
                fchk_id = row[0]  # New_ID
                ring_id = row[2]  # Ring_ID
                ring_atoms_list = []
                
                for index in ring_atoms_indices:
                    ring_atoms_str = row[index].strip()
                    if ring_atoms_str:
                        try:
                            atoms = ast.literal_eval(ring_atoms_str)
                            if isinstance(atoms, list):
                                # 原子序号加1
                                ring_atoms_list.append(','.join(str(atom + 1) for atom in atoms))
                        except (ValueError, SyntaxError, TypeError):
                            print(f"解析Ring_Atoms错误 fchk_id {fchk_id}: {ring_atoms_str}")
                            continue
                
                if fchk_id not in fchk_id_to_ring_atoms:
                    fchk_id_to_ring_atoms[fchk_id] = {}
                fchk_id_to_ring_atoms[fchk_id][ring_id] = ring_atoms_list
        
        # 生成TXT文件
        files_generated = 0
        for fchk_file in os.listdir(fchk_directory):
            if fchk_file.endswith('.fchk'):
                fchk_id = fchk_file.split('.')[0]
                if fchk_id in fchk_id_to_ring_atoms:
                    for ring_id, ring_atoms_lines in fchk_id_to_ring_atoms[fchk_id].items():
                        # 生成文件内容
                        txt_content = self.multiwfn_commands + '\n'.join(ring_atoms_lines) + '\nq\n'
                        
                        # 保存文件
                        txt_filename = f'{self.txt_prefix}{fchk_id}-ring{ring_id}.txt'
                        with open(os.path.join(output_directory, txt_filename), 
                                mode='w', encoding='utf-8') as txtfile:
                            txtfile.write(txt_content)
                        files_generated += 1
                else:
                    print(f'警告: CSV文件中未找到 {fchk_file} 的Ring_Atoms数据')
        
        print(f"{self.calculation_type.upper()}输入文件生成完成，共生成 {files_generated} 个文件")
        return files_generated
    
    def run_multiwfn_calculations(self, input_folder, fchk_folder, output_folder):
        """运行Multiwfn计算"""
        print(f"开始运行{self.calculation_type.upper()}计算...")
        
        # 确保输出目录存在
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        processed_files = 0
        for fchk_file in os.listdir(fchk_folder):
            if fchk_file.endswith('.fchk'):
                filename = fchk_file.split('.')[0]
                fchk_path = os.path.join(fchk_folder, fchk_file)
                
                # 检查环ID范围（1-10）
                for ring_id in range(1, 11):
                    input_filename = f"{self.txt_prefix}{filename}-ring{ring_id}.txt"
                    input_path = os.path.join(input_folder, input_filename)
                    output_path = os.path.join(output_folder, f"{input_filename}-out.txt")
                    
                    # 检查输入文件是否存在
                    if not os.path.exists(input_path):
                        continue
                    
                    try:
                        # 使用subprocess运行Multiwfn
                        with open(input_path, 'r') as input_file:
                            result = subprocess.run(
                                ['multiwfn', fchk_path],
                                stdin=input_file,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True
                            )
                        
                        # 保存输出
                        with open(output_path, 'w') as output_file:
                            output_file.write(result.stdout)
                        
                        processed_files += 1
                        print(f"处理完成: {fchk_file} ring_id {ring_id}")
                        
                    except Exception as e:
                        print(f"处理错误: {fchk_file} ring_id {ring_id}: {e}")
        
        print(f"{self.calculation_type.upper()}计算完成，共处理 {processed_files} 个文件")
        return processed_files
    
    def extract_results(self, output_folder, csv_file_path, result_csv_path):
        """提取计算结果并更新CSV文件"""
        print(f"开始提取{self.calculation_type.upper()}结果...")
        
        def extract_values(txt_file):
            """提取特定模式的值"""
            values = []
            with open(txt_file, 'r') as file:
                for line in file:
                    match = re.search(self.extract_pattern, line)
                    if match:
                        values.append(float(match.group(1)))
            return values
        
        def parse_filename(file_name):
            """解析文件名提取New_ID和Ring_ID"""
            pattern = rf'{self.txt_prefix}(\w+)-ring(\d+).txt-out.txt'
            match = re.search(pattern, file_name)
            if match:
                return match.group(1), int(match.group(2))
            else:
                raise ValueError(f"文件名格式不正确: {file_name}")
        
        # 读取原始CSV文件，确保数据类型一致
        # 指定New_ID为字符串类型，Ring_ID为整数类型
        df = pd.read_csv(csv_file_path, dtype={'New_ID': str, 'Ring_ID': int})
        
        # 确保输出文件不存在
        if os.path.exists(result_csv_path):
            os.remove(result_csv_path)
        
        processed_count = 0
        for filename in os.listdir(output_folder):
            if filename.startswith(self.txt_prefix) and filename.endswith('out.txt'):
                txt_file = os.path.join(output_folder, filename)
                
                try:
                    new_id, ring_id = parse_filename(filename)
                    values = extract_values(txt_file)
                    
                    # 检查数据是否存在 - 确保类型一致
                    new_id_str = str(new_id)  # 确保是字符串
                    ring_id_int = int(ring_id)  # 确保是整数
                    
                    # 使用.loc访问避免FutureWarning
                    new_id_exists = (df['New_ID'] == new_id_str).any()
                    ring_id_exists = (df['Ring_ID'] == ring_id_int).any()
                    
                    if new_id_exists and ring_id_exists:
                        # 过滤匹配的行
                        filtered_df = df[(df['New_ID'] == new_id_str) & 
                                       (df['Ring_ID'] == ring_id_int)].copy()
                        
                        # 添加结果列
                        max_rings = len(values)
                        for i in range(1, max_rings + 1):
                            col_name = self.result_column_prefix.format(i)
                            if col_name not in filtered_df.columns:
                                filtered_df[col_name] = None
                        
                        # 填充结果值
                        for i, value in enumerate(values):
                            filtered_df.loc[filtered_df.index[0], 
                                         self.result_column_prefix.format(i+1)] = value
                        
                        # 保存结果（追加模式）
                        mode = 'a' if os.path.exists(result_csv_path) else 'w'
                        header = mode == 'w'
                        filtered_df.to_csv(result_csv_path, mode=mode, 
                                         header=header, index=False)
                        
                        processed_count += 1
                        
                except Exception as e:
                    print(f"处理文件 {filename} 时出错: {e}")
        
        print(f"{self.calculation_type.upper()}结果提取完成，共处理 {processed_count} 个文件")
        return processed_count
    
    def run_full_pipeline(self, csv_filename, fchk_directory, input_txt_dir, 
                         output_dir, result_csv_path):
        """运行完整的处理流程"""
        print(f"=== 开始 {self.calculation_type.upper()} 完整处理流程 ===")
        
        # 1. 生成输入文件
        self.generate_input_files(csv_filename, fchk_directory, input_txt_dir)
        
        # 2. 运行计算
        self.run_multiwfn_calculations(input_txt_dir, fchk_directory, output_dir)
        
        # 3. 提取结果
        self.extract_results(output_dir, csv_filename, result_csv_path)
        
        print(f"=== {self.calculation_type.upper()} 处理流程完成 ===\n")



def main():
    """主函数 - 运行HOMA和MBCO两种计算"""
    base_path = "/home/ubuntu/cal/DPSCAL/lunci4"
    csv_filename = f"{base_path}/merged-lunci4-out.csv"
    fchk_directory = f"{base_path}/gjf"
    
    # HOMA处理流程
    homa_processor = MultiwfnProcessor(base_path, "homa")
    homa_processor.run_full_pipeline(
        csv_filename=csv_filename,
        fchk_directory=fchk_directory,
        input_txt_dir=f"{base_path}/homatxt",
        output_dir=f"{base_path}/OUTHOMA",
        result_csv_path=f"{base_path}/lunci4-homaout.csv"
    )
    
    # MBCO处理流程
    mbco_processor = MultiwfnProcessor(base_path, "mbco")
    mbco_processor.run_full_pipeline(
        csv_filename=csv_filename,
        fchk_directory=fchk_directory,
        input_txt_dir=f"{base_path}/mbcotxt",
        output_dir=f"{base_path}/OUTMBCO",
        result_csv_path=f"{base_path}/lunci4-mbcout.csv"
    )


if __name__ == "__main__":
    main()