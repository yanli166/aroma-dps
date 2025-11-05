import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

def get_ring_info(mol):
    if mol is None:
        return None
    # 计算2D坐标
    AllChem.Compute2DCoords(mol)
    
    # 获取分子中的环信息
    ring_info = mol.GetRingInfo()
    
    # 初始化环信息列表
    ring_data = []
    
    # 遍历所有环
    for ring in ring_info.AtomRings():
        ring_size = len(ring)
        ring_atoms = [atom_idx for atom_idx in ring]
        ring_data.append({
            'Ring_ID': len(ring_data) + 1,
            'Ring_Size': ring_size,
            'Ring_Atoms': ring_atoms
        })
    
    return ring_data

def process_smiles_file(input_file, output_file):
    # 读取CSV文件
    df = pd.read_csv(input_file)
    
    # 确保SMILES列存在
    if 'SMILES' not in df.columns:
        raise ValueError("CSV文件中没有'SMILES'列")
    
    # 初始化结果列表
    results = []

    # 初始化错误记录列表
    error_records = []

    # 遍历每一行，获取环信息和环上原子编号
    for index, row in df.iterrows():
        smiles = row['SMILES']
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            error_records.append({
                'Original_ID': row['no'],
                'SMILES': smiles,
                'Error': 'Invalid SMILES string or molecule could not be generated.'
            })
            continue
        ring_data = get_ring_info(mol)
        if ring_data is None:
            error_records.append({
                'Original_ID': row['no'],
                'SMILES': smiles,
                'Error': 'No valid molecule generated.'
            })
            continue
        
        # 为每个环创建一个单独的行
        for ring in ring_data:
            new_id = row['no']
            results.append([
                new_id,  # New_ID
                smiles,  # SMILES
                ring['Ring_ID'],  # Ring_ID
                ring['Ring_Size'],  # Ring_Size
                ring['Ring_Atoms']  # Ring_Atoms
            ])

    # 构建列名
    column_names = ['New_ID', 'SMILES', 'Ring_ID', 'Ring_Size', 'Ring_Atoms']
    
    # 创建DataFrame
    result_df = pd.DataFrame(results, columns=column_names)
    
    # 将结果写入新的CSV文件
    result_df.to_csv(output_file, index=False)

    # 如果有错误记录，保存到错误日志文件
    if error_records:
        error_df = pd.DataFrame(error_records)
        error_df.to_csv(output_file.replace('.csv', '_errors.csv'), index=False)

# 使用示例
input_file = r"/home/ubuntu/cal/DPSCAL/lunci4/lunci4-1.csv"  # 输入CSV文件路径
output_file = r"/home/ubuntu/cal/DPSCAL/lunci4/merged-lunci4-out.csv"  # 输出CSV文件路径
process_smiles_file(input_file, output_file)