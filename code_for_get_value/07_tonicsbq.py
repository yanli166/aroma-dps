import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from numpy.linalg import svd, norm
from itertools import combinations

# 第一部分：从.log文件中提取坐标并生成分子对象（保持不变）
def extract_coordinates_from_log(log_file_path):
    coordinates = []
    with open(log_file_path, 'r') as file:
        lines = file.readlines()

    standard_orientation_indices = []
    for i, line in enumerate(lines):
        if 'Standard orientation:' in line:
            standard_orientation_indices.append(i + 5)

    if not standard_orientation_indices:
        print("No 'Standard orientation' section found.")
        return coordinates

    last_coord_start_index = standard_orientation_indices[-1]

    for i in range(last_coord_start_index, len(lines)):
        line = lines[i].strip()
        if not line or '----' in line:
            break
        parts = line.split()
        if len(parts) == 6 and parts[0].isdigit() and parts[1].isdigit():
            try:
                atom_index = int(parts[0])
                atom_number = int(parts[1])
                x = float(parts[3])
                y = float(parts[4])
                z = float(parts[5])
                element_symbol = Chem.GetPeriodicTable().GetElementSymbol(atom_number)
                coordinates.append((element_symbol, x, y, z))
            except ValueError as e:
                print(f"Error parsing line {i+1}: {e}")
    return coordinates

def create_molecule_from_coordinates(coordinates):
    mol = Chem.RWMol()
    conf = Chem.Conformer()
    for i, (symbol, x, y, z) in enumerate(coordinates):
        atom = Chem.Atom(symbol)
        mol.AddAtom(atom)
        conf.SetAtomPosition(i, (x, y, z))
    mol.AddConformer(conf)
    return mol

# 第二部分：从SMILES字符串中提取环信息（保持不变）
def get_smiles_ring_info(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return []
    ring_info = mol.GetRingInfo()
    return ring_info.AtomRings()

# 第三部分：改进的Bq点计算（使用SVD平面拟合）
def calculate_average_plane(mol, ring_atom_indices, height=1.0):
    """
    使用SVD方法计算环平面和Bq点坐标
    :param mol: 带3D坐标的RDKit分子对象
    :param ring_atom_indices: 环原子索引列表
    :param height: Bq点距平面的高度(Å)
    :return: 两个Bq点坐标 [(x1,y1,z1), (x2,y2,z2)]
    """
    try:
        conf = mol.GetConformer()
        # 提取环原子坐标
        coords = np.array([list(conf.GetAtomPosition(i)) for i in ring_atom_indices])
        
        if len(coords) < 3:
            print(f"环原子数不足({len(coords)})，至少需要3个原子定义平面")
            return []
        
        # 1. 计算环几何中心
        center = np.mean(coords, axis=0)
        
        # 2. 中心化坐标并计算SVD
        centered = coords - center
        U, S, Vt = svd(centered)
        
        # 最小奇异值对应的向量即为法向量
        normal = Vt[2]
        
        # 3. 归一化法向量
        normal /= norm(normal)
        
        # 4. 计算Bq点坐标（中心±法向量×高度）
        bq1 = center + height * normal
        bq2 = center - height * normal
        
        return [tuple(bq1), tuple(bq2)]
    
    except Exception as e:
        print(f"平面计算错误: {e}")
        return []

# 第四部分：生成Gaussian输入文件（保持不变）
def create_gaussian_input_file(mol, bq_points, output_file_path, chk_file_name=None):
    with open(output_file_path, 'w') as gaussian_file:
        gaussian_file.write("%mem=200GB\n")
        gaussian_file.write("%nprocshared=32\n")
        gaussian_file.write("%rwf=\\temp\g16s\n")
        if chk_file_name:
            gaussian_file.write(f"%chk=B{chk_file_name}\n")
        gaussian_file.write("#P B3LYP/def2svp NMR nosymm\n\n")
        gaussian_file.write("Generated from log file coordinates\n\n")
        gaussian_file.write("0 1\n")

        conf = mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            gaussian_file.write(f"{mol.GetAtomWithIdx(i).GetSymbol()}    {pos.x:.6f}    {pos.y:.6f}    {pos.z:.6f}\n")

        for bq_point in bq_points:
            gaussian_file.write(f"Bq    {bq_point[0]:.6f}    {bq_point[1]:.6f}    {bq_point[2]:.6f}\n")
        gaussian_file.write("\n\n")

# 主程序流程
if __name__ == "__main__":
    log_directory = r"/home/ubuntu/cal/DPSCAL/lunci3/gjf-1/"
    csv_file_path = r"/home/ubuntu/cal/DPSCAL/lunci3/merged-lunci3.csv"
    output_directory = r"/home/ubuntu/cal/DPSCAL/lunci3/NICS1"

    os.makedirs(output_directory, exist_ok=True)
    df = pd.read_csv(csv_file_path)

    for log_file_name in os.listdir(log_directory):
        if not log_file_name.endswith('.log'):
            continue
            
        log_file_path = os.path.join(log_directory, log_file_name)
        prefix = os.path.splitext(log_file_name)[0]
        
        # 匹配CSV中的SMILES
        smiles_row = df[df['no'] == prefix]
        if smiles_row.empty:
            print(f"未找到{prefix}对应的SMILES")
            continue
            
        smiles = smiles_row.iloc[0]['SMILES']
        print(f"处理: {log_file_name} | SMILES: {smiles}")

        # 从log提取坐标
        coordinates = extract_coordinates_from_log(log_file_path)
        if not coordinates:
            print(f"坐标提取失败: {log_file_name}")
            continue
            
        # 创建分子对象
        mol = create_molecule_from_coordinates(coordinates)
        
        # 从SMILES获取环信息
        rings = get_smiles_ring_info(smiles)
        print(f"发现{len(rings)}个环结构")
        
        # 计算所有环的Bq点
        all_bq_points = []
        for i, ring in enumerate(rings):
            bq_points = calculate_average_plane(mol, ring, height=1.0)
            if bq_points:
                print(f"环{i+1} ({len(ring)}原子): 生成{len(bq_points)}个Bq点")
                all_bq_points.extend(bq_points)
            else:
                print(f"环{i+1} Bq点生成失败")
        
        # 生成Gaussian输入
        if all_bq_points:
            output_path = os.path.join(output_directory, f"B{prefix}.gjf")
            create_gaussian_input_file(mol, all_bq_points, output_path, f"{prefix}.chk")
            print(f"已生成: {output_path}\n")
        else:
            print(f"无有效Bq点，跳过生成\n")