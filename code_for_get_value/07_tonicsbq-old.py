import os
import pandas as pd
from rdkit import Chem
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from numpy.linalg import lstsq
from itertools import combinations

import re
from rdkit import Chem
import pandas as pd
import os
# 第一部分：从.log文件中提取坐标并生成分子对象
def extract_coordinates_from_log(log_file_path):
    coordinates = []
    with open(log_file_path, 'r') as file:
        lines = file.readlines()

    # 记录所有“Standard orientation”的起始索引
    standard_orientation_indices = []
    for i, line in enumerate(lines):
        if 'Standard orientation:' in line:
            standard_orientation_indices.append(i + 5)  # 坐标从 'Standard orientation:' 下五行开始

    if not standard_orientation_indices:
        print("No 'Standard orientation' section found in the log file.")
        return coordinates

    # 获取最后一组“Standard orientation”的索引
    last_coord_start_index = standard_orientation_indices[-1]

    # 提取最后一组坐标的范围
    for i in range(last_coord_start_index, len(lines)):
        line = lines[i].strip()
        if not line or '----' in line:  # 如果行为空或遇到分隔符，停止解析
            break
        parts = line.split()
        if len(parts) == 6 and parts[0].isdigit() and parts[1].isdigit():
            try:
                atom_index = int(parts[0])  # 原子索引
                atom_number = int(parts[1])  # 原子序号
                x = float(parts[3])  # x坐标
                y = float(parts[4])  # y坐标
                z = float(parts[5])  # z坐标
                element_symbol = Chem.GetPeriodicTable().GetElementSymbol(atom_number)
                coordinates.append((element_symbol, x, y, z))
            except ValueError as e:
                print(f"Error parsing line {i + 1} in {log_file_path}: {line}")
                print(f"Error: {e}")
    return coordinates

def create_molecule_from_coordinates(coordinates):
    mol = Chem.RWMol()
    conf = Chem.Conformer()
    for symbol, x, y, z in coordinates:
        atom = Chem.Atom(symbol)
        mol.AddAtom(atom)
        conf.SetAtomPosition(mol.GetNumAtoms() - 1, (x, y, z))
    mol.AddConformer(conf)
    return mol

# 第二部分：从SMILES字符串中提取环信息
def get_smiles_ring_info(smiles):
    mol1 = Chem.MolFromSmiles(smiles)
    ring_info = mol1.GetRingInfo()
    return ring_info.AtomRings()

# 第三部分：计算Bq点并生成Gaussian输入文件
def calculate_average_plane(mol, ring_atom_indices):
    try:
        conf = mol.GetConformer()
        ring_atom_coords = [conf.GetAtomPosition(idx) for idx in ring_atom_indices]
        if len(ring_atom_coords) < 3:
            print("Not enough atoms to form a plane.")
            return None, None

        ring_atom_coords = [list(coord) for coord in ring_atom_coords]
        atmcList = ring_atom_indices
        heigh = 1.0  # 高度值设置为1
        xyzCoors = {idx: coord for idx, coord in zip(atmcList, ring_atom_coords)}

        bq_points = calCoor(atmcList, heigh, xyzCoors)
        return bq_points
    except Exception as e:
        print(f"Error calculating average plane: {e}")
        return None, None

def projectOnPlane(a, b, c, d, x, y, z):
    # based on the Matlab function "projection" written by Neo Jing Ci, 11/7/18
    A = np.array([[1, 0, 0, -a], [0, 1, 0, -b], [0, 0, 1, -c], [a, b, c, 0]])
    B = np.array([[x], [y], [z], [d]])
    C = lstsq(A, B, rcond=None)[0]
    px = C[0][0]
    py = C[1][0]
    pz = C[2][0]
    return [px, py, pz]

# Calculate coordinates of ghost atoms
def calCoor(atmcList, heigh, xyzCoors):
    aveX = sum(xyzCoors[atm][0] for atm in atmcList) / len(atmcList)
    aveY = sum(xyzCoors[atm][1] for atm in atmcList) / len(atmcList)
    aveZ = sum(xyzCoors[atm][2] for atm in atmcList) / len(atmcList)
    userX = [xyzCoors[atm][0] for atm in atmcList]
    userY = [xyzCoors[atm][1] for atm in atmcList]
    userZ = [xyzCoors[atm][2] for atm in atmcList]

    a1 = np.column_stack([userX, userY, np.ones(len(atmcList))])
    a2 = np.column_stack([userZ])
    a3 = lstsq(a1, a2, rcond=None)[0]  # Finds the plane that best fits the selected atoms

    c1 = a3[0][0]
    c2 = a3[1][0]
    c3 = a3[2][0]

    for ii in range(len(userX)):
        projected_xyz = projectOnPlane(c1, c2, -1, -c3, userX[ii], userY[ii], userZ[ii])
        userX[ii] = projected_xyz[0]
        userY[ii] = projected_xyz[1]
        userZ[ii] = projected_xyz[2]

    para_a = (userY[1] - userY[0]) * (userZ[2] - userZ[0]) - (userY[2] - userY[0]) * (userZ[1] - userZ[0])
    para_b = (userZ[1] - userZ[0]) * (userX[2] - userX[0]) - (userZ[2] - userZ[0]) * (userX[1] - userX[0])
    para_c = (userX[1] - userX[0]) * (userY[2] - userY[0]) - (userX[2] - userX[0]) * (userY[1] - userY[0])

    if para_a != 0.0:
        para_A3 = 1 + para_b * para_b / para_a / para_a + para_c * para_c / para_a / para_a
        para_B3 = - 2 * aveX - 2 * para_b * para_b * aveX / para_a / para_a - 2 * para_c * para_c * aveX / para_a / para_a
        para_C3 = aveX * aveX + para_b * para_b * aveX * aveX / para_a / para_a + para_c * para_c * aveX * aveX / para_a / para_a - heigh * heigh
        deltaValue = para_B3 * para_B3 - 4 * para_A3 * para_C3
        if deltaValue != 0:
            bqN1X = (- para_B3 + np.sqrt(deltaValue)) / (2 * para_A3)
            bqN2X = (- para_B3 - np.sqrt(deltaValue)) / (2 * para_A3)
            bqN1Y = para_b / para_a * (bqN1X - aveX) + aveY
            bqN2Y = para_b / para_a * (bqN2X - aveX) + aveY
            bqN1Z = para_c / para_a * (bqN1X - aveX) + aveZ
            bqN2Z = para_c / para_a * (bqN2X - aveX) + aveZ
    else:
        # Handle the case where para_a == 0
        bqN1X, bqN1Y, bqN1Z, bqN2X, bqN2Y, bqN2Z = handle_para_a_zero_case(aveX, aveY, aveZ, userX, userY, userZ, heigh)

    return [(bqN1X, bqN1Y, bqN1Z), (bqN2X, bqN2Y, bqN2Z)]

def handle_para_a_zero_case(aveX, aveY, aveZ, userX, userY, userZ, heigh):
    # Handle the case where para_a == 0
    if userX[0] == userX[1] and userX[1] == userX[2]:
        bqN1X = float(heigh)
        bqN1Y = aveY
        bqN1Z = aveZ
        bqN2X = -float(heigh)
        bqN2Y = aveY
        bqN2Z = aveZ
    elif userY[0] == userY[1] and userY[1] == userY[2]:
        bqN1X = aveX
        bqN1Y = float(heigh)
        bqN1Z = aveZ
        bqN2X = aveX
        bqN2Y = -float(heigh)
        bqN2Z = aveZ
    elif userZ[0] == userZ[1] and userZ[1] == userZ[2]:
        bqN1X = aveX
        bqN1Y = aveY
        bqN1Z = float(heigh)
        bqN2X = aveX
        bqN2Y = aveY
        bqN2Z = -float(heigh)
    return bqN1X, bqN1Y, bqN1Z, bqN2X, bqN2Y, bqN2Z
    
def create_gaussian_input_file(mol, bq_points, output_file_path, chk_file_name=None):
    with open(output_file_path, 'w') as gaussian_file:
        # 写入内存、处理器数和检查点文件名
        gaussian_file.write("%mem=200GB\n")
        gaussian_file.write("%nprocshared=32\n")
        gaussian_file.write("%rwf=\\temp\g16s\n")
        if chk_file_name:
            gaussian_file.write(f"%chk=B{chk_file_name}\n")
        gaussian_file.write("#P B3LYP/def2svp NMR nosymm \n\n")
        gaussian_file.write("Generated from log file coordinates\n\n")
        gaussian_file.write("0 1\n")

        conf = mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(i)
            x, y, z = conf.GetAtomPosition(i)
            gaussian_file.write(f"{atom.GetSymbol()}    {x: .5f}    {y: .5f}    {z: .5f}\n")

        if bq_points:
            for bq_point in bq_points:
                bq_point_str = " ".join(f"{coord:.6f}" for coord in bq_point)
                gaussian_file.write(f"Bq {bq_point_str}\n")
        gaussian_file.write("\n\n")


# 主函数
if __name__ == "__main__":
    log_directory = r"/home/ubuntu/cal/DPSCAL/lunci3/gjf-1/"  # 文件夹1路径
    csv_file_path = r"/home/ubuntu/cal/DPSCAL/lunci3/merged-lunci3.csv"  # CSV文件路径
    output_directory = r"/home/ubuntu/cal/DPSCAL/lunci3/NICS1"  # 输出文件夹路径

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    df = pd.read_csv(csv_file_path)

    for log_file_name in os.listdir(log_directory):
        if log_file_name.endswith('.log'):
            log_file_path = os.path.join(log_directory, log_file_name)
            log_file_prefix = os.path.splitext(log_file_name)[0]  # 获取文件名前缀，如a1

            smiles_row = df[df['no'] == log_file_prefix]
            if not smiles_row.empty:
                smiles = smiles_row.iloc[0]['SMILES']
              # print(f"Processing {log_file_name} with SMILES: {smiles}")

                # 第一部分：提取坐标并生成分子对象
                coordinates = extract_coordinates_from_log(log_file_path)
                if coordinates:
                    log_mol = create_molecule_from_coordinates(coordinates)
                   #print(f"Molecule object created for {log_file_name}")

                    # 打印原始 coordinates 和 RDKit Conformer 中的坐标对比
                    conf = log_mol.GetConformer()
                   #print("----- Coordinate Comparison -----")
                    for i in range(len(coordinates)):
                        symbol, x, y, z = coordinates[i]
                        conf_x, conf_y, conf_z = conf.GetAtomPosition(i)
                      # print(f"Original (Atom {i}): {x:.5f} {y:.5f} {z:.5f}")
                      # print(f"Conformer (Atom {i}): {conf_x:.5f} {conf_y:.5f} {conf_z:.5f}")
                #   print("---------------------------------")
#
                else:
                    print(f"Failed to extract coordinates from {log_file_name}")
                    continue

                # 第二部分：从SMILES字符串中提取环信息
                smiles_rings = get_smiles_ring_info(smiles)
             #  print(f"SMILES rings: {smiles_rings}")

                # 第三部分：计算Bq点并生成Gaussian输入文件
                all_bq_points = []  # 用于存储所有环的Bq点
                for ring in smiles_rings:
                    bq_point1, bq_point2 = calculate_average_plane(log_mol, ring)
                    if bq_point1 and bq_point2:
                        all_bq_points.append(bq_point1)
                        all_bq_points.append(bq_point2)
                    else:
                        print(f"Failed to calculate Bq points for ring {ring} in {log_file_name}")

                # 如果有有效的Bq点，生成Gaussian输入文件
                if all_bq_points:
                    output_file_path = os.path.join(output_directory, f"B{log_file_prefix}.gjf")
                    chk_file_name = f"{log_file_prefix}.chk"
                    create_gaussian_input_file(log_mol, all_bq_points, output_file_path, chk_file_name)
                  #  print(f"Gaussian input file generated: {output_file_path}")
                else:
                    print(f"No valid Bq points calculated for {log_file_name}")
              
