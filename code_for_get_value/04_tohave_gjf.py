import os

def create_gaussian_input_file(xyz_file_path, output_file_path, title):
    """为分子创建Gaussian输入文件"""
    with open(xyz_file_path, 'r') as f:
        lines = f.readlines()
    
    # 获取原子数
    num_atoms = int(lines[0].strip())
    
    # 创建Gaussian输入文件
    with open(output_file_path, 'w') as gaussian_file:
        # 使用标题（即文件名）作为chk文件的名称
        gaussian_file.write("%%mem=200GB\n%%nprocshared=32\n%%rwf=\\temp\g16s\n%%chk=%s.chk\n#p B3LYP/def2svp Opt \n\n" % title)
        gaussian_file.write("%s\n\n" % title)
        gaussian_file.write("0 1\n")
        
        # 跳过前两行，从第三行开始读取原子信息
        for i in range(2, 2 + num_atoms):
            atom_info = lines[i].split()
            element = atom_info[0]
            x, y, z = map(float, atom_info[1:4])
            gaussian_file.write("%s    %.5f    %.5f    %.5f\n" % (element, x, y, z))
        gaussian_file.write("\n")
        gaussian_file.write("\n")
        gaussian_file.write("\n")

def process_xyz_files(input_dir, output_dir):
    """处理XYZ文件，为每个文件生成Gaussian输入文件"""
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 遍历目录中的所有子目录
    for subdir in os.listdir(input_dir):
        subdir_path = os.path.join(input_dir, subdir)
        if os.path.isdir(subdir_path):
            # 读取子目录中的xtbopt.xyz文件
            xyz_file_name = "xtbopt.xyz"
            xyz_file_path = os.path.join(subdir_path, xyz_file_name)
            
            if os.path.exists(xyz_file_path):
                # 以子目录的名称为GJF文件的标题
                title = subdir  # 直接使用子目录名称作为标题
                output_file_path = os.path.join(output_dir, f"{title}.gjf")
                
                # 创建Gaussian输入文件
                create_gaussian_input_file(xyz_file_path, output_file_path, title)
                print(f"已生成: {output_file_path}")

# 设置XYZ文件目录和输出目录
input_directory = r'/home/ubuntu/cal/DPSCAL/lunci3/xtb-out'  # 替换为XYZ文件目录路径
output_directory = r'/home/ubuntu/cal/DPSCAL/lunci3/gjf-1'  # 替换为输出目录路径

# 处理XYZ文件
process_xyz_files(input_directory, output_directory)
