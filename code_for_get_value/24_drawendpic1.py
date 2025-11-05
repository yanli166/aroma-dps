import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

def create_plots(df, x_cols, y_cols, hue_col1, hue_col2, output_dir):
    """
    创建多维分类散点图并保存
    :param df: 包含数据的数据框
    :param x_cols: 横坐标列名列表 (['MPP', 'SDP'])
    :param y_cols: 纵坐标列名列表 (['NICS','HOMA','MBCO','prediction'])
    :param hue_col1: 第一分类列 ('no1')
    :param hue_col2: 第二分类列 ('no3')
    :param output_dir: 输出目录路径
    """
    # 1. 数据预处理 - 确保数据完整性
    df = df.copy()  # 避免修改原始数据
    df[hue_col2] = df[hue_col2].fillna(0).astype(float)  # 填充NaN为0
    df[hue_col1] = df[hue_col1].astype(str)  # 转换为字符串类型
    
    # 2. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. 定义视觉编码系统 [3,6](@ref)
    # 3.1 no1 -> 颜色映射 (使用tab10色系)
    unique_no1 = sorted(df[hue_col1].unique())
    base_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_no1)))
    no1_color_map = {no1: base_colors[i] for i, no1 in enumerate(unique_no1)}
    
    # 3.2 no3 -> 形状+大小映射 (兼容0值)
    no3_markers = {
        1.0: ('o', 120),   # 圆形
        2.0: ('s', 100),   # 方形
        0.0: ('x', 80)     # 叉号 (用于填充的NaN值)
    }
    # 兜底配置 (处理未定义值)
    default_marker = ('D', 90)  # 菱形
    
    # 4. 创建组合图例
    def create_legend():
        legend_elements = []
        # 4.1 no1颜色图例
        for no1 in unique_no1:
            legend_elements.append(mpatches.Patch(
                color=no1_color_map[no1], 
                label=f'{hue_col1}={no1}'
            ))
        
        # 4.2 分隔线
        legend_elements.append(Line2D([0], [0], color='none'))
        
        # 4.3 no3形状图例
        for no3, (marker, size) in no3_markers.items():
            legend_elements.append(Line2D(
                [0], [0], 
                marker=marker, 
                color='w',
                markersize=size**0.5,  # 视觉大小调整
                markerfacecolor='gray',
                label=f'{hue_col2}={int(no3)}'
            ))
        return legend_elements

    # 5. 生成所有图表组合
    for x_col in x_cols:
        for y_col in y_cols:
            plt.figure(figsize=(10, 7))
            
            # 5.1 分层绘制数据点
            unique_no3_vals = sorted(df[hue_col2].unique())
            for no1_val in unique_no1:
                for no3_val in unique_no3_vals:
                    subset = df[(df[hue_col1] == no1_val) & (df[hue_col2] == no3_val)]
                    if subset.empty:
                        continue
                    
                    # 获取形状配置 (带错误保护)
                    marker_config = no3_markers.get(no3_val, default_marker)
                    
                    plt.scatter(
                        subset[x_col],
                        subset[y_col],
                        color=no1_color_map[no1_val],
                        marker=marker_config[0],
                        s=marker_config[1],
                        alpha=0.85,
                        edgecolor='w',
                        linewidth=1,
                        label=f"{hue_col1}={no1_val}, {hue_col2}={no3_val}"
                    )
            
            # 5.2 添加数据标签 [3](@ref)
            for _, row in df.iterrows():
                plt.annotate(
                    row['New_ID'], 
                    (row[x_col], row[y_col]),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha='center',
                    fontsize=9
                )
            
            # 5.3 设置图表属性
            plt.title(f"{y_col} vs {x_col}", fontweight='bold')
            plt.xlabel(f"{x_col} (Å)", fontweight='bold')
            plt.ylabel(y_col, fontweight='bold')
            plt.grid(True, linestyle=':', alpha=0.4)
            
            # 5.4 添加专业图例 [6](@ref)
            plt.legend(
                handles=create_legend(),
                loc='upper center',
                bbox_to_anchor=(0.5, -0.15),
                ncol=2,
                title="Classification Legend"
            )
            
            # 5.5 保存图表
            plt.tight_layout()
            output_path = os.path.join(output_dir, f"{x_col}_{y_col}_plot.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ 图表已保存: {output_path}")

# 使用示例
if __name__ == "__main__":
    # 配置参数
    csv_path = "/home/ubuntu/cal/DPSCAL/lunci3/lunci3-all-mark-1.csv"
    output_dir = "/home/ubuntu/cal/DPSCAL/lunci3/plots"
    
    # 读取数据 (添加错误处理)
    try:
        df = pd.read_csv(csv_path)
        print(f"数据读取成功: {len(df)} 行记录")
        
        # 验证必要列存在
        required_cols = ['New_ID', 'MPP', 'SDP', 'NICS', 'HOMA', 'MBCO', 'prediction', 'no1', 'no3']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺失必要列: {', '.join(missing_cols)}")
        
        # 执行绘图
        create_plots(
            df=df,
            x_cols=['MPP', 'SDP'],
            y_cols=['NICS', 'HOMA', 'MBCO', 'prediction'],
            hue_col1='no1',
            hue_col2='no3',
            output_dir=output_dir
        )
    except Exception as e:
        print(f"❌ 处理失败: {str(e)}")