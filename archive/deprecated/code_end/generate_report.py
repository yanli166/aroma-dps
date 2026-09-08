"""
汇总报告生成: 读取三层训练结果 + 四种泛化测试结果, 生成 md 报告
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_ROOT = os.path.join(PROJ_ROOT, 'code_end')


def load_layer_results(results_root, layer_name):
    """加载某一层的 per_seed_results.csv"""
    csv_path = os.path.join(results_root, layer_name, 'per_seed_results.csv')
    if not os.path.exists(csv_path):
        # 尝试从 summary.csv 恢复
        from multiseed.run_multiseed import recover_from_summaries
        sys.path.insert(0, CODE_ROOT)
        df = recover_from_summaries(os.path.join(results_root, layer_name), layer_name)
        if df is not None:
            return df
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def load_generalization_results(results_root):
    """加载泛化测试结果"""
    csv_path = os.path.join(results_root, 'generalization', 'generalization_summary.csv')
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def format_md_table(df, title, highlight_col=None):
    """将 DataFrame 转为 md 表格"""
    if df.empty:
        return f"### {title}\n\n无数据\n"

    lines = [f"### {title}\n"]
    # 表头
    cols = df.columns.tolist()
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")

    # 找最优行 (highlight_col 最大值)
    best_idx = None
    if highlight_col and highlight_col in df.columns:
        try:
            best_idx = df[highlight_col].astype(float).idxmax()
        except Exception:
            pass

    for i, (_, row) in enumerate(df.iterrows()):
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                if abs(v) < 0.01 and v != 0:
                    vals.append(f"{v:.6f}")
                else:
                    vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        if i == best_idx:
            vals = [f"**{v}**" for v in vals]
        lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(lines) + "\n"


def generate_report(results_root):
    """生成完整 md 报告"""
    md_lines = []
    md_lines.append("# 芳香性预测模型实验报告 (0716新数据)\n")
    md_lines.append(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md_lines.append("---\n")

    # ===== 三层训练结果 =====
    md_lines.append("## 1. 三层训练结果\n")

    # Layer 1
    df_l1 = load_layer_results(results_root, 'layer1_ml')
    if not df_l1.empty:
        agg_dict = {
            'cv_r2_mean': ('cv_r2', 'mean'), 'cv_r2_std': ('cv_r2', 'std'),
            'test_r2_mean': ('test_r2', 'mean'), 'test_r2_std': ('test_r2', 'std'),
            'test_mae_mean': ('test_mae', 'mean'), 'test_rmse_mean': ('test_rmse', 'mean'),
        }
        if 'train_time_sec' in df_l1.columns:
            agg_dict['train_time_mean'] = ('train_time_sec', 'mean')
        agg_l1 = df_l1.groupby(['task', 'model']).agg(**agg_dict).reset_index()
        for task in ['HOMA', 'NICS_1zz', 'MBCO']:
            sub = agg_l1[agg_l1['task'] == task].sort_values('test_r2_mean', ascending=False)
            if not sub.empty:
                md_lines.append(format_md_table(sub, f"Layer 1 ML - {task}", 'test_r2_mean'))
    else:
        md_lines.append("### Layer 1 ML\n\n结果未找到\n")

    # Layer 2
    df_l2 = load_layer_results(results_root, 'layer2_gnn')
    if not df_l2.empty:
        agg_l2 = df_l2.groupby(['task', 'model']).agg(
            cv_r2_mean=('cv_r2', 'mean'), cv_r2_std=('cv_r2', 'std'),
            test_r2_mean=('test_r2', 'mean'), test_r2_std=('test_r2', 'std'),
            test_mae_mean=('test_mae', 'mean'), test_rmse_mean=('test_rmse', 'mean'),
        ).reset_index()
        for task in ['HOMA', 'NICS_1zz', 'MBCO']:
            sub = agg_l2[agg_l2['task'] == task].sort_values('test_r2_mean', ascending=False)
            if not sub.empty:
                md_lines.append(format_md_table(sub, f"Layer 2 GNN - {task}", 'test_r2_mean'))
    else:
        md_lines.append("### Layer 2 GNN\n\n结果未找到\n")

    # Layer 3
    df_l3 = load_layer_results(results_root, 'layer3_ring')
    if not df_l3.empty:
        agg_l3 = df_l3.groupby(['task', 'model', 'encoding']).agg(
            cv_r2_mean=('cv_r2', 'mean'), cv_r2_std=('cv_r2', 'std'),
            test_r2_mean=('test_r2', 'mean'), test_r2_std=('test_r2', 'std'),
            test_mae_mean=('test_mae', 'mean'), test_rmse_mean=('test_rmse', 'mean'),
        ).reset_index()
        for task in ['HOMA', 'NICS_1zz', 'MBCO']:
            sub = agg_l3[agg_l3['task'] == task].sort_values('test_r2_mean', ascending=False)
            if not sub.empty:
                md_lines.append(format_md_table(sub, f"Layer 3 Ring Encoding - {task}", 'test_r2_mean'))
    else:
        md_lines.append("### Layer 3 Ring Encoding\n\n结果未找到\n")

    # ===== 三层最优对比 =====
    md_lines.append("\n## 2. 三层最优模型对比\n")
    md_lines.append("| 任务 | Layer 1 最优 | Layer 2 最优 | Layer 3 最优 |")
    md_lines.append("|------|-------------|-------------|-------------|")
    for task in ['HOMA', 'NICS_1zz', 'MBCO']:
        l1_best = "N/A"
        l2_best = "N/A"
        l3_best = "N/A"
        if not df_l1.empty:
            sub = df_l1[df_l1['task'] == task]
            if not sub.empty:
                best = sub.loc[sub['test_r2'].astype(float).idxmax()]
                l1_best = f"{best['model']} (R²={best['test_r2']:.4f})"
        if not df_l2.empty:
            sub = df_l2[df_l2['task'] == task]
            if not sub.empty:
                best = sub.loc[sub['test_r2'].astype(float).idxmax()]
                l2_best = f"{best['model']} (R²={best['test_r2']:.4f})"
        if not df_l3.empty:
            sub = df_l3[df_l3['task'] == task]
            if not sub.empty:
                best = sub.loc[sub['test_r2'].astype(float).idxmax()]
                l3_best = f"{best['model']}/{best['encoding']} (R²={best['test_r2']:.4f})"
        md_lines.append(f"| {task} | {l1_best} | {l2_best} | {l3_best} |")
    md_lines.append("")

    # ===== 泛化测试结果 =====
    md_lines.append("\n## 3. 集外测试结果\n")
    df_gen = load_generalization_results(results_root)
    if not df_gen.empty:
        for test_type in ['scaffold', 'ring_type', 'lunci6', 'lunci78']:
            sub = df_gen[df_gen['test_type'] == test_type].copy()
            if sub.empty:
                continue
            sub = sub.sort_values(['task', 'test_r2'], ascending=[True, False])
            md_lines.append(format_md_table(sub, f"集外测试 - {test_type}", 'test_r2'))
    else:
        md_lines.append("泛化测试结果未找到\n")

    # ===== 环编码范式对比 =====
    if not df_l3.empty:
        md_lines.append("\n## 4. 环编码范式对比 (Layer 3)\n")
        for task in ['HOMA', 'NICS_1zz', 'MBCO']:
            sub = df_l3[df_l3['task'] == task]
            if sub.empty:
                continue
            enc_agg = sub.groupby('encoding').agg(
                test_r2_mean=('test_r2', 'mean'),
                test_mae_mean=('test_mae', 'mean'),
            ).reset_index().sort_values('test_r2_mean', ascending=False)
            md_lines.append(format_md_table(enc_agg, f"环编码对比 - {task} (跨模型平均)", 'test_r2_mean'))

    md_lines.append("\n---\n")
    md_lines.append("*报告自动生成*\n")

    return "\n".join(md_lines)


def main():
    parser = argparse.ArgumentParser(description='生成实验报告')
    parser.add_argument('--results_root', type=str,
                        default=os.path.join(CODE_ROOT, 'results'))
    args = parser.parse_args()

    report = generate_report(args.results_root)
    out_path = os.path.join(args.results_root, 'final_report.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"报告已生成: {out_path}")
    print(f"\n报告预览:\n{'='*60}")
    print(report[:3000])


if __name__ == '__main__':
    main()
