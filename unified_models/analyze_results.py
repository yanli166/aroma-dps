"""
统一结果分析与可视化脚本
汇总所有15个模型的结果，生成对比表格和图表
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.path.join(PROJ_ROOT, '0427_unified_results')
OPTUNA_ROOT = os.path.join(PROJ_ROOT, 'unified_optuna')


def collect_results(results_dir):
    """收集所有模型的summary结果"""
    results = []
    for summary_path in sorted(glob.glob(os.path.join(results_dir, '*/summary.csv'))):
        model_dir = os.path.dirname(summary_path)
        exp_name = os.path.basename(model_dir)
        try:
            parts = exp_name.split('_')
            model = parts[0]
            mode = parts[1] if len(parts) > 1 else 'unknown'
        except Exception:
            model, mode = exp_name, 'unknown'

        try:
            df = pd.read_csv(summary_path)
            row = {'model': model, 'mode': mode, 'experiment': exp_name}
            for _, r in df.iterrows():
                row[r['metric']] = r['value']
            results.append(row)
        except Exception as e:
            print(f"跳过 {exp_name}: {e}")
    return pd.DataFrame(results)


def create_comparison_table(df):
    """创建对比表格"""
    cols = ['model', 'mode', 'best_val_loss', 'final_val_mae', 'final_val_r2',
            'final_val_rmse', 'max_val_r2', 'min_val_mae']
    available_cols = [c for c in cols if c in df.columns]
    table = df[available_cols].copy()

    for c in table.select_dtypes(include=[np.number]).columns:
        table[c] = table[c].round(4)

    table = table.sort_values(['model', 'mode'])
    return table


def plot_r2_comparison(df, save_dir):
    """绘制R²对比柱状图"""
    fig, ax = plt.subplots(figsize=(14, 7))

    models = df['model'].unique()
    modes = df['mode'].unique()
    x = np.arange(len(models))
    width = 0.25
    colors = {'label': '#4C72B0', 'mask': '#DD8452', 'pool': '#55A868'}

    for i, mode in enumerate(modes):
        vals = []
        for m in models:
            sub = df[(df['model'] == m) & (df['mode'] == mode)]
            if len(sub) > 0 and 'final_val_r2' in sub.columns:
                vals.append(sub['final_val_r2'].iloc[0])
            else:
                vals.append(0)
        ax.bar(x + i * width, vals, width, label=mode, color=colors.get(mode, 'gray'))

    ax.set_xlabel('Model', fontsize=13)
    ax.set_ylabel('Validation R²', fontsize=13)
    ax.set_title('Validation R² Comparison Across Models and Encoding Modes', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'r2_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_mae_comparison(df, save_dir):
    """绘制MAE对比柱状图"""
    fig, ax = plt.subplots(figsize=(14, 7))

    models = df['model'].unique()
    modes = df['mode'].unique()
    x = np.arange(len(models))
    width = 0.25
    colors = {'label': '#4C72B0', 'mask': '#DD8452', 'pool': '#55A868'}

    for i, mode in enumerate(modes):
        vals = []
        for m in models:
            sub = df[(df['model'] == m) & (df['mode'] == mode)]
            if len(sub) > 0 and 'final_val_mae' in sub.columns:
                vals.append(sub['final_val_mae'].iloc[0])
            else:
                vals.append(0)
        ax.bar(x + i * width, vals, width, label=mode, color=colors.get(mode, 'gray'))

    ax.set_xlabel('Model', fontsize=13)
    ax.set_ylabel('Validation MAE', fontsize=13)
    ax.set_title('Validation MAE Comparison Across Models and Encoding Modes', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mae_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_best_model_parity(results_dir, df, save_dir):
    """为最佳模型绘制parity plot"""
    if 'final_val_r2' not in df.columns:
        return
    best_row = df.loc[df['final_val_r2'].idxmax()]
    exp_name = best_row['experiment']
    val_data_path = os.path.join(results_dir, exp_name, 'val_set_data.csv')

    if not os.path.exists(val_data_path):
        print(f"找不到 {exp_name} 的验证集数据")
        return

    data = pd.read_csv(val_data_path)
    true = data['true'].values
    pred = data['pred'].values

    r2 = 1 - np.sum((true - pred) ** 2) / np.sum((true - true.mean()) ** 2)
    mae = np.mean(np.abs(true - pred))
    rmse = np.sqrt(np.mean((true - pred) ** 2))

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    ax.scatter(true, pred, alpha=0.4, s=8, c='steelblue')
    lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
    ax.plot(lims, lims, 'r-', linewidth=2, alpha=0.7)
    ax.set_xlabel('True HOMA Values', fontsize=13)
    ax.set_ylabel('Predicted HOMA Values', fontsize=13)
    ax.set_title(f'Best Model: {exp_name}\nR²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'best_model_parity.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"最佳模型: {exp_name} (R²={r2:.4f})")


def plot_all_parity_grid(results_dir, df, save_dir):
    """绘制所有模型的parity plot网格"""
    models = sorted(df['model'].unique())
    modes = ['label', 'mask', 'pool']
    fig, axes = plt.subplots(len(models), len(modes), figsize=(18, 30))

    for i, model in enumerate(models):
        for j, mode in enumerate(modes):
            ax = axes[i][j]
            sub = df[(df['model'] == model) & (df['mode'] == mode)]
            if len(sub) == 0:
                ax.set_visible(False)
                continue

            exp_name = sub.iloc[0]['experiment']
            val_path = os.path.join(results_dir, exp_name, 'val_set_data.csv')
            if not os.path.exists(val_path):
                ax.set_title(f'{model}-{mode}\n(无数据)')
                continue

            data = pd.read_csv(val_path)
            true, pred = data['true'].values, data['pred'].values
            r2 = 1 - np.sum((true - pred) ** 2) / np.sum((true - true.mean()) ** 2)

            ax.scatter(true, pred, alpha=0.3, s=3, c='steelblue')
            lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
            ax.plot(lims, lims, 'r-', linewidth=1, alpha=0.5)
            ax.set_title(f'{model}-{mode}\nR²={r2:.4f}', fontsize=11)
            ax.grid(True, alpha=0.3)

    plt.suptitle('All Models Parity Plots (Validation Set)', fontsize=16, y=1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'all_parity_grid.png'), dpi=200, bbox_inches='tight')
    plt.close()


def generate_report(df, save_dir):
    """生成Markdown报告"""
    report_path = os.path.join(save_dir, 'REPORT.md')
    with open(report_path, 'w') as f:
        f.write("# 统一模型对比结果报告\n\n")
        f.write(f"## 1. 实验概览\n\n")
        f.write(f"- 模型总数: {len(df)}\n")
        f.write(f"- 模型类型: {', '.join(sorted(df['model'].unique()))}\n")
        f.write(f"- 编码方式: {', '.join(sorted(df['mode'].unique()))}\n\n")

        f.write("## 2. 结果汇总表\n\n")
        table = create_comparison_table(df)
        f.write(table.to_markdown(index=False))
        f.write("\n\n")

        if 'final_val_r2' in df.columns:
            f.write("## 3. 关键发现\n\n")
            best = df.loc[df['final_val_r2'].idxmax()]
            f.write(f"- **最佳模型**: {best['experiment']} (R²={best['final_val_r2']:.4f})\n")

            for mode in ['label', 'mask', 'pool']:
                sub = df[df['mode'] == mode]
                if len(sub) > 0:
                    best_m = sub.loc[sub['final_val_r2'].idxmax()]
                    f.write(f"- **{mode}模式最佳**: {best_m['model']} (R²={best_m['final_val_r2']:.4f})\n")

            f.write("\n## 4. 模型排名 (按验证R²)\n\n")
            ranked = df.sort_values('final_val_r2', ascending=False)
            f.write("| 排名 | 实验 | R² | MAE | RMSE |\n")
            f.write("|------|------|-----|-----|------|\n")
            for i, (_, row) in enumerate(ranked.iterrows()):
                r2 = row.get('final_val_r2', 0)
                mae = row.get('final_val_mae', 0)
                rmse = row.get('final_val_rmse', 0)
                f.write(f"| {i+1} | {row['experiment']} | {r2:.4f} | {mae:.4f} | {rmse:.4f} |\n")

    print(f"报告已保存到: {report_path}")


def main():
    if not os.path.exists(RESULTS_ROOT):
        print(f"结果目录不存在: {RESULTS_ROOT}")
        return

    save_dir = os.path.join(RESULTS_ROOT, 'analysis')
    os.makedirs(save_dir, exist_ok=True)

    print("收集结果...")
    df = collect_results(RESULTS_ROOT)
    if df.empty:
        print("未找到任何结果!")
        return

    print(f"找到 {len(df)} 个实验结果")

    table = create_comparison_table(df)
    table.to_csv(os.path.join(save_dir, 'comparison_table.csv'), index=False)
    print("\n对比表格:")
    print(table.to_string(index=False))

    print("\n生成图表...")
    if 'final_val_r2' in df.columns:
        plot_r2_comparison(df, save_dir)
    if 'final_val_mae' in df.columns:
        plot_mae_comparison(df, save_dir)
    plot_best_model_parity(RESULTS_ROOT, df, save_dir)
    plot_all_parity_grid(RESULTS_ROOT, df, save_dir)

    print("\n生成报告...")
    generate_report(df, save_dir)

    print(f"\n所有分析结果保存到: {save_dir}")


if __name__ == "__main__":
    main()
