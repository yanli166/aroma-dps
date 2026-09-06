"""展示当前最优结果 - 测试集散点图与汇总"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib

# --- Auto path bootstrap (do not remove) ---
import os as _os
_THIS_FILE = _os.path.abspath(__file__)
_d = _os.path.dirname(_THIS_FILE)
while not _os.path.exists(_os.path.join(_d, 'unified_models')) and _d != '/':
    _d = _os.path.dirname(_d)
_PROJ_ROOT = _d
# --- End auto path bootstrap ---

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

RESULTS_ROOT = '_PROJ_ROOT/0427_unified_results'
SAVE_DIR = os.path.join(RESULTS_ROOT, 'best_results')
os.makedirs(SAVE_DIR, exist_ok=True)


def load_all_results():
    """加载所有15个模型的结果"""
    results = []
    for summary_path in sorted(glob.glob(os.path.join(RESULTS_ROOT, '*/summary.csv'))):
        exp_name = os.path.basename(os.path.dirname(summary_path))
        parts = exp_name.split('_')
        model, mode = parts[0], parts[1]
        df = pd.read_csv(summary_path)
        row = {'model': model, 'mode': mode, 'experiment': exp_name}
        for _, r in df.iterrows():
            row[r['metric']] = r['value']
        results.append(row)
    return pd.DataFrame(results).sort_values('final_val_r2', ascending=False)


def plot_best_scatter(df):
    """为Top 6模型生成测试集散点图"""
    top_n = min(6, len(df))
    top_df = df.head(top_n)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (_, row) in enumerate(top_df.iterrows()):
        ax = axes[idx]
        exp_name = row['experiment']
        data_path = os.path.join(RESULTS_ROOT, exp_name, 'val_set_data.csv')
        
        if not os.path.exists(data_path):
            ax.set_title(f'{exp_name}\n(无数据)', fontsize=11)
            ax.set_visible(True)
            continue
        
        data = pd.read_csv(data_path)
        true = data['true'].values
        pred = data['pred'].values
        
        r2 = 1 - np.sum((true - pred)**2) / np.sum((true - true.mean())**2)
        mae = np.mean(np.abs(true - pred))
        rmse = np.sqrt(np.mean((true - pred)**2))
        
        # 散点图
        ax.scatter(true, pred, alpha=0.3, s=8, c='steelblue', edgecolors='none')
        
        # 理想线
        lims = [min(true.min(), pred.min()) - 2, max(true.max(), pred.max()) + 2]
        ax.plot(lims, lims, 'r-', linewidth=1.5, alpha=0.8, label='Ideal')
        
        # 回归线
        if len(true) > 2:
            slope, intercept, r_val, _, _ = stats.linregress(true, pred)
            x_fit = np.linspace(lims[0], lims[1], 100)
            y_fit = slope * x_fit + intercept
            ax.plot(x_fit, y_fit, 'g--', linewidth=1, alpha=0.6, label=f'Fit (r={r_val:.4f})')
        
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('True HOMA', fontsize=10)
        ax.set_ylabel('Predicted HOMA', fontsize=10)
        ax.set_title(f'{exp_name}\nR²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}', fontsize=11)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal')
    
    plt.suptitle('Top 6 Models - Test Set Predictions (Scatter Plots)', fontsize=15, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, 'top6_scatter.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Top 6 散点图: {save_path}")


def plot_best_single(df):
    """最佳模型大散点图"""
    best = df.iloc[0]
    exp_name = best['experiment']
    data_path = os.path.join(RESULTS_ROOT, exp_name, 'val_set_data.csv')
    
    if not os.path.exists(data_path):
        print(f"找不到 {exp_name} 的数据")
        return
    
    data = pd.read_csv(data_path)
    true = data['true'].values
    pred = data['pred'].values
    
    r2 = 1 - np.sum((true - pred)**2) / np.sum((true - true.mean())**2)
    mae = np.mean(np.abs(true - pred))
    rmse = np.sqrt(np.mean((true - pred)**2))
    
    fig, ax = plt.subplots(figsize=(9, 9), dpi=300)
    
    # 颜色映射按误差
    errors = np.abs(true - pred)
    sc = ax.scatter(true, pred, c=errors, cmap='RdYlBu_r', alpha=0.5, s=12, edgecolors='none')
    plt.colorbar(sc, ax=ax, label='|Prediction Error|', shrink=0.8)
    
    # 理想线
    lims = [min(true.min(), pred.min()) - 2, max(true.max(), pred.max()) + 2]
    ax.plot(lims, lims, 'r-', linewidth=2, alpha=0.8, label='Ideal (y=x)')
    
    # 回归线
    slope, intercept, r_val, _, _ = stats.linregress(true, pred)
    x_fit = np.linspace(lims[0], lims[1], 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, 'g--', linewidth=1.5, alpha=0.7, 
            label=f'Linear Fit (y={slope:.3f}x+{intercept:.3f})')
    
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('True HOMA Value', fontsize=14)
    ax.set_ylabel('Predicted HOMA Value', fontsize=14)
    ax.set_title(f'Best Model: {exp_name}\nR²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}', 
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, 'best_model_scatter.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"最佳模型散点图: {save_path}")


def plot_ranking_bar(df):
    """所有15个模型的R²排名柱状图"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    labels = [f"{r['model']}-{r['mode']}" for _, r in df.iterrows()]
    r2_vals = df['final_val_r2'].values
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(df)))
    
    bars = ax.barh(range(len(df)), r2_vals, color=colors, edgecolor='white', linewidth=0.5)
    
    for i, (bar, val) in enumerate(zip(bars, r2_vals)):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{val:.4f}', va='center', fontsize=9, fontweight='bold')
    
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Test R²', fontsize=13)
    ax.set_title('All 15 Models - Test R² Ranking', fontsize=15, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, 'r2_ranking.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"R²排名图: {save_path}")


def print_summary_table(df):
    """打印汇总表"""
    print("\n" + "="*90)
    print("所有15个模型测试集结果汇总 (按R²降序)")
    print("="*90)
    print(f"{'排名':<4} {'实验':<20} {'R²':<10} {'MAE':<10} {'RMSE':<10} {'Val Loss':<10}")
    print("-"*90)
    for i, (_, row) in enumerate(df.iterrows()):
        r2 = row.get('final_val_r2', 0)
        mae = row.get('final_val_mae', 0)
        rmse = row.get('final_val_rmse', 0)
        vloss = row.get('best_val_loss', 0)
        print(f"{i+1:<4} {row['experiment']:<20} {r2:<10.4f} {mae:<10.4f} {rmse:<10.4f} {vloss:<10.4f}")
    print("="*90)


def main():
    df = load_all_results()
    print(f"加载 {len(df)} 个模型结果")
    
    print_summary_table(df)
    plot_best_single(df)
    plot_best_scatter(df)
    plot_ranking_bar(df)
    
    # 保存汇总表
    table = df[['model', 'mode', 'experiment', 'final_val_r2', 'final_val_mae', 
                'final_val_rmse', 'best_val_loss']].copy()
    table.columns = ['Model', 'Mode', 'Experiment', 'Test R²', 'Test MAE', 'Test RMSE', 'Best Val Loss']
    for c in ['Test R²', 'Test MAE', 'Test RMSE', 'Best Val Loss']:
        table[c] = table[c].round(4)
    table.to_csv(os.path.join(SAVE_DIR, 'summary_table.csv'), index=False)
    print(f"\n汇总表: {os.path.join(SAVE_DIR, 'summary_table.csv')}")


if __name__ == '__main__':
    main()
