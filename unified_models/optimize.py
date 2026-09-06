"""
Optuna超参数调优脚本
支持所有5种模型 × 3种编码方式
"""
import os
import sys
import argparse
import subprocess
import pandas as pd
import optuna
from optuna.trial import Trial

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)


def run_trial(trial: Trial, args, output_root):
    """运行单个Optuna试验"""
    params = {
        'n_conv_layers': trial.suggest_int('n_conv_layers', 2, 5),
        'n_hidden_layers': trial.suggest_int('n_hidden_layers', 1, 4),
        'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256]),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'p_dropout': trial.suggest_float('p_dropout', 0.1, 0.4),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
    }
    if args.model == 'gat':
        params['n_heads'] = trial.suggest_categorical('n_heads', [4, 8])

    exp_id = f"trial_{trial.number:04d}"
    exp_dir = os.path.join(output_root, exp_id)
    os.makedirs(exp_dir, exist_ok=True)

    cmd = [
        'python', '-u', os.path.join(PROJ_ROOT, 'unified_models/train.py'),
        '--model', args.model,
        '--mode', args.mode,
        '--dataset_path', args.dataset_path,
        '--output_dir', exp_dir,
        '--n_epochs', str(args.n_epochs),
        '--splitter', args.splitter,
        '--no_plot',
        '--seed', '42',
        '--n_threads', '4',
        '--patience', '20',
    ]
    for k, v in params.items():
        cmd.extend([f'--{k}', str(v)])

    print(f"\nTrial {trial.number}: {params}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True,
                                text=True, timeout=args.trial_timeout)
        summary_path = os.path.join(exp_dir, 'summary.csv')
        if os.path.exists(summary_path):
            df = pd.read_csv(summary_path)
            val_loss_row = df[df['metric'] == 'best_val_loss']
            if not val_loss_row.empty:
                val_loss = float(val_loss_row['value'].iloc[0])
                print(f"Trial {trial.number} val_loss={val_loss:.6f}")
                trial.set_user_attr('output_dir', exp_dir)
                return val_loss
        print(f"Trial {trial.number}: 无法提取结果")
        return float('inf')
    except subprocess.CalledProcessError as e:
        print(f"Trial {trial.number} failed: {e.stderr[-500:]}")
        return float('inf')
    except subprocess.TimeoutExpired:
        print(f"Trial {trial.number} timed out")
        return float('inf')


def main():
    parser = argparse.ArgumentParser(description='Optuna超参数调优')
    parser.add_argument('--model', type=str, required=True,
                       choices=['gat', 'gin', 'gnn', 'mpnn', 'graphsage'])
    parser.add_argument('--mode', type=str, required=True,
                       choices=['label', 'mask', 'pool'])
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--n_trials', type=int, default=20)
    parser.add_argument('--n_epochs', type=int, default=50,
                       help='每试验的epoch数 (默认50加速)')
    parser.add_argument('--splitter', type=str, default='random')
    parser.add_argument('--trial_timeout', type=int, default=1800)
    parser.add_argument('--study_name', type=str, default=None)
    args = parser.parse_args()

    study_name = args.study_name or f"{args.model}_{args.mode}_study"
    output_root = os.path.join(PROJ_ROOT, f'unified_optuna_{study_name}')
    os.makedirs(output_root, exist_ok=True)

    storage_url = f"sqlite:///{os.path.join(output_root, 'study.db')}"
    study = optuna.create_study(
        direction='minimize',
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3, n_warmup_steps=10, interval_steps=1
        )
    )

    print(f"开始 Optuna 调优: {args.model}-{args.mode}")
    print(f"试验数: {args.n_trials}, 每试验epoch: {args.n_epochs}")
    study.optimize(lambda trial: run_trial(trial, args, output_root),
                  n_trials=args.n_trials, show_progress_bar=True)

    print("\n" + "=" * 60)
    print(f"完成! 试验数: {len(study.trials)}")
    if study.best_trial:
        print(f"最佳 val_loss: {study.best_trial.value:.6f}")
        print("最佳参数:")
        for k, v in study.best_trial.params.items():
            print(f"  {k}: {v}")

    trials_df = study.trials_dataframe()
    trials_df.to_csv(os.path.join(output_root, 'all_trials.csv'), index=False)

    best_params = study.best_trial.params
    best_params['best_val_loss'] = study.best_trial.value
    pd.DataFrame([best_params]).to_csv(
        os.path.join(output_root, 'best_parameters.csv'), index=False)
    print(f"\n结果保存到: {output_root}")


if __name__ == "__main__":
    main()
