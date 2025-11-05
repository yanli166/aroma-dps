import os
import subprocess
import itertools
import random
import hashlib
import argparse
import pandas as pd
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

def combination_hash(combination):
    """生成参数组合的唯一哈希值"""
    return hashlib.md5(str(combination).encode()).hexdigest()[:8]

def run_experiment(args, combination, output_root):
    """运行单个参数组合的实验"""
    # 从args中解包需要的参数
    train_script = args.train_script
    dataset_path = args.dataset_path
    
    # 基础命令模板
    base_cmd = [
        "python", 
        train_script,
        "--dataset_path", dataset_path,
        "--n_epochs", "300",  # 每个实验训练300轮
        "--use_gpu"
    ]
    
    # 获取参数键列表
    keys = list(args.param_grid.keys())
    
    # 生成实验ID
    exp_id = f"exp_{combination_hash(combination)}"
    exp_dir = os.path.join(output_root, exp_id)
    
    # 构建完整命令
    cmd = base_cmd.copy()
    cmd += ["--output_dir", exp_dir]
    
    # 添加超参数
    for key, value in zip(keys, combination):
        cmd.extend([key, str(value)])
    
    print(f"Starting experiment {exp_id}")
    print("Command:", " ".join(cmd))
    
    # 执行训练命令
    try:
        # 使用subprocess.PIPE捕获输出，防止控制台混乱
        result = subprocess.run(
            cmd, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 打印成功信息
        print(f"Experiment {exp_id} completed successfully")
        print("Output:", result.stdout[:500])  # 只打印部分输出
        
        # 提取结果
        result_file = os.path.join(exp_dir, "cv_results.csv")
        if os.path.exists(result_file):
            with open(result_file, "r") as f:
                lines = f.readlines()
                if lines and len(lines) >= 2:  # 确保有足够行数
                    avg_line = lines[-1].strip()  # 最后一行是平均值
                    parts = avg_line.split(',')
                    if len(parts) >= 4:
                        avg_loss = parts[1]
                        avg_r2 = parts[3]
                        return exp_id, combination, avg_loss, avg_r2
        
        # 如果结果文件不存在或格式错误
        print(f"Warning: Result file missing or invalid for {exp_id}")
        return exp_id, combination, "MISSING", "MISSING"
        
    except subprocess.CalledProcessError as e:
        # 处理命令执行失败
        error_msg = f"Experiment {exp_id} failed with code {e.returncode}:\n"
        error_msg += f"Stdout: {e.stdout[:500]}\n" if e.stdout else ""
        error_msg += f"Stderr: {e.stderr[:500]}" if e.stderr else ""
        print(error_msg)
        return exp_id, combination, "FAILED", "FAILED"
    
    except Exception as e:
        # 处理其他异常
        print(f"Unexpected error in {exp_id}: {str(e)}")
        return exp_id, combination, "ERROR", "ERROR"

def main():
    parser = argparse.ArgumentParser(description='Parallel Hyperparameter Search for GAT Model')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to dataset CSV file')
    parser.add_argument('--train_script', type=str, default='train_gat.py',
                        help='Path to training script')
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Number of hyperparameter combinations to try')
    parser.add_argument('--max_workers', type=int, default=1,
                        help='Maximum number of parallel experiments')
    args = parser.parse_args()

    # 参数网格作为类的属性，而不是局部变量
    args.param_grid = {
        "--n_conv_layers": [2, 3, 4, 5],
        "--n_heads": [1, 2, 4, 8],
        "--learning_rate": [0.01, 0.001, 0.0001],
        "--p_dropout": [0.0, 0.1, 0.2, 0.3, 0.4],
        "--batch_size": [16, 32],
        "--n_hidden_layers": [1, 2, 3],
        "--weight_decay": [1e-5, 1e-4],
        "--node_vec_len": [60, 100],
    }

    # 创建所有可能的参数组合
    keys = list(args.param_grid.keys())
    values = list(args.param_grid.values())
    all_combinations = list(itertools.product(*values))
    
    # 随机选择指定数量的组合
    random.seed(18)
    selected_combinations = random.sample(all_combinations, min(args.n_trials, len(all_combinations)))
    
    # 结果目录和汇总文件
    output_root = "hyperparam_search_results"
    os.makedirs(output_root, exist_ok=True)
    summary_file = os.path.join(output_root, "param_summary.csv") 
    
    # 写入结果表头
    header = "exp_id," + ",".join(k.replace("--", "") for k in keys) + ",avg_val_loss,avg_val_r2\n"
    with open(summary_file, "w") as f:
        f.write(header)
    
    print(f"Starting parallel hyperparameter search with {len(selected_combinations)} trials")
    print(f"Using {args.max_workers} parallel workers")
    
    # 使用进程池并行执行
    completed = 0
    failed = 0
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        # 提交所有实验任务
        futures = {}
        for comb in selected_combinations:
            # 将args作为参数传递
            future = executor.submit(run_experiment, args, comb, output_root)
            futures[future] = comb
        
        # 处理完成的任务
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    exp_id, combination, avg_loss, avg_r2 = result
                    
                    # 保存结果到汇总文件
                    with open(summary_file, "a") as f:
                        params = ",".join(str(v) for v in combination)
                        f.write(f"{exp_id},{params},{avg_loss},{avg_r2}\n")
                    
                    completed += 1
                    print(f"Progress: {completed}/{len(selected_combinations)}")
                else:
                    failed += 1
            except Exception as e:
                print(f"Error processing future: {str(e)}")
                failed += 1
    
    # 分析结果
    print("\nHyperparameter search completed!")
    print(f"Successful: {completed}, Failed: {failed}")
    print(f"Results saved to: {summary_file}")
    
    # 简单结果分析
    try:
        df = pd.read_csv(summary_file)
        
        # 处理可能的空文件
        if df.empty:
            print("No results to analyze")
            return
        
        # 过滤无效结果
        valid_df = df[~df['avg_val_loss'].isin(["FAILED", "MISSING", "ERROR"])]
        
        if not valid_df.empty:
            # 转换为数值类型
            for col in valid_df.columns:
                if col != 'exp_id':
                    # 尝试转换为数值，无法转换的设为NaN
                    valid_df[col] = pd.to_numeric(valid_df[col], errors='coerce')
            
            # 找到最佳结果
            if not valid_df['avg_val_loss'].isnull().all():
                best_loss = valid_df.loc[valid_df['avg_val_loss'].idxmin()]
                print("\nBest parameters by validation loss:")
                print(best_loss.to_string())
            
            if not valid_df['avg_val_r2'].isnull().all():
                best_r2 = valid_df.loc[valid_df['avg_val_r2'].idxmax()]
                print("\nBest parameters by validation R²:")
                print(best_r2.to_string())
        else:
            print("No valid results to analyze")
            
    except Exception as e:
        print(f"Error analyzing results: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()