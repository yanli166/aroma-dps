import os
import subprocess
import itertools
import random
import hashlib
import argparse
import pandas as pd
import time

def combination_hash(combination):
    """生成参数组合的唯一哈希值"""
    return hashlib.md5(str(combination).encode()).hexdigest()[:8]

def main():
    parser = argparse.ArgumentParser(description='Sequential Hyperparameter Search for GAT Model')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to dataset CSV file')
    parser.add_argument('--train_script', type=str, default='train_gat.py',
                        help='Path to training script')
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Number of hyperparameter combinations to try')
    parser.add_argument('--use_cv', action='store_true',
                        help='Use cross-validation (k_folds=5) if set, else simple train/val split')
    args = parser.parse_args()

    # 简化参数网格
    param_grid = {
        "--n_conv_layers": [2, 3, 4, 5, 6],
        "--n_heads": [1, 2, 4, 8],
        "--learning_rate": [0.001, 0.0005, 0.0001],
        "--p_dropout": [0, 0.2, 0.4, 0.6],
        "--batch_size": [16, 32, 64],
        "--n_hidden_layers": [1, 2, 3, 4],
        "--weight_decay": [1e-5],
        "--node_vec_len": [60],
    }

    # 创建所有可能的参数组合
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combinations = list(itertools.product(*values))
    
    # 随机选择指定数量的组合
    random.seed(88)
    selected_combinations = random.sample(all_combinations, min(args.n_trials, len(all_combinations)))
    
    # 结果目录和汇总文件
    output_root = "param_search_results"
    os.makedirs(output_root, exist_ok=True)
    summary_file = os.path.join(output_root, "param_summary.csv")
    
    # 写入结果表头
    header = "exp_id," + ",".join(k.replace("--", "") for k in keys) + ",avg_val_loss,avg_val_r2,status\n"
    with open(summary_file, "w") as f:
        f.write(header)
    
    print(f"Starting sequential hyperparameter search with {len(selected_combinations)} trials")
    if args.use_cv:
        print("Using cross-validation (k_folds=5)")
    else:
        print("Using simple train/validation split (k_folds=0)")
    
    for i, combination in enumerate(selected_combinations):
        exp_id = f"exp_{combination_hash(combination)}"
        exp_dir = os.path.join(output_root, exp_id)
        os.makedirs(exp_dir, exist_ok=True)
        
        # 构建基础命令
        cmd = [
            "python", 
            args.train_script,
            "--dataset_path", args.dataset_path,
            "--output_dir", exp_dir,
            "--n_epochs", "300",  # 减少轮数以加速测试
            "--use_gpu",
        ]
        
        # 根据是否使用交叉验证添加参数
        if args.use_cv:
            cmd.extend(["--k_folds", "5"])
        else:
            # 使用默认训练集比例 (80%)
            cmd.extend(["--train_size", "0.8"])
        
        # 添加超参数
        for key, value in zip(keys, combination):
            cmd.extend([key, str(value)])
        
        print(f"\n\nStarting trial {i+1}/{len(selected_combinations)}: {exp_id}")
        print("Command:", " ".join(cmd))
        
        # 执行训练命令
        start_time = time.time()
        try:
            # 运行训练脚本
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            status = "SUCCESS"
            print(f"Experiment completed successfully in {time.time()-start_time:.1f} seconds")
            
            # 保存完整输出
            with open(os.path.join(exp_dir, "full_output.log"), "w") as f:
                f.write(result.stdout)
                if result.stderr:
                    f.write("\n\nSTDERR:\n")
                    f.write(result.stderr)
            
            # 提取结果（根据是否使用交叉验证）
            result_file = os.path.join(exp_dir, "cv_results.csv" if args.use_cv else "test_results.csv")
            avg_loss = "N/A"
            avg_r2 = "N/A"
            
            if os.path.exists(result_file):
                with open(result_file, "r") as f:
                    lines = f.readlines()
                    if lines and len(lines) >= 2:
                        if args.use_cv:
                            # 交叉验证结果（最后一行是平均值）
                            avg_line = lines[-1].strip()
                            parts = avg_line.split(',')
                            if len(parts) >= 4:
                                avg_loss = parts[1]
                                avg_r2 = parts[3]
                        else:
                            # 简单划分结果（第一行是验证集结果）
                            avg_line = lines[0].strip()
                            parts = avg_line.split(',')
                            if len(parts) >= 4:
                                avg_loss = parts[1]
                                avg_r2 = parts[3]
            
        except subprocess.CalledProcessError as e:
            status = "FAILED"
            print(f"Experiment failed with exit code {e.returncode}")
            
            # 保存错误信息
            error_log = os.path.join(exp_dir, "error.log")
            with open(error_log, "w") as f:
                f.write(f"Command: {' '.join(cmd)}\n\n")
                f.write(f"Exit code: {e.returncode}\n\n")
                f.write("=== STDOUT ===\n")
                f.write(e.stdout if e.stdout else "<empty>\n")
                f.write("\n=== STDERR ===\n")
                f.write(e.stderr if e.stderr else "<empty>\n")
            
        except Exception as e:
            status = "ERROR"
            print(f"Unexpected error: {str(e)}")
            
        # 更新汇总文件
        with open(summary_file, "a") as f:
            params = ",".join(str(v) for v in combination)
            f.write(f"{exp_id},{params},{avg_loss},{avg_r2},{status}\n")
        
        print(f"Progress: {i+1}/{len(selected_combinations)} completed")
    
    print("\nHyperparameter search completed!")
    print(f"Results summary saved to: {summary_file}")

if __name__ == "__main__":
    main()