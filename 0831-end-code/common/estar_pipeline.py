"""
[0831 重构] 共享 E*=median 训练-评估流水线 (Stage1/2/4 通用)

协议:
  1. 固定 80/20 holdout (split_seed=2026, 与 model seed 解耦)
  2. 80% development 内做 5-fold Group CV
  3. 每折训练: 早停, 记录 best_epoch (按 val MAE)
  4. E* = median(best_epochs) (取自 CV, 不用 test)
  5. 在完整 80% development 上按 E* 重训 final 模型 (固定 epoch 数)
  6. 在固定 20% test 上评估 (test 绝不参与任何选择)

支持自定义 GNN 与 PyG 模型, 通过传入 runner 函数实现:
  - runner_cv(backbone, ring_flag, train_idx, val_idx, ...) -> (model, val_mae, val_r2, val_rmse)
  - runner_final(backbone, ring_flag, train_idx, n_epochs, ...) -> model

也支持传统 ML:
  - ml_cv_final(model_factory, X, y, splits, ...) -> (cv_metrics, final_metrics)
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Callable, Optional, List, Dict, Any

import numpy as np
import torch


@dataclass
class EStarResult:
    cv_mae: float
    cv_r2: float
    cv_rmse: float
    best_epochs: List[int]
    estar: int
    n_ceiling_hits: int
    train_metrics: tuple  # (r2, mae, rmse)
    test_metrics: tuple   # (r2, mae, rmse)
    train_time_sec: float
    n_total: int
    n_test: int
    n_dev: int


def run_gnn_estar(
    backbone: str,
    ring_flag: int,
    splits: dict,
    params: dict,
    train_cv_fn: Callable,
    fit_final_fn: Callable,
    eval_fn: Callable,
    device,
    seed: int,
    max_epochs: int,
    patience: int,
    *,
    n_folds: int = 5,
    feature_mode: str = 'standard',
) -> EStarResult:
    """通用 E*=median GNN 训练流水线。

    train_cv_fn(backbone, ring_flag, train_idx, val_idx, n_epochs, patience, seed) -> model
        (已包含 early-stop, model.best_epoch 已记录, model.state_dict 为 best)
    fit_final_fn(backbone, ring_flag, train_idx, n_epochs, seed) -> model
        (固定 n_epochs, 无 early-stop)
    eval_fn(model, idx) -> (r2, mae, rmse)
    """
    folds = splits['folds']
    train_idx_dev = np.asarray(splits['train_idx'])
    test_idx = np.asarray(splits['test_idx'])

    # -------- 5 折 CV --------
    cv_rows = []
    best_epochs: List[int] = []
    for k, (tr, va) in enumerate(folds):
        m = train_cv_fn(backbone, ring_flag, np.asarray(tr), np.asarray(va),
                        max_epochs, patience, seed + k, feature_mode)
        r2, mae, rmse = eval_fn(m, va)
        cv_rows.append({'fold': k + 1, 'r2': r2, 'mae': mae, 'rmse': rmse})
        best_epochs.append(int(getattr(m, 'best_epoch', max_epochs)))
        del m

    cv_mae = float(np.mean([r['mae'] for r in cv_rows]))
    cv_r2 = float(np.mean([r['r2'] for r in cv_rows]))
    cv_rmse = float(np.mean([r['rmse'] for r in cv_rows]))
    estar = int(np.median(best_epochs))
    n_ceiling_hits = int(sum(1 for e in best_epochs if e >= max_epochs - 1))

    if n_ceiling_hits == len(best_epochs):
        print(f"  [warn] E*={estar} 撞上限 max_epochs={max_epochs} "
              f"(全部 {n_ceiling_hits} 折); 应提高 max_epochs。")

    # -------- Final: 完整 80% development 上训练 E* epochs --------
    t0 = time.time()
    fm = fit_final_fn(backbone, ring_flag, train_idx_dev, estar, seed, feature_mode)
    train_time = time.time() - t0
    train_metrics = eval_fn(fm, train_idx_dev)
    test_metrics = eval_fn(fm, test_idx)

    return EStarResult(
        cv_mae=cv_mae, cv_r2=cv_r2, cv_rmse=cv_rmse,
        best_epochs=best_epochs, estar=estar,
        n_ceiling_hits=n_ceiling_hits,
        train_metrics=train_metrics, test_metrics=test_metrics,
        train_time_sec=train_time,
        n_total=int(len(splits['train_idx']) + len(splits['test_idx'])),
        n_dev=int(len(train_idx_dev)), n_test=int(len(test_idx)),
    )
