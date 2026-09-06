"""
Metrics: unified evaluation metrics.

Consolidates:
- code_end/common/tasks.py::compute_metrics (sklearn r2_score)
- code_end/generalization_test/code/anchor_step3_run_models.py::compute_metrics (manual R²)

Uses sklearn.metrics.r2_score for consistency.
"""
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import spearmanr


def compute_metrics(y_true, y_pred):
    """Compute MAE, RMSE, R², and Spearman correlation.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        Dict with mae, rmse, r2, spearman_r, spearman_p.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    r2 = float(r2_score(y_true, y_pred))

    spearman_r, spearman_p = spearmanr(y_true, y_pred)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
    }
