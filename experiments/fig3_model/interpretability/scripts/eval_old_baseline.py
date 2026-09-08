# -*- coding: utf-8 -*-
"""
Step 3: 旧最优模型 (best_model_package, 只在 collet 全量上训练) 在相同的
       固定 a/b test 上的预测, 用于与 merged 重训模型 apples-to-apples 对比。

用法: python3 eval_old_baseline.py --gpu 2
输出:
  predictions/old_{TASK}_test_predictions.csv
  results/old_baseline_metrics.json
"""
import os
import sys
import json
import ast
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr

sys.path.insert(0, "/home/ubuntu/aroma-dps-code/best_model_package")
from graph_utils import build_graph
from model_arch import build_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
PRED_DIR = os.path.join(ROOT, "predictions")
RESULT_DIR = os.path.join(ROOT, "results")
OLD_PKG = "/home/ubuntu/aroma-dps-code/best_model_package"

NODE_VEC_LEN, MAX_ATOMS = 60, 75
BS = 64
OLD_FILES = {"HOMA": "homa_best.pt", "NICS_1zz": "nics_1zz_best.pt", "MBCO": "mbco_best.pt"}


def parse_aor(x):
    if isinstance(x, str):
        return ast.literal_eval(x)
    return list(x)


def load_ckpt(task, device):
    ckpt = torch.load(os.path.join(OLD_PKG, OLD_FILES[task]), map_location="cpu")
    cfg = ckpt["config"]
    model = build_model(use_projection=cfg["use_projection"], node_vec_len=cfg["node_vec_len"],
                        hidden_dim=cfg["hidden_dim"], n_conv=cfg["n_conv"],
                        n_hidden=cfg["n_hidden"], p_dropout=cfg["p_dropout"],
                        ring_flag_value=cfg["ring_flag_value"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--task", default=None, choices=list(OLD_FILES),
                    help="只评估某个任务; 缺省评估全部")
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[eval_old_baseline] device={device}")

    tasks = [args.task] if args.task else list(OLD_FILES)
    out_metrics = {}
    for task in tasks:
        merged = pd.read_csv(os.path.join(DATA_DIR, f"merged_{task}.csv"))
        merged["atom_on_ring"] = merged["atom_on_ring"].apply(parse_aor)
        test_df = merged[merged["split"] == "test"].reset_index(drop=True)
        y = test_df["y"].astype(float).values
        print(f"\n=== {task}: test n={len(test_df)} ===")

        model = load_ckpt(task, device)
        preds = []
        with torch.no_grad():
            for i in range(0, len(test_df), BS):
                smis = test_df["smiles"].iloc[i:i + BS].tolist()
                aors = test_df["atom_on_ring"].iloc[i:i + BS].tolist()
                nm, adj, ri = [], [], []
                for smi, aor in zip(smis, aors):
                    g = build_graph(smi, aor, NODE_VEC_LEN, MAX_ATOMS, ring_flag_value=1)
                    nm.append(g["node_mat"]); adj.append(g["adj_mat"]); ri.append(g["ring_indices"])
                nm = torch.tensor(np.array(nm, dtype=np.float32), device=device)
                adj = torch.tensor(np.array(adj, dtype=np.float32), device=device)
                ri = torch.tensor(np.array(ri, dtype=np.int64), device=device)
                p = model(nm, adj, ri).reshape(-1).cpu().numpy()
                preds.append(p)
        pv = np.concatenate(preds)
        r2 = r2_score(y, pv)
        mae = mean_absolute_error(y, pv)
        rmse = float(np.sqrt(mean_squared_error(y, pv)))
        spr = float(spearmanr(y, pv).correlation)
        print(f"  OLD baseline: R2={r2:.4f} MAE={mae:.4f} RMSE={rmse:.4f} Spearman={spr:.4f}")

        pred_df = test_df[["smiles", "atom_on_ring", "mol_id", "source"]].copy()
        pred_df["y_true"] = y
        pred_df["y_pred_old"] = pv
        pred_df.to_csv(os.path.join(PRED_DIR, f"old_{task}_test_predictions.csv"), index=False)

        out_metrics[task] = {"r2": float(r2), "mae": float(mae), "rmse": rmse,
                             "spearman": spr, "n_test": int(len(test_df))}

    with open(os.path.join(RESULT_DIR, "old_baseline_metrics.json"), "w") as f:
        json.dump(out_metrics, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {os.path.join(RESULT_DIR, 'old_baseline_metrics.json')}")


if __name__ == "__main__":
    main()
