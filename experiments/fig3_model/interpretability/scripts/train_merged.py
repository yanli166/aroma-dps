# -*- coding: utf-8 -*-
"""
Step 2: 在 base(a/b)+lunci10 合并数据集上用 Stage6 最终配置重训

协议 (完全复用 train_final.py / best_model_package):
  - 测试集: prepare_merged_data.py 固定的 collet a/b test (源于 full-collet 80/20 holdout seed=2026)
  - collet dev 内 87.5/12.5 group split (seed=2026) -> final_train/final_val (early stopping)
  - lunci10 全部并入 final_train
  - 超参/种子与 best_model_package 完全一致 (MODEL_SEED=11, hidden=128, 3conv, lr=1e-3, bs=64,
    patience=30, n_epochs=200, ReduceLROnPlateau)
  - 配置: HOMA proj=True; NICS_1zz/MBCO proj=False; ring_flag_value=1
  - BatchNorm1d 在 size=1 batch 上会出错 -> 保证 bs>1; 使用 squeeze(-1) 而非 squeeze()

用法: python3 train_merged.py --task HOMA --gpu 2
输出:
  models/{task}_merged_best.pt
  results/{task}_metrics.json        (val/test: R2, MAE, RMSE, Spearman)
  predictions/{task}_test_predictions.csv
"""
import os
import sys
import json
import ast
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr

sys.path.insert(0, "/home/ubuntu/aroma-dps-code/best_model_package")
from graph_utils import build_graph
from model_arch import build_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")
RESULT_DIR = os.path.join(ROOT, "results")
PRED_DIR = os.path.join(ROOT, "predictions")
for d in (MODEL_DIR, RESULT_DIR, PRED_DIR):
    os.makedirs(d, exist_ok=True)

NODE_VEC_LEN, MAX_ATOMS = 60, 75
SPLIT_SEED = 2026
MODEL_SEED = 11
PARAMS = {
    'hidden_dim': 128, 'n_conv_layers': 3, 'n_hidden_layers': 2,
    'learning_rate': 0.001, 'p_dropout': 0.2, 'batch_size': 64,
    'weight_decay': 1e-5, 'n_epochs': 200, 'patience': 30,
}

TASK_CFG = {
    "HOMA":     {"proj": True,  "csv": "merged_HOMA.csv",     "target": "HOMA"},
    "NICS_1zz": {"proj": False, "csv": "merged_NICS_1zz.csv", "target": "NICS(1)zz"},
    "MBCO":     {"proj": False, "csv": "merged_MBCO.csv",     "target": "MBCO"},
}


def parse_aor(x):
    if isinstance(x, str):
        return ast.literal_eval(x)
    return list(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASK_CFG))
    ap.add_argument("--gpu", type=int, default=2)
    args = ap.parse_args()
    cfg = TASK_CFG[args.task]

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[train_merged] task={args.task} device={device} seed={MODEL_SEED} split_seed={SPLIT_SEED}")

    df = pd.read_csv(os.path.join(DATA_DIR, cfg["csv"]))
    df["atom_on_ring"] = df["atom_on_ring"].apply(parse_aor)
    df = df[df["y"].notna()].reset_index(drop=True)

    dev_df = df[df["split"] == "dev"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    l10_df = df[df["split"] == "l10"].reset_index(drop=True)

    # collet dev -> 87.5/12.5 group split (group = mol_id, 与 train_final 相同)
    groups = dev_df["mol_id"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=SPLIT_SEED)
    tr_rel, va_rel = next(gss.split(np.arange(len(dev_df)), groups=groups))
    c_train = dev_df.iloc[tr_rel]
    c_val = dev_df.iloc[va_rel]
    assert len(set(c_train["mol_id"]) & set(c_val["mol_id"])) == 0
    assert len(set(c_train["mol_id"]) & set(test_df["mol_id"])) == 0
    assert len(set(c_val["mol_id"]) & set(l10_df["mol_id"])) == 0
    assert len(set(test_df["mol_id"]) & set(l10_df["mol_id"])) == 0
    print(f"  collet_dev={len(dev_df)} c_train={len(c_train)} c_val={len(c_val)} "
          f"l10={len(l10_df)} test={len(test_df)}")

    train_df = pd.concat([c_train, l10_df], ignore_index=True)
    print(f"  final_train={len(train_df)} (collet {len(c_train)} + l10 {len(l10_df)})")

    def build_rows(df):
        node_mats, adj_mats, ring_indices = [], [], []
        for smi, aor in zip(df["smiles"], df["atom_on_ring"]):
            g = build_graph(smi, aor, NODE_VEC_LEN, MAX_ATOMS, ring_flag_value=1)
            node_mats.append(g["node_mat"])
            adj_mats.append(g["adj_mat"])
            ring_indices.append(g["ring_indices"])
        return {
            "node_mats": torch.tensor(np.array(node_mats, dtype=np.float32)),
            "adj_mats": torch.tensor(np.array(adj_mats, dtype=np.float32)),
            "ring_indices": torch.tensor(np.array(ring_indices, dtype=np.int64)),
        }

    print("  building graphs ...")
    tr_data = build_rows(train_df)
    va_data = build_rows(c_val)
    te_data = build_rows(test_df)
    y_train = train_df["y"].astype(float).values
    y_val = c_val["y"].astype(float).values
    y_test = test_df["y"].astype(float).values

    # ---- 模型构建 (先设种子再建模型) ----
    torch.manual_seed(MODEL_SEED)
    np.random.seed(MODEL_SEED)
    import random
    random.seed(MODEL_SEED)
    torch.cuda.manual_seed_all(MODEL_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = build_model(use_projection=cfg["proj"], node_vec_len=NODE_VEC_LEN,
                        hidden_dim=PARAMS["hidden_dim"], n_conv=PARAMS["n_conv_layers"],
                        n_hidden=PARAMS["n_hidden_layers"], p_dropout=PARAMS["p_dropout"],
                        ring_flag_value=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=PARAMS["learning_rate"],
                                 weight_decay=PARAMS["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    loss_fn = torch.nn.MSELoss()
    bs = PARAMS["batch_size"]

    node_mats = tr_data["node_mats"].to(device)
    adj_mats = tr_data["adj_mats"].to(device)
    ring_indices = tr_data["ring_indices"].to(device)
    yt = torch.tensor(y_train, dtype=torch.float32, device=device)
    va_node = va_data["node_mats"].to(device)
    va_adj = va_data["adj_mats"].to(device)
    va_ring = va_data["ring_indices"].to(device)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device)

    def forward_all(nm, adj, ri, idx):
        return model(nm[idx], adj[idx], ri[idx]).reshape(-1)

    n_tr = len(train_df)
    tr_t = torch.arange(n_tr, device=device)
    va_t = torch.arange(len(c_val), device=device)

    best_vmae = float("inf")
    best_state = None
    best_epoch = 0
    bad = 0
    for ep in range(1, PARAMS["n_epochs"] + 1):
        model.train()
        perm = tr_t[torch.randperm(n_tr, device=device)]
        for i in range(0, n_tr, bs):
            bi = perm[i:i + bs]
            optimizer.zero_grad(set_to_none=True)
            pred = forward_all(node_mats, adj_mats, ring_indices, bi)
            loss = loss_fn(pred, yt[bi])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for i in range(0, len(c_val), bs):
                bi = va_t[i:i + bs]
                preds.append(forward_all(va_node, va_adj, va_ring, bi))
                truths.append(yv[bi])
        pv = torch.cat(preds)
        tv = torch.cat(truths)
        vmae = (pv - tv).abs().mean().item()
        scheduler.step(vmae)
        if vmae < best_vmae - 1e-6:
            best_vmae = vmae
            best_epoch = ep
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PARAMS["patience"]:
                break
        if ep % 20 == 0 or ep == 1:
            print(f"    ep={ep} val_mae={vmae:.4f} best={best_vmae:.4f}@{best_epoch}")

    model.load_state_dict(best_state)
    model.eval()

    def eval_rows(data, y, name):
        nm = data["node_mats"].to(device)
        adj = data["adj_mats"].to(device)
        ri = data["ring_indices"].to(device)
        preds = []
        with torch.no_grad():
            for i in range(0, len(y), bs):
                bi = torch.arange(i, min(i + bs, len(y)), device=device)
                preds.append(forward_all(nm, adj, ri, bi).cpu())
        pv = torch.cat(preds).numpy()
        r2 = r2_score(y, pv)
        mae = mean_absolute_error(y, pv)
        rmse = float(np.sqrt(mean_squared_error(y, pv)))
        spr = float(spearmanr(y, pv).correlation)
        print(f"  [{name}] n={len(y)} R2={r2:.4f} MAE={mae:.4f} RMSE={rmse:.4f} Spearman={spr:.4f}")
        return pv, {"r2": float(r2), "mae": float(mae), "rmse": rmse, "spearman": spr}

    _, val_met = eval_rows(va_data, y_val, "val")
    te_pred, te_met = eval_rows(te_data, y_test, "test")

    # 保存
    model_path = os.path.join(MODEL_DIR, f"{args.task}_merged_best.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "config": {"use_projection": cfg["proj"], "node_vec_len": NODE_VEC_LEN,
                   "hidden_dim": PARAMS["hidden_dim"], "n_conv": PARAMS["n_conv_layers"],
                   "n_hidden": PARAMS["n_hidden_layers"], "p_dropout": PARAMS["p_dropout"],
                   "ring_flag_value": 1},
        "seed": MODEL_SEED, "split_seed": SPLIT_SEED,
    }, model_path)

    metrics = {
        "task": args.task, "target": cfg["target"],
        "best_epoch": int(best_epoch),
        "n_collet_train": int(len(c_train)), "n_l10_train": int(len(l10_df)),
        "n_val": int(len(c_val)), "n_test": int(len(test_df)),
        "val": val_met, "test": te_met,
    }
    with open(os.path.join(RESULT_DIR, f"{args.task}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    pred_df = test_df[["smiles", "atom_on_ring", "mol_id", "source"]].copy()
    pred_df["y_true"] = y_test
    pred_df["y_pred"] = te_pred
    pred_df.to_csv(os.path.join(PRED_DIR, f"{args.task}_test_predictions.csv"), index=False)
    print(f"[saved] {model_path}")
    print(f"[saved] {os.path.join(RESULT_DIR, args.task + '_metrics.json')}")
    print(f"[saved] {os.path.join(PRED_DIR, args.task + '_test_predictions.csv')}")


if __name__ == "__main__":
    main()
