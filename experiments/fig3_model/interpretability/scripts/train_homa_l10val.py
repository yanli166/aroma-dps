# -*- coding: utf-8 -*-
"""
Step B: HOMA 重训 — 训练=collet a/b + l10 80%, 验证/测试=l10 20% (分子级留出)

Stage 6 最优配置 (proj=True), seed/超参与 best_model_package 一致。
early stopping 直接基于 l10 验证集 (用户确认: 用 l10 作为验证集以聚焦取代基信息)。

用法: CUDA_VISIBLE_DEVICES=2 python3 train_homa_l10val.py --gpu 0
输出: models/HOMA_l10val_best.pt
      results/HOMA_l10val_metrics.json
      predictions/HOMA_l10val_predictions.csv
"""
import os
import sys
import json
import ast
import random
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
MODEL_DIR = os.path.join(ROOT, "models")
RESULT_DIR = os.path.join(ROOT, "results")
PRED_DIR = os.path.join(ROOT, "predictions")

NODE_VEC_LEN, MAX_ATOMS = 60, 75
MODEL_SEED = 11
PARAMS = {"hidden_dim": 128, "n_conv_layers": 3, "n_hidden_layers": 2,
          "learning_rate": 0.001, "p_dropout": 0.2, "batch_size": 64,
          "weight_decay": 1e-5, "n_epochs": 200, "patience": 30}
PROJ = True  # HOMA 用 projection


def parse_aor(x):
    return ast.literal_eval(x) if isinstance(x, str) else list(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=2)
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[train_homa_l10val] device={device} seed={MODEL_SEED}")

    df = pd.read_csv(os.path.join(DATA_DIR, "homa_l10val.csv"))
    df["atom_on_ring"] = df["atom_on_ring"].apply(parse_aor)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "l10_val"].reset_index(drop=True)
    print(f"train={len(train_df)} (collet {int((train_df.source=='collet').sum())} + "
          f"l10 {int((train_df.source=='l10').sum())}), l10_val={len(val_df)}")

    def build_rows(sub):
        nm, adj, ri = [], [], []
        for smi, aor in zip(sub["smiles"], sub["atom_on_ring"]):
            g = build_graph(smi, aor, NODE_VEC_LEN, MAX_ATOMS, ring_flag_value=1)
            nm.append(g["node_mat"]); adj.append(g["adj_mat"]); ri.append(g["ring_indices"])
        return {"node_mats": torch.tensor(np.array(nm, dtype=np.float32)),
                "adj_mats": torch.tensor(np.array(adj, dtype=np.float32)),
                "ring_indices": torch.tensor(np.array(ri, dtype=np.int64))}

    print("  building graphs ...")
    tr_data = build_rows(train_df)
    va_data = build_rows(val_df)
    y_tr = train_df["y"].astype(float).values
    y_va = val_df["y"].astype(float).values

    random.seed(MODEL_SEED); np.random.seed(MODEL_SEED); torch.manual_seed(MODEL_SEED)
    torch.cuda.manual_seed_all(MODEL_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = build_model(use_projection=PROJ, node_vec_len=NODE_VEC_LEN,
                        hidden_dim=PARAMS["hidden_dim"], n_conv=PARAMS["n_conv_layers"],
                        n_hidden=PARAMS["n_hidden_layers"], p_dropout=PARAMS["p_dropout"],
                        ring_flag_value=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=PARAMS["learning_rate"],
                                 weight_decay=PARAMS["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                           factor=0.5, patience=10)
    loss_fn = torch.nn.MSELoss()
    bs = PARAMS["batch_size"]

    nm = tr_data["node_mats"].to(device); adj = tr_data["adj_mats"].to(device)
    ri = tr_data["ring_indices"].to(device)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=device)
    va_nm = va_data["node_mats"].to(device); va_adj = va_data["adj_mats"].to(device)
    va_ri = va_data["ring_indices"].to(device)
    yv = torch.tensor(y_va, dtype=torch.float32, device=device)
    n_tr = len(train_df)
    n_va = len(val_df)

    best_vmae = float("inf"); best_state = None; best_epoch = 0; bad = 0
    for ep in range(1, PARAMS["n_epochs"] + 1):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        for i in range(0, n_tr, bs):
            bi = perm[i:i + bs]
            optimizer.zero_grad(set_to_none=True)
            pred = model(nm[bi], adj[bi], ri[bi]).reshape(-1)
            loss = loss_fn(pred, yt[bi])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        pv_l, tv_l = [], []
        with torch.no_grad():
            for i in range(0, n_va, bs):
                bi = torch.arange(i, min(i + bs, n_va), device=device)
                pv_l.append(model(va_nm[bi], va_adj[bi], va_ri[bi]).reshape(-1))
                tv_l.append(yv[bi])
        pv = torch.cat(pv_l); tv = torch.cat(tv_l)
        vmae = (pv - tv).abs().mean().item()
        scheduler.step(vmae)
        if vmae < best_vmae - 1e-6:
            best_vmae = vmae; best_epoch = ep; bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PARAMS["patience"]:
                break
        if ep % 20 == 0 or ep == 1:
            print(f"    ep={ep} val_mae(l10)={vmae:.4f} best={best_vmae:.4f}@{best_epoch}")

    model.load_state_dict(best_state)
    model.eval()

    def predict(data):
        dnm = data["node_mats"].to(device); dadj = data["adj_mats"].to(device)
        dri = data["ring_indices"].to(device)
        outs = []
        with torch.no_grad():
            for i in range(0, len(dri), bs):
                bi = torch.arange(i, min(i + bs, len(dri)), device=device)
                outs.append(model(dnm[bi], dadj[bi], dri[bi]).reshape(-1).cpu())
        return torch.cat(outs).numpy()

    pv_va = predict(va_data)
    r2 = r2_score(y_va, pv_va)
    mae = mean_absolute_error(y_va, pv_va)
    rmse = float(np.sqrt(mean_squared_error(y_va, pv_va)))
    spr = float(spearmanr(y_va, pv_va).correlation)
    print(f"\n[l10_val] n={n_va} R2={r2:.4f} MAE={mae:.4f} RMSE={rmse:.4f} Spearman={spr:.4f}")

    # 保存
    model_path = os.path.join(MODEL_DIR, "HOMA_l10val_best.pt")
    torch.save({"state_dict": model.state_dict(),
                "config": {"use_projection": PROJ, "node_vec_len": NODE_VEC_LEN,
                           "hidden_dim": PARAMS["hidden_dim"], "n_conv": PARAMS["n_conv_layers"],
                           "n_hidden": PARAMS["n_hidden_layers"], "p_dropout": PARAMS["p_dropout"],
                           "ring_flag_value": 1},
                "seed": MODEL_SEED}, model_path)

    metrics = {"task": "HOMA", "val": "l10 20% (molecule-grouped)",
               "best_epoch": int(best_epoch), "n_train": int(n_tr),
               "n_collet_train": int((train_df.source == "collet").sum()),
               "n_l10_train": int((train_df.source == "l10").sum()),
               "n_val": int(n_va), "n_val_molecules": int(val_df["mol_id"].nunique()),
               "val_metrics": {"r2": float(r2), "mae": float(mae),
                               "rmse": rmse, "spearman": spr}}
    with open(os.path.join(RESULT_DIR, "HOMA_l10val_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    pred_df = val_df[["smiles", "atom_on_ring", "mol_id"]].copy()
    pred_df["y_true"] = y_va
    pred_df["y_pred"] = pv_va
    # 附加 l10 元数据 (ring_name / sub_name / sub_type)
    l10m = pd.read_csv("/home/ubuntu/aroma-dps-code/lunci10/lunci10_unified.csv")
    l10m = l10m[l10m["HOMA"].notna()].reset_index(drop=True)
    l10m["atom_on_ring"] = l10m["ring_atoms"].apply(parse_aor)
    recs = []
    for _, r in l10m.iterrows():
        recs.append({"smiles": r["smiles"], "atom_on_ring": r["atom_on_ring"],
                     "ring_name": r["ring_name"], "sub_name": r["sub_name"],
                     "sub_type": r["sub_type"], "ring_pos": r["ring_pos"]})
    l10_meta = pd.DataFrame(recs)
    key = lambda x: (str(x["smiles"]), str(sorted(x["atom_on_ring"])))
    lmap = {key(r): r for _, r in l10_meta.iterrows()}
    pred_df["ring_name"] = pred_df.apply(lambda r: lmap.get((str(r["smiles"]), str(sorted(r["atom_on_ring"]))), {}).get("ring_name", ""), axis=1)
    pred_df["sub_name"] = pred_df.apply(lambda r: lmap.get((str(r["smiles"]), str(sorted(r["atom_on_ring"]))), {}).get("sub_name", ""), axis=1)
    pred_df["sub_type"] = pred_df.apply(lambda r: lmap.get((str(r["smiles"]), str(sorted(r["atom_on_ring"]))), {}).get("sub_type", ""), axis=1)
    pred_df["ring_pos"] = pred_df.apply(lambda r: lmap.get((str(r["smiles"]), str(sorted(r["atom_on_ring"]))), {}).get("ring_pos", ""), axis=1)
    pred_df.to_csv(os.path.join(PRED_DIR, "HOMA_l10val_predictions.csv"), index=False)

    # 按环族汇总
    g = pred_df.groupby("ring_name").apply(
        lambda s: pd.Series({"n": len(s), "mae": mean_absolute_error(s["y_true"], s["y_pred"]),
                             "r2": r2_score(s["y_true"], s["y_pred"]),
                             "sub_n": s["sub_name"].nunique()}), include_groups=False)
    g.to_csv(os.path.join(RESULT_DIR, "HOMA_l10val_per_ringfamily.csv"))
    print(f"\n[saved] {model_path}")
    print(f"[saved] {os.path.join(RESULT_DIR, 'HOMA_l10val_metrics.json')}")
    print(f"[saved] {os.path.join(RESULT_DIR, 'HOMA_l10val_per_ringfamily.csv')}")
    print(f"[saved] {os.path.join(PRED_DIR, 'HOMA_l10val_predictions.csv')}")


if __name__ == "__main__":
    main()
