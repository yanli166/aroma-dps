# -*- coding: utf-8 -*-
"""抢救脚本: 复用已训练的 HOMA_l10val_best.pt 生成验证集预测 CSV (无需重训)"""
import os
import sys
import json
import ast
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

sys.path.insert(0, "/home/ubuntu/aroma-dps-code/best_model_package")
from graph_utils import build_graph
from model_arch import build_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR, MODEL_DIR = os.path.join(ROOT, "data"), os.path.join(ROOT, "models")
RESULT_DIR, PRED_DIR = os.path.join(ROOT, "results"), os.path.join(ROOT, "predictions")
NODE_VEC_LEN, MAX_ATOMS, BS = 60, 75, 64


def parse_aor(x):
    return ast.literal_eval(x) if isinstance(x, str) else list(x)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(os.path.join(MODEL_DIR, "HOMA_l10val_best.pt"), map_location="cpu")
    cfg = ckpt["config"]
    model = build_model(use_projection=cfg["use_projection"], node_vec_len=cfg["node_vec_len"],
                        hidden_dim=cfg["hidden_dim"], n_conv=cfg["n_conv"],
                        n_hidden=cfg["n_hidden"], p_dropout=cfg["p_dropout"],
                        ring_flag_value=cfg["ring_flag_value"]).to(device)
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    df = pd.read_csv(os.path.join(DATA_DIR, "homa_l10val.csv"))
    val_df = df[df["split"] == "l10_val"].reset_index(drop=True)
    val_df["atom_on_ring"] = val_df["atom_on_ring"].apply(parse_aor)
    y_va = val_df["y"].astype(float).values

    preds = []
    for i in range(0, len(val_df), BS):
        nm, adj, ri = [], [], []
        for smi, aor in zip(val_df["smiles"].iloc[i:i + BS],
                            val_df["atom_on_ring"].iloc[i:i + BS]):
            g = build_graph(smi, aor, NODE_VEC_LEN, MAX_ATOMS, ring_flag_value=1)
            nm.append(g["node_mat"]); adj.append(g["adj_mat"]); ri.append(g["ring_indices"])
        nm = torch.tensor(np.array(nm, dtype=np.float32), device=device)
        adj = torch.tensor(np.array(adj, dtype=np.float32), device=device)
        ri = torch.tensor(np.array(ri, dtype=np.int64), device=device)
        with torch.no_grad():
            preds.append(model(nm, adj, ri).reshape(-1).cpu().numpy())
    pv = np.concatenate(preds)

    pred_df = val_df[["smiles", "atom_on_ring", "mol_id"]].copy()
    pred_df["y_true"] = y_va
    pred_df["y_pred"] = pv
    # 元数据拼接
    l10m = pd.read_csv("/home/ubuntu/aroma-dps-code/lunci10/lunci10_unified.csv")
    l10m = l10m[l10m["HOMA"].notna()].reset_index(drop=True)
    recs = [{"smiles": r["smiles"], "atom_on_ring": parse_aor(r["ring_atoms"]),
             "ring_name": r["ring_name"], "sub_name": r["sub_name"],
             "sub_type": r["sub_type"], "ring_pos": r["ring_pos"]}
            for _, r in l10m.iterrows()]
    lmap = {(str(r["smiles"]), str(sorted(r["atom_on_ring"]))): r for r in recs}
    for col in ["ring_name", "sub_name", "sub_type", "ring_pos"]:
        pred_df[col] = pred_df.apply(
            lambda r: lmap.get((str(r["smiles"]), str(sorted(r["atom_on_ring"]))),
                               {}).get(col, ""), axis=1)
    pred_df.to_csv(os.path.join(PRED_DIR, "HOMA_l10val_predictions.csv"), index=False)
    print(f"[saved] {os.path.join(PRED_DIR, 'HOMA_l10val_predictions.csv')}  n={len(pred_df)}")

    g = pred_df.groupby("ring_name").apply(
        lambda s: pd.Series({"n": len(s), "mae": mean_absolute_error(s["y_true"], s["y_pred"]),
                             "r2": r2_score(s["y_true"], s["y_pred"]),
                             "sub_n": s["sub_name"].nunique()}), include_groups=False)
    g.to_csv(os.path.join(RESULT_DIR, "HOMA_l10val_per_ringfamily.csv"))
    print(f"[saved] {os.path.join(RESULT_DIR, 'HOMA_l10val_per_ringfamily.csv')}")
    print(g.to_string())


if __name__ == "__main__":
    main()
