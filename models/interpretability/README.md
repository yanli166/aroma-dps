# Fig.3f — 可解释性实验模型权重

- **面板归属**：Fig.3f（可解释性 / 归因分析）
- **来源**：`/home/ubuntu/aroma-dps-code/0908-end-code/models/`
- **迁移日期**：2026-09-08
- **文件数**：4 个 `.pt`（源目录全部 `.pt` 即此 4 个；均 <5MB，无排除项）。`diff -r` 已验证与源逐字节一致。

> ⚠ **本仓库不承载最终发布权重。** 这些 `.pt` 入仓只为 **provenance / 复现 Fig.3f**。**最终权重应走 Zenodo / GitHub Release**，在正文仓库内只保留 DOI + 文件名 + sha256 链接；`.gitignore` 若排除 `*.pt` 亦属正常。发布前请把本表每一项替换为 Release 资产地址。

---

## 1. 权重清单

| 文件 | 大小 | 实验线 | 训练方案 | 训练目标 | 早停依据 | best_epoch | 产出脚本 |
|---|---|---|---|---|---|---|---|
| `HOMA_merged_best.pt` | 1.84 MB | 线 1 merged | collet a/b(4633) + lunci10(2122) = 6755 训练；collet dev 内 87.5/12.5 group split 出 662 验证；test = collet a/b holdout 1335 | HOMA | collet dev 子验证集 MAE | 147 | `../experiments/fig3_model/interpretability/scripts/train_merged.py --task HOMA` |
| `HOMA_l10val_best.pt` | 1.84 MB | 线 2 l10val | collet a/b(6630) + lunci10 80% 分子(1686) = 8316 训练；验证 = lunci10 20% 留出 436 条 / 272 分子 | HOMA | **l10 验证集 MAE** | 146 | `train_homa_l10val.py` |
| `NICS_1zz_l10val_best.pt` | 1.78 MB | 线 2 l10val | collet a/b(6592) + lunci10 80%(1687) = 8279 训练；验证 = **同一组** 272 分子 / 436 条 | NICS(1)zz（l10 列名 `NICS_ZZ`） | **l10 验证集 MAE** | 182 | `train_task_l10val.py --task NICS_1zz` |
| `MBCO_l10val_best.pt` | 1.78 MB | 线 2 l10val | collet a/b(6699) + lunci10 80%(1687) = 8386 训练；验证 = **同一组** 272 分子 / 436 条 | MBCO | **l10 验证集 MAE** | 78 | `train_task_l10val.py --task MBCO` |

## 2. 共享超参与架构配置

四个 checkpoint 的 `config` 字段已实测读出，完全一致（除 `use_projection`）：

| 项 | 值 |
|---|---|
| `seed`（checkpoint 内 `seed` 字段） | **11**（四个权重全部实测 = 11） |
| `hidden_dim` | **128** |
| `n_conv`（脚本 `PARAMS` 中名为 `n_conv_layers`） | **3** |
| `n_hidden`（脚本 `PARAMS` 中名为 `n_hidden_layers`） | 2 |
| `p_dropout` | 0.2 |
| `node_vec_len` | 60 |
| `MAX_ATOMS` | 75 |
| `ring_flag_value` | **1**（二值环成员标记；与 Stage 6 最终方案一致） |
| `use_projection` | **HOMA 两个 = True**（`nn.Embedding` 可学习投影）；**NICS_1zz / MBCO = False** |
| `split_seed` | 2026（仅 `HOMA_merged_best.pt` 的 checkpoint 内保存该字段；l10val 三个未写入） |
| 优化器 / 调度 | Adam，lr=1e-3，weight_decay=1e-5；ReduceLROnPlateau(factor=0.5, patience=10) |
| batch_size / n_epochs / early-stop patience | 64 / 200 / 30 |
| 架构 | MPNN（GRU-style message passing）+ `fixed_avg` 读出；**无原生注意力**，故 Fig.3f 的归因图基于梯度 saliency（\|x·∇x\|）与 Integrated Gradients |

`state_dict` 张量数：HOMA 两个 = **63**（多出的 1 个即 projection embedding）；NICS/MBCO = **62**。可作为加载时的快速一致性检查。
checkpoint 顶层键：`{'state_dict','config','seed'}`（`HOMA_merged_best.pt` 另含 `'split_seed'`）。

## 3. 缺失的 merged 权重（重要，勿误判为迁移遗漏）

**`NICS_1zz_merged_best.pt` 与 `MBCO_merged_best.pt` 不存在**，源目录中即无此二文件。merged 方案只跑完 HOMA；`scripts/saliency_merged.py` 却硬编码引用这两个文件名，故该脚本独立运行必然失败。
详见 `../experiments/fig3_model/interpretability/pending/README.md`。补训命令：

```bash
python3 train_merged.py --task NICS_1zz --gpu <N>
python3 train_merged.py --task MBCO     --gpu <N>
```

## 4. 已知 provenance 瑕疵

`HOMA_l10val_best.pt` 本身是正常训练产物，但**由它派生的 HOMA l10val 指标 / 预测 CSV 并非训练脚本直接写出** —— `train_homa_l10val.py` 在保存预测阶段抛 `KeyError: 'atom_on_ring'` 崩溃，产物由抢救脚本 `resave_homa_l10val_preds.py` 复用本权重二次推理补写。权重未受影响（崩溃发生在 `[saved]` 之后的 CSV 写出阶段），但引用 HOMA l10val 数字时请留意该链路。详见 `../experiments/fig3_model/interpretability/scripts/README.md` §2 问题①。

## 5. 加载方式

模型构建依赖仓库外模块 `model_arch.build_model`（来自 `/home/ubuntu/aroma-dps-code/best_model_package/`），**尚未纳入本仓库**，属待相对化项：

```python
import torch, sys
sys.path.insert(0, "/home/ubuntu/aroma-dps-code/best_model_package")   # 待相对化
from model_arch import build_model

ckpt = torch.load("HOMA_l10val_best.pt", map_location="cpu", weights_only=False)
cfg = ckpt["config"]                       # 键名见 §2
model = build_model(use_projection=cfg["use_projection"],
                    node_vec_len=cfg["node_vec_len"],
                    hidden_dim=cfg["hidden_dim"],
                    n_conv=cfg["n_conv"],
                    n_hidden=cfg["n_hidden"],
                    p_dropout=cfg["p_dropout"],
                    ring_flag_value=cfg["ring_flag_value"]).to("cpu")
model.load_state_dict(ckpt["state_dict"]); model.eval()
```

> 以上调用形式与归档脚本 `resave_homa_l10val_preds.py:30-34` 完全一致（注意 config 里的键是 `n_conv` / `n_hidden`，而**不是**训练脚本 `PARAMS` 字典中的 `n_conv_layers` / `n_hidden_layers`）。
> 需 `torch_env` conda 环境（`/home/ubuntu/apps/anaconda3/envs/torch_env`）；系统默认 `python3` 无 torch。
