# 最终架构协议 (2026-09-01, seed=11 验证)

## 1. P1-1 sensitivity 测试结果 (ring_flag amplitude ∈ {0,1,5,10})

max_epochs=200, patience=30, MPNN, 仅 dev 5-fold CV, 4 档 amplitude:

| Task | ring_flag=0 | ring_flag=1 | ring_flag=5 | ring_flag=10 | |1-10| | |1-0| | 决定 |
|---|---|---|---|---|---|---|---|
| HOMA | 0.1649±0.0076 | 0.1603±0.0070 | 0.1452±0.0083 | **0.1416±0.0060** | 0.0187 | 0.0046 | 10 显著优于 1 (4×) → learnable projection |
| NICS_1zz | 1.2719±0.0546 | **1.1494±0.0521** | 1.1483±0.0374 | 1.1490±0.0401 | 0.0004 | 0.1225 | 1/5/10 几乎一样 → binary 1 够 |
| MBCO | 0.0262±0.0020 | **0.0234±0.0007** | 0.0232±0.0011 | 0.0234±0.0019 | 0.0000 | 0.0028 | 1/5/10 几乎一样 → binary 1 够 |

ceiling-hit: 全部 0/5 — 没有撞 max_epoch=200 上限，E* 都在 76-197 区间。

## 2. Stage 6 (final_membership) 4-方案对比 (binary + projection)

| Task | base | membership_1 | membership_10 | membership_proj | 推荐 |
|---|---|---|---|---|---|
| HOMA | 0.1649±0.0076 | 0.1603±0.0070 | 0.1416±0.0060 | **0.1419±0.0042** | membership_proj |
| NICS_1zz | 1.2719±0.0546 | **1.1494±0.0521** | 1.1490±0.0401 | 1.1265±0.0672 | membership_1 |
| MBCO | 0.0262±0.0020 | **0.0234±0.0007** | 0.0234±0.0019 | 0.0238±0.0014 | membership_1 |

## 3. 最终决定 (化学定义)

- **HOMA**: binary membership 0/1 (chemically defined) + `nn.Embedding(2, hidden_dim)` projection
  → 既保留化学二元定义, 又通过学习尺度恢复了原 amplitude-10 的收益
- **NICS_1zz**: binary membership 0/1 (chemically defined), 无 projection
- **MBCO**: binary membership 0/1 (chemically defined), 无 projection

正式 5-seed 跑时, HOMA 走 membership_proj; NICS / MBCO 走 membership_1。
不再根据结果随意改模型结构。

## 4. feature_mode 双轨 (正式 metadata 命名)

- `'standard'`                     : 全特征 (含显式芳香性)
- `'explicit_aromaticity_ablated'` : 移除 handcrafted explicit aromaticity flags/counts (n_aromatic_rings, ring_n_aromatic)
- metadata 明确写: "standard RDKit fingerprints retain aromaticity perception"
- fingerprint bits 本身 (MACCS / Morgan) 仍依赖 RDKit aromaticity perception, 仅移除了 handcrafted explicit 列

## 5. E*=median 训练协议 (锁定)

1. 固定 80/20 holdout, split_seed=2026, canonical SMILES 分组, 无分子泄漏
2. 80% development 内 5-fold Group CV (group-aware)
3. 每折: 早停 (val MAE), 记录 best_epoch, 取 median → E*
4. 完整 dev 上训练 E* epochs, 固定 test 上评估 (test 绝不上决策)
5. max_epochs=200, patience=30, ceiling-hit 全部 0/5
6. 5 seeds 只改 model init (torch/numpy/random seeds), 不改 holdout/CV folds
7. selection_rule: winner = min(cv_mae), final test 仅用于 reporting
