# 0831 JACS 协议重构 —— 子代理执行规格 (Stage 1/2/3/4)

工作区: `/home/ubuntu/aroma-dps-code/0831-end-code`
环境: `source /home/ubuntu/apps/anaconda3/etc/profile.d/conda.sh && conda activate torch_env`
数据根: `/home/ubuntu/data_90/alldata_in_3090/model1` (graphs.py 在那)

## 已完成的共享层 (不要动, 直接 import)
- `common/protocol.py`
  - `SPLIT_SEED=2026`, `MODEL_SEEDS=[11,22,33,44,55]`, `N_FOLDS=5`, `TEST_SIZE_RATIO=0.20`
  - `get_final_splits(n_samples, groups, split_seed=2026, persist=SPLITS_DIR)`
    → 返回 `{train_idx(80% dev), test_idx(20%), folds:[(tr,va),...], meta}` (索引面向全量 0..n-1)
  - `make_group_ids(smiles_list, use_inchikey=False)` → canonical SMILES 分组
  - `assert_no_leak(idxa,idxb,groups,tag)`, `completeness_check(actual_tuples, expected_tuples, label)`
- `common/train_eval.py` 已改: `.squeeze(-1)`→`.reshape(-1)` (batch=1 修复)；选模型/early-stop/`scheduler` 均按 **val-MAE** (反向传播仍 MSE)；训练函数给 `model.best_epoch`
- `common/constants.py` 新增 `RESULTS_V2_DIR`, `SPLITS_DIR`

## 协议铁律 (务必遵守)
1. **数据划分**: 用 `common.protocol.get_final_splits` 一次性生成固定 80/20 holdout + dev 内 5 折。
   **model seed 绝不进入 split**。同一 task 下不同 model seed 的 test 集必须逐字节相同。
2. **分组**: `groups = make_group_ids(df['smiles'], use_inchikey=False)` (canonical SMILES)。
   启动时调用 `assert_no_leak` 验证 dev/test 与每折 train/val 无分子重叠。
3. **模型选择只看 CV / info (val-MAE)**, 用 `cv_mae` 选 winner; **final test 绝不用于选择**。
4. **MAE 为主指标**。输出至少三份: `cv_results.csv`, `final_test_results.csv`, `per_seed_results.csv`。
   per_seed 每行含: `seed,task,model,config,n,cv_mae,test_mae,test_rmse,test_r2,run_status,error_message`。
5. **完整性**: 每组全部 run 完成后调用 `completeness_check` 校验理论组合无缺失; 有缺失须 `raise`。
6. 结果写入 `results_v2/<stage>/`，**不要写进** `results/` 或覆盖旧结果。

## Dry-run 配置 (本阶段只跑)
- 只跑 **1 个 model seed = 11**
- 会显著**降低 epoch**: CV/final 用 `n_epochs≈30, patience≈8`; Stage3 预训练 `n_epochs≈10`、微调 `n_epochs≈25`
- 若个别脚本默认 epoch 无法传参, 请在本任务内把默认改成上述小值
- 三个 task (HOMA / NICS_1zz / MBCO) 都要跑, 证明流程完整

## 各 Stage 职责与文档映射 (参考思路2.txt)
- Stage1 GNN: Fig.3a 插值基准。dry-run 至少 MPNN; 时间紧可只跑 1 backbone。
  收敛到: 每个 (task) 输出 `cv_mae`, `final test mae`.
- Stage1 traditional ML (P1 #9): 对 continuous 特征做 **train-only** scaling (SVM/KRR/MLP),
  树模型(RF/ET/XGB/LGB/CatBoost)不强制。dry-run 跑 1-2 个模型即可。
- Stage2 (P0 #7): 2×2 因子 `base/membership/learnable_readout/joint`, 只用 CV 选 winner, 不硬编码。
- Stage3 (P0 #14): 用 **Membership** backbone (`ring_flag=True, readout=fixed_avg`,` ring_value=10`) 的 encoder,
  比较 `direct_supervised / random_mask_pretrain / ring_mask_pretrain`。
- Stage4 (P0 #8): 只比较 `base(ring_flag=0,fixed_avg)` vs `ring_conditioned/membership(ring_flag=10,fixed_avg)`
  在 4 backbones (MPNN/GIN/GAT/DMPNN) 上; 报告 `ΔMAE`。**不要用 Joint 当对照**。

## 你的输出 (return)
1. 改了哪些文件 (路径) + 每处改什么
2. dry-run 跑出的每个 stage/task 的 `cv_mae` 与 `test_mae` 汇总表
3. 完整性检查是否通过; 是否有缺失/报错 (run_status)
4. 是否满足铁律 (固定test / 无泄漏 / MAE选模型 / test不进选择)