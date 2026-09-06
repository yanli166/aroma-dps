"""
Method 2: 电子效应单调性约束 (损失函数)

思路:
    在 MSE Loss 基础上加入单调性正则项。
    对于同一 batch 中的分子对 (i, j): 若分子 i 的 Σσ (取代基电子效应总强度) 大于
    分子 j 的 Σσ, 说明 i 更吸电子, 其 HOMA (芳香性) 应更低, 即 pred_i < pred_j。
    违反该顺序时施加惩罚: max(0, pred_i - pred_j + margin)。

    loss = mse_loss(preds, targets) + lambda * mean(penalty)

核心组件:
    - compute_sigma_sums(data): 数据预处理, 为每个分子计算 Σσ (基于 get_molecular_hammett_vector)
    - MonotonicityLoss: MSE + 单调性正则的联合损失
    - build_model: 返回标准 GNN/MPNN 模型实例 (本方法的创新在损失函数, 模型不变)
    - build_loss: 返回 MonotonicityLoss 实例
"""
import os
import sys


# --- Auto path bootstrap (do not remove) ---
import os as _os
_THIS_FILE = _os.path.abspath(__file__)
_d = _os.path.dirname(_THIS_FILE)
while not _os.path.exists(_os.path.join(_d, 'unified_models')) and _d != '/':
    _d = _os.path.dirname(_d)
_PROJ_ROOT = _d
# --- End auto path bootstrap ---

CODE_END_ROOT = '_PROJ_ROOT + "/code_end"'
ORIG_MODELS_ROOT = '_PROJ_ROOT + "/unified_models"'
for _p in (CODE_END_ROOT, ORIG_MODELS_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn as nn

from layer4_substituent.code.hammett_constants import get_molecular_hammett_vector
from unified_models.gnn.model import GNNModel
from unified_models.mpnn.model import MPNNModel


# ============== 数据预处理 ==============
def compute_sigma_sums(data, sigma_mode='both', device='cpu'):
    """为每个分子计算 Σσ (取代基电子效应总强度)

    使用 get_molecular_hammett_vector 获取分子级 Hammett 向量:
        [sum_sigma_m, sum_sigma_p, mean_sigma_m, mean_sigma_p,
         n_substituents, max_sigma_m, min_sigma_m, max_sigma_p, min_sigma_p]

    Args:
        data: load_adj_format 返回的字典, 需包含 'smiles'
        sigma_mode: Σσ 的计算方式
            - 'meta':  sum_sigma_m  (间位 σ 之和)
            - 'para':  sum_sigma_p  (对位 σ 之和)
            - 'both':  sum_sigma_m + sum_sigma_p  (间+对 σ 之和, 默认)
        device: 输出张量所在设备

    Returns:
        torch.Tensor: shape (N,), 每个分子的 Σσ
        同时向 data 中写入 'sigma_sums' 字段并返回更新后的 data
    """
    from rdkit import Chem

    smiles = data['smiles']
    n = len(smiles)
    sigma_sums = np.zeros(n, dtype=np.float32)

    for i in range(n):
        mol = Chem.MolFromSmiles(smiles[i])
        vec = get_molecular_hammett_vector(mol)  # (9,)
        sum_m = float(vec[0])  # sum_sigma_m
        sum_p = float(vec[1])  # sum_sigma_p
        if sigma_mode == 'meta':
            sigma_sums[i] = sum_m
        elif sigma_mode == 'para':
            sigma_sums[i] = sum_p
        elif sigma_mode == 'both':
            sigma_sums[i] = sum_m + sum_p
        else:
            raise ValueError(f"Unknown sigma_mode: {sigma_mode}")

    sigma_tensor = torch.tensor(sigma_sums, dtype=torch.float32, device=device)
    new_data = dict(data)
    new_data['sigma_sums'] = sigma_tensor
    new_data['sigma_mode'] = sigma_mode
    return new_data, sigma_tensor


# ============== 损失函数 ==============
class MonotonicityLoss(nn.Module):
    """MSE + 单调性正则的联合损失

    loss = mse_loss(preds, targets) + lambda_mono * monotonicity_penalty

    单调性约束:
        对于 batch 中所有分子对 (i, j):
            若 sigma_i > sigma_j (i 更吸电子), 则要求 pred_i < pred_j
            违反时惩罚 = max(0, pred_i - pred_j + margin)
        penalty = sum(惩罚) / (有效对数)

    Args:
        lambda_mono: 单调性正则项权重
        margin: 允许的松弛间隔 (margin 越大, 约束越严格)
        reduction: MSE 的归约方式 ('mean' 或 'sum')
    """

    def __init__(self, lambda_mono=0.1, margin=0.0, reduction='mean'):
        super().__init__()
        self.lambda_mono = float(lambda_mono)
        self.margin = float(margin)
        self.mse = nn.MSELoss(reduction=reduction)

    @staticmethod
    def _pairwise_penalty(preds, sigma_sums, margin):
        """计算所有分子对的成对单调性惩罚

        Args:
            preds: (B,) 模型预测
            sigma_sums: (B,) 每个分子的 Σσ
            margin: 松弛间隔

        Returns:
            scalar tensor: 平均惩罚 (已归一化)
        """
        # diff_sigma[i,j] = sigma_i - sigma_j; > 0 表示 i 更吸电子
        diff_sigma = sigma_sums.unsqueeze(1) - sigma_sums.unsqueeze(0)  # (B, B)
        # diff_pred[i,j] = pred_i - pred_j
        diff_pred = preds.unsqueeze(1) - preds.unsqueeze(0)  # (B, B)

        # 仅对 sigma_i > sigma_j 的对施加约束 (严格大于, 避免对称重复)
        mask = (diff_sigma > 0).float()

        # hinge: max(0, pred_i - pred_j + margin)
        penalty = torch.clamp(diff_pred + margin, min=0.0) * mask

        n_pairs = mask.sum()
        if float(n_pairs.item()) == 0.0:
            return preds.new_zeros(())
        return penalty.sum() / (n_pairs + 1e-8)

    def forward(self, preds, targets, sigma_sums):
        """计算联合损失

        Args:
            preds: (B,) 模型预测
            targets: (B,) 真实值
            sigma_sums: (B,) 每个分子的 Σσ (由 compute_sigma_sums 预计算)

        Returns:
            total_loss, (mse_loss, mono_penalty) 的元组 (便于记录)
        """
        preds = preds.view(-1)
        targets = targets.view(-1)
        sigma_sums = sigma_sums.view(-1).to(preds.device)

        mse_loss = self.mse(preds, targets)
        mono_penalty = self._pairwise_penalty(preds, sigma_sums, self.margin)
        total = mse_loss + self.lambda_mono * mono_penalty
        return total, (mse_loss, mono_penalty)


# ============== 构建模型 / 损失 ==============
def build_model(base='gnn', node_vec_len=60, hidden_dim=128, n_conv=3,
                n_hidden=2, n_outputs=1, p_dropout=0.2, mode='label', **kwargs):
    """返回标准 GNN/MPNN 模型 (Method 2 的创新在损失函数, 模型保持原样)

    Args:
        base: 'gnn' 或 'mpnn'
        其余参数同 GNNModel/MPNNModel
    """
    base = base.lower()
    if base == 'gnn':
        return GNNModel(node_vec_len, hidden_dim, n_conv, n_hidden, n_outputs,
                        p_dropout, mode)
    elif base == 'mpnn':
        return MPNNModel(node_vec_len, hidden_dim, n_conv, n_hidden, n_outputs,
                         p_dropout, mode)
    else:
        raise ValueError(f"Unsupported base model: {base} (choose 'gnn' or 'mpnn')")


def build_loss(lambda_mono=0.1, margin=0.0, reduction='mean'):
    """返回 MonotonicityLoss 实例"""
    return MonotonicityLoss(lambda_mono=lambda_mono, margin=margin, reduction=reduction)


if __name__ == '__main__':
    # 自测: 损失函数数值行为
    torch.manual_seed(0)
    preds = torch.tensor([0.9, 0.2, 0.5])       # 预测 (违反单调性的方向)
    targets = torch.tensor([0.3, 0.4, 0.5])
    sigma = torch.tensor([1.0, -1.0, 0.0])       # sigma[0] > sigma[1], 要求 pred[0] < pred[1], 但实际 0.9 > 0.2 → 应有惩罚
    loss_fn = MonotonicityLoss(lambda_mono=0.5, margin=0.0)
    total, (mse, mono) = loss_fn(preds, targets, sigma)
    print(f"[m2] total={total.item():.4f}, mse={mse.item():.4f}, mono_penalty={mono.item():.4f}")
    assert mono.item() > 0, "Monotonicity penalty should be > 0 for violated pairs"
    # 满足单调性的情况应无惩罚
    preds_ok = torch.tensor([0.1, 0.6, 0.4])
    _, (mse_ok, mono_ok) = loss_fn(preds_ok, targets, sigma)
    assert mono_ok.item() == 0.0, "No penalty expected when monotonicity holds"
    print("[m2] OK")
