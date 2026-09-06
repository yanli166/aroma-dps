"""
Stage3 掩码预训练模块 (GraphMAE 风格)

将 Ring Masking 作为独立的学习策略 (预训练) 进行评估。
固定网络结构 (Stage2 最佳配置), 通过掩码原子特征重建任务自监督预训练 encoder。

包含:
  - MaskedAutoencoder: 在 RingConditionedGNN 基础上加 decoder 重建原子特征
  - random_mask: 随机生成掩码 (15% 真实原子)
  - ring_mask: 环结构化掩码 (掩码所有目标环原子 + 10% 随机非环原子)
  - pretrain: 预训练循环 (自监督, 无标签)
"""
import torch
import torch.nn as nn

from models.ring_conditioned_gnn import RingConditionedGNN
from common.train_eval import set_full_seed


class MaskedAutoencoder(nn.Module):
    """掩码自编码器 (GraphMAE 风格)

    encoder = RingConditionedGNN (复用 init_transform + conv_layers 消息传递)
    decoder = MLP: hidden_dim -> node_vec_len (重建原始原子特征)

    forward 流程:
      1. 保存原始 node_mat 作为重建目标
      2. 将掩码原子的特征置零 (防止信息泄露)
      3. encoder 消息传递 (encode 获取节点级特征 node_fea)
      4. decoder 重建原子特征
      5. 返回重建值和原始值 (调用方仅计算掩码位置 loss)
    """

    def __init__(self, backbone, node_vec_len, hidden_dim, n_conv, n_hidden,
                 p_dropout, readout_mode, n_heads=4):
        super().__init__()
        # 复用 RingConditionedGNN 作为 encoder (包含完整消息传递 + readout + MLP head)
        self.encoder = RingConditionedGNN(
            backbone=backbone, node_vec_len=node_vec_len, hidden_dim=hidden_dim,
            n_conv=n_conv, n_hidden=n_hidden, n_outputs=1, p_dropout=p_dropout,
            readout_mode=readout_mode, n_heads=n_heads,
        )
        self.hidden_dim = hidden_dim
        self.node_vec_len = node_vec_len
        # 解码器: 重建原子原始特征 (hidden_dim -> node_vec_len)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_vec_len),
        )

    def forward(self, node_mat, adj_mat, ring_indices, mask):
        """
        Args:
            node_mat:     (batch, n_atoms, node_vec_len) 原始原子特征
            adj_mat:      (batch, n_atoms, n_atoms) 邻接矩阵
            ring_indices: (batch, n_atoms) 目标环原子位置标记 (-1=非环原子)
            mask:         (batch, n_atoms) 1.0=掩码, 0.0=保留
        Returns:
            pred:   (batch, n_atoms, node_vec_len) 重建的原子特征
            target: (batch, n_atoms, node_vec_len) 原始原子特征
        """
        # 1. 保存原始特征作为重建目标
        target = node_mat
        # 2. 将掩码原子的特征置零 (防止信息泄露)
        mask_exp = mask.unsqueeze(-1)  # (batch, n_atoms, 1)
        masked_node_mat = node_mat * (1.0 - mask_exp)
        # 3. encoder 消息传递 (encode 返回 pooled, node_fea; 取 node_fea 节点级表示)
        _, node_fea = self.encoder.encode(masked_node_mat, adj_mat, ring_indices)
        # 4. decoder 重建原子特征
        pred = self.decoder(node_fea)
        # 5. 返回重建值和原始值 (仅掩码位置计算 loss 由调用方处理)
        return pred, target


def _reduce_valid_mask(valid_mask, batch_size, n_atoms, device):
    """将 valid_mask 规范为 (batch, n_atoms) 2D 张量

    mask_mats 可能为 3D (batch, n_atoms, node_vec_len), 需沿特征维降维。
    """
    if valid_mask is None:
        return torch.ones(batch_size, n_atoms, device=device)
    valid_mask = valid_mask.float()
    if valid_mask.dim() == 3:
        # (batch, n_atoms, node_vec_len) -> (batch, n_atoms)
        valid_mask = (valid_mask.abs().sum(dim=-1) > 0).float()
    return valid_mask


def random_mask(node_mat, valid_mask=None, mask_ratio=0.15):
    """随机生成掩码 (默认 15% 真实原子)

    Args:
        node_mat:   (batch, n_atoms, node_vec_len)
        valid_mask: (batch, n_atoms) 或 (batch, n_atoms, node_vec_len); None=全部有效
        mask_ratio: 掩码比例
    Returns:
        mask: (batch, n_atoms) 1.0=掩码, 0.0=保留
    """
    batch_size, n_atoms, _ = node_mat.shape
    device = node_mat.device
    valid_mask = _reduce_valid_mask(valid_mask, batch_size, n_atoms, device)
    mask = torch.zeros(batch_size, n_atoms, device=device)
    for b in range(batch_size):
        valid_idx = torch.where(valid_mask[b] > 0)[0]
        n_valid = len(valid_idx)
        if n_valid == 0:
            continue
        n_mask = max(1, int(n_valid * mask_ratio))
        sel = valid_idx[torch.randperm(n_valid, device=device)[:n_mask]]
        mask[b, sel] = 1.0
    return mask


def ring_mask(ring_indices, valid_mask=None, non_ring_ratio=0.10):
    """环结构化掩码 (掩码所有目标环原子 + 10% 随机非环原子)

    优先掩码目标环上的所有原子, 再额外掩码少量随机非环原子,
    使总掩码比例与 random_mask 接近, 突出环结构信息的重建难度。

    Args:
        ring_indices:   (batch, n_atoms) 目标环原子位置标记 (-1=非环原子)
        valid_mask:     (batch, n_atoms) 或 (batch, n_atoms, node_vec_len); None=全部有效
        non_ring_ratio: 额外掩码非环原子比例
    Returns:
        mask: (batch, n_atoms) 1.0=掩码, 0.0=保留
    """
    batch_size, n_atoms = ring_indices.shape
    device = ring_indices.device
    valid_mask = _reduce_valid_mask(valid_mask, batch_size, n_atoms, device)
    # 环原子 (真实原子且在目标环上)
    ring_atom = ((ring_indices >= 0) & (valid_mask > 0)).float()
    # 非环真实原子
    non_ring_atom = ((ring_indices < 0) & (valid_mask > 0)).float()
    # 掩码所有目标环原子
    mask = ring_atom.clone()
    # 额外掩码 10% 随机非环原子
    for b in range(batch_size):
        non_ring_idx = torch.where(non_ring_atom[b] > 0)[0]
        n_non_ring = len(non_ring_idx)
        n_extra = int(n_non_ring * non_ring_ratio)
        if n_extra > 0:
            sel = non_ring_idx[torch.randperm(n_non_ring, device=device)[:n_extra]]
            mask[b, sel] = 1.0
    return mask


def pretrain(mae_model, data, train_idx, device, mask_type='random',
             n_epochs=100, batch_size=64, lr=1e-3, weight_decay=1e-5, seed=42):
    """预训练循环 (自监督, 无标签)

    使用掩码原子特征重建 (MSE) 作为自监督目标, 仅在 train_idx 上训练。
    重建目标: 被掩码原子的原始 node_mat 特征。

    Args:
        mae_model:  MaskedAutoencoder
        data:       dict (node_mats, adj_mats, ring_indices, mask_mats)
        train_idx:  训练样本索引 (torch tensor, on device)
        device:     torch device
        mask_type:  'random' or 'ring'
        n_epochs:   预训练轮数
    Returns:
        mae_model: 预训练后的模型
    """
    set_full_seed(seed)
    node_mats = data['node_mats']
    adj_mats = data['adj_mats']
    ring_indices = data['ring_indices']
    mask_mats = data.get('mask_mats', None)
    n_train = len(train_idx)

    optimizer = torch.optim.Adam(mae_model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    for epoch in range(1, n_epochs + 1):
        mae_model.train()
        perm = train_idx[torch.randperm(n_train, device=device)]
        total_loss = 0.0
        for i in range(0, n_train, batch_size):
            bi = perm[i:i + batch_size]
            node_mat = node_mats[bi]
            adj_mat = adj_mats[bi]
            ri = ring_indices[bi]
            vm = mask_mats[bi] if mask_mats is not None else None
            # 生成掩码
            if mask_type == 'random':
                m = random_mask(node_mat, valid_mask=vm)
            else:  # 'ring'
                m = ring_mask(ri, valid_mask=vm)
            optimizer.zero_grad(set_to_none=True)
            pred, target = mae_model(node_mat, adj_mat, ri, m)
            # 仅计算掩码位置的重建 loss
            m_exp = m.unsqueeze(-1).expand_as(pred)
            pred_m = pred[m_exp.bool()]
            target_m = target[m_exp.bool()]
            loss = loss_fn(pred_m, target_m)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mae_model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(bi)
        avg_loss = total_loss / n_train
        if epoch % 10 == 0 or epoch == 1:
            print(f"    [Pretrain/{mask_type}] epoch {epoch}/{n_epochs} recon_loss={avg_loss:.6f}", flush=True)

    del optimizer
    return mae_model
